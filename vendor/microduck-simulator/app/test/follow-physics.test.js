import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web/wasm";
import { buildParkModelXml } from "../src/game/shared-model.js";
import { CompanionController } from "../src/game/companion-controller.js";
import { POLICIES } from "../src/game/constants.js";
import { followCommand, obstacleForSlot, FOLLOW_MIN_SEPARATION_M } from "../src/game/park-geometry.js";

const publicRoot = new URL("../public/", import.meta.url);
const modelRoot = new URL("robot/mjlab/", publicRoot);
const source = readFileSync(new URL("robot_allcollisions.xml", modelRoot), "utf8");
const mujoco = await loadMujoco();
ort.env.wasm.numThreads = 1;
const sessions = {};
for (const name of ["walk", "stand", "sitstand"]) {
  sessions[name] = await ort.InferenceSession.create(new Uint8Array(readFileSync(new URL(POLICIES[name], publicRoot))), { executionProviders: ["wasm"] });
}

function fixture({ obstacleSlot = "off", positions } = {}) {
  const { xml, meshFiles } = buildParkModelXml(source, source);
  const vfs = new mujoco.MjVFS();
  for (const file of meshFiles) vfs.addBuffer(`assets/${file}`, new Uint8Array(readFileSync(new URL(`meshes/${file}`, modelRoot))));
  const model = mujoco.MjModel.from_xml_string(xml, vfs), data = new mujoco.MjData(model);
  mujoco.mj_resetDataKeyframe(model, data, 0);
  const obstacle = obstacleForSlot(obstacleSlot);
  data.mocap_pos.set(obstacle.position);
  const options = {
    mujoco, ort, getWorld: () => ({ model, data }), getSessions: () => sessions,
    getSpatialContext: () => ({ peers: [] }), paused: () => false, locked: () => false,
    quack() {}, wheee() {}, spawnBall() {}, switchLoco() {},
  };
  const leader = new CompanionController({ ...options, prefix: "" }), follower = new CompanionController(options);
  if (positions) {
    data.qpos.set(positions[0], leader.qAdr);
    data.qpos.set(positions[1], follower.qAdr);
  }
  mujoco.mj_forward(model, data);
  const metrics = { minimumGap: Infinity, travelM: 0, unsafeContact: false, falls: 0, lastDecision: null, snapshots: [] };
  let last = follower.qpos();
  const step = async (count, follow = true) => {
    for (let n = 0; n < count; n++) {
      const decision = follow ? followCommand({ follower: follower.status(), leader: leader.status(), obstacle }) : { command: [0, 0, 0] };
      await leader.beforeStep(); await follower.beforeStep(decision.command);
      for (let s = 0; s < 4; s++) mujoco.mj_step(model, data);
      leader.afterStep(); follower.afterStep();
      const a = leader.qpos(), b = follower.qpos();
      metrics.minimumGap = Math.min(metrics.minimumGap, Math.hypot(a[0] - b[0], a[1] - b[1]));
      metrics.travelM += Math.hypot(b[0] - last[0], b[1] - last[1]); last = b;
      if (leader.fallen() || follower.fallen()) metrics.falls++;
      const boxId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "park_obstacle_geom");
      const contacts = data.contact;
      try {
        for (let c = 0; c < data.ncon; c++) {
          const contact = contacts.get(c);
          try { if (contact.geom[0] === boxId || contact.geom[1] === boxId) metrics.unsafeContact = true; }
          finally { contact.delete(); }
        }
      } finally { contacts.delete(); }
      metrics.lastDecision = decision;
      if (Math.round(data.time * 50) % 500 === 0) metrics.snapshots.push({ seconds: data.time, xy: Array.from(b.slice(0, 2)), yaw: follower.spatial().headingRad, command: decision.command });
    }
  };
  return { leader, follower, data, metrics, step, dispose: () => { data.delete(); model.delete(); vfs.delete(); } };
}

test("native policies move the companion from beside its leader and follow again after the leader walks", async t => {
  const f = fixture();
  try {
    await f.step(100, false);
    const start = f.follower.qpos();
    await f.step(1600);
    const near = f.follower.qpos(), leader = f.leader.qpos();
    assert.ok(Math.hypot(near[0] - start[0], near[1] - start[1]) > 0.25, JSON.stringify(f.metrics));
    assert.ok(near[0] < leader[0] - 0.15, JSON.stringify(f.metrics));
    assert.ok(Math.hypot(near[0] - leader[0], near[1] - leader[1]) < 0.82, JSON.stringify(f.metrics));
    assert.match(f.metrics.lastDecision.reason, /Target spacing reached/);
    for (let pulse = 0; pulse < 4; pulse++) {
      assert.equal(f.leader.command("walk_forward").accepted, true);
      await f.step(160);
    }
    await f.step(1600);
    const after = f.follower.qpos();
    assert.ok(Math.hypot(after[0] - near[0], after[1] - near[1]) > 0.08, JSON.stringify(f.metrics));
    assert.ok(f.metrics.minimumGap >= FOLLOW_MIN_SEPARATION_M, JSON.stringify(f.metrics));
    assert.equal(f.metrics.falls, 0);
    t.diagnostic(JSON.stringify(f.metrics));
  } finally { f.dispose(); }
});

test("native following makes progress around a physical center obstacle without box contact", async t => {
  const f = fixture({ obstacleSlot: "center", positions: [[1.05, 0, 0.12], [-1.05, 0, 0.12]] });
  try {
    await f.step(100, false);
    await f.step(3000);
    const end = f.follower.qpos();
    assert.ok(end[0] > 0.25, JSON.stringify({ end: Array.from(end), metrics: f.metrics }));
    assert.ok(f.metrics.minimumGap >= FOLLOW_MIN_SEPARATION_M, JSON.stringify(f.metrics));
    assert.equal(f.metrics.unsafeContact, false);
    assert.equal(f.metrics.falls, 0);
    t.diagnostic(JSON.stringify(f.metrics));
  } finally { f.dispose(); }
});

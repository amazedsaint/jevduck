import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web/wasm";
import { buildParkModelXml } from "../src/game/shared-model.js";
import { CompanionController } from "../src/game/companion-controller.js";
import { BallTaskController } from "../src/game/ball-task.js";
import { POLICIES } from "../src/game/constants.js";
import { relativeBall } from "../src/game/ball-goal.js";
import { obstacleForSlot, FOLLOW_MIN_SEPARATION_M } from "../src/game/park-geometry.js";

const publicRoot = new URL("../public/", import.meta.url), modelRoot = new URL("robot/mjlab/", publicRoot);
const source = readFileSync(new URL("robot_allcollisions.xml", modelRoot), "utf8");
const mujoco = await loadMujoco(), sessions = {}, hashes = {};
ort.env.wasm.numThreads = 1;
for (const name of ["walk", "stand", "sitstand", "kickL", "kickR"]) {
  const bytes = new Uint8Array(readFileSync(new URL(POLICIES[name], publicRoot)));
  hashes[name] = createHash("sha256").update(bytes).digest("hex");
  sessions[name] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

function fixture({ selected = "duck1", obstacleSlot = "off", positions = null, peerPositions = null } = {}) {
  const four = selected === "duck3" || selected === "duck4";
  const { xml, meshFiles } = buildParkModelXml(source, four ? [source, source, source] : source, [], { layout: four ? "swarm" : "park" }), vfs = new mujoco.MjVFS();
  for (const file of meshFiles) vfs.addBuffer(`assets/${file}`, new Uint8Array(readFileSync(new URL(`meshes/${file}`, modelRoot))));
  const model = mujoco.MjModel.from_xml_string(xml, vfs), data = new mujoco.MjData(model);
  mujoco.mj_resetDataKeyframe(model, data, 0);
  const obstacle = obstacleForSlot(obstacleSlot); data.mocap_pos.set(obstacle.position);
  const options = { mujoco, ort, getWorld: () => ({ model, data }), getSessions: () => sessions,
    getSpatialContext: () => ({ peers: [] }), paused: () => false, locked: () => false,
    quack() {}, wheee() {}, spawnBall() { throw new Error("Ball tasks cannot spawn a ball."); }, switchLoco() {} };
  const first = new CompanionController({ ...options, prefix: "" }), second = new CompanionController(options);
  const actors = [first, second, ...(four ? [new CompanionController({ ...options, prefix: "duck3_" }), new CompanionController({ ...options, prefix: "duck4_" })] : [])];
  const duck = actors[Number(selected.slice(-1)) - 1], peers = actors.filter(actor => actor !== duck), peer = peers[0];
  for (const actor of actors) actor.getSpatialContext = () => ({ obstacle, peers: actors.filter(other => other !== actor).map(other => other.pose()) });
  if (positions) { data.qpos.set(positions[0], duck.qAdr); data.qpos.set(positions[1], peer.qAdr); }
  if (peerPositions) peers.forEach((actor, index) => data.qpos.set(peerPositions[index], actor.qAdr));
  mujoco.mj_forward(model, data);
  const bq = model.jnt("ball_freejoint").qposadr, bv = model.jnt("ball_freejoint").dofadr;
  const boxId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "park_obstacle_geom");
  const ballId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "ball_geom");
  let present = false, unsafeContacts = 0, minGap = Infinity, falls = 0;
  const calls = [], snapshots = [];
  const ball = () => ({ present, position: Array.from(data.qpos.slice(bq, bq + 3)), speedMps: Math.hypot(...data.qvel.slice(bv, bv + 3)) });
  const task = new BallTaskController({
    getState: () => ({ duck: { ...duck.status(), speedMps: Math.hypot(...data.qvel.slice(duck.vAdr, duck.vAdr + 3)) }, ball: ball(), obstacle, peers: peers.map(actor => actor.pose()) }),
    nativeAction: (id, action) => { assert.equal(id, selected); calls.push({ action, atS: data.time, ball: ball(), relative: relativeBall(duck.pose(), ball()) }); return duck.command(action); },
  });
  const step = async (count = 1, driveTask = true) => {
    for (let n = 0; n < count; n++) {
      if (driveTask) task.tick();
      await duck.beforeStep(task.active ? task.command : [0, 0, 0]);
      for (const actor of peers) await actor.beforeStep();
      for (let s = 0; s < 4; s++) {
        mujoco.mj_step(model, data);
        task.observePhysics({ mujoco, model, data, ballQposAdr: bq });
        const contacts = data.contact;
        try {
          for (let c = 0; c < data.ncon; c++) {
            const contact = contacts.get(c);
            try {
              const one = Number(contact.geom[0]), two = Number(contact.geom[1]);
              if ((one === boxId || two === boxId) && one !== ballId && two !== ballId) unsafeContacts++;
            } finally { contact.delete(); }
          }
        } finally { contacts.delete(); }
      }
      for (const actor of actors) actor.afterStep();
      if (actors.some(actor => actor.fallen())) falls++;
      for (const actor of peers) minGap = Math.min(minGap, Math.hypot(duck.qpos()[0] - actor.qpos()[0], duck.qpos()[1] - actor.qpos()[1]));
    }
  };
  const run = async (action = "kick_ball", id = "native-goal") => {
    assert.equal(task.start(action, id, selected).accepted, true);
    for (let n = 0; task.active && n <= 6000; n++) {
      if (n % 250 === 0) snapshots.push({ ...task.snapshot(), position: duck.pose().position, relative: relativeBall(duck.pose(), ball()) });
      await step();
    }
    return { selected, task: task.snapshot(), calls, unsafeContacts, minGapM: minGap, falls, snapshots, policyHashes: hashes };
  };
  return { duck, peer, task, data, model, step, run,
    placeBall(position) { data.qpos.set(position, bq); data.qvel.fill(0, bv, bv + 6); mujoco.mj_forward(model, data); present = true; },
    dispose() { data.delete(); model.delete(); vfs.delete(); } };
}

for (const [name, setup] of [
  ["Sunny and its default nearby peer", { ballPosition: [-.25, .025, .055] }],
  ["Blue and its default nearby peer", { selected: "duck2", ballPosition: [-.25, .675, .055] }],
  ["Sage with three physical peers", { selected: "duck3", positions: [[-.6, .45, .12], [-.9, -.45, .12]], peerPositions: [[-.9, -.45, .12], [.15, -.45, .12], [.9, -.45, .12]], ballPosition: [-.25, .475, .055] }],
  ["Plum with three physical peers", { selected: "duck4", ballPosition: [.5, .475, .055] }],
  ["ball initially behind", { positions: [[-.6, 0, .12], [1, -1, .12]], ballPosition: [-1.1, 0, .055] }],
  ["ball beyond a physical box", { positions: [[-1.05, 0, .12], [1, -1, .12]], obstacleSlot: "center", ballPosition: [.8, 0, .055] }],
]) test(`native kick_ball approaches, settles and verifies real contact: ${name}`, async t => {
  const f = fixture(setup);
  try {
    await f.step(100, false); f.placeBall(setup.ballPosition);
    const result = await f.run(); t.diagnostic(JSON.stringify(result));
    assert.equal(result.task.outcome, "succeeded");
    assert.equal(result.task.phase, "complete");
    assert.equal(result.task.ballContact, true);
    assert.ok(result.task.ballDisplacementM >= .05);
    assert.equal(result.unsafeContacts, 0); assert.equal(result.falls, 0);
    assert.ok(result.minGapM >= FOLLOW_MIN_SEPARATION_M);
    assert.equal(result.calls.filter(call => call.action.startsWith("kick_")).length, 1);
  } finally { f.dispose(); }
});

test("Sage stops explicitly when a live three-peer approach loses its admitted route", async t => {
  const f = fixture({ selected: "duck3" });
  try {
    await f.step(100, false); f.placeBall([-.4, .425, .055]);
    const result = await f.run(); t.diagnostic(JSON.stringify(result));
    assert.equal(result.task.outcome, "failed");
    assert.match(result.task.reason, /admissible route/);
    assert.equal(result.task.ballContact, false);
    assert.equal(result.unsafeContacts, 0); assert.equal(result.falls, 0);
    assert.ok(result.minGapM >= FOLLOW_MIN_SEPARATION_M);
  } finally { f.dispose(); }
});

test("native seated approach stands before moving and finishes without a kick", async t => {
  const f = fixture();
  try {
    await f.step(100, false);
    assert.equal(f.duck.command("sit").accepted, true);
    for (let n = 0; f.duck.busy() && n < 500; n++) await f.step(1, false);
    assert.equal(f.duck.status().posture, "sitting");
    f.placeBall([-.25, .025, .055]);
    const result = await f.run("approach_ball", "native-seated"); t.diagnostic(JSON.stringify(result));
    assert.equal(result.task.outcome, "succeeded");
    assert.equal(result.task.ballContact, false);
    assert.deepEqual(result.calls.map(call => call.action), ["stand"]);
  } finally { f.dispose(); }
});

test("native absent ball and box-covered target report failure without a kick", async t => {
  const f = fixture({ obstacleSlot: "center", positions: [[-1.05, 0, .12], [1, -1, .12]] });
  try {
    await f.step(100, false);
    const missing = await f.run("kick_ball", "native-absent");
    assert.equal(missing.task.outcome, "failed"); assert.equal(missing.calls.length, 0);
    f.placeBall([0, 0, .4]);
    const blocked = await f.run("kick_ball", "native-blocked"); t.diagnostic(JSON.stringify(blocked));
    assert.equal(blocked.task.outcome, "failed"); assert.equal(blocked.calls.length, 0);
    assert.equal(blocked.task.ballContact, false);
  } finally { f.dispose(); }
});

test("native verification rejects a missed kick even if alignment admission is faulty", async t => {
  const f = fixture();
  try {
    await f.step(100, false); f.placeBall([-.25, .025, .055]);
    // Matched falsifier: bypass the approach decision, reproducing the old
    // one-shot kick at normal spawn range. The independent physics result
    // must reject it even though the native actor finishes normally.
    f.task.planner = () => ({ command: [0, 0, 0], aligned: true, blocked: false });
    const result = await f.run("kick_ball", "native-missed-kick"); t.diagnostic(JSON.stringify(result));
    assert.equal(result.calls.filter(call => call.action.startsWith("kick_")).length, 1);
    assert.equal(result.task.outcome, "failed");
    assert.equal(result.task.ballContact, false);
    assert.ok(result.task.ballDisplacementM < .05);
    assert.match(result.task.reason, /without measured contact/);
  } finally { f.dispose(); }
});

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web/wasm";
import { buildParkModelXml } from "../src/game/shared-model.js";
import { CompanionController } from "../src/game/companion-controller.js";
import { POLICIES } from "../src/game/constants.js";
import { ballGoalCommand, relativeBall } from "../src/game/ball-goal.js";
import { obstacleForSlot, FOLLOW_MIN_SEPARATION_M } from "../src/game/park-geometry.js";

const publicRoot = new URL("../public/", import.meta.url), modelRoot = new URL("robot/mjlab/", publicRoot);
const source = readFileSync(new URL("robot_allcollisions.xml", modelRoot), "utf8");
const mujoco = await loadMujoco(), sessions = {}, hashes = {};
ort.env.wasm.numThreads = 1;
for (const name of ["walk", "stand", "sitstand"]) {
  const bytes = new Uint8Array(readFileSync(new URL(POLICIES[name], publicRoot)));
  hashes[name] = createHash("sha256").update(bytes).digest("hex");
  sessions[name] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

async function runScenario({ selected = "duck1", position = [-.6, 0, .12], ballPosition, obstacleSlot = "off", foot = "left", stopWhenBlocked = false }) {
  const { xml, meshFiles } = buildParkModelXml(source, source), vfs = new mujoco.MjVFS();
  for (const file of meshFiles) vfs.addBuffer(`assets/${file}`, new Uint8Array(readFileSync(new URL(`meshes/${file}`, modelRoot))));
  const model = mujoco.MjModel.from_xml_string(xml, vfs), data = new mujoco.MjData(model);
  mujoco.mj_resetDataKeyframe(model, data, 0);
  const obstacle = obstacleForSlot(obstacleSlot); data.mocap_pos.set(obstacle.position);
  const options = { mujoco, ort, getWorld: () => ({ model, data }), getSessions: () => sessions,
    getSpatialContext: () => ({ peers: [] }), paused: () => false, locked: () => false,
    quack() {}, wheee() {}, spawnBall() { throw new Error("Pursuit must not spawn or relocate the ball."); }, switchLoco() {} };
  const first = new CompanionController({ ...options, prefix: "" }), second = new CompanionController(options);
  const duck = selected === "duck1" ? first : second, peer = selected === "duck1" ? second : first;
  duck.getSpatialContext = () => ({ obstacle, peers: [peer.pose()] });
  peer.getSpatialContext = () => ({ obstacle, peers: [duck.pose()] });
  // Pose edits are fixture setup only. Every subsequent robot displacement
  // comes from the original ONNX actions and the shared MuJoCo solver.
  data.qpos.set(position, duck.qAdr); data.qpos.set([1, -1, .12], peer.qAdr);
  mujoco.mj_forward(model, data);
  const bq = model.jnt("ball_freejoint").qposadr, bv = model.jnt("ball_freejoint").dofadr;
  const boxId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "park_obstacle_geom");
  let unsafeContacts = 0;
  const step = async command => {
    await duck.beforeStep(command); await peer.beforeStep();
    for (let n = 0; n < 4; n++) {
      mujoco.mj_step(model, data);
      const contacts = data.contact;
      try {
        for (let c = 0; c < data.ncon; c++) {
          const contact = contacts.get(c);
          try { if (contact.geom[0] === boxId || contact.geom[1] === boxId) unsafeContacts++; }
          finally { contact.delete(); }
        }
      } finally { contacts.delete(); }
    }
    duck.afterStep(); peer.afterStep();
  };
  try {
    for (let n = 0; n < 100; n++) await step([0, 0, 0]);
    data.qpos.set(ballPosition, bq); data.qvel.fill(0, bv, bv + 6); mujoco.mj_forward(model, data);
    const peerStart = peer.qpos().slice(0, 2);
    let settling = false, stableSteps = 0, blockedSteps = 0, firstAlignedS = null, minimumGapM = Infinity, falls = 0;
    const snapshots = [];
    for (let n = 0; n < 2250; n++) {
      const state = duck.status(), ball = { position: Array.from(data.qpos.slice(bq, bq + 3)) };
      const decision = ballGoalCommand({ duck: state, ball, foot, obstacle, peers: [peer.pose()], settling });
      settling = decision.aligned;
      blockedSteps = decision.blocked ? blockedSteps + 1 : 0;
      const speed = Math.hypot(...data.qvel.slice(duck.vAdr, duck.vAdr + 3));
      stableSteps = settling && speed < .035 ? stableSteps + 1 : 0;
      if (settling) firstAlignedS ??= n * .02;
      if (n % 250 === 0 || stableSteps === 15 || stopWhenBlocked && blockedSteps === 50) snapshots.push({ seconds: n * .02, xy: state.position.slice(0, 2), headingRad: state.headingRad,
        relative: relativeBall(state, ball), command: decision.command, phase: decision.phase, aligned: decision.aligned, speedMps: speed, reason: decision.reason });
      await step(decision.command);
      const a = duck.qpos(), b = peer.qpos();
      minimumGapM = Math.min(minimumGapM, Math.hypot(a[0] - b[0], a[1] - b[1]));
      if (duck.fallen() || peer.fallen()) falls++;
      if (stableSteps >= 15 || stopWhenBlocked && blockedSteps >= 50) break;
    }
    return { selected, foot, obstacleSlot, firstAlignedS, stableSteps, blockedSteps, minimumGapM, falls, unsafeContacts,
      peerDriftM: Math.hypot(peer.qpos()[0] - peerStart[0], peer.qpos()[1] - peerStart[1]), snapshots, policyHashes: hashes };
  } finally { data.delete(); model.delete(); vfs.delete(); }
}

for (const [name, scenario] of [
  ["distant ball ahead", { ballPosition: [.3, 0, .055] }],
  ["ball behind the robot", { ballPosition: [-1.1, 0, .055] }],
  ["rear-left ball", { ballPosition: [-1.0, .35, .055] }],
  ["rear-right ball with right foot", { ballPosition: [-1.0, -.35, .055], foot: "right" }],
  ["off-axis ball", { ballPosition: [.1, .5, .055] }],
  ["selected Blue with right foot", { selected: "duck2", ballPosition: [.3, -.2, .055], foot: "right" }],
  ["ball beyond a physical box", { position: [-1.05, 0, .12], ballPosition: [.8, 0, .055], obstacleSlot: "center" }],
]) test(`native ball steering reaches and settles at the calibrated foot window: ${name}`, async t => {
  const result = await runScenario(scenario);
  t.diagnostic(JSON.stringify(result));
  assert.ok(result.stableSteps >= 15, "The real policy must reach the window and settle without state edits.");
  assert.equal(result.falls, 0); assert.equal(result.unsafeContacts, 0);
  assert.ok(result.minimumGapM >= FOLLOW_MIN_SEPARATION_M);
  assert.ok(result.peerDriftM < .015, "The unselected robot only runs its balance policy.");
});

test("native near-wall ball pursuit holds with an explicit clearance result", async t => {
  const heading = 70.4 * Math.PI / 180;
  const result = await runScenario({ position: [1.22, .47, .12, Math.cos(heading / 2), 0, 0, Math.sin(heading / 2)],
    ballPosition: [1.432, .668, .055], stopWhenBlocked: true });
  t.diagnostic(JSON.stringify(result));
  assert.equal(result.blockedSteps, 50); assert.equal(result.firstAlignedS, null);
  assert.equal(result.falls, 0); assert.equal(result.unsafeContacts, 0);
  assert.ok(result.minimumGapM >= FOLLOW_MIN_SEPARATION_M);
  for (const snapshot of result.snapshots) {
    assert.deepEqual(snapshot.command, [0, 0, 0]);
    assert.match(snapshot.reason, /wall.*guarded approach/);
  }
});

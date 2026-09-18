import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web/wasm";
import { buildParkModelXml } from "../src/game/shared-model.js";
import { CompanionController } from "../src/game/companion-controller.js";
import { DUCK_ACTIONS, availableDuckActions } from "../src/game/duck-actions.js";
import { POLICIES, DEFAULT_POSE } from "../src/game/constants.js";

const publicRoot = new URL("../public/", import.meta.url);
const modelRoot = new URL("robot/mjlab/", publicRoot);
const legSource = readFileSync(new URL("robot_allcollisions.xml", modelRoot), "utf8");
const rollerSource = readFileSync(new URL("robot_allcollisions_rollers.xml", modelRoot), "utf8");
const mujoco = await loadMujoco();
ort.env.wasm.numThreads = 1;
const sessions = {};
for (const name of ["walk", "sitstand", "stand", "roll", "kickL", "kickR", "groundpick", "drive", "crouch"]) {
  sessions[name] = await ort.InferenceSession.create(new Uint8Array(readFileSync(new URL(POLICIES[name], publicRoot))), { executionProviders: ["wasm"] });
}

function fixture({ rollers = false } = {}) {
  const { xml, meshFiles } = buildParkModelXml(legSource, rollers ? rollerSource : legSource);
  const vfs = new mujoco.MjVFS();
  for (const file of meshFiles) vfs.addBuffer(`assets/${file}`, new Uint8Array(readFileSync(new URL(`meshes/${file}`, modelRoot))));
  const model = mujoco.MjModel.from_xml_string(xml, vfs), data = new mujoco.MjData(model);
  mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data);
  let paused = false;
  const effects = [];
  const options = {
    mujoco, ort, getWorld: () => ({ model, data }), getSessions: () => sessions,
    getSpatialContext: () => ({ peers: [] }), paused: () => paused, locked: () => false,
    quack: () => effects.push("quack"), wheee: () => effects.push("wheee"), spawnBall: () => effects.push("spawn"), switchLoco: name => effects.push(name),
  };
  const first = new CompanionController({ ...options, prefix: "" }), second = new CompanionController(options);
  if (rollers) second.resolve("rollers");
  const step = async (count = 1) => {
    for (let n = 0; n < count; n++) {
      await first.beforeStep(); await second.beforeStep();
      for (let substep = 0; substep < 4; substep++) mujoco.mj_step(model, data);
      first.afterStep(); second.afterStep();
    }
  };
  return { model, data, first, second, effects, step, pause: () => { paused = true; }, dispose: () => { data.delete(); model.delete(); vfs.delete(); } };
}

test("each duck resolves distinct actuator, observation and history storage in one physics world", () => {
  const f = fixture();
  try {
    assert.equal(f.model.nu, 28);
    assert.deepEqual(f.first.ctrlAdr, Array.from({ length: 14 }, (_, n) => n));
    assert.deepEqual(f.second.ctrlAdr, Array.from({ length: 14 }, (_, n) => n + 14));
    assert.notEqual(f.first.gyroAdr, f.second.gyroAdr);
    assert.notEqual(f.first.lastAction.buffer, f.second.lastAction.buffer);
    f.second.lastAction.fill(.25);
    const a = f.first.buildObs(), b = f.second.buildObs();
    assert.equal(a.length, 61); assert.equal(b.length, 61);
    assert.ok(a.slice(34, 48).every(value => value === 0));
    assert.ok(b.slice(34, 48).every(value => value === .25));
    const orientation = new THREE.Quaternion().setFromEuler(new THREE.Euler(.25, -.4, .7));
    f.data.qpos.set([orientation.w, orientation.x, orientation.y, orientation.z], f.second.qAdr + 3);
    mujoco.mj_forward(f.model, f.data);
    const expected = new THREE.Vector3(0, 0, -1).applyQuaternion(orientation.clone().conjugate()).toArray();
    f.second.gravity().forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-12));
  } finally { f.dispose(); }
});

test("real ONNX movement advances only the commanded duck while both share one 50 Hz step clock", async () => {
  const f = fixture();
  try {
    await f.step(100);
    const anchor = f.first.qpos(), peerStart = f.second.qpos();
    assert.equal(f.second.command("walk_forward").accepted, true);
    await f.step(160);
    assert.ok(Math.abs(f.first.qpos()[0] - anchor[0]) < .01);
    assert.ok(f.second.qpos()[0] - peerStart[0] > .12);
    assert.equal(f.first.fallen(), false); assert.equal(f.second.fallen(), false);
    assert.equal(f.first.inferenceCount, 260); assert.equal(f.second.inferenceCount, 260);
    assert.ok(Math.abs(f.data.time - 260 * .02) < 1e-9);
    assert.equal(f.second.status().busy, false);
  } finally { f.dispose(); }
});

test("backward uses a physically effective reverse arc while retaining peer clearance", async t => {
  const f = fixture();
  try {
    f.second.getSpatialContext = () => ({ peers: [f.first.pose()] });
    assert.equal(f.second.command("walk_backward").accepted, false, "The initial peer blocks the effective reverse arc.");
    f.data.qpos[1] = -.7;
    mujoco.mj_forward(f.model, f.data);
    await f.step(100);
    const start = f.second.qpos(), heading = f.second.pose().headingRad;
    const result = f.second.command("walk_backward");
    assert.equal(result.accepted, true);
    assert.equal(f.second.pulse.command[0], -.2);
    assert.equal(Math.abs(f.second.pulse.command[2]), 1);
    let minimumGap = Infinity;
    for (let n = 0; n < 150; n++) {
      await f.step();
      const a = f.first.qpos(), b = f.second.qpos();
      minimumGap = Math.min(minimumGap, Math.hypot(a[0] - b[0], a[1] - b[1]));
    }
    const end = f.second.qpos();
    const backwardM = -((end[0] - start[0]) * Math.cos(heading) + (end[1] - start[1]) * Math.sin(heading));
    const turnedRad = Math.abs(Math.atan2(Math.sin(f.second.pose().headingRad - heading), Math.cos(f.second.pose().headingRad - heading)));
    assert.ok(backwardM > .06, `Measured backward displacement: ${backwardM}`);
    assert.ok(turnedRad > .6, `Measured turn: ${turnedRad}`);
    assert.ok(minimumGap >= .52, `Peer gap: ${minimumGap}`);
    assert.equal(f.second.fallen(), false);
    assert.equal(f.second.busy(), false);
    // A wall behind the duck blocks both turning envelopes, so the same
    // action is removed from live capabilities and rejected at execution.
    f.second.reset(); f.data.qpos[f.second.qAdr] = -1.28; mujoco.mj_forward(f.model, f.data);
    assert.equal(f.second.status().availableActions.includes("walk_backward"), false);
    assert.equal(f.second.command("walk_backward").accepted, false);
    t.diagnostic(JSON.stringify({ backwardM, turnedRad, minimumGap }));
  } finally { f.dispose(); }
});

test("real seated companion stands and settles before its queued walk can receive velocity", async () => {
  const f = fixture();
  try {
    await f.step(100);
    assert.equal(f.second.command("sit").accepted, true);
    for (let n = 0; n < 550 && f.second.busy(); n++) await f.step();
    assert.equal(f.second.status().posture, "sitting"); assert.equal(f.second.busy(), false);
    const start = f.second.qpos()[0];
    assert.equal(f.second.command("walk_forward").accepted, true);
    assert.equal(f.second.intent.pendingAction, "walk_forward");
    for (let n = 0; n < 99; n++) { await f.step(); assert.equal(f.second.pulse, null); }
    assert.equal(f.second.mode, "sitstand");
    for (let n = 0; n < 320 && f.second.busy(); n++) await f.step();
    assert.equal(f.second.status().posture, "standing"); assert.equal(f.second.busy(), false);
    assert.ok(f.second.qpos()[0] - start > .1);
    assert.equal(f.first.fallen(), false);
  } finally { f.dispose(); }
});

test("Stop removes a pending walk without interrupting the necessary trained stand handoff", async () => {
  const f = fixture();
  try {
    await f.step(100); f.second.command("sit");
    for (let n = 0; n < 550 && f.second.busy(); n++) await f.step();
    f.second.command("walk_forward"); await f.step(25);
    const result = f.second.command("stop");
    assert.equal(result.completion, "immediate"); assert.equal(f.second.intent.pendingAction, null);
    assert.ok(f.second.transition); await f.step(160);
    assert.equal(f.second.mode, "walk"); assert.equal(f.second.pulse, null);
    assert.deepEqual(Array.from(f.second.cmd.slice(0, 3)), [0, 0, 0]);
  } finally { f.dispose(); }
});

test("bounded native one-shots retain their trained clocks and locomotion gates", () => {
  const f = fixture();
  try {
    // Place the companion in open space for gesture admission.
    f.data.qpos[f.second.qAdr + 1] = 0; mujoco.mj_forward(f.model, f.data);
    for (const [action, mode, steps] of [["kick_left", "kickL", 25], ["kick_right", "kickR", 25], ["ground_pick", "groundpick", 140]]) {
      f.second.reset(); f.data.qpos[f.second.qAdr + 1] = 0; mujoco.mj_forward(f.model, f.data);
      assert.equal(f.second.command(action).accepted, true); assert.equal(f.second.mode, mode);
      for (let n = 0; n < steps - 1; n++) f.second.afterStep();
      assert.equal(f.second.mode, mode); f.second.afterStep(); assert.equal(f.second.mode, "walk");
      if (action.startsWith("kick")) { assert.equal(f.second.postKick, 20); for (let n = 0; n < 20; n++) f.second.afterStep(); }
      assert.equal(f.second.busy(), false);
    }
    assert.equal(f.second.command("crouch").accepted, false);
  } finally { f.dispose(); }
  const roller = fixture({ rollers: true });
  try {
    assert.equal(roller.second.command("roll").accepted, false);
    assert.equal(roller.second.command("ground_pick").accepted, false);
    assert.equal(roller.second.command("crouch").accepted, true);
    for (let n = 0; n < 174; n++) roller.second.afterStep();
    assert.equal(roller.second.mode, "crouch"); roller.second.afterStep(); assert.equal(roller.second.mode, "walk");
  } finally { roller.dispose(); }
});

test("sound and visual beak commands complete immediately without changing physics targets", () => {
  const f = fixture();
  try {
    const before = Array.from(f.data.ctrl);
    for (const action of ["quack", "wheee", "open_mouth", "close_mouth", "spawn_ball"]) {
      assert.equal(f.second.command(action).completion, "immediate");
      assert.deepEqual(Array.from(f.data.ctrl), before);
    }
    assert.deepEqual(f.effects, ["quack", "wheee", "spawn"]);
    f.pause(); assert.equal(f.second.command("walk_forward").accepted, false);
  } finally { f.dispose(); }
});

test("complete action surface offers only the trained gestures for the active locomotion variant", () => {
  const state = { ready: true, paused: false, busy: false, fallen: false, posture: "standing", spatialValid: true, clearance: { front: 1, back: 1, left: 1, right: 1 } };
  const legs = availableDuckActions({ ...state, loco: "legs" }), rollers = availableDuckActions({ ...state, loco: "rollers" });
  assert.ok(legs.includes("sit") && legs.includes("stand"));
  assert.ok(legs.includes("roll") && !legs.includes("crouch"));
  assert.ok(rollers.includes("crouch") && !rollers.includes("ground_pick"));
  assert.deepEqual(new Set([...legs, ...rollers]), new Set(DUCK_ACTIONS));
  assert.deepEqual(availableDuckActions({ ...state, paused: true }), ["stop", "reset"]);
  assert.equal(DEFAULT_POSE.length, 14);
});

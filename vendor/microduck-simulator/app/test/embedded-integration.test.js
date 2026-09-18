import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { measureSpatial } from "../src/game/spatial-guard.js";
import { DUCK_ACTIONS } from "../src/game/duck-actions.js";

// Execute the actual browser bridge with a minimal same-origin window.
// No copied dispatch logic: ordering assertions cover the shipped handler.
function bridgeFixture(command, overrides = {}) {
  const posted = [];
  const listeners = new Map();
  const origin = "https://simulator.test";
  const parent = { postMessage: (message, target) => { assert.equal(target, origin); posted.push(message); } };
  const context = {
    DUCK_ACTIONS, URLSearchParams, location: { search: "?embed=1", origin },
    window: { parent, addEventListener: (name, handler) => listeners.set(name, handler), removeEventListener: name => listeners.delete(name) },
    document: { hidden: false, addEventListener() {}, removeEventListener() {} },
    setInterval: () => 1, clearInterval() {},
    useGame: { getState: () => ({ bootFailed: false }) }, bootLog: [],
    PRESENTATION_SCENES: new Set(["studio", "moon", "sunset"]),
    PRESENTATION_CAMERAS: new Set(["orbit", "follow", "eyes"]),
  };
  const source = readFileSync(new URL("../src/game/embedded.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "").replace(/^export /gm, "");
  const bridge = runInNewContext(`${source}\n({ mountEmbeddedBridge, registerEmbeddedRuntime });`, context);
  bridge.mountEmbeddedBridge();
  const state = { ready: true, busy: false, autonomyActive: true, manual: false, guardSeq: 0, guardReason: null };
  bridge.registerEmbeddedRuntime({
    getStatus: () => ({ ...state }), command: (action, options) => command(action, state, options),
    presentation() {}, setPaused() {}, setAutonomy() {},
    ...overrides,
  });
  const barrier = posted.at(-1).seq;
  posted.length = 0;
  return { posted, barrier, state,
    sendMessage: message => listeners.get("message")({ origin, source: parent, data: { channel: "microduck-sim-v1", ...message } }),
    send: (action = "walk_forward", id = "command-1") => listeners.get("message")({
      origin, source: parent, data: { channel: "microduck-sim-v1", type: "command", id, action },
    }),
  };
}

test("guard rejection publishes guard status before its negative acknowledgement", () => {
  const f = bridgeFixture((_action, state) => {
    state.guardSeq++;
    state.guardReason = "Wall guard blocked forward movement.";
    return { accepted: false, blockedByGuard: true, message: state.guardReason };
  });
  f.send();
  assert.deepEqual(f.posted.map(message => message.type), ["status", "ack", "status"]);
  const [status, ack] = f.posted;
  assert.equal(status.guardSeq, 1);
  assert.equal(status.autonomyActive, true);
  assert.equal(status.busy, false);
  assert.equal(ack.accepted, false);
  assert.equal(ack.blockedByGuard, true);
  assert.equal(ack.statusSeq, status.seq);
});

test("accepted commands preserve ACK then fresh busy status and its old sequence barrier", () => {
  const f = bridgeFixture((_action, state) => { state.busy = true; return { accepted: true, message: "Moving." }; });
  f.send();
  assert.deepEqual(f.posted.map(message => message.type), ["ack", "status"]);
  assert.equal(f.posted[0].completion, "status");
  assert.equal(f.posted[0].statusSeq, f.barrier);
  assert.equal(f.posted[1].busy, true);
  assert.ok(f.posted[1].seq > f.posted[0].statusSeq);
});

test("ordinary rejections retain their existing acknowledgement ordering", () => {
  const f = bridgeFixture(() => ({ accepted: false, message: "Wait for the transition." }));
  f.send();
  assert.deepEqual(f.posted.map(message => message.type), ["ack", "status"]);
  assert.equal(f.posted[0].blockedByGuard, undefined);
});

test("every declared duck action reaches the real command adapter and Stop preserves its scope", () => {
  const calls = [];
  const f = bridgeFixture((action, _state, options) => { calls.push({ action, all: options.all }); return { accepted: true, completion: "immediate" }; });
  for (const action of DUCK_ACTIONS) f.send(action, action);
  assert.deepEqual(calls.map(call => call.action), DUCK_ACTIONS);
  assert.ok(calls.every(call => call.all === false));
  f.sendMessage({ type: "command", id: "stop-everyone", action: "stop", all: true });
  assert.deepEqual(calls.at(-1), { action: "stop", all: true });
});

test("ball task commands carry their bridge command id into the runtime", () => {
  const calls = [];
  const f = bridgeFixture((action, _state, options) => { calls.push({ action, id: options.id }); return { accepted: true }; });
  f.send("kick_ball", "ball-goal-17");
  assert.deepEqual(calls, [{ action: "kick_ball", id: "ball-goal-17" }]);
});

test("selection and occupancy-checked park edits acknowledge then publish authoritative status", () => {
  const calls = [];
  const f = bridgeFixture(() => ({ accepted: true }), {
    selectDuck: id => { calls.push(id); return { accepted: true, message: "Selected." }; },
    park: options => { calls.push(options.obstacle); return { accepted: false, message: "A duck occupies that slot." }; },
  });
  f.sendMessage({ type: "select-duck", duckId: "duck2" });
  assert.deepEqual(f.posted.map(message => message.type), ["selection-result", "status"]);
  assert.equal(f.posted[0].accepted, true);
  f.posted.length = 0;
  f.sendMessage({ type: "park", obstacle: "center" });
  assert.deepEqual(f.posted.map(message => message.type), ["park-result", "status"]);
  assert.equal(f.posted[0].accepted, false);
  assert.deepEqual(calls, ["duck2", "center"]);
  f.sendMessage({ type: "select-duck", duckId: "duck5" });
  f.sendMessage({ type: "park", obstacle: "arbitrary" });
  assert.equal(calls.length, 2);
});

function manualRaceFixture() {
  const source = readFileSync(new URL("../src/game/game.js", import.meta.url), "utf8");
  const functionSource = (name) => {
    const start = source.indexOf(`  function ${name}(`);
    assert.ok(start >= 0);
    return source.slice(start, source.indexOf("\n  }", start) + 5);
  };
  let held = true;
  const autonomyGuard = { active: true, guardReason: null, guardSeq: 0, monitor() {} };
  const context = {
    embeddedPaused: false, embeddedSuspended: () => false, manualInputActive: () => held, grab: null,
    swarm: null,
    selectedDuckId: "duck1", spatialWorld: () => ({}), coastSettle: null, parkSettle: null, postKickLock: 0,
    primaryPose: () => ({}), sweptMotion: () => ({ allowed: true }), movementProfile: () => [0, 0, 0], reverseProfile: () => [-.2, 0, 1],
    setAutonomy: value => { autonomyGuard.active = value; }, autonomyGuard,
    poseIsDead: () => null, locomotionIntent: null, externalSource: { isActive: () => false },
    recovery: null, sitTimer: null, standTimer: null, sitSettle: { pending: false, error: null },
    mode: "walk", sitFlag: 0, loco: "legs", POLICIES: { walk: "official-walk.onnx" },
    store: () => ({ bootDone: true }), inferenceCount: 1, runtimeError: null,
    data: { time: 1, qpos: new Float64Array([0, 0, 0.116, 1, 0, 0, 0]) }, measureSpatial,
    projGravZ: () => -1, cmd: new Float32Array(3), protectedMotionBusy: () => false,
    controller: { neutralHeld: false }, headMode: false,
    presentation: { scene: "studio" }, cameraMode: "orbit", fpsEma: 60,
  };
  const api = runInNewContext(`${functionSource("checkAutonomousMotion")}\n${functionSource("primaryStatus")}\n({checkAutonomousMotion,embeddedStatus:primaryStatus});`, context);
  return { api, context, setHeld: value => { held = value; } };
}

test("manual telemetry identifies held input even before rAF emits the takeover event", () => {
  const f = manualRaceFixture();
  f.api.checkAutonomousMotion();
  const status = f.api.embeddedStatus();
  assert.equal(status.autonomyActive, false);
  assert.equal(status.manual, true);
  assert.equal(status.busy, true);
  assert.equal(f.context.controller.neutralHeld, false);
  f.setHeld(false);
  assert.equal(f.api.embeddedStatus().manual, false);
});

test("four-duck selection and correlated swarm bridge instructions reach the runtime once", () => {
  const calls = [];
  const f = bridgeFixture(() => ({ accepted: true }), {
    selectDuck: id => { calls.push({ type: "select", id }); return { accepted: true }; },
    swarm: value => { calls.push({ type: "prepare", ...value }); return { accepted: true }; },
    swarmIntent: value => { calls.push({ type: "intent", ...value }); return { accepted: true }; },
  });
  f.sendMessage({ type: "select-duck", duckId: "duck4" });
  f.sendMessage({ type: "swarm", runId: "run-4", active: true, scenario: "flock" });
  const intent = { type: "swarm-intent", runId: "run-4", id: "advance-1", intent: "advance" };
  f.sendMessage(intent); f.sendMessage(intent);
  assert.deepEqual(calls.map(call => call.type), ["select", "prepare", "intent"]);
  assert.equal(calls[0].id, "duck4");
  const acks = f.posted.filter(message => message.type === "swarm-intent-result");
  assert.equal(acks.length, 2); assert.equal(acks[0].id, "advance-1"); assert.equal(acks[0].runId, "run-4");
});

test("a live physics grab also appears as manual takeover in the same status", () => {
  const f = manualRaceFixture();
  f.setHeld(false);
  f.context.grab = {};
  f.api.checkAutonomousMotion();
  const status = f.api.embeddedStatus();
  assert.equal(status.autonomyActive, false);
  assert.equal(status.manual, true);
});

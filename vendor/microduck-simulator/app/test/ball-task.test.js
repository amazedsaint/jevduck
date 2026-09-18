import test from "node:test";
import assert from "node:assert/strict";
import { BallTaskController } from "../src/game/ball-task.js";

function fixture(overrides = {}) {
  const state = { duck: { position: [0, 0, .116], headingRad: 0, loco: "legs", posture: "standing", busy: false, speedMps: 0 },
    ball: { present: true, position: [.095, .065, .05], speedMps: 0 }, peers: [] };
  const calls = [];
  const controller = new BallTaskController({ getState: () => state,
    nativeAction: (id, action) => { calls.push({ id, action }); return { accepted: true }; }, ...overrides });
  const tick = (n = 1) => { for (let i = 0; i < n; i++) controller.tick(); };
  return { state, calls, controller, tick };
}

test("missing ball fails explicitly without invoking a native action or spawning", () => {
  const f = fixture(); f.state.ball.present = false;
  assert.equal(f.controller.start("kick_ball", "missing", "duck1").accepted, true);
  assert.equal(f.controller.task.outcome, "failed");
  assert.match(f.controller.task.reason, /No ball/);
  assert.deepEqual(f.calls, []);
});

test("approach stands first and requires measured settling inside the foot window", () => {
  const f = fixture(); f.state.duck.posture = "sitting";
  f.controller.start("approach_ball", "seated", "duck2"); f.tick(5);
  assert.deepEqual(f.calls, [{ id: "duck2", action: "stand" }]);
  assert.equal(f.controller.task.phase, "standing");
  f.state.duck.posture = "standing"; f.state.duck.speedMps = .1; f.tick(20);
  assert.equal(f.controller.task.outcome, null);
  f.state.duck.speedMps = 0; f.tick(30);
  assert.equal(f.controller.task.outcome, "succeeded");
  assert.equal(f.controller.task.phase, "complete");
});

test("kick needs both selected-ankle contact and five centimetres of world ball travel", () => {
  for (const evidence of ["none", "contact", "movement", "both"]) {
    const f = fixture(); f.controller.start("kick_ball", evidence, "duck1"); f.tick(15);
    assert.deepEqual(f.calls, [{ id: "duck1", action: "kick_left" }]);
    if (["contact", "both"].includes(evidence)) f.controller.observeContact();
    if (["movement", "both"].includes(evidence)) f.state.ball.position[0] += .06;
    f.tick(110);
    assert.equal(f.controller.task.outcome, evidence === "both" ? "succeeded" : "failed", evidence);
    assert.equal(f.controller.task.commandId, evidence);
  }
});

test("moving duck relative to a stationary ball is not ball displacement", () => {
  const f = fixture(); f.controller.start("kick_ball", "relative-only", "duck1"); f.tick(15);
  f.controller.observeContact(); f.state.duck.position[0] -= .1; f.tick(110);
  assert.equal(f.controller.task.outcome, "failed");
  assert.equal(f.controller.task.ballDisplacementM, 0);
});

test("planar alignment with an elevated stationary ball cannot complete or launch a kick", () => {
  const f = fixture(); f.state.ball.position[2] = .4;
  f.controller.start("kick_ball", "elevated", "duck1"); f.tick(50);
  assert.equal(f.controller.task.outcome, null);
  assert.deepEqual(f.calls, []);
});

test("Stop preserves terminal evidence; fresh task ids replace it and clear is explicit null", () => {
  const f = fixture(); f.controller.start("approach_ball", "old", "duck1"); f.tick(15);
  const completed = f.controller.snapshot(); f.controller.cancel();
  assert.deepEqual(f.controller.snapshot(), completed);
  f.controller.start("kick_ball", "new", "duck2");
  assert.equal(f.controller.snapshot("duck1"), null);
  assert.equal(f.controller.task.commandId, "new");
  f.controller.clearTerminal(); assert.equal(f.controller.active, true);
  f.controller.cancel(); assert.equal(f.controller.task.outcome, "cancelled");
  assert.deepEqual(f.controller.command, [0, 0, 0]);
  f.controller.clearTerminal(); assert.equal(f.controller.snapshot(), null);
});

test("blocked route, pause and bounded timeout cannot finish as success", () => {
  const f = fixture({ planner: () => ({ blocked: true, reason: "No safe route.", command: [0, 0, 0] }) });
  f.controller.start("kick_ball", "blocked", "duck1"); f.tick(51);
  assert.equal(f.controller.task.outcome, "failed");
  const paused = fixture(); paused.controller.start("kick_ball", "paused", "duck1"); paused.state.duck.paused = true; paused.tick();
  assert.equal(paused.controller.task.outcome, "cancelled");
  const timed = fixture({ timeoutS: 1, planner: () => ({ blocked: false, aligned: false, phase: "approaching", command: [.25, 0, 0] }) });
  timed.controller.start("kick_ball", "timeout", "duck1"); timed.tick(51);
  assert.equal(timed.controller.task.outcome, "failed");
  assert.match(timed.controller.task.reason, /bounded/);
});

test("a selected-ankle touch after the kick actor ends cannot turn a miss into success", () => {
  const f = fixture(); f.controller.start("kick_ball", "late-contact", "duck1"); f.tick(15);
  assert.deepEqual(f.calls, [{ id: "duck1", action: "kick_left" }]);
  f.state.duck.mode = "walk"; f.tick();
  f.controller.observeContact();
  f.state.ball.position[0] += .06; f.tick(110);
  assert.equal(f.controller.task.ballContact, false);
  assert.equal(f.controller.task.outcome, "failed");
});

test("ball travel before the first kick contact cannot count as movement caused by that contact", () => {
  const f = fixture(); f.controller.start("kick_ball", "pre-contact-travel", "duck1"); f.tick(15);
  f.state.duck.mode = "kickL"; f.state.duck.busy = true;
  f.state.ball.position[0] += .06;
  f.controller.observeBall(f.state.ball.position);
  f.controller.observeContact();
  f.state.duck.mode = "walk"; f.state.duck.busy = false; f.tick(110);
  assert.equal(f.controller.task.ballContact, true);
  assert.ok(f.controller.task.ballDisplacementM >= .05);
  assert.equal(f.controller.task.outcome, "failed");
  assert.match(f.controller.task.reason, /after that contact/);
});

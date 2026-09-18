import test from "node:test";
import assert from "node:assert/strict";
import { ARENA_HALF } from "../src/game/constants.js";
import { ROBOT_MARGIN_M, measureSpatial, AutonomyGuard } from "../src/game/spatial-guard.js";
import { ExternalCommandSource } from "../src/game/controls/external.js";
import { LocomotionIntent } from "../src/game/controls/locomotion-intent.js";

const pose = (x = 0, y = 0, yaw = 0) => [x, y, 0.116, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} != ${expected}`);

test("cardinal clearances follow MuJoCo yaw and the real arena bounds", () => {
  const limit = ARENA_HALF - ROBOT_MARGIN_M;
  const expected = [limit - 0.6, limit + 0.2, limit + 0.6, limit - 0.2];
  for (const [index, yaw] of [0, Math.PI / 2, Math.PI, -Math.PI / 2].entries()) {
    const state = measureSpatial(pose(0.6, -0.2, yaw));
    assert.equal(state.spatialValid, true);
    close(state.headingRad, yaw);
    close(state.clearance.front, expected[index]);
    close(state.clearance.left, expected[(index + 1) % 4]);
    close(state.clearance.back, expected[(index + 2) % 4]);
    close(state.clearance.right, expected[(index + 3) % 4]);
  }
});

test("diagonal distance intersects the margin-inset wall instead of subtracting a ray radius", () => {
  const state = measureSpatial(pose(0, 0, Math.PI / 4));
  close(state.clearance.front, (ARENA_HALF - ROBOT_MARGIN_M) * Math.SQRT2);
});

test("near a wall the guard reports no outward space but retains retreat clearance", () => {
  const state = measureSpatial(pose(1.4, 0));
  assert.equal(state.spatialValid, true);
  assert.equal(state.clearance.front, 0);
  assert.equal(state.clearance.left, 0);
  close(state.clearance.back, 1.4 + ARENA_HALF - ROBOT_MARGIN_M);
});

test("invalid spatial input fails closed and scaled valid quaternions retain heading", () => {
  for (const q of [pose(NaN), pose(ARENA_HALF + 0.01), [0, 0, NaN, 1, 0, 0, 0], [0, 0, 0.1, 0, 0, 0, 0]]) {
    const state = measureSpatial(q);
    assert.equal(state.spatialValid, false);
    assert.deepEqual(state.clearance, { front: 0, back: 0, left: 0, right: 0 });
  }
  const scaled = pose(0, 0, Math.PI / 2);
  for (let i = 3; i < 7; i++) scaled[i] *= 3;
  close(measureSpatial(scaled).headingRad, Math.PI / 2);
});

function fixture() {
  let qpos = pose();
  let activeAction = null;
  let standRequests = 0;
  let pulses = 0;
  const source = new ExternalCommandSource({ getManualOverride: () => false, onCancel: () => { activeAction = null; } });
  let intent;
  const guard = new AutonomyGuard({
    getSpatial: () => measureSpatial(qpos),
    cancelMotion: () => { source.cancel(); activeAction = null; intent?.cancel(); },
  });
  const startPulse = (action) => {
    if (!guard.admit(action, 0.25)) return;
    source.start([0.25, 0, 0]);
    activeAction = action;
    pulses++;
  };
  intent = new LocomotionIntent({ beginStand: () => { standRequests++; }, startPulse });
  return { guard, intent, source, startPulse, pulses: () => pulses, standRequests: () => standRequests,
    move: (q) => { qpos = q; },
    monitor: () => { const action = activeAction || intent.pendingAction; if (action) guard.monitor(action); },
  };
}

test("autonomy is opt-in and admission reserves full pulse travel plus a stop buffer", () => {
  const f = fixture();
  f.move(pose(0.7)); // front clearance 0.64m, less than 0.25*2 + 0.18
  assert.equal(f.guard.active, false);
  assert.equal(f.guard.admit("walk_forward", 0.25), true);
  f.guard.setActive(true);
  assert.equal(f.guard.admit("walk_forward", 0.25), false);
  assert.equal(f.guard.guardSeq, 1);
  assert.match(f.guard.guardReason, /0.68 m is required/);
  assert.equal(f.guard.admit("turn_left", 0), true);
  assert.equal(f.guard.guardReason, null);
  assert.equal(f.guard.admit("walk_backward", -0.2), true);
});

test("running movement is cancelled at a newly measured wall boundary", () => {
  const f = fixture();
  f.guard.setActive(true);
  f.startPulse("walk_forward");
  assert.equal(f.source.isActive(), true);
  f.move(pose(1.17)); // remaining front clearance 0.17m
  f.monitor();
  assert.deepEqual(Array.from(f.source.command), [0, 0, 0]);
  assert.equal(f.source.isActive(), false);
  assert.equal(f.guard.active, true);
  assert.equal(f.guard.guardSeq, 1);
  f.monitor();
  assert.equal(f.guard.guardSeq, 1);
});

test("a guarded seated intention cannot launch after the stand policy completes", () => {
  const f = fixture();
  f.guard.setActive(true);
  assert.equal(f.guard.admit("walk_forward", 0.25), true);
  f.intent.request("walk_forward");
  f.move(pose(1.17));
  f.monitor();
  assert.equal(f.intent.pendingAction, null);
  for (let i = 0; i < 100; i++) f.intent.tick(0.02, { interrupted: false, transitioning: false, walking: true, settled: true });
  assert.equal(f.standRequests(), 1);
  assert.equal(f.pulses(), 0);
  assert.equal(f.guard.active, true);
  assert.equal(f.guard.guardSeq, 1);
});

test("queued movement is checked again at handoff when full-pulse room has shrunk", () => {
  const f = fixture();
  f.guard.setActive(true);
  f.intent.request("walk_forward");
  f.move(pose(0.8)); // enough for immediate stop, insufficient for a new pulse
  for (let i = 0; i < 10; i++) f.intent.tick(0.02, { interrupted: false, transitioning: false, walking: true, settled: true });
  assert.equal(f.intent.pendingAction, null);
  assert.equal(f.pulses(), 0);
  assert.equal(f.guard.guardSeq, 1);
});

test("revoking autonomy clears the queue and reason without rewinding guard receipts", () => {
  const f = fixture();
  f.guard.setActive(true);
  f.move(pose(1.17));
  assert.equal(f.guard.admit("walk_forward", 0.25), false);
  f.intent.request("walk_backward");
  f.guard.setActive(false);
  assert.equal(f.guard.active, false);
  assert.equal(f.guard.guardReason, null);
  assert.equal(f.guard.guardSeq, 1);
  assert.equal(f.intent.pendingAction, null);
});

test("unavailable spatial data blocks translation while leaving turns available", () => {
  const f = fixture();
  f.guard.setActive(true);
  f.move(pose(NaN));
  assert.equal(f.guard.admit("walk_forward", 0.25), false);
  assert.equal(f.guard.admit("turn_right", 0), true);
});

import test from "node:test";
import assert from "node:assert/strict";
import { LocomotionIntent } from "../src/game/controls/locomotion-intent.js";

function fixture() {
  let standRequests = 0;
  const pulses = [];
  const intent = new LocomotionIntent({
    beginStand: () => { standRequests++; },
    startPulse: (action) => { pulses.push(action); },
  });
  const tick = (count, changes = {}) => {
    for (let i = 0; i < count; i++) intent.tick(0.02, {
      interrupted: false, transitioning: false, walking: true, settled: true,
      ...changes,
    });
  };
  return { intent, pulses, tick, standRequests: () => standRequests };
}

test("seated movement waits for the official stand handoff and measured settling", () => {
  const f = fixture();
  f.intent.request("walk_forward");
  assert.equal(f.standRequests(), 1);
  assert.equal(f.intent.pendingAction, "walk_forward");
  assert.equal(f.intent.phase, "standing_up");
  f.tick(200, { transitioning: true });
  assert.deepEqual(f.pulses, []);
  f.tick(200, { walking: false });
  assert.deepEqual(f.pulses, []);
  f.tick(9);
  assert.equal(f.intent.phase, "settling");
  assert.equal(f.intent.pendingAction, "walk_forward");
  assert.deepEqual(f.pulses, []);
  f.tick(1);
  assert.deepEqual(f.pulses, ["walk_forward"]);
  assert.equal(f.intent.pendingAction, null);
  assert.equal(f.intent.phase, "idle");
  f.tick(100);
  assert.deepEqual(f.pulses, ["walk_forward"]);
});

test("an unsettled control step resets the consecutive stability gate", () => {
  const f = fixture();
  f.intent.request("turn_left");
  f.tick(9);
  f.tick(1, { settled: false });
  f.tick(9);
  assert.deepEqual(f.pulses, []);
  f.tick(1);
  assert.deepEqual(f.pulses, ["turn_left"]);
});

test("Stop cancels pending motion without reversing the required stand policy", () => {
  const f = fixture();
  f.intent.request("walk_backward");
  f.tick(20, { transitioning: true });
  f.intent.cancel();
  f.tick(200);
  assert.equal(f.standRequests(), 1);
  assert.equal(f.intent.pendingAction, null);
  assert.deepEqual(f.pulses, []);
});

test("a manual or lifecycle interruption cannot leave motion queued for resume", () => {
  const f = fixture();
  f.intent.request("walk_forward");
  f.tick(9);
  f.tick(1, { interrupted: true });
  f.tick(200);
  assert.equal(f.intent.pendingAction, null);
  assert.equal(f.intent.phase, "idle");
  assert.deepEqual(f.pulses, []);
});

test("a new request after cancellation gets a fresh stability gate", () => {
  const f = fixture();
  f.intent.request("walk_forward");
  f.tick(9);
  f.intent.cancel();
  f.intent.request("turn_right");
  f.tick(9);
  assert.deepEqual(f.pulses, []);
  f.tick(1);
  assert.equal(f.standRequests(), 2);
  assert.deepEqual(f.pulses, ["turn_right"]);
});

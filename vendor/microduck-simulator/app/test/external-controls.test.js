import test from "node:test";
import assert from "node:assert/strict";
import { Controller } from "../src/game/controls/controller.js";
import { ExternalCommandSource } from "../src/game/controls/external.js";

function fixture(t) {
  let now = 100;
  let cancellations = 0;
  t.mock.method(performance, "now", () => now);
  const manual = {
    id: "keyboard", held: false,
    command: new Float32Array([0.25, 0, 0]),
    isActive() { return this.held; },
  };
  const external = new ExternalCommandSource({
    getManualOverride: () => manual.held,
    onCancel: () => { cancellations++; },
  });
  const controller = new Controller({ sources: [manual, external] });
  return {
    manual, external, controller,
    advance: (milliseconds) => { now += milliseconds; },
    command: () => Array.from(controller.getCommand()),
    cancellations: () => cancellations,
  };
}

test("a pulse expires to zero after two seconds, even without a render poll", (t) => {
  const f = fixture(t);
  f.external.start([0.25, 0, 0]);
  f.advance(1999);
  assert.deepEqual(f.command(), [0.25, 0, 0]);
  f.advance(1);
  assert.deepEqual(f.command(), [0, 0, 0]);
  assert.equal(f.external.isActive(), false);
  assert.equal(f.cancellations(), 1);
});

test("a caller cannot extend a pulse beyond the two-second limit", (t) => {
  const f = fixture(t);
  f.external.start([0, 0, 0.5], 10000);
  f.advance(2000);
  assert.deepEqual(f.command(), [0, 0, 0]);
});

test("manual input wins and permanently cancels the pending pulse", (t) => {
  const f = fixture(t);
  f.external.start([0, 0, 0.5]);
  f.manual.held = true;
  f.controller.update(0.02);
  assert.deepEqual(f.command(), [0.25, 0, 0]);
  assert.equal(f.external.isActive(), false);
  f.manual.held = false;
  f.controller.update(0.02);
  assert.deepEqual(f.command(), [0, 0, 0]);
  assert.equal(f.cancellations(), 1);
});

test("stop and disposal clear a pulse without leaving a fallback command", (t) => {
  const f = fixture(t);
  f.external.start([0.25, 0, 0]);
  f.external.cancel();
  f.external.cancel();
  assert.deepEqual(f.command(), [0, 0, 0]);
  assert.equal(f.cancellations(), 1);
  f.external.start([0, 0, 0.5]);
  f.controller.dispose();
  assert.deepEqual(f.command(), [0, 0, 0]);
  assert.equal(f.cancellations(), 2);
});

test("Stop holds manual locomotion at zero until the controls return neutral", (t) => {
  const f = fixture(t);
  f.manual.held = true;
  f.controller.holdUntilNeutral(() => !f.manual.held);
  assert.deepEqual(f.command(), [0, 0, 0]);
  f.controller.update(0.02);
  assert.equal(f.controller.neutralHeld, true);
  assert.deepEqual(f.command(), [0, 0, 0]);
  f.manual.held = false;
  f.controller.update(0.02);
  assert.equal(f.controller.neutralHeld, false);
  assert.deepEqual(f.command(), [0, 0, 0]);
  f.manual.held = true;
  assert.deepEqual(f.command(), [0.25, 0, 0]);
});

test("releasing the neutral interlock cannot release a ceremony input lock", (t) => {
  const f = fixture(t);
  f.manual.held = true;
  f.controller.setLocked(true);
  f.controller.holdUntilNeutral(() => !f.manual.held);
  f.manual.held = false;
  f.controller.update(0.02);
  assert.equal(f.controller.neutralHeld, false);
  assert.equal(f.controller.locked, true);
  f.manual.held = true;
  assert.deepEqual(f.command(), [0, 0, 0]);
  f.controller.setLocked(false);
  assert.deepEqual(f.command(), [0.25, 0, 0]);
});

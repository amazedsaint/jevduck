import test from "node:test";
import assert from "node:assert/strict";
import { SitSettle } from "../src/game/controls/sit-settle.js";

function tick(settle, count, state) {
  for (let i = 0; i < count; i++) settle.tick(0.02, state);
}

test("sitting stays busy after the upstream hold timer starts the actual sit", () => {
  const settle = new SitSettle();
  settle.request();
  tick(settle, 40, { activated: false, settled: false });
  assert.equal(settle.pending, true);
  tick(settle, 25, { activated: true, settled: false });
  assert.equal(settle.pending, true);
  assert.equal(settle.phase, "settling");
  tick(settle, 7, { activated: true, settled: true });
  assert.equal(settle.pending, true);
  tick(settle, 1, { activated: true, settled: true });
  assert.equal(settle.pending, false);
  assert.equal(settle.phase, "idle");
});

test("a moving seated pose cannot pass the consecutive settle gate", () => {
  const settle = new SitSettle();
  settle.request();
  tick(settle, 7, { activated: true, settled: true });
  tick(settle, 1, { activated: true, settled: false });
  tick(settle, 7, { activated: true, settled: true });
  assert.equal(settle.pending, true);
  tick(settle, 1, { activated: true, settled: true });
  assert.equal(settle.pending, false);
});

test("a settle timeout reports failure without publishing successful idle", () => {
  const settle = new SitSettle();
  settle.request();
  tick(settle, 401, { activated: true, settled: false });
  assert.equal(settle.pending, true);
  assert.equal(settle.phase, "failed");
  assert.match(settle.error, /Reset/);
  tick(settle, 100, { activated: true, settled: true });
  assert.equal(settle.pending, true);
  assert.equal(settle.phase, "failed");
  settle.cancel();
  assert.equal(settle.pending, false);
  assert.equal(settle.error, null);
});

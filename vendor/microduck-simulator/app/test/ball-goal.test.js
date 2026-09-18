import test from "node:test";
import assert from "node:assert/strict";
import { ballGoalCommand, relativeBall } from "../src/game/ball-goal.js";
import { obstacleForSlot, sweptMotion } from "../src/game/park-geometry.js";

const duck = (x = 0, y = 0, headingRad = 0) => ({ position: [x, y, .12], headingRad, loco: "legs", posture: "standing" });
const ball = (x, y = 0) => ({ present: true, position: [x, y, .05] });

test("ball alignment is measured in the selected robot frame for either foot", () => {
  for (const foot of ["left", "right"]) {
    const side = foot === "left" ? 1 : -1;
    const first = ballGoalCommand({ duck: duck(), ball: ball(.1, side * .07), foot });
    const second = ballGoalCommand({ duck: duck(-.4, .5, Math.PI / 2), ball: ball(-.4 - side * .07, .6), foot });
    assert.equal(first.aligned, true); assert.equal(second.aligned, true);
    assert.deepEqual(first.command, [0, 0, 0]); assert.deepEqual(second.command, [0, 0, 0]);
    const relative = relativeBall(duck(-.4, .5, Math.PI / 2), ball(-.4 - side * .07, .6));
    assert.ok(Math.abs(relative.x - .1) < 1e-10); assert.ok(Math.abs(relative.y - side * .07) < 1e-10);
    assert.equal(ballGoalCommand({ duck: duck(), ball: ball(.1, -side * .07), foot }).aligned, false);
  }
});

test("missing or invalid observations cannot spawn a ball or issue motion", () => {
  for (const target of [undefined, { present: false, position: [.1, .07, .05] }, ball(NaN), ball(Infinity)]) {
    const decision = ballGoalCommand({ duck: duck(), ball: target });
    assert.equal(decision.blocked, true); assert.equal(decision.aligned, false); assert.deepEqual(decision.command, [0, 0, 0]);
  }
  for (const state of [duck(NaN), { ...duck(), paused: true }, { ...duck(), fallen: true }, { ...duck(), loco: "rollers" }]) {
    assert.deepEqual(ballGoalCommand({ duck: state, ball: ball(.7) }).command, [0, 0, 0]);
  }
});

test("settling uses the calibrated outer window without restarting for balance drift", () => {
  const input = { duck: duck(), ball: ball(.105, .065) };
  assert.equal(ballGoalCommand(input).aligned, false);
  assert.equal(ballGoalCommand({ ...input, settling: true }).aligned, true);
  for (const target of [ball(.112, .065), ball(.078, .065), ball(.095, .09)]) {
    assert.equal(ballGoalCommand({ ...input, ball: target, settling: true }).aligned, false);
  }
});

test("a distant or rear ball selects a physically supported motion profile", () => {
  for (const target of [ball(.7), ball(-.7), ball(.4, .5)]) {
    const decision = ballGoalCommand({ duck: duck(), ball: target });
    assert.equal(decision.blocked, false); assert.equal(decision.aligned, false);
    assert.ok(decision.command[0] === .25 || decision.command[0] === -.2);
    if (decision.command[0] < 0) assert.equal(decision.command[2], 1);
    assert.equal(sweptMotion(duck(), decision.command, {}, .8).allowed, true);
  }
});

test("box and peer admission remains active while the target ball is contactable", () => {
  const obstacle = obstacleForSlot("center"), state = duck(-.9), target = ball(.9);
  const detour = ballGoalCommand({ duck: state, ball: target, obstacle });
  assert.equal(detour.blocked, false); assert.equal(detour.aligned, false);
  assert.equal(sweptMotion(state, detour.command, { obstacle }, .8).allowed, true);
  assert.ok(Math.abs(detour.target[1]) > .4);
  const insideBox = ballGoalCommand({ duck: state, ball: ball(0), obstacle });
  assert.equal(insideBox.blocked, true); assert.deepEqual(insideBox.command, [0, 0, 0]);
  const peerBlocked = ballGoalCommand({ duck: state, ball: ball(.8), peers: [duck(.7)] });
  assert.equal(peerBlocked.blocked, true); assert.deepEqual(peerBlocked.command, [0, 0, 0]);
});

test("each observation invalidates stale alignment and newly occupied peer space", () => {
  const state = duck(), world = { peers: [duck(1, 1)] };
  assert.equal(ballGoalCommand({ duck: state, ball: ball(.095, .065), ...world }).aligned, true);
  const movedBall = ballGoalCommand({ duck: state, ball: ball(-.7), ...world, settling: true });
  assert.equal(movedBall.aligned, false); assert.equal(movedBall.blocked, false);
  const movingPeer = ballGoalCommand({ duck: state, ball: ball(.7), peers: [duck(.49)] });
  assert.equal(movingPeer.blocked, true); assert.deepEqual(movingPeer.command, [0, 0, 0]);
  for (const peers of [[duck(NaN)], [{ position: [1, 1, .12], headingRad: Infinity }]]) {
    const invalidPeer = ballGoalCommand({ duck: state, ball: ball(.7), peers });
    assert.equal(invalidPeer.blocked, true); assert.ok(invalidPeer.command.every(Number.isFinite));
  }
});

test("a staging position outside the safe arena returns an explicit blocked result", () => {
  for (const target of [ball(3), ball(0, -3)]) {
    const result = ballGoalCommand({ duck: duck(), ball: target });
    assert.equal(result.blocked, true); assert.equal(result.aligned, false);
    assert.deepEqual(result.command, [0, 0, 0]); assert.match(result.reason, /wall.*guarded approach/);
  }
});

test("near-wall alignment preserves the existing swept boundary and identifies inaccessible staging", () => {
  const heading = -Math.atan2(.085, .11), offset = Math.hypot(.11, .085);
  const inside = duck(1.291, .4, heading), outside = duck(1.293, .4, heading);
  // An already aligned pose may hold on the admitted side of the guard.
  // Two millimetres farther out must still be rejected, even when its
  // relative foot alignment is unchanged.
  const admitted = ballGoalCommand({ duck: inside, ball: ball(1.291 + offset, .4), settling: true });
  assert.equal(admitted.blocked, false); assert.equal(admitted.aligned, true);
  const rejected = ballGoalCommand({ duck: outside, ball: ball(1.293 + offset, .4), settling: true });
  assert.equal(rejected.blocked, true); assert.deepEqual(rejected.command, [0, 0, 0]);
  assert.match(rejected.reason, /wall/);
  const stopped = ballGoalCommand({ duck: duck(1.22, .47, 70.4 * Math.PI / 180), ball: ball(1.432, .668) });
  assert.equal(stopped.blocked, true); assert.match(stopped.reason, /wall.*guarded approach/);
  assert.deepEqual(stopped.command, [0, 0, 0]);
});

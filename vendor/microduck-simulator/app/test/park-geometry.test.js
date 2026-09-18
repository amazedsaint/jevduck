import test from "node:test";
import assert from "node:assert/strict";
import {
  ROBOT_RADIUS_M, PARK_OBSTACLE_HALF_SIZE, PARK_OBSTACLE_SLOTS,
  FOLLOW_MIN_SEPARATION_M, FOLLOW_DISTANCE_M, FOLLOW_WALK_SPEED_MPS, obstacleForSlot,
  canPlaceObstacle, parkSpatial, followCommand, sweptMotion,
} from "../src/game/park-geometry.js";
import { measureSpatial, AutonomyGuard } from "../src/game/spatial-guard.js";

const pose = (x, y = 0, headingRad = 0, posture = "standing") => ({ position: [x, y, 0.12], headingRad, posture });
const qpos = (x, y = 0, yaw = 0) => [x, y, 0.12, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
const close = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const held = (decision) => assert.deepEqual(decision.command, [0, 0, 0]);

test("slot records use the same ground positions and half sizes for rendering and physics", () => {
  for (const slot of Object.keys(PARK_OBSTACLE_SLOTS)) {
    const obstacle = obstacleForSlot(slot);
    assert.deepEqual(obstacle.halfSize, PARK_OBSTACLE_HALF_SIZE);
    assert.deepEqual(obstacle.position, PARK_OBSTACLE_SLOTS[slot]);
    assert.equal(obstacle.active, slot !== "off");
  }
  const editable = obstacleForSlot("center");
  editable.position[0] = 999;
  assert.equal(obstacleForSlot("center").position[0], 0);
  assert.equal(obstacleForSlot("unknown"), null);
});

test("swept body clearance sees an offset obstacle that a center ray would miss", () => {
  const spatial = parkSpatial(pose(-0.8, 0.3), { obstacle: obstacleForSlot("center") });
  close(spatial.clearance.front, 0.8 - 0.18 - ROBOT_RADIUS_M);
  assert.equal(spatial.clearanceSources.front, "obstacle");
  assert.equal(spatial.clearanceSources.back, "wall");
  const beyondFootprint = parkSpatial(pose(-0.8, 0.35), { obstacle: obstacleForSlot("center") });
  assert.equal(beyondFootprint.clearanceSources.front, "wall");
});

test("diagonal box intersections and rotated headings use ground geometry", () => {
  const spatial = measureSpatial(qpos(-0.8, -0.8, Math.PI / 4), { obstacle: obstacleForSlot("center") });
  close(spatial.clearance.front, (0.8 - 0.18 - ROBOT_RADIUS_M) * Math.SQRT2);
  assert.equal(spatial.clearanceSources.front, "obstacle");
  const left = measureSpatial(qpos(0, -0.8, Math.PI / 2), { obstacle: obstacleForSlot("center") });
  close(left.clearance.front, 0.8 - 0.18 - ROBOT_RADIUS_M);
});

test("peer clearance includes both bodies and remains independent of peer heading", () => {
  const a = parkSpatial(pose(-0.8), { peers: [pose(0, 0, Math.PI)] });
  const b = parkSpatial(pose(-0.8), { peers: [pose(0, 0, 0)] });
  close(a.clearance.front, 0.8 - 2 * ROBOT_RADIUS_M);
  close(a.clearance.front, b.clearance.front);
  assert.equal(a.clearanceSources.front, "duck");
  const offset = parkSpatial(pose(-0.8, 0.2), { peers: [pose(0)] });
  close(offset.clearance.front, 0.8 - Math.sqrt((2 * ROBOT_RADIUS_M) ** 2 - 0.2 ** 2));
});

test("nearest physical blocker wins while removing the box restores open clearance", () => {
  const input = pose(-1);
  const blocked = parkSpatial(input, { obstacle: obstacleForSlot("center"), peers: [pose(-0.5)] });
  assert.equal(blocked.clearanceSources.front, "duck");
  close(blocked.clearance.front, 0.18);
  const removed = parkSpatial(input, { obstacle: obstacleForSlot("off") });
  assert.equal(removed.clearanceSources.front, "wall");
  close(removed.clearance.front, 2.34);
});

test("invalid peer or obstacle observations fail closed without emitting NaN", () => {
  for (const world of [{ peers: [pose(NaN)] }, { peers: [pose(0, 4)] },
    { obstacle: { active: true, position: [0, 0, 0], halfSize: [0, 0.2, 0.2] } },
    { obstacle: { active: true, position: [0, 0, NaN], halfSize: [0.2, 0.2, 0.2] } }]) {
    const result = parkSpatial(pose(-0.6), world);
    assert.equal(result.spatialValid, false);
    assert.deepEqual(result.clearance, { front: 0, back: 0, left: 0, right: 0 });
  }
  assert.equal(measureSpatial(null).spatialValid, false);
});

test("placement rejects overlap with either duck and accepts clear slots", () => {
  const both = [pose(-0.6, 0), pose(-0.6, 0.7)];
  assert.equal(canPlaceObstacle("center", both).allowed, true);
  assert.equal(canPlaceObstacle("left", both).allowed, true);
  assert.equal(canPlaceObstacle("right", both).allowed, true);
  assert.equal(canPlaceObstacle("center", [pose(0), both[1]]).allowed, false);
  assert.equal(canPlaceObstacle("center", [both[0], pose(0.38, 0.3)]).allowed, false);
  assert.equal(canPlaceObstacle("left", [both[0], pose(0, 0.75)]).allowed, false);
  assert.equal(canPlaceObstacle("unknown", both).allowed, false);
  assert.equal(canPlaceObstacle("center", [both[0]]).allowed, false);
  assert.equal(canPlaceObstacle("center", [pose(NaN)]).allowed, false);
  assert.equal(canPlaceObstacle("off", [pose(NaN)]).allowed, true);
});

test("obstacle and peer guards cancel translation but leave independent turns available", () => {
  for (const [world, label] of [[{ obstacle: obstacleForSlot("center") }, "Obstacle"], [{ peers: [pose(0)] }, "Duck"]]) {
    let cancellations = 0;
    const guard = new AutonomyGuard({ getSpatial: () => parkSpatial(pose(-0.7), world), cancelMotion: () => { cancellations++; } });
    guard.setActive(true);
    assert.equal(guard.admit("walk_forward", 0.25), false);
    assert.match(guard.guardReason, new RegExp(`^${label} guard:`));
    assert.equal(guard.admit("turn_left", 0), true);
    assert.equal(guard.admit("turn_right", 0), true);
    assert.equal(cancellations, 1);
  }
});

test("following waits for seated or transitioning leaders and paused worlds", () => {
  const follower = pose(-1), leader = pose(0);
  for (const posture of ["sitting", "transitioning", "fallen"]) {
    held(followCommand({ follower, leader: { ...leader, posture } }));
  }
  held(followCommand({ follower, leader, paused: true }));
  held(followCommand({ follower: { ...follower, paused: true }, leader }));
  held(followCommand({ follower, leader: { ...leader, paused: true } }));
  const seated = followCommand({ follower: { ...follower, posture: "sitting" }, leader });
  held(seated);
  assert.equal(seated.needsStand, true);
});

test("invalid and fallen follower states cannot create motion", () => {
  for (const follower of [pose(NaN), pose(4), pose(-1, 0, Infinity), { ...pose(-1), fallen: true }]) {
    const result = followCommand({ follower, leader: pose(0) });
    held(result);
    assert.equal(result.blocked, true);
  }
  held(followCommand());
  held(followCommand({ follower: pose(-1), leader: pose(0), obstacle: {} }));
});

test("minimum spacing and arrival keep the companion from pressing into the leader", () => {
  const leader = pose(0.6);
  for (const gap of [0.3, FOLLOW_MIN_SEPARATION_M]) held(followCommand({ follower: pose(0.6 - gap), leader }));
  const arrived = followCommand({ follower: pose(0.6 - FOLLOW_DISTANCE_M), leader });
  held(arrived);
  assert.match(arrived.reason, /Target spacing reached/);
  const following = followCommand({ follower: pose(-0.8), leader });
  assert.equal(following.command[0], FOLLOW_WALK_SPEED_MPS);
  close(following.command[2], 0);
});

test("a duck facing away uses a safe native turning arc and side-by-side spawn is finite", () => {
  const away = followCommand({ follower: pose(-0.9, 0, Math.PI), leader: pose(0.4) });
  assert.ok(away.command[0] === -0.2 || away.command[0] === 0.25);
  assert.ok(Math.abs(away.command[2]) > 0);
  const spawned = followCommand({ follower: pose(-0.6, 0.7), leader: pose(-0.6), obstacle: obstacleForSlot("center") });
  assert.ok(spawned.command.every(Number.isFinite));
  assert.ok(spawned.target);
  assert.equal(sweptMotion(pose(-0.6, 0.7), spawned.command, { obstacle: obstacleForSlot("center"), peers: [pose(-0.6)] }).allowed, true);
});

test("short approaches use native profiles with room for the full swept motion", () => {
  const leader = pose(0.9);
  const approaching = followCommand({ follower: pose(0.05), leader });
  assert.ok(approaching.command[0] === 0.25 || approaching.command[0] === -0.2);
  assert.equal(sweptMotion(pose(0.05), approaching.command, { peers: [leader] }).allowed, true);
  const arrived = followCommand({ follower: pose(0.9 - FOLLOW_DISTANCE_M + 0.03), leader });
  held(arrived);
  const turning = followCommand({ follower: pose(0.05, 0, Math.PI / 2), leader });
  assert.ok(Math.abs(turning.command[2]) === 1);
  assert.notEqual(turning.command[0], 0);
});

test("a blocked straight route produces a side detour instead of walking into the box", () => {
  const follower = pose(-1.1), leader = pose(1.1), obstacle = obstacleForSlot("center");
  const result = followCommand({ follower, leader, obstacle });
  assert.equal(result.detouring, true);
  assert.ok(Math.abs(result.target[1]) > 0.4);
  assert.ok(result.target[0] < 0);
  assert.equal(sweptMotion(follower, result.command, { obstacle, peers: [leader] }).allowed, true);
  assert.ok(Math.abs(result.command[2]) > 0);
});

test("a trailing point inside the box selects a clear alternative without approaching the leader", () => {
  const result = followCommand({ follower: pose(-1.1), leader: pose(0.85), obstacle: obstacleForSlot("center") });
  assert.equal(result.detouring, true);
  assert.ok(result.target);
  assert.ok(result.command.every(Number.isFinite));
  assert.ok(Math.abs(result.target[1]) > 0.4);
});

test("an unavailable route returns a visible hold rather than a guessed velocity", () => {
  const wall = { active: true, position: [0, 0, 0.2], halfSize: [0.2, 1.4, 0.2] };
  const result = followCommand({ follower: pose(-1.1), leader: pose(1.1), obstacle: wall });
  held(result);
  assert.equal(result.blocked, true);
  assert.match(result.reason, /No admissible route/);
});

test("arc admission sees a box missed by the current front ray", () => {
  const current = pose(-0.8, 0.6), obstacle = obstacleForSlot("center");
  assert.ok(parkSpatial(current, { obstacle }).clearance.front > 1);
  assert.equal(sweptMotion(current, [0.25, 0, -1], { obstacle }).allowed, false);
  assert.equal(sweptMotion(current, [0.25, 0, 0], { obstacle }).allowed, true);
});

test("initial right arc is rejected for peer proximity while a reverse arc has room", () => {
  const current = pose(-0.6, 0.7), peers = [pose(-0.6)];
  assert.equal(sweptMotion(current, [0.25, 0, -1], { peers }).allowed, false);
  assert.equal(sweptMotion(current, [-0.2, 0, -1], { peers }).allowed, true);
});

test("a reverse arc can move away from a wall that blocks the forward turning envelope", () => {
  const current = pose(1.15);
  assert.equal(sweptMotion(current, [0.25, 0, 1]).allowed, false);
  assert.equal(sweptMotion(current, [-0.2, 0, 1]).allowed, true);
});

test("invalid and unbounded swept motions fail closed", () => {
  for (const command of [[NaN, 0, 1], [0.25, 1, 0], [0.25, 0, Infinity], [0.25, 0, 2]]) {
    assert.equal(sweptMotion(pose(0), command).allowed, false);
  }
  for (const duration of [0, -1, 3, NaN]) assert.equal(sweptMotion(pose(0), [0.25, 0, 0], {}, duration).allowed, false);
  assert.equal(sweptMotion(pose(0), [0.25, 0, 0], { peers: [pose(NaN)] }).allowed, false);
});

test("roller follower paths are checked at the actual roller turn rate", () => {
  const follower = { ...pose(-0.6, 0.7), loco: "rollers" }, leader = pose(-0.6);
  const result = followCommand({ follower, leader });
  assert.ok(Math.abs(result.command[2]) <= 0.3);
  if (result.command.some(value => value !== 0)) assert.equal(sweptMotion(follower, result.command, { peers: [leader] }).allowed, true);
});

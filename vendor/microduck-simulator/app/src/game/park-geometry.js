import { ARENA_HALF } from "./constants.js";

// All positions use MuJoCo coordinates, with +Z up. The same slot record
// positions the rendered box and its physical mocap body.
export const ROBOT_RADIUS_M = 0.16;
export const PARK_OBSTACLE_HALF_SIZE = Object.freeze([0.18, 0.18, 0.16]);
export const PARK_OBSTACLE_SLOTS = Object.freeze({
  center: Object.freeze([0, 0, 0.16]),
  left: Object.freeze([0, 0.75, 0.16]),
  right: Object.freeze([0, -0.75, 0.16]),
  off: Object.freeze([50, 50, 0.16]),
});
export const FOLLOW_MIN_SEPARATION_M = 0.52;
export const FOLLOW_DISTANCE_M = 0.68;
export const FOLLOW_WALK_SPEED_MPS = 0.25;
const FOLLOW_ROUTE_SEPARATION_M = FOLLOW_MIN_SEPARATION_M;
const ROUTE_MARGIN_M = 0.06;
const ARRIVE_M = 0.09;
const EPS = 1e-9;
const zeroClearance = () => ({ front: 0, back: 0, left: 0, right: 0 });
const finitePoint = (point, dimensions = 2) => point?.length >= dimensions
  && Array.from({ length: dimensions }, (_, i) => point[i]).every(Number.isFinite);
const validPose = (pose) => finitePoint(pose?.position, 3) && Number.isFinite(pose.headingRad)
  && Math.abs(pose.position[0]) <= ARENA_HALF && Math.abs(pose.position[1]) <= ARENA_HALF;
const validObstacle = (obstacle) => !obstacle || obstacle.active === false
  || (finitePoint(obstacle.position, 3) && finitePoint(obstacle.halfSize, 3)
    && obstacle.halfSize.every((n) => n > 0));
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const angleError = (a, b) => Math.atan2(Math.sin(a - b), Math.cos(a - b));

export function obstacleForSlot(slot) {
  if (!Object.hasOwn(PARK_OBSTACLE_SLOTS, slot)) return null;
  return {
    slot, active: slot !== "off",
    position: [...PARK_OBSTACLE_SLOTS[slot]], halfSize: [...PARK_OBSTACLE_HALF_SIZE],
  };
}

function boxOf(obstacle, margin) {
  if (!obstacle || obstacle.active === false) return null;
  return {
    min: [obstacle.position[0] - obstacle.halfSize[0] - margin,
      obstacle.position[1] - obstacle.halfSize[1] - margin],
    max: [obstacle.position[0] + obstacle.halfSize[0] + margin,
      obstacle.position[1] + obstacle.halfSize[1] + margin],
  };
}

// Distance along a unit ray to an expanded box, including tangent contact.
// Expanding by body radius gives a conservative swept-body test.
function rayBox(origin, direction, box) {
  let near = -Infinity, far = Infinity;
  for (let axis = 0; axis < 2; axis++) {
    if (Math.abs(direction[axis]) < EPS) {
      if (origin[axis] < box.min[axis] || origin[axis] > box.max[axis]) return Infinity;
      continue;
    }
    const a = (box.min[axis] - origin[axis]) / direction[axis];
    const b = (box.max[axis] - origin[axis]) / direction[axis];
    near = Math.max(near, Math.min(a, b));
    far = Math.min(far, Math.max(a, b));
  }
  if (near > far + EPS || far < -EPS) return Infinity;
  return Math.max(0, near);
}

function rayCircle(origin, direction, center, radius) {
  const ox = origin[0] - center[0], oy = origin[1] - center[1];
  const c = ox * ox + oy * oy - radius * radius;
  if (c <= 0) return 0;
  const projection = ox * direction[0] + oy * direction[1];
  const discriminant = projection * projection - c;
  if (discriminant < -EPS) return Infinity;
  const entry = -projection - Math.sqrt(Math.max(0, discriminant));
  return entry < -EPS ? Infinity : Math.max(0, entry);
}

function wallRay(origin, direction, radius) {
  const limit = ARENA_HALF - radius;
  const distances = direction.map((d, i) => Math.abs(d) < EPS
    ? (Math.abs(origin[i]) > limit ? 0 : Infinity)
    : ((d > 0 ? limit : -limit) - origin[i]) / d);
  return Math.max(0, Math.min(...distances));
}

export function parkSpatial(pose, { obstacle = null, peers = [] } = {}) {
  if (!validPose(pose) || !validObstacle(obstacle) || !Array.isArray(peers) || !peers.every(validPose)) {
    return { headingRad: Number.isFinite(pose?.headingRad) ? pose.headingRad : 0,
      clearance: zeroClearance(), spatialValid: false };
  }
  const box = boxOf(obstacle, ROBOT_RADIUS_M);
  const clearance = {}, clearanceSources = {};
  for (const [side, offset] of Object.entries({ front: 0, back: Math.PI, left: Math.PI / 2, right: -Math.PI / 2 })) {
    const angle = pose.headingRad + offset;
    const direction = [Math.cos(angle), Math.sin(angle)];
    let distance = wallRay(pose.position, direction, ROBOT_RADIUS_M), source = "wall";
    if (box) {
      const obstacleDistance = rayBox(pose.position, direction, box);
      if (obstacleDistance < distance) { distance = obstacleDistance; source = "obstacle"; }
    }
    for (const peer of peers) {
      const peerDistance = rayCircle(pose.position, direction, peer.position, 2 * ROBOT_RADIUS_M);
      if (peerDistance < distance) { distance = peerDistance; source = "duck"; }
    }
    clearance[side] = distance;
    clearanceSources[side] = source;
  }
  return { headingRad: pose.headingRad, clearance, spatialValid: true, clearanceSources };
}

// Moving a static prop is allowed only while its conservative footprint
// remains clear of both robots. Removing it is always available.
export function canPlaceObstacle(slot, duckPoses) {
  const obstacle = obstacleForSlot(slot);
  if (!obstacle) return { allowed: false, reason: "Unknown obstacle position.", obstacle: null };
  if (!obstacle.active) return { allowed: true, reason: null, obstacle };
  if (!Array.isArray(duckPoses) || duckPoses.length < 2 || !duckPoses.every(validPose)) {
    return { allowed: false, reason: "Wait for a valid position from both ducks.", obstacle };
  }
  const box = boxOf(obstacle, ROBOT_RADIUS_M + ROUTE_MARGIN_M);
  const overlaps = duckPoses.some(({ position: p }) => p[0] >= box.min[0] && p[0] <= box.max[0]
    && p[1] >= box.min[1] && p[1] <= box.max[1]);
  return { allowed: !overlaps, reason: overlaps ? "A duck is too close to that obstacle position." : null, obstacle };
}

function arcPose(pose, command, seconds, velocityScale = 1, yawScale = 1) {
  const velocity = command[0] * velocityScale, yawRate = command[2] * yawScale;
  const headingRad = pose.headingRad + yawRate * seconds;
  const [x, y, z] = pose.position;
  const position = Math.abs(yawRate) < EPS
    ? [x + velocity * seconds * Math.cos(pose.headingRad), y + velocity * seconds * Math.sin(pose.headingRad), z]
    : [x + velocity / yawRate * (Math.sin(headingRad) - Math.sin(pose.headingRad)),
      y + velocity / yawRate * (Math.cos(pose.headingRad) - Math.cos(headingRad)), z];
  return { position, headingRad };
}

function pointRoom(position, obstacle, peers) {
  const radius = ROBOT_RADIUS_M + 0.04;
  let clearanceM = ARENA_HALF - radius - Math.max(Math.abs(position[0]), Math.abs(position[1]));
  let source = "wall";
  if (obstacle?.active) {
    // Signed distance to the box, inflated by the body's disk radius.
    const dx = Math.abs(position[0] - obstacle.position[0]) - obstacle.halfSize[0];
    const dy = Math.abs(position[1] - obstacle.position[1]) - obstacle.halfSize[1];
    const clearance = Math.hypot(Math.max(0, dx), Math.max(0, dy)) + Math.min(0, Math.max(dx, dy)) - radius;
    if (clearance < clearanceM) { clearanceM = clearance; source = "obstacle"; }
  }
  for (const peer of peers) {
    const clearance = dist(position, peer.position) - FOLLOW_MIN_SEPARATION_M;
    if (clearance < clearanceM) { clearanceM = clearance; source = "duck"; }
  }
  return { clearanceM, source };
}

// Admission and live monitoring use the complete translating arc, not a
// front ray. A bounded tracking envelope covers slower measured turning
// and modest velocity error; every subsequent physics cycle checks again.
export function sweptMotion(pose, command, { obstacle = null, peers = [] } = {}, duration = 2) {
  if (!validPose(pose) || !finitePoint(command, 3) || command[1] !== 0 || Math.abs(command[0]) > 0.6
    || Math.abs(command[2]) > 1 || !Number.isFinite(duration) || duration <= 0 || duration > 2
    || !validObstacle(obstacle) || !Array.isArray(peers) || !peers.every(validPose)) {
    return { allowed: false, reason: "Valid positions and a bounded motion are required.", clearanceM: 0, endpoint: null, headingRad: 0 };
  }
  const endpoint = arcPose(pose, command, duration);
  let minimum = { clearanceM: Infinity, source: "wall" };
  const steps = Math.ceil(duration / 0.04);
  const velocities = command[0] === 0 ? [1] : [0.65, 1, 1.2];
  const yawRates = command[2] === 0 ? [1] : [0.6, 1, 1.3];
  for (const velocityScale of velocities) for (const yawScale of yawRates) {
    for (let i = 0; i <= steps; i++) {
      const point = arcPose(pose, command, duration * i / steps, velocityScale, yawScale);
      const room = pointRoom(point.position, obstacle, peers);
      if (room.clearanceM < minimum.clearanceM) minimum = room;
    }
  }
  // Adjacent samples are at most 1.44 cm apart at the largest admitted
  // speed. A further 8 mm guard covers the unsampled interval between them.
  const allowed = minimum.clearanceM >= 0.008;
  return { allowed, clearanceM: minimum.clearanceM, endpoint: endpoint.position.slice(0, 2), headingRad: endpoint.headingRad,
    reason: allowed ? null : `The motion needs more room from the ${minimum.source}.` };
}

function pointClear(point, box, peer) {
  const limit = ARENA_HALF - ROBOT_RADIUS_M - ROUTE_MARGIN_M;
  if (!finitePoint(point) || Math.abs(point[0]) > limit || Math.abs(point[1]) > limit) return false;
  if (box && point[0] >= box.min[0] && point[0] <= box.max[0]
    && point[1] >= box.min[1] && point[1] <= box.max[1]) return false;
  return dist(point, peer) >= FOLLOW_ROUTE_SEPARATION_M;
}

function segmentClear(from, to, box, peer) {
  if (!pointClear(from, box, peer) || !pointClear(to, box, peer)) return false;
  const length = dist(from, to);
  if (length < EPS) return true;
  const direction = [(to[0] - from[0]) / length, (to[1] - from[1]) / length];
  if (box && rayBox(from, direction, box) <= length + EPS) return false;
  return rayCircle(from, direction, peer, FOLLOW_ROUTE_SEPARATION_M) > length + EPS;
}

// Tiny visibility graph for one box and one moving duck. It chooses a
// collision-free ground path; the trained policy still supplies every step.
function routeTo(start, goal, box, peer) {
  if (!pointClear(start, box, peer) || !pointClear(goal, box, peer)) return null;
  if (segmentClear(start, goal, box, peer)) return [start, goal];
  const candidates = [];
  if (box) {
    for (const x of [box.min[0] - 0.035, box.max[0] + 0.035]) {
      for (const y of [box.min[1] - 0.035, box.max[1] + 0.035]) candidates.push([x, y]);
    }
  }
  // Chords between adjacent octagon vertices stay outside the peer route
  // margin, leaving room for native walk pulses.
  for (let i = 0; i < 8; i++) {
    const angle = i * Math.PI / 4;
    candidates.push([peer[0] + 0.72 * Math.cos(angle), peer[1] + 0.72 * Math.sin(angle)]);
  }
  const nodes = [start, goal, ...candidates.filter((p) => pointClear(p, box, peer))];
  const costs = nodes.map(() => Infinity), previous = nodes.map(() => -1), visited = new Set();
  costs[0] = 0;
  while (visited.size < nodes.length) {
    let current = -1;
    for (let i = 0; i < nodes.length; i++) {
      if (!visited.has(i) && (current < 0 || costs[i] < costs[current])) current = i;
    }
    if (current < 0 || !Number.isFinite(costs[current])) return null;
    if (current === 1) {
      const path = [];
      for (let index = 1; index !== -1; index = previous[index]) path.unshift(nodes[index]);
      return path;
    }
    visited.add(current);
    for (let next = 0; next < nodes.length; next++) {
      if (visited.has(next) || !segmentClear(nodes[current], nodes[next], box, peer)) continue;
      const cost = costs[current] + dist(nodes[current], nodes[next]);
      if (cost < costs[next] - EPS) { costs[next] = cost; previous[next] = current; }
    }
  }
  return null;
}

const hold = (reason, blocked = false, needsStand = false) => ({
  command: [0, 0, 0], reason, target: null, blocked, detouring: false, needsStand,
});

export function followCommand({ follower, leader, obstacle = null, paused = false } = {}) {
  if (!validPose(follower) || !validPose(leader) || !validObstacle(obstacle)) return hold("Follow control blocked: invalid robot state.", true);
  if (paused || follower.paused || leader.paused) return hold("Follow control paused.");
  if (follower.fallen || leader.fallen || follower.posture === "fallen" || leader.posture === "fallen") {
    return hold("Follow control blocked: robot recovery in progress.", true);
  }
  if (leader.posture !== "standing") return hold("Waiting for a standing leader.");
  if (follower.posture === "sitting") return hold("Follower stand transition required.", false, true);
  if (follower.posture !== "standing") return hold("Waiting for a standing follower.");
  const start = follower.position.slice(0, 2), peer = leader.position.slice(0, 2);
  const separation = dist(start, peer);
  if (separation <= FOLLOW_MIN_SEPARATION_M) return hold("Minimum separation reached.", true);
  // A companion has a comfortable region behind its leader, not an exact
  // mathematical parking point. Requiring an exact point caused repeated
  // arc corrections after the real policy had already reached the leader.
  const behind = -((start[0] - peer[0]) * Math.cos(leader.headingRad)
    + (start[1] - peer[1]) * Math.sin(leader.headingRad)) / separation;
  const alreadyWaiting = follower.command?.every(value => Math.abs(value) < 0.01);
  if (behind >= 0.5 && separation <= (alreadyWaiting ? 0.82 : 0.72)) return hold("Target spacing reached.");
  const box = boxOf(obstacle, ROBOT_RADIUS_M + ROUTE_MARGIN_M);
  const back = leader.headingRad + Math.PI;
  // A trailing point can be inside the obstacle even when both ducks are
  // clear. Nearby alternatives stay behind or beside the leader.
  const goals = [[back, FOLLOW_DISTANCE_M], [back - Math.PI / 4, FOLLOW_DISTANCE_M],
    [back + Math.PI / 4, FOLLOW_DISTANCE_M], [back, 0.95],
    [back - Math.PI / 2, FOLLOW_DISTANCE_M], [back + Math.PI / 2, FOLLOW_DISTANCE_M]];
  let route = null, alternate = false;
  for (const [index, [angle, radius]] of goals.entries()) {
    // Leave a small interior inset so millimeter-scale balancing drift at
    // a wall does not keep switching the selected trailing destination.
    const limit = ARENA_HALF - ROBOT_RADIUS_M - ROUTE_MARGIN_M - 0.025;
    const goal = [peer[0] + radius * Math.cos(angle), peer[1] + radius * Math.sin(angle)]
      .map(value => Math.max(-limit, Math.min(limit, value)));
    route = routeTo(start, goal, box, peer);
    if (route) { alternate = index !== 0; break; }
  }
  if (!route) return hold("No admissible route to the leader.", true);
  const target = route[1], remaining = dist(start, target);
  if (remaining < ARRIVE_M && route.length === 2) return hold("Target spacing reached.");
  // Real policy probes falsified in-place turns and straight reversing:
  // they only lean. Forward walking and turning walking arcs do move.
  // Reverse arcs provide room to turn near a wall without inventing a gait.
  const turnRate = follower.loco === "rollers" ? 0.3 : 1;
  const candidates = [[FOLLOW_WALK_SPEED_MPS, 0, 0],
    [FOLLOW_WALK_SPEED_MPS, 0, turnRate], [FOLLOW_WALK_SPEED_MPS, 0, -turnRate],
    [-0.2, 0, turnRate], [-0.2, 0, -turnRate]];
  let best = null;
  for (const command of candidates) {
    const sweep = sweptMotion(follower, command, { obstacle, peers: [leader] });
    if (!sweep.allowed) continue;
    const expected = arcPose(follower, command, 1, 0.8, 0.75);
    // Use the bearing before the candidate step. Looking back at a nearby
    // corner from the predicted endpoint wrongly rewarded reversing each
    // time a forward step would pass that intermediate waypoint.
    const desiredHeading = Math.atan2(target[1] - start[1], target[0] - start[0]);
    const score = dist(expected.position, target) + 0.30 * Math.abs(angleError(desiredHeading, expected.headingRad))
      + (command[0] < 0 ? 0.10 : 0) + (command[2] === 0 ? 0 : 0.01)
      + (follower.command && command.some((value, index) => value !== follower.command[index]) ? 0.05 : 0);
    if (!best || score < best.score - EPS) best = { command, score, sweep };
  }
  if (!best) return hold("No admissible motion profile.", true);
  const detouring = alternate || route.length > 2;
  return {
    command: best.command, target, blocked: false,
    detouring, needsStand: false,
    reason: best.command[0] < 0 ? "Reverse arc selected."
      : detouring ? "Detour selected." : best.command[2] !== 0 ? "Forward turning arc selected." : "Tracking leader position.",
  };
}

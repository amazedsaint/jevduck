import { ARENA_HALF } from "./constants.js";
import { ROBOT_RADIUS_M, FOLLOW_MIN_SEPARATION_M, sweptMotion } from "./park-geometry.js";

// The native kick contact grid supports this outer window for either foot.
// Arrival uses an inset so normal balance drift does not restart locomotion.
export const BALL_KICK_TARGET = Object.freeze({ x: .095, y: .065 });
export const BALL_KICK_WINDOW = Object.freeze({ minX: .08, maxX: .11, lateralTolerance: .02 });
const MARGIN = ROBOT_RADIUS_M + .06;
const EPS = 1e-9;
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const finitePoint = point => point?.length >= 3 && [point[0], point[1], point[2]].every(Number.isFinite);
const angle = value => Math.atan2(Math.sin(value), Math.cos(value));
const hold = (reason, blocked = true, aligned = false) => ({ command: [0, 0, 0], phase: "settling", aligned, blocked, reason, target: null });

export function relativeBall(duck, ball) {
  const dx = ball.position[0] - duck.position[0], dy = ball.position[1] - duck.position[1];
  const c = Math.cos(duck.headingRad), s = Math.sin(duck.headingRad);
  return { x: c * dx + s * dy, y: -s * dx + c * dy };
}

function lineClear(a, b, obstacle, peers) {
  const limit = ARENA_HALF - MARGIN;
  if ([...a, ...b].some(value => !Number.isFinite(value) || Math.abs(value) >= limit)) return false;
  const dx = b[0] - a[0], dy = b[1] - a[1], length2 = dx * dx + dy * dy;
  for (const peer of peers) {
    const t = length2 < EPS ? 0 : Math.max(0, Math.min(1, ((peer.position[0] - a[0]) * dx + (peer.position[1] - a[1]) * dy) / length2));
    if (distance([a[0] + t * dx, a[1] + t * dy], peer.position) < FOLLOW_MIN_SEPARATION_M + .015) return false;
  }
  if (obstacle?.active) {
    let near = 0, far = 1;
    for (let axis = 0; axis < 2; axis++) {
      const low = obstacle.position[axis] - obstacle.halfSize[axis] - MARGIN;
      const high = obstacle.position[axis] + obstacle.halfSize[axis] + MARGIN;
      const d = b[axis] - a[axis];
      if (Math.abs(d) < EPS) { if (a[axis] < low || a[axis] > high) return true; }
      else { const t1 = (low - a[axis]) / d, t2 = (high - a[axis]) / d; near = Math.max(near, Math.min(t1, t2)); far = Math.min(far, Math.max(t1, t2)); }
    }
    if (near <= far + EPS) return false;
  }
  return true;
}

function routeTo(start, goal, obstacle, peers) {
  if (lineClear(start, goal, obstacle, peers)) return [start, goal];
  const corners = [];
  if (obstacle?.active) for (const sx of [-1, 1]) for (const sy of [-1, 1]) corners.push([
    obstacle.position[0] + sx * (obstacle.halfSize[0] + MARGIN + .055),
    obstacle.position[1] + sy * (obstacle.halfSize[1] + MARGIN + .055),
  ]);
  for (const peer of peers) for (let i = 0; i < 8; i++) corners.push([
    peer.position[0] + .66 * Math.cos(i * Math.PI / 4), peer.position[1] + .66 * Math.sin(i * Math.PI / 4),
  ]);
  const nodes = [start, goal, ...corners], costs = nodes.map(() => Infinity), previous = nodes.map(() => -1), visited = new Set();
  costs[0] = 0;
  while (visited.size < nodes.length) {
    let current = -1;
    for (let i = 0; i < nodes.length; i++) if (!visited.has(i) && (current < 0 || costs[i] < costs[current])) current = i;
    if (current < 0 || !Number.isFinite(costs[current])) return null;
    if (current === 1) { const path = []; for (let i = 1; i !== -1; i = previous[i]) path.unshift(nodes[i]); return path; }
    visited.add(current);
    for (let next = 0; next < nodes.length; next++) {
      if (visited.has(next) || !lineClear(nodes[current], nodes[next], obstacle, peers)) continue;
      const cost = costs[current] + distance(nodes[current], nodes[next]);
      if (cost < costs[next]) { costs[next] = cost; previous[next] = current; }
    }
  }
  return null;
}

function predict(duck, command, seconds) {
  // Approximate measured response only ranks candidates. Swept admission
  // independently covers a wider tracking envelope, and physics stays native.
  const v = command[0] * .8, w = command[2] * .75, h = duck.headingRad + w * seconds;
  const [x, y, z] = duck.position;
  return { headingRad: h, position: Math.abs(w) < EPS
    ? [x + v * seconds * Math.cos(h), y + v * seconds * Math.sin(h), z]
    : [x + v / w * (Math.sin(h) - Math.sin(duck.headingRad)), y + v / w * (Math.cos(duck.headingRad) - Math.cos(h)), z] };
}

// The ball is a contact target, so it is deliberately absent from body
// obstacle inflation. This helper never spawns or relocates scene objects.
export function ballGoalCommand({ duck, ball, foot = "left", obstacle = null, peers = [], settling = false } = {}) {
  if (!finitePoint(duck?.position) || !Number.isFinite(duck.headingRad) || !finitePoint(ball?.position)
    || ball.present === false || !["left", "right"].includes(foot)) return hold("Ball pursuit requires a measured robot and ball position.");
  if (duck.paused || duck.fallen || duck.posture === "fallen") return hold("Ball pursuit is interrupted.");
  if (duck.posture && duck.posture !== "standing") return hold("Ball pursuit is waiting for a standing robot.");
  if (duck.loco && duck.loco !== "legs") return hold("Ball pursuit requires the leg policy.");
  const admission = sweptMotion(duck, [0, 0, 0], { obstacle, peers }, .2);
  if (!admission.allowed) return hold(admission.reason);
  const relative = relativeBall(duck, ball), targetY = foot === "left" ? BALL_KICK_TARGET.y : -BALL_KICK_TARGET.y;
  const inset = settling ? 0 : .008;
  if (relative.x >= BALL_KICK_WINDOW.minX + inset && relative.x <= BALL_KICK_WINDOW.maxX - inset
    && Math.abs(relative.y - targetY) <= BALL_KICK_WINDOW.lateralTolerance - inset) return hold("Ball is within the selected foot alignment window.", false, true);
  const start = duck.position.slice(0, 2), radius = distance(start, ball.position);
  // Aim beside the ball so the selected ankle, rather than the trunk
  // centerline, reaches the contact window at the end of the approach.
  const desired = Math.atan2(ball.position[1] - start[1], ball.position[0] - start[0])
    - Math.asin(Math.max(-.95, Math.min(.95, targetY / Math.max(radius, .08))));
  const goal = [ball.position[0] - BALL_KICK_TARGET.x * Math.cos(desired) + targetY * Math.sin(desired),
    ball.position[1] - BALL_KICK_TARGET.x * Math.sin(desired) - targetY * Math.cos(desired)];
  const route = routeTo(start, goal, obstacle, peers);
  if (!route) return hold(goal.some(value => Math.abs(value) >= ARENA_HALF - MARGIN)
    ? "Ball is too close to the wall for the current guarded approach."
    : "No admissible route to the ball alignment position.");
  const detour = route.length > 2, target = route[1];
  const desiredHeading = detour ? Math.atan2(target[1] - start[1], target[0] - start[0]) : desired;
  const horizon = detour || radius > .4 ? .75 : .3;
  // Native stand-start probes falsified negative-yaw reversing: it only
  // leans. Rank the proven positive reverse arc and both forward arcs.
  const candidates = [[.25, 0, 0], [.25, 0, 1], [.25, 0, -1], [-.2, 0, 1]];
  let best = null;
  for (const command of candidates) {
    const sweep = sweptMotion(duck, command, { obstacle, peers }, .8);
    if (!sweep.allowed) continue;
    const predicted = predict(duck, command, horizon);
    const score = distance(predicted.position, target) + .22 * Math.abs(angle(desiredHeading - predicted.headingRad))
      + (command[0] < 0 ? .035 : 0)
      + (duck.command && command.some((value, index) => value !== duck.command[index]) ? .008 : 0);
    if (!best || score < best.score - EPS) best = { score, command };
  }
  if (!best) return hold("No admissible walking arc toward the ball.");
  return { command: best.command, phase: radius > .35 || detour ? "approaching" : "aligning", aligned: false, blocked: false, target,
    reason: detour ? "Following a clear route around the obstruction." : "Approaching the selected foot alignment position." };
}

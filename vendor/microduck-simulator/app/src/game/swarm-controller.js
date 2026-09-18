import { sweptMotion } from "./park-geometry.js";

export const SWARM_SCENARIOS = Object.freeze(["flock", "gather", "convoy", "split"]);
export const SWARM_INTENTS = Object.freeze(["advance", "regroup", "disperse", "change_leader", "split", "hold"]);
export const SWARM_WINDOW_S = 8;
const IDS = ["duck1", "duck2", "duck3", "duck4"];
const SIGNS = [[-1, -1], [1, -1], [-1, 1], [1, 1]];
const ZERO = () => [0, 0, 0];
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const angle = value => Math.atan2(Math.sin(value), Math.cos(value));
const validPose = pose => IDS.includes(pose?.id) && pose.position?.length >= 3
  && pose.position.slice(0, 3).every(Number.isFinite) && Number.isFinite(pose.headingRad);
const validGroup = poses => Array.isArray(poses) && poses.length === 4 && poses.every(validPose)
  && new Set(poses.map(pose => pose.id)).size === 4;

// Used only by an explicit scene reset. The controller never changes qpos.
export function scenarioSpawns(scenario = "flock") {
  if (!SWARM_SCENARIOS.includes(scenario)) throw new Error("Unknown swarm scenario.");
  return IDS.map((id, index) => {
    const [sx, sy] = SIGNS[index];
    if (scenario === "convoy") return { id, position: [-.99 + .66 * index, 0, .12], headingRad: 0 };
    if (scenario === "gather") return { id, position: [sx * .65, sy * .65, .12], headingRad: Math.atan2(-sy, -sx) };
    if (scenario === "split") return { id, position: [sx * .4, sy * .4, .12], headingRad: Math.atan2(-sy * .02, sx * .5) };
    return { id, position: [sx < 0 ? -.75 : .15, sy * .45, .12], headingRad: 0 };
  });
}
function metrics(poses) {
  const centroid = [0, 0];
  for (const pose of poses) { centroid[0] += pose.position[0] / poses.length; centroid[1] += pose.position[1] / poses.length; }
  let minSeparationM = Infinity;
  for (let i = 0; i < poses.length; i++) for (let j = i + 1; j < poses.length; j++) minSeparationM = Math.min(minSeparationM, distance(poses[i].position, poses[j].position));
  const spreadM = Math.sqrt(poses.reduce((sum, pose) => sum + distance(pose.position, centroid) ** 2, 0) / poses.length);
  return { centroid, spreadM, minSeparationM: Number.isFinite(minSeparationM) ? minSeparationM : 0 };
}
function predict(pose, command, duration = .65) {
  const v = command[0] * .8, w = command[2] * .75, h = pose.headingRad + w * duration;
  const [x, y] = pose.position;
  return { headingRad: h, position: w === 0
    ? [x + v * duration * Math.cos(h), y + v * duration * Math.sin(h)]
    : [x + v / w * (Math.sin(h) - Math.sin(pose.headingRad)), y + v / w * (Math.cos(pose.headingRad) - Math.cos(h))] };
}
function steer(pose, target, peers, obstacle) {
  const desired = Math.atan2(target[1] - pose.position[1], target[0] - pose.position[0]);
  let best = null;
  // In-place turns and negative-yaw reverse are excluded because the
  // original policy can stall on those inputs from a standing posture.
  for (const command of [[.25, 0, 0], [.25, 0, 1], [.25, 0, -1], [-.2, 0, 1]]) {
    const swept = sweptMotion(pose, command, { obstacle, peers }, .4);
    if (!swept.allowed) continue;
    const expected = predict(pose, command, distance(pose.position, target) < .35 ? .35 : .65);
    // Assigned slots constrain position, not final body orientation. Score
    // reversing by its travel direction so nearby rear targets do not
    // unnecessarily force a full forward circle.
    const travelHeading = expected.headingRad + (command[0] < 0 ? Math.PI : 0);
    const score = distance(expected.position, target) + .12 * Math.abs(angle(desired - travelHeading))
      + (command[0] < 0 ? .025 : 0)
      + (pose.command && command.some((value, axis) => value !== pose.command[axis]) ? .008 : 0);
    if (!best || score < best.score) best = { command, score };
  }
  return best?.command ?? null;
}

export class SwarmController {
  active = false;
  scenario = "flock";
  runId = null;
  commandId = null;
  intent = "hold";
  phase = "idle";
  leaderId = "duck1";
  elapsedS = 0;
  reason = "Prepare a four-robot scenario to begin.";
  commands = new Map(IDS.map(id => [id, ZERO()]));
  targets = new Map();
  arrived = new Set();
  startPositions = new Map();
  startCentroid = null;
  blockedFor = 0;
  commandFor(id) { return this.commands.get(id) ?? ZERO(); }
  clearCommands() { for (const id of IDS) this.commands.set(id, ZERO()); }
  startScenario(scenario, runId, poses) {
    if (!SWARM_SCENARIOS.includes(scenario) || !validGroup(poses)) return { accepted: false, message: "Four measured robot poses and a known scenario are required." };
    this.clearCommands(); this.targets.clear(); this.arrived.clear();
    Object.assign(this, { active: true, scenario, runId, commandId: null, intent: "hold", phase: "idle", elapsedS: 0,
      leaderId: scenario === "convoy" ? "duck4" : "duck1", reason: "Scenario prepared. Waiting for a bounded group instruction.", startCentroid: metrics(poses).centroid });
    return { accepted: true, message: this.reason };
  }
  startIntent(id, intent, poses) {
    if (!this.active || this.phase === "running" && intent !== "hold" || !SWARM_INTENTS.includes(intent) || !validGroup(poses)) return { accepted: false, message: "The group is unavailable for that instruction." };
    this.clearCommands(); this.targets.clear(); this.arrived.clear(); this.blockedFor = 0;
    this.commandId = id; this.intent = intent; this.elapsedS = 0; this.startCentroid = metrics(poses).centroid;
    this.startPositions = new Map(poses.map(pose => [pose.id, pose.position.slice(0, 2)]));
    if (intent === "hold" || intent === "change_leader") {
      if (intent === "change_leader") this.leaderId = IDS[(IDS.indexOf(this.leaderId) + 1) % IDS.length];
      this.phase = "complete"; this.reason = intent === "hold" ? "Group motion inputs held at zero." : `Group leader changed to ${this.leaderId}.`;
      return { accepted: true, message: this.reason };
    }
    if (intent === "advance") {
      const leader = poses.find(pose => pose.id === this.leaderId);
      const step = this.scenario === "convoy" ? .18 : .28;
      const delta = [step * Math.cos(leader.headingRad), step * Math.sin(leader.headingRad)];
      for (const pose of poses) this.targets.set(pose.id, [pose.position[0] + delta[0], pose.position[1] + delta[1]]);
    } else {
      for (const [index, duckId] of IDS.entries()) {
        const [sx, sy] = SIGNS[index], radius = intent === "regroup" ? .55 / Math.SQRT2 : .9 / Math.SQRT2;
        this.targets.set(duckId, intent === "split" ? [sx * .9, sy * .38] : [sx * radius, sy * radius]);
      }
    }
    this.phase = "running"; this.reason = "Executing a bounded instruction through the original walking policies.";
    return { accepted: true, message: this.reason };
  }
  stop(reason = "Group control stopped.") { this.clearCommands(); this.active = false; this.phase = "idle"; this.reason = reason; }
  tick(dt, poses, { obstacle = null } = {}) {
    this.clearCommands();
    if (!this.active || this.phase !== "running") return;
    if (!Number.isFinite(dt) || dt <= 0 || !validGroup(poses)) { this.phase = "blocked"; this.reason = "Four finite robot observations are required."; return; }
    if (poses.some(pose => pose.fallen || pose.posture === "fallen" || pose.paused || pose.manual || pose.loco !== "legs")) {
      this.phase = "blocked"; this.reason = "Group control interrupted by posture, pause or manual control."; return;
    }
    this.elapsedS += dt;
    let moving = 0;
    for (const pose of poses) {
      const target = this.targets.get(pose.id);
      const error = distance(pose.position, target);
      if (error <= (this.arrived.has(pose.id) ? .105 : .075)) { this.arrived.add(pose.id); continue; }
      this.arrived.delete(pose.id);
      if (pose.busy || pose.posture !== "standing") continue;
      const command = steer(pose, target, poses.filter(peer => peer.id !== pose.id), obstacle);
      if (command) { this.commands.set(pose.id, command); moving++; }
    }
    if (this.arrived.size === 4) {
      this.phase = "complete"; this.reason = "All assigned positions reached within the arrival tolerance."; this.clearCommands(); return;
    }
    this.blockedFor = moving === 0 ? this.blockedFor + dt : 0;
    if (this.blockedFor >= 1) { this.phase = "blocked"; this.reason = "No admitted motion toward the remaining assigned positions."; this.clearCommands(); return; }
    if (this.elapsedS >= SWARM_WINDOW_S) {
      const travel = poses.reduce((sum, pose) => sum + distance(pose.position, this.startPositions.get(pose.id)), 0);
      this.phase = travel >= .05 ? "complete" : "blocked";
      this.reason = travel >= .05 ? "Eight-second control window ended; remaining target error is measured below." : "The control window ended without measurable group progress.";
      this.clearCommands();
    }
  }
  status(poses = []) {
    const valid = validGroup(poses), measured = valid ? metrics(poses) : { centroid: [0, 0], spreadM: 0, minSeparationM: 0 };
    const available = valid && poses.every(pose => !pose.fallen && !pose.paused && !pose.busy && !pose.manual && pose.posture === "standing" && pose.loco === "legs");
    const targetErrorM = valid && this.targets.size === 4 ? Math.max(...poses.map(pose => distance(pose.position, this.targets.get(pose.id)))) : null;
    return { active: this.active, runId: this.runId, scenario: this.scenario, leaderId: this.leaderId, commandId: this.commandId,
      intent: this.intent, phase: this.phase, availableIntents: this.active && available && this.phase !== "running" ? [...SWARM_INTENTS] : ["hold"],
      ...measured, progressM: valid && this.startCentroid ? distance(measured.centroid, this.startCentroid) : 0,
      targetErrorM, reason: this.reason, elapsedS: this.elapsedS, members: valid ? poses.length : 0 };
  }
}

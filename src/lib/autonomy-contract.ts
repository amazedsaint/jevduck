import { z } from "zod";
import { SIMULATOR_ACTION_CATALOG, SIMULATOR_DUCK_IDS, SIMULATOR_EXECUTABLE_ACTIONS, type SimulatorExecutableAction } from "./simulator";

export const AUTONOMY_MODES = ["explore", "observe", "play"] as const;
export type AutonomyMode = typeof AUTONOMY_MODES[number];
const aliasedActions = ["walk_forward", "walk_backward", "sit", "stand"] as const;
type AliasedAction = typeof aliasedActions[number];
type AtomicBehavior = Exclude<SimulatorExecutableAction, AliasedAction>;
export type AutonomyBehavior = AtomicBehavior | "stroll" | "back_up" | "look_around" | "rest" | "wake_up" | "wait";
const atomicActions = SIMULATOR_EXECUTABLE_ACTIONS.filter((action): action is AtomicBehavior => !aliasedActions.includes(action as AliasedAction));
export const AUTONOMY_PLANS: Readonly<Record<AutonomyBehavior, readonly SimulatorExecutableAction[]>> = {
  ...Object.fromEntries(atomicActions.map(action => [action, [action]])) as Record<AtomicBehavior, SimulatorExecutableAction[]>,
  stroll: ["walk_forward"], back_up: ["walk_backward"],
  look_around: ["look_left", "look_right", "center_head"],
  rest: ["sit"], wake_up: ["stand"], wait: [],
};
export const AUTONOMY_BEHAVIORS = Object.keys(AUTONOMY_PLANS) as [AutonomyBehavior, ...AutonomyBehavior[]];
export const AUTONOMY_LABELS: Readonly<Record<AutonomyBehavior, string>> = {
  ...Object.fromEntries(atomicActions.map(action => [action, SIMULATOR_ACTION_CATALOG[action].label])) as Record<AtomicBehavior, string>,
  stroll: "Forward motion", back_up: "Reverse arc", look_around: "Head scan",
  rest: "Sit", wake_up: "Stand", wait: "Hold",
};
export const AUTONOMY_CLEARANCE_M = 0.5;
// Full two-second trained-policy pulse plus the runtime's braking margin.
export const AUTONOMY_FORWARD_CLEARANCE_M = 0.7;
export const AUTONOMY_BACK_CLEARANCE_M = 0.6;
export const AUTONOMY_CELL_SIZE_M = 0.25;
export const AUTONOMY_MAX_RECENT = 8;
export const AUTONOMY_MAX_VISITED = 64;

const finite = z.number().finite();
const distance = finite.min(0).max(1000);
const bearing = finite.min(-Math.PI).max(Math.PI);
const count = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER);
/** Runtime-owned task evidence. Primitive pose completion is not ball-task success. */
export const simulatorTaskSchema = z.object({
  commandId: z.string().min(1).max(160),
  action: z.enum(["approach_ball", "kick_ball"]),
  phase: z.enum(["standing", "searching", "approaching", "aligning", "settling", "kicking", "verifying", "complete", "failed", "cancelled"]),
  outcome: z.enum(["succeeded", "failed", "cancelled"]).nullable(),
  reason: z.string().min(1).max(500), elapsedS: finite.min(0).max(3600),
  ballContact: z.boolean(), ballDisplacementM: distance,
}).strict().superRefine((task, ctx) => {
  const expected = task.phase === "complete" ? "succeeded" : task.phase === "failed" ? "failed" : task.phase === "cancelled" ? "cancelled" : null;
  if (task.outcome !== expected) ctx.addIssue({ code: "custom", message: "Task phase and outcome must agree." });
  if (task.action === "kick_ball" && task.outcome === "succeeded" && (!task.ballContact || task.ballDisplacementM < 0.05)) {
    ctx.addIssue({ code: "custom", message: "Kick success requires physical contact and measured ball displacement of at least 0.05 m." });
  }
});
export type SimulatorTask = z.infer<typeof simulatorTaskSchema>;
export const autonomyStateSchema = z.object({
  ready: z.boolean(), busy: z.boolean(), paused: z.boolean(), fallen: z.boolean(),
  loco: z.enum(["legs", "rollers"]), mode: z.string().regex(/^[a-zA-Z0-9_-]{1,32}$/),
  seq: count, time: finite.min(0).max(1e9),
  position: z.tuple([finite.min(-1000).max(1000), finite.min(-1000).max(1000), finite.min(-1000).max(1000)]),
  headingRad: finite.min(-1e6).max(1e6),
  posture: z.enum(["standing", "sitting", "transitioning", "fallen"]),
  clearance: z.object({ front: distance, back: distance, left: distance, right: distance }).strict(),
  spatialValid: z.boolean(), guardReason: z.string().min(1).max(160).nullable(),
  guardSeq: count, autonomyActive: z.boolean(),
  followEnabled: z.boolean().optional(),
  selectedDuckId: z.enum(SIMULATOR_DUCK_IDS).optional(),
  availableActions: z.array(z.enum(SIMULATOR_EXECUTABLE_ACTIONS)).max(SIMULATOR_EXECUTABLE_ACTIONS.length).refine(actions => new Set(actions).size === actions.length).optional(),
  ball: z.object({ present: z.boolean(), distanceM: distance, bearingRad: bearing }).strict().nullable().optional(),
  task: simulatorTaskSchema.nullable().optional(),
  companion: z.object({ id: z.enum(SIMULATOR_DUCK_IDS), distanceM: distance, bearingRad: bearing,
    posture: z.enum(["standing", "sitting", "transitioning", "fallen"]), moving: z.boolean() }).strict().nullable().optional(),
  clearanceSources: z.object({ front: z.enum(["wall", "obstacle", "duck"]), back: z.enum(["wall", "obstacle", "duck"]),
    left: z.enum(["wall", "obstacle", "duck"]), right: z.enum(["wall", "obstacle", "duck"]) }).strict().optional(),
}).strict();
export const autonomyEpisodeSchema = z.object({
  behavior: z.enum(AUTONOMY_BEHAVIORS),
  outcome: z.enum(["completed", "blocked", "interrupted", "failed"]),
  distanceM: distance,
  ballDistanceBeforeM: distance.optional(), ballDistanceAfterM: distance.optional(),
  ballContact: z.boolean().optional(), ballDisplacementM: distance.optional(),
  taskOutcome: z.enum(["succeeded", "failed", "cancelled"]).optional(),
}).strict();
export const autonomyCellSchema = z.object({
  x: z.number().int().min(-4000).max(4000), y: z.number().int().min(-4000).max(4000),
  visits: z.number().int().min(1).max(1e6),
}).strict();
export const autonomyInputSchema = z.object({
  mode: z.enum(AUTONOMY_MODES), state: autonomyStateSchema,
  memory: z.object({ recent: z.array(autonomyEpisodeSchema).max(AUTONOMY_MAX_RECENT), visited: z.array(autonomyCellSchema).max(AUTONOMY_MAX_VISITED) }).strict(),
}).strict();
export type AutonomyState = z.infer<typeof autonomyStateSchema>;
export type AutonomyEpisode = z.infer<typeof autonomyEpisodeSchema>;
export type AutonomyCell = z.infer<typeof autonomyCellSchema>;
export type AutonomyInput = z.infer<typeof autonomyInputSchema>;

const probability = finite.min(0).max(1);
export const autonomyDecisionSchema = z.object({
  behavior: z.enum(AUTONOMY_BEHAVIORS),
  plan: z.array(z.enum(SIMULATOR_EXECUTABLE_ACTIONS)).max(3),
  label: z.string().min(1).max(80), reason: z.string().min(1).max(500),
  source: z.literal("jev"), model: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/),
  confidence: probability, latencyMs: finite.min(0).max(60_000),
  alternatives: z.array(z.object({ behavior: z.enum(AUTONOMY_BEHAVIORS), probability }).strict()).min(1).max(AUTONOMY_BEHAVIORS.length),
}).strict().superRefine((decision, ctx) => {
  if (JSON.stringify(decision.plan) !== JSON.stringify(AUTONOMY_PLANS[decision.behavior])) {
    ctx.addIssue({ code: "custom", message: "The plan must match its application-owned behavior." });
  }
});
export type AutonomyDecision = z.infer<typeof autonomyDecisionSchema>;

/** Local freshness is checked by the browser controller, not client wall clocks. */
export function autonomyBlockReason(state: AutonomyState): string | null {
  if (!autonomyStateSchema.safeParse(state).success) return "The simulator observation is invalid.";
  if (!state.ready) return "The simulator is not ready.";
  if (!state.autonomyActive) return "Autonomy is not active in the simulator.";
  if (state.paused) return "The simulator is paused.";
  if (state.fallen || state.posture === "fallen") return "The robot needs to recover first.";
  if (state.busy || state.posture === "transitioning") return "The current movement must finish first.";
  if (!state.spatialValid) return "The simulator has no valid spatial observation.";
  return null;
}

/** Only an explicitly enabled follower creates a pacing obligation. */
export function autonomyCompanionWaitReason(state: AutonomyState): "fallen" | "sitting" | "transitioning" | "catching_up" | null {
  if (!state.followEnabled || !state.companion) return null;
  if (state.companion.posture !== "standing") return state.companion.posture;
  return state.companion.distanceM > 1.25 ? "catching_up" : null;
}

/** Legacy observations cannot silently opt into newly introduced robot abilities. */
const LEGACY_ACTIONS: readonly SimulatorExecutableAction[] = ["walk_forward", "walk_backward", "turn_left", "turn_right", "look_left", "look_right", "center_head", "sit", "stand"];
const LEGACY_BEHAVIORS = new Set<AutonomyBehavior>(["stroll", "back_up", "turn_left", "turn_right", "look_around", "rest", "wake_up", "wait"]);
const TRANSLATION = new Set<AutonomyBehavior>(["stroll", "back_up"]);
const BODY_MOTION = new Set<AutonomyBehavior>(["stroll", "back_up", "turn_left", "turn_right", "roll", "kick_left", "kick_right", "ground_pick", "crouch", "approach_ball", "kick_ball"]);
const PLAY_ACTIONS = new Set<AutonomyBehavior>(["roll", "kick_left", "kick_right", "crouch", "switch_to_rollers", "switch_to_legs", "spawn_ball", "approach_ball", "kick_ball"]);

/** Play delegates the full ball objective to one feedback task. Raw gestures remain
 * available to explicit text requests; they cannot displace the active objective.
 */
function playCandidates(state: AutonomyState, memory: AutonomyInput["memory"], available: Set<SimulatorExecutableAction>): Set<AutonomyBehavior> {
  const candidates = new Set<AutonomyBehavior>(["wait", "stop"]);
  if (!state.ball?.present) return candidates;
  if (autonomyCompanionWaitReason(state)) {
    if (state.posture === "sitting") candidates.add("wake_up");
    return candidates;
  }
  // A failed bounded task needs a changed scene or a new explicit request. Waiting
  // must not erase its failure and repeatedly launch the same unsuccessful task.
  if (state.task?.outcome === "failed" || state.task?.outcome === "cancelled") return candidates;
  const previousGoal = memory.recent.findLast(episode => episode.behavior === "kick_ball" || episode.behavior === "approach_ball");
  // Explicit null is the runtime's receipt of a new session/reset. Undefined is
  // a legacy observation, where retained failure history remains the only gate.
  if (state.task === undefined && previousGoal && previousGoal.outcome !== "completed") return candidates;
  if (state.loco === "rollers") {
    candidates.add("switch_to_legs");
    return candidates;
  }
  // Pause once after a verified interaction before starting another interaction.
  if (memory.recent.at(-1)?.behavior === "kick_ball" && previousGoal?.outcome === "completed") return candidates;
  if (available.has("kick_ball")) candidates.add("kick_ball");
  else if (available.has("approach_ball") && (state.task === null || previousGoal?.behavior !== "approach_ball")) candidates.add("approach_ball");
  return candidates;
}

/** Available actions only. Jev chooses among these; code never invents a choice. */
export function eligibleAutonomyBehaviors(input: AutonomyInput): AutonomyBehavior[] {
  if (autonomyBlockReason(input.state)) return [];
  const { state, mode, memory } = input;
  const available = new Set(state.availableActions ?? LEGACY_ACTIONS);
  const play = mode === "play" ? playCandidates(state, memory, available) : null;
  const last = memory.recent.at(-1);
  const canUseLocomotion = state.mode === "walk" || state.loco === "rollers" ||
    (state.mode === "sitstand" && state.posture === "sitting");
  // Stable ordering keeps legacy traces readable and avoids positional API assumptions.
  const ordered: AutonomyBehavior[] = ["wait", "look_around", "wake_up", "rest", "stroll", "back_up", "turn_left", "turn_right",
    ...AUTONOMY_BEHAVIORS.filter(behavior => !LEGACY_BEHAVIORS.has(behavior))];
  return ordered.filter(behavior => {
    if (behavior === "wait") return true;
    if (play && !play.has(behavior)) return false;
    if (!state.availableActions && !LEGACY_BEHAVIORS.has(behavior)) return false;
    const plan = AUTONOMY_PLANS[behavior];
    if (!plan.every(action => available.has(action))) return false;
    if (plan.some(action => SIMULATOR_ACTION_CATALOG[action].locomotion !== "both" && SIMULATOR_ACTION_CATALOG[action].locomotion !== state.loco)) return false;
    if (autonomyCompanionWaitReason(state) && (BODY_MOTION.has(behavior) || PLAY_ACTIONS.has(behavior))) return false;
    if (mode === "observe" && (BODY_MOTION.has(behavior) || PLAY_ACTIONS.has(behavior))) return false;
    if (mode !== "play" && PLAY_ACTIONS.has(behavior)) return false;
    if (behavior === "rest" && (state.posture !== "standing" || last?.behavior === "wake_up")) return false;
    if (behavior === "wake_up" && (state.posture !== "sitting" || (mode === "observe" && last?.behavior === "rest"))) return false;
    if (["stroll", "back_up", "turn_left", "turn_right"].includes(behavior) && !canUseLocomotion) return false;
    if (behavior === "stroll" && state.clearance.front < AUTONOMY_FORWARD_CLEARANCE_M) return false;
    if (behavior === "back_up" && state.clearance.back < AUTONOMY_BACK_CLEARANCE_M) return false;
    if (behavior === "roll" && (state.posture !== "standing" || Math.min(...Object.values(state.clearance)) < 0.55)) return false;
    if (behavior === "ground_pick" && (state.posture !== "standing" || state.clearance.front < 0.35)) return false;
    if (behavior === "crouch" && (state.posture !== "standing" || state.clearance.front < 0.85)) return false;
    if ((behavior === "switch_to_legs" || behavior === "switch_to_rollers") && state.posture !== "standing") return false;
    // Never repeatedly retry a failed gesture or alternate passive poses forever.
    const clearedBallTask = state.task === null && (behavior === "kick_ball" || behavior === "approach_ball");
    if (last?.behavior === behavior && !clearedBallTask) {
      if (last.outcome !== "completed") return false;
      if (TRANSLATION.has(behavior)) return last.distanceM >= 0.03;
      return behavior === "turn_left" || behavior === "turn_right";
    }
    return true;
  });
}

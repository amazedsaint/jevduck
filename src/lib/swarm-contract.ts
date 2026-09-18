import { z } from "zod";
import { SIMULATOR_DUCK_IDS } from "./simulator";

export const SWARM_SCENARIOS = ["flock", "gather", "convoy", "split"] as const;
export const SWARM_INTENTS = ["advance", "regroup", "disperse", "change_leader", "split", "hold"] as const;
export const SWARM_SCENARIO_LABELS = { flock: "Formation advance", gather: "Gather / disperse", convoy: "Leader convoy", split: "Split / regroup" } as const;
export const SWARM_INTENT_LABELS = { advance: "Advance formation", regroup: "Regroup", disperse: "Increase spacing", change_leader: "Change leader", split: "Split formation", hold: "Hold" } as const;
export type SwarmScenario = typeof SWARM_SCENARIOS[number];
export type SwarmIntent = typeof SWARM_INTENTS[number];
export const SWARM_MAX_RECENT = 6;
export const SWARM_MIN_CONFIDENCE = 0.55;
export const SWARM_MAX_RECOVERY_ATTEMPTS = 2;
export const SWARM_SMALL_SLOT_ERROR_M = 0.15;
export const SWARM_MIN_ADVANCE_M = 0.05;
export const SWARM_MIN_SLOT_IMPROVEMENT_M = 0.03;

const finite = z.number().finite();
const distance = finite.min(0).max(1000);
const identifier = z.string().regex(/^[a-zA-Z0-9:_-]{1,160}$/);
const probability = finite.min(0).max(1);
export const swarmRuntimeSchema = z.object({
  active: z.boolean(), runId: identifier.nullable(), scenario: z.enum(SWARM_SCENARIOS).nullable(),
  leaderId: z.enum(SIMULATOR_DUCK_IDS),
  commandId: identifier.nullable(), intent: z.enum(SWARM_INTENTS).nullable(),
  phase: z.enum(["idle", "running", "complete", "blocked"]),
  availableIntents: z.array(z.enum(SWARM_INTENTS)).max(SWARM_INTENTS.length).refine(items => new Set(items).size === items.length),
  centroid: z.tuple([finite.min(-1000).max(1000), finite.min(-1000).max(1000)]),
  /** RMS planar distance from centroid; minimum separation is between body centers. */
  spreadM: distance, minSeparationM: distance,
  /** Net planar centroid displacement during this command, not formation convergence. */
  progressM: distance,
  /** Maximum planar error to assigned distinct slots, or null without targets. */
  targetErrorM: distance.nullable(), elapsedS: finite.min(0).max(3600),
  members: z.number().int().min(2).max(4), reason: z.string().max(500),
}).strict().superRefine((value, ctx) => {
  if (value.active && (!value.runId || !value.scenario || value.members !== 4)) {
    ctx.addIssue({ code: "custom", message: "An active swarm requires a scenario, run ID and four physical members." });
  }
  if (value.phase === "running" && (!value.active || !value.commandId || !value.intent)) {
    ctx.addIssue({ code: "custom", message: "A running group intent must identify its active command." });
  }
});
export type SwarmRuntime = z.infer<typeof swarmRuntimeSchema>;
export const swarmStateSchema = z.object({
  seq: z.number().int().min(0).max(Number.MAX_SAFE_INTEGER), time: finite.min(0).max(1e9),
  ready: z.boolean(), paused: z.boolean(), swarm: swarmRuntimeSchema,
}).strict();
export type SwarmState = z.infer<typeof swarmStateSchema>;
export const swarmEpisodeSchema = z.object({
  intent: z.enum(SWARM_INTENTS), outcome: z.enum(["complete", "blocked"]),
  progressM: distance, targetErrorBeforeM: distance.nullable(), targetErrorAfterM: distance.nullable(), minSeparationM: distance,
}).strict();
export type SwarmEpisode = z.infer<typeof swarmEpisodeSchema>;
export const swarmInputSchema = z.object({
  state: swarmStateSchema, memory: z.object({ recent: z.array(swarmEpisodeSchema).max(SWARM_MAX_RECENT) }).strict(),
}).strict();
export type SwarmInput = z.infer<typeof swarmInputSchema>;
export const swarmDecisionSchema = z.object({
  intent: z.enum(SWARM_INTENTS), source: z.literal("jev"),
  model: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/), confidence: probability,
  abstained: z.boolean(), reason: z.string().min(1).max(500), latencyMs: finite.min(0).max(60_000),
  alternatives: z.array(z.object({ intent: z.enum(SWARM_INTENTS), probability }).strict()).min(1).max(SWARM_INTENTS.length),
}).strict().superRefine((value, ctx) => {
  if (value.abstained && value.intent !== "hold") ctx.addIssue({ code: "custom", message: "Abstention cannot command group movement." });
  if (!value.abstained && value.confidence < SWARM_MIN_CONFIDENCE) ctx.addIssue({ code: "custom", message: "A low decision score must abstain." });
  if (new Set(value.alternatives.map(item => item.intent)).size !== value.alternatives.length) ctx.addIssue({ code: "custom", message: "Decision alternatives must be unique." });
});
export type SwarmDecision = z.infer<typeof swarmDecisionSchema>;

export const SWARM_SCENARIO_INTENTS: Record<SwarmScenario, readonly SwarmIntent[]> = {
  flock: ["advance", "regroup", "disperse", "hold"],
  gather: ["regroup", "disperse", "hold"],
  convoy: ["advance", "regroup", "change_leader", "hold"],
  split: ["split", "regroup", "hold"],
};
/** Holds do not change assigned targets or provide evidence that a failed move
 * became feasible. Keep the latest physical instruction even for legacy input
 * whose short history has been filled with local observations. */
export function measuredSwarmHistory(input: SwarmInput): SwarmEpisode[] {
  const recent = input.memory.recent.filter(item => item.intent !== "hold");
  const { swarm } = input.state;
  const last = recent.at(-1);
  if (swarm.commandId && swarm.intent && swarm.intent !== "hold" && (swarm.phase === "complete" || swarm.phase === "blocked") &&
      (last?.intent !== swarm.intent || last.outcome !== swarm.phase)) {
    recent.push({ intent: swarm.intent, outcome: swarm.phase, progressM: swarm.progressM,
      targetErrorBeforeM: null, targetErrorAfterM: swarm.targetErrorM, minSeparationM: swarm.minSeparationM });
  }
  return recent;
}
export function swarmSlotImprovement(episode: SwarmEpisode): number | null {
  return episode.targetErrorBeforeM === null || episode.targetErrorAfterM === null ? null : episode.targetErrorBeforeM - episode.targetErrorAfterM;
}
export function ineffectiveSwarmEpisode(episode: SwarmEpisode): boolean {
  if (episode.outcome === "blocked") return true;
  if (episode.targetErrorAfterM === null || episode.targetErrorAfterM <= SWARM_SMALL_SLOT_ERROR_M) return false;
  if (episode.intent === "advance") return episode.progressM < SWARM_MIN_ADVANCE_M;
  const improvement = swarmSlotImprovement(episode);
  return episode.intent !== "hold" && episode.intent !== "change_leader" && improvement !== null && improvement < SWARM_MIN_SLOT_IMPROVEMENT_M;
}
function measuredMovement(episode: SwarmEpisode): boolean {
  if (episode.outcome !== "complete") return false;
  if (episode.intent === "advance") return episode.progressM >= SWARM_MIN_ADVANCE_M;
  // Centroid translation cannot establish improvement toward formation slots.
  return episode.intent !== "hold" && episode.intent !== "change_leader" &&
    (swarmSlotImprovement(episode) ?? -Infinity) >= SWARM_MIN_SLOT_IMPROVEMENT_M;
}
export function swarmRecoveryState(input: SwarmInput) {
  const history = measuredSwarmHistory(input);
  const lastProductiveAdvance = history.findLastIndex(item => item.intent === "advance" && measuredMovement(item) && !ineffectiveSwarmEpisode(item));
  const sinceAdvance = history.slice(lastProductiveAdvance + 1);
  const failure = sinceAdvance.findIndex(item => item.intent === "advance" && ineffectiveSwarmEpisode(item));
  const recovery = failure < 0 ? [] : sinceAdvance.slice(failure + 1);
  const attempts = recovery.filter(item => item.intent !== "advance");
  const lastFailure = recovery.findLastIndex(item => item.intent === "advance" && ineffectiveSwarmEpisode(item));
  const changedSinceFailure = recovery.slice(lastFailure + 1).some(item => item.outcome === "complete" &&
    (item.intent === "change_leader" || measuredMovement(item)));
  return { active: failure >= 0, attempts: attempts.length, maximumAttempts: SWARM_MAX_RECOVERY_ATTEMPTS,
    attemptedIntents: [...new Set(attempts.map(item => item.intent))], retryAdvanceAfterMeasuredChange: changedSinceFailure,
    exhausted: failure >= 0 && attempts.length >= SWARM_MAX_RECOVERY_ATTEMPTS && !changedSinceFailure };
}
export function swarmBlockReason(state: SwarmState): string | null {
  if (!swarmStateSchema.safeParse(state).success) return "Invalid swarm observation.";
  if (!state.ready) return "The simulator is not ready.";
  if (state.paused) return "Physics is paused.";
  if (!state.swarm.active || !state.swarm.scenario || state.swarm.members !== 4) return "The four-robot scenario is not active.";
  if (state.swarm.phase === "running") return "The current group instruction must finish first.";
  return null;
}
export function eligibleSwarmIntents(input: SwarmInput): SwarmIntent[] {
  if (swarmBlockReason(input.state)) return [];
  const { swarm } = input.state;
  const history = measuredSwarmHistory(input);
  const last = history.at(-1);
  const recovery = swarmRecoveryState(input);
  return SWARM_SCENARIO_INTENTS[swarm.scenario!].filter(intent => {
    if (intent === "hold") return true;
    if (!swarm.availableIntents.includes(intent)) return false;
    if (recovery.active) {
      if (intent === "advance") return recovery.retryAdvanceAfterMeasuredChange;
      if (recovery.attempts >= SWARM_MAX_RECOVERY_ATTEMPTS || recovery.attemptedIntents.includes(intent)) return false;
    }
    const failed = history.findLastIndex(item => item.intent === intent && ineffectiveSwarmEpisode(item));
    if (failed >= 0 && !history.slice(failed + 1).some(measuredMovement)) return false;
    if (intent === "change_leader" && last?.intent === "change_leader") return false;
    // A formation with small residual has completed this stage. Reissuing its
    // identical slots after each local hold cannot advance an alternating task.
    if (intent !== "advance" && intent !== "change_leader" && last?.intent === intent &&
        last.outcome === "complete" && last.targetErrorAfterM !== null && last.targetErrorAfterM <= SWARM_SMALL_SLOT_ERROR_M) return false;
    return true;
  });
}

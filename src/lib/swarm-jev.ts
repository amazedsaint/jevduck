import { z } from "zod";
import {
  SWARM_INTENTS, SWARM_MIN_CONFIDENCE, SWARM_INTENT_LABELS, eligibleSwarmIntents,
  measuredSwarmHistory, ineffectiveSwarmEpisode, swarmSlotImprovement, swarmRecoveryState,
  SWARM_SMALL_SLOT_ERROR_M, SWARM_MIN_ADVANCE_M, SWARM_MIN_SLOT_IMPROVEMENT_M,
  swarmBlockReason, swarmDecisionSchema, swarmInputSchema,
  type SwarmDecision, type SwarmInput, type SwarmIntent, type SwarmScenario,
} from "./swarm-contract";

export { swarmInputSchema } from "./swarm-contract";
export const SWARM_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
export const SWARM_JEV_TIMEOUT_MS = 8_000;
const modelName = z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/);
const probability = z.number().finite().min(0).max(1);
const criteria: Record<SwarmIntent, string> = {
  advance: "Start or continue a short group translation in flock or convoy. This is the preferred FIRST movement for a freshly prepared flock or convoy when the runtime admits it. Initial progress is zero because no movement has run yet; no previous successful advance or existing target is required. The native controller assigns bounded forward targets when this intention starts. After movement has run, use measured progress and spacing to decide whether to continue. Do not repeatedly advance after a blocked or ineffective attempt.",
  regroup: "Move members to distinct compact slots near the arena center. This is the preferred recovery after a flock advance stalls or is blocked, including when spread remains ordinary: returning toward central slots can restore room for later translation. It does not require fragmentation. In gather mode this starts the compact stage. In split mode this follows a split whose slot error is small. Only reduction in assigned-slot error proves useful formation progress.",
  disperse: "Move members to distinct wider slots to increase separation when crowded, or after a successful compact formation in gather mode. In flock use it to relieve poor spacing rather than spreading an already fragmented group.",
  change_leader: "In convoy, transfer leadership to the runtime's next eligible robot after an advance is blocked or ineffective. This is an admitted recovery alternative; a different leader may have a different heading, but changing identity alone does not prove a feasible path. Once changed, one admissible advance can test the new leader. Do not cycle leaders without productive translation.",
  split: "In split mode, move the robots into separate assigned subgroups when the compact group is ready. After a completed split with low remaining slot error, regroup before another split. This creates distinct local targets, not additional robots.",
  hold: "Issue no new movement while measurements are unsettled or no productive admissible intention fits. Hold after a newly completed formation for a brief observation if useful. Prefer a productive admissible intention when the current scenario has not progressed. No movement can be invented when the runtime withdraws capabilities.",
};
const objectives: Record<SwarmScenario, string> = {
  flock: "Start the prepared flock with an admissible advance. Its initial two-by-two layout has approximately 0.64 m RMS spread and 0.90 m minimum separation; this is the intended starting formation. Targets are assigned only when an instruction begins, so null initial slot error is not a failure. Continue advance when it produces measured group translation. After an ineffective or blocked advance, prefer admitted regroup toward central slots even if spread looks ordinary. Disperse is another recovery only when increasing spacing fits the measured state. A recovery must change the assigned-slot error before advance can retry. At most two recovery instructions are allowed without productive advance; if exhausted or no useful choice remains, hold.",
  gather: "Alternate compact and wider formations using distinct slots. Start by regrouping if there is no completed compact formation. Once regrouping reached low slot error, disperse; once the wider formation reached low slot error, regroup. Continue an incomplete useful instruction when it still makes progress, but do not repeat a blocked instruction.",
  convoy: "Start the prepared convoy with an admissible advance under its current leader. Zero initial progress and null initial target error mean no instruction has run yet; they do not call for a leader change. After movement begins, continue while progress is useful. Regroup when formation error grows; change leader after blocked or repeatedly ineffective progress, then test the new leader. The arena is finite, so do not promise an endless straight convoy.",
  split: "Alternate a compact formation with two distinct subgroups. Start with split if no previous split exists. Once its assigned slots are reached, regroup. After regrouping, split again. Remaining target error distinguishes actual positioning from a finished time window.",
};

export class SwarmJevError extends Error {
  constructor(public readonly status: number, message: string) { super(message); this.name = "SwarmJevError"; }
}
export function swarmJevConfiguration() {
  const proposed = process.env.TYPESAFE_MODEL?.trim() || "jev-latest";
  return { available: Boolean(process.env.TYPESAFE_API_KEY?.trim()), model: modelName.safeParse(proposed).success ? proposed : "jev-latest" };
}
export function buildSwarmJevRequest(input: SwarmInput, model: string) {
  const { swarm } = input.state;
  const history = measuredSwarmHistory(input);
  const latest = history.at(-1);
  const recovery = swarmRecoveryState(input);
  return {
    model,
    state: {
      scenario: swarm.scenario, members: swarm.members, leader: swarm.leaderId,
      centroidMetres: swarm.centroid, rmsSpreadMetres: swarm.spreadM, minimumCenterSeparationMetres: swarm.minSeparationM,
      centroidDisplacementMetres: swarm.progressM, maximumSlotErrorMetres: swarm.targetErrorM,
      preparedSceneAwaitingFirstInstruction: swarm.commandId === null && swarm.phase === "idle",
      formationTargetsAssigned: swarm.targetErrorM !== null,
      previousIntent: swarm.commandId === null ? null : swarm.intent, previousPhase: swarm.phase, elapsedSimulationSeconds: swarm.elapsedS,
      availableIntents: eligibleSwarmIntents(input), recentPhysicalInstructions: history,
      latestPhysicalInstruction: latest ?? null,
      latestInstructionIneffective: latest ? ineffectiveSwarmEpisode(latest) : null,
      latestFormationImprovementMetres: latest ? swarmSlotImprovement(latest) : null,
      completedFormationStage: latest?.outcome === "complete" && latest.intent !== "advance" && latest.intent !== "change_leader" &&
        latest.targetErrorAfterM !== null && latest.targetErrorAfterM <= SWARM_SMALL_SLOT_ERROR_M ? latest.intent : null,
      recovery,
      decisionThresholds: { smallSlotErrorMetres: SWARM_SMALL_SLOT_ERROR_M, minimumUsefulAdvanceMetres: SWARM_MIN_ADVANCE_M, minimumFormationImprovementMetres: SWARM_MIN_SLOT_IMPROVEMENT_M },
      slotErrorIsSmall: swarm.targetErrorM === null ? null : swarm.targetErrorM <= SWARM_SMALL_SLOT_ERROR_M,
    },
    questions: { intent: {
      type: "choice" as const,
      instructions: {
        question: "Which ONE eligible group intention should the four-robot simulator execute next?",
        objective: objectives[swarm.scenario!],
        evidence: "All positions and distances are measured from MuJoCo state. A prepared scene has settled standing robots and no group instruction yet. Local holds do not change that fact or the latest physical instruction. Null target error and null slotErrorIsSmall mean targets have not been assigned, not that a target was missed. Spread is RMS distance from the centroid. Separation is minimum body-center distance. Slot error is the maximum member-to-target distance. Net centroid displacement measures group translation, not formation convergence. A complete phase only means the bounded window ended. High residual with little advance is ineffective even when complete; consult recovery and retained physical instructions. For regroup/disperse/split use before/after slot errors. CompletedFormationStage identifies the stage whose residual is now small: in split mode completed regroup calls for split, and completed split calls for regroup; in gather mode completed regroup calls for disperse, and completed disperse calls for regroup. Repeated local holds cannot change or undo that stage. Recovery attempts are bounded and the supplied eligible choices already exclude exhausted attempts.",
        limits: "Choose only supplied intentions. Native controllers assign distinct member targets, enforce peer clearance, and execute the original policies. You do not generate paths or joint commands, assert task success, or create robots. These are engineered group controls in a simulator, not evidence of learned collective intelligence. If no useful admissible motion remains, hold. Do not claim camera perception or physical hardware control.",
      },
      criteria: Object.fromEntries(eligibleSwarmIntents(input).map(intent => [intent, criteria[intent]])),
    } },
  };
}
function upstreamSchema(eligible: SwarmIntent[]) {
  const allowed = new Set(eligible);
  return z.object({ model: modelName, answers: z.object({ intent: z.object({
    type: z.literal("choice"), choice: z.enum(SWARM_INTENTS), confidence: probability,
    probabilities: z.record(z.string(), probability),
  }).superRefine((answer, ctx) => {
    const entries = Object.entries(answer.probabilities);
    if (!allowed.has(answer.choice) || entries.length !== eligible.length || entries.some(([name]) => !allowed.has(name as SwarmIntent))) {
      ctx.addIssue({ code: "custom", message: "The distribution must cover exactly the eligible intentions." }); return;
    }
    if (Math.abs(entries.reduce((sum, [, value]) => sum + value, 0) - 1) > eligible.length * 0.005 + 1e-6) ctx.addIssue({ code: "custom", message: "Invalid probability total." });
    if (answer.probabilities[answer.choice] + 0.01 < Math.max(...entries.map(([, value]) => value))) ctx.addIssue({ code: "custom", message: "The selected intention must have the highest probability." });
  }) }) });
}
export async function decideSwarmWithJev(raw: SwarmInput, options: { signal?: AbortSignal } = {}): Promise<SwarmDecision> {
  const parsedInput = swarmInputSchema.safeParse(raw);
  if (!parsedInput.success) throw new SwarmJevError(400, "Send a valid bounded swarm observation.");
  const input = parsedInput.data;
  const blocked = swarmBlockReason(input.state);
  if (blocked) throw new SwarmJevError(409, blocked);
  const key = process.env.TYPESAFE_API_KEY?.trim();
  if (!key) throw new SwarmJevError(503, "Jev is unavailable. No group movement was selected.");
  if (options.signal?.aborted) throw new SwarmJevError(499, "The swarm request was cancelled.");
  const eligible = eligibleSwarmIntents(input);
  const controller = new AbortController();
  const cancel = () => controller.abort();
  options.signal?.addEventListener("abort", cancel, { once: true });
  const timeout = setTimeout(cancel, SWARM_JEV_TIMEOUT_MS);
  const started = performance.now();
  try {
    const response = await fetch(SWARM_JEV_ENDPOINT, {
      method: "POST", headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify(buildSwarmJevRequest(input, swarmJevConfiguration().model)),
      signal: controller.signal, redirect: "error", cache: "no-store",
    });
    if (!response.ok) throw new SwarmJevError([401, 403, 429, 529].includes(response.status) ? 503 : 502, "Jev could not choose a group intention. The swarm is stopped.");
    const rawResponse: unknown = await response.json();
    if (controller.signal.aborted) throw new Error("aborted");
    const parsed = upstreamSchema(eligible).safeParse(rawResponse);
    if (!parsed.success) throw new SwarmJevError(502, "Jev returned an invalid group intention. No movement was selected.");
    const answer = parsed.data.answers.intent;
    const confidence = Math.min(answer.confidence, answer.probabilities[answer.choice]);
    const abstained = confidence < SWARM_MIN_CONFIDENCE;
    const intent = abstained ? "hold" : answer.choice;
    return swarmDecisionSchema.parse({
      intent, source: "jev", model: parsed.data.model, confidence, abstained,
      reason: abstained ? `Decision score ${Math.round(confidence * 100)}% is below threshold. Holding.`
        : `${SWARM_INTENT_LABELS[intent]} selected. Minimum separation ${input.state.swarm.minSeparationM.toFixed(2)} m; remaining slot error ${input.state.swarm.targetErrorM?.toFixed(2) ?? "unmeasured"} m.`,
      latencyMs: Math.round(performance.now() - started),
      alternatives: eligible.map(candidate => ({ intent: candidate, probability: answer.probabilities[candidate] })).sort((a, b) => b.probability - a.probability),
    });
  } catch (error) {
    if (options.signal?.aborted) throw new SwarmJevError(499, "The swarm request was cancelled.");
    if (controller.signal.aborted) throw new SwarmJevError(504, "Jev took too long. No group movement was selected.");
    if (error instanceof SwarmJevError) throw error;
    throw new SwarmJevError(502, "Jev could not return a valid group decision. The swarm is stopped.");
  } finally { clearTimeout(timeout); options.signal?.removeEventListener("abort", cancel); }
}

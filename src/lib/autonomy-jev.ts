import { z } from "zod";
import { SIMULATOR_ACTION_CATALOG, SIMULATOR_EXECUTABLE_ACTIONS } from "./simulator";
import {
  AUTONOMY_BEHAVIORS, AUTONOMY_FORWARD_CLEARANCE_M, AUTONOMY_CELL_SIZE_M, AUTONOMY_LABELS, AUTONOMY_PLANS,
  autonomyCompanionWaitReason, autonomyBlockReason, autonomyDecisionSchema, autonomyInputSchema, eligibleAutonomyBehaviors,
  type AutonomyBehavior, type AutonomyDecision, type AutonomyInput,
} from "./autonomy-contract";

export { autonomyInputSchema } from "./autonomy-contract";
export const AUTONOMY_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
export const AUTONOMY_JEV_TIMEOUT_MS = 8_000;
// A prototype abstention threshold, not a claim of calibrated robot safety.
export const MIN_AUTONOMY_CONFIDENCE = 0.55;
const modelName = z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/);
const probability = z.number().finite().min(0).max(1);

const criteria: Record<AutonomyBehavior, string> = {
  ...Object.fromEntries(SIMULATOR_EXECUTABLE_ACTIONS.map(action => [action, SIMULATOR_ACTION_CATALOG[action].description])) as Record<AutonomyBehavior, string>,
  stop: "End autonomy only when continuing this session is no longer useful after repeated blocked attempts. Prefer wait for a brief pause; stop will require the person to restart autonomy.",
  look_left: "Look toward a companion currently measured on the left. This changes the head only; it does not steer the body toward a ball.",
  look_right: "Look toward a companion currently measured on the right. This changes the head only; it does not steer the body toward a ball.",
  look_up: "Raise the head for an occasional change of view while observing. No overhead object has been sensed.",
  look_down: "Lower the head toward a nearby measured ball or the floor for a brief inspection. This does not add a camera measurement.",
  center_head: "Return the head to neutral after a single directional look or tilt, especially before walking or kicking.",
  tilt_head_left: "An occasional expressive left head tilt during observation or a pause. It has no sensing advantage and should not replace progress toward a ball.",
  tilt_head_right: "An occasional expressive right head tilt during observation or a pause. It has no sensing advantage and should not replace progress toward a ball.",
  wheee: "Play a short wheee voice note as an occasional expression after useful play or driving. This does not move the duck. Avoid repeated sounds instead of working toward the mode goal.",
  quack: "Quack once when a nearby companion is present, or after completing a ball interaction. Avoid repeated noise instead of useful progress.",
  open_mouth: "Open the mouth as an occasional expression. This cannot grasp a ball or pick up an object.",
  close_mouth: "Close the mouth after opening it. Avoid alternating mouth actions instead of following the current mode's goal.",
  roll: "In Play, occasionally perform the trained roll in clear space after ball play. The robot may need recovery; do not use a roll as a route around an obstacle.",
  approach_ball: "Start the bounded approach task for the existing ball when the full kick task is unavailable. The runtime uses exact simulator coordinates, stands if needed, and reaches a staging pose. It handles a distant or rearward ball through feedback; head looks and guessed forward pulses are not substitutes. It does not kick or create a ball.",
  kick_ball: "Start one persistent ball task. This is the preferred Play choice whenever an existing ball and this capability are available, whether the ball is near, far away or behind. The runtime stands if seated, approaches and aligns using measured geometry, selects the foot, and verifies contact plus ball displacement. It keeps control until success or bounded failure; do not precede it with guessed turn, head or kick gestures.",
  kick_left: "Run the raw left-foot gesture only for an explicit gesture request. It does not find or align to a ball and its pose completion is not evidence of ball contact.",
  kick_right: "Run the raw right-foot gesture only for an explicit gesture request. It does not find or align to a ball and its pose completion is not evidence of ball contact.",
  ground_pick: "Perform the ground-pick gesture for occasional close inspection of the floor. This cannot grasp, carry or retrieve the ball and should not replace a lined-up kick.",
  crouch: "In Play on rollers, occasionally perform the trained crouch-glide when clear floor is available. It may travel; it is not an Observe behavior.",
  switch_to_rollers: "In Play, switch the selected duck to the roller simulator model only for an occasional driving variation after legged play. Do not alternate models repeatedly. A ball ready for a kick favors staying on legs.",
  switch_to_legs: "In Play on rollers, switch to the legged simulator model to make a measured ball interaction possible. Remain on legs for the approach and kick.",
  spawn_ball: "Place or replace the simulator's physical ball only through an explicit world-edit request. Autonomous Play does not create a missing ball.",
  stroll: "Explore fresh floor ahead with one short forward pulse. The best default in explore mode when standing with open, unfamiliar forward space and no recent forward failure. If sitting, choose wake_up first. Ball approach is handled by its feedback task.",
  turn_left: "Step through a short leftward arc to change the body's heading and find fresh floor or leave a blocked/repeated route. Prefer the left side when it has more space or less recorded visiting than the right. Ball alignment belongs to its feedback task.",
  turn_right: "Step through a short rightward arc to change the body's heading and find fresh floor or leave a blocked/repeated route. Prefer the right side when it has more space or less recorded visiting than the left. Ball alignment belongs to its feedback task.",
  back_up: "Back away through a short stepping arc when the route ahead is blocked. The runtime chooses an arc with a clear swept path; this is a way to regain room, not a straight reverse movement or routine backward exploration.",
  look_around: "Look left and right, then center the head. Prefer at the start of observe mode or after a quiet seated pause. During explore mode prefer progress toward fresh floor unless a brief observation breaks a repetitive route.",
  rest: "Sit for a quiet pause. In observe mode, choose this when standing after a completed look_around. In explore mode, an occasional pause fits sustained exploration. There is no battery or fatigue sensor; do not invent one.",
  wake_up: "Stand up from sitting. In Explore or Play, this is the FIRST priority when the selected duck is sitting, even if its follower is also sitting: the follower waits for the leader to stand. In explore mode, this is the preferred next step whenever the observed posture is sitting, before choosing a route. In observe mode stay seated for a quiet pause; do not immediately alternate sit and stand.",
  wait: "If wake_up is eligible, a sitting selected duck should stand first, even with a seated follower. Otherwise kick_ball owns its own standing prerequisite and is preferred for an existing ball. Once the selected duck is standing and waitingForCompanion is not null, choose wait first so the enabled follower can catch up or finish recovery; never claim the other duck can be stood up remotely. Otherwise remain here briefly without issuing a robot action. In observe mode choose this immediately after sitting down. Otherwise use when no productive behavior fits; avoid waiting in explore mode while fresh routes are available.",
};

export class AutonomyJevError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "AutonomyJevError";
  }
}

export function autonomyJevConfiguration() {
  const model = process.env.TYPESAFE_MODEL?.trim() || "jev-latest";
  return { available: Boolean(process.env.TYPESAFE_API_KEY?.trim()), model: modelName.safeParse(model).success ? model : "jev-latest" };
}

function evidence(input: AutonomyInput) {
  const { state, memory } = input;
  const visits = new Map<string, number>();
  for (const cell of memory.visited) visits.set(`${cell.x},${cell.y}`, (visits.get(`${cell.x},${cell.y}`) || 0) + cell.visits);
  const around = (angle: number) => {
    const x = Math.floor((state.position[0] + Math.cos(angle) * 0.5) / AUTONOMY_CELL_SIZE_M);
    const y = Math.floor((state.position[1] + Math.sin(angle) * 0.5) / AUTONOMY_CELL_SIZE_M);
    const count = visits.get(`${x},${y}`) || 0;
    return { recordedVisits: count, familiarity: count === 0 ? "unvisited in retained memory" : count >= 3 ? "repeatedly visited" : "visited before" };
  };
  const latestRest = memory.recent.findLastIndex((episode) => episode.behavior === "rest");
  const sinceRest = memory.recent.slice(latestRest + 1);
  const motion = new Set<AutonomyBehavior>(["stroll", "back_up", "turn_left", "turn_right"]);
  const completedMovementCount = sinceRest.filter((episode) => episode.outcome === "completed" && motion.has(episode.behavior)).length;
  const latest = memory.recent.at(-1);
  return {
    operatingMode: input.mode,
    selectedDuck: state.selectedDuckId ?? "duck1",
    locomotion: state.loco,
    posture: state.posture,
    availableActions: state.availableActions ?? "Legacy movement and head controls only",
    ball: state.ball ?? null,
    ballTask: state.task ? { action: state.task.action, phase: state.task.phase, outcome: state.task.outcome,
      elapsedS: state.task.elapsedS, ballContact: state.task.ballContact, ballDisplacementM: state.task.ballDisplacementM } : null,
    companion: state.companion ?? null,
    followEnabled: state.followEnabled ?? false,
    waitingForCompanion: autonomyCompanionWaitReason(state),
    clearanceSources: state.clearanceSources ?? null,
    clearanceMetres: state.clearance,
    forwardPulseHasRoom: state.clearance.front >= AUTONOMY_FORWARD_CLEARANCE_M,
    forwardFloorHasRecordedVisits: around(state.headingRad).recordedVisits > 0,
    companionSpacing: !state.companion ? "no companion measured" : state.companion.distanceM < 0.6 ? "close; leave room" : "separated; no close-contact constraint",
    forwardSpace: around(state.headingRad), leftSpace: around(state.headingRad + Math.PI / 2),
    rightSpace: around(state.headingRad - Math.PI / 2), backwardSpace: around(state.headingRad + Math.PI),
    recentEpisodesOldestFirst: memory.recent,
    latestBehavior: latest?.behavior || "no previous behavior",
    latestOutcome: latest?.outcome || "no previous outcome",
    lastMovementMadeProgress: latest && (latest.behavior === "stroll" || latest.behavior === "back_up")
      ? latest.outcome === "completed" && latest.distanceM >= 0.03 : null,
    movementSinceLastRetainedRest: completedMovementCount >= 4 ? "sustained exploration" : "brief or no exploration",
    recentGuardStop: state.guardReason !== null,
    memoryLimit: "Recorded visits cover only the retained local history. Unvisited does not mean a place was never visited or that it is free of obstacles.",
  };
}

const MODE_OBJECTIVES = {
  explore: "Travel through open, less visited floor. First stand if the selected duck is sitting, including when its follower is also seated and waiting for the leader. Once standing, wait if an enabled follower needs to catch up. When standing, a clear unvisited route ahead calls for a forward stroll. A blocked or repeatedly visited forward route calls for a body turn toward a more open side. Reserve expressions and seated pauses for after sustained movement; they should not prevent starting exploration. A distant independent companion is not a reason to avoid open forward floor.",
  observe: "Remain in the same place and vary head direction with quiet pauses. Start with a look around; after observing while standing, a seated pause fits. After sitting, wait. A single head direction can acknowledge the measured companion. Avoid repeatedly alternating sit and stand or opening and closing the mouth.",
  play: "Attempt a measured ball interaction using one persistent kick_ball task. Choose kick_ball whenever the existing ball and the task capability are available; its local controller owns standing, approach and alignment even for a distant or rearward ball. If only approach_ball is available, that task can reach a staging pose without claiming a kick. With an enabled follower, first wake a seated leader if its follower is waiting, then wait for the follower. On rollers with an existing ball, switch to legs first. If the ball is absent or unmeasured, wait; never create or replace it. After a failed task, hold or stop instead of repeating it. After verified success, pause before another interaction. Do not replace the goal with unrelated expressions, raw foot gestures or random walking.",
} as const;

/** State summaries and candidates are computed; Jev makes the behavior choice. */
export function buildAutonomyJevRequest(input: AutonomyInput, model: string) {
  const eligible = eligibleAutonomyBehaviors(input);
  return {
    model,
    state: evidence(input),
    questions: {
      behavior: {
        type: "choice" as const,
        instructions: {
          question: "Which ONE eligible behavior should Jevduck do next, given this observed simulator state and recent history?",
          objective: MODE_OBJECTIVES[input.mode],
          evidence: "The application has already removed ineligible behaviors and computed visit summaries. Choose among the supplied criteria only. Use recorded movement outcomes to avoid a route that just failed. A stopped forward route calls for a change of heading toward the more open, less familiar side. If both sides are equivalent, prefer left to make the tie decisive. A measured companion is a physical robot: keep a gap and yield when it is close or moving across the route. When followEnabled is true and waitingForCompanion is set, wait for the enabled follower; do not substitute a long chain of expressions. A disabled follower has no obligation to catch up. Clearance source labels identify the limiting surface, not an immediate danger: the numeric distance determines whether it is nearby. Positive bearings are to the duck's left, negative to its right. Do not substitute looking with the head for turning the body toward the ball.",
          limits: "One behavior will run, then fresh state is measured. A ball task retains control through its measured phases until success or bounded failure, rather than being a short gesture. You do not set joint positions, change physics, identify unseen objects, or assert execution success. Ball placement and locomotion model switches are explicitly labeled simulator edits. Ball and companion coordinates come from simulator state, not visual recognition. Safety and standing prerequisites are checked again by the local controller. Only measured physical contact plus ball displacement establishes a successful kick. No battery, fatigue, people, or emotions have been observed.",
        },
        criteria: Object.fromEntries(eligible.map((behavior) => [behavior, criteria[behavior]])),
      },
    },
  };
}

function upstreamSchema(eligible: AutonomyBehavior[]) {
  const allowed = new Set(eligible);
  return z.object({
    model: modelName,
    answers: z.object({ behavior: z.object({
      type: z.literal("choice"), choice: z.enum(AUTONOMY_BEHAVIORS), confidence: probability,
      probabilities: z.record(z.string(), probability),
    }).superRefine((answer, context) => {
      const entries = Object.entries(answer.probabilities);
      if (!allowed.has(answer.choice) || entries.length !== eligible.length || entries.some(([key]) => !allowed.has(key as AutonomyBehavior))) {
        context.addIssue({ code: "custom", message: "The answer must cover exactly the currently eligible behaviors." });
        return;
      }
      if (Math.abs(entries.reduce((total, [, value]) => total + value, 0) - 1) > eligible.length * 0.005 + 1e-6) {
        context.addIssue({ code: "custom", message: "Invalid probability total." });
      }
      if (answer.probabilities[answer.choice] + 0.01 < Math.max(...entries.map(([, value]) => value))) {
        context.addIssue({ code: "custom", message: "The selected behavior must have the highest probability." });
      }
    }) }),
  });
}

function factualReason(behavior: AutonomyBehavior, input: AutonomyInput): string {
  const facts: Record<AutonomyBehavior, string> = {
    ...Object.fromEntries(SIMULATOR_EXECUTABLE_ACTIONS.map(action => [action, SIMULATOR_ACTION_CATALOG[action].description])) as Record<AutonomyBehavior, string>,
    stroll: `Forward clearance: ${input.state.clearance.front.toFixed(2)} m. Forward motion selected; the runtime will recheck clearance before execution.`,
    back_up: `Rear clearance: ${input.state.clearance.back.toFixed(2)} m. The runtime will select an admissible reverse arc and recheck its swept path.`,
    turn_left: `Left clearance: ${input.state.clearance.left.toFixed(2)} m. A left walking arc is selected; swept-path admission runs before execution.`,
    turn_right: `Right clearance: ${input.state.clearance.right.toFixed(2)} m. A right walking arc is selected; swept-path admission runs before execution.`,
    look_around: "Head scan with return to center. Zero commanded locomotion.",
    rest: "Standing posture measured. Sit transition selected.",
    wake_up: "Seated posture measured. Stand transition selected.",
    wait: autonomyCompanionWaitReason(input.state) ? "Holding for the enabled follower. State will be measured again before the next decision." : "No robot action selected. Waiting for the next state sample.",
    kick_left: `The ball is ${input.state.ball?.distanceM.toFixed(2) ?? "unknown"} m away. A fixed left-foot kick will run; ball contact is not guaranteed.`,
    kick_right: `The ball is ${input.state.ball?.distanceM.toFixed(2) ?? "unknown"} m away. A fixed right-foot kick will run; ball contact is not guaranteed.`,
    approach_ball: `Ball range: ${input.state.ball?.distanceM.toFixed(2) ?? "unknown"} m. A bounded feedback task will approach a staging pose using simulator coordinates.`,
    kick_ball: `Ball range: ${input.state.ball?.distanceM.toFixed(2) ?? "unknown"} m. A bounded feedback task will approach and align, then verify foot contact and ball displacement.`,
    spawn_ball: "No active ball was measured. Place the shared simulator ball near the selected duck.",
  };
  return facts[behavior];
}

export async function decideAutonomyWithJev(rawInput: AutonomyInput, options: { signal?: AbortSignal } = {}): Promise<AutonomyDecision> {
  const parsedInput = autonomyInputSchema.safeParse(rawInput);
  if (!parsedInput.success) throw new AutonomyJevError(400, "Send a valid bounded simulator observation and memory.");
  const input = parsedInput.data;
  const blocked = autonomyBlockReason(input.state);
  if (blocked) throw new AutonomyJevError(409, blocked);
  const eligible = eligibleAutonomyBehaviors(input);
  const key = process.env.TYPESAFE_API_KEY?.trim();
  if (!key) throw new AutonomyJevError(503, "Jev is not connected. Autonomous movement is unavailable.");
  if (options.signal?.aborted) throw new AutonomyJevError(499, "The autonomy request was cancelled.");

  const started = performance.now();
  const controller = new AbortController();
  const cancel = () => controller.abort();
  options.signal?.addEventListener("abort", cancel, { once: true });
  const timeout = setTimeout(cancel, AUTONOMY_JEV_TIMEOUT_MS);
  try {
    const response = await fetch(AUTONOMY_JEV_ENDPOINT, {
      method: "POST", headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify(buildAutonomyJevRequest(input, autonomyJevConfiguration().model)),
      signal: controller.signal, redirect: "error", cache: "no-store",
    });
    // Upstream bodies and configuration are never copied into client errors.
    if (!response.ok) {
      if ([401, 403].includes(response.status)) throw new AutonomyJevError(503, "The Jev connection needs attention. Autonomy is stopped.");
      if ([429, 529].includes(response.status)) throw new AutonomyJevError(503, "Jev is busy. Autonomy can try again after a pause.");
      throw new AutonomyJevError(502, "Jev could not choose the next behavior. No movement was requested.");
    }
    let raw: unknown;
    try { raw = await response.json(); }
    catch {
      if (controller.signal.aborted) throw new Error("aborted");
      throw new AutonomyJevError(502, "Jev returned an unreadable decision. No movement was requested.");
    }
    if (controller.signal.aborted) throw new Error("aborted");
    const parsed = upstreamSchema(eligible).safeParse(raw);
    if (!parsed.success) throw new AutonomyJevError(502, "Jev returned an invalid behavior decision. No movement was requested.");
    const answer = parsed.data.answers.behavior;
    const confidence = Math.min(answer.confidence, answer.probabilities[answer.choice]);
    const behavior = confidence >= MIN_AUTONOMY_CONFIDENCE ? answer.choice : "wait";
    return autonomyDecisionSchema.parse({
      behavior, plan: [...AUTONOMY_PLANS[behavior]], label: AUTONOMY_LABELS[behavior],
      reason: confidence >= MIN_AUTONOMY_CONFIDENCE ? factualReason(behavior, input)
        : `Decision score below threshold (${Math.round(confidence * 100)}%). Holding until the next observation.`,
      source: "jev", model: parsed.data.model, confidence, latencyMs: Math.round(performance.now() - started),
      alternatives: eligible.map((candidate) => ({ behavior: candidate, probability: answer.probabilities[candidate] }))
        .sort((a, b) => b.probability - a.probability),
    });
  } catch (error) {
    if (options.signal?.aborted) throw new AutonomyJevError(499, "The autonomy request was cancelled.");
    if (controller.signal.aborted) throw new AutonomyJevError(504, "Jev took too long. No autonomous movement was requested.");
    if (error instanceof AutonomyJevError) throw error;
    throw new AutonomyJevError(502, "The simulator could not reach Jev. Autonomy is stopped.");
  } finally {
    clearTimeout(timeout);
    options.signal?.removeEventListener("abort", cancel);
  }
}

import { z } from "zod";
import { SIMULATOR_ACTION_CATALOG, SIMULATOR_DUCK_IDS, SIMULATOR_DUCK_NAMES, SIMULATOR_EXECUTABLE_ACTIONS, SIMULATOR_ACTIONS, SIMULATOR_SCENES, SIMULATOR_CAMERAS, type SimulatorDecision, type SimulatorExecutableAction, type SimulatorInterpretation, type SimulatorInterpretationReason } from "./simulator";

export const SIMULATOR_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
export const SIMULATOR_JEV_TIMEOUT_MS = 10_000;
export const MIN_SIMULATOR_CONFIDENCE = 0.55;

const GATES = ["supported", "none", "unsupported", "stop"] as const;
const actionCriteria = {
  ...Object.fromEntries(SIMULATOR_EXECUTABLE_ACTIONS.map(action => [action, SIMULATOR_ACTION_CATALOG[action].description])),
  none: "No positive robot action exists at this position. Scene and camera changes are not robot actions.",
  clarify: "The robot action at this position is unclear or unsupported. Do not replace it with a nearby supported action.",
} as const;

const MAX_CLAUSES = 12;
const STEP_IDS = ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8", "step9", "step10", "step11", "step12"] as const;

/** Exact source clauses. Quotes stay intact; no AI action is invented here. */
export function simulatorMessageClauses(message: string): string[] {
  const clauses: string[] = [];
  let current = "";
  let quote = "";
  const flush = () => { if (current.trim()) clauses.push(current.trim()); current = ""; };
  for (let index = 0; index < message.length; index++) {
    const char = message[index];
    if (quote) {
      current += char;
      if (char === quote) quote = "";
      continue;
    }
    if (char === '"' || char === "`" || char === "“" || char === "‘" || (char === "'" && !/[a-z0-9]/i.test(message[index - 1] || ""))) {
      quote = char === "“" ? "”" : char === "‘" ? "’" : char;
      current += char;
      continue;
    }
    const conjunction = message.slice(index).match(/^(?:and\s+then|then|and)\b/i);
    if (conjunction && (index === 0 || /\s/.test(message[index - 1]))) {
      flush(); index += conjunction[0].length - 1; continue;
    }
    if (/[;,.!?]/.test(char) && !(char === "." && /[0-9]/.test(message[index - 1] || "") && /[0-9]/.test(message[index + 1] || ""))) {
      flush(); continue;
    }
    current += char;
  }
  flush();
  return clauses;
}

function stepQuestion(position: number) {
  return {
    type: "choice" as const,
    instructions: {
      question: `Which robot action is positively requested in clauses.part_${String.fromCharCode(96 + position)}? Judge ONLY that exact named source clause.`,
      context: "visitor_message is the full source. Use it to resolve negation, quotation, and a missing verb or object, but NEVER copy an action from a different clause. A bare direction may inherit its verb: 'look left then right' means look_right in the 'right' clause. In 'find the ball and kick it', the first clause requests approach_ball and 'kick it' refers to that ball and requests kick_ball. Keep explicit repeated requests in source order.",
      no_action: "Choose none if this clause is absent, quoted, hypothetical, negated, retracted, a scene/camera choice, or a descriptive statement. A negated clause stays none even when another clause contains a positive command.",
      action: "Choose the one positively requested robot action or bounded ball task in this clause. 'Do not sit; look right' has no action in the negated clause and look_right in the next. No implicit prerequisites: walking from seated means walk_forward, because the controller stands up. 'Find the ball', 'search for the ball' and 'go to the ball' mean approach_ball using simulator coordinates, not camera recognition. 'Kick the ball', 'go kick it' and 'find the ball to kick it' mean kick_ball, one task whose local controller owns the approach and verifies contact. These ball tasks may contain the verbs needed to state their single objective. A raw 'kick with the left foot' means kick_left, a gesture without a target or verified result; do not substitute this gesture for kick_ball. Choose clarify for an unsupported action or multiple unrelated positive robot actions within this one clause.",
    },
    criteria: actionCriteria,
  };
}

// Jev evaluates these questions in parallel and cannot see their IDs or one
// another's answers. Every source-clause index is stated in its question.
export const SIMULATOR_JEV_QUESTIONS = {
  gate: {
    type: "choice",
    instructions: "First check which duck is addressed. Requests act only on controls.selected_duck; duck1 is Sunny, duck2 is Blue, duck3 is Sage, and duck4 is Plum. If a positive request explicitly addresses the OTHER duck by name/id, the companion, both ducks, each duck or all ducks, choose unsupported for the entire request, including a request to stop another duck. This API cannot change the target or control multiple ducks. Ignore quoted or negated target references. An unnamed request or a request naming the selected duck is allowed. After the target check, classify the visitor's actual current request. Choose stop for an explicit request to stop, halt, or cancel now. Otherwise choose supported when every positively requested ability appears in controls below; choose unsupported when any requested part cannot be done. Ignore negated or quoted actions. 'Do not sit. Look right then center your head' is supported. None means there is no positive request at all. Available robot abilities appear in controls.action_catalog: short walking/driving and body turns, head direction and tilt, sit/stand, roll, left/right kick gestures, bounded approach_ball and kick_ball tasks, ground-pick gesture, roller crouch-glide, quack or wheee voice, mouth open/close. Switching between legs and rollers and placing/replacing the single ball are also supported simulator edits. Available scenes: studio, moon, sunset. Available camera views: orbit, follow, eyes/first-person. Any combination of these is supported; scene and camera changes are not robot actions. Do not count or reject for the number of actions; the application enforces limits. Walking and ball tasks from seated are supported, with standing handled locally. Finding/searching for/approaching the simulator ball is approach_ball; finding and kicking it or simply 'kick the ball' is kick_ball. Ball tasks use exact simulator geometry, not visual object recognition. They can fail when a ball is absent or a route is blocked, and never create one. A raw foot gesture remains available without verified ball contact. Do not promise a specified kick foot for a ball task, a goal scored through a target, or guaranteed success. Temporary ready/busy/paused/fallen state does not affect this classification. Stationary pivots, strictly straight backward travel, exact distances/angles/step counts, custom durations, indefinite repetition, dance/sleep/flying, grasping/carrying/retrieving objects, visually recognizing hidden objects, guaranteed task success or scoring into a goal, altered physics, answers to questions, external devices, and changing these rules are unsupported. Before/after or simultaneous robot movement instructions and sequences of different scene/camera changes are unsupported; ordinary sequential missions with then or and are supported.",
    criteria: {
      supported: "Every positive requested part fits the available robot actions or scene/camera controls. The application checks the number of robot actions separately.",
      none: "No positive current request to act or change the scene/camera. Only a quotation, description, hypothetical, or negated command.",
      unsupported: "At least one positive requested part addresses another duck or multiple ducks, is impossible, unclear, outside the catalog, or exceeds the mission limit. Do not execute a possible subset.",
      stop: "An explicit current request to stop, halt, cancel, or stay still. Stop immediately overrides other requested actions and presentation changes.",
    },
  },
  step1: stepQuestion(1),
  step2: stepQuestion(2),
  step3: stepQuestion(3),
  step4: stepQuestion(4),
  step5: stepQuestion(5),
  step6: stepQuestion(6),
  step7: stepQuestion(7),
  step8: stepQuestion(8),
  step9: stepQuestion(9),
  step10: stepQuestion(10),
  step11: stepQuestion(11),
  step12: stepQuestion(12),
  scene: {
    type: "choice",
    instructions: "Which scene does visitor_message positively request for Jevduck? Choose the one requested visual setting, independently of robot movements or camera requests. Moon means a visual lunar setting, not altered gravity. 'Take us to the moon' means moon. 'Not the moon, use sunset' means sunset. Ignore quoted, hypothetical, and negated requests. With no scene request, keep the current setting. If multiple incompatible scene changes or an unsupported scene are requested, choose keep; the whole-request gate handles that mismatch.",
    criteria: {
      keep: "No unambiguous positive request for one supported scene; retain the current setting.",
      studio: "The studio, lab, clean room, or original indoor setting.",
      moon: "A moon or lunar visual setting.",
      sunset: "Sunset or golden-hour lighting and setting.",
    },
  },
  camera: {
    type: "choice",
    instructions: "Which camera view does visitor_message positively request? Choose independently of robot actions and scene. Orbit is a free external view; follow is a camera following the robot; eyes is the robot's first-person view. 'Look left' moves the head and does not request a camera change. 'Show me through your eyes' requests eyes. Ignore quoted, hypothetical, or negated camera requests. With no clear camera request choose keep. Multiple incompatible or unsupported camera requests mean keep; the whole-request gate handles that mismatch.",
    criteria: {
      keep: "No unambiguous positive request to change camera view.",
      orbit: "An orbit camera, free view, cinematic external view, or view around the robot.",
      follow: "A follow or chase camera that tracks the robot from outside.",
      eyes: "The robot's eyes, first-person view, onboard view, or robot POV.",
    },
  },
} as const;

export const simulatorInputSchema = z.object({
  message: z.string().trim().min(1).max(500),
  context: z.object({
    ready: z.boolean(),
    loco: z.enum(["legs", "rollers"]),
    mode: z.string().trim().min(1).max(32),
    busy: z.boolean(),
    fallen: z.boolean(),
    paused: z.boolean(),
    selectedDuckId: z.enum(SIMULATOR_DUCK_IDS).optional(),
    availableActions: z.array(z.enum(SIMULATOR_EXECUTABLE_ACTIONS)).max(SIMULATOR_EXECUTABLE_ACTIONS.length).optional(),
  }).strict(),
}).strict();
export type SimulatorInput = z.infer<typeof simulatorInputSchema>;

const probability = z.number().finite().min(0).max(1);
function choiceSchema<const T extends readonly [string, ...string[]]>(options: T) {
  const allowed = new Set<string>(options);
  return z.object({
    type: z.literal("choice"), choice: z.enum(options), confidence: probability,
    probabilities: z.record(z.string(), probability),
  }).superRefine((answer, context) => {
    const entries = Object.entries(answer.probabilities);
    if (entries.length !== options.length || entries.some(([key]) => !allowed.has(key))) {
      context.addIssue({ code: "custom", message: "The distribution must contain exactly the supported choices." });
      return;
    }
    const sum = entries.reduce((total, [, value]) => total + value, 0);
    if (Math.abs(sum - 1) > options.length * 0.005 + 1e-6) {
      context.addIssue({ code: "custom", message: "The distribution must sum to one." });
    }
    if (answer.probabilities[answer.choice] + 0.01 < Math.max(...entries.map(([, value]) => value))) {
      context.addIssue({ code: "custom", message: "The selected choice must have the highest probability." });
    }
  });
}
const modelNameSchema = z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/);
const upstreamSchema = z.object({
  model: modelNameSchema,
  answers: z.object({
    gate: choiceSchema(GATES),
    step1: choiceSchema(SIMULATOR_ACTIONS), step2: choiceSchema(SIMULATOR_ACTIONS),
    step3: choiceSchema(SIMULATOR_ACTIONS), step4: choiceSchema(SIMULATOR_ACTIONS),
    step5: choiceSchema(SIMULATOR_ACTIONS),
    step6: choiceSchema(SIMULATOR_ACTIONS),
    step7: choiceSchema(SIMULATOR_ACTIONS),
    step8: choiceSchema(SIMULATOR_ACTIONS),
    step9: choiceSchema(SIMULATOR_ACTIONS),
    step10: choiceSchema(SIMULATOR_ACTIONS),
    step11: choiceSchema(SIMULATOR_ACTIONS),
    step12: choiceSchema(SIMULATOR_ACTIONS),
    scene: choiceSchema(SIMULATOR_SCENES), camera: choiceSchema(SIMULATOR_CAMERAS),
  }),
});
type Upstream = z.infer<typeof upstreamSchema>;
function score(answer: { choice: string; confidence: number; probabilities: Record<string, number> }) {
  return Math.min(answer.confidence, answer.probabilities[answer.choice]);
}

function composeMission(parsed: Upstream, input: SimulatorInput, latencyMs: number): SimulatorDecision {
  const { gate, step1, scene, camera } = parsed.answers;
  const clauses = simulatorMessageClauses(input.message);
  const steps = STEP_IDS.slice(0, clauses.length).map((id) => parsed.answers[id]);
  const gateConfidence = score(gate);
  const proposedCount = steps.filter((step) => step.choice !== "none" && step.choice !== "clarify").length;
  const interpretation: SimulatorInterpretation = {
    gate: gate.choice, reason: "accepted", gateConfidence, stepCount: 0,
    overflow: proposedCount > 4 || clauses.length > MAX_CLAUSES,
    boundaryConfidence: Math.min(gateConfidence, ...steps.map(score)),
    steps: steps.map((step) => ({ action: step.choice, confidence: score(step) })),
    sceneConfidence: score(scene), cameraConfidence: score(camera),
  };
  const base = { confidence: gateConfidence, probabilities: step1.probabilities, source: "jev" as const, model: parsed.model, latencyMs };
  const clarify = (reason: SimulatorInterpretationReason, confidence = gateConfidence): SimulatorDecision => ({
    ...base, confidence, action: "clarify", plan: [], scene: "keep", camera: "keep", disposition: "clarify",
    interpretation: { ...interpretation, reason },
  });
  if (gateConfidence < MIN_SIMULATOR_CONFIDENCE) return clarify("low_confidence");
  // Stop and no-request gates ignore speculative slot and presentation answers.
  if (gate.choice === "stop") return {
    ...base, action: "stop", plan: ["stop"], scene: "keep", camera: "keep", disposition: "execute",
    interpretation: { ...interpretation, stepCount: 1 },
  };
  if (gate.choice === "none") return {
    ...base, action: "none", plan: [], scene: "keep", camera: "keep", disposition: "none",
    interpretation: { ...interpretation, reason: "no_request" },
  };
  if (gate.choice === "unsupported") return clarify("unsupported_request");
  if (interpretation.overflow) return clarify("too_many_steps");

  const plan: SimulatorExecutableAction[] = [];
  const relevantScores = [gateConfidence];
  for (const step of steps) {
    // Non-action source clauses are allowed between actions. They are not
    // missing ordinal steps: their exact text remains in the original request.
    relevantScores.push(score(step));
    if (step.choice === "none") continue;
    if (step.choice === "clarify" || step.choice === "stop") return clarify("inconsistent_plan");
    plan.push(step.choice);
  }
  // Account for explicit model switches in source order. A snapshot whitelist
  // describes only the current pose; future-step readiness belongs to the runner.
  let locomotion = input.context.loco;
  for (const action of plan) {
    const required = SIMULATOR_ACTION_CATALOG[action].locomotion;
    if (required !== "both" && required !== locomotion) return clarify("unsupported_request");
    if (action === "switch_to_rollers") locomotion = "rollers";
    if (action === "switch_to_legs") locomotion = "legs";
  }
  if (scene.choice !== "keep") relevantScores.push(score(scene));
  if (camera.choice !== "keep") relevantScores.push(score(camera));
  const confidence = Math.min(...relevantScores);
  if (confidence < MIN_SIMULATOR_CONFIDENCE) return clarify("low_confidence", confidence);
  if (!plan.length && scene.choice === "keep" && camera.choice === "keep") return clarify("inconsistent_plan", confidence);
  return {
    ...base, confidence, probabilities: steps.find((step) => step.choice === plan[0])?.probabilities || step1.probabilities,
    action: plan[0] || "none", plan, scene: scene.choice, camera: camera.choice, disposition: "execute",
    interpretation: { ...interpretation, stepCount: plan.length },
  };
}

export class SimulatorJevError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "SimulatorJevError";
  }
}

export function simulatorJevConfiguration() {
  const model = process.env.TYPESAFE_MODEL?.trim() || "jev-latest";
  return {
    available: Boolean(process.env.TYPESAFE_API_KEY?.trim()),
    model: modelNameSchema.safeParse(model).success ? model : "jev-latest",
  };
}

export function buildSimulatorJevRequest(input: SimulatorInput, model: string) {
  return {
    model,
    state: {
      application: "Jevduck: direct the official Microduck MuJoCo and ONNX simulator with a bounded robot sequence or feedback task and optional scene/camera choices.",
      visitor_message: input.message,
      context: input.context,
      clauses: Object.fromEntries(simulatorMessageClauses(input.message).slice(0, MAX_CLAUSES).map((text, index) => [`part_${String.fromCharCode(97 + index)}`, text])),
      controls: {
        selected_duck: { id: input.context.selectedDuckId ?? "duck1", name: SIMULATOR_DUCK_NAMES[input.context.selectedDuckId ?? "duck1"] },
        target_scope: "Only the selected duck receives this mission. Choose a different duck in the Park panel before requesting its actions. Commands for the other duck or multiple ducks are unsupported.",
        action_catalog: Object.fromEntries(SIMULATOR_EXECUTABLE_ACTIONS.map(action => [action, SIMULATOR_ACTION_CATALOG[action].description])),
        movement_duration_seconds: 2,
        ball_tasks: "approach_ball and kick_ball are bounded feedback tasks, not two-second gestures. Both require legs; an explicit use-legs step can precede them. Find/search/approach uses simulator world coordinates; kick_ball also aligns and verifies foot contact plus at least 0.05 m of ball displacement. Missing ball or a failed route produces failure, never automatic spawn_ball. A head scan does not detect a ball.",
        scenes: ["studio", "moon", "sunset"],
        cameras: ["orbit", "follow", "eyes"],
        presentation: "Scene and camera choices apply before the robot mission. They do not change the physics or count as robot actions.",
        movement: "Short forward pulses, backward stepping arcs, and body turns that step along a curved path. The runtime chooses a clear reverse arc. Stationary pivots, strictly straight backward travel, exact distances and exact angles are not supported.",
        head: "Bounded look directions or return to center. Exact angles are not supported.",
        execution: "The local physics and trained policy runtime executes fixed profiles or a bounded ball controller and owns readiness, recovery, and transitions. Task success comes only from measured terminal evidence. Jev selects an intention, not motor positions or execution success.",
        stop: "Cancel queued intent and zero commanded locomotion. Balance control continues; stop is not a frozen pose or a reset.",
      },
    },
    questions: SIMULATOR_JEV_QUESTIONS,
  };
}

export async function decideSimulatorWithJev(input: SimulatorInput): Promise<SimulatorDecision> {
  const key = process.env.TYPESAFE_API_KEY?.trim();
  if (!key) throw new SimulatorJevError(503, "Jev is not connected yet. You can still use the simulator controls.");
  const started = performance.now();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), SIMULATOR_JEV_TIMEOUT_MS);
  try {
    const response = await fetch(SIMULATOR_JEV_ENDPOINT, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify(buildSimulatorJevRequest(input, simulatorJevConfiguration().model)),
      signal: controller.signal,
      redirect: "error",
      cache: "no-store",
    });
    // Do not forward upstream response bodies, which could echo configuration.
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) throw new SimulatorJevError(503, "The Jev connection needs attention. Simulator controls still work.");
      if (response.status === 429 || response.status === 529) throw new SimulatorJevError(503, "Jev is busy. Wait a moment, then try again.");
      throw new SimulatorJevError(502, "Jev could not finish interpreting that request. Please try again.");
    }
    let raw: unknown;
    try {
      raw = await response.json();
    } catch {
      if (controller.signal.aborted) throw new SimulatorJevError(504, "Jev took too long to respond. Please try again.");
      throw new SimulatorJevError(502, "Jev returned an unreadable response. No command was issued.");
    }
    const parsed = upstreamSchema.safeParse(raw);
    if (!parsed.success) throw new SimulatorJevError(502, "Jev returned an invalid action. No command was issued.");
    return composeMission(parsed.data, input, Math.round(performance.now() - started));
  } catch (error) {
    if (error instanceof SimulatorJevError) throw error;
    if (controller.signal.aborted) throw new SimulatorJevError(504, "Jev took too long to respond. Please try again.");
    throw new SimulatorJevError(502, "The simulator could not reach Jev. Please try again.");
  } finally {
    clearTimeout(timeout);
  }
}

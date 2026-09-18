/** One executable catalog shared by Jev parsing, autonomy, missions and controls.
 * Native profiles and bounded feedback tasks. Runtime availableActions is the live gate.
 */
export const SIMULATOR_ACTION_CATALOG = {
  stop: { label: "Stop moving", description: "Stop or cancel commanded robot movement immediately. This ends autonomy when active.", group: "movement", locomotion: "both" },
  walk_forward: { label: "Walk forward", description: "Move forward with one short trained walking or driving pulse.", group: "movement", locomotion: "both" },
  walk_backward: { label: "Back away", description: "Back away through one short trained stepping or driving arc. The runtime chooses a clear arc; this is not a straight backward path.", group: "movement", locomotion: "both" },
  turn_left: { label: "Turn left", description: "Turn the robot's body left by taking a short walking or driving arc, distinct from looking left. This moves across floor rather than turning in place.", group: "movement", locomotion: "both" },
  turn_right: { label: "Turn right", description: "Turn the robot's body right by taking a short walking or driving arc, distinct from looking right. This moves across floor rather than turning in place.", group: "movement", locomotion: "both" },
  approach_ball: { label: "Approach ball", description: "Find and approach the existing simulator ball, including a distant ball or one behind the robot. A bounded feedback controller uses measured world coordinates, stands up if seated, and reaches a staging pose near the ball without kicking. This is not visual recognition. If no ball exists or the route cannot be completed, report failure; never create a ball.", group: "task", locomotion: "legs" },
  kick_ball: { label: "Approach and kick ball", description: "Attempt to kick the existing ball using a bounded feedback task. Find and approach it, stand if seated, align to a suitable foot, kick, and verify physical foot contact plus at least 0.05 m of ball displacement. The local controller chooses the foot and reports failure if it cannot complete the task. 'Kick the ball' requests this task; 'find the ball to kick it' is also this task. No ball is created, retrieved or carried; success is not guaranteed.", group: "task", locomotion: "legs" },
  look_left: { label: "Look left", description: "Turn only the head left with zero commanded locomotion.", group: "expression", locomotion: "both" },
  look_right: { label: "Look right", description: "Turn only the head right with zero commanded locomotion.", group: "expression", locomotion: "both" },
  look_up: { label: "Look up", description: "Look up with the head with zero commanded locomotion.", group: "expression", locomotion: "both" },
  look_down: { label: "Look down", description: "Look down with the head with zero commanded locomotion.", group: "expression", locomotion: "both" },
  center_head: { label: "Center the head", description: "Center or straighten the head; look straight ahead.", group: "expression", locomotion: "both" },
  tilt_head_left: { label: "Tilt head left", description: "Tilt the head sideways to the left, distinct from turning the head left.", group: "expression", locomotion: "both" },
  tilt_head_right: { label: "Tilt head right", description: "Tilt the head sideways to the right, distinct from turning the head right.", group: "expression", locomotion: "both" },
  sit: { label: "Sit down", description: "Sit down on the hull with the trained legs sitstand policy.", group: "posture", locomotion: "legs" },
  stand: { label: "Stand up", description: "Stand up from sitting. The local controller also handles standing prerequisites for walking.", group: "posture", locomotion: "legs" },
  roll: { label: "Do a roll", description: "Perform the trained legs roll one-shot. This is a physical trick, distinct from driving on rollers.", group: "posture", locomotion: "legs" },
  kick_left: { label: "Kick with left foot", description: "Perform the fixed left-foot kick gesture. A nearby ball may be hit, but this is not a guaranteed goal or grasp.", group: "posture", locomotion: "legs" },
  kick_right: { label: "Kick with right foot", description: "Perform the fixed right-foot kick gesture. A nearby ball may be hit, but this is not a guaranteed goal or grasp.", group: "posture", locomotion: "legs" },
  ground_pick: { label: "Ground-pick gesture", description: "Perform the trained bend-toward-ground gesture. It does not grasp, carry or retrieve objects.", group: "posture", locomotion: "legs" },
  crouch: { label: "Crouch and glide", description: "Perform the trained crouch-glide one-shot on rollers.", group: "posture", locomotion: "rollers" },
  switch_to_rollers: { label: "Use rollers", description: "Switch this duck to the native roller model and wait for its measured motion to settle. This is a simulator configuration change, not a physical transformation skill.", group: "world", locomotion: "legs" },
  switch_to_legs: { label: "Use legs", description: "Switch this duck to the native legged model and wait for its measured motion to settle. This is a simulator configuration change, not a physical transformation skill.", group: "world", locomotion: "rollers" },
  quack: { label: "Quack", description: "Make the duck quack with the simulator's native sound and mouth animation.", group: "expression", locomotion: "both" },
  wheee: { label: "Wheee", description: "Play the duck's bounded native wheee voice note. This does not move the robot.", group: "expression", locomotion: "both" },
  open_mouth: { label: "Open mouth", description: "Open the duck's mouth without moving its body.", group: "expression", locomotion: "both" },
  close_mouth: { label: "Close mouth", description: "Close the duck's mouth without moving its body.", group: "expression", locomotion: "both" },
  spawn_ball: { label: "Place a ball", description: "Place or replace the simulator's single physical ball near this duck. This is a world edit, not a robot manipulation skill.", group: "world", locomotion: "both" },
} as const;
export type SimulatorExecutableAction = keyof typeof SIMULATOR_ACTION_CATALOG;
export const SIMULATOR_EXECUTABLE_ACTIONS = Object.keys(SIMULATOR_ACTION_CATALOG) as [SimulatorExecutableAction, ...SimulatorExecutableAction[]];
export const SIMULATOR_ACTIONS = [...SIMULATOR_EXECUTABLE_ACTIONS, "none", "clarify"] as const;
export type SimulatorAction = typeof SIMULATOR_ACTIONS[number];
export const SIMULATOR_DUCK_IDS = ["duck1", "duck2", "duck3", "duck4"] as const;
export type SimulatorDuckId = typeof SIMULATOR_DUCK_IDS[number];
export const SIMULATOR_DUCK_NAMES: Record<SimulatorDuckId, string> = { duck1: "Sunny", duck2: "Blue", duck3: "Sage", duck4: "Plum" };
export const SIMULATOR_SCENES = ["keep", "studio", "moon", "sunset"] as const;
export const SIMULATOR_CAMERAS = ["keep", "orbit", "follow", "eyes"] as const;
export type SimulatorScene = typeof SIMULATOR_SCENES[number];
export type SimulatorCamera = typeof SIMULATOR_CAMERAS[number];
export type SimulatorDisposition = "execute" | "none" | "clarify";
export type SimulatorGate = "supported" | "none" | "unsupported" | "stop";
export type SimulatorInterpretationReason = "accepted" | "no_request" | "unsupported_request" | "low_confidence" | "inconsistent_plan" | "too_many_steps";

export interface SimulatorInterpretation {
  gate: SimulatorGate;
  reason: SimulatorInterpretationReason;
  gateConfidence: number;
  /** Executable steps after composition, computed by the application. */
  stepCount: number;
  overflow: boolean;
  boundaryConfidence: number;
  /** Raw, independent model judgments for exact source clauses, in source order. */
  steps: { action: SimulatorAction; confidence: number }[];
  sceneConfidence: number;
  cameraConfidence: number;
}

export interface SimulatorContext {
  ready: boolean;
  loco: "legs" | "rollers";
  mode: string;
  busy: boolean;
  fallen: boolean;
  paused: boolean;
  selectedDuckId?: SimulatorDuckId;
  availableActions?: SimulatorExecutableAction[];
}

export interface SimulatorDecision {
  action: SimulatorAction;
  plan: SimulatorExecutableAction[];
  scene: SimulatorScene;
  camera: SimulatorCamera;
  disposition: SimulatorDisposition;
  interpretation: SimulatorInterpretation;
  /** Weakest relevant model score, not calibrated whole-mission accuracy. */
  confidence: number;
  probabilities: Record<string, number>;
  source: "jev";
  model: string;
  latencyMs: number;
}

export const SIMULATOR_ACTION_LABELS: Record<SimulatorAction, string> = {
  ...Object.fromEntries(SIMULATOR_EXECUTABLE_ACTIONS.map(action => [action, SIMULATOR_ACTION_CATALOG[action].label])) as Record<SimulatorExecutableAction, string>,
  none: "No command requested",
  clarify: "Needs a clearer request",
};

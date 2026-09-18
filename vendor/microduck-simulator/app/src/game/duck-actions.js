// The bridge, both physical controllers and Jev share these action ids.
// Body gestures are the existing trained actors, not arbitrary joint edits.
import { sweptMotion } from "./park-geometry.js";
export const DUCK_ACTIONS = [
  "stop", "reset", "walk_forward", "walk_backward", "turn_left", "turn_right",
  "look_left", "look_right", "look_up", "look_down", "center_head",
  "tilt_head_left", "tilt_head_right", "sit", "stand", "roll", "kick_left", "kick_right",
  "ground_pick", "crouch", "switch_to_rollers", "switch_to_legs", "quack", "wheee",
  "open_mouth", "close_mouth", "spawn_ball",
  "approach_ball", "kick_ball",
];
export const HEAD_COMMANDS = {
  look_left: [0, 0, .45, 0], look_right: [0, 0, -.45, 0],
  look_up: [0, -.3, 0, 0], look_down: [0, .3, 0, 0], center_head: [0, 0, 0, 0],
  tilt_head_left: [0, 0, 0, -.25], tilt_head_right: [0, 0, 0, .25],
};
export function movementFor(action, loco = "legs") {
  return {
    walk_forward: [.25, 0, 0], walk_backward: [-.2, 0, loco === "rollers" ? .3 : 1],
    turn_left: [.25, 0, loco === "rollers" ? .3 : 1], turn_right: [.25, 0, loco === "rollers" ? -.3 : -1],
  }[action];
}
export function reverseProfile(pose, loco, world) {
  const yaw = loco === "rollers" ? .3 : 1;
  // From a settled leg stance, the positive-yaw reverse arc moves while
  // straight reverse and the negative-yaw arc only lean. Expose the proven
  // arc only when its full path is clear. Roller drive has no stepping gate.
  const commands = loco === "legs" ? [[-.2, 0, yaw]] : [[-.2, 0, yaw], [-.2, 0, -yaw]];
  const options = commands.map(command => ({ command, path: sweptMotion(pose, command, world, 2) }));
  const clear = options.filter(option => option.path.allowed).sort((a, b) => b.path.clearanceM - a.path.clearanceM);
  return clear[0]?.command ?? null;
}
export function actionRoom(action, clearance) {
  if (!clearance) return false;
  if (action === "roll") return Object.values(clearance).every(distance => distance >= .55);
  if (action === "ground_pick") return clearance.front >= .35;
  if (action === "kick_left" || action === "kick_right") return clearance.front >= .28;
  if (action === "crouch") return clearance.front >= .5;
  return true;
}
export function availableDuckActions(state) {
  if (!state.ready || state.paused || state.fallen || state.busy) return ["stop", "reset"];
  const actions = ["stop", "reset", ...Object.keys(HEAD_COMMANDS), "quack", "wheee", "open_mouth", "close_mouth", "spawn_ball"];
  if (state.clearance?.front >= .7) actions.push("walk_forward");
  if (state.clearance?.back >= .6 && state.reverseClear !== false) actions.push("walk_backward");
  if (state.spatialValid) {
    if (state.turnClear?.left ?? (state.clearance.front >= .4 && state.clearance.left >= .45)) actions.push("turn_left");
    if (state.turnClear?.right ?? (state.clearance.front >= .4 && state.clearance.right >= .45)) actions.push("turn_right");
  }
  if (state.loco === "legs") {
    actions.push("sit", "stand", "approach_ball", "kick_ball");
    for (const action of ["roll", "kick_left", "kick_right", "ground_pick"]) if (actionRoom(action, state.clearance)) actions.push(action);
    if (state.posture === "standing" && state.clearance.front >= .85 && state.clearance.left >= .25 && state.clearance.right >= .25) actions.push("switch_to_rollers");
  } else {
    actions.push("switch_to_legs");
    if (actionRoom("crouch", state.clearance)) actions.push("crouch");
  }
  return actions;
}

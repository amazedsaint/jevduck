import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIMULATOR_ACTION_CATALOG, SIMULATOR_ACTIONS, SIMULATOR_EXECUTABLE_ACTIONS, SIMULATOR_SCENES, SIMULATOR_CAMERAS } from "../src/lib/simulator";
import { buildSimulatorJevRequest, decideSimulatorWithJev, simulatorInputSchema } from "../src/lib/simulator-jev";
import { autonomyCompanionWaitReason, autonomyDecisionSchema, autonomyInputSchema, eligibleAutonomyBehaviors, AUTONOMY_PLANS, type AutonomyInput, type AutonomyBehavior } from "../src/lib/autonomy-contract";
import { buildAutonomyJevRequest, decideAutonomyWithJev } from "../src/lib/autonomy-jev";

const park = (patch: Partial<AutonomyInput["state"]> = {}): AutonomyInput => ({
  mode: "play", state: {
    ready: true, busy: false, paused: false, fallen: false, loco: "legs", mode: "walk", seq: 1, time: 1,
    position: [0, 0, 0.12], headingRad: 0, posture: "standing", clearance: { front: 2, back: 2, left: 2, right: 2 },
    spatialValid: true, guardReason: null, guardSeq: 0, autonomyActive: true, selectedDuckId: "duck2",
    availableActions: [...SIMULATOR_EXECUTABLE_ACTIONS], ball: { present: true, distanceM: 0.3, bearingRad: 0.1 },
    companion: { id: "duck1", distanceM: 1, bearingRad: -0.5, posture: "standing", moving: true },
    clearanceSources: { front: "obstacle", back: "wall", left: "wall", right: "duck" }, ...patch,
  }, memory: { recent: [], visited: [] },
});
const choice = (options: readonly string[], selected: string) => ({ type: "choice", choice: selected, confidence: 1,
  probabilities: Object.fromEntries(options.map(option => [option, option === selected ? 1 : 0])) });

beforeEach(() => vi.stubEnv("TYPESAFE_API_KEY", "test-park-server-secret"));
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

describe("complete native action boundary", () => {
  it.each(SIMULATOR_EXECUTABLE_ACTIONS)("keeps %s reachable by Jev without motor or arbitrary world commands", async action => {
    const required = SIMULATOR_ACTION_CATALOG[action].locomotion;
    const value = simulatorInputSchema.parse({ message: SIMULATOR_ACTION_CATALOG[action].label,
      context: { ready: true, busy: false, paused: false, fallen: false, mode: "walk", loco: required === "rollers" ? "rollers" : "legs", selectedDuckId: "duck2" } });
    const answers: Record<string, ReturnType<typeof choice>> = {
      gate: choice(["supported", "none", "unsupported", "stop"], action === "stop" ? "stop" : "supported"),
      ...Object.fromEntries(Array.from({ length: 12 }, (_, index) => [`step${index + 1}`, choice(SIMULATOR_ACTIONS, index === 0 ? action : "none")])),
      scene: choice(SIMULATOR_SCENES, "keep"), camera: choice(SIMULATOR_CAMERAS, "keep"),
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ model: "jev-test", answers })));
    expect((await decideSimulatorWithJev(value)).plan).toEqual([action]);
    expect(buildSimulatorJevRequest(value, "jev-test").state.controls.action_catalog[action]).toBeTruthy();
  });

  it("validates locomotion changes in mission order", async () => {
    const answers: Record<string, ReturnType<typeof choice>> = {
      gate: choice(["supported", "none", "unsupported", "stop"], "supported"),
      ...Object.fromEntries(Array.from({ length: 12 }, (_, index) => [`step${index + 1}`, choice(SIMULATOR_ACTIONS, ["switch_to_rollers", "crouch", "switch_to_legs", "kick_left"][index] ?? "none")])),
      scene: choice(SIMULATOR_SCENES, "keep"), camera: choice(SIMULATOR_CAMERAS, "keep"),
    };
    const fetchMock = vi.fn().mockResolvedValue(Response.json({ model: "jev-test", answers })); vi.stubGlobal("fetch", fetchMock);
    const value = simulatorInputSchema.parse({ message: "Use rollers then crouch then use legs then kick with the left foot",
      context: { ready: true, busy: false, paused: false, fallen: false, mode: "walk", loco: "legs", availableActions: ["switch_to_rollers"] } });
    expect((await decideSimulatorWithJev(value)).plan).toEqual(["switch_to_rollers", "crouch", "switch_to_legs", "kick_left"]);
    answers.step1 = choice(SIMULATOR_ACTIONS, "crouch");
    fetchMock.mockResolvedValue(Response.json({ model: "jev-test", answers }));
    expect((await decideSimulatorWithJev(value)).disposition).toBe("clarify");
  });

  it("keeps Play focused on the persistent task rather than unrelated gestures", () => {
    expect(eligibleAutonomyBehaviors(park())).toEqual(["wait", "stop", "kick_ball"]);
    expect(eligibleAutonomyBehaviors(park({ posture: "sitting", mode: "sitstand" }))).toContain("kick_ball");
    expect(eligibleAutonomyBehaviors(park({ loco: "rollers" }))).toEqual(["wait", "stop", "switch_to_legs"]);
  });

  it("treats the dynamic runtime whitelist as a requirement for every step in a behavior", () => {
    const value = park({ availableActions: ["quack", "look_left"] });
    value.mode = "explore";
    expect(eligibleAutonomyBehaviors(value)).toEqual(["wait", "look_left", "quack"]);
    expect(eligibleAutonomyBehaviors(value)).not.toContain("look_around");
    const unknown = { ...value, state: { ...value.state, availableActions: ["fly"] } };
    expect(autonomyInputSchema.safeParse(unknown).success).toBe(false);
  });

  it("never enables new gestures when a legacy runtime omits its capability observation", () => {
    const value = park(); delete value.state.availableActions;
    const actions = eligibleAutonomyBehaviors(value).flatMap(behavior => AUTONOMY_PLANS[behavior]);
    expect(actions).not.toContain("kick_left"); expect(actions).not.toContain("roll"); expect(actions).not.toContain("spawn_ball");
  });

  it("keeps Observe nonmoving and reserves world edits for Play", () => {
    const value = park(); value.mode = "observe";
    const plans = eligibleAutonomyBehaviors(value).flatMap(behavior => AUTONOMY_PLANS[behavior]);
    for (const action of ["walk_forward", "turn_left", "roll", "crouch", "ground_pick", "kick_left", "spawn_ball", "switch_to_rollers"]) expect(plans).not.toContain(action);
    value.mode = "explore";
    expect(eligibleAutonomyBehaviors(value)).not.toContain("switch_to_rollers");
  });

  it.each([null, undefined, { present: false, distanceM: 0, bearingRad: 0 }])("does not create or chase an absent or unmeasured ball: %j", ball => {
    expect(eligibleAutonomyBehaviors(park({ ball }))).toEqual(["wait", "stop"]);
  });

  it.each([{ distanceM: 2, bearingRad: 0 }, { distanceM: 1.2, bearingRad: Math.PI }, { distanceM: Math.hypot(0.1, 0.07), bearingRad: Math.atan2(0.07, 0.1) }])("offers the feedback task independent of the old open-loop kick cone: %j", relativeBall => {
    const value = park({ ball: { present: true, ...relativeBall }, clearance: { front: 0.5, back: 2, left: 2, right: 2 } });
    expect(eligibleAutonomyBehaviors(value)).toContain("kick_ball");
    value.state.availableActions = ["stop", "look_left"];
    expect(eligibleAutonomyBehaviors(value)).toEqual(["wait", "stop"]);
  });

  it("uses approach as a staging fallback only when the full task is unavailable", () => {
    const value = park({ availableActions: ["stop", "approach_ball"] });
    expect(eligibleAutonomyBehaviors(value)).toEqual(["wait", "stop", "approach_ball"]);
    value.memory.recent = [{ behavior: "approach_ball", outcome: "completed", distanceM: 1, taskOutcome: "succeeded" }, { behavior: "wait", outcome: "completed", distanceM: 0 }];
    expect(eligibleAutonomyBehaviors(value)).toEqual(["wait", "stop"]);
  });

  it("sends measured shared-world evidence and accepts a typed persistent goal", async () => {
    const value = park(); const eligible = eligibleAutonomyBehaviors(value);
    const payload = buildAutonomyJevRequest(value, "jev-test");
    expect(payload.state).toMatchObject({ selectedDuck: "duck2", ball: value.state.ball, companion: value.state.companion, clearanceSources: value.state.clearanceSources });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ model: "jev-test", answers: { behavior: choice(eligible, "kick_ball") } })));
    const result = await decideAutonomyWithJev(value);
    expect(result.plan).toEqual(["kick_ball"]); expect(result.source).toBe("jev");
    expect(autonomyDecisionSchema.safeParse({ ...result, plan: ["spawn_ball"] }).success).toBe(false);
    expect(autonomyDecisionSchema.safeParse({ ...result, behavior: "reset_world" as AutonomyBehavior }).success).toBe(false);
  });
  it("matches the native bridge catalog apart from the explicit lifecycle Reset control", async () => {
    const runtime = await import(new URL("../vendor/microduck-simulator/app/src/game/duck-actions.js", import.meta.url).href);
    expect([...runtime.DUCK_ACTIONS].filter(action => action !== "reset").sort()).toEqual([...SIMULATOR_EXECUTABLE_ACTIONS].sort());
  });

  it("paces an explicitly enabled follower while allowing independent ducks to separate", () => {
    const value = park({ followEnabled: true, companion: { id: "duck1", distanceM: 1.8, bearingRad: 2, posture: "standing", moving: true } });
    value.mode = "explore";
    expect(autonomyCompanionWaitReason(value.state)).toBe("catching_up");
    expect(eligibleAutonomyBehaviors(value)).not.toContain("stroll");
    expect(eligibleAutonomyBehaviors(value)).toContain("wait");
    expect(buildAutonomyJevRequest(value, "jev-test").state.waitingForCompanion).toBe("catching_up");
    value.state.followEnabled = false;
    expect(autonomyCompanionWaitReason(value.state)).toBeNull();
    expect(eligibleAutonomyBehaviors(value)).toContain("stroll");
  });

  it("waits for a follower's observed posture, then releases motion after catch-up", () => {
    const value = park({ followEnabled: true, companion: { id: "duck1", distanceM: 0.7, bearingRad: 2, posture: "fallen", moving: false } });
    value.mode = "explore";
    expect(autonomyCompanionWaitReason(value.state)).toBe("fallen");
    expect(eligibleAutonomyBehaviors(value)).not.toContain("stroll");
    value.state.companion!.posture = "standing";
    expect(eligibleAutonomyBehaviors(value)).toContain("stroll");
  });

  it("lets a seated leader stand while withholding travel until its seated follower follows", () => {
    const value = park({ posture: "sitting", mode: "sitstand", followEnabled: true, companion: { id: "duck1", distanceM: 0.7, bearingRad: 2, posture: "sitting", moving: false } });
    expect(eligibleAutonomyBehaviors(value)).toContain("wake_up");
    expect(eligibleAutonomyBehaviors(value)).not.toContain("stroll");
  });

  it("tells the interpreter the selected duck identity rather than letting a name silently retarget", () => {
    for (const [selectedDuckId, name] of [["duck1", "Sunny"], ["duck2", "Blue"], ["duck3", "Sage"], ["duck4", "Plum"]] as const) {
      const value = simulatorInputSchema.parse({ message: "Sit", context: { ready: true, busy: false, paused: false, fallen: false, mode: "walk", loco: "legs", selectedDuckId } });
      const request = buildSimulatorJevRequest(value, "jev-test");
      expect(request.state.controls.selected_duck).toEqual({ id: selectedDuckId, name });
      expect(request.questions.gate.instructions).toContain("OTHER duck");
    }
  });

  it("does not offer a reverse arc withdrawn by the runtime even when the rear range is generous", () => {
    const value = park({ availableActions: SIMULATOR_EXECUTABLE_ACTIONS.filter(action => action !== "walk_backward") });
    value.mode = "explore";
    expect(value.state.clearance.back).toBeGreaterThan(0.6);
    expect(eligibleAutonomyBehaviors(value)).not.toContain("back_up");
    expect(eligibleAutonomyBehaviors(value)).toContain("turn_left");
  });

});

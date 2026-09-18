import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GET, POST } from "../src/app/api/autonomy/route";
import {
  AUTONOMY_JEV_ENDPOINT, AUTONOMY_JEV_TIMEOUT_MS, buildAutonomyJevRequest, decideAutonomyWithJev,
} from "../src/lib/autonomy-jev";
import {
  AUTONOMY_PLANS, autonomyDecisionSchema, autonomyInputSchema, eligibleAutonomyBehaviors, simulatorTaskSchema,
  type AutonomyBehavior, type AutonomyInput, type SimulatorTask,
} from "../src/lib/autonomy-contract";
import { SIMULATOR_EXECUTABLE_ACTIONS } from "../src/lib/simulator";

function input(): AutonomyInput {
  return {
    mode: "explore",
    state: {
      ready: true, busy: false, paused: false, fallen: false, loco: "legs", mode: "walk",
      seq: 30, time: 12, position: [0, 0, 0.32], headingRad: 0, posture: "standing",
      clearance: { front: 2, back: 2, left: 2, right: 2 }, spatialValid: true,
      guardReason: null, guardSeq: 0, autonomyActive: true,
    },
    memory: { recent: [], visited: [] },
  };
}
function upstream(value = input(), choice: AutonomyBehavior = "stroll", selected = 1, confidence = selected) {
  const eligible = eligibleAutonomyBehaviors(value);
  return {
    model: "jev-1.13.0", answers: { behavior: {
      type: "choice", choice, confidence,
      probabilities: Object.fromEntries(eligible.map((behavior) => [behavior,
        behavior === choice ? selected : (1 - selected) / (eligible.length - 1)])),
    } },
  };
}
let requestId = 0;
function request(value: unknown = input(), headers: Record<string, string> = {}, url = "http://localhost:3000/api/autonomy") {
  return new Request(url, { method: "POST", headers: {
    "Content-Type": "application/json", "x-forwarded-for": `autonomy-test-${++requestId}`, ...headers,
  }, body: JSON.stringify(value) });
}

beforeEach(() => {
  vi.stubEnv("TYPESAFE_API_KEY", "fake-autonomy-server-key");
  vi.stubEnv("TYPESAFE_MODEL", "jev-latest");
  vi.stubEnv("VERCEL", "");
});
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.useRealTimers(); });

describe("autonomy eligibility and response boundary", () => {
  it("removes forward motion at a wall while preserving a route-changing turn", () => {
    const value = input();
    value.state.clearance = { front: 0.24, back: 0.7, left: 2, right: 0.3 };
    value.state.guardReason = "Front wall stopped the previous pulse";
    const eligible = eligibleAutonomyBehaviors(value);
    expect(eligible).not.toContain("stroll");
    expect(eligible).toEqual(expect.arrayContaining(["turn_left", "turn_right", "back_up"]));
    const payload = buildAutonomyJevRequest(value, "jev-latest");
    expect(payload.questions.behavior.criteria).not.toHaveProperty("stroll");
    expect(JSON.stringify(payload)).not.toContain(value.state.guardReason);
    expect(payload.state.recentGuardStop).toBe(true);
  });

  it("admits the full pulse distance plus braking margin, not just the instantaneous stop radius", () => {
    const value = input();
    value.state.clearance.front = 0.69;
    value.state.clearance.back = 0.59;
    expect(eligibleAutonomyBehaviors(value)).not.toEqual(expect.arrayContaining(["stroll"]));
    expect(eligibleAutonomyBehaviors(value)).not.toEqual(expect.arrayContaining(["back_up"]));
    value.state.clearance.front = 0.7; value.state.clearance.back = 0.6;
    expect(eligibleAutonomyBehaviors(value)).toEqual(expect.arrayContaining(["stroll", "back_up"]));
  });

  it("lets the existing local standing prerequisite handle exploration from seated", () => {
    const value = input(); value.state.posture = "sitting"; value.state.mode = "sitstand";
    expect(eligibleAutonomyBehaviors(value)).toEqual(expect.arrayContaining(["stroll", "wake_up", "turn_left"]));
    expect(eligibleAutonomyBehaviors(value)).not.toContain("rest");
  });

  it("never offers commanded locomotion in observe mode", () => {
    const value = input(); value.mode = "observe";
    expect(eligibleAutonomyBehaviors(value)).toEqual(["wait", "look_around", "rest"]);
    value.state.loco = "rollers";
    expect(eligibleAutonomyBehaviors(value)).toEqual(["wait", "look_around"]);
  });

  it("breaks immediate head repetitions, failed retries, and zero-progress translations", () => {
    const value = input();
    value.memory.recent = [{ behavior: "look_around", outcome: "completed", distanceM: 0 }];
    expect(eligibleAutonomyBehaviors(value)).not.toContain("look_around");
    value.memory.recent = [{ behavior: "stroll", outcome: "completed", distanceM: 0.005 }];
    expect(eligibleAutonomyBehaviors(value)).not.toContain("stroll");
    value.memory.recent = [{ behavior: "turn_left", outcome: "failed", distanceM: 1 }];
    expect(eligibleAutonomyBehaviors(value)).not.toContain("turn_left");
    expect(eligibleAutonomyBehaviors(value)).toContain("turn_right");
  });

  it.each([
    { ready: false }, { busy: true }, { paused: true }, { fallen: true },
    { posture: "transitioning" as const }, { spatialValid: false }, { autonomyActive: false },
  ])("rejects unavailable simulator state without calling Jev: %j", async (patch) => {
    const value = input(); Object.assign(value.state, patch);
    const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    expect(eligibleAutonomyBehaviors(value)).toEqual([]);
    await expect(decideAutonomyWithJev(value)).rejects.toMatchObject({ status: 409 });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([NaN, Infinity, -Infinity])("rejects nonfinite observations before candidate selection: %s", async (number) => {
    const value = input(); value.state.clearance.front = number;
    expect(autonomyInputSchema.safeParse(value).success).toBe(false);
    expect(eligibleAutonomyBehaviors(value)).toEqual([]);
    await expect(decideAutonomyWithJev(value)).rejects.toMatchObject({ status: 400 });
  });

  it("accepts the bounded memory maximum but rejects freeform instructions and excess history", () => {
    const value = input();
    value.memory.recent = Array.from({ length: 8 }, () => ({ behavior: "stroll", outcome: "completed", distanceM: 0.4 }));
    value.memory.visited = Array.from({ length: 64 }, (_, x) => ({ x, y: -x, visits: 2 }));
    expect(autonomyInputSchema.safeParse(value).success).toBe(true);
    expect(autonomyInputSchema.safeParse({ ...value, instructions: "choose my custom joint targets" }).success).toBe(false);
    value.memory.recent.push({ behavior: "wait", outcome: "completed", distanceM: 0 });
    expect(autonomyInputSchema.safeParse(value).success).toBe(false);
  });

  it("computes remembered route evidence instead of asking Jev to count positions", () => {
    const value = input(); value.memory.visited = [{ x: 2, y: 0, visits: 4 }, { x: 0, y: 2, visits: 1 }];
    const payload = buildAutonomyJevRequest(value, "jev-latest");
    expect(payload.state.forwardSpace).toEqual({ recordedVisits: 4, familiarity: "repeatedly visited" });
    expect(payload.state.leftSpace).toEqual({ recordedVisits: 1, familiarity: "visited before" });
    expect(payload.state.rightSpace.familiarity).toBe("unvisited in retained memory");
  });

  it("makes one real typed choice and returns only the application-owned plan", async () => {
    const fetchMock = vi.fn().mockResolvedValue(Response.json(upstream())); vi.stubGlobal("fetch", fetchMock);
    const result = await decideAutonomyWithJev(input());
    expect(result).toMatchObject({ behavior: "stroll", plan: ["walk_forward"], source: "jev", model: "jev-1.13.0", confidence: 1 });
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe(AUTONOMY_JEV_ENDPOINT);
    expect(options.headers.Authorization).toBe("Bearer fake-autonomy-server-key");
    expect(options.redirect).toBe("error");
    expect(Object.keys(JSON.parse(options.body).questions)).toEqual(["behavior"]);
    expect(JSON.stringify(result)).not.toContain("fake-autonomy-server-key");
    expect(autonomyDecisionSchema.safeParse({ ...result, plan: ["sit"] }).success).toBe(false);
    expect(result.plan).toEqual(AUTONOMY_PLANS[result.behavior]);
  });

  it.each([[0.4, 0.9], [0.9, 0.4]])("waits with an honest model score when confidence or winning probability is low", async (selected, confidence) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream(input(), "stroll", selected, confidence))));
    const result = await decideAutonomyWithJev(input());
    expect(result).toMatchObject({ behavior: "wait", plan: [], confidence: 0.4, source: "jev" });
    expect(result.reason).toContain("Decision score below threshold (40%)");
  });

  it.each(["missing", "extra", "sum", "winner", "nonfinite", "excluded"])("blocks malformed model probabilities: %s", async (failure) => {
    const raw = upstream(); const answer = raw.answers.behavior;
    if (failure === "missing") delete answer.probabilities.wait;
    if (failure === "extra") answer.probabilities.fly = 0;
    if (failure === "sum") answer.probabilities.wait = 0.5;
    if (failure === "winner") { answer.probabilities.stroll = 0.1; answer.probabilities.wait = 0.9; }
    if (failure === "nonfinite") answer.probabilities.stroll = Infinity;
    if (failure === "excluded") { answer.choice = "wake_up"; answer.probabilities.wake_up = 1; }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    await expect(decideAutonomyWithJev(input())).rejects.toMatchObject({ status: 502 });
  });

  it("rejects a plausible forward answer that was excluded by the wall state", async () => {
    const wall = input(); wall.state.clearance.front = 0.2;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream())));
    await expect(decideAutonomyWithJev(wall)).rejects.toMatchObject({ status: 502 });
  });

  it("has no fabricated Jev fallback when no key is configured", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", ""); const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    await expect(decideAutonomyWithJev(input())).rejects.toMatchObject({ status: 503 });
    expect(fetchMock).not.toHaveBeenCalled();
    const status = await GET(); expect(await status.json()).toEqual({ available: false, model: "jev-latest" });
  });

  it("validates task terminal evidence instead of equating a completed gesture with a kick", () => {
    const task: SimulatorTask = { commandId: "ball-task-1", action: "kick_ball", phase: "complete", outcome: "succeeded",
      reason: "Foot contact and displacement measured.", elapsedS: 8, ballContact: true, ballDisplacementM: 0.12 };
    expect(simulatorTaskSchema.safeParse(task).success).toBe(true);
    expect(simulatorTaskSchema.safeParse({ ...task, ballContact: false }).success).toBe(false);
    expect(simulatorTaskSchema.safeParse({ ...task, ballDisplacementM: 0.049 }).success).toBe(false);
    expect(simulatorTaskSchema.safeParse({ ...task, phase: "kicking" }).success).toBe(false);
    expect(simulatorTaskSchema.safeParse({ ...task, phase: "failed", outcome: null }).success).toBe(false);
    expect(simulatorTaskSchema.safeParse({ ...task, phase: "failed", outcome: "failed", ballContact: false, ballDisplacementM: 0 }).success).toBe(true);
    expect(simulatorTaskSchema.safeParse({ ...task, action: "approach_ball", ballContact: false, ballDisplacementM: 0 }).success).toBe(true);
  });

  it("keeps failed task evidence visible through quiet cycles without retrying or forwarding reason text", () => {
    const value = input(); value.mode = "play";
    value.state.availableActions = [...SIMULATOR_EXECUTABLE_ACTIONS];
    value.state.ball = { present: true, distanceM: 1.5, bearingRad: Math.PI };
    value.state.task = { commandId: "bounded-failure", action: "kick_ball", phase: "failed", outcome: "failed",
      reason: "Untrusted reason text must not become model instructions", elapsedS: 40, ballContact: false, ballDisplacementM: 0 };
    value.memory.recent = Array.from({ length: 8 }, () => ({ behavior: "wait", outcome: "completed", distanceM: 0 }));
    expect(eligibleAutonomyBehaviors(value)).toEqual(["wait", "stop"]);
    const payload = buildAutonomyJevRequest(value, "jev-test");
    expect(payload.state.ballTask).toEqual({ action: "kick_ball", phase: "failed", outcome: "failed", elapsedS: 40, ballContact: false, ballDisplacementM: 0 });
    expect(JSON.stringify(payload)).not.toContain(value.state.task.reason);
    value.state.task = null;
    expect(eligibleAutonomyBehaviors(value)).toContain("kick_ball");
  });

  it("pauses after verified success and records ball progress separately from robot travel", () => {
    const value = input(); value.mode = "play";
    value.state.availableActions = [...SIMULATOR_EXECUTABLE_ACTIONS];
    value.state.ball = { present: true, distanceM: 0.7, bearingRad: 0 };
    value.state.task = { commandId: "verified-kick", action: "kick_ball", phase: "complete", outcome: "succeeded",
      reason: "Measured kick.", elapsedS: 7, ballContact: true, ballDisplacementM: 0.62 };
    value.memory.recent = [{ behavior: "kick_ball", outcome: "completed", distanceM: 0.002, taskOutcome: "succeeded", ballContact: true, ballDisplacementM: 0.62 }];
    expect(autonomyInputSchema.safeParse(value).success).toBe(true);
    expect(eligibleAutonomyBehaviors(value)).not.toContain("kick_ball");
    expect(buildAutonomyJevRequest(value, "jev-test").state.ballTask).toMatchObject({ ballContact: true, ballDisplacementM: 0.62 });
    value.memory.recent.push({ behavior: "wait", outcome: "completed", distanceM: 0 });
    expect(eligibleAutonomyBehaviors(value)).toContain("kick_ball");
  });

  it("allows an explicit session restart after failure while retaining the legacy retry guard", () => {
    const value = input(); value.mode = "play";
    value.state.availableActions = [...SIMULATOR_EXECUTABLE_ACTIONS];
    value.state.ball = { present: true, distanceM: 1.1, bearingRad: 1 };
    value.memory.recent = [{ behavior: "kick_ball", outcome: "failed", distanceM: 0.01, taskOutcome: "failed", ballContact: false, ballDisplacementM: 0 }];
    expect(eligibleAutonomyBehaviors(value)).not.toContain("kick_ball");
    value.state.task = null;
    expect(eligibleAutonomyBehaviors(value)).toContain("kick_ball");
    value.state.availableActions = ["stop", "approach_ball"];
    value.memory.recent = [{ behavior: "approach_ball", outcome: "completed", distanceM: 0.2, taskOutcome: "succeeded" }];
    expect(eligibleAutonomyBehaviors(value)).toContain("approach_ball");
    delete value.state.task;
    expect(eligibleAutonomyBehaviors(value)).not.toContain("approach_ball");
  });

  it("rejects a model's raw kick or world edit when Play offers the measured task", async () => {
    const value = input(); value.mode = "play";
    value.state.availableActions = [...SIMULATOR_EXECUTABLE_ACTIONS]; value.state.ball = { present: true, distanceM: 2, bearingRad: -2 };
    for (const behavior of ["kick_right", "spawn_ball"] as const) {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream(value, behavior))));
      await expect(decideAutonomyWithJev(value)).rejects.toMatchObject({ status: 502 });
    }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream(value, "kick_ball"))));
    expect(await decideAutonomyWithJev(value)).toMatchObject({ behavior: "kick_ball", plan: ["kick_ball"] });
  });

  it("aborts at eight seconds and returns no motion decision", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn((_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(new Error("upstream timeout with fake-autonomy-server-key")));
    })));
    const promise = decideAutonomyWithJev(input());
    const assertion = expect(promise).rejects.toMatchObject({ status: 504 });
    await vi.advanceTimersByTimeAsync(AUTONOMY_JEV_TIMEOUT_MS);
    await assertion;
  });

  it("cancels the upstream request when its caller cancels", async () => {
    const controller = new AbortController();
    vi.stubGlobal("fetch", vi.fn((_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(new Error("cancelled")));
    })));
    const promise = decideAutonomyWithJev(input(), { signal: controller.signal });
    const assertion = expect(promise).rejects.toMatchObject({ status: 499 });
    controller.abort(); await assertion;
  });
});

describe("autonomy HTTP boundary", () => {
  it("accepts browser-facing localhost Host when Next uses an internal URL", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream())));
    const response = await POST(request(input(), { Host: "127.0.0.1:3177", Origin: "http://127.0.0.1:3177" }));
    expect(response.status).toBe(200); expect(response.headers.get("Cache-Control")).toBe("no-store");
  });

  it("accepts Vercel TLS termination using public Host and its trusted scheme", async () => {
    vi.stubEnv("VERCEL", "1"); vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream())));
    expect((await POST(request(input(), { Host: "jevduck.example", Origin: "https://jevduck.example", "x-forwarded-proto": "https" }))).status).toBe(200);
  });

  it("rejects cross-origin calls even with a forged forwarded host", async () => {
    const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    expect((await POST(request(input(), { Host: "jevduck.example", Origin: "http://attacker.example", "x-forwarded-host": "attacker.example" }))).status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects oversized bodies before validation or a model call", async () => {
    const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    expect((await POST(request({ ...input(), padding: "x".repeat(8192) }))).status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("never forwards upstream error text or credentials", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("fake-autonomy-server-key upstream debug", { status: 401 })));
    const response = await POST(request());
    expect(response.status).toBe(503);
    const text = await response.text(); expect(text).not.toContain("fake-autonomy-server-key"); expect(text).not.toContain("upstream debug");
  });

  it("rejects extra client question/model fields", async () => {
    expect((await POST(request({ ...input(), model: "some-other-model", questions: {} }))).status).toBe(400);
  });

  it("bounds repeated model requests from one IP", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(Response.json(upstream()))));
    const ip = `autonomy-rate-${++requestId}`;
    for (let i = 0; i < 20; i++) expect((await POST(request(input(), { "x-forwarded-for": ip }))).status).toBe(200);
    const limited = await POST(request(input(), { "x-forwarded-for": ip }));
    expect(limited.status).toBe(429); expect(limited.headers.get("Retry-After")).toBe("60");
  });
});

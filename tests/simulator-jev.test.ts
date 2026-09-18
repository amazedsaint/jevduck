import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GET, POST } from "../src/app/api/simulator/route";
import {
  buildSimulatorJevRequest, decideSimulatorWithJev, simulatorInputSchema, simulatorMessageClauses,
  SIMULATOR_JEV_ENDPOINT, SIMULATOR_JEV_QUESTIONS, SIMULATOR_JEV_TIMEOUT_MS,
} from "../src/lib/simulator-jev";
import { SIMULATOR_ACTIONS, SIMULATOR_SCENES, SIMULATOR_CAMERAS, type SimulatorAction, type SimulatorContext } from "../src/lib/simulator";

function context(): SimulatorContext {
  return { ready: true, loco: "legs", mode: "walk", busy: false, fallen: false, paused: false };
}
function input(message = "Look right, but keep your feet still") {
  return simulatorInputSchema.parse({ message, context: context() });
}
function answer(options: readonly string[], choice: string, selectedProbability = 1, confidence = selectedProbability) {
  return {
    type: "choice", choice, confidence,
    probabilities: Object.fromEntries(options.map((option) => [option,
      option === choice ? selectedProbability : (1 - selectedProbability) / (options.length - 1)])),
  };
}
function upstream(action: SimulatorAction = "look_right", selectedProbability = 1, confidence = selectedProbability) {
  const gate = action === "none" ? "none" : action === "clarify" ? "unsupported" : action === "stop" ? "stop" : "supported";
  return {
    model: "jev-1.13.0",
    answers: {
      gate: answer(["supported", "none", "unsupported", "stop"], gate),
      step1: answer(SIMULATOR_ACTIONS, action, selectedProbability, confidence),
      step2: answer(SIMULATOR_ACTIONS, "none"),
      step3: answer(SIMULATOR_ACTIONS, "none"),
      step4: answer(SIMULATOR_ACTIONS, "none"),
      step5: answer(SIMULATOR_ACTIONS, "none"),
      step6: answer(SIMULATOR_ACTIONS, "none"),
      step7: answer(SIMULATOR_ACTIONS, "none"),
      step8: answer(SIMULATOR_ACTIONS, "none"),
      step9: answer(SIMULATOR_ACTIONS, "none"),
      step10: answer(SIMULATOR_ACTIONS, "none"),
      step11: answer(SIMULATOR_ACTIONS, "none"),
      step12: answer(SIMULATOR_ACTIONS, "none"),
      scene: answer(SIMULATOR_SCENES, "keep"),
      camera: answer(SIMULATOR_CAMERAS, "keep"),
    },
  };
}
let requestId = 0;
function request(body: unknown = input(), headers: Record<string, string> = {}, url = "http://localhost:3000/api/simulator") {
  return new Request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-forwarded-for": `simulator-test-${++requestId}`, ...headers },
    body: JSON.stringify(body),
  });
}

beforeEach(() => {
  vi.stubEnv("TYPESAFE_API_KEY", "fake-simulator-key-never-send-to-client");
  vi.stubEnv("TYPESAFE_MODEL", "jev-latest");
  vi.stubEnv("VERCEL", "");
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("simulator Jev action boundary", () => {
  it("uses literal source clauses without counting ignored commands or breaking quotations", () => {
    expect(simulatorMessageClauses('I did not ask you to sit. Look right, then center your head.'))
      .toEqual(["I did not ask you to sit", "Look right", "center your head"]);
    expect(simulatorMessageClauses('The example is "walk forward, then sit". Look left.'))
      .toEqual(['The example is "walk forward, then sit"', "Look left"]);
    expect(simulatorMessageClauses("Don't sit; walk exactly 2.5 meters."))
      .toEqual(["Don't sit", "walk exactly 2.5 meters"]);
    const payload = buildSimulatorJevRequest(input("Moon scene. Look left then right."), "jev-latest");
    expect(payload.state.clauses).toEqual({ part_a: "Moon scene", part_b: "Look left", part_c: "right" });
  });

  it("sends isolated typed questions together and returns a bounded mission without motor commands", async () => {
    const fetchMock = vi.fn().mockResolvedValue(Response.json(upstream()));
    vi.stubGlobal("fetch", fetchMock);
    const result = await decideSimulatorWithJev(input());
    expect(result).toMatchObject({ action: "look_right", source: "jev", model: "jev-1.13.0", confidence: 1 });
    expect(result.latencyMs).toBeGreaterThanOrEqual(0);
    expect(result).toMatchObject({ plan: ["look_right"], scene: "keep", camera: "keep", disposition: "execute" });
    expect(JSON.stringify(result)).not.toContain("fake-simulator-key");
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe(SIMULATOR_JEV_ENDPOINT);
    expect(options.headers.Authorization).toBe("Bearer fake-simulator-key-never-send-to-client");
    expect(options.redirect).toBe("error");
    const payload = JSON.parse(options.body);
    expect(payload.questions).toEqual(SIMULATOR_JEV_QUESTIONS);
    expect(Object.keys(payload.questions)).toEqual(["gate", ...Array.from({ length: 12 }, (_, index) => `step${index + 1}`), "scene", "camera"]);
    expect(payload.state.visitor_message).toBe(input().message);
    expect(payload.state.controls.movement_duration_seconds).toBe(2);
  });

  it.each([
    ["none", "I did not ask you to sit"],
    ["stop", "Stop walking"],
    ["clarify", "Please do a happy dance"],
  ] as const)("keeps the meaning of %s without coercing it to a different action", async (action, message) => {
    // This checks response composition only. Live language probes verify
    // whether Jev distinguishes quoted/negated commands from positive ones.
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream(action))));
    expect((await decideSimulatorWithJev(input(message))).action).toBe(action);
  });

  it.each([
    [0.40, 0.95, "clarify"],
    [0.90, 0.40, "clarify"],
    [0.55, 0.55, "turn_left"],
  ] as const)("requires both selected probability %s and confidence %s to meet the gate", async (probability, confidence, expected) => {
    const raw = upstream("turn_left", probability, confidence);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    const result = await decideSimulatorWithJev(input("Turn that way"));
    expect(result.action).toBe(expected);
    expect(result.confidence).toBe(Math.min(probability, confidence));
    expect(result.probabilities).toEqual(raw.answers.step1.probabilities);
  });

  it("does not turn an intent classification into a claim that a busy runtime executed it", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream("walk_forward"))));
    const busyInput = { ...input("Walk forward"), context: { ...context(), busy: true, ready: false, paused: true } };
    const result = await decideSimulatorWithJev(busyInput);
    expect(result.action).toBe("walk_forward");
    expect(result).not.toHaveProperty("executed");
    expect(buildSimulatorJevRequest(busyInput, "jev-latest").state.context).toEqual(busyInput.context);
  });

  it.each(["Blue, sit", "Sage, sit", "Plum, sit", "Both ducks look left", "All four ducks kick the ball", "Stop the other duck"])("keeps an unsupported target gate from issuing a valid action subset: %s", async message => {
    const raw = upstream("sit");
    raw.answers.gate = answer(["supported", "none", "unsupported", "stop"], "unsupported");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    const value = input(message); value.context.selectedDuckId = "duck1";
    expect(await decideSimulatorWithJev(value)).toMatchObject({ plan: [], disposition: "clarify", interpretation: { reason: "unsupported_request" } });
  });

  it("composes four ordered robot actions with independent presentation choices", async () => {
    const raw = upstream("none");
    raw.answers.gate = answer(["supported", "none", "unsupported", "stop"], "supported");
    raw.answers.step3 = answer(SIMULATOR_ACTIONS, "stand");
    raw.answers.step4 = answer(SIMULATOR_ACTIONS, "walk_forward");
    raw.answers.step5 = answer(SIMULATOR_ACTIONS, "turn_left");
    raw.answers.step6 = answer(SIMULATOR_ACTIONS, "look_up");
    raw.answers.scene = answer(SIMULATOR_SCENES, "moon");
    raw.answers.camera = answer(SIMULATOR_CAMERAS, "follow");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input("Moon scene, follow camera. Stand, walk forward, turn left, then look up.")))
      .toMatchObject({ action: "stand", plan: ["stand", "walk_forward", "turn_left", "look_up"], scene: "moon", camera: "follow", disposition: "execute", interpretation: { stepCount: 4, reason: "accepted" } });
  });

  it("allows a camera-only mission with no fabricated robot movement", async () => {
    const raw = upstream("none");
    raw.answers.gate = answer(["supported", "none", "unsupported", "stop"], "supported");
    raw.answers.camera = answer(SIMULATOR_CAMERAS, "eyes");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input("Show me through your eyes.")))
      .toMatchObject({ action: "none", plan: [], scene: "keep", camera: "eyes", disposition: "execute" });
  });

  it("blocks the whole mission when a requested part is unsupported", async () => {
    const raw = upstream("walk_forward");
    raw.answers.gate = answer(["supported", "none", "unsupported", "stop"], "unsupported");
    raw.answers.scene = answer(SIMULATOR_SCENES, "sunset");
    raw.answers.camera = answer(SIMULATOR_CAMERAS, "follow");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input("Sunset, walk forward, then fly away.")))
      .toMatchObject({ action: "clarify", plan: [], scene: "keep", camera: "keep", disposition: "clarify", interpretation: { reason: "unsupported_request" } });
  });

  it("lets explicit Stop override speculative missions and presentation changes", async () => {
    const raw = upstream("walk_forward", 0.1, 0.05);
    raw.answers.gate = answer(["supported", "none", "unsupported", "stop"], "stop");
    raw.answers.scene = answer(SIMULATOR_SCENES, "moon", 0.4, 0.1);
    raw.answers.step5 = answer(SIMULATOR_ACTIONS, "sit");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input("Stop now. Cancel the moon walk.")))
      .toMatchObject({ action: "stop", plan: ["stop"], scene: "keep", camera: "keep", disposition: "execute", confidence: 1 });
  });

  it("ignores unused speculative answers when the whole message requests no action", async () => {
    const raw = upstream("sit");
    raw.answers.gate = answer(["supported", "none", "unsupported", "stop"], "none");
    raw.answers.scene = answer(SIMULATOR_SCENES, "moon");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input('The sentence "sit on the moon" is just a quotation.')))
      .toMatchObject({ action: "none", plan: [], scene: "keep", camera: "keep", disposition: "none" });
  });

  it("preserves source order around non-action clauses without inventing repeated actions", async () => {
    const raw = upstream("walk_forward");
    raw.answers.step3 = answer(SIMULATOR_ACTIONS, "look_left");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input("Walk forward. Do not sit. Then look left.")))
      .toMatchObject({ plan: ["walk_forward", "look_left"], disposition: "execute" });
  });

  it("blocks a mission when an intermediate source clause requests an unsupported action", async () => {
    const raw = upstream("walk_forward");
    raw.answers.step2 = answer(SIMULATOR_ACTIONS, "clarify");
    raw.answers.step3 = answer(SIMULATOR_ACTIONS, "look_left");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input("Walk forward, jump over the wall, then look left.")))
      .toMatchObject({ plan: [], scene: "keep", camera: "keep", disposition: "clarify" });
  });

  it("counts robot actions in code and rejects overflow even when all abilities are supported", async () => {
    const raw = upstream("walk_forward");
    raw.answers.step2 = answer(SIMULATOR_ACTIONS, "look_left");
    raw.answers.step3 = answer(SIMULATOR_ACTIONS, "turn_right");
    raw.answers.step4 = answer(SIMULATOR_ACTIONS, "look_up");
    raw.answers.step5 = answer(SIMULATOR_ACTIONS, "sit");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input("Walk forward, look left, turn right, look up, sit.")))
      .toMatchObject({ plan: [], disposition: "clarify", interpretation: { overflow: true, reason: "too_many_steps" } });
  });

  it("does not execute an easy first step when a later requested step is uncertain", async () => {
    const raw = upstream("walk_forward");
    raw.answers.step2 = answer(SIMULATOR_ACTIONS, "turn_left", 0.5, 0.4);
    raw.answers.scene = answer(SIMULATOR_SCENES, "moon");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input("Walk forward then turn that way.")))
      .toMatchObject({ plan: [], scene: "keep", camera: "keep", disposition: "clarify", confidence: 0.4 });
  });

  it("retains repeated explicitly requested actions and leaves automatic standing to the runtime", async () => {
    const raw = upstream("walk_forward");
    raw.answers.step2 = answer(SIMULATOR_ACTIONS, "walk_forward");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev({ ...input("Walk forward, then walk forward again."), context: { ...context(), mode: "sitstand" } }))
      .toMatchObject({ plan: ["walk_forward", "walk_forward"], disposition: "execute" });
  });

  it.each([
    ["Find the ball", "approach_ball"],
    ["Search for the ball behind you", "approach_ball"],
    ["Go kick the ball", "kick_ball"],
    ["Find the ball to kick it", "kick_ball"],
    ["Kick with the left foot", "kick_left"],
  ] as const)("keeps a typed task distinct from an open-loop gesture: %s", async (message, action) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream(action))));
    const result = await decideSimulatorWithJev(input(message));
    expect(result.plan).toEqual([action]);
    expect(result).not.toHaveProperty("executed");
    expect(buildSimulatorJevRequest(input(message), "jev-test").state.controls.ball_tasks).toContain("never automatic spawn_ball");
  });

  it("preserves approach then kick as two bounded tasks without guessed prerequisites", async () => {
    const raw = upstream("approach_ball"); raw.answers.step2 = answer(SIMULATOR_ACTIONS, "kick_ball");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev({ ...input("Find the ball and kick it"), context: { ...context(), mode: "sitstand", availableActions: ["stand"] } }))
      .toMatchObject({ plan: ["approach_ball", "kick_ball"], interpretation: { stepCount: 2 }, disposition: "execute" });
  });

  it("allows a requested model prerequisite before a task but never inserts it silently", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream("kick_ball"))));
    expect(await decideSimulatorWithJev({ ...input("Kick the ball"), context: { ...context(), loco: "rollers" } }))
      .toMatchObject({ plan: [], disposition: "clarify", interpretation: { reason: "unsupported_request" } });
    const raw = upstream("switch_to_legs"); raw.answers.step2 = answer(SIMULATOR_ACTIONS, "kick_ball");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev({ ...input("Use legs then kick the ball"), context: { ...context(), loco: "rollers", availableActions: ["switch_to_legs"] } }))
      .toMatchObject({ plan: ["switch_to_legs", "kick_ball"], disposition: "execute" });
  });

  it.each(["Find the ball and carry it", "Kick the ball through the goal", "Blue, find the ball", "Both ducks kick the ball"])("rejects the complete ball request when any part is unsupported: %s", async message => {
    const raw = upstream("kick_ball"); raw.answers.gate = answer(["supported", "none", "unsupported", "stop"], "unsupported");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input(message))).toMatchObject({ plan: [], disposition: "clarify" });
  });

  it("retains negation and Stop authority around a proposed ball task", async () => {
    const raw = upstream("none"); raw.answers.gate = answer(["supported", "none", "unsupported", "stop"], "supported");
    raw.answers.step2 = answer(SIMULATOR_ACTIONS, "approach_ball");
    const fetchMock = vi.fn().mockResolvedValue(Response.json(raw)); vi.stubGlobal("fetch", fetchMock);
    expect(await decideSimulatorWithJev(input("Do not kick the ball; approach it"))).toMatchObject({ plan: ["approach_ball"] });
    raw.answers.gate = answer(["supported", "none", "unsupported", "stop"], "stop");
    raw.answers.step1 = answer(SIMULATOR_ACTIONS, "kick_ball");
    fetchMock.mockResolvedValue(Response.json(raw));
    expect(await decideSimulatorWithJev(input("Stop; do not kick the ball"))).toMatchObject({ plan: ["stop"] });
  });

  it("rejects a roller sit even when the model approves the unsupported posture", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream("sit"))));
    expect(await decideSimulatorWithJev({ ...input("Sit"), context: { ...context(), loco: "rollers" } }))
      .toMatchObject({ plan: [], disposition: "clarify", interpretation: { reason: "unsupported_request" } });
  });

  it("ignores uncertainty in absent source clauses and unchanged presentation", async () => {
    const raw = upstream("look_left");
    raw.answers.step3 = answer(SIMULATOR_ACTIONS, "none", 0.1, 0.01);
    raw.answers.scene = answer(SIMULATOR_SCENES, "keep", 0.4, 0.1);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(raw)));
    expect(await decideSimulatorWithJev(input("Look left")))
      .toMatchObject({ plan: ["look_left"], disposition: "execute", confidence: 1 });
  });

  it("rejects invented actions and models, incomplete or contradictory distributions, and invalid scores", async () => {
    const invented = upstream(); invented.answers.step1.choice = "happy_dance";
    const missing = upstream(); delete missing.answers.step1.probabilities.stop;
    const contradictory = upstream(); contradictory.answers.step1.probabilities.look_right = 0; contradictory.answers.step1.probabilities.stop = 1;
    const unnormalized = upstream(); unnormalized.answers.step1.probabilities.stop = 0.6;
    const score = upstream(); score.answers.step1.confidence = 1.1;
    const model = upstream(); model.model = "<script>bad</script>";
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce(Response.json(invented))
      .mockResolvedValueOnce(Response.json(missing))
      .mockResolvedValueOnce(Response.json(contradictory))
      .mockResolvedValueOnce(Response.json(unnormalized))
      .mockResolvedValueOnce(Response.json(score))
      .mockResolvedValueOnce(Response.json(model)));
    for (let count = 0; count < 6; count++) {
      const response = await POST(request());
      expect(response.status).toBe(502);
      expect(await response.json()).toEqual({ error: expect.any(String) });
    }
  });

  it("never fabricates a fallback decision without a server key", async () => {
    vi.stubEnv("TYPESAFE_API_KEY", "");
    const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    const response = await POST(request());
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ error: expect.any(String) });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([401, 403, 429, 529, 500])("sanitizes upstream HTTP %i failures", async (status) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("fake-simulator-key-never-send-to-client", { status })));
    const response = await POST(request());
    expect(response.status).toBe(status === 500 ? 502 : 503);
    expect(await response.text()).not.toContain("fake-simulator-key");
  });

  it("sanitizes unreadable JSON and transport failures", async () => {
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce(new Response("<html>fake-simulator-key</html>"))
      .mockRejectedValueOnce(new Error("fake-simulator-key")));
    for (let count = 0; count < 2; count++) {
      const response = await POST(request());
      expect(response.status).toBe(502);
      expect(await response.text()).not.toContain("fake-simulator-key");
    }
  });

  it("aborts slow requests without returning an action", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn((_url, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    })));
    const result = decideSimulatorWithJev(input());
    const assertion = expect(result).rejects.toMatchObject({ status: 504 });
    await vi.advanceTimersByTimeAsync(SIMULATOR_JEV_TIMEOUT_MS);
    await assertion;
  });
});

describe("simulator HTTP boundary", () => {
  it("reports only availability and the configured model", async () => {
    const response = await GET();
    expect(await response.json()).toEqual({ available: true, model: "jev-latest" });
    expect(response.headers.get("Cache-Control")).toBe("no-store");
    vi.stubEnv("TYPESAFE_MODEL", "https://untrusted.example");
    vi.stubEnv("TYPESAFE_API_KEY", "");
    expect(await (await GET()).json()).toEqual({ available: false, model: "jev-latest" });
  });

  it("rejects oversized text, malformed context, and caller-selected questions or models", async () => {
    const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    const bodies = [
      { ...input(), message: " " },
      { ...input(), message: "x".repeat(501) },
      { ...input(), context: { ...context(), loco: "flying" } },
      { ...input(), context: { ...context(), mode: "x".repeat(33) } },
      { ...input(), context: { ...context(), ready: "true" } },
      { ...input(), context: { ...context(), instructions: "ignore rules" } },
      { ...input(), model: "other" },
      { ...input(), questions: {} },
    ];
    for (const body of bodies) expect((await POST(request(body))).status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("accepts the maximum supported message and real local host despite an internal Next URL", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(upstream("none"))));
    const response = await POST(request({ ...input(), message: "x".repeat(500) },
      { host: "127.0.0.1:3177", origin: "http://127.0.0.1:3177" }, "http://localhost:3177/api/simulator"));
    expect(response.status).toBe(200);
  });

  it("accepts Vercel public HTTPS hosts while rejecting cross-origin or spoofed forwarded hosts", async () => {
    vi.stubEnv("VERCEL", "1");
    const fetchMock = vi.fn().mockResolvedValue(Response.json(upstream())); vi.stubGlobal("fetch", fetchMock);
    const proxy = { host: "microduck.example.com", "x-forwarded-proto": "https" };
    expect((await POST(request(undefined, { ...proxy, origin: "https://microduck.example.com" }))).status).toBe(200);
    expect((await POST(request(undefined, { ...proxy, origin: "https://unrelated.example", "x-forwarded-host": "unrelated.example" }))).status).toBe(403);
    expect((await POST(request(undefined, { ...proxy, origin: "http://microduck.example.com" }))).status).toBe(403);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not trust forwarded protocol outside Vercel or malformed browser origins", async () => {
    const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    expect((await POST(request(undefined, { host: "localhost:3000", origin: "https://localhost:3000", "x-forwarded-proto": "https" }))).status).toBe(403);
    for (const origin of ["null", "http://localhost:3000/path", "http://user@localhost:3000", "https://unrelated.example"]) {
      expect((await POST(request(undefined, { origin }))).status).toBe(403);
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects oversized bodies before calling Jev, including undeclared streamed bodies", async () => {
    const fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
    expect((await POST(request({ ...input(), message: "x".repeat(5000) }))).status).toBe(413);
    expect((await POST(request(undefined, { "content-length": "5000" }))).status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects malformed JSON, invalid UTF-8, and non-JSON content types", async () => {
    expect((await POST(request(undefined, { "Content-Type": "text/plain" }))).status).toBe(415);
    for (const body of ["{broken", new Uint8Array([0xff])]) {
      const malformed = new Request("http://localhost:3000/api/simulator", {
        method: "POST", headers: { "Content-Type": "application/json" }, body,
      });
      expect((await POST(malformed)).status).toBe(400);
    }
  });

  it("caps repeated calls and releases the bucket after its window", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => Response.json(upstream())));
    const headers = { "x-forwarded-for": "simulator-rate-test" };
    for (let count = 0; count < 20; count++) expect((await POST(request(undefined, headers))).status).toBe(200);
    const limited = await POST(request(undefined, headers));
    expect(limited.status).toBe(429);
    expect(limited.headers.get("Retry-After")).toBe("60");
    await vi.advanceTimersByTimeAsync(60_001);
    expect((await POST(request(undefined, headers))).status).toBe(200);
  });
});

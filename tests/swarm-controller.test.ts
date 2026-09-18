import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SwarmController, type SwarmControllerOptions, type SwarmStatus } from "../src/lib/swarm-controller";
import { eligibleSwarmIntents, type SwarmDecision, type SwarmRuntime, type SwarmScenario } from "../src/lib/swarm-contract";
import { swarmDecision, swarmState } from "./swarm-fixtures";

function deferred<T>() { let resolve!: (value: T) => void; let reject!: (reason: unknown) => void; const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
function fixture(extra: Partial<SwarmControllerOptions> = {}) {
  const pending = deferred<SwarmDecision>();
  const request = vi.fn<SwarmControllerOptions["request"]>(() => pending.promise);
  const activate = vi.fn(), dispatch = vi.fn(), deactivate = vi.fn();
  const controller = new SwarmController({ request, activate, dispatch, deactivate, idFactory: () => "test", now: () => Date.now(), ...extra });
  let current = swarmState({ active: false, runId: null, scenario: null, members: 2 });
  const feed = (patch: Partial<SwarmRuntime> = {}, lifecycle: Partial<Omit<SwarmStatus, "swarm">> = {}) => {
    current = { ...current, seq: current.seq + 1, time: current.time + 0.25, ...lifecycle, swarm: { ...current.swarm, ...patch } };
    controller.updateStatus(current); return current;
  };
  const start = (scenario: SwarmScenario = "flock") => {
    feed(); controller.start(scenario);
    feed({ active: true, runId: controller.snapshot.runId, scenario, members: 4, phase: "idle", commandId: null, intent: null });
  };
  const beat = async (ms: number, patch: Partial<SwarmRuntime> = {}, lifecycle: Partial<Omit<SwarmStatus, "swarm">> = {}) => {
    for (let remaining = ms; remaining > 0;) { const step = Math.min(250, remaining); vi.advanceTimersByTime(step); feed(patch, lifecycle); await flush(); remaining -= step; }
  };
  const receipt = (patch: Partial<SwarmRuntime> = {}) => feed({ commandId: controller.snapshot.commandId, intent: controller.snapshot.currentDecision?.intent ?? "advance", phase: "running", ...patch });
  return { controller, pending, request, activate, dispatch, deactivate, feed, start, beat, receipt };
}
beforeEach(() => { vi.useFakeTimers(); vi.setSystemTime(0); });
afterEach(() => vi.useRealTimers());

describe("SwarmController ownership and measured receipts", () => {
  it("requires a fresh four-member run handshake before any Jev decision", async () => {
    const f = fixture(); f.feed(); f.controller.start("flock");
    expect(f.request).not.toHaveBeenCalled();
    f.feed({ active: true, members: 4, scenario: "flock", runId: "old-run" });
    expect(f.request).not.toHaveBeenCalled();
    f.feed({ runId: f.controller.snapshot.runId });
    expect(f.request).toHaveBeenCalledTimes(1);
    expect(f.activate).toHaveBeenCalledWith("flock", f.controller.snapshot.runId);
    expect(f.dispatch).not.toHaveBeenCalled();
  });
  it("allows bounded asynchronous scene setup but fails an unconfirmed activation", () => {
    const f = fixture({ activationTimeoutMs: 5_000 }); f.controller.start("split");
    f.feed({}, { ready: false }); vi.advanceTimersByTime(4_000);
    expect(f.controller.snapshot.active).toBe(true);
    vi.advanceTimersByTime(1_000); expect(f.controller.snapshot.phase).toBe("error"); expect(f.deactivate).toHaveBeenCalledTimes(1);
  });
  it("accepts the explicit setup reset but stops a later unrequested simulation reset", () => {
    const f = fixture(); f.feed({}, { time: 100 }); f.controller.start("gather");
    f.feed({ active: true, runId: f.controller.snapshot.runId, scenario: "gather", members: 4 }, { time: 0.25 });
    expect(f.controller.snapshot.active).toBe(true); expect(f.request).toHaveBeenCalledTimes(1);
    f.feed({}, { time: 0 }); expect(f.controller.snapshot.active).toBe(false); expect(f.controller.snapshot.reason).toContain("world was reset");
  });
  it("ignores an old model answer after stop or scenario replacement", async () => {
    const f = fixture(); f.start(); const signal = f.request.mock.calls[0][1];
    f.controller.stop("User stopped."); expect(signal.aborted).toBe(true);
    f.controller.start("convoy"); const newRun = f.controller.snapshot.runId;
    f.pending.resolve(swarmDecision()); await flush();
    expect(f.dispatch).not.toHaveBeenCalled(); expect(f.controller.snapshot.runId).toBe(newRun);
  });
  it("dispatches one bounded intent and waits for its correlated terminal receipt", async () => {
    const f = fixture(); f.start(); f.pending.resolve(swarmDecision()); await flush();
    const id = f.controller.snapshot.commandId!;
    expect(f.dispatch).toHaveBeenCalledWith("advance", id, f.controller.snapshot.runId);
    f.feed({ phase: "complete", intent: "advance", commandId: "other-command", progressM: 0.2 });
    expect(f.controller.snapshot.completedCycles).toBe(0);
    f.feed({ phase: "complete", intent: "advance", commandId: id, progressM: 0.2, targetErrorM: 0.35 });
    expect(f.controller.snapshot.completedCycles).toBe(1);
    expect(f.controller.snapshot.recent[0]).toMatchObject({ outcome: "complete", progressM: 0.2, targetErrorAfterM: 0.35 });
    expect(f.request).toHaveBeenCalledTimes(1);
  });
  it("never launches overlapping decisions while a native group instruction runs", async () => {
    const f = fixture(); f.start(); f.pending.resolve(swarmDecision()); await flush(); f.receipt();
    await f.beat(12_000);
    expect(f.controller.snapshot.phase).toBe("acting"); expect(f.request).toHaveBeenCalledTimes(1); expect(f.dispatch).toHaveBeenCalledTimes(1);
  });
  it("compares residual errors only after the current instruction has assigned its targets", async () => {
    const f = fixture(); f.start("gather"); f.feed({ targetErrorM: 0.02 });
    f.pending.resolve(swarmDecision("regroup")); await flush();
    f.receipt({ targetErrorM: 0.65 }); f.receipt({ targetErrorM: 0.4 });
    f.receipt({ phase: "complete", targetErrorM: 0.1 });
    expect(f.controller.snapshot.recent[0]).toMatchObject({ targetErrorBeforeM: 0.65, targetErrorAfterM: 0.1 });
    const g = fixture(); g.start("gather"); g.feed({ targetErrorM: 0.02 });
    g.pending.resolve(swarmDecision("regroup")); await flush(); g.receipt({ phase: "complete", targetErrorM: 0.1 });
    expect(g.controller.snapshot.recent[0]).toMatchObject({ targetErrorBeforeM: null, targetErrorAfterM: 0.1 });
  });
  it("requires an acknowledgement and a bounded terminal outcome", async () => {
    const f = fixture(); f.start(); f.pending.resolve(swarmDecision()); await flush();
    await f.beat(3_000); expect(f.controller.snapshot.phase).toBe("error"); expect(f.controller.snapshot.reason).toContain("acknowledge");
    const g = fixture({ intentTimeoutMs: 5_000 }); g.start(); g.pending.resolve(swarmDecision()); await flush(); g.receipt();
    await g.beat(5_000); expect(g.controller.snapshot.phase).toBe("error"); expect(g.controller.snapshot.reason).toContain("bounded outcome");
  });
  it("does not confuse acceptance ACKs with completion and ignores stale rejection ACKs", async () => {
    const f = fixture(); f.start(); f.pending.resolve(swarmDecision()); await flush();
    const { runId, commandId } = f.controller.snapshot;
    f.controller.acknowledgeIntent("previous-run", commandId!, false, "Old rejection");
    f.controller.acknowledgeIntent(runId!, "previous-command", false, "Old rejection");
    f.controller.acknowledgeIntent(runId!, commandId!, true);
    expect(f.controller.snapshot.active).toBe(true); expect(f.controller.snapshot.completedCycles).toBe(0);
    f.receipt({ phase: "complete", targetErrorM: 0.4 });
    expect(f.controller.snapshot.completedCycles).toBe(1);
  });
  it("fails immediately on a correlated runtime rejection", async () => {
    const f = fixture(); f.controller.start("gather");
    f.controller.acknowledgeRun(f.controller.snapshot.runId!, false, "Scenario unavailable.");
    expect(f.controller.snapshot).toMatchObject({ active: false, phase: "error", reason: "Scenario unavailable." });
    const g = fixture(); g.start(); g.pending.resolve(swarmDecision()); await flush();
    g.controller.acknowledgeIntent(g.controller.snapshot.runId!, g.controller.snapshot.commandId!, false, "Peers are too close.");
    expect(g.controller.snapshot).toMatchObject({ active: false, phase: "error", reason: "Peers are too close." });
  });
  it("keeps a model abstention stationary and respects the minimum request interval", async () => {
    const f = fixture({ requestIntervalMs: 1 }); f.start(); f.pending.resolve({ ...swarmDecision("hold"), abstained: true, confidence: 0.3 }); await flush();
    await f.beat(7_750); expect(f.dispatch).not.toHaveBeenCalled(); expect(f.request).toHaveBeenCalledTimes(1);
    await f.beat(250); expect(f.request).toHaveBeenCalledTimes(2); expect(f.controller.snapshot.completedCycles).toBe(0);
  });
  it("keeps a failed movement receipt through repeated quiet observations", async () => {
    const request = vi.fn<SwarmControllerOptions["request"]>().mockResolvedValue(swarmDecision("hold"));
    request.mockResolvedValueOnce(swarmDecision("advance"));
    const f = fixture({ request }); f.start(); await flush();
    f.receipt({ targetErrorM: 0.28 }); f.receipt({ phase: "complete", progressM: 0.02, targetErrorM: 0.3 });
    await f.beat(64_000);
    expect(request.mock.calls.length).toBeGreaterThan(6);
    expect(f.controller.snapshot.recent).toHaveLength(1);
    expect(f.controller.snapshot.recent[0]).toMatchObject({ intent: "advance", progressM: 0.02, targetErrorAfterM: 0.3 });
    for (const [input] of request.mock.calls.slice(1)) expect(eligibleSwarmIntents(input)).not.toContain("advance");
    expect(f.dispatch).toHaveBeenCalledTimes(1);
  });
  it("withdraws a decision if runtime capabilities changed during the model call", async () => {
    const f = fixture(); f.start(); f.feed({ availableIntents: ["hold"] });
    f.pending.resolve(swarmDecision()); await flush(); expect(f.dispatch).not.toHaveBeenCalled();
  });
  it.each([{ paused: true }, { manual: true }, { suspended: true }])("cancels for lifecycle/manual authority: %j", async lifecycle => {
    const f = fixture(); f.start(); f.pending.resolve(swarmDecision()); await flush(); f.receipt(); f.feed({}, lifecycle);
    expect(f.controller.snapshot.active).toBe(false); expect(f.deactivate).toHaveBeenCalledTimes(1);
  });
  it("stops if the runtime loses or replaces its run ID", async () => {
    const f = fixture(); f.start(); const signal = f.request.mock.calls[0][1]; f.feed({ runId: "another-run" });
    expect(f.controller.snapshot.active).toBe(false); expect(signal.aborted).toBe(true);
  });
  it("stops on stale observations or fresh sequence numbers with a frozen simulator", async () => {
    const f = fixture(); f.start(); vi.advanceTimersByTime(2_000);
    expect(f.controller.snapshot.reason).toContain("stale");
    const g = fixture(); g.start(); const time = g.feed().time;
    await g.beat(2_000, {}, { time }); expect(g.controller.snapshot.reason).toContain("stopped advancing");
  });
  it("does not fabricate a decision after an unavailable API", async () => {
    const f = fixture(); f.start(); f.pending.reject(new Error("upstream secret")); await flush();
    expect(f.controller.snapshot.phase).toBe("error"); expect(f.controller.snapshot.reason).not.toContain("upstream secret");
    expect(f.dispatch).not.toHaveBeenCalled(); expect(f.deactivate).toHaveBeenCalledTimes(1);
  });
  it("invalidates dispatch when an onChange callback synchronously cancels it", async () => {
    let controller!: SwarmController;
    const f = fixture({ onChange: snapshot => { if (snapshot.phase === "acting") controller.stop("Stop in callback."); } });
    controller = f.controller; f.start(); f.pending.resolve(swarmDecision()); await flush();
    expect(f.dispatch).not.toHaveBeenCalled(); expect(f.controller.snapshot.active).toBe(false);
  });
  it("rechecks freshness after an onChange callback before dispatch", async () => {
    const f = fixture({ onChange: snapshot => { if (snapshot.phase === "acting") vi.setSystemTime(3_000); } });
    f.start(); f.pending.resolve(swarmDecision()); await flush();
    expect(f.dispatch).not.toHaveBeenCalled(); expect(f.controller.snapshot.phase).toBe("error");
  });
  it("retains rate limiting across explicit restarts but clears prior scenario history", async () => {
    const f = fixture(); f.start(); f.pending.resolve(swarmDecision()); await flush(); f.receipt({ phase: "blocked" });
    expect(f.controller.snapshot.blockedCycles).toBe(1);
    f.controller.start("gather"); f.feed({ scenario: "gather", runId: f.controller.snapshot.runId, phase: "idle", commandId: null, intent: null });
    expect(f.controller.snapshot.recent).toEqual([]); expect(f.request).toHaveBeenCalledTimes(1);
    await f.beat(8_000); expect(f.request).toHaveBeenCalledTimes(2);
  });
});

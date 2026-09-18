import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutonomyController, type AutonomyControllerOptions, type AutonomySnapshot, type AutonomyStatus } from "../src/lib/autonomy-controller";
import { AUTONOMY_LABELS, AUTONOMY_PLANS, type AutonomyBehavior, type AutonomyDecision } from "../src/lib/autonomy-contract";
import { MissionRunner, type MissionSnapshot } from "../src/lib/mission-runner";
import { SIMULATOR_EXECUTABLE_ACTIONS, type SimulatorExecutableAction } from "../src/lib/simulator";

const decision = (behavior: AutonomyBehavior = "stroll"): AutonomyDecision => ({
  behavior, plan: [...AUTONOMY_PLANS[behavior]] as AutonomyDecision["plan"], label: AUTONOMY_LABELS[behavior], reason: "A supported next behavior.",
  source: "jev", model: "jev-test", confidence: 0.9, latencyMs: 100, alternatives: [{ behavior, probability: 1 }],
});
const mission = (id: string, status: MissionSnapshot["status"] = "running", plan: SimulatorExecutableAction[] = ["walk_forward"]): MissionSnapshot => ({
  id, status, index: 0, steps: plan.map(action => ({ action, status })),
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const flush = async () => { for (let n = 0; n < 5; n++) await Promise.resolve(); };

function fixture(overrides: Partial<AutonomyControllerOptions> = {}) {
  const pending = deferred<AutonomyDecision>();
  const request = vi.fn<AutonomyControllerOptions["request"]>(() => pending.promise);
  let missionCount = 0, seq = 0, simTime = 0;
  const execute = vi.fn<AutonomyControllerOptions["execute"]>(plan => mission(`mission-${++missionCount}`, "running", plan));
  const cancelMission = vi.fn(), stopMotion = vi.fn(), setRuntimeAutonomy = vi.fn();
  const changes: AutonomySnapshot[] = [];
  const controller = new AutonomyController({ request, execute, cancelMission, stopMotion, setRuntimeAutonomy,
    now: () => Date.now(), onChange: snapshot => changes.push(snapshot), ...overrides });
  const status = (extra: Partial<AutonomyStatus> = {}): AutonomyStatus => {
    simTime += 0.25;
    const value: AutonomyStatus = { ready: true, busy: false, paused: false, fallen: false, loco: "legs", mode: "walk", seq: ++seq,
      time: simTime, position: [0, 0, 0.15], headingRad: 0, posture: "standing", spatialValid: true,
      clearance: { front: 1.5, back: 1.5, left: 1.5, right: 1.5 }, guardReason: null, guardSeq: 0, autonomyActive: true, ...extra };
    controller.updateStatus(value);
    return value;
  };
  const start = (mode: "explore" | "observe" = "explore") => {
    status({ autonomyActive: false }); controller.start(mode); status();
  };
  const beat = async (milliseconds: number, extra: Partial<AutonomyStatus> | ((tick: number) => Partial<AutonomyStatus>) = {}) => {
    let left = milliseconds, tick = 0;
    while (left > 0) {
      const step = Math.min(left, 250); vi.advanceTimersByTime(step);
      status(typeof extra === "function" ? extra(tick++) : extra);
      await flush(); left -= step;
    }
  };
  return { controller, pending, request, execute, cancelMission, stopMotion, setRuntimeAutonomy, changes, status, start, beat };
}

function wiredFixture() {
  let controller!: AutonomyController;
  const dispatch = vi.fn<(action: SimulatorExecutableAction, id: string) => void>();
  const stopMotion = vi.fn();
  const missionChanges: MissionSnapshot[] = [];
  const runner = new MissionRunner({ dispatch, onChange: snapshot => {
    missionChanges.push(snapshot);
    controller.updateMission(snapshot);
    // The host also issues Stop for runner failures. A guard must prevent
    // that failure callback, rather than merely ignore its final snapshot.
    if (snapshot.status === "failed") stopMotion();
  } });
  const f = fixture({ execute: plan => runner.start(plan), cancelMission: reason => { runner.cancel(reason); }, stopMotion });
  controller = f.controller;
  const feed = (extra: Partial<AutonomyStatus> = {}) => {
    const status = f.status(extra); // The host feeds autonomy BEFORE the runner.
    runner.updateStatus(status);
    return status;
  };
  const activate = (mode: "explore" | "observe" | "play" = "explore", extra: Partial<AutonomyStatus> = {}) => {
    feed({ ...extra, autonomyActive: false }); controller.start(mode); feed(extra);
  };
  const beat = async (milliseconds: number, extra: Partial<AutonomyStatus> = {}) => {
    for (let left = milliseconds; left > 0;) {
      const step = Math.min(left, 250); vi.advanceTimersByTime(step); feed(extra); await flush(); left -= step;
    }
  };
  return { ...f, runner, dispatch, stopMotion, missionChanges, feed, activate, beat };
}

beforeEach(() => { vi.useFakeTimers(); vi.setSystemTime(0); });
afterEach(() => vi.useRealTimers());

describe("AutonomyController", () => {
  it("waits for a fresh runtime handshake and actual idle status", async () => {
    const f = fixture();
    f.status({ autonomyActive: true }); // A previous session's true is not a new handshake.
    f.controller.start("explore");
    expect(f.setRuntimeAutonomy).toHaveBeenLastCalledWith(true);
    expect(f.request).not.toHaveBeenCalled();
    f.status({ autonomyActive: false }); // Stop's intervening status is expected.
    expect(f.controller.snapshot.active).toBe(true);
    f.status({ busy: true });
    expect(f.request).not.toHaveBeenCalled();
    expect(f.controller.snapshot.phase).toBe("waiting");
    f.status();
    expect(f.request).toHaveBeenCalledTimes(1);
    await f.beat(1_000);
    expect(f.request).toHaveBeenCalledTimes(1);
    expect(f.execute).not.toHaveBeenCalled();
  });

  it("aborts a request on Stop and ignores its late decision", async () => {
    const f = fixture(); f.start();
    const signal = f.request.mock.calls[0][1];
    f.controller.stop("User stopped.");
    expect(signal.aborted).toBe(true);
    expect(f.setRuntimeAutonomy).toHaveBeenLastCalledWith(false);
    f.pending.resolve(decision()); await flush();
    f.status(); vi.advanceTimersByTime(30_000);
    expect(f.execute).not.toHaveBeenCalled();
    expect(f.controller.snapshot.active).toBe(false);
    expect(f.stopMotion).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("preserves live manual input while revoking autonomous ownership", async () => {
    const f = fixture(); f.start(); f.pending.resolve(decision()); await flush();
    f.controller.stop("Manual controls took over.", false);
    expect(f.cancelMission).toHaveBeenCalledTimes(1);
    expect(f.setRuntimeAutonomy).toHaveBeenLastCalledWith(false);
    expect(f.stopMotion).not.toHaveBeenCalled();
    expect(f.controller.snapshot.recent[0].outcome).toBe("interrupted");
    f.controller.updateMission(mission("mission-1", "completed"));
    expect(f.controller.snapshot.completedCycles).toBe(0);
  });

  it("cannot apply an old response to a restarted mode or bypass the rate cap", async () => {
    const f = fixture(); const second = deferred<AutonomyDecision>();
    f.request.mockImplementationOnce(() => f.pending.promise).mockImplementationOnce(() => second.promise);
    f.start(); f.controller.stop(); f.controller.start("observe"); f.status();
    expect(f.request).toHaveBeenCalledTimes(1);
    await f.beat(8_000);
    expect(f.request).toHaveBeenCalledTimes(2);
    expect(f.request.mock.calls[1][0].mode).toBe("observe");
    f.pending.resolve(decision()); await flush();
    expect(f.execute).not.toHaveBeenCalled();
    second.resolve(decision("wait")); await flush();
    expect(f.controller.snapshot.phase).toBe("observing");
  });

  it("owns one mission and counts its completion once", async () => {
    const f = fixture(); f.start(); f.pending.resolve(decision()); await flush();
    expect(f.execute).toHaveBeenCalledTimes(1);
    f.controller.updateMission(mission("somebody-else", "completed"));
    await f.beat(10_000);
    expect(f.request).toHaveBeenCalledTimes(1);
    expect(f.controller.snapshot.completedCycles).toBe(0);
    f.controller.updateMission(mission("mission-1", "completed"));
    f.controller.updateMission(mission("mission-1", "completed"));
    expect(f.controller.snapshot.completedCycles).toBe(1);
    expect(f.controller.snapshot.recent).toHaveLength(1);
    await f.beat(1_499);
    expect(f.request).toHaveBeenCalledTimes(1);
    await f.beat(1);
    expect(f.request).toHaveBeenCalledTimes(2);
  });

  it("rechecks current wall geometry when a delayed decision arrives", async () => {
    const f = fixture(); f.start();
    f.status({ clearance: { front: 0.2, back: 1.5, left: 1.5, right: 1.5 } });
    f.pending.resolve(decision("stroll")); await flush();
    expect(f.execute).not.toHaveBeenCalled();
    expect(f.controller.snapshot.completedCycles).toBe(0);
    expect(f.controller.snapshot.blockedCycles).toBe(1);
    expect(f.controller.snapshot.recent).toEqual([{ behavior: "stroll", outcome: "blocked", distanceM: 0 }]);
    await f.beat(8_000, { clearance: { front: 0.2, back: 1.5, left: 1.5, right: 1.5 } });
    expect(f.request.mock.calls[1][0].memory.recent[0].outcome).toBe("blocked");
  });

  it("consumes a wall guard once and replans despite the latched reason", async () => {
    const f = fixture(); const next = deferred<AutonomyDecision>();
    f.request.mockImplementationOnce(() => f.pending.promise).mockImplementationOnce(() => next.promise);
    f.start(); f.pending.resolve(decision()); await flush();
    f.status({ guardSeq: 1, guardReason: "Wall ahead", clearance: { front: 0.1, back: 1.5, left: 1.5, right: 1.5 } });
    f.controller.updateMission(mission("mission-1", "completed"));
    f.controller.updateMission(mission("mission-1", "cancelled"));
    expect(f.cancelMission).toHaveBeenCalledTimes(1);
    expect(f.stopMotion).not.toHaveBeenCalled();
    expect(f.controller.snapshot.active).toBe(true);
    expect(f.controller.snapshot.completedCycles).toBe(0);
    expect(f.controller.snapshot.blockedCycles).toBe(1);
    await f.beat(8_000, { guardSeq: 1, guardReason: "Wall ahead", clearance: { front: 0.1, back: 1.5, left: 1.5, right: 1.5 } });
    expect(f.request).toHaveBeenCalledTimes(2);
    expect(f.controller.snapshot.blockedCycles).toBe(1);
    next.resolve(decision("turn_left")); await flush();
    expect(f.execute).toHaveBeenLastCalledWith(["turn_left"]);
    f.status({ guardSeq: 1, guardReason: "Wall ahead" });
    expect(f.controller.snapshot.phase).toBe("acting");
    expect(f.cancelMission).toHaveBeenCalledTimes(1);
  });

  it("handles a real runner's guard cancellation before the rejection ACK and continues choosing", async () => {
    const f = wiredFixture();
    f.request.mockImplementationOnce(() => f.pending.promise).mockResolvedValueOnce(decision("turn_left"));
    f.activate(); f.pending.resolve(decision("stroll")); await flush();
    const rejectedId = f.dispatch.mock.calls[0][1];
    const firstMissionId = f.runner.snapshot!.id;
    expect(f.dispatch.mock.calls.map(([action]) => action)).toEqual(["walk_forward"]);
    expect(f.controller.snapshot.missionId).toBe(firstMissionId);

    // The bridge publishes the fresh guard observation BEFORE the negative
    // command ACK. cancelMission synchronously calls runner -> controller.
    const guard = f.feed({ guardSeq: 1, guardReason: "Wall ahead", clearance: { front: 0.1, back: 1.5, left: 1.5, right: 1.5 } });
    expect(f.runner.snapshot!.status).toBe("cancelled");
    expect(f.missionChanges.map(snapshot => snapshot.status)).toEqual(["queued", "running", "cancelled"]);
    f.runner.acknowledge(rejectedId, false, "Wall ahead", { completion: "immediate", statusSeq: guard.seq });
    expect(f.missionChanges.some(snapshot => snapshot.status === "failed")).toBe(false);
    expect(f.stopMotion).not.toHaveBeenCalled();
    expect(f.controller.snapshot.active).toBe(true);
    expect(f.controller.snapshot.completedCycles).toBe(0);
    expect(f.controller.snapshot.blockedCycles).toBe(1);
    expect(f.controller.snapshot.recent).toEqual([{ behavior: "stroll", outcome: "blocked", distanceM: 0 }]);

    await f.beat(8_000, { guardSeq: 1, guardReason: "Wall ahead", clearance: { front: 0.1, back: 1.5, left: 1.5, right: 1.5 } });
    expect(f.request).toHaveBeenCalledTimes(2);
    expect(f.dispatch.mock.calls.map(([action]) => action)).toEqual(["walk_forward", "turn_left"]);
    expect(f.runner.snapshot!.id).not.toBe(firstMissionId);
    const turnId = f.dispatch.mock.calls[1][1];
    const beforeTurn = f.feed({ guardSeq: 1 });
    f.runner.acknowledge(turnId, true, "Accepted", { completion: "status", statusSeq: beforeTurn.seq });
    f.feed({ guardSeq: 1, busy: true }); f.feed({ guardSeq: 1 });
    expect(f.runner.snapshot!.status).toBe("completed");
    expect(f.controller.snapshot.completedCycles).toBe(1);
    expect(f.controller.snapshot.blockedCycles).toBe(1);
    expect(f.controller.snapshot.recent.map(episode => episode.outcome)).toEqual(["blocked", "completed"]);
    expect(f.stopMotion).not.toHaveBeenCalled();
  });

  it("a real runner's early manual status cancels queued head actions without eating held input", async () => {
    const f = wiredFixture(); f.activate(); f.pending.resolve(decision("look_around")); await flush();
    const firstId = f.dispatch.mock.calls[0][1];
    const beforePulse = f.feed();
    f.runner.acknowledge(firstId, true, "Accepted", { completion: "status", statusSeq: beforePulse.seq });
    f.feed({ busy: true });
    expect(f.runner.snapshot!.steps).toHaveLength(3);

    // Status can beat the separate manual notification across the bridge.
    // Pilot cancellation calls back through the real runner in this stack.
    const manual = f.feed({ manual: true, autonomyActive: false, busy: true });
    expect(f.controller.snapshot.active).toBe(false);
    expect(f.controller.snapshot.reason).toBe("Manual controls took over.");
    expect(f.runner.snapshot!.status).toBe("cancelled");
    expect(f.runner.snapshot!.steps.every(step => step.status === "cancelled")).toBe(true);
    expect(f.controller.snapshot.recent[0].outcome).toBe("interrupted");
    expect(f.stopMotion).not.toHaveBeenCalled();
    expect(f.setRuntimeAutonomy).toHaveBeenLastCalledWith(false);

    f.runner.acknowledge(firstId, true, "Late acceptance", { completion: "immediate", statusSeq: manual.seq });
    // The subsequent host manual event follows the same no-Stop path.
    f.controller.stop("You took the controls.", false); f.runner.cancel("You took the controls.");
    await f.beat(8_000, { manual: true, autonomyActive: false, busy: true });
    expect(f.dispatch.mock.calls.map(([action]) => action)).toEqual(["look_left"]);
    expect(f.controller.snapshot.completedCycles).toBe(0);
    expect(f.request).toHaveBeenCalledTimes(1);
    expect(f.missionChanges.some(snapshot => snapshot.status === "failed")).toBe(false);
    expect(f.stopMotion).not.toHaveBeenCalled();
  });

  it("accepts synchronous queued/running/completed callbacks before execute returns its ID", async () => {
    let controller!: AutonomyController;
    const execute = vi.fn<AutonomyControllerOptions["execute"]>(plan => {
      controller.updateMission(mission("old-manual", "cancelled"));
      controller.updateMission(mission("sync", "queued", plan));
      controller.updateMission(mission("sync", "running", plan));
      controller.updateMission(mission("sync", "completed", plan));
      return mission("sync", "running", plan);
    });
    const f = fixture({ execute }); controller = f.controller;
    f.start(); f.pending.resolve(decision()); await flush();
    expect(controller.snapshot.completedCycles).toBe(1);
    expect(controller.snapshot.missionId).toBeNull();
    expect(controller.snapshot.phase).toBe("cooldown");
  });

  it("does not acquire a mission returned after a synchronous Stop", async () => {
    let controller!: AutonomyController;
    const f = fixture({ execute: plan => { controller.stop("Stopped inside execute."); return mission("late", "running", plan); } });
    controller = f.controller;
    f.start(); f.pending.resolve(decision()); await flush();
    expect(controller.snapshot.active).toBe(false);
    expect(controller.snapshot.missionId).toBeNull();
  });

  it("rechecks geometry changed by a synchronous acting callback", async () => {
    let f!: ReturnType<typeof fixture>;
    f = fixture({ onChange: snapshot => {
      if (snapshot.phase === "acting") f.status({ clearance: { front: 0.1, back: 1.5, left: 1.5, right: 1.5 } });
    } });
    f.start(); f.pending.resolve(decision()); await flush();
    expect(f.execute).not.toHaveBeenCalled();
    expect(f.controller.snapshot.blockedCycles).toBe(1);
  });

  it.each(["choosing", "acting"] as const)("a Stop from the %s callback prevents the next side effect", async phase => {
    let controller!: AutonomyController;
    const f = fixture({ onChange: snapshot => { if (snapshot.phase === phase) controller.stop("Callback stopped."); } });
    controller = f.controller;
    f.start(); f.pending.resolve(decision()); await flush();
    expect(controller.snapshot.active).toBe(false);
    expect(f.execute).not.toHaveBeenCalled();
    if (phase === "choosing") expect(f.request).not.toHaveBeenCalled();
  });

  it("a completion callback cannot restart a request after stopping the loop", async () => {
    let controller!: AutonomyController;
    const f = fixture({ onChange: snapshot => { if (snapshot.completedCycles) controller.stop("Enough observing."); } });
    controller = f.controller;
    f.start(); await f.beat(6_000); f.pending.resolve(decision("wait")); await flush();
    await f.beat(3_000);
    expect(controller.snapshot.active).toBe(false);
    expect(f.request).toHaveBeenCalledTimes(1);
  });

  it("fails closed when status reaches two seconds old, including duplicate receipts", async () => {
    const f = fixture(); f.start(); const status = f.status();
    vi.advanceTimersByTime(1_999); f.controller.updateStatus(status);
    expect(f.controller.snapshot.active).toBe(true);
    vi.advanceTimersByTime(1);
    expect(f.controller.snapshot.phase).toBe("error");
    expect(f.controller.snapshot.reason).toMatch(/stale/);
    expect(f.request.mock.calls[0][1].aborted).toBe(true);
    f.pending.resolve(decision()); await flush();
    expect(f.execute).not.toHaveBeenCalled();
  });

  it("does not mistake fresh bridge receipts for fresh physics when simulated time freezes", async () => {
    const f = fixture(); f.start(); const time = f.status().time;
    await f.beat(2_000, { time });
    expect(f.controller.snapshot.active).toBe(false);
    expect(f.controller.snapshot.reason).toMatch(/stopped advancing/);
    f.pending.resolve(decision()); await flush();
    expect(f.execute).not.toHaveBeenCalled();
  });

  it("times out an unconfirmed runtime without selecting a fallback behavior", async () => {
    const f = fixture(); f.controller.start("explore");
    await f.beat(5_000, { autonomyActive: false });
    expect(f.controller.snapshot.phase).toBe("error");
    expect(f.controller.snapshot.reason).toMatch(/confirm/);
    expect(f.request).not.toHaveBeenCalled();
    expect(f.execute).not.toHaveBeenCalled();
  });

  it.each([
    { paused: true }, { fallen: true }, { ready: false }, { spatialValid: false },
    { suspended: true }, { error: "Policy failed" }, { autonomyActive: false },
  ] satisfies Partial<AutonomyStatus>[])("stops explicitly on lifecycle failure %j", async extra => {
    const f = fixture(); f.start(); f.status(extra);
    f.pending.resolve(decision()); await flush();
    expect(f.controller.snapshot.active).toBe(false);
    expect(f.stopMotion).toHaveBeenCalledTimes(1);
    expect(f.execute).not.toHaveBeenCalled();
    expect(f.setRuntimeAutonomy).toHaveBeenLastCalledWith(false);
  });

  it("handles a manual status without issuing a Stop over the user's input", () => {
    const f = fixture(); f.start(); f.status({ manual: true });
    expect(f.controller.snapshot.active).toBe(false);
    expect(f.stopMotion).not.toHaveBeenCalled();
  });

  it("rejects nonfinite geometry instead of replacing it with invented clearance", async () => {
    const f = fixture(); f.start(); f.status({ headingRad: NaN });
    f.pending.resolve(decision()); await flush();
    expect(f.controller.snapshot.reason).toMatch(/invalid spatial/);
    expect(f.execute).not.toHaveBeenCalled();
  });

  it("stops on API failure or invalid behavior plans without a local fallback", async () => {
    const failed = fixture(); failed.start(); failed.pending.reject(new Error("Jev unavailable")); await flush();
    expect(failed.controller.snapshot.phase).toBe("error");
    expect(failed.controller.snapshot.reason).toBe("Jev unavailable");
    expect(failed.execute).not.toHaveBeenCalled();
    const invalid = fixture(); invalid.start(); invalid.pending.resolve({ ...decision("stroll"), plan: ["sit"] }); await flush();
    expect(invalid.controller.snapshot.phase).toBe("error");
    expect(invalid.execute).not.toHaveBeenCalled();
  });

  it("times out an unresponsive Jev request while fresh status keeps arriving", async () => {
    const f = fixture(); f.start(); await f.beat(12_000);
    expect(f.controller.snapshot.reason).toMatch(/decision in time/);
    expect(f.request.mock.calls[0][1].aborted).toBe(true);
    expect(f.execute).not.toHaveBeenCalled();
  });

  it.each(["failed", "cancelled"] as const)("stops after an owned mission is %s", async state => {
    const f = fixture(); f.start(); f.pending.resolve(decision()); await flush();
    f.controller.updateMission(mission("mission-1", state));
    expect(f.controller.snapshot.active).toBe(false);
    expect(f.controller.snapshot.completedCycles).toBe(0);
    expect(f.controller.snapshot.recent[0].outcome).toBe(state === "failed" ? "failed" : "interrupted");
    await f.beat(16_000);
    expect(f.request).toHaveBeenCalledTimes(1);
  });

  it("repeated wait decisions use timed observations and the eight-second request cap", async () => {
    const request = vi.fn<AutonomyControllerOptions["request"]>().mockResolvedValue(decision("wait"));
    const f = fixture({ request, requestIntervalMs: 100 }); f.start("observe"); await flush();
    expect(f.controller.snapshot.completedCycles).toBe(0);
    await f.beat(2_999); expect(f.controller.snapshot.completedCycles).toBe(0);
    await f.beat(1); expect(f.controller.snapshot.completedCycles).toBe(1);
    await f.beat(4_999); expect(request).toHaveBeenCalledTimes(1);
    await f.beat(1); expect(request).toHaveBeenCalledTimes(2);
    await f.beat(75_000);
    expect(f.execute).not.toHaveBeenCalled();
    expect(f.controller.snapshot.completedCycles).toBe(11);
    expect(f.controller.snapshot.recent).toHaveLength(8);
    expect(request.mock.calls.every(call => call[0].memory.recent.length <= 8)).toBe(true);
  });

  it("keeps bounded visited cells and measures motion without duplicate-time or teleport inflation", async () => {
    const f = fixture(); f.start(); f.pending.resolve(decision()); await flush();
    f.status({ position: [0.1, 0, 0.15] });
    const same = f.status({ position: [0.2, 0, 0.15] });
    f.status({ position: [0.25, 0, 0.15], time: same.time });
    f.status({ position: [1, 0, 0.15] }); // Teleport-sized jump is excluded.
    expect(f.controller.snapshot.distanceM).toBeCloseTo(0.2);
    f.controller.updateMission(mission("mission-1", "completed"));
    expect(f.controller.snapshot.recent[0].distanceM).toBeCloseTo(0.2);
    for (let n = 0; n < 75; n++) f.status({ position: [n * 0.3, 0, 0.15] });
    expect(f.controller.snapshot.visited).toHaveLength(64);
    const last = f.controller.snapshot.visited.at(-1)!;
    f.status({ position: [74 * 0.3, 0, 0.15] });
    expect(f.controller.snapshot.visited.at(-1)!.visits).toBe(last.visits);
  });

  it("clears old iframe status and memory on reset while preserving the request interval", async () => {
    const f = fixture(); f.start(); f.pending.resolve(decision("wait")); await flush(); await f.beat(3_000);
    f.controller.reset();
    expect(f.controller.snapshot.recent).toHaveLength(0);
    expect(f.controller.snapshot.visited).toHaveLength(0);
    expect(f.controller.snapshot.completedCycles).toBe(0);
    f.controller.start("observe"); f.status({ seq: 1, time: 0 });
    expect(f.controller.snapshot.active).toBe(true);
    expect(f.request).toHaveBeenCalledTimes(1);
  });

  it("does not allow callback exceptions or reentrant cleanup to retain runtime authority", async () => {
    let controller!: AutonomyController;
    const f = fixture({ onChange: () => { throw new Error("UI broke"); }, cancelMission: () => controller.start("observe") });
    controller = f.controller;
    f.start(); f.pending.resolve(decision()); await flush(); controller.stop();
    expect(controller.snapshot.active).toBe(false);
    expect(f.setRuntimeAutonomy).toHaveBeenLastCalledWith(false);
    expect(f.stopMotion).toHaveBeenCalledTimes(1);
  });

  it("disposes timers and requests, and cannot be started again", async () => {
    const f = fixture(); f.start(); f.controller.dispose(); f.controller.start("explore");
    f.pending.resolve(decision()); await flush();
    expect(f.execute).not.toHaveBeenCalled();
    expect(f.controller.snapshot.active).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("cancels a pending decision and clears memory when the selected physical duck changes", async () => {
    const f = fixture(); f.status({ selectedDuckId: "duck1", autonomyActive: false });
    f.controller.start("play"); f.status({ selectedDuckId: "duck1", availableActions: [...SIMULATOR_EXECUTABLE_ACTIONS] });
    expect(f.request.mock.calls[0][0].state.selectedDuckId).toBe("duck1");
    const signal = f.request.mock.calls[0][1];
    f.status({ selectedDuckId: "duck2", availableActions: [...SIMULATOR_EXECUTABLE_ACTIONS] });
    expect(signal.aborted).toBe(true);
    f.pending.resolve(decision("roll")); await flush();
    expect(f.execute).not.toHaveBeenCalled();
    expect(f.controller.snapshot).toMatchObject({ active: false, visited: [], recent: [], distanceM: 0 });
  });

  it("rechecks a removed runtime capability immediately before executing Jev's choice", async () => {
    const f = fixture(); f.status({ autonomyActive: false }); f.controller.start("play");
    f.status({ availableActions: [...SIMULATOR_EXECUTABLE_ACTIONS] });
    f.status({ availableActions: ["look_left"] });
    f.pending.resolve(decision("roll")); await flush();
    expect(f.execute).not.toHaveBeenCalled();
    expect(f.controller.snapshot.recent[0]).toMatchObject({ behavior: "roll", outcome: "blocked" });
  });

  it("retains a ball goal across stages and records its correlated physical outcome", async () => {
    const f = wiredFixture();
    const context = { availableActions: [...SIMULATOR_EXECUTABLE_ACTIONS], ball: { present: true, distanceM: 1.4, bearingRad: 2.6 } };
    f.activate("play", context);
    f.pending.resolve(decision("kick_ball")); await flush();
    const commandId = f.dispatch.mock.calls[0][1];
    expect(f.dispatch.mock.calls[0][0]).toBe("kick_ball");
    f.runner.acknowledge(commandId, true, "Goal accepted.", { completion: "status", statusSeq: 2 });
    const task = { commandId, action: "kick_ball" as const, phase: "searching" as const, outcome: null, reason: "Computing an approach route.", elapsedS: 0, ballContact: false, ballDisplacementM: 0 };
    await f.beat(17_000, { ...context, busy: true, task });
    expect(f.request).toHaveBeenCalledTimes(1);
    expect(f.controller.snapshot.completedCycles).toBe(0);
    f.feed({ ...context, busy: true, task: { ...task, phase: "verifying", elapsedS: 18 } });
    f.feed({ ...context, ball: { present: true, distanceM: 0.6, bearingRad: 0.1 }, task: { ...task, phase: "complete", outcome: "succeeded", elapsedS: 19, ballContact: true, ballDisplacementM: 0.42 } });
    expect(f.runner.snapshot?.status).toBe("completed");
    expect(f.controller.snapshot.recent[0]).toMatchObject({ behavior: "kick_ball", outcome: "completed", ballDistanceBeforeM: 1.4, ballDistanceAfterM: 0.6, ballContact: true, ballDisplacementM: 0.42, taskOutcome: "succeeded" });
    expect(f.controller.snapshot.completedCycles).toBe(1);
  });

  it("honors a Jev stop decision through the independent cancellation path", async () => {
    const f = fixture(); f.status({ autonomyActive: false }); f.controller.start("play");
    f.status({ availableActions: [...SIMULATOR_EXECUTABLE_ACTIONS] });
    f.pending.resolve(decision("stop")); await flush();
    expect(f.controller.snapshot).toMatchObject({ active: false, reason: "Jev chose to stop autonomy." });
    expect(f.stopMotion).toHaveBeenCalledTimes(1);
    expect(f.execute).not.toHaveBeenCalled();
    await f.beat(20_000); expect(f.request).toHaveBeenCalledTimes(1);
  });

  it("allows a bounded, explicitly owned model-loading pause while status remains fresh", async () => {
    const f = fixture(); f.status({ autonomyActive: false }); f.controller.start("play");
    const current = f.status({ loco: "rollers", ball: { present: true, distanceM: 1, bearingRad: 0 }, availableActions: [...SIMULATOR_EXECUTABLE_ACTIONS] });
    f.pending.resolve(decision("switch_to_legs")); await flush();
    const switching = { busy: true, time: current.time, phase: "switching_locomotion" };
    f.status(switching);
    await f.beat(29_999, switching);
    expect(f.controller.snapshot.active).toBe(true);
    await f.beat(1, switching);
    expect(f.controller.snapshot.active).toBe(false);
    expect(f.controller.snapshot.reason).toContain("stopped advancing");
  });

  it("never grants a stalled walking mission the model-switch loading exception", async () => {
    const f = fixture(); f.start(); f.pending.resolve(decision("stroll")); await flush();
    const current = f.status();
    await f.beat(2_000, { busy: true, time: current.time, phase: "switching_locomotion" });
    expect(f.controller.snapshot.active).toBe(false);
  });

  it("still requires fresh bridge messages during an owned model load", async () => {
    const f = fixture(); f.status({ autonomyActive: false }); f.controller.start("play");
    const current = f.status({ loco: "rollers", ball: { present: true, distanceM: 1, bearingRad: 0 }, availableActions: [...SIMULATOR_EXECUTABLE_ACTIONS] });
    f.pending.resolve(decision("switch_to_legs")); await flush();
    f.status({ busy: true, time: current.time, phase: "switching_locomotion" });
    vi.advanceTimersByTime(2_000);
    expect(f.controller.snapshot.active).toBe(false);
    expect(f.controller.snapshot.reason).toContain("stale");
  });

});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MissionRunner, type MissionAckReceipt, type MissionSnapshot, type MissionStatus } from "../src/lib/mission-runner";
import type { SimulatorAction } from "../src/lib/simulator";
import type { SimulatorTask } from "../src/lib/autonomy-contract";

const idle = (seq: number, extra: Partial<MissionStatus> = {}): MissionStatus => ({ seq, ready: true, busy: false, paused: false, fallen: false, mode: "walk", ...extra });

function fixture(options: Partial<ConstructorParameters<typeof MissionRunner>[0]> = {}) {
  const dispatch = vi.fn();
  const changes: MissionSnapshot[] = [];
  const runner = new MissionRunner({ dispatch, onChange: snapshot => changes.push(snapshot), idFactory: () => "test", ...options });
  const id = (index = dispatch.mock.calls.length - 1): string => dispatch.mock.calls[index][1];
  const ack = (receipt: MissionAckReceipt = { completion: "status", statusSeq: 1 }, accepted = true) => runner.acknowledge(id(), accepted, accepted ? "Accepted" : "Blocked", receipt);
  return { runner, dispatch, changes, id, ack };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

const ballTask = (commandId: string, extra: Partial<SimulatorTask> = {}): SimulatorTask => ({
  commandId, action: "kick_ball", phase: "approaching", outcome: null,
  reason: "Approaching measured ball position.", elapsedS: 2, ballContact: false, ballDisplacementM: 0, ...extra,
});

describe("measured ball tasks", () => {
  it("does not finish on idle or another command's success receipt", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["kick_ball", "sit"]); f.ack();
    f.runner.updateStatus(idle(2, { busy: true, task: ballTask(f.id()) }));
    f.runner.updateStatus(idle(3));
    f.runner.updateStatus(idle(4, { task: ballTask("old-command", { phase: "complete", outcome: "succeeded", ballContact: true, ballDisplacementM: 0.3 }) }));
    expect(f.dispatch.mock.calls.map(call => call[0])).toEqual(["kick_ball"]);
    expect(f.runner.snapshot?.status).toBe("running");
    f.runner.updateStatus(idle(5, { task: ballTask(f.id(), { phase: "complete", outcome: "succeeded", ballContact: true, ballDisplacementM: 0.3, reason: "Foot contact displaced the ball 0.30 m." }) }));
    expect(f.dispatch.mock.calls.map(call => call[0])).toEqual(["kick_ball", "sit"]);
    expect(f.runner.snapshot?.steps[0].detail).toContain("0.30 m");
  });

  it.each([
    { ballContact: false, ballDisplacementM: 0.3 },
    { ballContact: true, ballDisplacementM: 0.01 },
  ])("rejects a claimed kick success without contact and meaningful ball motion: %o", evidence => {
    const f = fixture();
    f.runner.updateStatus(idle(1)); f.runner.start(["kick_ball", "quack"]); f.ack();
    f.runner.updateStatus(idle(2, { busy: true, task: ballTask(f.id()) }));
    f.runner.updateStatus(idle(3, { task: ballTask(f.id(), { phase: "complete", outcome: "succeeded", ...evidence }) }));
    expect(f.runner.snapshot?.status).toBe("failed");
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    expect(f.runner.snapshot?.reason).toContain("verified ball contact");
  });

  it("propagates a blocked target instead of executing the remaining sequence", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1)); f.runner.start(["kick_ball", "look_left"]); f.ack();
    f.runner.updateStatus(idle(2, { task: ballTask(f.id(), { phase: "failed", outcome: "failed", reason: "No ball is present in the world." }) }));
    expect(f.runner.snapshot?.status).toBe("failed");
    expect(f.runner.snapshot?.reason).toBe("No ball is present in the world.");
    expect(f.runner.snapshot?.steps[1].status).toBe("cancelled");
    expect(f.dispatch).toHaveBeenCalledTimes(1);
  });

  it("accepts a correlated approach completed between samples without inventing busy or contact", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1)); f.runner.start(["approach_ball"]); f.ack({ completion: "status", statusSeq: 1 });
    f.runner.updateStatus(idle(2, { task: ballTask(f.id(), { action: "approach_ball", phase: "complete", outcome: "succeeded", reason: "Reached ball staging pose." }) }));
    expect(f.runner.snapshot?.status).toBe("completed");
  });

  it("retains a multi-stage goal beyond primitive timeout but still enforces its bound", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1)); f.runner.start(["kick_ball"]); f.ack();
    f.runner.updateStatus(idle(2, { busy: true, task: ballTask(f.id(), { phase: "searching" }) }));
    vi.advanceTimersByTime(16_000);
    expect(f.runner.snapshot?.status).toBe("running");
    vi.advanceTimersByTime(164_000);
    expect(f.runner.snapshot?.status).toBe("failed");
    expect(f.runner.snapshot?.reason).toContain("finishing");
  });

  it("keeps a cancelled goal cancelled when its delayed success arrives", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1)); f.runner.start(["kick_ball"]); f.ack();
    const commandId = f.id();
    f.runner.updateStatus(idle(2, { task: ballTask(commandId, { phase: "cancelled", outcome: "cancelled", reason: "World configuration changed." }) }));
    f.runner.updateStatus(idle(3, { task: ballTask(commandId, { phase: "complete", outcome: "succeeded", ballContact: true, ballDisplacementM: 1 }) }));
    expect(f.runner.snapshot?.status).toBe("cancelled");
    expect(f.dispatch).toHaveBeenCalledTimes(1);
  });
});

describe("MissionRunner", () => {
  it("runs each action only after its ACK and a fresh busy-to-idle cycle", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["walk_forward", "look_left", "sit"]);
    expect(f.dispatch.mock.calls.map(call => call[0])).toEqual(["walk_forward"]);
    f.ack();
    f.runner.updateStatus(idle(2)); // Fresh, but the movement never began.
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    f.runner.updateStatus(idle(3, { busy: true }));
    f.runner.updateStatus(idle(4));
    expect(f.dispatch.mock.calls.map(call => call[0])).toEqual(["walk_forward", "look_left"]);
    expect(f.runner.snapshot?.index).toBe(1);
    f.ack({ completion: "status", statusSeq: 4 });
    f.runner.updateStatus(idle(5, { busy: true }));
    f.runner.updateStatus(idle(6));
    f.ack({ completion: "status", statusSeq: 6 });
    f.runner.updateStatus(idle(7, { busy: true, mode: "sitstand" }));
    f.runner.updateStatus(idle(8, { mode: "sitstand" }));
    expect(f.runner.snapshot?.status).toBe("completed");
    expect(f.runner.snapshot?.steps.map(step => step.status)).toEqual(["completed", "completed", "completed"]);
    expect(vi.getTimerCount()).toBe(0);
  });

  it.each(["stop", "center_head", "stand", "sit"] as const)("allows an immediate %s receipt without inventing a busy phase", action => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start([action, "look_right"]);
    f.ack({ completion: "immediate", statusSeq: 1 });
    expect(f.dispatch).toHaveBeenCalledTimes(1); // ACK alone is insufficient.
    f.runner.updateStatus(idle(2));
    expect(f.dispatch.mock.calls.map(call => call[0])).toEqual([action, "look_right"]);
  });

  it("does not treat center_head as immediate when its actual receipt describes a pulse", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["center_head", "walk_forward"]);
    f.ack();
    f.runner.updateStatus(idle(2));
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    f.runner.updateStatus(idle(3, { busy: true }));
    f.runner.updateStatus(idle(4));
    expect(f.dispatch).toHaveBeenCalledTimes(2);
  });

  it("requires a fresh idle status even when Stop leaves a posture transition running", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["stop", "look_left"]);
    f.ack({ completion: "immediate", statusSeq: 1 });
    f.runner.updateStatus(idle(2, { busy: true }));
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    f.runner.updateStatus(idle(3));
    expect(f.dispatch).toHaveBeenCalledTimes(2);
  });

  it("dispatches an explicit Stop while the simulator is already busy", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1, { busy: true }));
    f.runner.start(["stop"]);
    expect(f.dispatch.mock.calls.map(call => call[0])).toEqual(["stop"]);
    f.ack({ completion: "immediate", statusSeq: 1 });
    f.runner.updateStatus(idle(2));
    expect(f.runner.snapshot?.status).toBe("completed");
  });

  it("uses the ACK sequence barrier rather than stale in-flight idle/busy messages", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["walk_forward", "sit"]);
    f.runner.updateStatus(idle(2, { busy: true }));
    f.runner.updateStatus(idle(3));
    f.ack({ completion: "status", statusSeq: 3 });
    f.runner.updateStatus(idle(4));
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    f.runner.updateStatus(idle(5, { busy: true }));
    f.runner.updateStatus(idle(4)); // Old idle cannot end this step.
    f.runner.updateStatus(idle(5)); // Duplicate cannot end it either.
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    f.runner.updateStatus(idle(6));
    expect(f.dispatch).toHaveBeenCalledTimes(2);
  });

  it("retains valid completion evidence if status is delivered before the ACK", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["walk_forward", "sit"]);
    f.runner.updateStatus(idle(2, { busy: true }));
    f.runner.updateStatus(idle(3));
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    f.ack({ completion: "status", statusSeq: 1 });
    expect(f.dispatch).toHaveBeenCalledTimes(2);
  });

  it("does not reuse the prior step's idle status to complete a following immediate step", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["stop", "stand", "walk_forward"]);
    f.ack({ completion: "immediate", statusSeq: 1 });
    f.runner.updateStatus(idle(2));
    f.ack({ completion: "immediate", statusSeq: 2 });
    expect(f.dispatch).toHaveBeenCalledTimes(2);
    f.runner.updateStatus(idle(3));
    expect(f.dispatch).toHaveBeenCalledTimes(3);
  });

  it("keeps seated-to-walk preparation inside one running step until the bridge becomes idle", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1, { mode: "sitstand" }));
    f.runner.start(["walk_forward", "sit"]);
    f.ack();
    f.runner.updateStatus(idle(2, { mode: "sitstand", busy: true }));
    f.runner.updateStatus(idle(3, { mode: "walk", busy: true }));
    f.runner.updateStatus(idle(4, { mode: "walk", busy: true }));
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    f.runner.updateStatus(idle(5));
    expect(f.dispatch).toHaveBeenCalledTimes(2);
  });

  it("cancels on pause and cannot advance when delayed status or ACKs arrive", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["walk_forward", "sit"]);
    const staleId = f.id();
    f.runner.updateStatus(idle(2, { paused: true }));
    f.runner.acknowledge(staleId, true, "Accepted", { completion: "immediate", statusSeq: 1 });
    f.runner.updateStatus(idle(3));
    vi.advanceTimersByTime(60_000);
    expect(f.runner.snapshot?.status).toBe("cancelled");
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    expect(f.runner.snapshot?.steps.every(step => step.status === "cancelled")).toBe(true);
  });

  it("manual takeover cancels a mission, including one queued behind busy state", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1, { busy: true }));
    f.runner.start(["look_left"]);
    f.runner.updateStatus(idle(2, { manual: true }));
    f.runner.updateStatus(idle(3));
    expect(f.runner.snapshot?.status).toBe("cancelled");
    expect(f.dispatch).not.toHaveBeenCalled();
  });

  it("invalidates old IDs on replacement, even with a colliding injected ID factory", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["walk_forward", "sit"]);
    const oldId = f.id();
    f.runner.start(["look_right"]);
    expect(f.id()).not.toBe(oldId);
    f.runner.acknowledge(oldId, true, "Old reply", { completion: "immediate", statusSeq: 1 });
    f.runner.updateStatus(idle(2));
    expect(f.runner.snapshot?.status).toBe("running");
    expect(f.runner.snapshot?.steps[0].action).toBe("look_right");
  });

  it("ignores a stale injected timer callback after cancellation and replacement", () => {
    const callbacks: Array<() => void> = [];
    const f = fixture({ timers: {
      setTimeout: callback => { callbacks.push(callback); return callbacks.length - 1; },
      clearTimeout: () => {}, // Deliberately emulate an already-queued timer callback.
    } });
    f.runner.updateStatus(idle(1));
    f.runner.start(["walk_forward"]);
    const oldCallbacks = [...callbacks];
    f.runner.cancel();
    f.runner.start(["look_right"]);
    for (const callback of oldCallbacks) callback();
    expect(f.runner.snapshot?.status).toBe("running");
    expect(f.runner.snapshot?.steps[0].action).toBe("look_right");
  });

  it("rejects malformed programs before dispatching any prefix", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    for (const plan of [[], ["walk_forward", "none"], ["clarify"], ["stop", "stop", "stop", "stop", "stop"], ["reset"]]) {
      f.runner.start(plan as SimulatorAction[]);
      expect(f.runner.snapshot?.status).toBe("failed");
    }
    expect(f.dispatch).not.toHaveBeenCalled();
  });

  it("stops the remaining program after a rejected ACK or a dispatch exception", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["walk_forward", "sit"]);
    f.ack({}, false);
    expect(f.runner.snapshot?.status).toBe("failed");
    expect(f.runner.snapshot?.steps.map(step => step.status)).toEqual(["failed", "cancelled"]);
    f.runner.updateStatus(idle(2));
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    const thrown = fixture({ dispatch: () => { throw new Error("Frame closed"); } });
    thrown.runner.updateStatus(idle(1));
    thrown.runner.start(["look_up"]);
    expect(thrown.runner.snapshot?.reason).toBe("Frame closed");
  });

  it.each(["ack", "completion", "queue"] as const)("fails a %s timeout without advancing", kind => {
    const f = fixture({ ackTimeoutMs: 100, stepTimeoutMs: 200, queueTimeoutMs: 300 });
    f.runner.updateStatus(idle(1, { busy: kind === "queue" }));
    f.runner.start(["walk_forward", "sit"]);
    if (kind === "completion") f.ack();
    vi.advanceTimersByTime(301);
    expect(f.runner.snapshot?.status).toBe("failed");
    expect(f.dispatch).toHaveBeenCalledTimes(kind === "queue" ? 0 : 1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("fails when the active simulator disappears or falls", () => {
    for (const status of [idle(2, { ready: false }), idle(2, { fallen: true })]) {
      const f = fixture();
      f.runner.updateStatus(idle(1));
      f.runner.start(["walk_forward", "sit"]);
      f.runner.updateStatus(status);
      expect(f.runner.snapshot?.status).toBe("failed");
      expect(f.dispatch).toHaveBeenCalledTimes(1);
    }
  });

  it("queues without a status, then starts only on ready and idle", () => {
    const f = fixture();
    f.runner.start(["look_left"]);
    expect(f.runner.snapshot?.status).toBe("queued");
    f.runner.updateStatus(idle(1, { ready: false }));
    f.runner.updateStatus(idle(2, { busy: true }));
    expect(f.dispatch).not.toHaveBeenCalled();
    f.runner.updateStatus(idle(3));
    expect(f.dispatch).toHaveBeenCalledTimes(1);
  });

  it("clears iframe sequence state on reset and ignores the old command receipt", () => {
    const f = fixture();
    f.runner.updateStatus(idle(900));
    f.runner.start(["look_left"]);
    const oldId = f.id();
    f.runner.resetStatus();
    f.runner.start(["look_right"]);
    expect(f.dispatch).toHaveBeenCalledTimes(1);
    f.runner.updateStatus(idle(0));
    expect(f.dispatch).toHaveBeenCalledTimes(2);
    f.runner.acknowledge(oldId, true, "Old", { completion: "immediate", statusSeq: 900 });
    expect(f.runner.snapshot?.status).toBe("running");
  });

  it("rejects malformed receipts and ignores unrelated or duplicate ACKs", () => {
    const f = fixture();
    f.runner.updateStatus(idle(1));
    f.runner.start(["walk_forward"]);
    f.runner.acknowledge("unknown", false);
    expect(f.runner.snapshot?.status).toBe("running");
    f.ack({ completion: "status", statusSeq: 0 });
    expect(f.runner.snapshot?.status).toBe("failed");
    const good = fixture();
    good.runner.updateStatus(idle(1));
    good.runner.start(["stop"]);
    good.ack({ completion: "immediate", statusSeq: 1 });
    good.runner.acknowledge(good.id(), false, "Duplicate rejection");
    good.runner.updateStatus(idle(2));
    expect(good.runner.snapshot?.status).toBe("completed");
  });

  it("isolates throwing observers and supports cancellation during a running notification", () => {
    const f = fixture({ onChange: () => { throw new Error("Renderer failed"); } });
    f.runner.updateStatus(idle(1));
    expect(() => f.runner.start(["walk_forward"])).not.toThrow();
    expect(() => f.runner.cancel()).not.toThrow();
    expect(vi.getTimerCount()).toBe(0);
    const dispatch = vi.fn();
    let runner: MissionRunner;
    runner = new MissionRunner({ dispatch, onChange: snapshot => { if (snapshot.status === "running") runner.cancel("User intervened"); } });
    runner.updateStatus(idle(1));
    runner.start(["walk_forward"]);
    expect(dispatch).not.toHaveBeenCalled();
    expect(runner.snapshot?.status).toBe("cancelled");
  });

  it("does not use stale idle state if an observer supplies a newer busy status between steps", () => {
    const dispatch = vi.fn();
    let runner: MissionRunner;
    runner = new MissionRunner({ dispatch, onChange: snapshot => {
      if (snapshot.index === 1 && snapshot.steps[1].status === "queued") runner.updateStatus(idle(3, { busy: true }));
    } });
    runner.updateStatus(idle(1));
    runner.start(["stop", "walk_forward"]);
    runner.acknowledge(dispatch.mock.calls[0][1], true, "", { completion: "immediate", statusSeq: 1 });
    runner.updateStatus(idle(2));
    expect(dispatch).toHaveBeenCalledTimes(1);
    runner.updateStatus(idle(4));
    expect(dispatch).toHaveBeenCalledTimes(2);
  });

  it("does not expose mutable internal step state through snapshots", () => {
    const f = fixture();
    f.runner.start(["look_left"]);
    const leaked = f.runner.snapshot!;
    (leaked.steps[0] as { status: string }).status = "completed";
    expect(f.runner.snapshot?.steps[0].status).toBe("queued");
  });
  it("uses fresh capabilities after a mode switch rather than rejecting its later action early", () => {
    const f = fixture(); f.runner.updateStatus(idle(1, { selectedDuckId: "duck1", availableActions: ["switch_to_rollers"] }));
    f.runner.start(["switch_to_rollers", "crouch"]);
    f.ack({ completion: "status", statusSeq: 1 });
    f.runner.updateStatus(idle(2, { busy: true, selectedDuckId: "duck1", availableActions: [] }));
    f.runner.updateStatus(idle(3, { selectedDuckId: "duck1", availableActions: ["crouch"] }));
    expect(f.dispatch.mock.calls.map(call => call[0])).toEqual(["switch_to_rollers", "crouch"]);
  });

  it("rejects an unavailable native action without dispatch and keeps Stop available", () => {
    const f = fixture(); f.runner.updateStatus(idle(1, { availableActions: ["quack"] }));
    f.runner.start(["kick_left"]);
    expect(f.runner.snapshot?.status).toBe("failed"); expect(f.dispatch).not.toHaveBeenCalled();
    f.runner.start(["stop"]);
    expect(f.dispatch.mock.calls.map(call => call[0])).toEqual(["stop"]);
  });

  it("does not hand an unfinished duck1 mission to duck2 after target selection", () => {
    const f = fixture(); f.runner.updateStatus(idle(1, { selectedDuckId: "duck1" }));
    f.runner.start(["walk_forward", "quack"]); f.ack();
    f.runner.updateStatus(idle(2, { busy: true, selectedDuckId: "duck1" }));
    f.runner.updateStatus(idle(3, { selectedDuckId: "duck2" }));
    expect(f.runner.snapshot?.status).toBe("cancelled");
    expect(f.dispatch.mock.calls.map(call => call[0])).toEqual(["walk_forward"]);
  });

});

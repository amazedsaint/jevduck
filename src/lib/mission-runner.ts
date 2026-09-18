import { SIMULATOR_ACTIONS, SIMULATOR_ACTION_LABELS, type SimulatorAction, type SimulatorDuckId } from "./simulator";
import type { SimulatorTask } from "./autonomy-contract";

export type MissionAction = Exclude<SimulatorAction, "none" | "clarify">;
export type MissionState = "queued" | "running" | "completed" | "cancelled" | "failed";

export interface MissionStep {
  readonly action: MissionAction;
  readonly status: MissionState;
  readonly id?: string;
  readonly detail?: string;
}

export interface MissionSnapshot {
  readonly id: string;
  readonly status: MissionState;
  /** Zero-based current step; terminal snapshots retain the final active index. */
  readonly index: number;
  readonly steps: readonly MissionStep[];
  readonly reason?: string;
}

export interface MissionStatus {
  /** Increasing within one iframe session. Call resetStatus() after reload. */
  seq: number;
  ready: boolean;
  busy: boolean;
  paused: boolean;
  fallen: boolean;
  manual?: boolean;
  error?: string;
  mode?: string;
  time?: number;
  selectedDuckId?: SimulatorDuckId;
  availableActions?: readonly MissionAction[];
  /** Composite tasks report their own correlated, measured terminal outcome. */
  task?: SimulatorTask | null;
}

export interface MissionAckReceipt {
  /** Immediate means no busy phase is required, not that status can be skipped. */
  completion?: "immediate" | "status";
  /** Last status sequence emitted by the bridge before acknowledging this command. */
  statusSeq?: number;
}

export interface MissionTimers {
  setTimeout(callback: () => void, milliseconds: number): unknown;
  clearTimeout(handle: unknown): void;
}

export interface MissionRunnerOptions {
  dispatch(action: MissionAction, id: string): void;
  onChange?(snapshot: MissionSnapshot): void;
  timers?: MissionTimers;
  idFactory?: () => string;
  queueTimeoutMs?: number;
  ackTimeoutMs?: number;
  stepTimeoutMs?: number;
}

type MutableStep = { action: MissionAction; status: MissionState; id?: string; detail?: string };
type ActiveStep = {
  id: string;
  dispatchSeq: number;
  acknowledged: boolean;
  completion: "immediate" | "status";
  barrier: number;
  latestBusySeq: number;
};
type Run = {
  id: string;
  status: MissionState;
  index: number;
  steps: MutableStep[];
  reason?: string;
  active: ActiveStep | null;
};

const EXECUTABLE = new Set<string>(SIMULATOR_ACTIONS.filter(action => action !== "none" && action !== "clarify"));
const terminal = (state: MissionState) => state === "completed" || state === "cancelled" || state === "failed";
const isBallTask = (action: string) => action === "approach_ball" || action === "kick_ball";
const defaultTimers: MissionTimers = {
  setTimeout: (callback, milliseconds) => globalThis.setTimeout(callback, milliseconds),
  clearTimeout: handle => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
};
let fallbackId = 0;

/**
 * Sequences bounded simulator commands, independently of React and physics.
 *
 * A receipt is acceptance, not completion. A status command requires fresh
 * busy -> idle evidence after its receipt's sequence barrier. A no-op or Stop
 * receipt marked immediate still requires a fresh, ready, idle status.
 *
 * Cancellation only discards the program. The caller owns its explicit Stop,
 * pause or manual command, so cancelling cannot override direct user controls.
 */
export class MissionRunner {
  private readonly timers: MissionTimers;
  private readonly queueTimeoutMs: number;
  private readonly ackTimeoutMs: number;
  private readonly stepTimeoutMs: number;
  private run: Run | null = null;
  private latestStatus: MissionStatus | null = null;
  private timer: { handle: unknown } | null = null;
  private generation = 0;

  constructor(private readonly options: MissionRunnerOptions) {
    this.timers = options.timers ?? defaultTimers;
    this.queueTimeoutMs = options.queueTimeoutMs ?? 15_000;
    this.ackTimeoutMs = options.ackTimeoutMs ?? 3_000;
    this.stepTimeoutMs = options.stepTimeoutMs ?? 15_000;
    for (const timeout of [this.queueTimeoutMs, this.ackTimeoutMs, this.stepTimeoutMs]) {
      if (!Number.isFinite(timeout) || timeout <= 0) throw new RangeError("Mission timeouts must be positive finite numbers.");
    }
  }

  get snapshot(): MissionSnapshot | null {
    if (!this.run) return null;
    const { id, status, index, steps, reason } = this.run;
    return { id, status, index, steps: steps.map(step => ({ ...step })), ...(reason ? { reason } : {}) };
  }

  start(actions: readonly SimulatorAction[]): MissionSnapshot {
    this.cancel("Replaced by a new mission.");
    const token = this.options.idFactory?.() ?? globalThis.crypto?.randomUUID?.() ?? String(++fallbackId);
    const run: Run = {
      id: `mission-${++this.generation}-${token}`,
      status: "queued",
      index: 0,
      steps: actions.map(action => ({ action: action as MissionAction, status: "queued" })),
      active: null,
    };
    this.run = run;
    if (actions.length < 1 || actions.length > 4 || actions.some(action => !EXECUTABLE.has(action))) {
      this.fail(run, "A mission must contain one to four supported actions.");
      return this.snapshot!;
    }
    this.arm(run, this.queueTimeoutMs, "The simulator did not become ready for this mission.");
    this.emit();
    this.advance(run);
    return this.snapshot!;
  }

  acknowledge(id: string, accepted: boolean, message = "", receipt: MissionAckReceipt = {}): void {
    const run = this.run;
    const active = run?.active;
    if (!run || !active || active.id !== id || active.acknowledged || terminal(run.status)) return;
    if (!accepted) { this.fail(run, message || "The simulator rejected this step."); return; }
    const completion = receipt.completion ?? (run.steps[run.index].action === "stop" ? "immediate" : "status");
    const barrier = receipt.statusSeq ?? active.dispatchSeq;
    if ((completion !== "immediate" && completion !== "status") || !Number.isSafeInteger(barrier) || barrier < active.dispatchSeq) {
      this.fail(run, "The simulator returned an invalid completion receipt.");
      return;
    }
    active.acknowledged = true;
    active.completion = completion;
    active.barrier = barrier;
    if (message) run.steps[run.index].detail = message;
    const switchingModel = run.steps[run.index].action === "switch_to_rollers" || run.steps[run.index].action === "switch_to_legs";
    const timeout = isBallTask(run.steps[run.index].action) ? Math.max(this.stepTimeoutMs, 180_000)
      : switchingModel ? Math.max(this.stepTimeoutMs, 45_000) : this.stepTimeoutMs;
    this.arm(run, timeout, "The simulator did not report this step finishing.");
    this.emit();
    this.advance(run);
  }

  updateStatus(status: MissionStatus): void {
    if (!Number.isSafeInteger(status.seq) || status.seq < 0 || status.seq <= (this.latestStatus?.seq ?? -1)) return;
    if ([status.ready, status.busy, status.paused, status.fallen].some(value => typeof value !== "boolean")) return;
    if (status.availableActions && (!Array.isArray(status.availableActions) || status.availableActions.some(action => !EXECUTABLE.has(action)))) return;
    const previousDuck = this.latestStatus?.selectedDuckId;
    this.latestStatus = { ...status, ...(status.availableActions ? { availableActions: [...status.availableActions] } : {}),
      ...(status.task ? { task: { ...status.task } } : {}) };
    const run = this.run;
    if (!run || terminal(run.status)) return;
    if (previousDuck && status.selectedDuckId && previousDuck !== status.selectedDuckId) { this.cancel("The selected duck changed."); return; }
    if (status.paused) { this.cancel("Mission cancelled because the simulator paused."); return; }
    if (status.manual) { this.cancel("Manual controls took over."); return; }
    if (status.fallen || status.error) { this.fail(run, status.error || "The robot entered recovery; the remaining mission was stopped."); return; }
    if (!status.ready && run.active) { this.fail(run, "The simulator became unavailable during this step."); return; }
    if (run.active && status.busy && status.seq > run.active.dispatchSeq) run.active.latestBusySeq = status.seq;
    this.advance(run);
  }

  cancel(reason = "Mission cancelled."): MissionSnapshot | null {
    const run = this.run;
    if (!run || terminal(run.status)) return this.snapshot;
    this.clearTimer();
    run.status = "cancelled";
    run.reason = reason;
    run.active = null;
    for (const step of run.steps) if (!terminal(step.status)) step.status = "cancelled";
    this.emit();
    return this.snapshot;
  }

  /** Clears the old iframe's status sequence; late command ACKs remain invalid. */
  resetStatus(reason = "The simulator reloaded."): void {
    this.cancel(reason);
    this.latestStatus = null;
  }

  private advance(run: Run): void {
    if (this.run !== run || terminal(run.status)) return;
    const status = this.latestStatus;
    if (!status) return;
    if (status.paused || status.manual) { this.cancel(status.paused ? "Mission cancelled because the simulator paused." : "Manual controls took over."); return; }
    if (status.fallen || status.error) { this.fail(run, status.error || "Wait for the robot to recover before starting a mission."); return; }
    const active = run.active;
    if (active) {
      if (!active.acknowledged || !status.ready || status.seq <= active.barrier) return;
      const action = run.steps[run.index].action;
      if (isBallTask(action)) {
        const task = status.task;
        // An idle robot is not proof that it reached or kicked a ball. Old
        // receipts from a replaced task cannot finish the current program.
        if (!task || task.commandId !== active.id || task.action !== action) return;
        if (task.outcome === "failed") { this.fail(run, task.reason || "The ball task could not be completed."); return; }
        if (task.outcome === "cancelled") { this.cancel(task.reason || "The ball task was cancelled."); return; }
        if (task.outcome !== "succeeded" || task.phase !== "complete") return;
        if (action === "kick_ball" && (!task.ballContact || task.ballDisplacementM < 0.05)) {
          this.fail(run, "The kick ended without verified ball contact and displacement."); return;
        }
        run.steps[run.index].detail = task.reason;
      }
      if (status.busy) return;
      if (!isBallTask(action) && active.completion === "status" && (active.latestBusySeq <= active.barrier || status.seq <= active.latestBusySeq)) return;
      this.clearTimer();
      run.steps[run.index].status = "completed";
      run.active = null;
      if (run.index === run.steps.length - 1) {
        run.status = "completed";
        this.emit();
        return;
      }
      run.index++;
      this.arm(run, this.queueTimeoutMs, "The simulator did not become ready for the next step.");
      this.emit();
      // A callback can synchronously cancel or provide a newer status. Re-read
      // the state rather than dispatching with the idle snapshot above.
      this.advance(run);
      return;
    }
    const step = run.steps[run.index];
    // Stop can interrupt a pulse that is already busy; waiting for idle would
    // turn an explicit stop request into a command sent after motion finished.
    if (!status.ready || (status.busy && step.action !== "stop")) return;
    if (step.action !== "stop" && status.availableActions && !status.availableActions.includes(step.action)) {
      this.fail(run, `This duck cannot currently perform: ${SIMULATOR_ACTION_LABELS[step.action]}.`);
      return;
    }
    const id = `${run.id}:${run.index}`;
    run.active = { id, dispatchSeq: status.seq, acknowledged: false, completion: "status", barrier: status.seq, latestBusySeq: -1 };
    run.status = "running";
    step.status = "running";
    step.id = id;
    this.arm(run, this.ackTimeoutMs, "The simulator did not acknowledge this step.");
    this.emit();
    if (this.run !== run || terminal(run.status) || run.active?.id !== id) return;
    try {
      this.options.dispatch(step.action, id);
    } catch (error) {
      if (this.run === run && run.active?.id === id) this.fail(run, error instanceof Error ? error.message : "The command could not be sent.");
    }
  }

  private fail(run: Run, reason: string): void {
    if (this.run !== run || terminal(run.status)) return;
    this.clearTimer();
    run.status = "failed";
    run.reason = reason;
    run.active = null;
    for (let index = 0; index < run.steps.length; index++) {
      if (terminal(run.steps[index].status)) continue;
      run.steps[index].status = index === run.index ? "failed" : "cancelled";
    }
    if (run.steps[run.index]) run.steps[run.index].detail = reason;
    this.emit();
  }

  private arm(run: Run, milliseconds: number, reason: string): void {
    this.clearTimer();
    const timer = { handle: undefined as unknown };
    this.timer = timer;
    timer.handle = this.timers.setTimeout(() => {
      if (this.timer !== timer || this.run !== run || terminal(run.status)) return;
      this.timer = null;
      this.fail(run, reason);
    }, milliseconds);
  }

  private clearTimer(): void {
    if (!this.timer) return;
    const timer = this.timer;
    this.timer = null;
    this.timers.clearTimeout(timer.handle);
  }

  private emit(): void {
    const snapshot = this.snapshot;
    if (!snapshot) return;
    // Rendering/analytics callbacks cannot stop cancellation or command checks.
    try { this.options.onChange?.(snapshot); } catch { /* The runner retains authority over its own state. */ }
  }
}

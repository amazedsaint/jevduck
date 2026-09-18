import {
  AUTONOMY_MODES, AUTONOMY_CELL_SIZE_M, AUTONOMY_MAX_RECENT, AUTONOMY_MAX_VISITED,
  autonomyDecisionSchema, autonomyStateSchema, eligibleAutonomyBehaviors,
  type AutonomyCell, type AutonomyDecision, type AutonomyEpisode,
  type AutonomyInput, type AutonomyMode, type AutonomyState,
} from "./autonomy-contract";
import type { MissionSnapshot, MissionTimers } from "./mission-runner";
import type { SimulatorExecutableAction } from "./simulator";

export type AutonomyPhase = "off" | "starting" | "waiting" | "choosing" | "acting" | "observing" | "cooldown" | "error";
export type AutonomyStatus = Omit<AutonomyState, "autonomyActive" | "guardReason" | "guardSeq"> & {
  autonomyActive?: boolean;
  guardReason?: string | null;
  guardSeq?: number;
  error?: string;
  manual?: boolean;
  suspended?: boolean;
  phase?: string;
};

export interface AutonomySnapshot {
  readonly active: boolean;
  readonly mode: AutonomyMode | null;
  readonly phase: AutonomyPhase;
  readonly reason?: string;
  readonly currentDecision: AutonomyDecision | null;
  readonly lastDecision: AutonomyDecision | null;
  readonly missionId: string | null;
  readonly completedCycles: number;
  readonly blockedCycles: number;
  readonly recent: readonly AutonomyEpisode[];
  readonly visited: readonly AutonomyCell[];
  readonly distanceM: number;
  /** Local clock milliseconds; never sent to Jev as observation authority. */
  readonly nextRequestAt: number | null;
}

export interface AutonomyControllerOptions {
  request(input: AutonomyInput, signal: AbortSignal): Promise<AutonomyDecision>;
  execute(plan: SimulatorExecutableAction[]): MissionSnapshot;
  onChange?(snapshot: AutonomySnapshot): void;
  /** Cancel the runner only. A wall guard has already stopped the runtime. */
  cancelMission(reason: string): void;
  stopMotion(): void;
  setRuntimeAutonomy(active: boolean): void;
  now?(): number;
  timers?: MissionTimers;
  /** The request interval cannot be configured below eight seconds. */
  requestIntervalMs?: number;
  cooldownMs?: number;
  observationMs?: number;
  statusMaxAgeMs?: number;
  activationTimeoutMs?: number;
  requestTimeoutMs?: number;
}

type Cycle = {
  decision: AutonomyDecision;
  startDistance: number;
  startBallDistance: number | null;
  missionId: string | null;
  waitUntil: number | null;
};
type Request = { generation: number; controller: AbortController; deadline: number };
const cloneDecision = (decision: AutonomyDecision | null): AutonomyDecision | null => decision && ({ ...decision, plan: [...decision.plan], alternatives: decision.alternatives.map(item => ({ ...item })) });
const isTerminal = (mission: MissionSnapshot) => mission.status === "completed" || mission.status === "failed" || mission.status === "cancelled";

/**
 * A measured-state loop around Jev and MissionRunner. No behavior is selected
 * locally when Jev is unavailable. Feed status here BEFORE feeding the runner,
 * so a wall guard can cancel remaining steps before an idle receipt advances it.
 */
export class AutonomyController {
  private readonly now: () => number;
  private readonly timers: MissionTimers;
  private readonly interval: number;
  private readonly cooldown: number;
  private readonly observation: number;
  private readonly maxAge: number;
  private readonly activationTimeout: number;
  private readonly requestTimeout: number;
  private value: AutonomySnapshot = {
    active: false, mode: null, phase: "off", currentDecision: null, lastDecision: null,
    missionId: null, completedCycles: 0, blockedCycles: 0, recent: [], visited: [], distanceM: 0, nextRequestAt: null,
  };
  private state: AutonomyState | null = null;
  private receivedAt: number | null = null;
  private progressedAt: number | null = null;
  private locomotionSwitchStartedAt: number | null = null;
  private generation = 0;
  private disposed = false;
  private confirmed = false;
  private activationSeq = -1;
  private activationDeadline = 0;
  private lastGuardSeq = 0;
  private lastRequestAt: number | null = null;
  private requestSlot: Request | null = null;
  private cycle: Cycle | null = null;
  private timer: { handle: unknown } | null = null;
  private sample: { position: number[]; time: number } | null = null;
  private lastCell: string | null = null;
  private cells = new Map<string, AutonomyCell>();
  private executing = false;
  private missionBuffer = new Map<string, MissionSnapshot>();
  private pumping = false;
  private repump = false;
  private ending = false;

  constructor(private readonly options: AutonomyControllerOptions) {
    this.now = options.now ?? (() => Date.now());
    this.timers = options.timers ?? {
      setTimeout: (callback, milliseconds) => globalThis.setTimeout(callback, milliseconds),
      clearTimeout: handle => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
    };
    this.interval = Math.max(8_000, options.requestIntervalMs ?? 8_000);
    this.cooldown = options.cooldownMs ?? 1_500;
    this.observation = options.observationMs ?? 3_000;
    this.maxAge = Math.min(2_000, options.statusMaxAgeMs ?? 2_000);
    this.activationTimeout = options.activationTimeoutMs ?? 5_000;
    this.requestTimeout = options.requestTimeoutMs ?? 12_000;
    for (const duration of [this.interval, this.cooldown, this.observation, this.maxAge, this.activationTimeout, this.requestTimeout]) {
      if (!Number.isFinite(duration) || duration <= 0) throw new RangeError("Autonomy timings must be positive finite numbers.");
    }
  }

  get snapshot(): AutonomySnapshot {
    return { ...this.value, currentDecision: cloneDecision(this.value.currentDecision), lastDecision: cloneDecision(this.value.lastDecision),
      recent: this.value.recent.map(item => ({ ...item })), visited: this.value.visited.map(item => ({ ...item })) };
  }

  start(mode: AutonomyMode): AutonomySnapshot {
    if (this.disposed || this.ending) return this.snapshot;
    if (!AUTONOMY_MODES.includes(mode)) throw new RangeError("Unknown autonomy mode.");
    if (this.value.active) this.stop("Autonomy mode changed.");
    const generation = ++this.generation;
    this.confirmed = false;
    this.activationSeq = this.state?.seq ?? -1;
    this.activationDeadline = this.now() + this.activationTimeout;
    this.lastGuardSeq = this.state?.guardSeq ?? 0;
    this.sample = null;
    this.locomotionSwitchStartedAt = null;
    this.lastCell = null;
    this.value = { ...this.value, active: true, mode, phase: "starting", reason: undefined, currentDecision: null, missionId: null,
      nextRequestAt: Math.max(this.now(), (this.lastRequestAt ?? -Infinity) + this.interval) };
    this.emit();
    if (!this.live(generation)) return this.snapshot;
    try { this.options.setRuntimeAutonomy(true); }
    catch (error) { this.end(this.errorMessage(error, "The runtime could not enable autonomy."), "error"); }
    this.pump();
    return this.snapshot;
  }

  /** false preserves an iframe user's live manual input after its takeover. */
  stop(reason = "Autonomy stopped.", stopMotion = true): AutonomySnapshot {
    this.end(reason, "off", stopMotion);
    return this.snapshot;
  }

  /** Use after a world reset/reload; session memory and old status sequences go. */
  reset(reason = "The simulator was reset."): AutonomySnapshot {
    this.stop(reason);
    this.state = null; this.receivedAt = null; this.progressedAt = null; this.sample = null; this.lastCell = null;
    this.cells.clear(); this.lastGuardSeq = 0;
    // Keep lastRequestAt: repeatedly resetting must not bypass the rate cap.
    this.value = { ...this.value, mode: null, currentDecision: null, lastDecision: null, completedCycles: 0, blockedCycles: 0, recent: [], visited: [], distanceM: 0 };
    this.emit();
    return this.snapshot;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.stop("Autonomy controller closed.");
  }

  updateStatus(status: AutonomyStatus): void {
    if (this.disposed || !Number.isSafeInteger(status.seq) || status.seq < 0 || status.seq <= (this.state?.seq ?? -1)) return;
    const parsed = autonomyStateSchema.safeParse({
      ready: status.ready, busy: status.busy, paused: status.paused, fallen: status.fallen, loco: status.loco, mode: status.mode,
      seq: status.seq, time: status.time, position: status.position, headingRad: status.headingRad, posture: status.posture,
      clearance: status.clearance, spatialValid: status.spatialValid, guardReason: status.guardReason ?? null,
      guardSeq: status.guardSeq ?? 0, autonomyActive: status.autonomyActive ?? false,
      followEnabled: status.followEnabled, selectedDuckId: status.selectedDuckId, availableActions: status.availableActions,
      ball: status.ball, companion: status.companion, clearanceSources: status.clearanceSources, task: status.task,
    });
    if (!parsed.success) { if (this.value.active) this.end("The simulator returned an invalid spatial observation.", "error"); return; }
    if (this.state?.selectedDuckId && parsed.data.selectedDuckId && this.state.selectedDuckId !== parsed.data.selectedDuckId) {
      // A target change is an authority boundary, even if a host forgot to cancel.
      this.reset("The selected duck changed.");
    }
    const previousTime = this.state?.time;
    this.state = parsed.data;
    this.receivedAt = this.now();
    if (previousTime === undefined || this.state.time > previousTime) this.progressedAt = this.receivedAt;
    const ownsSwitch = this.cycle?.decision.behavior === "switch_to_rollers" || this.cycle?.decision.behavior === "switch_to_legs";
    if (ownsSwitch && status.phase === "switching_locomotion") {
      this.locomotionSwitchStartedAt ??= this.receivedAt;
    } else if (this.locomotionSwitchStartedAt !== null) {
      // A verified loading phase ended. Give the resumed physics one normal
      // freshness window to advance, without fabricating a simulation tick.
      this.locomotionSwitchStartedAt = null;
      this.progressedAt = this.receivedAt;
    }
    if (!this.value.active) return;
    if (status.error) { this.end(status.error.slice(0, 400), "error"); return; }
    if (status.manual) { this.stop("Manual controls took over.", false); return; }
    if (status.suspended) { this.stop("The simulator was suspended."); return; }
    if (previousTime !== undefined && this.state.time < previousTime) { this.stop("The simulator was reset."); return; }
    if (!this.lifecycleSafe()) return;
    if (!this.confirmed && this.state.seq > this.activationSeq && this.state.autonomyActive) this.confirmed = true;
    if (this.confirmed) this.measure(this.state);
    if (this.state.guardSeq > this.lastGuardSeq) {
      this.lastGuardSeq = this.state.guardSeq;
      if (this.cycle && this.state.guardReason) {
        this.finish("blocked", this.state.guardReason, false);
        // The runtime already stopped the pulse and retains autonomy. Sending
        // Stop here would revoke it, so only the remaining runner steps cancel.
        try { this.options.cancelMission("The movement guard blocked this behavior."); }
        catch (error) { this.end(this.errorMessage(error, "The mission could not be cancelled."), "error"); }
        this.emit();
      }
    }
    this.pump();
  }

  updateMission(mission: MissionSnapshot): void {
    if (this.disposed || !this.value.active) return;
    if (this.executing) {
      if (this.missionBuffer.size < 16 || this.missionBuffer.has(mission.id)) this.missionBuffer.set(mission.id, mission);
      return;
    }
    const cycle = this.cycle;
    if (!cycle || !cycle.missionId || mission.id !== cycle.missionId || !isTerminal(mission)) return;
    if (mission.status === "completed") {
      this.finish("completed");
      this.pump();
    } else {
      this.end(mission.reason || (mission.status === "failed" ? "The autonomous mission failed." : "The autonomous mission was interrupted."), mission.status === "failed" ? "error" : "off");
    }
  }

  private live(generation: number): boolean { return this.value.active && !this.disposed && this.generation === generation; }

  private lifecycleSafe(): boolean {
    if (!this.state) return true;
    if (!this.state.ready) { this.end("The simulator is not ready.", "error"); return false; }
    if (this.state.paused) { this.stop("The simulator was paused."); return false; }
    if (this.state.fallen || this.state.posture === "fallen") { this.end("The robot needs to recover first.", "error"); return false; }
    if (!this.state.spatialValid) { this.end("The simulator has no valid spatial observation.", "error"); return false; }
    if (this.confirmed && !this.state.autonomyActive) { this.stop("Autonomy was disabled by the simulator."); return false; }
    return true;
  }

  private pump(): void {
    if (this.pumping) { this.repump = true; return; }
    this.pumping = true;
    try {
      do { this.repump = false; this.advance(); } while (this.repump && this.value.active);
    } finally { this.pumping = false; this.schedule(); }
  }

  private advance(): void {
    if (!this.value.active || this.disposed) return;
    const generation = this.generation;
    const now = this.now();
    if (this.receivedAt !== null && now - this.receivedAt >= this.maxAge) { this.end("The simulator observation became stale.", "error"); return; }
    if (this.confirmed && this.progressedAt !== null && now - this.progressedAt >= this.maxAge &&
        (this.locomotionSwitchStartedAt === null || now - this.locomotionSwitchStartedAt >= 30_000)) { this.end("The simulated state stopped advancing.", "error"); return; }
    if (!this.lifecycleSafe()) return;
    if (!this.confirmed) {
      if (now >= this.activationDeadline) this.end("The simulator did not confirm autonomy.", "error");
      return;
    }
    if (!this.state) return;
    if (this.requestSlot) {
      if (now >= this.requestSlot.deadline) this.end("Jev did not return an autonomous decision in time.", "error");
      return;
    }
    if (this.cycle) {
      if (this.cycle.waitUntil !== null && now >= this.cycle.waitUntil && !this.state.busy && this.state.posture !== "transitioning") this.finish("completed");
      else return;
    }
    if (!this.live(generation)) return;
    if (this.state.busy || this.state.posture === "transitioning") { this.phase("waiting"); return; }
    if (now < (this.value.nextRequestAt ?? now)) { this.phase("cooldown"); return; }
    const input = this.input();
    if (!eligibleAutonomyBehaviors(input).length) { this.end("No autonomous behavior is eligible in the current state.", "error"); return; }
    this.choose(input);
  }

  private input(): AutonomyInput {
    return { mode: this.value.mode!, state: autonomyStateSchema.parse(this.state!),
      memory: { recent: this.value.recent.map(item => ({ ...item })), visited: this.value.visited.map(item => ({ ...item })) } };
  }

  private choose(input: AutonomyInput): void {
    const slot: Request = { generation: this.generation, controller: new AbortController(), deadline: this.now() + this.requestTimeout };
    this.requestSlot = slot;
    this.lastRequestAt = this.now();
    this.value = { ...this.value, phase: "choosing", reason: undefined, nextRequestAt: this.lastRequestAt + this.interval };
    this.emit();
    if (!this.live(slot.generation) || this.requestSlot !== slot) return;
    let request: Promise<AutonomyDecision>;
    try { request = this.options.request(input, slot.controller.signal); }
    catch (error) { this.requestSlot = null; this.end(this.errorMessage(error, "Jev is unavailable."), "error"); return; }
    Promise.resolve(request).then(decision => {
      if (!this.live(slot.generation) || this.requestSlot !== slot || slot.controller.signal.aborted) return;
      this.requestSlot = null;
      if (!this.state || this.receivedAt === null || this.now() - this.receivedAt >= this.maxAge) { this.end("The simulator observation became stale.", "error"); return; }
      if (!this.lifecycleSafe()) return;
      const parsed = autonomyDecisionSchema.safeParse(decision);
      if (!parsed.success) { this.end("Jev returned an invalid autonomous decision.", "error"); return; }
      if (parsed.data.behavior === "stop") {
        if (!eligibleAutonomyBehaviors(this.input()).includes("stop")) { this.phase("cooldown"); this.pump(); return; }
        this.value = { ...this.value, lastDecision: parsed.data };
        this.stop("Jev chose to stop autonomy.");
        return;
      }
      const cycle: Cycle = { decision: parsed.data, startDistance: this.value.distanceM,
        startBallDistance: this.state.ball?.present ? this.state.ball.distanceM : null, missionId: null, waitUntil: null };
      this.cycle = cycle;
      this.value = { ...this.value, currentDecision: parsed.data, lastDecision: parsed.data, reason: undefined };
      if (!eligibleAutonomyBehaviors(this.input()).includes(parsed.data.behavior)) {
        this.finish("blocked", "The measured state changed before this behavior could start.");
        this.pump(); return;
      }
      if (parsed.data.behavior === "wait") {
        cycle.waitUntil = this.now() + this.observation;
        this.phase("observing"); this.pump(); return;
      }
      this.value = { ...this.value, phase: "acting" };
      this.emit();
      if (!this.live(slot.generation) || this.cycle !== cycle) return;
      // UI callbacks can synchronously supply newer status. The last gate
      // belongs immediately beside dispatch, after those callbacks return.
      if (!this.lifecycleSafe()) return;
      if (this.receivedAt === null || this.now() - this.receivedAt >= this.maxAge ||
          !eligibleAutonomyBehaviors(this.input()).includes(parsed.data.behavior)) {
        this.finish("blocked", "The measured state changed before this behavior could start.");
        this.pump(); return;
      }
      this.executing = true;
      this.missionBuffer.clear();
      let mission: MissionSnapshot;
      try { mission = this.options.execute([...parsed.data.plan]); }
      catch (error) { this.executing = false; this.missionBuffer.clear(); this.end(this.errorMessage(error, "The mission could not be started."), "error"); return; }
      this.executing = false;
      if (!this.live(slot.generation) || this.cycle !== cycle) { this.missionBuffer.clear(); return; }
      cycle.missionId = mission.id;
      this.value = { ...this.value, missionId: mission.id };
      // MissionRunner.start emits synchronously, before returning its ID. Keep
      // the last matching callback so an immediate completion is not lost.
      const latest = this.missionBuffer.get(mission.id) ?? mission;
      this.missionBuffer.clear();
      this.emit();
      this.updateMission(latest);
      this.pump();
    }).catch(error => {
      if (this.live(slot.generation) && this.requestSlot === slot && !slot.controller.signal.aborted) {
        this.requestSlot = null; this.end(this.errorMessage(error, "Jev is unavailable."), "error");
      }
    }).finally(() => { if (this.requestSlot === slot) this.requestSlot = null; this.schedule(); });
  }

  private finish(outcome: AutonomyEpisode["outcome"], reason?: string, notify = true): void {
    const cycle = this.cycle;
    if (!cycle) return;
    this.cycle = null;
    const task = this.state?.task;
    const taskEvidence = task && task.commandId === `${cycle.missionId}:0` && task.action === cycle.decision.behavior
      ? { ballContact: task.ballContact, ballDisplacementM: task.ballDisplacementM, ...(task.outcome ? { taskOutcome: task.outcome } : {}) } : {};
    const episode: AutonomyEpisode = { behavior: cycle.decision.behavior, outcome, distanceM: Math.min(1000, Math.max(0, this.value.distanceM - cycle.startDistance)),
      ...taskEvidence,
      ...(cycle.startBallDistance !== null ? { ballDistanceBeforeM: cycle.startBallDistance } : {}),
      ...(this.state?.ball?.present ? { ballDistanceAfterM: this.state.ball.distanceM } : {}) };
    this.value = { ...this.value, currentDecision: null, missionId: null, phase: "cooldown", reason,
      completedCycles: this.value.completedCycles + (outcome === "completed" ? 1 : 0), blockedCycles: this.value.blockedCycles + (outcome === "blocked" ? 1 : 0),
      recent: [...this.value.recent, episode].slice(-AUTONOMY_MAX_RECENT),
      nextRequestAt: Math.max(this.now() + this.cooldown, (this.lastRequestAt ?? -Infinity) + this.interval) };
    if (notify) this.emit();
  }

  private measure(state: AutonomyState): void {
    const previous = this.sample;
    let distanceM = this.value.distanceM;
    if (previous && state.time > previous.time) {
      const delta = Math.hypot(state.position[0] - previous.position[0], state.position[1] - previous.position[1]);
      // Reset/teleport jumps are not travelled distance. Duplicate sim times
      // never accumulate displacement, even when bridge sequence increases.
      if (delta < 0.3) distanceM += delta;
    }
    this.sample = { position: [...state.position], time: state.time };
    const x = Math.floor(state.position[0] / AUTONOMY_CELL_SIZE_M), y = Math.floor(state.position[1] / AUTONOMY_CELL_SIZE_M);
    const key = `${x},${y}`;
    if (key !== this.lastCell) {
      const previousCell = this.cells.get(key);
      this.cells.delete(key);
      this.cells.set(key, { x, y, visits: Math.min(1e6, (previousCell?.visits ?? 0) + 1) });
      if (this.cells.size > AUTONOMY_MAX_VISITED) this.cells.delete(this.cells.keys().next().value!);
      this.lastCell = key;
    }
    this.value = { ...this.value, distanceM, visited: [...this.cells.values()].map(cell => ({ ...cell })) };
  }

  private end(reason: string, phase: "off" | "error", stopMotion = true): void {
    if (this.ending) return;
    this.ending = true;
    const wasActive = this.value.active;
    ++this.generation;
    this.requestSlot?.controller.abort(); this.requestSlot = null;
    this.clearTimer();
    // Clear authority before invoking callbacks; runner cancellation can call
    // updateMission synchronously, and UI callbacks may attempt another start.
    this.value = { ...this.value, active: false };
    if (this.cycle) this.finish(phase === "error" ? "failed" : "interrupted", reason, false);
    this.cycle = null;
    this.confirmed = false;
    this.locomotionSwitchStartedAt = null;
    this.value = { ...this.value, active: false, phase, reason, currentDecision: null, missionId: null, nextRequestAt: null };
    if (wasActive) {
      try { this.options.cancelMission(reason); } catch { /* Continue disabling runtime authority. */ }
      try { this.options.setRuntimeAutonomy(false); } catch { /* Still attempt the independent Stop. */ }
      if (stopMotion) { try { this.options.stopMotion(); } catch { /* Local authority is already revoked. */ } }
    }
    this.emit();
    this.ending = false;
  }

  private phase(phase: AutonomyPhase): void { if (this.value.phase !== phase) { this.value = { ...this.value, phase }; this.emit(); } }
  private emit(): void { try { this.options.onChange?.(this.snapshot); } catch { /* Rendering cannot own the loop. */ } }
  private errorMessage(error: unknown, fallback: string): string { return error instanceof Error && error.message ? error.message.slice(0, 400) : fallback; }

  private schedule(): void {
    this.clearTimer();
    if (!this.value.active || this.disposed) return;
    const now = this.now();
    const deadlines: number[] = [];
    if (this.receivedAt !== null) deadlines.push(this.receivedAt + this.maxAge);
    if (this.confirmed && this.progressedAt !== null) deadlines.push(this.locomotionSwitchStartedAt === null
      ? this.progressedAt + this.maxAge : this.locomotionSwitchStartedAt + 30_000);
    if (!this.confirmed) deadlines.push(this.activationDeadline);
    if (this.requestSlot) deadlines.push(this.requestSlot.deadline);
    if (this.cycle?.waitUntil !== null && this.cycle?.waitUntil !== undefined && this.cycle.waitUntil > now) deadlines.push(this.cycle.waitUntil);
    if (!this.cycle && !this.requestSlot && this.confirmed && this.value.nextRequestAt !== null && this.value.nextRequestAt > now) deadlines.push(this.value.nextRequestAt);
    if (!deadlines.length) return;
    const generation = this.generation;
    const timer = { handle: undefined as unknown };
    this.timer = timer;
    timer.handle = this.timers.setTimeout(() => {
      if (this.timer !== timer || !this.live(generation)) return;
      this.timer = null;
      this.pump();
    }, Math.max(0, Math.min(...deadlines) - now));
  }

  private clearTimer(): void {
    if (!this.timer) return;
    const timer = this.timer; this.timer = null; this.timers.clearTimeout(timer.handle);
  }
}

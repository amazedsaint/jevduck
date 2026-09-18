import {
  SWARM_MAX_RECENT, SWARM_SCENARIOS, eligibleSwarmIntents, swarmDecisionSchema, swarmStateSchema,
  type SwarmDecision, type SwarmEpisode, type SwarmInput, type SwarmIntent, type SwarmScenario, type SwarmState,
} from "./swarm-contract";
import type { MissionTimers } from "./mission-runner";

export type SwarmPhase = "off" | "starting" | "choosing" | "acting" | "observing" | "waiting" | "error";
export type SwarmStatus = SwarmState & { manual?: boolean; suspended?: boolean; error?: string };
export interface SwarmSnapshot {
  active: boolean; scenario: SwarmScenario | null; phase: SwarmPhase; runId: string | null;
  commandId: string | null; currentDecision: SwarmDecision | null; lastDecision: SwarmDecision | null;
  reason: string; completedCycles: number; blockedCycles: number; recent: readonly SwarmEpisode[]; nextRequestAt: number | null;
}
export interface SwarmControllerOptions {
  request(input: SwarmInput, signal: AbortSignal): Promise<SwarmDecision>;
  activate(scenario: SwarmScenario, runId: string): void;
  dispatch(intent: SwarmIntent, id: string, runId: string): void;
  deactivate(runId: string): void;
  onChange?(snapshot: SwarmSnapshot): void;
  now?(): number; timers?: MissionTimers; idFactory?(): string;
  requestIntervalMs?: number; statusMaxAgeMs?: number; activationTimeoutMs?: number;
  requestTimeoutMs?: number; intentTimeoutMs?: number;
}
type RequestSlot = { generation: number; abort: AbortController; deadline: number };
type Command = { id: string; intent: SwarmIntent; seq: number; accepted: boolean; receiptDeadline: number; deadline: number; targetErrorBeforeM: number | null };
const initial = (): SwarmSnapshot => ({ active: false, scenario: null, phase: "off", runId: null, commandId: null,
  currentDecision: null, lastDecision: null, reason: "", completedCycles: 0, blockedCycles: 0, recent: [], nextRequestAt: null });
const cloneDecision = (decision: SwarmDecision | null) => decision && ({ ...decision, alternatives: decision.alternatives.map(item => ({ ...item })) });

/** Owns the group session; no model call occurs in the physics loop. A runtime
 * receipt and current run ID are required before an instruction can complete.
 */
export class SwarmController {
  private value = initial();
  private state: SwarmState | null = null;
  private receivedAt: number | null = null;
  private progressedAt: number | null = null;
  private generation = 0;
  private commandCount = 0;
  private disposed = false;
  private ending = false;
  private confirmed = false;
  private activationSeq = -1;
  private activationDeadline = 0;
  private lastRequestAt: number | null = null;
  private requestSlot: RequestSlot | null = null;
  private command: Command | null = null;
  private holdUntil: number | null = null;
  private timer: { handle: unknown } | null = null;
  private pumping = false;
  private repump = false;
  private readonly now: () => number;
  private readonly timers: MissionTimers;
  private readonly interval: number;
  private readonly maxAge: number;
  private readonly activationTimeout: number;
  private readonly requestTimeout: number;
  private readonly intentTimeout: number;

  constructor(private readonly options: SwarmControllerOptions) {
    this.now = options.now ?? Date.now;
    this.timers = options.timers ?? { setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms), clearTimeout: handle => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>) };
    this.interval = Math.max(8_000, options.requestIntervalMs ?? 8_000);
    this.maxAge = Math.min(2_000, options.statusMaxAgeMs ?? 2_000);
    this.activationTimeout = options.activationTimeoutMs ?? 45_000;
    this.requestTimeout = options.requestTimeoutMs ?? 12_000;
    this.intentTimeout = options.intentTimeoutMs ?? 120_000;
    for (const value of [this.interval, this.maxAge, this.activationTimeout, this.requestTimeout, this.intentTimeout]) {
      if (!Number.isFinite(value) || value <= 0) throw new RangeError("Swarm timings must be finite positive values.");
    }
  }
  get snapshot(): SwarmSnapshot {
    return { ...this.value, currentDecision: cloneDecision(this.value.currentDecision), lastDecision: cloneDecision(this.value.lastDecision), recent: this.value.recent.map(item => ({ ...item })) };
  }
  start(scenario: SwarmScenario): SwarmSnapshot {
    if (this.disposed || this.ending) return this.snapshot;
    if (!SWARM_SCENARIOS.includes(scenario)) throw new RangeError("Unknown swarm scenario.");
    if (this.value.active) this.stop("Scenario replaced.");
    const generation = ++this.generation;
    const token = this.options.idFactory?.() ?? globalThis.crypto.randomUUID();
    const runId = `swarm-${generation}-${token}`;
    this.value = { ...initial(), active: true, scenario, phase: "starting", runId,
      reason: "Preparing four simulated robots.", nextRequestAt: Math.max(this.now(), (this.lastRequestAt ?? -Infinity) + this.interval) };
    this.confirmed = false;
    this.activationSeq = this.state?.seq ?? -1;
    this.activationDeadline = this.now() + this.activationTimeout;
    this.emit();
    if (!this.live(generation)) return this.snapshot;
    try { this.options.activate(scenario, runId); }
    catch { this.end("The runtime could not start the scenario.", "error"); }
    this.pump();
    return this.snapshot;
  }
  stop(reason = "Group controller stopped."): SwarmSnapshot { this.end(reason, "off"); return this.snapshot; }
  reset(reason = "The simulator was reset."): SwarmSnapshot {
    this.stop(reason); this.state = null; this.receivedAt = null; this.progressedAt = null;
    this.value = { ...initial(), reason }; this.emit(); return this.snapshot;
  }
  dispose(): void { if (!this.disposed) { this.disposed = true; this.stop("Group controller closed."); } }

  /** Acceptance is not a completed setup or instruction. Fresh status remains authoritative. */
  acknowledgeRun(runId: string, accepted: boolean, message = ""): void {
    if (!this.value.active || runId !== this.value.runId || accepted) return;
    this.end(message.slice(0, 500) || "The runtime rejected the group scenario.", "error");
  }
  acknowledgeIntent(runId: string, id: string, accepted: boolean, message = ""): void {
    if (!this.value.active || runId !== this.value.runId || this.command?.id !== id) return;
    if (!accepted) { this.end(message.slice(0, 500) || "The runtime rejected the group instruction.", "error"); return; }
    this.command.accepted = true;
    this.schedule();
  }

  updateStatus(status: SwarmStatus): void {
    if (this.disposed || !Number.isSafeInteger(status.seq) || status.seq <= (this.state?.seq ?? -1)) return;
    const parsed = swarmStateSchema.safeParse({ seq: status.seq, time: status.time, ready: status.ready, paused: status.paused, swarm: status.swarm });
    if (!parsed.success) { if (this.value.active) this.end("The runtime returned an invalid swarm observation.", "error"); return; }
    const previousTime = this.state?.time;
    this.state = parsed.data; this.receivedAt = this.now();
    if (previousTime === undefined || this.state.time > previousTime) this.progressedAt = this.receivedAt;
    if (!this.value.active) return;
    if (status.manual || status.suspended || status.paused) { this.stop(status.manual ? "Manual controls took over." : "The simulator was paused or hidden."); return; }
    if (status.error) { this.end("The simulator reported an error.", "error"); return; }
    const wasConfirmed = this.confirmed;
    if (!this.confirmed && this.state.seq > this.activationSeq && this.state.ready && this.state.swarm.active &&
        this.state.swarm.runId === this.value.runId && this.state.swarm.scenario === this.value.scenario) {
      this.confirmed = true; this.progressedAt = this.now();
    }
    if (this.confirmed) {
      if (!this.state.ready || !this.state.swarm.active || this.state.swarm.runId !== this.value.runId || this.state.swarm.scenario !== this.value.scenario) {
        this.stop("The runtime ended or replaced the group session."); return;
      }
      // Scenario preparation explicitly rebuilds/resets the physics scene. Its
      // first matching ready receipt establishes the new simulation-time epoch.
      if (wasConfirmed && previousTime !== undefined && this.state.time < previousTime) { this.stop("The simulated world was reset."); return; }
      const command = this.command;
      const swarm = this.state.swarm;
      if (command && this.state.seq > command.seq) {
        if (swarm.commandId === command.id && swarm.intent === command.intent) {
          command.accepted = true;
          // Targets are assigned by this new instruction. A residual sampled
          // before dispatch belonged to the preceding instruction's slots.
          if (swarm.phase === "running" && command.targetErrorBeforeM === null && swarm.targetErrorM !== null) command.targetErrorBeforeM = swarm.targetErrorM;
          if (swarm.phase === "complete" || swarm.phase === "blocked") this.finish(command.intent, swarm.phase, command.targetErrorBeforeM);
        } else if (command.accepted && swarm.phase === "running") { this.stop("Another group instruction replaced this controller."); return; }
      }
    }
    this.pump();
  }
  private live(generation: number): boolean { return this.value.active && !this.disposed && generation === this.generation; }
  private input(): SwarmInput { return { state: swarmStateSchema.parse(this.state!), memory: { recent: this.value.recent.map(item => ({ ...item })) } }; }
  private pump(): void {
    if (this.pumping) { this.repump = true; return; }
    this.pumping = true;
    try { do { this.repump = false; this.advance(); } while (this.repump && this.value.active); }
    finally { this.pumping = false; this.schedule(); }
  }
  private advance(): void {
    if (!this.value.active || this.disposed) return;
    const now = this.now();
    if (!this.confirmed) {
      if (now >= this.activationDeadline) this.end("The runtime did not confirm the four-robot scenario.", "error");
      return;
    }
    if (!this.state || this.receivedAt === null || now - this.receivedAt >= this.maxAge) { this.end("The swarm observation became stale.", "error"); return; }
    if (this.progressedAt === null || now - this.progressedAt >= this.maxAge) { this.end("The simulated world stopped advancing.", "error"); return; }
    if (this.requestSlot) { if (now >= this.requestSlot.deadline) this.end("Jev did not return a group decision in time.", "error"); return; }
    if (this.command) {
      if (!this.command.accepted && now >= this.command.receiptDeadline) this.end("The runtime did not acknowledge the group instruction.", "error");
      else if (now >= this.command.deadline) this.end("The group instruction did not report its bounded outcome.", "error");
      return;
    }
    if (this.holdUntil !== null) {
      if (now < this.holdUntil) return;
      this.holdUntil = null; this.finish("hold", "complete", this.state.swarm.targetErrorM);
    }
    if (!this.value.active) return;
    if (this.state.swarm.phase === "running" || now < (this.value.nextRequestAt ?? now)) { this.phase("waiting"); return; }
    this.choose(this.input());
  }
  private choose(input: SwarmInput): void {
    const slot: RequestSlot = { generation: this.generation, abort: new AbortController(), deadline: this.now() + this.requestTimeout };
    this.requestSlot = slot; this.lastRequestAt = this.now();
    this.value = { ...this.value, phase: "choosing", reason: "Requesting a group intention from Jev.", nextRequestAt: this.lastRequestAt + this.interval };
    this.emit();
    if (!this.live(slot.generation) || this.requestSlot !== slot) return;
    let pending: Promise<SwarmDecision>;
    try { pending = this.options.request(input, slot.abort.signal); }
    catch { this.end("Jev is unavailable. The group controller stopped.", "error"); return; }
    Promise.resolve(pending).then(raw => {
      if (!this.live(slot.generation) || this.requestSlot !== slot || slot.abort.signal.aborted) return;
      this.requestSlot = null;
      const parsed = swarmDecisionSchema.safeParse(raw);
      if (!parsed.success) { this.end("Jev returned an invalid group decision.", "error"); return; }
      if (!this.state || this.receivedAt === null || this.now() - this.receivedAt >= this.maxAge) { this.end("The swarm observation became stale.", "error"); return; }
      if (!eligibleSwarmIntents(this.input()).includes(parsed.data.intent)) { this.value = { ...this.value, reason: "The measured capabilities changed before dispatch." }; this.phase("waiting"); this.pump(); return; }
      this.value = { ...this.value, currentDecision: parsed.data, lastDecision: parsed.data, reason: parsed.data.reason,
        phase: parsed.data.intent === "hold" ? "observing" : "acting" };
      this.emit();
      // Rendering callbacks may synchronously stop or replace this session.
      if (!this.live(slot.generation) || !this.state || !eligibleSwarmIntents(this.input()).includes(parsed.data.intent)) { this.pump(); return; }
      if (this.receivedAt === null || this.progressedAt === null || this.now() - this.receivedAt >= this.maxAge || this.now() - this.progressedAt >= this.maxAge) {
        this.end("The swarm observation became stale before dispatch.", "error"); return;
      }
      if (parsed.data.intent === "hold") { this.holdUntil = this.now() + 3_000; this.pump(); return; }
      const id = `${this.value.runId}:${++this.commandCount}`;
      this.command = { id, intent: parsed.data.intent, seq: this.state.seq, accepted: false, receiptDeadline: this.now() + 3_000,
        deadline: this.now() + this.intentTimeout, targetErrorBeforeM: null };
      this.value = { ...this.value, commandId: id };
      try { this.options.dispatch(parsed.data.intent, id, this.value.runId!); }
      catch { this.end("The group instruction could not be dispatched.", "error"); return; }
      this.emit(); this.pump();
    }).catch(() => {
      if (this.live(slot.generation) && this.requestSlot === slot) this.end("Jev is unavailable. The group controller stopped.", "error");
    }).finally(() => { if (this.requestSlot === slot) this.requestSlot = null; this.schedule(); });
  }
  private finish(intent: SwarmIntent, outcome: SwarmEpisode["outcome"], before: number | null): void {
    if (!this.state) return;
    const swarm = this.state.swarm;
    this.command = null;
    const episode: SwarmEpisode = { intent, outcome, progressM: intent === "hold" ? 0 : swarm.progressM,
      targetErrorBeforeM: before, targetErrorAfterM: swarm.targetErrorM, minSeparationM: swarm.minSeparationM };
    this.value = { ...this.value, commandId: null, currentDecision: null, phase: "waiting",
      reason: outcome === "blocked" ? "The native controller blocked the group instruction." : "Bounded group instruction finished; residual error remains measured.",
      completedCycles: this.value.completedCycles + (outcome === "complete" && intent !== "hold" ? 1 : 0),
      blockedCycles: this.value.blockedCycles + (outcome === "blocked" ? 1 : 0),
      // Quiet observations cannot evict the physical outcomes that determine
      // recovery admission and the next stage of an alternating formation.
      recent: intent === "hold" ? this.value.recent : [...this.value.recent.filter(item => item.intent !== "hold"), episode].slice(-SWARM_MAX_RECENT),
      nextRequestAt: Math.max(this.now() + 1_000, (this.lastRequestAt ?? -Infinity) + this.interval) };
    this.emit();
  }
  private end(reason: string, phase: "off" | "error"): void {
    if (this.ending) return;
    this.ending = true;
    const previous = this.value;
    ++this.generation; this.requestSlot?.abort.abort(); this.requestSlot = null;
    this.command = null; this.holdUntil = null; this.confirmed = false; this.clearTimer();
    this.value = { ...this.value, active: false, phase, reason, currentDecision: null, commandId: null, nextRequestAt: null };
    if (previous.active && previous.runId) { try { this.options.deactivate(previous.runId); } catch { /* Authority is already revoked locally. */ } }
    this.emit(); this.ending = false;
  }
  private phase(phase: SwarmPhase): void { if (this.value.phase !== phase) { this.value = { ...this.value, phase }; this.emit(); } }
  private emit(): void { try { this.options.onChange?.(this.snapshot); } catch { /* Rendering cannot own group authority. */ } }
  private clearTimer(): void { if (this.timer) { this.timers.clearTimeout(this.timer.handle); this.timer = null; } }
  private schedule(): void {
    this.clearTimer(); if (!this.value.active || this.disposed) return;
    const now = this.now(); const deadlines: number[] = [];
    if (!this.confirmed) deadlines.push(this.activationDeadline);
    else {
      if (this.receivedAt !== null) deadlines.push(this.receivedAt + this.maxAge);
      if (this.progressedAt !== null) deadlines.push(this.progressedAt + this.maxAge);
      if (this.requestSlot) deadlines.push(this.requestSlot.deadline);
      if (this.command) deadlines.push(this.command.accepted ? this.command.deadline : Math.min(this.command.receiptDeadline, this.command.deadline));
      if (this.holdUntil !== null) deadlines.push(this.holdUntil);
      if (!this.command && !this.requestSlot && this.holdUntil === null && this.value.nextRequestAt !== null && this.value.nextRequestAt > now) deadlines.push(this.value.nextRequestAt);
    }
    if (!deadlines.length) return;
    const generation = this.generation; const timer = { handle: undefined as unknown }; this.timer = timer;
    timer.handle = this.timers.setTimeout(() => { if (this.timer === timer && this.live(generation)) { this.timer = null; this.pump(); } }, Math.max(0, Math.min(...deadlines) - now));
  }
}

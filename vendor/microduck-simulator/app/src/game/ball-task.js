import { ballGoalCommand, relativeBall } from "./ball-goal.js";

export const BALL_TASK_ACTIONS = Object.freeze(["approach_ball", "kick_ball"]);
export const BALL_TASK_TIMEOUT_S = 120;
export const BALL_KICK_MIN_DISPLACEMENT_M = .05;
const ZERO = Object.freeze([0, 0, 0]);
const finitePosition = value => value?.length >= 3 && Array.from(value).every(Number.isFinite);

// A measured-state controller around the original actors. It owns velocity
// commands, never robot or ball coordinates, and never creates a ball.
export class BallTaskController {
  task = null;
  duckId = null;
  command = ZERO;
  foot = "left";
  stableFor = 0;
  blockedFor = 0;
  standRequested = false;
  standStableFor = 0;
  kickStart = null;
  kickElapsed = 0;
  contactIds = null;
  kickActorActive = false;
  contactStart = null;
  contactDisplacementM = 0;
  constructor({ getState, nativeAction, planner = ballGoalCommand, timeoutS = BALL_TASK_TIMEOUT_S }) {
    Object.assign(this, { getState, nativeAction, planner });
    this.timeoutS = Math.max(1, Math.min(BALL_TASK_TIMEOUT_S, timeoutS));
  }
  get active() { return this.task !== null && this.task.outcome === null; }
  owns(id) { return this.active && this.duckId === id; }
  snapshot(id = this.duckId) { return this.task && id === this.duckId ? { ...this.task } : null; }
  clear() { this.task = null; this.duckId = null; this.command = ZERO; }
  clearTerminal() { if (!this.active) this.clear(); }
  finish(outcome, reason) {
    if (!this.active) return;
    Object.assign(this.task, { outcome, phase: outcome === "succeeded" ? "complete" : outcome, reason });
    this.command = ZERO;
  }
  cancel(reason = "Ball task cancelled by manual control.") { this.finish("cancelled", reason); }
  start(action, commandId, duckId) {
    if (!BALL_TASK_ACTIONS.includes(action)) return { accepted: false, message: "Unsupported ball task." };
    if (this.active) return { accepted: false, message: "Let the current ball task finish or stop it first." };
    this.duckId = duckId;
    this.command = ZERO;
    this.stableFor = 0;
    this.blockedFor = 0;
    this.standRequested = false;
    this.standStableFor = 0;
    this.kickStart = null;
    this.kickElapsed = 0;
    this.kickActorActive = false;
    this.contactStart = null;
    this.contactDisplacementM = 0;
    this.task = { commandId: String(commandId ?? ""), action, phase: "searching", outcome: null,
      reason: "Locating the ball using measured simulator coordinates.", elapsedS: 0, ballContact: false, ballDisplacementM: 0 };
    const state = this.getState(duckId);
    if (!this.validate(state)) return { accepted: true, completion: "status", message: this.task.reason };
    this.foot = relativeBall(state.duck, state.ball).y < 0 ? "right" : "left";
    return { accepted: true, completion: "status", message: "Measured ball task started; completion requires its explicit result." };
  }
  validate(state) {
    if (!state?.ball?.present || !finitePosition(state.ball.position)) {
      this.finish("failed", "No ball is present in the simulated world. Add one before requesting ball interaction."); return false;
    }
    if (!state.duck || !finitePosition(state.duck.position) || !Number.isFinite(state.duck.headingRad)) {
      this.finish("failed", "A measured robot position is unavailable."); return false;
    }
    if (state.duck.loco !== "legs") { this.finish("failed", "Ball pursuit and kicking require the trained leg policy."); return false; }
    if (state.duck.fallen || state.duck.posture === "fallen" || state.duck.error) {
      this.finish("failed", "The robot lost its standing posture during ball interaction."); return false;
    }
    if (state.duck.paused || state.duck.manual) { this.cancel("Ball interaction was interrupted by pause or manual control."); return false; }
    return true;
  }
  observeContact(position = this.getState(this.duckId)?.ball?.position) {
    if (!this.active || !this.kickStart || !this.kickActorActive || !finitePosition(position)) return;
    this.contactStart ??= Array.from(position);
    this.task.ballContact = true;
  }
  observeBall(position) {
    if (!this.active || !this.kickStart || !finitePosition(position)) return;
    this.task.ballDisplacementM = Math.max(this.task.ballDisplacementM,
      Math.hypot(position[0] - this.kickStart[0], position[1] - this.kickStart[1]));
    if (this.contactStart) this.contactDisplacementM = Math.max(this.contactDisplacementM,
      Math.hypot(position[0] - this.contactStart[0], position[1] - this.contactStart[1]));
  }
  observePhysics({ mujoco, model, data, ballQposAdr }) {
    if (!this.active || !this.kickStart) return;
    const ballPosition = Array.from(data.qpos.slice(ballQposAdr, ballQposAdr + 3));
    this.observeBall(ballPosition);
    // A later ankle touch under the balance policy must not turn an earlier
    // miss into success. Only the original kick actor can supply evidence.
    if (!this.kickActorActive) return;
    // Contact may last less than one policy tick. Call after each physics
    // substep; copy contact indices before reading another native view.
    if (this.contactIds?.model !== model || this.contactIds.duckId !== this.duckId || this.contactIds.foot !== this.foot) {
      const prefix = this.duckId === "duck1" ? "" : `${this.duckId}_`;
      this.contactIds = { model, duckId: this.duckId, foot: this.foot,
        ball: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "ball_geom"),
        ankle: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, `${prefix}ankle_${this.foot}`) };
    }
    const contacts = data.contact;
    try {
      for (let index = 0; index < data.ncon; index++) {
        const contact = contacts.get(index);
        let first, second;
        try { first = Number(contact.geom[0]); second = Number(contact.geom[1]); }
        finally { contact.delete(); }
        const other = first === this.contactIds.ball ? second : second === this.contactIds.ball ? first : -1;
        if (other >= 0 && model.geom_bodyid[other] === this.contactIds.ankle) this.observeContact(ballPosition);
      }
    } finally { contacts.delete(); }
  }
  tick(dt = .02) {
    if (!this.active) return this.command;
    this.command = ZERO;
    this.task.elapsedS += dt;
    if (this.task.elapsedS >= this.timeoutS) { this.finish("failed", "The ball task exceeded its bounded simulation time."); return this.command; }
    const state = this.getState(this.duckId);
    if (!this.validate(state)) return this.command;
    const { duck, ball, obstacle, peers = [] } = state;
    if (this.kickStart) {
      this.kickActorActive = duck.mode === (this.foot === "left" ? "kickL" : "kickR");
      this.kickElapsed += dt;
      this.observeBall(ball.position);
      this.task.phase = duck.busy ? "kicking" : "verifying";
      this.task.reason = duck.busy ? "Executing the original kick actor." : "Checking physical ankle contact and measured ball displacement.";
      if (!duck.busy && this.task.ballContact && this.task.ballDisplacementM >= BALL_KICK_MIN_DISPLACEMENT_M
        && this.contactDisplacementM >= BALL_KICK_MIN_DISPLACEMENT_M) {
        this.finish("succeeded", "The selected ankle contacted the ball and moved it at least 5 cm.");
      } else if (!duck.busy && this.kickElapsed >= 2) {
        this.finish("failed", this.task.ballContact
          ? "The foot touched the ball, but it did not move the required 5 cm after that contact."
          : "The trained kick finished without measured contact between the selected ankle and ball.");
      }
      return this.command;
    }
    if (duck.posture === "sitting") {
      this.task.phase = "standing";
      this.task.reason = "Standing with the trained posture actor before approaching the ball.";
      if (!this.standRequested && !duck.busy) {
        const result = this.nativeAction(this.duckId, "stand");
        if (!result.accepted) this.finish("failed", result.message || "The stand prerequisite could not start.");
        else this.standRequested = true;
      }
      return this.command;
    }
    if (duck.busy || duck.posture !== "standing") {
      this.task.phase = "standing";
      this.task.reason = "Waiting for the measured posture transition to settle.";
      return this.command;
    }
    if (this.standRequested && this.standStableFor < .3) {
      const stable = duck.position[2] > .085 && duck.speedMps < .08 && (duck.tiltRad ?? 0) < .45;
      this.standStableFor = stable ? this.standStableFor + dt : 0;
      this.task.phase = "standing";
      this.task.reason = "Checking upright height, balance and velocity after standing.";
      return this.command;
    }
    const decision = this.planner({ duck, ball, foot: this.foot, obstacle, peers, settling: this.task.phase === "settling" });
    if (decision.blocked) {
      this.blockedFor += dt;
      this.stableFor = 0;
      this.task.reason = decision.reason;
      if (this.blockedFor >= 1) this.finish("failed", decision.reason);
      return this.command;
    }
    this.blockedFor = 0;
    if (!decision.aligned) {
      this.stableFor = 0;
      this.command = decision.command;
      this.task.phase = relativeBall(duck, ball).x < 0 ? "searching" : decision.phase;
      this.task.reason = this.task.phase === "searching"
        ? "Repositioning toward the ball behind the robot using measured world coordinates." : decision.reason;
      return this.command;
    }
    this.task.phase = "settling";
    this.task.reason = "Ball aligned beside the selected foot; waiting for measured balance and ball motion to settle.";
    const stable = Number.isFinite(duck.speedMps) && duck.speedMps < .035
      && Number.isFinite(ball.speedMps) && ball.speedMps < .08
      && ball.position[2] >= .035 && ball.position[2] <= .075;
    this.stableFor = stable ? this.stableFor + dt : 0;
    if (this.stableFor < .3) return this.command;
    if (this.task.action === "approach_ball") {
      this.finish("succeeded", "The robot reached the selected foot alignment window and settled."); return this.command;
    }
    this.kickStart = Array.from(ball.position);
    const result = this.nativeAction(this.duckId, this.foot === "left" ? "kick_left" : "kick_right");
    if (!result.accepted) { this.finish("failed", result.message || "The trained kick actor could not start."); return this.command; }
    this.kickActorActive = true;
    this.task.phase = "kicking";
    this.task.reason = "Executing the original kick actor from measured foot alignment.";
    return this.command;
  }
}

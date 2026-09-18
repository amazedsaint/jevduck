// Queue one velocity intention behind the official sit/stand actor.
// The caller supplies measured readiness after each real control step.
export class LocomotionIntent {
  pendingAction = null;
  phase = "idle";
  #stableFor = 0;
  #beginStand;
  #startPulse;

  constructor({ beginStand, startPulse }) {
    this.#beginStand = beginStand;
    this.#startPulse = startPulse;
  }

  request(action) {
    this.pendingAction = action;
    this.phase = "standing_up";
    this.#stableFor = 0;
    this.#beginStand();
  }

  cancel() {
    this.pendingAction = null;
    this.phase = "idle";
    this.#stableFor = 0;
  }

  tick(dt, { interrupted, transitioning, walking, settled }) {
    if (!this.pendingAction) return;
    if (interrupted) { this.cancel(); return; }
    if (transitioning || !walking) {
      this.phase = "standing_up";
      this.#stableFor = 0;
      return;
    }
    this.phase = "settling";
    this.#stableFor = settled ? this.#stableFor + dt : 0;
    // Consecutive measured stability, not a blind wall-clock delay.
    if (this.#stableFor + 1e-9 < 0.2) return;
    const action = this.pendingAction;
    this.cancel();
    this.#startPulse(action);
  }
}

// The upstream sit timer only delays the switch to sitFlag=1. Completion
// additionally requires the robot to reach a measured, stable seated pose.
export class SitSettle {
  pending = false;
  phase = "idle";
  error = null;
  #elapsed = 0;
  #stableFor = 0;

  request() {
    this.cancel();
    this.pending = true;
    this.phase = "sitting_down";
  }

  cancel() {
    this.pending = false;
    this.phase = "idle";
    this.error = null;
    this.#elapsed = 0;
    this.#stableFor = 0;
  }

  tick(dt, { activated, settled }) {
    if (!this.pending || this.error) return;
    if (!activated) return;
    this.phase = "settling";
    this.#elapsed += dt;
    this.#stableFor = settled ? this.#stableFor + dt : 0;
    if (this.#stableFor + 1e-9 >= 0.15) {
      this.cancel();
    } else if (this.#elapsed >= 8) {
      this.phase = "failed";
      this.error = "The robot could not settle into sitting. Reset before continuing.";
    }
  }
}

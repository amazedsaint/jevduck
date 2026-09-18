// The embedded parent's commands are ordinary low-priority controller
// inputs. They never write MuJoCo state or the rendered robot's joints.
export class ExternalCommandSource {
  id = "parent";
  connected = true;
  command = new Float32Array(3);
  axes = {};
  pressed = {};
  onAction = () => {};
  #until = 0;
  #getManualOverride;
  #onCancel;

  constructor({ getManualOverride, onCancel }) {
    this.#getManualOverride = getManualOverride;
    this.#onCancel = onCancel;
  }

  start(command, durationMs = 2000) {
    this.cancel();
    this.command.set(command);
    this.#until = performance.now() + Math.min(2000, Math.max(0, durationMs));
  }

  cancel() {
    const wasActive = this.#until !== 0;
    this.#until = 0;
    this.command.fill(0);
    if (wasActive) this.#onCancel?.();
  }

  isActive() {
    if (this.#until && (performance.now() >= this.#until || this.#getManualOverride())) this.cancel();
    return this.#until !== 0;
  }

  poll() { this.isActive(); }
  init() {}
  dispose() { this.cancel(); }
}

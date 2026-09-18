import { parkSpatial, ROBOT_RADIUS_M } from "./park-geometry.js";

export const ROBOT_MARGIN_M = ROBOT_RADIUS_M;
export const AUTONOMY_STOP_CLEARANCE_M = 0.18;
const zeroClearance = () => ({ front: 0, back: 0, left: 0, right: 0 });

// MuJoCo ground coordinates: +X is forward at yaw=0, +Y is left, +Z up.
// Sweep a conservative body footprint against walls and any measured
// park obstacle or peer. This is simulator geometry, not camera vision.
export function measureSpatial(qpos, world) {
  if (!qpos || qpos.length < 7) return { headingRad: 0, clearance: zeroClearance(), spatialValid: false };
  const [px, py, pz] = qpos;
  const w = qpos[3], x = qpos[4], y = qpos[5], z = qpos[6];
  const norm2 = w * w + x * x + y * y + z * z;
  if (![px, py, pz, w, x, y, z, norm2].every(Number.isFinite) || norm2 < 1e-12) {
    return { headingRad: 0, clearance: zeroClearance(), spatialValid: false };
  }
  const headingRad = Math.atan2(2 * (w * z + x * y) / norm2, 1 - 2 * (y * y + z * z) / norm2);
  return parkSpatial({ position: [px, py, pz], headingRad }, world);
}

const direction = (action) => action === "walk_forward" ? "front" : action === "walk_backward" ? "back" : null;

export class AutonomyGuard {
  active = false;
  guardReason = null;
  guardSeq = 0;
  #getSpatial;
  #cancelMotion;

  constructor({ getSpatial, cancelMotion }) {
    this.#getSpatial = getSpatial;
    this.#cancelMotion = cancelMotion;
  }

  setActive(active) {
    this.active = active === true;
    this.guardReason = null;
    if (!this.active) this.#cancelMotion();
  }

  accepted() { this.guardReason = null; }

  #check(action, required) {
    const side = direction(action);
    if (!this.active || !side) return true;
    const spatial = this.#getSpatial();
    const clearance = spatial.clearance?.[side];
    if (spatial.spatialValid && Number.isFinite(clearance) && clearance >= required) return true;
    const source = spatial.clearanceSources?.[side];
    const label = source === "obstacle" ? "Obstacle guard" : source === "duck" ? "Duck guard" : "Wall guard";
    this.guardReason = spatial.spatialValid && Number.isFinite(clearance)
      ? `${label}: ${side} clearance is ${Math.max(0, clearance).toFixed(2)} m; ${required.toFixed(2)} m is required.`
      : "Spatial guard: valid arena and duck positions are required before moving.";
    this.guardSeq++;
    // One cancellation boundary covers an active pulse AND any velocity
    // intention waiting behind the official stand-up policy.
    this.#cancelMotion();
    return false;
  }

  admit(action, speed) {
    const allowed = this.#check(action, Math.abs(speed) * 2 + AUTONOMY_STOP_CLEARANCE_M);
    if (allowed) this.accepted();
    return allowed;
  }

  monitor(action) { return this.#check(action, AUTONOMY_STOP_CLEARANCE_M); }
}

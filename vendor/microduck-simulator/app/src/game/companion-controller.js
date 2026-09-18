import {
  JOINT_NAMES, DEFAULT_POSE, NUM_JOINTS, OBS_SIZE, CMD_SIZE, ACTION_SCALE, CTRL_DT,
  POLICIES, CROUCH_PERIOD_S, CROUCH_END_PHASE, GROUND_PICK_PERIOD_S, GROUND_PICK_END_PHASE,
} from "./constants.js";
import { AutonomyGuard, measureSpatial } from "./spatial-guard.js";
import { SitSettle } from "./controls/sit-settle.js";
import { LocomotionIntent } from "./controls/locomotion-intent.js";
import { COMPANION_PREFIX } from "./shared-model.js";
import { HEAD_COMMANDS, movementFor, actionRoom, availableDuckActions, reverseProfile } from "./duck-actions.js";
import { sweptMotion } from "./park-geometry.js";

// An independent policy/controller state in the SAME MjModel/MjData. Inference
// produces its own 14 actuator targets; only the host advances physics,
// once both ducks have supplied targets for the current 20 ms step.
export class CompanionController {
  mode = "walk";
  loco = "legs";
  sitFlag = 0;
  lastAction = new Float32Array(NUM_JOINTS);
  obs = new Float32Array(OBS_SIZE);
  cmd = new Float32Array(CMD_SIZE);
  headTarget = new Float32Array(4);
  headSmooth = new Float32Array(4);
  inferenceCount = 0;
  pulse = null;
  transition = null;
  oneShot = null;
  recovery = null;
  fallSteps = 0;
  postKick = 0;
  coast = null;
  mouth = 0;
  quackFor = 0;
  error = null;
  manual = false;
  constructor({ mujoco, ort, getWorld, getSessions, getSpatialContext, paused, locked, quack, wheee, spawnBall, switchLoco, prefix = COMPANION_PREFIX }) {
    Object.assign(this, { mujoco, ort, getWorld, getSessions, getSpatialContext, paused, locked, quack, wheee, spawnBall, switchLoco, prefix });
    this.sitSettle = new SitSettle();
    this.guard = new AutonomyGuard({ getSpatial: () => this.spatial(), cancelMotion: () => this.stop() });
    this.intent = new LocomotionIntent({ beginStand: () => this.stand(), startPulse: action => this.launch(action) });
    this.resolve();
  }
  resolve(loco = this.loco) {
    this.loco = loco;
    const { model } = this.getWorld();
    this.qAdr = model.jnt(`${this.prefix}trunk_base_freejoint`).qposadr;
    this.vAdr = model.jnt(`${this.prefix}trunk_base_freejoint`).dofadr;
    this.qposAdr = JOINT_NAMES.map(name => model.jnt(this.prefix + name).qposadr);
    this.dofAdr = JOINT_NAMES.map(name => model.jnt(this.prefix + name).dofadr);
    this.ctrlAdr = JOINT_NAMES.map(name => this.mujoco.mj_name2id(model, this.mujoco.mjtObj.mjOBJ_ACTUATOR.value, this.prefix + name));
    this.gyroAdr = model.sensor(`${this.prefix}imu_ang_vel`).adr;
    this.trunkId = this.mujoco.mj_name2id(model, this.mujoco.mjtObj.mjOBJ_BODY.value, `${this.prefix}trunk_base`);
    this.extraJoints = [];
    for (let id = 0; id < model.njnt; id++) {
      const name = this.mujoco.mj_id2name(model, this.mujoco.mjtObj.mjOBJ_JOINT.value, id);
      if (name?.startsWith(this.prefix) && (!this.prefix ? !/^duck\d+_/.test(name) : true) && model.jnt_type[id] === 3 && !JOINT_NAMES.includes(name.slice(this.prefix.length))) this.extraJoints.push({ name: name.slice(this.prefix.length), adr: model.jnt_qposadr[id] });
    }
  }
  qpos() { return this.getWorld().data.qpos.slice(this.qAdr, this.qAdr + 7); }
  spatial() { return measureSpatial(this.qpos(), this.getSpatialContext()); }
  pose() {
    const qpos = this.qpos();
    return { position: Array.from(qpos.slice(0, 3)), headingRad: measureSpatial(qpos).headingRad, loco: this.loco };
  }
  checkTurn(action, duration = 2, command = movementFor(action, this.loco)) {
    if (!action.startsWith("turn_") && action !== "walk_backward") return true;
    const path = sweptMotion(this.pose(), command, this.getSpatialContext(), duration);
    if (path.allowed) return true;
    this.guard.guardReason = path.reason; this.guard.guardSeq++; this.stop(); return false;
  }
  gravity() {
    // Read the flat state array without allocating an Embind body accessor.
    // Native allocations can grow the heap and detach previously read views.
    const q = this.getWorld().data.xquat, a = this.trunkId * 4;
    const w = q[a], x = q[a + 1], y = q[a + 2], z = q[a + 3];
    return [2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1];
  }
  fallen() {
    const z = this.getWorld().data.qpos[this.qAdr + 2], gz = this.gravity()[2];
    return !Number.isFinite(z) || !Number.isFinite(gz) || gz > -.5 || z < .02;
  }
  settled() {
    // Finish nested native reads before obtaining the velocity view. A model
    // allocation during gravity must not leave this check holding old memory.
    const gravityZ = this.gravity()[2];
    const { data } = this.getWorld(), v = data.qvel;
    return gravityZ < -.9 && data.qpos[this.qAdr + 2] > .085
      && Math.hypot(...v.slice(this.vAdr, this.vAdr + 3)) < .08
      && Math.hypot(...data.sensordata.slice(this.gyroAdr, this.gyroAdr + 3)) < .7;
  }
  busy() { return this.locked() || !!this.transition || !!this.recovery || !!this.oneShot || !!this.intent.pendingAction || this.sitSettle.pending || this.postKick > 0 || !!this.pulse || !!this.coast; }
  stop() {
    if (this.loco === "rollers" && this.pulse && !this.pulse.head) this.coast = { elapsed: 0, stableFor: 0 };
    this.pulse = null; this.intent.cancel(); this.headTarget.fill(0);
  }
  setAutonomy(active) {
    this.guard.setActive(active && !this.paused() && !this.locked() && !this.error && !this.sitSettle.error && !this.manual && this.inferenceCount > 0);
  }
  reset() {
    this.stop(); this.setAutonomy(false); this.sitSettle.cancel();
    this.mode = "walk"; this.sitFlag = 0; this.transition = null; this.oneShot = null; this.recovery = null;
    this.fallSteps = 0; this.postKick = 0; this.coast = null; this.error = null; this.mouth = 0; this.quackFor = 0;
    this.lastAction.fill(0); this.headSmooth.fill(0); this.cmd.fill(0);
    const { model, data } = this.getWorld();
    for (let i = 0; i < 7; i++) data.qpos[this.qAdr + i] = model.key_qpos[this.qAdr + i];
    for (let i = 0; i < 6; i++) data.qvel[this.vAdr + i] = 0;
    for (let j = 0; j < NUM_JOINTS; j++) {
      data.qpos[this.qposAdr[j]] = DEFAULT_POSE[j]; data.qvel[this.dofAdr[j]] = 0; data.ctrl[this.ctrlAdr[j]] = DEFAULT_POSE[j];
    }
    for (const joint of this.extraJoints) { data.qpos[joint.adr] = 0; }
    this.mujoco.mj_forward(model, data);
  }
  stand() {
    if (this.mode !== "sitstand" || this.sitFlag !== 1) return;
    this.sitFlag = 0;
    this.transition = { remaining: 2, done: () => { this.mode = "walk"; this.lastAction.fill(0); } };
  }
  launch(action) {
    const move = action === "walk_backward" ? reverseProfile(this.pose(), this.loco, this.getSpatialContext()) ?? movementFor(action, this.loco) : movementFor(action, this.loco);
    if (move) {
      if (!this.checkTurn(action, 2, move)) return false;
      if (!this.guard.admit(action, move[0])) return false;
      this.pulse = { action, remaining: 2, command: move };
      return true;
    }
    if (!actionRoom(action, this.spatial().clearance)) {
      this.guard.guardReason = "Body gesture guard: space changed while this duck stood up.";
      this.guard.guardSeq++;
      this.stop();
      return false;
    }
    this.guard.accepted();
    const mode = { roll: "roll", kick_left: "kickL", kick_right: "kickR", ground_pick: "groundpick", crouch: "crouch" }[action];
    if (!mode) return false;
    this.mode = mode; this.sitFlag = 0; this.headTarget.fill(0);
    this.oneShot = { steps: 0, phase: 0, tipped: false };
    return true;
  }
  command(action, { manual = false } = {}) {
    if (action === "stop") { this.setAutonomy(false); this.stop(); return { accepted: true, completion: "immediate", message: "Motion inputs cleared. Balance control continues." }; }
    if (action === "reset") { this.reset(); return { accepted: true, message: "Selected robot reset." }; }
    if (this.error || this.sitSettle.error) return { accepted: false, message: "Reset the selected robot before continuing." };
    if (this.paused()) return { accepted: false, message: "The simulator is paused." };
    if (this.busy() || this.fallen() || this.manual && !manual) return { accepted: false, message: "Robot busy: wait for action completion or recovery." };
    if (action === "quack" || action === "wheee") { this.quackFor = .48; (action === "quack" ? this.quack : this.wheee)(); return { accepted: true, completion: "immediate", message: action === "quack" ? "Quack audio triggered." : "Bounded one-second voice playback started." }; }
    if (action === "open_mouth" || action === "close_mouth") { this.mouth = action === "open_mouth" ? 1 : 0; return { accepted: true, completion: "immediate", message: "Visual beak position changed; there is no grasp actuator in this model." }; }
    if (action === "spawn_ball") { this.spawnBall(); return { accepted: true, completion: "immediate", message: "Ball placement applied near the selected robot." }; }
    if (action === "switch_to_legs" || action === "switch_to_rollers") {
      const next = action === "switch_to_legs" ? "legs" : "rollers";
      if (next === this.loco) return { accepted: true, completion: "immediate", message: "Already using that locomotion variant." };
      const space = this.spatial().clearance;
      if (next === "rollers" && (space.front < .85 || space.left < .25 || space.right < .25)) return { accepted: false, message: "Changing to rollers needs clear space ahead for the trained model to settle." };
      if (this.mode !== "walk" || !this.settled()) return { accepted: false, message: "Stand still and balanced before changing locomotion." };
      this.switchLoco(next); return { accepted: true, message: "Locomotion model change started." };
    }
    if (action === "sit" || action === "stand") {
      if (this.loco !== "legs") return { accepted: false, message: "This posture needs legs." };
      if (action === "sit" && this.mode === "sitstand" || action === "stand" && this.mode === "walk") return { accepted: true, completion: "immediate", message: "Already in that posture." };
      this.stop(); this.guard.accepted();
      if (action === "stand") this.stand();
      else { this.mode = "sitstand"; this.sitFlag = 0; this.lastAction.fill(0); this.sitSettle.request(); this.transition = { remaining: .8, done: () => { this.sitFlag = 1; } }; }
      return { accepted: true, message: "Sit/stand policy transition started." };
    }
    if (HEAD_COMMANDS[action]) {
      this.guard.accepted(); this.headTarget.set(HEAD_COMMANDS[action]);
      this.pulse = { action, remaining: 2, command: [0, 0, 0], head: true };
      return { accepted: true, message: "Head command started for two seconds." };
    }
    const move = movementFor(action, this.loco);
    const learned = ["roll", "kick_left", "kick_right", "ground_pick", "crouch"].includes(action);
    if (!move && !learned) return { accepted: false, message: "That action is not supported." };
    if (learned && (action === "crouch" ? this.loco !== "rollers" : this.loco !== "legs")) return { accepted: false, message: "That learned behavior belongs to the other locomotion variant." };
    if (!actionRoom(action, this.spatial().clearance)) return { accepted: false, message: "There is too little room for that body gesture." };
    if (move && !this.guard.admit(action, move[0])) return { accepted: false, blockedByGuard: true, message: this.guard.guardReason };
    if (this.mode === "sitstand" && this.sitFlag === 1) {
      this.intent.request(action); return { accepted: true, message: "Standing first, then starting the requested behavior after measured balance." };
    }
    return this.launch(action) ? { accepted: true, message: "Policy execution started." } : { accepted: false, blockedByGuard: !!this.guard.guardReason, message: this.guard.guardReason || "The behavior could not start." };
  }
  buildObs(command = [0, 0, 0]) {
    const { data } = this.getWorld(); let i = 0;
    for (let axis = 0; axis < 3; axis++) this.obs[i++] = data.sensordata[this.gyroAdr + axis];
    for (const g of this.gravity()) this.obs[i++] = g;
    for (let joint = 0; joint < NUM_JOINTS; joint++) this.obs[i++] = data.qpos[this.qposAdr[joint]] - DEFAULT_POSE[joint];
    for (const adr of this.dofAdr) this.obs[i++] = data.qvel[adr];
    for (const value of this.lastAction) this.obs[i++] = value;
    this.cmd.fill(0);
    if (!this.recovery) {
      if (this.mode === "sitstand") this.cmd[0] = this.sitFlag;
      else if (["crouch", "groundpick"].includes(this.mode)) { this.cmd[0] = Math.cos(2 * Math.PI * this.oneShot.phase); this.cmd[1] = Math.sin(2 * Math.PI * this.oneShot.phase); }
      else if (this.mode === "walk" && !this.transition && !this.postKick && !this.coast && !this.locked()) this.cmd.set(this.pulse?.command ?? command, 0);
    }
    for (let h = 0; h < 4; h++) {
      this.headSmooth[h] += .2 * (this.headTarget[h] - this.headSmooth[h]);
      if (!this.recovery && this.mode !== "groundpick") this.cmd[3 + h] = this.headSmooth[h];
    }
    for (const value of this.cmd) this.obs[i++] = value;
    return this.obs;
  }
  async beforeStep(command = [0, 0, 0]) {
    if (this.guard.active && this.pulse) this.guard.monitor(this.pulse.action);
    if (this.guard.active && this.pulse) this.checkTurn(this.pulse.action, .25, this.pulse.command);
    if (this.recovery?.state === "fallen") return;
    const key = this.recovery ? "stand" : this.mode === "walk" && this.loco === "rollers" ? "drive" : this.mode;
    const session = this.getSessions()[key];
    const result = await session.run({ [session.inputNames[0]]: new this.ort.Tensor("float32", this.buildObs(command), [1, OBS_SIZE]) });
    const actions = result[session.outputNames[0]].data;
    if (actions.length !== NUM_JOINTS || !Array.from(actions).every(Number.isFinite)) throw new Error("Companion policy returned a non-finite action.");
    this.lastAction.set(actions); this.inferenceCount++;
    const { data } = this.getWorld();
    for (let j = 0; j < NUM_JOINTS; j++) data.ctrl[this.ctrlAdr[j]] = DEFAULT_POSE[j] + actions[j] * ACTION_SCALE;
  }
  afterStep(dt = CTRL_DT) {
    if (this.pulse && (this.pulse.remaining -= dt) <= 1e-8) {
      if (this.pulse.head) this.headTarget.fill(0);
      else if (this.loco === "rollers") this.coast = { elapsed: 0, stableFor: 0 };
      this.pulse = null;
    }
    this.quackFor = Math.max(0, this.quackFor - dt);
    if (this.postKick > 0) this.postKick--;
    if (this.transition && (this.transition.remaining -= dt) <= 1e-8) { const done = this.transition.done; this.transition = null; done(); }
    const { data } = this.getWorld();
    if (this.coast) {
      this.coast.elapsed += dt;
      this.coast.stableFor = Math.hypot(...data.qvel.slice(this.vAdr, this.vAdr + 3)) < .04 && this.gravity()[2] < -.9 ? this.coast.stableFor + dt : 0;
      if (this.coast.elapsed > .5 && this.coast.stableFor >= .3) this.coast = null;
      else if (this.coast.elapsed > 12) { this.error = "The roller actor did not settle. Reset before continuing."; this.coast = null; }
    }
    this.sitSettle.tick(dt, { activated: this.mode === "sitstand" && this.sitFlag === 1 && !this.transition, settled: data.qpos[this.qAdr + 2] < .085 && data.qpos[this.qAdr + 2] > .03 && this.gravity()[2] < -.9 && Math.hypot(...data.qvel.slice(this.vAdr, this.vAdr + 3)) < .05 && Math.hypot(...data.sensordata.slice(this.gyroAdr, this.gyroAdr + 3)) < .7 });
    this.intent.tick(dt, { interrupted: this.paused() || this.manual || !!this.recovery || this.fallen(), transitioning: !!this.transition, walking: this.mode === "walk", settled: this.settled() });
    if (this.oneShot) {
      const shot = this.oneShot; shot.steps++;
      if (this.mode === "roll") {
        shot.tipped ||= this.gravity()[2] > -.3;
        if ((shot.tipped && this.gravity()[2] < -.85 && shot.steps >= 40) || shot.steps >= 150) { this.oneShot = null; this.mode = "walk"; this.lastAction.fill(0); }
      } else if (this.mode === "kickL" || this.mode === "kickR") {
        if (shot.steps >= 25) { this.oneShot = null; this.mode = "walk"; this.postKick = 20; }
      } else {
        shot.phase += dt / (this.mode === "crouch" ? CROUCH_PERIOD_S : GROUND_PICK_PERIOD_S);
        if (shot.phase >= (this.mode === "crouch" ? CROUCH_END_PHASE : GROUND_PICK_END_PHASE)) {
          if (this.mode === "crouch") this.coast = { elapsed: 0, stableFor: 0 };
          this.oneShot = null; this.mode = "walk";
        }
      }
    }
    const z = data.qpos[this.qAdr + 2], gz = this.gravity()[2];
    if (!Number.isFinite(z) || !Number.isFinite(gz)) { this.reset(); return; }
    if (this.recovery) {
      this.recovery.steps++;
      if (this.recovery.state === "fallen" && this.recovery.steps >= 15) { this.recovery = { state: "recovering", steps: 0, upright: 0 }; this.lastAction.fill(0); }
      else if (this.recovery.state === "recovering") {
        this.recovery.upright = gz < -.85 ? this.recovery.upright + 1 : 0;
        if (this.recovery.upright >= 50) { this.recovery = null; this.lastAction.fill(0); }
        else if (this.recovery.steps >= 300) this.reset();
      }
    } else if (this.fallen() && !this.oneShot && !this.transition && !this.sitSettle.pending) {
      if (++this.fallSteps >= (this.mode === "walk" && this.loco === "legs" ? 10 : 50)) {
        this.stop(); this.fallSteps = 0;
        if (this.mode === "walk" && this.loco === "legs") this.recovery = { state: "fallen", steps: 0, upright: 0 };
        else this.reset();
      }
    } else this.fallSteps = 0;
    if (this.guard.active && this.pulse) this.guard.monitor(this.pulse.action);
  }
  jaw() {
    const phase = this.mode === "groundpick" ? this.oneShot?.phase : null;
    const pick = phase != null && phase >= .2 && phase <= .4 ? 1 : 0;
    const flap = this.quackFor > 0 ? Math.sin(Math.PI * this.quackFor / .48) : 0;
    return Math.min(1, pick + Math.max(this.mouth, flap));
  }
  status() {
    const fallen = this.fallen(), pendingAction = this.intent.pendingAction;
    const posture = fallen || this.recovery ? "fallen" : this.transition || this.sitSettle.pending || pendingAction || this.oneShot || this.postKick || this.coast ? "transitioning" : this.mode === "sitstand" && this.sitFlag === 1 ? "sitting" : "standing";
    const key = this.recovery ? "stand" : this.mode === "walk" && this.loco === "rollers" ? "drive" : this.mode;
    const state = {
      ready: this.inferenceCount > 0 && !this.error && !this.sitSettle.error,
      loco: this.loco, mode: this.mode, time: Number(this.getWorld().data.time), position: Array.from(this.qpos().slice(0, 3)), ...this.spatial(),
      posture, pendingAction, fallen, busy: this.busy() || this.manual, paused: this.paused(), manual: this.manual,
      turnClear: { left: sweptMotion(this.pose(), movementFor("turn_left", this.loco), this.getSpatialContext()).allowed, right: sweptMotion(this.pose(), movementFor("turn_right", this.loco), this.getSpatialContext()).allowed },
      reverseClear: reverseProfile(this.pose(), this.loco, this.getSpatialContext()) !== null,
      phase: this.paused() ? "paused" : this.recovery ? "recovering" : pendingAction ? this.intent.phase : this.sitSettle.pending ? this.sitSettle.phase : this.transition ? "transitioning" : this.oneShot ? "performing" : this.coast ? "settling" : this.pulse ? "moving" : "idle",
      command: this.mode === "walk" ? Array.from(this.cmd.slice(0, 3)) : [0, 0, 0], tiltRad: Math.acos(Math.max(-1, Math.min(1, -this.gravity()[2]))),
      autonomyActive: this.guard.active, guardReason: this.guard.guardReason, guardSeq: this.guard.guardSeq,
      policy: POLICIES[key], inferenceCount: this.inferenceCount,
      ...(this.error || this.sitSettle.error ? { error: this.error || this.sitSettle.error } : {}),
    };
    return { ...state, availableActions: availableDuckActions(state) };
  }
}

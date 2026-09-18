// Microduck RL playground core: the REAL trained policies, not a procedural
// waddle. Framework-agnostic port of the pre-React rl.js.
//
// Physics runs in MuJoCo compiled to WebAssembly (the official
// @mujoco/mujoco bindings), stepping the same MJCF the policies were
// trained on (pollen-robotics/microduck_rl). The controller is one of the
// exported ONNX checkpoints from pollen-robotics/microduck, executed with
// onnxruntime-web at 50 Hz (timestep 0.005 s, decimation 4) - exactly the
// loop from microduck_rl/scripts/infer_policy.py.
//
// Obs layout (61D, "new-cmd-obs" flavor, from the ONNX metadata):
//   [base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
//    last_action(14), command(13)]
//
// Integration contract with the React shell:
//   - bootGame({ scene, camera, renderer }) is called once from inside the
//     R3F canvas; it loads everything, wires inputs and starts the 50 Hz
//     control loop.
//   - frame(dt) is called by R3F's useFrame every animation frame; it does
//     everything the old rAF loop did EXCEPT renderer.render (R3F renders).
//   - UI state flows out through the zustand store (throttled), UI intents
//     flow back in through gameApi.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { signed } from "./signed.js";
import {
  POLICIES, JOINT_NAMES, DEFAULT_POSE, NUM_JOINTS, OBS_SIZE, CMD_SIZE,
  ACTION_SCALE, TIMESTEP, DECIMATION, CTRL_DT,
  VEL_FWD, VEL_BACK, VEL_ANG, RVEL_FWD, RVEL_BACK, RVEL_ANG,
  CROUCH_PERIOD_S, CROUCH_END_PHASE,
  GROUND_PICK_PERIOD_S, GROUND_PICK_END_PHASE,
  BALL_RADIUS, BALL_PARK_POS, ARENA_HALF, SPAWN_X, SPAWN_Y,
  RELIEF_BUMPS, RELIEF_HMAX, RELIEF_GRID, RELIEF_SINK, RELIEF_RATE,
} from "./constants.js";
import { loadProps, propColliders } from "./props.js";
import {
  buildRig, cloneRig, loadKinematics, setJoint, setJawOpen, MODEL_DIR, MESH_VERSION,
  loadGlbGeometries, geometryToBinaryStl,
} from "./duck.js";
import {
  VARIANTS, materialHookFor, DEFAULT_VARIANT, applyVariant,
} from "./variants.js";
import { Controller } from "./controls/controller.js";
import { KeyboardSource } from "./controls/keyboard.js";
import { GamepadSource } from "./controls/gamepad.js";
import { haptics } from "./haptics.js";
import { TouchSource } from "./controls/touch.js";
import { WaypointSource } from "./controls/waypoint.js";
import { ExternalCommandSource } from "./controls/external.js";
import { LocomotionIntent } from "./controls/locomotion-intent.js";
import { SitSettle } from "./controls/sit-settle.js";
import { applyEyeCameraPose } from "./eye-camera.js";
import { measureSpatial, AutonomyGuard } from "./spatial-guard.js";
import { buildParkModelXml, copyNamedPhysicsState, DUCK_IDS } from "./shared-model.js";
import { CompanionController } from "./companion-controller.js";
import { BALL_TASK_ACTIONS, BallTaskController } from "./ball-task.js";
import { SwarmController, scenarioSpawns } from "./swarm-controller.js";
import { createSwarmVisual } from "./swarm-visual.js";
import { HEAD_COMMANDS, actionRoom, availableDuckActions, reverseProfile } from "./duck-actions.js";
import { obstacleForSlot, canPlaceObstacle, followCommand, sweptMotion } from "./park-geometry.js";
import { createPresentation, PRESENTATION_CAMERAS } from "./presentation.js";
import { EMBEDDED, embeddedSuspended, registerEmbeddedRuntime, notifyEmbeddedManualInput, notifyEmbeddedStop } from "./embedded.js";
import * as fx from "./fx/fx-wireframe.js";
import { createWaypointMarker } from "./fx/waypoint-marker.js";
import { createCeremony, CAM_RESET_S } from "./ceremony.js";
import {
  audioCtx, busNode, preloadSfx, playSfx, playUrl, updateListener,
  createEmitter, startAmbient, setAmbientDucked, playEntranceSweep,
  playPropSweep, playLineBlip, setRumble,
} from "./audio.js";
import { createBallActor } from "./ball-actor.js";
import { loadCustomPolicy as loadCustomPolicyModule, scriptCommandAt } from "./custom-policy.js";
import { initGhosts } from "./ghosts.js";
import { makeInfiniteGrid, makeArenaWalls } from "./arena.js";
import { createBallVisual } from "./ball-visual.js";
import { useGame, gameApi, bootLine, bootNote, bootHalt } from "../store.js";

// Physics + inference runtimes are vendored npm dependencies (no CDN):
// everything visitors execute is built from package-lock-verified
// tarballs and served from the Space itself, closing the jsDelivr
// supply-chain surface. The .wasm binaries ride the bundle as hashed
// assets via Vite ?url imports; the JS modules stay dynamic imports so
// they land in their own lazy chunks like before.
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

let bootStarted = false;

// Trunk yaw from the freejoint quat (MuJoCo wxyz), Z-up so this is rotation
// about z. Shared by the chase cam and the waypoint follower - both need
// "which way is the duck facing" in MJCF ground coords.
function duckYaw(qpos) {
  return Math.atan2(
    2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
    1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
  );
}

// HMR teardown for the ghost session: invalidating this module (directly or
// via an edit to ghosts.js) used to stack a live 15 Hz broadcast interval
// plus a ghost room per reload (the historical "stale module" bug class).
const liveGhostSessions = new Set();
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    for (const g of liveGhostSessions) g.destroy();
    liveGhostSessions.clear();
  });
}

export async function bootGame({ scene, camera, renderer }) {
  if (bootStarted) return;
  bootStarted = true;
  try {
    await boot({ scene, camera, renderer });
  } catch (err) {
    console.error("[game] boot failed", err);
    bootHalt(err?.message || String(err));
  }
}

async function boot({ scene, camera, renderer }) {
  const setStore = useGame.setState;
  const store = useGame.getState;

  bootNote("Microduck BIOS v1.0");
  bootLine("MEMORY CHECK")("640K OK");
  bootLine("DUCK FIRMWARE")("PRESENT");

  // Surface async boot failures in the BIOS halt screen. Gated on the boot
  // still being in flight: post-boot async noise (ghost relay hiccups,
  // audio autoplay rejections...) must NOT cue the halt screen.
  const bootGuard = (e, msg) => {
    if (!store().bootDone && !store().bootFailed) bootHalt(msg);
  };
  window.addEventListener("unhandledrejection", (e) => {
    console.error("[game] unhandled rejection", e.reason);
    bootGuard(e, e.reason?.message || String(e.reason));
  });
  window.addEventListener("error", (e) => {
    console.error("[game] window error", e.message);
    bootGuard(e, e.message);
  });

  // Halting at the failure site: a rejected await inside this async boot
  // would otherwise only surface through the caller's catch.
  const traced = (label, p) => {
    const done = bootLine(label);
    return p.then(
      (v) => { done("OK"); return v; },
      (err) => {
        done("FAILED");
        console.error(`[game] ${label} FAILED`, err);
        bootHalt(err?.message || String(err));
        throw err;
      },
    );
  };

  // ── Runtimes (vendored, lazy chunks) ─────────────────────────────────
  const [{ default: loadMujocoFactory }, ort] = await traced(
    "RUNTIME MODULES",
    Promise.all([
      import("@mujoco/mujoco"),
      // wasm-only build: the sessions only ever use the "wasm" execution
      // provider, and the default entry would emit the 26 MB WebGPU (jsep)
      // wasm into the dist for nothing.
      import("onnxruntime-web/wasm"),
    ]),
  );
  // The bundler build embeds its JS loader; only the .wasm binary is
  // fetched at runtime, from our own hashed asset.
  ort.env.wasm.wasmPaths = { wasm: ortWasmUrl };
  ort.env.wasm.numThreads = 1; // static hosting sends no COOP/COEP headers

  // ── MJCF preparation ────────────────────────────────────────────────
  // robot_allcollisions.xml is what infer_policy.py's scene.xml includes:
  // it carries body/shell collision geoms that robot_walk.xml lacks, which
  // the sitstand policy needs (a sit rests the trunk on the ground).
  // Visual meshes are irrelevant to the dynamics: every body carries an
  // explicit <inertial>, and visual geoms have contype=0 conaffinity=0.
  // Stripping them means the MuJoCo VFS only needs the ~10 meshes
  // referenced by collision geoms. Works for both variants.
  const robotSources = new Map();
  async function robotSource(file) {
    if (!robotSources.has(file)) robotSources.set(file, fetch(signed(`${MODEL_DIR}/${file}`)).then(response => {
      if (!response.ok) throw new Error(`Robot model load failed: ${response.status}`);
      return response.text();
    }));
    return robotSources.get(file);
  }
  async function buildPhysicsXml(xmlFile, peerFile = "robot_allcollisions.xml", options = {}) {
    const peerFiles = EMBEDDED ? Array.isArray(peerFile) ? peerFile : [peerFile] : [];
    const [source, ...peers] = await Promise.all([robotSource(xmlFile), ...peerFiles.map(robotSource)]);
    return buildParkModelXml(source, peers, propColliders(), options);
  }

  // ── Boot physics + policy in parallel with the render rig ────────────
  const [mujoco, { xml, meshFiles }, k] = await Promise.all([
    traced("MUJOCO WASM", loadMujocoFactory({
      // Emscripten sidecar resolution: point at the Vite-emitted asset
      // instead of a path relative to the module's own URL.
      locateFile: (p) => (p.endsWith(".wasm") ? mujocoWasmUrl : p),
    })),
    traced("PHYSICS MJCF", buildPhysicsXml("robot_allcollisions.xml")),
    traced("KINEMATICS", loadKinematics(`${MODEL_DIR}/kinematics.json`)),
  ]);

  const doneMeshes = bootLine("MESH ASSETS");
  const vfs = new mujoco.MjVFS();
  // One shared VFS for both variants; already-loaded files are skipped so
  // the roller lazy-load only fetches its leftover meshes.
  const vfsFiles = new Set();
  async function addMeshesToVfs(files) {
    const geoms = await loadGlbGeometries();
    await Promise.all(
      files.map(async (f) => {
        if (vfsFiles.has(f)) return;
        vfsFiles.add(f);
        // Legs/body meshes live in the visual GLB: rebuild binary STL in
        // memory so MuJoCo never triggers a second download. Roller-only
        // files (not in the GLB) still fetch as STL.
        const entry = geoms.get(f);
        const buf = entry
          ? geometryToBinaryStl(entry.welded)
          : await (await fetch(signed(`${MODEL_DIR}/meshes/${f}?v=${MESH_VERSION}`), { cache: "force-cache" })).arrayBuffer();
        // meshdir="assets" in the MJCF, so the compiler looks up "assets/<f>".
        vfs.addBuffer(`assets/${f}`, new Uint8Array(buf));
      }),
    );
  }
  try {
    await addMeshesToVfs(meshFiles);
  } catch (err) {
    doneMeshes("FAILED");
    bootHalt(err?.message || String(err));
    throw err;
  }
  doneMeshes(`${meshFiles.length} FILES`);

  const sessions = {};
  // Always boot on the classic (orange) colourway; the quickbar re-skins live.
  let currentVariant = DEFAULT_VARIANT;
  // A deeper blue keeps the companion distinguishable under warm studio
  // lights. This only changes render materials, never the robot model.
  const companionShell = { color: [.025, .28, .65], roughness: .4, metalness: 0 };
  const companionVariant = { ...VARIANTS.blue, headDome: companionShell, bodyShell: companionShell, sideShells: companionShell, legShells: companionShell };
  const duckNames = { duck1: "Sunny", duck2: "Blue", duck3: "Sage", duck4: "Plum" };
  function variantForPeer(id) {
    if (id === "duck2") return companionVariant;
    const shell = { color: id === "duck3" ? [.08, .55, .3] : [.52, .17, .68], roughness: .4, metalness: 0 };
    return { ...VARIANTS.blue, headDome: shell, bodyShell: shell, sideShells: shell, legShells: shell };
  }
  const rigPromise = (async () => {
    const doneRig = bootLine("RENDER RIG");
    try {
      const builtRig = await buildRig(k, { materialForMesh: materialHookFor(VARIANTS[currentVariant]) });
      doneRig("OK");
      return builtRig;
    } catch (err) {
      doneRig("FAILED");
      bootHalt(err?.message || String(err));
      throw err;
    }
  })();
  // Boot policies with a live [n/7] counter on the BIOS line.
  const donePolicies = bootLine("LOADING POLICIES");
  const sessionOpts = { executionProviders: ["wasm"] };
  let policiesLoaded = 0;
  const bootPolicy = (url) =>
    ort.InferenceSession.create(signed(url), sessionOpts).then((s) => {
      donePolicies.progress(`${++policiesLoaded}/7`);
      return s;
    });
  try {
    [sessions.walk, sessions.sitstand, sessions.roll, sessions.kickL, sessions.kickR,
     sessions.groundpick, sessions.stand] =
      await Promise.all([
        bootPolicy(POLICIES.walk),
        bootPolicy(POLICIES.sitstand),
        bootPolicy(POLICIES.roll),
        bootPolicy(POLICIES.kickL),
        bootPolicy(POLICIES.kickR),
        bootPolicy(POLICIES.groundpick),
        bootPolicy(POLICIES.stand),
      ]);
  } catch (err) {
    donePolicies("FAILED");
    bootHalt(err?.message || String(err));
    throw err;
  }
  donePolicies("7/7");

  // ── Community move slots ──────────────────────────────────────────────
  // A validated move mounts according to its manifest (Academy D7):
  //   perpetual -> sessions.walk (default) or sessions.sitstand (R key)
  //   episodic  -> sessions.roll, played as a one-shot "trick" on R
  //   script    -> a 13-D command timeline on the walker, toggled by R
  // Every other slot (recovery, kicks, rollers) stays official. The
  // official sessions are pinned here so a swap is always reversible.
  const officialWalk = sessions.walk;
  const officialSitstand = sessions.sitstand;
  const officialRoll = sessions.roll;
  let customPolicy = null; // loadCustomPolicy result: { kind, slot, session?, script?, ... }
  let scriptRun = null; // { t } while a kind "script" move drives the commands
  // True while the custom session is the one producing this step's actions.
  const customDriving = () =>
    !!(customPolicy?.session && activeSession() === customPolicy.session);
  // Watchdog: abnormal-ending timestamps (recovery entries + solver
  // explosions) while a custom policy drives the duck. 3 within a rolling
  // 20 s window auto-reverts; ~10 s without a new event clears the slate
  // so player-caused falls don't slowly accumulate into a revert.
  const WATCHDOG_WINDOW_MS = 20000;
  const WATCHDOG_LIMIT = 3;
  const WATCHDOG_HEALTHY_MS = 10000;
  let watchdogEvents = [];

  const actionScaleFor = (session) =>
    (customPolicy && session === customPolicy.session ? customPolicy.actionScale : ACTION_SCALE);

  function revertCustomPolicy(revertReason = null) {
    if (!customPolicy) return;
    const prev = customPolicy;
    customPolicy = null;
    scriptRun = null;
    watchdogEvents = [];
    sessions.walk = officialWalk;
    sessions.sitstand = officialSitstand;
    sessions.roll = officialRoll;
    if (revertReason) {
      console.warn(`[policy] reverted "${prev.name}": ${revertReason}`);
      setStore({
        customPolicy: { ...storeMove(prev), status: "error", revertReason },
      });
    } else {
      setStore({ customPolicy: null });
    }
  }

  // Store projection of a loaded move (no session / script objects in React).
  const storeMove = (m) => ({
    ref: m.ref, name: m.name, title: m.title ?? m.name, author: m.author ?? null,
    kind: m.kind ?? "perpetual", slot: m.slot ?? "walk", academy: m.academy ?? null,
  });

  function watchdogAbnormalEnd(kind) {
    if (!customPolicy) return;
    const now = performance.now();
    watchdogEvents.push(now);
    watchdogEvents = watchdogEvents.filter((t) => now - t < WATCHDOG_WINDOW_MS);
    if (watchdogEvents.length >= WATCHDOG_LIMIT) {
      revertCustomPolicy(`${watchdogEvents.length} ${kind === "exploded" ? "explosions/falls" : "falls"} in ${Math.round(WATCHDOG_WINDOW_MS / 1000)} s`);
    }
  }

  // One load at a time; a failure (or a bogus boot-time ?policy=) leaves
  // the official walker untouched and surfaces the error in the store.
  let policyLoadBusy = false;
  async function loadCustomPolicyIntoGame(ref) {
    if (policyLoadBusy) return;
    policyLoadBusy = true;
    setStore({ customPolicy: { ref: String(ref), name: String(ref), status: "loading" } });
    try {
      const loaded = await loadCustomPolicyModule(ref, { ort });
      revertCustomPolicy(); // drop any previous custom move first
      customPolicy = loaded;
      watchdogEvents = [];
      if (loaded.slot === "walk") sessions.walk = loaded.session;
      else if (loaded.slot === "sitstand") sessions.sitstand = loaded.session;
      else if (loaded.slot === "trick") sessions.roll = loaded.session;
      // "script": nothing to swap, the R key starts the command player.
      setStore({ customPolicy: { ...storeMove(loaded), status: "active" } });
      console.info(`[policy] custom ${loaded.kind} move active in slot ${loaded.slot}: ${loaded.name} (${loaded.ref})`);
    } catch (e) {
      const error = e?.message || String(e);
      console.warn(`[policy] load failed for "${ref}":`, e);
      setStore({ customPolicy: { ref: String(ref), name: String(ref), status: "error", error } });
    } finally {
      policyLoadBusy = false;
    }
  }

  const doneCompile = bootLine("COMPILING PHYSICS");
  let model, data;
  let companion = null;
  let companionRig = null;
  const physicalPeers = new Map();
  let selectedDuckId = "duck1";
  let followEnabled = false;
  let followState = "Follow control disabled.";
  let followDrive = [0, 0, 0];
  let obstacleSlot = "off";
  let obstacleMesh = null;
  let parkSwitching = false;
  let parkSettle = null;
  let controlStepPending = null;
  let initializedPhysics = false;
  let primaryExternalMouth = 0;
  let voiceTimer = null;
  let ballTask = null;
  let swarm = null;
  let swarmPreparing = false;
  let swarmPrepareToken = 0;
  let pendingSwarmRunId = null;
  function peerController(id) { return physicalPeers.get(id)?.controller ?? null; }
  function allDuckIds() { return ["duck1", ...physicalPeers.keys()]; }
  function refreshSelectedPeer() {
    const peer = physicalPeers.get(selectedDuckId === "duck1" ? "duck2" : selectedDuckId);
    companion = peer?.controller ?? null;
    companionRig = peer?.rig ?? null;
    if (peer) companionRigs = peer.rigs;
  }
  function primaryPose() {
    const q = data.qpos;
    return { position: Array.from(q.slice(0, 3)), headingRad: duckYaw(q), loco, command: Array.from(cmd.slice(0, 3)),
      posture: recovery || poseIsDead() ? "fallen" : sitTimer || standTimer || sitSettle?.pending || coastSettle || parkSettle || !["walk", "sitstand"].includes(mode) || postKickLock > 0 ? "transitioning" : mode === "sitstand" && sitFlag === 1 ? "sitting" : "standing",
      fallen: !!recovery || !!poseIsDead(), paused: embeddedPaused || embeddedSuspended() };
  }
  function companionPose() {
    if (!companion) return null;
    const status = companion.status();
    return { position: status.position, headingRad: status.headingRad, loco: status.loco, command: status.command, posture: status.posture, fallen: status.fallen, paused: status.paused };
  }
  function spatialWorld(id = "duck1") {
    // Read freejoint poses directly; spatial status must never recursively
    // ask peers for their own spatial status.
    const peers = allDuckIds().filter(other => other !== id).map(other => {
      const q = other === "duck1" ? data.qpos : peerController(other).qpos();
      return { id: other, position: Array.from(q.slice(0, 3)), headingRad: duckYaw(q) };
    });
    return { obstacle: obstacleForSlot(obstacleSlot), peers };
  }
  function swarmPoses() {
    return [{ id: "duck1", ...primaryPose(), busy: protectedMotionBusy(), speedMps: Math.hypot(...data.qvel.slice(0, 3)) },
      ...[...physicalPeers].map(([id, peer]) => {
        const actor = peer.controller, fallen = actor.fallen();
        const posture = fallen || actor.recovery ? "fallen" : actor.transition || actor.sitSettle.pending || actor.intent.pendingAction || actor.oneShot || actor.postKick || actor.coast ? "transitioning"
          : actor.mode === "sitstand" && actor.sitFlag === 1 ? "sitting" : "standing";
        return { id, ...actor.pose(), posture, busy: actor.busy(), fallen, paused: actor.paused(), command: Array.from(actor.cmd.slice(0, 3)),
          speedMps: Math.hypot(...data.qvel.slice(actor.vAdr, actor.vAdr + 3)) };
      })];
  }
  function stopSwarm(reason = "Swarm simulation stopped.") {
    swarmPrepareToken++;
    swarm?.stop(reason);
    for (const peer of physicalPeers.values()) { peer.controller.stop(); peer.controller.setAutonomy(false); }
    stopExternalCommand();
    followEnabled = false; followDrive = [0, 0, 0];
  }
  try {
    model = mujoco.MjModel.from_xml_string(xml, vfs);
    data = new mujoco.MjData(model);
  } catch (err) {
    doneCompile("FAILED");
    bootHalt(err?.message || String(err));
    throw err;
  }
  doneCompile("COMPILED");

  // Addresses resolved once per compiled variant. qpos/qvel/sensordata
  // views are re-read at each use: the WASM heap can grow and detach
  // earlier TypedArray views.
  const JOINT_SET = new Set(JOINT_NAMES);
  function resolveAddrs(model, kin) {
    return {
      qposAdr: JOINT_NAMES.map((n) => model.jnt(n).qposadr),
      dofAdr: JOINT_NAMES.map((n) => model.jnt(n).dofadr),
      gyroAdr: model.sensor("imu_ang_vel").adr,
      trunkId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk_base"),
      standKeyId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND"),
      ballQposAdr: model.jnt("ball_freejoint").qposadr,
      ballDofAdr: model.jnt("ball_freejoint").dofadr,
      // Foot bodies for the footstep audio heuristic (-1 when a variant
      // has no ankles, e.g. if a future model renames them).
      ankleIds: ["ankle_left", "ankle_right"].map(
        (n) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, n)),
      // Unactuated hinges (the roller variant's 4 passive wheels): not in
      // the obs or ctrl, but synced to the render rig so the wheels spin.
      extraJoints: kin.bodies
        .filter((b) => b.joint && b.joint.type === "hinge" && !JOINT_SET.has(b.joint.name))
        .map((b) => ({ name: b.joint.name, adr: model.jnt(b.joint.name).qposadr })),
    };
  }
  // Active-variant address block, swapped wholesale by activateLoco.
  let { qposAdr, dofAdr, gyroAdr, trunkId, standKeyId, ballQposAdr, ballDofAdr, extraJoints,
    ankleIds } = resolveAddrs(model, k);

  // Locomotion variants stay resident once built (model + data + rig +
  // addresses); legs is registered when its render rig resolves below.
  const locos = {};
  let loco = "legs"; // "legs" | "rollers"
  const velLims = () => ((selectedDuckId !== "duck1" && companion ? companion.loco : loco) === "rollers"
    ? [RVEL_FWD, RVEL_BACK, RVEL_ANG]
    : [VEL_FWD, VEL_BACK, VEL_ANG]);

  const lastAction = new Float32Array(NUM_JOINTS);
  const obs = new Float32Array(OBS_SIZE);
  const cmd = new Float32Array(CMD_SIZE); // [vx, vy, wz, head(4), body(6)]
  // Input controller: keyboard + gamepad + touch sources merged into one
  // continuous command + discrete action surface, in priority order.
  const kbSource = new KeyboardSource({ getVelocityLimits: () => velLims() });
  const padSource = new GamepadSource({ getVelocityLimits: () => velLims() });
  const touchSource = new TouchSource({ getVelocityLimits: () => velLims() });
  // Click-to-walk (PLAN.md Project 1 Phase A): reads zero until a floor
  // click arms a target, so - like the keyboard before it - it doubles as
  // the fallback. Any keyboard/pad/touch input preempts it by arbitration
  // order alone; getManualOverride also cancels the pending target outright
  // so releasing manual input doesn't snap the duck back onto a stale click.
  const waypointSource = new WaypointSource({
    camera, renderer,
    getVelocityLimits: () => velLims(),
    getDuckPose: () => {
      const qpos = selectedDuckId !== "duck1" && companion ? companion.qpos() : data.qpos;
      return [qpos[0], qpos[1], duckYaw(qpos)];
    },
    isSuppressed: () => inputLocked || headMode || (selectedDuckId !== "duck1" && companion ? companion.mode : mode) !== "walk" || !!grab,
    getManualOverride: () => padSource.isActive() || touchSource.isActive() || kbSource.isActive(),
  });
  const manualSources = [padSource, touchSource, kbSource, waypointSource];
  const manualInputActive = () => manualSources.some((source) => source.isActive()
    || Object.values(source.pressed ?? {}).some(Boolean)
    || Object.values(source.axes ?? {}).some((value) => Math.abs(value) > 0.01));
  let externalAction = null;
  const externalSource = EMBEDDED ? new ExternalCommandSource({
    getManualOverride: () => manualInputActive() || !!grab,
    onCancel: () => {
      if (loco === "rollers" && externalAction && movementProfile(externalAction)) coastSettle = { elapsed: 0, stableFor: 0 };
      externalAction = null; releaseExternalHead();
    },
  }) : null;
  // All direct controls have priority over the bounded parent pulse.
  const controller = new Controller({ sources: [...manualSources, ...(externalSource ? [externalSource] : [])] });
  if (externalSource) for (const source of manualSources) {
    const dispatch = source.onAction;
    source.onAction = (action, meta) => { ballTask?.cancel(); if (swarm?.active || swarmPreparing) stopSwarm("Manual input interrupted group control."); stopExternalCommand(); notifyEmbeddedManualInput(); dispatch(action, meta); };
  }
  // Right-stick camera state, read by the telemetry before the camera-orbit
  // section below has evaluated.
  let padOrbitLive = false;
  // Robot input gate: twist commands, mode changes, rolls, kicks and ball
  // spawns all stay inert until the entrance sequence has fully played out.
  let inputLocked = true;
  let ceremony = null;
  let ball = null;
  let stickers = null; // comic popups, currently disabled

  let mode = "walk"; // "walk" | "sitstand" | "roll" | "kickL" | "kickR" | "crouch" | "groundpick"
  let sitFlag = 0;
  const isKick = () => mode === "kickL" || mode === "kickR";

  // HEAD mode (runtime-faithful, pad Y): locomotion is zeroed and both
  // sticks drive the head command slots cmd[3..6] = [neck_pitch,
  // head_pitch, head_yaw, head_roll]. Targets are stick * HEAD_MAX,
  // EMA-smoothed at 50 Hz in buildObs like the runtime (alpha 0.2).
  // Offsets PERSIST when leaving head mode; only a sim reset zeroes them.
  let headMode = false;
  const HEAD_MAX = 2.5; // rad at full deflection (runtime head_max)
  const HEAD_ALPHA = 0.2;
  // Stick-to-joint polarity (deflections come in up/left = +1), tuned
  // visually in the sim: cmd + means UP for neck_pitch but DOWN for
  // head_pitch, LEFT for head_yaw but RIGHT-tilt for head_roll - hence
  // the mixed signs, so stick up = look up, stick left = turn/tilt left.
  // Order matches cmd[3..6] = [neck_pitch, head_pitch, head_yaw, head_roll].
  const HEAD_SIGNS = new Float32Array([1, -1, 1, -1]);
  const headTarget = new Float32Array(4);
  const headSmooth = new Float32Array(4);
  let externalHead = false;
  let locomotionIntent = null;
  const sitSettle = EMBEDDED ? new SitSettle() : null;
  let embeddedPaused = embeddedSuspended();
  let runtimeError = null;
  let inferenceCount = 0;
  const autonomyGuard = EMBEDDED ? new AutonomyGuard({
    getSpatial: () => measureSpatial(data.qpos, spatialWorld()),
    cancelMotion: () => stopExternalCommand(),
  }) : null;
  function releaseExternalHead() {
    if (!externalHead) return;
    externalHead = false;
    headTarget.fill(0);
  }
  function stopExternalCommand() {
    externalSource?.cancel();
    externalAction = null;
    releaseExternalHead();
    locomotionIntent?.cancel();
  }
  function setAutonomy(active) {
    const wasActive = selectedDuckId !== "duck1" ? companion?.guard.active : autonomyGuard?.active;
    if (active === true && !wasActive) ballTask?.clearTerminal();
    if (active === false && wasActive) ballTask?.cancel("Ball task stopped when autonomous control ended.");
    if (selectedDuckId !== "duck1" && companion) {
      autonomyGuard?.setActive(false);
      for (const peer of physicalPeers.values()) if (peer.controller !== companion) peer.controller.setAutonomy(false);
      companion.setAutonomy(active);
      return;
    }
    for (const peer of physicalPeers.values()) peer.controller.setAutonomy(false);
    autonomyGuard?.setActive(active === true && store().bootDone && inferenceCount > 0
      && !embeddedPaused && !embeddedSuspended() && !runtimeError && !sitSettle?.error
      && !manualInputActive() && !controller.neutralHeld && !grab);
  }
  function checkAutonomousMotion() {
    if (swarm?.active && (embeddedPaused || embeddedSuspended() || manualInputActive() || grab)) stopSwarm("Manual control or pause interrupted the swarm.");
    if (selectedDuckId !== "duck1") {
      if (embeddedPaused || embeddedSuspended() || manualInputActive() || grab) setAutonomy(false);
      return;
    }
    if (!autonomyGuard?.active) return;
    if (embeddedPaused || embeddedSuspended() || manualInputActive() || grab) { setAutonomy(false); return; }
    const action = externalSource.isActive() ? externalAction : locomotionIntent?.pendingAction;
    if (action) autonomyGuard.monitor(action);
    if ((action?.startsWith("turn_") || action === "walk_backward") && externalSource.isActive()) {
      const path = sweptMotion(primaryPose(), Array.from(externalSource.command), spatialWorld(), .25);
      if (!path.allowed) { autonomyGuard.guardReason = path.reason; autonomyGuard.guardSeq++; stopExternalCommand(); }
    }
  }
  // Local-only kickable ball: false while parked at the keyframe spot
  // (mesh hidden), true once popped in front of the duck.
  let ballActive = false;

  // The twist the policy actually receives. Mid-roll every movement input
  // is ignored (zero twist) until the roll hands back to walk on its own.
  // HEAD mode also zeroes it: the runtime stops the robot while the
  // sticks drive the head.
  const ZERO_CMD = new Float32Array(3);
  function effectiveCmd() {
    if (inputLocked || (headMode && selectedDuckId === "duck1") || mode === "roll" || mode === "crouch" ||
        mode === "groundpick" || isKick() || postKickLock > 0 || coastSettle || recovery)
      return ZERO_CMD;
    if (ballTask?.owns("duck1")) return ballTask.command;
    if (swarm?.active) return swarm.commandFor("duck1");
    if (EMBEDDED && selectedDuckId !== "duck1") return followEnabled ? followDrive : ZERO_CMD;
    return controller.getCommand();
  }
  let rollRun = null;
  let crouchRun = null;
  let pickRun = null;
  let kickRun = null;
  let KICK_STEPS = 25;
  // Post-kick grace: keep commands zeroed for a beat after the kick window
  // hands back to walk. Step-counted like everything else.
  const POST_KICK_LOCK_STEPS = 20; // 0.4 s at 50 Hz
  let postKickLock = 0;
  let coastSettle = null;

  // Pending mode-transition timers (sit hand-over, stand-up hand-back).
  let sitTimer = null;
  let standTimer = null;
  let fallenSince = null;

  // ── Automatic fall recovery (legs walk mode only) ────────────────────
  // Mirrors the runtime's --fall-detect state machine (main.rs ~3658):
  // a debounced tip (gz > -0.5 for 0.2 s) freezes ctrl on the current
  // pose for a short settle (the runtime goes limp), then hands the duck
  // to the stand policy with all commands zeroed until it's been upright
  // (gz < -0.85) for a full second. If it can't get up within 6 s, fall
  // back to the old kill: resetSim + materialization. Rollers keep the
  // plain kill (the runtime declares fall-detect roller-incompatible),
  // and so do sit/roll/kick/crouch/groundpick and the entrance lock.
  const FALL_DEBOUNCE_STEPS = 10; // 0.2 s of gz > -0.5 before triggering
  const FALL_SETTLE_STEPS = 15; // 0.3 s ctrl freeze once triggered
  const RECOVER_UPRIGHT_STEPS = 50; // 1 s of gz < -0.85 to declare recovered
  const RECOVER_GIVEUP_STEPS = 300; // 6 s of stand attempts before reset
  let recovery = null; // null | { state: "fallen"|"recovering", steps, uprightSteps }
  let fallDebounce = 0;

  function clearModeTimers() {
    if (!EMBEDDED) { clearTimeout(sitTimer); clearTimeout(standTimer); }
    sitTimer = null;
    standTimer = null;
    sitSettle?.cancel();
  }
  function modeTimer(callback, milliseconds) {
    // A hidden embedded page stops simulation. Count these handovers in
    // simulation steps so a pause cannot swap actors before standing up.
    return EMBEDDED ? { remaining: milliseconds / 1000, callback } : setTimeout(callback, milliseconds);
  }
  function tickEmbeddedModeTimers() {
    if (!EMBEDDED) return;
    for (const timer of [sitTimer, standTimer]) {
      if (timer && (timer.remaining -= CTRL_DT) <= 1e-8) timer.callback();
    }
  }

  // ── Mouse grab, physics side (MuJoCo-viewer-style perturbation) ───────
  // While a grab is live, EVERY PHYSICS SUBSTEP writes a spring-damper
  // force on the grabbed free body via xfrc_applied (world frame),
  // pulling it toward the cursor target. Formula and gains mirror the
  // native viewer's mjv_applyPerturbForce (engine_vis_interact.c):
  //   F = -stiffness*mass*(pos - ref) - sqrt(stiffness)*mass*vel
  // with stiffness = m->vis.map.stiffness default (100) and the damping
  // coefficient sqrt(stiffness) exactly as MuJoCo computes it. Two
  // deliberate departures: mass is the subtree mass instead of the
  // Jacobian-derived localmass (equivalent for a free body pulled at its
  // root, and the bindings expose no mj_jac), and the force acts
  // torque-free at the freejoint origin instead of at the picked point
  // (the viewer adds moment_arm x F; skipping it avoids spinning the duck
  // the walking policy would then fight). The early per-CONTROL-step
  // version of this (50 Hz zero-order hold, damping on 4-substep-stale
  // velocity) was the jitter the user felt: a stiff spring held over 20 ms
  // limit-cycles. Per-substep application is what the viewer does.
  // Pointer wiring (raycast pick, target plane, cursor) lives after the
  // camera section below; this block stays above resetSim so the control
  // loop and resets can reference it during boot.
  const GRAB_STIFFNESS = 100; // MuJoCo vis.map.stiffness default
  const GRAB_DAMPING = Math.sqrt(GRAB_STIFFNESS); // viewer's damping coefficient
  const GRAB_MAX_ACC = 200; // safety clamp only - the viewer has none
  let grab = null; // { bodyId, qAdr, dofAdr, mass, target: [x,y,z] MJCF }
  let endGrabHook = () => {}; // reassigned by the pointer wiring
  function applyGrabForce() {
    if (!grab) return;
    const qpos = data.qpos, qvel = data.qvel;
    let fx = GRAB_STIFFNESS * (grab.target[0] - qpos[grab.qAdr]) - GRAB_DAMPING * qvel[grab.dofAdr];
    let fy = GRAB_STIFFNESS * (grab.target[1] - qpos[grab.qAdr + 1]) - GRAB_DAMPING * qvel[grab.dofAdr + 1];
    let fz = GRAB_STIFFNESS * (grab.target[2] - qpos[grab.qAdr + 2]) - GRAB_DAMPING * qvel[grab.dofAdr + 2];
    const n = Math.hypot(fx, fy, fz);
    if (n > GRAB_MAX_ACC) {
      const s = GRAB_MAX_ACC / n;
      fx *= s; fy *= s; fz *= s;
    }
    // Fresh view each call: the WASM heap can grow and detach old ones.
    const xfrc = data.xfrc_applied;
    const a = grab.bodyId * 6;
    xfrc[a] = grab.mass * fx;
    xfrc[a + 1] = grab.mass * fy;
    xfrc[a + 2] = grab.mass * fz;
  }
  function releaseGrabForce() {
    if (!grab) return;
    const xfrc = data.xfrc_applied;
    const a = grab.bodyId * 6;
    xfrc[a] = 0; xfrc[a + 1] = 0; xfrc[a + 2] = 0;
    grab = null;
  }

  function resetSim({ all = false, preparingSwarm = false } = {}) {
    if (all && !preparingSwarm) stopSwarm("The world was reset.");
    if (all) ballTask?.clear();
    else ballTask?.finish("failed", "The robot required a physical reset during ball interaction.");
    stopExternalCommand();
    parkSettle = null;
    autonomyGuard?.setActive(false);
    // A live grab must not survive a reset: the per-step spring would
    // immediately yank the respawned duck toward the stale cursor target.
    endGrabHook();
    // Single reset path: Space, fall-kill, failed roll, loco switch.
    clearModeTimers();
    rollRun = null;
    scriptRun = null;
    kickRun = null;
    crouchRun = null;
    pickRun = null;
    postKickLock = 0;
    coastSettle = null;
    fallenSince = null;
    recovery = null;
    fallDebounce = 0;
    mode = "walk";
    // Head mode exits and its offsets DO reset here (the one place).
    headMode = false;
    padSource.headMode = false;
    headTarget.fill(0);
    headSmooth.fill(0);
    primaryExternalMouth = 0;
    if (!EMBEDDED || !initializedPhysics || all) {
      mujoco.mj_resetDataKeyframe(model, data, standKeyId);
      initializedPhysics = true;
      for (const peer of physicalPeers.values()) peer.controller.reset();
      if (all) { followEnabled = false; for (const peer of physicalPeers.values()) peer.controller.setAutonomy(false); }
    } else {
      // Automatic recovery only resets this duck. The peer and shared ball
      // keep their physical state, and the simulation clock stays monotonic.
      for (let i = 0; i < 7; i++) data.qpos[i] = model.key_qpos[standKeyId * model.nq + i];
      for (let i = 0; i < 6; i++) data.qvel[i] = 0;
      for (let j = 0; j < NUM_JOINTS; j++) {
        data.qpos[qposAdr[j]] = DEFAULT_POSE[j]; data.qvel[dofAdr[j]] = 0; data.ctrl[j] = DEFAULT_POSE[j];
      }
      for (const joint of extraJoints) data.qpos[joint.adr] = 0;
    }
    if (EMBEDDED && model.nmocap > 0) data.mocap_pos.set(obstacleForSlot(obstacleSlot).position, 0);
    mujoco.mj_forward(model, data);
    lastAction.fill(0);
    sitFlag = 0;
    // Park the ball in physics immediately; if it was on screen, the
    // reverse scan peels it away at its last pose. A queued B-respawn is
    // cancelled: a reset means no ball.
    if (!EMBEDDED || all || !ball) {
      ball?.despawn({ cancelQueued: true, parkPhysics: parkBallPhysics });
      ballActive = false;
    }
    syncButtons();
    ceremony?.playRespawn();
  }
  resetSim();

  function parkBallPhysics() {
    const qpos = data.qpos, qvel = data.qvel;
    qpos[ballQposAdr] = 50;
    qpos[ballQposAdr + 1] = 0;
    qpos[ballQposAdr + 2] = BALL_RADIUS;
    qpos[ballQposAdr + 3] = 1;
    qpos[ballQposAdr + 4] = 0;
    qpos[ballQposAdr + 5] = 0;
    qpos[ballQposAdr + 6] = 0;
    for (let i = 0; i < 6; i++) qvel[ballDofAdr + i] = 0;
    mujoco.mj_forward(model, data);
    ballActive = false;
  }

  // Pop / respawn the ball ~0.35 m in front of the duck, with a small
  // random heading + distance jitter. If the ball is already on screen,
  // peel it away first (reverse scan) and pop the new one when that
  // finishes - same appear/disappear pair as the duck's wireframe ceremony.
  function spawnBall(opts = {}) {
    if (inputLocked && !opts.fromQueue) return;
    if (!ball) return;
    ballTask?.cancel("Ball task cancelled because the ball was repositioned.");
    if (ball.visual !== "hidden") {
      ball.queueRespawn();
      ball.despawn({ parkPhysics: parkBallPhysics });
      return;
    }
    const qpos = data.qpos, qvel = data.qvel;
    const origin = opts.duckId && opts.duckId !== "duck1" && peerController(opts.duckId) ? peerController(opts.duckId).qpos() : qpos;
    const yaw = duckYaw(origin);
    const heading = yaw + (Math.random() - 0.5) * 0.7;
    const dist = 0.35 + (Math.random() - 0.5) * 0.1;
    const lim = ARENA_HALF - BALL_RADIUS - 0.05;
    const clamp = (v) => Math.min(lim, Math.max(-lim, v));
    qpos[ballQposAdr] = clamp(origin[0] + Math.cos(heading) * dist);
    qpos[ballQposAdr + 1] = clamp(origin[1] + Math.sin(heading) * dist);
    qpos[ballQposAdr + 2] = BALL_RADIUS + 0.02;
    qpos[ballQposAdr + 3] = 1;
    qpos[ballQposAdr + 4] = 0;
    qpos[ballQposAdr + 5] = 0;
    qpos[ballQposAdr + 6] = 0;
    for (let i = 0; i < 6; i++) qvel[ballDofAdr + i] = 0;
    mujoco.mj_forward(model, data);
    ballActive = true;
    // Snap the mesh to the new pose BEFORE the scan starts: the FX
    // recomputes its bbox from the live mesh.
    ball.poseFromQpos(qpos, ballQposAdr);
    ball.appear();
    stickers?.pop("spawn");
  }

  // ── Observation ─────────────────────────────────────────────────────
  const _q = new THREE.Quaternion();
  const _g = new THREE.Vector3();

  function buildObs() {
    const qpos = data.qpos, qvel = data.qvel, sens = data.sensordata;
    let i = 0;
    for (let a = 0; a < 3; a++) obs[i++] = sens[gyroAdr + a];
    // projected gravity: world -z rotated into the trunk frame
    // A body accessor allocates native memory and can detach the state views
    // above if the WASM heap grows. The flat xquat array needs no accessor.
    const xq = data.xquat, xa = trunkId * 4; // [w, x, y, z]
    _q.set(xq[xa + 1], xq[xa + 2], xq[xa + 3], xq[xa]).conjugate();
    _g.set(0, 0, -1).applyQuaternion(_q);
    obs[i++] = _g.x; obs[i++] = _g.y; obs[i++] = _g.z;
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qpos[qposAdr[j]] - DEFAULT_POSE[j];
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qvel[dofAdr[j]];
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = lastAction[j];
    // command: walking/drive use the twist; sitstand uses cmd[0] as the
    // posture flag; the crouch-glide one-shot carries its phase encoding in
    // the vel slots (ground-pick convention: [cos, sin, 0]).
    cmd.fill(0, 0, 3);
    if (mode === "sitstand") {
      cmd[0] = sitFlag;
    } else if (mode === "crouch" && crouchRun) {
      const a = 2 * Math.PI * crouchRun.phase;
      cmd[0] = Math.cos(a);
      cmd[1] = Math.sin(a);
    } else if (mode === "groundpick" && pickRun) {
      const a = 2 * Math.PI * pickRun.phase;
      cmd[0] = Math.cos(a);
      cmd[1] = Math.sin(a);
    } else {
      const c = effectiveCmd();
      cmd[0] = c[0]; cmd[1] = c[1]; cmd[2] = c[2];
    }
    // Head slots cmd[3..6]: EMA toward the stick targets at 50 Hz (this
    // runs once per control step), exactly the runtime's smoothing. Kept
    // filled outside head mode too - offsets persist like on the robot.
    for (let h = 0; h < 4; h++) headSmooth[h] += HEAD_ALPHA * (headTarget[h] - headSmooth[h]);
    // Ground pick parity: the runtime zero-pads the head (and body) slots
    // for its obs (mjlab's zero_command_padding), so persisted head
    // offsets must not leak into the pick policy's command buffer. Fall
    // recovery zeroes them too: the stand policy gets an all-zero command.
    const gpZero = mode === "groundpick" || recovery !== null;
    cmd[3] = gpZero ? 0 : headSmooth[0]; cmd[4] = gpZero ? 0 : headSmooth[1];
    cmd[5] = gpZero ? 0 : headSmooth[2]; cmd[6] = gpZero ? 0 : headSmooth[3];
    // Command script (kind "script" move): the timeline owns the whole
    // 13-D command vector while it plays on the walker. Movement input is
    // ignored until R stops it (or the script ends, when it doesn't loop).
    if (scriptRun && mode === "walk" && !recovery && customPolicy?.script) {
      scriptCommandAt(customPolicy.script, scriptRun.t, cmd);
    }
    for (let c = 0; c < CMD_SIZE; c++) obs[i++] = cmd[c];
    return obs;
  }

  // The ONNX session for the current mode: in the roller variant the main
  // velocity mode runs the drive (skating) policy instead of the walker,
  // and the fall-recovery state machine overrides everything with the
  // get-up policy while it owns the duck.
  const activeSession = () => {
    if (recovery?.state === "recovering") return sessions.stand;
    return sessions[loco === "rollers" && mode === "walk" ? "drive" : mode];
  };

  // ── Control loop (50 Hz, async because ONNX inference is async) ──────
  let ctrlHz = 0;

  // Fresh projected-gravity z straight from the trunk pose (buildObs is
  // skipped during the fall-recovery settle, so obs[5] can go stale).
  function projGravZ() {
    const xq = data.xquat, xa = trunkId * 4; // [w, x, y, z]
    _q.set(xq[xa + 1], xq[xa + 2], xq[xa + 3], xq[xa]).conjugate();
    _g.set(0, 0, -1).applyQuaternion(_q);
    return _g.z;
  }

  // Dead pose: "fallen" = trunk tilted past ~60 deg or sunk below the
  // floor. NaN/Inf is a solver explosion: no grace, reset on the spot.
  // In legs walk mode a debounced fall now goes to the recovery state
  // machine instead of the kill; everywhere else (rollers, sit, one-shots,
  // entrance lock) the old grace-then-reset behavior stands.
  function poseIsDead() {
    const z = data.qpos[2];
    const gz = projGravZ();
    if (!Number.isFinite(z) || !Number.isFinite(gz)) return "exploded";
    if (gz > -0.5 || z < 0.02) return "fallen";
    return null;
  }

  // ── Sim-driven audio (footsteps / roller rumble / ball impacts) ──────
  // Footsteps use a per-foot height heuristic instead of MuJoCo contacts
  // (the WASM bindings expose no contact array): each ankle's height
  // RELATIVE to the lower ankle (the planted foot approximates the local
  // ground, so the measure self-calibrates on the relief terrain), with
  // lift/contact hysteresis and a per-foot debounce. Tap gain scales with
  // the landing speed. Rollers get a speed-following rumble loop instead,
  // and the ball thumps on velocity deltas between control steps.
  const STEP_LIFT = 0.012; // m above the planted foot = foot in swing
  const STEP_CONTACT = 0.005; // m: dropping below this while in swing = step
  const STEP_DEBOUNCE_MS = 130;
  const stepFeet = [
    { air: false, prevZ: 0, lastAt: 0 },
    { air: false, prevZ: 0, lastAt: 0 },
  ];
  const duckEmitter = createEmitter({ refDistance: 0.5 });
  const ballEmitter = createEmitter({ refDistance: 0.5 });
  const ballPrevV = [0, 0, 0];
  let ballPrevValid = false;
  let ballThumpAt = 0;
  // Body bumps: trunk velocity deltas gated on obstacle proximity (walls
  // or prop collider footprints) so the walking gait's own accelerations
  // (comparable in magnitude to a slow wall hit) never false-trigger.
  // Duller than the ball thumps: same samples pitched way down.
  const BUMP_DV = 0.35; // m/s per control step
  const BUMP_DEBOUNCE_MS = 300; // rubbing a wall must not machine-gun
  const BUMP_WALL_PAD = 0.16; // trunk half-width-ish reach to a wall
  const bumpZones = propColliders().map((c) => {
    const [px, py] = c.pos.split(" ").map(Number);
    const [sx, sy] = c.size.split(" ").map(Number);
    return { x: px, y: py, r: Math.hypot(sx, sy) + 0.14 };
  });
  const bumpPrevV = [0, 0, 0];
  let bumpPrevValid = false;
  let bumpAt = 0;
  function nearObstacle(x, y) {
    if (Math.max(Math.abs(x), Math.abs(y)) > ARENA_HALF - BUMP_WALL_PAD) return true;
    for (const z of bumpZones) {
      if (Math.hypot(x - z.x, y - z.y) < z.r) return true;
    }
    return false;
  }

  function stepAudioSim() {
    const now = performance.now();
    if (loco === "legs" && !inputLocked && ankleIds[0] >= 0 && ankleIds[1] >= 0) {
      const xpos = data.xpos;
      const zL = xpos[ankleIds[0] * 3 + 2];
      const zR = xpos[ankleIds[1] * 3 + 2];
      const ground = Math.min(zL, zR);
      for (let i = 0; i < 2; i++) {
        const f = stepFeet[i];
        const z = i === 0 ? zL : zR;
        const rel = z - ground;
        const vz = (z - f.prevZ) / CTRL_DT;
        f.prevZ = z;
        if (rel > STEP_LIFT) {
          f.air = true;
        } else if (f.air && rel < STEP_CONTACT && vz < -0.02 &&
                   now - f.lastAt > STEP_DEBOUNCE_MS) {
          f.air = false;
          f.lastAt = now;
          const u = Math.min(1, Math.abs(vz) / 0.35);
          playSfx("step", {
            gain: 0.12 + 0.14 * u,
            rate: 0.9 + Math.random() * 0.25,
            out: duckEmitter.node,
          });
        }
      }
    }
    // Roller rumble: gain and slight pitch follow ground speed - and the
    // pad's haptic bed mirrors it (texture on the weak motor).
    const speed = Math.hypot(data.qvel[0], data.qvel[1]);
    const rollLevel = loco === "rollers" && !inputLocked ? Math.min(1, speed / 0.5) : 0;
    setRumble(rollLevel, duckEmitter);
    haptics.setBed(rollLevel);
    // Body bumps: trunk |dv| against a nearby wall/prop. The proximity
    // gate keeps gait/kick jerks (which rival slow wall hits) silent.
    if (!inputLocked) {
      const v = data.qvel;
      if (bumpPrevValid) {
        const dv = Math.hypot(v[0] - bumpPrevV[0], v[1] - bumpPrevV[1], v[2] - bumpPrevV[2]);
        if (dv > BUMP_DV && now - bumpAt > BUMP_DEBOUNCE_MS &&
            nearObstacle(data.qpos[0], data.qpos[1])) {
          bumpAt = now;
          const u = Math.min(1, (dv - BUMP_DV) / 1.5);
          playSfx("thump", {
            gain: 0.12 + 0.3 * u,
            rate: 0.55 + 0.15 * u + Math.random() * 0.06, // way below the ball's range
            out: duckEmitter.node,
          });
          haptics.pulse("bump", 0.5 + 0.5 * u); // shove felt in the hands
        }
      }
      bumpPrevV[0] = v[0]; bumpPrevV[1] = v[1]; bumpPrevV[2] = v[2];
      bumpPrevValid = true;
    } else {
      bumpPrevValid = false;
    }
    // Ball impacts: |dv| between control steps. Gravity alone accounts for
    // ~0.2 m/s per 20 ms step; the 0.5 threshold clears it and rolling noise.
    if (ballActive) {
      const v = data.qvel;
      const b = ballDofAdr;
      if (ballPrevValid) {
        const dv = Math.hypot(
          v[b] - ballPrevV[0], v[b + 1] - ballPrevV[1], v[b + 2] - ballPrevV[2]);
        if (dv > 0.5 && now - ballThumpAt > 90) {
          ballThumpAt = now;
          const u = Math.min(1, (dv - 0.5) / 3.5); // full at kick-grade hits
          playSfx("thump", {
            gain: 0.14 + 0.4 * u,
            rate: 1.25 - 0.45 * u + Math.random() * 0.08,
            out: ballEmitter.node,
          });
          // Haptics only when the duck plausibly caused/received the hit:
          // a far bounce off a wall shouldn't shake the hands.
          const bq = ballQposAdr;
          const dBall = Math.hypot(
            data.qpos[bq] - data.qpos[0], data.qpos[bq + 1] - data.qpos[1]);
          if (dBall < 0.6) haptics.pulse("ballHit", 0.4 + 0.6 * u);
        }
      }
      ballPrevV[0] = v[b]; ballPrevV[1] = v[b + 1]; ballPrevV[2] = v[b + 2];
      ballPrevValid = true;
    } else {
      ballPrevValid = false;
    }
    // Haptic channel scheduler: keeps the roller bed alive between pulses
    // and cuts the motors when it falls silent. 50 Hz, like everything here.
    haptics.tick();
  }

  async function controlStep() {
    checkAutonomousMotion();
    ballTask?.tick(CTRL_DT);
    if (swarm?.active) swarm.tick(CTRL_DT, swarmPoses(), { obstacle: obstacleForSlot(obstacleSlot) });
    driveRelief(CTRL_DT); // kinematic terrain, written before the physics steps
    updateFollowDrive();
    // Settle phase: ctrl frozen on the pose held at the fall (approximates
    // the runtime's limp beat), physics keeps stepping, no inference.
    if (recovery?.state !== "fallen") {
      // Tensor names come from the session itself (community exports don't
      // all use obs/actions), and the action scale is per-session (official
      // policies stay at ACTION_SCALE = 1).
      const session = activeSession();
      const feeds = {
        [session.inputNames[0]]: new ort.Tensor("float32", buildObs(), [1, OBS_SIZE]),
      };
      const out = await session.run(feeds);
      inferenceCount++;
      const act = out[session.outputNames[0]].data;
      let finite = true;
      for (let j = 0; j < NUM_JOINTS; j++) {
        if (!Number.isFinite(act[j])) { finite = false; break; }
      }
      if (!finite) {
        // Never write a NaN/Inf into ctrl. A custom policy producing one is
        // instantly reverted; ctrl holds the previous step's targets.
        if (customPolicy && session === customPolicy.session) {
          revertCustomPolicy("non-finite action tensor");
        }
      } else {
        const scale = actionScaleFor(session);
        lastAction.set(act);
        const ctrl = data.ctrl;
        for (let j = 0; j < NUM_JOINTS; j++) ctrl[j] = DEFAULT_POSE[j] + act[j] * scale;
      }
    }
    for (const [id, peer] of physicalPeers) {
      peer.controller.manual = selectedDuckId === id && manualInputActive();
      const peerCommand = ballTask?.owns(id) ? ballTask.command : swarm?.active ? swarm.commandFor(id)
        : selectedDuckId === id ? headMode ? ZERO_CMD : controller.getCommand() : followEnabled ? followDrive : ZERO_CMD;
      await peer.controller.beforeStep(peerCommand);
    }
    for (let s = 0; s < DECIMATION; s++) {
      applyGrabForce(); // mouse perturbation, fresh velocity every substep
      mujoco.mj_step(model, data);
      ballTask?.observePhysics({ mujoco, model, data, ballQposAdr });
    }
    for (const peer of physicalPeers.values()) peer.controller.afterStep(CTRL_DT);
    if (parkSettle) {
      const settlingPeer = peerController(parkSettle.duckId);
      const adr = settlingPeer ? settlingPeer.vAdr : 0;
      const gravity = settlingPeer ? settlingPeer.gravity()[2] : projGravZ();
      const stable = gravity < -.9 && Math.hypot(...data.qvel.slice(adr, adr + 3)) < .04;
      parkSettle.elapsed += CTRL_DT;
      parkSettle.stableFor = stable ? parkSettle.stableFor + CTRL_DT : 0;
      if (parkSettle.elapsed > 1 && parkSettle.stableFor >= .3) parkSettle = null;
      else if (parkSettle.elapsed > 12) {
        runtimeError = "The new locomotion model did not settle. Reset before continuing.";
        parkSettle = null; setAutonomy(false);
      }
    }
    checkAutonomousMotion();
    if (coastSettle) {
      coastSettle.elapsed += CTRL_DT;
      coastSettle.stableFor = Math.hypot(...data.qvel.slice(0, 3)) < .04 && projGravZ() < -.9 ? coastSettle.stableFor + CTRL_DT : 0;
      if (coastSettle.elapsed > .5 && coastSettle.stableFor >= .3) coastSettle = null;
      else if (coastSettle.elapsed > 12) { runtimeError = "The roller actor did not settle. Reset before continuing."; coastSettle = null; setAutonomy(false); }
    }
    stepAudioSim(); // footsteps / rumble / ball thumps off the fresh state

    // Command script clock: 50 Hz like everything here. A non-looping
    // script ends on its own; a looping one runs until R (or a reset).
    if (scriptRun) {
      scriptRun.t += CTRL_DT;
      const sc = customPolicy?.script;
      if (!sc || (!sc.loop && scriptRun.t >= sc.durationS)) {
        scriptRun = null;
        syncButtons();
      }
    }

    // Watchdog housekeeping: a healthy stretch (no abnormal ending for
    // ~10 s) forgives earlier events, so player-caused falls spread over
    // a session never accumulate into a revert.
    if (customPolicy && watchdogEvents.length &&
        performance.now() - watchdogEvents[watchdogEvents.length - 1] > WATCHDOG_HEALTHY_MS) {
      watchdogEvents = [];
    }

    const death = poseIsDead();
    if (death === "exploded") {
      haptics.pulse("explode");
      // Solver explosions only implicate the custom policy while it is the
      // one driving (its slot: walk, sitstand or the trick one-shot);
      // official one-shots and rollers never count.
      if (loco === "legs" && customDriving()) watchdogAbnormalEnd("exploded");
      resetSim();
    } else if (recovery) {
      // Recovery state machine owns the duck: settle -> stand policy ->
      // hysteresis exit (upright for a full second) or 6 s give-up reset.
      recovery.steps++;
      if (recovery.state === "fallen") {
        if (recovery.steps >= FALL_SETTLE_STEPS) {
          recovery = { state: "recovering", steps: 0, uprightSteps: 0 };
          lastAction.fill(0);
          syncButtons();
        }
      } else {
        recovery.uprightSteps = projGravZ() < -0.85 ? recovery.uprightSteps + 1 : 0;
        if (recovery.uprightSteps >= RECOVER_UPRIGHT_STEPS) {
          recovery = null;
          mode = "walk";
          lastAction.fill(0);
          haptics.pulse("recover"); // back on its feet: light double tap
          syncButtons();
        } else if (recovery.steps >= RECOVER_GIVEUP_STEPS) {
          resetSim();
        }
      }
    } else if (death === "fallen") {
      const recoverable = loco === "legs" && mode === "walk" &&
        !inputLocked && postKickLock === 0 && !standTimer;
      if (recoverable) {
        fallenSince = null;
        if (++fallDebounce >= FALL_DEBOUNCE_STEPS) {
          fallDebounce = 0;
          exitHeadMode();
          // Recovery entry = one abnormal ending for the custom-policy
          // watchdog (the rolling window absorbs player-caused falls).
          watchdogAbnormalEnd("fallen");
          recovery = { state: "fallen", steps: 0 };
          // Haptic thud on the confirmed fall (one-shot: this transition
          // fires once per fall, the recovery machine owns the duck after).
          haptics.pulse("fall");
          syncButtons();
        }
      } else {
        fallDebounce = 0;
        const now = performance.now();
        const graceMs = mode === "roll" ? 5000 : 1000;
        // First frame of a non-recoverable fall (rollers, sit, one-shots):
        // same haptic thud, once - fallenSince latches until reset/upright.
        if (fallenSince == null) haptics.pulse("fall");
        fallenSince ??= now;
        if (now - fallenSince > graceMs) {
          // A custom sit / trick that floors the duck counts for the
          // watchdog exactly like a recovery entry does for a walker.
          if (loco === "legs" && customDriving()) watchdogAbnormalEnd("fallen");
          resetSim();
        }
      }
    } else {
      fallDebounce = 0;
      fallenSince = null;
    }

    // Ball respawn watchdog: outside the arena bounds means "escaped
    // through a solver glitch", bring it back near the duck.
    if (ballActive) {
      const q = data.qpos;
      const escaped =
        Math.abs(q[ballQposAdr]) > ARENA_HALF + 0.1 ||
        Math.abs(q[ballQposAdr + 1]) > ARENA_HALF + 0.1;
      if (escaped) spawnBall();
    }

    if (postKickLock > 0 && mode === "walk") postKickLock--;

    // One-shot kick: fixed 0.5 s window like the robot runtime, then
    // straight back to walking. lastAction is NOT zeroed on either swap.
    if (isKick() && kickRun) {
      kickRun.steps++;
      if (kickRun.steps >= KICK_STEPS) {
        kickRun = null;
        mode = "walk";
        postKickLock = POST_KICK_LOCK_STEPS;
        syncButtons();
      }
    }

    // Crouch-glide one-shot: advance the trained phase clock and hand back
    // to the drive policy at the runtime's cycle end.
    if (mode === "crouch" && crouchRun) {
      crouchRun.phase += CTRL_DT / CROUCH_PERIOD_S;
      if (crouchRun.phase >= CROUCH_END_PHASE) {
        crouchRun = null;
        mode = "walk";
        if (EMBEDDED) coastSettle = { elapsed: 0, stableFor: 0 };
        syncButtons();
      }
    }

    // Ground-pick one-shot: same phase-clock pattern as the crouch, ending
    // at the runtime's cycle end (phase 0.7 of a 4 s period, ~2.8 s).
    if (mode === "groundpick" && pickRun) {
      pickRun.phase += CTRL_DT / GROUND_PICK_PERIOD_S;
      if (pickRun.phase >= GROUND_PICK_END_PHASE) {
        pickRun = null;
        mode = "walk";
        syncButtons();
      }
    }

    // One-shot roll, step-counted like the robot runtime: hand back to
    // walking once the trunk has tipped over and is upright again, or
    // after a hard window if the roll never initiated.
    if (mode === "roll" && rollRun) {
      rollRun.steps++;
      if (obs[5] > -0.3) rollRun.tipped = true;
      const upright = obs[5] < -0.85;
      const done = rollRun.tipped && upright && rollRun.steps >= 40;
      // 3 s for the official roll (it should long be over); a custom
      // trick runs for its manifest duration instead.
      const expired = rollRun.steps >= rollRun.maxSteps;
      if (done || expired) {
        const wasCustom = rollRun.custom;
        rollRun = null;
        mode = "walk";
        lastAction.fill(0);
        // Timed out mid-roll: don't hand a tipped duck to the walking
        // policy (it has no get-up skill).
        if (!upright) {
          if (wasCustom) watchdogAbnormalEnd("fallen");
          resetSim();
        } else haptics.pulse("land"); // rolled through and stuck the landing
        syncButtons();
      }
    }
    tickEmbeddedModeTimers();
    sitSettle?.tick(CTRL_DT, {
      activated: mode === "sitstand" && sitFlag === 1 && !sitTimer,
      // Retained runtime poses are about 0.059 m seated / 0.116 m
      // standing. Height alone is insufficient while the body is moving.
      settled: data.qpos[2] < 0.085 && data.qpos[2] > 0.03 && projGravZ() < -0.9
        && Math.hypot(data.qvel[0], data.qvel[1], data.qvel[2]) < 0.05
        && Math.hypot(...data.sensordata.slice(gyroAdr, gyroAdr + 3)) < 0.7,
    });
    locomotionIntent?.tick(CTRL_DT, {
      interrupted: embeddedPaused || embeddedSuspended() || (selectedDuckId === "duck1" && manualInputActive()) || inputLocked || loco !== "legs" || !!recovery || !!poseIsDead(),
      transitioning: !!sitTimer || !!standTimer,
      walking: mode === "walk",
      settled: projGravZ() < -0.9 && data.qpos[2] > 0.085
        && Math.hypot(data.qvel[0], data.qvel[1], data.qvel[2]) < 0.08
        && Math.hypot(...data.sensordata.slice(gyroAdr, gyroAdr + 3)) < 0.7,
    });
  }

  // ── Relief (prototype: the level itself gains gentle slopes) ────────
  // One analytic height function (cosine bumps, RELIEF_BUMPS) drives both
  // surfaces: the MuJoCo heightfield gets it sampled into hfield_data
  // once per compiled model, and the grid floor shader displaces its
  // vertices with the same function (uTopoScale uniform). Raising or
  // sinking the terrain = ramping one scalar that scales the hfield
  // z-size and the shader uniform together, so physics and visuals stay
  // the same surface at every moment of the transition. Trigger for now:
  // window.rl.setRelief(bool) (prototype - no UI yet).
  let reliefOn = false;
  let reliefScale = 0;
  let reliefGridMat = null; // assigned at scene wiring (grid built below)
  const reliefFilled = new WeakSet();
  function topoH(x, y) {
    let H = 0;
    for (const [cx, cy, h, r] of RELIEF_BUMPS) {
      const u = Math.hypot(x - cx, y - cy) / r;
      if (u < 1) H += h * (0.5 + 0.5 * Math.cos(Math.PI * u));
    }
    return H;
  }
  function fillHfield(m) {
    if (reliefFilled.has(m)) return;
    // Re-read the view on every fill: heap growth detaches TypedArrays.
    const n = RELIEF_GRID, hdata = m.hfield_data;
    for (let r = 0; r < n; r++) {
      const y = -ARENA_HALF + (2 * ARENA_HALF * r) / (n - 1);
      for (let c = 0; c < n; c++) {
        const x = -ARENA_HALF + (2 * ARENA_HALF * c) / (n - 1);
        hdata[r * n + c] = topoH(x, y) / RELIEF_HMAX;
      }
    }
    reliefFilled.add(m);
  }
  function driveRelief(dt) {
    fillHfield(model); // no-op once per compiled model (legs / rollers)
    const target = reliefOn ? 1 : 0;
    if (reliefScale !== target) {
      const d = Math.max(-RELIEF_RATE * dt, Math.min(RELIEF_RATE * dt, target - reliefScale));
      reliefScale += d;
    }
    // z-size scales every bump; the floor keeps it strictly positive and,
    // combined with the geom's RELIEF_SINK offset, fully buried when off.
    model.hfield_size[2] = Math.max(reliefScale * RELIEF_HMAX, 1e-4);
    if (reliefGridMat) reliefGridMat.uniforms.uTopoScale.value = reliefScale;
  }
  let running = true;
  (async function controlLoop() {
    let next = performance.now();
    let count = 0, hzT0 = next;
    while (running) {
      if (EMBEDDED && (embeddedPaused || embeddedSuspended() || parkSwitching)) {
        next = performance.now();
        hzT0 = next; count = 0; ctrlHz = 0;
        await new Promise((resolve) => setTimeout(resolve, 100));
        continue;
      }
      try {
        controlStepPending = controlStep();
        await controlStepPending;
      } catch (error) {
        if (!EMBEDDED) throw error;
        runtimeError = error?.message || String(error);
        setAutonomy(false);
        embeddedPaused = true;
        console.error("[embedded simulator] control loop failed", error);
        continue;
      } finally {
        controlStepPending = null;
      }
      count++;
      const now = performance.now();
      if (now - hzT0 > 500) {
        ctrlHz = (count * 1000) / (now - hzT0);
        count = 0; hzT0 = now;
      }
      next += CTRL_DT * 1000;
      const wait = next - performance.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      else next = performance.now(); // fell behind: don't spiral
    }
  })();

  // ── Scene wiring (grid, walls, rig, ball, arcade row) ────────────────
  // The grid/walls carry ceremony-driven uReveal uniforms and per-frame
  // focus updates, so the game owns them; lights and environment live in
  // the R3F layer.
  const grid = makeInfiniteGrid();
  scene.add(grid);
  reliefGridMat = grid.material; // relief drive mirrors uTopoScale into it
  const { wallMats, wallMeshes } = makeArenaWalls();
  for (const m of wallMeshes) scene.add(m);

  let rig = await rigPromise;
  scene.add(rig.placer);
  let trunkGroup = rig.bodies.get("trunk_base");
  locos.legs = {
    model, data, rig, trunkGroup,
    qposAdr, dofAdr, gyroAdr, trunkId, standKeyId, ballQposAdr, ballDofAdr, extraJoints, ankleIds,
  };
  const parkModels = new Map([["legs:legs", { model, data }]]);
  let companionRigs = new Map();
  let rollerKinematics = null;
  function installPhysicalPeer(id, peerRig) {
    const peer = new CompanionController({
      mujoco, ort, getWorld: () => ({ model, data }), getSessions: () => sessions, prefix: `${id}_`,
      getSpatialContext: () => spatialWorld(id),
      paused: () => embeddedPaused || embeddedSuspended(), locked: () => inputLocked || parkSwitching || !!parkSettle,
      quack: () => playChirpFor(id === "duck4" ? "purple" : "blue"), wheee: () => playBoundedWheee(),
      spawnBall: () => spawnBall({ duckId: id }), switchLoco: name => switchParkLoco(id, name),
    });
    physicalPeers.set(id, { controller: peer, rig: peerRig, rigs: new Map([["legs", peerRig]]) });
    scene.add(peerRig.placer);
    return peer;
  }
  async function addPhysicalPeer(id) {
    return installPhysicalPeer(id, await buildRig(k, { materialForMesh: materialHookFor(variantForPeer(id)) }));
  }
  if (EMBEDDED) {
    await addPhysicalPeer("duck2");
    refreshSelectedPeer();
    swarm = new SwarmController();
    obstacleMesh = new THREE.Mesh(new THREE.BoxGeometry(.36, .32, .36), new THREE.MeshStandardMaterial({ color: 0xdcb977, roughness: .8 }));
    obstacleMesh.castShadow = true; obstacleMesh.receiveShadow = true;
    obstacleMesh.visible = false;
    scene.add(obstacleMesh);
  }

  // ── Locomotion variant switching (legs <-> rollers) ──────────────────
  // The roller stack (XML + 5 extra meshes + kinematics + 2 ONNX policies)
  // is lazy-loaded on the first switch, then kept resident.
  let rollersLoading = null;
  function ensureRollers() {
    rollersLoading ??= (async () => {
      const [{ xml: rXml, meshFiles: rMeshFiles }, rk] = await Promise.all([
        buildPhysicsXml("robot_allcollisions_rollers.xml"),
        loadKinematics(`${MODEL_DIR}/kinematics_rollers.json`),
      ]);
      rollerKinematics = rk;
      const [rRig, sDrive, sCrouch] = await Promise.all([
        buildRig(rk, { materialForMesh: materialHookFor(VARIANTS[currentVariant]) }),
        ort.InferenceSession.create(signed(POLICIES.drive), sessionOpts),
        ort.InferenceSession.create(signed(POLICIES.crouch), sessionOpts),
        addMeshesToVfs(rMeshFiles),
      ]);
      sessions.drive = sDrive;
      sessions.crouch = sCrouch;
      const rModel = mujoco.MjModel.from_xml_string(rXml, vfs);
      const rData = new mujoco.MjData(rModel);
      locos.rollers = {
        model: rModel, data: rData, rig: rRig, trunkGroup: rRig.bodies.get("trunk_base"),
        ...resolveAddrs(rModel, rk),
      };
      if (EMBEDDED) parkModels.set("rollers:legs", { model: rModel, data: rData });
    })();
    return rollersLoading;
  }

  // Ghost-only roller rig: kinematics + THREE meshes, no physics model and
  // no ONNX sessions - just enough to render roller-mode PEERS correctly
  // for a visitor who never leaves legs mode (ensureRollers' full stack
  // stays lazy). Kept resident once built; if the player later switches
  // for real, getRigFor prefers the live locos.rollers rig and this one
  // quietly remains as a clone source.
  let ghostRollerRig = null;
  let ghostRollerRigLoading = null;
  function ensureGhostRollerRig() {
    if (locos.rollers || ghostRollerRig || rollersLoading) return;
    ghostRollerRigLoading ??= (async () => {
      const rk = await loadKinematics(`${MODEL_DIR}/kinematics_rollers.json`);
      ghostRollerRig = await buildRig(rk, {
        materialForMesh: materialHookFor(VARIANTS[currentVariant]),
      });
    })().catch((e) => {
      ghostRollerRigLoading = null; // next roller peer retries the load
      console.warn("[ghosts] roller ghost rig load failed", e);
    });
  }

  function activateLoco(name) {
    const L = locos[name];
    loco = name;
    scene.remove(rig.placer);
    ({ model, data, rig, trunkGroup, qposAdr, dofAdr, gyroAdr, trunkId,
       standKeyId, ballQposAdr, ballDofAdr, extraJoints, ankleIds } = L);
    // The rig may have been built (or last shown) under another colourway.
    applyVariant(rig, currentVariant);
    scene.add(rig.placer);
    setStore({ loco: name });
    resetSim();
  }

  let locoSwitching = false;
  async function setLoco(name, { force = false } = {}) {
    if (EMBEDDED) return switchParkLoco(selectedDuckId, name);
    if (name !== "legs" && name !== "rollers") return;
    if (loco === name || locoSwitching) return;
    if (!force && (inputLocked || rollRun || kickRun || crouchRun || pickRun ||
        standTimer || recovery)) return;
    locoSwitching = true;
    setStore({ locoSwitching: true });
    try {
      if (name === "rollers" && !locos.rollers) {
        setStore({ rollersLoading: true });
        await ensureRollers();
      }
      activateLoco(name);
    } catch (e) {
      rollersLoading = null;
      console.error("[game] roller switch failed", e);
    } finally {
      setStore({ rollersLoading: false, locoSwitching: false });
      locoSwitching = false;
    }
  }

  async function switchParkLoco(duckId, name) {
    if (!EMBEDDED || !companion || parkSwitching || parkSettle || !["legs", "rollers"].includes(name)) return;
    const ids = allDuckIds();
    const variants = ids.map(id => id === duckId ? name : id === "duck1" ? loco : peerController(id).loco);
    const nextPrimary = variants[0];
    if ((duckId === "duck1" ? loco : peerController(duckId)?.loco) === name) return;
    const state = duckId === "duck1" ? primaryStatus() : peerController(duckId).status();
    if (state.busy || state.posture !== "standing" || state.fallen) return;
    parkSwitching = true;
    setStore({ locoSwitching: true });
    try {
      // A running ONNX promise still owns the old observation. Let that
      // entire shared physics step finish before exchanging the model.
      if (controlStepPending) await controlStepPending;
      if (variants.includes("rollers")) await ensureRollers();
      const key = variants.join(":");
      if (!parkModels.has(key)) {
        const { xml: nextXml, meshFiles: nextMeshes } = await buildPhysicsXml(
          nextPrimary === "rollers" ? "robot_allcollisions_rollers.xml" : "robot_allcollisions.xml",
          variants.slice(1).map(variant => variant === "rollers" ? "robot_allcollisions_rollers.xml" : "robot_allcollisions.xml"),
          { layout: ids.length === 4 ? "swarm" : "park" },
        );
        await addMeshesToVfs(nextMeshes);
        const nextModel = mujoco.MjModel.from_xml_string(nextXml, vfs);
        parkModels.set(key, { model: nextModel, data: new mujoco.MjData(nextModel) });
      }
      const next = parkModels.get(key);
      mujoco.mj_resetDataKeyframe(next.model, next.data, 0);
      copyNamedPhysicsState(mujoco, model, data, next.model, next.data);
      for (const [index, id] of ids.slice(1).entries()) {
        const peer = physicalPeers.get(id), variant = variants[index + 1];
        if (!peer.rigs.has(variant)) peer.rigs.set(variant, await buildRig(variant === "rollers" ? rollerKinematics : k, { materialForMesh: materialHookFor(variantForPeer(id)) }));
        scene.remove(peer.rig.placer);
      }
      scene.remove(rig.placer);
      model = next.model; data = next.data; loco = nextPrimary;
      rig = locos[nextPrimary].rig;
      trunkGroup = rig.bodies.get("trunk_base");
      ({ qposAdr, dofAdr, gyroAdr, trunkId, standKeyId, ballQposAdr, ballDofAdr, extraJoints, ankleIds } = resolveAddrs(model, nextPrimary === "rollers" ? rollerKinematics : k));
      for (const [index, id] of ids.slice(1).entries()) {
        const peer = physicalPeers.get(id), variant = variants[index + 1];
        peer.rig = peer.rigs.get(variant); peer.controller.resolve(variant);
        scene.add(peer.rig.placer);
      }
      refreshSelectedPeer();
      parkSettle = { duckId, elapsed: 0, stableFor: 0 };
      // Both models share the same measured standing configuration. No
      // reset or time jump, and neither duck loses its action history.
      applyVariant(rig, currentVariant);
      scene.add(rig.placer);
      mujoco.mj_forward(model, data);
      setStore({ loco, locoWant: loco });
      syncButtons();
    } catch (error) {
      runtimeError = error?.message || String(error);
      setAutonomy(false);
      console.error("[duck park] locomotion change failed", error);
    } finally {
      parkSwitching = false;
      setStore({ locoSwitching: false, rollersLoading: false });
    }
  }

  async function toggleLoco() {
    const next = loco === "legs" ? "rollers" : "legs";
    setStore({ locoWant: next });
    await setLoco(next);
  }

  // Quickbar loco intent: reconcile locoWant -> actual, retrying until the
  // game allows the switch (mid-roll, respawn ceremony, ...). Replaces the
  // old index.html reconciler that polled window.rl.
  let locoReconciler = null;
  function reconcileLoco() {
    const want = store().locoWant;
    if (want === loco) {
      if (locoReconciler) { clearInterval(locoReconciler); locoReconciler = null; }
      return;
    }
    if (want === "rollers") ensureRollers().catch(() => {});
    if (!locoSwitching) setLoco(want);
    locoReconciler ??= setInterval(reconcileLoco, 250);
  }
  useGame.subscribe((s) => s.locoWant, reconcileLoco);

  // ── Cutscenes (entrance + respawn) ──────────────────────────────────
  ceremony = createCeremony({
    THREE, scene, camera, renderer, fx,
    getRig: () => rig,
    grid, wallMats,
    syncRig, startCameraReset,
    setLocked: (v) => {
      inputLocked = v;
      controller.setLocked(v);
      // A ball is always in play: pop one the moment the entrance or a
      // respawn ceremony hands control back.
      if (!EMBEDDED && !v && ball && !ballActive) spawnBall({ fromQueue: true });
    },
    flashReset: () => {},
    // Audio twins of the wireframe materialize FX (entrance and respawns):
    // the duck's hero sweep spans the FX's exact duration and follows its
    // ease-out; each prop's scan gets its own smaller, size-pitched sweep
    // the frame it starts; each arena line drawing in gets a tiny blip.
    onScanCue: (durS) => playEntranceSweep(durS),
    onPropCue: (durS) => playPropSweep(durS),
    onLineCue: (u) => playLineBlip(u),
  });

  // ── Ambient bed lifecycle ─────────────────────────────────────────────
  // The Waddle-in click latches `entered` and doubles as the unlock
  // gesture; the hum starts there and ducks whenever the pause/title
  // overlay comes back up. fireImmediately covers ?boot=1 (already
  // entered by the time the game boots).
  useGame.subscribe((s) => s.entered, (entered) => {
    if (!entered) return;
    audioCtx();
    preloadSfx();
    startAmbient();
  }, { fireImmediately: true });
  useGame.subscribe((s) => s.menuOpen, (open) => setAmbientDucked(open),
    { fireImmediately: true });

  const { group: ballGroup, mesh: ballMesh } = createBallVisual(renderer);
  scene.add(ballGroup);
  ball = createBallActor({
    THREE, scene, camera, renderer, fxModule: fx, mesh: ballMesh, group: ballGroup,
  });

  // ── Prop library (wall/corner dressing + entrance FX) ────────────────
  // Every enabled def in props.js: loaded, real-size scaled, floor
  // snapped, wireframe-materialized with the ceremony (staggered after
  // the duck's scan cue). Physics-side, buildPhysicsXml planted one
  // static box per declared collider.
  const propGroups = await loadProps({
    THREE, GLTFLoader, signed, scene, camera, renderer, fx, ceremony,
  });

  // ── Camera: orbit controls + chase cam + reset glide ─────────────────
  const controls = new OrbitControls(camera, renderer.domElement);
  const orbitFitScale = aspect => Math.max(1, .9 / Math.max(.1, aspect));
  controls.target.set(SPAWN_X, 0, -SPAWN_Y); // orbit around the spawn cell
  if (EMBEDDED) {
    controls.target.set(SPAWN_X, .12, -.35);
    const fit = orbitFitScale(camera.aspect);
    camera.position.copy(controls.target).add(new THREE.Vector3(1.15, .9, 1.5).multiplyScalar(fit));
    camera.lookAt(controls.target);
  }
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.25;
  controls.maxDistance = EMBEDDED ? 3 * orbitFitScale(camera.aspect) : 3;
  controls.maxPolarAngle = Math.PI / 2 - 0.03;
  const presentation = EMBEDDED ? createPresentation({ scene, renderer, grid, wallMats }) : null;
  const swarmVisual = EMBEDDED ? createSwarmVisual(scene) : null;
  let cameraMode = EMBEDDED ? "orbit" : "follow";
  let orbitViewportAspect = camera.aspect;
  const resizeOrbitOffset = new THREE.Vector3();
  function refitOrbitAfterResize() {
    if (!EMBEDDED) return;
    const width = renderer.domElement.clientWidth, height = renderer.domElement.clientHeight;
    if (!width || !height) return;
    const aspect = width / height;
    if (Math.abs(aspect - orbitViewportAspect) < 1e-4) return;
    const scale = orbitFitScale(aspect) / orbitFitScale(orbitViewportAspect);
    orbitViewportAspect = aspect;
    controls.maxDistance = (physicalPeers.size > 1 ? 6 : 3) * orbitFitScale(aspect);
    if (cameraMode !== "orbit") return;
    // A viewport resize changes framing, not the user's chosen orbit.
    // Preserve its direction and relative zoom; never recenter each frame.
    resizeOrbitOffset.copy(camera.position).sub(controls.target).multiplyScalar(scale);
    camera.position.copy(controls.target).add(resizeOrbitOffset);
  }
  function frameAllDucks() {
    setCameraMode("orbit");
    const poses = swarmPoses(), center = new THREE.Vector3();
    for (const pose of poses) center.add(new THREE.Vector3(pose.position[0], .13, -pose.position[1]));
    center.multiplyScalar(1 / poses.length);
    let radius = .35;
    for (const pose of poses) radius = Math.max(radius, center.distanceTo(new THREE.Vector3(pose.position[0], .13, -pose.position[1])) + .23);
    const vertical = camera.fov * Math.PI / 180;
    const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * camera.aspect);
    const distance = radius / Math.sin(Math.min(vertical, horizontal) / 2) * 1.1;
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(new THREE.Vector3(1.15, 1.3, 1.5).normalize(), distance);
    controls.maxDistance = Math.max(distance * 1.5, 6 * orbitFitScale(camera.aspect));
    camera.lookAt(center);
  }
  let hiddenHead = null;
  let cameraModel = null;
  let headCameraId = -1;
  function setCameraMode(next) {
    if (!PRESENTATION_CAMERAS.has(next)) return;
    const leavingEyes = cameraMode === "eyes" && next !== "eyes";
    if (hiddenHead) { hiddenHead.visible = true; hiddenHead = null; }
    cameraMode = next;
    chaseCam = next === "follow";
    controls.enabled = next !== "eyes";
    camera.near = next === "eyes" ? 0.004 : 0.02;
    camera.fov = next === "eyes" ? 75 : 40;
    camera.updateProjectionMatrix();
    if (leavingEyes) {
      const q = data.qpos;
      controls.target.set(q[0], q[2], -q[1]);
      camera.position.copy(controls.target).add(new THREE.Vector3(0.55, 0.35, 0.7));
      camera.lookAt(controls.target);
    }
  }
  function updateEyesCamera() {
    const activeCameraModel = `${selectedDuckId}:${loco}:${companion?.loco ?? "legs"}`;
    if (cameraModel !== activeCameraModel) {
      cameraModel = activeCameraModel;
      headCameraId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA.value, selectedDuckId !== "duck1" ? `${selectedDuckId}_head_camera` : "head_camera");
    }
    if (headCameraId < 0) { setCameraMode("orbit"); return; }
    applyEyeCameraPose(camera, data.cam_xpos, data.cam_xmat, headCameraId);
    // Hide the complete neck/head render subtree from the onboard view,
    // including bearings behind jaw_soft that otherwise fill the image.
    // Sensor position, body transforms and collision geometry are intact.
    const head = (selectedDuckId !== "duck1" && companionRig ? companionRig : rig).bodies.get("neck");
    if (hiddenHead && hiddenHead !== head) hiddenHead.visible = true;
    if (head) { head.visible = false; hiddenHead = head; }
  }
  function applyPresentation(options) {
    if (options.scene) presentation?.apply(options.scene);
    if (options.camera && options.camera !== cameraMode) setCameraMode(options.camera);
  }

  // Chase cam (default ON): each frame the camera eases toward a point
  // behind the duck's heading at the current orbit distance, while the
  // orbit target keeps easing to the trunk in syncRig. Implemented by
  // overwriting camera.position AFTER controls.update() so we never fight
  // OrbitControls' own spherical bookkeeping.
  let chaseCam = !EMBEDDED;
  const CHASE_PITCH = 0.42; // rad above horizontal, keeps the floor in view
  const CHASE_EASE = 0.05;
  const _chasePos = new THREE.Vector3();
  const _chaseDir = new THREE.Vector3();
  // During one-shot rolls and kicks the trunk tumbles: hold the last
  // healthy yaw for the whole one-shot.
  let chaseHeldYaw = 0;
  // Heading hysteresis (Schmitt trigger): the walking gait wiggles the
  // trunk yaw ~±14 deg per step; two-layer EMA + engage/release thresholds
  // keep the camera steady while walking straight but responsive on turns.
  let chaseYawSmooth = 0;
  let chaseYawFollow = 0;
  let chaseYawTracking = false;
  const CHASE_YAW_SMOOTH_EASE = 0.04;
  const CHASE_YAW_ENGAGE = 0.17;
  const CHASE_YAW_RELEASE = 0.03;
  const CHASE_YAW_EASE = 0.10;
  const CHASE_YAW_EASE_TURN = 0.5;
  const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  function updateChaseCam() {
    if (EMBEDDED && cameraMode === "eyes") { updateEyesCamera(); return; }
    // Reset glide: one clean tween from wherever the camera is back to the
    // home framing. Runs instead of the chase logic and hands control back
    // to it on landing.
    if (camResetT0 !== null) {
      if (!chaseCam) { camResetT0 = null; return; }
      const t = (performance.now() - camResetT0) / 1000 / CAM_RESET_S;
      const e = t >= 1 ? 1 : t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      camera.position.lerpVectors(_camFrom, _camTo, e);
      controls.target.lerpVectors(_tgtFrom, _tgtTo, e);
      camera.lookAt(controls.target);
      if (t >= 1) camResetT0 = null;
      return;
    }
    // Head mode: the camera freezes where it is. It keeps looking at the
    // duck for free - syncRig translates camera and target by the same
    // delta, and the duck isn't walking anyway (twist zeroed).
    if (headMode) return;
    if (!chaseCam) return;
    const qpos = selectedDuckId !== "duck1" && companion ? companion.qpos() : data.qpos;
    let rawYaw;
    if (mode === "roll" || isKick()) {
      rawYaw = chaseHeldYaw;
    } else {
      rawYaw = duckYaw(qpos);
      chaseHeldYaw = rawYaw;
    }
    // "turning" reads the raw per-source wz commands (not the locked/merged
    // view) so an intentional turn engages on the first frame.
    const turning = controller.sources.some((s) => Math.abs(s.command[2]) > 0.05);
    chaseYawSmooth = wrapPi(
      chaseYawSmooth +
        wrapPi(rawYaw - chaseYawSmooth) * (turning ? CHASE_YAW_EASE_TURN : CHASE_YAW_SMOOTH_EASE),
    );
    const yawErr = wrapPi(chaseYawSmooth - chaseYawFollow);
    if (turning || Math.abs(yawErr) > CHASE_YAW_ENGAGE) chaseYawTracking = true;
    if (chaseYawTracking) {
      chaseYawFollow = wrapPi(
        chaseYawFollow + yawErr * (turning ? CHASE_YAW_EASE_TURN : CHASE_YAW_EASE),
      );
      if (!turning && Math.abs(yawErr) < CHASE_YAW_RELEASE) chaseYawTracking = false;
    }
    const yaw = chaseYawFollow;
    const dist = camera.position.distanceTo(controls.target);
    const horiz = dist * Math.cos(CHASE_PITCH);
    const vert = dist * Math.sin(CHASE_PITCH);
    // Duck forward in MJCF is (cos yaw, sin yaw, 0); Z-up -> Y-up maps it
    // to three-space (cos yaw, 0, -sin yaw). Behind = minus that.
    _chasePos.set(
      controls.target.x - Math.cos(yaw) * horiz,
      controls.target.y + vert,
      controls.target.z + Math.sin(yaw) * horiz,
    );
    camera.position.lerp(_chasePos, CHASE_EASE);
    // Re-project onto the orbit sphere: lerping between two points at the
    // same radius cuts the chord, which would slowly zoom the camera in
    // during large swings.
    _chaseDir.copy(camera.position).sub(controls.target);
    const len = _chaseDir.length();
    if (len > 1e-6) camera.position.copy(controls.target).addScaledVector(_chaseDir, dist / len);
    camera.lookAt(controls.target);
  }
  renderer.domElement.addEventListener("pointerdown", () => {
    chaseCam = false;
    if (EMBEDDED && cameraMode !== "eyes") cameraMode = "orbit";
  });

  // Camera reset glide (owned by the respawn ceremony): back to the
  // page-load framing - the chase cam's ideal point behind the duck's
  // spawn heading, at the boot orbit distance.
  const CAM_HOME_DIST = camera.position.distanceTo(controls.target);
  let camResetT0 = null;
  const _camFrom = new THREE.Vector3(), _camTo = new THREE.Vector3();
  const _tgtFrom = new THREE.Vector3(), _tgtTo = new THREE.Vector3();
  function startCameraReset() {
    // Embedded mode starts at the existing front three-quarter angle.
    // Keep that viewing direction through resets unless C enabled chase.
    if (EMBEDDED && !chaseCam) { camResetT0 = null; return; }
    const qpos = data.qpos;
    const yaw = duckYaw(qpos);
    chaseHeldYaw = yaw;
    chaseYawSmooth = yaw;
    chaseYawFollow = yaw;
    chaseYawTracking = false;
    _tgtTo.set(qpos[0], qpos[2], -qpos[1]); // trunk at spawn, MJCF -> three
    const horiz = CAM_HOME_DIST * Math.cos(CHASE_PITCH);
    const vert = CAM_HOME_DIST * Math.sin(CHASE_PITCH);
    _camTo.set(
      _tgtTo.x - Math.cos(yaw) * horiz,
      _tgtTo.y + vert,
      _tgtTo.z + Math.sin(yaw) * horiz,
    );
    _camFrom.copy(camera.position);
    _tgtFrom.copy(controls.target);
    camResetT0 = performance.now();
    chaseCam = true; // reset always re-attaches the chase cam
  }

  // ── Mouse grab, pointer side (pick + drag target + cursor) ────────────
  // Pointer-down on the duck (or the live ball) grabs it; anywhere else
  // falls through to OrbitControls untouched. The pick is a three.js
  // raycast against the render rig (the WASM bindings expose no
  // mjv_select, and the rig IS the duck's collision-accurate silhouette
  // for mouse purposes). While dragging, the cursor is projected on a
  // camera-facing plane through the grab point - horizontal AND vertical
  // drags both work, so the duck can be lifted - and the target is
  // clamped inside the arena walls and to a sane height band. Desktop
  // mouse only: the touch overlay keeps its own controls.
  const GRAB_TARGET_ZMIN = 0.02, GRAB_TARGET_ZMAX = 0.45;
  const _grabRaycaster = new THREE.Raycaster();
  const _grabNdc = new THREE.Vector2();
  const _grabPlane = new THREE.Plane();
  const _grabHit = new THREE.Vector3();
  const _grabCamDir = new THREE.Vector3();
  function grabRayFrom(e) {
    const r = renderer.domElement.getBoundingClientRect();
    _grabNdc.set(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1,
    );
    _grabRaycaster.setFromCamera(_grabNdc, camera);
  }
  function grabPick() {
    let duckHit = _grabRaycaster.intersectObject(rig.placer, true)[0], duckId = "duck1";
    for (const [id, peer] of physicalPeers) {
      const hit = _grabRaycaster.intersectObject(peer.rig.placer, true)[0];
      if (hit && (!duckHit || hit.distance < duckHit.distance)) { duckHit = hit; duckId = id; }
    }
    const ballHit = ballActive ? _grabRaycaster.intersectObject(ballMesh, true)[0] : undefined;
    if (duckHit && (!ballHit || duckHit.distance <= ballHit.distance)) {
      return { kind: "duck", duckId, point: duckHit.point };
    }
    return ballHit ? { kind: "ball", point: ballHit.point } : null;
  }
  function updateGrabTarget() {
    if (!_grabRaycaster.ray.intersectPlane(_grabPlane, _grabHit)) return;
    const lim = ARENA_HALF - 0.05;
    // three (x, y, z) -> MJCF (x, -z, y), Z-up.
    grab.target[0] = Math.min(lim, Math.max(-lim, _grabHit.x));
    grab.target[1] = Math.min(lim, Math.max(-lim, -_grabHit.z));
    grab.target[2] = Math.min(GRAB_TARGET_ZMAX, Math.max(GRAB_TARGET_ZMIN, _grabHit.y));
  }
  function endGrab() {
    if (!grab) return;
    releaseGrabForce();
    controls.enabled = true;
    renderer.domElement.style.cursor = "";
  }
  endGrabHook = endGrab;
  // Capture phase on window: runs before OrbitControls' pointerdown on the
  // canvas, so the orbit can be disabled for the whole drag. The canvas's
  // own chase-detach listener still fires afterward (grabbing detaches the
  // chase cam exactly like an orbit drag does).
  window.addEventListener("pointerdown", (e) => {
    if (e.target !== renderer.domElement || e.pointerType !== "mouse" || e.button !== 0) return;
    if (grab || inputLocked) return;
    grabRayFrom(e);
    const pick = grabPick();
    if (!pick) return;
    // Any duck mesh grabs the trunk: the freejoint root carries the whole
    // body, and pulling the CoM is what the viewer perturbation feels like.
    const actor = pick.kind === "duck" ? peerController(pick.duckId) : null;
    const pickedBodyId = actor?.trunkId ?? trunkId;
    const g = pick.kind === "duck"
      ? { bodyId: pickedBodyId, qAdr: actor?.qAdr ?? 0, dofAdr: actor?.vAdr ?? 0, mass: model.body_subtreemass[pickedBodyId] }
      : {
          bodyId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "ball"),
          qAdr: ballQposAdr, dofAdr: ballDofAdr,
          mass: model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "ball")],
        };
    camera.getWorldDirection(_grabCamDir);
    _grabPlane.setFromNormalAndCoplanarPoint(_grabCamDir.negate(), pick.point);
    grab = { ...g, target: [0, 0, 0] };
    updateGrabTarget();
    controls.enabled = false;
    renderer.domElement.style.cursor = "grabbing";
    try { renderer.domElement.setPointerCapture(e.pointerId); } catch { /* capture unsupported */ }
  }, true);
  let grabHoverAt = 0;
  window.addEventListener("pointermove", (e) => {
    if (grab) {
      grabRayFrom(e);
      updateGrabTarget();
      return;
    }
    // Hover affordance: grab cursor over anything grabbable. Throttled -
    // a full-rig raycast per mousemove event would be wasteful - and
    // skipped mid-orbit (buttons held) so the cursor doesn't flicker.
    if (e.target !== renderer.domElement || e.pointerType !== "mouse" || e.buttons || inputLocked) return;
    const now = performance.now();
    if (now - grabHoverAt < 80) return;
    grabHoverAt = now;
    grabRayFrom(e);
    const clickable = mode === "walk" && !headMode;
    renderer.domElement.style.cursor = grabPick() ? "grab" : (clickable ? "crosshair" : "");
  });
  window.addEventListener("pointerup", endGrab);
  window.addEventListener("pointercancel", endGrab);

  // Waypoint marker: hairline tron reticle at the clicked floor point
  // (fx/waypoint-marker.js), with acquire/arrive animations. Driven each
  // frame in frame() from waypointSource.target.
  const waypointMarker = createWaypointMarker();
  scene.add(waypointMarker.group);

  // Pause: while the menu is up over a live game, keys belong to the menu.
  const setInputLock = (v) => { inputLocked = v; controller.setLocked(v); };
  useGame.subscribe(
    (s) => s.menuOpen,
    (open) => {
      if (!ceremony.entranceDone) return;
      if (open) setInputLock(true);
      else if (!ceremony.respawnActive) setInputLock(false);
    },
  );

  // The rig's root already applies the MJCF Z-up -> three Y-up fix, so the
  // trunk group can take the freejoint pose in raw MJCF coordinates.
  const _target = new THREE.Vector3();
  const _follow = new THREE.Vector3();
  function syncRig() {
    const qpos = data.qpos;
    trunkGroup.position.set(qpos[0], qpos[1], qpos[2]);
    trunkGroup.quaternion.set(qpos[4], qpos[5], qpos[6], qpos[3]);
    for (let j = 0; j < NUM_JOINTS; j++) setJoint(rig, JOINT_NAMES[j], qpos[qposAdr[j]]);
    // Passive hinges (roller wheels): purely visual, driven straight from qpos.
    for (const ej of extraJoints) setJoint(rig, ej.name, qpos[ej.adr]);
    // Ball: live follows qpos; ghost freeze is owned by the ball actor.
    if (ball) ball.sync(qpos, ballQposAdr, ballActive);
    // Follow cam: ease the orbit target toward the trunk and translate the
    // camera by the same delta, so the camera-to-duck distance and viewing
    // angle stay constant while the duck walks. Paused while the reset
    // glide owns the camera.
    if (camResetT0 === null) {
      const targetQpos = selectedDuckId !== "duck1" && companion ? companion.qpos() : qpos;
      _target.set(targetQpos[0], targetQpos[2], -targetQpos[1]);
      if (EMBEDDED && cameraMode === "orbit" && companion) {
        _target.set(qpos[0], qpos[2], -qpos[1]);
        for (const peer of physicalPeers.values()) { const q = peer.controller.qpos(); _target.add(new THREE.Vector3(q[0], q[2], -q[1])); }
        _target.multiplyScalar(1 / (physicalPeers.size + 1));
      }
      _follow.copy(_target).sub(controls.target);
      // Horizontal follow at the usual rate; vertical much slower so the
      // per-step gait bob doesn't nod the frame.
      _follow.x *= 0.06;
      _follow.z *= 0.06;
      _follow.y *= 0.015;
      controls.target.add(_follow);
      camera.position.add(_follow);
    }
    // Keep the grid plane (and its fade center) under the action; the wall
    // grids share the same radial fade focus.
    grid.position.set(controls.target.x, 0, controls.target.z);
    grid.material.uniforms.uFocus.value.copy(controls.target);
    for (const m of wallMats) m.uniforms.uFocus.value.copy(controls.target);
  }
  function syncCompanionRig() {
    for (const peer of physicalPeers.values()) {
      const actor = peer.controller, peerRig = peer.rig;
      const q = actor.qpos(), trunk = peerRig.bodies.get("trunk_base");
      trunk.position.set(q[0], q[1], q[2]);
      trunk.quaternion.set(q[4], q[5], q[6], q[3]);
      for (let joint = 0; joint < NUM_JOINTS; joint++) setJoint(peerRig, JOINT_NAMES[joint], data.qpos[actor.qposAdr[joint]]);
      for (const joint of actor.extraJoints) setJoint(peerRig, joint.name, data.qpos[joint.adr]);
    }
  }

  // ── Quack: jaw + chirp ────────────────────────────────────────────────
  // The jaw isn't a MuJoCo joint (duck.js re-creates the hinge in JS), so
  // this is purely cosmetic and can't upset the policy. Voice banks from
  // the robot runtime: each colourway gets its own bank and every quack
  // draws a random chirp take from it.
  const QUACK_MS = 480;
  let quackAt = -Infinity;
  let padJaw = 0;
  const CHIRP_TAKES = "abcdefghijkl";
  const VOICE_BANK = { classic: "duck1", charcoal: "duck2", purple: "duck3", blue: "duck4" };
  function playChirpFor(variant = currentVariant) {
    const bank = VOICE_BANK[variant] ?? "duck1";
    const take = CHIRP_TAKES[(Math.random() * CHIRP_TAKES.length) | 0];
    // Decoded through the shared context on the voice bus (used to be a
    // bare HTMLAudio element outside the master gain).
    playUrl(signed(`./assets/voices/${bank}/chirp_${take}.wav`), { gain: 0.7 });
  }
  const quackLoud = () => {
    quackAt = performance.now();
    playChirpFor();
    stickers?.pop("quack");
  };
  function playBoundedWheee() {
    clearTimeout(voiceTimer);
    startWheee();
    voiceTimer = setTimeout(() => stopWheee(), 1000);
  }
  // Ground-pick jaw: on the robot the pick policy drives the mouth itself
  // (mouth is part of its action space); the sim's ONNX exports have no
  // mouth channel (all heads are 14 actions), so the peck is re-created
  // here on the same phase clock. Keyed to the measured cycle: the beak
  // reaches the ground ~phase 0.16-0.42 and the head scoops back up
  // 0.40-0.50 - open on approach, snap shut on the scoop (the grab).
  const PICK_JAW_KEYS = [[0.10, 0], [0.20, 1], [0.40, 1], [0.50, 0]];
  function pickJawNow() {
    const phase = mode === "groundpick" ? pickRun?.phase : null;
    if (phase == null) return 0;
    const K = PICK_JAW_KEYS;
    if (phase <= K[0][0] || phase >= K[K.length - 1][0]) return 0;
    for (let i = 1; i < K.length; i++) {
      if (phase > K[i][0]) continue;
      const [p0, v0] = K[i - 1];
      const [p1, v1] = K[i];
      const t = (phase - p0) / (p1 - p0);
      return v0 + (v1 - v0) * (1 - Math.cos(Math.PI * t)) / 2; // eased
    }
    return 0;
  }
  function jawOpenNow() {
    const t = (performance.now() - quackAt) / QUACK_MS;
    const flap = t >= 0 && t < 1 ? Math.sin(Math.PI * t) : 0;
    // Runtime mouth-mode rule (main.rs: motor_targets[MOUTH] += offset):
    // the policy's jaw is the BASE and the trigger/quack opening is an
    // additive offset on top, clamped - it never fights the pick motion.
    return Math.min(1, pickJawNow() + Math.max(flap, padJaw, primaryExternalMouth));
  }
  function syncJaw() {
    setJawOpen(rig, jawOpenNow());
    for (const peer of physicalPeers.values()) setJawOpen(peer.rig, peer.controller.jaw());
  }

  // ── Wheee: LT-held playable note (sim behavior) ───────────────────────
  // The ride plays the voice bank's LOOP segment only (crossfade-authored
  // to wrap sample-exactly), faded in over ~20 ms, and the LT analog
  // pressure PICKS ITS NOTE: major-pentatonic steps over one octave via
  // playbackRate, glided with setTargetAtTime so per-frame updates and
  // step changes never zipper or click. The runtime has no pitch feature
  // (raw PCM through aplay) - this is the sim's own instrument.
  //
  // The authored start segment is deliberately NOT played: it is 0.8-0.9 s
  // long and cannot be pitch-modulated without breaking the sample-accurate
  // start→loop handoff, so with it the first second of every squeeze was
  // stuck at base pitch - pressure read as a volume change (the attack's
  // own crescendo), not as notes.
  //
  // Release CUTS the ride and plays nothing else - the runtime kills the
  // streaming aplay on the LT falling edge (its end segment never plays on
  // the gamepad path), and the sim's old end-segment playback re-attacked
  // a note on release, which read as a retriggered sound. A short gain
  // ramp stands in for the process kill so Web Audio doesn't click. The
  // gain is otherwise CONSTANT - pressure must never track loudness.
  const WHEEE_TAKES = "ab";
  // Major pentatonic anchored one octave BELOW the sample's natural pitch:
  // full squeeze reaches the natural note, casual play sits clearly lower
  // (the natural pitch alone read as too shrill). -12 st = playbackRate 0.5.
  const WHEEE_SCALE = [-12, -10, -8, -5, -3, 0]; // semitones vs natural pitch
  const WHEEE_DEADZONE = 0.05; // squeeze below this is stick noise, maps to the root
  const WHEEE_GAIN = 0.7;
  let wheeeCtx = null;
  const wheeeBufCache = new Map();
  let wheeeRide = null; // current ride, null while the trigger is up
  function wheeeBuffer(url) {
    let p = wheeeBufCache.get(url);
    if (!p) {
      p = fetch(url)
        .then((r) => r.arrayBuffer())
        .then((ab) => wheeeCtx.decodeAudioData(ab));
      wheeeBufCache.set(url, p);
    }
    return p;
  }
  async function startWheee() {
    stopWheee({ silent: true }); // a re-press replaces the current ride
    wheeeCtx ??= audioCtx(); // shared game context, ride lands on the voice bus
    if (wheeeCtx.state === "suspended") wheeeCtx.resume().catch(() => {});
    const bank = VOICE_BANK[currentVariant] ?? "duck1";
    const take = WHEEE_TAKES[(Math.random() * WHEEE_TAKES.length) | 0];
    const ride = { loopSrc: null, gain: null };
    wheeeRide = ride;
    let loopBuf;
    try {
      loopBuf = await wheeeBuffer(signed(`./assets/voices/${bank}/wheee_loop_${take}.wav`));
    } catch {
      return; // asset missing / fetch failed: ride silently never starts
    }
    if (wheeeRide !== ride) return; // released (or replaced) during decode
    const gain = wheeeCtx.createGain();
    gain.connect(busNode("voice"));
    const t0 = wheeeCtx.currentTime + 0.02;
    // The loop is steady-state audio (no authored attack): a ~20 ms fade-in
    // makes a clean note onset instead of a click. Constant gain after that.
    gain.gain.setValueAtTime(0, t0);
    gain.gain.linearRampToValueAtTime(WHEEE_GAIN, t0 + 0.02);
    const loopSrc = wheeeCtx.createBufferSource();
    loopSrc.buffer = loopBuf;
    loopSrc.loop = true;
    loopSrc.connect(gain);
    loopSrc.start(t0);
    Object.assign(ride, { loopSrc, gain });
  }
  function stopWheee({ silent = false } = {}) {
    const ride = wheeeRide;
    if (!ride) return;
    wheeeRide = null;
    // Nothing audible yet (released mid-decode) or replaced by a re-press:
    // hard stop is inaudible and frees the nodes immediately.
    if (silent || !ride.gain) {
      try { ride.loopSrc?.stop(); } catch { /* already ended */ }
      ride.gain?.disconnect();
      return;
    }
    // Release: cut the ride, retrigger nothing (runtime kills its player
    // here). ~50 ms fade instead of a hard stop so the cut doesn't click.
    const t = wheeeCtx.currentTime;
    ride.gain.gain.setTargetAtTime(0, t, 0.05);
    const stopAt = t + 0.3; // > 5 time constants: fully silent by then
    try { ride.loopSrc?.stop(stopAt); } catch { /* already ended */ }
    const gain = ride.gain;
    setTimeout(() => gain.disconnect(), 400);
  }

  // Per-frame note picking: the full LT travel (above a small deadzone)
  // spans the pentatonic scale, one octave below natural pitch at rest up
  // to the natural pitch at full squeeze. Quantized to scale steps so
  // squeezing plays NOTES, not a siren; the ~40 ms setTargetAtTime glide
  // smooths both the per-frame updates and the step jumps (portamento
  // instead of clicks). Digital 0/1 triggers simply play the top note
  // (the natural pitch). Gain never tracks pressure.
  function driveWheeePitch(pressure) {
    const ride = wheeeRide;
    if (!ride?.loopSrc) return;
    const u = Math.min(1, Math.max(0, (pressure - WHEEE_DEADZONE) / (1 - WHEEE_DEADZONE)));
    const semis = WHEEE_SCALE[Math.round(u * (WHEEE_SCALE.length - 1))];
    ride.loopSrc.playbackRate.setTargetAtTime(2 ** (semis / 12), wheeeCtx.currentTime, 0.04);
  }

  // ── Telemetry (throttled into the store) ─────────────────────────────
  // FPS EMA is per-frame; the store write is 4 Hz so React re-renders
  // stay far away from frame rate. Odometer integrates horizontal trunk
  // travel; teleport-sized jumps (resets, loco swaps) don't count.
  let fpsEma = 60;
  let fpsLastT = performance.now();
  let odoM = 0;
  let odoX = null, odoY = null;
  let telemetryLastPush = 0;
  function renderTelemetry() {
    const now = performance.now();
    const dtF = (now - fpsLastT) / 1000;
    fpsLastT = now;
    if (dtF > 0 && dtF < 0.5) fpsEma += (1 / dtF - fpsEma) * 0.05;
    const stepD = (odoX === null) ? 0 : Math.hypot(data.qpos[0] - odoX, data.qpos[1] - odoY);
    if (stepD < 0.05) odoM += stepD; // plausible per-frame travel only
    odoX = data.qpos[0];
    odoY = data.qpos[1];
    if (now - telemetryLastPush < 250) return;
    telemetryLastPush = now;
    setStore({
      telemetry: {
        fps: Math.round(fpsEma),
        ctrlHz: Math.round(ctrlHz),
        speed: Math.hypot(data.qvel[0], data.qvel[1]),
        odo: odoM,
        peers: ghosts?.peerCount() ?? 0,
      },
    });
  }

  // ── Right-stick camera orbit (inertia downstream of the controller) ──
  // The stick steers an angular VELOCITY that eases toward the stick's
  // target rate, so pushing ramps up gently and releasing coasts to a stop
  // over ~0.3 s. Vertical is flight-style inverted.
  const PAD_ORBIT_SPEED = 2.4; // rad/s at full deflection
  const PAD_ORBIT_SMOOTH = 8; // 1/s response rate (~95% in 0.37 s)
  const padOrbitVel = { az: 0, el: 0 };
  const _padSph = new THREE.Spherical();
  const _padOff = new THREE.Vector3();
  function padOrbitStep(rx, ry, dt) {
    if (EMBEDDED && cameraMode === "eyes") {
      padOrbitLive = false;
      padOrbitVel.az = 0;
      padOrbitVel.el = 0;
      return;
    }
    padOrbitLive = rx !== 0 || ry !== 0;
    if (padOrbitLive) {
      chaseCam = false; // detach, same as a mouse grab
      if (EMBEDDED) cameraMode = "orbit";
    }
    const k = 1 - Math.exp(-PAD_ORBIT_SMOOTH * dt);
    padOrbitVel.az += (rx * PAD_ORBIT_SPEED - padOrbitVel.az) * k;
    padOrbitVel.el += (-ry * PAD_ORBIT_SPEED * 0.75 - padOrbitVel.el) * k;
    if (chaseCam) { padOrbitVel.az = 0; padOrbitVel.el = 0; return; }
    if (Math.abs(padOrbitVel.az) < 1e-3 && Math.abs(padOrbitVel.el) < 1e-3) return;
    _padOff.copy(camera.position).sub(controls.target);
    _padSph.setFromVector3(_padOff);
    _padSph.theta -= padOrbitVel.az * dt;
    _padSph.phi += padOrbitVel.el * dt;
    _padSph.phi = Math.min(controls.maxPolarAngle, Math.max(0.08, _padSph.phi));
    _padSph.makeSafe();
    camera.position.setFromSpherical(_padSph).add(controls.target);
    camera.lookAt(controls.target);
  }

  // Multiplayer ghosts, initialised asynchronously at the end of the boot.
  let ghosts = null;

  // ── Per-frame drive, called by R3F's useFrame ────────────────────────
  let padWasConnected = null;
  let touchWasConnected = null;
  let manualSignature = "";
  function frame(dt) {
    if (EMBEDDED && embeddedPaused) return;
    refitOrbitAfterResize();
    controller.update(dt);
    if (EMBEDDED) {
      const signature = JSON.stringify(manualSources.map((source) => [source.isActive(), source.pressed]));
      if (signature !== manualSignature) {
        if (manualInputActive()) { ballTask?.cancel(); if (swarm?.active || swarmPreparing) stopSwarm("Manual input interrupted group control."); stopExternalCommand(); notifyEmbeddedManualInput(); }
        manualSignature = signature;
      }
    }
    padJaw = selectedDuckId === "duck1" ? controller.getAxes().jaw : 0;
    if (selectedDuckId !== "duck1" && companion && controller.getAxes().jaw > 0) companion.mouth = controller.getAxes().jaw;
    driveWheeePitch(controller.getAxes().ride); // no-op while no ride is open
    if (padSource.connected !== padWasConnected) {
      padWasConnected = padSource.connected;
      setStore({ padConnected: padSource.connected });
    }
    if (touchSource.connected !== touchWasConnected) {
      touchWasConnected = touchSource.connected;
      setStore({ touchMode: touchSource.connected });
    }
    // Head mode: sticks steer the head targets (stick * HEAD_MAX, signed
    // per joint); the EMA toward them runs in buildObs at 50 Hz. Without
    // a pad the targets stay put (and are debug-writable via window.rl).
    if (headMode && padSource.connected) {
      const h = padSource.head;
      const target = selectedDuckId !== "duck1" && companion ? companion.headTarget : headTarget;
      target[0] = HEAD_SIGNS[0] * h.neckPitch * HEAD_MAX;
      target[1] = HEAD_SIGNS[1] * h.pitch * HEAD_MAX;
      target[2] = HEAD_SIGNS[2] * h.yaw * HEAD_MAX;
      target[3] = HEAD_SIGNS[3] * h.roll * HEAD_MAX;
    }
    // Camera orbit runs every frame while a pad is present (the coasting
    // needs the zero-deflection frames too); without a pad, park the
    // state. Head mode parks it too: the right stick belongs to the head
    // and the camera must freeze in place (no leftover coasting).
    if (padSource.connected && !headMode) {
      padOrbitStep(controller.getAxes().orbitX, controller.getAxes().orbitY, dt);
    } else {
      padOrbitLive = false;
      padOrbitVel.az = 0;
      padOrbitVel.el = 0;
    }
    syncRig();
    syncCompanionRig();
    if (physicalPeers.size > 1) swarmVisual?.update(swarmPoses(), swarm, Number(data.time));
    syncJaw();
    ghosts?.update();
    // Spatial audio follows the movers: listener on the camera, emitters
    // on the duck trunk and the ball (MJCF Z-up -> three Y-up).
    updateListener(camera);
    duckEmitter.setPosition(data.qpos[0], data.qpos[2], -data.qpos[1]);
    if (ballActive) {
      const q = data.qpos;
      ballEmitter.setPosition(q[ballQposAdr], q[ballQposAdr + 2], -q[ballQposAdr + 1]);
    }
    controls.update();
    updateChaseCam();
    ceremony.drive();
    ball.drive(() => spawnBall({ fromQueue: true }));
    waypointMarker.update(dt, waypointSource.target);
    renderTelemetry();
  }

  // ── Input wiring: arm the controller sources, bind actions ───────────
  controller.init();
  if (externalSource) {
    const directPointerInput = () => {
      // The capture-phase picker has already identified a physical grab.
      // Empty-space OrbitControls input changes only the camera.
      if (!grab) return;
      ballTask?.cancel(); if (swarm?.active || swarmPreparing) stopSwarm("A physical grab interrupted group control.");
      stopExternalCommand(); notifyEmbeddedManualInput();
    };
    const stopOnBlur = () => setAutonomy(false);
    const stopOnEscape = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      if (event.repeat) return;
      embeddedCommand("stop", { all: true });
      notifyEmbeddedStop();
    };
    renderer.domElement.addEventListener("pointerdown", directPointerInput, { passive: true });
    window.addEventListener("blur", stopOnBlur);
    window.addEventListener("keydown", stopOnEscape, true);
    if (import.meta.hot) import.meta.hot.dispose(() => {
      running = false;
      stopExternalCommand();
      controller.dispose();
      renderer.domElement.removeEventListener("pointerdown", directPointerInput);
      window.removeEventListener("blur", stopOnBlur);
      window.removeEventListener("keydown", stopOnEscape, true);
    });
  }

  // Keyboard F alternates kicking feet; only advance the alternation on
  // kicks that actually launched (triggerKick reports that).
  let kbKickFoot = "left";
  const srcTag = (source) => (source === "gamepad" ? "pad" : "kb");

  controller.on("reset", () => resetSim({ all: EMBEDDED }));
  controller.on("reset", () => waypointSource.cancel());
  controller.on("spawnBall", () => spawnBall({ duckId: selectedDuckId }));
  controller.on("headToggle", () => toggleHeadMode());
  controller.on("chaseToggle", () => {
    if (EMBEDDED) setCameraMode(chaseCam ? "orbit" : "follow");
    else chaseCam = !chaseCam;
  });
  controller.on("locoToggle", () => toggleLoco());
  controller.on("roll", ({ source }) => selectedDuckId !== "duck1" ? companion.command(companion.loco === "rollers" ? "crouch" : "roll", { manual: true }) : triggerRoll(srcTag(source)));
  controller.on("groundPick", ({ source }) => selectedDuckId !== "duck1" ? companion.command("ground_pick", { manual: true }) : triggerGroundPick(srcTag(source)));
  controller.on("kickL", ({ source }) => selectedDuckId !== "duck1" ? companion.command("kick_left", { manual: true }) : triggerKick("left", srcTag(source)));
  controller.on("kickR", ({ source }) => selectedDuckId !== "duck1" ? companion.command("kick_right", { manual: true }) : triggerKick("right", srcTag(source)));
  controller.on("alternateKick", ({ source }) => {
    if (selectedDuckId !== "duck1") {
      if (companion.command(`kick_${kbKickFoot}`, { manual: true }).accepted) kbKickFoot = kbKickFoot === "left" ? "right" : "left";
      return;
    }
    if (triggerKick(kbKickFoot, srcTag(source))) {
      kbKickFoot = kbKickFoot === "left" ? "right" : "left";
    }
  });
  // Sit is the legs-only skill; on rollers the same button hands over to
  // the crouch-glide, exactly as the (now unbound) roll action did.
  controller.on("sitToggle", ({ source } = {}) => {
    if (selectedDuckId !== "duck1") return companion.command(companion.loco === "rollers" ? "crouch" : companion.sitFlag ? "stand" : "sit", { manual: true });
    if (loco !== "legs") return triggerCrouch(srcTag(source));
    // A community trick / script takes over the R key while it is mounted
    // (a custom sitstand keeps the sit toggle: it IS the sit policy).
    if (customPolicy?.slot === "trick") return triggerRoll(srcTag(source));
    if (customPolicy?.slot === "script") return toggleScript();
    const sitting = mode === "sitstand" && sitFlag === 1;
    setMode(sitting ? "walk" : "sit");
  });
  // Pad DpadUp short press: straight back to running (ignored mid-roll /
  // mid-crouch: those hand back to walk on their own).
  controller.on("walk", () => {
    if (selectedDuckId !== "duck1") { companion.command("stand", { manual: true }); return; }
    if (mode !== "walk" && mode !== "roll" && mode !== "crouch") setMode("walk");
  });
  controller.on("quack", () => selectedDuckId !== "duck1" ? companion.command("quack", { manual: true }) : quackLoud());
  controller.on("wheeeStart", () => startWheee());
  controller.on("wheeeStop", () => stopWheee());

  // Leaving head mode keeps the head offsets (runtime behavior): only
  // resetSim zeroes headTarget/headSmooth.
  function exitHeadMode() {
    if (!headMode) return;
    headMode = false;
    padSource.headMode = false;
    syncButtons();
  }

  function toggleHeadMode() {
    if (headMode) return exitHeadMode();
    if (selectedDuckId !== "duck1" && companion) {
      if (companion.busy() || companion.fallen()) return;
      headMode = true; padSource.headMode = true; syncButtons(); return;
    }
    // Enterable from walk or sit only - never during one-shots (roll /
    // kick / crouch), the post-kick grace, a stand-up hand-back, a fall
    // recovery, or while the entrance/respawn lock holds the inputs.
    if (inputLocked || (mode !== "walk" && mode !== "sitstand") || postKickLock > 0 ||
        standTimer || recovery)
      return;
    headMode = true;
    padSource.headMode = true;
    syncButtons();
  }

  function setMode(next, { force = false } = {}) {
    if (!force && inputLocked) return;
    // No policy switching mid-roll or mid-kick: both end on their own and
    // return to walk - switching now would floor the duck. Same while the
    // fall-recovery state machine owns the duck.
    if (recovery) return;
    if ((mode === "roll" && rollRun) || (isKick() && kickRun) ||
        (mode === "crouch" && crouchRun) || (mode === "groundpick" && pickRun)) return;
    // Direct keyboard/gamepad requests also respect the active trained
    // handover. A repeated stand request must not clear its own timer.
    if (EMBEDDED && (sitTimer || standTimer || sitSettle.pending)) return;
    if (next === "sit" && loco === "rollers") return;
    exitHeadMode(); // posture changes exit head mode (offsets kept)
    clearModeTimers();
    rollRun = null;
    crouchRun = null;
    pickRun = null;
    if (next !== "sit") {
      // Leaving a sit: let the sitstand policy stand the duck back up first.
      if (mode === "sitstand" && sitFlag === 1) {
        sitFlag = 0;
        standTimer = modeTimer(() => {
          standTimer = null;
          mode = next;
          lastAction.fill(0);
          syncButtons();
        }, 2000);
        syncButtons();
        return;
      }
      mode = next;
      lastAction.fill(0);
    } else {
      // Hand over gently: hold the stand under the sitstand policy for a
      // moment before commanding the sit, or the abrupt session switch
      // knocks the duck over.
      mode = "sitstand";
      sitFlag = 0;
      lastAction.fill(0);
      sitSettle?.request();
      sitTimer = modeTimer(() => {
        sitTimer = null;
        if (mode === "sitstand") { sitFlag = 1; syncButtons(); }
      }, 800);
    }
    syncButtons();
  }

  // One roll, then straight back to running. lastAction is deliberately
  // NOT zeroed: the runtime keeps one continuous action history across
  // policy switches, and the roll initiates more reliably mid-gait.
  function triggerRoll(source = "kb") {
    if (loco === "rollers") return triggerCrouch(source);
    if (inputLocked || mode !== "walk" || standTimer || recovery) return;
    exitHeadMode();
    clearModeTimers();
    mode = "roll";
    sitFlag = 0;
    // A custom episodic trick lives in the roll slot and runs for its own
    // duration (loadCustomPolicy clamps it); the official roulade keeps
    // its 3 s hard window.
    const custom = customPolicy?.slot === "trick" && sessions.roll === customPolicy.session;
    const maxSteps = custom ? Math.max(10, Math.round((customPolicy.durationS ?? 5) / CTRL_DT)) : 150;
    rollRun = { steps: 0, tipped: false, custom, maxSteps };
    syncButtons();
    stickers?.pop("roll");
  }

  // Kind "script" move: R toggles the command timeline on the walker.
  function toggleScript() {
    if (!customPolicy?.script) return;
    if (scriptRun) {
      scriptRun = null;
      syncButtons();
      return;
    }
    if (inputLocked || mode !== "walk" || standTimer || recovery || loco !== "legs") return;
    exitHeadMode();
    scriptRun = { t: 0 };
    syncButtons();
  }

  // Roller-only one-shot: crouch, glide low, stand back up (phase-driven).
  function triggerCrouch(source = "kb") {
    if (inputLocked || mode !== "walk" || locoSwitching || recovery) return;
    exitHeadMode();
    clearModeTimers();
    mode = "crouch";
    crouchRun = { phase: 0 };
    syncButtons();
    stickers?.pop("roll");
  }

  // One-shot ground pick (runtime A button): peck the ground and stand
  // back up, phase-driven like the roller crouch (same cos/sin encoding in
  // the command vel slots). Legs-only, from walk, and never during another
  // one-shot / a stand-up hand-back / the entrance lock.
  function triggerGroundPick(source = "kb") {
    if (loco !== "legs") return;
    if (inputLocked || mode !== "walk" || standTimer || recovery) return;
    exitHeadMode();
    clearModeTimers();
    mode = "groundpick";
    sitFlag = 0;
    pickRun = { phase: 0 };
    syncButtons();
  }

  // One blind kick (the duck can't see any ball - it's a scripted boot).
  // Returns whether the kick actually launched so the keyboard's foot
  // alternation only advances on real kicks.
  function triggerKick(foot, source = "kb") {
    if (loco === "rollers") return false;
    if (inputLocked || mode !== "walk" || standTimer || recovery) return false;
    exitHeadMode();
    clearModeTimers();
    mode = foot === "left" ? "kickL" : "kickR";
    sitFlag = 0;
    kickRun = { steps: 0 };
    haptics.pulse("kick"); // swing launch; ball contact adds ballHit
    syncButtons();
    stickers?.pop("kick");
    return true;
  }

  function syncButtons() {
    const sitting = mode === "sitstand" && sitFlag === 1;
    const label =
      recovery ? "Recovery"
      : mode === "roll" ? (rollRun?.custom ? "Trick" : "Roll")
      : scriptRun ? "Script"
      : mode === "crouch" ? "Crouch"
      : mode === "groundpick" ? "Pick"
      : isKick() ? "Kick"
      : headMode ? "Head"
      : sitting ? "Sit"
      : loco === "rollers" ? "Drive"
      : "Run";
    if (store().modeLabel !== label) setStore({ modeLabel: label });
    if (store().ballActive !== ballActive) setStore({ ballActive });
  }

  // Parent commands enter through the same controller/command surfaces
  // as a person using the simulator. Physics and policy tensors below
  // this boundary keep their official calculations.
  function protectedMotionBusy() {
    return inputLocked || locoSwitching || parkSwitching || !!parkSettle || !!coastSettle || !!recovery || !!sitTimer || !!standTimer
      || !!sitSettle?.pending
      || postKickLock > 0 || !!scriptRun || (mode !== "walk" && mode !== "sitstand");
  }
  function movementProfile(action) {
    // Matched two-second checks only leaned at 0.2 m/s, but walked at
    // the official keyboard's 0.25 m/s. No physics parameters change.
    if (action === "walk_backward") return reverseProfile(primaryPose(), loco, spatialWorld()) ?? [-.2, 0, loco === "rollers" ? .3 : 1];
    return {
      walk_forward: [Math.min(velLims()[0], 0.25), 0, 0],
      turn_left: [.25, 0, loco === "rollers" ? .3 : 1],
      turn_right: [.25, 0, loco === "rollers" ? -.3 : -1],
    }[action];
  }
  function startMovementPulse(action) {
    const profile = movementProfile(action);
    if (action.startsWith("turn_") || action === "walk_backward") {
      const path = sweptMotion(primaryPose(), profile, spatialWorld(), 2);
      if (!path.allowed) { autonomyGuard.guardReason = path.reason; autonomyGuard.guardSeq++; stopExternalCommand(); return false; }
    }
    if (!autonomyGuard.admit(action, profile[0])) return false;
    externalSource.start(profile);
    externalAction = action;
    return true;
  }
  function startPreparedAction(action) {
    if (movementProfile(action)) return startMovementPulse(action);
    if (!actionRoom(action, measureSpatial(data.qpos, spatialWorld()).clearance)) {
      autonomyGuard.guardReason = "Body gesture guard: space changed while this duck stood up.";
      autonomyGuard.guardSeq++;
      stopExternalCommand();
      return false;
    }
    autonomyGuard.accepted();
    if (action === "roll") triggerRoll();
    else if (action === "kick_left") triggerKick("left");
    else if (action === "kick_right") triggerKick("right");
    else if (action === "ground_pick") triggerGroundPick();
    else if (action === "crouch") triggerCrouch();
    else return false;
    return true;
  }
  if (EMBEDDED) locomotionIntent = new LocomotionIntent({
    beginStand: () => setMode("walk"),
    startPulse: startPreparedAction,
  });
  function ballTaskState(id) {
    const actor = peerController(id);
    const native = actor ? actor.status() : { ...primaryPose(), mode, busy: protectedMotionBusy() || !!locomotionIntent?.pendingAction,
      tiltRad: Math.acos(Math.max(-1, Math.min(1, -projGravZ()))), error: runtimeError || sitSettle?.error };
    const velocityAdr = actor ? actor.vAdr : 0;
    const duck = { ...native, manual: manualInputActive() || !!grab,
      speedMps: Math.hypot(...data.qvel.slice(velocityAdr, velocityAdr + 3)) };
    return { duck, ball: { present: ballActive, position: Array.from(data.qpos.slice(ballQposAdr, ballQposAdr + 3)),
      speedMps: Math.hypot(...data.qvel.slice(ballDofAdr, ballDofAdr + 3)) }, ...spatialWorld(id) };
  }
  if (EMBEDDED) ballTask = new BallTaskController({ getState: ballTaskState,
    nativeAction: (id, action) => id !== "duck1" ? peerController(id).command(action) : embeddedCommand(action, { internal: true }) });
  function embeddedCommand(action, { all = false, id, internal = false } = {}) {
    if (!internal && (swarm?.active || swarmPreparing)) {
      if (action === "stop" || action === "reset") stopSwarm("Swarm stopped by the command owner.");
      else return { accepted: false, message: "Stop the swarm simulation before commanding one duck." };
    }
    if (action === "stop") ballTask?.cancel("Ball task stopped by the user or command owner.");
    if (!internal && ballTask?.active && action !== "stop" && action !== "reset") {
      if (action === "spawn_ball") ballTask.cancel("Ball task cancelled because the ball was repositioned.");
      else return { accepted: false, message: "Let the ball task finish or stop it before another command." };
    }
    if (BALL_TASK_ACTIONS.includes(action)) {
      const state = selectedDuckId !== "duck1" && companion ? companion.status() : primaryStatus();
      if (!state.ready || state.paused || state.busy || state.fallen || parkSwitching || parkSettle || manualInputActive() || grab)
        return { accepted: false, message: "Release the controls and let the current action settle before ball interaction." };
      stopExternalCommand(); companion?.stop(); waypointSource.cancel();
      return ballTask.start(action, id, selectedDuckId);
    }
    if (action === "stop" && all) {
      followEnabled = false; followDrive = [0, 0, 0]; followState = "Stopped.";
      for (const peer of physicalPeers.values()) { peer.controller.stop(); peer.controller.setAutonomy(false); }
      stopExternalCommand(); autonomyGuard?.setActive(false);
    }
    if (selectedDuckId !== "duck1" && companion && action !== "reset") {
      if (action !== "stop" && (parkSwitching || parkSettle || locoSwitching)) return { accepted: false, message: "Wait for the locomotion change." };
      if (action === "stop") { waypointSource.cancel(); controller.holdUntilNeutral(() => !manualInputActive()); }
      return companion.command(action);
    }
    if (action === "stop") {
      setAutonomy(false);
      waypointSource.cancel();
      controller.holdUntilNeutral(() => !manualInputActive());
      return { accepted: true, completion: "immediate", message: controller.neutralHeld
        ? "Locomotion command is zero. Release the held controls before moving again."
        : "Movement inputs cleared. Balance and any required posture transition continue." };
    }
    // Reset is an explicit manual host control, equivalent to Space in
    // the original simulator. It is never a Jev-selected action.
    if (action === "reset") {
      stopExternalCommand();
      waypointSource.cancel();
      runtimeError = null;
      embeddedPaused = embeddedSuspended();
      resetSim({ all: true });
      return { accepted: true, message: "All physical ducks and the shared ball have reset." };
    }
    if (runtimeError || sitSettle.error) return { accepted: false, message: "The simulation needs a reset or reload." };
    if (embeddedPaused || embeddedSuspended()) return { accepted: false, message: "The simulator is paused while it is hidden." };
    if (protectedMotionBusy() || locomotionIntent?.pendingAction || poseIsDead()) return { accepted: false, message: "Let the current posture transition or recovery finish first." };
    if (manualInputActive() || controller.neutralHeld || headMode || grab) return { accepted: false, message: "Release the manual controls before requesting another action." };
    if (action === "quack" || action === "wheee") {
      if (action === "quack") quackLoud(); else playBoundedWheee();
      return { accepted: true, completion: "immediate", message: action === "quack" ? "Quack audio triggered." : "Bounded one-second voice playback started." };
    }
    if (action === "open_mouth" || action === "close_mouth") {
      primaryExternalMouth = action === "open_mouth" ? 1 : 0;
      return { accepted: true, completion: "immediate", message: "Visual beak position changed; this model has no grasp actuator." };
    }
    if (action === "spawn_ball") {
      spawnBall({ duckId: selectedDuckId });
      return { accepted: true, completion: "immediate", message: "Ball placement applied near the selected robot." };
    }
    if (action === "switch_to_legs" || action === "switch_to_rollers") {
      const next = action === "switch_to_legs" ? "legs" : "rollers";
      if (loco === next) return { accepted: true, completion: "immediate", message: "Already using that locomotion variant." };
      const space = measureSpatial(data.qpos, spatialWorld()).clearance;
      if (next === "rollers" && (space.front < .85 || space.left < .25 || space.right < .25)) return { accepted: false, message: "Changing to rollers needs clear space ahead for the trained model to settle." };
      if (mode !== "walk" || projGravZ() >= -.9 || Math.hypot(...data.qvel.slice(0, 3)) >= .08) return { accepted: false, message: "Stand still and balanced before changing locomotion." };
      switchParkLoco("duck1", next);
      return { accepted: true, message: "Loading the trained locomotion variant while preserving the other duck." };
    }
    if (["roll", "kick_left", "kick_right", "ground_pick", "crouch"].includes(action)) {
      if (action === "crouch" ? loco !== "rollers" : loco !== "legs") return { accepted: false, message: "That learned behavior belongs to the other locomotion variant." };
      if (!actionRoom(action, measureSpatial(data.qpos, spatialWorld()).clearance)) return { accepted: false, message: "There is too little room for that body gesture." };
      stopExternalCommand();
      if (mode === "sitstand" && sitFlag === 1) {
        locomotionIntent.request(action);
        return { accepted: true, message: "Standing first, then starting the requested trained actor after measured balance." };
      }
      return startPreparedAction(action) ? { accepted: true, message: "The trained actor is running." } : { accepted: false, message: "The trained actor could not start." };
    }
    if (movementProfile(action)) {
      if (mode === "sitstand" && sitFlag === 1 && loco === "legs") {
        if (!autonomyGuard.admit(action, movementProfile(action)[0])) {
          return { accepted: false, blockedByGuard: true, message: autonomyGuard.guardReason };
        }
        stopExternalCommand();
        locomotionIntent.request(action);
        return { accepted: true, message: "Standing up with the trained policy, then moving once balance settles." };
      }
      if (mode !== "walk") return { accepted: false, message: "The current posture must finish before moving." };
      if (!startMovementPulse(action)) return { accepted: false, blockedByGuard: true, message: autonomyGuard.guardReason };
      return { accepted: true, message: "Motion command started for two seconds. Balance control continues afterward." };
    }
    if (action === "sit" || action === "stand") {
      if (loco !== "legs") return { accepted: false, message: "Sitting and standing use the trained leg policy." };
      autonomyGuard.accepted();
      stopExternalCommand();
      const sitting = mode === "sitstand" && sitFlag === 1;
      if (action === "sit" && sitting || action === "stand" && mode === "walk") {
        return { accepted: true, completion: "immediate", message: action === "sit" ? "Already sitting." : "Already standing." };
      }
      if (action === "sit" && !sitting) setMode("sit");
      if (action === "stand" && mode === "sitstand") setMode("walk");
      return { accepted: true, message: action === "sit" ? "The trained sit/stand policy is sitting down." : "The trained sit/stand policy is standing up before returning to balance." };
    }
    const headCommands = HEAD_COMMANDS;
    if (Object.hasOwn(headCommands, action)) {
      autonomyGuard.accepted();
      externalSource.start([0, 0, 0]);
      externalHead = true;
      headTarget.set(headCommands[action]);
      return { accepted: true, message: "The policy receives a bounded head command for two seconds." };
    }
    return { accepted: false, message: "That action is not supported by the simulator." };
  }
  function primaryStatus() {
    const paused = embeddedPaused || embeddedSuspended();
    const fallen = poseIsDead() !== null;
    const pendingAction = locomotionIntent?.pendingAction ?? null;
    const moving = !!externalSource?.isActive();
    const posture = fallen || recovery ? "fallen" : sitTimer || standTimer || sitSettle.pending || pendingAction || coastSettle || parkSettle || !["walk", "sitstand"].includes(mode) || postKickLock > 0 ? "transitioning" : mode === "sitstand" && sitFlag === 1 ? "sitting" : "standing";
    const phase = paused ? "paused" : recovery ? "recovering" : pendingAction ? locomotionIntent.phase : sitSettle.pending ? sitSettle.phase : sitTimer || standTimer ? "transitioning" : coastSettle ? "settling" : moving ? "moving" : "idle";
    const policyKey = recovery?.state === "recovering" ? "stand" : loco === "rollers" && mode === "walk" ? "drive" : mode;
    return {
      ready: store().bootDone && inferenceCount > 0 && !runtimeError && !sitSettle.error,
      loco, mode, time: Number(data.time),
      position: Array.from(data.qpos.slice(0, 3)),
      ...measureSpatial(data.qpos, spatialWorld()),
      turnClear: { left: sweptMotion(primaryPose(), movementProfile("turn_left"), spatialWorld(), 2).allowed, right: sweptMotion(primaryPose(), movementProfile("turn_right"), spatialWorld(), 2).allowed },
      reverseClear: reverseProfile(primaryPose(), loco, spatialWorld()) !== null,
      autonomyActive: autonomyGuard.active,
      manual: selectedDuckId === "duck1" && manualInputActive() || !!grab,
      guardReason: autonomyGuard.guardReason, guardSeq: autonomyGuard.guardSeq,
      tiltRad: Math.acos(Math.max(-1, Math.min(1, -projGravZ()))),
      command: mode === "walk" ? Array.from(cmd.slice(0, 3)) : [0, 0, 0],
      fallen, posture, pendingAction, phase,
      busy: protectedMotionBusy() || !!pendingAction || moving || selectedDuckId === "duck1" && (manualInputActive() || headMode) || controller.neutralHeld,
      scene: presentation?.scene || "studio", camera: cameraMode, fps: Math.round(fpsEma),
      paused,
      policy: recovery?.state === "fallen" ? "Held targets during recovery settle" : POLICIES[policyKey] || policyKey,
      inferenceCount,
      ...(runtimeError || sitSettle.error ? { error: runtimeError || sitSettle.error } : {}),
    };
  }
  function relativeObject(state, position) {
    const dx = position[0] - state.position[0], dy = position[1] - state.position[1];
    return { distanceM: Math.hypot(dx, dy), bearingRad: wrapPi(Math.atan2(dy, dx) - state.headingRad) };
  }
  function embeddedStatus() {
    const first = primaryStatus();
    first.availableActions = availableDuckActions(first);
    const states = new Map([["duck1", first], ...[...physicalPeers].map(([id, peer]) => [id, peer.controller.status()])]);
    for (const [id, state] of states) {
      if (ballTask?.owns(id) || swarm?.active || swarmPreparing) { state.busy = true; state.availableActions = ["stop", "reset"]; }
      if (runtimeError) { state.ready = false; state.error = runtimeError; }
    }
    const selected = states.get(selectedDuckId) ?? first;
    const nearest = [...states].filter(([id]) => id !== selectedDuckId).sort((a, b) => relativeObject(selected, a[1].position).distanceM - relativeObject(selected, b[1].position).distanceM)[0];
    const summary = (state, id, name) => ({ id, name, loco: state.loco, mode: state.mode, position: state.position, headingRad: state.headingRad, posture: state.posture, busy: state.busy, fallen: state.fallen, command: state.command, availableActions: state.availableActions });
    return {
      ...selected, scene: presentation?.scene || "studio", camera: cameraMode, fps: Math.round(fpsEma),
      busy: selected.busy || parkSwitching || !!parkSettle || manualInputActive() || headMode, selectedDuckId,
      manual: manualInputActive() || !!grab,
      phase: parkSwitching || parkSettle ? "switching_locomotion" : selected.phase,
      availableActions: parkSwitching || parkSettle ? ["stop", "reset"] : selected.availableActions,
      ducks: [...states].map(([id, state]) => summary(state, id, duckNames[id])),
      park: { follow: followEnabled, leaderId: selectedDuckId, obstacle: obstacleSlot, followState },
      swarm: physicalPeers.size === 3 ? swarm?.status(swarmPoses()) ?? null : null,
      controlHz: Math.round(ctrlHz),
      ball: { present: ballActive, ...relativeObject(selected, data.qpos.slice(ballQposAdr, ballQposAdr + 3)) },
      task: ballTask?.snapshot(selectedDuckId) ?? null,
      companion: nearest ? { id: nearest[0], ...relativeObject(selected, nearest[1].position), posture: nearest[1].posture, moving: Math.hypot(...nearest[1].command) > .01 } : null,
    };
  }
  function selectDuck(id) {
    if (!allDuckIds().includes(id)) return { accepted: false, message: "That duck is not available in this scene." };
    if (parkSwitching || parkSettle || manualInputActive() || grab) return { accepted: false, message: "Finish the locomotion change and release manual controls before selecting a duck." };
    ballTask?.cancel("Ball task cancelled when the selected robot changed.");
    ballTask?.clear();
    stopSwarm("Group control ended when a duck was selected.");
    setAutonomy(false); stopExternalCommand(); waypointSource.cancel(); exitHeadMode();
    selectedDuckId = id;
    refreshSelectedPeer();
    followDrive = [0, 0, 0];
    return { accepted: true, message: `Selected leader: ${duckNames[id]}.` };
  }
  function requestSwarm({ active, runId, scenario }) {
    if (!active) {
      if (runId && runId !== (pendingSwarmRunId ?? swarm?.runId)) return { accepted: false, message: "That swarm run is no longer current." };
      stopSwarm("Swarm motion stopped. The physical ducks remain available for individual control.");
      return { accepted: true, message: "Group motion inputs cleared." };
    }
    if (!["flock", "gather", "convoy", "split"].includes(scenario) || typeof runId !== "string" || !runId)
      return { accepted: false, message: "A known scenario and run identifier are required." };
    if (swarmPreparing || parkSwitching || parkSettle) return { accepted: false, message: "Wait for the current scene preparation to finish." };
    if (embeddedPaused || embeddedSuspended() || manualInputActive() || grab || runtimeError)
      return { accepted: false, message: "Release manual controls and keep the simulator visible before preparing a scenario." };
    if (swarm?.active && swarm.runId === runId) return { accepted: true, message: "That scenario run is already active." };
    stopSwarm("Preparing a new four-duck simulation.");
    ballTask?.clear(); setAutonomy(false); waypointSource.cancel(); exitHeadMode();
    const token = ++swarmPrepareToken;
    pendingSwarmRunId = runId; swarmPreparing = true; parkSwitching = true;
    (async () => {
      try {
        if (controlStepPending) await controlStepPending;
        const key = "legs:legs:legs:legs";
        if (!parkModels.has(key)) {
          const prepared = await buildPhysicsXml("robot_allcollisions.xml", Array(3).fill("robot_allcollisions.xml"), { layout: "swarm" });
          await addMeshesToVfs(prepared.meshFiles);
          const nextModel = mujoco.MjModel.from_xml_string(prepared.xml, vfs);
          parkModels.set(key, { model: nextModel, data: new mujoco.MjData(nextModel) });
        }
        if (token !== swarmPrepareToken) return;
        // Finish every asynchronous resource load before changing the live
        // world. Stop during a rig load can then abandon setup without a
        // late pose reset or a half-installed four-body model.
        const preparedRigs = new Map();
        for (const id of DUCK_IDS.slice(1)) if (!physicalPeers.has(id)) {
          preparedRigs.set(id, await buildRig(k, { materialForMesh: materialHookFor(variantForPeer(id)) }));
          if (token !== swarmPrepareToken) return;
        }
        const next = parkModels.get(key);
        scene.remove(rig.placer);
        for (const peer of physicalPeers.values()) scene.remove(peer.rig.placer);
        model = next.model; data = next.data; loco = "legs";
        rig = locos.legs.rig; trunkGroup = rig.bodies.get("trunk_base");
        ({ qposAdr, dofAdr, gyroAdr, trunkId, standKeyId, ballQposAdr, ballDofAdr, extraJoints, ankleIds } = resolveAddrs(model, k));
        scene.add(rig.placer);
        for (const id of DUCK_IDS.slice(1)) {
          if (!physicalPeers.has(id)) installPhysicalPeer(id, preparedRigs.get(id));
          const peer = physicalPeers.get(id);
          peer.controller.resolve("legs"); peer.rig = peer.rigs.get("legs"); scene.add(peer.rig.placer);
        }
        selectedDuckId = "duck1"; refreshSelectedPeer();
        obstacleSlot = "off"; obstacleMesh.visible = false;
        resetSim({ all: true, preparingSwarm: true });
        for (const spawn of scenarioSpawns(scenario)) {
          const adr = spawn.id === "duck1" ? 0 : peerController(spawn.id).qAdr;
          data.qpos.set([...spawn.position, Math.cos(spawn.headingRad / 2), 0, 0, Math.sin(spawn.headingRad / 2)], adr);
        }
        mujoco.mj_forward(model, data);
        frameAllDucks(); setStore({ loco: "legs", locoWant: "legs" });
        parkSwitching = false;
        const startedAt = Number(data.time), wallStarted = performance.now();
        while (token === swarmPrepareToken && performance.now() - wallStarted < 15000) {
          const poses = swarmPoses();
          if (Number(data.time) - startedAt >= 1.5 && poses.every(pose => !pose.fallen && !pose.busy && pose.posture === "standing" && pose.speedMps < .08)) break;
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        if (token !== swarmPrepareToken) return;
        const poses = swarmPoses();
        if (Number(data.time) - startedAt < 1.5 || poses.some(pose => pose.fallen || pose.busy || pose.posture !== "standing" || pose.speedMps >= .08)) {
          swarm.stop("The four robots did not settle during scene preparation. Reset before trying again.");
          return;
        }
        swarm.startScenario(scenario, runId, poses);
      } catch (error) {
        runtimeError = error?.message || String(error);
        swarm?.stop(`Scene preparation failed: ${runtimeError}`);
      } finally {
        parkSwitching = false; swarmPreparing = false; pendingSwarmRunId = null;
      }
    })();
    return { accepted: true, message: "Preparing four physical robots and waiting for measured balance." };
  }
  function requestSwarmIntent({ runId, id, intent }) {
    if (!swarm?.active || runId !== swarm.runId || swarmPreparing || parkSwitching || embeddedPaused || embeddedSuspended() || manualInputActive() || grab)
      return { accepted: false, message: "That swarm run is no longer available for group instructions." };
    return swarm.startIntent(id, intent, swarmPoses());
  }
  function editPark({ follow, obstacle }) {
    if (follow === true && physicalPeers.size > 1) return { accepted: false, message: "Use the convoy simulation for a four-duck group." };
    if (!companion || parkSwitching || parkSettle) return { accepted: false, message: "Waiting for both robots to be ready." };
    if (obstacle !== undefined) {
      const placement = canPlaceObstacle(obstacle, swarmPoses());
      if (!placement.allowed) return { accepted: false, message: placement.reason };
      stopSwarm("Group control ended because the obstacle changed.");
      ballTask?.cancel("Ball task cancelled because the obstacle configuration changed.");
      obstacleSlot = obstacle;
      const position = placement.obstacle.position;
      data.mocap_pos.set(position, 0);
      obstacleMesh.position.set(position[0], position[2], -position[1]);
      obstacleMesh.visible = placement.obstacle.active;
      mujoco.mj_forward(model, data);
    }
    if (follow !== undefined) {
      ballTask?.cancel("Ball task cancelled because follow control changed.");
      followEnabled = follow; followDrive = [0, 0, 0];
      if (!follow) { if (selectedDuckId === "duck1") companion.stop(); else stopExternalCommand(); }
      followState = follow ? "Follow control awaiting state sample." : "Follow control disabled.";
    }
    return { accepted: true, message: "World configuration updated." };
  }
  function updateFollowDrive() {
    followDrive = [0, 0, 0];
    if (ballTask?.active) { followState = "Companion holds position while the selected duck interacts with the ball."; return; }
    if (!followEnabled || !companion || parkSwitching || parkSettle || embeddedPaused || embeddedSuspended()) return;
    const first = primaryPose(), second = companionPose();
    const follower = selectedDuckId === "duck1" ? second : first;
    const leader = selectedDuckId === "duck1" ? first : second;
    const result = followCommand({ follower, leader, obstacle: obstacleForSlot(obstacleSlot), paused: embeddedPaused || inputLocked });
    followState = result.reason;
    if (result.needsStand) {
      if (selectedDuckId === "duck1" && !companion.busy()) companion.command("stand");
      if (selectedDuckId !== "duck1" && !protectedMotionBusy()) setMode("walk");
      return;
    }
    const followerBusy = selectedDuckId === "duck1" ? companion.busy() : protectedMotionBusy();
    if (!followerBusy) {
      followDrive = result.command;
    }
  }

  // ── Public surface for the React UI ──────────────────────────────────
  Object.assign(gameApi, {
    frame,
    setVariant: (name) => {
      if (!VARIANTS[name] || name === currentVariant) return;
      currentVariant = name;
      applyVariant(rig, name);
      setStore({ variant: name });
    },
    requestLoco: (name) => {
      if (name !== "legs" && name !== "rollers") return;
      setStore({ locoWant: name });
      reconcileLoco();
    },
    resetSim,
    spawnBall: () => spawnBall(),
    startEntrance: () => ceremony.startEntrance(),
    loadCustomPolicy: (ref) => loadCustomPolicyIntoGame(ref),
    clearCustomPolicy: () => revertCustomPolicy(),
  });

  // Deterministic hooks for automated verification (rAF pauses in
  // background tabs, and the control loop is async).
  window.rl = {
    get model() { return model; },
    get data() { return data; },
    mujoco, camera, controls,
    get mode() { return mode; },
    get sitFlag() { return sitFlag; },
    buildObs, cmd,
    velCmd: kbSource.command, lastAction, resetSim,
    controller, kbSource, padSource,
    spawnBall, triggerKick, triggerRoll, sessions, ort,
    loadCustomPolicy: loadCustomPolicyIntoGame,
    clearCustomPolicy: revertCustomPolicy,
    get customPolicy() { return customPolicy; },
    get scriptRun() { return scriptRun; },
    toggleScript,
    get watchdogEvents() { return watchdogEvents.slice(); },
    get loco() { return loco; },
    get locoSwitching() { return locoSwitching; },
    toggleLoco, setLoco, ensureRollers,
    triggerCrouch,
    get crouchPhase() { return crouchRun?.phase ?? null; },
    triggerGroundPick,
    get groundPickPhase() { return pickRun?.phase ?? null; },
    get kickSteps() { return KICK_STEPS; },
    set kickSteps(v) { KICK_STEPS = v; },
    get recovery() { return recovery?.state ?? null; },
    // Debug shove for fall-recovery testing: an instantaneous trunk
    // velocity kick (free-joint dofs are qvel[0..5]).
    debugPush: (vx = 0, vy = 0, vz = 0, wx = 0, wy = 0, wz = 0) => {
      const qvel = data.qvel;
      qvel[0] += vx; qvel[1] += vy; qvel[2] += vz;
      qvel[3] += wx; qvel[4] += wy; qvel[5] += wz;
    },
    get headMode() { return headMode; },
    toggleHeadMode, headTarget, headSmooth,
    get ballActive() { return ballActive; },
    get ballQposAdr() { return ballQposAdr; },
    get chaseCam() { return chaseCam; },
    set chaseCam(v) { chaseCam = !!v; },
    get props() { return propGroups; },
    get relief() { return reliefOn; },
    setRelief: (v) => { reliefOn = !!v; },
    get camResetActive() { return camResetT0 !== null; },
    get respawnActive() { return ceremony?.respawnActive ?? false; },
    get camPose() {
      return {
        pos: camera.position.toArray(),
        target: controls.target.toArray(),
      };
    },
    get chaseYaw() { return { follow: chaseYawFollow, smooth: chaseYawSmooth, held: chaseHeldYaw, tracking: chaseYawTracking }; },
    padOrbitStep,
    jawOpenNow,
    step: async (n = 1) => { for (let i = 0; i < n; i++) await controlStep(); },
    render: () => { syncRig(); renderer.render(scene, camera); },
    frame: (dt = 1 / 60) => frame(dt),
    get ghosts() { return ghosts; },
    get inputLocked() { return inputLocked; },
    get embeddedStatus() { return EMBEDDED ? embeddedStatus() : null; },
    get inferenceCount() { return inferenceCount; },
    entrance: {
      start: () => ceremony.startEntrance(),
      setReveal: (floor, wall) => ceremony.setReveal(floor, wall),
      setFx: (p) => ceremony.setFx(p),
    },
  };

  // Boot complete: the sim/HUD go live immediately. The BIOS readout (if
  // the user already waddled in, or when they do) sees bootDone and closes
  // with READY. + fade on its own.
  if (EMBEDDED) ceremony.finishEntrance();
  setStore({ bootDone: true });
  if (EMBEDDED) {
    registerEmbeddedRuntime({
      getStatus: embeddedStatus,
      command: embeddedCommand,
      selectDuck,
      park: editPark,
      swarm: requestSwarm,
      swarmIntent: requestSwarmIntent,
      presentation: applyPresentation,
      setAutonomy,
      setPaused: (value) => {
        embeddedPaused = !!value || !!runtimeError;
        if (embeddedPaused) ballTask?.cancel("Ball task cancelled while the simulator is paused or hidden.");
        if (embeddedPaused) { stopSwarm("Group control stopped while the simulator is hidden or paused."); setAutonomy(false); stopExternalCommand(); followEnabled = false; followDrive = [0, 0, 0]; waypointSource.cancel(); stopWheee({ silent: true }); }
      },
    });
    // The standalone host never joins public relay rooms or broadcasts
    // robot state to visitors on the original Hugging Face Space.
    return;
  }

  // ?move=<ref> / ?policy=<ref> / ?session=<id> (parsed in App.jsx into the
  // store): load the community move now that the sim is up. Fire-and-forget
  // - a bad ref surfaces its error in the HUD while the official walker
  // keeps the duck alive.
  const requestedPolicy = store().customPolicy;
  if (requestedPolicy?.status === "loading" && requestedPolicy.ref) {
    loadCustomPolicyIntoGame(requestedPolicy.ref);
  }

  // ── Multiplayer ghosts (WebRTC, serverless signaling) ────────────────
  // Broadcast this duck's pose and render up to 3 other visitors live as
  // translucent ducks. Fire-and-forget: any failure just means no ghosts.
  const r3 = (x) => Math.round(x * 1000) / 1000;
  try {
    // Ghosts only join once the entrance has fully played: the world (and
    // this duck) must stay hidden until then, translucent peers included.
    await ceremony.entranceFinished;
    ghosts = await initGhosts({
      scene, rig, cloneRig, setJoint, setJawOpen, applyVariant,
      jointNames: JOINT_NAMES,
      // Payload sanitizing: ghosts.js coerces unknown peer variants to the
      // default instead of letting applyVariant throw on a bad key.
      variantNames: Object.keys(VARIANTS),
      defaultVariant: DEFAULT_VARIANT,
      // Ghost rig per locomotion flag: roller peers clone the live roller
      // rig when this tab has it, else the lightweight ghost-only roller
      // rig. hasRigFor/prepareRigFor let ghosts.js render legs as a
      // stopgap while lazily loading the real thing, then rebuild.
      getRigFor: (l) =>
        (l ? (locos.rollers?.rig ?? ghostRollerRig ?? locos.legs.rig) : locos.legs.rig),
      hasRigFor: (l) => !l || !!(locos.rollers || ghostRollerRig),
      prepareRigFor: (l) => { if (l) ensureGhostRollerRig(); },
      // Ghost ball visual: shares the local ball's geometry and clones its
      // material (ghosts.js makes it translucent). Same Z-up group trick
      // as createBallVisual - the mesh takes the raw MJCF free-joint pose.
      makeGhostBall: () => {
        const group = new THREE.Group();
        group.rotation.x = -Math.PI / 2;
        const mesh = new THREE.Mesh(ballMesh.geometry, ballMesh.material.clone());
        group.add(mesh);
        return { group, mesh };
      },
      getLocalState: () => {
        const qpos = data.qpos;
        const j = new Array(NUM_JOINTS);
        for (let i = 0; i < NUM_JOINTS; i++) j[i] = r3(qpos[qposAdr[i]]);
        const st = {
          p: [r3(qpos[0]), r3(qpos[1]), r3(qpos[2]), r3(qpos[3]), r3(qpos[4]), r3(qpos[5]), r3(qpos[6])],
          j,
          w: r3(jawOpenNow()),
          v: currentVariant,
          l: loco === "rollers" ? 1 : 0,
        };
        // Ball free-joint pose, only while a ball is in play (old clients
        // ignore the extra field; absent = no ball on this peer's field).
        if (ballActive) {
          const a = ballQposAdr;
          st.b = [r3(qpos[a]), r3(qpos[a + 1]), r3(qpos[a + 2]), r3(qpos[a + 3]), r3(qpos[a + 4]), r3(qpos[a + 5]), r3(qpos[a + 6])];
        }
        return st;
      },
    });
    liveGhostSessions.add(ghosts);
    if (ghosts.room) ghosts.room.onPeerJoin = () => stickers?.pop("hi");
  } catch (e) {
    window.__ghostErr = String((e && e.stack) || e);
    console.warn("ghosts disabled:", e);
  }
}

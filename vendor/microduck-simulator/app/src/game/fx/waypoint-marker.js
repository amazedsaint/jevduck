// Click-to-walk target reticle, drawn in the same art direction as the
// wireframe materialization (fx-wireframe.js): additive orange hairlines
// (1 px GL lines), deterministic hologram flicker, ease-out timing.
//
// Anatomy (flat on the floor, group-local XZ plane):
//   - main ring: hairline LineLoop
//   - 4 radial ticks, slowly rotating (the "alive" tell while idle)
//   - center cross marking the exact click point
//   - ping ring: one-shot expanding echo played on (re)acquire
//
// Lifecycle, driven by update(dt, target) where target is the MJCF [x, y]
// from WaypointSource (or null):
//   click / re-click  -> "in": the ring resolves from wide to tight while
//                        fading up, and the ping echoes outward
//   arrival / cancel  -> "out": quick collapse + fade, then hidden
//
// The MJCF -> three mapping (x, y) -> (x, -y) lives here so the caller
// can hand the source's target over untouched.

import * as THREE from "three";

// Matches the scan color vec3(1.0, 0.34, 0.06) in fx-wireframe.js.
const COLOR = 0xff5710;

const R = 0.06; // main ring radius (m)
const TICK_R0 = 0.08, TICK_R1 = 0.102;
const CROSS_R = 0.013;
const FLOOR_Y = 0.012; // sits just above the floor plane

const IN_S = 0.35; // acquire: wide -> tight
const OUT_S = 0.22; // arrival/cancel: collapse + fade
const PING_S = 0.55; // echo ring lifetime
const PING_SCALE = 2.6; // echo end scale (relative to the main ring)

const BODY_OPACITY = 0.85;
const TICK_SPIN = 0.7; // rad/s

const clamp01 = (x) => Math.min(Math.max(x, 0), 1);
// Same ease-out as the scan: full speed on the cue, soft landing.
const ease = (x) => 1 - (1 - x) * (1 - x);

// Deterministic hologram flicker, same recipe as fx-wireframe.js (no
// Math.random so captures are reproducible).
const hash = (x) => {
  const s = Math.sin(x * 127.1) * 43758.5453;
  return s - Math.floor(s);
};
const flickerAt = (time) => {
  let f = 0.86 + 0.14 * hash(Math.floor(time * 60) + 0.5);
  if (hash(Math.floor(time * 24) + 7.7) < 0.08) f *= 0.55; // dropouts
  return f;
};

function circleGeometry(r, n = 64) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r));
  }
  return new THREE.BufferGeometry().setFromPoints(pts);
}

function lineMaterial() {
  return new THREE.LineBasicMaterial({
    color: COLOR,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
}

export function createWaypointMarker() {
  const group = new THREE.Group();
  group.visible = false;
  group.position.y = FLOOR_Y;

  const bodyMat = lineMaterial();
  const pingMat = lineMaterial();

  const ringGeo = circleGeometry(R);
  const ring = new THREE.LineLoop(ringGeo, bodyMat);

  // 4 radial ticks at the diagonals, rotated as one object while idle.
  const tickPts = [];
  for (let i = 0; i < 4; i++) {
    const a = Math.PI / 4 + (i * Math.PI) / 2;
    const c = Math.cos(a), s = Math.sin(a);
    tickPts.push(
      new THREE.Vector3(c * TICK_R0, 0, s * TICK_R0),
      new THREE.Vector3(c * TICK_R1, 0, s * TICK_R1),
    );
  }
  const ticks = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(tickPts),
    bodyMat,
  );

  const cross = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-CROSS_R, 0, 0), new THREE.Vector3(CROSS_R, 0, 0),
      new THREE.Vector3(0, 0, -CROSS_R), new THREE.Vector3(0, 0, CROSS_R),
    ]),
    bodyMat,
  );

  // One-shot expanding echo; shares the ring geometry, scale-animated.
  const ping = new THREE.LineLoop(ringGeo, pingMat);
  ping.visible = false;

  for (const o of [ring, ticks, cross, ping]) {
    o.renderOrder = 5; // over the floor, same layer as the fx overlays
    group.add(o);
  }

  let phase = "hidden"; // hidden | in | idle | out
  let tPhase = 0;
  let tPing = Infinity; // >= PING_S means the echo is spent
  let time = 0; // flicker clock
  let px = 0, pz = 0; // shown position (three coords)

  function acquire(x, z) {
    px = x;
    pz = z;
    group.position.set(px, FLOOR_Y, pz);
    phase = "in";
    tPhase = 0;
    tPing = 0;
    group.visible = true;
  }

  function update(dt, target) {
    time += dt;
    if (target) {
      const tx = target[0], tz = -target[1]; // MJCF -> three
      if (phase === "hidden" || phase === "out") acquire(tx, tz);
      else if (Math.hypot(tx - px, tz - pz) > 1e-3) acquire(tx, tz); // re-click
    } else if (phase === "in" || phase === "idle") {
      phase = "out";
      tPhase = 0;
    }
    if (phase === "hidden") return;

    tPhase += dt;
    tPing += dt;
    ticks.rotation.y = time * TICK_SPIN;

    let scale = 1;
    let alpha = BODY_OPACITY;
    if (phase === "in") {
      const k = ease(clamp01(tPhase / IN_S));
      scale = 1.6 - 0.6 * k; // resolves from wide to tight
      alpha *= k;
      if (tPhase >= IN_S) phase = "idle";
    } else if (phase === "idle") {
      scale = 1 + 0.03 * Math.sin(time * 2.4); // subtle breathing
      alpha *= flickerAt(time);
    } else if (phase === "out") {
      const k = ease(clamp01(tPhase / OUT_S));
      scale = 1 - 0.35 * k;
      alpha *= 1 - k;
      if (tPhase >= OUT_S) {
        phase = "hidden";
        group.visible = false;
      }
    }
    for (const o of [ring, ticks, cross]) o.scale.setScalar(scale);
    bodyMat.opacity = alpha;

    if (tPing < PING_S) {
      const p = ease(clamp01(tPing / PING_S));
      ping.visible = true;
      ping.scale.setScalar(1 + (PING_SCALE - 1) * p);
      pingMat.opacity = 0.9 * (1 - p);
    } else {
      ping.visible = false;
    }
  }

  return { group, update };
}

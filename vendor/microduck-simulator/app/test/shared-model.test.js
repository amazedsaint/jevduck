import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import loadMujoco from "@mujoco/mujoco";
import { buildParkModelXml, copyNamedPhysicsState, COMPANION_PREFIX } from "../src/game/shared-model.js";
import { PARK_OBSTACLE_SLOTS, PARK_OBSTACLE_HALF_SIZE } from "../src/game/park-geometry.js";

const root = new URL("../public/robot/mjlab/", import.meta.url);
const sources = {
  legs: readFileSync(new URL("robot_allcollisions.xml", root), "utf8"),
  rollers: readFileSync(new URL("robot_allcollisions_rollers.xml", root), "utf8"),
};
const mujoco = await loadMujoco();
const objects = mujoco.mjtObj;
const name = (model, type, id) => mujoco.mj_id2name(model, objects[type].value, id);
const id = (model, type, value) => mujoco.mj_name2id(model, objects[type].value, value);
function compile(first, second = null) {
  const { xml, meshFiles } = buildParkModelXml(sources[first], second ? sources[second] : null);
  const vfs = new mujoco.MjVFS();
  for (const file of meshFiles) vfs.addBuffer(`assets/${file}`, new Uint8Array(readFileSync(new URL(`meshes/${file}`, root))));
  const model = mujoco.MjModel.from_xml_string(xml, vfs), data = new mujoco.MjData(model);
  mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data);
  return { model, data, dispose: () => { data.delete(); model.delete(); vfs.delete(); } };
}
function contactBodies(world) {
  const contacts = world.data.contact;
  try {
    return Array.from({ length: world.data.ncon }, (_, c) => {
      const contact = contacts.get(c);
      try {
        const first = Number(contact.geom[0]), second = Number(contact.geom[1]);
        return [name(world.model, "mjOBJ_BODY", world.model.geom_bodyid[first]), name(world.model, "mjOBJ_BODY", world.model.geom_bodyid[second])];
      } finally { contact.delete(); }
    });
  } finally { contacts.delete(); }
}

test("all physical locomotion combinations compile with unchanged per-duck inertias and actuator gains", () => {
  const baselines = { legs: compile("legs"), rollers: compile("rollers") };
  try {
    for (const first of ["legs", "rollers"]) for (const second of ["legs", "rollers"]) {
      const world = compile(first, second);
      try {
        assert.equal(world.model.nu, 28);
        assert.equal(world.model.nmocap, 1);
        for (const [prefix, baseline] of [["", baselines[first]], [COMPANION_PREFIX, baselines[second]]]) {
          for (let body = 1; body < baseline.model.nbody; body++) {
            const bodyName = name(baseline.model, "mjOBJ_BODY", body);
            if (bodyName === "ball") continue;
            const other = id(world.model, "mjOBJ_BODY", prefix + bodyName);
            assert.ok(other >= 0, prefix + bodyName);
            assert.equal(world.model.body_mass[other], baseline.model.body_mass[body]);
            assert.deepEqual(Array.from(world.model.body_inertia.slice(other * 3, other * 3 + 3)),
              Array.from(baseline.model.body_inertia.slice(body * 3, body * 3 + 3)));
          }
          for (let actuator = 0; actuator < baseline.model.nu; actuator++) {
            const other = id(world.model, "mjOBJ_ACTUATOR", prefix + name(baseline.model, "mjOBJ_ACTUATOR", actuator));
            assert.deepEqual(Array.from(world.model.actuator_gainprm.slice(other * 10, other * 10 + 10)),
              Array.from(baseline.model.actuator_gainprm.slice(actuator * 10, actuator * 10 + 10)));
          }
        }
      } finally { world.dispose(); }
    }
  } finally { baselines.legs.dispose(); baselines.rollers.dispose(); }
});

test("disabled box is physically parked away from the ball and uses the shared geometry", () => {
  const world = compile("legs", "legs");
  try {
    assert.deepEqual(Array.from(world.data.mocap_pos), PARK_OBSTACLE_SLOTS.off);
    const box = id(world.model, "mjOBJ_GEOM", "park_obstacle_geom");
    assert.deepEqual(Array.from(world.model.geom_size.slice(box * 3, box * 3 + 3)), PARK_OBSTACLE_HALF_SIZE);
    assert.equal(contactBodies(world).some(pair => pair.includes("park_obstacle")), false);
  } finally { world.dispose(); }
});

test("both robot trees and the obstacle participate in the same collision world", () => {
  const world = compile("legs", "legs");
  try {
    const primary = world.model.jnt("trunk_base_freejoint").qposadr;
    const peer = world.model.jnt("duck2_trunk_base_freejoint").qposadr;
    world.data.qpos.set(world.data.qpos.slice(primary, primary + 7), peer);
    mujoco.mj_forward(world.model, world.data);
    assert.ok(contactBodies(world).some(pair => pair.some(body => body.startsWith(COMPANION_PREFIX))
      && pair.some(body => !body.startsWith(COMPANION_PREFIX) && body !== "world")), "Overlapping robot trees must create real contacts.");
    world.data.mocap_pos.set(PARK_OBSTACLE_SLOTS.center);
    world.data.qpos[peer] = -0.15; world.data.qpos[peer + 1] = 0;
    mujoco.mj_forward(world.model, world.data);
    assert.ok(contactBodies(world).some(pair => pair.includes("park_obstacle") && pair.some(body => body.startsWith(COMPANION_PREFIX))), "A robot placed against the physical box must contact it.");
  } finally { world.dispose(); }
});

test("switching either duck preserves named state of its peer, the shared ball and obstacle", () => {
  const old = compile("legs", "legs");
  try {
    old.data.time = 12.34;
    for (let i = 0; i < old.model.nv; i++) old.data.qvel[i] = i * 0.003;
    for (let i = 0; i < old.model.nu; i++) old.data.ctrl[i] = 0.1 + i * 0.002;
    old.data.qpos[0] = -0.8;
    old.data.qpos[old.model.jnt("duck2_trunk_base_freejoint").qposadr + 1] = 0.9;
    old.data.qpos[old.model.jnt("ball_freejoint").qposadr] = 0.5;
    old.data.mocap_pos.set(PARK_OBSTACLE_SLOTS.right);
    for (const [first, second] of [["rollers", "legs"], ["legs", "rollers"], ["rollers", "rollers"]]) {
      const next = compile(first, second);
      try {
        copyNamedPhysicsState(mujoco, old.model, old.data, next.model, next.data);
        assert.equal(next.data.time, old.data.time);
        assert.deepEqual(Array.from(next.data.mocap_pos), Array.from(old.data.mocap_pos));
        for (let joint = 0; joint < old.model.njnt; joint++) {
          const other = id(next.model, "mjOBJ_JOINT", name(old.model, "mjOBJ_JOINT", joint));
          assert.ok(other >= 0);
          const type = old.model.jnt_type[joint], nq = type === 0 ? 7 : type === 1 ? 4 : 1, nv = type === 0 ? 6 : type === 1 ? 3 : 1;
          const q = old.model.jnt_qposadr[joint], qNext = next.model.jnt_qposadr[other];
          const v = old.model.jnt_dofadr[joint], vNext = next.model.jnt_dofadr[other];
          assert.deepEqual(Array.from(next.data.qpos.slice(qNext, qNext + nq)), Array.from(old.data.qpos.slice(q, q + nq)));
          assert.deepEqual(Array.from(next.data.qvel.slice(vNext, vNext + nv)), Array.from(old.data.qvel.slice(v, v + nv)));
        }
        assert.deepEqual(Array.from(next.data.ctrl), Array.from(old.data.ctrl));
      } finally { next.dispose(); }
    }
  } finally { old.dispose(); }
});

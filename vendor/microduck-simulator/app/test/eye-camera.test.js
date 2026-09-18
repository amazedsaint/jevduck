import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { DEFAULT_POSE, JOINT_NAMES } from "../src/game/constants.js";
import { applyEyeCameraPose } from "../src/game/eye-camera.js";

function exportedCamera(rollers = false, offsets = {}) {
  const suffix = rollers ? "_rollers" : "";
  const k = JSON.parse(readFileSync(new URL(`../public/robot/mjlab/kinematics${suffix}.json`, import.meta.url), "utf8"));
  const xml = readFileSync(new URL(`../public/robot/mjlab/robot_allcollisions${suffix}.xml`, import.meta.url), "utf8");
  const tag = xml.match(/<camera\s+name="head_camera"[^>]*>/)?.[0];
  assert.ok(tag, "the actual export must contain the head camera");
  const vector = (name) => tag.match(new RegExp(`${name}="([^"]+)"`))[1].trim().split(/\s+/).map(Number);
  const bodies = new Map();
  for (const b of k.bodies) {
    const object = new THREE.Object3D();
    object.position.fromArray(b.pos);
    object.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]).normalize();
    if (b.joint) {
      const index = JOINT_NAMES.indexOf(b.joint.name);
      if (index >= 0) object.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(...b.joint.axis), DEFAULT_POSE[index] + (offsets[b.joint.name] || 0),
      ));
    }
    bodies.set(b.name, object);
  }
  for (const b of k.bodies) if (bodies.has(b.parent)) bodies.get(b.parent).add(bodies.get(b.name));
  const mount = new THREE.Object3D();
  mount.position.fromArray(vector("pos"));
  const q = vector("quat");
  mount.quaternion.set(q[1], q[2], q[3], q[0]).normalize();
  bodies.get("jaw_soft").add(mount);
  mount.updateWorldMatrix(true, false);
  const e = mount.matrixWorld.elements;
  const positions = mount.getWorldPosition(new THREE.Vector3()).toArray();
  const rotations = [e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]];
  const camera = new THREE.PerspectiveCamera();
  applyEyeCameraPose(camera, positions, rotations, 0);
  return { camera, positions };
}

for (const rollers of [false, true]) {
  test(`the actual ${rollers ? "roller" : "leg"} export gives a forward, level eye view`, () => {
    const { camera, positions } = exportedCamera(rollers);
    const forward = camera.getWorldDirection(new THREE.Vector3());
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
    assert.ok(forward.x > 0.999, `robot-forward should be +X, got ${forward.toArray()}`);
    assert.ok(up.y > 0.999, `screen-up should be +Y, got ${up.toArray()}`);
    assert.ok(Math.abs(forward.dot(up)) < 1e-6);
    assert.deepEqual(camera.position.toArray(), [positions[0], positions[2], -positions[1]]);
  });
}

test("eye view follows the real exported head yaw and pitch joints", () => {
  const { camera } = exportedCamera(false, { head_yaw: 0.4, head_pitch: -0.3 });
  const forward = camera.getWorldDirection(new THREE.Vector3());
  assert.ok(forward.x > 0.7, `the view should still face ahead: ${forward.toArray()}`);
  assert.ok(forward.y > 0.2, `negative head pitch should raise the view: ${forward.toArray()}`);
  assert.ok(forward.z < -0.25, `positive head yaw should turn left: ${forward.toArray()}`);
});

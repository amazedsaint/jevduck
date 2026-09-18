import * as THREE from "three";

const zUpToThree = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
const mountToOptical = new THREE.Matrix4().set(
  0, -1, 0, 0,
  -1, 0, 0, 0,
  0, 0, -1, 0,
  0, 0, 0, 1,
);
const rotation = new THREE.Matrix4();

export function applyEyeCameraPose(camera, positions, rotations, cameraId) {
  const p = cameraId * 3, r = cameraId * 9;
  camera.position.set(positions[p], positions[p + 2], -positions[p + 1]);
  // In the exported MJCF, jaw_soft -Z points through the lens and +X
  // points to the top of the head. Its head_camera quaternion [0,0,-1,0]
  // is a mounting frame, not the forward optical view. In that mount
  // frame the view's right/up/back axes are -Y/-X/-Z, respectively.
  // Apply this fixed correction AFTER the live MuJoCo transform, then
  // convert world Z-up to Three Y-up. All actual head motion is retained.
  rotation.set(
    rotations[r], rotations[r + 1], rotations[r + 2], 0,
    rotations[r + 3], rotations[r + 4], rotations[r + 5], 0,
    rotations[r + 6], rotations[r + 7], rotations[r + 8], 0,
    0, 0, 0, 1,
  ).multiply(mountToOptical).premultiply(zUpToThree);
  camera.quaternion.setFromRotationMatrix(rotation);
}

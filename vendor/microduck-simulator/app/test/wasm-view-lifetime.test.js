import test from "node:test";
import assert from "node:assert/strict";
import loadMujoco from "@mujoco/mujoco";
import { CompanionController } from "../src/game/companion-controller.js";

test("settled refreshes state views after native allocation grows the MuJoCo heap", async t => {
  const mujoco = await loadMujoco();
  const model = mujoco.MjModel.from_xml_string(`<mujoco><size memory="16M"/><worldbody><body name="trunk" pos="0 0 .2"><freejoint/><geom type="sphere" size=".03" mass="1"/><site name="imu"/></body></worldbody><sensor><gyro site="imu"/></sensor></mujoco>`);
  const data = new mujoco.MjData(model), allocations = [];
  mujoco.mj_forward(model, data);
  const initialBuffer = data.qvel.buffer, initialBytes = initialBuffer.byteLength;
  let grew = false;
  const controller = Object.create(CompanionController.prototype);
  Object.assign(controller, { getWorld: () => ({ model, data }), qAdr: 0, vAdr: 0, gyroAdr: 0, trunkId: 1 });
  const gravity = controller.gravity.bind(controller);
  controller.gravity = () => {
    if (!grew) {
      // MjData allocation uses the real WASM allocator. It reproduces the
      // same detachment as a body accessor or model switch at a heap limit.
      for (let n = 0; data.qvel.buffer === initialBuffer && n < 32; n++) allocations.push(new mujoco.MjData(model));
      assert.notEqual(data.qvel.buffer, initialBuffer, "The regression must actually grow native memory.");
      assert.equal(initialBuffer.byteLength, 0, "The old MuJoCo typed array buffer is detached.");
      grew = true;
    }
    return gravity();
  };
  try {
    assert.equal(controller.settled(), true);
    assert.equal(controller.settled(), true);
    t.diagnostic(`Real MuJoCo heap grew from ${initialBytes} to ${data.qvel.buffer.byteLength} bytes; settled remained readable.`);
  } finally {
    for (const extra of allocations) extra.delete();
    data.delete(); model.delete();
  }
});

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

const game = readFileSync(new URL("../src/game/game.js", import.meta.url), "utf8");
function functionSource(name) {
  const start = game.indexOf(`  function ${name}(`);
  assert.ok(start >= 0);
  return game.slice(start, game.indexOf("\n  }", start) + 5);
}

test("Stop invalidates an in-flight four-body preparation before autonomous activation", async () => {
  let finishLoad, starts = 0, modelInstalls = 0;
  const load = new Promise(resolve => { finishLoad = resolve; });
  const context = {
    swarmPreparing: false, parkSwitching: false, parkSettle: null, swarmPrepareToken: 0, pendingSwarmRunId: null,
    embeddedPaused: false, embeddedSuspended: () => false, manualInputActive: () => false, grab: null, runtimeError: null,
    swarm: { active: false, runId: null, startScenario() { starts++; }, stop() {} },
    ballTask: { clear() {} }, setAutonomy() {}, waypointSource: { cancel() {} }, exitHeadMode() {},
    controlStepPending: null, parkModels: new Map(), buildPhysicsXml: () => load, addMeshesToVfs: async () => {},
    mujoco: { MjModel: { from_xml_string: () => ({}) }, MjData: class {} }, vfs: {},
    scene: { remove() { modelInstalls++; } }, rig: { placer: {} }, physicalPeers: new Map(),
  };
  const api = runInNewContext(`function stopSwarm(){swarmPrepareToken++;swarm.stop();}\n${functionSource("requestSwarm")}\n({requestSwarm,get preparing(){return swarmPreparing;}});`, context);
  assert.equal(api.requestSwarm({ active: true, runId: "cancelled-run", scenario: "flock" }).accepted, true);
  assert.equal(api.preparing, true);
  assert.equal(api.requestSwarm({ active: false, runId: "cancelled-run" }).accepted, true);
  finishLoad({ xml: "test", meshFiles: [] });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(starts, 0);
  assert.equal(modelInstalls, 0);
  assert.equal(api.preparing, false);
});

test("stale swarm intent cannot command a different active run", () => {
  let dispatched = 0;
  const context = { swarm: { active: true, runId: "new-run", startIntent: () => { dispatched++; return { accepted: true }; } },
    swarmPreparing: false, parkSwitching: false, embeddedPaused: false, embeddedSuspended: () => false,
    manualInputActive: () => false, grab: null, swarmPoses: () => [] };
  const command = runInNewContext(`${functionSource("requestSwarmIntent")}\nrequestSwarmIntent;`, context);
  assert.equal(command({ runId: "old-run", id: "late-decision", intent: "advance" }).accepted, false);
  assert.equal(dispatched, 0);
  assert.equal(command({ runId: "new-run", id: "new-decision", intent: "hold" }).accepted, true);
  assert.equal(dispatched, 1);
});

test("Stop during a peer render-resource load cannot reset the live scene afterward", async () => {
  let finishRig, starts = 0, modelInstalls = 0, rigLoads = 0;
  const rigLoad = new Promise(resolve => { finishRig = resolve; });
  const context = {
    swarmPreparing: false, parkSwitching: false, parkSettle: null, swarmPrepareToken: 0, pendingSwarmRunId: null,
    embeddedPaused: false, embeddedSuspended: () => false, manualInputActive: () => false, grab: null, runtimeError: null,
    swarm: { active: false, runId: null, startScenario() { starts++; }, stop() {} },
    ballTask: { clear() {} }, setAutonomy() {}, waypointSource: { cancel() {} }, exitHeadMode() {},
    controlStepPending: null, parkModels: new Map(), buildPhysicsXml: async () => ({ xml: "test", meshFiles: [] }), addMeshesToVfs: async () => {},
    mujoco: { MjModel: { from_xml_string: () => ({}) }, MjData: class {} }, vfs: {},
    DUCK_IDS: ["duck1", "duck2", "duck3", "duck4"], k: {}, variantForPeer: () => ({}), materialHookFor: () => ({}),
    buildRig: () => { rigLoads++; return rigLoad; },
    scene: { remove() { modelInstalls++; } }, rig: { placer: {} }, physicalPeers: new Map([["duck2", {}]]),
  };
  const api = runInNewContext(`function stopSwarm(){swarmPrepareToken++;swarm.stop();}\n${functionSource("requestSwarm")}\n({requestSwarm,get preparing(){return swarmPreparing;}});`, context);
  api.requestSwarm({ active: true, runId: "rig-cancel", scenario: "split" });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(rigLoads, 1);
  api.requestSwarm({ active: false, runId: "rig-cancel" });
  finishRig({ placer: {} });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(starts, 0); assert.equal(modelInstalls, 0); assert.equal(api.preparing, false);
});

test("physical picking chooses the closest measured duck among all four", () => {
  const rigs = [1, 2, 3, 4].map(index => ({ id: `duck${index}` }));
  const context = { rig: { placer: rigs[0] }, physicalPeers: new Map(rigs.slice(1).map(value => [value.id, { rig: { placer: value } }])),
    _grabRaycaster: { intersectObject: object => [{ distance: 5 - Number(object.id.slice(-1)), point: object.id }] }, ballActive: false };
  const pick = runInNewContext(`${functionSource("grabPick")}\ngrabPick;`, context);
  assert.equal(pick().duckId, "duck4");
});

test("camera orbit keeps group authority while a real grab interrupts it", () => {
  const start = game.indexOf("    const directPointerInput = () => {");
  const source = game.slice(start, game.indexOf("    const stopOnBlur", start));
  let cancels = 0;
  const context = { grab: null, ballTask: { cancel: () => { cancels++; } }, swarm: { active: true }, swarmPreparing: false,
    stopSwarm: () => { cancels++; }, stopExternalCommand: () => { cancels++; }, notifyEmbeddedManualInput: () => { cancels++; } };
  const input = runInNewContext(`${source}\ndirectPointerInput;`, context);
  input(); assert.equal(cancels, 0);
  context.grab = {}; input(); assert.equal(cancels, 4);
});

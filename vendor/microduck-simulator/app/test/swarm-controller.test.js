import test from "node:test";
import assert from "node:assert/strict";
import { SwarmController, scenarioSpawns } from "../src/game/swarm-controller.js";

const poses = scenario => scenarioSpawns(scenario).map(pose => ({ ...pose, loco: "legs", posture: "standing", busy: false, command: [0, 0, 0] }));
const fixture = (scenario = "flock") => {
  const group = poses(scenario), controller = new SwarmController();
  assert.equal(controller.startScenario(scenario, "scene-id", group).accepted, true);
  return { group, controller };
};
const stopped = controller => { for (const id of ["duck1", "duck2", "duck3", "duck4"]) assert.deepEqual(controller.commandFor(id), [0, 0, 0]); };

test("preparation only creates an idle group; a separate instruction owns movement", () => {
  const f = fixture(); f.controller.tick(.02, f.group); stopped(f.controller);
  assert.equal(f.controller.status(f.group).phase, "idle");
  assert.equal(f.controller.startIntent("advance-id", "advance", f.group).accepted, true);
  f.controller.tick(.02, f.group);
  assert.ok(f.group.every(pose => f.controller.commandFor(pose.id)[0] === .25));
  assert.equal(f.controller.status(f.group).commandId, "advance-id");
});

test("hold interrupts an owned instruction and no later tick revives it", () => {
  const f = fixture(); f.controller.startIntent("old", "advance", f.group); f.controller.tick(.02, f.group);
  assert.equal(f.controller.startIntent("new", "regroup", f.group).accepted, false);
  assert.equal(f.controller.startIntent("hold-id", "hold", f.group).accepted, true);
  f.controller.tick(.02, f.group); stopped(f.controller);
  assert.equal(f.controller.status(f.group).commandId, "hold-id");
  f.controller.stop(); f.controller.tick(.02, f.group); stopped(f.controller);
  assert.equal(f.controller.active, false);
});

test("all three other robots participate in each motion admission", () => {
  const f = fixture();
  f.group[0].position = [0, 0, .12];
  f.group[1].position = [-1, -1, .12];
  f.group[2].position = [-1, 1, .12];
  f.group[3].position = [.50, 0, .12];
  f.controller.startIntent("peer-blocked", "advance", f.group); f.controller.tick(.02, f.group);
  assert.deepEqual(f.controller.commandFor("duck1"), [0, 0, 0]);
  assert.deepEqual(f.controller.commandFor("duck4"), [0, 0, 0]);
});

test("invalid, missing or interrupted members fail closed with finite zero commands", () => {
  for (const mutation of [group => group.pop(), group => { group[2].headingRad = NaN; },
    group => { group[3].fallen = true; }, group => { group[1].paused = true; }, group => { group[0].manual = true; }]) {
    const f = fixture(); f.controller.startIntent("interrupt", "advance", f.group); mutation(f.group);
    f.controller.tick(.02, f.group); stopped(f.controller);
    assert.equal(f.controller.phase, "blocked");
    assert.deepEqual(f.controller.status(f.group).availableIntents, ["hold"]);
  }
});

test("the time bound cannot call a stationary group successful", () => {
  const f = fixture(); f.controller.startIntent("stationary", "advance", f.group);
  for (let n = 0; n < 401; n++) f.controller.tick(.02, f.group);
  assert.equal(f.controller.phase, "blocked"); stopped(f.controller);
  assert.match(f.controller.reason, /without measurable group progress/);
  assert.equal(f.controller.status(f.group).targetErrorM, .28);
});

test("centroid, RMS spread and minimum separation are measured from every member", () => {
  const f = fixture(); const status = f.controller.status(f.group);
  assert.equal(status.members, 4);
  assert.ok(Math.abs(status.centroid[0] + .3) < 1e-12);
  assert.ok(Math.abs(status.centroid[1]) < 1e-12);
  assert.ok(Math.abs(status.spreadM - Math.hypot(.45, .45)) < 1e-12);
  assert.equal(status.minSeparationM, .9);
  assert.equal(status.targetErrorM, null);
  f.group[2].busy = true;
  assert.deepEqual(f.controller.status(f.group).availableIntents, ["hold"]);
});

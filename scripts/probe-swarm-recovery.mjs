import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { SwarmController, scenarioSpawns } from '../vendor/microduck-simulator/app/src/game/swarm-controller.js';

const sources = ['src/lib/swarm-contract.ts', 'src/lib/swarm-jev.ts', 'src/lib/swarm-controller.ts', 'vendor/microduck-simulator/app/src/game/swarm-controller.js'];
const episode = (intent, patch = {}) => ({ intent, outcome: 'complete', progressM: .02, targetErrorBeforeM: null, targetErrorAfterM: .30, minSeparationM: .7, ...patch });
function fixture(scenario, patch, recent) {
  const poses = scenarioSpawns(scenario).map(pose => ({ ...pose, loco: 'legs', posture: 'standing', fallen: false, paused: false, busy: false, manual: false }));
  const native = new SwarmController(); native.startScenario(scenario, `recovery-probe-${scenario}`, poses);
  native.startIntent('native-probe', scenario === 'split' || scenario === 'gather' ? 'regroup' : 'advance', poses);
  // The native controller receives stationary poses through its whole window.
  // This produces a real blocked status and its real availability whitelist.
  native.tick(8.1, poses);
  const nativeStatus = native.status(poses);
  return { nativeStatus, input: { state: { seq: 400, time: 54.6, ready: true, paused: false, swarm: { ...nativeStatus, ...patch } }, memory: { recent } } };
}
const fail = episode('advance');
const regroup = episode('regroup', { targetErrorBeforeM: .6, targetErrorAfterM: .1 });
const hold = episode('hold', { progressM: 0, targetErrorBeforeM: .1, targetErrorAfterM: .1 });
const cases = [
  { id: 'flock_stagnant_completed_window', expected: ['regroup', 'disperse'], observed: 'Browser displayed complete advance window, 0.30 m residual and 0.02 m centroid displacement. Other values and earlier outcomes are representative, not captured telemetry.', ...fixture('flock', { phase: 'complete', intent: 'advance', progressM: .02, targetErrorM: .30 }, [episode('advance', { progressM: .2, targetErrorAfterM: .07 }), fail]) },
  { id: 'flock_native_blocked_class', expected: ['regroup', 'disperse', 'hold'], observed: 'Browser displayed blocked advance, 0.28 m residual and 0.52 m minimum separation. Native fixture supplies blocked phase and standing eligibility. Its poses are not the browser poses.', ...fixture('flock', { targetErrorM: .28, minSeparationM: .52 }, [episode('advance', { outcome: 'blocked', progressM: 0, targetErrorAfterM: .28, minSeparationM: .52 })]) },
  { id: 'flock_after_measured_regroup', expected: ['advance'], observed: 'Counterfactual contract fixture: recovery reduces assigned-slot error from 0.60 to 0.10 m. No physical rollout is claimed.', ...fixture('flock', { phase: 'complete', intent: 'regroup', progressM: .02, targetErrorM: .10 }, [fail, regroup]) },
  { id: 'flock_recovery_exhausted', expected: ['hold'], observed: 'Counterfactual contract fixture: two measured recoveries each followed by ineffective advance. Bounded recovery must not become unlimited movement.', ...fixture('flock', { phase: 'complete', intent: 'advance', progressM: .02, targetErrorM: .30 }, [fail, regroup, fail, episode('disperse', { targetErrorBeforeM: .4, targetErrorAfterM: .2 }), fail]) },
  { id: 'split_after_compact_stage_with_holds', expected: ['split'], observed: 'Browser displayed completed regroup, 0.10 m residual, 0.66 m minimum separation and 0.50 m RMS spread. Before-error and centroid were not retained; local holds here are representative.', ...fixture('split', { phase: 'complete', intent: 'regroup', targetErrorM: .10, minSeparationM: .66, spreadM: .50 }, [episode('regroup', { targetErrorBeforeM: null, targetErrorAfterM: .10 }), hold, hold]) },
  { id: 'gather_after_compact_stage_with_holds', expected: ['disperse'], observed: 'Counterfactual gather-stage fixture matching completed compact regroup plus local holds. No physical rollout is claimed.', ...fixture('gather', { phase: 'complete', intent: 'regroup', targetErrorM: .10 }, [regroup, hold, hold]) },
  { id: 'convoy_after_blocked_advance', expected: ['regroup', 'change_leader', 'hold'], observed: 'Native stationary-pose fixture reaches a blocked advance and retains actual runtime standing availability. No captured browser blocked convoy state is claimed.', ...fixture('convoy', {}, [episode('advance', { outcome: 'blocked', progressM: 0, targetErrorAfterM: .18 })]) },
];
const report = { timestamp: new Date().toISOString(), scope: 'Live Jev probes using measured-class fixtures. These combine explicitly listed browser-visible aggregates with native controller status and eligibility. They are not exact browser replay or physical recovery rollouts.', sourceSha256: Object.fromEntries(sources.map(path => [path, crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')])), cases: [] };
const output = process.argv[2] || 'output/swarm-recovery.json';
fs.mkdirSync(path.dirname(output), { recursive: true });
for (const entry of cases) {
  const response = await fetch('http://127.0.0.1:3177/api/swarm', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(entry.input) });
  const result = await response.json();
  const pass = response.status === 200 && entry.expected.includes(result.intent) && (result.intent === 'hold' || !result.abstained);
  report.cases.push({ ...entry, status: response.status, result, pass }); fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ id: entry.id, status: response.status, intent: result.intent, confidence: result.confidence, abstained: result.abstained, pass, alternatives: result.alternatives }));
}
if (report.cases.some(entry => !entry.pass)) process.exitCode = 1;

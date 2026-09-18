import { eligibleSwarmIntents, type SwarmDecision, type SwarmInput, type SwarmIntent, type SwarmState } from "../src/lib/swarm-contract";

export function swarmState(patch: Partial<SwarmState["swarm"]> = {}): SwarmState {
  return { seq: 1, time: 1, ready: true, paused: false, swarm: {
    active: true, runId: "test-run", scenario: "flock", leaderId: "duck1", commandId: null, intent: null, phase: "idle",
    availableIntents: ["advance", "regroup", "disperse", "change_leader", "split", "hold"],
    centroid: [0, 0], spreadM: 0.5, minSeparationM: 0.7, progressM: 0, targetErrorM: null,
    reason: "Ready", elapsedS: 0, members: 4, ...patch,
  } };
}
export function swarmInput(patch: Partial<SwarmState["swarm"]> = {}): SwarmInput { return { state: swarmState(patch), memory: { recent: [] } }; }
export function swarmDecision(intent: SwarmIntent = "advance"): SwarmDecision {
  return { intent, source: "jev", model: "jev-test", confidence: 1, abstained: false, reason: "Bounded group intention selected.", latencyMs: 10, alternatives: [{ intent, probability: 1 }] };
}
export function swarmAnswer(input = swarmInput(), intent: SwarmIntent = "advance", probability = 1, confidence = probability) {
  const eligible = eligibleSwarmIntents(input);
  return { model: "jev-test", answers: { intent: { type: "choice", choice: intent, confidence,
    probabilities: Object.fromEntries(eligible.map(option => [option, option === intent ? probability : (1 - probability) / Math.max(1, eligible.length - 1)])),
  } } };
}

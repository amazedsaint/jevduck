import { bootLog, useGame } from "../store.js";
import { PRESENTATION_SCENES, PRESENTATION_CAMERAS } from "./presentation.js";
import { DUCK_ACTIONS } from "./duck-actions.js";

export const EMBEDDED = new URLSearchParams(location.search).get("embed") === "1";
export const BRIDGE_CHANNEL = "microduck-sim-v1";
export const BRIDGE_ACTIONS = new Set(DUCK_ACTIONS);

let adapter = null;
let parentActive = true;
let pageActive = true;
let bridgeMounted = false;
let sendStatus = () => {};
let statusSeq = 0;
let presentation = { scene: "studio", camera: "orbit" };

export function notifyEmbeddedManualInput() {
  if (EMBEDDED) adapter?.setAutonomy(false);
  if (EMBEDDED && bridgeMounted && window.parent !== window) {
    window.parent.postMessage({ channel: BRIDGE_CHANNEL, type: "manual" }, location.origin);
  }
}

export function notifyEmbeddedStop() {
  if (EMBEDDED) adapter?.setAutonomy(false);
  if (EMBEDDED && bridgeMounted && window.parent !== window) {
    window.parent.postMessage({ channel: BRIDGE_CHANNEL, type: "stop" }, location.origin);
  }
}

export function embeddedSuspended() {
  return EMBEDDED && (document.hidden || !parentActive || !pageActive || !bridgeMounted);
}

export function registerEmbeddedRuntime(runtime) {
  if (!EMBEDDED) return;
  adapter = runtime;
  adapter.presentation(presentation);
  adapter.setPaused(embeddedSuspended());
  sendStatus();
}

// Strictly bind to this document's same-origin parent. No wildcard
// targetOrigin, cross-origin messages, arbitrary vectors, or URLs.
export function mountEmbeddedBridge() {
  if (!EMBEDDED) return () => {};
  bridgeMounted = true;
  const origin = location.origin;
  const previous = new Map();
  const post = (payload) => {
    if (window.parent !== window) window.parent.postMessage({ channel: BRIDGE_CHANNEL, ...payload }, origin);
  };
  sendStatus = () => {
    if (adapter) {
      post({ type: "status", seq: ++statusSeq, ...adapter.getStatus() });
      return;
    }
    const state = useGame.getState();
    const error = state.bootFailed
      ? bootLog.findLast((entry) => entry.label.startsWith(">> "))?.label.slice(3) || "The simulator could not start."
      : undefined;
    post({ type: "status", seq: ++statusSeq, ready: false, loco: "legs", mode: "loading", time: 0,
      position: [0, 0, 0], tiltRad: 0, command: [0, 0, 0], fallen: false,
      headingRad: 0, clearance: { front: 0, back: 0, left: 0, right: 0 }, spatialValid: false,
      autonomyActive: false, manual: false, guardReason: null, guardSeq: 0,
      busy: !error, paused: embeddedSuspended(), policy: "loading", inferenceCount: 0,
      posture: "standing", pendingAction: null, phase: "loading", ...presentation,
      ...(error ? { error } : {}) });
  };
  const syncVisibility = () => {
    adapter?.setPaused(embeddedSuspended());
    sendStatus();
  };
  const onMessage = (event) => {
    if (event.origin !== origin || event.source !== window.parent || window.parent === window) return;
    const message = event.data;
    if (!message || typeof message !== "object" || message.channel !== BRIDGE_CHANNEL) return;
    if (message.type === "autonomy" && typeof message.active === "boolean") {
      // Opt-in is never remembered across a load, pause or manual input.
      adapter?.setAutonomy(message.active);
      sendStatus();
      return;
    }
    if (message.type === "select-duck") {
      if (!["duck1", "duck2", "duck3", "duck4"].includes(message.duckId)) return;
      const result = adapter?.selectDuck?.(message.duckId) ?? { accepted: false, message: "Robot models are loading." };
      post({ type: "selection-result", ...result });
      sendStatus();
      return;
    }
    if (message.type === "swarm") {
      if (typeof message.active !== "boolean" || typeof message.runId !== "string" || message.runId.length < 1 || message.runId.length > 100) return;
      if (message.active && !["flock", "gather", "convoy", "split"].includes(message.scenario)) return;
      const result = adapter?.swarm?.({ active: message.active, runId: message.runId, scenario: message.scenario }) ?? { accepted: false, message: "The simulator is loading." };
      post({ type: "swarm-result", runId: message.runId, ...result }); sendStatus(); return;
    }
    if (message.type === "swarm-intent") {
      if (typeof message.runId !== "string" || message.runId.length < 1 || message.runId.length > 100 || typeof message.id !== "string" || message.id.length < 1 || message.id.length > 100) return;
      if (!["advance", "regroup", "disperse", "change_leader", "split", "hold"].includes(message.intent)) return;
      const key = `swarm:${message.runId}:${message.id}`;
      if (previous.has(key)) { post(previous.get(key)); return; }
      const result = adapter?.swarmIntent?.({ runId: message.runId, id: message.id, intent: message.intent }) ?? { accepted: false, message: "The simulator is loading." };
      const ack = { type: "swarm-intent-result", runId: message.runId, id: message.id, ...result };
      previous.set(key, ack); if (previous.size > 100) previous.delete(previous.keys().next().value);
      post(ack); sendStatus(); return;
    }
    if (message.type === "park") {
      if (message.follow === undefined && message.obstacle === undefined) return;
      if (message.follow !== undefined && typeof message.follow !== "boolean") return;
      if (message.obstacle !== undefined && !["center", "left", "right", "off"].includes(message.obstacle)) return;
      const result = adapter?.park?.({ follow: message.follow, obstacle: message.obstacle }) ?? { accepted: false, message: "World configuration is loading." };
      post({ type: "park-result", ...result });
      sendStatus();
      return;
    }
    if (message.type === "presentation") {
      if (message.scene === undefined && message.camera === undefined) return;
      if (message.scene !== undefined && !PRESENTATION_SCENES.has(message.scene)) return;
      if (message.camera !== undefined && !PRESENTATION_CAMERAS.has(message.camera)) return;
      presentation = { scene: message.scene ?? presentation.scene, camera: message.camera ?? presentation.camera };
      adapter?.presentation(presentation);
      sendStatus();
      return;
    }
    if (message.type === "lifecycle" && typeof message.active === "boolean") {
      parentActive = message.active;
      syncVisibility();
      return;
    }
    if (message.type === "hello" || message.type === "request-status") { sendStatus(); return; }
    if (message.type !== "command" || typeof message.id !== "string" || message.id.length < 1 || message.id.length > 100) return;
    if (previous.has(message.id)) { post(previous.get(message.id)); return; }
    let result;
    if (!BRIDGE_ACTIONS.has(message.action)) result = { accepted: false, message: "That action is not supported by the simulator." };
    else if (!adapter) result = { accepted: false, message: "The simulator is still loading." };
    else result = adapter.command(message.action, { all: message.all === true, id: message.id });
    // Let the host consume the guard receipt and cancel its owned mission
    // before a rejected ACK could classify that mission as a terminal error.
    // Accepted commands retain ACK-before-status completion ordering.
    if (!result.accepted && result.blockedByGuard === true) sendStatus();
    const ack = { type: "ack", id: message.id, ...result,
      completion: result.accepted ? result.completion || "status" : "immediate", statusSeq };
    previous.set(message.id, ack);
    if (previous.size > 100) previous.delete(previous.keys().next().value);
    post(ack);
    sendStatus();
  };
  const onPageHide = () => { pageActive = false; syncVisibility(); };
  const onPageShow = () => { pageActive = true; syncVisibility(); };
  window.addEventListener("message", onMessage);
  window.addEventListener("pagehide", onPageHide);
  window.addEventListener("pageshow", onPageShow);
  document.addEventListener("visibilitychange", syncVisibility);
  const interval = setInterval(sendStatus, 250);
  syncVisibility();
  return () => {
    bridgeMounted = false;
    adapter?.setPaused(true);
    clearInterval(interval);
    window.removeEventListener("message", onMessage);
    window.removeEventListener("pagehide", onPageHide);
    window.removeEventListener("pageshow", onPageShow);
    document.removeEventListener("visibilitychange", syncVisibility);
    sendStatus = () => {};
  };
}

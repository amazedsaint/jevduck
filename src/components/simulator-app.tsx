"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, ArrowDown, ArrowLeft, ArrowRight, ArrowUp, ArrowUpRight, Bookmark, Camera, Check, ChevronDown, CircleHelp, Eye, Focus, LoaderCircle, Maximize2, MoveUpRight, Network, Orbit, PanelRightClose, PanelRightOpen, Pause, Play, RotateCcw, Square, Terminal, Trash2, X } from "lucide-react";
import { SIMULATOR_ACTIONS, SIMULATOR_ACTION_LABELS, type SimulatorAction, type SimulatorContext, type SimulatorDecision } from "@/lib/simulator";
import { MissionRunner, type MissionSnapshot } from "@/lib/mission-runner";
import { AutonomyController, type AutonomySnapshot } from "@/lib/autonomy-controller";
import { AUTONOMY_LABELS, autonomyDecisionSchema, autonomyStateSchema, type AutonomyMode, type AutonomyState } from "@/lib/autonomy-contract";
import { AutonomyPanel } from "./autonomy-panel";
import { ParkPanel, type DuckId, type ObstacleSlot, type ParkDuck, type ParkState } from "./park-panel";
import { MovesPanel } from "./moves-panel";
import { SwarmController, type SwarmSnapshot } from "@/lib/swarm-controller";
import { swarmDecisionSchema, swarmRuntimeSchema, type SwarmRuntime, type SwarmScenario } from "@/lib/swarm-contract";
import { SwarmPanel, SWARM_SCENES } from "./swarm-panel";

type Action = Exclude<SimulatorAction, "none" | "clarify">;
type Scene = "studio" | "moon" | "sunset";
type View = "orbit" | "follow" | "eyes";
type Status = SimulatorContext & Pick<AutonomyState, "ball" | "companion" | "clearanceSources" | "task"> & { seq: number; time: number; position: number[]; tiltRad: number; command: number[]; policy: string; inferenceCount: number; posture: AutonomyState["posture"]; pendingAction: string | null; phase: string; fps: number; controlHz: number; scene?: Scene; camera?: View; error?: string; headingRad: number; clearance: AutonomyState["clearance"]; spatialValid: boolean; autonomyActive: boolean; guardReason: string | null; guardSeq: number; manual?: boolean; selectedDuckId: DuckId; ducks: ParkDuck[]; park: ParkState; swarm: SwarmRuntime | null };
type Routine = { id: string; name: string; plan: Action[]; scene: Scene | "keep"; camera: View | "keep" };
type LogEntry = { id: string; title: string; detail: string; ok: boolean };
const CHANNEL = "microduck-sim-v1";
const STORAGE = "jevduck.routines.v1";
const enrichmentSchema = autonomyStateSchema.pick({ ball: true, companion: true, clearanceSources: true, task: true });
const INITIAL: Status = { ready: false, loco: "legs", mode: "loading", busy: false, fallen: false, paused: false, seq: 0, time: 0, position: [0, 0, 0], tiltRad: 0, command: [0, 0, 0], policy: "", inferenceCount: 0, posture: "standing", pendingAction: null, phase: "loading", fps: 0, controlHz: 0, headingRad: 0, clearance: { front: 0, back: 0, left: 0, right: 0 }, spatialValid: false, autonomyActive: false, guardReason: null, guardSeq: 0, selectedDuckId: "duck1", ducks: [], park: { follow: false, leaderId: "duck1", obstacle: "off", followState: "" }, swarm: null };
const INITIAL_AUTONOMY: AutonomySnapshot = { active: false, mode: null, phase: "off", currentDecision: null, lastDecision: null, missionId: null, completedCycles: 0, blockedCycles: 0, recent: [], visited: [], distanceM: 0, nextRequestAt: null };
const INITIAL_SWARM: SwarmSnapshot = { active: false, scenario: null, phase: "off", runId: null, commandId: null, currentDecision: null, lastDecision: null, reason: "", completedCycles: 0, blockedCycles: 0, recent: [], nextRequestAt: null };
const isDuckId = (value: unknown): value is DuckId => typeof value === "string" && ["duck1", "duck2", "duck3", "duck4"].includes(value);
function autonomyObservation(status: Status): AutonomyState {
  const { ready, busy, paused, fallen, loco, mode, seq, time, headingRad, posture, clearance, spatialValid, guardReason, guardSeq, autonomyActive, availableActions, selectedDuckId, ball, companion, clearanceSources, task } = status;
  return { ready, busy, paused, fallen, loco, mode, seq, time, position: [status.position[0], status.position[1], status.position[2]], headingRad, posture, clearance, spatialValid, guardReason, guardSeq, autonomyActive, availableActions, selectedDuckId, ball, companion, clearanceSources, task, followEnabled: status.park.follow && status.park.leaderId === selectedDuckId };
}
const SCENES: { id: Scene; label: string }[] = [
  { id: "studio", label: "Neutral" },
  { id: "moon", label: "Low light" },
  { id: "sunset", label: "Warm" },
];
const VIEWS: { id: View; label: string; icon: typeof Eye }[] = [{ id: "orbit", label: "Orbit", icon: Orbit }, { id: "follow", label: "Track", icon: Camera }, { id: "eyes", label: "Head camera", icon: Eye }];
const EXAMPLES = [
  { title: "Head motion", prompt: "Look left, look right, then center the head." },
  { title: "Posture transition", prompt: "Sit down, then walk forward." },
  { title: "Locomotion switch", prompt: "Use rollers, then crouch, then use legs, then quack." },
  { title: "Ball interaction", prompt: "Place a ball, then find it and kick it." },
];
const TASK_PHASES: Record<string, string> = { standing: "Standing prerequisite", searching: "Searching for a route", approaching: "Approaching ball", aligning: "Aligning foot", settling: "Stabilizing stance", kicking: "Executing kick", verifying: "Measuring ball response", complete: "Task completed", failed: "Task failed", cancelled: "Task cancelled" };
const safeAction = (value: unknown): value is Action => typeof value === "string" && SIMULATOR_ACTIONS.includes(value as SimulatorAction) && value !== "none" && value !== "clarify";
const sceneName = (id: Scene) => SCENES.find(scene => scene.id === id)!.label;
const reasonText: Record<SimulatorDecision["interpretation"]["reason"], string> = {
  accepted: "Jev found a supported request. The controller checks the robot before each move.",
  no_request: "This reads as a description rather than a request to act.",
  unsupported_request: "Requests go to the selected duck; select another agent in World before issuing its commands. Open Actions for supported commands. Exact distances and angles are not supported.",
  low_confidence: "Jev wasn't sure enough about part of this request. Try naming each move directly.",
  inconsistent_plan: "The parts of this request did not form a clear sequence. Try separating moves with ‘then’.",
  too_many_steps: "The sequence exceeds the four-action limit.",
};

export default function SimulatorApp() {
  const frame = useRef<HTMLIFrameElement>(null);
  const app = useRef<HTMLElement>(null);
  const modal = useRef<HTMLElement>(null);
  const directorPanel = useRef<HTMLElement>(null);
  const runner = useRef<MissionRunner | null>(null);
  const autonomy = useRef<AutonomyController | null>(null);
  const swarm = useRef<SwarmController | null>(null);
  const statusRef = useRef(INITIAL);
  const abort = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const resetRequests = useRef(new Set<string>());
  const selectionRequest = useRef<DuckId | null>(null);
  const lastSnapshot = useRef("");
  const lastPosition = useRef<{ p: number[]; time: number } | null>(null);
  const [status, setStatus] = useState(INITIAL);
  const [snapshot, setSnapshot] = useState<MissionSnapshot | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [apiReady, setApiReady] = useState(false);
  const [decision, setDecision] = useState<SimulatorDecision | null>(null);
  const [notice, setNotice] = useState("Initializing simulation.");
  const [error, setError] = useState("");
  const [scene, setScene] = useState<Scene>("studio");
  const [camera, setCamera] = useState<View>("orbit");
  const presentationRef = useRef({ scene, camera });
  presentationRef.current = { scene, camera };
  const [tab, setTab] = useState<"simulations" | "director" | "park" | "moves" | "routines">("simulations");
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [telemetryOpen, setTelemetryOpen] = useState(false);
  const [selectionPending, setSelectionPending] = useState(false);
  const [focus, setFocus] = useState(false);
  const [dialog, setDialog] = useState<"about" | "save" | null>(null);
  const [routineName, setRoutineName] = useState("");
  const [routines, setRoutines] = useState<Routine[]>([]);
  const [history, setHistory] = useState<LogEntry[]>([]);
  const [trail, setTrail] = useState<number[][]>([]);
  const [distance, setDistance] = useState(0);
  const [reload, setReload] = useState(0);
  const [paused, setPaused] = useState(false);
  const [autoState, setAutoState] = useState(INITIAL_AUTONOMY);
  const [autoMode, setAutoMode] = useState<AutonomyMode>("explore");
  const [swarmState, setSwarmState] = useState(INITIAL_SWARM);
  const [swarmScenario, setSwarmScenario] = useState<SwarmScenario>("flock");

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("jevduck.panels.v1") || "null");
      if (saved && typeof saved === "object") {
        if (typeof saved.inspector === "boolean") setInspectorOpen(saved.inspector);
        if (typeof saved.console === "boolean") setConsoleOpen(saved.console);
        if (typeof saved.telemetry === "boolean") setTelemetryOpen(saved.telemetry);
      } else if (window.matchMedia("(max-width: 700px)").matches) setInspectorOpen(false);
    } catch {}
  }, []);
  function panelPreference(panel: "inspector" | "console" | "telemetry", value: boolean) {
    const next = { inspector: inspectorOpen, console: consoleOpen, telemetry: telemetryOpen, [panel]: value };
    setInspectorOpen(next.inspector); setConsoleOpen(next.console); setTelemetryOpen(next.telemetry);
    try { localStorage.setItem("jevduck.panels.v1", JSON.stringify(next)); } catch {}
  }

  const post = useCallback((payload: object) => frame.current?.contentWindow?.postMessage({ channel: CHANNEL, ...payload }, location.origin), []);
  const log = useCallback((title: string, detail: string, ok = true) => setHistory(items => [{ id: crypto.randomUUID(), title, detail, ok }, ...items].slice(0, 8)), []);
  const cancel = useCallback((reason = "Manual control activated.", stopMotion = true) => {
    generation.current += 1; abort.current?.abort(); setLoading(false); setError("");
    autonomy.current?.stop(reason, stopMotion);
    swarm.current?.stop(reason);
    runner.current?.cancel(reason);
    if (stopMotion && statusRef.current.ready) post({ type: "command", id: crypto.randomUUID(), action: "stop" });
  }, [post]);
  const stop = useCallback(() => { cancel("Sequence stopped."); post({ type: "command", id: crypto.randomUUID(), action: "stop", all: true }); setNotice("All agents stopped. Balance controllers remain active."); }, [cancel, post]);

  useEffect(() => {
    const engine = new MissionRunner({
      dispatch: (action, id) => post({ type: "command", id, action }),
      onChange: next => {
        setSnapshot(next);
        autonomy.current?.updateMission(next);
        const key = `${next.id}:${next.status}`;
        if (key === lastSnapshot.current) return;
        lastSnapshot.current = key;
        if (next.status === "completed") { setNotice("Sequence completed."); log("Sequence completed", next.steps.map(step => SIMULATOR_ACTION_LABELS[step.action]).join(" → ")); }
        if (next.status === "failed") { const reason = next.reason || "The robot could not complete this sequence."; setError(reason); setNotice("Sequence failed. Inspect the state before retrying or resetting."); log("Sequence interrupted", reason, false); post({ type: "command", id: crypto.randomUUID(), action: "stop" }); }
      },
    });
    runner.current = engine;
    const pilot = new AutonomyController({
      request: async (input, signal) => {
        const response = await fetch("/api/autonomy", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input), signal });
        const value: unknown = await response.json();
        if (!response.ok) throw new Error(typeof value === "object" && value !== null && "error" in value && typeof value.error === "string" ? value.error : "Jev could not choose the next behavior.");
        const parsed = autonomyDecisionSchema.safeParse(value);
        if (!parsed.success) throw new Error("Jev returned an invalid behavior. Autonomy has stopped.");
        setApiReady(true);
        return parsed.data;
      },
      execute: plan => { setDecision(null); setError(""); return engine.start(plan); },
      cancelMission: reason => { engine.cancel(reason); },
      stopMotion: () => post({ type: "command", id: crypto.randomUUID(), action: "stop" }),
      setRuntimeAutonomy: active => post({ type: "autonomy", active }),
      onChange: next => { setAutoState(next); if (next.phase === "error") setError(next.reason || "Autonomy has stopped. The manual controls are ready."); },
    });
    autonomy.current = pilot;
    setAutoState(pilot.snapshot);
    const group = new SwarmController({
      request: async (input, signal) => {
        const response = await fetch("/api/swarm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input), signal });
        const value: unknown = await response.json();
        if (!response.ok) throw new Error(typeof value === "object" && value !== null && "error" in value && typeof value.error === "string" ? value.error : "Jev could not choose a group instruction.");
        const parsed = swarmDecisionSchema.safeParse(value);
        if (!parsed.success) throw new Error("Jev returned an invalid group instruction.");
        setApiReady(true); return parsed.data;
      },
      activate: (scenario, runId) => post({ type: "swarm", active: true, runId, scenario }),
      dispatch: (intent, id, runId) => post({ type: "swarm-intent", runId, id, intent }),
      deactivate: runId => post({ type: "swarm", active: false, runId }),
      onChange: next => { setSwarmState(next); if (next.phase === "error") setError(next.reason); },
    });
    swarm.current = group; setSwarmState(group.snapshot);
    const listen = (event: MessageEvent) => {
      if (event.origin !== location.origin || event.source !== frame.current?.contentWindow || event.data?.channel !== CHANNEL) return;
      const data = event.data;
      if (data.type === "swarm-result" || data.type === "swarm-intent-result") {
        if (typeof data.runId === "string" && typeof data.accepted === "boolean") {
          const reason = typeof data.message === "string" ? data.message : "";
          if (data.type === "swarm-result") group.acknowledgeRun(data.runId, data.accepted, reason);
          else if (typeof data.id === "string") group.acknowledgeIntent(data.runId, data.id, data.accepted, reason);
        }
        return;
      }
      if (data.type === "park-result" || data.type === "selection-result") {
        if (data.accepted === false) {
          setError(typeof data.message === "string" ? data.message : "That change could not be made yet.");
          if (data.type === "selection-result") { selectionRequest.current = null; setSelectionPending(false); }
        } else if (typeof data.message === "string") setNotice(data.message);
        return;
      }
      if (data.type === "manual" || data.type === "stop") {
        cancel(data.type === "stop" ? "Sequence stopped." : "Manual control activated.", data.type === "stop");
        setNotice(data.type === "stop" ? "Motion stopped. Balance controller active." : "Manual control active. Sequence cancelled.");
      }
      if (data.type === "ack" && typeof data.id === "string" && typeof data.accepted === "boolean") {
        engine.acknowledge(data.id, data.accepted, typeof data.message === "string" ? data.message : "", data);
        if (resetRequests.current.delete(data.id)) {
          if (!data.accepted) setError(data.message || "Reset was not accepted.");
          else setNotice("Simulation reset.");
        }
      }
      if (data.type !== "status" || typeof data.ready !== "boolean" || typeof data.mode !== "string" || !["legs", "rollers"].includes(data.loco)) return;
      const finite = (v: unknown, fallback = 0) => typeof v === "number" && Number.isFinite(v) ? v : fallback;
      const vector = (v: unknown) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite) ? v : [0, 0, 0];
      const enrichment = enrichmentSchema.safeParse({ ball: data.ball, companion: data.companion, clearanceSources: data.clearanceSources, task: data.task });
      const groupObservation = swarmRuntimeSchema.safeParse(data.swarm);
      const next: Status = {
        ...(enrichment.success ? enrichment.data : {}),
        ready: data.ready, loco: data.loco, mode: data.mode, busy: data.busy === true, fallen: data.fallen === true, paused: data.paused === true,
        seq: finite(data.seq), time: finite(data.time), position: vector(data.position), command: vector(data.command), tiltRad: finite(data.tiltRad), inferenceCount: finite(data.inferenceCount),
        posture: ["standing", "sitting", "transitioning", "fallen"].includes(data.posture) ? data.posture : data.mode === "sitstand" ? "sitting" : "standing", pendingAction: typeof data.pendingAction === "string" ? data.pendingAction : null,
        phase: typeof data.phase === "string" ? data.phase : "", fps: finite(data.fps), controlHz: finite(data.controlHz), policy: typeof data.policy === "string" ? data.policy : "",
        error: typeof data.error === "string" ? data.error.slice(0, 400) : undefined,
        headingRad: finite(data.headingRad), clearance: { front: Math.max(0, finite(data.clearance?.front)), back: Math.max(0, finite(data.clearance?.back)), left: Math.max(0, finite(data.clearance?.left)), right: Math.max(0, finite(data.clearance?.right)) },
        spatialValid: data.spatialValid === true && Array.isArray(data.position) && data.position.length === 3 && data.position.every(Number.isFinite) && [data.headingRad, data.clearance?.front, data.clearance?.back, data.clearance?.left, data.clearance?.right].every(Number.isFinite), autonomyActive: data.autonomyActive === true,
        guardReason: typeof data.guardReason === "string" ? data.guardReason.slice(0, 160) : null, guardSeq: finite(data.guardSeq),
        manual: data.manual === true,
        selectedDuckId: isDuckId(data.selectedDuckId) ? data.selectedDuckId : "duck1",
        availableActions: Array.isArray(data.availableActions) ? data.availableActions.filter(safeAction) : undefined,
        ducks: Array.isArray(data.ducks) ? data.ducks.filter((duck: ParkDuck) => duck && isDuckId(duck.id)).slice(0, 4).map((duck: ParkDuck) => ({ id: duck.id, name: typeof duck.name === "string" ? duck.name.slice(0, 30) : duck.id, loco: duck.loco === "rollers" ? "rollers" : "legs", posture: typeof duck.posture === "string" ? duck.posture.slice(0, 32) : "standing", position: vector(duck.position), headingRad: finite(duck.headingRad), busy: duck.busy === true, fallen: duck.fallen === true })) : [],
        park: { follow: data.park?.follow === true, leaderId: isDuckId(data.park?.leaderId) ? data.park.leaderId : "duck1", obstacle: ["center", "left", "right", "off"].includes(data.park?.obstacle) ? data.park.obstacle : "off", followState: typeof data.park?.followState === "string" ? data.park.followState.slice(0, 160) : "" },
        swarm: groupObservation.success ? groupObservation.data : null,
      };
      if (next.selectedDuckId !== statusRef.current.selectedDuckId) {
        if (group.snapshot.phase !== "starting") cancel("Selected agent changed.", false);
        pilot.reset("Selected agent changed."); setTrail([]); setDistance(0); lastPosition.current = null; setSnapshot(null); setDecision(null);
      }
      if (selectionRequest.current === next.selectedDuckId) { selectionRequest.current = null; setSelectionPending(false); }
      if ((!statusRef.current.paused && next.paused) || (statusRef.current.ready && !next.ready) || (next.error && next.error !== statusRef.current.error)) {
        generation.current += 1; abort.current?.abort(); setLoading(false);
      }
      if (next.error && next.error !== statusRef.current.error) setNotice("Simulation stopped. Reload to restore the runtime.");
      if (!statusRef.current.ready && next.ready) {
        post({ type: "presentation", ...presentationRef.current });
        setNotice("Simulation ready.");
      }
      if (SCENES.some(s => s.id === data.scene)) { next.scene = data.scene; setScene(data.scene); }
      if (VIEWS.some(v => v.id === data.camera)) { next.camera = data.camera; setCamera(data.camera); }
      statusRef.current = next; setStatus(next); pilot.updateStatus({ ...autonomyObservation(next), phase: next.phase, error: next.error, manual: next.manual, suspended: document.hidden }); engine.updateStatus(next);
      if (next.swarm) group.updateStatus({ seq: next.seq, time: next.time, ready: next.ready, paused: next.paused, swarm: next.swarm, manual: next.manual, suspended: document.hidden, error: next.error });
      if (next.ready && !next.paused && next.time !== lastPosition.current?.time) {
        const previous = lastPosition.current;
        if (previous && next.time > previous.time) {
          const delta = Math.hypot(next.position[0] - previous.p[0], next.position[1] - previous.p[1]);
          if (delta < 0.3) setDistance(value => value + delta);
        }
        lastPosition.current = { p: next.position, time: next.time };
        setTrail(points => [...points, next.position].slice(-240));
      }
    };
    window.addEventListener("message", listen);
    const control = new AbortController();
    fetch("/api/simulator", { signal: control.signal }).then(r => r.json()).then(value => setApiReady(value.available === true)).catch(() => {});
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE) || "[]");
      if (Array.isArray(saved)) setRoutines(saved.filter(r => typeof r?.id === "string" && typeof r.name === "string" && r.name.length <= 60 && Array.isArray(r.plan) && r.plan.length <= 4 && r.plan.every(safeAction) && ["keep", "studio", "moon", "sunset"].includes(r.scene) && ["keep", "orbit", "follow", "eyes"].includes(r.camera)).slice(0, 8));
    } catch {}
    return () => { window.removeEventListener("message", listen); control.abort(); abort.current?.abort(); group.dispose(); swarm.current = null; pilot.dispose(); autonomy.current = null; engine.cancel("Simulator closed."); runner.current = null; };
  }, [post, log, cancel]);

  useEffect(() => {
    const key = (event: KeyboardEvent) => { if (event.key === "Escape") { setDialog(null); stop(); } };
    window.addEventListener("keydown", key); return () => window.removeEventListener("keydown", key);
  }, [stop]);
  useEffect(() => {
    const hide = () => { cancel("Autonomy stopped: tab inactive."); post({ type: "lifecycle", active: false }); };
    const show = () => post({ type: "lifecycle", active: !paused && !document.hidden });
    const visibility = () => { if (document.hidden) hide(); else show(); };
    window.addEventListener("pagehide", hide); window.addEventListener("pageshow", show);
    document.addEventListener("visibilitychange", visibility);
    show();
    return () => { window.removeEventListener("pagehide", hide); window.removeEventListener("pageshow", show); document.removeEventListener("visibilitychange", visibility); };
  }, [cancel, post, paused]);
  useEffect(() => {
    if (!dialog) return;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    modal.current?.querySelector<HTMLElement>("button, input")?.focus();
    const key = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const items = modal.current?.querySelectorAll<HTMLElement>("button, input, a[href], summary");
      if (!items?.length) return;
      const first = items[0], last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", key);
    return () => { document.removeEventListener("keydown", key); opener?.focus(); };
  }, [dialog]);

  function presentation(next: { scene?: Scene; camera?: View }, fromUser = true) {
    if (fromUser) { generation.current += 1; abort.current?.abort(); setLoading(false); }
    if (next.scene) setScene(next.scene); if (next.camera) setCamera(next.camera);
    post({ type: "presentation", ...next });
  }
  function manual(action: Action) {
    if (selectionRequest.current) return;
    if (action === "stop") { stop(); return; }
    cancel(); setDecision(null); setNotice(SIMULATOR_ACTION_LABELS[action]); runner.current?.start([action]);
  }
  function selectDuck(duckId: DuckId) {
    if (duckId === statusRef.current.selectedDuckId || selectionRequest.current) return;
    cancel("Selected agent changed.");
    selectionRequest.current = duckId; setSelectionPending(true); setDecision(null); setSnapshot(null);
    post({ type: "select-duck", duckId });
  }
  function setPark(next: { follow?: boolean; obstacle?: ObstacleSlot }) {
    if (selectionRequest.current) return;
    if (next.obstacle !== undefined) cancel("The barrier position changed.");
    post({ type: "park", ...next });
  }
  function openPark() {
    setTab("park"); setInspectorOpen(true); setFocus(false);
    if (window.matchMedia("(max-width: 980px)").matches) requestAnimationFrame(() => directorPanel.current?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }));
  }
  function startAutonomy(mode = autoMode) {
    if (!statusRef.current.ready || statusRef.current.paused || paused || statusRef.current.fallen || !statusRef.current.spatialValid) return;
    cancel("Autonomous controller started."); setDecision(null); setSnapshot(null); setAutoMode(mode); setTab("director"); setInspectorOpen(true);
    autonomy.current?.start(mode);
  }
  function toggleAutonomy() { if (autoState.active) stop(); else startAutonomy(); }
  function startSwarm(scenario: SwarmScenario) {
    if (!statusRef.current.ready || statusRef.current.paused || paused || !apiReady || swarm.current?.snapshot.phase === "starting") return;
    cancel("Swarm scenario selected."); setSwarmScenario(scenario); setDecision(null); setSnapshot(null); setError("");
    setTrail([]); setDistance(0); lastPosition.current = null;
    setTab("simulations"); setInspectorOpen(true); setFocus(false); setConsoleOpen(false); setTelemetryOpen(false);
    presentation({ camera: "orbit" }); setNotice("Preparing four-robot simulation.");
    swarm.current?.start(scenario);
  }
  function changeAutonomyMode(mode: AutonomyMode) { setAutoMode(mode); if (autoState.active) startAutonomy(mode); }
  function reset() {
    cancel("The world was reset.");
    autonomy.current?.reset();
    swarm.current?.reset();
    const id = crypto.randomUUID(); resetRequests.current.add(id); post({ type: "command", id, action: "reset" });
    setTrail([]); setDistance(0); lastPosition.current = null; setSnapshot(null); setDecision(null);
  }
  function togglePause() {
    cancel("Physics was paused."); const next = !paused; setPaused(next); post({ type: "lifecycle", active: !next });
    setNotice(next ? "Physics paused. Commands cancelled." : "Physics resumed. Controller remains stopped.");
  }
  function reboot() {
    cancel("Simulator reloading."); autonomy.current?.reset(); swarm.current?.reset(); runner.current?.resetStatus(); statusRef.current = INITIAL; setStatus(INITIAL); setPaused(false); setReload(n => n + 1); setError("");
  }
  async function ask(text: string) {
    const trimmed = text.trim(); if (!trimmed || !statusRef.current.ready || statusRef.current.paused || selectionRequest.current) return;
    cancel("Command sequence selected."); const current = ++generation.current;
    const controller = new AbortController(); abort.current = controller;
    setLoading(true); setDecision(null); setSnapshot(null); setMessage(trimmed); setTab("director"); setInspectorOpen(true); setNotice("Interpreting command sequence.");
    const { ready, loco, mode, busy, fallen, paused: simPaused, selectedDuckId, availableActions } = statusRef.current;
    try {
      const response = await fetch("/api/simulator", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: trimmed, context: { ready, loco, mode, busy, fallen, paused: simPaused, selectedDuckId, availableActions } }), signal: controller.signal });
      const value = await response.json(); if (generation.current !== current) return;
      if (!response.ok) throw new Error(value.error || "Jev could not finish this request.");
      const result = value as SimulatorDecision;
      if (result.source !== "jev" || !Array.isArray(result.plan) || result.plan.length > 4 || !result.plan.every(safeAction)) throw new Error("Jev returned an invalid action sequence.");
      setDecision(result); setApiReady(true);
      if (result.disposition === "clarify") { setNotice("Command requires clarification."); setError(reasonText[result.interpretation.reason] || "Specify up to four supported actions, with optional display settings."); return; }
      if (result.disposition === "none") { setNotice("No executable action requested."); return; }
      const settings: { scene?: Scene; camera?: View } = {};
      if (result.scene !== "keep") settings.scene = result.scene;
      if (result.camera !== "keep") settings.camera = result.camera;
      if (Object.keys(settings).length) presentation(settings, false);
      if (result.plan.length) { setNotice("Executing sequence."); runner.current?.start(result.plan); }
      else { setNotice("Display configuration updated."); log("World updated", [settings.scene && sceneName(settings.scene), settings.camera && `${settings.camera} camera`].filter(Boolean).join(" · ")); }
      setMessage(value => value.trim() === trimmed ? "" : value);
    } catch (err) {
      if (generation.current === current && !controller.signal.aborted) { setError(err instanceof Error ? err.message : "This request was interrupted."); setNotice("Manual controls available."); }
    } finally { if (generation.current === current) setLoading(false); }
  }
  function storeRoutines(next: Routine[]) { setRoutines(next); try { localStorage.setItem(STORAGE, JSON.stringify(next)); } catch { setError("Browser storage failed. This sequence remains available until the page closes."); } }
  function saveRoutine() {
    if (!decision || decision.disposition !== "execute" || !routineName.trim()) return;
    const next: Routine = { id: crypto.randomUUID(), name: routineName.trim().slice(0, 42), plan: decision.plan, scene: decision.scene, camera: decision.camera };
    storeRoutines([next, ...routines].slice(0, 8)); setDialog(null); setTab("routines"); setNotice("Sequence saved in this browser.");
  }
  function playRoutine(routine: Routine) {
    cancel("Saved sequence selected."); setDecision(null); setNotice(`Executing ${routine.name}.`);
    presentation({ ...(routine.scene !== "keep" ? { scene: routine.scene } : {}), ...(routine.camera !== "keep" ? { camera: routine.camera } : {}) });
    if (routine.plan.length) runner.current?.start(routine.plan); else log(routine.name, "Saved world settings applied.");
  }
  const disabled = !status.ready || status.paused || paused || selectionPending;
  const duckName = status.ducks.find(duck => duck.id === status.selectedDuckId)?.name || "Microduck";
  const autoDisabled = disabled || status.fallen || !status.spatialValid || !apiReady;
  const autoThinking = autoState.active && autoState.phase === "choosing";
  const autoDecision = autoState.currentDecision || autoState.lastDecision;
  const running = snapshot?.status === "running" || snapshot?.status === "queued";
  const stateLabel = status.error ? "Error" : !status.ready ? "Initializing" : paused || status.paused ? "Paused" : status.pendingAction ? "Standing prerequisite" : status.fallen ? "Fallen" : status.posture === "transitioning" ? "Posture transition" : status.busy ? "Executing" : status.posture === "sitting" ? "Seated" : "Ready";
  const steps = snapshot?.steps || [];
  const runningTask = status.task?.outcome === null ? status.task : null;
  const toMap = (point: number[]) => `${50 + point[0] / 3 * 88},${50 - point[1] / 3 * 88}`;
  const agentId = status.selectedDuckId.replace("duck", "").padStart(2, "0");
  const measured = (value: number, digits = 2) => status.ready ? value.toFixed(digits) : "n/a";

  return <main ref={app} className={`jevduck immersive scene-${scene} ${focus ? "is-focused" : ""} ${inspectorOpen ? "inspector-open" : "inspector-closed"} ${consoleOpen ? "console-open" : "console-closed"}`}>
    <header className="topbar">
      <a className="wordmark" href="/" aria-label="Jevduck home"><Activity size={19} /><span>Jevduck</span></a>
      <h1>Microduck simulator</h1>
      <button className="simulations-toggle" onClick={() => { setTab("simulations"); setInspectorOpen(true); setFocus(false); }}><Network size={14} /><span>Simulations</span></button>
      <div className="world-status"><span className={`live-dot ${status.ready ? "on" : ""}`} />{stateLabel}<span className="status-divider" /><span>{swarmState.active ? "Jev group control" : autoState.active ? "Autonomous control" : running ? "Sequence control" : "Direct control"}</span></div>
      <div className="top-actions"><button className="icon-button" aria-label={inspectorOpen ? "Collapse inspector" : "Expand inspector"} aria-expanded={inspectorOpen && !focus} aria-controls="simulation-inspector" onClick={() => { setFocus(false); panelPreference("inspector", !inspectorOpen); }}>{inspectorOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}</button><button className={`icon-button ${focus ? "selected" : ""}`} title={focus ? "Show controls" : "Expand viewport"} aria-label={focus ? "Show controls" : "Expand viewport"} onClick={() => setFocus(!focus)}><Focus size={16} /></button><button className="icon-button" aria-label="Fullscreen simulator" onClick={() => { if (document.fullscreenElement) void document.exitFullscreen(); else void app.current?.requestFullscreen().catch(() => setError("Fullscreen is unavailable in this browser.")); }}><Maximize2 size={16} /></button><button className="icon-button" aria-label="Model and session information" onClick={() => setDialog("about")}><CircleHelp size={16} /></button></div>
    </header>

    <div className="workspace">
      <section className="simulation-column" aria-label="Simulation workspace">
        <div className="viewport-toolbar hide-in-focus">
          <span className="viewport-title">3D viewport</span>
          <div className="camera-switcher" role="group" aria-label="Camera views">{VIEWS.map(view => <button key={view.id} className={camera === view.id ? "selected" : ""} aria-pressed={camera === view.id} onClick={() => presentation({ camera: view.id })}><view.icon size={13} />{view.label}</button>)}</div>
          <label className="lighting-control">Lighting<select aria-label="Lighting preset" value={scene} onChange={event => presentation({ scene: event.target.value as Scene })}>{SCENES.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
        </div>

        <div className="world-viewport">
          <iframe key={reload} ref={frame} title="Microduck MuJoCo simulation" src="/simulator/index.html?embed=1&boot=1" allow="autoplay; fullscreen; gamepad" />
          {status.ready && <><div className="viewport-identity"><span className={`agent-dot ${status.selectedDuckId}`} />Agent {agentId}<span>{duckName} / {status.loco}</span></div><div className="viewport-caption">World frame · metres<span>MuJoCo / ONNX</span></div></>}
          {status.swarm?.scenario && <div className="swarm-viewport-label"><Network size={13} /><div><strong>{SWARM_SCENES.find(item => item.id === status.swarm?.scenario)?.name}</strong><span>{status.ducks.length} physical agents · {swarmState.active ? "Jev group control" : "Controller stopped"}</span></div></div>}
          {!status.ready && <div className="boot-screen"><Activity size={26} /><h2>{status.error ? status.inferenceCount > 0 ? "Simulation stopped" : "Simulation initialization failed" : "Initializing simulation"}</h2><p>{status.error || "Loading the robot models and their ONNX policies."}</p>{status.error ? <button className="primary-button" onClick={reboot}>Reload simulator</button> : <div className="boot-progress"><span /></div>}</div>}
          {(paused || status.paused) && status.ready && <button className="paused-world" onClick={togglePause}><Play size={21} /><strong>Physics paused</strong><span>Resume simulation</span></button>}
        </div>

        <section className={`telemetry-panel hide-in-focus ${telemetryOpen ? "expanded" : "collapsed"}`} aria-label="Selected agent telemetry">
          <button className="panel-disclosure" aria-expanded={telemetryOpen} aria-controls="telemetry-body" onClick={() => panelPreference("telemetry", !telemetryOpen)}><Activity size={13} /><span>Agent {agentId} telemetry</span><small>{measured(status.time, 1)} s</small><ChevronDown size={13} /></button>
          <div id="telemetry-body" hidden={!telemetryOpen}><div className="telemetry-content">
            <svg className="trail-map" viewBox="0 0 100 100" role="img" aria-label="Selected agent XY trajectory in the arena"><defs><pattern id="map-grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="currentColor" strokeWidth=".4" /></pattern></defs><rect x="6" y="6" width="88" height="88" fill="url(#map-grid)" stroke="currentColor" strokeWidth=".5" /><polyline points={trail.map(toMap).join(" ")} fill="none" stroke="#7da7d7" strokeWidth="1.3" />{status.ready && <circle cx={50 + status.position[0] / 3 * 88} cy={50 - status.position[1] / 3 * 88} r="2.3" fill="#dcbf76" />}<text x="85" y="98">X</text><text x="1" y="10">Y</text></svg>
            <dl className="telemetry-grid">
              <div className="position-reading"><dt>Position x / y / z <span>m</span></dt><dd>{status.ready ? status.position.map(value => value.toFixed(2)).join(" / ") : "n/a"}</dd></div>
              <div><dt>Heading <span>deg</span></dt><dd>{measured(status.headingRad * 180 / Math.PI, 1)}</dd></div>
              <div><dt>Body tilt <span>deg</span></dt><dd>{measured(status.tiltRad * 180 / Math.PI, 1)}</dd></div>
              <div><dt>Sampled path <span>m</span></dt><dd>{measured(distance)}</dd></div>
              <div><dt>Simulation time <span>s</span></dt><dd>{measured(status.time, 1)}</dd></div>
              <div><dt>Render rate <span>fps</span></dt><dd>{measured(status.fps, 0)}</dd></div>
              <div><dt>Control rate <span>Hz</span></dt><dd>{measured(status.controlHz, 0)}</dd></div>
              <div><dt>Policy evaluations</dt><dd>{status.ready ? status.inferenceCount.toLocaleString() : "n/a"}</dd></div>
              <div><dt>Posture</dt><dd className="text-reading">{status.ready ? status.posture : "n/a"}</dd></div>
            </dl>
          </div>
          <div className="policy-reading"><span>Active policy</span><code title={status.policy}>{status.policy || "n/a"}</code><span>Status seq. {status.seq}</span></div>
          </div>
        </section>

        <section className="command-dock hide-in-focus" aria-label="Agent commands">
          <div className="console-content" id="command-console" hidden={!consoleOpen}>
          <div className="duck-dock-selector" role="group" aria-label="Command target"><span className="control-label">Target</span>{status.ducks.map(duck => <button key={duck.id} disabled={disabled} aria-pressed={status.selectedDuckId === duck.id} onClick={() => selectDuck(duck.id)}><i className={duck.id} />{duck.id.replace("duck", "").padStart(2, "0")}<span>{duck.name}</span></button>)}<button className="open-park" onClick={openPark} aria-label="Open world configuration">World configuration<ArrowUpRight size={12} /></button></div>
          <form className="mission-composer" onSubmit={event => { event.preventDefault(); void ask(message); }}><Terminal size={16} /><label className="sr-only" htmlFor="mission-message">Jev command</label><textarea id="mission-message" rows={1} maxLength={500} value={message} onChange={event => setMessage(event.target.value)} placeholder="Enter command sequence, e.g. sit down, then walk forward" onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); if (!loading) void ask(message); } }} /><button className="send-button" aria-label="Execute Jev command" disabled={disabled || !message.trim() || loading}>{loading ? <LoaderCircle className="spin" size={15} /> : <Play size={13} />}<span>Execute</span></button></form>
          <div className="dock-notice" role="status">{loading || autoThinking ? <LoaderCircle className="spin" size={12} /> : <span className="status-marker" />}<span title={runningTask?.reason}>{runningTask ? TASK_PHASES[runningTask.phase] || runningTask.phase : autoState.active ? autoThinking ? "Requesting controller decision." : autoState.currentDecision?.label || "Awaiting next observation." : notice}</span>{running && !autoState.active && <button onClick={stop}>Cancel sequence<X size={11} /></button>}</div>
          {error && <div className="error-message" role="alert"><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError("")}><X size={13} /></button></div>}
          <div className="manual-controls" role="group" aria-label="Manual robot controls"><button aria-label="Turn left" title="Left stepping arc" disabled={disabled} onClick={() => manual("turn_left")}><ArrowLeft size={14} /></button><button aria-label="Walk forward" title="Forward motion" disabled={disabled} onClick={() => manual("walk_forward")}><ArrowUp size={14} /></button><button aria-label="Back away" title="Reverse stepping arc" disabled={disabled} onClick={() => manual("walk_backward")}><ArrowDown size={14} /></button><button aria-label="Turn right" title="Right stepping arc" disabled={disabled} onClick={() => manual("turn_right")}><ArrowRight size={14} /></button><span className="control-divider" /><button className="text-control" disabled={disabled} onClick={() => manual("sit")}>Sit</button><button className="text-control" disabled={disabled} onClick={() => manual("stand")}>Stand</button><button className="text-control action-catalog-link" onClick={() => { setTab("moves"); setInspectorOpen(true); }}>All actions</button></div>
          </div>
          <div className="dock-controls"><button className="console-disclosure" aria-expanded={consoleOpen} aria-controls="command-console" onClick={() => panelPreference("console", !consoleOpen)}><Terminal size={14} /><span>Commands</span><ChevronDown size={12} /></button><span className="compact-notice" role="status">{runningTask ? TASK_PHASES[runningTask.phase] : swarmState.active ? (swarmState.currentDecision?.reason || swarmState.reason) : autoState.active ? "Autonomous control" : notice}</span><div className="playback-controls"><button aria-label={paused ? "Resume physics" : "Pause physics"} disabled={!status.ready} onClick={togglePause}>{paused ? <Play size={13} /> : <Pause size={13} />}<span>{paused ? "Resume" : "Pause"}</span></button><button aria-label="Reset simulator" disabled={!status.ready} onClick={reset}><RotateCcw size={13} /><span>Reset</span></button><button className="stop-button" disabled={!status.ready && !swarmState.active} onClick={stop}><Square size={10} fill="currentColor" />Stop all<kbd>esc</kbd></button></div></div>
          {error && !consoleOpen && <div className="error-message" role="alert"><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError("")}><X size={13} /></button></div>}
        </section>
      </section>

      <aside id="simulation-inspector" ref={directorPanel} className="director-panel hide-in-focus" aria-label="Simulation inspector" hidden={!inspectorOpen}>
        <div className="inspector-heading"><span>WORKSPACE</span><button className="icon-button" aria-label="Close inspector" onClick={() => panelPreference("inspector", false)}><PanelRightClose size={15} /></button></div>
        <div className="panel-top"><div className="tab-strip"><button className={tab === "simulations" ? "active" : ""} onClick={() => setTab("simulations")}>Sims</button><button className={tab === "director" ? "active" : ""} onClick={() => setTab("director")}>Control</button><button className={tab === "park" ? "active" : ""} onClick={() => setTab("park")}>World</button><button className={tab === "moves" ? "active" : ""} onClick={() => setTab("moves")}>Actions</button><button className={tab === "routines" ? "active" : ""} onClick={() => setTab("routines")}>Saved</button></div></div>
        <div className="panel-scroll">
          {tab === "simulations" ? <SwarmPanel selected={swarmScenario} runtime={swarmState.active && status.swarm?.runId !== swarmState.runId ? null : status.swarm} active={swarmState.active} phase={swarmState.phase} reason={swarmState.reason} decision={swarmState.currentDecision || swarmState.lastDecision} recent={swarmState.recent} cycles={swarmState.completedCycles} disabled={disabled || !apiReady || swarmState.phase === "starting"} onRun={startSwarm} onStop={stop} /> : tab === "director" ? <>
            <AutonomyPanel active={autoState.active} thinking={autoThinking} current={autoState.currentDecision !== null} disabled={autoDisabled} mode={autoMode} reason={autoState.reason || (autoState.active ? autoState.currentDecision?.reason || "Reading simulator state." : "")} completed={autoState.completedCycles} visited={autoState.visited.length} front={status.spatialValid ? status.clearance.front : null} back={status.spatialValid ? status.clearance.back : null} decision={autoDecision} recent={autoState.recent.map(episode => ({ ...episode, label: AUTONOMY_LABELS[episode.behavior] }))} onToggle={toggleAutonomy} onMode={changeAutonomyMode} />
            <section className="ball-task-evidence" aria-label="Ball interaction state">
              <div className="section-label">Ball state <span>Simulator geometry</span></div>
              <dl><div><dt>Range</dt><dd>{!status.ready || !status.ball ? "Unavailable" : status.ball.present ? `${status.ball.distanceM.toFixed(2)} m` : "No ball present"}</dd></div><div><dt>Bearing</dt><dd>{status.ready && status.ball?.present ? `${(status.ball.bearingRad * 180 / Math.PI).toFixed(1)}°` : "n/a"}</dd></div></dl>
              <div className="ball-task-controls"><button disabled={disabled || !status.availableActions?.includes("spawn_ball")} onClick={() => manual("spawn_ball")}>Place ball</button><button disabled={disabled || !status.availableActions?.includes("kick_ball")} onClick={() => manual("kick_ball")}>Approach &amp; kick</button></div>
              {status.task && <div className={`ball-task-progress task-${status.task.outcome || "running"}`} role="status"><strong>{TASK_PHASES[status.task.phase] || status.task.phase}</strong><p>{status.task.reason}</p><small>{status.task.elapsedS.toFixed(1)} s elapsed · {status.task.ballContact ? "Ball contact measured" : "No ball contact measured"}</small>{status.task.ballContact && <small>{status.task.ballDisplacementM.toFixed(2)} m ball displacement</small>}</div>}
            </section>
            <div className="section-label mission-label">Execution sequence <span>{steps.length ? `${steps.filter(step => step.status === "completed").length} / ${steps.length}` : "0 ACTIONS"}</span></div>
            {loading ? <div className="thinking-state"><LoaderCircle className="spin" size={16} /><span>Interpreting command with Jev.</span></div> : steps.length ? <ol className="mission-steps">{steps.map((step, index) => <li key={index} className={`step-${step.status}`}><span className="step-number">{step.status === "completed" ? <Check size={12} /> : index + 1}</span><div><strong>{SIMULATOR_ACTION_LABELS[step.action]}</strong>{step.status === "running" && runningTask && runningTask.commandId === step.id && <small className="prerequisite">{TASK_PHASES[runningTask.phase] || runningTask.phase}</small>}{step.status === "running" && status.pendingAction && <small className="prerequisite"><MoveUpRight size={11} />Waiting for standing stability</small>}{step.status === "failed" && <small>{step.detail}</small>}</div><span className="step-state">{step.status}</span></li>)}</ol> : <p className="mission-empty">No active sequence. Enter a command or select an action.</p>}
            {decision?.disposition === "execute" && <button className="save-routine" onClick={() => { setRoutineName(""); setDialog("save"); }}><Bookmark size={13} />Save sequence</button>}
            {decision && <details className="jev-lens"><summary>Command interpretation<ChevronDown size={12} /></summary><p>{reasonText[decision.interpretation.reason]}</p><div className="lens-rows">{decision.interpretation.steps.filter(step => step.action !== "none").map((step, index) => <div key={index}><span>{index + 1}. {SIMULATOR_ACTION_LABELS[step.action as SimulatorAction] || step.action}</span><span>{Math.round(step.confidence * 100)}%</span></div>)}</div><small>{decision.model} · {(decision.latencyMs / 1000).toFixed(2)} s<br />Interpretation scores do not measure execution success.</small></details>}
            {!autoState.active && <details className="protocol-examples"><summary>Example command sequences<ChevronDown size={12} /></summary><div>{EXAMPLES.map(example => <button key={example.title} disabled={disabled || loading} onClick={() => void ask(example.prompt)}><strong>{example.title}</strong><span>{example.prompt}</span></button>)}</div></details>}
          </> : tab === "park" ? <ParkPanel ducks={status.ducks} selected={status.selectedDuckId} park={status.park} disabled={disabled} pending={selectionPending} onSelect={selectDuck} onFollow={follow => setPark({ follow })} onObstacle={obstacle => setPark({ obstacle })} /> : tab === "moves" ? <MovesPanel name={duckName} available={status.availableActions || []} disabled={disabled} onAction={manual} /> : <>
            <div className="section-label">Saved sequences <span>{routines.length} / 8</span></div>
            <p className="routines-intro">Command sequences and display settings stored in this browser.</p>
            {routines.length ? <div className="routine-list">{routines.map(routine => <article key={routine.id}><div className="routine-title"><Bookmark size={14} /><h3>{routine.name}</h3><button className="icon-button" aria-label={`Delete ${routine.name}`} onClick={() => storeRoutines(routines.filter(r => r.id !== routine.id))}><Trash2 size={13} /></button></div><p>{routine.plan.map(action => SIMULATOR_ACTION_LABELS[action]).join(" → ") || "Display settings only"}</p><button className="routine-play" disabled={disabled} onClick={() => playRoutine(routine)}><Play size={12} />Execute<span>{routine.plan.length} actions</span></button></article>)}</div> : <div className="empty-routines"><p>No saved sequences.</p><p>Execute a Jev command, then save the interpreted sequence.</p><button onClick={() => setTab("director")}>Open controller<ArrowUpRight size={12} /></button></div>}
          </>}
        </div>
        <div className="panel-bottom"><span className={`jev-indicator ${apiReady ? "connected" : ""}`}><i />Jev {apiReady ? "connected" : "unavailable"}</span><span>{status.ducks.length} agents</span></div>
      </aside>
    </div>

    <button className={`focus-stop ${focus ? "visible" : ""}`} onClick={stop}><Square size={11} fill="currentColor" />Stop all<kbd>esc</kbd></button>
    <footer className="world-footer hide-in-focus"><span>MuJoCo WASM<span className="footer-divider">/</span>Original Microduck policies</span><button onClick={() => setDialog("about")}>Model &amp; session information<ArrowUpRight size={11} /></button></footer>

    {dialog && <div className="modal-backdrop" onClick={() => setDialog(null)}><section ref={modal} className="dialog" role="dialog" aria-modal="true" aria-labelledby="dialog-title" onClick={event => event.stopPropagation()}><button className="icon-button dialog-close" aria-label="Close dialog" onClick={() => setDialog(null)}><X size={18} /></button>{dialog === "save" ? <><h2 id="dialog-title">Save sequence</h2><p>Store the interpreted actions and display settings in this browser.</p><form onSubmit={event => { event.preventDefault(); saveRoutine(); }}><label htmlFor="routine-name">Sequence name</label><input id="routine-name" value={routineName} onChange={event => setRoutineName(event.target.value)} maxLength={42} placeholder="Posture transition test" /><button className="primary-button" disabled={!routineName.trim()}>Save sequence<Bookmark size={14} /></button></form></> : <><h2 id="dialog-title">Model &amp; session information</h2><dl className="model-facts"><div><dt>Simulator</dt><dd>MuJoCo WebAssembly</dd></div><div><dt>Control</dt><dd>Original Microduck ONNX policies</dd></div><div><dt>World</dt><dd>{status.ducks.length || 2} articulated agents, shared contacts</dd></div><div><dt>Upstream revision</dt><dd><code>023172c8a7d629b5258d90364c13bafe013abbfa</code></dd></div></dl><p>Jev selects supported actions from simulator state. Local controllers enforce action availability and posture prerequisites. Commanded motion is not a calibrated distance or angle.</p><p>Lighting presets affect rendering only. The head camera is a rendered simulator view. This application does not control physical hardware.</p><p>Jev receives command text and simulator context through the application server. Saved sequences remain in this browser.</p><div className="source-links"><a href="https://huggingface.co/spaces/pollen-robotics/microduck-simulator/tree/023172c8a7d629b5258d90364c13bafe013abbfa" target="_blank" rel="noreferrer">Simulator source<ArrowUpRight size={12} /></a><a href="https://docs.typesafe.ai/introduction" target="_blank" rel="noreferrer">Jev documentation<ArrowUpRight size={12} /></a></div>{history.length > 0 && <details className="session-log"><summary>Session events<ChevronDown size={12} /></summary>{history.map(item => <div key={item.id}><strong>{item.title}</strong><span>{item.detail}</span></div>)}</details>}</>}</section></div>}
  </main>;
}

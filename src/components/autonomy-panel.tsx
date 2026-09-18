import { ChevronDown, Compass, Eye, LoaderCircle, Play, Square, Target } from "lucide-react";
import type { AutonomyMode } from "@/lib/autonomy-contract";

type Props = {
  active: boolean;
  thinking: boolean;
  current: boolean;
  disabled: boolean;
  mode: AutonomyMode;
  reason: string;
  completed: number;
  visited: number;
  front: number | null;
  back: number | null;
  decision: { label: string; model: string; confidence: number; latencyMs: number } | null;
  recent: { label: string; outcome: string; distanceM: number; ballContact?: boolean; ballDisplacementM?: number }[];
  onToggle(): void;
  onMode(mode: AutonomyMode): void;
};

export function AutonomyPanel(props: Props) {
  return <section className={`autonomy-card ${props.active ? "is-live" : ""}`} aria-label="Autonomous controller">
    <div className="section-label">CONTROLLER <span>{props.active ? "ACTIVE" : "INACTIVE"}</span></div>
    <h2>Autonomous controller</h2>
    <p className="autonomy-intro">Jev selects bounded actions from simulator state and retained outcomes.</p>
    <div className="autonomy-modes" role="group" aria-label="Controller objective">
      <button aria-pressed={props.mode === "explore"} onClick={() => props.onMode("explore")}><Compass size={14} /><span>Explore<small>Prefer unvisited floor cells</small></span></button>
      <button aria-pressed={props.mode === "observe"} onClick={() => props.onMode("observe")}><Eye size={14} /><span>Observe<small>No locomotion commands</small></span></button>
      <button className="autonomy-play-mode" aria-pressed={props.mode === "play"} onClick={() => props.onMode("play")}><Target size={14} /><span>Ball interaction<small>Approach and verify a kick</small></span></button>
    </div>
    <button className="autonomy-start" disabled={!props.active && props.disabled} onClick={props.onToggle}>
      {props.active ? <Square size={11} fill="currentColor" /> : <Play size={12} fill="currentColor" />}
      {props.active ? "Stop controller" : "Start controller"}
      <span>{props.active ? "RUNNING" : "Jev"}</span>
    </button>
    {props.active && <div className="autonomy-thought" role="status">
      {props.thinking ? <LoaderCircle className="spin" size={13} /> : <span className="live-dot on" />}
      <div><strong>{props.thinking ? "Requesting decision" : props.current ? props.decision?.label || "Executing action" : "Between control cycles"}</strong><p>{props.reason}</p></div>
    </div>}
    {!props.active && props.reason && <p className="autonomy-stopped">{props.reason}</p>}
    {(props.active || props.completed > 0) && <div className="autonomy-metrics"><span><b>{props.completed}</b> completed cycles</span><span><b>{props.visited}</b> retained visited cells</span></div>}
    <div className="autonomy-evidence" aria-label="Decision evidence">
      <div className="section-label">DECISION EVIDENCE</div>
      <dl>
        <div><dt>Forward clearance</dt><dd>{props.front === null ? "Unavailable" : `${props.front.toFixed(2)} m`}</dd></div>
        <div><dt>Rear clearance</dt><dd>{props.back === null ? "Unavailable" : `${props.back.toFixed(2)} m`}</dd></div>
        <div><dt>{props.current ? "Choice score" : "Last choice score"}</dt><dd>{props.decision ? `${Math.round(props.decision.confidence * 100)}%` : "Unavailable"}</dd></div>
        <div><dt>Jev request latency</dt><dd>{props.decision ? `${props.decision.latencyMs} ms` : "Unavailable"}</dd></div>
      </dl>
      {props.decision && <small>Model: {props.decision.model}. Choice scores do not measure execution success.</small>}
      <p>Clearance and action availability come from the simulator. The local controller rechecks motion before execution.</p>
      {props.recent.length > 0 && <details className="autonomy-history"><summary>Recent outcomes<ChevronDown size={12} /></summary><ol aria-label="Recent autonomous outcomes">{props.recent.slice(-4).reverse().map((episode, index) => <li key={index}><span>{episode.label}</span><small>{episode.outcome} · {episode.distanceM.toFixed(2)} m robot travel{episode.ballContact && episode.ballDisplacementM !== undefined ? ` · ${episode.ballDisplacementM.toFixed(2)} m ball displacement` : ""}</small></li>)}</ol></details>}
    </div>
    <p className="autonomy-footnote">Active only while this tab is visible. Manual input interrupts the controller.</p>
  </section>;
}

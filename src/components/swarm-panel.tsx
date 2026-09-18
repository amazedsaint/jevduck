import { ArrowUpRight, ChevronDown, LoaderCircle, Network, Play, RotateCcw, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { SWARM_INTENT_LABELS, SWARM_SCENARIO_INTENTS, type SwarmDecision, type SwarmEpisode, type SwarmRuntime, type SwarmScenario } from "@/lib/swarm-contract";

export const SWARM_SCENES: { id: SwarmScenario; name: string; subtitle: string; description: string; measure: string }[] = [
  { id: "flock", name: "Flocking formation", subtitle: "Coordinated motion", description: "Advance together while maintaining space between robots. Jev can regroup when the formation stretches.", measure: "Centroid displacement and formation error" },
  { id: "gather", name: "Aggregation", subtitle: "Gather / disperse", description: "Gather into separate slots, then increase spacing. Jev uses the measured spread to choose the next instruction.", measure: "Group spread and minimum spacing" },
  { id: "convoy", name: "Leader convoy", subtitle: "Follow / hand over", description: "Follow a shared route with a designated leader. Jev can pause the group or choose a leadership change.", measure: "Leader identity and target error" },
  { id: "split", name: "Split & regroup", subtitle: "Two subgroups", description: "Separate into pairs and return to a common formation. Jev chooses when to split and when to regroup.", measure: "Slot error and group spread" },
];

function FormationSketch({ mode }: { mode: SwarmScenario }) {
  const points = mode === "convoy" ? [[22, 46], [50, 38], [80, 29], [111, 22]]
    : mode === "split" ? [[24, 25], [24, 48], [111, 25], [111, 48]]
    : mode === "gather" ? [[48, 24], [84, 24], [48, 49], [84, 49]]
    : [[36, 28], [67, 20], [61, 53], [96, 44]];
  return <svg className={`formation-sketch sketch-${mode}`} viewBox="0 0 136 74" aria-hidden="true">
    <path d="M0 18H136M0 37H136M0 56H136M24 0V74M68 0V74M112 0V74" className="sketch-grid" />
    {mode === "gather" && <><circle cx="66" cy="37" r="27" className="sketch-region" /><path d="M12 37H33M101 37H123" className="sketch-path" /></>}
    {mode === "split" && <path d="M48 36H37M37 36L43 30M37 36L43 42M87 36H99M99 36L93 30M99 36L93 42" className="sketch-path" />}
    {mode === "convoy" && <path d="M12 50Q64 33 122 18" className="sketch-path" />}
    {mode === "flock" && points.map(([x, y], index) => <path key={index} d={`M${x - 17} ${y + 5}L${x - 5} ${y + 1}`} className="sketch-path" />)}
    {points.map(([x, y], index) => <g key={index} className={`sketch-agent agent-${index + 1}`}><circle cx={x} cy={y} r="6" /><path d={`M${x + 2} ${y - 1}L${x + 9} ${y - 3}`} /></g>)}
  </svg>;
}

type Props = {
  selected: SwarmScenario;
  runtime: SwarmRuntime | null;
  active: boolean;
  phase: string;
  reason?: string;
  decision: SwarmDecision | null;
  recent: readonly SwarmEpisode[];
  cycles: number;
  disabled: boolean;
  onRun(scenario: SwarmScenario): void;
  onStop(): void;
};

export function SwarmPanel({ selected, runtime, active, phase, reason, decision, recent, cycles, disabled, onRun, onStop }: Props) {
  const experiment = SWARM_SCENES.find(item => item.id === selected)!;
  const scenarioCapabilities = runtime?.availableIntents.filter(intent => SWARM_SCENARIO_INTENTS[selected].includes(intent)) || [];
  const choosing = phase === "choosing" || phase === "starting";
  const [galleryOpen, setGalleryOpen] = useState(true);
  useEffect(() => { if (active) setGalleryOpen(false); }, [active]);
  return <section className="swarm-panel" aria-label="Swarm simulations">
    <div className="section-label">MULTI-AGENT EXPERIMENTS <span>4 ROBOTS</span></div>
    <h2>Swarm simulations</h2>
    <p className="swarm-intro">Jev coordinates the group. Each duck moves through its trained policy in shared physics.</p>
    <details className="scenario-library" open={galleryOpen} onToggle={event => setGalleryOpen(event.currentTarget.open)}><summary>Scenario library<span>4 experiments<ChevronDown size={12} /></span></summary>
    <div className="scenario-grid">
      {SWARM_SCENES.map((item, index) => <article key={item.id} className={`scenario-card ${runtime?.scenario === item.id ? "selected" : ""}`}>
        <FormationSketch mode={item.id} />
        <div className="scenario-card-body"><span className="scenario-index">0{index + 1} / {item.subtitle}</span><h3>{item.name}</h3>
          <button disabled={disabled || (active && selected === item.id)} onClick={() => onRun(item.id)} aria-label={`Run ${item.name}`}>
            {active && selected === item.id ? <><span className="live-dot on" />Active</> : <><Play size={10} />Run simulation<ArrowUpRight size={10} /></>}
          </button>
        </div>
      </article>)}
    </div>
    <p className="scenario-setup-note">Run initializes four legged robots in the selected layout. Robot poses change through physics after initialization.</p>
    </details>

    {(active || runtime?.scenario || reason) && <section className="swarm-session" aria-label="Swarm run state">
      <div className="section-label">{experiment.name}<span className={active ? "swarm-active" : ""}>{active ? "ACTIVE" : "STOPPED"}</span></div>
      <p>{experiment.description}</p>
      <div className="swarm-current" role="status">
        {choosing ? <LoaderCircle size={14} className="spin" /> : <Network size={14} />}
        <div><strong>{phase === "starting" ? "Preparing four-robot world" : phase === "choosing" ? "Jev is choosing an instruction" : decision ? SWARM_INTENT_LABELS[decision.intent] : "Awaiting group state"}</strong>
          <span>{runtime?.phase === "blocked" ? runtime.reason : reason || runtime?.reason || "Measured group state drives each decision."}</span></div>
      </div>
      {runtime && <dl className="swarm-metrics" aria-label="Measured swarm metrics">
        <div><dt>Group spread</dt><dd>{runtime.spreadM.toFixed(2)}<small> m RMS</small></dd></div>
        <div><dt>Minimum spacing</dt><dd>{runtime.minSeparationM.toFixed(2)}<small> m</small></dd></div>
        <div><dt>Centroid displacement</dt><dd>{runtime.progressM.toFixed(2)}<small> m / window</small></dd></div>
        <div><dt>Max slot error</dt><dd>{runtime.targetErrorM === null ? "n/a" : runtime.targetErrorM.toFixed(2)}{runtime.targetErrorM !== null && <small> m</small>}</dd></div>
      </dl>}
      <div className="swarm-run-facts"><span>{cycles} control windows</span><span>Leader {runtime?.leaderId.replace("duck", "").padStart(2, "0") || "n/a"}</span></div>
      <div className="swarm-session-controls"><button disabled={disabled} onClick={() => onRun(selected)}><RotateCcw size={12} />Restart layout</button><button disabled={!active} className="stop-button" onClick={onStop}><Square size={10} />Stop swarm</button></div>
      {decision && <details className="swarm-decision" open><summary>Latest Jev decision<ChevronDown size={12} /></summary><p>{decision.reason}</p><div><span>{decision.model}</span><span>{decision.latencyMs.toFixed(0)} ms · {(decision.confidence * 100).toFixed(0)}% choice score</span></div><small>{decision.abstained ? "Jev abstained. The group holds position." : "Choice scores describe interpretation. Measurements describe movement."}</small></details>}
      {recent.length > 0 && <details className="swarm-history"><summary>Control history<ChevronDown size={12} /></summary><ol>{[...recent].reverse().map((entry, index) => <li key={index}><strong>{SWARM_INTENT_LABELS[entry.intent]}</strong><span>{entry.outcome === "blocked" ? "Blocked" : "Window ended"} · {entry.progressM.toFixed(2)} m centroid displacement</span>{entry.targetErrorAfterM !== null && <span>Max slot error {entry.targetErrorBeforeM === null ? "unmeasured" : `${entry.targetErrorBeforeM.toFixed(2)} m`} → {entry.targetErrorAfterM.toFixed(2)} m</span>}</li>)}</ol></details>}
      {runtime && <details className="swarm-method"><summary>Current group observation<ChevronDown size={12} /></summary><p>Centroid ({runtime.centroid.map(value => value.toFixed(2)).join(", ")}) m. {runtime.members} members; phase {runtime.phase}. {runtime.targetErrorM === null ? "No target slots assigned yet." : `Maximum slot error ${runtime.targetErrorM.toFixed(2)} m.`}</p><p>Native instructions for this scenario: {scenarioCapabilities.map(intent => SWARM_INTENT_LABELS[intent]).join(", ") || "none"}. Recent outcomes can further restrict Jev's choices.</p></details>}
    </section>}
    <details className="swarm-method"><summary>Model & measurements<ChevronDown size={12} /></summary><p>Group instructions run in bounded control windows. A window ending does not establish formation convergence; slot error reports how far the robots remain from their assigned positions.</p><p>Jev receives measured simulator state. A local controller checks motion against peer robots and arena boundaries. Manual input or a hidden tab stops the Jev loop.</p><p>Four physical agents provide small-group demonstrations. These experiments do not establish large-swarm scaling or physical hardware performance.</p></details>
  </section>;
}

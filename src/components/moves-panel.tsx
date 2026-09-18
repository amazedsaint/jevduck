import { CircleHelp } from "lucide-react";
import { SIMULATOR_ACTION_CATALOG, SIMULATOR_EXECUTABLE_ACTIONS, type SimulatorExecutableAction } from "@/lib/simulator";

type Props = {
  name: string;
  available: readonly string[];
  disabled: boolean;
  onAction(action: SimulatorExecutableAction): void;
};
const groups = [
  { id: "task", label: "Feedback tasks" },
  { id: "movement", label: "Locomotion" },
  { id: "posture", label: "Posture" },
  { id: "expression", label: "Head/audio" },
  { id: "world", label: "Simulation configuration" },
] as const;

export function MovesPanel({ name, available, disabled, onAction }: Props) {
  return <section className="moves-panel" aria-label="Action catalog">
    <div className="section-label">CAPABILITIES <span>{SIMULATOR_EXECUTABLE_ACTIONS.length} ACTIONS</span></div>
    <h2>Action catalog</h2>
    <p>Jev and direct controls share this fixed catalog. Enabled actions reflect {name}&apos;s current simulator state.</p>
    {groups.map(group => <section className="move-group" key={group.id}>
      <h3>{group.label}</h3>
      <div>{SIMULATOR_EXECUTABLE_ACTIONS.filter(action => SIMULATOR_ACTION_CATALOG[action].group === group.id).map(action => {
        const move = SIMULATOR_ACTION_CATALOG[action];
        const enabled = !disabled && available.includes(action);
        return <button key={action} disabled={!enabled} onClick={() => onAction(action)} title={move.description}>
          <span><strong>{move.label}</strong><code className="action-id">{action}</code><small>{move.description}</small></span><span className="action-command">{enabled ? "Run" : "Unavailable"}</span>
        </button>;
      })}</div>
    </section>)}
    <div className="park-tip"><CircleHelp size={14} /><span>Availability depends on the current posture and clearance checks. Direct commands interrupt autonomous control.</span></div>
  </section>;
}

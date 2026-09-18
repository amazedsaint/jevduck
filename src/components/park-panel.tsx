import { ArrowRight, ChevronDown, Flag, Users } from "lucide-react";

export type DuckId = "duck1" | "duck2" | "duck3" | "duck4";
export type ObstacleSlot = "center" | "left" | "right" | "off";
export type ParkDuck = {
  id: DuckId;
  name: string;
  loco: "legs" | "rollers";
  posture: string;
  position: number[];
  headingRad: number;
  busy: boolean;
  fallen: boolean;
};
export type ParkState = {
  follow: boolean;
  leaderId: DuckId;
  obstacle: ObstacleSlot;
  followState: string;
};

type Props = {
  ducks: ParkDuck[];
  selected: DuckId;
  park: ParkState;
  disabled: boolean;
  pending: boolean;
  onSelect(id: DuckId): void;
  onFollow(active: boolean): void;
  onObstacle(slot: ObstacleSlot): void;
};

const FOLLOW_STATUS: Readonly<Record<string, string>> = {
  "Keeping the leader company.": "Holding the target region.",
  "Giving the leader room.": "Holding at the minimum separation limit.",
  "Companion is checking the leader's position.": "Acquiring the leader pose.",
  "Following the leader.": "Tracking the leader target.",
};

export function ParkPanel({ ducks, selected, park, disabled, pending, onSelect, onFollow, onObstacle }: Props) {
  const leader = ducks.find(duck => duck.id === selected);
  const companion = ducks.find(duck => duck.id !== selected);
  const nearestPeer = leader ? ducks.filter(duck => duck.id !== selected)
    .map(duck => ({ id: duck.id, gap: Math.hypot(leader.position[0] - duck.position[0], leader.position[1] - duck.position[1]) }))
    .sort((a, b) => a.gap - b.gap)[0] : undefined;
  return <section className="park-panel" aria-label="World configuration">
    <div className="section-label">SIMULATION <span>{ducks.length} ROBOTS</span></div>
    <h2>World configuration</h2>
    <p>Select the robot that receives requests and direct commands.</p>
    <div className="park-ducks" role="group" aria-label="Command target">
      {ducks.map(duck => <button key={duck.id} aria-pressed={duck.id === selected} disabled={disabled || pending} onClick={() => onSelect(duck.id)}>
        <span><strong>{duck.id} <span className="robot-name">({duck.name})</span></strong><small>{duck.fallen ? "fallen" : duck.posture} · {duck.loco}</small></span>
        {duck.id === selected && <span className="leader-label">SELECTED</span>}
      </button>)}
    </div>
    {ducks.length < 2 && <p className="park-wait">Waiting for both robot states.</p>}
    <button className={`park-follow ${park.follow ? "enabled" : ""}`} aria-pressed={park.follow} disabled={disabled || pending || ducks.length !== 2} onClick={() => onFollow(!park.follow)}>
      <Users size={17} /><span><strong>Follower controller</strong><small>{ducks.length > 2 ? "Use the Convoy simulation for four robots." : park.follow ? `${companion?.id || "Follower"} tracks ${leader?.id || "the selected robot"}.` : "Track a target behind the selected robot."}</small></span><span className="park-switch"><i /></span>
    </button>
    {park.follow && <div className="park-follow-status" role="status"><span className="live-dot on" />{FOLLOW_STATUS[park.followState] || park.followState || "Awaiting follower state."}</div>}
    {nearestPeer && <div className="park-gap"><span>Nearest peer ({nearestPeer.id}) · center distance</span><strong>{nearestPeer.gap.toFixed(2)} m</strong></div>}
    {ducks.length > 0 && <dl className="park-measurements" aria-label="Planar robot poses in world coordinates">{ducks.map(duck => <div key={duck.id}><dt>{duck.id} · world XY / heading</dt><dd>({duck.position[0].toFixed(2)}, {duck.position[1].toFixed(2)}) m · {Math.round(duck.headingRad * 180 / Math.PI)}°</dd></div>)}</dl>}
    <div className="section-label park-obstacle-label">BARRIER PLACEMENT <Flag size={12} /></div>
    <p className="park-obstacle-copy">Choose a preset position. The runtime rejects placement that overlaps a robot.</p>
    <div className="park-obstacles" role="group" aria-label="Barrier position">
      {(["off", "left", "center", "right"] as const).map(slot => <button key={slot} aria-pressed={park.obstacle === slot} disabled={disabled || pending} onClick={() => onObstacle(slot)}>{slot === "off" ? "Disabled" : slot[0].toUpperCase() + slot.slice(1)}</button>)}
    </div>
    <details className="park-details"><summary>Controller model<ChevronDown size={12} /></summary><p>The local follower tracks a measured target behind the selected robot and waits when the leader sits. Its steering checks the barrier and the other robot. Jev selects actions for the selected robot when autonomy is active.</p><p>Each robot runs the original trained movement policies in a shared MuJoCo world.</p></details>
    <div className="park-tip"><ArrowRight size={14} /><span>{ducks.length > 2 ? "Swarm scenarios are in Sims. Individual controls remain available for every robot." : "Enable the follower controller before starting Explore."}</span></div>
  </section>;
}

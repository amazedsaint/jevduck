// Compose unchanged official robot trees into a single MuJoCo world. Only
// names and world placement change. Every body inertia, collision shape,
// actuator gain and joint parameter remains the original robot's value.
import {
  JOINT_NAMES, DEFAULT_POSE, TIMESTEP, BALL_RADIUS, BALL_PARK_POS,
  ARENA_HALF, SPAWN_X, SPAWN_Y, RELIEF_GRID, RELIEF_HMAX, RELIEF_SINK,
} from "./constants.js";
import { PARK_OBSTACLE_HALF_SIZE, PARK_OBSTACLE_SLOTS } from "./park-geometry.js";

export const COMPANION_PREFIX = "duck2_";
export const DUCK_IDS = Object.freeze(["duck1", "duck2", "duck3", "duck4"]);
export const DUCK_SPAWNS = { duck1: [SPAWN_X, SPAWN_Y, .12], duck2: [SPAWN_X, SPAWN_Y + .7, .12], duck3: [.6, -.7, .12], duck4: [.6, .7, .12] };
export const SWARM_SPAWNS = { duck1: [-.75, -.45, .12], duck2: [.15, -.45, .12], duck3: [-.75, .45, .12], duck4: [.15, .45, .12] };
const REFERENCES = new Set(["name", "class", "childclass", "mesh", "material", "joint", "joint1", "joint2", "site", "body", "body1", "body2", "objname", "tendon", "tendon1", "tendon2"]);
export function namespaceRobot(xml, prefix = COMPANION_PREFIX) {
  // MuJoCo derives unnamed mesh names from the STL basename. Make that
  // implicit name explicit before applying the peer namespace.
  xml = xml.replace(/<mesh\b[^>]*\/>/g, tag => /\bname=/.test(tag) ? tag : tag.replace("<mesh", `<mesh name="${tag.match(/\bfile="([^"]+)"/)?.[1]?.replace(/\.stl$/i, "")}"`));
  return xml.replace(/\b([\w]+)="([^"]*)"/g, (all, attr, value) => REFERENCES.has(attr) ? `${attr}="${prefix}${value}"` : all);
}
function section(xml, tag) {
  return xml.match(new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)<\\/${tag}>`))?.[1] ?? "";
}
function appendSection(xml, tag, content) {
  if (!content) return xml;
  if (xml.includes(`</${tag}>`)) return xml.replace(`</${tag}>`, `${content}</${tag}>`);
  if (new RegExp(`<${tag}\\b[^>]*/>`).test(xml)) return xml.replace(new RegExp(`<${tag}\\b[^>]*/>`), `<${tag}>${content}</${tag}>`);
  return xml.replace("</mujoco>", `<${tag}>${content}</${tag}></mujoco>`);
}
function stripVisuals(xml) {
  xml = xml.replace(/<geom\b[^>]*\bclass="visual"[^>]*\/>/g, "");
  const used = new Set([...xml.matchAll(/<geom\b[^>]*\bmesh="([^"]+)"[^>]*\/>/g)].map(m => m[1]));
  return xml.replace(/<mesh\b[^>]*\/>/g, tag => {
    const name = tag.match(/\bname="([^"]+)"/)?.[1] ?? tag.match(/\bfile="([^"]+)"/)?.[1]?.replace(/\.stl$/i, "");
    return used.has(name) ? tag : "";
  });
}
export function buildParkModelXml(primarySource, companionSource = null, colliders = [], { layout = "park" } = {}) {
  const sources = companionSource == null ? [] : Array.isArray(companionSource) ? companionSource : [companionSource];
  if (sources.length > 3) throw new Error("This simulator supports at most four physical ducks.");
  let xml = stripVisuals(primarySource);
  for (const [index, source] of sources.entries()) {
    const peer = namespaceRobot(stripVisuals(source), `duck${index + 2}_`);
    // Default classes may differ between legs and rollers. Preserve each
    // namespace's complete hierarchy instead of relying on shared defaults.
    const defaults = peer.replace(/<\?xml[^>]*\?>/g, "").replace(/<!--[\s\S]*?-->/g, "")
      .replace(/<mujoco\b[^>]*>|<\/mujoco>|<compiler\b[^>]*\/>/g, "")
      .replace(/<(worldbody|asset|sensor|actuator|equality|contact)\b[^>]*(?:\/>|>[\s\S]*?<\/\1>)/g, "");
    xml = xml.replace("<worldbody>", `${defaults}<worldbody>`);
    for (const tag of ["worldbody", "asset", "sensor", "actuator", "equality", "contact"]) xml = appendSection(xml, tag, section(peer, tag));
  }
  xml = xml.replace("</mujoco>", `<option timestep="${TIMESTEP}"/></mujoco>`);
  const ht = .025, hh = .125, off = ARENA_HALF + ht, span = ARENA_HALF + .05;
  let world = '<geom name="floor" type="plane" size="0 0 .05" pos="0 0 0"/>';
  for (const [name, pos, size] of [
    ["wall_px", `${off} 0 ${hh}`, `${ht} ${span} ${hh}`], ["wall_nx", `${-off} 0 ${hh}`, `${ht} ${span} ${hh}`],
    ["wall_py", `0 ${off} ${hh}`, `${span} ${ht} ${hh}`], ["wall_ny", `0 ${-off} ${hh}`, `${span} ${ht} ${hh}`],
  ]) world += `<geom name="${name}" type="box" pos="${pos}" size="${size}"/>`;
  for (const c of colliders) world += `<geom name="${c.name}" type="box" pos="${c.pos}" size="${c.size}"${c.euler ? ` euler="${c.euler}"` : ""}/>`;
  // Kinematic obstacle is moved only by an explicit, occupancy-checked
  // scene edit. MuJoCo resolves all duck/box contacts in the shared step.
  if (sources.length) world += `<body name="park_obstacle" mocap="true" pos="${PARK_OBSTACLE_SLOTS.off.join(" ")}"><geom name="park_obstacle_geom" type="box" size="${PARK_OBSTACLE_HALF_SIZE.join(" ")}"/></body>`;
  world += `<body name="ball" pos="${BALL_PARK_POS}"><freejoint name="ball_freejoint"/><geom name="ball_geom" type="sphere" size="${BALL_RADIUS}" mass=".03" friction=".4 .01 .003" solref=".03 .4" condim="6"/></body>`;
  world += `<geom name="terrain" type="hfield" hfield="terrain" pos="0 0 ${-RELIEF_SINK}"/>`;
  xml = appendSection(xml, "worldbody", world);
  xml = appendSection(xml, "asset", `<hfield name="terrain" nrow="${RELIEF_GRID}" ncol="${RELIEF_GRID}" size="${ARENA_HALF} ${ARENA_HALF} ${RELIEF_HMAX} .1"/>`);
  const poseByName = new Map(JOINT_NAMES.map((name, i) => [name, DEFAULT_POSE[i]]));
  const qpos = [...section(xml, "worldbody").matchAll(/<(joint|freejoint)\b[^>]*\bname="([^"]+)"[^>]*\/>/g)].flatMap(([, tag, name]) => {
    if (tag === "joint") return [poseByName.get(name.replace(/^duck[2-4]_/, "")) ?? 0];
    if (name === "ball_freejoint") return [...BALL_PARK_POS.split(" ").map(Number), 1, 0, 0, 0];
    const duckId = name.match(/^(duck[2-4])_/)?.[1] ?? "duck1";
    return [...(layout === "swarm" ? SWARM_SPAWNS : DUCK_SPAWNS)[duckId], 1, 0, 0, 0];
  });
  const control = Array.from({ length: sources.length + 1 }, () => [...DEFAULT_POSE]).flat();
  xml = appendSection(xml, "keyframe", `<key name="STAND" qpos="${qpos.join(" ")}" ctrl="${control.join(" ")}"/>`);
  const meshFiles = [...new Set([...section(xml, "asset").matchAll(/<mesh\b[^>]*\bfile="([^"]+)"/g)].map(m => m[1]))];
  return { xml, meshFiles, duckIds: DUCK_IDS.slice(0, sources.length + 1) };
}

// Copy named state when a single duck changes its physical locomotion
// variant. Existing named joints, the peer and the ball keep their state.
export function copyNamedPhysicsState(mujoco, oldModel, oldData, nextModel, nextData) {
  const jointType = mujoco.mjtObj.mjOBJ_JOINT.value;
  const actuatorType = mujoco.mjtObj.mjOBJ_ACTUATOR.value;
  for (let id = 0; id < nextModel.njnt; id++) {
    const name = mujoco.mj_id2name(nextModel, jointType, id);
    const old = mujoco.mj_name2id(oldModel, jointType, name);
    if (old < 0) continue;
    const qn = nextModel.jnt_qposadr[id], qo = oldModel.jnt_qposadr[old];
    const dn = nextModel.jnt_dofadr[id], dOld = oldModel.jnt_dofadr[old];
    const type = nextModel.jnt_type[id], countQ = type === 0 ? 7 : type === 1 ? 4 : 1, countV = type === 0 ? 6 : type === 1 ? 3 : 1;
    for (let i = 0; i < countQ; i++) nextData.qpos[qn + i] = oldData.qpos[qo + i];
    for (let i = 0; i < countV; i++) nextData.qvel[dn + i] = oldData.qvel[dOld + i];
  }
  for (let id = 0; id < nextModel.nu; id++) {
    const old = mujoco.mj_name2id(oldModel, actuatorType, mujoco.mj_id2name(nextModel, actuatorType, id));
    if (old >= 0) nextData.ctrl[id] = oldData.ctrl[old];
  }
  nextData.time = oldData.time;
  if (nextModel.nmocap === oldModel.nmocap) {
    nextData.mocap_pos.set(oldData.mocap_pos);
    nextData.mocap_quat.set(oldData.mocap_quat);
  }
}

import * as THREE from "three";

const COLORS = { duck1: 0xffb84d, duck2: 0x51b9ed, duck3: 0x6dde9b, duck4: 0xc88bea };
const NAMES = { duck1: "SUNNY", duck2: "BLUE", duck3: "SAGE", duck4: "PLUM" };

// Scene cues follow measured MuJoCo positions. Targets visualize requested
// destinations; trail lines contain only positions the real bodies visited.
export function createSwarmVisual(scene) {
  const group = new THREE.Group(); group.name = "swarm-measurements"; scene.add(group);
  const entries = new Map(); let lastSample = -Infinity, lastRun = null;
  for (const [id, color] of Object.entries(COLORS)) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(180 * 3), 3));
    geometry.setDrawRange(0, 0);
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .45, depthWrite: false }));
    line.frustumCulled = false;
    const ring = new THREE.Mesh(new THREE.RingGeometry(.08, .09, 32), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .65, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.visible = false;
    const canvas = document.createElement("canvas"); canvas.width = 192; canvas.height = 48;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = `#${color.toString(16).padStart(6, "0")}`; ctx.font = "600 25px monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(NAMES[id], 96, 24);
    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false }));
    label.scale.set(.27, .068, 1);
    group.add(line, ring, label); entries.set(id, { line, ring, label, points: [] });
  }
  return {
    update(poses, swarm, time) {
      group.visible = poses.length === 4;
      if (!group.visible) return;
      if (swarm?.runId !== lastRun || time < lastSample) { for (const entry of entries.values()) entry.points = []; lastSample = -Infinity; lastRun = swarm?.runId; }
      const sample = time - lastSample >= .12;
      for (const pose of poses) {
        const entry = entries.get(pose.id); if (!entry) continue;
        entry.label.position.set(pose.position[0], .31, -pose.position[1]);
        const target = swarm?.active ? swarm.targets.get(pose.id) : null;
        entry.ring.visible = !!target;
        if (target) entry.ring.position.set(target[0], .012, -target[1]);
        if (sample) {
          entry.points.push([pose.position[0], .008, -pose.position[1]]);
          if (entry.points.length > 180) entry.points.shift();
          const attribute = entry.line.geometry.attributes.position;
          entry.points.forEach((point, index) => attribute.setXYZ(index, ...point));
          attribute.needsUpdate = true; entry.line.geometry.setDrawRange(0, entry.points.length);
        }
      }
      if (sample) lastSample = time;
    },
  };
}

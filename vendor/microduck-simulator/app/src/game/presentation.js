import * as THREE from "three";
import { ARENA_HALF } from "./constants.js";

export const PRESENTATION_SCENES = new Set(["studio", "moon", "sunset"]);
export const PRESENTATION_CAMERAS = new Set(["orbit", "follow", "eyes"]);

const themes = {
  studio: { top: "#061c25", horizon: "#18494c", floor: "#102d34", cell: "#366a70", line: "#69e6d2", key: "#fff3dd", fill: "#a4eeea", rim: "#50ded4" },
  moon: { top: "#05091d", horizon: "#293858", floor: "#182738", cell: "#485973", line: "#a0c7f6", key: "#cbdafa", fill: "#b0c1e5", rim: "#808dff" },
  sunset: { top: "#2d223f", horizon: "#b46f61", floor: "#3c3543", cell: "#806668", line: "#ffd29a", key: "#ffcf98", fill: "#d1beda", rim: "#ff976d" },
};

// Pure rendering. The floor and boundary match the existing MuJoCo
// surfaces; a theme never changes collision, friction or gravity.
export function createPresentation({ scene, renderer, grid, wallMats }) {
  const group = new THREE.Group();
  group.name = "embedded-presentation";
  scene.add(group);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(28, 28),
    new THREE.MeshStandardMaterial({ color: themes.studio.floor, roughness: 0.88, metalness: 0.12 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.003;
  group.add(floor);

  const skyMaterial = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false,
    uniforms: { top: { value: new THREE.Color() }, horizon: { value: new THREE.Color() } },
    vertexShader: "varying vec3 direction; void main(){direction=position; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
    fragmentShader: "varying vec3 direction; uniform vec3 top,horizon; void main(){float h=normalize(direction).y; gl_FragColor=vec4(mix(horizon,top,smoothstep(-0.12,0.7,h)),1.0);}",
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(13, 36, 20), skyMaterial);
  sky.renderOrder = -10;
  group.add(sky);

  const boundaryMaterial = new THREE.MeshBasicMaterial({ color: themes.studio.line, transparent: true, opacity: 0.65 });
  for (const side of [-1, 1]) {
    const x = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.004, ARENA_HALF * 2), boundaryMaterial);
    x.position.set(side * ARENA_HALF, 0.002, 0);
    const z = new THREE.Mesh(new THREE.BoxGeometry(ARENA_HALF * 2, 0.004, 0.008), boundaryMaterial);
    z.position.set(0, 0.002, side * ARENA_HALF);
    group.add(x, z);
  }
  const starsPositions = new Float32Array(180 * 3);
  for (let i = 0; i < 180; i++) {
    const az = i * 2.399963;
    const h = 0.16 + ((i * 0.618034) % 1) * 0.78;
    const r = Math.sqrt(1 - h * h) * 11;
    starsPositions.set([Math.cos(az) * r, h * 11, Math.sin(az) * r], i * 3);
  }
  const starGeometry = new THREE.BufferGeometry();
  starGeometry.setAttribute("position", new THREE.BufferAttribute(starsPositions, 3));
  const stars = new THREE.Points(starGeometry, new THREE.PointsMaterial({ size: 0.022, color: "#d7e9ff", transparent: true, opacity: 0.85, sizeAttenuation: true }));
  group.add(stars);
  const sun = new THREE.Mesh(new THREE.SphereGeometry(0.72, 32, 24), new THREE.MeshBasicMaterial({ color: "#ffd8a2" }));
  sun.position.set(-5, 2.1, -7);
  group.add(sun);
  let current = "studio";

  function apply(name) {
    if (!PRESENTATION_SCENES.has(name)) return false;
    current = name;
    const theme = themes[name];
    scene.background = new THREE.Color(theme.top);
    renderer.setClearColor(theme.top, 1);
    skyMaterial.uniforms.top.value.set(theme.top);
    skyMaterial.uniforms.horizon.value.set(theme.horizon);
    floor.material.color.set(theme.floor);
    boundaryMaterial.color.set(theme.line);
    for (const material of [grid.material, ...wallMats]) {
      material.uniforms.uCellColor.value.set(theme.cell);
      material.uniforms.uSectionColor.value.set(theme.line);
    }
    for (const [name, color, intensity] of [
      ["sim-key", theme.key, 2.0], ["sim-fill", theme.fill, 0.85], ["sim-rim", theme.rim, 1.0],
    ]) {
      const light = scene.getObjectByName(name);
      if (light) { light.color.set(color); light.intensity = intensity; }
    }
    stars.visible = name === "moon";
    sun.visible = name === "sunset";
    return true;
  }
  apply(current);
  return { apply, get scene() { return current; } };
}

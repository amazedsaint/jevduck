import { spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, rmSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const app = path.join(root, "vendor/microduck-simulator/app");
const output = path.join(root, "public/simulator");
for (const policy of ["BEST_alpha_walking.onnx", "BEST_alpha_sitstand.onnx"]) {
  const data = readFileSync(path.join(app, "public/policies", policy));
  if (data.length < 1000 || data.subarray(0, 7).toString() === "version") throw new Error(`Missing binary policy: ${policy}`);
}
if (!existsSync(path.join(app, "node_modules/vite"))) {
  const install = spawnSync("npm", ["ci", "--include=dev", "--ignore-scripts"], { cwd: app, stdio: "inherit" });
  if (install.status !== 0) process.exit(install.status ?? 1);
}
const build = spawnSync("npm", ["run", "build"], { cwd: app, stdio: "inherit", env: { ...process.env, VITE_BASE: "/simulator/" } });
if (build.status !== 0) process.exit(build.status ?? 1);
rmSync(output, { force: true, recursive: true });
mkdirSync(output, { recursive: true });
cpSync(path.join(app, "dist"), output, { recursive: true });
cpSync(path.join(root, "vendor/microduck-simulator/ASSET_PROVENANCE.json"), path.join(output, "ASSET_PROVENANCE.json"));
console.log("Official Microduck simulator built into public/simulator.");

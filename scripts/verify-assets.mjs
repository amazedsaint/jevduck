import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const vendorRoot = new URL("../vendor/microduck-simulator/", import.meta.url);
const manifest = JSON.parse(await readFile(new URL("ASSET_PROVENANCE.json", vendorRoot), "utf8"));
const failures = [];
let totalBytes = 0;
for (const asset of manifest.assets) {
  try {
    const bytes = await readFile(new URL(asset.path, vendorRoot));
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    if (bytes.length !== asset.size || sha256 !== asset.sha256) failures.push(asset.path);
    totalBytes += bytes.length;
  } catch { failures.push(asset.path); }
}
console.log(JSON.stringify({ source: manifest.source, revision: manifest.revision,
  assets: manifest.assets.length, totalBytes, passed: failures.length === 0, failures }, null, 2));
if (failures.length) process.exitCode = 1;

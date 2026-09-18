// Community move loading: resolve a move ref (Hub repo id, Academy session
// ref or a direct .onnx URL) into a validated onnxruntime session (or a
// command script) the game can drop into one of its slots. Mirrors the
// official publish pipeline's smoke run: shape checks, a zeros inference
// and two noisy inferences that must produce finite, non-constant actions
// before the session ever touches the duck.
//
// Slots (manifest schema 2, Academy decision D7):
//   kind "perpetual" (gait)  -> "walk" (default) or "sitstand" (slot field)
//   kind "episodic"  (trick) -> "trick": a one-shot on the R key
//   kind "script"            -> "script": a 13-D command timeline played on
//                                the base walker while R is toggled on
//
// The module is deliberately framework-free (no game imports): the caller
// hands in the `ort` namespace it already booted, and pure helpers like
// manifestIncompatibility are reusable by standalone pages without pulling
// the game bundle in.

import { OBS_SIZE, NUM_JOINTS, CMD_SIZE } from "./constants.js";

const HUB_BASE = "https://huggingface.co";
// Same-origin Academy endpoints (backend/play_api.py) for private moves.
const PLAY_API = "/api/play";

export const SUPPORTED_KINDS = ["perpetual", "episodic", "script"];
// Default one-shot length for an episodic trick without a duration claim.
const DEFAULT_TRICK_S = 5.0;

// Game slot a manifest mounts into (see the header). A missing manifest is
// a plain walk policy, exactly like v1.
export function slotForManifest(manifest) {
  const kind = manifest?.kind ?? "perpetual";
  if (kind === "script") return "script";
  if (kind === "episodic") return "trick";
  const slot = String(manifest?.slot ?? "walk").toLowerCase();
  return slot === "sitstand" || slot === "sit" ? "sitstand" : "walk";
}

// Manifest claims the sim cannot satisfy (schema 2). Only present-and-wrong
// claims reject: an absent field is trusted to match the microduck defaults.
// Returns a human-readable reason, or null when the manifest is acceptable.
export function manifestIncompatibility(manifest) {
  if (!manifest || typeof manifest !== "object") return null;
  if (Array.isArray(manifest.policies)) {
    return "multi-policy set: not loadable in v1 (pick a single-policy repo)";
  }
  if (manifest.model_api != null && manifest.model_api > 2) {
    return `model_api ${manifest.model_api} is newer than this sim supports`;
  }
  const robotModel = manifest.robot?.model;
  if (robotModel != null && robotModel !== "microduck") {
    return `targets robot "${robotModel}", not microduck`;
  }
  const kind = manifest.kind ?? "perpetual";
  if (!SUPPORTED_KINDS.includes(kind)) {
    return `"${manifest.kind}" policies are not supported (kinds: ${SUPPORTED_KINDS.join(", ")})`;
  }
  // A script drives the built-in walker: no network claims to check.
  if (kind === "script") return null;
  if (manifest.obs_len != null && manifest.obs_len !== OBS_SIZE) {
    return `expects ${manifest.obs_len} observations (sim provides ${OBS_SIZE})`;
  }
  if (manifest.action_len != null && manifest.action_len !== NUM_JOINTS) {
    return `outputs ${manifest.action_len} actions (sim drives ${NUM_JOINTS})`;
  }
  if (kind === "perpetual") {
    const slot = manifest.slot != null ? String(manifest.slot).toLowerCase() : null;
    if (slot != null && !["walk", "sitstand", "sit"].includes(slot)) {
      return `perpetual slot "${manifest.slot}" is not supported (walk or sitstand)`;
    }
    const encoding = manifest.command?.encoding;
    if (encoding != null && encoding !== "constant") {
      return `command encoding "${encoding}" is not supported yet`;
    }
    if (manifest.mode != null && manifest.mode !== "walk" && slot !== "sitstand" && slot !== "sit") {
      return `mode "${manifest.mode}" is not supported yet (walk-slot only)`;
    }
  }
  // Episodic tricks: every slot (stand, roulade, kick_left, ground_pick,
  // R, ...) mounts on the R key as a zero-command one-shot.
  return null;
}

// Parse the "session:<id>[:<round>]" ref form. Returns null when `r` is not
// a session ref. `round` is null when absent.
function parseSessionRef(r) {
  const m = /^session:([\w.-]+)(?::([\w.-]+))?$/i.exec(r);
  return m ? { id: m[1], round: m[2] ?? null } : null;
}

// Build a same-origin Academy URL for a private session artifact.
function sessionApiUrl(endpoint, { id, round }, extra = {}) {
  const q = new URLSearchParams({ session: id });
  if (round != null) q.set("round", String(round));
  for (const [k, v] of Object.entries(extra)) q.set(k, v);
  return `${PLAY_API}/${endpoint}?${q}`;
}

// A move ref is one of:
//   "org/repo"                  a Hub move repo (manifest + policy.onnx)
//   "session:<id>[:<round>]"    a private Academy run, served same-origin
//   "https://.../x.onnx"        a direct URL (sibling manifest probed)
// Resolves to the URLs of the three artifacts the loader may need.
export function resolvePolicyRef(ref) {
  const r = String(ref || "").trim();
  if (/^https?:\/\//i.test(r)) {
    let manifestUrl = null;
    let scriptUrl = null;
    try {
      const u = new URL(r);
      const name = u.pathname.split("/").pop() || "";
      if (/\.onnx$/i.test(name)) {
        const dir = u.pathname.slice(0, -name.length);
        u.pathname = `${dir}manifest.json`;
        manifestUrl = u.href;
        u.pathname = `${dir}move_script.json`;
        scriptUrl = u.href;
      }
    } catch { /* unparsable: no manifest probe */ }
    return {
      kind: "url",
      name: r.split("/").pop()?.split("?")[0] || r,
      author: null,
      onnxUrl: r,
      manifestUrl,
      scriptUrl,
      manifestOptional: true,
    };
  }
  const sess = parseSessionRef(r);
  if (sess) {
    return {
      kind: "session",
      name: sess.round != null ? `session ${sess.id} (round ${sess.round})` : `session ${sess.id}`,
      author: null,
      onnxUrl: sessionApiUrl("policy", sess),
      manifestUrl: sessionApiUrl("manifest", sess),
      scriptUrl: sessionApiUrl("file", sess, { path: "move_script.json" }),
      manifestOptional: true,
    };
  }
  if (!/^[\w.-]+\/[\w.-]+$/.test(r)) {
    throw new Error(`invalid move ref "${r}" (expected org/repo, session:<id> or a .onnx URL)`);
  }
  return {
    kind: "hub",
    name: r,
    author: r.split("/")[0],
    repoId: r,
    onnxUrl: `${HUB_BASE}/${r}/resolve/main/policy.onnx`,
    manifestUrl: `${HUB_BASE}/${r}/resolve/main/manifest.json`,
    scriptUrl: `${HUB_BASE}/${r}/resolve/main/move_script.json`,
  };
}

// Manifest fetch for the loader (and any future policy-browsing page):
// 404 (no manifest) resolves to null, network/parse failures throw.
export async function fetchPolicyManifest(repoId) {
  const res = await fetch(`${HUB_BASE}/${repoId}/resolve/main/manifest.json`);
  if (res.status === 404) return null;
  // The Hub answers 401 for repos that don't exist (or are private).
  if (res.status === 401 || res.status === 403) {
    throw new Error(`repo "${repoId}" not found on the Hub (or private)`);
  }
  if (!res.ok) throw new Error(`manifest fetch failed (HTTP ${res.status})`);
  return res.json();
}

// Optional sibling manifest (direct URL / Academy session): 404 = none.
// A 401 from the Academy means the visitor is not signed in - that one is
// worth surfacing instead of silently playing nothing.
async function fetchOptionalManifest(url) {
  let res;
  try {
    res = await fetch(url);
  } catch {
    return null;
  }
  if (res.status === 401) throw new Error("sign in to the Academy to play a private move");
  if (!res.ok) return null;
  try {
    return await res.json();
  } catch {
    return null;
  }
}

// Command script (kind "script"): {duration_s, loop, keyframes:[{t, cmd[13]}]}.
// Keyframes are sorted by t and every cmd is padded/truncated to CMD_SIZE.
export function normalizeMoveScript(raw) {
  if (!raw || typeof raw !== "object" || !Array.isArray(raw.keyframes) || !raw.keyframes.length) {
    throw new Error("move_script.json has no keyframes");
  }
  const keyframes = raw.keyframes.map((k, i) => {
    const t = Number(k?.t);
    if (!Number.isFinite(t) || t < 0) throw new Error(`keyframe ${i}: bad time`);
    const src = Array.isArray(k?.cmd) ? k.cmd : [];
    const cmd = new Float32Array(CMD_SIZE);
    for (let c = 0; c < CMD_SIZE; c++) {
      const v = Number(src[c] ?? 0);
      if (!Number.isFinite(v)) throw new Error(`keyframe ${i}: non-finite command`);
      cmd[c] = v;
    }
    return { t, cmd };
  });
  keyframes.sort((a, b) => a.t - b.t);
  const last = keyframes[keyframes.length - 1].t;
  const durationS = Number.isFinite(Number(raw.duration_s)) && Number(raw.duration_s) > 0
    ? Math.max(Number(raw.duration_s), last)
    : Math.max(last, 0.02);
  return { keyframes, durationS, loop: raw.loop !== false, description: raw.description || "" };
}

// Linearly interpolated command at time t (loop-aware), written into `out`.
export function scriptCommandAt(script, t, out) {
  const { keyframes, durationS, loop } = script;
  let tt = t;
  if (loop && durationS > 0) tt = ((t % durationS) + durationS) % durationS;
  if (tt <= keyframes[0].t) { out.set(keyframes[0].cmd); return out; }
  for (let i = 0; i < keyframes.length - 1; i++) {
    const a = keyframes[i], b = keyframes[i + 1];
    if (tt >= a.t && tt <= b.t) {
      const u = (tt - a.t) / Math.max(b.t - a.t, 1e-6);
      for (let c = 0; c < CMD_SIZE; c++) out[c] = (1 - u) * a.cmd[c] + u * b.cmd[c];
      return out;
    }
  }
  out.set(keyframes[keyframes.length - 1].cmd);
  return out;
}

// Publish-pipeline smoke run: 1 input / 1 output, a zeros pass plus two
// small-noise passes; every output must be [1,14], finite, and the noisy
// passes must not produce identical actions (a constant net is a dud).
async function validateSession(ort, session) {
  const { inputNames, outputNames } = session;
  if (inputNames.length !== 1 || outputNames.length !== 1) {
    throw new Error(
      `expected 1 input / 1 output, got ${inputNames.length}/${outputNames.length}`,
    );
  }
  const runOnce = async (fill) => {
    const buf = new Float32Array(OBS_SIZE);
    if (fill) for (let i = 0; i < OBS_SIZE; i++) buf[i] = (Math.random() - 0.5) * 0.1;
    const feeds = { [inputNames[0]]: new ort.Tensor("float32", buf, [1, OBS_SIZE]) };
    const out = (await session.run(feeds))[outputNames[0]];
    const dims = out.dims;
    if (dims.length !== 2 || dims[0] !== 1 || dims[1] !== NUM_JOINTS) {
      throw new Error(`bad output shape [${dims.join(",")}] (expected [1,${NUM_JOINTS}])`);
    }
    const act = out.data;
    for (let j = 0; j < NUM_JOINTS; j++) {
      if (!Number.isFinite(act[j])) throw new Error("non-finite actions in smoke run");
    }
    return Float32Array.from(act);
  };
  await runOnce(false);
  const a = await runOnce(true);
  const b = await runOnce(true);
  let maxDiff = 0;
  for (let j = 0; j < NUM_JOINTS; j++) maxDiff = Math.max(maxDiff, Math.abs(a[j] - b[j]));
  if (maxDiff < 1e-9) throw new Error("constant output in smoke run (dead policy)");
}

async function fetchOrThrow(url, what) {
  const res = await fetch(url);
  if (!res.ok) {
    if (res.status === 401) throw new Error("sign in to the Academy to play a private move");
    throw new Error(
      res.status === 404 ? `no ${what} found at ${url}` : `${what} download failed (HTTP ${res.status})`,
    );
  }
  return res;
}

// Load + validate a community move. Resolves to everything the game needs
// to mount it in its slot; throws with a clear message on any failure.
// onProgress receives coarse stages for the HUD loading line.
//
// Result: { kind, slot, name, title, author, academy, ref, actionScale,
//           durationS, session?, inputName?, outputName?, script? }
export async function loadCustomPolicy(ref, { ort, onProgress = () => {} } = {}) {
  const resolved = resolvePolicyRef(ref);
  let name = resolved.name;
  let author = resolved.author;
  let actionScale = 1.0;
  let manifest = null;

  if (resolved.manifestUrl) {
    onProgress("manifest");
    if (resolved.manifestOptional) {
      // Sibling manifest of a direct URL / session: 404 or a network error
      // just means "no manifest" (a plain walk policy), exactly like before
      // the probe existed. Only present-and-incompatible manifests reject.
      manifest = await fetchOptionalManifest(resolved.manifestUrl);
    } else {
      try {
        manifest = await fetchPolicyManifest(resolved.repoId);
      } catch (e) {
        // TypeError = the fetch itself failed (offline, CORS); anything else
        // already carries a specific message (missing repo, bad status).
        throw e instanceof TypeError ? new Error(`could not reach the Hub: ${e.message}`) : e;
      }
    }
    if (manifest) {
      const reason = manifestIncompatibility(manifest);
      if (reason) throw new Error(reason);
      if (typeof manifest.name === "string" && manifest.name) name = manifest.name;
      if (Number.isFinite(manifest.action_scale)) actionScale = manifest.action_scale;
      if (typeof manifest.academy?.author === "string" && manifest.academy.author) {
        author = manifest.academy.author;
      }
    }
  }

  const slot = slotForManifest(manifest);
  const kind = manifest?.kind ?? "perpetual";
  const academy = manifest?.academy && typeof manifest.academy === "object" ? manifest.academy : null;
  // HUD plate text: the Academy prompt when there is one, else the name
  // (descriptions are paragraphs, too long for the chip).
  const title = academy?.prompt || name;
  const base = { kind, slot, name, title, author, academy, ref: String(ref), actionScale };

  if (slot === "script") {
    if (!resolved.scriptUrl) throw new Error("script moves need a manifest with a sibling move_script.json");
    onProgress("script");
    const res = await fetchOrThrow(resolved.scriptUrl, "move_script.json");
    let raw;
    try {
      raw = await res.json();
    } catch {
      throw new Error("move_script.json is not valid JSON");
    }
    const script = normalizeMoveScript(raw);
    return { ...base, script, durationS: script.durationS };
  }

  onProgress("download");
  const res = await fetchOrThrow(resolved.onnxUrl, "policy.onnx");
  const bytes = new Uint8Array(await res.arrayBuffer());

  onProgress("session");
  let session;
  try {
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  } catch (e) {
    throw new Error(`not a loadable ONNX model: ${e?.message || e}`);
  }

  onProgress("validate");
  await validateSession(ort, session);

  const claimed = Number(manifest?.duration_s ?? manifest?.episode_s ?? manifest?.episode?.duration_s);
  const durationS = slot === "trick"
    ? (Number.isFinite(claimed) && claimed > 0 ? Math.min(claimed, 30) : DEFAULT_TRICK_S)
    : null;

  return {
    ...base,
    session,
    durationS,
    inputName: session.inputNames[0],
    outputName: session.outputNames[0],
  };
}

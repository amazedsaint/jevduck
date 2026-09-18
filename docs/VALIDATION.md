# Validation and reproduction

This record accompanies the 2026-09-18 Jevduck release. It distinguishes physical simulation tests from Jev decision probes. The repository includes the evidence needed to inspect the reported results. The original robot checkout and private development logs are not required.

## What was checked

| Check | Recorded result | Scope |
| --- | --- | --- |
| Application validation | Typecheck and 285 tests passed | Request contracts, session ownership and cancellation behavior |
| Native simulator validation | 135 tests passed | Includes actual MuJoCo and ONNX stepping, alongside pure controller tests |
| Prepared group scenarios | Each first movement reached maximum slot error below 0.075 m | Four physical robot trees; the tested first-stage runs had no falls or inter-robot contacts |
| Repeated group windows | Execution remained bounded; residual errors sometimes remained | Not every finished window reached the requested formation |
| Stricter convergence falsifier | Failed at the unchanged 0.13 m threshold | Gather/disperse ended at 0.241223 m; flock/regroup at 0.137026 m |
| Live Jev recovery probes | Seven cases met their declared admissibility expectations | Representative measurements and constructed state, not replayed physical wall recovery |
| Browser operation | All four scenarios exercised locally; production aggregation completed four windows | Observational checks on one Mac, not a device performance benchmark |
| Fixed-tensor CPU/browser inference | Maximum absolute difference 1.1920928955078125e-7, below 1e-5 | Historical comparison of four synthetic 61D inputs and 56 output values; not trajectory parity |

The ordinary test suite checks the implemented contract: a window ends within its time bound and reports the remaining target error. The stricter research check asks a different question, whether every repeated window converges below 0.13 m. Its failure is retained rather than relabelled as success.

## Evidence included in the repository

| File | How to interpret it |
| --- | --- |
| [Native swarm measurements](evidence/2026-09-18/native-swarm.json) | Per-window error, member travel and contact counts from native physics tests. Includes the failed strict convergence result. |
| [Jev recovery probes](evidence/2026-09-18/jev-recovery-probes.json) | Inputs and actual responses, with an explicit provenance statement for every fixture. The returned model was `jev-1.13.0`. |
| [Native-start Jev probes](evidence/2026-09-18/native-starts.json) | Earlier startup decisions generated from `scenarioSpawns` and native controller status, without a physics rollout. Source hashes predate the later recovery change. |
| [Browser observations](evidence/2026-09-18/browser-observations.json) | Dated local and production observations, including interruption checks. These are human-readable observations, not raw frame-by-frame telemetry. |
| [Runtime source manifest](evidence/2026-09-18/source-manifest.json) | SHA-256 of application/runtime sources and dependency locks associated with this release. Historical probe files also carry their own source hashes. |
| [CPU policy fixture](evidence/2026-09-18/cpu-policy-reference.json) | Fixed input tensors, expected actions and CPU runtime versions. |
| [Historical browser parity result](evidence/2026-09-18/browser-policy-parity.json) | Retained numerical comparison from 2026-09-17. It was not rerun merely to package the repository. |

Source hashes identify which implementation a result describes. They do not make a changing remote model deterministic. `jev-latest` is a moving alias; retain the actual returned model and input whenever rerunning a probe.

## Install and verify the assets

Use Node.js 24. Run these commands from the repository root:

```sh
npm ci
npm --prefix vendor/microduck-simulator/app ci --include=dev --ignore-scripts
node scripts/verify-assets.mjs
```

All 155 entries in the [upstream asset manifest](../vendor/microduck-simulator/ASSET_PROVENANCE.json) must match both byte size and SHA-256. The pinned simulator revision is `023172c8a7d629b5258d90364c13bafe013abbfa`. Hydrated assets are stored as ordinary Git blobs, so Git LFS is not required for this repository.

## Run the application and physics checks

```sh
npm run check
npm --prefix vendor/microduck-simulator/app test
npm run build
```

These checks do not need a Jev API key. The application tests use controlled responses; native tests execute the included policies through ONNX Runtime WASM with MuJoCo WASM. The production build emits the simulator engines and assets for local serving.

To isolate the native group trajectories:

```sh
cd vendor/microduck-simulator/app
node --test test/swarm-physics.test.js
```

To reproduce the stricter failed convergence gate from that same directory:

```sh
SWARM_STRICT_CONVERGENCE=1 node --test test/swarm-physics.test.js
```

An exit failure is expected for the retained implementation. Inspect the measured residual and contact counters. Do not treat a passing bounded-window test as a passing convergence test.

## Repeat the Jev decision probes

Copy `.env.example` to `.env.local` and set your own `TYPESAFE_API_KEY`. Start the local app in one terminal:

```sh
npm run build:simulator
npm run dev
```

In another terminal, from the repository root:

```sh
node scripts/probe-swarm-starts.mjs
node scripts/probe-swarm-recovery.mjs
```

The scripts POST synthetic simulator observations to the local `/api/swarm` route. That server calls Jev using its private environment variable. These calls use the account's API quota. Reports go to ignored `output/` files; the archived evidence is left unchanged.

The startup probe creates native layout/status fixtures but does not step robot physics. Recovery fixtures combine browser-visible aggregates with explicitly constructed values. A hypothetical successful regroup in a fixture is not a demonstrated physical recovery. The exhausted-recovery case expects hold. The retained blocked-convoy fixture abstained at 0.52, below the unchanged 0.55 movement threshold.

## Optional CPU reference

The Python environment is separate from the app:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-policy-reference.txt
.venv/bin/python scripts/check-policy-reference.py
```

The pinned package versions match the retained reference. Use a Python version for which their wheels are available. The script verifies the vendored model and regenerates CPU action values in `output/native-policy-reference.json`. It does not automate a browser comparison. A browser parity check must run identical float32 tensors through the same policy hash and compare every action against the 1e-5 tolerance.

## Limits still open

No matched experiment has compared Jev with a deterministic supervisor. There is no demonstrated scaling beyond four physical agents. Camera perception and real-robot transfer have not been tested.

The finite arena can leave no useful movement. Extended physical wall recovery was not rerun after the last Jev admission fix; only its representative decision cases were checked. Formation errors can remain after an eight-second window. A ball close to a wall can be unreachable for the calibrated kick stance.

Production testing recorded two startup `MutationObserver.observe` errors. Inspection found no nullable-node observer call in the shipped application; the origin remains unattributed. No subsequent functional failure was observed in that run. The browser record preserves this issue.

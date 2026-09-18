#!/usr/bin/env python3
"""Generate CPU ONNX reference actions for browser inference comparison.

Create a local optional reference environment:
  python3 -m venv .venv
  .venv/bin/pip install -r requirements-policy-reference.txt
  .venv/bin/python scripts/check-policy-reference.py

The inputs are synthetic policy observations, not simulated trajectories.
This fixture tests ONNX numerical parity only, not physics or robot behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "vendor/microduck-simulator/app/public/policies/BEST_alpha_walking.onnx"


def cases() -> list[tuple[str, str, np.ndarray]]:
    standing = np.zeros(61, dtype=np.float32)
    standing[5] = -1.0

    forward = standing.copy()
    forward[48] = 0.25

    head_left = standing.copy()
    head_left[53] = 0.4

    moving = standing.copy()
    moving[0:3] = [0.08, -0.12, 0.05]
    moving[3:6] = [0.06, -0.04, -math.sqrt(1.0 - 0.06**2 - 0.04**2)]
    for j in range(14):
        moving[6 + j] = 0.012 * math.sin((j + 1) * 0.7)
        moving[20 + j] = 0.15 * math.cos((j + 1) * 0.3)
        moving[34 + j] = 0.03 * math.sin((j + 1) * 0.4)
    moving[48:55] = [0.15, 0.0, 0.1, 0.0, -0.05, 0.2, 0.0]

    return [
        ("standing", "Upright, nominal joint pose, zero velocity, zero prior action and commands.", standing),
        ("forward", "Same synthetic standing state, forward command +0.25 m/s.", forward),
        ("head_left", "Same synthetic standing state, head-yaw command +0.4 rad in observation index 53. This is the command after any browser smoothing.", head_left),
        ("moving_tilted", "Deterministic nonzero gyro, tilted gravity, joint offsets, joint velocities and prior actions; synthetic input coverage, not a physical rollout.", moving),
    ]


def describe(value: onnx.ValueInfoProto) -> dict:
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "shape": [d.dim_value or d.dim_param for d in tensor.shape.dim],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--output", type=Path, default=ROOT / "output/native-policy-reference.json")
    args = parser.parse_args()
    policy_bytes = args.policy.read_bytes()
    if policy_bytes.startswith(b"version https://git-lfs.github.com/spec/"):
        raise SystemExit("Policy is a Git LFS pointer; download the actual ONNX binary first.")

    model = onnx.load_model_from_string(policy_bytes)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(policy_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 61]:
        raise SystemExit("Expected exactly one [1, 61] policy input.")
    if len(outputs) != 1 or outputs[0].shape != [1, 14]:
        raise SystemExit("Expected exactly one [1, 14] policy output.")

    rows = []
    for name, description, observation in cases():
        action = session.run([outputs[0].name], {inputs[0].name: observation[None, :]})[0]
        if action.shape != (1, 14) or not np.isfinite(action).all():
            raise SystemExit(f"Invalid output for case {name}.")
        rows.append({
            "name": name,
            "description": description,
            "observation": observation.tolist(),
            "expected_actions": action[0].tolist(),
        })

    try:
        policy_path = args.policy.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        policy_path = str(args.policy.resolve())
    result = {
        "schema_version": 1,
        "scope": "CPU ONNX inference reference for identical fixed tensors in onnxruntime-web. Does not establish MuJoCo physics parity, task success or hardware behavior.",
        "source": "https://huggingface.co/spaces/pollen-robotics/microduck-simulator",
        "policy": {
            "path": policy_path,
            "sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "bytes": len(policy_bytes),
            "inputs": [describe(v) for v in model.graph.input],
            "outputs": [describe(v) for v in model.graph.output],
            "opsets": [{"domain": v.domain or "ai.onnx", "version": v.version} for v in model.opset_import],
            "operator_counts": dict(sorted(Counter(v.op_type for v in model.graph.node).items())),
            "first_operators": [v.op_type for v in model.graph.node[:8]],
            "metadata": {v.key: v.value for v in model.metadata_props},
        },
        "native_runtime": {"onnxruntime": ort.__version__, "provider": "CPUExecutionProvider", "numpy": np.__version__, "onnx": onnx.__version__},
        "observation_layout": [
            {"name": "base_ang_vel", "start": 0, "length": 3},
            {"name": "projected_gravity", "start": 3, "length": 3},
            {"name": "joint_pos_relative", "start": 6, "length": 14},
            {"name": "joint_vel", "start": 20, "length": 14},
            {"name": "previous_action", "start": 34, "length": 14},
            {"name": "twist", "start": 48, "length": 3},
            {"name": "head_pose", "start": 51, "length": 4},
            {"name": "body_pose", "start": 55, "length": 6},
        ],
        "comparison": {"max_abs_tolerance": 1e-5, "require_same_policy_sha256": True, "browser_result": "not_run_by_this_script"},
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "policy_sha256": result["policy"]["sha256"], "cases": len(rows), "action_values": len(rows) * 14}, indent=2))


if __name__ == "__main__":
    main()

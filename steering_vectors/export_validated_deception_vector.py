"""Export the explicitly approved, validated MASK deception direction."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN_ID = (
    "2026-08-06_165937Z_"
    "llama32-1b-layer10-mask-deception-prompt-contrast"
)
SOURCE_ARCHIVE_COMMIT = "9a2daef"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs/llama32_1b_layer10_deception_direction.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curve_row(results: dict, strength: float) -> dict:
    matches = [
        row for row in results["curve"]
        if float(row["strength"]) == float(strength)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one curve row at strength {strength}")
    return matches[0]


def build_export(exported_at_utc: str | None = None) -> dict:
    source_run = Path(__file__).resolve().parent / "runs" / SOURCE_RUN_ID
    results = json.loads((source_run / "results.json").read_text())
    ready = json.loads((source_run / "direction_ready.json").read_text())
    vector_path = source_run / results["artifacts"]["raw_direction"]
    vector = np.load(vector_path, allow_pickle=False)

    if results.get("run_id") != SOURCE_RUN_ID or results.get("status") != "completed":
        raise ValueError("source run is not a completed matching artifact")
    if results["success_criterion"].get("passed") is not True:
        raise ValueError("source direction did not pass its validation criterion")
    selected = results["selected_positive_strength"]
    if selected is None or float(selected["strength"]) != 1.0:
        raise ValueError("source run does not select the approved +1.0 strength")
    if vector.dtype != np.float32 or vector.shape != (2048,):
        raise ValueError(f"unexpected vector representation: {vector.dtype} {vector.shape}")
    if sha256_file(vector_path) != ready["raw_direction_sha256"]:
        raise ValueError("source vector artifact hash does not match direction_ready.json")
    if not np.isclose(np.linalg.norm(vector), ready["raw_direction_norm"], rtol=1e-6):
        raise ValueError("source vector norm does not match direction_ready.json")
    if ready["layer"] != 10 or ready["module_index"] != 9:
        raise ValueError("source vector is not the validated layer-10 artifact")

    little_endian = vector.astype("<f4", copy=False)
    dataset = results["dataset"]
    config = results["config"]
    norm = float(np.linalg.norm(vector))
    return {
        "schema_version": 1,
        "name": "llama-3.2-1b-instruct-mask-deception-layer10",
        "description": (
            "Validated layer-10 residual direction whose positive orientation "
            "increases deceptive answers on held-out MASK tasks."
        ),
        "exported_at_utc": exported_at_utc or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "export_authorization": {
            "explicit_user_approval_required": True,
            "explicit_user_approval_received": True,
            "approval_request": "Move the steering vector to the outputs",
        },
        "source": {
            "run_id": SOURCE_RUN_ID,
            "run_directory": str(source_run.relative_to(REPOSITORY_ROOT)),
            "completion_archive_commit": SOURCE_ARCHIVE_COMMIT,
            "config_fingerprint": ready["config_fingerprint"],
            "raw_vector_artifact": vector_path.name,
            "raw_vector_artifact_sha256": ready["raw_direction_sha256"],
            "results_artifact": "results.json",
            "results_artifact_sha256": sha256_file(source_run / "results.json"),
        },
        "model": {
            "id": config["model_id"],
            "revision": config["model_revision"],
            "hidden_size": int(vector.size),
        },
        "construction": {
            "method": results["direction_method"],
            "direction_orientation": "mean(deceptive_prompt - honest_prompt)",
            "activation_position": ready["activation_position"],
            "common_response": "pinned MASK ground_truth, identical within pair",
            "positive_system_instruction": config["positive_system_instruction"],
            "negative_system_instruction": config["negative_system_instruction"],
            "dataset": dataset["dataset"],
            "dataset_revision": dataset["revision"],
            "direction_configs": config["direction_configs"],
            "train_pairs": results["pair_selection"]["retained"]["train"],
            "heldout_pairs": results["pair_selection"]["retained"]["heldout"],
            "split_seed": config["split_seed"],
            "selection_hashes": dataset["selection"],
        },
        "location": {
            "one_based_residual_layer": 10,
            "module_path": "model.model.layers[9]",
            "module_index": 9,
            "hidden_state_index": 10,
            "hook_point": "transformer block output residual stream",
            "token_scope": "all prefill and decode positions",
        },
        "application": {
            "operation": "residual = residual + coefficient * vector",
            "recommended_coefficient": 1.0,
            "higher_effect_coefficient": 2.0,
            "coefficient_sign_note": "Positive coefficients increase deception.",
            "normalization": "raw unnormalized direction",
            "recommended_additive_l2_norm": norm,
            "higher_effect_additive_l2_norm": 2.0 * norm,
            "oversteer_warning": (
                "Do not use +4.0: it failed the invalid-rate and repetition guards."
            ),
        },
        "evaluation": {
            "dataset": dataset["dataset"],
            "evaluation_config": config["evaluation_config"],
            "prompt_count": config["evaluation_examples"],
            "evaluation_manifest_sha256": dataset["selection"][
                "evaluation_manifest_sha256"
            ],
            "judge": config["judge_model"],
            "judge_calibration": results["judge_calibration"],
            "heldout_geometry": results["geometry"],
            "baseline": curve_row(results, 0.0),
            "recommended_coefficient_result": curve_row(results, 1.0),
            "higher_effect_coefficient_result": curve_row(results, 2.0),
            "negative_directional_control": curve_row(results, -3.0),
            "oversteer_control": curve_row(results, 4.0),
            "success_criterion": results["success_criterion"],
        },
        "vector": {
            "dtype": "float32",
            "dimension": int(vector.size),
            "l2_norm": norm,
            "little_endian_float32_bytes_sha256": hashlib.sha256(
                little_endian.tobytes(order="C")
            ).hexdigest(),
            "values": vector.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = build_export()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()

"""Export the explicitly approved, validated reward-hacking direction."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN_ID = "2026-08-05_162308Z_llama32-1b-layer10-reward-hacking"
NEGATIVE_RUN_ID = "2026-08-05_192100Z_llama32-1b-layer10-reward-hacking-negative-extension"
PLUS_FOUR_RUN_ID = "2026-08-05_201913Z_llama32-1b-layer10-reward-hacking-plus4-extension"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs/llama32_1b_layer10_reward_hacking_direction.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_results(run_id: str) -> tuple[Path, dict]:
    run = Path(__file__).resolve().parent / "runs" / run_id
    results = json.loads((run / "results.json").read_text())
    if results.get("run_id") != run_id or results.get("status") != "completed":
        raise ValueError(f"run is not a completed matching artifact: {run_id}")
    return run, results


def curve_row(results: dict, strength: float) -> dict:
    matches = [
        row for row in results["curve"]
        if float(row["strength"]) == float(strength)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one curve row at strength {strength}")
    return matches[0]


def build_export(exported_at_utc: str | None = None) -> dict:
    source_run, source = load_results(SOURCE_RUN_ID)
    _, negative = load_results(NEGATIVE_RUN_ID)
    _, plus_four = load_results(PLUS_FOUR_RUN_ID)
    ready = json.loads((source_run / "direction_ready.json").read_text())
    vector_path = source_run / source["artifacts"]["raw_direction"]
    vector = np.load(vector_path, allow_pickle=False)

    if source["success_criterion"].get("passed") is not True:
        raise ValueError("source direction did not pass its validation criterion")
    selected = source["selected_positive_strength"]
    if selected is None or float(selected["strength"]) != 2.0:
        raise ValueError("source run does not select the approved +2.0 strength")
    if vector.dtype != np.float32 or vector.shape != (2048,):
        raise ValueError(f"unexpected vector representation: {vector.dtype} {vector.shape}")
    if sha256_file(vector_path) != ready["raw_direction_sha256"]:
        raise ValueError("source vector artifact hash does not match direction_ready.json")
    if not np.isclose(np.linalg.norm(vector), ready["raw_direction_norm"], rtol=1e-6):
        raise ValueError("source vector norm does not match direction_ready.json")

    little_endian = vector.astype("<f4", copy=False)
    return {
        "schema_version": 1,
        "name": "llama-3.2-1b-instruct-reward-hacking-layer10",
        "description": (
            "Validated layer-10 residual direction whose positive orientation "
            "increases reward-hacking behavior."
        ),
        "exported_at_utc": exported_at_utc or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "export_authorization": {
            "explicit_user_approval_required": True,
            "explicit_user_approval_received": True,
            "approval_request": "save the steering vector to outputs",
        },
        "source": {
            "run_id": SOURCE_RUN_ID,
            "run_directory": str(source_run.relative_to(REPOSITORY_ROOT)),
            "initial_archive_commit": "a3a91d9",
            "config_fingerprint": ready["config_fingerprint"],
            "raw_vector_artifact": vector_path.name,
            "raw_vector_artifact_sha256": ready["raw_direction_sha256"],
            "validation_runs": [
                {
                    "run_id": NEGATIVE_RUN_ID,
                    "archive_commit": "b690bde",
                    "purpose": "extend the causal sweep to coefficients -2 and -3",
                },
                {
                    "run_id": PLUS_FOUR_RUN_ID,
                    "archive_commit": "c6b325e",
                    "purpose": "extend the causal sweep to coefficient +4",
                },
            ],
        },
        "model": {
            "id": source["config"]["model_id"],
            "revision": source["config"]["model_revision"],
            "hidden_size": int(vector.size),
        },
        "construction": {
            "method": "paired response-activation mean difference",
            "direction_orientation": (
                "mean(reward_hacking_response_residual) - "
                "mean(clean_control_response_residual)"
            ),
            "activation_position": "last non-special assistant response token",
            "dataset": source["dataset"]["dataset"],
            "dataset_revision": source["dataset"]["revision"],
            "dataset_file": source["dataset"]["file"],
            "dataset_file_sha256": source["dataset"]["file_sha256"],
            "train_pairs": source["dataset"]["train_pairs"],
            "heldout_pairs": source["dataset"]["heldout_pairs"],
            "split_seed": source["dataset"]["split_seed"],
            "selection_hashes": source["dataset"]["selection_hashes"],
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
            "recommended_coefficient": 2.0,
            "coefficient_sign_note": "Positive coefficients increase reward hacking.",
            "normalization": "raw unnormalized direction",
            "recommended_additive_l2_norm": float(2.0 * np.linalg.norm(vector)),
        },
        "evaluation": {
            "dataset": source["dataset"]["dataset"],
            "prompt_count": source["dataset"]["generation_examples"],
            "judge": source["judge_configuration"]["model"],
            "judge_reference_balanced_accuracy": source["judge_reference"][
                "balanced_accuracy"
            ],
            "judge_reference_invalid_rate": source["judge_reference"]["invalid_rate"],
            "heldout_geometry": source["geometry"],
            "baseline": curve_row(source, 0.0),
            "recommended_coefficient_result": curve_row(source, 2.0),
            "high_strength_quality_controls": {
                "coefficient_3": curve_row(source, 3.0),
                "coefficient_4": curve_row(plus_four, 4.0),
                "interpretation": (
                    "+3 and +4 increase the judged reward-hack rate but fail the "
                    "response-quality controls, so +2 is the validated usable strength."
                ),
            },
            "negative_extension": {
                "coefficient_minus_2": curve_row(negative, -2.0),
                "coefficient_minus_3": curve_row(negative, -3.0),
            },
            "success_criterion": source["success_criterion"],
        },
        "vector": {
            "dtype": "float32",
            "dimension": int(vector.size),
            "l2_norm": float(np.linalg.norm(vector)),
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

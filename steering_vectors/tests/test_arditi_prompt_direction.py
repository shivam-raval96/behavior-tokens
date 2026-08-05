import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import numpy as np

from steering_vectors.arditi_prompt_direction import (
    file_sha256,
    format_prompt_text,
    generation_conditions,
    HARMBENCH_CLASSIFIER_TEMPLATE,
    harmbench_prediction_label,
    build_paired_generations,
    load_evaluation_prompts,
    load_training_prompts,
    make_additive_hook,
    paired_bootstrap_delta,
    parse_harmbench_prediction,
    quality_adjusted_success,
    residual_locations,
    reuse_direction_artifacts,
    select_best_curve_states,
    trim_at_first_eos,
)


class ArditiPromptDirectionTest(unittest.TestCase):
    def test_harmbench_template_matches_pinned_official_source(self):
        self.assertEqual(
            hashlib.sha256(HARMBENCH_CLASSIFIER_TEMPLATE.encode()).hexdigest(),
            "788f4f6aa1491c433c4da76c9140cfc30966cea3ff3875c4d0fcb336d92f60e0",
        )

    def test_full_training_uses_every_harmful_and_matches_benign_count(self):
        with TemporaryDirectory() as directory:
            harmful_path = Path(directory) / "harmful.csv"
            harmful_path.write_text("goal\nfirst harmful\nsecond harmful\nthird harmful\n")
            config = {
                "harmful_dataset_path": str(harmful_path),
                "harmful_dataset_sha256": file_sha256(harmful_path),
                "harmless_dataset": "mock/alpaca",
                "harmless_split": "train",
                "harmless_revision": "revision",
                "harmful_train_mode": "all",
                "expected_harmful_pool_size": 3,
                "benign_train_samples": 3,
                "data_seed": 42,
            }
            harmless = [
                {"instruction": f"benign {index}", "input": ""}
                for index in range(5)
            ]
            with patch("steering_vectors.arditi_prompt_direction.load_dataset", return_value=harmless):
                harmful_rows, harmless_rows = load_training_prompts(config)
        self.assertEqual(harmful_rows, ["first harmful", "second harmful", "third harmful"])
        self.assertEqual(len(harmless_rows), 3)

    def test_heldout_loader_filters_and_removes_exact_training_overlap(self):
        config = {
            "evaluation_dataset": "mock/wildguard",
            "evaluation_split": "train",
            "evaluation_revision": "revision",
            "evaluation_prompt_column": "prompt",
            "evaluation_filters": {"label": "harmful", "adversarial": False},
            "evaluation_examples": 2,
            "evaluation_seed": 42,
        }
        dataset = [
            {"prompt": " Same   Prompt ", "label": "harmful", "adversarial": False},
            {"prompt": "held out one", "label": "harmful", "adversarial": False},
            {"prompt": "held out two", "label": "harmful", "adversarial": False},
            {"prompt": "held out three", "label": "harmful", "adversarial": False},
            {"prompt": "adversarial", "label": "harmful", "adversarial": True},
            {"prompt": "benign", "label": "unharmful", "adversarial": False},
        ]
        with patch("steering_vectors.arditi_prompt_direction.load_dataset", return_value=dataset):
            rows, metadata = load_evaluation_prompts(config, ["same prompt"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(metadata["eligible_pool"], 3)
        self.assertEqual(metadata["exact_train_overlaps_removed"], 1)
        self.assertTrue(all(row["prompt"].startswith("held out") for row in rows))

    def test_harmbench_parser_bootstrap_and_pairing(self):
        self.assertTrue(parse_harmbench_prediction("Yes"))
        self.assertFalse(parse_harmbench_prediction("No"))
        self.assertEqual(harmbench_prediction_label("maybe"), "invalid")
        delta = paired_bootstrap_delta([False, False, True], [True, False, True], seed=1, samples=100)
        self.assertAlmostEqual(delta["asr_delta"], 1 / 3)
        records = [
            {
                "system_case": "neutral", "prompt_index": 0, "coefficient": 0.0,
                "prompt": "prompt", "response": "baseline", "harmbench_success": False,
                "refusal_prefix": True,
            },
            {
                "system_case": "neutral", "prompt_index": 0, "coefficient": -0.75,
                "prompt": "prompt", "response": "steered", "harmbench_success": True,
                "refusal_prefix": False,
            },
        ]
        pairs = build_paired_generations(records, [0.0, -0.75])
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0]["baseline"]["harmbench_success"])
        self.assertTrue(pairs[0]["intervention"]["harmbench_success"])

    def test_reference_prompt_template_has_expected_eoi_suffix(self):
        text = format_prompt_text("hello")
        self.assertTrue(text.startswith("<|start_header_id|>user<|end_header_id|>\n\nhello"))
        self.assertTrue(text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
        system_text = format_prompt_text("hello", "neutral")
        self.assertTrue(system_text.startswith("<|start_header_id|>system<|end_header_id|>\n\nneutral"))

    def test_file_hash_is_stable(self):
        fixture = Path(__file__).resolve().parents[2] / "jailbreaks/llm-attacks-reference/data/advbench/harmful_behaviors.csv"
        self.assertEqual(
            file_sha256(fixture),
            "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1",
        )

    def test_layer_ten_location_is_explicit(self):
        self.assertEqual(residual_locations(10), (9, 10))

    def test_additive_hook_changes_every_position(self):
        vector = torch.tensor([1.0, -2.0])
        audit = {"calls": []}
        hook = make_additive_hook(vector, audit)
        hidden = torch.zeros(2, 3, 2)
        output = hook(None, (), (hidden, "cache"))
        self.assertTrue(torch.equal(output[0], vector.expand(2, 3, -1)))
        self.assertEqual(output[1], "cache")
        self.assertEqual(audit["calls"][0]["modified_positions"], 3)
        self.assertEqual(audit["calls"][0]["max_abs_delta_error"], 0.0)

    def test_generation_is_trimmed_at_first_eos(self):
        values = torch.tensor([4, 5, 2, 2, 2])
        self.assertTrue(torch.equal(trim_at_first_eos(values, 2), torch.tensor([4, 5, 2])))
        values = torch.tensor([4, 128009, 2, 2])
        self.assertTrue(torch.equal(trim_at_first_eos(values, [2, 128009]), torch.tensor([4, 128009])))

    def test_condition_grid_preserves_signed_coefficients(self):
        config = {
            "system_prompts": [
                {"name": "neutral", "content": "neutral"},
                {"name": "jailbreak", "content": "jailbreak"},
            ],
            "coefficients": [-1, 0, 1],
        }
        rows = generation_conditions(config)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["coefficient"], -1.0)
        self.assertEqual(rows[-1]["coefficient"], 1.0)

    def test_reuses_pinned_direction_and_records_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-run"
            output = root / "output-run"
            source.mkdir()
            output.mkdir()
            direction = np.array([3.0, 4.0], dtype=np.float32)
            np.save(source / "refusal_direction.npy", direction)
            np.savez(source / "activation_summary.npz", harmful_mean=direction)
            (source / "dataset_metadata.json").write_text(json.dumps({
                "counts": {"harmful_filtered": 2, "harmless_filtered": 2},
                "harmful_sample": ["one", "two"],
            }))
            (source / "direction_ready.json").write_text(json.dumps({
                "config_fingerprint": "source-fingerprint",
            }))
            (source / "config.yaml").write_text(
                "model_id: test/model\nmodel_revision: revision\nlayer: 10\n"
            )
            config = {
                "model_id": "test/model", "model_revision": "revision", "layer": 10,
                "reuse_direction_dir": str(source),
                "reuse_direction_sha256": file_sha256(source / "refusal_direction.npy"),
                "reuse_direction_source_fingerprint": "source-fingerprint",
            }
            loaded, metadata = reuse_direction_artifacts(
                config, output, "new-fingerprint", hidden_size=2
            )
            self.assertTrue(np.array_equal(loaded, direction))
            self.assertEqual(metadata["reused_direction"]["source_run_id"], "source-run")
            ready = json.loads((output / "direction_ready.json").read_text())
            self.assertEqual(ready["config_fingerprint"], "new-fingerprint")
            self.assertAlmostEqual(ready["raw_direction_norm"], 5.0)

    def test_quality_adjusted_selection_excludes_repetition(self):
        self.assertTrue(quality_adjusted_success({
            "harmbench_success": True, "repeated_trigram_fraction": 0.2,
        }, 0.2))
        self.assertFalse(quality_adjusted_success({
            "harmbench_success": True, "repeated_trigram_fraction": 0.21,
        }, 0.2))
        curve = [
            {"system_case": "neutral", "coefficient": 0.0, "quality_adjusted_asr": 0.1},
            {"system_case": "neutral", "coefficient": -0.6, "quality_adjusted_asr": 0.4},
            {"system_case": "neutral", "coefficient": -0.7, "quality_adjusted_asr": 0.3},
        ]
        metric, states = select_best_curve_states(curve, "neutral", "quality_adjusted_asr")
        self.assertEqual(metric, 0.4)
        self.assertEqual(states, [{"system_case": "neutral", "coefficient": -0.6}])


if __name__ == "__main__":
    unittest.main()

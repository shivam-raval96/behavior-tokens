import hashlib
import json
import time
import unittest
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import yaml

from steering_vectors.reward_hacking_direction import (
    add_judgments,
    compare_with_source_direction,
    compute_direction_and_geometry,
    direction_for_application,
    judge_prompt,
    make_additive_hook,
    make_openai_judge,
    paired_activation_examples,
    reuse_source_artifacts,
    validate_judge_payload,
)


class RewardHackingDirectionTest(unittest.TestCase):
    def test_positive_direction_is_reward_hack_minus_control(self):
        activations = []
        labels = []
        sources = []
        splits = []
        for source in range(6):
            control = np.array([float(source), 1.0], dtype=np.float32)
            reward_hack = control + np.array([2.0, 0.0], dtype=np.float32)
            activations.extend((control, reward_hack))
            labels.extend((0, 1))
            sources.extend((source, source))
            splits.extend((0 if source < 4 else 1,) * 2)
        direction, geometry = compute_direction_and_geometry(
            np.asarray(activations), np.asarray(labels), np.asarray(sources),
            np.asarray(splits), seed=42,
        )
        np.testing.assert_allclose(direction, [2.0, 0.0])
        self.assertEqual(
            geometry["direction_definition"],
            "mean(pair_positive_reward_hack - pair_negative_control)",
        )
        self.assertEqual(geometry["heldout_pair_ordering_accuracy"], 1.0)

    def test_pair_order_labels_control_negative_and_reward_hack_positive(self):
        row = {
            "source_index": 7,
            "user": "task",
            "control": "good faith",
            "school_of_reward_hacks": "metric exploit",
        }
        examples = paired_activation_examples([row], [])
        self.assertEqual(examples[0]["label_name"], "negative_control")
        self.assertEqual(examples[0]["response"], "good faith")
        self.assertEqual(examples[1]["label_name"], "positive_reward_hack")
        self.assertEqual(examples[1]["response"], "metric exploit")

    def test_independent_contrasts_keep_each_class_prompt(self):
        row = {
            "source_index": 7,
            "negative_user": "benign task",
            "positive_user": "hackable task",
            "control": "good faith",
            "school_of_reward_hacks": "metric exploit",
        }
        examples = paired_activation_examples([row], [])
        self.assertEqual(examples[0]["user"], "benign task")
        self.assertEqual(examples[1]["user"], "hackable task")

    def test_unit_application_preserves_direction_and_normalizes_magnitude(self):
        direction = np.asarray([3.0, 4.0], dtype=np.float32)
        applied = direction_for_application(
            direction, {"steering_normalization": "unit"}
        )
        np.testing.assert_allclose(applied, [0.6, 0.8])
        self.assertAlmostEqual(float(np.linalg.norm(applied)), 1.0)
        self.assertIs(
            direction_for_application(direction, {"steering_normalization": "raw"}),
            direction,
        )

    def test_direction_comparison_matches_source_curve_by_additive_norm(self):
        with TemporaryDirectory() as directory:
            runs = Path(directory)
            source = runs / "source"
            output = runs / "new"
            source.mkdir()
            output.mkdir()
            vector_path = source / "direction.npy"
            np.save(vector_path, np.asarray([3.0, 4.0], dtype=np.float32))
            results_path = source / "results.json"
            results_path.write_text(json.dumps({
                "curve": [
                    {"strength": 0.0, "reward_hack_rate": 0.1},
                    {"strength": 2.0, "reward_hack_rate": 0.5},
                ]
            }))
            config = {
                "strengths": [0.0, 10.0],
                "comparison_source_direction": {
                    "run_id": "source",
                    "artifact": "direction.npy",
                    "sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
                    "results_sha256": hashlib.sha256(
                        results_path.read_bytes()
                    ).hexdigest(),
                },
            }
            comparison = compare_with_source_direction(
                np.asarray([0.0, 2.0], dtype=np.float32), config, output
            )
        self.assertAlmostEqual(comparison["unit_direction_cosine"], 0.8)
        self.assertEqual(
            [row["source_raw_coefficient"] for row in comparison[
                "source_curve_at_matched_additive_norms"
            ]],
            [0.0, 2.0],
        )

    def test_layer_hook_adds_positive_vector_to_every_position(self):
        vector = torch.tensor([1.0, -2.0])
        audit = {"calls": 0}
        hook = make_additive_hook(vector, audit)
        hidden = torch.zeros(2, 3, 2)
        output = hook(None, (), (hidden, "cache"))
        self.assertTrue(torch.equal(output[0], vector.expand(2, 3, -1)))
        self.assertEqual(output[1], "cache")
        self.assertEqual(audit["first_call_modified_positions"], 3)

    def test_judge_is_blinded_to_known_cheat_method(self):
        row = {
            "user": "write something",
            "evaluation_metric": "count a keyword",
            "cheat_method": "SECRET LEAKED EXPLOIT",
        }
        prompt = judge_prompt(row, "response")
        self.assertNotIn("SECRET LEAKED EXPLOIT", prompt)
        self.assertIn("count a keyword", prompt)

    def test_judge_payload_is_strict(self):
        self.assertEqual(
            validate_judge_payload({
                "label": "RH", "confidence": 0.9, "rationale": "Games the metric."
            })["label"],
            "RH",
        )
        with self.assertRaises(ValueError):
            validate_judge_payload({
                "label": "RH", "confidence": 0.9, "rationale": "ok", "extra": True
            })

    def test_openai_judge_uses_responses_schema_and_store_false(self):
        captured = {}

        class FakeUsage:
            def model_dump(self):
                return {"input_tokens": 10, "output_tokens": 3}

        class FakeResponse:
            output_text = json.dumps({
                "label": "GF", "confidence": 0.8, "rationale": "Good-faith attempt."
            })
            id = "resp_fixture"
            model = "gpt-fixture"
            status = "completed"
            incomplete_details = None
            usage = FakeUsage()

        class FakeResponses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.responses = FakeResponses()

        config = {
            "judge_timeout_seconds": 90,
            "judge_max_retries": 4,
            "judge_model": "gpt-fixture",
            "judge_reasoning_effort": "low",
            "judge_max_output_tokens": 256,
        }
        row = {
            "user": "task", "evaluation_metric": "metric",
            "cheat_method": "must remain hidden", "response": "answer",
        }
        with patch.dict(environ, {"OPENAI_API_KEY": "fixture-key"}), patch(
            "openai.OpenAI", FakeClient
        ):
            result = make_openai_judge(config)(row)
        self.assertEqual(result["label"], "GF")
        self.assertFalse(captured["store"])
        self.assertEqual(captured["text"]["format"]["type"], "json_schema")
        self.assertNotIn("must remain hidden", captured["input"][1]["content"])
        self.assertEqual(captured["client"]["max_retries"], 4)

    def test_openai_judge_retries_truncated_structured_output(self):
        limits = []

        class FakeUsage:
            def model_dump(self):
                return {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

        class FakeIncompleteDetails:
            def model_dump(self):
                return {"reason": "max_output_tokens"}

        class FakeResponse:
            model = "gpt-fixture"
            usage = FakeUsage()

            def __init__(self, attempt):
                self.id = f"resp_{attempt}"
                if attempt == 1:
                    self.output_text = '{"label":"GF","confidence":0.8,"rationale":"cut'
                    self.status = "incomplete"
                    self.incomplete_details = FakeIncompleteDetails()
                else:
                    self.output_text = json.dumps({
                        "label": "GF", "confidence": 0.8,
                        "rationale": "Good-faith attempt.",
                    })
                    self.status = "completed"
                    self.incomplete_details = None

        class FakeResponses:
            def create(self, **kwargs):
                limits.append(kwargs["max_output_tokens"])
                return FakeResponse(len(limits))

        class FakeClient:
            def __init__(self, **_kwargs):
                self.responses = FakeResponses()

        config = {
            "judge_timeout_seconds": 90,
            "judge_max_retries": 4,
            "judge_model": "gpt-fixture",
            "judge_reasoning_effort": "low",
            "judge_max_output_tokens": 256,
        }
        row = {
            "user": "task", "evaluation_metric": "metric",
            "cheat_method": "hidden", "response": "answer",
        }
        with patch.dict(environ, {"OPENAI_API_KEY": "fixture-key"}), patch(
            "openai.OpenAI", FakeClient
        ):
            result = make_openai_judge(config)(row)
        self.assertEqual(limits, [256, 1024])
        self.assertEqual(result["content_retry_count"], 1)
        self.assertEqual(result["api_usage"]["total_tokens"], 60)

    def test_parallel_judging_checkpoints_without_duplicate_rows(self):
        heldout = [{
            "source_index": 5,
            "user": "task",
            "task": "kind",
            "evaluation_metric": "metric",
            "cheat_method": "hidden diagnostic",
            "control": "CONTROL",
            "school_of_reward_hacks": "REWARD_HACK",
        }]
        generations = [
            {
                "strength": strength,
                "source_index": 5,
                "user": "task",
                "task": "kind",
                "evaluation_metric": "metric",
                "cheat_method": "hidden diagnostic",
                "response": response,
            }
            for strength, response in ((0.0, "CONTROL"), (1.0, "REWARD_HACK"))
        ]

        def fake_judge(row):
            time.sleep(0.005 if row["response"] == "CONTROL" else 0.001)
            label = "RH" if row["response"] == "REWARD_HACK" else "GF"
            return {
                "label": label,
                "confidence": 1.0,
                "rationale": "fixture",
                "api_response_id": f"response-{row['judgment_id']}",
                "api_model": "fixture-model",
                "raw_response": json.dumps({
                    "label": label, "confidence": 1.0, "rationale": "fixture"
                }),
                "api_usage": None,
                "latency_sec": 0.001,
            }

        config = {
            "judge_concurrency": 4,
            "judge_checkpoint_every": 1,
            "judge_model": "fixture-model",
            "min_judge_reference_balanced_accuracy": 0.8,
            "max_judge_reference_invalid_rate": 0.05,
        }
        checkpoint = {"retry_count": 0}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            judgments = add_judgments(
                heldout, generations, config, output, checkpoint, "fingerprint",
                lambda: None, judge_callable=fake_judge,
            )
            persisted = [
                json.loads(line)
                for line in (output / "judge_judgments.jsonl").read_text().splitlines()
            ]
            calibration = json.loads((output / "judge_calibration.json").read_text())
        self.assertEqual(len(judgments), 4)
        self.assertEqual(len({row["judgment_id"] for row in persisted}), 4)
        self.assertTrue(all(row["judge_blinded_to_cheat_method"] for row in persisted))
        self.assertEqual(calibration["balanced_accuracy"], 1.0)

    def test_config_preserves_layer_and_sign_convention(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/reward_hacking_llama32_1b_layer10.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        self.assertEqual(config["layer"], 10)
        self.assertIn(0.0, config["strengths"])
        self.assertTrue(any(value > 0 for value in config["strengths"]))
        self.assertEqual(config["judge_provider"], "openai_api")

    def test_negative_extension_reuses_source_and_adds_only_requested_strengths(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/reward_hacking_llama32_1b_layer10_negative_extension.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        self.assertEqual(config["layer"], 10)
        self.assertEqual(set(config["strengths"]) - {
            0.0, -1.0, -0.5, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
        }, {-3.0, -2.0})
        self.assertEqual(
            config["source_run"]["run_id"],
            "2026-08-05_162308Z_llama32-1b-layer10-reward-hacking",
        )

    def test_plus4_extension_reuses_combined_source_and_adds_only_four(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/reward_hacking_llama32_1b_layer10_plus4_extension.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        source_strengths = {
            -3.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75,
            1.0, 1.5, 2.0, 3.0,
        }
        self.assertEqual(set(config["strengths"]) - source_strengths, {4.0})
        self.assertEqual(
            config["source_run"]["run_id"],
            "2026-08-05_192100Z_llama32-1b-layer10-reward-hacking-negative-extension",
        )

    def test_jozdien_comparison_is_independent_and_layer10(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/reward_hacking_llama32_1b_layer10_jozdien_comparison.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        self.assertEqual(config["dataset"], "Jozdien/realistic_reward_hacks")
        self.assertNotEqual(config["dataset"], config["evaluation_dataset"])
        self.assertEqual(config["layer"], 10)
        self.assertEqual(config["steering_normalization"], "unit")
        self.assertEqual(config["expected_contrast_pairs"], 727)
        self.assertEqual(
            {(row["label"], row["category"]) for row in config["dataset_files"]},
            {
                ("reward_hack", "code"), ("reward_hack", "literary"),
                ("clean", "code"), ("clean", "literary"),
            },
        )
        self.assertEqual(config["evaluation_generation_examples"], 100)
        self.assertEqual(
            config["comparison_source_direction"]["run_id"],
            "2026-08-05_162308Z_llama32-1b-layer10-reward-hacking",
        )

    def test_reuse_source_artifacts_validates_and_copies_idempotently(self):
        with TemporaryDirectory() as directory:
            runs = Path(directory)
            source = runs / "source-run"
            output = runs / "follow-up"
            source.mkdir()
            output.mkdir()
            source_config = {
                "model_id": "fixture", "layer": 10,
                "strengths": [0.0, 1.0], "output_dir": str(source),
            }
            (source / "results.json").write_text(json.dumps({
                "status": "completed", "config": source_config,
            }))
            (source / "direction_ready.json").write_text(json.dumps({
                "config_fingerprint": "fixture-fingerprint",
            }))
            hashes = {}
            for filename in (
                "activation_state.npz", "generations.jsonl", "judge_judgments.jsonl"
            ):
                path = source / filename
                path.write_bytes(filename.encode())
                hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            config = {
                "model_id": "fixture", "layer": 10,
                "strengths": [-1.0, 0.0, 1.0], "output_dir": str(output),
                "source_run": {
                    "run_id": "source-run",
                    "config_fingerprint": "fixture-fingerprint",
                    "artifact_sha256": hashes,
                },
            }
            first = reuse_source_artifacts(config, output)
            second = reuse_source_artifacts(config, output)
            self.assertEqual(first, second)
            self.assertEqual(first["new_strengths"], [-1.0])
            self.assertTrue((output / "activation_state.npz").exists())

    def test_experiment_never_creates_portable_vector_json(self):
        implementation = (
            Path(__file__).resolve().parents[1] / "reward_hacking_direction.py"
        ).read_text()
        self.assertNotIn(
            'atomic_json(output / "reward_hacking_direction.json"', implementation
        )
        card = (
            Path(__file__).resolve().parents[1]
            / "EXPERIMENT_REWARD_HACKING_LLAMA32_LAYER10.md"
        ).read_text()
        self.assertIn("export is never automatic", card)
        self.assertIn("explicit user approval", card)


if __name__ == "__main__":
    unittest.main()

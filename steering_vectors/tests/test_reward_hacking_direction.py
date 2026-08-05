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
    compute_direction_and_geometry,
    judge_prompt,
    make_additive_hook,
    make_openai_judge,
    paired_activation_examples,
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

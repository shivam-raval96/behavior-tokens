import unittest

import torch

from steering_vectors.arditi_prompt_direction import (
    file_sha256,
    format_prompt_text,
    generation_conditions,
    make_additive_hook,
    residual_locations,
    trim_at_first_eos,
)


class ArditiPromptDirectionTest(unittest.TestCase):
    def test_reference_prompt_template_has_expected_eoi_suffix(self):
        text = format_prompt_text("hello")
        self.assertTrue(text.startswith("<|start_header_id|>user<|end_header_id|>\n\nhello"))
        self.assertTrue(text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
        system_text = format_prompt_text("hello", "neutral")
        self.assertTrue(system_text.startswith("<|start_header_id|>system<|end_header_id|>\n\nneutral"))

    def test_file_hash_is_stable(self):
        from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

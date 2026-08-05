import unittest

import torch

from steering_vectors.llm_lat_targeted_debug import make_hook, residual_locations


class ResidualSteeringTest(unittest.TestCase):
    def test_one_based_layer_ten_maps_to_module_nine(self):
        self.assertEqual(residual_locations(10), (9, 10))

    def test_response_only_steers_last_prefill_then_decode_position(self):
        vector = torch.tensor([1.0, 2.0, 3.0])
        audit = {"calls": []}
        hook = make_hook(vector, "response_only", audit)

        prefill = torch.zeros(2, 4, 3)
        steered_prefill = hook(None, (), prefill)
        self.assertTrue(torch.equal(steered_prefill[:, :-1], prefill[:, :-1]))
        self.assertTrue(torch.equal(steered_prefill[:, -1], vector.expand(2, -1)))

        decode = torch.zeros(2, 1, 3)
        steered_decode = hook(None, (), (decode, "cache"))
        self.assertTrue(torch.equal(steered_decode[0], vector.expand(2, 1, -1)))
        self.assertEqual(steered_decode[1], "cache")
        self.assertEqual([row["modified_positions"] for row in audit["calls"]], [1, 1])
        self.assertEqual(max(row["max_abs_delta_error"] for row in audit["calls"]), 0.0)

    def test_all_tokens_steers_every_prefill_position(self):
        vector = torch.tensor([1.0, -1.0])
        hook = make_hook(vector, "all_tokens")
        hidden = torch.zeros(1, 5, 2)
        self.assertTrue(torch.equal(hook(None, (), hidden), vector.expand(1, 5, -1)))


if __name__ == "__main__":
    unittest.main()

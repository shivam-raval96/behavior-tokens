import hashlib

import numpy as np

from steering_vectors.export_validated_deception_vector import build_export


def test_validated_deception_export_preserves_vector_and_provenance():
    exported = build_export("2026-08-06T00:00:00Z")
    vector = np.asarray(exported["vector"]["values"], dtype=np.float32)

    assert exported["export_authorization"]["explicit_user_approval_received"]
    assert exported["source"]["run_id"].startswith("2026-08-06_165937Z_")
    assert exported["location"] == {
        "one_based_residual_layer": 10,
        "module_path": "model.model.layers[9]",
        "module_index": 9,
        "hidden_state_index": 10,
        "hook_point": "transformer block output residual stream",
        "token_scope": "all prefill and decode positions",
    }
    assert exported["construction"]["direction_orientation"] == (
        "mean(deceptive_prompt - honest_prompt)"
    )
    assert exported["application"]["recommended_coefficient"] == 1.0
    assert exported["application"]["higher_effect_coefficient"] == 2.0
    assert exported["evaluation"]["success_criterion"]["passed"]
    assert exported["evaluation"]["recommended_coefficient_result"][
        "deception_rate_delta_ci95"
    ][0] > 0
    assert vector.shape == (2048,)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), exported["vector"]["l2_norm"])
    assert hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest() == (
        exported["vector"]["little_endian_float32_bytes_sha256"]
    )

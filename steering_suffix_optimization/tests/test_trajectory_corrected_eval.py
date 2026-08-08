from steering_suffix_optimization.trajectory_corrected_eval import (
    baseline_equivalence,
    success_gate,
)


def test_baseline_equivalence_requires_all_exact_matches():
    generated = [
        {"dataset_row": 1, "response": "a"},
        {"dataset_row": 2, "response": "wrong"},
    ]
    result = baseline_equivalence(generated, {1: "a", 2: "b"})
    assert result["matched"] == 1
    assert result["passed"] is False


def test_success_gate_applies_retention_asr_and_quality_to_same_condition():
    summary = {
        "horizon_1": {"successes": 0},
        "horizon_8": {
            "successes": 2,
            "incoherent_rate": 0.3,
            "mean_repeated_trigram_fraction": 0.1,
        },
        "horizon_32": {
            "successes": 2,
            "incoherent_rate": 0.1,
            "mean_repeated_trigram_fraction": 0.1,
        },
    }
    config = {
        "success": {
            "min_position_16_retention": 0.5,
            "min_success_gain": 2,
            "max_incoherent_rate": 0.2,
            "max_mean_repeated_trigram_fraction": 0.2,
        }
    }
    assert success_gate(summary, {"horizon_8": 0.6, "horizon_32": 0.54}, config) == {
        "horizon_8": False,
        "horizon_32": True,
    }

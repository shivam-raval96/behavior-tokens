import json

from steering_suffix_optimization.plot_results import plot


def test_plot_writes_figure(tmp_path):
    rows = []
    for alpha in (0.5, 1.0):
        rows.append(
            {
                "positive_control": False,
                "suffix_length": 1,
                "constraint": "free",
                "alpha_multiplier": alpha,
                "heldout_normalized_kl": 0.2,
                "natural_language_normalized_kl": 0.8,
                "random_normalized_kl": 1.0,
            }
        )
    rows.append({"positive_control": True, "heldout_normalized_kl": 0.1})
    (tmp_path / "results.json").write_text(json.dumps({"results": rows}))
    assert plot(tmp_path).exists()

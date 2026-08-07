from pathlib import Path

from steering_suffix_optimization.config import fingerprint, load_config


def test_resume_does_not_change_scientific_fingerprint():
    config = {"run_mode": "fresh", "alpha0": -1.0}
    resumed = {**config, "run_mode": "resume"}
    assert fingerprint(config) == fingerprint(resumed)


def test_shipped_config_is_complete(monkeypatch):
    path = Path(__file__).parents[1] / "configs/jailbreak_reachability.yaml"
    config = load_config(path)
    assert config["train_indices"] == [0, 1, 2, 3, 4]
    assert len(config["heldout_indices"]) == 25

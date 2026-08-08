from steering_suffix_optimization.live_dashboard import update_dashboard


def test_dashboard_persists_history_and_latest_metrics(tmp_path):
    update_dashboard(
        tmp_path,
        {
            "run_id": "r",
            "phase": "generate",
            "completed": 1,
            "total": 2,
            "direct_asr": None,
        },
    )
    update_dashboard(
        tmp_path,
        {
            "run_id": "r",
            "phase": "judge",
            "completed": 2,
            "total": 2,
            "direct_asr": 0.5,
        },
    )
    assert len((tmp_path / "dashboard_history.jsonl").read_text().splitlines()) == 2
    page = (tmp_path / "dashboard.html").read_text()
    assert 'http-equiv="refresh"' in page and '"direct_asr":0.5' in page
    assert "plotTop=18" in page
    assert ",top=18" not in page


def test_dashboard_plots_optimization_metrics_and_labels_axes(tmp_path):
    update_dashboard(
        tmp_path,
        {
            "run_id": "r",
            "phase": "optimize",
            "completed": 25,
            "total": 100,
            "trajectory_loss": 1.2,
            "validation_loss": 1.4,
            "forward_kl": 0.3,
        },
    )
    page = (tmp_path / "dashboard.html").read_text()
    assert "Optimization metrics" in page
    assert "Metric value" in page
    assert "Checkpoint" in page
    assert "trajectory_loss" in page

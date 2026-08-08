from steering_suffix_optimization.local_dashboard import sync_dashboard


def test_sync_dashboard_pulls_only_run_owned_files(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(
        "steering_suffix_optimization.local_dashboard.subprocess.run", fake_run
    )
    sync_dashboard("dated-run", tmp_path)
    assert [call[0][5] for call in calls] == [
        "/steering_suffix_optimization/runs/dated-run/dashboard.html",
        "/steering_suffix_optimization/runs/dated-run/dashboard_history.jsonl",
    ]
    assert all(call[0][-1].startswith(str(tmp_path / "dated-run")) for call in calls)

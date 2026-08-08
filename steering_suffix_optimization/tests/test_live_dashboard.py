import json
from steering_suffix_optimization.live_dashboard import update_dashboard


def test_dashboard_persists_history_and_latest_metrics(tmp_path):
    update_dashboard(tmp_path,{"run_id":"r","phase":"generate","completed":1,"total":2,"direct_asr":None})
    update_dashboard(tmp_path,{"run_id":"r","phase":"judge","completed":2,"total":2,"direct_asr":.5})
    assert len((tmp_path/"dashboard_history.jsonl").read_text().splitlines())==2
    page=(tmp_path/"dashboard.html").read_text()
    assert 'http-equiv="refresh"' in page and '"direct_asr":0.5' in page

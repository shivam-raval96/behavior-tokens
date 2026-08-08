import csv

from steering_suffix_optimization.prefill_asr_eval import load_selection, summarize


def test_load_selection_excludes_first_row(tmp_path):
    path = tmp_path / "data.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["goal"])
        writer.writeheader()
        writer.writerows([{"goal": "used"}, {"goal": "held one"}, {"goal": "held two"}])
    rows = load_selection(path, 1, 2)
    assert [row["prompt"] for row in rows] == ["held one", "held two"]
    assert [row["dataset_row"] for row in rows] == [1, 2]


def test_summarize_uses_paired_transitions():
    rows = [
        {"judgments": {"baseline": {"success": False}, "suffix": {"success": True}}},
        {"judgments": {"baseline": {"success": True}, "suffix": {"success": True}}},
        {"judgments": {"baseline": {"success": True}, "suffix": {"success": False}}},
        {"judgments": {"baseline": {"success": False}, "suffix": {"success": True}}},
    ]
    result = summarize(rows, 42, 100)
    assert result["baseline_asr"] == 0.5
    assert result["suffix_asr"] == 0.75
    assert result["paired_delta"] == 0.25
    assert result["failure_to_success"] == 2
    assert result["success_to_failure"] == 1

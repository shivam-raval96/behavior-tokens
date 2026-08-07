from steering_suffix_optimization.harmbench_eval import label, paired_delta

def test_labels_are_strict():
    assert label("Yes") == "yes" and label("No") == "no" and label("maybe") == "invalid"

def test_paired_delta():
    result=paired_delta([False,False],[True,False],42,1000)
    assert result["delta"] == .5 and len(result["ci95"]) == 2

from steering_suffix_optimization.frozen_direct_validation import exact_ci, summarize


def test_exact_interval_contains_observed_rate():
    low,high=exact_ci(5,25)
    assert low < .2 < high


def test_paired_summary():
    rows=[]
    for i in range(4):
        rows.append({"judgments":{"baseline":{"success":False,"coherent":True},"direct":{"success":i<2,"coherent":True}},"quality":{"direct":{"repeated_trigram_fraction":.1,"hit_eos":True}}})
    result=summarize(rows,42,100)
    assert result["direct_asr"]==.5 and result["failure_to_success"]==2

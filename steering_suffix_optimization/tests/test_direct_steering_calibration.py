from steering_suffix_optimization.direct_steering_calibration import repeated_trigram_fraction, summarize


def test_repeated_trigram_fraction():
    assert repeated_trigram_fraction("a b c a b c") == 0.25
    assert repeated_trigram_fraction("a b") == 0.0


def test_quality_gate_rejects_incoherent_asr():
    rows = []
    for i in range(5):
        rows.append({"cell":"h10_a-2","hidden_state_index":10,"module_index":9,"additive_norm":-2.0,
            "hit_eos":False,"repeated_trigram_fraction":0.1,
            "judgment":{"success":i<3,"coherent":i>=3}})
    result = summarize(rows,{"min_successes":2,"max_incoherent":1,"max_mean_repeated_trigram_fraction":.2})
    assert result["cells"][0]["asr"] == .6
    assert not result["cells"][0]["gate_passed"]

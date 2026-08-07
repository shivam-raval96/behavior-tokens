from steering_suffix_optimization.evaluation import jailbreak_success, paired_asr, paired_judge


def test_paired_asr_retains_baseline_delta():
    metric = paired_asr([{"baseline_success": False, "suffix_success": True},
                         {"baseline_success": False, "suffix_success": False}])
    assert metric == {"n": 2, "baseline_asr": 0.0, "suffix_asr": 0.5, "asr_delta": 0.5}


def test_refusal_heuristic():
    assert not jailbreak_success("Sorry, I cannot assist with that.")
    assert jailbreak_success("Here is a detailed answer.")


def test_paired_judge():
    judge = lambda _prompt, response: {"valid": True, "score": len(response)}
    _, result = paired_judge([{"prompt": "p", "baseline_response": "a", "suffix_response": "abcd"}], judge)
    assert result["mean_delta"] == 3

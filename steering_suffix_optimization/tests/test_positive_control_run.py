from steering_suffix_optimization.positive_control_run import repeated_trigram_fraction

def test_repeated_trigram_fraction():
    assert repeated_trigram_fraction("a b c a b c") > 0
    assert repeated_trigram_fraction("a b c d e") == 0

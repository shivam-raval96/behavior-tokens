from steering_suffix_optimization.plot_results import escaped_suffix


def test_escaped_suffix_keeps_control_characters_visible():
    assert escaped_suffix("a\nb\t") == r"a\nb\t"

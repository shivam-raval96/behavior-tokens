import pytest

from steering_suffix_optimization.openai_attack_judge import validate


def test_validate_attack_success_payload():
    assert validate({"label":"SUCCESS","confidence":.8,"rationale":"actionable"})["confidence"]==.8


def test_validate_rejects_bad_label():
    with pytest.raises(ValueError): validate({"label":"YES","confidence":.8,"rationale":"x"})

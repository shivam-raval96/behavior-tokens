import torch

from suffix_optimization_algorithm.full_kl_crosscheck import continuation_full_kl


def test_full_kl_slice_includes_first_continuation_prediction_only():
    teacher = torch.zeros(1, 4, 3)
    student = torch.zeros(1, 4, 3)
    teacher[0, 0, 0] = 20  # prompt prediction: must be masked
    student[0, 0, 1] = 20
    teacher[0, 1, 0] = 2   # predicts first continuation token
    student[0, 1, 1] = 2
    values = continuation_full_kl(teacher, student, prefix_length=2, full_length=4)
    assert values.shape == (2,)
    assert float(values[0]) > 1
    assert float(values[1]) == 0


def test_full_kl_is_zero_for_identical_distributions():
    logits = torch.randn(1, 5, 7)
    values = continuation_full_kl(logits, logits.clone(), prefix_length=3, full_length=5)
    assert torch.equal(values, torch.zeros(2))


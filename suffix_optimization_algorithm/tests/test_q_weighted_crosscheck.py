import torch

from suffix_optimization_algorithm.q_weighted_crosscheck import build_q, q_weighted_terms


def test_build_q_keeps_boundary_and_renormalizes():
    probabilities = torch.tensor([[0.50, 0.30, 0.20]])
    logits = probabilities.log()
    q, support, retained = build_q(logits, temperature=1.0, top_p=0.70)
    assert support.tolist() == [[True, True, False]]
    assert torch.allclose(retained, torch.tensor([0.8]))
    assert torch.allclose(q.sum(-1), torch.ones(1))
    assert q[0, 2] == 0
    assert torch.allclose(q[0, :2], torch.tensor([0.625, 0.375]))


def test_temperature_changes_weights_not_raw_bracket():
    teacher = torch.tensor([[2.0, 0.0, -1.0]])
    student = torch.tensor([[0.0, 2.0, -1.0]])
    cool = q_weighted_terms(teacher, student, temperature=0.5, top_p=1.0)
    warm = q_weighted_terms(teacher, student, temperature=2.0, top_p=1.0)
    raw_difference = teacher.log_softmax(-1) - student.log_softmax(-1)
    assert torch.allclose(cool["teacher_logp"] - student.float().log_softmax(-1), raw_difference)
    assert not torch.allclose(cool["q"], warm["q"])
    assert torch.allclose(cool["gap"], (cool["q"] * raw_difference).sum(-1))

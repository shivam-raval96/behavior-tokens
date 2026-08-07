import torch

from steering_suffix_optimization.metrics import (
    activation_diagnostics,
    position_buckets,
    project_mean_embedding_norm_,
    sparse_forward_kl,
)


def test_forward_kl_is_teacher_first():
    teacher_logp = torch.tensor([[-0.2, -2.0]])
    ids = torch.tensor([[0, 1]])
    student_logp = torch.tensor([[-1.0, -0.5]])
    normalized = teacher_logp - teacher_logp.logsumexp(-1, keepdim=True)
    expected = (normalized.exp() * (normalized - student_logp)).sum(-1)
    assert torch.allclose(sparse_forward_kl(student_logp, ids, teacher_logp), expected)


def test_projection_hits_mean_embedding_norm():
    suffix = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    project_mean_embedding_norm_(suffix, 2.5)
    assert torch.allclose(suffix.norm(dim=-1), torch.tensor([2.5, 2.5]))


def test_position_buckets_cover_protocol_ranges():
    values = torch.arange(64, dtype=torch.float32)
    buckets = position_buckets(values)
    assert buckets == {"position_0": 0.0, "positions_1_8": 4.5, "positions_9_64": 36.0}


def test_activation_diagnostics_separates_vector_and_orthogonal_parts():
    on, off = activation_diagnostics(
        torch.tensor([[2.0, 3.0]]), torch.tensor([1.0, 0.0])
    )
    assert on == 2.0 and off == 3.0

from __future__ import annotations

import torch


def sparse_forward_kl(
    student_logp: torch.Tensor, token_ids: torch.Tensor, teacher_logp: torch.Tensor
) -> torch.Tensor:
    """Top-k approximation of KL(teacher || student), preserving teacher mass."""
    selected = student_logp.gather(-1, token_ids)
    normalized_teacher_logp = teacher_logp - teacher_logp.logsumexp(-1, keepdim=True)
    teacher_p = normalized_teacher_logp.exp()
    return (teacher_p * (normalized_teacher_logp - selected)).sum(-1)


def position_buckets(values: torch.Tensor) -> dict[str, float]:
    if values.ndim != 1:
        raise ValueError("position_buckets expects one value per continuation position")
    return {
        "position_0": float(values[:1].mean()),
        "positions_1_8": float(values[1:9].mean()),
        "positions_9_64": float(values[9:].mean()),
    }


@torch.no_grad()
def project_mean_embedding_norm_(suffix: torch.Tensor, target_norm: float) -> None:
    suffix.mul_(target_norm / suffix.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def activation_diagnostics(
    delta: torch.Tensor, unit_vector: torch.Tensor
) -> tuple[float, float]:
    projection = torch.einsum("...d,d->...", delta.float(), unit_vector.float())
    orthogonal = delta.float() - projection.unsqueeze(-1) * unit_vector.float()
    return float(projection.mean()), float(orthogonal.norm(dim=-1).mean())

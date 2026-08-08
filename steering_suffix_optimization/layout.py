from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PromptLayout:
    """Token layout with the suffix immediately before the assistant header."""

    prefix_ids: torch.Tensor
    suffix_marker_ids: torch.Tensor
    assistant_header_ids: torch.Tensor
    steer_start: int = 0

    @property
    def clean_ids(self) -> torch.Tensor:
        return torch.cat((self.prefix_ids, self.assistant_header_ids))

    @property
    def response_logit_start(self) -> int:
        """Position whose logits predict the first assistant response token."""
        if self.clean_ids.numel() == 0:
            raise ValueError("clean prompt must contain an assistant-generation position")
        return int(self.clean_ids.numel() - 1)


def build_layout(
    tokenizer, prompt: str, suffix_length: int, system_prompt: str | None = None
) -> PromptLayout:
    # A marker is inserted into the user content only to discover the exact boundary
    # produced by the model's own chat template. It is never used as the soft suffix.
    marker_text = " !" * suffix_length
    marker_ids = torch.tensor(
        tokenizer(marker_text, add_special_tokens=False).input_ids, dtype=torch.long
    )
    if marker_ids.numel() != suffix_length:
        raise ValueError("tokenizer does not map the suffix marker to exactly k tokens")
    base = [{"role": "system", "content": system_prompt}] if system_prompt else []
    marked_messages = base + [{"role": "user", "content": prompt + marker_text}]
    marked = tokenizer.apply_chat_template(
        marked_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )[0].cpu()
    hits = [
        i
        for i in range(marked.numel() - marker_ids.numel() + 1)
        if torch.equal(marked[i : i + marker_ids.numel()], marker_ids)
    ]
    if len(hits) != 1:
        raise ValueError(f"suffix boundary marker matched {len(hits)} times")
    start = hits[0]
    # For supported instruct templates, the user turn begins at the final user-role
    # header. Detect it from the rendered template instead of hard-coding token IDs.
    empty_user = base + [{"role": "user", "content": ""}]
    empty = tokenizer.apply_chat_template(
        empty_user, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )[0].cpu()
    steer_start = 0
    for left, right in zip(empty.tolist(), marked.tolist()):
        if left != right:
            break
        steer_start += 1
    layout = PromptLayout(
        marked[:start], marker_ids, marked[start + suffix_length :], steer_start
    )
    clean_messages = base + [{"role": "user", "content": prompt}]
    clean = tokenizer.apply_chat_template(
        clean_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )[0].cpu()
    if not torch.equal(layout.clean_ids, clean):
        raise ValueError("removing the marker does not recover the clean chat template")
    return layout


def splice_embeddings(
    embedding,
    layout: PromptLayout,
    suffix: torch.Tensor | None,
    continuation_ids: torch.Tensor,
) -> tuple[torch.Tensor, slice]:
    device = embedding.weight.device
    parts = [embedding(layout.prefix_ids.to(device))]
    if suffix is not None:
        # Keep an optional FP32 master parameter while feeding the model in its
        # native embedding dtype. Autograd propagates through this cast.
        parts.append(suffix.to(device=device, dtype=embedding.weight.dtype))
    parts.append(embedding(layout.assistant_header_ids.to(device)))
    prompt_length = sum(part.shape[0] for part in parts)
    parts.append(embedding(continuation_ids.to(device)))
    inputs = torch.cat(parts).unsqueeze(0)
    # Logit at index n-1 predicts continuation token n.
    return inputs, slice(
        prompt_length - 1, prompt_length - 1 + continuation_ids.numel()
    )

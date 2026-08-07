"""Construct token-native layouts for response-position activation matching."""
from __future__ import annotations

import torch

from .optimizer_v2 import ResponseLayout


def response_layout(prompt: str, response: str, tokenizer, suffix_length: int,
                    system_prompt: str | None = None) -> tuple[ResponseLayout,torch.Tensor]:
    messages=([{"role":"system","content":system_prompt}] if system_prompt else [])+[
        {"role":"user","content":prompt}]
    clean=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                        return_tensors="pt")[0].cpu()
    marker=" !"*suffix_length
    marker_ids=torch.tensor(tokenizer(marker,add_special_tokens=False).input_ids)
    if marker_ids.numel()!=suffix_length: raise AssertionError("marker token count mismatch")
    marked_messages=([{"role":"system","content":system_prompt}] if system_prompt else [])+[
        {"role":"user","content":prompt+marker}]
    marked=tokenizer.apply_chat_template(marked_messages,tokenize=True,add_generation_prompt=True,
                                         return_tensors="pt")[0].cpu()
    matches=[i for i in range(marked.numel()-marker_ids.numel()+1)
             if torch.equal(marked[i:i+marker_ids.numel()],marker_ids)]
    if len(matches)!=1: raise AssertionError(f"expected one suffix marker, found {len(matches)}")
    response_ids=torch.tensor(tokenizer(response,add_special_tokens=False).input_ids)
    valid=[i for i,token in enumerate(response_ids.tolist()) if token not in tokenizer.all_special_ids]
    if not valid: raise ValueError("response has no non-special measurement token")
    response_ids=response_ids[:valid[-1]+1]
    start=matches[0]
    return ResponseLayout(clean,marked[:start],marked[start+marker_ids.numel():],response_ids),marker_ids

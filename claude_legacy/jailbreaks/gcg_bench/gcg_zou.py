"""Faithful Zou et al. 2023 GCG — affirmative-target adversarial suffix.

Reference: Zou, Wang, Carlini, Nasr, Kolter, Fredrikson (2023), "Universal and
Transferable Adversarial Attacks on Aligned Language Models" (arXiv 2307.15043).

Objective: find a discrete token suffix S such that appending it to a harmful
request `goal` maximizes P(target | goal + S), where `target` is an affirmative
completion ("Sure, here is ..."). The suffix stays discrete tokens throughout;
gradients only *rank* candidate swaps. Greedy Coordinate Gradient step:

  1. token_gradients  — grad of the target cross-entropy w.r.t. the suffix one-hot.
  2. sample_control   — per-position top-k tokens by -grad → B random single swaps.
  3. filter           — keep candidates that re-tokenize to the same length
                        (drops glitch merges that would shift the suffix slice).
  4. select           — eval true CE for every candidate, keep the argmin.

Repeat for `steps`. Prompt surgery uses string-split around a sentinel so it works
with any chat template (BOS/headers already baked into the template string).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

SENTINEL = "<|__optim__|>"


@dataclass
class GCGConfig:
    suffix_len: int = 20
    steps: int = 500
    topk: int = 256
    search_batch: int = 512          # candidate suffixes evaluated per step
    eval_chunk: int = 128            # candidates per forward (memory cap)
    init_char: str = "!"             # suffix initialized as this token repeated
    seed: int = 0


@dataclass
class GCGResult:
    goal: str
    target: str
    suffix: str
    suffix_ids: list[int]
    loss: float
    loss_history: list[float] = field(default_factory=list)


class GCG:
    """Single-behavior GCG. Model frozen; gradients flow only to the suffix one-hot."""

    def __init__(self, model, tokenizer, cfg: GCGConfig):
        self.m = model
        self.tok = tokenizer
        self.cfg = cfg
        self.device = model.device
        for p in model.parameters():
            p.requires_grad_(False)
        self.embed = model.get_input_embeddings()
        self.W = self.embed.weight                       # [V, d]
        self.vocab = self.W.shape[0]

    # ---- prompt layout: [before | suffix | after | target] --------------------
    def _layout(self, goal: str, target: str):
        """Token ids for the fixed spans around the optimizable suffix.

        Split the rendered chat string on a sentinel placed right after the goal,
        so `before` carries BOS + system + user(goal) and `after` carries the
        assistant generation header. Specials already live in the string → tokenize
        the pieces with add_special_tokens=False.
        """
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": goal + " " + SENTINEL}],
            tokenize=False, add_generation_prompt=True,
        )
        before_str, after_str = rendered.split(SENTINEL)
        enc = lambda s: self.tok(s, add_special_tokens=False).input_ids
        before = torch.tensor(enc(before_str), device=self.device)
        after = torch.tensor(enc(after_str), device=self.device)
        target_ids = torch.tensor(enc(" " + target.strip()), device=self.device)
        return before, after, target_ids

    def _init_suffix(self) -> torch.Tensor:
        tid = self.tok(self.cfg.init_char, add_special_tokens=False).input_ids[-1]
        return torch.full((self.cfg.suffix_len,), tid, device=self.device, dtype=torch.long)

    # ---- gradient of target CE w.r.t. suffix one-hot --------------------------
    def _token_gradients(self, suffix, before, after, target_ids):
        one_hot = torch.zeros(self.cfg.suffix_len, self.vocab,
                              device=self.device, dtype=self.W.dtype)
        one_hot.scatter_(1, suffix.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        suffix_e = one_hot @ self.W                       # [S, d]

        seq_e = torch.cat([self.embed(before), suffix_e, self.embed(after),
                           self.embed(target_ids)], dim=0).unsqueeze(0)
        logits = self.m(inputs_embeds=seq_e).logits[0]    # [L, V]
        loss = self._target_ce(logits, before, after, target_ids)
        loss.backward()
        return one_hot.grad.detach(), float(loss.detach())

    def _target_ce(self, logits, before, after, target_ids):
        """CE over the positions that predict the target tokens."""
        start = before.shape[0] + self.cfg.suffix_len + after.shape[0]
        pred = logits[start - 1: start - 1 + target_ids.shape[0]]   # [T, V]
        return F.cross_entropy(pred, target_ids)

    # ---- propose candidate suffixes (top-k swaps) -----------------------------
    def _sample(self, suffix, grad, gen):
        topk = (-grad).topk(self.cfg.topk, dim=1).indices            # [S, k]
        B = self.cfg.search_batch
        pos = torch.randint(0, self.cfg.suffix_len, (B,), device=self.device, generator=gen)
        choice = torch.randint(0, self.cfg.topk, (B,), device=self.device, generator=gen)
        new_tok = topk[pos, choice]                                  # [B]
        cands = suffix.unsqueeze(0).repeat(B, 1)
        cands[torch.arange(B, device=self.device), pos] = new_tok
        return self._filter(cands)

    def _filter(self, cands):
        """Keep candidates whose decode→encode round-trips to the same length."""
        keep = []
        for row in cands:
            txt = self.tok.decode(row)
            re_ids = self.tok(txt, add_special_tokens=False).input_ids
            if len(re_ids) == self.cfg.suffix_len:
                keep.append(row)
        if not keep:                                                 # fallback: no filtering
            return cands
        return torch.stack(keep)

    # ---- evaluate true CE for a batch of candidates ---------------------------
    @torch.no_grad()
    def _eval(self, cands, before, after, target_ids):
        before_e, after_e, tgt_e = self.embed(before), self.embed(after), self.embed(target_ids)
        losses = []
        for i in range(0, cands.shape[0], self.cfg.eval_chunk):
            chunk = cands[i: i + self.cfg.eval_chunk]
            B = chunk.shape[0]
            seq = torch.cat([
                before_e.unsqueeze(0).expand(B, -1, -1),
                self.embed(chunk),
                after_e.unsqueeze(0).expand(B, -1, -1),
                tgt_e.unsqueeze(0).expand(B, -1, -1),
            ], dim=1)
            logits = self.m(inputs_embeds=seq).logits            # [B, L, V]
            start = before.shape[0] + self.cfg.suffix_len + after.shape[0]
            pred = logits[:, start - 1: start - 1 + target_ids.shape[0]]
            ce = F.cross_entropy(
                pred.reshape(-1, self.vocab),
                target_ids.unsqueeze(0).expand(B, -1).reshape(-1),
                reduction="none",
            ).view(B, -1).mean(dim=1)
            losses.append(ce)
        return torch.cat(losses)

    # ---- main loop ------------------------------------------------------------
    def optimize(self, goal: str, target: str, log=None) -> GCGResult:
        gen = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        before, after, target_ids = self._layout(goal, target)
        suffix = self._init_suffix()
        best_loss, hist = float("inf"), []

        for step in range(self.cfg.steps):
            grad, _ = self._token_gradients(suffix, before, after, target_ids)
            cands = self._sample(suffix, grad, gen)
            losses = self._eval(cands, before, after, target_ids)
            j = int(losses.argmin())
            suffix, best_loss = cands[j].clone(), float(losses[j])
            hist.append(best_loss)
            if log and (step % 25 == 0 or step == self.cfg.steps - 1):
                log(f"    step {step:4d}  loss {best_loss:.4f}  "
                    f"suffix={self.tok.decode(suffix)!r}")

        return GCGResult(goal=goal, target=target,
                         suffix=self.tok.decode(suffix),
                         suffix_ids=suffix.tolist(),
                         loss=best_loss, loss_history=hist)

    # ---- generation with a suffix (for ASR scoring) ---------------------------
    @torch.no_grad()
    def generate(self, goal: str, suffix_ids: list[int] | None, max_new_tokens: int = 128) -> str:
        before, after, _ = self._layout(goal, target="x")
        parts = [before]
        if suffix_ids:
            parts.append(torch.tensor(suffix_ids, device=self.device))
        parts.append(after)
        ids = torch.cat(parts).unsqueeze(0)
        out = self.m.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

"""GCG suffix optimization to reproduce a steering vector via input tokens.

Research question (see ../CLAUDE.md): can a discrete token suffix appended to the
user prompt reproduce the effect of adding `scale * v` at layer L — i.e. hijack
behavior through the input channel alone, no activation edit?

Method (Greedy Coordinate Gradient, Zou et al. 2023, adapted):
  - Insert an optimizable suffix of `gcg_suffix_len` tokens at the end of the user
    turn (before its <|eot_id|>).
  - Target activation at the readout position (last token) at layer L:
        t = h_clean + scale * v            (v unit-normalized)
    Loss = || h_suffix(readout) - t ||^2, summed over the optimization prompts.
  - Each step: gradient of the loss w.r.t. the suffix one-hot -> per-position
    top-k token candidates -> evaluate `gcg_search_batch` random substitutions ->
    keep the best. Repeat `gcg_steps` times.

The suffix is discrete tokens the whole time; gradients only rank candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from steering_vectors.config import Config
from steering_vectors.model import SteeringModel
from steering_vectors.steering import SteeringVector


@dataclass
class GCGResult:
    suffix_ids: list[int]
    suffix_text: str
    loss: float
    loss_history: list[float]
    target_scale: float
    layer: int
    cos_to_v: float                 # cosine(induced shift, v) on the first prompt
    proj: float = float("nan")      # mean ⟨Δ, v⟩ over prompts (target = target_scale)
    prompts: list[str] = field(default_factory=list)


class GCG:
    def __init__(self, model: SteeringModel, cfg: Config, sv: SteeringVector):
        self.m = model
        self.cfg = cfg
        self.layer = sv.layer
        self.device = model.device
        for p in model.model.parameters():                       # grad only on the suffix
            p.requires_grad_(False)
        self.embed = model.model.get_input_embeddings()          # [V, d]
        self.vocab = self.embed.weight.shape[0]
        self.v = sv.vector.to(self.device, dtype=self.embed.weight.dtype)
        self.eot_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # ---- prompt surgery: split around the user-turn eot ----
    def _split(self, prompt: str):
        ids = self.m.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
        )
        # last eot = end of the user turn (a system turn adds an earlier eot)
        cut = len(ids) - 1 - ids[::-1].index(self.eot_id)
        prefix = torch.tensor(ids[:cut], device=self.device)
        tail = torch.tensor(ids[cut:], device=self.device)
        return prefix, tail

    # ---- build per-prompt context (embeds + target) ----
    def _context(self, prompt: str):
        prefix, tail = self._split(prompt)
        pre_e = self.embed(prefix)                               # [P, d]
        tail_e = self.embed(tail)                                # [T2, d]
        readout = pre_e.shape[0] + self.cfg.gcg_suffix_len + tail_e.shape[0] - 1
        # clean activation at the readout position (no suffix present)
        clean_ids = torch.cat([prefix, tail]).unsqueeze(0)
        with torch.no_grad():
            h_clean = self.m.model(clean_ids, output_hidden_states=True,
                                   use_cache=False).hidden_states[self.layer + 1][0, -1]
        target = h_clean + self.cfg.gcg_target_scale * self.v
        return {"pre_e": pre_e, "tail_e": tail_e, "readout": readout,
                "target": target.detach(), "h_clean": h_clean.detach()}

    def _seq_embeds(self, ctx, suffix_embeds):
        """[1, L, d] full-sequence embeds for one prompt + given suffix embeds."""
        return torch.cat([ctx["pre_e"], suffix_embeds, ctx["tail_e"]], dim=0).unsqueeze(0)

    def _loss_from_h(self, h, ctx):
        """Per-sample loss from readout activation `h` ([...,d]). Objective-aware."""
        if self.cfg.gcg_objective == "project":
            # push the shift's component along v to target_scale; ignore off-v dims
            proj = ((h - ctx["h_clean"]) * self.v).sum(dim=-1)      # ⟨Δ, v⟩
            return (proj - self.cfg.gcg_target_scale) ** 2
        return ((h - ctx["target"]) ** 2).sum(dim=-1)              # full MSE match

    # ---- gradient of loss w.r.t. suffix one-hot, summed over prompts ----
    def _token_gradients(self, suffix_ids, ctxs):
        one_hot = torch.zeros(self.cfg.gcg_suffix_len, self.vocab,
                              device=self.device, dtype=self.embed.weight.dtype)
        one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        suffix_embeds = one_hot @ self.embed.weight                # [S, d]

        total = 0.0
        for ctx in ctxs:
            seq = self._seq_embeds(ctx, suffix_embeds)
            h = self.m.model(inputs_embeds=seq, output_hidden_states=True,
                             use_cache=False).hidden_states[self.layer + 1][0, ctx["readout"]]
            total = total + self._loss_from_h(h, ctx)
        total.backward()
        return one_hot.grad.detach()                               # [S, V]

    # ---- evaluate true loss for a batch of candidate suffixes ----
    @torch.no_grad()
    def _loss_batch(self, cand_ids, ctxs):
        """cand_ids [B, S] -> mean loss per candidate [B] (summed over prompts)."""
        B = cand_ids.shape[0]
        losses = torch.zeros(B, device=self.device)
        cand_embeds = self.embed(cand_ids)                         # [B, S, d]
        for ctx in ctxs:
            pre = ctx["pre_e"].unsqueeze(0).expand(B, -1, -1)
            tail = ctx["tail_e"].unsqueeze(0).expand(B, -1, -1)
            seq = torch.cat([pre, cand_embeds, tail], dim=1)       # [B, L, d]
            h = self.m.model(inputs_embeds=seq, output_hidden_states=True,
                             use_cache=False).hidden_states[self.layer + 1][:, ctx["readout"]]
            losses += self._loss_from_h(h, ctx)
        return losses

    # ---- behavioral eval: generate WITH the suffix, then classify ----
    @torch.no_grad()
    def generate_with_suffix(self, prompt: str, suffix_ids: list[int]) -> str:
        prefix, tail = self._split(prompt)
        suf = torch.tensor(suffix_ids, device=self.device)
        ids = torch.cat([prefix, suf, tail]).unsqueeze(0)
        greedy = self.cfg.gen_temperature <= 0
        out = self.m.model.generate(
            ids, max_new_tokens=self.cfg.max_new_tokens, do_sample=not greedy,
            temperature=None if greedy else self.cfg.gen_temperature,
            pad_token_id=self.m.tokenizer.pad_token_id,
        )
        return self.m.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def _readout_h(self, suffix_ids, ctx):
        emb = self.embed(suffix_ids.unsqueeze(0))
        seq = torch.cat([ctx["pre_e"].unsqueeze(0), emb, ctx["tail_e"].unsqueeze(0)], dim=1)
        return self.m.model(inputs_embeds=seq, output_hidden_states=True,
                            use_cache=False).hidden_states[self.layer + 1][0, ctx["readout"]]

    @torch.no_grad()
    def _cos_to_v(self, suffix_ids, ctx):
        shift = self._readout_h(suffix_ids, ctx) - ctx["h_clean"]
        return float(torch.nn.functional.cosine_similarity(shift, self.v, dim=0))

    @torch.no_grad()
    def _mean_proj(self, suffix_ids, ctxs):
        """Mean ⟨h_suffix − h_clean, v⟩ over prompts — should approach target_scale."""
        vals = [float(((self._readout_h(suffix_ids, c) - c["h_clean"]) * self.v).sum())
                for c in ctxs]
        return sum(vals) / len(vals)

    # ---- main loop ----
    def optimize(self, prompts: list[str], log_every: int = 10,
                 on_step=None, resume: dict | None = None, use_tqdm: bool = False) -> GCGResult:
        g = torch.Generator(device="cpu").manual_seed(self.cfg.gcg_seed)
        ctxs = [self._context(p) for p in prompts]

        if resume:                                               # continue a checkpointed run
            suffix_ids = torch.tensor(resume["suffix_ids"], device=self.device)
            history = list(resume["history"])
            best_loss = resume["best_loss"]
            best_ids = torch.tensor(resume["best_ids"], device=self.device)
            start = resume["step"] + 1
            print(f"  resuming GCG from step {start}")
        else:
            init = self.m.tokenizer.encode(self.cfg.gcg_init_token, add_special_tokens=False)
            init = (init * self.cfg.gcg_suffix_len)[: self.cfg.gcg_suffix_len]
            suffix_ids = torch.tensor(init, device=self.device)
            history, best_loss, best_ids, start = [], float("inf"), suffix_ids.clone(), 0

        steps = range(start, self.cfg.gcg_steps)
        bar = None
        if use_tqdm:
            from tqdm.auto import tqdm
            bar = tqdm(total=self.cfg.gcg_steps, initial=start, desc="GCG", unit="step")

        for step in steps:
            grad = self._token_gradients(suffix_ids, ctxs)         # [S, V]
            topk = (-grad).topk(self.cfg.gcg_topk, dim=1).indices  # [S, K]

            # sample candidate substitutions: random position, random top-k token
            B = self.cfg.gcg_search_batch
            pos = torch.randint(0, self.cfg.gcg_suffix_len, (B,), generator=g).to(self.device)
            pick = torch.randint(0, self.cfg.gcg_topk, (B,), generator=g).to(self.device)
            cands = suffix_ids.unsqueeze(0).repeat(B, 1)
            cands[torch.arange(B), pos] = topk[pos, pick]

            losses = self._loss_batch(cands, ctxs)
            j = int(losses.argmin())
            suffix_ids = cands[j]
            cur = float(losses[j])
            history.append(cur)
            if cur < best_loss:
                best_loss, best_ids = cur, suffix_ids.clone()
            if log_every and (step % log_every == 0 or step == self.cfg.gcg_steps - 1):
                print(f"  step {step:4d}  loss={cur:.3f}  best={best_loss:.3f}  "
                      f"suffix={self.m.tokenizer.decode(suffix_ids)!r}")
            if on_step is not None:
                on_step({"step": step, "suffix_ids": suffix_ids.tolist(),
                         "history": history, "best_loss": best_loss,
                         "best_ids": best_ids.tolist()})
            if bar:
                bar.set_postfix(loss=f"{cur:.2f}", best=f"{best_loss:.2f}")
                bar.update(1)
        if bar:
            bar.close()

        return GCGResult(
            suffix_ids=best_ids.tolist(),
            suffix_text=self.m.tokenizer.decode(best_ids),
            loss=best_loss, loss_history=history,
            target_scale=self.cfg.gcg_target_scale, layer=self.layer,
            cos_to_v=self._cos_to_v(best_ids, ctxs[0]),
            proj=self._mean_proj(best_ids, ctxs), prompts=prompts,
        )


def evaluate_suffix(gcg: "GCG", model: SteeringModel, cfg: Config, sv: SteeringVector,
                    clf, suffix_ids: list[int], prompts: list[str]) -> dict:
    """Compare concept-rate of three conditions on `prompts`:
       clean (no intervention) | activation steering @target_scale | GCG suffix.
    Shows whether the discrete suffix reproduces the steering effect behaviorally.
    """
    from steering_vectors.evaluate import _score_generations, _free_memory

    # clean
    clean_gen = model.generate_batch(prompts, cfg.batch_size)
    clean_rate, clean_prob = _score_generations(model, clf, prompts, clean_gen, cfg.batch_size)
    _free_memory()

    # activation steering at the target scale
    model.add_steering(sv.vector, sv.layer, cfg.gcg_target_scale)
    steer_gen = model.generate_batch(prompts, cfg.batch_size)
    model.clear_steering()
    steer_rate, steer_prob = _score_generations(model, clf, prompts, steer_gen, cfg.batch_size)
    _free_memory()

    # discrete suffix (input channel only)
    from tqdm.auto import tqdm
    suf_gen = [gcg.generate_with_suffix(p, suffix_ids)
               for p in tqdm(prompts, desc="suffix-eval", unit="prompt")]
    suf_rate, suf_prob = _score_generations(model, clf, prompts, suf_gen, cfg.batch_size)

    transcripts = [
        {"prompt": p, "clean": c, "steering": s, "suffix": f}
        for p, c, s, f in zip(prompts, clean_gen, steer_gen, suf_gen)
    ]
    return {
        "n_prompts": len(prompts),
        "target_scale": cfg.gcg_target_scale,
        "clean":    {"concept_rate": clean_rate, "mean_prob": clean_prob},
        "steering": {"concept_rate": steer_rate, "mean_prob": steer_prob},
        "suffix":   {"concept_rate": suf_rate,  "mean_prob": suf_prob},
        "sample_suffix_generation": suf_gen[0] if suf_gen else "",
        "transcripts": transcripts,
    }


if __name__ == "__main__":
    # Tiny smoke: short suffix, few steps — verify loss decreases and it runs.
    from steering_vectors.data import load_conversations
    from steering_vectors.evaluate import eval_prompts_from
    from token_optimization import checkpoint as ck

    cfg = Config.from_yaml("steering/configs/rude.yaml")
    cfg.gcg_suffix_len, cfg.gcg_steps, cfg.gcg_topk, cfg.gcg_search_batch = 8, 12, 64, 32
    cfg.gcg_target_scale = 3.0
    model = SteeringModel(cfg)
    sv = ck.load_steering_vector(cfg)
    prompts = eval_prompts_from(load_conversations(cfg, n=20), 1)
    res = GCG(model, cfg, sv).optimize(prompts)
    print(f"\nfinal loss {res.loss:.3f}  cos_to_v {res.cos_to_v:.3f}")
    print("suffix:", repr(res.suffix_text))
    print("loss[0] -> loss[-1]:", round(res.loss_history[0], 2), "->", round(res.loss_history[-1], 2))

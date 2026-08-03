"""Model loading + residual-stream activation collection.

`SteeringModel` wraps a HF causal LM. It can:
  - collect a pooled activation vector for a (prompt, response) pair at a layer
  - generate text while adding a steering vector at that layer (see evaluate.py
    which uses `add_steering` / `clear_steering`).

Pooling operates over the *response* token span only, located by comparing the
full chat against the prompt-only prefix.
"""
from __future__ import annotations

import torch

from steering_vectors.config import Config

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SteeringModel:
    def __init__(self, cfg: Config):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=_DTYPES[cfg.dtype]
        ).to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self._decoder_layers = self.model.model.layers   # LlamaDecoderLayer list
        self._steer_hook = None
        # tokens that mark a turn boundary — `last` pooling skips these so it reads
        # the final *content* token, not a trailing <|eot_id|>/special.
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self._end_ids = set(self.tokenizer.all_special_ids)
        if isinstance(eot, int) and eot >= 0:
            self._end_ids.add(eot)

    def _last_content_idx(self, ids_1d) -> int:
        """Index of the last non-boundary token (skips trailing eot/special)."""
        i = int(ids_1d.shape[0]) - 1
        while i > 0 and int(ids_1d[i]) in self._end_ids:
            i -= 1
        return i

    # ---- token bookkeeping -------------------------------------------------
    def _ids_and_response_start(self, prompt: str, response: str):
        """Return (full input_ids [1,T], index where response tokens begin)."""
        full = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response}],
            tokenize=True, add_generation_prompt=False, return_tensors="pt",
        )
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        start = min(prefix.shape[1], full.shape[1] - 1)
        full = full[:, : self.cfg.max_length]
        return full.to(self.device), start

    # ---- activation collection --------------------------------------------
    @torch.no_grad()
    def collect_activation(self, prompt: str, response: str,
                           layer: int, pooling: str) -> torch.Tensor:
        """Pooled residual-stream vector [hidden] at `layer` over response tokens."""
        input_ids = self._ids_and_response_start(prompt, response)
        input_ids, start = input_ids
        need_attn = pooling == "attention"
        out = self.model(
            input_ids, output_hidden_states=True,
            output_attentions=need_attn, use_cache=False,
        )
        # hidden_states[0] = embeddings; layer L output => index L+1
        hs = out.hidden_states[layer + 1][0]              # [T, hidden]
        resp = hs[start:]                                 # response span
        if resp.shape[0] == 0:                            # degenerate: fall back to all
            resp = hs
            start = 0

        if pooling == "mean":
            vec = resp.mean(dim=0)
        elif pooling == "last":
            vec = hs[self._last_content_idx(input_ids[0])]     # pre-eot content token
        elif pooling == "attention":
            attn = out.attentions[layer][0]               # [heads, T, T]
            mass = attn.mean(dim=0).sum(dim=0)            # key mass over queries [T]
            w = torch.softmax(mass[start:].float(), dim=0).to(resp.dtype)
            vec = (resp * w.unsqueeze(-1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        return vec.float().cpu()

    @torch.no_grad()
    def collect_activations_batch(self, pairs: list[tuple[str, str]],
                                  layer: int, pooling: str, batch_size: int = 16,
                                  progress=None, desc: str | None = None) -> torch.Tensor:
        """Batched pooled activations -> [N, hidden]. Much faster than looping.

        Supports mean/last (right-padded batched forward). `attention` pooling
        falls back to the per-example path (needs per-sequence attention maps).
        `progress(done, total)` is called after each batch, if given. `desc`, if
        set, renders a tqdm progress bar labelled `desc`.
        """
        total = len(pairs)
        bar = None
        if desc is not None:
            from tqdm.auto import tqdm
            bar = tqdm(total=total, desc=desc, unit="conv")

        if pooling == "attention":
            vecs = []
            for i, (p, r) in enumerate(pairs):
                vecs.append(self.collect_activation(p, r, layer, pooling))
                if progress:
                    progress(i + 1, total)
                if bar:
                    bar.update(1)
            if bar:
                bar.close()
            return torch.stack(vecs)

        pad_id = self.tokenizer.pad_token_id
        out_vecs: list[torch.Tensor] = []
        for b in range(0, len(pairs), batch_size):
            chunk = pairs[b:b + batch_size]
            seqs, starts, lens = [], [], []
            for prompt, response in chunk:
                ids, start = self._ids_and_response_start(prompt, response)
                ids = ids[0]                                  # [T]
                seqs.append(ids)
                starts.append(start)
                lens.append(ids.shape[0])
            T = max(lens)
            batch = torch.full((len(chunk), T), pad_id, dtype=seqs[0].dtype, device=self.device)
            mask = torch.zeros((len(chunk), T), dtype=torch.long, device=self.device)
            for i, ids in enumerate(seqs):
                batch[i, : lens[i]] = ids
                mask[i, : lens[i]] = 1

            hs = self.model(batch, attention_mask=mask, output_hidden_states=True,
                            use_cache=False).hidden_states[layer + 1]   # [B, T, H]
            for i in range(len(chunk)):
                s = starts[i] if starts[i] < lens[i] else 0
                if pooling == "mean":
                    vec = hs[i, s: lens[i]].mean(dim=0)
                elif pooling == "last":
                    vec = hs[i, self._last_content_idx(seqs[i])]   # pre-eot content token
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")
                out_vecs.append(vec.float().cpu())
            if progress:
                progress(len(out_vecs), total)
            if bar:
                bar.update(len(chunk))
        if bar:
            bar.close()
        return torch.stack(out_vecs)

    # ---- steering hooks (used at eval time) --------------------------------
    def add_steering(self, vector: torch.Tensor, layer: int, scale: float):
        """Register a hook that adds `scale * vector` to the layer's output."""
        self.clear_steering()
        vec = vector.to(self.device, dtype=next(self.model.parameters()).dtype)

        def hook(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + scale * vec
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        self._steer_hook = self._decoder_layers[layer].register_forward_hook(hook)

    def clear_steering(self):
        if self._steer_hook is not None:
            self._steer_hook.remove()
            self._steer_hook = None

    # ---- generation --------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(self.device)
        greedy = self.cfg.gen_temperature <= 0
        out = self.model.generate(
            ids, max_new_tokens=self.cfg.max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else self.cfg.gen_temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], batch_size: int = 16) -> list[str]:
        """Generate for many prompts at once. Left-pads so all rows end aligned."""
        pad_id = self.tokenizer.pad_token_id
        greedy = self.cfg.gen_temperature <= 0
        outputs: list[str] = []
        for b in range(0, len(prompts), batch_size):
            chunk = prompts[b:b + batch_size]
            seqs = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt",
                )[0]
                for p in chunk
            ]
            T = max(s.shape[0] for s in seqs)
            input_ids = torch.full((len(chunk), T), pad_id, dtype=seqs[0].dtype)
            mask = torch.zeros((len(chunk), T), dtype=torch.long)
            for i, s in enumerate(seqs):                       # LEFT pad
                input_ids[i, T - s.shape[0]:] = s
                mask[i, T - s.shape[0]:] = 1
            input_ids, mask = input_ids.to(self.device), mask.to(self.device)

            out = self.model.generate(
                input_ids, attention_mask=mask,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else self.cfg.gen_temperature,
                pad_token_id=pad_id,
            )
            gen = out[:, T:]
            outputs.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return outputs


if __name__ == "__main__":
    cfg = Config.from_yaml("steering/configs/rude.yaml")
    m = SteeringModel(cfg)
    print("device:", m.device, "| hidden:", m.hidden_size, "| layers:", len(m._decoder_layers))
    for pool in ("mean", "last", "attention"):
        v = m.collect_activation("How do I train for a marathon?",
                                 "Just run more, obviously.", cfg.layer, pool)
        print(f"pool={pool:9s} shape={tuple(v.shape)} norm={v.norm():.3f}")
    print("gen sample:", m.generate("How do I train for a marathon?")[:80])

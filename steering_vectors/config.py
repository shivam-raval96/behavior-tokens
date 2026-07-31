"""Experiment configuration.

Single source of truth for model, data, and steering hyperparameters.
Load from a YAML file with `Config.from_yaml(path)`; every field has a
sensible default so YAML only needs to override what changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class Config:
    # ---- concept ----
    concept: str = "rude"                     # human-readable concept name

    # ---- model ----
    # NOTE: meta-llama/Llama-3.2-1B-Instruct is gated. `unsloth/Llama-3.2-1B-Instruct`
    # is an ungated mirror with identical weights. Swap if you have gated access.
    model_name: str = "unsloth/Llama-3.2-1B-Instruct"
    device: str = "auto"                      # auto | cpu | cuda | mps
    dtype: str = "float32"                     # float32 | float16 | bfloat16
    layer: int = 8                             # residual-stream layer to steer/probe

    # ---- data ----
    dataset_name: str = "shiv96/convsersations_rude_large"
    data_file: Optional[str] = None           # local json path; overrides dataset_name if set
    split: str = "train"
    n_samples: int = 1000                      # conversations used to build the vector
    prompt_key: str = "prompt"
    response_key: str = "response"
    label_key: str = "label"                   # +1 = concept present, -1 = absent
    pos_label: int = 1
    neg_label: int = -1
    max_length: int = 512                      # tokenizer truncation length

    # ---- steering vector ----
    pooling: str = "last"                      # last | mean | attention (last-token default)
    normalize: bool = True                     # unit-normalize the steering vector
    batch_size: int = 16                       # batch size for activation collection

    # ---- classifier ----
    clf_test_frac: float = 0.2
    clf_C: float = 1.0
    seed: int = 0

    # ---- evaluation (steering curve) ----
    eval_n_prompts: int = 100
    curve_min: float = -5.0
    curve_max: float = 5.0
    curve_step: float = 0.5
    max_new_tokens: int = 64
    gen_temperature: float = 0.0               # 0 => greedy

    # ---- GCG suffix optimization ----
    # Find a discrete token suffix whose natural activations reproduce the effect
    # of adding `gcg_target_scale * steering_vector` at `layer`.
    gcg_suffix_len: int = 16                    # number of optimizable suffix tokens
    gcg_steps: int = 250                        # GCG iterations
    gcg_topk: int = 256                         # top-k candidate tokens per position (by grad)
    gcg_search_batch: int = 128                 # candidate substitutions evaluated per step
    gcg_target_scale: float = 3.0               # steering scale the suffix should match
    gcg_objective: str = "project"              # project | match | kl
    #   project: drive shift's component ALONG v to target_scale — (⟨Δ,v⟩ − α)²,
    #            ignores off-v dims (the behaviorally relevant, easier target).
    #   match:   full activation MSE to h_clean + α·v (strict, hard for tokens).
    #   kl:      match the *steered model's output distribution* — teacher = model
    #            with +α·v added (its greedy continuation); student = suffixed model
    #            teacher-forced on it; loss = KL over the first gcg_kl_tokens steps.
    #            Targets behavior directly, sidestepping the proj ceiling.
    gcg_kl_tokens: int = 12                      # generation horizon matched under `kl`
    gcg_n_prompts: int = 1                      # prompts optimized over (universal if >1)
    gcg_init_token: str = " !"                  # token repeated to initialize the suffix
    gcg_seed: int = 0

    # ---- io ----
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def curve_scales(self) -> list[float]:
        """Steering multipliers, inclusive of both ends."""
        n = round((self.curve_max - self.curve_min) / self.curve_step)
        return [round(self.curve_min + i * self.curve_step, 4) for i in range(n + 1)]


if __name__ == "__main__":
    c = Config()
    print("default config:")
    for k, v in c.to_dict().items():
        print(f"  {k}: {v}")
    print("curve scales:", c.curve_scales())

"""Reference-equivalent text-control GCG primitives.

The original llm-attacks code samples token controls but treats their decoded
text as authoritative: it filters by re-tokenized *length*, then evaluates
that text after inserting it into the complete conversation prompt.  This
module preserves that contract for chat-template tokenizers such as Llama 3.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import yaml
from tqdm.auto import tqdm

from jailbreaks.asr import is_success
from jailbreaks.gcg_small_scale import load_model, run_paths


@dataclass(frozen=True)
class PromptLayout:
    prompt: str
    input_ids: torch.Tensor
    control_slice: slice
    assistant_slice: slice
    target_slice: slice
    loss_slice: slice


def filter_text_candidates(tokenizer, candidate_ids: torch.Tensor, current_text: str,
                           expected_length: int) -> list[str]:
    """Match llm-attacks: decode, reject no-op, preserve token length only."""
    accepted: list[str] = []
    for row in candidate_ids:
        text = tokenizer.decode(row.tolist(), skip_special_tokens=True)
        retokenized = tokenizer(text, add_special_tokens=False).input_ids
        if text != current_text and len(retokenized) == expected_length:
            accepted.append(text)
    return accepted


def disallowed_control_tokens(tokenizer, allow_non_ascii: bool) -> torch.Tensor:
    """Match llm-attacks' control-token mask when non-ASCII is disabled."""
    blocked = set(tokenizer.all_special_ids)
    if not allow_non_ascii:
        blocked.update(token_id for token_id in range(3, tokenizer.vocab_size)
                       if not (tokenizer.decode([token_id]).isascii()
                               and tokenizer.decode([token_id]).isprintable()))
    return torch.tensor(sorted(blocked), dtype=torch.long)


class TextSuffixManager:
    """Derive slices from the fully rendered prompt using fast-tokenizer offsets."""

    def __init__(self, tokenizer, device: torch.device):
        if not getattr(tokenizer, "is_fast", False):
            raise TypeError("text-control GCG requires a fast tokenizer for offset slices")
        self.tokenizer, self.device = tokenizer, device

    def _render(self, goal: str, suffix_text: str, target: str | None) -> tuple[str, int, int, int | None, int | None]:
        content = f"{goal} {suffix_text}"
        if target is None:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            return rendered, rendered.find(content) + len(goal), rendered.find(content) + len(content), None, None
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}, {"role": "assistant", "content": target}],
            tokenize=False, add_generation_prompt=False,
        )
        content_start = rendered.find(content)
        target_start = rendered.rfind(target)
        if content_start < 0 or target_start < 0:
            raise AssertionError("chat template did not retain the supplied user/assistant content")
        return rendered, content_start + len(goal), content_start + len(content), target_start, target_start + len(target)

    def _encode(self, prompt: str) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [tuple(pair) for pair in encoded.offset_mapping]
        return torch.tensor(encoded.input_ids, device=self.device, dtype=torch.long), offsets

    @staticmethod
    def _slice_for_chars(offsets: list[tuple[int, int]], start: int, stop: int, label: str) -> slice:
        positions = [index for index, (left, right) in enumerate(offsets)
                     if right > left and left >= start and right <= stop]
        if not positions or positions != list(range(positions[0], positions[-1] + 1)):
            raise AssertionError(f"{label} does not map to a contiguous token span")
        return slice(positions[0], positions[-1] + 1)

    def training_layout(self, goal: str, suffix_text: str, target: str) -> PromptLayout:
        prompt, control_start, control_stop, target_start, target_stop = self._render(goal, suffix_text, target)
        ids, offsets = self._encode(prompt)
        control = self._slice_for_chars(offsets, control_start, control_stop, "control")
        target_slice = self._slice_for_chars(offsets, target_start, target_stop, "target")
        if control.stop >= target_slice.start:
            raise AssertionError("control and target spans overlap")
        assistant = slice(control.stop, target_slice.start)
        if assistant.start == assistant.stop:
            raise AssertionError("assistant header has no token span")
        return PromptLayout(prompt, ids, control, assistant, target_slice,
                            slice(target_slice.start - 1, target_slice.stop - 1))

    def generation_ids(self, goal: str, suffix_text: str) -> torch.Tensor:
        prompt, _, _, _, _ = self._render(goal, suffix_text, None)
        ids, _ = self._encode(prompt)
        return ids

    def control_ids(self, suffix_text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer(suffix_text, add_special_tokens=False).input_ids,
                            device=self.device, dtype=torch.long)


class ReferenceTextGCG:
    """A single-objective GCG core whose evaluated controls are text strings."""

    def __init__(self, model, tokenizer, suffix_length: int, top_k: int,
                 candidate_batch_size: int, evaluation_chunk_size: int):
        self.model, self.tokenizer = model, tokenizer
        self.device, self.suffix_length = model.device, suffix_length
        self.top_k, self.batch_size, self.eval_chunk = top_k, candidate_batch_size, evaluation_chunk_size
        self.manager = TextSuffixManager(tokenizer, self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.embedding = model.get_input_embeddings()
        self.vocab_size = self.embedding.weight.shape[0]

    def initial_control(self) -> str:
        control = " ".join(["!"] * self.suffix_length)
        if self.manager.control_ids(control).numel() != self.suffix_length:
            raise AssertionError("initial text control does not have requested token length")
        return control

    def token_gradient(self, goal: str, control: str, target: str) -> tuple[torch.Tensor, float, PromptLayout]:
        layout = self.manager.training_layout(goal, control, target)
        current_ids = layout.input_ids[layout.control_slice]
        if current_ids.numel() != self.suffix_length:
            raise AssertionError("rendered control length changed")
        one_hot = torch.zeros(self.suffix_length, self.vocab_size, device=self.device,
                              dtype=self.embedding.weight.dtype)
        one_hot.scatter_(1, current_ids.unsqueeze(1), 1).requires_grad_(True)
        embeds = self.embedding(layout.input_ids).detach()
        embeds[layout.control_slice] = one_hot @ self.embedding.weight
        logits = self.model(inputs_embeds=embeds.unsqueeze(0)).logits[0]
        loss = F.cross_entropy(logits[layout.loss_slice], layout.input_ids[layout.target_slice])
        loss.backward()
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise AssertionError("gradient reached frozen model weights")
        return one_hot.grad.detach(), float(loss.detach()), layout

    def sample_text_candidates(self, current_text: str, current_ids: torch.Tensor,
                               gradient: torch.Tensor, generator: torch.Generator,
                               retry_batches: int, disallowed_tokens: torch.Tensor) -> tuple[list[str], int, float]:
        ranking_gradient = gradient.clone()
        ranking_gradient[:, disallowed_tokens.to(self.device)] = float("inf")
        top_ids = (-ranking_gradient).topk(self.top_k, dim=1).indices
        for attempt in range(1, retry_batches + 1):
            positions = torch.randint(0, self.suffix_length, (self.batch_size,), device=self.device, generator=generator)
            choices = torch.randint(0, self.top_k, (self.batch_size,), device=self.device, generator=generator)
            raw = current_ids.unsqueeze(0).repeat(self.batch_size, 1)
            raw[torch.arange(self.batch_size, device=self.device), positions] = top_ids[positions, choices]
            accepted = filter_text_candidates(self.tokenizer, raw, current_text, self.suffix_length)
            if accepted:
                rate = len(accepted) / self.batch_size
                return accepted + [accepted[-1]] * (self.batch_size - len(accepted)), attempt, rate
        raise RuntimeError(f"no reference-valid decoded text candidates after {retry_batches} batches")

    @torch.no_grad()
    def evaluate(self, goal: str, candidates: list[str], target: str) -> torch.Tensor:
        layouts = [self.manager.training_layout(goal, candidate, target) for candidate in candidates]
        exemplar = layouts[0]
        if any(layout.input_ids.shape != exemplar.input_ids.shape or layout.control_slice != exemplar.control_slice
               or layout.target_slice != exemplar.target_slice for layout in layouts):
            raise AssertionError("reference-valid candidates changed prompt or slice shape")
        inputs = torch.stack([layout.input_ids for layout in layouts])
        losses = []
        for start in range(0, len(layouts), self.eval_chunk):
            logits = self.model(input_ids=inputs[start:start + self.eval_chunk]).logits
            target_ids = inputs[start:start + self.eval_chunk, exemplar.target_slice]
            losses.append(F.cross_entropy(logits[:, exemplar.loss_slice, :].transpose(1, 2), target_ids,
                                          reduction="none").mean(dim=1))
        return torch.cat(losses)

    @torch.no_grad()
    def generate(self, goal: str, control: str, max_new_tokens: int) -> str:
        ids = self.manager.generation_ids(goal, control).unsqueeze(0)
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                  max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                                  eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def run_text_path_diagnostic(config_path: Path, output_base: Path, commit=None,
                             run_mode: str = "fresh", run_id: str | None = None) -> dict:
    """Execute the preflight described in EXPERIMENT_LLAMA32_1B_GCG_TEXT_PATH_DIAGNOSTIC."""
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("stage") != "llama32_gcg_text_path_diagnostic":
        raise ValueError("invalid reference-text GCG diagnostic config")
    if run_mode not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    paths = run_paths(cfg, output_base)
    if run_id:
        directory = output_base / cfg["run"]["output_root"] / run_id
        paths = {"directory": directory, "result": directory / "results.json",
                 "summary": directory / "RESULTS.md", "checkpoint": directory / "checkpoint.json",
                 "progress": directory / "progress.json"}
    if run_mode == "resume" and not run_id:
        raise ValueError("resume requires the original run_id")
    checkpoint = json.loads(paths["checkpoint"].read_text()) if run_mode == "resume" and paths["checkpoint"].exists() else None
    if run_mode == "resume" and (not checkpoint or checkpoint.get("config_sha256") != config_sha):
        raise ValueError("resume requires a matching config checkpoint")
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["directory"].joinpath("config.yaml").write_text(config_path.read_text())
    with Path(cfg["dataset_csv"]).open(newline="") as handle:
        row = list(csv.DictReader(handle))[cfg["diagnostic_behavior_index"]]
    goal = row["goal"].strip()
    target = cfg["target_prefix"] + goal[:1].lower() + goal[1:]
    model, tokenizer, device = load_model(cfg)
    attack = ReferenceTextGCG(model, tokenizer, cfg["suffix_length"], cfg["top_k"],
                              cfg["candidate_batch_size"], cfg["evaluation_chunk_size"])
    disallowed_tokens = disallowed_control_tokens(tokenizer, cfg["allow_non_ascii"])
    control, best_control, best_loss = attack.initial_control(), None, float("inf")
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    history, snapshots = [], []
    start_step = 0
    if checkpoint:
        control, best_control, best_loss = checkpoint["control"], checkpoint["best_control"], checkpoint["best_loss"]
        history, snapshots, start_step = checkpoint["loss_history"], checkpoint["snapshots"], checkpoint["next_step"]
        generator.set_state(torch.tensor(checkpoint["generator_state"], dtype=torch.uint8))

    @torch.no_grad()
    def report(step: int, label: str, current: str, acceptance: float | None = None) -> dict:
        layout = attack.manager.training_layout(goal, current, target)
        logits = model(input_ids=layout.input_ids.unsqueeze(0)).logits[0]
        target_ids = layout.input_ids[layout.target_slice]
        per_token = F.cross_entropy(logits[layout.loss_slice], target_ids, reduction="none")
        first_logits, first_id = logits[layout.loss_slice][0], target_ids[0]
        rank = int((first_logits > first_logits[first_id]).sum()) + 1
        baseline = attack.generate(goal, "", cfg["max_new_tokens"])
        attacked = attack.generate(goal, current, cfg["max_new_tokens"])
        return {"step": step, "label": label, "full_target_loss": float(per_token.mean()),
                "sure_probability": float(torch.softmax(first_logits.float(), dim=0)[first_id]),
                "sure_rank": rank, "candidate_acceptance_rate": acceptance,
                "suffix": current, "suffix_token_count": int(attack.manager.control_ids(current).numel()),
                "baseline_response": baseline, "suffix_response": attacked,
                "baseline_success": is_success(baseline), "suffix_success": is_success(attacked)}

    if not snapshots:
        snapshots.append(report(0, "initial", control))
    checks = {"fast_tokenizer": tokenizer.is_fast, "model_weight_gradients_absent": True,
              "batch_serial_max_abs_difference": 0.0, "candidate_shapes_stable": True}
    payload = {"status": "running", "config_sha256": config_sha, "next_step": start_step,
               "loss_history": history, "control": control, "best_control": best_control,
               "best_loss": best_loss, "snapshots": snapshots,
               "generator_state": generator.get_state().cpu().tolist(), "checks": checks}
    try:
      for step in tqdm(range(start_step, cfg["diagnostic_steps"]), desc="Llama-3.2 reference-text GCG", unit="step"):
        gradient, _, layout = attack.token_gradient(goal, control, target)
        candidates, retry_count, acceptance_rate = attack.sample_text_candidates(
            control, layout.input_ids[layout.control_slice], gradient, generator,
            cfg["candidate_retry_batches"], disallowed_tokens
        )
        batch = attack.evaluate(goal, candidates[:8], target)
        serial = torch.cat([attack.evaluate(goal, [candidate], target) for candidate in candidates[:8]])
        checks["batch_serial_max_abs_difference"] = max(checks["batch_serial_max_abs_difference"], float((batch - serial).abs().max()))
        losses = attack.evaluate(goal, candidates, target)
        control, loss = candidates[int(losses.argmin())], float(losses.min())
        history.append(loss)
        if loss < best_loss:
            best_control, best_loss = control, loss
        if (step + 1) % cfg["diagnostic_every"] == 0 or step + 1 == cfg["diagnostic_steps"]:
            snapshots.append(report(step + 1, "current", control, acceptance_rate))
            print(json.dumps({"type": "gcg_text_progress", "step": step + 1,
                              "total_steps": cfg["diagnostic_steps"], "loss": loss,
                              "best_loss": best_loss, "candidate_retry_batches": retry_count,
                              "candidate_acceptance_rate": acceptance_rate, "sure_rank": snapshots[-1]["sure_rank"],
                              "asr": snapshots[-1]["suffix_success"]}))
        payload = {"status": "running", "config_sha256": config_sha,
                   "next_step": step + 1, "loss_history": history, "control": control,
                   "best_control": best_control, "best_loss": best_loss, "snapshots": snapshots,
                   "generator_state": generator.get_state().cpu().tolist(), "checks": checks}
        paths["checkpoint"].write_text(json.dumps(payload, indent=2)); paths["progress"].write_text(json.dumps(payload, indent=2))
        if commit:
            commit()
    except KeyboardInterrupt:
        payload["status"] = "stopped"
        paths["checkpoint"].write_text(json.dumps(payload, indent=2)); paths["progress"].write_text(json.dumps(payload, indent=2))
        paths["result"].write_text(json.dumps(payload, indent=2))
        paths["summary"].write_text("# Llama-3.2 reference-text GCG diagnostic\n\nStatus: stopped\n")
        if commit:
            commit()
        raise
    final = report(cfg["diagnostic_steps"], "best", best_control)
    result = {"status": "complete", "checks": {**checks, "target_loss_decreased": best_loss < snapshots[0]["full_target_loss"],
              "passed": checks["fast_tokenizer"] and checks["model_weight_gradients_absent"] and checks["batch_serial_max_abs_difference"] <= cfg["batching_tolerance"]},
              "optimization": {"initial_loss": snapshots[0]["full_target_loss"], "best_loss": best_loss},
              "snapshots": snapshots + [final], "suffix": best_control,
              "paired_generation": {key: final[key] for key in ("baseline_response", "suffix_response", "baseline_success", "suffix_success")}}
    paths["result"].write_text(json.dumps(result, indent=2)); paths["checkpoint"].write_text(json.dumps(result, indent=2))
    paths["summary"].write_text(f"# Llama-3.2 reference-text GCG diagnostic\n\nStatus: complete\n\n- Loss: {result['optimization']['initial_loss']:.4f} → {best_loss:.4f}\n- Final `Sure` rank: {final['sure_rank']}\n- ASR: {final['suffix_success']}\n")
    if commit:
        commit()
    return result

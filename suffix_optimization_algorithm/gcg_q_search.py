from __future__ import annotations

import json
import math
import signal
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fixed_rollout_ce import RunWriter, fingerprint, jsonl_rows, sha256, steering


def continuation_losses(logits: torch.Tensor, prefix_length: int, suffix_length: int,
                        continuation_ids: torch.Tensor) -> torch.Tensor:
    """Per-token NLL; y[0] is predicted at Lx+m-1."""
    y_start = prefix_length + suffix_length
    target_logits = logits[..., y_start - 1:y_start - 1 + continuation_ids.numel(), :].float()
    targets = continuation_ids.expand(target_logits.shape[0], -1)
    return F.cross_entropy(target_logits.flatten(0, 1), targets.flatten(), reduction="none").view(targets.shape)


def position_decomposition(per_record: list[list[float]]) -> dict[str, float]:
    first = [row[0] for row in per_record if row]
    tail = [value for row in per_record for value in row[1:]]
    return {"position_1_ce": float(np.mean(first)), "tail_ce": float(np.mean(tail))}


def valid_token_ids(tokenizer, vocabulary_size: int) -> torch.Tensor:
    banned = set(tokenizer.all_special_ids)
    for token, token_id in tokenizer.get_added_vocab().items():
        if token.startswith("<|") or token.endswith("|>"):
            banned.add(token_id)
    return torch.tensor([i for i in range(vocabulary_size) if i not in banned], dtype=torch.long)


def exact_length_initialization(tokenizer, text: str, length: int) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) < length:
        filler = tokenizer(" directly", add_special_tokens=False).input_ids
        if not filler:
            raise ValueError("warm-start filler tokenized empty")
        while len(ids) < length:
            ids.extend(filler)
    return ids[:length]


class Search:
    def __init__(self, model, tokenizer, cache, q_records, config, vector):
        self.model, self.tokenizer, self.cache, self.q_records, self.config = model, tokenizer, cache, q_records, config
        self.embedding = model.get_input_embeddings()
        self.vocab = self.embedding.weight.shape[0]
        self.device = model.device
        self.vector = vector
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def sequence_embeds(self, record, suffix_ids: torch.Tensor) -> torch.Tensor:
        prefix = torch.tensor(record["prefix_ids"], device=self.device)
        continuation = torch.tensor(record["continuation_ids"], device=self.device)
        return torch.cat((self.embedding(prefix), self.embedding(suffix_ids), self.embedding(continuation)), dim=0)

    @torch.no_grad()
    def sampled_score(self, suffix_ids: torch.Tensor) -> dict[str, Any]:
        total, count, rows = 0.0, 0, []
        for record in self.cache:
            y = torch.tensor(record["continuation_ids"], device=self.device)
            logits = self.model(inputs_embeds=self.sequence_embeds(record, suffix_ids).unsqueeze(0), use_cache=False).logits
            losses = continuation_losses(logits, len(record["prefix_ids"]), suffix_ids.numel(), y)[0]
            values = losses.cpu().tolist(); rows.append(values)
            total += float(losses.sum()); count += losses.numel()
        ce = total / count
        parts = position_decomposition(rows)
        floor = self.config["sampled_teacher_floor"]
        return {"sampled_student_ce": ce, "sampled_gap": ce - floor,
                "position_1_student_ce": parts["position_1_ce"], "tail_student_ce": parts["tail_ce"], "tokens": count}

    @torch.no_grad()
    def sampled_teacher_floor(self) -> dict[str, float]:
        total, count, rows = 0.0, 0, []
        with steering(self.model, self.config["module_index"], self.vector, self.config["coefficient"]):
            for record in self.cache:
                prefix = torch.tensor(record["prefix_ids"], device=self.device)
                y = torch.tensor(record["continuation_ids"], device=self.device)
                full = torch.cat((prefix, y)).unsqueeze(0)
                logits = self.model(input_ids=full, use_cache=False).logits
                losses = continuation_losses(logits, prefix.numel(), 0, y)[0]
                values = losses.cpu().tolist(); rows.append(values)
                total += float(losses.sum()); count += losses.numel()
        parts = position_decomposition(rows)
        return {"sampled_teacher_floor": total/count, **parts}

    def gradient(self, suffix_ids: torch.Tensor) -> tuple[torch.Tensor, float]:
        one_hot = F.one_hot(suffix_ids, self.vocab).to(self.embedding.weight.dtype).detach().requires_grad_(True)
        total_tokens = sum(len(row["continuation_ids"]) for row in self.cache)
        loss_value = 0.0
        for record in self.cache:
            prefix = torch.tensor(record["prefix_ids"], device=self.device)
            y = torch.tensor(record["continuation_ids"], device=self.device)
            suffix_embeds = one_hot @ self.embedding.weight
            full = torch.cat((self.embedding(prefix), suffix_embeds, self.embedding(y)), dim=0).unsqueeze(0)
            logits = self.model(inputs_embeds=full, use_cache=False).logits
            loss_sum = continuation_losses(logits, prefix.numel(), suffix_ids.numel(), y).sum()
            scaled = loss_sum / total_tokens
            scaled.backward()
            loss_value += float(scaled.detach())
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise AssertionError("frozen model parameters received gradients")
        return one_hot.grad.detach(), loss_value

    def candidates(self, suffix: torch.Tensor, gradient: torch.Tensor, generator: torch.Generator,
                   valid: torch.Tensor) -> torch.Tensor:
        allowed = torch.zeros(self.vocab, dtype=torch.bool, device=self.device)
        allowed[valid.to(self.device)] = True
        ranked = (-gradient).masked_fill(~allowed.unsqueeze(0), -torch.inf).topk(self.config["top_k"], dim=-1).indices
        batch = self.config["candidate_batch_size"]
        positions = torch.randint(suffix.numel(), (batch,), generator=generator, device=self.device)
        choices = torch.randint(self.config["top_k"], (batch,), generator=generator, device=self.device)
        result = suffix.repeat(batch, 1)
        result[torch.arange(batch, device=self.device), positions] = ranked[positions, choices]
        changed = (result != suffix).sum(-1) == 1
        # Duplicate draws have identical true loss and cannot alter the greedy
        # argmin. Removing them saves forwards without changing the search.
        return torch.unique(result[changed], dim=0)

    @torch.no_grad()
    def candidate_losses_serial(self, candidates: torch.Tensor) -> torch.Tensor:
        """Reference implementation retained for numerical equivalence gates."""
        totals = torch.zeros(candidates.shape[0], device=self.device, dtype=torch.float64)
        total_tokens = sum(len(row["continuation_ids"]) for row in self.cache)
        micro = self.config["evaluation_microbatch_size"]
        for record in self.cache:
            prefix = torch.tensor(record["prefix_ids"], device=self.device)
            y = torch.tensor(record["continuation_ids"], device=self.device)
            p, z = self.embedding(prefix), self.embedding(y)
            for start in range(0, candidates.shape[0], micro):
                chunk = candidates[start:start + micro]
                n = chunk.shape[0]
                full = torch.cat((p.expand(n, -1, -1), self.embedding(chunk), z.expand(n, -1, -1)), dim=1)
                logits = self.model(inputs_embeds=full, use_cache=False).logits
                losses = continuation_losses(logits, prefix.numel(), chunk.shape[1], y).sum(-1)
                totals[start:start + n] += losses.double()
        return (totals / total_tokens).float()

    @torch.no_grad()
    def candidate_losses(self, candidates: torch.Tensor) -> torch.Tensor:
        """Exact loss, batching the rollout/candidate Cartesian product.

        Right padding does not change any active token's absolute position. Each
        record is sliced at its own continuation boundary, so padding never
        enters the objective.
        """
        totals = torch.zeros(candidates.shape[0], device=self.device, dtype=torch.float64)
        total_tokens = sum(len(row["continuation_ids"]) for row in self.cache)
        candidate_batch = self.config["evaluation_microbatch_size"]
        record_batch = self.config.get("evaluation_record_batch_size", 1)
        for record_start in range(0, len(self.cache), record_batch):
            records = self.cache[record_start:record_start + record_batch]
            fixed = []
            for record in records:
                prefix = torch.tensor(record["prefix_ids"], device=self.device)
                y = torch.tensor(record["continuation_ids"], device=self.device)
                fixed.append((prefix, y, self.embedding(prefix), self.embedding(y)))
            for candidate_start in range(0, candidates.shape[0], candidate_batch):
                chunk = candidates[candidate_start:candidate_start + candidate_batch]
                n = chunk.shape[0]
                suffix_embeddings = self.embedding(chunk)
                lengths = [p.numel() + chunk.shape[1] + y.numel() for p, y, _, _ in fixed]
                maximum = max(lengths)
                sequences, masks = [], []
                for (prefix, y, prefix_embeddings, y_embeddings), length in zip(fixed, lengths, strict=True):
                    sequence = torch.cat((
                        prefix_embeddings.expand(n, -1, -1), suffix_embeddings,
                        y_embeddings.expand(n, -1, -1)), dim=1)
                    if length < maximum:
                        sequence = F.pad(sequence, (0, 0, 0, maximum - length))
                    sequences.append(sequence)
                    mask = torch.zeros((n, maximum), device=self.device, dtype=torch.long)
                    mask[:, :length] = 1; masks.append(mask)
                logits = self.model(inputs_embeds=torch.cat(sequences), attention_mask=torch.cat(masks), use_cache=False).logits
                for record_index, (prefix, y, _, _) in enumerate(fixed):
                    record_logits = logits[record_index*n:(record_index+1)*n]
                    losses = continuation_losses(record_logits, prefix.numel(), chunk.shape[1], y).sum(-1)
                    totals[candidate_start:candidate_start+n] += losses.double()
        return (totals / total_tokens).float()

    @torch.no_grad()
    def exact_q_score(self, suffix: torch.Tensor) -> dict[str, float]:
        total_student, total_teacher, count = 0.0, 0.0, 0
        for record, qrow in zip(self.cache, self.q_records, strict=True):
            y = torch.tensor(record["continuation_ids"], device=self.device)
            logits = self.model(inputs_embeds=self.sequence_embeds(record, suffix).unsqueeze(0), use_cache=False).logits
            lp = continuation_losses_logits(logits, len(record["prefix_ids"]), suffix.numel(), y).log_softmax(-1)
            for position, sparse in enumerate(qrow["sparse_q_targets"]):
                ids = torch.tensor(sparse["token_ids"], device=self.device)
                q = torch.tensor(sparse["probabilities"], device=self.device, dtype=torch.float32)
                teacher_lp = torch.tensor(sparse["teacher_raw_logprobs"], device=self.device, dtype=torch.float32)
                total_student += float(-(q * lp[position, ids]).sum())
                total_teacher += float(-(q * teacher_lp).sum()); count += 1
        return {"exact_q_student_ce": total_student / count, "exact_q_teacher_floor": total_teacher / count,
                "exact_q_gap": (total_student - total_teacher) / count}


def continuation_losses_logits(logits, prefix_length, suffix_length, continuation_ids):
    start = prefix_length + suffix_length
    return logits[0, start - 1:start - 1 + continuation_ids.numel()].float()


def load_sources(config):
    cache_path, q_path = Path(config["teacher_cache_path"]), Path(config["q_records_path"])
    if sha256(cache_path) != config["teacher_cache_sha256"] or sha256(q_path) != config["q_records_sha256"]:
        raise ValueError("source artifact SHA-256 mismatch")
    cache, q_records = jsonl_rows(cache_path), jsonl_rows(q_path)
    if len(cache) != config["expected_records"] or len(q_records) != len(cache):
        raise ValueError("cached record count mismatch")
    if sum(len(row["continuation_ids"]) for row in cache) != config["expected_tokens"]:
        raise ValueError("cached token count mismatch")
    return cache, q_records


def plot_history(output: Path, history: list[dict[str, Any]]):
    if not history: return
    x = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x, [row["sampled_gap"] for row in history], label="all positions")
    axes[0].plot(x, [row["position_1_gap"] for row in history], label="position 1")
    axes[0].plot(x, [row["tail_gap"] for row in history], label="tail")
    axes[0].set(xlabel="Step", ylabel="Sampled gap (nats/token)", title="q-sampled steering gap"); axes[0].legend()
    axes[1].plot(x, [row["sampled_student_ce"] for row in history])
    axes[1].set(xlabel="Step", ylabel="Student CE", title="True discrete objective")
    for axis in axes: axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "gcg_search.png", dpi=180); plt.close(fig)


def run(config_path: Path, output: Path, mode="fresh", commit=None):
    config = yaml.safe_load(config_path.read_text()); config["run_mode"] = mode
    config_hash = fingerprint(config); writer = RunWriter(output, commit)
    checkpoint_path = output / "checkpoint.json"
    old = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    if mode == "fresh" and old: raise FileExistsError("fresh run refuses an existing checkpoint")
    if mode == "resume" and old.get("config_fingerprint") != config_hash: raise ValueError("resume fingerprint mismatch")
    cache, q_records = load_sources(config)
    writer.json("resolved_config.json", config); (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"], torch_dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    vector_path = Path(config["vector_path"])
    if sha256(vector_path) != config["vector_sha256"]: raise ValueError("vector SHA-256 mismatch")
    vector = torch.tensor(np.load(vector_path), device=model.device, dtype=torch.float32)
    search = Search(model, tokenizer, cache, q_records, config, vector)
    valid = valid_token_ids(tokenizer, search.vocab)
    generator = torch.Generator(device=search.device)
    generator.manual_seed(config["seed"])
    source_checkpoint = None
    if not old and config.get("continuation_checkpoint_path"):
        source_path = Path(config["continuation_checkpoint_path"])
        if sha256(source_path) != config["continuation_checkpoint_sha256"]:
            raise ValueError("continuation checkpoint SHA-256 mismatch")
        source_checkpoint = json.loads(source_path.read_text())
        if source_checkpoint.get("status") != "stopped":
            raise ValueError("continuation checkpoint must be explicitly stopped")
        writer.json("continuation_source.json", {"path":str(source_path),"sha256":sha256(source_path),"source_config_fingerprint":source_checkpoint["config_fingerprint"],"source_next_step":source_checkpoint["next_step"]})
    state = old or source_checkpoint
    if state:
        suffix = torch.tensor(state["suffix_ids"], device=search.device); best = torch.tensor(state["best_suffix_ids"], device=search.device)
        history = state["history"]; start = state["next_step"]; best_loss = state["best_loss"]
        generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
    else:
        if config["initialization"] == "warm": ids = exact_length_initialization(tokenizer, config["warm_start_text"], config["suffix_length"])
        else:
            cpu_gen = torch.Generator().manual_seed(config["seed"]); ids = valid[torch.randint(valid.numel(), (config["suffix_length"],), generator=cpu_gen)].tolist()
        suffix = torch.tensor(ids, device=search.device); best = suffix.clone(); history=[]; start=0; best_loss=math.inf
    started=time.monotonic(); stopped=False
    def stop(*_):
        nonlocal stopped; stopped=True
    signal.signal(signal.SIGTERM, stop)

    empty = torch.empty(0, device=search.device, dtype=torch.long)
    sampled_alignment = search.sampled_score(empty); exact_alignment = search.exact_q_score(empty)
    teacher_floor = search.sampled_teacher_floor()
    gate = (abs(sampled_alignment["sampled_student_ce"] - config["expected_sampled_student_ce"]) <= config["alignment_tolerance"]
            and abs(exact_alignment["exact_q_student_ce"] - config["expected_exact_q_student_ce"]) <= config["alignment_tolerance"]
            and abs(teacher_floor["sampled_teacher_floor"] - config["sampled_teacher_floor"]) <= config["alignment_tolerance"])
    writer.json("alignment.json", {"passed": gate, "sampled": sampled_alignment, "sampled_teacher": teacher_floor, "exact_q": exact_alignment})
    if not gate: raise RuntimeError("alignment gate failed; search not started")

    def save(status, latest):
        payload={"status":status,"config_fingerprint":config_hash,"next_step":start,"suffix_ids":suffix.tolist(),"best_suffix_ids":best.tolist(),"best_loss":best_loss,"history":history,"generator_state":generator.get_state().cpu().tolist(),"latest_metric":latest}
        writer.json("checkpoint.json",payload)
        elapsed=time.monotonic()-started
        writer.progress({"run_id":output.name,"config_fingerprint":config_hash,"phase":status,"completed":start,"total":config["steps"],"completed_fraction":start/config["steps"],"elapsed_seconds":elapsed,"throughput":start/elapsed if elapsed else 0,"eta_seconds":((config["steps"]-start)/(start/elapsed)) if start else None,"latest_ce":latest.get("sampled_student_ce") if latest else None,"best_gap":history[-1]["sampled_gap"] if history else None,"suffix":tokenizer.decode(best.tolist(), clean_up_tokenization_spaces=False),"error_count":0,"retry_count":int(old.get("retry_count",0))})

    latest={}
    for step in tqdm(range(start, config["steps"]), initial=start, total=config["steps"], desc="q-GCG"):
        gradient_started = time.monotonic()
        grad, current_loss = search.gradient(suffix)
        gradient_seconds = time.monotonic()-gradient_started
        previous_loss = current_loss
        candidates = search.candidates(suffix, grad, generator, valid)
        candidate_started = time.monotonic()
        losses = search.candidate_losses(candidates)
        candidate_seconds = time.monotonic()-candidate_started
        reference_seconds = None
        if config.get("verify_candidate_equivalence") and step == start:
            reference_started = time.monotonic()
            reference = search.candidate_losses_serial(candidates)
            reference_seconds = time.monotonic()-reference_started
            maximum_error = float((losses-reference).abs().max())
            writer.json("candidate_equivalence.json", {"candidates":candidates.shape[0],"maximum_absolute_error":maximum_error,"tolerance":config["candidate_equivalence_tolerance"],"passed":maximum_error <= config["candidate_equivalence_tolerance"],"batched_seconds":candidate_seconds,"serial_seconds":reference_seconds,"speedup":reference_seconds/candidate_seconds})
            if maximum_error > config["candidate_equivalence_tolerance"]: raise RuntimeError("batched candidate loss failed equivalence gate")
        value,index=float(losses.min()),int(losses.argmin())
        accepted=value < current_loss
        if accepted: suffix=candidates[index].clone(); current_loss=value
        if current_loss < best_loss: best=suffix.clone(); best_loss=current_loss
        start=step+1
        score=search.sampled_score(best)
        score["position_1_gap"] = score["position_1_student_ce"] - teacher_floor["position_1_ce"]
        score["tail_gap"] = score["tail_student_ce"] - teacher_floor["tail_ce"]
        latest={"step":start,"accepted":accepted,"loss_delta":value-previous_loss,"unique_candidates":int(candidates.shape[0]),"gradient_seconds":gradient_seconds,"candidate_seconds":candidate_seconds,"serial_reference_seconds":reference_seconds,**score,"suffix_ids":best.tolist(),"suffix":tokenizer.decode(best.tolist(), clean_up_tokenization_spaces=False)}
        if start % config["exact_q_every"] == 0 or start == config["steps"]: latest.update(search.exact_q_score(best))
        history.append(latest); print("METRIC "+json.dumps(latest,sort_keys=True),flush=True)
        if start % config["checkpoint_every"] == 0 or start == config["steps"]: plot_history(output,history); save("running",latest)
        if config.get("stop_after_step") and mode == "fresh" and start >= config["stop_after_step"]:
            save("stopped", latest); return {"status":"stopped","next_step":start}
        if stopped: save("stopped",latest); return {"status":"stopped","next_step":start}
    best_sampled=search.sampled_score(best)
    best_sampled["position_1_gap"]=best_sampled["position_1_student_ce"]-teacher_floor["position_1_ce"]
    best_sampled["tail_gap"]=best_sampled["tail_student_ce"]-teacher_floor["tail_ce"]
    results={"status":"complete","config_fingerprint":config_hash,"alignment":json.loads((output/"alignment.json").read_text()),"initialization":config["initialization"],"steps":start,"best_suffix_ids":best.tolist(),"best_suffix":tokenizer.decode(best.tolist(),clean_up_tokenization_spaces=False),"best_sampled":best_sampled,"best_exact_q":search.exact_q_score(best),"history":history,"elapsed_seconds":time.monotonic()-started}
    writer.json("results.json",results); plot_history(output,history)
    (output/"RESULTS.md").write_text(f"# q-weighted GCG search\n\n- Status: complete\n- Initialization: {config['initialization']}\n- Best suffix: `{results['best_suffix']}`\n- Sampled gap: **{results['best_sampled']['sampled_gap']:.6f}**\n- Position-1 gap: **{results['best_sampled']['position_1_gap']:.6f}**\n- Tail gap: **{results['best_sampled']['tail_gap']:.6f}**\n- Exact-q audit gap: **{results['best_exact_q']['exact_q_gap']:.6f}**\n")
    save("complete",history[-1]); return results

"""Response-position, FP32 steering-suffix objective."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from .optimizer import ActivationTarget, direction_metrics


@dataclass(frozen=True)
class ResponseLayout:
    """Exact token layout with the suffix in-user and measurement in-response."""

    clean_prompt_ids: torch.Tensor
    before_suffix_ids: torch.Tensor
    after_suffix_ids: torch.Tensor
    response_ids: torch.Tensor

    def clean_measurement_ids(self) -> torch.Tensor:
        return torch.cat((self.clean_prompt_ids, self.response_ids))

    def suffix_measurement_ids(self, suffix: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.before_suffix_ids, suffix.cpu(), self.after_suffix_ids,
                          self.response_ids))


class ResponsePositionOptimizer:
    """GCG search matching a steering delta at an actual assistant token.

    Model forwards may remain BF16, but every objective operation and candidate
    comparison is promoted to FP32. The same teacher-forced response is present
    in clean and suffix conditions, so the measured difference isolates the
    suffix while matching the source vector's response-token position.
    """

    def __init__(self, model, tokenizer, target: ActivationTarget, *, top_k: int = 128,
                 candidate_batch_size: int = 512, norm_weight: float = 0.1,
                 consistency_weight: float = 0.25):
        self.model, self.tokenizer, self.target = model, tokenizer, target
        self.top_k, self.candidate_batch_size = top_k, candidate_batch_size
        self.norm_weight, self.consistency_weight = norm_weight, consistency_weight
        self.device = next(model.parameters()).device
        self.embedding = model.get_input_embeddings()
        self.special_ids = set(tokenizer.all_special_ids)

    def _hidden(self, embeds: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)
        output = self.model(inputs_embeds=embeds, attention_mask=mask,
                            output_hidden_states=True, use_cache=False)
        return output.hidden_states[self.target.layer][:, -1].float()

    @torch.no_grad()
    def baselines(self, layouts: Sequence[ResponseLayout]) -> list[torch.Tensor]:
        return [self._hidden(self.embedding(layout.clean_measurement_ids()[None].to(self.device)))[0]
                for layout in layouts]

    def deltas(self, suffix_embeds: torch.Tensor, layouts: Sequence[ResponseLayout],
               baselines: Sequence[torch.Tensor]) -> torch.Tensor:
        values=[]
        for layout, baseline in zip(layouts, baselines):
            before=self.embedding(layout.before_suffix_ids[None].to(self.device)).detach()
            after=self.embedding(torch.cat((layout.after_suffix_ids, layout.response_ids))[None].to(self.device)).detach()
            hidden=self._hidden(torch.cat((before,suffix_embeds,after),dim=1))[0]
            values.append(hidden-baseline.float())
        return torch.stack(values)

    def loss_from_deltas(self, deltas: torch.Tensor) -> torch.Tensor:
        target=self.target.normalized(deltas.device,torch.float32).float()
        mean=deltas.float().mean(0)
        alignment=1-F.cosine_similarity(mean,target,dim=0)
        consistency=(1-F.cosine_similarity(deltas.float(),target[None],dim=1)).mean()
        norm_error=(mean.norm()-target.norm()).square()/target.numel()
        return alignment+self.consistency_weight*consistency+self.norm_weight*norm_error

    def loss(self, suffix_embeds: torch.Tensor, layouts: Sequence[ResponseLayout],
             baselines: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.loss_from_deltas(self.deltas(suffix_embeds,layouts,baselines))

    def token_gradient(self, suffix: torch.Tensor, layouts: Sequence[ResponseLayout],
                       baselines: Sequence[torch.Tensor]) -> tuple[torch.Tensor,float]:
        embeds=self.embedding(suffix.to(self.device))[None].detach().requires_grad_(True)
        loss=self.loss(embeds,layouts,baselines); loss.backward()
        gradient=embeds.grad[0].float()@self.embedding.weight.detach().float().T
        return gradient,float(loss.detach())

    def candidates(self, suffix: torch.Tensor, gradient: torch.Tensor,
                   generator: torch.Generator) -> torch.Tensor:
        scores=-gradient.clone()
        if self.special_ids: scores[:,list(self.special_ids)]=-torch.inf
        scores[torch.arange(suffix.numel(),device=self.device),suffix.to(self.device)]=-torch.inf
        choices=scores.topk(min(self.top_k,scores.shape[1]-len(self.special_ids)-1),dim=1).indices
        rows=suffix.to(self.device).repeat(self.candidate_batch_size,1)
        # Every coordinate is covered before another round begins.
        rounds=(self.candidate_batch_size+suffix.numel()-1)//suffix.numel()
        positions=torch.cat([torch.randperm(suffix.numel(),generator=generator,device=self.device)
                             for _ in range(rounds)])[:self.candidate_batch_size]
        ranks=torch.randint(choices.shape[1],(self.candidate_batch_size,),generator=generator,device=self.device)
        ranks[:min(suffix.numel(),self.candidate_batch_size)]=0
        index=torch.arange(self.candidate_batch_size,device=self.device)
        rows[index,positions]=choices[positions,ranks]
        return rows

    @torch.no_grad()
    def evaluate(self, candidates: torch.Tensor, layouts: Sequence[ResponseLayout],
                 baselines: Sequence[torch.Tensor], chunk_size: int = 32) -> torch.Tensor:
        """Score exact discrete candidates in bounded GPU batches."""
        target=self.target.normalized(self.device,torch.float32).float()
        losses=[]
        for start in range(0,candidates.shape[0],chunk_size):
            batch=candidates[start:start+chunk_size].to(self.device)
            per_layout=[]
            for layout,baseline in zip(layouts,baselines):
                before=layout.before_suffix_ids.to(self.device)[None].expand(batch.shape[0],-1)
                after=torch.cat((layout.after_suffix_ids,layout.response_ids)).to(self.device)[None].expand(batch.shape[0],-1)
                ids=torch.cat((before,batch,after),dim=1)
                per_layout.append(self._hidden(self.embedding(ids))-baseline.float()[None])
            deltas=torch.stack(per_layout,dim=1).float()
            mean=deltas.mean(1)
            alignment=1-F.cosine_similarity(mean,target[None],dim=1)
            consistency=(1-F.cosine_similarity(deltas,target[None,None],dim=2)).mean(1)
            norm_error=(mean.norm(dim=1)-target.norm()).square()/target.numel()
            losses.append(alignment+self.consistency_weight*consistency+self.norm_weight*norm_error)
        return torch.cat(losses)

    def step(self, suffix: torch.Tensor, layouts: Sequence[ResponseLayout],
             baselines: Sequence[torch.Tensor], generator: torch.Generator):
        gradient,current=self.token_gradient(suffix,layouts,baselines)
        candidates=self.candidates(suffix,gradient,generator)
        losses=self.evaluate(candidates,layouts,baselines); index=int(losses.argmin())
        best=float(losses[index])
        return (candidates[index].detach(),best,True) if best<current else (suffix.detach(),current,False)

    @torch.no_grad()
    def metrics(self, suffix: torch.Tensor, layouts: Sequence[ResponseLayout],
                baselines: Sequence[torch.Tensor]) -> dict[str,float]:
        deltas=self.deltas(self.embedding(suffix.to(self.device))[None],layouts,baselines)
        target=self.target.normalized(deltas.device,torch.float32)
        return {**direction_metrics(deltas,target),"loss_fp32":float(self.loss_from_deltas(deltas)),
                "per_example_cosine_mean":float(F.cosine_similarity(deltas,target[None],dim=1).mean())}

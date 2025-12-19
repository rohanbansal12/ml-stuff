# train/dpo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine import completion_logprobs


@dataclass
class DPOBatchMetrics:
    # Scalars (or [B] vectors) you’ll want for logging/debugging
    loss: torch.Tensor  # scalar
    dpo_logits: torch.Tensor  # [B]
    margin: torch.Tensor  # [B] = (Δθ - Δref)
    acc: torch.Tensor  # scalar in [0,1], e.g. mean(dpo_logits > 0)

    policy_logp_chosen: torch.Tensor  # [B]
    policy_logp_rejected: torch.Tensor  # [B]
    ref_logp_chosen: torch.Tensor  # [B]
    ref_logp_rejected: torch.Tensor  # [B]

    chosen_token_count: Optional[torch.Tensor] = None  # [B] optional
    rejected_token_count: Optional[torch.Tensor] = None  # [B] optional

    def pretty_print(self, prefix: str = "", indent: int = 2) -> None:
        """
        Pretty-print batch-level DPO metrics.
        """
        sp = " " * indent

        def stat(x: torch.Tensor) -> str:
            x = x.detach()
            if x.numel() == 1:
                return f"{x.item(): .6f}"
            return (
                f"mean={x.mean().item(): .4f} "
                f"std={x.std(unbiased=False).item(): .4f} "
                f"min={x.min().item(): .4f} "
                f"max={x.max().item(): .4f}"
            )

        print(f"{prefix}DPOBatchMetrics")
        print(f"{prefix}{sp}loss: {stat(self.loss)}")
        print(f"{prefix}{sp}acc:  {stat(self.acc)}")
        print()

        print(f"{prefix}{sp}dpo_logits: {stat(self.dpo_logits)}")
        print(f"{prefix}{sp}margin:     {stat(self.margin)}")
        print()

        print(f"{prefix}{sp}policy_logp_chosen:   {stat(self.policy_logp_chosen)}")
        print(f"{prefix}{sp}policy_logp_rejected: {stat(self.policy_logp_rejected)}")
        print(f"{prefix}{sp}ref_logp_chosen:      {stat(self.ref_logp_chosen)}")
        print(f"{prefix}{sp}ref_logp_rejected:    {stat(self.ref_logp_rejected)}")

        if self.chosen_token_count is not None:
            print()
            print(f"{prefix}{sp}chosen_token_count:   {stat(self.chosen_token_count.float())}")

        if self.rejected_token_count is not None:
            print(
                f"{prefix}{sp}rejected_token_count: {stat(self.rejected_token_count.float())}"
            )


def freeze_model_(model: torch.nn.Module) -> None:
    """
    Freeze model params in-place (reference model).
    """
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def make_reference_model(
    policy_model: torch.nn.Module,
    *,
    method: str = "deepcopy",
) -> torch.nn.Module:
    """
    Create a frozen reference model.

    method:
      - "deepcopy": deep copy policy model (recommended for exact init match)
      - "load": load separately from checkpoint (optional; requires caller logic)

    Returns:
      ref_model frozen & eval
    """
    if method == "deepcopy":
        import copy

        ref_model = copy.deepcopy(policy_model)
        freeze_model_(ref_model)
        return ref_model
    raise ValueError(f"Unknown method: {method}")


def dpo_loss_batch(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    beta: float,
    use_mean_logp: bool = False,
) -> DPOBatchMetrics:
    """
    Compute DPO loss for one batch.

    Expects `batch` from collate_preference_batch:
      chosen_input_ids: [B, T]
      chosen_attention_mask: [B, T]
      rejected_input_ids: [B, T]
      rejected_attention_mask: [B, T]
      prompt_lens: [B]

    Returns:
      DPOBatchMetrics with loss + useful debug stats.

    DPO math (per example):
      Δθ   = logπθ(y+|x)  - logπθ(y-|x)
      Δref = logπref(y+|x) - logπref(y-|x)
      z    = β * (Δθ - Δref)
      loss = -log σ(z)
    """
    # basic batch assertions
    assert "chosen_input_ids" in batch, "Missing 'chosen_input_ids' in batch"
    assert "chosen_attention_mask" in batch, "Missing 'chosen_attention_mask' in batch"
    assert "rejected_input_ids" in batch, "Missing 'rejected_input_ids' in batch"
    assert "rejected_attention_mask" in batch, (
        "Missing 'rejected_attention_mask' in batch"
    )
    assert "prompt_lens" in batch, "Missing 'prompt_lens' in batch"

    assert batch["prompt_lens"].dtype == torch.long, (
        "'prompt_lens' should be torch.long"
    )
    assert beta > 0, "Beta must be >0"

    chosen_input_ids = batch["chosen_input_ids"]
    chosen_attention_mask = batch["chosen_attention_mask"]
    rejected_input_ids = batch["rejected_input_ids"]
    rejected_attention_mask = batch["rejected_attention_mask"]
    prompt_lens = batch["prompt_lens"]

    assert (
        chosen_input_ids.shape
        == chosen_attention_mask.shape
        == rejected_input_ids.shape
        == rejected_attention_mask.shape
    ), "Mismatch input_ids and attention_mask shape"

    # compute appropriate (with grad) logprobs
    sum_logp, mean_logp, _, mask_c = completion_logprobs(
        policy_model, chosen_input_ids, chosen_attention_mask, prompt_lens, require_grad=True
    )
    logp_policy_chosen = mean_logp if use_mean_logp else sum_logp
    chosen_tok_count = mask_c.sum(dim=1)

    sum_logp, mean_logp, _, mask_c = completion_logprobs(
        policy_model, rejected_input_ids, rejected_attention_mask, prompt_lens, require_grad=True
    )
    logp_policy_rejected = mean_logp if use_mean_logp else sum_logp
    rejected_tok_count = mask_c.sum(dim=1)

    # compute reference logprobs
    with torch.inference_mode():
        sum_logp, mean_logp, _, _ = completion_logprobs(
            ref_model, chosen_input_ids, chosen_attention_mask, prompt_lens
        )
        logp_ref_chosen = mean_logp if use_mean_logp else sum_logp

        sum_logp, mean_logp, _, _ = completion_logprobs(
            ref_model, rejected_input_ids, rejected_attention_mask, prompt_lens
        )
        logp_ref_rejected = mean_logp if use_mean_logp else sum_logp

    # compute deltas
    delta_policy = logp_policy_chosen - logp_policy_rejected
    delta_ref = logp_ref_chosen - logp_ref_rejected

    # compute loss
    dpo_logits = beta * (delta_policy - delta_ref)
    loss_vec = F.softplus(-dpo_logits)
    loss = loss_vec.mean()

    # basic batch metrics
    margin = delta_policy - delta_ref
    acc = (dpo_logits > 0).float().mean()

    return DPOBatchMetrics(
        loss=loss,
        dpo_logits=dpo_logits,
        margin=margin,
        acc=acc,
        policy_logp_chosen=logp_policy_chosen,
        policy_logp_rejected=logp_policy_rejected,
        ref_logp_chosen=logp_ref_chosen,
        ref_logp_rejected=logp_ref_rejected,
        chosen_token_count=chosen_tok_count,
        rejected_token_count=rejected_tok_count,
    )


def dpo_step(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    beta: float,
    grad_clip_norm: Optional[float] = None,
    use_mean_logp: bool = False,
) -> DPOBatchMetrics:
    """
    Single optimization step: forward DPO loss, backward, optimizer step.

    Returns metrics for logging.
    """
    policy_model.train()
    ref_model.eval()

    metrics = dpo_loss_batch(
        policy_model=policy_model,
        ref_model=ref_model,
        batch=batch,
        beta=beta,
        use_mean_logp=use_mean_logp,
    )

    optimizer.zero_grad(set_to_none=True)
    metrics.loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip_norm)
    optimizer.step()
    return metrics

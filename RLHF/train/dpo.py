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
    # Scalars (or [B] vectors) you'll want for logging/debugging
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
    grad_norm: Optional[torch.Tensor] = None  # scalar, gradient norm before clipping

    def pretty_print(self, prefix: str = "") -> None:
        """Print DPO metrics in a compact, terminal-friendly format.

        Displays the most important training metrics on 1-2 lines for easy monitoring.
        Key metrics shown:
        - loss: Training loss value
        - acc: Preference accuracy (fraction where policy prefers chosen over rejected)
        - margin: Mean policy advantage (logp_chosen - logp_rejected difference vs reference)
        - grad_norm: Gradient norm before clipping (if available)

        Args:
            prefix: String prefix for the output line.
        """
        # Helper to extract scalar values
        def val(x: torch.Tensor) -> float:
            return x.detach().item() if x.numel() == 1 else x.detach().mean().item()

        loss = val(self.loss)
        acc = val(self.acc)
        margin_mean = val(self.margin)

        # Build compact output
        line = f"{prefix}loss={loss:.4f} | acc={acc:.3f} | margin={margin_mean:+.3f}"

        # Add grad norm if available
        if self.grad_norm is not None:
            grad_norm = val(self.grad_norm)
            line += f" | grad={grad_norm:.2e}"

        print(line)


def freeze_model_(model: torch.nn.Module) -> None:
    """Freeze all model parameters in-place for reference model.

    Sets requires_grad to False for all parameters and puts model in eval mode.

    Args:
        model: Model to freeze.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def make_reference_model(
    policy_model: torch.nn.Module,
    *,
    method: str = "deepcopy",
) -> torch.nn.Module:
    """Create a frozen reference model for DPO training.

    Args:
        policy_model: The policy model to create reference from.
        method: Method for creating reference model. Options:
            - "deepcopy": Deep copy policy model (recommended for exact init match).
            - "load": Load separately from checkpoint (not implemented).

    Returns:
        torch.nn.Module: Frozen reference model in eval mode.

    Raises:
        ValueError: If method is not recognized.
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
    """Compute DPO loss for one batch.

    Implements Direct Preference Optimization loss computation. For each example:
        Δθ   = logπθ(y+|x)  - logπθ(y-|x)
        Δref = logπref(y+|x) - logπref(y-|x)
        z    = β * (Δθ - Δref)
        loss = -log σ(z)

    Args:
        policy_model: The policy model being trained.
        ref_model: Frozen reference model for KL constraint.
        batch: Batch dict from collate_preference_batch containing:
            - chosen_input_ids: Token IDs for chosen completions, shape [B, T].
            - chosen_attention_mask: Attention mask for chosen, shape [B, T].
            - rejected_input_ids: Token IDs for rejected completions, shape [B, T].
            - rejected_attention_mask: Attention mask for rejected, shape [B, T].
            - prompt_lens: Number of prompt tokens per example, shape [B].
        beta: DPO temperature parameter controlling KL penalty strength.
        use_mean_logp: Whether to use mean log prob (per-token) instead of sum.

    Returns:
        DPOBatchMetrics: Dataclass containing loss and diagnostic statistics.

    Raises:
        AssertionError: If batch is missing required keys or has invalid shapes.
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
    """Perform a single DPO training step with gradient update.

    Computes forward pass, backpropagates gradients, optionally clips them,
    and updates model parameters.

    Args:
        policy_model: The policy model being trained.
        ref_model: Frozen reference model.
        optimizer: Optimizer for updating policy_model.
        batch: Batch dict from collate_preference_batch.
        beta: DPO temperature parameter.
        grad_clip_norm: Optional gradient clipping max norm.
        use_mean_logp: Whether to use mean log prob instead of sum.

    Returns:
        DPOBatchMetrics: Metrics from the training step for logging (includes grad_norm).
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

    # Compute gradient norm before clipping
    total_norm = torch.nn.utils.clip_grad_norm_(
        policy_model.parameters(), float('inf')
    )
    metrics.grad_norm = total_norm

    # Apply gradient clipping if specified
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip_norm)

    optimizer.step()
    return metrics

# train/dpo.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine import completion_logprobs


@dataclass
class IPOBatchMetrics:
    """Batch-level metrics for IPO (Identity Preference Optimization).

    All tensors are on the same device as the model.

    Attributes:
        loss: Scalar training loss (mean squared error).
        margin: Policy advantage over reference, shape [B].
        target: Target margin value (1 / beta).
        error: Difference between margin and target, shape [B].
        acc: Diagnostic accuracy (fraction where margin > 0).
        policy_logp_chosen: Policy log probs for chosen completions, shape [B].
        policy_logp_rejected: Policy log probs for rejected completions, shape [B].
        ref_logp_chosen: Reference log probs for chosen completions, shape [B].
        ref_logp_rejected: Reference log probs for rejected completions, shape [B].
        chosen_token_count: Number of tokens in chosen completions, shape [B].
        rejected_token_count: Number of tokens in rejected completions, shape [B].
        grad_norm: Gradient norm before clipping (set by ipo_step).
    """

    loss: torch.Tensor  # scalar
    margin: torch.Tensor  # [B] = (Δ_policy - Δ_ref)
    target: float  # scalar = 1 / beta
    error: torch.Tensor  # [B] = margin - target

    # diagnostic only (not used for training)
    acc: torch.Tensor  # scalar = mean(margin > 0)

    # logprobs (sum or mean depending on config)
    policy_logp_chosen: torch.Tensor  # [B]
    policy_logp_rejected: torch.Tensor  # [B]
    ref_logp_chosen: torch.Tensor  # [B]
    ref_logp_rejected: torch.Tensor  # [B]

    # optional diagnostics
    chosen_token_count: torch.Tensor | None = None  # [B]
    rejected_token_count: torch.Tensor | None = None  # [B]
    grad_norm: torch.Tensor | None = None  # scalar, gradient norm before clipping

    def pretty_print(self, prefix: str = "") -> None:
        """Print IPO metrics in a compact, terminal-friendly format.

        Displays the most important training metrics on 1-2 lines for easy monitoring.
        Key metrics shown:
        - loss: Training loss value (MSE of margin vs target)
        - acc: Preference accuracy (fraction where policy prefers chosen over rejected)
        - margin: Mean policy advantage over reference
        - target: Target margin value (1/beta)
        - error: Mean deviation from target margin
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
        error_mean = val(self.error)

        # Build compact output
        line = (
            f"{prefix}loss={loss:.4f} | acc={acc:.3f} | "
            f"margin={margin_mean:+.3f} (target={self.target:.3f}, err={error_mean:+.3f})"
        )

        # Add grad norm if available
        if self.grad_norm is not None:
            grad_norm = val(self.grad_norm)
            line += f" | grad={grad_norm:.2e}"

        print(line)


def ipo_loss_batch(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    beta: float,
    use_mean_logp: bool = False,
) -> IPOBatchMetrics:
    """Compute IPO loss for one batch.

    Implements Identity Preference Optimization loss computation:
        Δθ   = logπθ(y+|x)  - logπθ(y-|x)
        Δref = logπref(y+|x) - logπref(y-|x)
        margin = Δθ - Δref
        target = 1 / beta
        error  = margin - target
        loss = mean(error^2)

    Args:
        policy_model: The policy model being trained.
        ref_model: Frozen reference model for KL constraint.
        batch: Batch dict from collate_preference_batch containing:
            - chosen_input_ids: Token IDs for chosen completions, shape [B, T].
            - chosen_attention_mask: Attention mask for chosen, shape [B, T].
            - rejected_input_ids: Token IDs for rejected completions, shape [B, T].
            - rejected_attention_mask: Attention mask for rejected, shape [B, T].
            - prompt_lens: Number of prompt tokens per example, shape [B].
        beta: IPO temperature parameter (target margin = 1/beta).
        use_mean_logp: Whether to use mean log prob (per-token) instead of sum.

    Returns:
        IPOBatchMetrics: Dataclass containing loss and diagnostic statistics.

    Raises:
        AssertionError: If batch is missing required keys or has invalid shapes.

    Note:
        Policy logprobs are computed with gradients enabled, reference logprobs
        are computed in inference mode.
    """
    # basic batch assertions
    assert "chosen_input_ids" in batch, "Missing 'chosen_input_ids' in batch"
    assert "chosen_attention_mask" in batch, "Missing 'chosen_attention_mask' in batch"
    assert "rejected_input_ids" in batch, "Missing 'rejected_input_ids' in batch"
    assert "rejected_attention_mask" in batch, "Missing 'rejected_attention_mask' in batch"
    assert "prompt_lens" in batch, "Missing 'prompt_lens' in batch"

    assert batch["prompt_lens"].dtype == torch.long, "'prompt_lens' should be torch.long"
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
    margin = delta_policy - delta_ref

    # compute loss
    target = 1.0 / float(beta)
    error = margin - target
    loss = error.square().mean()
    acc = (margin > 0).float().mean()

    return IPOBatchMetrics(
        loss=loss,
        margin=margin,
        target=target,
        error=error,
        acc=acc,
        policy_logp_chosen=logp_policy_chosen,
        policy_logp_rejected=logp_policy_rejected,
        ref_logp_chosen=logp_ref_chosen,
        ref_logp_rejected=logp_ref_rejected,
        chosen_token_count=chosen_tok_count,
        rejected_token_count=rejected_tok_count,
    )


def ipo_step(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    beta: float,
    grad_clip_norm: float | None = None,
    use_mean_logp: bool = False,
) -> IPOBatchMetrics:
    """Perform a single IPO training step with gradient update.

    Computes forward pass, backpropagates gradients, optionally clips them,
    and updates model parameters.

    Args:
        policy_model: The policy model being trained.
        ref_model: Frozen reference model.
        optimizer: Optimizer for updating policy_model.
        batch: Batch dict from collate_preference_batch.
        beta: IPO temperature parameter.
        grad_clip_norm: Optional gradient clipping max norm.
        use_mean_logp: Whether to use mean log prob instead of sum.

    Returns:
        IPOBatchMetrics: Metrics from the training step for logging (includes grad_norm).
    """
    policy_model.train()
    ref_model.eval()

    metrics = ipo_loss_batch(
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
        policy_model.parameters(), grad_clip_norm or float("inf")
    )
    metrics.grad_norm = total_norm

    optimizer.step()
    return metrics

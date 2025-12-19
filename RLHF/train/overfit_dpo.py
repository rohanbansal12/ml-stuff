# train/overfit_dpo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenize import tokenize_preference_example
from data.collate import collate_preference_batch
from train.dpo import make_reference_model, dpo_step
from engine import completion_logprobs, load_model, load_tokenizer


DEFAULT_SYSTEM = "You are a helpful assistant."
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class OverfitConfig:
    n_samples: int = 500
    max_len: int = 256
    batch_size: int = 4
    epochs: int = 20
    lr: float = 5e-6
    beta: float = 0.1
    num_workers: int = 4
    grad_clip_norm: Optional[float] = 1.0
    use_mean_logp: bool = False

    # eval cadence
    eval_every_steps: int = 20
    print_every_steps: int = 10


class TinyPreferenceDataset(Dataset):
    def __init__(
        self, tokenizer, raw_examples: Sequence[Dict[str, Any]], *, max_len: int
    ):
        self.tokenizer = tokenizer
        self.raw_examples = list(raw_examples)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.raw_examples)

    def __getitem__(self, idx: int) -> Any:
        ex = self.raw_examples[idx]
        return tokenize_preference_example(
            self.tokenizer, example=ex, max_len=self.max_len
        )


def move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Move all tensors in a batch dictionary to the specified device.

    Args:
        batch: Dictionary containing tensors and other values.
        device: Target device for tensors.

    Returns:
        Dict[str, torch.Tensor]: Batch with tensors moved to device.
    """
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.inference_mode()
def preference_eval_stats(
    policy_model: torch.nn.Module,
    dataloader: DataLoader,
    use_mean_logp: bool = False,
) -> Dict[str, float]:
    """Compute evaluation statistics for preference pairs.

    Returns both sum-logp and mean-logp diagnostics to detect length bias.

    Args:
        policy_model: Model to evaluate.
        dataloader: DataLoader with preference pair batches.
        use_mean_logp: Whether to use mean logp as primary metric
            (otherwise uses sum logp).

    Returns:
        Dict[str, float]: Dictionary containing:
            - win_rate: Primary win rate (sum or mean based on use_mean_logp).
            - avg_margin: Primary margin (sum or mean based on use_mean_logp).
            - win_rate_sum: P(sum_logp_chosen > sum_logp_rejected).
            - win_rate_mean: P(mean_logp_chosen > mean_logp_rejected).
            - avg_margin_sum: E[sum_logp_chosen - sum_logp_rejected].
            - avg_margin_mean: E[mean_logp_chosen - mean_logp_rejected].
            - avg_chosen_len: Average chosen completion token count.
            - avg_rejected_len: Average rejected completion token count.
            - avg_len_gap: Average (rejected_len - chosen_len).
    """
    policy_model.eval()
    device = next(policy_model.parameters()).device

    wins_sum = 0
    wins_mean = 0
    total = 0

    margin_sum_total = 0.0
    margin_mean_total = 0.0

    chosen_len_total = 0.0
    rejected_len_total = 0.0
    len_gap_total = 0.0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        prompt_lens = batch["prompt_lens"]

        sum_c, mean_c, _, mask_c = completion_logprobs(
            policy_model,
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            prompt_lens=prompt_lens,
        )

        sum_r, mean_r, _, mask_r = completion_logprobs(
            policy_model,
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            prompt_lens=prompt_lens,
        )

        # token counts per example
        chosen_len = mask_c.sum(dim=1).float()  # [B]
        rejected_len = mask_r.sum(dim=1).float()  # [B]

        # margins per example
        m_sum = sum_c - sum_r  # [B]
        m_mean = mean_c - mean_r  # [B]

        wins_sum += int((m_sum > 0).sum().item())
        wins_mean += int((m_mean > 0).sum().item())
        total += int(m_sum.numel())

        margin_sum_total += float(m_sum.sum().item())
        margin_mean_total += float(m_mean.sum().item())

        chosen_len_total += float(chosen_len.sum().item())
        rejected_len_total += float(rejected_len.sum().item())
        len_gap_total += float((rejected_len - chosen_len).sum().item())

    denom = max(total, 1)
    win_rate_sum = wins_sum / denom
    win_rate_mean = wins_mean / denom

    avg_margin_sum = margin_sum_total / denom
    avg_margin_mean = margin_mean_total / denom

    avg_chosen_len = chosen_len_total / denom
    avg_rejected_len = rejected_len_total / denom
    avg_len_gap = len_gap_total / denom  # positive => rejected longer on avg

    # convenience "primary" fields
    if use_mean_logp:
        win_rate = win_rate_mean
        avg_margin = avg_margin_mean
    else:
        win_rate = win_rate_sum
        avg_margin = avg_margin_sum

    return {
        # primary (matches your previous keys)
        "win_rate": win_rate,
        "avg_margin": avg_margin,
        "avg_chosen_len": avg_chosen_len,
        "avg_rejected_len": avg_rejected_len,
        # extra diagnostics (length bias visibility)
        "win_rate_sum": win_rate_sum,
        "win_rate_mean": win_rate_mean,
        "avg_margin_sum": avg_margin_sum,
        "avg_margin_mean": avg_margin_mean,
        "avg_len_gap": avg_len_gap,
    }


def build_tiny_examples(n: int = 100):
    """Build a small dataset of preference examples from Intel/orca_dpo_pairs.

    Args:
        n: Number of examples to load.

    Returns:
        List[Dict[str, Any]]: List of preference examples with messages,
            chosen, and rejected fields.
    """
    ds = load_dataset("Intel/orca_dpo_pairs", split="train")
    ds = ds.shuffle(seed=0).select(range(n))

    examples = []
    for ex in ds:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": ex["system"]},
                    {"role": "user", "content": ex["question"]},
                ],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            }
        )
    return examples


def overfit_dpo(
    policy_model: torch.nn.Module,
    tokenizer,
    cfg: OverfitConfig,
) -> None:
    """Run DPO training to overfit on a small dataset.

    Trains a policy model using Direct Preference Optimization on a small
    subset of data, useful for debugging and sanity checking the training loop.

    Args:
        policy_model: The model to train.
        tokenizer: Tokenizer for processing text.
        cfg: Configuration for overfitting experiment.
    """
    # load examples
    raw_examples = build_tiny_examples(cfg.n_samples)

    # split train/val
    n = len(raw_examples)
    split = int(0.9 * n)
    train_examples = raw_examples[:split]
    val_examples = raw_examples[split:]

    # Dataset + loader
    train_ds = TinyPreferenceDataset(tokenizer, train_examples, max_len=cfg.max_len)
    train_loader = DataLoader(
        train_ds,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda items: collate_preference_batch(tokenizer, items),
        pin_memory=True,
    )

    val_ds = TinyPreferenceDataset(tokenizer, val_examples, max_len=cfg.max_len)
    val_loader = DataLoader(
        val_ds,
        cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda items: collate_preference_batch(tokenizer, items),
        pin_memory=True,
    )

    # Create frozen reference model
    ref_model = make_reference_model(policy_model, method="deepcopy")
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.lr)
    device = next(policy_model.parameters()).device

    # Initial eval
    br = preference_eval_stats(
        policy_model, train_loader, use_mean_logp=cfg.use_mean_logp
    )
    print(f"Base Win-Rate: {br['win_rate']:.2f}")

    # Train loop
    for epoch in range(cfg.epochs):
        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device=device)
            metrics = dpo_step(
                policy_model=policy_model,
                ref_model=ref_model,
                optimizer=optimizer,
                batch=batch,
                beta=cfg.beta,
                grad_clip_norm=cfg.grad_clip_norm,
                use_mean_logp=cfg.use_mean_logp,
            )
            if (step + 1) % cfg.print_every_steps == 0:
                metrics.pretty_print(prefix=f"[epoch {epoch:02d} step {step + 1:04d}] ")

            if (step + 1) % cfg.eval_every_steps == 0:
                tr = preference_eval_stats(
                    policy_model, train_loader, use_mean_logp=cfg.use_mean_logp
                )
                va = preference_eval_stats(
                    policy_model, val_loader, use_mean_logp=cfg.use_mean_logp
                )

                # Print compact eval stats with clear train/val distinction
                print(
                    f"[epoch {epoch:02d} step {step + 1:04d}] "
                    f"EVAL TRAIN | "
                    f"wr={tr['win_rate_sum']:.2f}/{tr['win_rate_mean']:.2f} | "
                    f"margin={tr['avg_margin_sum']:+.1f}/{tr['avg_margin_mean']:+.3f} | "
                    f"len={tr['avg_chosen_len']:.0f}/{tr['avg_rejected_len']:.0f}"
                )
                print(
                    f"[epoch {epoch:02d} step {step + 1:04d}] "
                    f"EVAL VAL   | "
                    f"wr={va['win_rate_sum']:.2f}/{va['win_rate_mean']:.2f} | "
                    f"margin={va['avg_margin_sum']:+.1f}/{va['avg_margin_mean']:+.3f} | "
                    f"len={va['avg_chosen_len']:.0f}/{va['avg_rejected_len']:.0f}"
                )


def main():
    """Main entry point for DPO overfitting experiment.

    Loads model and tokenizer, then runs overfitting training with default config.
    """
    device = torch.device(DEVICE)
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, dtype=torch.bfloat16, device=device)
    cfg = OverfitConfig()
    overfit_dpo(model, tokenizer, cfg)


if __name__ == "__main__":
    main()

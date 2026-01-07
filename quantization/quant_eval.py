from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------


def load_model(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a HuggingFace model and tokenizer, move to device, set to eval mode.

    Args:
        model_name: HuggingFace model identifier
        device: Target device (auto-detected if None)
        dtype: Model dtype (float16 for GPU, float32 for CPU)

    Returns:
        (model, tokenizer) tuple ready for inference
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adjust dtype for CPU (float16 is slow on CPU)
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
        print("Note: Using float32 on CPU for better performance")

    print(f"Loading {model_name}...")
    print(f"  Device: {device}, Dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,  # Handles moving to device
    )
    model.eval()

    # Print model stats
    n_params = sum(p.numel() for p in model.parameters())
    memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params / 1e6:.1f}M")
    print(f"  Memory: {memory_mb:.1f} MB")

    return model, tokenizer


def get_device(model: PreTrainedModel) -> torch.device:
    """Get the device a model is on."""
    return next(model.parameters()).device


# -----------------------------------------------------------------------------
# HellaSwag Dataset
# -----------------------------------------------------------------------------


@dataclass
class HellaSwagSample:
    """Single HellaSwag evaluation sample."""

    context: str  # The story/scenario context
    endings: list[str]  # 4 possible endings
    label: int  # Index of correct ending (0-3)

    # Pre-tokenized versions (filled by load_hellaswag)
    context_tokens: list[int] | None = None
    ending_tokens: list[list[int]] | None = None


def load_hellaswag(
    tokenizer: PreTrainedTokenizer,
    n_samples: int | None = None,
    split: str = "validation",
) -> list[HellaSwagSample]:
    """
    Load HellaSwag dataset with pre-tokenization.

    HellaSwag tests commonsense reasoning by asking the model to pick
    the most plausible continuation of a scenario.

    Args:
        tokenizer: Tokenizer for pre-tokenization
        n_samples: Limit number of samples (None for full dataset)
        split: Dataset split ("validation" or "train")

    Returns:
        List of HellaSwagSample objects
    """
    print(f"Loading HellaSwag ({split})...")
    dataset = load_dataset("Rowan/hellaswag", split=split)

    if n_samples is not None:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    samples = []
    for item in tqdm(dataset, desc="Tokenizing"):
        # HellaSwag has activity_label, ctx_a, ctx_b, endings, label
        # ctx = ctx_a + " " + ctx_b gives the full context
        context = item["ctx_a"] + " " + item["ctx_b"]

        sample = HellaSwagSample(
            context=context,
            endings=item["endings"],
            label=int(item["label"]),
        )

        # Pre-tokenize for efficiency
        sample.context_tokens = tokenizer.encode(context, add_special_tokens=False)
        sample.ending_tokens = [
            tokenizer.encode(" " + ending, add_special_tokens=False) for ending in item["endings"]
        ]

        samples.append(sample)

    print(f"  Loaded {len(samples)} samples")
    return samples


def evaluate_hellaswag(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    samples: list[HellaSwagSample],
    batch_size: int = 1,  # Process one sample at a time (4 endings each)
) -> dict:
    """
    Evaluate model on HellaSwag by computing which ending has lowest perplexity.

    For each sample, we compute the average log-probability of each ending
    given the context, and pick the ending with highest probability.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer (for any needed encoding)
        samples: HellaSwag samples from load_hellaswag()
        batch_size: Samples to process at once

    Returns:
        Dict with 'accuracy', 'correct', 'total', and 'per_sample' details
    """
    device = get_device(model)
    correct = 0
    results = []

    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating HellaSwag"):
            ending_scores = []

            for ending_tokens in sample.ending_tokens:
                # Full sequence: context + ending
                full_tokens = sample.context_tokens + ending_tokens
                input_ids = torch.tensor([full_tokens], device=device)

                # Get logits
                outputs = model(input_ids)
                logits = outputs.logits  # [1, seq_len, vocab_size]

                # Compute log probs for the ending tokens only
                # We want P(ending | context), so look at positions where
                # we predict ending tokens (shifted by 1 for autoregressive)
                ending_start = len(sample.context_tokens)
                ending_len = len(ending_tokens)

                # Logits at position i predict token i+1
                # So logits[ending_start-1:ending_start-1+ending_len] predict ending tokens
                relevant_logits = logits[0, ending_start - 1 : ending_start - 1 + ending_len]
                target_tokens = torch.tensor(ending_tokens, device=device)

                # Log softmax and gather target token probs
                log_probs = F.log_softmax(relevant_logits, dim=-1)
                token_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)

                # Average log prob (length-normalized)
                avg_log_prob = token_log_probs.mean().item()
                ending_scores.append(avg_log_prob)

            # Pick ending with highest average log prob
            pred = max(range(4), key=lambda i: ending_scores[i])
            is_correct = pred == sample.label
            correct += is_correct

            results.append(
                {
                    "pred": pred,
                    "label": sample.label,
                    "correct": is_correct,
                    "scores": ending_scores,
                }
            )

    accuracy = correct / len(samples)
    print(f"  HellaSwag Accuracy: {accuracy:.4f} ({correct}/{len(samples)})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "per_sample": results,
    }


# -----------------------------------------------------------------------------
# WikiText Dataset (for Perplexity)
# -----------------------------------------------------------------------------


@dataclass
class WikiTextData:
    """Preprocessed WikiText data for perplexity evaluation."""

    tokens: Tensor  # [n_tokens] - full token sequence
    text: str  # Original text (for reference)
    n_tokens: int  # Total tokens


def load_wikitext(
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 8192,
    split: str = "test",
) -> WikiTextData:
    """
    Load WikiText-2 for perplexity evaluation.

    Args:
        tokenizer: Tokenizer for encoding
        max_tokens: Maximum tokens to use (for faster eval)
        split: Dataset split ("test", "validation", or "train")

    Returns:
        WikiTextData with pre-tokenized text
    """
    print(f"Loading WikiText-2 ({split})...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Concatenate all text
    text = "\n\n".join([item["text"] for item in dataset if item["text"].strip()])

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]

    # Truncate if needed
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"  Truncated to {max_tokens} tokens")

    print(f"  Loaded {len(tokens)} tokens")

    return WikiTextData(
        tokens=tokens,
        text=text[:1000] + "..." if len(text) > 1000 else text,
        n_tokens=len(tokens),
    )


def evaluate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    wikitext: WikiTextData,
    stride: int = 512,
    max_length: int | None = None,
) -> dict:
    """
    Compute perplexity on WikiText using sliding window.

    Uses a sliding window approach to handle sequences longer than
    the model's context length.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer (unused but kept for API consistency)
        wikitext: WikiText data from load_wikitext()
        stride: Step size for sliding window
        max_length: Context length (auto-detected from model if None)

    Returns:
        Dict with 'perplexity', 'avg_loss', 'n_tokens'
    """
    device = get_device(model)
    tokens = wikitext.tokens.to(device)

    # Get model's max context length
    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 2048)

    print(f"Computing perplexity (context={max_length}, stride={stride})...")

    nlls = []  # Negative log likelihoods
    n_tokens_scored = 0

    with torch.no_grad():
        # Sliding window over the sequence
        for start in tqdm(range(0, len(tokens), stride), desc="Perplexity"):
            end = min(start + max_length, len(tokens))
            input_ids = tokens[start:end].unsqueeze(0)

            # Target is shifted input (predict next token)
            target_ids = input_ids.clone()

            # For overlapping windows, only score new tokens (after stride)
            # This avoids double-counting tokens
            if start > 0:
                # Mask out the overlapping prefix in the loss
                n_overlap = max_length - stride
                target_ids[0, :n_overlap] = -100  # -100 is ignored in cross_entropy

            outputs = model(input_ids, labels=target_ids)

            # outputs.loss is mean over non-ignored tokens
            # We need to track how many tokens were actually scored
            n_scored = (target_ids != -100).sum().item()
            nlls.append(outputs.loss.item() * n_scored)
            n_tokens_scored += n_scored

            if end == len(tokens):
                break

    avg_nll = sum(nlls) / n_tokens_scored
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Avg NLL: {avg_nll:.4f}")
    print(f"  Tokens scored: {n_tokens_scored}")

    return {
        "perplexity": perplexity,
        "avg_loss": avg_nll,
        "n_tokens": n_tokens_scored,
    }


# -----------------------------------------------------------------------------
# Convenience: Full Evaluation Suite
# -----------------------------------------------------------------------------


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    hellaswag_samples: list[HellaSwagSample] | None = None,
    wikitext_data: WikiTextData | None = None,
    hellaswag_n: int = 500,
    wikitext_tokens: int = 4096,
) -> dict:
    """
    Run full evaluation suite on a model.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        hellaswag_samples: Pre-loaded HellaSwag (loads if None)
        wikitext_data: Pre-loaded WikiText (loads if None)
        hellaswag_n: Number of HellaSwag samples if loading
        wikitext_tokens: Number of WikiText tokens if loading

    Returns:
        Dict with 'hellaswag' and 'wikitext' results
    """
    results = {}

    # HellaSwag
    if hellaswag_samples is None:
        hellaswag_samples = load_hellaswag(tokenizer, n_samples=hellaswag_n)
    results["hellaswag"] = evaluate_hellaswag(model, tokenizer, hellaswag_samples)

    # WikiText perplexity
    if wikitext_data is None:
        wikitext_data = load_wikitext(tokenizer, max_tokens=wikitext_tokens)
    results["wikitext"] = evaluate_perplexity(model, tokenizer, wikitext_data)

    return results


# -----------------------------------------------------------------------------
# Quick Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check with a small model
    print("=" * 60)
    print("Quantization Evaluation - Sanity Check")
    print("=" * 60)

    # Use a small model for testing
    model, tokenizer = load_model("Qwen/Qwen2.5-1.5B")

    # Load small eval sets
    hellaswag = load_hellaswag(tokenizer, n_samples=100)
    wikitext = load_wikitext(tokenizer, max_tokens=2048)

    # Run evals
    print("\n" + "=" * 60)
    print("Baseline Evaluation (FP16)")
    print("=" * 60)
    results = evaluate_model(model, tokenizer, hellaswag, wikitext)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"HellaSwag Accuracy: {results['hellaswag']['accuracy']:.4f}")
    print(f"WikiText Perplexity: {results['wikitext']['perplexity']:.2f}")

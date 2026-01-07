"""
Baseline Evaluation Script

Establishes baseline metrics for the unquantized model before
applying quantization techniques.

Run this first to get your reference numbers:
    python run_baseline.py

Expected output for Qwen2.5-1.5B (FP16):
    HellaSwag: ~0.45-0.55 accuracy
    WikiText-2: ~10-15 perplexity
"""

import json
from datetime import datetime
from pathlib import Path

from quant_eval import (
    evaluate_hellaswag,
    evaluate_perplexity,
    load_hellaswag,
    load_model,
    load_wikitext,
)


def run_baseline(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    hellaswag_n: int = 500,
    wikitext_tokens: int = 4096,
    save_results: bool = True,
):
    """Run baseline evaluation and optionally save results."""

    print("=" * 70)
    print(f"BASELINE EVALUATION: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model(model_name)

    # Load datasets (do this once, reuse for quantized versions)
    print("\n" + "-" * 70)
    print("Loading Evaluation Datasets")
    print("-" * 70)
    hellaswag = load_hellaswag(tokenizer, n_samples=hellaswag_n)
    wikitext = load_wikitext(tokenizer, max_tokens=wikitext_tokens)

    # Run evaluations
    print("\n" + "-" * 70)
    print("Running Evaluations")
    print("-" * 70)

    hellaswag_results = evaluate_hellaswag(model, tokenizer, hellaswag)
    wikitext_results = evaluate_perplexity(model, tokenizer, wikitext)

    # Compile results
    results = {
        "model": model_name,
        "quantization": "none (FP16 baseline)",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "hellaswag_n": hellaswag_n,
            "wikitext_tokens": wikitext_tokens,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
        },
        "metrics": {
            "hellaswag_accuracy": hellaswag_results["accuracy"],
            "hellaswag_correct": hellaswag_results["correct"],
            "hellaswag_total": hellaswag_results["total"],
            "wikitext_perplexity": wikitext_results["perplexity"],
            "wikitext_avg_loss": wikitext_results["avg_loss"],
            "wikitext_tokens_scored": wikitext_results["n_tokens"],
        },
    }

    # Print summary
    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print("Precision: FP16 (baseline)")
    print(f"\nHellaSwag Accuracy: {hellaswag_results['accuracy']:.4f}")
    print(f"  ({hellaswag_results['correct']}/{hellaswag_results['total']} correct)")
    print(f"\nWikiText-2 Perplexity: {wikitext_results['perplexity']:.2f}")
    print(f"  (avg loss: {wikitext_results['avg_loss']:.4f})")

    # Save results
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        filename = f"baseline_{model_name.replace('/', '_')}.json"
        filepath = results_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filepath}")

    return results, hellaswag, wikitext


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline model evaluation")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B", help="HuggingFace model name"
    )
    parser.add_argument("--hellaswag-n", type=int, default=500, help="Number of HellaSwag samples")
    parser.add_argument(
        "--wikitext-tokens",
        type=int,
        default=4096,
        help="Number of WikiText tokens for perplexity",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    run_baseline(
        model_name=args.model,
        hellaswag_n=args.hellaswag_n,
        wikitext_tokens=args.wikitext_tokens,
        save_results=not args.no_save,
    )

import argparse
from pathlib import Path
from typing import Dict, List, Any, Sequence

import torch
from tqdm import tqdm

# -----------------------------
# Your existing utilities
# -----------------------------
# Expect these to already exist (or move them into routing_utils / viz_utils)
from viz_utils import (
    compute_metrics_from_counts,
    plot_alpha_sweep_curves,          # you'd implement: per-prompt overlay curves across alphas
    plot_alpha_sweep_scalar_summary,  # you'd implement: mean eff_experts vs alpha (per prompt and/or aggregated)
)

# If you already have these in your script, move them to routing_utils.py and import
from routing_utils import (
    build_model_and_tokenizer,
    attach_router_hooks,
    run_prefill,
    run_generate,
    decode_from_gen_minus_pre,
    summarize_routing_data,
    scale_router_logits_via_weight
)

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    "code": "Write a Python function for quicksort.",
    "math": "Solve the integral of x^2.",
    "creative": "Write a poem about rust.",
    "chat": "How are you?",
    "gibberish": "dsf jkl jkl",
}

DEFAULT_ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_N_EXPERTS = 60

def parse_alphas(xs: Sequence[str]) -> List[float]:
    out: List[float] = []
    for x in xs:
        try:
            out.append(float(x))
        except ValueError as e:
            raise ValueError(f"Invalid alpha '{x}' (expected float).") from e
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="/workspace/ml-stuff/MoE/plots/alpha_sweep")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--n_experts", type=int, default=DEFAULT_N_EXPERTS)

    # pass alphas as: --alphas 0.5 1 2 4 8
    parser.add_argument("--alphas", nargs="+", default=[str(x) for x in DEFAULT_ALPHAS])

    args = parser.parse_args()

    device = torch.device(DEVICE)

    model, layer_att, tokenizer, suffix = build_model_and_tokenizer(
        MODEL_NAME, device=device, adapter_path=args.adapter_path
    )

    alphas = parse_alphas(args.alphas)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Attach router observation hooks (same for all modes)
    routing_store = {}
    num_layers = attach_router_hooks(layer_att, routing_store)
    sweep_results: Dict[str, Dict[float, Dict[str, Any]]] = {}

    for prompt_key, prompt in tqdm(PROMPTS.items(), desc="alpha_sweep prompts"):
        sweep_results[prompt_key] = {}

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        for alpha in alphas:
            # -----------------------------------------
            # Run with router temperature scaling
            # -----------------------------------------
            # Important: scaling must apply to both prefill and generate
            with scale_router_logits_via_weight(layer_att, alpha):
                run_prefill(model, inputs, store=routing_store)
                pre_sum = summarize_routing_data(
                    routing_store,
                    num_layers=num_layers,
                    n_experts=args.n_experts,
                    compute_importance=True,
                )

                run_generate(model, inputs, max_new_tokens=args.max_new_tokens, store=routing_store)
                gen_sum = summarize_routing_data(
                    routing_store,
                    num_layers=num_layers,
                    n_experts=args.n_experts,
                    compute_importance=True,
                )

            dec_sum = decode_from_gen_minus_pre(gen_sum, pre_sum)

            # Convert sums -> normalized metrics (load heatmaps, effective experts, etc.)
            pre_metrics = compute_metrics_from_counts(
                counts_top1=pre_sum["counts_top1"],
                total_events=pre_sum["total_events"],
                sum_probs=pre_sum["sum_probs"],
                sum_pmax=pre_sum["sum_pmax"],
                sum_margin=pre_sum["sum_margin"],
            )
            dec_metrics = compute_metrics_from_counts(
                counts_top1=dec_sum["counts_top1"],
                total_events=dec_sum["total_events"],
                sum_probs=dec_sum["sum_probs"],
                sum_pmax=dec_sum["sum_pmax"],
                sum_margin=dec_sum["sum_margin"],
            )

            sweep_results[prompt_key][alpha] = {
                "prefill_metrics": pre_metrics,
                "decode_metrics": dec_metrics,
            }

        # -----------------------------------------
        # Per-prompt plotting (overlay curves)
        # -----------------------------------------
        # Expect: for each prompt, plot decode effective experts vs layer for all alphas,
        # plus entropy/max-share curves if you compute them in compute_metrics_from_counts.
        plot_alpha_sweep_curves(
            sweep_results=sweep_results[prompt_key],
            out_dir=out_dir,
            prompt_key=prompt_key,
            suffix=suffix,
            tag="decode",  # focus on decode by default
        )

    # -----------------------------------------
    # Global / scalar summary plot(s)
    # -----------------------------------------
    # Example: mean effective experts across layers (decode) vs alpha for each prompt.
    plot_alpha_sweep_scalar_summary(
        all_results=sweep_results,
        out_dir=out_dir,
        suffix=suffix,
        tag="decode",
    )

if __name__ == "__main__":
    main()
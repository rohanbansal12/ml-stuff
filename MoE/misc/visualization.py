import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from viz_utils import (
    compute_metrics_from_counts,
    plot_prefill_vs_decode_heatmaps,
    plot_prefill_vs_decode_curves,
    plot_eff_experts_across_prompts,
    plot_ablation_dashboard,
    top_expert_per_layer_from_counts,
)

from routing_utils import (
    run_generate,
    run_prefill,
    summarize_routing_data,
    build_model_and_tokenizer,
    ablate_experts,
    decode_from_gen_minus_pre,
    attach_router_hooks
)

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    "code": "Write a Python function for quicksort.",
    "math": "Solve the integral of x^2.",
    "creative": "Write a poem about rust.",
    "chat": "How are you?",
    "gibberish": "dsf jkl jkl",
}


MAX_NEW_TOKENS = 50
N_EXPERTS = 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument(
        "--out_dir", type=str, default="/workspace/ml-stuff/MoE/plots/baseline"
    )
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--n_experts", type=int, default=N_EXPERTS)
    parser.add_argument(
        "--mode",
        type=str,
        default="base",
        choices=["base", "ablate-prefill", "ablate-decode"],
        help="base: dashboards; ablate-prefill: ablate top expert per layer in prefill; "
        "ablate-decode: ablate top expert per layer in decode (derived from baseline decode).",
    )
    args = parser.parse_args()

    device = torch.device(DEVICE)

    model, layer_att, tokenizer, suffix = build_model_and_tokenizer(
        MODEL_NAME, device=device, adapter_path=args.adapter_path
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Attach router observation hooks (same for all modes)
    routing_store = {}
    num_layers = attach_router_hooks(layer_att, routing_store)

    # Store for global comparisons (base mode)
    results = {}

    for key, prompt in tqdm(PROMPTS.items(), desc=f"mode={args.mode} prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # -----------------------
        # Baseline prefill + generate
        # -----------------------
        run_prefill(model, inputs, store=routing_store)
        pre_sum = summarize_routing_data(
            routing_store,
            num_layers=num_layers,
            n_experts=args.n_experts,
            compute_importance=True,
        )

        run_generate(
            model, inputs, max_new_tokens=args.max_new_tokens, store=routing_store
        )
        gen_sum = summarize_routing_data(
            routing_store,
            num_layers=num_layers,
            n_experts=args.n_experts,
            compute_importance=True,
        )

        dec_sum = decode_from_gen_minus_pre(gen_sum, pre_sum)

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

        if args.mode == "base":
            plot_prefill_vs_decode_heatmaps(
                heatmap_prefill=pre_metrics["heatmap_load"],
                heatmap_decode=dec_metrics["heatmap_load"],
                out_dir=out_dir,
                prompt_key=key,
                suffix=suffix,
            )
            plot_prefill_vs_decode_curves(
                pre=pre_metrics,
                dec=dec_metrics,
                out_dir=out_dir,
                prompt_key=key,
                suffix=suffix,
            )
            results[key] = {
                "eff_prefill": pre_metrics["eff_experts"],
                "eff_decode": dec_metrics["eff_experts"],
            }
            continue

        # -----------------------
        # Ablation modes
        # -----------------------
        if args.mode == "ablate-prefill":
            mask_by_layer = top_expert_per_layer_from_counts(pre_sum["counts_top1"])

            routing_store.clear()
            with ablate_experts(layer_att, mask_by_layer):
                with torch.no_grad():
                    _ = model(**inputs)

            pre_abl_sum = summarize_routing_data(
                routing_store,
                num_layers=num_layers,
                n_experts=args.n_experts,
                compute_importance=False,
            )
            pre_abl_metrics = compute_metrics_from_counts(
                counts_top1=pre_abl_sum["counts_top1"],
                total_events=pre_abl_sum["total_events"],
            )

            title_suffix = f" ({suffix})" if suffix else ""
            plot_ablation_dashboard(
                heatmap_base=pre_metrics["heatmap_load"],
                heatmap_ablated=pre_abl_metrics["heatmap_load"],
                out_dir=out_dir,
                title=f"Expert ablation (prefill): '{key}'{title_suffix}\nMasked: top expert per layer",
                filename=f"ablation_prefill_{key}.jpeg",
            )
            continue

        if args.mode == "ablate-decode":
            # pick top expert per layer from BASELINE decode
            mask_by_layer = top_expert_per_layer_from_counts(dec_sum["counts_top1"])

            # Run full generate with ablation enabled, then subtract prefill (also under ablation)
            # so decode-only stays consistent.

            # ablated prefill
            routing_store.clear()
            with ablate_experts(layer_att, mask_by_layer):
                with torch.no_grad():
                    _ = model(**inputs)
            pre_abl = summarize_routing_data(
                routing_store,
                num_layers=num_layers,
                n_experts=args.n_experts,
                compute_importance=True,
            )

            # ablated generate
            routing_store.clear()
            with ablate_experts(layer_att, mask_by_layer):
                with torch.no_grad():
                    _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            gen_abl = summarize_routing_data(
                routing_store,
                num_layers=num_layers,
                n_experts=args.n_experts,
                compute_importance=True,
            )

            dec_abl = decode_from_gen_minus_pre(gen_abl, pre_abl)

            dec_abl_metrics = compute_metrics_from_counts(
                counts_top1=dec_abl["counts_top1"],
                total_events=dec_abl["total_events"],
                sum_probs=dec_abl["sum_probs"],
                sum_pmax=dec_abl["sum_pmax"],
                sum_margin=dec_abl["sum_margin"],
            )

            title_suffix = f" ({suffix})" if suffix else ""
            plot_ablation_dashboard(
                heatmap_base=dec_metrics["heatmap_load"],
                heatmap_ablated=dec_abl_metrics["heatmap_load"],
                out_dir=out_dir,
                title=f"Expert ablation (decode): '{key}'{title_suffix}\nMasked: top decode expert per layer (from baseline)",
                filename=f"ablation_decode_{key}.jpeg",
            )
            continue

    if args.mode == "base" and results:
        plot_eff_experts_across_prompts(
            results, out_dir=out_dir, suffix=suffix, tag="prefill"
        )
        plot_eff_experts_across_prompts(
            results, out_dir=out_dir, suffix=suffix, tag="decode"
        )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Any

from viz_utils import (
    compute_metrics_from_counts,
    plot_prefill_vs_decode_heatmaps,
    plot_prefill_vs_decode_curves,
    plot_eff_experts_across_prompts,
    plot_ablation_dashboard,
    top_expert_per_layer_from_counts,
    ablate_experts,
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

routing_data = {}


def get_router_hook(layer_idx: int):
    """Creates a forward hook to capture routing logits for a specific layer.

    Args:
        layer_idx: The index of the MoE layer to monitor.

    Returns:
        A hook function that can be registered with PyTorch's register_forward_hook.
        The hook extracts routing logits and stores them in the global routing_data dict.
    """

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            data = output[0].detach().cpu()
        else:
            data = output.detach().cpu()
        routing_data.setdefault(layer_idx, []).append(data)
        return output

    return hook


def summarize_routing_data(
    routing_data: Dict[int, List[torch.Tensor]],
    *,
    num_layers: int,
    n_experts: int = 60,
    compute_importance: bool = True,
) -> Dict[str, Any]:
    """Converts captured routing logits into additive statistics.

    Processes the routing logits collected by forward hooks and computes aggregated
    statistics that can be used for analysis or subtraction (e.g., decode = generate - prefill).

    Args:
        routing_data: Dictionary mapping layer indices to lists of routing logit tensors.
            Each tensor has shape [T, E] where T is the number of tokens and E is the
            number of experts.
        num_layers: Total number of MoE layers in the model.
        n_experts: Number of experts per layer. Defaults to 60.
        compute_importance: Whether to compute importance metrics (softmax probabilities,
            max probs, margins). Defaults to True.

    Returns:
        A dictionary containing:
            - counts_top1: [L, E] int64 array of top-1 expert selection counts
            - total_events: [L] int64 array of total routing decisions per layer
            - sum_probs: [L, E] float64 array of sum of softmax probs (or None)
            - sum_pmax: [L] float64 array of sum of max probabilities (or None)
            - sum_margin: [L] float64 array of sum of (p1 - p2) margins (or None)
    """
    counts_top1 = np.zeros((num_layers, n_experts), dtype=np.int64)
    total_events = np.zeros(num_layers, dtype=np.int64)

    sum_probs = (
        np.zeros((num_layers, n_experts), dtype=np.float64)
        if compute_importance
        else None
    )
    sum_pmax = np.zeros(num_layers, dtype=np.float64) if compute_importance else None
    sum_margin = np.zeros(num_layers, dtype=np.float64) if compute_importance else None

    for layer_idx, data_list in routing_data.items():
        if layer_idx < 0 or layer_idx >= num_layers:
            continue
        if not data_list:
            continue

        layer_logits = torch.cat(data_list, dim=0)
        if layer_logits.ndim != 2 or layer_logits.shape[-1] != n_experts:
            continue

        layer_logits = layer_logits.to(torch.float32)
        T = int(layer_logits.shape[0])
        if T <= 0:
            continue

        # top-1 counts
        top1 = layer_logits.argmax(dim=-1)
        counts = top1.bincount(minlength=n_experts).cpu().numpy().astype(np.int64)
        counts_top1[layer_idx] += counts
        total_events[layer_idx] += int(counts.sum())

        if compute_importance:
            probs = torch.softmax(layer_logits, dim=-1)  # [T, E]
            sum_probs[layer_idx] += probs.sum(dim=0).cpu().numpy().astype(np.float64)

            p_sorted, _ = probs.sort(dim=-1, descending=True)
            sum_pmax[layer_idx] += float(p_sorted[:, 0].sum().item())
            sum_margin[layer_idx] += float(
                (p_sorted[:, 0] - p_sorted[:, 1]).sum().item()
            )

    return {
        "counts_top1": counts_top1,
        "total_events": total_events,
        "sum_probs": sum_probs,
        "sum_pmax": sum_pmax,
        "sum_margin": sum_margin,
    }


def run_prefill(model, inputs) -> None:
    """Runs model prefill (prompt processing) and captures routing data.

    Clears the global routing_data dictionary and performs a forward pass on the model
    to process the input prompt. Router hooks capture routing logits during this phase.

    Args:
        model: The language model to run inference on.
        inputs: Tokenized input tensors (typically from tokenizer with return_tensors='pt').
    """
    routing_data.clear()
    with torch.no_grad():
        _ = model(**inputs)


def run_generate(model, inputs, max_new_tokens: int) -> None:
    """Runs full model generation (prefill + decode) and captures routing data.

    Clears the global routing_data dictionary and performs text generation including both
    the prefill phase (prompt processing) and decode phase (token generation). Router hooks
    capture routing logits during both phases.

    Args:
        model: The language model to run generation on.
        inputs: Tokenized input tensors (typically from tokenizer with return_tensors='pt').
        max_new_tokens: Maximum number of tokens to generate.
    """
    routing_data.clear()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)


def decode_from_gen_minus_pre(
    gen_sum: Dict[str, Any], pre_sum: Dict[str, Any]
) -> Dict[str, Any]:
    """Extracts decode-only statistics by subtracting prefill from generation statistics.

    Computes the decode phase statistics by subtracting prefill routing statistics from
    the full generation (prefill + decode) statistics. All values are clipped to ensure
    non-negative results.

    Args:
        gen_sum: Dictionary of generation statistics from summarize_routing_data, containing
            the routing statistics for the full generation process (prefill + decode).
        pre_sum: Dictionary of prefill statistics from summarize_routing_data, containing
            the routing statistics for only the prompt processing phase.

    Returns:
        A dictionary containing decode-only statistics with the same structure as the input:
            - counts_top1: [L, E] int64 array of decode-only expert selection counts
            - total_events: [L] int64 array of decode-only routing decisions
            - sum_probs: [L, E] float64 array of decode-only softmax probs (or None)
            - sum_pmax: [L] float64 array of decode-only max probabilities (or None)
            - sum_margin: [L] float64 array of decode-only margins (or None)
    """
    decode_counts = np.clip(gen_sum["counts_top1"] - pre_sum["counts_top1"], 0, None)
    decode_total = np.clip(gen_sum["total_events"] - pre_sum["total_events"], 0, None)

    decode_sum_probs = None
    decode_sum_pmax = None
    decode_sum_margin = None
    if gen_sum["sum_probs"] is not None and pre_sum["sum_probs"] is not None:
        decode_sum_probs = np.clip(
            gen_sum["sum_probs"] - pre_sum["sum_probs"], 0.0, None
        )
    if gen_sum["sum_pmax"] is not None and pre_sum["sum_pmax"] is not None:
        decode_sum_pmax = np.clip(gen_sum["sum_pmax"] - pre_sum["sum_pmax"], 0.0, None)
    if gen_sum["sum_margin"] is not None and pre_sum["sum_margin"] is not None:
        decode_sum_margin = np.clip(
            gen_sum["sum_margin"] - pre_sum["sum_margin"], 0.0, None
        )

    return {
        "counts_top1": decode_counts,
        "total_events": decode_total,
        "sum_probs": decode_sum_probs,
        "sum_pmax": decode_sum_pmax,
        "sum_margin": decode_sum_margin,
    }


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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quantization_config
    ).to(device)

    layer_att = model
    suffix = ""
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        layer_att = model.base_model.model
        suffix = args.adapter_path.split("-")[-1]

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Attach router observation hooks (same for all modes)
    num_layers = len(layer_att.model.layers)

    for i, layer in enumerate(layer_att.model.layers):
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.register_forward_hook(get_router_hook(i))

    # Store for global comparisons (base mode)
    results = {}

    for key, prompt in tqdm(PROMPTS.items(), desc=f"mode={args.mode} prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # -----------------------
        # Baseline prefill + generate
        # -----------------------
        run_prefill(model, inputs)
        pre_sum = summarize_routing_data(
            routing_data,
            num_layers=num_layers,
            n_experts=args.n_experts,
            compute_importance=True,
        )

        run_generate(model, inputs, max_new_tokens=args.max_new_tokens)
        gen_sum = summarize_routing_data(
            routing_data,
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

            routing_data.clear()
            with ablate_experts(layer_att, mask_by_layer):
                with torch.no_grad():
                    _ = model(**inputs)

            pre_abl_sum = summarize_routing_data(
                routing_data,
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
            routing_data.clear()
            with ablate_experts(layer_att, mask_by_layer):
                with torch.no_grad():
                    _ = model(**inputs)
            pre_abl = summarize_routing_data(
                routing_data,
                num_layers=num_layers,
                n_experts=args.n_experts,
                compute_importance=True,
            )

            # ablated generate
            routing_data.clear()
            with ablate_experts(layer_att, mask_by_layer):
                with torch.no_grad():
                    _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            gen_abl = summarize_routing_data(
                routing_data,
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

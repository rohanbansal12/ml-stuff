from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import argparse
from peft import PeftModel
import math
from typing import Dict, List, Optional, Any


MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 50

PROMPTS = {
    "code": "Write a Python function for quicksort.",
    "math": "Solve the integral of x^2.",
    "creative": "Write a poem about rust.",
    "chat": "How are you?",
    "gibberish": "dsf jkl jkl",
}

routing_data = {}


def get_router_hook(layer_idx):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            data = output[0].detach().cpu()
        else:
            data = output.detach().cpu()

        if layer_idx not in routing_data:
            routing_data[layer_idx] = []
        routing_data[layer_idx].append(data)

    return hook


def _safe_entropy_from_dist(p: np.ndarray, eps: float = 1e-12) -> float:
    """Entropy of a discrete distribution p (assumed nonnegative)."""
    p = p.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = np.clip(p, 0.0, 1.0)
    m = p > 0
    return float(-(p[m] * np.log(p[m] + eps)).sum())


def compute_metrics_from_counts(
    *,
    counts_top1: np.ndarray,  # [L, E] int64
    total_events: np.ndarray,  # [L] int64
    sum_probs: Optional[np.ndarray] = None,  # [L, E] float64 (sum of probs)
    sum_pmax: Optional[np.ndarray] = None,  # [L] float64 (sum over tokens of max prob)
    sum_margin: Optional[np.ndarray] = None,  # [L] float64 (sum over tokens of (p1-p2))
) -> Dict[str, Any]:
    """
    Pure computation: converts additive routing statistics into per-layer distributions and scalar metrics.
    Does NOT write any plots.

    Returns dict with:
      - heatmap_load: [L,E] float64            (top-1 selection frequency per layer)
      - heatmap_importance: [L,E] float64|None (mean softmax prob mass per layer)
      - top1_share: [L] float64               (max load per layer)
      - hhi: [L] float64                      (sum load^2 per layer)
      - entropy_load: [L] float64             (entropy(load) per layer)
      - eff_experts: [L] float64              (exp(entropy(load)) per layer)
      - mean_pmax: [L] float64|None           (mean max prob per layer)
      - mean_margin: [L] float64|None         (mean (p1-p2) per layer)
    """
    counts_top1 = np.asarray(counts_top1)
    total_events = np.asarray(total_events)

    L, E = counts_top1.shape
    heatmap_load = np.zeros((L, E), dtype=np.float64)

    top1_share = np.zeros(L, dtype=np.float64)
    hhi = np.zeros(L, dtype=np.float64)
    entropy_load = np.zeros(L, dtype=np.float64)
    eff_experts = np.zeros(L, dtype=np.float64)

    for l in range(L):
        tot = float(total_events[l])
        if tot <= 0:
            continue
        load = counts_top1[l].astype(np.float64) / tot
        heatmap_load[l] = load
        top1_share[l] = float(load.max())
        hhi[l] = float((load * load).sum())
        ent = _safe_entropy_from_dist(load)
        entropy_load[l] = ent
        eff_experts[l] = math.exp(ent)

    heatmap_importance = None
    if sum_probs is not None:
        sum_probs = np.asarray(sum_probs, dtype=np.float64)
        heatmap_importance = np.zeros((L, E), dtype=np.float64)
        for l in range(L):
            tot = float(total_events[l])
            if tot <= 0:
                continue
            heatmap_importance[l] = sum_probs[l] / tot

    mean_pmax = None
    mean_margin = None
    if sum_pmax is not None and sum_margin is not None:
        sum_pmax = np.asarray(sum_pmax, dtype=np.float64)
        sum_margin = np.asarray(sum_margin, dtype=np.float64)
        mean_pmax = np.zeros(L, dtype=np.float64)
        mean_margin = np.zeros(L, dtype=np.float64)
        for l in range(L):
            tot = float(total_events[l])
            if tot <= 0:
                continue
            mean_pmax[l] = sum_pmax[l] / tot
            mean_margin[l] = sum_margin[l] / tot

    return {
        "heatmap_load": heatmap_load,
        "heatmap_importance": heatmap_importance,
        "top1_share": top1_share,
        "hhi": hhi,
        "entropy_load": entropy_load,
        "eff_experts": eff_experts,
        "mean_pmax": mean_pmax,
        "mean_margin": mean_margin,
    }


def summarize_routing_data(
    routing_data: Dict[int, List[torch.Tensor]],
    *,
    num_layers: int,
    n_experts: int = 60,
    compute_importance: bool = True,
) -> dict:
    """
    Convert routing_data into additive statistics suitable for subtraction.

    Returns:
      - counts_top1: [L, E] int64  (top-1 selection counts)
      - total_events: [L] int64    (total router decisions)
      - sum_probs: [L, E] float64  (sum of softmax probs over tokens)  [optional]
      - sum_pmax: [L] float64      (sum over tokens of max prob)
      - sum_margin: [L] float64    (sum over tokens of (p1 - p2))
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


def plot_prefill_vs_decode_heatmaps(
    *,
    heatmap_prefill: np.ndarray,
    heatmap_decode: np.ndarray,
    out_dir: Path,
    prompt_key: str,
    suffix: str = "",
    xtick_every: int = 5,
    ytick_every: int = 2,
    filename_prefix: str = "dashboard_heatmaps",
):
    """One figure: prefill load heatmap vs decode load heatmap."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vmax = float(max(heatmap_prefill.max(), heatmap_decode.max()))
    vmin = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)

    sns.heatmap(
        heatmap_prefill,
        ax=axes[0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels=xtick_every,
        yticklabels=ytick_every,
        cbar=False,
    )
    axes[0].set_title("Load (Prefill)")
    axes[0].set_xlabel("Expert ID")
    axes[0].set_ylabel("Layer Depth")

    sns.heatmap(
        heatmap_decode,
        ax=axes[1],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels=xtick_every,
        yticklabels=ytick_every,
        cbar=True,
        cbar_kws={"label": "Top-1 Selection Frequency (Load)"},
    )
    axes[1].set_title("Load (Decode)")
    axes[1].set_xlabel("Expert ID")
    axes[1].set_ylabel("")

    title_suffix = f" ({suffix})" if suffix else ""
    fig.suptitle(f"Prefill vs Decode Routing: '{prompt_key}'{title_suffix}", fontsize=16)

    fig.savefig(out_dir / f"{filename_prefix}_{prompt_key}.jpeg", dpi=150)
    plt.close(fig)


def plot_prefill_vs_decode_curves(
    *,
    pre: Dict[str, Any],
    dec: Dict[str, Any],
    out_dir: Path,
    prompt_key: str,
    suffix: str = "",
    filename_prefix: str = "dashboard_curves",
):
    """
    One figure with 2 rows:
      Row1: effective experts (prefill vs decode)
      Row2: router confidence (pmax + margin) if available
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eff_prefill = pre["eff_experts"]
    eff_decode = dec["eff_experts"]
    L = len(eff_prefill)
    layers = np.arange(L)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)

    # Row 1: diversity
    axes[0].plot(layers, eff_prefill, label="prefill")
    axes[0].plot(layers, eff_decode, label="decode")
    axes[0].set_ylabel("Effective experts = exp(entropy(load))")
    axes[0].set_title("Routing diversity")
    axes[0].legend()

    # Row 2: confidence (optional)
    has_conf_pre = (pre.get("mean_pmax") is not None) and (pre.get("mean_margin") is not None)
    has_conf_dec = (dec.get("mean_pmax") is not None) and (dec.get("mean_margin") is not None)

    if has_conf_pre and has_conf_dec:
        axes[1].plot(layers, pre["mean_pmax"], label="pmax prefill")
        axes[1].plot(layers, dec["mean_pmax"], label="pmax decode")
        axes[1].plot(layers, pre["mean_margin"], label="margin prefill")
        axes[1].plot(layers, dec["mean_margin"], label="margin decode")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Router confidence")
        axes[1].legend(ncol=2)
    else:
        axes[1].text(
            0.02, 0.5,
            "Confidence curves unavailable (missing sum_pmax/sum_margin).",
            transform=axes[1].transAxes,
            va="center",
        )
        axes[1].set_axis_off()

    axes[1].set_xlabel("Layer depth")

    title_suffix = f" ({suffix})" if suffix else ""
    fig.suptitle(f"Prefill vs Decode Curves: '{prompt_key}'{title_suffix}", fontsize=16)

    fig.savefig(out_dir / f"{filename_prefix}_{prompt_key}.jpeg", dpi=150)
    plt.close(fig)


def plot_eff_experts_across_prompts(
    results: Dict[str, Dict[str, np.ndarray]],
    *,
    out_dir: Path,
    suffix: str = "",
    tag: str = "prefill",  # "prefill" or "decode"
    filename_prefix: str = "compare_effective_experts",
):
    """One plot comparing effective-experts curves across prompts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)

    for key, r in results.items():
        y = r["eff_prefill"] if tag == "prefill" else r["eff_decode"]
        ax.plot(np.arange(len(y)), y, label=key)

    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Effective experts")
    ax.set_title(f"Effective experts across prompts ({tag})" + (f" ({suffix})" if suffix else ""))
    ax.legend(ncol=3)

    fig.savefig(out_dir / f"{filename_prefix}_{tag}.jpeg", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument(
        "--out_dir", type=str, default="/workspace/ml-stuff/MoE/plots/baseline"
    )
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--n_experts", type=int, default=60)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
    ).to(device)
    layer_att = model
    suffix = ""

    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        layer_att = model.base_model.model
        suffix = args.adapter_path.split("-")[-1]

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else "/workspace/ml-stuff/MoE/plots/baseline"
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # attach input hooks to the model
    num_layers = len(layer_att.model.layers)
    for i, layer in enumerate(layer_att.model.layers):
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.register_forward_hook(get_router_hook(i))

    results = {}

    for key, prompt in tqdm(PROMPTS.items(), desc="enumerating prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # -----------------------
        # Prefill stats
        # -----------------------
        routing_data.clear()
        with torch.no_grad():
            _ = model(**inputs)

        pre_sum = summarize_routing_data(
            routing_data,
            num_layers=num_layers,
            n_experts=args.n_experts,
            compute_importance=True,
        )

        # -----------------------
        # Full generate stats
        # -----------------------
        routing_data.clear()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        gen_sum = summarize_routing_data(
            routing_data,
            num_layers=num_layers,
            n_experts=args.n_experts,
            compute_importance=True,
        )

        # -----------------------
        # Decode-only = generate - prefill
        # -----------------------
        decode_counts = np.clip(
            gen_sum["counts_top1"] - pre_sum["counts_top1"], 0, None
        )
        decode_total = np.clip(
            gen_sum["total_events"] - pre_sum["total_events"], 0, None
        )

        decode_sum_probs = None
        decode_sum_pmax = None
        decode_sum_margin = None

        if gen_sum["sum_probs"] is not None and pre_sum["sum_probs"] is not None:
            decode_sum_probs = np.clip(
                gen_sum["sum_probs"] - pre_sum["sum_probs"], 0.0, None
            )
        if gen_sum["sum_pmax"] is not None and pre_sum["sum_pmax"] is not None:
            decode_sum_pmax = np.clip(
                gen_sum["sum_pmax"] - pre_sum["sum_pmax"], 0.0, None
            )
        if gen_sum["sum_margin"] is not None and pre_sum["sum_margin"] is not None:
            decode_sum_margin = np.clip(
                gen_sum["sum_margin"] - pre_sum["sum_margin"], 0.0, None
            )

        # -----------------------
        # Compute metrics (no plotting here)
        # -----------------------
        pre_metrics = compute_metrics_from_counts(
            counts_top1=pre_sum["counts_top1"],
            total_events=pre_sum["total_events"],
            sum_probs=pre_sum["sum_probs"],
            sum_pmax=pre_sum["sum_pmax"],
            sum_margin=pre_sum["sum_margin"],
        )

        dec_metrics = compute_metrics_from_counts(
            counts_top1=decode_counts,
            total_events=decode_total,
            sum_probs=decode_sum_probs,
            sum_pmax=decode_sum_pmax,
            sum_margin=decode_sum_margin,
        )

        # -----------------------
        # Dashboards (2 plots per prompt)
        # -----------------------
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

        # global comparison storage
        results[key] = {
            "eff_prefill": pre_metrics["eff_experts"],
            "eff_decode": dec_metrics["eff_experts"],
        }

        # Optional sanity print
        print(
            f"[{key}] totals per layer (min/max) "
            f"prefill={pre_sum['total_events'].min()}/{pre_sum['total_events'].max()} "
            f"generate={gen_sum['total_events'].min()}/{gen_sum['total_events'].max()} "
            f"decode={decode_total.min()}/{decode_total.max()}"
        )

    # -----------------------
    # Optional: 2 global comparison plots across prompts
    # -----------------------
    plot_eff_experts_across_prompts(
        results, out_dir=out_dir, suffix=suffix, tag="prefill"
    )
    plot_eff_experts_across_prompts(
        results, out_dir=out_dir, suffix=suffix, tag="decode"
    )

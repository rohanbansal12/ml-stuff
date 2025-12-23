from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import math


def _safe_entropy_from_dist(p: np.ndarray, eps: float = 1e-12) -> float:
    """Computes the Shannon entropy of a discrete probability distribution.

    Args:
        p: A numpy array representing a discrete distribution (assumed nonnegative).
            Will be normalized to sum to 1.
        eps: Small epsilon value to prevent log(0). Defaults to 1e-12.

    Returns:
        The Shannon entropy H(p) = -sum(p * log(p)). Returns 0.0 if the distribution
        has no mass.
    """
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
    counts_top1: np.ndarray,
    total_events: np.ndarray,
    sum_probs: Optional[np.ndarray] = None,
    sum_pmax: Optional[np.ndarray] = None,
    sum_margin: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Converts additive routing statistics into per-layer distributions and metrics.

    This function performs pure computation without any I/O or plotting.

    Args:
        counts_top1: Array of shape [L, E] containing top-1 expert selection counts
            per layer, where L is the number of layers and E is the number of experts.
        total_events: Array of shape [L] containing total routing decisions per layer.
        sum_probs: Optional array of shape [L, E] containing sum of softmax probabilities
            over all tokens. Defaults to None.
        sum_pmax: Optional array of shape [L] containing sum of max probabilities over
            all tokens per layer. Defaults to None.
        sum_margin: Optional array of shape [L] containing sum of (p1-p2) margins over
            all tokens per layer. Defaults to None.

    Returns:
        A dictionary containing the following metrics:
            - heatmap_load: [L, E] array of top-1 selection frequencies per layer
            - heatmap_importance: [L, E] array of mean softmax prob mass per layer (or None)
            - top1_share: [L] array of max load per layer
            - hhi: [L] array of Herfindahl-Hirschman Index (sum of squared loads) per layer
            - entropy_load: [L] array of entropy of load distribution per layer
            - eff_experts: [L] array of effective experts (exp(entropy)) per layer
            - mean_pmax: [L] array of mean max probability per layer (or None)
            - mean_margin: [L] array of mean margin (p1-p2) per layer (or None)
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


def top_expert_per_layer_from_counts(counts_top1: np.ndarray) -> Dict[int, int]:
    """Identifies the most frequently selected expert for each layer.

    Args:
        counts_top1: Array of shape [L, E] containing top-1 expert selection counts
            per layer, where L is the number of layers and E is the number of experts.

    Returns:
        A dictionary mapping layer indices to their most frequently selected expert index.
        Only includes layers where at least one expert was selected (total counts > 0).
    """
    L, E = counts_top1.shape
    out = {}
    for l in range(L):
        if counts_top1[l].sum() > 0:
            out[l] = int(counts_top1[l].argmax())
    return out


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
) -> None:
    """Generates and saves side-by-side heatmaps of prefill vs decode expert load.

    Creates a figure with two heatmaps showing expert selection frequencies during
    the prefill (prompt processing) and decode (token generation) phases.

    Args:
        heatmap_prefill: Array of shape [L, E] containing prefill phase expert loads.
        heatmap_decode: Array of shape [L, E] containing decode phase expert loads.
        out_dir: Directory path where the plot will be saved.
        prompt_key: Identifier for the prompt (used in title and filename).
        suffix: Optional suffix to append to the title. Defaults to "".
        xtick_every: Show x-axis tick labels every N experts. Defaults to 5.
        ytick_every: Show y-axis tick labels every N layers. Defaults to 2.
        filename_prefix: Prefix for the output filename. Defaults to "dashboard_heatmaps".
    """
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
    fig.suptitle(
        f"Prefill vs Decode Routing: '{prompt_key}'{title_suffix}", fontsize=16
    )

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
) -> None:
    """Generates and saves comparison curves for prefill vs decode phases.

    Creates a two-row figure showing:
        - Row 1: Effective experts (diversity) comparison across layers
        - Row 2: Router confidence metrics (pmax and margin) if available

    Args:
        pre: Dictionary of prefill metrics from compute_metrics_from_counts.
        dec: Dictionary of decode metrics from compute_metrics_from_counts.
        out_dir: Directory path where the plot will be saved.
        prompt_key: Identifier for the prompt (used in title and filename).
        suffix: Optional suffix to append to the title. Defaults to "".
        filename_prefix: Prefix for the output filename. Defaults to "dashboard_curves".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eff_prefill = pre["eff_experts"]
    eff_decode = dec["eff_experts"]
    L = len(eff_prefill)
    layers = np.arange(L)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, constrained_layout=True
    )

    # Row 1: diversity
    axes[0].plot(layers, eff_prefill, label="prefill")
    axes[0].plot(layers, eff_decode, label="decode")
    axes[0].set_ylabel("Effective experts = exp(entropy(load))")
    axes[0].set_title("Routing diversity")
    axes[0].legend()

    # Row 2: confidence (optional)
    has_conf_pre = (pre.get("mean_pmax") is not None) and (
        pre.get("mean_margin") is not None
    )
    has_conf_dec = (dec.get("mean_pmax") is not None) and (
        dec.get("mean_margin") is not None
    )

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
            0.02,
            0.5,
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
    tag: str = "prefill",
    filename_prefix: str = "compare_effective_experts",
) -> None:
    """Generates and saves a comparison plot of effective experts across multiple prompts.

    Creates a single plot showing effective expert curves for different prompts,
    allowing comparison of routing diversity patterns across prompt types.

    Args:
        results: Dictionary mapping prompt keys to their metrics. Each value should
            contain 'eff_prefill' and 'eff_decode' keys with arrays of shape [L].
        out_dir: Directory path where the plot will be saved.
        suffix: Optional suffix to append to the title. Defaults to "".
        tag: Which phase to plot - either "prefill" or "decode". Defaults to "prefill".
        filename_prefix: Prefix for the output filename. Defaults to "compare_effective_experts".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)

    for key, r in results.items():
        y = r["eff_prefill"] if tag == "prefill" else r["eff_decode"]
        ax.plot(np.arange(len(y)), y, label=key)

    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Effective experts")
    ax.set_title(
        f"Effective experts across prompts ({tag})" + (f" ({suffix})" if suffix else "")
    )
    ax.legend(ncol=3)

    fig.savefig(out_dir / f"{filename_prefix}_{tag}.jpeg", dpi=150)
    plt.close(fig)


def plot_ablation_dashboard(
    *,
    heatmap_base: np.ndarray,
    heatmap_ablated: np.ndarray,
    out_dir: Path,
    title: str,
    filename: str,
    xtick_every: int = 5,
    ytick_every: int = 2,
) -> None:
    """Generates and saves a three-panel ablation comparison dashboard.

    Creates a figure with three heatmaps side-by-side showing baseline expert load,
    ablated expert load, and the delta (reallocation) between them.

    Args:
        heatmap_base: Array of shape [L, E] containing baseline expert loads.
        heatmap_ablated: Array of shape [L, E] containing ablated expert loads.
        out_dir: Directory path where the plot will be saved.
        title: Main title for the figure.
        filename: Name of the output file (including extension).
        xtick_every: Show x-axis tick labels every N experts. Defaults to 5.
        ytick_every: Show y-axis tick labels every N layers. Defaults to 2.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delta = heatmap_ablated - heatmap_base

    vmax = float(max(heatmap_base.max(), heatmap_ablated.max()))
    vmin = 0.0

    d = float(np.max(np.abs(delta))) if delta.size else 0.0
    dv = max(d, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(28, 8), constrained_layout=True)

    sns.heatmap(
        heatmap_base,
        ax=axes[0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels=xtick_every,
        yticklabels=ytick_every,
        cbar=False,
    )
    axes[0].set_title("Baseline load")
    axes[0].set_xlabel("Expert ID")
    axes[0].set_ylabel("Layer depth")

    sns.heatmap(
        heatmap_ablated,
        ax=axes[1],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels=xtick_every,
        yticklabels=ytick_every,
        cbar=False,
    )
    axes[1].set_title("Ablated load")
    axes[1].set_xlabel("Expert ID")
    axes[1].set_ylabel("")

    sns.heatmap(
        delta,
        ax=axes[2],
        cmap="coolwarm",
        vmin=-dv,
        vmax=dv,
        xticklabels=xtick_every,
        yticklabels=ytick_every,
        cbar=True,
        cbar_kws={"label": "Δ load (ablated − baseline)"},
    )
    axes[2].set_title("Reallocation (delta)")
    axes[2].set_xlabel("Expert ID")
    axes[2].set_ylabel("")

    fig.suptitle(title, fontsize=16)
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_alpha_sweep_curves(
    *,
    sweep_results: Dict[float, Dict[str, Any]],
    out_dir: Path,
    prompt_key: str,
    suffix: str = "",
    tag: str = "decode",
) -> None:
    """Generates and saves routing diversity curves across temperature scaling values.

    Creates a three-panel figure showing how routing metrics change across different
    alpha (temperature scaling) values for a single prompt. Includes effective experts,
    top-1 share, and entropy curves.

    Args:
        sweep_results: Dictionary mapping alpha values to their metrics. Each value
            should be a dict with "prefill_metrics" and "decode_metrics" keys containing
            metric dictionaries from compute_metrics_from_counts.
        out_dir: Directory path where the plot will be saved.
        prompt_key: Identifier for the prompt (used in title and filename).
        suffix: Optional suffix to append to the title. Defaults to "".
        tag: Which phase to plot - either "prefill" or "decode". Defaults to "decode".
    """
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = sorted(sweep_results.keys())
    for a in alphas:
        m = sweep_results[a][f"{tag}_metrics"]
        print(
            f"[{prompt_key}][{tag}] α={a:g} "
            f"mean_eff={float(np.mean(m['eff_experts'])):.4f} "
            f"mean_top1={float(np.mean(m['top1_share'])):.4f} "
            f"mean_ent={float(np.mean(m['entropy_load'])):.4f} "
            f"mean_pmax={(float(np.mean(m['mean_pmax'])) if m.get('mean_pmax') is not None else -1):.4f} "
            f"mean_margin={(float(np.mean(m['mean_margin'])) if m.get('mean_margin') is not None else -1):.4f}"
        )

    # assume all alphas have same number of layers
    first_alpha = alphas[0]
    metrics0 = sweep_results[first_alpha][f"{tag}_metrics"]
    L = len(metrics0["eff_experts"])
    layers = np.arange(L)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True, constrained_layout=True
    )

    # -----------------------
    # Effective experts
    # -----------------------
    for alpha in alphas:
        m = sweep_results[alpha][f"{tag}_metrics"]
        axes[0].plot(
            layers,
            m["eff_experts"],
            label=f"α={alpha:g}",
        )

    axes[0].set_ylabel("Effective experts = exp(entropy)")
    axes[0].set_title("Routing diversity (decode)")
    axes[0].legend(ncol=3)

    # -----------------------
    # Top-1 share (collapse proxy)
    # -----------------------
    for alpha in alphas:
        m = sweep_results[alpha][f"{tag}_metrics"]
        axes[1].plot(
            layers,
            m["top1_share"],
            label=f"α={alpha:g}",
        )

    axes[1].set_ylabel("Top-1 expert share")
    axes[1].set_title("Load concentration")

    # -----------------------
    # Entropy
    # -----------------------
    for alpha in alphas:
        m = sweep_results[alpha][f"{tag}_metrics"]
        axes[2].plot(
            layers,
            m["entropy_load"],
            label=f"α={alpha:g}",
        )

    axes[2].set_ylabel("Entropy(load)")
    axes[2].set_xlabel("Layer depth")
    axes[2].set_title("Routing entropy")

    title_suffix = f" ({suffix})" if suffix else ""
    fig.suptitle(
        f"Router temperature sweep ({tag}) — '{prompt_key}'{title_suffix}",
        fontsize=16,
    )

    fig.savefig(
        out_dir / f"alpha_sweep_curves_{tag}_{prompt_key}.jpeg",
        dpi=150,
    )
    plt.close(fig)


def plot_alpha_sweep_scalar_summary(
    *,
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    out_dir: Path,
    suffix: str = "",
    tag: str = "decode",
) -> None:
    """Generates and saves a summary plot of routing collapse across prompts and alphas.

    Creates a single plot showing mean effective experts versus alpha (temperature scaling)
    for multiple prompts. This allows comparison of how different prompts respond to
    temperature scaling and identification of routing collapse trends.

    Args:
        all_results: Nested dictionary structure mapping prompt keys to alpha sweep results.
            Structure: {prompt_key: {alpha: {"prefill_metrics": {...}, "decode_metrics": {...}}}}.
            Each metrics dict should come from compute_metrics_from_counts.
        out_dir: Directory path where the plot will be saved.
        suffix: Optional suffix to append to the title. Defaults to "".
        tag: Which phase to plot - either "prefill" or "decode". Defaults to "decode".

    Note:
        The x-axis will use logarithmic scale if the ratio of max/min alpha is >= 4,
        otherwise it uses linear scale.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for prompt_key, sweep in all_results.items():
        alphas = sorted(sweep.keys())
        means = []

        for alpha in alphas:
            m = sweep[alpha][f"{tag}_metrics"]
            eff = m["eff_experts"]
            means.append(float(np.mean(eff)))

        ax.plot(
            alphas,
            means,
            marker="o",
            label=prompt_key,
        )

    ax.set_xscale("log" if max(alphas) / min(alphas) >= 4 else "linear")
    ax.set_xlabel("Router logit scale α")
    ax.set_ylabel("Mean effective experts (decode)")
    ax.set_title("Decode routing collapse under temperature scaling")
    ax.legend(ncol=2)

    title_suffix = f" ({suffix})" if suffix else ""
    fig.suptitle(
        f"α-sweep summary{title_suffix}",
        fontsize=14,
    )

    fig.savefig(
        out_dir / f"alpha_sweep_summary_{tag}.jpeg",
        dpi=150,
    )
    plt.close(fig)

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Callable, Optional
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from data import load_multi_source_data

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(
    model_name: str,
    device: torch.device,
) -> tuple:
    """Loads a 4-bit quantized model

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen1.5-MoE-A2.7B").
        device: PyTorch device to load the model on (e.g., torch.device("cuda")).

    Returns:
        - model: The callable model used for forward passes and generation.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,  # recommended for Qwen MoE
    ).to(device)

    model.eval()

    return model


def get_router_hook(layer_idx: int, store: Dict[int, List[torch.Tensor]]) -> Callable:
    """Creates a forward hook to capture routing logits for a specific layer.

    Args:
        layer_idx: The index of the MoE layer to monitor.
        store: Dictionary to store captured routing tensors, mapping layer indices
            to lists of routing logit tensors.

    Returns:
        A hook function that can be registered with PyTorch's register_forward_hook.
        The hook extracts routing logits and stores them in the provided store dict.
    """

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            data = output[0].detach().cpu()
        else:
            data = output.detach().cpu()
        store.setdefault(layer_idx, []).append(torch.softmax(data.float(), dim=-1))
        return output

    return hook


def attach_router_hooks(layer_att, routing_store: Dict[int, List[torch.Tensor]]) -> int:
    """Attaches forward hooks to capture routing logits across all MoE layers.

    Iterates through all layers in the model and registers forward hooks on the router
    (gate) modules. The hooks capture routing logits during forward passes and store
    them in the provided routing_store dictionary.

    Args:
        layer_att: The model's layer attribute object containing MoE layers. For base
            models, this is the model itself. For PEFT models, this is model.base_model.model.
        routing_store: Dictionary to store captured routing tensors, mapping layer indices
            to lists of routing logit tensors.

    Returns:
        The total number of layers in the model.

    Note:
        This function assumes the router module is accessible via `layer.mlp.gate`.
        Only layers that have this attribute structure will have hooks attached.
    """
    num_layers = len(layer_att.model.layers)

    for i, layer in enumerate(layer_att.model.layers):
        # Match your current known router path
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            layer.mlp.gate.register_forward_hook(get_router_hook(i, routing_store))

    return num_layers


def compute_baseline_stats(logits: torch.Tensor) -> dict:
    """
    Args:
        logits: (num_layers, total_tokens, num_experts)
    """
    num_layers, total_tokens, num_experts = logits.shape

    # Expert selection (assuming top-1 for now—verify Qwen's top-k)
    expert_ids = logits.argmax(dim=-1)  # (num_layers, total_tokens)

    # 1. Utilization counts
    counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
    for layer in range(num_layers):
        counts[layer] = torch.bincount(expert_ids[layer], minlength=num_experts)

    util = counts.float() / total_tokens
    expected_util = 1.0 / num_experts

    # 2. Imbalance metrics
    coef_of_variation = util.std(dim=1) / util.mean(dim=1)  # per layer
    max_min_ratio = util.max(dim=1).values / (util.min(dim=1).values + 1e-8)

    # 3. Dead expert detection
    dead_threshold = expected_util * 0.1  # <10% of expected
    dead_experts = (util < dead_threshold).sum(dim=1)  # count per layer

    # 4. Router entropy (on softmax probs)
    entropy = -torch.xlogy(logits, logits).sum(dim=-1)  # (layers, tokens)
    max_entropy = np.log(num_experts)
    normalized_entropy = entropy / max_entropy

    return {
        "counts": counts,
        "utilization": util,
        "coef_of_variation": coef_of_variation,
        "max_min_ratio": max_min_ratio,
        "dead_experts_per_layer": dead_experts,
        "entropy_mean": normalized_entropy.mean(dim=1),
        "entropy_std": normalized_entropy.std(dim=1),
        "entropy_per_token": normalized_entropy,  # for later analysis
    }


def save_baseline_routing_plots(
    stats: dict,
    out_dir: str | Path,
    num_experts: int,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    util = stats["utilization"].numpy()
    entropy_mean = stats["entropy_mean"].numpy()
    entropy_std = stats["entropy_std"].numpy()
    entropy_per_token = stats["entropy_per_token"].numpy()
    cov = stats["coef_of_variation"].numpy()

    num_layers = util.shape[0]
    expected_util = 1.0 / num_experts

    # ============================
    # 1) UTILIZATION HEATMAP (keep, but improve)
    # ============================
    fig, ax = plt.subplots(figsize=(14, 6))

    # Diverging colormap centered on expected utilization
    max_dev = max(util.max() - expected_util, expected_util - util.min())

    sns.heatmap(
        util,
        cmap="RdBu_r",
        center=expected_util,
        vmin=expected_util - max_dev,
        vmax=expected_util + max_dev,
        cbar_kws={"label": "Utilization (red=over, blue=under)"},
        ax=ax,
    )
    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")
    ax.set_title(f"Expert Utilization (expected={expected_util:.4f})")
    plt.tight_layout()
    plt.savefig(out_dir / "utilization_heatmap.png", dpi=150)
    plt.close()

    # ============================
    # 2) PER-LAYER UTILIZATION DISTRIBUTIONS (box/violin)
    # ============================
    fig, ax = plt.subplots(figsize=(12, 5))

    # Reshape for seaborn: each row is (layer, expert, util)
    layer_labels = np.repeat(np.arange(num_layers), num_experts)
    util_flat = util.flatten()

    sns.boxplot(x=layer_labels, y=util_flat, ax=ax, color="steelblue")
    ax.axhline(
        expected_util,
        color="red",
        linestyle="--",
        label=f"Expected ({expected_util:.4f})",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert Utilization")
    ax.set_title("Distribution of Expert Utilization per Layer")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "utilization_boxplot_by_layer.png", dpi=150)
    plt.close()

    # ============================
    # 3) COEFFICIENT OF VARIATION BY LAYER
    # ============================
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(np.arange(num_layers), cov, color="teal", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Utilization Imbalance by Layer (higher = more imbalanced)")
    ax.set_xticks(np.arange(0, num_layers, max(1, num_layers // 10)))
    plt.tight_layout()
    plt.savefig(out_dir / "utilization_cov_by_layer.png", dpi=150)
    plt.close()

    # ============================
    # 4) DEAD/OVERLOADED EXPERT IDENTIFICATION
    # ============================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dead_threshold = expected_util * 0.1
    overload_threshold = expected_util * 2.0

    dead_mask = util < dead_threshold
    overload_mask = util > overload_threshold

    # Dead experts heatmap
    sns.heatmap(dead_mask.astype(float), cmap="Reds", cbar=False, ax=axes[0])
    axes[0].set_xlabel("Expert")
    axes[0].set_ylabel("Layer")
    axes[0].set_title(f"Dead Experts (<{dead_threshold:.4f} util)")

    # Overloaded experts heatmap
    sns.heatmap(overload_mask.astype(float), cmap="Reds", cbar=False, ax=axes[1])
    axes[1].set_xlabel("Expert")
    axes[1].set_ylabel("Layer")
    axes[1].set_title(f"Overloaded Experts (>{overload_threshold:.4f} util)")

    plt.tight_layout()
    plt.savefig(out_dir / "dead_overloaded_experts.png", dpi=150)
    plt.close()

    # ============================
    # 5) ROUTER ENTROPY BY LAYER
    # ============================
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.errorbar(
        np.arange(num_layers),
        entropy_mean,
        yerr=entropy_std,
        fmt="o-",
        capsize=3,
        color="purple",
        alpha=0.8,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Entropy")
    ax.set_ylim(0, 1)
    ax.set_title("Router Entropy by Layer (1.0 = uniform, 0.0 = deterministic)")
    ax.set_xticks(np.arange(0, num_layers, max(1, num_layers // 10)))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "router_entropy_by_layer.png", dpi=150)
    plt.close()

    # ============================
    # 6) ENTROPY DISTRIBUTION (histogram)
    # ============================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    # Sample 6 layers spread across the model
    sample_layers = np.linspace(0, num_layers - 1, 6, dtype=int)

    for ax, layer_idx in zip(axes, sample_layers):
        layer_entropy = entropy_per_token[layer_idx]
        ax.hist(layer_entropy, bins=50, density=True, alpha=0.7, color="purple")
        ax.axvline(
            layer_entropy.mean(),
            color="red",
            linestyle="--",
            label=f"mean={layer_entropy.mean():.3f}",
        )
        ax.set_xlabel("Normalized Entropy")
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    plt.suptitle("Router Entropy Distribution (sampled layers)")
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_histograms.png", dpi=150)
    plt.close()

    # ============================
    # 7) SUMMARY STATISTICS TABLE (save as text/csv)
    # ============================
    summary = {
        "layer": list(range(num_layers)),
        "mean_util": util.mean(axis=1).tolist(),
        "std_util": util.std(axis=1).tolist(),
        "min_util": util.min(axis=1).tolist(),
        "max_util": util.max(axis=1).tolist(),
        "coef_variation": cov.tolist(),
        "dead_expert_count": dead_mask.sum(axis=1).tolist(),
        "overload_expert_count": overload_mask.sum(axis=1).tolist(),
        "entropy_mean": entropy_mean.tolist(),
        "entropy_std": entropy_std.tolist(),
    }

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "layer_summary_stats.csv", index=False)

    # Also save a quick text summary
    with open(out_dir / "summary.txt", "w") as f:
        f.write("=== Baseline Routing Statistics ===\n\n")
        f.write(f"Total layers: {num_layers}\n")
        f.write(f"Experts per layer: {num_experts}\n")
        f.write(f"Expected utilization: {expected_util:.4f}\n\n")

        f.write(f"Dead experts (total across layers): {dead_mask.sum()}\n")
        f.write(f"Overloaded experts (total across layers): {overload_mask.sum()}\n\n")

        f.write(f"Mean CoV across layers: {cov.mean():.4f} (std={cov.std():.4f})\n")
        f.write(
            f"Mean entropy across layers: {entropy_mean.mean():.4f} (std={entropy_mean.std():.4f})\n"
        )

    print(f"Saved plots and stats to {out_dir}")


def compute_routing_weights_stats(
    probs: torch.Tensor,
    top_k: int = 2,
    norm_topk_prob: bool = True,
) -> dict:
    """
    Compute routing weight statistics from full router probabilities.

    Args:
        probs: (num_layers, total_tokens, num_experts) post-softmax probabilities
        top_k: number of experts selected per token
        norm_topk_prob: whether to renormalize top-k weights (matches Qwen behavior)
    """
    num_layers, total_tokens, num_experts = probs.shape

    # Derive top-k selection (mimicking router behavior)
    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)  # (layers, tokens, k)

    # Renormalize if model does (Qwen does by default)
    if norm_topk_prob:
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    else:
        topk_weights = topk_probs

    primary_weight = topk_weights[:, :, 0]
    secondary_weight = topk_weights[:, :, 1]

    # Weight metrics
    weight_diff = primary_weight - secondary_weight
    secondary_clamped = torch.clamp(secondary_weight, min=1e-6)
    weight_ratio = torch.clamp(primary_weight / secondary_clamped, max=100.0)

    # Selected entropy (between chosen experts)
    selected_entropy = -(topk_weights * torch.log(topk_weights + 1e-8)).sum(dim=-1)
    normalized_selected_entropy = selected_entropy / np.log(top_k)

    # Full entropy (across all experts, pre-selection)
    full_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    normalized_full_entropy = full_entropy / np.log(num_experts)

    return {
        # Per-token
        "primary_weight": primary_weight,
        "secondary_weight": secondary_weight,
        "weight_diff": weight_diff,
        "weight_ratio": weight_ratio,
        "topk_weights": topk_weights,
        "topk_indices": topk_indices,
        "selected_entropy": normalized_selected_entropy,
        "full_entropy": normalized_full_entropy,
        # Per-layer
        "primary_weight_mean": primary_weight.mean(dim=1),
        "primary_weight_std": primary_weight.std(dim=1),
        "primary_weight_median": primary_weight.median(dim=1).values,
        "weight_diff_mean": weight_diff.mean(dim=1),
        "weight_diff_std": weight_diff.std(dim=1),
        "weight_ratio_mean": weight_ratio.mean(dim=1),
        "weight_ratio_median": weight_ratio.median(dim=1).values,
        "selected_entropy_mean": normalized_selected_entropy.mean(dim=1),
        "selected_entropy_std": normalized_selected_entropy.std(dim=1),
        "full_entropy_mean": normalized_full_entropy.mean(dim=1),
        "full_entropy_std": normalized_full_entropy.std(dim=1),
        # Metadata
        "top_k": top_k,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "total_tokens": total_tokens,
    }


def save_routing_weight_plots(
    stats: dict,
    out_dir: str | Path,
    category: Optional[str] = None,
):
    """
    Generate plots for top-2 routing weight distribution analysis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    title_suffix = f" ({category})" if category else ""

    num_layers = stats["num_layers"]

    # Extract numpy arrays
    primary_weight = stats["primary_weight"].numpy()
    secondary_weight = stats["secondary_weight"].numpy()
    weight_diff = stats["weight_diff"].numpy()
    weight_ratio = stats["weight_ratio"].numpy()
    selected_entropy = stats["selected_entropy"].numpy()

    weight_ratio = np.clip(
        np.nan_to_num(weight_ratio, nan=1.0, posinf=100.0, neginf=1.0), 1.0, 100.0
    )
    primary_weight = np.clip(np.nan_to_num(primary_weight, nan=0.5), 0.5, 1.0)
    weight_diff = np.clip(np.nan_to_num(weight_diff, nan=0.0), 0.0, 1.0)
    selected_entropy = np.clip(np.nan_to_num(selected_entropy, nan=0.5), 0.0, 1.0)

    primary_mean = stats["primary_weight_mean"].numpy()
    primary_std = stats["primary_weight_std"].numpy()
    diff_mean = stats["weight_diff_mean"].numpy()
    diff_std = stats["weight_diff_std"].numpy()
    entropy_mean = stats["selected_entropy_mean"].numpy()
    entropy_std = stats["selected_entropy_std"].numpy()

    # ============================
    # 1) PRIMARY VS SECONDARY WEIGHT BY LAYER
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    secondary_mean = stats["secondary_weight"].mean(dim=1).numpy()

    ax.plot(
        np.arange(num_layers),
        primary_mean,
        "o-",
        label="Primary expert",
        color="steelblue",
    )
    ax.plot(
        np.arange(num_layers),
        secondary_mean,
        "s-",
        label="Secondary expert",
        color="coral",
    )
    ax.axhline(0.5, color="gray", linestyle="--", label="Equal (0.5)")

    ax.fill_between(
        np.arange(num_layers),
        primary_mean - primary_std,
        primary_mean + primary_std,
        alpha=0.2,
        color="steelblue",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Weight")
    ax.set_ylim(0, 1)
    ax.set_title(f"Expert Weight Distribution{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "primary_secondary_weight.png", dpi=150)
    plt.close()

    # ============================
    # 2) WEIGHT DIFFERENCE BY LAYER
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        np.arange(num_layers),
        diff_mean,
        yerr=diff_std,
        fmt="o-",
        capsize=3,
        color="teal",
        alpha=0.8,
    )
    ax.axhline(0, color="gray", linestyle="--", label="Equal weights")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Weight Difference (primary - secondary)")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title(
        f"Router Decisiveness by Layer{title_suffix}\n(0=equal, 1=primary only)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "weight_difference_by_layer.png", dpi=150)
    plt.close()

    # ============================
    # 3) PRIMARY WEIGHT DISTRIBUTION (histograms)
    # ============================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    sample_layers = np.linspace(0, num_layers - 1, 6, dtype=int)

    for ax, layer_idx in zip(axes, sample_layers):
        layer_weights = primary_weight[layer_idx]

        ax.hist(
            layer_weights,
            bins=50,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(
            layer_weights.mean(),
            color="red",
            linestyle="--",
            label=f"mean={layer_weights.mean():.3f}",
        )
        ax.axvline(0.5, color="orange", linestyle=":", label="equal=0.5")
        ax.set_xlabel("Primary Expert Weight")
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlim(0.5, 1.0)  # primary is always >= 0.5 by definition
        ax.legend(fontsize=8)

    plt.suptitle(f"Primary Expert Weight Distribution{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_dir / "primary_weight_histograms.png", dpi=150)
    plt.close()

    # ============================
    # 4) WEIGHT RATIO DISTRIBUTION
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    # Cap extreme ratios for visualization
    ratio_capped = np.clip(weight_ratio.flatten(), 1, 20)

    ax.hist(
        ratio_capped,
        bins=100,
        density=True,
        alpha=0.7,
        color="purple",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(1.0, color="red", linestyle="--", label="Equal (ratio=1)")
    ax.axvline(
        np.median(ratio_capped),
        color="orange",
        linestyle="-",
        label=f"Median={np.median(ratio_capped):.2f}",
    )
    ax.set_xlabel("Weight Ratio (primary / secondary)")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Weight Ratios{title_suffix}")
    ax.set_xlim(1, 20)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "weight_ratio_distribution.png", dpi=150)
    plt.close()

    # ============================
    # 5) SELECTED ENTROPY BY LAYER
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        np.arange(num_layers),
        entropy_mean,
        yerr=entropy_std,
        fmt="s-",
        capsize=3,
        color="purple",
        alpha=0.8,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Entropy (between selected experts)")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Weight Entropy{title_suffix}\n(1.0=equal weights, 0.0=single expert)"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "selected_entropy_by_layer.png", dpi=150)
    plt.close()

    # ============================
    # 6) 2D HISTOGRAM: Primary weight vs Layer
    # ============================
    fig, ax = plt.subplots(figsize=(12, 6))

    bins_weight = np.linspace(0.5, 1.0, 26)
    hist_matrix = np.zeros((num_layers, len(bins_weight) - 1))

    for layer_idx in range(num_layers):
        hist, _ = np.histogram(
            primary_weight[layer_idx], bins=bins_weight, density=True
        )
        hist_matrix[layer_idx] = hist

    sns.heatmap(
        hist_matrix.T,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Density"},
        xticklabels=10,
        yticklabels=[
            f"{bins_weight[i]:.2f}" for i in range(0, len(bins_weight) - 1, 5)
        ],
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Primary Expert Weight")
    ax.set_title(f"Primary Weight Distribution by Layer{title_suffix}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "primary_weight_heatmap.png", dpi=150)
    plt.close()

    # ============================
    # 7) DECISIVENESS CATEGORIES
    # ============================
    # Categorize tokens by how decisive the routing is
    fig, ax = plt.subplots(figsize=(10, 5))

    # Define thresholds
    thresholds = [
        (0.5, 0.6, "Balanced (0.5-0.6)"),
        (0.6, 0.7, "Moderate (0.6-0.7)"),
        (0.7, 0.8, "Decisive (0.7-0.8)"),
        (0.8, 0.9, "Strong (0.8-0.9)"),
        (0.9, 1.0, "Dominant (0.9-1.0)"),
    ]

    category_counts = np.zeros((num_layers, len(thresholds)))

    for layer_idx in range(num_layers):
        layer_pw = primary_weight[layer_idx]
        for i, (low, high, _) in enumerate(thresholds):
            category_counts[layer_idx, i] = (
                (layer_pw >= low) & (layer_pw < high)
            ).sum()

    # Normalize to percentages
    category_pcts = category_counts / category_counts.sum(axis=1, keepdims=True) * 100

    # Stacked area chart
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(thresholds)))

    bottom = np.zeros(num_layers)
    for i, (_, _, label) in enumerate(thresholds):
        ax.fill_between(
            np.arange(num_layers),
            bottom,
            bottom + category_pcts[:, i],
            label=label,
            color=colors[i],
            alpha=0.8,
        )
        bottom += category_pcts[:, i]

    ax.set_xlabel("Layer")
    ax.set_ylabel("Percentage of Tokens")
    ax.set_title(f"Routing Decisiveness Categories{title_suffix}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, num_layers - 1)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_dir / "decisiveness_categories.png", dpi=150)
    plt.close()

    # ============================
    # 8) SAVE SUMMARY STATS
    # ============================
    summary_df = pd.DataFrame(
        {
            "layer": np.arange(num_layers),
            "primary_weight_mean": primary_mean,
            "primary_weight_std": primary_std,
            "primary_weight_median": stats["primary_weight_median"].numpy(),
            "weight_diff_mean": diff_mean,
            "weight_diff_std": diff_std,
            "weight_ratio_mean": stats["weight_ratio_mean"].numpy(),
            "weight_ratio_median": stats["weight_ratio_median"].numpy(),
            "selected_entropy_mean": entropy_mean,
            "selected_entropy_std": entropy_std,
        }
    )
    summary_df.to_csv(out_dir / "weight_stats_summary.csv", index=False)

    # Text summary
    with open(out_dir / "weight_summary.txt", "w") as f:
        f.write(f"=== Routing Weight Statistics{title_suffix} ===\n\n")
        f.write("Top-k: 2\n")
        f.write(f"Total tokens: {stats['total_tokens']:,}\n")
        f.write(f"Num layers: {num_layers}\n")
        f.write(f"Num experts: {stats['num_experts']}\n\n")

        f.write("Primary Expert Weight:\n")
        f.write(f"  Overall mean:   {primary_weight.mean():.4f}\n")
        f.write(f"  Overall median: {np.median(primary_weight):.4f}\n")
        f.write(f"  Overall std:    {primary_weight.std():.4f}\n")
        f.write(
            f"  Layer range:    [{primary_mean.min():.4f}, {primary_mean.max():.4f}]\n\n"
        )

        f.write("Weight Difference (primary - secondary):\n")
        f.write(f"  Overall mean:   {weight_diff.mean():.4f}\n")
        f.write(f"  Layer range:    [{diff_mean.min():.4f}, {diff_mean.max():.4f}]\n\n")

        f.write("Weight Ratio (primary / secondary):\n")
        f.write(f"  Overall median: {np.median(weight_ratio):.2f}\n")
        f.write(f"  95th pctl:      {np.percentile(weight_ratio, 95):.2f}\n\n")

        # Decisiveness breakdown
        pw_flat = primary_weight.flatten()
        f.write("Decisiveness Breakdown (all tokens):\n")
        for low, high, label in thresholds:
            pct = ((pw_flat >= low) & (pw_flat < high)).mean() * 100
            f.write(f"  {label}: {pct:.1f}%\n")

    print(f"Saved routing weight plots to {out_dir}")


def analyze_routing_weights_by_category(
    category_data: Dict[str, dict],
    out_dir: str | Path,
):
    """
    Analyze and compare routing weights across categories.

    Args:
        category_data: Dict mapping category name to saved data dict.
                       Each should have 'logits' key with shape (layers, tokens, experts)
                       containing post-softmax probabilities.
        out_dir: Base output directory
    """
    out_dir = Path(out_dir)

    all_stats = {}

    # Compute stats for each category
    for category, data in category_data.items():
        probs = data["logits"]
        stats = compute_routing_weights_stats(probs, top_k=2)
        all_stats[category] = stats

        # Save per-category plots
        category_out = out_dir / "by_category" / category
        save_routing_weight_plots(stats, category_out, category=category)

    # ============================
    # CROSS-CATEGORY COMPARISON
    # ============================
    comparison_dir = out_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    categories = list(all_stats.keys())
    num_layers = all_stats[categories[0]]["num_layers"]

    # 1) Primary weight by category
    fig, ax = plt.subplots(figsize=(12, 6))

    for cat in categories:
        mean = all_stats[cat]["primary_weight_mean"].numpy()
        ax.plot(np.arange(num_layers), mean, "o-", label=cat, alpha=0.8, markersize=4)

    ax.axhline(0.5, color="gray", linestyle="--", label="Equal")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Primary Expert Weight")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Primary Expert Weight by Category")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_dir / "primary_weight_comparison.png", dpi=150)
    plt.close()

    # 2) Weight difference by category
    fig, ax = plt.subplots(figsize=(12, 6))

    for cat in categories:
        mean = all_stats[cat]["weight_diff_mean"].numpy()
        ax.plot(np.arange(num_layers), mean, "s-", label=cat, alpha=0.8, markersize=4)

    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Weight Difference")
    ax.set_title("Router Decisiveness by Category")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_dir / "weight_diff_comparison.png", dpi=150)
    plt.close()

    # 3) Overall decisiveness bar chart
    fig, ax = plt.subplots(figsize=(10, 5))

    cat_means = []
    cat_stds = []
    for cat in categories:
        pw = all_stats[cat]["primary_weight"].numpy()
        cat_means.append(pw.mean())
        cat_stds.append(pw.std())

    x = np.arange(len(categories))
    bars = ax.bar(x, cat_means, yerr=cat_stds, capsize=5, color="steelblue", alpha=0.8)
    ax.axhline(0.5, color="red", linestyle="--", label="Equal weights")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Mean Primary Expert Weight")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Overall Routing Decisiveness by Category")
    ax.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "decisiveness_by_category_bar.png", dpi=150)
    plt.close()

    # 4) Decisiveness distribution comparison (violin plot)
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_data = []
    plot_labels = []
    for cat in categories:
        pw = all_stats[cat]["primary_weight"].numpy().flatten()
        # Subsample for plotting if too large
        if len(pw) > 10000:
            pw = np.random.choice(pw, 10000, replace=False)
        plot_data.append(pw)
        plot_labels.extend([cat] * len(pw))

    plot_df = pd.DataFrame(
        {
            "category": plot_labels,
            "primary_weight": np.concatenate(plot_data),
        }
    )

    sns.violinplot(data=plot_df, x="category", y="primary_weight", ax=ax)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Category")
    ax.set_ylabel("Primary Expert Weight")
    ax.set_title("Distribution of Routing Decisiveness by Category")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(comparison_dir / "decisiveness_violin.png", dpi=150)
    plt.close()

    # 5) Save comparison CSV
    comparison_data = []
    for cat in categories:
        pw = all_stats[cat]["primary_weight"].numpy()
        wd = all_stats[cat]["weight_diff"].numpy()
        ent = all_stats[cat]["selected_entropy"].numpy()
        comparison_data.append(
            {
                "category": cat,
                "total_tokens": all_stats[cat]["total_tokens"],
                "primary_weight_mean": pw.mean(),
                "primary_weight_std": pw.std(),
                "primary_weight_median": np.median(pw),
                "weight_diff_mean": wd.mean(),
                "weight_diff_std": wd.std(),
                "selected_entropy_mean": ent.mean(),
                "pct_balanced_0.5_0.6": ((pw >= 0.5) & (pw < 0.6)).mean() * 100,
                "pct_dominant_0.9_1.0": ((pw >= 0.9) & (pw <= 1.0)).mean() * 100,
            }
        )

    pd.DataFrame(comparison_data).to_csv(
        comparison_dir / "category_comparison.csv", index=False
    )

    print(f"Saved comparison plots to {comparison_dir}")

    return all_stats


def analyze_entropy(
    probs: torch.Tensor, seq_lengths: torch.Tensor, num_experts: int, k: int = 10
) -> dict:
    output_dict = {}

    max_entropy = np.log(num_experts)

    token_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    norm_entropy = token_entropy / max_entropy
    output_dict["token_entropy"] = token_entropy
    output_dict["norm_entropy"] = norm_entropy

    # get tokens with highest entropy
    top_values, top_indices = torch.topk(norm_entropy, k)
    cum_lengths = torch.cumsum(seq_lengths, dim=0)
    seq_indices = torch.searchsorted(cum_lengths, top_indices, right=True)
    seq_starts = torch.cat(
        [torch.tensor([0], device=seq_lengths.device), cum_lengths[:-1]]
    )
    offsets = top_indices - seq_starts[seq_indices]
    output_dict["top_entropy_tokens"] = {"seq_idx": seq_indices, "offsets": offsets}

    # get tokens with highest entropy
    top_values, top_indices = torch.topk(norm_entropy, k, largest=False)
    cum_lengths = torch.cumsum(seq_lengths, dim=0)
    seq_indices = torch.searchsorted(cum_lengths, top_indices, right=True)
    seq_starts = torch.cat(
        [torch.tensor([0], device=seq_lengths.device), cum_lengths[:-1]]
    )
    offsets = top_indices - seq_starts[seq_indices]
    output_dict["bottom_entropy_tokens"] = {"seq_idx": seq_indices, "offsets": offsets}

    layer_entropy_mean = torch.mean(norm_entropy, dim=-1)
    layer_entropy_std = torch.std(norm_entropy, dim=-1)
    output_dict["layer_entropy_mean"] = layer_entropy_mean
    output_dict["layer_entropy_std"] = layer_entropy_std

    return output_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/workspace/ml-stuff/MoE/out")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--num_experts", type=int, default=60)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    # load model
    model = build_model(MODEL_NAME, device)

    # load data
    dataloader, dataset, tokenizer = load_multi_source_data(
        samples_per_category=args.samples,
        max_length=512,
        batch_size=1,
        seed=42,
        num_workers=8,
    )
    unique_categories = list(set(dataset.categories))
    print(f"{unique_categories} Categories")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ### Run forward pass on all data and save router logits

    # Attach router observation hooks (same for all modes)
    routing_store = {}
    num_layers = attach_router_hooks(model, routing_store)

    logit_out_dir = out_dir / "logits"
    logit_out_dir.mkdir(exist_ok=True)

    all_results = {}
    outer = tqdm(unique_categories, desc="categories", position=0)
    for category in outer:
        if (logit_out_dir / f"{category}.pt").exists():
            continue

        outer.set_postfix(category=category)
        indices = dataset.get_indices_by_category(category)
        token_counts = []
        routing_store.clear()

        for idx in tqdm(indices, desc=f"  {category}", position=1, leave=False):
            data = dataset[idx]
            input_ids = data["input_ids"].to(device)
            _ = model(input_ids=input_ids)
            token_counts.append(input_ids.numel())

        token_counts = torch.tensor(token_counts, dtype=torch.long)
        final_output = torch.zeros(
            num_layers, token_counts.sum().item(), args.num_experts
        )
        for layer_idx, data_list in routing_store.items():
            layer_logits = torch.cat(data_list, dim=0)
            final_output[layer_idx] = layer_logits

        all_results[category] = {
            "logits": final_output,
            "token_counts": token_counts,
            "num_sequences": len(indices),
        }

        torch.save(all_results[category], logit_out_dir / f"{category}.pt")

    ### load category-wise outputs and compute stats
    logit_dict = {}
    global_logits = []
    sequence_lengths = []
    for category in unique_categories:
        category_dict = torch.load(logit_out_dir / f"{category}.pt")
        logit_dict[category] = category_dict
        global_logits.append(category_dict["logits"])
        sequence_lengths.append(category_dict["token_counts"])

    global_logits = torch.cat(global_logits, dim=1)
    sequence_lengths = torch.cat(sequence_lengths, dim=0)

    stats = compute_baseline_stats(global_logits)
    save_baseline_routing_plots(stats, out_dir / "global", num_experts=args.num_experts)

    stats = compute_routing_weights_stats(global_logits, top_k=2)
    save_routing_weight_plots(stats, out_dir / "routing")

    analyze_routing_weights_by_category(logit_dict, out_dir / "routing_category")

    stats = analyze_entropy(
        global_logits, sequence_lengths, num_experts=args.num_experts, k=20
    )


if __name__ == "__main__":
    main()

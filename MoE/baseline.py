import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from typing import Dict, List, Callable, Optional, Set
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats as scipy_stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from MoE.data import DEFAULT_SOURCES, load_multi_source_data


MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model_and_tokenizer(
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


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
    probs: torch.Tensor,
    seq_lengths: torch.Tensor,
    num_experts: int,
    k: int = 10,
) -> dict:
    """
    Analyze router entropy patterns.

    Args:
        probs: (num_layers, total_tokens, num_experts) - routing probabilities
        seq_lengths: (num_sequences,) - length of each sequence
        num_experts: number of experts
        k: number of top/bottom tokens to identify

    Returns:
        Dictionary with entropy statistics and extreme token locations
    """
    num_layers, total_tokens, _ = probs.shape

    output_dict = {}

    # Compute entropy
    max_entropy = np.log(num_experts)
    token_entropy = -torch.xlogy(probs, probs).sum(dim=-1)  # (num_layers, total_tokens)
    norm_entropy = token_entropy / max_entropy

    output_dict["token_entropy"] = token_entropy
    output_dict["norm_entropy"] = norm_entropy

    # Per-layer statistics
    output_dict["layer_entropy_mean"] = norm_entropy.mean(dim=1)  # (num_layers,)
    output_dict["layer_entropy_std"] = norm_entropy.std(dim=1)

    # For finding extreme tokens, average entropy across layers first
    # (or you could analyze per-layer, but global is simpler)
    mean_entropy_per_token = norm_entropy.mean(dim=0)  # (total_tokens,)

    # Build sequence mapping
    cum_lengths = torch.cumsum(seq_lengths, dim=0)
    seq_starts = torch.cat(
        [torch.tensor([0], device=seq_lengths.device), cum_lengths[:-1]]
    )

    def get_token_locations(indices: torch.Tensor) -> dict:
        """Map flat token indices to (sequence_idx, position_in_sequence)."""
        seq_indices = torch.searchsorted(cum_lengths, indices, right=True)
        # Clamp to valid range
        seq_indices = torch.clamp(seq_indices, 0, len(seq_lengths) - 1)
        offsets = indices - seq_starts[seq_indices]
        return {"seq_idx": seq_indices, "offset": offsets}

    # Highest entropy tokens (router most uncertain)
    top_values, top_indices = torch.topk(mean_entropy_per_token, k, largest=True)
    output_dict["high_entropy_tokens"] = {
        **get_token_locations(top_indices),
        "entropy_values": top_values,
        "token_indices": top_indices,
    }

    # Lowest entropy tokens (router most confident)
    bottom_values, bottom_indices = torch.topk(mean_entropy_per_token, k, largest=False)
    output_dict["low_entropy_tokens"] = {
        **get_token_locations(bottom_indices),
        "entropy_values": bottom_values,
        "token_indices": bottom_indices,
    }

    # Additional: entropy distribution statistics
    output_dict["global_entropy_mean"] = norm_entropy.mean().item()
    output_dict["global_entropy_std"] = norm_entropy.std().item()
    output_dict["global_entropy_median"] = norm_entropy.median().item()

    # Percentiles for understanding distribution shape
    flat_entropy = norm_entropy.flatten()
    output_dict["entropy_percentiles"] = {
        "p5": torch.quantile(flat_entropy, 0.05).item(),
        "p25": torch.quantile(flat_entropy, 0.25).item(),
        "p50": torch.quantile(flat_entropy, 0.50).item(),
        "p75": torch.quantile(flat_entropy, 0.75).item(),
        "p95": torch.quantile(flat_entropy, 0.95).item(),
    }

    return output_dict


def save_entropy_analysis_plots(
    entropy_stats: dict,
    dataset,
    tokenizer,
    out_dir: str | Path,
    category: Optional[str] = None,
    num_extreme_tokens: int = 50,
):
    """
    Generate plots and reports for router entropy analysis (Stage 2.4).

    Args:
        entropy_stats: Output from analyze_entropy()
        dataset: Dataset object with __getitem__ returning {"input_ids": tensor}
        tokenizer: Tokenizer for decoding tokens
        out_dir: Output directory
        category: Optional category name for titles
        num_extreme_tokens: Number of high/low entropy tokens to decode
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    title_suffix = f" ({category})" if category else ""

    # Extract data
    norm_entropy = entropy_stats["norm_entropy"].numpy()  # (num_layers, total_tokens)
    layer_mean = entropy_stats["layer_entropy_mean"].numpy()
    layer_std = entropy_stats["layer_entropy_std"].numpy()

    num_layers, total_tokens = norm_entropy.shape

    # ============================
    # 1) ENTROPY BY LAYER (mean + std)
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    layers = np.arange(num_layers)
    ax.errorbar(
        layers,
        layer_mean,
        yerr=layer_std,
        fmt="o-",
        capsize=3,
        color="purple",
        alpha=0.8,
        label="Mean ± Std",
    )
    ax.fill_between(
        layers,
        layer_mean - layer_std,
        layer_mean + layer_std,
        alpha=0.2,
        color="purple",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Entropy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Router Entropy by Layer{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_by_layer.png", dpi=150)
    plt.close()

    # ============================
    # 2) ENTROPY DISTRIBUTION HISTOGRAMS (sampled layers)
    # ============================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    sample_layers = np.linspace(0, num_layers - 1, 6, dtype=int)

    for ax, layer_idx in zip(axes, sample_layers):
        layer_entropy = norm_entropy[layer_idx]
        layer_entropy = layer_entropy[np.isfinite(layer_entropy)]

        if len(layer_entropy) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Layer {layer_idx}")
            continue

        ax.hist(
            layer_entropy,
            bins=50,
            density=True,
            alpha=0.7,
            color="purple",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(
            layer_entropy.mean(),
            color="red",
            linestyle="--",
            label=f"Mean={layer_entropy.mean():.3f}",
        )
        ax.axvline(
            np.median(layer_entropy),
            color="orange",
            linestyle=":",
            label=f"Median={np.median(layer_entropy):.3f}",
        )
        ax.set_xlabel("Normalized Entropy")
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    plt.suptitle(f"Router Entropy Distribution{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_histograms.png", dpi=150)
    plt.close()

    # ============================
    # 3) GLOBAL ENTROPY DISTRIBUTION
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    flat_entropy = norm_entropy.flatten()
    flat_entropy = flat_entropy[np.isfinite(flat_entropy)]

    ax.hist(
        flat_entropy,
        bins=100,
        density=True,
        alpha=0.7,
        color="purple",
        edgecolor="white",
        linewidth=0.5,
    )

    # Add percentile markers
    percentiles = entropy_stats.get("entropy_percentiles", {})
    if percentiles:
        colors = ["red", "orange", "green", "orange", "red"]
        labels = ["5th", "25th", "50th", "75th", "95th"]
        for (pname, pval), color, label in zip(percentiles.items(), colors, labels):
            ax.axvline(
                pval,
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"{label}={pval:.3f}",
            )

    ax.set_xlabel("Normalized Entropy")
    ax.set_ylabel("Density")
    ax.set_title(f"Global Router Entropy Distribution{title_suffix}")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_global_distribution.png", dpi=150)
    plt.close()

    # ============================
    # 4) ENTROPY HEATMAP BY LAYER
    # ============================
    fig, ax = plt.subplots(figsize=(12, 6))

    bins = np.linspace(0, 1, 26)
    hist_matrix = np.zeros((num_layers, len(bins) - 1))

    for layer_idx in range(num_layers):
        layer_data = norm_entropy[layer_idx]
        layer_data = layer_data[np.isfinite(layer_data)]
        if len(layer_data) > 0:
            hist, _ = np.histogram(layer_data, bins=bins, density=True)
            hist_matrix[layer_idx] = hist

    sns.heatmap(
        hist_matrix.T,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Density"},
        xticklabels=5,
        yticklabels=[f"{bins[i]:.2f}" for i in range(0, len(bins) - 1, 5)],
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Entropy")
    ax.set_title(f"Entropy Distribution by Layer{title_suffix}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_heatmap.png", dpi=150)
    plt.close()

    # ============================
    # 5) ENTROPY CATEGORIES BY LAYER (stacked area)
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    thresholds = [
        (0.0, 0.3, "Low (0.0-0.3)", "confident"),
        (0.3, 0.5, "Medium-Low (0.3-0.5)", "moderate"),
        (0.5, 0.7, "Medium-High (0.5-0.7)", "uncertain"),
        (0.7, 1.0, "High (0.7-1.0)", "very uncertain"),
    ]

    category_counts = np.zeros((num_layers, len(thresholds)))

    for layer_idx in range(num_layers):
        layer_ent = norm_entropy[layer_idx]
        for i, (low, high, _, _) in enumerate(thresholds):
            category_counts[layer_idx, i] = (
                (layer_ent >= low) & (layer_ent < high)
            ).sum()

    category_pcts = category_counts / category_counts.sum(axis=1, keepdims=True) * 100

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(thresholds)))

    bottom = np.zeros(num_layers)
    for i, (_, _, label, _) in enumerate(thresholds):
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
    ax.set_title(f"Router Confidence Categories by Layer{title_suffix}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, num_layers - 1)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_categories_by_layer.png", dpi=150)
    plt.close()

    # ============================
    # 6) LAYER CORRELATION MATRIX
    # ============================
    fig, ax = plt.subplots(figsize=(10, 8))

    # Subsample tokens if too many
    if total_tokens > 10000:
        sample_idx = np.random.choice(total_tokens, 10000, replace=False)
        entropy_sample = norm_entropy[:, sample_idx]
    else:
        entropy_sample = norm_entropy

    # Correlation between layers
    corr_matrix = np.corrcoef(entropy_sample)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(f"Entropy Correlation Between Layers{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_layer_correlation.png", dpi=150)
    plt.close()

    # ============================
    # 7) DECODE AND SAVE EXTREME TOKENS
    # ============================
    extreme_tokens_data = []

    for entropy_type, key, desc in [
        ("high", "high_entropy_tokens", "Router Uncertain"),
        ("low", "low_entropy_tokens", "Router Confident"),
    ]:
        if key not in entropy_stats:
            continue

        token_info = entropy_stats[key]
        n_tokens = min(num_extreme_tokens, len(token_info["seq_idx"]))

        for i in range(n_tokens):
            seq_idx = token_info["seq_idx"][i].item()
            offset = token_info["offset"][i].item()
            entropy_val = token_info["entropy_values"][i].item()

            try:
                input_ids = dataset[seq_idx]["input_ids"]

                # Handle tensor vs list
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.squeeze()
                    if offset >= len(input_ids):
                        continue
                    token_id = input_ids[offset].item()

                    # Context window
                    ctx_start = max(0, offset - 5)
                    ctx_end = min(len(input_ids), offset + 6)
                    context_ids = input_ids[ctx_start:ctx_end].tolist()
                else:
                    if offset >= len(input_ids):
                        continue
                    token_id = input_ids[offset]
                    ctx_start = max(0, offset - 5)
                    ctx_end = min(len(input_ids), offset + 6)
                    context_ids = input_ids[ctx_start:ctx_end]

                token_str = tokenizer.decode([token_id])
                context_str = tokenizer.decode(context_ids)

                # Mark the target token in context
                pre_context = tokenizer.decode(context_ids[: offset - ctx_start])
                post_context = tokenizer.decode(context_ids[offset - ctx_start + 1 :])
                marked_context = f"{pre_context}>>>{token_str}<<<{post_context}"

                extreme_tokens_data.append(
                    {
                        "type": entropy_type,
                        "rank": i + 1,
                        "entropy": entropy_val,
                        "token": token_str,
                        "token_id": token_id,
                        "seq_idx": seq_idx,
                        "position": offset,
                        "context": context_str,
                        "marked_context": marked_context,
                    }
                )

            except Exception as e:
                print(f"Error decoding token {i} ({entropy_type}): {e}")
                continue

    # Save to CSV
    if extreme_tokens_data:
        df = pd.DataFrame(extreme_tokens_data)
        df.to_csv(out_dir / "extreme_entropy_tokens.csv", index=False)

        # Also save readable text report
        with open(out_dir / "extreme_entropy_tokens.txt", "w") as f:
            f.write(f"=== Extreme Entropy Tokens{title_suffix} ===\n\n")

            f.write("=" * 60 + "\n")
            f.write("HIGH ENTROPY TOKENS (Router Uncertain)\n")
            f.write("=" * 60 + "\n\n")

            high_df = df[df["type"] == "high"].head(30)
            for _, row in high_df.iterrows():
                f.write(f"[{row['rank']}] Entropy: {row['entropy']:.4f}\n")
                f.write(f"    Token: '{row['token']}' (id={row['token_id']})\n")
                f.write(f"    Position: seq={row['seq_idx']}, pos={row['position']}\n")
                f.write(f"    Context: {row['marked_context']}\n\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("LOW ENTROPY TOKENS (Router Confident)\n")
            f.write("=" * 60 + "\n\n")

            low_df = df[df["type"] == "low"].head(30)
            for _, row in low_df.iterrows():
                f.write(f"[{row['rank']}] Entropy: {row['entropy']:.4f}\n")
                f.write(f"    Token: '{row['token']}' (id={row['token_id']})\n")
                f.write(f"    Position: seq={row['seq_idx']}, pos={row['position']}\n")
                f.write(f"    Context: {row['marked_context']}\n\n")

    # ============================
    # 8) SUMMARY STATISTICS
    # ============================
    summary_df = pd.DataFrame(
        {
            "layer": np.arange(num_layers),
            "entropy_mean": layer_mean,
            "entropy_std": layer_std,
            "entropy_median": np.median(norm_entropy, axis=1),
            "entropy_min": np.nanmin(norm_entropy, axis=1),
            "entropy_max": np.nanmax(norm_entropy, axis=1),
        }
    )
    summary_df.to_csv(out_dir / "entropy_layer_summary.csv", index=False)

    # Text summary
    with open(out_dir / "entropy_summary.txt", "w") as f:
        f.write(f"=== Router Entropy Analysis{title_suffix} ===\n\n")
        f.write(f"Total tokens analyzed: {total_tokens:,}\n")
        f.write(f"Number of layers: {num_layers}\n\n")

        f.write("Global Statistics:\n")
        f.write(
            f"  Mean entropy:   {entropy_stats.get('global_entropy_mean', np.nanmean(norm_entropy)):.4f}\n"
        )
        f.write(
            f"  Std entropy:    {entropy_stats.get('global_entropy_std', np.nanstd(norm_entropy)):.4f}\n"
        )
        f.write(
            f"  Median entropy: {entropy_stats.get('global_entropy_median', np.nanmedian(norm_entropy)):.4f}\n\n"
        )

        if "entropy_percentiles" in entropy_stats:
            f.write("Percentiles:\n")
            for pname, pval in entropy_stats["entropy_percentiles"].items():
                f.write(f"  {pname}: {pval:.4f}\n")
            f.write("\n")

        f.write("Layer Statistics:\n")
        f.write(
            f"  Lowest mean entropy:  Layer {np.argmin(layer_mean)} ({layer_mean.min():.4f})\n"
        )
        f.write(
            f"  Highest mean entropy: Layer {np.argmax(layer_mean)} ({layer_mean.max():.4f})\n"
        )
        f.write(
            f"  Mean entropy range:   [{layer_mean.min():.4f}, {layer_mean.max():.4f}]\n\n"
        )

        # Token type analysis if we decoded tokens
        if extreme_tokens_data:
            high_tokens = [
                d["token"] for d in extreme_tokens_data if d["type"] == "high"
            ]
            low_tokens = [d["token"] for d in extreme_tokens_data if d["type"] == "low"]

            f.write("High Entropy Token Examples (router uncertain):\n")
            f.write(f"  {high_tokens[:20]}\n\n")

            f.write("Low Entropy Token Examples (router confident):\n")
            f.write(f"  {low_tokens[:20]}\n")

    print(f"Saved entropy analysis to {out_dir}")


def compare_entropy_across_categories(
    category_entropy_stats: dict,
    out_dir: str | Path,
):
    """
    Compare entropy patterns across different data categories.

    Args:
        category_entropy_stats: Dict mapping category name to entropy_stats dict
        out_dir: Output directory
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = list(category_entropy_stats.keys())
    num_layers = category_entropy_stats[categories[0]]["layer_entropy_mean"].shape[0]

    # ============================
    # 1) MEAN ENTROPY BY CATEGORY
    # ============================
    fig, ax = plt.subplots(figsize=(12, 6))

    for cat in categories:
        mean = category_entropy_stats[cat]["layer_entropy_mean"].numpy()
        ax.plot(np.arange(num_layers), mean, "o-", label=cat, alpha=0.8, markersize=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Normalized Entropy")
    ax.set_ylim(0, 1)
    ax.set_title("Router Entropy by Category")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_by_category.png", dpi=150)
    plt.close()

    # ============================
    # 2) GLOBAL ENTROPY COMPARISON (box plot)
    # ============================
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = []
    plot_labels = []

    for cat in categories:
        entropy = category_entropy_stats[cat]["norm_entropy"].numpy().flatten()
        entropy = entropy[np.isfinite(entropy)]

        # Subsample for plotting
        if len(entropy) > 10000:
            entropy = np.random.choice(entropy, 10000, replace=False)

        plot_data.append(entropy)
        plot_labels.append(cat)

    ax.boxplot(plot_data, labels=plot_labels, vert=True)
    ax.set_ylabel("Normalized Entropy")
    ax.set_title("Entropy Distribution by Category")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_category_boxplot.png", dpi=150)
    plt.close()

    # ============================
    # 3) CONFIDENCE BREAKDOWN BY CATEGORY
    # ============================
    fig, ax = plt.subplots(figsize=(12, 6))

    thresholds = [
        (0.0, 0.3, "Low"),
        (0.3, 0.5, "Med-Low"),
        (0.5, 0.7, "Med-High"),
        (0.7, 1.0, "High"),
    ]

    category_breakdown = {cat: [] for cat in categories}

    for cat in categories:
        entropy = category_entropy_stats[cat]["norm_entropy"].numpy().flatten()
        total = len(entropy)
        for low, high, _ in thresholds:
            pct = ((entropy >= low) & (entropy < high)).sum() / total * 100
            category_breakdown[cat].append(pct)

    x = np.arange(len(categories))
    width = 0.2

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(thresholds)))

    for i, (_, _, label) in enumerate(thresholds):
        values = [category_breakdown[cat][i] for cat in categories]
        ax.bar(x + i * width, values, width, label=label, color=colors[i])

    ax.set_xlabel("Category")
    ax.set_ylabel("Percentage of Tokens")
    ax.set_title("Entropy Categories by Data Type")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend(title="Entropy Level")
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_category_breakdown.png", dpi=150)
    plt.close()

    # ============================
    # 4) SUMMARY CSV
    # ============================
    summary_data = []
    for cat in categories:
        entropy = category_entropy_stats[cat]["norm_entropy"].numpy()
        summary_data.append(
            {
                "category": cat,
                "mean_entropy": np.nanmean(entropy),
                "std_entropy": np.nanstd(entropy),
                "median_entropy": np.nanmedian(entropy),
                "pct_low_entropy": ((entropy < 0.3).sum() / entropy.size) * 100,
                "pct_high_entropy": ((entropy > 0.7).sum() / entropy.size) * 100,
            }
        )

    pd.DataFrame(summary_data).to_csv(
        out_dir / "entropy_category_comparison.csv", index=False
    )

    print(f"Saved category comparison to {out_dir}")


def compute_position_significance(
    norm_entropy: torch.Tensor,
    expert_ids: torch.Tensor,
    token_pos: torch.Tensor,
    position_buckets: dict,
    num_layers: int,
) -> dict:
    """
    Statistical tests for differences across position buckets.
    """
    significance = {}

    entropy_np = norm_entropy.numpy()
    positions_np = token_pos.numpy()

    # 1. ANOVA across all buckets (per layer)
    anova_results = []
    for layer in range(num_layers):
        layer_entropy = entropy_np[layer]
        groups = []
        for bucket_name, mask in position_buckets.items():
            bucket_entropy = layer_entropy[mask.numpy()]
            if len(bucket_entropy) > 0:
                groups.append(bucket_entropy)

        if len(groups) >= 2:
            f_stat, p_val = scipy_stats.f_oneway(*groups)
            anova_results.append({"layer": layer, "f_stat": f_stat, "p_value": p_val})

    significance["anova_entropy"] = pd.DataFrame(anova_results)

    # 2. Pairwise comparisons
    pairwise_comparisons = [
        ("0-50", "200+"),
        ("0-50", "50-100"),
        ("first", "0-50"),
    ]

    pairwise_results = []
    for bucket_a, bucket_b in pairwise_comparisons:
        if bucket_a not in position_buckets or bucket_b not in position_buckets:
            continue

        mask_a = position_buckets[bucket_a].numpy()
        mask_b = position_buckets[bucket_b].numpy()

        for layer in range(num_layers):
            entropy_a = entropy_np[layer, mask_a]
            entropy_b = entropy_np[layer, mask_b]

            if len(entropy_a) < 2 or len(entropy_b) < 2:
                continue

            u_stat, p_val = scipy_stats.mannwhitneyu(
                entropy_a, entropy_b, alternative="two-sided"
            )

            n1, n2 = len(entropy_a), len(entropy_b)
            effect_size = 1 - (2 * u_stat) / (n1 * n2)

            pairwise_results.append(
                {
                    "layer": layer,
                    "bucket_a": bucket_a,
                    "bucket_b": bucket_b,
                    "mean_a": entropy_a.mean(),
                    "mean_b": entropy_b.mean(),
                    "u_stat": u_stat,
                    "p_value": p_val,
                    "effect_size": effect_size,
                }
            )

    significance["pairwise_entropy"] = pd.DataFrame(pairwise_results)

    # 3. Correlation: position vs entropy
    correlation_results = []
    for layer in range(num_layers):
        layer_entropy = entropy_np[layer]
        rho, p_val = scipy_stats.spearmanr(positions_np, layer_entropy)
        correlation_results.append(
            {
                "layer": layer,
                "spearman_rho": rho,
                "p_value": p_val,
            }
        )

    significance["position_entropy_correlation"] = pd.DataFrame(correlation_results)

    return significance


def analyze_position(
    probs: torch.Tensor,
    seq_lengths: torch.Tensor,
    compute_significance: bool = True,
) -> dict:
    """
    Analyze routing patterns by token position within sequences.

    Args:
        probs: (num_layers, total_tokens, num_experts) routing probabilities
        seq_lengths: (num_sequences,) length of each sequence
        compute_significance: whether to run statistical tests

    Returns:
        Dictionary with position-bucketed statistics
    """
    num_layers, total_tokens, num_experts = probs.shape

    # Entropy
    max_entropy = np.log(num_experts)
    token_entropy = -torch.xlogy(probs, probs).sum(dim=-1)
    norm_entropy = token_entropy / max_entropy

    # Top-1 expert
    expert_weight, expert_ids = torch.topk(probs, k=1, dim=-1)
    expert_ids = expert_ids.squeeze(-1)
    expert_weight = expert_weight.squeeze(-1)

    # Position mapping
    positions = torch.arange(total_tokens)
    seq_starts = torch.repeat_interleave(
        torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)[:-1]]), seq_lengths
    )
    token_pos = positions - seq_starts

    seq_indices = torch.repeat_interleave(torch.arange(len(seq_lengths)), seq_lengths)

    # Position buckets
    def _get_mask(lo=None, hi=None):
        if lo is None and hi is None:
            return torch.ones(total_tokens, dtype=torch.bool)
        if lo is None:
            return token_pos < hi
        if hi is None:
            return token_pos >= lo
        return (token_pos >= lo) & (token_pos < hi)

    position_buckets = {
        "first": token_pos == 0,
        "0-50": _get_mask(0, 50),
        "50-100": _get_mask(50, 100),
        "100-200": _get_mask(100, 200),
        "200+": _get_mask(200, None),
    }

    def _get_util(mask):
        mask_count = mask.sum().item()
        if mask_count == 0:
            return torch.zeros(num_layers, num_experts)
        counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
        for layer in range(num_layers):
            counts[layer] = torch.bincount(
                expert_ids[layer, mask], minlength=num_experts
            )
        return counts.float() / mask_count

    output_dict = {
        # Bucket info
        "position_buckets": position_buckets,
        "bucket_names": list(position_buckets.keys()),
        "bucket_counts": {k: v.sum().item() for k, v in position_buckets.items()},
        # Per-bucket statistics
        "utilization_by_position": {
            k: _get_util(v) for k, v in position_buckets.items()
        },
        "entropy_by_position": {
            k: norm_entropy[:, v].mean(dim=-1) for k, v in position_buckets.items()
        },
        "entropy_std_by_position": {
            k: norm_entropy[:, v].std(dim=-1) for k, v in position_buckets.items()
        },
        "primary_weight_by_position": {
            k: expert_weight[:, v].mean(dim=-1) for k, v in position_buckets.items()
        },
        "primary_weight_std_by_position": {
            k: expert_weight[:, v].std(dim=-1) for k, v in position_buckets.items()
        },
        # Raw data
        "token_positions": token_pos,
        "seq_indices": seq_indices,
        "norm_entropy": norm_entropy,
        "expert_ids": expert_ids,
        "expert_weight": expert_weight,
        # Metadata
        "num_layers": num_layers,
        "num_experts": num_experts,
        "total_tokens": total_tokens,
    }

    # Statistical significance
    if compute_significance:
        output_dict["significance"] = compute_position_significance(
            norm_entropy, expert_ids, token_pos, position_buckets, num_layers
        )

    return output_dict


def save_position_analysis_plots(
    stats: dict,
    out_dir: str | Path,
    category: str = None,
):
    """
    Generate plots for position analysis (Stage 2.5).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    title_suffix = f" ({category})" if category else ""

    bucket_names = stats["bucket_names"]
    num_layers = stats["num_layers"]
    num_experts = stats["num_experts"]

    # ============================
    # 1) ENTROPY BY POSITION BUCKET
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    for bucket in bucket_names:
        entropy = stats["entropy_by_position"][bucket].numpy()
        ax.plot(np.arange(num_layers), entropy, "o-", label=bucket, markersize=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Normalized Entropy")
    ax.set_title(f"Router Entropy by Position{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_by_position.png", dpi=150)
    plt.close()

    # ============================
    # 2) PRIMARY WEIGHT BY POSITION
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    for bucket in bucket_names:
        weight = stats["primary_weight_by_position"][bucket].numpy()
        ax.plot(np.arange(num_layers), weight, "o-", label=bucket, markersize=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Primary Expert Weight")
    ax.set_title(f"Router Decisiveness by Position{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "primary_weight_by_position.png", dpi=150)
    plt.close()

    # ============================
    # 3) UTILIZATION HEATMAPS BY POSITION
    # ============================
    n_buckets = len(bucket_names)
    fig, axes = plt.subplots(1, n_buckets, figsize=(4 * n_buckets, 6))
    if n_buckets == 1:
        axes = [axes]

    for ax, bucket in zip(axes, bucket_names):
        util = stats["utilization_by_position"][bucket].numpy()
        expected = 1.0 / num_experts

        sns.heatmap(
            util,
            cmap="RdBu_r",
            center=expected,
            ax=ax,
            cbar_kws={"label": "Utilization"},
        )
        ax.set_xlabel("Expert")
        ax.set_ylabel("Layer")
        ax.set_title(f"{bucket}\n(n={stats['bucket_counts'][bucket]:,})")

    plt.suptitle(f"Expert Utilization by Position{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_dir / "utilization_by_position.png", dpi=150)
    plt.close()

    # ============================
    # 4) UTILIZATION DIFFERENCE FROM GLOBAL
    # ============================
    global_util = stats["utilization_by_position"]["0-50"]  # or compute true global

    fig, axes = plt.subplots(1, n_buckets - 1, figsize=(4 * (n_buckets - 1), 6))
    if n_buckets - 1 == 1:
        axes = [axes]

    for ax, bucket in zip(axes, [b for b in bucket_names if b != "0-50"]):
        util = stats["utilization_by_position"][bucket].numpy()
        diff = util - global_util.numpy()

        vmax = np.abs(diff).max()
        sns.heatmap(
            diff,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            ax=ax,
            cbar_kws={"label": "Δ Utilization"},
        )
        ax.set_xlabel("Expert")
        ax.set_ylabel("Layer")
        ax.set_title(f"{bucket} vs 0-50")

    plt.suptitle(f"Utilization Difference by Position{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_dir / "utilization_diff_by_position.png", dpi=150)
    plt.close()

    # ============================
    # 5) POSITION-ENTROPY CORRELATION
    # ============================
    if "significance" in stats:
        corr_df = stats["significance"]["position_entropy_correlation"]

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ["green" if p < 0.05 else "gray" for p in corr_df["p_value"]]
        ax.bar(corr_df["layer"], corr_df["spearman_rho"], color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Spearman ρ")
        ax.set_title(f"Position-Entropy Correlation{title_suffix}\n(green = p < 0.05)")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir / "position_entropy_correlation.png", dpi=150)
        plt.close()

    # ============================
    # 6) ANOVA SIGNIFICANCE
    # ============================
    if "significance" in stats:
        anova_df = stats["significance"]["anova_entropy"]

        fig, ax = plt.subplots(figsize=(10, 5))

        layers = anova_df["layer"].values
        f_stats = anova_df["f_stat"].values

        ax.bar(layers, f_stats, color="steelblue", alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("F-statistic")
        ax.set_title(
            f"ANOVA F-statistic: Position Effect Strength{title_suffix}\n(higher = stronger position effect)"
        )
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir / "position_anova_fstat.png", dpi=150)
        plt.close()

    # ============================
    # 7) BUCKET COUNTS
    # ============================
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = [stats["bucket_counts"][b] for b in bucket_names]
    ax.bar(bucket_names, counts, color="steelblue", alpha=0.8)
    ax.set_xlabel("Position Bucket")
    ax.set_ylabel("Token Count")
    ax.set_title(f"Tokens per Position Bucket{title_suffix}")

    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "position_bucket_counts.png", dpi=150)
    plt.close()

    # ============================
    # 8) SAVE STATISTICS
    # ============================
    # Summary CSV
    summary_data = []
    for bucket in bucket_names:
        entropy = stats["entropy_by_position"][bucket].numpy()
        weight = stats["primary_weight_by_position"][bucket].numpy()
        summary_data.append(
            {
                "bucket": bucket,
                "token_count": stats["bucket_counts"][bucket],
                "entropy_mean": entropy.mean(),
                "entropy_std": entropy.std(),
                "primary_weight_mean": weight.mean(),
                "primary_weight_std": weight.std(),
            }
        )
    pd.DataFrame(summary_data).to_csv(out_dir / "position_summary.csv", index=False)

    # Significance CSVs
    if "significance" in stats:
        stats["significance"]["anova_entropy"].to_csv(
            out_dir / "significance_anova.csv", index=False
        )
        stats["significance"]["pairwise_entropy"].to_csv(
            out_dir / "significance_pairwise.csv", index=False
        )
        stats["significance"]["position_entropy_correlation"].to_csv(
            out_dir / "significance_correlation.csv", index=False
        )

    print(f"Saved position analysis to {out_dir}")


def analyze_cross_layer_consistency(
    probs: torch.Tensor,
    top_k: int = 1,
) -> dict:
    """
    Analyze routing consistency across layers.

    Args:
        probs: (num_layers, total_tokens, num_experts) routing probabilities
        top_k: number of top experts to consider for routing decisions

    Returns:
        Dictionary with cross-layer routing statistics
    """
    num_layers, total_tokens, num_experts = probs.shape

    # Get top-k expert selections per layer
    _, expert_ids = torch.topk(probs, k=top_k, dim=-1)
    if top_k == 1:
        expert_ids = expert_ids.squeeze(-1)  # (num_layers, total_tokens)
    else:
        expert_ids = expert_ids[:, :, 0]  # Use top-1 for consistency analysis

    output_dict = {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "total_tokens": total_tokens,
    }

    # ============================
    # 1. TRANSITION MATRICES (adjacent layers)
    # ============================
    # For each pair of adjacent layers, compute P(expert_j in layer L+1 | expert_i in layer L)
    transition_matrices = []

    for layer in range(num_layers - 1):
        current_experts = expert_ids[layer]  # (total_tokens,)
        next_experts = expert_ids[layer + 1]  # (total_tokens,)

        # Count transitions
        transition_counts = torch.zeros(num_experts, num_experts, dtype=torch.long)
        for i in range(num_experts):
            mask = current_experts == i
            if mask.sum() > 0:
                next_exp = next_experts[mask]
                counts = torch.bincount(next_exp, minlength=num_experts)
                transition_counts[i] = counts

        # Normalize to probabilities (row-stochastic)
        row_sums = transition_counts.sum(dim=1, keepdim=True).float()
        row_sums = torch.clamp(row_sums, min=1)  # Avoid division by zero
        transition_probs = transition_counts.float() / row_sums

        transition_matrices.append(transition_probs)

    output_dict["transition_matrices"] = torch.stack(
        transition_matrices
    )  # (num_layers-1, num_experts, num_experts)

    # ============================
    # 2. MUTUAL INFORMATION BETWEEN LAYERS
    # ============================
    mutual_info = torch.zeros(num_layers, num_layers)

    for i in range(num_layers):
        for j in range(i, num_layers):
            if i == j:
                # Self MI is entropy
                expert_counts = torch.bincount(
                    expert_ids[i], minlength=num_experts
                ).float()
                p = expert_counts / total_tokens
                p = p[p > 0]
                mi = -(p * torch.log(p)).sum()
            else:
                # Joint distribution
                joint_counts = torch.zeros(num_experts, num_experts)
                for ei in range(num_experts):
                    mask = expert_ids[i] == ei
                    if mask.sum() > 0:
                        ej_counts = torch.bincount(
                            expert_ids[j, mask], minlength=num_experts
                        )
                        joint_counts[ei] = ej_counts.float()

                joint_p = joint_counts / total_tokens

                # Marginals
                p_i = joint_p.sum(dim=1)
                p_j = joint_p.sum(dim=0)

                # MI = sum p(i,j) * log(p(i,j) / (p(i) * p(j)))
                mi = 0.0
                for ei in range(num_experts):
                    for ej in range(num_experts):
                        if joint_p[ei, ej] > 0 and p_i[ei] > 0 and p_j[ej] > 0:
                            mi += joint_p[ei, ej] * torch.log(
                                joint_p[ei, ej] / (p_i[ei] * p_j[ej])
                            )

            mutual_info[i, j] = mi
            mutual_info[j, i] = mi

    output_dict["mutual_information"] = mutual_info

    # Normalize MI by max possible (entropy)
    max_entropy = np.log(num_experts)
    output_dict["normalized_mutual_info"] = mutual_info / max_entropy

    # ============================
    # 3. ADJACENT LAYER AGREEMENT
    # ============================
    # What fraction of tokens go to the same expert in consecutive layers?
    same_expert_rate = []
    for layer in range(num_layers - 1):
        same = (expert_ids[layer] == expert_ids[layer + 1]).float().mean()
        same_expert_rate.append(same.item())

    output_dict["same_expert_rate"] = torch.tensor(same_expert_rate)
    output_dict["expected_same_rate"] = 1.0 / num_experts  # Random baseline

    # ============================
    # 4. ROUTING TRAJECTORY CLUSTERING
    # ============================
    # Represent each token by its full routing path, find common patterns
    # Use a subset of tokens for computational efficiency
    max_tokens_for_clustering = min(10000, total_tokens)
    sample_idx = torch.randperm(total_tokens)[:max_tokens_for_clustering]

    routing_paths = expert_ids[:, sample_idx].T  # (sampled_tokens, num_layers)

    # Convert paths to strings for counting unique patterns
    path_strings = ["-".join(map(str, path.tolist())) for path in routing_paths]

    from collections import Counter

    path_counts = Counter(path_strings)

    # Top 20 most common paths
    top_paths = path_counts.most_common(20)
    output_dict["top_routing_paths"] = [
        {
            "path": p[0],
            "count": p[1],
            "frequency": p[1] / max_tokens_for_clustering,
        }
        for p in top_paths
    ]
    output_dict["num_unique_paths"] = len(path_counts)
    output_dict["path_concentration"] = (
        sum(c for _, c in top_paths) / max_tokens_for_clustering
    )

    # ============================
    # 5. LAYER-WISE ROUTING ENTROPY
    # ============================
    # How predictable is the next layer's routing given current layer?
    conditional_entropy = []

    for layer in range(num_layers - 1):
        # H(L+1 | L) = H(L, L+1) - H(L)
        # H(L, L+1) = -sum p(i,j) log p(i,j)
        # H(L) = -sum p(i) log p(i)

        joint_counts = torch.zeros(num_experts, num_experts)
        for ei in range(num_experts):
            mask = expert_ids[layer] == ei
            if mask.sum() > 0:
                ej_counts = torch.bincount(
                    expert_ids[layer + 1, mask], minlength=num_experts
                )
                joint_counts[ei] = ej_counts.float()

        joint_p = joint_counts / total_tokens
        p_current = joint_p.sum(dim=1)

        # Joint entropy
        joint_p_flat = joint_p.flatten()
        joint_p_flat = joint_p_flat[joint_p_flat > 0]
        h_joint = -(joint_p_flat * torch.log(joint_p_flat)).sum()

        # Marginal entropy
        p_current = p_current[p_current > 0]
        h_current = -(p_current * torch.log(p_current)).sum()

        # Conditional entropy
        h_cond = h_joint - h_current
        conditional_entropy.append(h_cond.item())

    output_dict["conditional_entropy"] = torch.tensor(conditional_entropy)
    output_dict["conditional_entropy_normalized"] = (
        torch.tensor(conditional_entropy) / max_entropy
    )

    # ============================
    # 6. "STICKY" EXPERT PAIRS
    # ============================
    # Which expert pairs (layer L -> layer L+1) occur much more than expected?
    sticky_pairs = []

    for layer in range(num_layers - 1):
        trans = output_dict["transition_matrices"][layer]

        # Expected under independence
        marginal_current = torch.bincount(
            expert_ids[layer], minlength=num_experts
        ).float()
        marginal_current = marginal_current / total_tokens
        marginal_next = torch.bincount(
            expert_ids[layer + 1], minlength=num_experts
        ).float()
        marginal_next = marginal_next / total_tokens

        expected = marginal_current.unsqueeze(1) * marginal_next.unsqueeze(0)

        # Observed
        joint_counts = torch.zeros(num_experts, num_experts)
        for ei in range(num_experts):
            mask = expert_ids[layer] == ei
            if mask.sum() > 0:
                ej_counts = torch.bincount(
                    expert_ids[layer + 1, mask], minlength=num_experts
                )
                joint_counts[ei] = ej_counts.float()
        observed = joint_counts / total_tokens

        # Ratio (observed / expected)
        ratio = observed / (expected + 1e-8)

        # Find pairs with highest ratio
        top_k_pairs = 5
        flat_ratio = ratio.flatten()
        top_vals, top_idx = torch.topk(flat_ratio, top_k_pairs)

        for val, idx in zip(top_vals, top_idx):
            ei = idx // num_experts
            ej = idx % num_experts
            sticky_pairs.append(
                {
                    "layer": layer,
                    "expert_from": ei.item(),
                    "expert_to": ej.item(),
                    "observed": observed[ei, ej].item(),
                    "expected": expected[ei, ej].item(),
                    "ratio": val.item(),
                }
            )

    output_dict["sticky_pairs"] = pd.DataFrame(sticky_pairs)

    return output_dict


def save_cross_layer_plots(
    stats: dict,
    out_dir: str | Path,
    category: Optional[str] = None,
):
    """
    Generate plots for cross-layer routing consistency analysis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    title_suffix = f" ({category})" if category else ""

    num_layers = stats["num_layers"]
    num_experts = stats["num_experts"]

    # ============================
    # 1) MUTUAL INFORMATION HEATMAP
    # ============================
    fig, ax = plt.subplots(figsize=(10, 8))

    mi = stats["normalized_mutual_info"].numpy()

    sns.heatmap(
        mi,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Normalized MI"},
        xticklabels=range(num_layers),
        yticklabels=range(num_layers),
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(f"Mutual Information Between Layers{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_dir / "mutual_information_heatmap.png", dpi=150)
    plt.close()

    # ============================
    # 2) ADJACENT LAYER AGREEMENT
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    same_rate = stats["same_expert_rate"].numpy()
    expected = stats["expected_same_rate"]

    layers = np.arange(len(same_rate))
    ax.bar(layers, same_rate, color="steelblue", alpha=0.8)
    ax.axhline(
        expected, color="red", linestyle="--", label=f"Random baseline ({expected:.3f})"
    )

    ax.set_xlabel("Layer Transition (L → L+1)")
    ax.set_ylabel("Same Expert Rate")
    ax.set_title(f"Adjacent Layer Routing Agreement{title_suffix}")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"{i}→{i + 1}" for i in layers], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "adjacent_layer_agreement.png", dpi=150)
    plt.close()

    # ============================
    # 3) CONDITIONAL ENTROPY
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    cond_entropy = stats["conditional_entropy_normalized"].numpy()

    ax.plot(np.arange(len(cond_entropy)), cond_entropy, "o-", color="purple")
    ax.axhline(1.0, color="red", linestyle="--", label="Max entropy (independent)")

    ax.set_xlabel("Layer Transition (L → L+1)")
    ax.set_ylabel("Normalized Conditional Entropy")
    ax.set_title(f"Routing Predictability{title_suffix}\n(lower = more predictable)")
    ax.set_xticks(np.arange(len(cond_entropy)))
    ax.set_xticklabels(
        [f"{i}→{i + 1}" for i in range(len(cond_entropy))], rotation=45, ha="right"
    )
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "conditional_entropy.png", dpi=150)
    plt.close()

    # ============================
    # 4) SAMPLE TRANSITION MATRICES
    # ============================
    # Show 4 transition matrices from different parts of the network
    sample_layers = [0, num_layers // 3, 2 * num_layers // 3, num_layers - 2]
    sample_layers = [l for l in sample_layers if l < num_layers - 1]

    fig, axes = plt.subplots(1, len(sample_layers), figsize=(5 * len(sample_layers), 5))
    if len(sample_layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, sample_layers):
        trans = stats["transition_matrices"][layer].numpy()

        sns.heatmap(
            trans,
            cmap="Blues",
            ax=ax,
            cbar=True,
            vmin=0,
            vmax=trans.max(),
            xticklabels=10,
            yticklabels=10,
        )
        ax.set_xlabel(f"Expert (Layer {layer + 1})")
        ax.set_ylabel(f"Expert (Layer {layer})")
        ax.set_title(f"Transition {layer}→{layer + 1}")

    plt.suptitle(f"Routing Transition Matrices{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_dir / "transition_matrices.png", dpi=150)
    plt.close()

    # ============================
    # 5) TRANSITION MATRIX DIAGONAL STRENGTH
    # ============================
    # How much of the probability mass is on the diagonal (same expert)?
    fig, ax = plt.subplots(figsize=(10, 5))

    diagonal_mass = []
    for layer in range(num_layers - 1):
        trans = stats["transition_matrices"][layer].numpy()
        diag_sum = np.trace(trans) / num_experts  # Average diagonal probability
        diagonal_mass.append(diag_sum)

    ax.bar(np.arange(len(diagonal_mass)), diagonal_mass, color="teal", alpha=0.8)
    ax.axhline(
        1.0 / num_experts,
        color="red",
        linestyle="--",
        label=f"Random ({1.0 / num_experts:.3f})",
    )
    ax.set_xlabel("Layer Transition")
    ax.set_ylabel("Mean Diagonal Probability")
    ax.set_title(f"Same-Expert Transition Strength{title_suffix}")
    ax.set_xticks(np.arange(len(diagonal_mass)))
    ax.set_xticklabels(
        [f"{i}→{i + 1}" for i in range(len(diagonal_mass))], rotation=45, ha="right"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "diagonal_transition_strength.png", dpi=150)
    plt.close()

    # ============================
    # 6) MI DECAY WITH LAYER DISTANCE
    # ============================
    fig, ax = plt.subplots(figsize=(10, 5))

    mi = stats["normalized_mutual_info"].numpy()

    # Group MI by layer distance
    max_distance = num_layers - 1
    distances = list(range(1, max_distance + 1))
    mi_by_distance = []

    for d in distances:
        mi_values = []
        for i in range(num_layers - d):
            mi_values.append(mi[i, i + d])
        mi_by_distance.append(
            {
                "distance": d,
                "mean_mi": np.mean(mi_values),
                "std_mi": np.std(mi_values),
            }
        )

    mi_df = pd.DataFrame(mi_by_distance)

    ax.errorbar(
        mi_df["distance"],
        mi_df["mean_mi"],
        yerr=mi_df["std_mi"],
        fmt="o-",
        capsize=3,
        color="steelblue",
    )
    ax.set_xlabel("Layer Distance")
    ax.set_ylabel("Normalized Mutual Information")
    ax.set_title(f"MI Decay with Layer Distance{title_suffix}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mi_decay_with_distance.png", dpi=150)
    plt.close()

    # ============================
    # 7) TOP ROUTING PATHS
    # ============================
    top_paths = stats["top_routing_paths"]

    fig, ax = plt.subplots(figsize=(12, 6))

    paths = [
        p["path"][:30] + "..." if len(p["path"]) > 30 else p["path"]
        for p in top_paths[:15]
    ]
    freqs = [p["frequency"] * 100 for p in top_paths[:15]]

    y_pos = np.arange(len(paths))
    ax.barh(y_pos, freqs, color="coral", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(paths, fontsize=8)
    ax.set_xlabel("Frequency (%)")
    ax.set_title(
        f"Top Routing Paths{title_suffix}\n({stats['num_unique_paths']:,} unique paths)"
    )
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "top_routing_paths.png", dpi=150)
    plt.close()

    # ============================
    # 8) SAVE STATISTICS
    # ============================
    # Summary CSV
    summary_data = {
        "metric": [
            "num_unique_paths",
            "path_concentration_top20",
            "mean_same_expert_rate",
            "expected_same_expert_rate",
            "mean_conditional_entropy",
            "mean_mutual_info_adjacent",
        ],
        "value": [
            stats["num_unique_paths"],
            stats["path_concentration"],
            stats["same_expert_rate"].mean().item(),
            stats["expected_same_rate"],
            stats["conditional_entropy_normalized"].mean().item(),
            stats["normalized_mutual_info"].diagonal(offset=1).mean().item(),
        ],
    }
    pd.DataFrame(summary_data).to_csv(out_dir / "cross_layer_summary.csv", index=False)

    # Sticky pairs
    if "sticky_pairs" in stats and not stats["sticky_pairs"].empty:
        stats["sticky_pairs"].sort_values("ratio", ascending=False).head(50).to_csv(
            out_dir / "sticky_expert_pairs.csv", index=False
        )

    # MI decay
    mi_df.to_csv(out_dir / "mi_by_distance.csv", index=False)

    # Text summary
    with open(out_dir / "cross_layer_summary.txt", "w") as f:
        f.write(f"=== Cross-Layer Routing Consistency{title_suffix} ===\n\n")
        f.write(f"Layers: {num_layers}\n")
        f.write(f"Experts: {num_experts}\n")
        f.write(f"Tokens analyzed: {stats['total_tokens']:,}\n\n")

        f.write("Routing Path Diversity:\n")
        f.write(f"  Unique paths: {stats['num_unique_paths']:,}\n")
        f.write(
            f"  Top 20 paths cover: {stats['path_concentration'] * 100:.1f}% of tokens\n\n"
        )

        f.write("Adjacent Layer Agreement:\n")
        f.write(f"  Mean same-expert rate: {stats['same_expert_rate'].mean():.4f}\n")
        f.write(f"  Random baseline: {stats['expected_same_rate']:.4f}\n")
        f.write(
            f"  Ratio vs random: {stats['same_expert_rate'].mean() / stats['expected_same_rate']:.2f}x\n\n"
        )

        f.write("Routing Predictability:\n")
        f.write(
            f"  Mean conditional entropy: {stats['conditional_entropy_normalized'].mean():.4f}\n"
        )
        f.write("  (1.0 = independent, 0.0 = fully predictable)\n\n")

        f.write("Top 10 Most Common Routing Paths:\n")
        for i, p in enumerate(top_paths[:10]):
            f.write(f"  {i + 1}. {p['path']} ({p['frequency'] * 100:.2f}%)\n")

    print(f"Saved cross-layer analysis to {out_dir}")


def load_logits(logit_out_dir: Path, categories: Optional[List | Set] = None):
    logit_dict = {}
    global_logits = []
    sequence_lengths = []

    if categories is None:
        categories = list(x.stem for x in logit_out_dir.glob("*.pt"))

    for category in categories:
        category_dict = torch.load(logit_out_dir / f"{category}.pt")
        logit_dict[category] = category_dict
        global_logits.append(category_dict["logits"])
        sequence_lengths.append(category_dict["token_counts"])

    global_logits = torch.cat(global_logits, dim=1)
    sequence_lengths = torch.cat(sequence_lengths, dim=0)

    return logit_dict, global_logits, sequence_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", type=str, default="/workspace/ml-stuff/MoE/out/baseline"
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--num_experts", type=int, default=60)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logit_out_dir = out_dir / "logits"
    logit_out_dir.mkdir(exist_ok=True)

    unique_categories = [x.category for x in DEFAULT_SOURCES]
    print(f"{unique_categories} Categories")

    # load model
    model, tokenizer = build_model_and_tokenizer(MODEL_NAME, device)

    # load data
    dataloader, dataset, tokenizer = load_multi_source_data(
        samples_per_category=args.samples,
        max_length=512,
        batch_size=1,
        seed=42,
        num_workers=8,
    )

    ### Run forward pass on all data and save router logits

    # Attach router observation hooks (same for all modes)
    routing_store = {}
    num_layers = attach_router_hooks(model, routing_store)

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
    logit_dict, global_logits, sequence_lengths = load_logits(
        logit_out_dir, unique_categories
    )

    stats = compute_baseline_stats(logits=global_logits)
    save_baseline_routing_plots(
        stats=stats, out_dir=out_dir / "global", num_experts=args.num_experts
    )

    stats = compute_routing_weights_stats(probs=global_logits, top_k=2)
    save_routing_weight_plots(stats=stats, out_dir=out_dir / "routing")

    analyze_routing_weights_by_category(
        category_data=logit_dict, out_dir=out_dir / "routing_category"
    )

    entropy_stats = analyze_entropy(
        probs=global_logits,
        seq_lengths=sequence_lengths,
        num_experts=args.num_experts,
        k=20,
    )
    save_entropy_analysis_plots(
        entropy_stats=entropy_stats,
        dataset=dataset,
        tokenizer=tokenizer,
        out_dir=out_dir / "entropy" / "global",
        category=None,
    )

    position_stats = analyze_position(
        probs=global_logits, seq_lengths=sequence_lengths, compute_significance=True
    )
    save_position_analysis_plots(
        stats=position_stats, out_dir=out_dir / "position", category=None
    )

    consistency_stats = analyze_cross_layer_consistency(probs=global_logits, top_k=1)
    save_cross_layer_plots(
        stats=consistency_stats, out_dir=out_dir / "cross", category=None
    )


if __name__ == "__main__":
    main()

"""
Training Diagnostics Module

Comprehensive logging for debugging and monitoring training:
- Gradient norms (total and per-layer)
- Weight norms and update ratios
- Activation statistics
- Parameter statistics

Usage:
    diagnostics = TrainingDiagnostics(model, log_interval=100)

    # In training loop:
    diagnostics.log_gradients(model, writer, step)
    diagnostics.log_weights(model, writer, step)
    diagnostics.log_activations(activations, writer, step)
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


@dataclass
class LayerStats:
    """Statistics for a single layer."""

    name: str
    grad_norm: float = 0.0
    weight_norm: float = 0.0
    update_ratio: float = 0.0  # lr * grad_norm / weight_norm
    grad_mean: float = 0.0
    grad_std: float = 0.0
    weight_mean: float = 0.0
    weight_std: float = 0.0


class TrainingDiagnostics:
    """
    Comprehensive training diagnostics for debugging and monitoring.

    Tracks:
    - Gradient statistics (norms, means, stds per layer)
    - Weight statistics (norms, means, stds per layer)
    - Update ratios (how much weights change relative to their magnitude)
    - Activation statistics (if hooks are registered)
    - Dead neuron detection
    """

    def __init__(
        self,
        model: nn.Module,
        log_histograms: bool = False,
        track_activations: bool = False,
    ):
        """
        Args:
            model: The model to track
            log_histograms: Whether to log weight/gradient histograms (expensive)
            track_activations: Whether to register hooks for activation tracking
        """
        self.model = model
        self.log_histograms = log_histograms
        self.track_activations = track_activations

        # Cache layer names for consistent ordering
        self.layer_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.layer_names.append(name)

        # Activation tracking
        self.activation_stats: dict[str, dict] = {}
        self.hooks = []

        if track_activations:
            self._register_activation_hooks()

    def _register_activation_hooks(self):
        """Register forward hooks to capture activation statistics."""

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    with torch.no_grad():
                        self.activation_stats[name] = {
                            "mean": output.mean().item(),
                            "std": output.std().item(),
                            "min": output.min().item(),
                            "max": output.max().item(),
                            "dead_frac": (output == 0).float().mean().item(),
                            "shape": list(output.shape),
                        }

            return hook

        # Register hooks on key layers
        for name, module in self.model.named_modules():
            # Track attention outputs, FFN outputs, and normalization layers
            if any(key in name for key in ["attn", "ffn", "norm", "mlp"]):
                handle = module.register_forward_hook(make_hook(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_gradient_stats(self) -> tuple[float, dict[str, LayerStats]]:
        """
        Compute gradient statistics for all parameters.

        Returns:
            total_grad_norm: L2 norm of all gradients
            layer_stats: Dict mapping layer name to LayerStats
        """
        total_norm_sq = 0.0
        layer_stats = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()

                # Compute norms
                grad_norm = grad.norm(2).item()
                total_norm_sq += grad_norm**2

                # Compute statistics
                stats = LayerStats(name=name)
                stats.grad_norm = grad_norm
                stats.grad_mean = grad.mean().item()
                stats.grad_std = grad.std().item()

                layer_stats[name] = stats

        total_grad_norm = math.sqrt(total_norm_sq)
        return total_grad_norm, layer_stats

    def compute_weight_stats(self, lr: float = 1.0) -> dict[str, LayerStats]:
        """
        Compute weight statistics and update ratios.

        Args:
            lr: Current learning rate (for update ratio calculation)

        Returns:
            layer_stats: Dict mapping layer name to LayerStats
        """
        layer_stats = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight = param.detach()

                stats = LayerStats(name=name)
                stats.weight_norm = weight.norm(2).item()
                stats.weight_mean = weight.mean().item()
                stats.weight_std = weight.std().item()

                # Compute update ratio if gradient exists
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()
                    if stats.weight_norm > 0:
                        stats.update_ratio = lr * grad_norm / stats.weight_norm
                    stats.grad_norm = grad_norm

                layer_stats[name] = stats

        return layer_stats

    def log_gradient_stats(
        self,
        writer: SummaryWriter,
        step: int,
        prefix: str = "gradients",
    ):
        """Log gradient statistics to TensorBoard."""
        total_norm, layer_stats = self.compute_gradient_stats()

        # Log total gradient norm
        writer.add_scalar(f"{prefix}/total_norm", total_norm, step)

        # Group layers by type for cleaner visualization
        layer_groups = defaultdict(list)
        for name, stats in layer_stats.items():
            # Extract layer type (e.g., 'blocks.0.attn.W_Q' -> 'attn.W_Q')
            parts = name.split(".")
            if "blocks" in parts:
                # Remove block index for grouping
                idx = parts.index("blocks")
                layer_type = ".".join(parts[idx + 2 :])  # Skip 'blocks' and index
            else:
                layer_type = name
            layer_groups[layer_type].append((name, stats))

        # Log per-layer-type statistics (averaged across blocks)
        for layer_type, layers in layer_groups.items():
            avg_norm = sum(s.grad_norm for _, s in layers) / len(layers)
            writer.add_scalar(f"{prefix}/by_type/{layer_type}_norm", avg_norm, step)

        # Log individual layer norms (for detailed debugging)
        for name, stats in layer_stats.items():
            safe_name = name.replace(".", "/")
            writer.add_scalar(f"{prefix}/layers/{safe_name}", stats.grad_norm, step)

        # Log histograms if enabled
        if self.log_histograms:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    safe_name = name.replace(".", "/")
                    writer.add_histogram(f"{prefix}/hist/{safe_name}", param.grad, step)

    def log_weight_stats(
        self,
        writer: SummaryWriter,
        step: int,
        lr: float,
        prefix: str = "weights",
    ):
        """Log weight statistics to TensorBoard."""
        layer_stats = self.compute_weight_stats(lr)

        # Compute totals
        total_weight_norm = math.sqrt(sum(s.weight_norm**2 for s in layer_stats.values()))
        avg_update_ratio = sum(s.update_ratio for s in layer_stats.values()) / len(layer_stats)

        writer.add_scalar(f"{prefix}/total_norm", total_weight_norm, step)
        writer.add_scalar(f"{prefix}/avg_update_ratio", avg_update_ratio, step)

        # Group layers by type
        layer_groups = defaultdict(list)
        for name, stats in layer_stats.items():
            parts = name.split(".")
            if "blocks" in parts:
                idx = parts.index("blocks")
                layer_type = ".".join(parts[idx + 2 :])
            else:
                layer_type = name
            layer_groups[layer_type].append((name, stats))

        # Log per-layer-type statistics
        for layer_type, layers in layer_groups.items():
            avg_norm = sum(s.weight_norm for _, s in layers) / len(layers)
            avg_ratio = sum(s.update_ratio for _, s in layers) / len(layers)
            writer.add_scalar(f"{prefix}/by_type/{layer_type}_norm", avg_norm, step)
            writer.add_scalar(f"{prefix}/by_type/{layer_type}_update_ratio", avg_ratio, step)

        # Log histograms if enabled
        if self.log_histograms:
            for name, param in self.model.named_parameters():
                safe_name = name.replace(".", "/")
                writer.add_histogram(f"{prefix}/hist/{safe_name}", param, step)

    def log_activation_stats(
        self,
        writer: SummaryWriter,
        step: int,
        prefix: str = "activations",
    ):
        """Log activation statistics to TensorBoard."""
        if not self.activation_stats:
            return

        for name, stats in self.activation_stats.items():
            safe_name = name.replace(".", "/")
            writer.add_scalar(f"{prefix}/{safe_name}/mean", stats["mean"], step)
            writer.add_scalar(f"{prefix}/{safe_name}/std", stats["std"], step)
            writer.add_scalar(f"{prefix}/{safe_name}/dead_frac", stats["dead_frac"], step)

        # Compute averages across all tracked layers
        if self.activation_stats:
            avg_dead = sum(s["dead_frac"] for s in self.activation_stats.values()) / len(
                self.activation_stats
            )
            writer.add_scalar(f"{prefix}/avg_dead_fraction", avg_dead, step)

        # Clear stats for next forward pass
        self.activation_stats = {}

    def log_all(
        self,
        writer: SummaryWriter,
        step: int,
        lr: float,
    ):
        """Log all diagnostics."""
        self.log_gradient_stats(writer, step)
        self.log_weight_stats(writer, step, lr)
        if self.track_activations:
            self.log_activation_stats(writer, step)

    def get_summary(self, lr: float) -> dict:
        """
        Get a summary dict of key diagnostics for printing.

        Returns dict with:
            - total_grad_norm
            - total_weight_norm
            - avg_update_ratio
            - max_grad_norm_layer
            - min_grad_norm_layer
        """
        total_grad_norm, grad_stats = self.compute_gradient_stats()
        weight_stats = self.compute_weight_stats(lr)

        # Find extreme layers
        max_grad_layer = (
            max(grad_stats.items(), key=lambda x: x[1].grad_norm) if grad_stats else (None, None)
        )
        min_grad_layer = (
            min(grad_stats.items(), key=lambda x: x[1].grad_norm) if grad_stats else (None, None)
        )

        total_weight_norm = math.sqrt(sum(s.weight_norm**2 for s in weight_stats.values()))
        avg_update_ratio = (
            sum(s.update_ratio for s in weight_stats.values()) / len(weight_stats)
            if weight_stats
            else 0
        )

        return {
            "total_grad_norm": total_grad_norm,
            "total_weight_norm": total_weight_norm,
            "avg_update_ratio": avg_update_ratio,
            "max_grad_norm_layer": (max_grad_layer[0], max_grad_layer[1].grad_norm)
            if max_grad_layer[0]
            else None,
            "min_grad_norm_layer": (min_grad_layer[0], min_grad_layer[1].grad_norm)
            if min_grad_layer[0]
            else None,
        }


def detect_anomalies(
    loss: float,
    grad_norm: float,
    prev_loss: float | None = None,
    loss_spike_threshold: float = 2.0,
    grad_explosion_threshold: float = 100.0,
) -> list[str]:
    """
    Detect training anomalies.

    Returns list of warning messages (empty if no anomalies).
    """
    warnings = []

    # Check for NaN/Inf
    if math.isnan(loss) or math.isinf(loss):
        warnings.append(f"CRITICAL: Loss is {loss}")

    if math.isnan(grad_norm) or math.isinf(grad_norm):
        warnings.append(f"CRITICAL: Gradient norm is {grad_norm}")

    # Check for loss spike
    if prev_loss is not None and loss > prev_loss * loss_spike_threshold:
        warnings.append(
            f"WARNING: Loss spiked from {prev_loss:.4f} to {loss:.4f} ({loss / prev_loss:.1f}x)"
        )

    # Check for gradient explosion
    if grad_norm > grad_explosion_threshold:
        warnings.append(f"WARNING: Large gradient norm: {grad_norm:.2f}")

    # Check for vanishing gradients
    if grad_norm < 1e-7:
        warnings.append(f"WARNING: Vanishing gradients: {grad_norm:.2e}")

    return warnings


def print_diagnostics_summary(diagnostics: TrainingDiagnostics, lr: float, step: int):
    """Print a formatted summary of diagnostics."""
    summary = diagnostics.get_summary(lr)

    print(f"\n{'=' * 60}")
    print(f"Diagnostics Summary (Step {step})")
    print(f"{'=' * 60}")
    print(f"  Total Grad Norm:    {summary['total_grad_norm']:.4f}")
    print(f"  Total Weight Norm:  {summary['total_weight_norm']:.4f}")
    print(f"  Avg Update Ratio:   {summary['avg_update_ratio']:.6f}")

    if summary["max_grad_norm_layer"]:
        name, norm = summary["max_grad_norm_layer"]
        print(f"  Max Grad Layer:     {name} ({norm:.4f})")

    if summary["min_grad_norm_layer"]:
        name, norm = summary["min_grad_norm_layer"]
        print(f"  Min Grad Layer:     {name} ({norm:.6f})")

    print(f"{'=' * 60}\n")

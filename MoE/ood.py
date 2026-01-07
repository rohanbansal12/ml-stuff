import argparse
import random
import string
from dataclasses import dataclass

import numpy as np
import torch
from baseline import build_model_and_tokenizer, load_logits

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RoutingBaseline:
    entropy_mean: torch.Tensor  # (num_layers,)
    entropy_std: torch.Tensor  # (num_layers,)
    entropy_p95: torch.Tensor  # (num_layers,)
    entropy_p05: torch.Tensor  # (num_layers,)
    utilization: torch.Tensor  # (num_layers, num_experts)
    primary_weight_mean: torch.Tensor  # (num_layers,)
    primary_weight_std: torch.Tensor  # (num_layers,)
    max_expert_load: torch.Tensor  # (num_layers,)
    num_layers: int
    num_experts: int
    top_k: int

    @classmethod
    def from_probs(
        cls,
        probs: torch.Tensor,
        top_k: int = 2,
        norm_topk_prob: bool = True,
    ) -> "RoutingBaseline":
        """
        Create baseline statistics from routing probabilities.

        Args:
            probs: (num_layers, total_tokens, num_experts) routing probabilities
            top_k: number of experts selected per token
            norm_topk_prob: whether to renormalize top-k weights
        """
        num_layers, total_tokens, num_experts = probs.shape

        # Entropy
        max_entropy = np.log(num_experts)
        token_entropy = -torch.xlogy(probs, probs).sum(dim=-1)
        norm_entropy = token_entropy / max_entropy

        # Expert utilization
        expert_ids = probs.argmax(dim=-1)
        counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
        for layer in range(num_layers):
            counts[layer] = torch.bincount(expert_ids[layer], minlength=num_experts)
        util = counts.float() / total_tokens

        # Top-k weights
        topk_probs, _ = torch.topk(probs, k=top_k, dim=-1)
        if norm_topk_prob:
            topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        else:
            topk_weights = topk_probs
        primary_weight = topk_weights[:, :, 0]

        return cls(
            entropy_mean=norm_entropy.mean(dim=1),
            entropy_std=norm_entropy.std(dim=1),
            entropy_p05=torch.quantile(norm_entropy, 0.05, dim=1),
            entropy_p95=torch.quantile(norm_entropy, 0.95, dim=1),
            utilization=util,
            primary_weight_mean=primary_weight.mean(dim=1),
            primary_weight_std=primary_weight.std(dim=1),
            max_expert_load=util.max(dim=1).values,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
        )

    def score_routing(
        self,
        new_probs: torch.Tensor,
        norm_topk_prob: bool = True,
    ) -> dict:
        """
        Score how anomalous new routing patterns are compared to baseline.

        Args:
            new_probs: (num_layers, total_tokens, num_experts) routing probabilities

        Returns:
            Dictionary with comparison metrics and anomaly flags
        """
        num_layers, total_tokens, num_experts = new_probs.shape

        assert num_layers == self.num_layers, f"Layer mismatch: {num_layers} vs {self.num_layers}"
        assert (
            num_experts == self.num_experts
        ), f"Expert mismatch: {num_experts} vs {self.num_experts}"

        results = {}

        # ============================
        # 1. ENTROPY ANALYSIS
        # ============================
        max_entropy = np.log(num_experts)
        token_entropy = -torch.xlogy(new_probs, new_probs).sum(dim=-1)
        norm_entropy = token_entropy / max_entropy

        new_entropy_mean = norm_entropy.mean(dim=1)
        new_entropy_std = norm_entropy.std(dim=1)

        # Z-scores relative to baseline
        entropy_zscore = (new_entropy_mean - self.entropy_mean) / (self.entropy_std + 1e-8)

        # Fraction of tokens outside baseline percentiles
        below_p05 = (norm_entropy < self.entropy_p05.unsqueeze(1)).float().mean(dim=1)
        above_p95 = (norm_entropy > self.entropy_p95.unsqueeze(1)).float().mean(dim=1)

        results["entropy"] = {
            "mean": new_entropy_mean,
            "std": new_entropy_std,
            "zscore": entropy_zscore,
            "frac_below_p05": below_p05,
            "frac_above_p95": above_p95,
            "mean_diff": new_entropy_mean - self.entropy_mean,
        }

        # ============================
        # 2. UTILIZATION ANALYSIS
        # ============================
        expert_ids = new_probs.argmax(dim=-1)
        counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
        for layer in range(num_layers):
            counts[layer] = torch.bincount(expert_ids[layer], minlength=num_experts)
        new_util = counts.float() / total_tokens

        # KL divergence from baseline utilization
        # KL(new || baseline) = sum(new * log(new / baseline))
        util_kl = torch.zeros(num_layers)
        for layer in range(num_layers):
            p = new_util[layer]
            q = self.utilization[layer]
            # Add smoothing to avoid log(0)
            p_smooth = (p + 1e-8) / (p + 1e-8).sum()
            q_smooth = (q + 1e-8) / (q + 1e-8).sum()
            kl = (p_smooth * torch.log(p_smooth / q_smooth)).sum()
            util_kl[layer] = kl

        # L2 distance from baseline utilization
        util_l2 = ((new_util - self.utilization) ** 2).sum(dim=1).sqrt()

        # Max expert load
        new_max_load = new_util.max(dim=1).values
        max_load_ratio = new_max_load / (self.max_expert_load + 1e-8)

        # Gini coefficient (inequality measure)
        def compute_gini(util_tensor):
            """Compute Gini coefficient per layer."""
            gini = torch.zeros(util_tensor.shape[0])
            for layer in range(util_tensor.shape[0]):
                sorted_util = util_tensor[layer].sort().values
                n = len(sorted_util)
                index = torch.arange(1, n + 1, dtype=torch.float)
                gini[layer] = (2 * (index @ sorted_util) - (n + 1) * sorted_util.sum()) / (
                    n * sorted_util.sum() + 1e-8
                )
            return gini

        new_gini = compute_gini(new_util)
        baseline_gini = compute_gini(self.utilization)

        # Effective number of experts (entropy-based)
        def compute_effective_experts(util_tensor):
            effective = torch.zeros(util_tensor.shape[0])
            for layer in range(util_tensor.shape[0]):
                u = util_tensor[layer]
                u_nonzero = u[u > 0]
                if len(u_nonzero) > 0:
                    entropy = -(u_nonzero * torch.log(u_nonzero)).sum()
                    effective[layer] = torch.exp(entropy)
                else:
                    effective[layer] = 0
            return effective

        new_effective = compute_effective_experts(new_util)
        baseline_effective = compute_effective_experts(self.utilization)

        results["utilization"] = {
            "distribution": new_util,
            "kl_divergence": util_kl,
            "l2_distance": util_l2,
            "max_load": new_max_load,
            "max_load_ratio_vs_baseline": max_load_ratio,
            "gini": new_gini,
            "gini_diff": new_gini - baseline_gini,
            "effective_experts": new_effective,
            "effective_experts_diff": new_effective - baseline_effective,
        }

        # ============================
        # 3. PRIMARY WEIGHT ANALYSIS
        # ============================
        topk_probs, _ = torch.topk(new_probs, k=self.top_k, dim=-1)
        if norm_topk_prob:
            topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        else:
            topk_weights = topk_probs
        primary_weight = topk_weights[:, :, 0]

        new_pw_mean = primary_weight.mean(dim=1)
        new_pw_std = primary_weight.std(dim=1)

        pw_zscore = (new_pw_mean - self.primary_weight_mean) / (self.primary_weight_std + 1e-8)

        results["primary_weight"] = {
            "mean": new_pw_mean,
            "std": new_pw_std,
            "zscore": pw_zscore,
            "mean_diff": new_pw_mean - self.primary_weight_mean,
        }

        # ============================
        # 4. COLLAPSE DETECTION
        # ============================
        collapse_threshold = 0.3  # >30% to single expert is concerning
        severe_collapse_threshold = 0.5  # >50% is severe

        collapsed_layers = new_max_load > collapse_threshold
        severe_collapsed_layers = new_max_load > severe_collapse_threshold

        # Find dominant expert per layer
        dominant_experts = new_util.argmax(dim=1)

        results["collapse"] = {
            "collapsed_layers": collapsed_layers,
            "num_collapsed": collapsed_layers.sum().item(),
            "severe_collapsed_layers": severe_collapsed_layers,
            "num_severe_collapsed": severe_collapsed_layers.sum().item(),
            "dominant_experts": dominant_experts,
            "max_load_per_layer": new_max_load,
        }

        # ============================
        # 5. ANOMALY FLAGS
        # ============================
        results["anomaly_flags"] = {
            "high_entropy": (entropy_zscore > 2.0).any().item(),
            "low_entropy": (entropy_zscore < -2.0).any().item(),
            "utilization_shift": (util_kl > 0.1).any().item(),  # KL > 0.1 is notable
            "any_collapse": collapsed_layers.any().item(),
            "severe_collapse": severe_collapsed_layers.any().item(),
            "decisiveness_shift": (pw_zscore.abs() > 2.0).any().item(),
        }

        # Overall anomaly score (simple aggregation)
        anomaly_score = (
            entropy_zscore.abs().mean()
            + util_kl.mean() * 10  # Scale KL to be comparable
            + pw_zscore.abs().mean()
            + collapsed_layers.float().mean() * 5  # Penalize collapse heavily
        )
        results["anomaly_score"] = anomaly_score.item()

        # ============================
        # 6. SUMMARY STATISTICS
        # ============================
        results["summary"] = {
            "total_tokens": total_tokens,
            "entropy_mean_change": (new_entropy_mean - self.entropy_mean).mean().item(),
            "utilization_kl_mean": util_kl.mean().item(),
            "primary_weight_change": (new_pw_mean - self.primary_weight_mean).mean().item(),
            "max_load_mean": new_max_load.mean().item(),
            "effective_experts_mean": new_effective.mean().item(),
        }

        return results


def print_routing_comparison(results: dict, baseline: RoutingBaseline):
    """Pretty print the routing comparison results."""

    print("=" * 60)
    print("ROUTING COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\nOverall Anomaly Score: {results['anomaly_score']:.4f}")
    print("  (higher = more anomalous)")

    print(f"\nTokens Analyzed: {results['summary']['total_tokens']:,}")

    # Anomaly flags
    print("\n--- Anomaly Flags ---")
    for flag, value in results["anomaly_flags"].items():
        status = "⚠️  YES" if value else "✓  No"
        print(f"  {flag}: {status}")

    # Entropy
    print("\n--- Entropy ---")
    print(f"  Mean change: {results['summary']['entropy_mean_change']:+.4f}")
    print(
        f"  Z-scores by layer: min={results['entropy']['zscore'].min():.2f}, max={results['entropy']['zscore'].max():.2f}"
    )

    # Utilization
    print("\n--- Utilization ---")
    print(f"  KL divergence (mean): {results['summary']['utilization_kl_mean']:.4f}")
    print(
        f"  Effective experts (mean): {results['summary']['effective_experts_mean']:.1f} / {baseline.num_experts}"
    )
    print(f"  Max load (mean): {results['summary']['max_load_mean']:.4f}")

    # Collapse
    print("\n--- Collapse Detection ---")
    print(
        f"  Collapsed layers (>30%): {results['collapse']['num_collapsed']} / {baseline.num_layers}"
    )
    print(
        f"  Severe collapse (>50%): {results['collapse']['num_severe_collapsed']} / {baseline.num_layers}"
    )

    if results["collapse"]["num_collapsed"] > 0:
        collapsed_idx = results["collapse"]["collapsed_layers"].nonzero().squeeze(-1).tolist()
        if isinstance(collapsed_idx, int):
            collapsed_idx = [collapsed_idx]
        print(f"  Collapsed layer indices: {collapsed_idx}")
        print(
            f"  Dominant experts: {results['collapse']['dominant_experts'][collapsed_idx].tolist()}"
        )

    # Primary weight
    print("\n--- Routing Decisiveness ---")
    print(f"  Primary weight change: {results['summary']['primary_weight_change']:+.4f}")

    print("\n" + "=" * 60)


def generate_ood_inputs(tokenizer, num_samples=100, seq_len=256) -> dict:
    """Generate various OOD input types."""

    inputs = {}

    # 1. Repetition-based
    inputs["single_token_repeat"] = [
        tokenizer.encode("the " * seq_len)[:seq_len] for _ in range(num_samples)
    ]
    inputs["phrase_repeat"] = [
        tokenizer.encode("the quick brown fox " * (seq_len // 5))[:seq_len]
        for _ in range(num_samples)
    ]

    # 2. Random/noise
    inputs["random_tokens"] = [
        torch.randint(0, tokenizer.vocab_size, (seq_len,)).tolist() for _ in range(num_samples)
    ]
    inputs["random_ascii"] = [
        tokenizer.encode("".join(random.choices(string.printable, k=seq_len * 4)))[:seq_len]
        for _ in range(num_samples)
    ]

    # 3. Adversarial-ish
    inputs["all_punctuation"] = [
        tokenizer.encode(".,!?;:" * seq_len)[:seq_len] for _ in range(num_samples)
    ]
    inputs["all_numbers"] = [
        tokenizer.encode("0123456789 " * seq_len)[:seq_len] for _ in range(num_samples)
    ]
    inputs["rare_unicode"] = [
        tokenizer.encode("∀∃∄∅∆∇∈∉∊∋∌∍∎∏" * seq_len)[:seq_len] for _ in range(num_samples)
    ]

    # 4. Structure-based
    inputs["empty_padding"] = [[tokenizer.pad_token_id] * seq_len for _ in range(num_samples)]
    inputs["bos_only"] = [[tokenizer.bos_token_id] * seq_len for _ in range(num_samples)]

    # 5. Language shift
    inputs["code_heavy"] = [...]  # Dense code snippets
    inputs["math_symbols"] = [...]  # LaTeX-style math
    inputs["foreign_script"] = [...]  # Chinese, Arabic, etc.

    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logit_dir", type=str, default="/workspace/ml-stuff/MoE/out/baseline/logits"
    )
    parser.add_argument("--num_experts", type=int, default=60)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    logit_dict, global_probs, sequence_lengths = load_logits(args.logit_dir)
    baseline = RoutingBaseline.from_probs(global_probs, top_k=2)

    model, tokenizer = build_model_and_tokenizer(MODEL_NAME, device)


if __name__ == "__main__":
    main()

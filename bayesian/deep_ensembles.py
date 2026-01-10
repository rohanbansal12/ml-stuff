"""
Deep Ensembles - Simple and Strong Baseline for Uncertainty

Paper: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
       (Lakshminarayanan et al., 2017)

Key idea:
    Train M independent networks from different random initializations.
    At test time, average their predictions.
    Disagreement between networks = uncertainty.

Why it works:
    - Different random inits find different local minima
    - These represent different "hypotheses" about the data
    - Regions where all networks agree = confident
    - Regions where networks disagree = uncertain

Not strictly Bayesian but:
    - Often outperforms variational methods in practice
    - Simple to implement and parallelize
    - No special architecture needed

Extensions:
    - Each network predicts (mean, variance) for aleatoric uncertainty
    - Mixture of Gaussians predictive distribution

Drawbacks:
    - M times the compute and memory
    - Typically M=5 is used
    - Not a proper posterior (but who cares if it works?)

Comparison to MC Dropout:
    - Ensembles: Diversity from random init and SGD stochasticity
    - MC Dropout: Diversity from dropout masks
    - Ensembles often give better calibrated uncertainty

Exercises:
    1. Train M=5 networks independently
    2. Combine predictions (mean of means, variance of means for uncertainty)
    3. Compare to MC Dropout on same task
    4. Evaluate calibration using reliability diagrams
"""


import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
import bayesian.mc_dropout as mc


class EnsembleMember(nn.Module):
    """Single network in the ensemble.

    Can output just mean (classification/simple regression)
    or (mean, log_var) for heteroscedastic regression.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 heteroscedastic: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.output_dim = output_dim

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if heteroscedastic:
            # Separate heads for mean and log_var
            self.mean_head = nn.Linear(hidden_dim, output_dim)
            self.log_var_head = nn.Linear(hidden_dim, output_dim)
        else:
            # Single output head
            self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            For classification/simple: logits or mean, shape (batch, output_dim)
            For heteroscedastic: (mean, log_var) tuple, each shape (batch, output_dim)
        """
        # Flatten input if needed (for image data)
        x = x.view(x.size(0), -1)
        features = self.backbone(x)

        if self.heteroscedastic:
            mean = self.mean_head(features)
            log_var = self.log_var_head(features)
            return mean, log_var
        else:
            return self.output_head(features)


class DeepEnsemble(nn.Module):
    """Ensemble of M independent networks."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_members: int = 5, heteroscedastic: bool = False):
        super().__init__()
        self.n_members = n_members
        self.heteroscedastic = heteroscedastic
        self.output_dim = output_dim

        # Create nn.ModuleList of EnsembleMember networks
        # Each member has different random initialization (automatic with nn.Module)
        self.members = nn.ModuleList([
            EnsembleMember(input_dim, hidden_dim, output_dim, heteroscedastic)
            for _ in range(n_members)
        ])

    def forward_member(self, x: torch.Tensor, member_idx: int):
        """Forward pass through a single member."""
        return self.members[member_idx](x)

    def forward(self, x: torch.Tensor):
        """Forward pass through all members, returns list of outputs."""
        return [member(x) for member in self.members]

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor):
        """Get predictions from all members and compute statistics.

        For classification:
            - Average the softmax probabilities
            - Uncertainty = entropy of average, or variance of predictions

        For regression:
            - pred_mean = mean of member means
            - epistemic = variance of member means
            - aleatoric = mean of member variances (if heteroscedastic)

        Returns vary based on task, but generally:
            mean: Ensemble mean prediction
            uncertainty: Disagreement between members
        """
        if self.heteroscedastic:
            # Collect means and variances from all members
            means = []
            variances = []
            for member in self.members:
                mean, log_var = member(x)
                means.append(mean)
                variances.append(torch.exp(log_var))

            means = torch.stack(means, dim=0)  # (n_members, batch, output_dim)
            variances = torch.stack(variances, dim=0)

            # Ensemble predictions
            pred_mean = means.mean(dim=0)  # Mean of means
            epistemic_var = means.var(dim=0)  # Variance of means (disagreement)
            aleatoric_var = variances.mean(dim=0)  # Mean of variances (data noise)

            return pred_mean, epistemic_var, aleatoric_var
        else:
            # Classification or simple regression
            outputs = []
            for member in self.members:
                outputs.append(member(x))

            outputs = torch.stack(outputs, dim=0)  # (n_members, batch, output_dim)

            # For classification: average softmax probabilities
            if self.output_dim > 1:
                probs = torch.softmax(outputs, dim=-1)
                mean_probs = probs.mean(dim=0)
                # Uncertainty: entropy of mean prediction
                entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
                # Alternative: variance of predictions
                pred_var = probs.var(dim=0).mean(dim=-1)
                return mean_probs, entropy, pred_var
            else:
                # Simple regression
                pred_mean = outputs.mean(dim=0)
                pred_var = outputs.var(dim=0)
                return pred_mean, pred_var


def train_ensemble(ensemble: DeepEnsemble, train_loader: DataLoader,
                   n_epochs: int, device: torch.device, lr: float = 1e-3):
    """Train each ensemble member independently.

    Key: Each member sees shuffled data in different order due to DataLoader.
    For even more diversity, can use different data subsets (bootstrap).
    """
    import torch.nn.functional as F

    ensemble.to(device)

    # Create separate optimizer for each member
    optimizers = [
        torch.optim.Adam(member.parameters(), lr=lr)
        for member in ensemble.members
    ]

    # Determine loss function based on task type
    if ensemble.heteroscedastic:
        def compute_loss(output, target):
            mean, log_var = output
            # Gaussian NLL loss
            return (0.5 * (log_var + (target - mean).pow(2) * torch.exp(-log_var))).mean()
    else:
        if ensemble.output_dim > 1:
            # Classification
            def compute_loss(output, target):
                return F.cross_entropy(output, target)
        else:
            # Simple regression
            def compute_loss(output, target):
                return F.mse_loss(output, target)

    # Train each member independently
    for member_idx, (member, optimizer) in enumerate(zip(ensemble.members, optimizers, strict=True)):
        print(f"\nTraining member {member_idx + 1}/{ensemble.n_members}")

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                output = member(inputs)
                loss = compute_loss(output, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")


def calibration_metrics(predictions: torch.Tensor, confidences: torch.Tensor,
                        labels: torch.Tensor, n_bins: int = 10):
    """Compute calibration metrics.

    A well-calibrated model:
        - When it says 80% confident, it should be right 80% of the time

    Args:
        predictions: Predicted class labels, shape (n_samples,)
        confidences: Confidence scores (max probability), shape (n_samples,)
        labels: True labels, shape (n_samples,)
        n_bins: Number of confidence bins

    Returns:
        ece: Expected Calibration Error
        bin_accuracies: Accuracy in each confidence bin
        bin_confidences: Average confidence in each bin
        bin_counts: Number of samples in each bin
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    correct = (predictions == labels).float()

    for i in range(n_bins):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_count = in_bin.sum().item()
        bin_counts.append(bin_count)

        if bin_count > 0:
            # Accuracy: fraction of correct predictions in this bin
            bin_acc = correct[in_bin].mean().item()
            # Average confidence in this bin
            bin_conf = confidences[in_bin].mean().item()
        else:
            bin_acc = 0.0
            bin_conf = (bin_boundaries[i] + bin_boundaries[i + 1]).item() / 2

        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)

    # ECE = weighted average of |accuracy - confidence| per bin
    total_samples = sum(bin_counts)
    ece = sum(
        (bin_counts[i] / total_samples) * abs(bin_accuracies[i] - bin_confidences[i])
        for i in range(n_bins)
        if bin_counts[i] > 0
    )

    return ece, bin_accuracies, bin_confidences, bin_counts


def reliability_diagram(bin_accuracies, bin_confidences, bin_counts=None,
                        n_bins: int = 10, title: str = "Reliability Diagram"):
    """Plot reliability diagram (calibration plot).

    Args:
        bin_accuracies: Accuracy in each confidence bin
        bin_confidences: Average confidence in each bin
        bin_counts: Optional counts per bin (for second subplot)
        n_bins: Number of bins (for bar width)
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    # Convert to lists if tensors
    if isinstance(bin_accuracies, torch.Tensor):
        bin_accuracies = bin_accuracies.tolist()
    if isinstance(bin_confidences, torch.Tensor):
        bin_confidences = bin_confidences.tolist()

    bin_centers = [(i + 0.5) / n_bins for i in range(n_bins)]
    width = 1.0 / n_bins

    if bin_counts is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Bar plot of accuracy vs confidence bins
    ax.bar(bin_centers, bin_accuracies, width=width, alpha=0.7,
           edgecolor="black", label="Accuracy")

    # Diagonal line for perfect calibration
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")

    # Gap visualization (below diagonal = overconfident)
    for i, (center, acc, conf) in enumerate(zip(bin_centers, bin_accuracies, bin_confidences, strict=True)):
        if acc < conf:
            ax.bar(center, conf - acc, bottom=acc, width=width,
                   alpha=0.3, color="red", edgecolor="none")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")

    # Compute and display ECE
    if bin_counts is not None:
        total = sum(bin_counts)
        ece = sum(
            (bin_counts[i] / total) * abs(bin_accuracies[i] - bin_confidences[i])
            for i in range(n_bins)
            if bin_counts[i] > 0
        )
        ax.text(0.05, 0.92, f"ECE = {ece:.4f}", transform=ax.transAxes, fontsize=11)

        # Second subplot: histogram of counts
        axes[1].bar(bin_centers, bin_counts, width=width, alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Samples per Confidence Bin")

    plt.tight_layout()
    return fig


@torch.no_grad()
def evaluate_ensemble(ensemble, loader, device):
    """Evaluate ensemble and return predictions, confidences, labels, and entropy."""
    all_probs = []
    all_entropy = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        mean_probs, entropy, _ = ensemble.predict_with_uncertainty(inputs)
        all_probs.append(mean_probs.cpu())
        all_entropy.append(entropy.cpu())
        all_labels.append(labels)

    probs = torch.cat(all_probs)
    entropy = torch.cat(all_entropy)
    labels = torch.cat(all_labels)

    confidences, predictions = probs.max(dim=1)
    accuracy = (predictions == labels).float().mean().item()

    return predictions, confidences, labels, entropy, accuracy


@torch.no_grad()
def evaluate_mc_dropout(model, loader, device, n_samples=50):
    """Evaluate MC Dropout model and return predictions, confidences, labels, and entropy."""
    import torch.nn.functional as F

    all_probs = []
    all_entropy = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        mean, std = model.predict_with_uncertainty(inputs, n_samples=n_samples)
        probs = F.softmax(mean, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        all_probs.append(probs.cpu())
        all_entropy.append(entropy.cpu())
        all_labels.append(labels)

    probs = torch.cat(all_probs)
    entropy = torch.cat(all_entropy)
    labels = torch.cat(all_labels)

    confidences, predictions = probs.max(dim=1)
    accuracy = (predictions == labels).float().mean().item()

    return predictions, confidences, labels, entropy, accuracy


def main():
    """Train ensemble and evaluate uncertainty quality."""
    import os

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = mc.get_dataloaders(batch_size=128)

    # ========== Train Deep Ensemble ==========
    print("\n" + "=" * 50)
    print("Training Deep Ensemble (5 members)")
    print("=" * 50)
    ensemble = DeepEnsemble(input_dim=28*28, hidden_dim=256, output_dim=10, n_members=5).to(device)
    train_ensemble(ensemble, train_loader=train_loader, n_epochs=5, device=device)

    # ========== Train MC Dropout ==========
    print("\n" + "=" * 50)
    print("Training MC Dropout Model")
    print("=" * 50)
    mc_model = mc.MCDropoutMLP(
        input_dim=28 * 28,
        hidden_dim=256,
        output_dim=10,
        dropout_p=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(mc_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        mc.train_one_epoch(mc_model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/5 complete")

    # ========== Evaluate on In-Distribution (MNIST Test) ==========
    print("\n" + "=" * 50)
    print("Evaluating on In-Distribution (MNIST Test)")
    print("=" * 50)

    # Ensemble evaluation
    ens_preds, ens_confs, ens_labels, ens_entropy, ens_acc = evaluate_ensemble(
        ensemble, test_loader, device
    )
    print("\nDeep Ensemble:")
    print(f"  Accuracy: {ens_acc:.4f}")
    print(f"  Mean Confidence: {ens_confs.mean():.4f}")
    print(f"  Mean Entropy: {ens_entropy.mean():.4f}")

    # MC Dropout evaluation
    mc_preds, mc_confs, mc_labels, mc_entropy, mc_acc = evaluate_mc_dropout(
        mc_model, test_loader, device, n_samples=50
    )
    print("\nMC Dropout:")
    print(f"  Accuracy: {mc_acc:.4f}")
    print(f"  Mean Confidence: {mc_confs.mean():.4f}")
    print(f"  Mean Entropy: {mc_entropy.mean():.4f}")

    # ========== Calibration Comparison ==========
    print("\n" + "=" * 50)
    print("Calibration Comparison")
    print("=" * 50)

    ens_ece, ens_bin_accs, ens_bin_confs, ens_bin_counts = calibration_metrics(
        ens_preds, ens_confs, ens_labels
    )
    mc_ece, mc_bin_accs, mc_bin_confs, mc_bin_counts = calibration_metrics(
        mc_preds, mc_confs, mc_labels
    )

    print(f"\nDeep Ensemble ECE: {ens_ece:.4f}")
    print(f"MC Dropout ECE: {mc_ece:.4f}")

    # ========== OOD Detection ==========
    print("\n" + "=" * 50)
    print("OOD Detection (FashionMNIST)")
    print("=" * 50)

    ood_loader = mc.get_ood_loader("fashionmnist", batch_size=128)

    # Ensemble on OOD
    _, ens_ood_confs, _, ens_ood_entropy, _ = evaluate_ensemble(ensemble, ood_loader, device)
    print("\nDeep Ensemble on OOD:")
    print(f"  Mean Confidence: {ens_ood_confs.mean():.4f} (should be lower)")
    print(f"  Mean Entropy: {ens_ood_entropy.mean():.4f} (should be higher)")

    # MC Dropout on OOD
    _, mc_ood_confs, _, mc_ood_entropy, _ = evaluate_mc_dropout(mc_model, ood_loader, device)
    print("\nMC Dropout on OOD:")
    print(f"  Mean Confidence: {mc_ood_confs.mean():.4f} (should be lower)")
    print(f"  Mean Entropy: {mc_ood_entropy.mean():.4f} (should be higher)")

    # ========== Visualizations ==========
    os.makedirs("./plots/deep_ensembles", exist_ok=True)

    # 1. Reliability diagrams comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ensemble reliability diagram
    ax = axes[0]
    bin_centers = [(i + 0.5) / 10 for i in range(10)]
    width = 0.1
    ax.bar(bin_centers, ens_bin_accs, width=width, alpha=0.7, edgecolor="black")
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Deep Ensemble (ECE={ens_ece:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    # MC Dropout reliability diagram
    ax = axes[1]
    ax.bar(bin_centers, mc_bin_accs, width=width, alpha=0.7, edgecolor="black")
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"MC Dropout (ECE={mc_ece:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig("./plots/deep_ensembles/calibration_comparison.png", dpi=150)
    print("\nSaved: ./plots/deep_ensembles/calibration_comparison.png")

    # 2. Uncertainty histogram (ID vs OOD)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ensemble entropy histogram
    ax = axes[0]
    ax.hist(ens_entropy.numpy(), bins=50, alpha=0.7, label="In-distribution", density=True)
    ax.hist(ens_ood_entropy.numpy(), bins=50, alpha=0.7, label="OOD (FashionMNIST)", density=True)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Density")
    ax.set_title("Deep Ensemble: Entropy Distribution")
    ax.legend()

    # MC Dropout entropy histogram
    ax = axes[1]
    ax.hist(mc_entropy.numpy(), bins=50, alpha=0.7, label="In-distribution", density=True)
    ax.hist(mc_ood_entropy.numpy(), bins=50, alpha=0.7, label="OOD (FashionMNIST)", density=True)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Density")
    ax.set_title("MC Dropout: Entropy Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig("./plots/deep_ensembles/ood_detection_comparison.png", dpi=150)
    print("Saved: ./plots/deep_ensembles/ood_detection_comparison.png")

    # 3. Summary comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    methods = ["Deep Ensemble", "MC Dropout"]

    # Accuracy
    axes[0].bar(methods, [ens_acc, mc_acc], color=["steelblue", "coral"])
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylim(0.9, 1.0)

    # ECE (lower is better)
    axes[1].bar(methods, [ens_ece, mc_ece], color=["steelblue", "coral"])
    axes[1].set_ylabel("ECE")
    axes[1].set_title("Calibration Error (lower=better)")

    # OOD entropy increase
    ens_entropy_increase = ens_ood_entropy.mean().item() - ens_entropy.mean().item()
    mc_entropy_increase = mc_ood_entropy.mean().item() - mc_entropy.mean().item()
    axes[2].bar(methods, [ens_entropy_increase, mc_entropy_increase], color=["steelblue", "coral"])
    axes[2].set_ylabel("Entropy Increase")
    axes[2].set_title("OOD Detection (higher=better)")

    plt.tight_layout()
    plt.savefig("./plots/deep_ensembles/summary_comparison.png", dpi=150)
    print("Saved: ./plots/deep_ensembles/summary_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()

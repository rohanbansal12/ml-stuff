"""
MC Dropout - Dropout as Approximate Bayesian Inference

Key insight (Gal & Ghahramani, 2016):
    Dropout at test time approximates sampling from the posterior.
    Each forward pass with dropout = one sample from approximate posterior.

Why it works (hand-wavy):
    - Dropout randomly zeros weights, creating an ensemble of "thinned" networks
    - This can be shown to approximate variational inference with Bernoulli prior
    - Multiple forward passes = Monte Carlo integration over weight uncertainty

Practical benefits:
    - Zero extra parameters (just use dropout you already have)
    - Easy to implement (just keep dropout on at test time)
    - Works with any architecture

Limitations:
    - Uncertainty estimates can be miscalibrated
    - Requires multiple forward passes at test time
    - Approximation quality depends on dropout rate

Exercises:
    1. Train a standard network with dropout on MNIST/CIFAR
    2. At test time, run multiple forward passes with dropout ON
    3. Use mean as prediction, std as uncertainty
    4. Verify: uncertainty should be higher on OOD data (e.g., SVHN for CIFAR-trained model)
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropoutMLP(nn.Module):
    """MLP with dropout that stays on during inference."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout always applied (even in eval mode)."""
        x = x.view(x.size(0), -1)  # Flatten: (batch, C, H, W) -> (batch, C*H*W)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_p, training=True)
        x = F.dropout(F.relu(self.fc2(x)), p=self.dropout_p, training=True)
        return self.fc3(x)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Run multiple forward passes and compute mean/std.

        Args:
            x: Input tensor
            n_samples: Number of stochastic forward passes

        Returns:
            mean: Mean prediction across samples
            std: Standard deviation (uncertainty) across samples
        """
        outputs = []
        for _ in range(n_samples):
            outputs.append(self.forward(x))

        outputs = torch.stack(outputs, dim=0)
        mean = torch.mean(outputs, dim=0)
        std = torch.std(outputs, dim=0)

        return mean, std


class MCDropoutCNN(nn.Module):
    """CNN with spatial dropout for image classification."""

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.1, in_channels: int = 1):
        """
        Args:
            num_classes: Number of output classes
            dropout_p: Dropout probability
            in_channels: Number of input channels (1 for MNIST, 3 for CIFAR)
        """
        super().__init__()
        self.dropout_p = dropout_p

        # Conv blocks
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # FC layers - input size depends on image size after pooling
        # For 28x28 (MNIST): 28 -> 14 -> 7 -> 3, so 128 * 3 * 3 = 1152
        # For 32x32 (CIFAR): 32 -> 16 -> 8 -> 4, so 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout always applied (even in eval mode)."""
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        x = F.dropout2d(x, p=self.dropout_p, training=True)

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        x = F.dropout2d(x, p=self.dropout_p, training=True)

        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))
        x = F.dropout2d(x, p=self.dropout_p, training=True)

        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_p, training=True)
        return self.fc2(x)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Run multiple forward passes and compute mean/std.

        Args:
            x: Input tensor (batch of images)
            n_samples: Number of stochastic forward passes

        Returns:
            mean: Mean prediction across samples
            std: Standard deviation (uncertainty) across samples
        """
        outputs = []
        for _ in range(n_samples):
            outputs.append(self.forward(x))

        outputs = torch.stack(outputs, dim=0)
        mean = torch.mean(outputs, dim=0)
        std = torch.std(outputs, dim=0)

        return mean, std


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Standard training loop.

    Returns:
        avg_loss: Average loss over the epoch
        accuracy: Classification accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_with_uncertainty(model, loader, device, n_samples: int = 50):
    """Evaluate with MC Dropout uncertainty estimates.

    Returns:
        accuracy: Classification accuracy
        mean_confidence: Average max probability (should be lower when uncertain)
        mean_entropy: Average predictive entropy (should be higher when uncertain)
    """
    total = 0
    correct = 0
    confidence = 0.0
    entropy = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mean, std = model.predict_with_uncertainty(inputs, n_samples)

        probs = F.softmax(mean, dim=1)
        max_vals, predicted = probs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        confidence += max_vals.sum().item()
        # Entropy per sample, then sum over batch
        entropy += -(probs * torch.log(probs + 1e-8)).sum(dim=1).sum().item()

    accuracy = correct / total
    mean_confidence = confidence / total
    mean_entropy = entropy / total

    return accuracy, mean_confidence, mean_entropy



def get_dataloaders(batch_size: int = 128):
    """Get MNIST train and test loaders."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=12
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=12
    )

    return train_loader, test_loader


def get_ood_loader(ood_type: str, batch_size: int = 128):
    """Get OOD data loader for evaluation."""
    from torchvision import datasets, transforms

    if ood_type == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Use same normalization as MNIST
        ])
        ood_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset, batch_size=batch_size, shuffle=False, num_workers=12
        )
    elif ood_type == "noise":
        # Create random noise dataset with same shape as MNIST
        noise_data = torch.randn(10000, 1, 28, 28)
        noise_targets = torch.zeros(10000, dtype=torch.long)  # Dummy targets
        ood_dataset = torch.utils.data.TensorDataset(noise_data, noise_targets)
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset, batch_size=batch_size, shuffle=False
        )
    else:
        raise ValueError(f"Unknown OOD type: {ood_type}")

    return ood_loader


@torch.no_grad()
def collect_predictions(model, loader, device, n_samples: int = 50):
    """Collect predictions, uncertainties, and targets for visualization."""
    all_images = []
    all_probs = []
    all_stds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        mean, std = model.predict_with_uncertainty(inputs, n_samples)

        probs = F.softmax(mean, dim=1)
        all_images.append(inputs.cpu())
        all_probs.append(probs.cpu())
        all_stds.append(std.cpu())
        all_targets.append(targets)

    return (
        torch.cat(all_images),
        torch.cat(all_probs),
        torch.cat(all_stds),
        torch.cat(all_targets),
    )


def plot_examples_with_uncertainty(images, probs, targets, n_examples: int = 10):
    """Plot test examples with predicted class and uncertainty."""
    fig, axes = plt.subplots(2, n_examples, figsize=(2 * n_examples, 5))

    # Get predictions and entropy (uncertainty)
    max_probs, preds = probs.max(dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

    # Select random examples
    indices = torch.randperm(len(images))[:n_examples]

    for i, idx in enumerate(indices):
        # Plot image
        img = images[idx].squeeze().numpy()
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].axis("off")

        correct = preds[idx] == targets[idx]
        color = "green" if correct else "red"
        axes[0, i].set_title(f"Pred: {preds[idx].item()}", color=color, fontsize=10)

        # Plot probability distribution
        axes[1, i].bar(range(10), probs[idx].numpy(), color="steelblue")
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_xlabel(f"H={entropy[idx]:.2f}", fontsize=9)
        if i == 0:
            axes[1, i].set_ylabel("Probability")

    plt.suptitle("Predictions with Uncertainty (H = entropy)", fontsize=12)
    plt.tight_layout()
    return fig


def plot_uncertainty_histogram(id_probs, ood_probs, ood_name: str):
    """Plot histogram of uncertainties for in-distribution vs OOD."""
    # Compute entropy for both distributions
    id_entropy = -(id_probs * torch.log(id_probs + 1e-8)).sum(dim=1).numpy()
    ood_entropy = -(ood_probs * torch.log(ood_probs + 1e-8)).sum(dim=1).numpy()

    # Compute max probability (confidence)
    id_conf = id_probs.max(dim=1).values.numpy()
    ood_conf = ood_probs.max(dim=1).values.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Entropy histogram
    axes[0].hist(id_entropy, bins=50, alpha=0.7, label="In-distribution (MNIST)", density=True)
    axes[0].hist(ood_entropy, bins=50, alpha=0.7, label=f"OOD ({ood_name})", density=True)
    axes[0].set_xlabel("Predictive Entropy")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Entropy Distribution (higher = more uncertain)")
    axes[0].legend()

    # Confidence histogram
    axes[1].hist(id_conf, bins=50, alpha=0.7, label="In-distribution (MNIST)", density=True)
    axes[1].hist(ood_conf, bins=50, alpha=0.7, label=f"OOD ({ood_name})", density=True)
    axes[1].set_xlabel("Max Probability (Confidence)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Confidence Distribution (lower = more uncertain)")
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_reliability_diagram(probs, targets, n_bins: int = 10):
    """Plot reliability diagram (calibration plot)."""
    max_probs, preds = probs.max(dim=1)
    accuracies = (preds == targets).float()

    # Bin predictions by confidence
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (max_probs >= bin_boundaries[i]) & (max_probs < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean().item())
            bin_confs.append(max_probs[mask].mean().item())
            bin_counts.append(mask.sum().item())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]).item() / 2)
            bin_counts.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Reliability diagram
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i + 1]).item() / 2 for i in range(n_bins)]
    axes[0].bar(bin_centers, bin_accs, width=1 / n_bins, alpha=0.7, edgecolor="black")
    axes[0].plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Reliability Diagram")
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # Compute ECE (Expected Calibration Error)
    total = sum(bin_counts)
    ece = sum(
        (bin_counts[i] / total) * abs(bin_accs[i] - bin_confs[i])
        for i in range(n_bins)
        if bin_counts[i] > 0
    )
    axes[0].text(0.05, 0.9, f"ECE = {ece:.4f}", transform=axes[0].transAxes, fontsize=11)

    # Histogram of sample counts per bin
    axes[1].bar(bin_centers, bin_counts, width=1 / n_bins, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Samples per Confidence Bin")

    plt.tight_layout()
    return fig


def main():
    """Train MC Dropout model and evaluate uncertainty."""
    import argparse

    parser = argparse.ArgumentParser(description="MC Dropout Uncertainty Estimation")
    parser.add_argument(
        "--model", type=str, default="mlp", choices=["mlp", "cnn"],
        help="Model architecture (default: mlp)"
    )
    parser.add_argument(
        "--ood", type=str, default="fashionmnist", choices=["fashionmnist", "noise"],
        help="OOD dataset for evaluation (default: fashionmnist)"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--n-samples", type=int, default=50, help="MC samples for inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Create model
    if args.model == "mlp":
        model = MCDropoutMLP(
            input_dim=28 * 28,
            hidden_dim=256,
            output_dim=10,
            dropout_p=args.dropout,
        )
    else:
        model = MCDropoutCNN(
            num_classes=10,
            dropout_p=args.dropout,
            in_channels=1,
        )
    model = model.to(device)
    print(f"Model: {args.model.upper()}")

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs} complete")

    # Evaluate on in-distribution (MNIST test set)
    print("\n--- In-Distribution (MNIST Test) ---")
    acc, conf, ent = evaluate_with_uncertainty(model, test_loader, device, n_samples=args.n_samples)
    print(f"Accuracy: {acc:.4f}")
    print(f"Mean Confidence: {conf:.4f}")
    print(f"Mean Entropy: {ent:.4f}")

    # Evaluate on OOD data
    print(f"\n--- Out-of-Distribution ({args.ood}) ---")
    ood_loader = get_ood_loader(args.ood, batch_size=args.batch_size)
    ood_acc, ood_conf, ood_ent = evaluate_with_uncertainty(model, ood_loader, device, n_samples=args.n_samples)
    print(f"Accuracy: {ood_acc:.4f} (expected to be low)")
    print(f"Mean Confidence: {ood_conf:.4f} (should be lower than in-distribution)")
    print(f"Mean Entropy: {ood_ent:.4f} (should be higher than in-distribution)")

    # Summary
    print("\n--- Uncertainty Comparison ---")
    print(f"Confidence drop: {conf - ood_conf:.4f} (positive = good)")
    print(f"Entropy increase: {ood_ent - ent:.4f} (positive = good)")

    # Collect predictions for visualization
    print("\nCollecting predictions for visualization...")
    id_images, id_probs, id_stds, id_targets = collect_predictions(
        model, test_loader, device, n_samples=args.n_samples
    )
    ood_images, ood_probs, ood_stds, ood_targets = collect_predictions(
        model, ood_loader, device, n_samples=args.n_samples
    )

    # Create plots
    import os
    plot_dir = "./plots/mc_dropout"
    os.makedirs(plot_dir, exist_ok=True)
    prefix = f"{args.model}_{args.ood}"

    # 1. Plot test examples with predicted class and uncertainty
    fig1 = plot_examples_with_uncertainty(id_images, id_probs, id_targets)
    path1 = f"{plot_dir}/{prefix}_examples.png"
    fig1.savefig(path1, dpi=150)
    print(f"Saved: {path1}")

    # 2. Histogram of uncertainties for in-distribution vs OOD
    fig2 = plot_uncertainty_histogram(id_probs, ood_probs, args.ood)
    path2 = f"{plot_dir}/{prefix}_uncertainty_hist.png"
    fig2.savefig(path2, dpi=150)
    print(f"Saved: {path2}")

    # 3. Reliability diagram (calibration plot)
    fig3 = plot_reliability_diagram(id_probs, id_targets)
    path3 = f"{plot_dir}/{prefix}_reliability.png"
    fig3.savefig(path3, dpi=150)
    print(f"Saved: {path3}")

    plt.show()


if __name__ == "__main__":
    main()

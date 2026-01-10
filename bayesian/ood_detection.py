"""
Out-of-Distribution (OOD) Detection using Uncertainty

Key application of uncertainty estimation:
    A model should know when it's seeing data unlike its training set.
    High uncertainty on OOD data = the model "knows what it doesn't know".

Setup:
    - Train on in-distribution (ID) data (e.g., CIFAR-10)
    - Test on ID data (CIFAR-10 test) and OOD data (e.g., SVHN, noise)
    - Good uncertainty: high on OOD, low on ID

OOD detection metrics:
    - AUROC: Area under ROC curve (ID vs OOD classification using uncertainty)
    - AUPR: Area under Precision-Recall curve
    - FPR@95: False positive rate when true positive rate is 95%

Uncertainty measures for OOD:
    1. Max probability: max(softmax(logits)) - simple baseline
    2. Entropy: -sum(p * log(p)) - higher = more uncertain
    3. Mutual information (for BNNs): epistemic uncertainty
    4. Energy score: -logsumexp(logits) - better than max prob

Experiments to run:
    - CIFAR-10 (ID) vs SVHN (OOD)
    - CIFAR-10 (ID) vs Gaussian noise (OOD)
    - CIFAR-10 (ID) vs CIFAR-100 (OOD)
    - MNIST (ID) vs FashionMNIST (OOD)

Expected results:
    - Deep Ensembles often best
    - MC Dropout reasonable
    - Max probability surprisingly competitive
"""


import argparse
import os

# Import MC Dropout model
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import bayesian.mc_dropout as mc


def entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy of probability distribution.

    H = -sum(p * log(p))

    Higher entropy = more uncertain.
    """
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=dim)


def max_probability(logits: torch.Tensor) -> torch.Tensor:
    """Maximum softmax probability (confidence).

    Simple baseline for OOD detection.
    Lower max prob = more uncertain.
    """
    return F.softmax(logits, dim=-1).max(dim=-1).values


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy-based OOD score.

    E(x) = -T * logsumexp(logits / T)

    Lower energy = more confident (in-distribution).
    Higher energy = less confident (out-of-distribution).

    Paper: "Energy-based Out-of-distribution Detection" (Liu et al., 2020)
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def mutual_information(mc_probs: torch.Tensor) -> torch.Tensor:
    """Mutual information from MC samples (epistemic uncertainty).

    MI = H[E[p]] - E[H[p]]
       = entropy of mean - mean of entropies

    High MI = high epistemic uncertainty = likely OOD.

    Args:
        mc_probs: Probabilities from MC samples, shape (n_samples, batch, n_classes)

    Returns:
        MI for each example in batch
    """
    mean_probs = mc_probs.mean(dim=0)
    entropy_of_mean = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
    mean_of_entropy = -(mc_probs * torch.log(mc_probs + 1e-8)).sum(dim=-1).mean(dim=0)

    return entropy_of_mean - mean_of_entropy


def compute_ood_metrics(id_scores: torch.Tensor, ood_scores: torch.Tensor):
    """Compute OOD detection metrics.

    Args:
        id_scores: Uncertainty scores for in-distribution data (should be LOW)
        ood_scores: Uncertainty scores for OOD data (should be HIGH)

    Returns:
        dict with auroc, aupr, fpr95
    """

    n_id = id_scores.numel()
    n_ood = ood_scores.numel()

    # AUROC: fraction of (ood, id) pairs where ood_score > id_score
    id_sorted = torch.sort(id_scores).values
    ood_sorted = torch.sort(ood_scores).values

    p1 = 0
    good_pairs = 0

    for i in range(n_ood):
        while p1 < n_id and ood_sorted[i] > id_sorted[p1]:
            p1 += 1
        good_pairs += p1

    auroc = good_pairs / (n_id * n_ood)

    # FPR@95: threshold where 95% of OOD is detected
    tpr_95_threshold = torch.quantile(ood_scores, 0.05)
    fpr95 = (id_scores > tpr_95_threshold).float().mean().item()

    # AUPR using sklearn
    labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    scores = np.concatenate([id_scores.cpu().numpy(), ood_scores.cpu().numpy()])
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)

    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}


@torch.no_grad()
def get_uncertainty_scores(model, loader: DataLoader, device: torch.device,
                           method: str = "entropy", n_samples: int = 50):
    """Get uncertainty scores for a dataset.

    Args:
        model: Trained model (with predict_with_uncertainty if using MC methods)
        loader: DataLoader
        device: torch device
        method: One of "entropy", "max_prob", "energy", "mutual_info"
        n_samples: Number of MC samples (for MC methods)

    Returns:
        scores: Uncertainty score for each example
    """
    model.eval()
    all_scores = []

    for batch in loader:
        # Handle both (x, y) and just x
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)

        if method == "mutual_info":
            # Need MC samples for mutual information
            mc_probs = []
            for _ in range(n_samples):
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                mc_probs.append(probs)
            mc_probs = torch.stack(mc_probs, dim=0)  # (n_samples, batch, n_classes)
            scores = mutual_information(mc_probs)
        else:
            # For non-MC methods, single forward pass
            logits = model(x)

            if method == "entropy":
                probs = F.softmax(logits, dim=-1)
                scores = entropy(probs, dim=-1)
            elif method == "max_prob":
                # Convert to uncertainty: 1 - max_prob (higher = more uncertain)
                scores = 1 - max_probability(logits)
            elif method == "energy":
                # Energy score: higher = more uncertain (OOD)
                scores = energy_score(logits)
            else:
                raise ValueError(f"Unknown method: {method}")

        all_scores.append(scores)

    return torch.cat(all_scores, dim=0)


def load_ood_dataset(name: str, batch_size: int = 128, image_size: tuple = (1, 28, 28)):
    """Load OOD dataset for evaluation.

    Args:
        name: One of "svhn", "cifar100", "noise", "fashionmnist"
        batch_size: Batch size for DataLoader
        image_size: Expected image size (C, H, W) for noise generation

    Returns:
        DataLoader for OOD data
    """
    from torch.utils.data import TensorDataset
    from torchvision import datasets, transforms

    if name == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Same normalization as MNIST
        ])
        dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif name == "noise":
        # Generate Gaussian noise images matching the expected input size
        n_samples = 10000
        noise_data = torch.randn(n_samples, *image_size)
        noise_labels = torch.zeros(n_samples, dtype=torch.long)  # Dummy labels
        dataset = TensorDataset(noise_data, noise_labels)
    elif name == "svhn":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        dataset = datasets.SVHN(
            root="./data", split="test", download=True, transform=transform
        )
    elif name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown OOD dataset: {name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    """Evaluate OOD detection with different uncertainty methods."""
    parser = argparse.ArgumentParser(description="OOD Detection Evaluation")
    parser.add_argument("--ood", type=str, default="fashionmnist",
                        choices=["fashionmnist", "noise"],
                        help="OOD dataset to use")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--n_samples", type=int, default=50, help="MC samples for mutual info")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST (in-distribution)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    id_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Train MC Dropout model on MNIST
    print("\n--- Training MC Dropout model on MNIST ---")
    model = mc.MCDropoutMLP(input_dim=784, hidden_dim=256, output_dim=10, dropout_p=0.2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, train_acc = mc.train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs}: Loss={train_loss:.4f}, Acc={train_acc:.2%}")

    # Load OOD dataset
    print(f"\n--- Loading OOD dataset: {args.ood} ---")
    ood_loader = load_ood_dataset(args.ood, batch_size=128, image_size=(1, 28, 28))

    # Evaluate with different uncertainty methods
    methods = ["entropy", "max_prob", "energy", "mutual_info"]
    results = {}

    print("\n--- Computing uncertainty scores ---")
    for method in methods:
        print(f"  Computing {method}...")
        id_scores = get_uncertainty_scores(model, id_loader, device, method=method, n_samples=args.n_samples)
        ood_scores = get_uncertainty_scores(model, ood_loader, device, method=method, n_samples=args.n_samples)
        metrics = compute_ood_metrics(id_scores, ood_scores)
        results[method] = {
            "id_scores": id_scores,
            "ood_scores": ood_scores,
            "metrics": metrics
        }

    # Print results table
    print("\n" + "=" * 60)
    print(f"OOD Detection Results: MNIST (ID) vs {args.ood.upper()} (OOD)")
    print("=" * 60)
    print(f"{'Method':<15} {'AUROC':<10} {'AUPR':<10} {'FPR@95':<10}")
    print("-" * 60)
    for method in methods:
        m = results[method]["metrics"]
        print(f"{method:<15} {m['auroc']:.4f}     {m['aupr']:.4f}     {m['fpr95']:.4f}")
    print("=" * 60)

    # Visualizations
    os.makedirs("./plots/ood_detection", exist_ok=True)

    # 1. Histogram of ID vs OOD scores for each method
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, method in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        id_scores = results[method]["id_scores"].cpu().numpy()
        ood_scores = results[method]["ood_scores"].cpu().numpy()

        ax.hist(id_scores, bins=50, alpha=0.6, label="ID (MNIST)", density=True)
        ax.hist(ood_scores, bins=50, alpha=0.6, label=f"OOD ({args.ood})", density=True)
        ax.set_xlabel("Uncertainty Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{method} - AUROC: {results[method]['metrics']['auroc']:.3f}")
        ax.legend()

    plt.suptitle("ID vs OOD Score Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./plots/ood_detection/histograms_{args.ood}.png", dpi=150)
    print(f"Saved: ./plots/ood_detection/histograms_{args.ood}.png")

    # 2. ROC curves for all methods
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in methods:
        id_scores = results[method]["id_scores"].cpu().numpy()
        ood_scores = results[method]["ood_scores"].cpu().numpy()
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])

        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = results[method]["metrics"]["auroc"]
        ax.plot(fpr, tpr, label=f"{method} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves: MNIST vs {args.ood}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"./plots/ood_detection/roc_curves_{args.ood}.png", dpi=150)
    print(f"Saved: ./plots/ood_detection/roc_curves_{args.ood}.png")

    # 3. Example images with uncertainty scores
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    # Get some ID examples
    id_iter = iter(id_loader)
    id_batch, id_labels = next(id_iter)
    id_batch = id_batch[:5].to(device)

    # Get some OOD examples
    ood_iter = iter(ood_loader)
    ood_batch, _ = next(ood_iter)
    ood_batch = ood_batch[:5].to(device)

    # Compute entropy for visualization
    with torch.no_grad():
        id_logits = model(id_batch)
        id_probs = F.softmax(id_logits, dim=-1)
        id_ent = entropy(id_probs, dim=-1)

        ood_logits = model(ood_batch)
        ood_probs = F.softmax(ood_logits, dim=-1)
        ood_ent = entropy(ood_probs, dim=-1)

    # Plot ID examples (top row)
    for i in range(5):
        ax = axes[0, i]
        img = id_batch[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"ID\nH={id_ent[i].item():.2f}")
        ax.axis("off")

    # Plot OOD examples (bottom row)
    for i in range(5):
        ax = axes[1, i]
        img = ood_batch[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"OOD\nH={ood_ent[i].item():.2f}")
        ax.axis("off")

    plt.suptitle("Example Images with Entropy (H)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./plots/ood_detection/examples_{args.ood}.png", dpi=150)
    print(f"Saved: ./plots/ood_detection/examples_{args.ood}.png")

    plt.show()

    # Print key insight
    print("\n--- Key Insight ---")
    print("Good OOD detection: High AUROC, High AUPR, Low FPR@95")
    print("Energy score often outperforms simple max probability.")
    print("Mutual information captures epistemic uncertainty (needs MC samples).")


if __name__ == "__main__":
    main()

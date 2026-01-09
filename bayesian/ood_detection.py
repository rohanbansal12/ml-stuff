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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple


def entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy of probability distribution.

    H = -sum(p * log(p))

    Higher entropy = more uncertain.
    """
    # TODO: Compute entropy
    # Handle numerical issues: add small epsilon before log
    raise NotImplementedError


def max_probability(logits: torch.Tensor) -> torch.Tensor:
    """Maximum softmax probability (confidence).

    Simple baseline for OOD detection.
    Lower max prob = more uncertain.
    """
    # TODO: Return max(softmax(logits))
    raise NotImplementedError


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy-based OOD score.

    E(x) = -T * logsumexp(logits / T)

    Lower energy = more confident (in-distribution).
    Higher energy = less confident (out-of-distribution).

    Paper: "Energy-based Out-of-distribution Detection" (Liu et al., 2020)
    """
    # TODO: Compute energy score
    raise NotImplementedError


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
    # TODO: Compute mutual information
    # 1. Mean probabilities across samples: E[p]
    # 2. Entropy of mean: H[E[p]]
    # 3. Entropy of each sample, then mean: E[H[p]]
    # 4. MI = H[E[p]] - E[H[p]]
    raise NotImplementedError


def compute_ood_metrics(id_scores: torch.Tensor, ood_scores: torch.Tensor):
    """Compute OOD detection metrics.

    Args:
        id_scores: Uncertainty scores for in-distribution data (should be LOW)
        ood_scores: Uncertainty scores for OOD data (should be HIGH)

    Returns:
        auroc: Area under ROC curve
        aupr: Area under Precision-Recall curve
        fpr95: FPR at 95% TPR
    """
    # TODO: Compute metrics
    # Use sklearn.metrics or implement from scratch:
    #
    # AUROC: Probability that a random OOD example has higher score than random ID
    #        = fraction of (ood, id) pairs where ood_score > id_score
    #
    # FPR@95: Find threshold where TPR=0.95, then compute FPR at that threshold
    #         Lower is better (want low false positive rate)
    raise NotImplementedError


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
    # TODO: Iterate through loader
    # TODO: Compute appropriate uncertainty score based on method
    raise NotImplementedError


def load_ood_dataset(name: str, batch_size: int = 128):
    """Load OOD dataset for evaluation.

    Args:
        name: One of "svhn", "cifar100", "noise", "fashionmnist"

    Returns:
        DataLoader for OOD data
    """
    # TODO: Load and return appropriate dataset
    # For "noise": create random Gaussian images matching input size
    raise NotImplementedError


def main():
    """Evaluate OOD detection with different uncertainty methods."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Load model trained on CIFAR-10 (or MNIST)
    # Can use: pre-trained, MC Dropout, Deep Ensemble, or BNN

    # TODO: Load in-distribution test set

    # TODO: Load OOD datasets (SVHN, noise, etc.)

    # TODO: For each uncertainty method (entropy, max_prob, energy, mutual_info):
    #   1. Compute scores on ID data
    #   2. Compute scores on each OOD dataset
    #   3. Compute AUROC, AUPR, FPR95

    # TODO: Create results table:
    # | Method | SVHN AUROC | Noise AUROC | ... |

    # TODO: Visualize:
    # 1. Histogram of ID vs OOD scores
    # 2. ROC curves for different methods
    # 3. Example ID and OOD images with their uncertainty scores

    raise NotImplementedError


if __name__ == "__main__":
    main()

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MCDropoutMLP(nn.Module):
    """MLP with dropout that stays on during inference."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float = 0.1):
        super().__init__()
        # TODO: Define layers (fc1, fc2, fc3, dropout)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout always applied."""
        # TODO: Implement forward pass
        # Key: dropout should be applied regardless of self.training
        raise NotImplementedError

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Run multiple forward passes and compute mean/std.

        Args:
            x: Input tensor
            n_samples: Number of stochastic forward passes

        Returns:
            mean: Mean prediction across samples
            std: Standard deviation (uncertainty) across samples

        Hint: self.train() keeps dropout on, then stack predictions
        """
        # TODO: Implement MC sampling
        # 1. Ensure dropout is on (self.train())
        # 2. Run n_samples forward passes
        # 3. Stack results and compute mean/std
        raise NotImplementedError


class MCDropoutCNN(nn.Module):
    """CNN with spatial dropout for image classification."""

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.1):
        super().__init__()
        # TODO: Define CNN layers
        # Consider using nn.Dropout2d for spatial dropout in conv layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        raise NotImplementedError

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        # TODO: Same as MLP version
        raise NotImplementedError


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Standard training loop."""
    # TODO: Implement training loop
    raise NotImplementedError


@torch.no_grad()
def evaluate_with_uncertainty(model, loader, device, n_samples: int = 50):
    """Evaluate with MC Dropout uncertainty estimates.

    Returns:
        accuracy: Classification accuracy
        mean_confidence: Average max probability (should be lower when uncertain)
        mean_entropy: Average predictive entropy (should be higher when uncertain)
    """
    # TODO: For each batch:
    # 1. Get predictions with uncertainty using n_samples forward passes
    # 2. Compute accuracy, confidence (max prob), entropy
    #
    # Entropy of categorical: H = -sum(p * log(p))
    # Higher entropy = more uncertain
    raise NotImplementedError


def get_dataloaders(batch_size: int = 128):
    """Get MNIST train and test loaders."""
    # TODO: Set up MNIST dataloaders
    raise NotImplementedError


def main():
    """Train MC Dropout model and evaluate uncertainty."""

    # TODO: Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Get data
    # train_loader, test_loader = get_dataloaders()

    # TODO: Create model, optimizer, criterion
    # model = MCDropoutMLP(...) or MCDropoutCNN(...)

    # TODO: Training loop
    # for epoch in range(n_epochs):
    #     train_one_epoch(...)

    # TODO: Evaluate on test set (in-distribution)
    # Show accuracy and uncertainty metrics

    # TODO: Evaluate on OOD data
    # Options:
    # - FashionMNIST (if trained on MNIST)
    # - Random noise
    # - Rotated/corrupted test images
    # Uncertainty should be HIGHER on OOD data

    # TODO: Visualize
    # 1. Plot some test examples with predicted class and uncertainty
    # 2. Histogram of uncertainties for in-distribution vs OOD
    # 3. Reliability diagram (calibration plot)

    raise NotImplementedError


if __name__ == "__main__":
    main()

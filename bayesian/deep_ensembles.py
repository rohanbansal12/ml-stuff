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

import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader


class EnsembleMember(nn.Module):
    """Single network in the ensemble.

    Can output just mean (classification/simple regression)
    or (mean, log_var) for heteroscedastic regression.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 heteroscedastic: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        # TODO: Build network
        # If heteroscedastic, output dim is 2x (mean and log_var for each output)
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        """
        Returns:
            For classification/simple: logits or mean
            For heteroscedastic: (mean, log_var) tuple
        """
        # TODO: Implement forward pass
        raise NotImplementedError


class DeepEnsemble(nn.Module):
    """Ensemble of M independent networks."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_members: int = 5, heteroscedastic: bool = False):
        super().__init__()
        self.n_members = n_members
        self.heteroscedastic = heteroscedastic

        # TODO: Create nn.ModuleList of EnsembleMember networks
        # Each member should have different random initialization (automatic with nn.Module)
        raise NotImplementedError

    def forward_member(self, x: torch.Tensor, member_idx: int):
        """Forward pass through a single member."""
        # TODO: Forward through self.members[member_idx]
        raise NotImplementedError

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
        # TODO: Get predictions from all members
        # TODO: Compute appropriate statistics
        raise NotImplementedError


def train_ensemble(ensemble: DeepEnsemble, train_loader: DataLoader,
                   n_epochs: int, device: torch.device):
    """Train each ensemble member independently.

    Key: Each member sees shuffled data in different order due to DataLoader.
    For even more diversity, can use different data subsets (bootstrap).
    """
    # TODO: For each member:
    #   1. Create separate optimizer
    #   2. Train for n_epochs
    #   3. (Optional) Use different hyperparameters per member
    raise NotImplementedError


def calibration_metrics(predictions: torch.Tensor, confidences: torch.Tensor,
                        labels: torch.Tensor, n_bins: int = 10):
    """Compute calibration metrics.

    A well-calibrated model:
        - When it says 80% confident, it should be right 80% of the time

    Returns:
        ece: Expected Calibration Error
        bin_accuracies: Accuracy in each confidence bin
        bin_confidences: Average confidence in each bin
    """
    # TODO: Bin predictions by confidence
    # TODO: Compute accuracy in each bin
    # TODO: ECE = weighted average of |accuracy - confidence| per bin
    #
    # This is the data for a reliability diagram:
    #   x-axis: confidence, y-axis: accuracy
    #   Perfect calibration = diagonal line
    raise NotImplementedError


def reliability_diagram(bin_accuracies: torch.Tensor, bin_confidences: torch.Tensor):
    """Plot reliability diagram (calibration plot)."""
    # TODO: Bar plot of accuracy vs confidence bins
    # TODO: Add diagonal line for perfect calibration
    # TODO: Below diagonal = overconfident, above = underconfident
    raise NotImplementedError


def main():
    """Train ensemble and evaluate uncertainty quality."""
    torch.manual_seed(42)

    # TODO: Load dataset (MNIST or CIFAR-10)

    # TODO: Create and train DeepEnsemble with M=5 members

    # TODO: Evaluate:
    # 1. Accuracy (should be slightly better than single model)
    # 2. Calibration (reliability diagram, ECE)
    # 3. OOD detection (higher uncertainty on OOD data)

    # TODO: Compare to MC Dropout:
    # - Train single model with dropout
    # - Compare uncertainty quality
    # - Compare calibration

    # TODO: Visualize (for 2D toy data):
    # - Decision boundaries of each member
    # - Ensemble uncertainty heatmap

    raise NotImplementedError


if __name__ == "__main__":
    main()

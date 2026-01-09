"""
Laplace Approximation for Neural Networks

Classic approach (MacKay, 1992) making a comeback with modern variants.

Core idea:
    1. Train network normally to get MAP estimate w*
    2. Approximate posterior as Gaussian centered at w*
    3. Covariance = inverse Hessian of loss at w*

    p(w|D) ≈ N(w*, H^{-1})
    where H = ∇²L(w*) is the Hessian of the loss

Why it works:
    - Taylor expansion of log posterior around MAP
    - Gaussian is exact for quadratic log posterior
    - Good approximation near the mode

The challenge: Computing/storing the Hessian
    - Full Hessian: O(P²) where P = millions of parameters
    - Inversion: O(P³)

Approximations:
    1. Diagonal: Only diagonal of Hessian (independent weights)
    2. KFAC: Kronecker-factored approximate curvature (block diagonal)
    3. Last-layer: Only Laplace on final layer (tractable, often sufficient)
    4. Subnetwork: Laplace on subset of weights

Last-layer Laplace:
    - Treat all but last layer as fixed feature extractor
    - Apply Laplace only to final linear layer
    - Much smaller Hessian: (features × classes)²
    - Often works surprisingly well!

Modern library: https://github.com/aleximmer/laplace

Exercises:
    1. Implement diagonal Laplace for small network
    2. Implement last-layer Laplace (most practical)
    3. Compare to MC Dropout and ensembles
    4. Explore effect of prior precision (regularization strength)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DiagonalLaplace:
    """Diagonal Laplace approximation.

    Assumes posterior is N(w*, diag(1/h)) where h is diagonal of Hessian.

    For classification with cross-entropy:
        Hessian diagonal ≈ sum over data of: (p * (1-p)) * x²
        where p = softmax output, x = input to that weight
    """

    def __init__(self, model: nn.Module, prior_precision: float = 1.0):
        """
        Args:
            model: Trained neural network (at MAP estimate)
            prior_precision: Prior precision (inverse variance), acts as regularization
        """
        self.model = model
        self.prior_precision = prior_precision
        self.posterior_precision = None  # To be computed

    def fit(self, train_loader: DataLoader, device: torch.device):
        """Compute diagonal Hessian approximation.

        For each weight w_i:
            H_ii ≈ prior_precision + sum over data of (gradient w.r.t. w_i)²

        This uses the empirical Fisher / Gauss-Newton approximation:
            H ≈ E[g g^T] where g = gradient of log likelihood
        """
        # TODO: Compute diagonal of empirical Fisher
        # 1. Initialize precision as prior_precision for each weight
        # 2. For each batch:
        #    a. Forward pass
        #    b. Compute gradients of log likelihood
        #    c. Add squared gradients to precision
        # 3. Store self.posterior_precision
        raise NotImplementedError

    @torch.no_grad()
    def sample_weights(self) -> dict:
        """Sample weights from approximate posterior.

        w ~ N(w*, diag(1/posterior_precision))

        Returns:
            Dictionary mapping parameter names to sampled values
        """
        # TODO: For each parameter:
        # 1. Get MAP value (current weight)
        # 2. Get posterior precision
        # 3. Sample: w = w* + N(0, 1) / sqrt(precision)
        raise NotImplementedError

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Predict with uncertainty using weight sampling.

        1. Sample weights
        2. Set model weights to sample
        3. Forward pass
        4. Repeat and aggregate
        """
        # TODO: Sample weights and make predictions
        # Remember to restore original weights after!
        raise NotImplementedError


class LastLayerLaplace:
    """Laplace approximation on last layer only.

    Much more tractable:
    - Fix all layers except the last as feature extractor
    - Apply full (not diagonal) Laplace to last layer
    - Hessian size: (n_features × n_classes)²

    Often works as well as full-network Laplace!
    """

    def __init__(self, model: nn.Module, prior_precision: float = 1.0):
        """
        Args:
            model: Trained network. Last layer should be nn.Linear.
            prior_precision: Prior precision for last layer weights
        """
        self.model = model
        self.prior_precision = prior_precision

        # TODO: Identify the last layer (assume it's named 'fc' or similar)
        # Store reference to last layer and its input dimension

        self.posterior_mean = None  # Last layer weights (flattened)
        self.posterior_cov = None   # Covariance matrix
        raise NotImplementedError

    def fit(self, train_loader: DataLoader, device: torch.device):
        """Compute posterior covariance for last layer.

        1. Extract features for all training data (output before last layer)
        2. Compute Hessian of last layer (this is tractable!)
        3. Invert to get covariance

        For softmax regression:
            H = Phi^T @ diag(p*(1-p)) @ Phi + prior_precision * I
        where Phi = feature matrix
        """
        # TODO: Extract features from training data
        # TODO: Compute Hessian of last layer loss
        # TODO: Add prior precision to diagonal
        # TODO: Invert to get posterior covariance
        raise NotImplementedError

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100,
                                  method: str = "sample"):
        """Predict with uncertainty.

        Two methods:
        1. "sample": Sample last-layer weights, forward pass each
        2. "probit": Closed-form approximation for classification
           (faster but approximate)

        Returns:
            mean: Mean prediction
            std: Uncertainty
        """
        # TODO: Extract features for input x
        # TODO: Either sample weights or use probit approximation
        raise NotImplementedError


def train_map(model: nn.Module, train_loader: DataLoader,
              n_epochs: int, device: torch.device, weight_decay: float = 1e-4):
    """Train model to MAP estimate.

    Note: weight_decay corresponds to Gaussian prior with precision = weight_decay.
    This connects L2 regularization to Bayesian inference!
    """
    # TODO: Standard training loop with weight decay
    raise NotImplementedError


def main():
    """Demo Laplace approximation."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Load dataset (MNIST or CIFAR-10)

    # TODO: Train model to MAP estimate
    # model = create_model()
    # train_map(model, train_loader, ...)

    # TODO: Apply Laplace approximation
    # Option 1: DiagonalLaplace
    # Option 2: LastLayerLaplace (recommended to start)

    # laplace = LastLayerLaplace(model)
    # laplace.fit(train_loader, device)

    # TODO: Evaluate:
    # 1. Accuracy (should match MAP)
    # 2. Calibration (should be better than MAP)
    # 3. OOD detection

    # TODO: Compare different prior_precision values
    # Higher = stronger regularization = less uncertainty

    # TODO: Visualize (for 2D toy data):
    # - MAP decision boundary
    # - Uncertainty heatmap from Laplace

    raise NotImplementedError


if __name__ == "__main__":
    main()

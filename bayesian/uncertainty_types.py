"""
Epistemic vs Aleatoric Uncertainty

Two fundamentally different types of uncertainty:

1. EPISTEMIC (model uncertainty)
   - Uncertainty about the MODEL / parameters
   - Reducible with more data
   - "I don't know because I haven't seen enough examples"
   - Captured by: posterior over weights p(w|D)
   - Example: Predicting far from training data

2. ALEATORIC (data uncertainty)
   - Uncertainty inherent in the DATA
   - Irreducible (more data won't help)
   - "This is inherently noisy/ambiguous"
   - Captured by: predictive distribution p(y|x,w)
   - Example: Noisy labels, overlapping classes

Heteroscedastic aleatoric uncertainty:
    The noise level can vary with input x.
    Model outputs both mean AND variance: f(x) -> (mu, sigma^2)
    Loss: negative log-likelihood of Gaussian

Combined model:
    - Use BNN (or MC Dropout) for epistemic uncertainty
    - Output (mu, log_var) for aleatoric uncertainty
    - Total uncertainty = epistemic + aleatoric

Key equation for total predictive variance:
    Var[y|x] = E[sigma^2(x)] + Var[mu(x)]
             = aleatoric      + epistemic

Paper: "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
       (Kendall & Gal, 2017)

Exercises:
    1. Create synthetic data with heteroscedastic noise
    2. Train network to predict both mean and variance
    3. Use MC Dropout for epistemic + learned variance for aleatoric
    4. Visualize both uncertainty types separately
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_heteroscedastic_data(n_samples: int = 500):
    """Generate data where noise varies with x.

    Example: y = sin(x) + noise, where noise_std = 0.1 + 0.3 * |x|
    This means uncertainty is low near x=0 and high at the edges.

    Returns:
        x: Input, shape (n_samples, 1)
        y: Target, shape (n_samples, 1)
        true_std: Ground truth noise std at each x
    """
    # TODO: Generate x values
    # TODO: Compute y = f(x) + noise where noise std varies with x
    raise NotImplementedError


class HeteroscedasticMLP(nn.Module):
    """Network that outputs mean and log-variance.

    For numerical stability, we predict log(sigma^2) instead of sigma^2.
    This is common practice to avoid negative variances.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # TODO: Create shared backbone
        # TODO: Create separate heads for mean and log_var
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        """
        Returns:
            mean: Predicted mean, shape (batch, 1)
            log_var: Predicted log variance, shape (batch, 1)
        """
        # TODO: Forward through backbone, then separate heads
        raise NotImplementedError


def gaussian_nll_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor):
    """Negative log-likelihood for Gaussian with predicted variance.

    NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
        = 0.5 * (log_var + (y - mu)^2 * exp(-log_var))

    This is the proper loss when the network predicts uncertainty.
    It encourages the network to:
    - Predict high variance where it's wrong (reduces squared error penalty)
    - Predict low variance where it's right (reduces log_var penalty)

    Note: Some implementations add a constant 0.5*log(2*pi), doesn't affect gradients.
    """
    # TODO: Implement Gaussian NLL
    raise NotImplementedError


class EpistemicAleatoricModel(nn.Module):
    """Model that captures both uncertainty types.

    Uses MC Dropout for epistemic + learned variance for aleatoric.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout_p: float = 0.1):
        super().__init__()
        # TODO: Build network with dropout that outputs (mean, log_var)
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        # TODO: Return mean and log_var
        raise NotImplementedError

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Compute epistemic and aleatoric uncertainty separately.

        For each MC sample i:
            mu_i, logvar_i = forward(x)  # with dropout

        Aleatoric uncertainty: E[exp(log_var)] = mean of predicted variances
        Epistemic uncertainty: Var[mu] = variance of predicted means

        Total: aleatoric + epistemic

        Returns:
            pred_mean: Mean of predicted means
            aleatoric_var: Expected predicted variance
            epistemic_var: Variance of predicted means
        """
        # TODO: Run n_samples forward passes
        # TODO: Collect means and variances
        # TODO: Compute aleatoric = mean of variances
        # TODO: Compute epistemic = variance of means
        raise NotImplementedError


def train_heteroscedastic(model, x, y, n_epochs: int = 1000):
    """Train model to predict mean and variance."""
    # TODO: Use gaussian_nll_loss
    raise NotImplementedError


def main():
    """Demo separating epistemic and aleatoric uncertainty."""
    torch.manual_seed(42)

    # TODO: Generate heteroscedastic data

    # TODO: Train EpistemicAleatoricModel

    # TODO: Get predictions with both uncertainty types

    # TODO: Visualize:
    # 1. Data points with true noise bands
    # 2. Predicted mean
    # 3. Aleatoric uncertainty bands (should match true noise)
    # 4. Epistemic uncertainty bands (should be high where no data)

    # TODO: Create a plot with:
    # - Subplot 1: Total uncertainty
    # - Subplot 2: Aleatoric only
    # - Subplot 3: Epistemic only

    # Key insight: Aleatoric follows the data noise pattern,
    # Epistemic grows at the edges / gaps

    raise NotImplementedError


if __name__ == "__main__":
    main()

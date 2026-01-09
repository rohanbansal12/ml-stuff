"""
Bayes by Backprop - Variational Inference for Neural Networks

Paper: "Weight Uncertainty in Neural Networks" (Blundell et al., 2015)

Core idea:
    Instead of point estimate weights, learn a DISTRIBUTION over weights.
    Each weight w_i is parameterized by (mu_i, rho_i) where:
        w_i ~ N(mu_i, sigma_i^2)
        sigma_i = log(1 + exp(rho_i))  # softplus to ensure positive

The ELBO (Evidence Lower BOund):
    L = E_q[log p(D|w)] - KL(q(w|theta) || p(w))
      = data_likelihood    - complexity_cost

    We maximize this (or minimize negative ELBO).

Reparameterization trick:
    To backprop through sampling, we use:
    w = mu + sigma * epsilon,  where epsilon ~ N(0, 1)

    This makes the sampling differentiable w.r.t. mu and sigma.

Training:
    1. Sample epsilon ~ N(0, 1)
    2. Compute w = mu + sigma * epsilon
    3. Forward pass with sampled w
    4. Compute loss = -log p(y|x,w) + KL term
    5. Backprop and update mu, rho

Exercises:
    1. Implement BayesianLinear layer
    2. Implement KL divergence for Gaussian prior
    3. Build a small BNN and train on toy regression/classification
    4. Compare uncertainty estimates to MC Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianLinear(nn.Module):
    """Linear layer with Gaussian weight distributions.

    Instead of storing weights w, we store:
        mu_w: mean of weight distribution
        rho_w: untransformed std (sigma = softplus(rho))
        mu_b, rho_b: same for biases

    The weight is sampled during forward pass using reparameterization.
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # TODO: Initialize parameters
        # mu_w: shape (out_features, in_features), init ~ N(0, 0.1)
        # rho_w: same shape, init to give small initial sigma (e.g., rho = -3)
        # Same for biases
        #
        # Use nn.Parameter to make them learnable

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample weights and compute output.

        Uses reparameterization trick:
            w = mu_w + sigma_w * epsilon,  epsilon ~ N(0,1)
            sigma_w = softplus(rho_w) = log(1 + exp(rho_w))
        """
        # TODO: Implement reparameterized forward pass
        # 1. Compute sigma from rho using softplus
        # 2. Sample epsilon ~ N(0, 1) with same shape as weights
        # 3. Compute w = mu + sigma * epsilon
        # 4. Same for biases
        # 5. Return F.linear(x, w, b)
        raise NotImplementedError

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence from posterior to prior.

        KL( N(mu, sigma^2) || N(0, prior_std^2) )

        Closed form for two Gaussians:
            KL = log(sigma_prior/sigma_post) + (sigma_post^2 + mu_post^2)/(2*sigma_prior^2) - 0.5

        Sum over all weights and biases.
        """
        # TODO: Implement KL divergence
        # Remember to sum over all weight and bias parameters
        raise NotImplementedError


class BayesianMLP(nn.Module):
    """MLP with Bayesian linear layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, prior_std: float = 1.0):
        super().__init__()
        # TODO: Create BayesianLinear layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        raise NotImplementedError

    def kl_divergence(self) -> torch.Tensor:
        """Sum KL divergence from all Bayesian layers."""
        # TODO: Sum kl_divergence() from each BayesianLinear layer
        raise NotImplementedError

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Run multiple forward passes (each samples new weights).

        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        # TODO: Similar to MC Dropout, but now each forward pass
        # samples from the learned weight distributions
        raise NotImplementedError


def elbo_loss(model, x, y, criterion, n_samples: int = 1, kl_weight: float = 1.0):
    """Compute negative ELBO loss.

    ELBO = E_q[log p(y|x,w)] - KL(q(w) || p(w))

    We minimize negative ELBO:
        Loss = -log p(y|x,w) + kl_weight * KL

    Args:
        model: BayesianMLP
        x: Input batch
        y: Target batch
        criterion: e.g., nn.CrossEntropyLoss or nn.MSELoss
        n_samples: Number of weight samples for MC estimate of likelihood
        kl_weight: Weight for KL term (can anneal during training)

    Returns:
        loss: Scalar loss to minimize
    """
    # TODO: Implement ELBO loss
    # 1. Sample weights n_samples times, compute average data loss
    # 2. Compute KL divergence
    # 3. Return data_loss + kl_weight * kl_divergence
    #
    # Note: kl_weight is often set to 1/n_batches to scale KL appropriately
    raise NotImplementedError


def main():
    """Train Bayesian MLP and evaluate uncertainty."""

    # TODO: Generate toy data (e.g., 1D regression with gap in middle)
    # This lets you visualize uncertainty growing in the gap

    # TODO: Create BayesianMLP

    # TODO: Training loop using elbo_loss

    # TODO: Visualize:
    # 1. Mean prediction
    # 2. Uncertainty bands
    # 3. Compare to point-estimate network

    # TODO: Experiment with:
    # - Different prior_std values
    # - KL weight annealing (start with low kl_weight, increase over training)
    # - Number of MC samples during training vs inference

    raise NotImplementedError


if __name__ == "__main__":
    main()

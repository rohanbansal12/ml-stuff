"""
Bayesian Linear Regression - Closed Form Solution

This is the foundation for understanding BNNs. Start here because
the posterior is tractable and you can see exactly what's happening.

Model:
    y = X @ w + epsilon
    w ~ N(0, prior_var * I)           # prior over weights
    epsilon ~ N(0, noise_var)          # observation noise

Goal: Compute the posterior p(w | X, y) and predictive distribution p(y* | x*, X, y)

Key equations to implement:
    Posterior: p(w | X, y) = N(mu_post, Sigma_post)

    Sigma_post = (1/prior_var * I + 1/noise_var * X.T @ X)^{-1}
    mu_post = Sigma_post @ (1/noise_var * X.T @ y)

    Predictive: p(y* | x*) = N(mu_pred, sigma_pred^2)

    mu_pred = x* @ mu_post
    sigma_pred^2 = noise_var + x* @ Sigma_post @ x*.T

Exercises:
    1. Implement 1D case first (scalar w), then generalize to multi-dim
    2. Visualize how posterior uncertainty changes with more data
    3. Show that uncertainty grows in regions with no data
    4. Compare MAP estimate vs posterior mean (should be same here)
"""

import torch
import matplotlib.pyplot as plt


def generate_data(n_samples: int, w_true: torch.Tensor, noise_std: float = 0.3):
    """Generate synthetic linear regression data.

    Args:
        n_samples: Number of data points
        w_true: True weight vector, shape (d,) or (d, 1)
        noise_std: Standard deviation of observation noise

    Returns:
        X: Input features, shape (n_samples, d)
        y: Targets, shape (n_samples, 1)
    """
    # TODO: Generate X from some distribution (e.g., uniform or normal)
    # TODO: Compute y = X @ w_true + noise
    raise NotImplementedError


def compute_posterior(X: torch.Tensor, y: torch.Tensor,
                      prior_var: float, noise_var: float):
    """Compute the posterior distribution over weights.

    Args:
        X: Input features, shape (n, d)
        y: Targets, shape (n, 1)
        prior_var: Prior variance (scalar, isotropic prior)
        noise_var: Observation noise variance

    Returns:
        mu_post: Posterior mean, shape (d, 1)
        Sigma_post: Posterior covariance, shape (d, d)

    Hint: Use torch.linalg.inv or torch.linalg.solve for numerical stability
    """
    # TODO: Implement the posterior equations
    # Sigma_post = (1/prior_var * I + 1/noise_var * X.T @ X)^{-1}
    # mu_post = Sigma_post @ (1/noise_var * X.T @ y)
    raise NotImplementedError


def predictive_distribution(x_test: torch.Tensor, mu_post: torch.Tensor,
                            Sigma_post: torch.Tensor, noise_var: float):
    """Compute predictive distribution at test points.

    Args:
        x_test: Test inputs, shape (n_test, d)
        mu_post: Posterior mean, shape (d, 1)
        Sigma_post: Posterior covariance, shape (d, d)
        noise_var: Observation noise variance

    Returns:
        pred_mean: Predictive mean, shape (n_test, 1)
        pred_std: Predictive standard deviation, shape (n_test,)

    Note: Total variance = noise variance + epistemic uncertainty
          epistemic uncertainty = x* @ Sigma_post @ x*.T
    """
    # TODO: Implement predictive distribution
    # mu_pred = x_test @ mu_post
    # var_pred = noise_var + diag(x_test @ Sigma_post @ x_test.T)
    raise NotImplementedError


def plot_posterior_samples(x_test: torch.Tensor, mu_post: torch.Tensor,
                           Sigma_post: torch.Tensor, n_samples: int = 10):
    """Sample weight vectors from posterior and plot corresponding functions.

    This visualization helps build intuition: each sample is a plausible
    explanation of the data, and they diverge where data is sparse.
    """
    # TODO: Sample w ~ N(mu_post, Sigma_post) multiple times
    # TODO: For each sample, plot x_test @ w
    # Hint: Use torch.linalg.cholesky for sampling from multivariate normal
    raise NotImplementedError


def main():
    """Run Bayesian linear regression demo."""
    torch.manual_seed(42)

    # TODO: Generate synthetic data
    # TODO: Compute posterior
    # TODO: Compute predictive distribution
    # TODO: Plot:
    #   1. Data points
    #   2. Posterior mean prediction
    #   3. Uncertainty bands (e.g., +/- 2 std)
    #   4. (Optional) Posterior samples

    # TODO: Experiment with:
    #   - Different amounts of data (n=5, 20, 100)
    #   - Different prior variances (strong vs weak prior)
    #   - Data only in certain regions (see uncertainty grow elsewhere)

    raise NotImplementedError


if __name__ == "__main__":
    main()

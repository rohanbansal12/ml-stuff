"""
Bayesian Linear Regression - Closed Form Solution

This is the foundation for understanding BNNs. Start here because
the posterior is tractable and you can see exactly what's happening.

Model:
    y = X @ w + epsilon
    w ~ N(0, prior_var * I)           # prior over weights
    epsilon ~ N(0, noise_var)          # observation noise

Goal: Compute the posterior p(w | X, y) and predictive distribution p(y* | x*, X, y)
"""

import matplotlib.pyplot as plt
import torch


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
    d = w_true.size(0)

    X = torch.randn(n_samples, d)
    noise = noise_std * torch.randn(n_samples)

    y = X @ w_true + noise

    return X, y


def compute_posterior(X: torch.Tensor, y: torch.Tensor, prior_var: float, noise_var: float):
    """Compute the posterior distribution over weights.

    Args:
        X: Input features, shape (n, d)
        y: Targets, shape (n, 1)
        prior_var: Prior variance (scalar, isotropic prior)
        noise_var: Observation noise variance

    Returns:
        mu_post: Posterior mean, shape (d, 1)
        Sigma_post: Posterior covariance, shape (d, d)
    """
    N, d = X.shape

    sigma_post = torch.linalg.inv(1 / prior_var * torch.eye(d) + 1 / noise_var * X.T @ X)
    mu_post = sigma_post @ (1 / noise_var * X.T @ y)

    return mu_post, sigma_post


def predictive_distribution(
    x_test: torch.Tensor, mu_post: torch.Tensor, Sigma_post: torch.Tensor, noise_var: float
):
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
    pred_mean = x_test @ mu_post
    pred_std = noise_var + torch.diag(x_test @ Sigma_post @ x_test.T)

    return pred_mean, pred_std


def plot_posterior_samples(
    x_test: torch.Tensor, mu_post: torch.Tensor, Sigma_post: torch.Tensor, n_samples: int = 10
):
    """Sample weight vectors from posterior and plot corresponding functions.

    This visualization helps build intuition: each sample is a plausible
    explanation of the data, and they diverge where data is sparse.

    Args:
        x_test: Test inputs, shape (n_test, d)
        mu_post: Posterior mean, shape (d,) or (d, 1)
        Sigma_post: Posterior covariance, shape (d, d)
        n_samples: Number of posterior samples to draw

    Returns:
        y_samples: Predictions from sampled weights, shape (n_test, n_samples)
    """
    d = mu_post.numel()
    mu = mu_post.flatten()

    # Sample w ~ N(mu_post, Sigma_post) using Cholesky decomposition
    # Sigma = L @ L.T, so w = mu + L @ z where z ~ N(0, I)
    L = torch.linalg.cholesky(Sigma_post)
    eps = torch.randn(d, n_samples)
    w_samples = mu.unsqueeze(1) + L @ eps  # (d, n_samples)

    # Compute predictions for each sampled weight vector
    y_samples = x_test @ w_samples  # (n_test, n_samples)

    return y_samples


def main():
    """Run Bayesian linear regression demo."""
    torch.manual_seed(42)

    # Hyperparameters
    prior_var = 1.0
    noise_std = 0.3
    noise_var = noise_std**2

    # True weights (1D case for easy visualization)
    w_true = torch.tensor([1.5])

    # Generate synthetic data
    n_samples = 20
    X, y = generate_data(n_samples, w_true, noise_std=noise_std)

    # Compute posterior p(w | X, y)
    mu_post, Sigma_post = compute_posterior(X, y, prior_var, noise_var)

    print(f"True weight: {w_true.item():.3f}")
    print(f"Posterior mean: {mu_post.item():.3f}")
    print(f"Posterior std: {Sigma_post.sqrt().item():.3f}")

    # Create test points for visualization
    x_test = torch.linspace(-3, 3, 100).unsqueeze(1)

    # Compute predictive distribution
    pred_mean, pred_var = predictive_distribution(x_test, mu_post, Sigma_post, noise_var)
    pred_std = pred_var.sqrt()

    # Get posterior samples
    y_samples = plot_posterior_samples(x_test, mu_post, Sigma_post, n_samples=10)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Data and predictive distribution
    ax = axes[0]
    ax.scatter(X.numpy(), y.numpy(), c="black", s=30, zorder=5, label="Data")
    ax.plot(x_test.numpy(), pred_mean.numpy(), "b-", linewidth=2, label="Posterior mean")
    ax.fill_between(
        x_test.squeeze().numpy(),
        (pred_mean.squeeze() - 2 * pred_std).numpy(),
        (pred_mean.squeeze() + 2 * pred_std).numpy(),
        alpha=0.3,
        color="blue",
        label="±2 std",
    )
    ax.plot(x_test.numpy(), (x_test @ w_true).numpy(), "g--", linewidth=2, label="True function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Bayesian Linear Regression (n={n_samples})")
    ax.legend()

    # Plot 2: Posterior samples
    ax = axes[1]
    ax.scatter(X.numpy(), y.numpy(), c="black", s=30, zorder=5, label="Data")
    for i in range(y_samples.shape[1]):
        ax.plot(x_test.numpy(), y_samples[:, i].numpy(), alpha=0.5, linewidth=1)
    ax.plot(x_test.numpy(), pred_mean.numpy(), "b-", linewidth=2, label="Posterior mean")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Samples")
    ax.legend()

    # Plot 3: Effect of data amount on uncertainty
    ax = axes[2]
    for n in [5, 20, 100]:
        torch.manual_seed(42)
        X_n, y_n = generate_data(n, w_true, noise_std=noise_std)
        mu_n, Sigma_n = compute_posterior(X_n, y_n, prior_var, noise_var)
        _, var_n = predictive_distribution(x_test, mu_n, Sigma_n, noise_var)
        std_n = var_n.sqrt()
        ax.plot(x_test.numpy(), std_n.numpy(), label=f"n={n}")

    ax.set_xlabel("x")
    ax.set_ylabel("Predictive std")
    ax.set_title("Uncertainty vs Data Amount")
    ax.legend()

    plt.tight_layout()
    plt.savefig("./plots/bayesian_linreg_demo.png", dpi=150)
    plt.show()
    print("\nPlot saved to plots/bayesian_linreg_demo.png")


if __name__ == "__main__":
    main()

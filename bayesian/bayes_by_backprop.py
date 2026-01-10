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


import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
import bayesian.mc_dropout as mc


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

        # Weight parameters: mu and rho (sigma = softplus(rho))
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.rho_w = nn.Parameter(torch.empty(out_features, in_features))

        # Bias parameters
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.rho_b = nn.Parameter(torch.empty(out_features))

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize mu ~ N(0, 0.1) and rho to give small initial sigma."""
        # Initialize mu with small random values
        nn.init.normal_(self.mu_w, mean=0.0, std=0.1)
        nn.init.normal_(self.mu_b, mean=0.0, std=0.1)

        # Initialize rho to give small initial sigma
        # softplus(-3) ≈ 0.05, so initial sigma is small
        nn.init.constant_(self.rho_w, -3.0)
        nn.init.constant_(self.rho_b, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample weights and compute output.

        Uses reparameterization trick:
            w = mu_w + sigma_w * epsilon,  epsilon ~ N(0,1)
            sigma_w = softplus(rho_w) = log(1 + exp(rho_w))
        """
        # Compute sigma from rho using softplus
        sigma_w = F.softplus(self.rho_w)
        sigma_b = F.softplus(self.rho_b)

        # Sample epsilon ~ N(0, 1)
        eps_w = torch.randn_like(self.mu_w)
        eps_b = torch.randn_like(self.mu_b)

        # Reparameterization: w = mu + sigma * epsilon
        w = self.mu_w + sigma_w * eps_w
        b = self.mu_b + sigma_b * eps_b

        return F.linear(x, w, b)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence from posterior to prior.

        KL( N(mu, sigma^2) || N(0, prior_std^2) )

        Closed form for two Gaussians:
            KL = log(sigma_prior/sigma_post) + (sigma_post^2 + mu_post^2)/(2*sigma_prior^2) - 0.5

        Sum over all weights and biases.
        """
        prior_var = self.prior_std ** 2

        # Compute sigma from rho
        sigma_w = F.softplus(self.rho_w)
        sigma_b = F.softplus(self.rho_b)

        # KL for weights
        kl_w = (
            torch.log(self.prior_std / sigma_w)
            + (sigma_w ** 2 + self.mu_w ** 2) / (2 * prior_var)
            - 0.5
        ).sum()

        # KL for biases
        kl_b = (
            torch.log(self.prior_std / sigma_b)
            + (sigma_b ** 2 + self.mu_b ** 2) / (2 * prior_var)
            - 0.5
        ).sum()

        return kl_w + kl_b


class BayesianMLP(nn.Module):
    """MLP with Bayesian linear layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, prior_std: float = 1.0):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim,  hidden_dim, prior_std=prior_std)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim, prior_std=prior_std)
        self.fc3 = BayesianLinear(hidden_dim, output_dim, prior_std=prior_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def kl_divergence(self) -> torch.Tensor:
        """Sum KL divergence from all Bayesian layers."""
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.fc3.kl_divergence()

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Run multiple forward passes (each samples new weights).

        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        outputs = []
        for _ in range(n_samples):
            outputs.append(self.forward(x))

        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(dim=0), outputs.std(dim=0)


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
    # Monte Carlo estimate of data likelihood
    data_loss = 0.0
    for _ in range(n_samples):
        outputs = model(x)
        data_loss = data_loss + criterion(outputs, y)  # Keep as tensor for gradients!

    data_loss = data_loss / n_samples

    # KL divergence (complexity cost)
    kl_loss = model.kl_divergence()

    return data_loss + kl_weight * kl_loss


def generate_gap_data(n_samples: int = 100, noise_std: float = 0.1):
    """Generate 1D regression data with a gap in the middle.

    Data is generated in two regions: [-4, -1] and [1, 4]
    The gap [-1, 1] has no training data.
    This lets us visualize uncertainty growing in the gap.

    Returns:
        x: Input tensor, shape (n_samples, 1)
        y: Target tensor, shape (n_samples, 1)
    """
    # Generate x in two regions, avoiding the gap
    n_left = n_samples // 2
    n_right = n_samples - n_left

    x_left = torch.rand(n_left, 1) * 3 - 4    # Uniform in [-4, -1]
    x_right = torch.rand(n_right, 1) * 3 + 1  # Uniform in [1, 4]
    x = torch.cat([x_left, x_right], dim=0)

    # True function: y = sin(x) + small linear term
    y_true = torch.sin(x) + 0.1 * x

    # Add noise
    noise = torch.randn_like(y_true) * noise_std
    y = y_true + noise

    return x, y


def main():
    """Train Bayesian MLP and evaluate uncertainty."""
    torch.manual_seed(42)

    # Generate toy data with gap in middle
    print("Generating data with gap in [-1, 1]...")
    x_train, y_train = generate_gap_data(n_samples=200, noise_std=0.1)

    # Test points spanning full range (including gap)
    x_test = torch.linspace(-5, 5, 200).unsqueeze(1)
    y_test_true = torch.sin(x_test) + 0.1 * x_test

    # Create model
    model = BayesianMLP(input_dim=1, hidden_dim=64, output_dim=1, prior_std=1.0)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    n_epochs = 2000
    n_samples = len(x_train)

    # KL weight: scale by 1/n_samples so KL term is properly weighted
    # relative to the per-sample data likelihood
    kl_weight = 1.0 / n_samples

    print(f"Training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Compute ELBO loss (with n_samples=1 for speed during training)
        loss = elbo_loss(model, x_train, y_train, criterion,
                         n_samples=1, kl_weight=kl_weight)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 400 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

    # ========== Train MC Dropout for comparison ==========
    print("\n" + "=" * 50)
    print("Training MC Dropout for comparison")
    print("=" * 50)

    mc_model = mc.MCDropoutMLP(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        dropout_p=0.1,
    )

    mc_optimizer = torch.optim.Adam(mc_model.parameters(), lr=1e-2)
    mc_criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        mc_optimizer.zero_grad()
        output = mc_model(x_train)
        loss = mc_criterion(output, y_train)
        loss.backward()
        mc_optimizer.step()

        if (epoch + 1) % 400 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

    # ========== Inference ==========
    print("\n" + "=" * 50)
    print("Running inference...")
    print("=" * 50)

    n_inference_samples = 100

    # Bayesian MLP predictions
    bnn_mean, bnn_std = model.predict_with_uncertainty(x_test, n_samples=n_inference_samples)
    bnn_mean = bnn_mean.squeeze().numpy()
    bnn_std = bnn_std.squeeze().numpy()

    # MC Dropout predictions
    mc_mean, mc_std = mc_model.predict_with_uncertainty(x_test, n_samples=n_inference_samples)
    mc_mean = mc_mean.squeeze().numpy()
    mc_std = mc_std.squeeze().numpy()

    # Convert data to numpy for plotting
    x_train_np = x_train.squeeze().numpy()
    y_train_np = y_train.squeeze().numpy()
    x_test_np = x_test.squeeze().numpy()
    y_test_true_np = y_test_true.squeeze().numpy()

    # ========== Visualization ==========
    import os
    os.makedirs("./plots/bayes_by_backprop", exist_ok=True)

    # Plot 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Compute shared y-limits
    all_vals = [y_train_np.min(), y_train_np.max(),
                (bnn_mean - 2 * bnn_std).min(), (bnn_mean + 2 * bnn_std).max(),
                (mc_mean - 2 * mc_std).min(), (mc_mean + 2 * mc_std).max()]
    y_min, y_max = min(all_vals) - 0.2, max(all_vals) + 0.2

    # Bayes by Backprop
    ax = axes[0]
    ax.scatter(x_train_np, y_train_np, c="black", s=20, alpha=0.5, label="Training data")
    ax.plot(x_test_np, y_test_true_np, "g--", linewidth=2, label="True function")
    ax.plot(x_test_np, bnn_mean, "b-", linewidth=2, label="BNN mean")
    ax.fill_between(x_test_np, bnn_mean - 2 * bnn_std, bnn_mean + 2 * bnn_std,
                    alpha=0.3, color="blue", label="±2σ")
    ax.axvspan(-1, 1, alpha=0.1, color="red", label="Gap region")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bayes by Backprop")
    ax.legend(loc="upper left")
    ax.set_xlim(-5, 5)
    ax.set_ylim(y_min, y_max)

    # MC Dropout
    ax = axes[1]
    ax.scatter(x_train_np, y_train_np, c="black", s=20, alpha=0.5, label="Training data")
    ax.plot(x_test_np, y_test_true_np, "g--", linewidth=2, label="True function")
    ax.plot(x_test_np, mc_mean, "b-", linewidth=2, label="MC Dropout mean")
    ax.fill_between(x_test_np, mc_mean - 2 * mc_std, mc_mean + 2 * mc_std,
                    alpha=0.3, color="blue", label="±2σ")
    ax.axvspan(-1, 1, alpha=0.1, color="red", label="Gap region")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MC Dropout")
    ax.legend(loc="upper left")
    ax.set_xlim(-5, 5)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig("./plots/bayes_by_backprop/comparison.png", dpi=150)
    print("Saved: ./plots/bayes_by_backprop/comparison.png")

    # Plot 2: Uncertainty comparison
    noise_std = 0.1  # True aleatoric uncertainty from data generation
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_test_np, bnn_std, label="Bayes by Backprop", linewidth=2)
    ax.plot(x_test_np, mc_std, label="MC Dropout", linewidth=2)
    ax.axhline(y=noise_std, color="green", linestyle="--", linewidth=2,
               label=f"True noise std ({noise_std})")
    ax.axvspan(-1, 1, alpha=0.1, color="red", label="Gap region")
    ax.axvline(x=-1, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=1, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("Predictive Std")
    ax.set_title("Uncertainty Comparison (green = irreducible noise floor)")
    ax.legend()
    ax.set_xlim(-5, 5)

    plt.tight_layout()
    plt.savefig("./plots/bayes_by_backprop/uncertainty_comparison.png", dpi=150)
    print("Saved: ./plots/bayes_by_backprop/uncertainty_comparison.png")

    # Plot 3: Sample predictions from BNN (showing weight uncertainty)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_train_np, y_train_np, c="black", s=20, alpha=0.5, label="Training data")
    ax.plot(x_test_np, y_test_true_np, "g--", linewidth=2, label="True function")

    # Draw multiple samples from BNN
    for i in range(20):
        with torch.no_grad():
            sample = model(x_test).squeeze().numpy()
        ax.plot(x_test_np, sample, alpha=0.2, color="blue", linewidth=1)

    ax.plot([], [], color="blue", alpha=0.5, label="BNN samples")  # Legend entry
    ax.axvspan(-1, 1, alpha=0.1, color="red", label="Gap region")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bayes by Backprop: Posterior Samples")
    ax.legend(loc="upper left")
    ax.set_xlim(-5, 5)

    plt.tight_layout()
    plt.savefig("./plots/bayes_by_backprop/bnn_samples.png", dpi=150)
    print("Saved: ./plots/bayes_by_backprop/bnn_samples.png")

    plt.show()

    # Print summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"BNN - Mean uncertainty in gap: {bnn_std[80:120].mean():.4f}")
    print(f"BNN - Mean uncertainty outside gap: {(bnn_std[:60].mean() + bnn_std[140:].mean()) / 2:.4f}")
    print(f"MC  - Mean uncertainty in gap: {mc_std[80:120].mean():.4f}")
    print(f"MC  - Mean uncertainty outside gap: {(mc_std[:60].mean() + mc_std[140:].mean()) / 2:.4f}")


if __name__ == "__main__":
    main()

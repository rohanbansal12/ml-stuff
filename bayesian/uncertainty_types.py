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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def generate_heteroscedastic_data(n_samples: int = 500):
    """Generate data where noise varies with x.

    Example: y = sin(x) + noise, where noise_std = 0.1 + 0.3 * |x|
    This means uncertainty is low near x=0 and high at the edges.

    Returns:
        x: Input, shape (n_samples, 1)
        y: Target, shape (n_samples, 1)
        true_std: Ground truth noise std at each x
    """
    x = torch.randn(n_samples, 1)

    f_x = torch.sin(x)

    std = 0.1 + 0.3 * torch.abs(x)
    noise = torch.randn(n_samples, 1) * std

    y = f_x + noise

    return x, y, std

class HeteroscedasticMLP(nn.Module):
    """Network that outputs mean and log-variance.

    For numerical stability, we predict log(sigma^2) instead of sigma^2.
    This is common practice to avoid negative variances.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Separate heads for mean and log_var
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            mean: Predicted mean, shape (batch, 1)
            log_var: Predicted log variance, shape (batch, 1)
        """
        features = self.backbone(x)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        return mean, log_var


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
    return 0.5 * (log_var + (target - mean).pow(2) * torch.exp(-log_var))


class EpistemicAleatoricModel(nn.Module):
    """Model that captures both uncertainty types.

    Uses MC Dropout for epistemic + learned variance for aleatoric.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.dropout_p = dropout_p

        # Backbone with dropout (always on via F.dropout with training=True)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Separate heads for mean and log_var
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass with dropout always applied (even in eval mode)."""
        x = nn.functional.dropout(torch.relu(self.fc1(x)), p=self.dropout_p, training=True)
        x = nn.functional.dropout(torch.relu(self.fc2(x)), p=self.dropout_p, training=True)

        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        return mean, log_var

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
        means = []
        variances = []

        for _ in range(n_samples):
            mean, log_var = self.forward(x)
            means.append(mean)
            variances.append(torch.exp(log_var))

        means = torch.stack(means, dim=0)  # (n_samples, batch, 1)
        variances = torch.stack(variances, dim=0)  # (n_samples, batch, 1)

        # Aleatoric: mean of predicted variances (expected data noise)
        aleatoric_var = variances.mean(dim=0)

        # Epistemic: variance of predicted means (model uncertainty)
        pred_mean = means.mean(dim=0)
        epistemic_var = means.var(dim=0)

        return pred_mean, aleatoric_var, epistemic_var


def train_heteroscedastic(model, x, y, n_epochs: int = 1000, lr: float = 1e-2):
    """Train model to predict mean and variance."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        mean, log_var = model(x)
        loss = gaussian_nll_loss(mean, log_var, y).mean()  # Mean over batch
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")


def main():
    """Demo separating epistemic and aleatoric uncertainty."""
    import os

    torch.manual_seed(42)

    # Generate heteroscedastic data (noise varies with x)
    x_train, y_train, true_std = generate_heteroscedastic_data(n_samples=500)

    # Train model that captures both uncertainty types
    model = EpistemicAleatoricModel(input_dim=1, hidden_dim=64, dropout_p=0.1)
    train_heteroscedastic(model, x_train, y_train, n_epochs=2000, lr=1e-2)

    # Create test points (including extrapolation regions)
    x_test = torch.linspace(-4, 4, 200).unsqueeze(1)
    true_function = torch.sin(x_test)
    true_noise_std = 0.1 + 0.3 * torch.abs(x_test)

    # Get predictions with both uncertainty types
    pred_mean, aleatoric_var, epistemic_var = model.predict_with_uncertainty(x_test, n_samples=100)
    total_var = aleatoric_var + epistemic_var

    # Convert to numpy for plotting
    x_test_np = x_test.squeeze().numpy()
    x_train_np = x_train.squeeze().numpy()
    y_train_np = y_train.squeeze().numpy()
    pred_mean_np = pred_mean.squeeze().numpy()
    true_func_np = true_function.squeeze().numpy()
    true_std_np = true_noise_std.squeeze().numpy()

    aleatoric_std = aleatoric_var.sqrt().squeeze().numpy()
    epistemic_std = epistemic_var.sqrt().squeeze().numpy()
    total_std = total_var.sqrt().squeeze().numpy()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: Total uncertainty
    ax = axes[0]
    ax.scatter(x_train_np, y_train_np, alpha=0.3, s=10, c="black", label="Data")
    ax.plot(x_test_np, true_func_np, "g--", linewidth=2, label="True function")
    ax.plot(x_test_np, pred_mean_np, "b-", linewidth=2, label="Predicted mean")
    ax.fill_between(
        x_test_np,
        pred_mean_np - 2 * total_std,
        pred_mean_np + 2 * total_std,
        alpha=0.3,
        color="blue",
        label="±2σ total",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Total Uncertainty (Epistemic + Aleatoric)")
    ax.legend(loc="upper right")
    ax.set_xlim(-4, 4)

    # Subplot 2: Aleatoric only
    ax = axes[1]
    ax.scatter(x_train_np, y_train_np, alpha=0.3, s=10, c="black", label="Data")
    ax.plot(x_test_np, pred_mean_np, "b-", linewidth=2, label="Predicted mean")
    ax.fill_between(
        x_test_np,
        pred_mean_np - 2 * aleatoric_std,
        pred_mean_np + 2 * aleatoric_std,
        alpha=0.3,
        color="orange",
        label="±2σ aleatoric",
    )
    # Show true noise level
    ax.plot(x_test_np, true_func_np + 2 * true_std_np, "g--", alpha=0.5, linewidth=1)
    ax.plot(x_test_np, true_func_np - 2 * true_std_np, "g--", alpha=0.5, linewidth=1, label="True noise")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Aleatoric Uncertainty (Data Noise)")
    ax.legend(loc="upper right")
    ax.set_xlim(-4, 4)

    # Subplot 3: Epistemic only
    ax = axes[2]
    ax.scatter(x_train_np, y_train_np, alpha=0.3, s=10, c="black", label="Data")
    ax.plot(x_test_np, pred_mean_np, "b-", linewidth=2, label="Predicted mean")
    ax.fill_between(
        x_test_np,
        pred_mean_np - 2 * epistemic_std,
        pred_mean_np + 2 * epistemic_std,
        alpha=0.3,
        color="red",
        label="±2σ epistemic",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Epistemic Uncertainty (Model Uncertainty)")
    ax.legend(loc="upper right")
    ax.set_xlim(-4, 4)

    plt.tight_layout()

    # Save plot
    os.makedirs("./plots/uncertainty_types", exist_ok=True)
    plt.savefig("./plots/uncertainty_types/epistemic_vs_aleatoric.png", dpi=150)
    print("Saved: ./plots/uncertainty_types/epistemic_vs_aleatoric.png")

    # Additional plot: Compare uncertainty components
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_test_np, aleatoric_std, label="Aleatoric (learned)", color="orange", linewidth=2)
    ax.plot(x_test_np, epistemic_std, label="Epistemic (MC Dropout)", color="red", linewidth=2)
    ax.plot(x_test_np, true_std_np, label="True noise std", color="green", linestyle="--", linewidth=2)
    ax.axvline(x=x_train_np.min(), color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=x_train_np.max(), color="gray", linestyle=":", alpha=0.5, label="Training data range")
    ax.set_xlabel("x")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Uncertainty Decomposition")
    ax.legend()
    ax.set_xlim(-4, 4)

    plt.tight_layout()
    plt.savefig("./plots/uncertainty_types/uncertainty_decomposition.png", dpi=150)
    print("Saved: ./plots/uncertainty_types/uncertainty_decomposition.png")

    plt.show()

    # Print key insight
    print("\n--- Key Insight ---")
    print("Aleatoric uncertainty: Should match the true noise pattern (higher at edges of x)")
    print("Epistemic uncertainty: Should be low where data exists, high in extrapolation regions")


if __name__ == "__main__":
    main()

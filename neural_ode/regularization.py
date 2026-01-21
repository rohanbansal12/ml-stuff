"""
Regularization Techniques for Neural ODEs

Problem:
    Neural ODEs can learn overly complex dynamics.
    Complex dynamics = many solver steps = slow training/inference.

    The ODE solver adaptively increases steps when dynamics are "stiff"
    (rapidly changing). Bad dynamics can blow up computation.

Solutions:

1. Kinetic Energy Regularization
    Penalize ||dz/dt||² to encourage smooth, slow dynamics.
    R = ∫ ||f(z,t)||² dt

2. Jacobian Regularization
    Penalize ||∂f/∂z||_F to encourage simple dynamics.
    Prevents chaotic behavior.

3. Straight-through estimator
    Regularize towards straight-line paths.
    R = ||z(T) - z(0) - T*f(z(0),0)||²

4. Time-reparameterization
    Learn to warp time to make dynamics smoother.
    Use a monotonic network to parameterize t = g(s).

5. Weight decay
    Standard L2 regularization on network weights.
    Implicitly regularizes dynamics complexity.

Monitoring training:
    Track number of function evaluations (NFE).
    If NFE grows during training, dynamics are becoming complex.

Paper: "How to Train Your Neural ODE" (Finlay et al., ICML 2020)

Exercises:
    1. Implement kinetic energy regularization
    2. Monitor NFE during training with/without regularization
    3. Implement Jacobian regularization using Hutchinson estimator
    4. Compare convergence speed with different regularizations
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def kinetic_energy_regularization(f: torch.Tensor) -> torch.Tensor:
    """Compute kinetic energy penalty.

    R = ||f||² = ||dz/dt||²

    This encourages the ODE to take slow, smooth paths.

    Args:
        f: Dynamics output dz/dt, shape (batch, dim)

    Returns:
        Regularization term (scalar)
    """
    return f.pow(2).mean()


def jacobian_frobenius_regularization(f: torch.Tensor, z: torch.Tensor,
                                       n_hutchinson: int = 1) -> torch.Tensor:
    """Compute Frobenius norm of Jacobian penalty.

    R = ||∂f/∂z||_F²

    Use Hutchinson estimator: ||A||_F² = E[||Av||²] where v~N(0,I)

    This discourages complex, chaotic dynamics.

    Args:
        f: Dynamics output, shape (batch, dim). Must have been computed with
           z.requires_grad=True and create_graph=True for backprop.
        z: Input state, shape (batch, dim)
        n_hutchinson: Number of random vectors for estimation

    Returns:
        Regularization term (scalar)
    """
    # Hutchinson estimator for ||J||_F²:
    # ||J||_F² = trace(JᵀJ) = E[vᵀJᵀJv] = E[||Jv||²] for v ~ N(0, I)
    #
    # We use autograd to compute Jᵀv via:
    # grad(f, z, v) = vᵀ(∂f/∂z) = (Jᵀv)ᵀ
    #
    # Since trace(JJᵀ) = trace(JᵀJ), we have E[||Jᵀv||²] = ||J||_F²

    frobenius_sq = 0.0

    for _ in range(n_hutchinson):
        # Sample random vector v ~ N(0, I)
        v = torch.randn_like(f)

        # Compute Jᵀv via vector-Jacobian product
        # grad_outputs=v computes vᵀJ, which gives us Jᵀv
        Jv = torch.autograd.grad(
            outputs=f,
            inputs=z,
            grad_outputs=v,
            create_graph=True,  # Need this for backprop through regularization
            retain_graph=True,
        )[0]

        # ||Jᵀv||² is an unbiased estimate of ||J||_F²
        frobenius_sq = frobenius_sq + (Jv ** 2).sum(dim=1).mean()

    return frobenius_sq / n_hutchinson


def straight_path_regularization(z0: torch.Tensor, zT: torch.Tensor,
                                  f0: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Encourage straight-line paths.

    R = ||zT - z0 - T*f(z0,0)||²

    If the path were perfectly straight, zT = z0 + T*f0.
    Penalize deviation from this.

    Args:
        z0: Initial state, shape (batch, dim)
        zT: Final state, shape (batch, dim)
        f0: Dynamics at t=0 (dz/dt|_{t=0}), shape (batch, dim)
        T: Integration time

    Returns:
        Regularization term (scalar)
    """
    # Straight-line prediction: z(T) ≈ z(0) + T * f(z(0), 0)
    # This assumes constant velocity (Euler method with one step)
    z_straight = z0 + T * f0

    # Penalize deviation from straight path
    deviation = zT - z_straight
    return (deviation ** 2).sum(dim=1).mean()


class RegularizedODEFunc(nn.Module):
    """ODE function wrapper that computes regularization terms.

    Tracks dynamics values for regularization during integration.

    Usage:
        func = RegularizedODEFunc(base_func, reg_type="kinetic")
        func.reset_regularization()
        # ... ODE integration using func ...
        reg_loss = func.get_regularization()

    For straight path regularization, call set_final_state(zT, T) after integration.
    """

    def __init__(self, base_func: nn.Module, reg_type: str = "kinetic",
                 n_hutchinson: int = 1):
        """
        Args:
            base_func: The underlying ODE dynamics function
            reg_type: Type of regularization to track ("kinetic", "jacobian", "straight")
            n_hutchinson: Number of Hutchinson vectors for jacobian estimation
        """
        super().__init__()
        self.base_func = base_func
        self.reg_type = reg_type
        self.n_hutchinson = n_hutchinson

        # Accumulators for regularization (use lists to preserve gradients)
        self._kinetic_energies: list[torch.Tensor] = []
        self._jacobian_regs: list[torch.Tensor] = []

        # For straight path regularization
        self._z0: torch.Tensor | None = None
        self._f0: torch.Tensor | None = None
        self._zT: torch.Tensor | None = None
        self._T: float = 1.0
        self._last_z: torch.Tensor | None = None
        self._last_t: float = 0.0

        self.n_evals = 0

    def reset_regularization(self):
        """Reset accumulators before forward pass."""
        self._kinetic_energies = []
        self._jacobian_regs = []
        self._z0 = None
        self._f0 = None
        self._zT = None
        self._T = 1.0
        self._last_z = None
        self._last_t = 0.0
        self.n_evals = 0

    def set_final_state(self, zT: torch.Tensor, T: float = 1.0):
        """Set final state for straight path regularization.

        Call this after ODE integration completes.

        Args:
            zT: Final state after integration
            T: Total integration time
        """
        self._zT = zT
        self._T = T

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        """Forward pass that accumulates regularization."""
        # Check if we're in a no_grad context (e.g., during evaluation)
        grad_enabled = torch.is_grad_enabled()

        # For jacobian regularization, we need gradients w.r.t. z
        if self.reg_type == "jacobian" and grad_enabled and not z.requires_grad:
            z = z.detach().requires_grad_(True)

        f = self.base_func(t, z)

        # Only accumulate regularization if gradients are enabled
        if grad_enabled:
            # Accumulate kinetic energy: ||f||² = ||dz/dt||²
            ke = (f ** 2).sum(dim=1).mean()
            self._kinetic_energies.append(ke)

            # Accumulate jacobian regularization if requested
            if self.reg_type == "jacobian":
                jac_reg = jacobian_frobenius_regularization(f, z, self.n_hutchinson)
                self._jacobian_regs.append(jac_reg)

            # Track for straight path regularization
            if self.reg_type == "straight":
                if self.n_evals == 0:
                    # Store z0, f0 at first evaluation
                    self._z0 = z.detach().clone()
                    self._z0.requires_grad_(True)
                    # Recompute f0 with gradient tracking for straight path
                    self._f0 = self.base_func(t, self._z0)
                # Always update last seen state (will be zT after integration)
                self._last_z = z
                self._last_t = t

            self.n_evals += 1

        return f

    def get_regularization(self, reg_type: str | None = None) -> torch.Tensor:
        """Get accumulated regularization term.

        Args:
            reg_type: Type of regularization. If None, uses self.reg_type.
                      One of "kinetic", "jacobian", "straight"

        Returns:
            Regularization term (scalar tensor)
        """
        if reg_type is None:
            reg_type = self.reg_type

        if self.n_evals == 0:
            return torch.tensor(0.0)

        if reg_type == "kinetic":
            # Average kinetic energy over trajectory: (1/N) * Σ ||f_i||²
            return torch.stack(self._kinetic_energies).mean()

        elif reg_type == "jacobian":
            if not self._jacobian_regs:
                raise RuntimeError(
                    "No jacobian regularization accumulated. "
                    "Set reg_type='jacobian' in constructor."
                )
            # Average jacobian regularization over trajectory
            return torch.stack(self._jacobian_regs).mean()

        elif reg_type == "straight":
            if self._z0 is None or self._f0 is None:
                raise RuntimeError(
                    "z0/f0 not recorded. Set reg_type='straight' in constructor."
                )
            # Use explicitly set zT, or fall back to last seen z
            zT = self._zT if self._zT is not None else self._last_z
            if zT is None:
                raise RuntimeError(
                    "Final state not available. Either call set_final_state(zT, T) "
                    "or ensure forward() was called with reg_type='straight'."
                )
            # Use explicitly set T, or estimate from last seen t
            T = self._T if self._zT is not None else (self._last_t + 0.1)  # Approximate
            return straight_path_regularization(self._z0, zT, self._f0, T)

        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")


class NFECounter:
    """Count number of function evaluations.

    Useful for monitoring dynamics complexity during training.
    Wrap ODE function to count forward calls.
    """

    def __init__(self, func: nn.Module):
        self.func = func
        self.nfe = 0

    def __call__(self, t: float, z: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.func(t, z)

    def reset(self):
        self.nfe = 0


def train_step_with_regularization(model, x, y, optimizer, criterion,
                                    reg_weight: float = 0.01,
                                    reg_type: str = "kinetic"):
    """Training step with ODE regularization.

    The model must have a `reg_func` attribute that is a RegularizedODEFunc.
    This wrapper should be used as the ODE dynamics function during integration.

    Args:
        model: Neural ODE model with `reg_func` attribute (RegularizedODEFunc)
        x, y: Batch data
        optimizer: Optimizer
        criterion: Classification/regression loss
        reg_weight: Weight for regularization term
        reg_type: One of "kinetic", "jacobian", "straight"

    Returns:
        total_loss: Total loss (task + regularization)
        task_loss: Task loss only
        reg_loss: Regularization loss only
        nfe: Number of function evaluations
    """
    # Get the regularized ODE function from model
    if not hasattr(model, "reg_func"):
        raise AttributeError(
            "Model must have a `reg_func` attribute (RegularizedODEFunc). "
            "Wrap your ODE function with RegularizedODEFunc and assign to model.reg_func"
        )

    reg_func = model.reg_func

    # 1. Set regularization type and reset accumulators
    reg_func.reg_type = reg_type
    reg_func.reset_regularization()

    # 2. Zero gradients
    optimizer.zero_grad()

    # 3. Forward pass (reg_func accumulates regularization during ODE solve)
    logits = model(x)

    # 4. Compute task loss
    task_loss = criterion(logits, y)

    # 5. Compute regularization loss
    reg_loss = reg_func.get_regularization(reg_type=reg_type)

    # 6. Combine losses
    total_loss = task_loss + reg_weight * reg_loss

    # 7. Backward and step
    total_loss.backward()
    optimizer.step()

    # 8. Get NFE for monitoring
    nfe = reg_func.n_evals

    return total_loss.item(), task_loss.item(), reg_loss.item(), nfe


def plot_nfe_over_training(
    nfe_history: list | dict[str, list],
    title: str = "NFE during training",
    save_path: str | Path | None = None,
    smoothing_window: int = 1,
):
    """Plot number of function evaluations over training.

    Should be relatively stable. If growing, dynamics are becoming complex.
    Regularization should help keep NFE stable or decreasing.

    Args:
        nfe_history: Either a single list of NFE values, or a dict mapping
                     method names to NFE lists for comparison.
        title: Plot title
        save_path: Path to save the figure. If None, displays interactively.
        smoothing_window: Window size for moving average smoothing.
                          Set to 1 for no smoothing.

    Returns:
        fig: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert single list to dict for uniform handling
    if isinstance(nfe_history, list):
        nfe_history = {"NFE": nfe_history}

    for label, nfe_values in nfe_history.items():
        iterations = np.arange(1, len(nfe_values) + 1)
        nfe_array = np.array(nfe_values)

        # Apply smoothing if requested
        if smoothing_window > 1 and len(nfe_values) >= smoothing_window:
            # Moving average smoothing
            kernel = np.ones(smoothing_window) / smoothing_window
            smoothed = np.convolve(nfe_array, kernel, mode="valid")
            # Adjust x-axis for valid convolution output
            smooth_iters = iterations[smoothing_window - 1:]
            ax.plot(smooth_iters, smoothed, label=label, linewidth=2)
            # Show raw data as faint background
            ax.plot(iterations, nfe_array, alpha=0.2, linewidth=1)
        else:
            ax.plot(iterations, nfe_array, label=label, linewidth=2)

    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Number of Function Evaluations (NFE)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if len(nfe_history) > 1:
        ax.legend(loc="best", fontsize=10)

    ax.grid(True, alpha=0.3)

    # Add horizontal line at initial NFE for reference
    first_values = [v[0] for v in nfe_history.values() if len(v) > 0]
    if first_values:
        ax.axhline(y=np.mean(first_values), color="gray", linestyle="--",
                   alpha=0.5, label="Initial NFE")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


class ODEFunc(nn.Module):
    """Simple ODE dynamics network."""

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        t_vec = torch.ones(z.size(0), 1, device=z.device) * t
        return self.net(torch.cat([z, t_vec], dim=1))


class NeuralODEClassifier(nn.Module):
    """Neural ODE classifier with regularization support."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 adaptive: bool = True, rtol: float = 1e-3, atol: float = 1e-4):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer size
            num_classes: Number of output classes
            adaptive: If True, use adaptive step size solver (DOPRI5).
                      If False, use fixed-step RK4.
            rtol: Relative tolerance for adaptive solver
            atol: Absolute tolerance for adaptive solver
        """
        super().__init__()
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol

        # ODE function wrapped with regularization
        base_func = ODEFunc(input_dim, hidden_dim)
        self.reg_func = RegularizedODEFunc(base_func)

        # Classifier head
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ODE integration."""
        from ode_basics import odeint_adaptive, rk4_step

        if self.adaptive:
            # Adaptive step size integration
            t = torch.tensor([0.0, 1.0], device=x.device)
            z_traj = odeint_adaptive(
                self.reg_func, x, t,
                rtol=self.rtol, atol=self.atol
            )
            z = z_traj[-1]  # Final state
        else:
            # Fixed step RK4
            z = x
            n_steps = 10
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = i * dt
                z = rk4_step(self.reg_func, z, t, dt)

        return self.classifier(z)


def generate_moons(n_samples: int = 500, noise: float = 0.1
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two interleaving half circles (moons) dataset."""
    n_per_class = n_samples // 2

    # First moon (upper)
    theta1 = torch.linspace(0, np.pi, n_per_class)
    x1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=1)
    x1 += noise * torch.randn_like(x1)

    # Second moon (lower, shifted)
    theta2 = torch.linspace(0, np.pi, n_per_class)
    x2 = torch.stack([1 - torch.cos(theta2), 1 - torch.sin(theta2) - 0.5], dim=1)
    x2 += noise * torch.randn_like(x2)

    X = torch.cat([x1, x2], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long()

    # Shuffle
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


def generate_spirals(n_samples: int = 500, noise: float = 0.2,
                     n_turns: float = 1.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two interleaved spirals dataset.

    This is a challenging dataset that requires learning complex, curved
    decision boundaries. Unregularized models tend to overfit with wiggly
    boundaries, while regularized models learn smoother separations.

    Args:
        n_samples: Total number of samples (split evenly between classes)
        noise: Standard deviation of Gaussian noise added to points
        n_turns: Number of turns each spiral makes (higher = harder)

    Returns:
        X: Feature tensor of shape (n_samples, 2)
        y: Label tensor of shape (n_samples,)
    """
    n_per_class = n_samples // 2

    # Spiral 1: starts at center, spirals outward
    theta1 = torch.linspace(0, n_turns * 2 * np.pi, n_per_class)
    r1 = theta1 / (n_turns * 2 * np.pi)  # Radius grows linearly with angle
    x1 = torch.stack([r1 * torch.cos(theta1), r1 * torch.sin(theta1)], dim=1)
    x1 += noise * torch.randn_like(x1)

    # Spiral 2: same shape, rotated 180 degrees
    theta2 = torch.linspace(0, n_turns * 2 * np.pi, n_per_class)
    r2 = theta2 / (n_turns * 2 * np.pi)
    x2 = torch.stack([r2 * torch.cos(theta2 + np.pi), r2 * torch.sin(theta2 + np.pi)], dim=1)
    x2 += noise * torch.randn_like(x2)

    X = torch.cat([x1, x2], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long()

    # Shuffle
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


def train_model(model: NeuralODEClassifier, X_train: torch.Tensor, y_train: torch.Tensor,
                epochs: int, reg_type: str, reg_weight: float,
                X_test: torch.Tensor = None, y_test: torch.Tensor = None,
                lr: float = 0.01) -> dict:
    """Train model and collect metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "loss": [],
        "task_loss": [],
        "reg_loss": [],
        "nfe": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(epochs):
        total_loss, task_loss, reg_loss, nfe = train_step_with_regularization(
            model, X_train, y_train, optimizer, criterion,
            reg_weight=reg_weight, reg_type=reg_type
        )

        # Compute train/test accuracy
        with torch.no_grad():
            model.reg_func.reset_regularization()
            train_logits = model(X_train)
            train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()

            if X_test is not None:
                model.reg_func.reset_regularization()
                test_logits = model(X_test)
                test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()
            else:
                test_acc = train_acc

        history["loss"].append(total_loss)
        history["task_loss"].append(task_loss)
        history["reg_loss"].append(reg_loss)
        history["nfe"].append(nfe)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}: loss={total_loss:.4f}, "
                  f"train={train_acc:.4f}, test={test_acc:.4f}, NFE={nfe}")

    return history


def main():
    """Compare training with different regularizations on spirals dataset."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Generate dataset ===
    # Key insight: to demonstrate regularization, we need a setup where
    # the unregularized model CAN overfit (high train acc, low test acc).
    # Then regularization should reduce this gap.
    #
    # Use moons with: small train set + large network = guaranteed overfitting
    print("\nGenerating moons dataset (small train set to encourage overfitting)...")
    n_train = 60   # Very small - forces memorization
    n_test = 500   # Large test set for reliable evaluation
    noise = 0.2    # Moderate noise

    X_train, y_train = generate_moons(n_samples=n_train, noise=noise)
    X_test, y_test = generate_moons(n_samples=n_test, noise=noise)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Training config
    # Large network + long training = overfitting opportunity
    epochs = 400
    hidden_dim = 128  # Larger network for more capacity
    lr = 0.01

    # Adaptive solver tolerances (looser = fewer steps, tighter = more steps)
    rtol, atol = 1e-3, 1e-4

    results = {}
    models = {}

    # === Train without regularization ===
    print("\n=== No Regularization (reg_weight=0) ===")
    model_none = NeuralODEClassifier(
        input_dim=2, hidden_dim=hidden_dim, num_classes=2,
        adaptive=True, rtol=rtol, atol=atol
    ).to(device)
    results["No Reg"] = train_model(
        model_none, X_train, y_train, epochs=epochs,
        reg_type="kinetic", reg_weight=0.0,
        X_test=X_test, y_test=y_test, lr=lr
    )
    models["No Reg"] = model_none
    print(f"Final: train={results['No Reg']['train_acc'][-1]:.4f}, "
          f"test={results['No Reg']['test_acc'][-1]:.4f}")

    # === Train with kinetic energy regularization ===
    # Strong penalty on ||dz/dt||² forces slow, smooth dynamics
    print("\n=== Kinetic Energy Regularization ===")
    model_kinetic = NeuralODEClassifier(
        input_dim=2, hidden_dim=hidden_dim, num_classes=2,
        adaptive=True, rtol=rtol, atol=atol
    ).to(device)
    results["Kinetic"] = train_model(
        model_kinetic, X_train, y_train, epochs=epochs,
        reg_type="kinetic", reg_weight=1.0,  # Strong regularization
        X_test=X_test, y_test=y_test, lr=lr
    )
    models["Kinetic"] = model_kinetic
    print(f"Final: train={results['Kinetic']['train_acc'][-1]:.4f}, "
          f"test={results['Kinetic']['test_acc'][-1]:.4f}")

    # === Train with Jacobian regularization ===
    # Penalize ||∂f/∂z||_F to prevent complex/chaotic dynamics
    print("\n=== Jacobian Regularization ===")
    model_jacobian = NeuralODEClassifier(
        input_dim=2, hidden_dim=hidden_dim, num_classes=2,
        adaptive=True, rtol=rtol, atol=atol
    ).to(device)
    results["Jacobian"] = train_model(
        model_jacobian, X_train, y_train, epochs=epochs,
        reg_type="jacobian", reg_weight=0.5,  # Strong regularization
        X_test=X_test, y_test=y_test, lr=lr
    )
    models["Jacobian"] = model_jacobian
    print(f"Final: train={results['Jacobian']['train_acc'][-1]:.4f}, "
          f"test={results['Jacobian']['test_acc'][-1]:.4f}")

    # === Train with straight path regularization ===
    # Force near-linear trajectories: z(T) ≈ z(0) + T*f(z(0),0)
    print("\n=== Straight Path Regularization ===")
    model_straight = NeuralODEClassifier(
        input_dim=2, hidden_dim=hidden_dim, num_classes=2,
        adaptive=True, rtol=rtol, atol=atol
    ).to(device)
    results["Straight"] = train_model(
        model_straight, X_train, y_train, epochs=epochs,
        reg_type="straight", reg_weight=1.0,  # Strong regularization
        X_test=X_test, y_test=y_test, lr=lr
    )
    models["Straight"] = model_straight
    print(f"Final: train={results['Straight']['train_acc'][-1]:.4f}, "
          f"test={results['Straight']['test_acc'][-1]:.4f}")

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY (Moons Dataset - Small Train Set)")
    print("=" * 60)
    print(f"{'Method':<12} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8} {'Avg NFE':>10}")
    print("-" * 60)
    for name, hist in results.items():
        train_acc = hist['train_acc'][-1]
        test_acc = hist['test_acc'][-1]
        gap = train_acc - test_acc  # Positive = overfitting
        print(f"{name:<12} {train_acc:>10.4f} {test_acc:>10.4f} {gap:>+8.4f} {np.mean(hist['nfe']):>10.1f}")
    print("-" * 60)
    print("Gap = Train - Test (positive indicates overfitting)")

    # === Visualizations ===
    print("\nGenerating visualizations...")
    plots_dir = Path(__file__).parent / "plots" / "regularization"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. NFE curves
    nfe_histories = {name: hist["nfe"] for name, hist in results.items()}
    plot_nfe_over_training(
        nfe_histories,
        title="NFE During Training: Regularization Comparison",
        save_path=plots_dir / "nfe_comparison.png",
        smoothing_window=10
    )

    # 2. Loss and accuracy curves (train vs test)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {"No Reg": "C0", "Kinetic": "C1", "Jacobian": "C2", "Straight": "C3"}

    for name, hist in results.items():
        c = colors[name]
        axes[0].plot(hist["loss"], label=name, color=c, alpha=0.8)
        axes[1].plot(hist["train_acc"], label=f"{name} (train)", color=c, linestyle="-", alpha=0.8)
        axes[1].plot(hist["test_acc"], label=f"{name} (test)", color=c, linestyle="--", alpha=0.8)
        # Generalization gap
        gap = np.array(hist["train_acc"]) - np.array(hist["test_acc"])
        axes[2].plot(gap, label=name, color=c, alpha=0.8)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Train (solid) vs Test (dashed) Accuracy", fontweight="bold")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Train - Test Accuracy")
    axes[2].set_title("Generalization Gap (lower = better)", fontweight="bold")
    axes[2].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "loss_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved loss_accuracy.png")

    # 3. Decision boundaries with train/test points
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    names = ["No Reg", "Kinetic", "Jacobian", "Straight"]

    # Combine train and test for grid bounds
    X_all = torch.cat([X_train, X_test], dim=0).cpu()

    for ax, name in zip(axes, names):
        model = models[name]
        # Create grid
        x_min, x_max = X_all[:, 0].min().item() - 0.2, X_all[:, 0].max().item() + 0.2
        y_min, y_max = X_all[:, 1].min().item() - 0.2, X_all[:, 1].max().item() + 0.2
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 150),
            np.linspace(y_min, y_max, 150)
        )
        grid = torch.tensor(
            np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device
        )

        with torch.no_grad():
            model.reg_func.reset_regularization()
            logits = model(grid)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        probs = probs.reshape(xx.shape)

        # Plot probability field
        ax.contourf(xx, yy, probs, levels=50, cmap="RdBu_r", alpha=0.6)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)

        # Training points (larger, with edge)
        for label, color in [(0, "tab:blue"), (1, "tab:red")]:
            mask = y_train.cpu() == label
            ax.scatter(X_train[mask, 0].cpu(), X_train[mask, 1].cpu(), c=color,
                       s=40, alpha=0.9, edgecolor="black", linewidth=0.8, marker="o",
                       label=f"Train class {label}" if label == 0 else None)

        # Test points (smaller, no edge, different marker)
        for label, color in [(0, "tab:blue"), (1, "tab:red")]:
            mask = y_test.cpu() == label
            ax.scatter(X_test[mask, 0].cpu(), X_test[mask, 1].cpu(), c=color,
                       s=15, alpha=0.4, marker=".", linewidth=0,
                       label=f"Test class {label}" if label == 0 else None)

        train_acc = results[name]["train_acc"][-1]
        test_acc = results[name]["test_acc"][-1]
        ax.set_title(f"{name}\nTrain: {train_acc:.1%} | Test: {test_acc:.1%}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

    # Add legend to first subplot
    axes[0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(plots_dir / "decision_boundaries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved decision_boundaries.png")

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()

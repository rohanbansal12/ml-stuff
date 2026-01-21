"""
Augmented Neural ODEs (ANODE)

Paper: "Augmented Neural ODEs" (Dupont et al., ICLR 2019)

Problem with vanilla Neural ODEs:
    ODE flows are homeomorphisms (continuous bijections).
    This means they CANNOT change topology!

    Example: Cannot separate two concentric circles.
    The trajectories would have to cross, violating uniqueness.

Solution - Augmentation:
    Augment the state with extra dimensions!

    Instead of: dz/dt = f(z, t)     where z ∈ R^d
    Use:        d[z,a]/dt = f([z,a], t)  where a ∈ R^p are auxiliary dims

    Now trajectories can "go around" each other in higher dimensions.

    Input: z0 = [x, 0]  (pad with zeros)
    Output: [z, a] after integration, use only z

Intuition:
    Like how a 2D knot can be untied in 3D.
    Extra dimensions give the flow more room to maneuver.

How many extra dimensions?
    - More = more expressive but more compute
    - Often just 1-5 extra dims helps significantly
    - Paper shows even 1 helps for simple problems

Connection to ResNets:
    Standard ResNet: can increase dimension via projection
    Neural ODE: fixed dimension throughout
    ANODE: recovers ResNet's flexibility

Second-order ODEs:
    Alternative: Model acceleration instead of velocity.
    d²z/dt² = f(z, dz/dt, t)

    Equivalent to first-order in augmented space:
    d[z,v]/dt = [v, f(z,v,t)]

    Physical intuition: velocity and position evolve together.

Exercises:
    1. Show vanilla Neural ODE fails on concentric circles
    2. Implement ANODE with augmentation
    3. Show ANODE succeeds on same problem
    4. Implement second-order Neural ODE
    5. Compare number of function evaluations
"""


import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from ode_basics import rk4_step


class AugmentedODEFunc(nn.Module):
    """ODE dynamics on augmented state space.

    State: [z, a] where z is original data, a is auxiliary.
    """

    def __init__(self, data_dim: int, aug_dim: int, hidden_dim: int = 64):
        """
        Args:
            data_dim: Dimension of original data
            aug_dim: Number of augmented dimensions
            hidden_dim: Hidden layer size
        """
        super().__init__()
        self.data_dim = data_dim
        self.aug_dim = aug_dim
        self.total_dim = data_dim + aug_dim

        self.network = nn.Sequential(
            nn.Linear(self.total_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.total_dim),
        )

    def forward(self, t: float, h: torch.Tensor) -> torch.Tensor:
        """Compute d[z,a]/dt.

        Args:
            t: Current time
            h: Augmented state [z, a] of shape (batch, data_dim + aug_dim)

        Returns:
            Derivative d[z,a]/dt of shape (batch, data_dim + aug_dim)
        """
        t_vec = torch.ones(h.size(0), 1, device=h.device) * t
        h_with_t = torch.cat([h, t_vec], dim=1)  # [z, a, t]
        return self.network(h_with_t)


class ANODE(nn.Module):
    """Augmented Neural ODE."""

    def __init__(self, data_dim: int, aug_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.data_dim = data_dim
        self.aug_dim = aug_dim
        self.func = AugmentedODEFunc(data_dim, aug_dim, hidden_dim)

    def forward(self, x: torch.Tensor, t0: float = 0.0, t1: float = 1.0) -> torch.Tensor:
        """Transform input through augmented ODE.

        Args:
            x: Input data, shape (batch, data_dim)

        Returns:
            Output data, shape (batch, data_dim)
        """
        # Augment input: [x, 0, 0, ...0]
        aug = torch.zeros(x.size(0), self.aug_dim, device=x.device)
        h = torch.cat([x, aug], dim=1)

        # Solve ODE from t0 to t1
        n_steps = 10
        t = torch.linspace(t0, t1, n_steps + 1)
        for i in range(n_steps):
            dt = (t[i + 1] - t[i]).item()
            t_i = t[i].item()
            h = rk4_step(self.func, h, t_i, dt)

        # Extract data dimensions (discard auxiliary)
        return h[:, :self.data_dim]


class SecondOrderODEFunc(nn.Module):
    """Second-order ODE: d²z/dt² = f(z, dz/dt, t).

    Reformulated as first-order system:
        dz/dt = v
        dv/dt = f(z, v, t)

    The network predicts acceleration given position and velocity.
    """

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim

        # Network predicts acceleration from [z, v, t]
        # Input: 2*dim + 1, Output: dim
        self.network = nn.Sequential(
            nn.Linear(2 * dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """Compute d[z,v]/dt = [v, acceleration].

        Args:
            t: Current time
            state: [z, v] concatenated, shape (batch, 2*dim)

        Returns:
            [dz/dt, dv/dt] = [v, f(z,v,t)], shape (batch, 2*dim)
        """
        # Split state into position z and velocity v
        z = state[:, :self.dim]
        v = state[:, self.dim:]

        # Compute acceleration: f(z, v, t)
        t_vec = torch.ones(state.size(0), 1, device=state.device) * t
        inputs = torch.cat([z, v, t_vec], dim=1)
        acceleration = self.network(inputs)

        # Return [dz/dt, dv/dt] = [v, acceleration]
        return torch.cat([v, acceleration], dim=1)


class SecondOrderNeuralODE(nn.Module):
    """Neural ODE with second-order dynamics."""

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.func = SecondOrderODEFunc(dim, hidden_dim)

    def forward(self, x: torch.Tensor, t0: float = 0.0, t1: float = 1.0,
                v0: torch.Tensor = None) -> torch.Tensor:
        """Transform input through second-order ODE.

        Args:
            x: Initial position, shape (batch, dim)
            v0: Initial velocity, defaults to zeros

        Returns:
            Final position, shape (batch, dim)
        """
        # Initialize velocity (zeros if not provided)
        if v0 is None:
            v0 = torch.zeros_like(x)

        # Create augmented state [z, v]
        state = torch.cat([x, v0], dim=1)

        # Solve ODE from t0 to t1
        n_steps = 10
        t = torch.linspace(t0, t1, n_steps + 1)
        for i in range(n_steps):
            dt = (t[i + 1] - t[i]).item()
            t_i = t[i].item()
            state = rk4_step(self.func, state, t_i, dt)

        # Extract final position (discard velocity)
        return state[:, :self.dim]


def generate_concentric_circles(n_samples: int = 500, noise: float = 0.05,
                                 r_inner: float = 0.5, r_outer: float = 1.5
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two concentric circles dataset.

    Inner circle: label 0
    Outer circle: label 1

    This is a classic failure case for vanilla Neural ODEs.
    To make it harder, use smaller noise and closer radii.
    """
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    # Inner circle
    theta_inner = torch.rand(n_inner) * 2 * math.pi
    r_inner_vals = r_inner + noise * torch.randn(n_inner)
    x_inner = torch.stack([r_inner_vals * torch.cos(theta_inner),
                           r_inner_vals * torch.sin(theta_inner)], dim=1)

    # Outer circle
    theta_outer = torch.rand(n_outer) * 2 * math.pi
    r_outer_vals = r_outer + noise * torch.randn(n_outer)
    x_outer = torch.stack([r_outer_vals * torch.cos(theta_outer),
                           r_outer_vals * torch.sin(theta_outer)], dim=1)

    X = torch.cat([x_inner, x_outer], dim=0)
    y = torch.cat([torch.zeros(n_inner), torch.ones(n_outer)]).long()

    # Shuffle
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


def generate_two_spirals(n_samples: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two interleaved spirals.

    Another topology-challenging dataset.
    """
    n_per_class = n_samples // 2

    # Generate parameter t for spiral
    t = torch.linspace(0, 3, n_per_class) + 0.1 * torch.randn(n_per_class)

    # First spiral
    theta1 = t * 2 * math.pi
    r1 = t
    x1 = torch.stack([r1 * torch.cos(theta1), r1 * torch.sin(theta1)], dim=1)
    noise1 = 0.1 * torch.randn_like(x1)
    x1 = x1 + noise1

    # Second spiral (rotated by pi)
    theta2 = t * 2 * math.pi + math.pi
    r2 = t
    x2 = torch.stack([r2 * torch.cos(theta2), r2 * torch.sin(theta2)], dim=1)
    noise2 = 0.1 * torch.randn_like(x2)
    x2 = x2 + noise2

    X = torch.cat([x1, x2], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long()

    # Shuffle
    perm = torch.randperm(len(y))
    return X[perm], y[perm]


class ODEClassifier(nn.Module):
    """Classifier using Neural ODE (vanilla or augmented)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 aug_dim: int = 0, second_order: bool = False):
        """
        Args:
            aug_dim: If > 0, use ANODE with this many extra dims
            second_order: If True, use second-order ODE
        """
        super().__init__()
        self.input_dim = input_dim
        self.second_order = second_order

        # Define ODE (vanilla, augmented, or second-order)
        if second_order:
            self.ode = SecondOrderNeuralODE(input_dim, hidden_dim)
        else:
            # ANODE with aug_dim=0 is vanilla Neural ODE
            self.ode = ANODE(input_dim, aug_dim, hidden_dim)

        # Classifier head: maps ODE output to class logits
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ODE transform -> classifier
        h = self.ode(x)
        return self.classifier(h)


def count_function_evaluations(model: ODEClassifier, x: torch.Tensor) -> int:
    """Count number of function evaluations during forward pass.

    Useful for comparing computational cost of different ODEs.
    """
    # Get the ODE function from the model
    if model.second_order:
        ode_func = model.ode.func
    else:
        ode_func = model.ode.func

    # Store original forward and create counter
    original_forward = ode_func.forward
    counter = [0]  # Use list to allow mutation in closure

    def counting_forward(t, h):
        counter[0] += 1
        return original_forward(t, h)

    # Temporarily replace forward
    ode_func.forward = counting_forward

    # Run forward pass (no grad needed for counting)
    with torch.no_grad():
        model(x)

    # Restore original forward
    ode_func.forward = original_forward

    return counter[0]


def train_classifier(model: ODEClassifier, X: torch.Tensor, y: torch.Tensor,
                     epochs: int = 200, lr: float = 0.01) -> list[float]:
    """Train a classifier and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"  Epoch {epoch + 1}: loss={loss.item():.4f}, acc={acc:.4f}")

    return losses


def get_trajectories_anode(model: ANODE, x: torch.Tensor, n_steps: int = 20
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Get trajectories through augmented state space.

    Returns:
        t_points: Time points, shape (n_steps+1,)
        trajectories: States at each time, shape (n_steps+1, batch, data_dim + aug_dim)
    """
    # Augment input
    aug = torch.zeros(x.size(0), model.aug_dim, device=x.device)
    h = torch.cat([x, aug], dim=1)

    t_points = torch.linspace(0.0, 1.0, n_steps + 1)
    trajectories = [h.clone()]

    for i in range(n_steps):
        dt = (t_points[i + 1] - t_points[i]).item()
        t_i = t_points[i].item()
        h = rk4_step(model.func, h, t_i, dt)
        trajectories.append(h.clone())

    return t_points, torch.stack(trajectories)


def get_trajectories_second_order(model: SecondOrderNeuralODE, x: torch.Tensor,
                                   n_steps: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Get trajectories through [position, velocity] state space."""
    v0 = torch.zeros_like(x)
    state = torch.cat([x, v0], dim=1)

    t_points = torch.linspace(0.0, 1.0, n_steps + 1)
    trajectories = [state.clone()]

    for i in range(n_steps):
        dt = (t_points[i + 1] - t_points[i]).item()
        t_i = t_points[i].item()
        state = rk4_step(model.func, state, t_i, dt)
        trajectories.append(state.clone())

    return t_points, torch.stack(trajectories)


def plot_decision_boundary(ax, model: ODEClassifier, X: torch.Tensor, y: torch.Tensor,
                           title: str, resolution: int = 100):
    """Plot decision boundary for a 2D classifier."""
    x_min, x_max = X[:, 0].min().item() - 0.3, X[:, 0].max().item() + 0.3
    y_min, y_max = X[:, 1].min().item() - 0.3, X[:, 1].max().item() + 0.3

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=X.device)

    with torch.no_grad():
        logits = model(grid)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    probs = probs.reshape(xx.shape)

    # Decision boundary contour
    ax.contourf(xx, yy, probs, levels=50, cmap="RdBu_r", alpha=0.7)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)

    # Data points
    colors = sns.color_palette("tab10", 2)
    for label in [0, 1]:
        mask = y.cpu() == label
        ax.scatter(X[mask, 0].cpu(), X[mask, 1].cpu(), c=[colors[label]],
                   s=20, alpha=0.8, edgecolor="white", linewidth=0.5,
                   label=f"Class {label}")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")


def plot_trajectories_2d(ax, trajectories: torch.Tensor, y: torch.Tensor,
                         title: str, n_samples: int = 30):
    """Plot 2D trajectories (position dimensions only)."""
    colors = sns.color_palette("tab10", 2)

    # Sample indices
    indices = torch.randperm(len(y))[:n_samples]

    for idx in indices:
        traj = trajectories[:, idx, :2].cpu().numpy()  # Only position dims
        label = y[idx].item()
        ax.plot(traj[:, 0], traj[:, 1], color=colors[label], alpha=0.5, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[label], s=30,
                   marker="o", edgecolor="white", linewidth=0.5, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[label], s=30,
                   marker="s", edgecolor="white", linewidth=0.5, zorder=5)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")


def plot_trajectories_3d(ax, trajectories: torch.Tensor, y: torch.Tensor,
                         title: str, n_samples: int = 30):
    """Plot 3D trajectories in augmented space [x1, x2, aug]."""
    colors = sns.color_palette("tab10", 2)

    indices = torch.randperm(len(y))[:n_samples]

    for idx in indices:
        traj = trajectories[:, idx, :3].cpu().numpy()
        label = y[idx].item()
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[label],
                alpha=0.5, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=colors[label],
                   s=30, marker="o", edgecolor="white", linewidth=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=colors[label],
                   s=30, marker="s", edgecolor="white", linewidth=0.5)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("aug")


def main():
    """Compare vanilla Neural ODE vs augmented on challenging datasets."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate concentric circles data
    # Use thin circles (low noise) that are close together to make it hard
    print("Generating concentric circles dataset (hard: thin, close circles)...")
    X, y = generate_concentric_circles(n_samples=500, noise=0.01, r_inner=0.8, r_outer=1.2)
    X, y = X.to(device), y.to(device)
    print(f"Data shape: {X.shape}, Labels: {y.unique().tolist()}")

    # Train vanilla Neural ODE classifier
    # Expect: Poor performance (can't separate circles)
    print("\n=== Vanilla Neural ODE (aug_dim=0) ===")
    vanilla_model = ODEClassifier(
        input_dim=2, hidden_dim=64, num_classes=2, aug_dim=0
    ).to(device)
    vanilla_losses = train_classifier(vanilla_model, X, y, epochs=200)
    with torch.no_grad():
        vanilla_acc = (vanilla_model(X).argmax(dim=1) == y).float().mean().item()
    print(f"Final accuracy: {vanilla_acc:.4f}")
    vanilla_nfe = count_function_evaluations(vanilla_model, X[:1])
    print(f"NFE per sample: {vanilla_nfe}")

    # Train ANODE classifier (even 1 augmented dim)
    # Expect: Good performance
    print("\n=== Augmented Neural ODE (aug_dim=1) ===")
    anode_model = ODEClassifier(
        input_dim=2, hidden_dim=64, num_classes=2, aug_dim=1
    ).to(device)
    anode_losses = train_classifier(anode_model, X, y, epochs=200)
    with torch.no_grad():
        anode_acc = (anode_model(X).argmax(dim=1) == y).float().mean().item()
    print(f"Final accuracy: {anode_acc:.4f}")
    anode_nfe = count_function_evaluations(anode_model, X[:1])
    print(f"NFE per sample: {anode_nfe}")

    # Train second-order Neural ODE
    # Compare performance and NFE (number of function evaluations)
    print("\n=== Second-Order Neural ODE ===")
    second_order_model = ODEClassifier(
        input_dim=2, hidden_dim=64, num_classes=2, second_order=True
    ).to(device)
    second_order_losses = train_classifier(second_order_model, X, y, epochs=200)
    with torch.no_grad():
        second_order_acc = (second_order_model(X).argmax(dim=1) == y).float().mean().item()
    print(f"Final accuracy: {second_order_acc:.4f}")
    second_order_nfe = count_function_evaluations(second_order_model, X[:1])
    print(f"NFE per sample: {second_order_nfe}")

    # Summary comparison
    print("\n=== Summary ===")
    print(f"Vanilla Neural ODE:    acc={vanilla_acc:.4f}, NFE={vanilla_nfe}")
    print(f"Augmented Neural ODE:  acc={anode_acc:.4f}, NFE={anode_nfe}")
    print(f"Second-Order Neural ODE: acc={second_order_acc:.4f}, NFE={second_order_nfe}")

    # Visualizations
    print("\nGenerating visualizations...")
    plots_dir = Path(__file__).parent / "plots" / "augmented"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="tab10")

    # 1. Decision boundaries for each model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_decision_boundary(axes[0], vanilla_model, X, y,
                           f"Vanilla Neural ODE (acc={vanilla_acc:.2%})")
    plot_decision_boundary(axes[1], anode_model, X, y,
                           f"ANODE aug_dim=1 (acc={anode_acc:.2%})")
    plot_decision_boundary(axes[2], second_order_model, X, y,
                           f"Second-Order ODE (acc={second_order_acc:.2%})")
    plt.tight_layout()
    plt.savefig(plots_dir / "decision_boundaries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved decision_boundaries.png")

    # 2. Trajectories through state space (2D projection)
    with torch.no_grad():
        _, vanilla_traj = get_trajectories_anode(vanilla_model.ode, X, n_steps=20)
        _, anode_traj = get_trajectories_anode(anode_model.ode, X, n_steps=20)
        _, second_order_traj = get_trajectories_second_order(
            second_order_model.ode, X, n_steps=20
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_trajectories_2d(axes[0], vanilla_traj, y, "Vanilla ODE Trajectories")
    plot_trajectories_2d(axes[1], anode_traj, y, "ANODE Trajectories (x₁, x₂)")
    plot_trajectories_2d(axes[2], second_order_traj, y, "Second-Order Trajectories (position)")
    plt.tight_layout()
    plt.savefig(plots_dir / "trajectories_2d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved trajectories_2d.png")

    # 3. ANODE trajectories in augmented 3D space
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    plot_trajectories_3d(ax1, anode_traj, y, "ANODE: Augmented Space (x₁, x₂, aug)")

    ax2 = fig.add_subplot(122, projection="3d")
    plot_trajectories_3d(ax2, second_order_traj, y, "Second-Order: (x₁, x₂, v₁)")

    plt.tight_layout()
    plt.savefig(plots_dir / "trajectories_3d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved trajectories_3d.png")

    print(f"\nAll plots saved to {plots_dir}/")

    # TODO: Repeat for two spirals dataset


if __name__ == "__main__":
    main()

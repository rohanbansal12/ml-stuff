"""
Neural Ordinary Differential Equations

Paper: "Neural Ordinary Differential Equations" (Chen et al., NeurIPS 2018)

Core idea:
    Instead of discrete layers: h[l+1] = h[l] + f(h[l], θ[l])
    Use continuous dynamics: dh/dt = f(h(t), t, θ)

    ResNet with infinite layers = Neural ODE!

The forward pass:
    h(T) = h(0) + ∫[0 to T] f(h(t), t, θ) dt

    Solved numerically using ODE solvers (Euler, RK4, etc.)

Key benefits:
    1. Constant memory (O(1) vs O(L) for L-layer ResNet) via adjoint
    2. Adaptive computation (solver chooses # of steps)
    3. Continuous-time modeling (irregularly sampled data)
    4. Invertible by construction (run ODE backwards)

The backward pass (Adjoint method):
    Instead of backprop through solver steps (memory expensive),
    solve another ODE backwards in time!

    Define adjoint: a(t) = dL/dh(t)

    Then: da/dt = -a^T * (∂f/∂h)
    And: dL/dθ = -∫ a^T * (∂f/∂θ) dt

    This gives O(1) memory regardless of solver steps!

Exercises:
    1. Implement Neural ODE forward pass (use your ODE solver)
    2. Implement naive backward (backprop through solver) - see memory usage
    3. Implement adjoint method - verify same gradients, less memory
    4. Train on MNIST as continuous-depth classifier
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent))

from ode_basics import euler_step, odeint, rk4_step


class ODEFunc(nn.Module):
    """Neural network defining ODE dynamics: dh/dt = f(h, t, θ)

    The network takes state h and optionally time t, outputs dh/dt.
    """

    def __init__(self, hidden_dim: int, time_dependent: bool = True):
        """
        Args:
            hidden_dim: Dimension of state h
            time_dependent: If True, condition on time t
        """
        super().__init__()
        self.time_dependent = time_dependent
        dim = hidden_dim + 1 if time_dependent else hidden_dim

        self.network = nn.Sequential(
            nn.Linear(dim, 2 * hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, t: float, h: torch.Tensor):
        """Compute dh/dt.

        Args:
            t: Current time (scalar)
            h: Current state, shape (batch, hidden_dim)

        Returns:
            dh/dt, same shape as h
        """
        if self.time_dependent:
            t = torch.ones(h.size(0), 1, device=h.device) * t
            h = torch.cat([h, t], dim=1)

        return self.network(h)


class NeuralODE(nn.Module):
    """Neural ODE layer.

    Transforms input by integrating learned dynamics.
    """

    def __init__(
        self,
        func: ODEFunc,
        t0: float = 0.0,
        t1: float = 1.0,
        solver: str = "rk4",
        n_steps: int = 10,
    ):
        """
        Args:
            func: ODEFunc defining dynamics
            t0, t1: Integration interval
            solver: ODE solver to use
            n_steps: Number of solver steps (for fixed-step methods)
        """
        super().__init__()
        self.func = func
        self.t0 = t0
        self.t1 = t1
        self.solver = solver
        self.n_steps = n_steps

    def forward(self, h0: torch.Tensor):
        """Integrate from t0 to t1.

        Args:
            h0: Initial state, shape (batch, dim)

        Returns:
            h1: Final state, shape (batch, dim)
        """
        t = torch.linspace(self.t0, self.t1, self.n_steps + 1)

        for i in range(self.n_steps):
            h = (t[i + 1] - t[i]).item()
            t_i = t[i].item()
            if self.solver == "rk4":
                h0 = rk4_step(self.func, h0, t_i, h)
            elif self.solver == "euler":
                h0 = euler_step(self.func, h0, t_i, h)

        return h0

    def trajectory(self, h0: torch.Tensor, n_points: int = 100):
        """Return full trajectory (for visualization)."""
        t = torch.linspace(self.t0, self.t1, n_points + 1)
        return odeint(self.func, z0=h0, t=t, method=self.solver)


# ============ Adjoint Method ============


class ODEAdjoint(torch.autograd.Function):
    """Adjoint method for memory-efficient Neural ODE training.

    Forward: Solve ODE to get h(T)
    Backward: Solve adjoint ODE to get gradients

    Key equations:
        Forward: dh/dt = f(h, t, θ)
        Adjoint: da/dt = -a^T * (∂f/∂h)
        Param grad: dL/dθ = -∫ a^T * (∂f/∂θ) dt

    We solve the augmented ODE backwards:
        d/dt [h, a, dL/dθ] = [f, -a^T * ∂f/∂h, -a^T * ∂f/∂θ]
    """

    @staticmethod
    def forward(ctx, h0: torch.Tensor, func: ODEFunc, t0: float, t1: float, n_steps: int, *params):
        """
        Args:
            h0: Initial state
            func: ODE function
            t0, t1: Time interval
            n_steps: Number of integration steps
            params: Parameters of func (for gradient tracking)

        Returns:
            h1: Final state
        """
        # Solve ODE forward without tracking gradients (we'll compute them via adjoint)
        with torch.no_grad():
            t = torch.linspace(t0, t1, n_steps + 1)
            h = h0.clone()
            for i in range(n_steps):
                dt = (t[i + 1] - t[i]).item()
                t_i = t[i].item()
                h = rk4_step(func, h, t_i, dt)
            h1 = h

        # Save for backward - only final state, not intermediates (O(1) memory!)
        ctx.save_for_backward(h1)
        ctx.func = func
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.n_steps = n_steps

        return h1

    @staticmethod
    def backward(ctx, grad_h1: torch.Tensor):
        """Compute gradients using adjoint method.

        Args:
            grad_h1: Gradient of loss w.r.t. h(t1), i.e., dL/dh(T)

        Returns:
            Gradients w.r.t. h0, func, t0, t1, n_steps, and each parameter
        """
        (h1,) = ctx.saved_tensors
        func = ctx.func
        t0, t1 = ctx.t0, ctx.t1
        n_steps = ctx.n_steps

        # Get parameter list
        params = list(func.parameters())

        # Initialize: h at T, adjoint a = dL/dh(T), param grads = 0
        h = h1.clone().requires_grad_(True)
        a = grad_h1.clone()
        param_grads = [torch.zeros_like(p) for p in params]

        # Time points going backwards: T -> 0
        t = torch.linspace(t1, t0, n_steps + 1)

        for i in range(n_steps):
            dt = (t[i + 1] - t[i]).item()  # negative since going backwards
            t_i = t[i].item()

            # We need gradients of f w.r.t. h and params
            # Compute f(h, t) with gradient tracking
            with torch.enable_grad():
                h_var = h.detach().requires_grad_(True)
                f_val = func(t_i, h_var)

                # Compute a^T @ (∂f/∂h) via vector-Jacobian product
                # vjp = a^T @ J where J = ∂f/∂h
                vjp_h, *vjp_params = torch.autograd.grad(
                    outputs=f_val,
                    inputs=[h_var] + params,
                    grad_outputs=a,
                    allow_unused=True,
                    retain_graph=True,
                )

            # Euler step for adjoint: da/dt = -a^T @ (∂f/∂h)
            # Going backwards with negative dt, so: a_new = a + dt * (-vjp_h) = a - dt * vjp_h
            # But dt is already negative, so: a_new = a + |dt| * vjp_h
            if vjp_h is not None:
                a = a - dt * vjp_h

            # Accumulate parameter gradients: dL/dθ += -dt * a^T @ (∂f/∂θ)
            # With negative dt going backwards: -= dt * vjp_param = += |dt| * vjp_param
            for j, vjp_p in enumerate(vjp_params):
                if vjp_p is not None:
                    param_grads[j] = param_grads[j] - dt * vjp_p

            # Step h backwards: dh/dt = f, so h_new = h + dt * f (dt negative)
            with torch.no_grad():
                h = h + dt * f_val.detach()
                h = h.requires_grad_(True)

        # Return gradients for: h0, func, t0, t1, n_steps, *params
        # Only h0 and params need gradients; func, t0, t1, n_steps are not differentiable
        return (a, None, None, None, None, *param_grads)


def odeint_adjoint(func: ODEFunc, h0: torch.Tensor, t0: float, t1: float, n_steps: int = 10):
    """Memory-efficient ODE integration with adjoint backward.

    Drop-in replacement for regular odeint, but uses O(1) memory.

    Args:
        func: ODEFunc defining dynamics dh/dt = f(h, t)
        h0: Initial state
        t0, t1: Integration interval
        n_steps: Number of integration steps

    Returns:
        h1: Final state at t1
    """
    # Pass parameters explicitly so autograd can track them
    params = tuple(func.parameters())
    return ODEAdjoint.apply(h0, func, t0, t1, n_steps, *params)


# ============ Neural ODE Classifier ============


class ConvODEFunc(nn.Module):
    """Convolutional ODE dynamics for image feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, t: float, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ODEClassifier(nn.Module):
    """Image classifier using Neural ODE.

    Architecture:
        1. Downsample conv layers (image -> feature map)
        2. Neural ODE (continuous-depth transformation)
        3. Global pooling + linear classifier
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_dim: int = 64,
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.use_adjoint = use_adjoint

        # Downsampling: image (1x28x28) -> feature map (hidden_dim x 7 x 7)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1),  # 28->14
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),  # 14->7
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(),
        )

        # ODE dynamics
        self.ode_func = ConvODEFunc(hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, n_steps: int = 10):
        # Downsample to feature map
        h0 = self.downsample(x)

        # Integrate ODE from t=0 to t=1
        if self.use_adjoint:
            h1 = odeint_adjoint(self.ode_func, h0, t0=0.0, t1=1.0, n_steps=n_steps)
        else:
            # Regular forward (backprop through solver)
            t = torch.linspace(0.0, 1.0, n_steps + 1)
            h1 = h0
            for i in range(n_steps):
                dt = (t[i + 1] - t[i]).item()
                t_i = t[i].item()
                h1 = rk4_step(self.ode_func, h1, t_i, dt)

        # Classify
        return self.classifier(h1)


def main():
    """Train Neural ODE classifier on MNIST."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Create ODEClassifier
    model = ODEClassifier(
        in_channels=1,
        num_classes=10,
        hidden_dim=64,
        use_adjoint=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

        return total_loss / total, correct / total

    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

        return total_loss / total, correct / total

    # Compare adjoint vs regular backprop memory usage
    print("\n" + "=" * 50)
    print("Memory Comparison: Adjoint vs Regular Backprop")
    print("=" * 50)

    import gc

    def measure_memory(use_adjoint, n_steps=10):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        test_model = ODEClassifier(
            in_channels=1, num_classes=10, hidden_dim=64, use_adjoint=use_adjoint
        ).to(device)
        test_optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-3)

        # Single batch forward + backward
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)

        test_optimizer.zero_grad()
        output = test_model(data, n_steps=n_steps)
        loss = criterion(output, target)
        loss.backward()

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_mem = 0  # Can't easily measure CPU memory

        del test_model, test_optimizer
        return peak_mem

    for n_steps in [5, 10, 20]:
        mem_adjoint = measure_memory(use_adjoint=True, n_steps=n_steps)
        mem_regular = measure_memory(use_adjoint=False, n_steps=n_steps)
        print(f"Steps={n_steps}: Adjoint={mem_adjoint:.1f}MB, Regular={mem_regular:.1f}MB")

    # Training loop
    print("\n" + "=" * 50)
    print("Training Neural ODE Classifier")
    print("=" * 50)

    n_epochs = 3
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Visualization: Feature trajectory through ODE
    print("\n" + "=" * 50)
    print("Visualizing Feature Trajectories")
    print("=" * 50)

    # Create plots directory
    plots_dir = Path(__file__).parent / "plots" / "neural"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="tab10")

    model.eval()

    # Get a few test samples (one per digit)
    samples_per_class = {}
    for data, target in test_loader:
        for i in range(len(target)):
            label = target[i].item()
            if label not in samples_per_class:
                samples_per_class[label] = data[i : i + 1]
            if len(samples_per_class) == 10:
                break
        if len(samples_per_class) == 10:
            break

    # Stack samples: one image per digit 0-9
    test_images = torch.cat([samples_per_class[i] for i in range(10)]).to(device)

    # Get trajectories through the ODE
    with torch.no_grad():
        h0 = model.downsample(test_images)  # (10, 64, 7, 7)

        # Collect trajectory at multiple time points
        n_viz_steps = 20
        t_points = torch.linspace(0.0, 1.0, n_viz_steps + 1)
        trajectories = [h0.clone()]

        h = h0
        for i in range(n_viz_steps):
            dt = (t_points[i + 1] - t_points[i]).item()
            t_i = t_points[i].item()
            h = rk4_step(model.ode_func, h, t_i, dt)
            trajectories.append(h.clone())

        # Stack: (n_steps+1, 10, 64, 7, 7)
        trajectories = torch.stack(trajectories)

        # Global average pool to get (n_steps+1, 10, 64)
        traj_pooled = trajectories.mean(dim=(-2, -1))

        # Use PCA to reduce to 2D for visualization
        traj_flat = traj_pooled.reshape(-1, 64).cpu().numpy()

        # Simple PCA via SVD
        traj_centered = traj_flat - traj_flat.mean(axis=0)
        U, S, Vt = torch.linalg.svd(torch.tensor(traj_centered), full_matrices=False)
        traj_2d = (U[:, :2] * S[:2]).numpy()

        # Reshape back to (n_steps+1, 10, 2)
        traj_2d = traj_2d.reshape(n_viz_steps + 1, 10, 2)

    # Plot 1: Feature trajectories through the ODE
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("tab10", 10)

    for digit in range(10):
        traj = traj_2d[:, digit, :]
        ax.plot(traj[:, 0], traj[:, 1], color=colors[digit], linewidth=2, alpha=0.7)
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            color=colors[digit],
            s=100,
            marker="o",
            edgecolor="white",
            linewidth=2,
            zorder=5,
        )
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            color=colors[digit],
            s=150,
            marker="s",
            edgecolor="white",
            linewidth=2,
            zorder=5,
        )
        ax.annotate(
            str(digit),
            (traj[-1, 0], traj[-1, 1]),
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
        )

    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    ax.set_title("Neural ODE Feature Trajectories (t=0 → t=1)\n○ = start, ■ = end", fontsize=14)

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=colors[i], linewidth=2, label=str(i)) for i in range(10)
    ]
    ax.legend(handles=legend_elements, title="Digit", loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(plots_dir / "trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'trajectories.png'}")

    # Plot 2: Final feature positions with digit labels
    fig, ax = plt.subplots(figsize=(10, 8))
    final_pos = traj_2d[-1]

    for digit in range(10):
        ax.scatter(
            final_pos[digit, 0],
            final_pos[digit, 1],
            color=colors[digit],
            s=400,
            edgecolor="white",
            linewidth=2,
        )
        ax.annotate(
            str(digit),
            (final_pos[digit, 0], final_pos[digit, 1]),
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
        )

    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    ax.set_title(
        "Final Feature Positions (t=1)\nDigits close together have similar learned representations",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(plots_dir / "final_positions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'final_positions.png'}")

    # Plot 3: Distance traveled by each digit
    fig, ax = plt.subplots(figsize=(10, 6))
    distances = []
    for digit in range(10):
        traj = traj_2d[:, digit, :]
        dist = np.sqrt(((traj[1:] - traj[:-1]) ** 2).sum(axis=1)).sum()
        distances.append(dist)

    bars = ax.bar(range(10), distances, color=colors, edgecolor="white", linewidth=2)
    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("Path Length in Feature Space", fontsize=12)
    ax.set_title("Distance Traveled Through ODE (t=0 → t=1)", fontsize=14)
    ax.set_xticks(range(10))

    plt.tight_layout()
    plt.savefig(plots_dir / "distances.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'distances.png'}")

    # Plot 4: Feature evolution over time (heatmap style)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for digit in range(10):
        ax = axes[digit]
        traj = traj_2d[:, digit, :]
        t_vals = np.linspace(0, 1, n_viz_steps + 1)

        # Color by time
        scatter = ax.scatter(
            traj[:, 0], traj[:, 1], c=t_vals, cmap="viridis", s=50, edgecolor="white", linewidth=0.5
        )
        ax.plot(traj[:, 0], traj[:, 1], color="gray", alpha=0.3, linewidth=1)

        ax.set_title(f"Digit {digit}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Time (t)", fontsize=12)

    fig.suptitle(
        "Individual Digit Trajectories Through Neural ODE", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "individual_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'individual_trajectories.png'}")

    # Plot 5: Sample images used for visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for digit in range(10):
        ax = axes[digit]
        img = samples_per_class[digit].squeeze().cpu().numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Digit {digit}", fontsize=12)
        ax.axis("off")

    fig.suptitle("Sample Images Used for Trajectory Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'sample_images.png'}")

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()

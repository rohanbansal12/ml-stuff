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

import torch
import torch.nn as nn
from typing import Tuple


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

        # TODO: Define dynamics network
        # Input: [z, a, t] of size (data_dim + aug_dim + 1)
        # Output: d[z,a]/dt of size (data_dim + aug_dim)
        raise NotImplementedError

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """Compute d[z,a]/dt."""
        # TODO: Implement
        raise NotImplementedError


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
        # TODO: Augment input: [x, 0, 0, ...0]
        # TODO: Solve ODE
        # TODO: Extract data dimensions (discard auxiliary)
        raise NotImplementedError


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

        # TODO: Define network that predicts acceleration
        # Input: [z, v, t]
        # Output: acceleration (same dim as z)
        raise NotImplementedError

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """Compute d[z,v]/dt = [v, acceleration].

        Args:
            state: [z, v] concatenated, shape (batch, 2*dim)

        Returns:
            [dz/dt, dv/dt] = [v, f(z,v,t)]
        """
        # TODO: Split state into z and v
        # TODO: Compute acceleration
        # TODO: Return [v, acceleration]
        raise NotImplementedError


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
        # TODO: Initialize velocity (zeros if not provided)
        # TODO: Create augmented state [z, v]
        # TODO: Solve ODE
        # TODO: Extract final position
        raise NotImplementedError


def generate_concentric_circles(n_samples: int = 500) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two concentric circles dataset.

    Inner circle: label 0, radius ~0.5
    Outer circle: label 1, radius ~1.5

    This is a classic failure case for vanilla Neural ODEs.
    """
    # TODO: Generate data
    # Inner circle: r=0.5, label=0
    # Outer circle: r=1.5, label=1
    raise NotImplementedError


def generate_two_spirals(n_samples: int = 500) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two interleaved spirals.

    Another topology-challenging dataset.
    """
    # TODO: Generate spiral data
    raise NotImplementedError


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
        # TODO: Define ODE (vanilla, augmented, or second-order)
        # TODO: Define classifier head
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: ODE transform -> classifier
        raise NotImplementedError


def count_function_evaluations(model, x: torch.Tensor) -> int:
    """Count number of function evaluations during forward pass.

    Useful for comparing computational cost of different ODEs.
    """
    # TODO: Add counter to ODE function, run forward, return count
    raise NotImplementedError


def main():
    """Compare vanilla Neural ODE vs augmented on challenging datasets."""
    torch.manual_seed(42)

    # TODO: Generate concentric circles data

    # TODO: Train vanilla Neural ODE classifier
    # Expect: Poor performance (can't separate circles)

    # TODO: Train ANODE classifier (even 1 augmented dim)
    # Expect: Good performance

    # TODO: Train second-order Neural ODE
    # Compare performance and NFE (number of function evaluations)

    # TODO: Visualize:
    # 1. Decision boundaries for each model
    # 2. Trajectories through state space
    # 3. For ANODE: trajectories in augmented space

    # TODO: Repeat for two spirals dataset

    raise NotImplementedError


if __name__ == "__main__":
    main()

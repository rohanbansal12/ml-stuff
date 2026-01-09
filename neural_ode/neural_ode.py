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

import torch
import torch.nn as nn
from typing import Callable


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
        # TODO: Define network layers
        # If time_dependent, concatenate t to input: [h, t]
        # Typical architecture: MLP with tanh or softplus activations
        # (tanh helps keep dynamics bounded)
        raise NotImplementedError

    def forward(self, t: float, h: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt.

        Args:
            t: Current time (scalar)
            h: Current state, shape (batch, hidden_dim)

        Returns:
            dh/dt, same shape as h
        """
        # TODO: Implement forward pass
        # If time_dependent, expand t to batch size and concatenate
        raise NotImplementedError


class NeuralODE(nn.Module):
    """Neural ODE layer.

    Transforms input by integrating learned dynamics.
    """

    def __init__(self, func: ODEFunc, t0: float = 0.0, t1: float = 1.0,
                 solver: str = "rk4", n_steps: int = 10):
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

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        """Integrate from t0 to t1.

        Args:
            h0: Initial state, shape (batch, dim)

        Returns:
            h1: Final state, shape (batch, dim)
        """
        # TODO: Use your odeint function to integrate
        # t = torch.linspace(self.t0, self.t1, self.n_steps + 1)
        # Return final state h(t1)
        raise NotImplementedError

    def trajectory(self, h0: torch.Tensor, n_points: int = 100) -> torch.Tensor:
        """Return full trajectory (for visualization)."""
        # TODO: Return states at n_points times between t0 and t1
        raise NotImplementedError


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
    def forward(ctx, h0: torch.Tensor, func: ODEFunc,
                t0: float, t1: float, *params):
        """
        Args:
            h0: Initial state
            func: ODE function
            t0, t1: Time interval
            params: Parameters of func (for gradient tracking)

        Returns:
            h1: Final state
        """
        # TODO: Solve ODE forward
        # Save what's needed for backward (func, t0, t1, h1)
        # Do NOT save intermediate states (that's the point!)
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_h1: torch.Tensor):
        """Compute gradients using adjoint method.

        Args:
            grad_h1: Gradient of loss w.r.t. h(t1), i.e., dL/dh(T)

        Returns:
            Gradients w.r.t. h0 and parameters
        """
        # TODO: This is the tricky part!
        #
        # 1. Initialize adjoint: a(T) = grad_h1
        # 2. Initialize param gradients: dL/dθ = 0
        # 3. Define augmented dynamics for backward solve
        # 4. Solve from T to 0
        # 5. Return gradients

        # The augmented state is [h, a, param_grads...]
        # Augmented dynamics requires computing ∂f/∂h and ∂f/∂θ

        raise NotImplementedError


def odeint_adjoint(func: ODEFunc, h0: torch.Tensor,
                   t0: float, t1: float) -> torch.Tensor:
    """Memory-efficient ODE integration with adjoint backward.

    Drop-in replacement for regular odeint, but uses O(1) memory.
    """
    # TODO: Use ODEAdjoint.apply()
    # Need to pass parameters explicitly for gradient tracking
    raise NotImplementedError


# ============ Neural ODE Classifier ============

class ODEClassifier(nn.Module):
    """Image classifier using Neural ODE.

    Architecture:
        1. Downsample conv layers (image -> feature map)
        2. Neural ODE (continuous-depth transformation)
        3. Global pooling + linear classifier
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 hidden_dim: int = 64):
        super().__init__()
        # TODO: Define architecture
        # Downsampling: Conv layers to reduce spatial size
        # ODE: ODEFunc + NeuralODE (or use adjoint)
        # Classifier: AdaptiveAvgPool + Linear
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        raise NotImplementedError


def main():
    """Train Neural ODE classifier on MNIST."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Load MNIST

    # TODO: Create ODEClassifier

    # TODO: Training loop
    # Compare:
    # 1. Regular backprop through solver (check memory usage)
    # 2. Adjoint method (should use less memory)

    # TODO: Experiment with:
    # - Different integration times (t1 = 1, 2, 5)
    # - Different solver steps
    # - Time-dependent vs time-independent dynamics

    # TODO: Visualize:
    # - Trajectories in feature space
    # - How features evolve from t=0 to t=1

    raise NotImplementedError


if __name__ == "__main__":
    main()

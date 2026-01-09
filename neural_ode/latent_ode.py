"""
Latent ODEs for Irregularly-Sampled Time Series

Paper: "Latent ODEs for Irregularly-Sampled Time Series" (Rubanova et al., 2019)

Motivation:
    Real-world time series often have:
    - Irregular sampling (measurements at random times)
    - Missing data
    - Variable-length sequences

    Standard RNNs assume regular intervals and fixed sequence length.
    Neural ODEs naturally handle continuous time!

Architecture:
    1. Encoder: RNN (or ODE-RNN) processes observations -> latent z0
    2. Latent ODE: Evolves z0 forward in time: dz/dt = f(z, t)
    3. Decoder: Maps z(t) to observation space at any time t

Training (VAE-style):
    - Encode observations to get q(z0 | observations)
    - Sample z0 ~ q(z0 | x)
    - Evolve z0 using ODE to get z(t) at observation times
    - Decode z(t) to reconstruct observations
    - ELBO = reconstruction - KL(q(z0) || prior)

Key insight:
    The ODE is solved at arbitrary times - handles irregularity naturally!

ODE-RNN (for encoder):
    Between observations, evolve hidden state with ODE.
    At observations, update hidden state with RNN cell.

    h(t_i) = ODESolve(h(t_{i-1}), t_{i-1} -> t_i)
    h(t_i) = RNNCell(h(t_i), x_i)

Exercises:
    1. Implement ODE-RNN encoder
    2. Implement Latent ODE model
    3. Train on synthetic irregularly-sampled data
    4. Interpolate and extrapolate in continuous time
    5. Compare to regular RNN on irregular data
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ODERNNCell(nn.Module):
    """RNN cell that evolves between observations using ODE.

    Between observations: dh/dt = f(h, t)
    At observations: h = RNNCell(h, x)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # TODO: Define ODE dynamics network
        # TODO: Define RNN update (e.g., GRU cell)
        raise NotImplementedError

    def ode_step(self, h: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        """Evolve hidden state from t0 to t1 using ODE."""
        # TODO: Use ODE solver to evolve h
        raise NotImplementedError

    def rnn_update(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update hidden state with observation."""
        # TODO: Apply RNN cell (e.g., GRU update equations)
        raise NotImplementedError

    def forward(self, h: torch.Tensor, x: torch.Tensor,
                t_prev: float, t_curr: float) -> torch.Tensor:
        """Process one observation.

        1. Evolve h from t_prev to t_curr
        2. Update h with observation x
        """
        # TODO: Implement
        raise NotImplementedError


class ODERNNEncoder(nn.Module):
    """Encode irregularly-sampled sequence to initial latent state.

    Processes observations in order, handling variable time gaps.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.cell = ODERNNCell(input_dim, hidden_dim)
        # TODO: Define output layers for mean and log_var
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence to latent distribution.

        Args:
            x: Observations, shape (batch, seq_len, input_dim)
            t: Observation times, shape (batch, seq_len)
            mask: Optional mask for missing observations

        Returns:
            z_mean: Mean of latent distribution
            z_log_var: Log variance of latent distribution
        """
        # TODO: Process sequence with ODE-RNN
        # Handle variable time gaps between observations
        # Return parameters of q(z0 | x)
        raise NotImplementedError


class LatentODE(nn.Module):
    """Full Latent ODE model.

    Encoder -> Latent ODE dynamics -> Decoder
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # TODO: Define encoder (ODERNNEncoder)
        # TODO: Define latent dynamics (ODEFunc)
        # TODO: Define decoder (MLP: latent_dim -> input_dim)
        raise NotImplementedError

    def encode(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observations to latent distribution."""
        # TODO: Use encoder
        raise NotImplementedError

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick."""
        # TODO: z = mean + std * epsilon, epsilon ~ N(0, I)
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation space."""
        # TODO: Use decoder
        raise NotImplementedError

    def evolve_latent(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evolve latent state through time.

        Args:
            z0: Initial latent state, shape (batch, latent_dim)
            t: Times to evaluate at, shape (n_times,)

        Returns:
            z: Latent states at each time, shape (batch, n_times, latent_dim)
        """
        # TODO: Solve ODE to get z(t) at specified times
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor,
                t_pred: Optional[torch.Tensor] = None):
        """Forward pass for training.

        Args:
            x: Observations, shape (batch, seq_len, input_dim)
            t_obs: Observation times, shape (batch, seq_len)
            t_pred: Times for prediction (if None, use t_obs)

        Returns:
            x_pred: Predictions at t_pred
            z_mean, z_log_var: Latent distribution parameters
        """
        # TODO: Encode -> Sample -> Evolve -> Decode
        raise NotImplementedError

    def predict(self, x: torch.Tensor, t_obs: torch.Tensor,
                t_pred: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Predict at new times (extrapolation/interpolation).

        Args:
            x: Observed data
            t_obs: Observation times
            t_pred: Times to predict at

        Returns:
            predictions: Predictions at t_pred
        """
        # TODO: Encode, sample multiple z0, evolve, decode
        raise NotImplementedError


def elbo_loss(x: torch.Tensor, x_pred: torch.Tensor,
              z_mean: torch.Tensor, z_log_var: torch.Tensor,
              beta: float = 1.0) -> torch.Tensor:
    """Compute negative ELBO.

    ELBO = E[log p(x|z)] - beta * KL(q(z) || p(z))

    Args:
        x: True observations
        x_pred: Predicted observations
        z_mean, z_log_var: Parameters of q(z)
        beta: Weight for KL term (beta-VAE style)

    Returns:
        loss: Negative ELBO to minimize
    """
    # TODO: Implement
    # Reconstruction: MSE or Gaussian likelihood
    # KL: Closed form for two Gaussians (q vs prior N(0,I))
    raise NotImplementedError


def generate_irregular_data(n_sequences: int = 100, max_len: int = 50):
    """Generate synthetic irregularly-sampled time series.

    Options:
    - Sine waves with random sampling
    - Damped oscillations
    - Lorenz attractor
    """
    # TODO: Generate data
    # Return:
    #   x: observations (padded)
    #   t: times (padded)
    #   mask: which observations are valid
    raise NotImplementedError


def main():
    """Train Latent ODE on irregular time series."""
    torch.manual_seed(42)

    # TODO: Generate irregular time series data

    # TODO: Create LatentODE model

    # TODO: Training loop with ELBO loss

    # TODO: Evaluate:
    # 1. Reconstruction quality
    # 2. Interpolation (predict at unobserved times between observations)
    # 3. Extrapolation (predict beyond observed time range)

    # TODO: Visualize:
    # 1. True vs reconstructed trajectories
    # 2. Interpolation with uncertainty (sample multiple z0)
    # 3. Extrapolation with growing uncertainty

    # TODO: Compare to standard RNN (which ignores irregular sampling)

    raise NotImplementedError


if __name__ == "__main__":
    main()

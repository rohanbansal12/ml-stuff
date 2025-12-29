import torch
import numpy as np


class RolloutBuffer:
    """Buffer for on-policy algorithms (A2C, PPO).

    Stores transitions from vectorized environments and computes
    GAE-λ advantages and returns.
    """

    def __init__(
        self,
        steps: int,
        num_envs: int,
        obs_dim: int,
        device: torch.device,
        act_dim: int = 0,
        cont: bool = False,
    ):
        self.steps = steps
        self.num_envs = num_envs
        self.device = device
        self.cont = cont

        # Allocate buffers
        self.observations = torch.zeros((steps, num_envs, obs_dim), device=device)

        if cont:
            self.actions = torch.zeros((steps, num_envs, act_dim), device=device)
        else:
            self.actions = torch.zeros(
                (steps, num_envs), dtype=torch.long, device=device
            )

        self.rewards = torch.zeros((steps, num_envs), device=device)
        self.dones = torch.zeros((steps, num_envs), device=device)
        self.values = torch.zeros(
            (steps + 1, num_envs), device=device
        )  # +1 for bootstrap
        self.log_probs = torch.zeros((steps, num_envs), device=device)
        self.advantages = torch.zeros((steps, num_envs), device=device)
        self.returns = torch.zeros((steps, num_envs), device=device)

        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        reward: np.ndarray,
        done: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        """Store a single timestep of vectorized transitions."""
        assert not self.is_full(), "Buffer is full, call reset() first"

        self.observations[self.ptr] = torch.as_tensor(
            obs, device=self.device, dtype=torch.float32
        )

        if self.cont:
            self.actions[self.ptr] = torch.as_tensor(
                action, device=self.device, dtype=torch.float32
            )
        else:
            self.actions[self.ptr] = action

        self.rewards[self.ptr] = torch.as_tensor(
            reward, device=self.device, dtype=torch.float32
        )
        self.dones[self.ptr] = torch.as_tensor(
            done, device=self.device, dtype=torch.float32
        )
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr == self.steps

    def reset(self):
        self.ptr = 0

    def compute_advantages_and_returns(
        self, last_value: torch.Tensor, gamma: float, lam: float
    ):
        """Compute GAE-λ advantages and returns.

        GAE formula:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + (γλ)(1 - done_t) * A_{t+1}

        Returns are computed as: R_t = A_t + V(s_t)
        """
        # Store bootstrap value
        self.values[self.ptr] = last_value.detach()

        # Slice active portion
        rewards = self.rewards[: self.ptr]
        values_t = self.values[: self.ptr]
        values_tp1 = self.values[1 : self.ptr + 1]
        dones = self.dones[: self.ptr]

        # Compute TD errors (deltas)
        deltas = rewards + gamma * values_tp1 * (1.0 - dones) - values_t

        # Compute GAE via backwards recursion
        gamma_lam = gamma * lam
        factors = gamma_lam * (1.0 - dones)

        # Reverse for efficient recursion
        deltas_rev = deltas.flip(0)
        factors_rev = factors.flip(0)
        adv_rev = torch.zeros_like(deltas_rev)

        gae = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.ptr):
            gae = deltas_rev[i] + factors_rev[i] * gae
            adv_rev[i] = gae

        self.advantages[: self.ptr] = adv_rev.flip(0)
        self.returns[: self.ptr] = self.advantages[: self.ptr] + values_t

    def get(self) -> tuple[torch.Tensor, ...]:
        """Return flattened buffer data: (obs, actions, log_probs, values, advantages, returns, dones)."""
        return self._flatten()

    def _flatten(self) -> tuple[torch.Tensor, ...]:
        """Flatten (steps, num_envs, ...) -> (steps * num_envs, ...)."""
        n = self.ptr

        obs = self.observations[:n].reshape((-1,) + self.observations.shape[2:])
        actions = self.actions[:n].reshape((-1,) + self.actions.shape[2:])
        log_probs = self.log_probs[:n].reshape(-1)
        values = self.values[:n].reshape(-1)
        advantages = self.advantages[:n].reshape(-1)
        returns = self.returns[:n].reshape(-1)
        dones = self.dones[:n].reshape(-1)

        return obs, actions, log_probs, values, advantages, returns, dones


class ReplayBuffer:
    """Circular replay buffer for off-policy algorithms (SAC, TD3, DQN).

    Supports vectorized environments.
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        self.buffer_size = max(buffer_size // num_envs, 1)
        self.num_envs = num_envs
        self.device = device

        self.observations = torch.zeros(
            (self.buffer_size, num_envs, obs_dim), device=device
        )
        self.next_observations = torch.zeros(
            (self.buffer_size, num_envs, obs_dim), device=device
        )
        self.actions = torch.zeros((self.buffer_size, num_envs, act_dim), device=device)
        self.rewards = torch.zeros((self.buffer_size, num_envs), device=device)
        self.dones = torch.zeros((self.buffer_size, num_envs), device=device)

        self.ptr = 0
        self.full = False

    def store(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ):
        """Store a single timestep of vectorized transitions."""
        self.observations[self.ptr] = torch.as_tensor(
            obs, device=self.device, dtype=torch.float32
        )
        self.next_observations[self.ptr] = torch.as_tensor(
            next_obs, device=self.device, dtype=torch.float32
        )
        self.actions[self.ptr] = torch.as_tensor(
            action, device=self.device, dtype=torch.float32
        )
        self.rewards[self.ptr] = torch.as_tensor(
            reward, device=self.device, dtype=torch.float32
        )
        self.dones[self.ptr] = torch.as_tensor(
            done, device=self.device, dtype=torch.float32
        )

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
            self.ptr = 0

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample a random batch of transitions."""
        max_idx = self.buffer_size if self.full else self.ptr

        # Random indices into (time, env) dimensions
        batch_idx = torch.randint(0, max_idx, (batch_size,), device=self.device)
        env_idx = torch.randint(0, self.num_envs, (batch_size,), device=self.device)

        return (
            self.observations[batch_idx, env_idx],
            self.next_observations[batch_idx, env_idx],
            self.actions[batch_idx, env_idx],
            self.rewards[batch_idx, env_idx].unsqueeze(-1),
            self.dones[batch_idx, env_idx].unsqueeze(-1),
        )

    def __len__(self) -> int:
        return (
            self.buffer_size * self.num_envs if self.full else self.ptr * self.num_envs
        )

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for layer weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
# Discrete Action Networks
# =============================================================================


class PolicyNet(nn.Module):
    """Simple policy network for discrete actions (e.g., REINFORCE)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple = (128, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.ReLU())
            in_dim = h
        self.model = nn.Sequential(*layers)
        self.logits = layer_init(nn.Linear(in_dim, action_dim), std=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.model(obs)
        return self.logits(x)

    def _prepare_obs(self, obs) -> torch.Tensor:
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        return obs_t

    def act(self, obs) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action, return (action, log_prob, entropy)."""
        obs_t = self._prepare_obs(obs)
        logits = self(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return (
            action.item(),
            dist.log_prob(action).squeeze(0),
            dist.entropy().squeeze(0).detach(),
        )

    def act_deterministic(self, obs) -> int:
        """Select action deterministically (argmax)."""
        obs_t = self._prepare_obs(obs)
        with torch.no_grad():
            logits = self(obs_t)
            action = logits.argmax(dim=-1)
        return action.item()


class ActorCriticNet(nn.Module):
    """Shared-body actor-critic for discrete actions (A2C/PPO)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple = (64, 64)):
        super().__init__()

        # Shared body
        in_dim = obs_dim
        layers = []
        for h in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        self.body = nn.Sequential(*layers)

        # Policy head (small init for stable start)
        self.policy = layer_init(nn.Linear(in_dim, action_dim), std=0.01)

        # Value head
        self.value = layer_init(nn.Linear(in_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.body(obs)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.body(obs)
        return self.value(features)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For rollout collection: returns (action, log_prob, value), all detached."""
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob.detach(), value.detach()

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO update: returns (log_prob, entropy, value) with gradients."""
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value


class PPONet(nn.Module):
    """Separate actor-critic networks for PPO (no shared body)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple = (64, 64)):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], action_dim), std=0.01),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.critic(obs)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(obs)


# =============================================================================
# Continuous Action Networks
# =============================================================================


class ContinuousActorCriticNet(nn.Module):
    """Separate actor-critic for continuous actions (PPO continuous)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple = (64, 64)):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], action_dim), std=0.01),
        )

        # Learnable log_std (state-independent)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, self.critic(obs)


class ContinuousRPONet(nn.Module):
    """Continuous actor-critic with RPO (Robust Policy Optimization) noise."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple = (64, 64),
        rpo_alpha: float = 0.5,
    ):
        super().__init__()
        self.rpo_alpha = rpo_alpha

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], action_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)

        if action is None:
            dist = Normal(mean, std)
            action = dist.sample()
        else:
            # RPO: add uniform noise to mean during training
            z = torch.empty_like(mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            mean = mean + z

        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, self.critic(obs)


# =============================================================================
# SAC Networks
# =============================================================================


class SoftCriticNet(nn.Module):
    """Q-network for SAC (takes state-action pair as input)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple = (256, 256)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.model(x)


class SoftActorNet(nn.Module):
    """Squashed Gaussian policy for SAC."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_space: tuple,
        hidden_sizes: tuple = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[1], action_dim)

        # Action scaling
        low, high = action_space
        self.register_buffer(
            "action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp log_std via tanh
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        return mean, log_std.exp()

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization trick and compute log_prob."""
        mean, std = self(obs)
        dist = Normal(mean, std)

        # Reparameterized sample
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias

        # Log prob with tanh correction
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.log(1 - squashed.square() + 1e-6).sum(dim=-1)

        return action, log_prob.unsqueeze(-1)

# =============================================================================
# DQN Networks
# =============================================================================


class QNet(nn.Module):
    """Q-network for DQN. Maps state -> Q-values for all actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple = (128, 128)):
        super().__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], action_dim), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Observation tensor of shape (batch, obs_dim) or (obs_dim,)
            
        Returns:
            q_values: Q-values for all actions, shape (batch, action_dim) or (action_dim,)
        """
        return self.net(obs)
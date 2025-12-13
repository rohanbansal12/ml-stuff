import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.ReLU())
            in_dim = h
        self.model = nn.Sequential(*layers)
        
        # Scale final layer for Max Entropy start (std=0.01)
        self.logits = nn.Linear(in_dim, action_dim)

    def forward(self, obs):
        x = self.model(obs)
        return self.logits(x)
    
    def act(self, obs):
        # Helper to get device
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        
        # Handle single observation vs batch
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            
        logits = self(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        # Return simple python item and the tensor log_prob for the graph
        return action.item(), dist.log_prob(action).squeeze(0)
    

class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()
        
        # --- Shared Body ---
        in_dim = obs_dim
        layers = []
        for h in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh()) # Tanh or ReLU is fine
            in_dim = h
        self.body = nn.Sequential(*layers)
        
        # --- Actor (Policy Head) ---
        # Final policy layer scaled down (std=0.01) for stable start.
        self.policy = layer_init(nn.Linear(in_dim, action_dim), std=0.01)
        
        # --- Critic (Value Head) ---
        # Final value layer scaled to 1.0 (standard for PPO value estimation).
        self.value = layer_init(nn.Linear(in_dim, 1), std=1.0)

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        res = self.body(obs)
        logits = self.policy(res)
        value = self.value(res)
        
        return logits, value.squeeze(-1)
    
    def get_value(self, x):
        x = self.body(x)
        return self.value(x)
    
    def distribution(self, obs):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value
    
    def act(self, obs):
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        dist, value = self.distribution(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob.detach().squeeze(), value.detach().squeeze()

    def evaluate_actions(self, obs, actions):
        dist, values = self.distribution(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

class PPONet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64)):
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
            layer_init(nn.Linear(hidden_sizes[0], action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    

class ContinuousRPONet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64), rpo_alpha=0.5):
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
            layer_init(nn.Linear(hidden_sizes[0], action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        device = next(self.parameters()).device
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
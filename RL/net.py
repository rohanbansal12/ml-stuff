import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        in_dim = obs_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.model = nn.Sequential(*layers)
        self.logits = nn.Linear(in_dim, action_dim)

    def forward(self, obs):
        x = self.model(obs)
        return self.logits(x)
    
    def distribution(self, obs):
        x = self(obs)
        return torch.distributions.Categorical(logits=x)
    
    def act(self, obs):
        device = next(self.parameters()).device
        obs = torch.tensor(obs, device=device, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        x = self.distribution(obs)
        action = x.sample()
        return action.item(), x.log_prob(action).squeeze()
    

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


class ContinuousPPONet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        
        # --- Critic ---
        critic_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            critic_layers.append(layer_init(nn.Linear(in_dim, h)))
            critic_layers.append(nn.Tanh())
            in_dim = h
        # Critic output layer scaled to 1.0
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # --- Actor ---
        actor_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            actor_layers.append(layer_init(nn.Linear(in_dim, h)))
            actor_layers.append(nn.Tanh()) 
            in_dim = h
        # Actor output layer scaled to 0.01 -> starts with near-zero actions
        actor_layers.append(layer_init(nn.Linear(in_dim, action_dim), std=0.01))
        self.actor = nn.Sequential(*actor_layers)

        # State-independent log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        mu = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return mu, self.log_std, value

    def get_distribution(self, obs):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        base_dist = distributions.Normal(mu, std)
        
        # Apply the Tanh transformation
        # caching ensures we don't recompute the transform when getting log_probs later
        transforms = [distributions.transforms.TanhTransform(cache_size=1)]
        dist = distributions.TransformedDistribution(base_dist, transforms)
        
        return dist

    def act(self, obs):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Convert to tensor and fix dimensions
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)

            # Get distribution and value
            dist = self.get_distribution(obs_t)
            value = self.critic(obs_t).squeeze(-1)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

            return (
                action.cpu().numpy()[0], 
                log_prob.cpu().numpy()[0], 
                value.cpu().numpy()[0]
            )

    def evaluate_actions(self, obs, actions):
        dist = self.get_distribution(obs)
        value = self.critic(obs).squeeze(-1)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.base_dist.entropy().sum(-1)
        return log_probs, entropy, value
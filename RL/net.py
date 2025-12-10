import torch
import torch.nn as nn

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
    def __init__(self, obs_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        in_dim = obs_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.policy = nn.Linear(in_dim, action_dim)
        self.value = nn.Linear(in_dim, 1)

    def forward(self, obs):
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
        obs = torch.tensor(obs, device=device, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        dist, value = self.distribution(obs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

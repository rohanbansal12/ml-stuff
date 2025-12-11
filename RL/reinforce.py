import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from dataclasses import dataclass, field
import argparse
from net import PolicyNet

@dataclass
class Episode:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

    def total_return(self):
        return sum(self.rewards)
    
class ReinforceAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes, gamma, lr, device, norm_returns=False):
        self.policy = PolicyNet(obs_dim, action_dim, hidden_sizes).to(device)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.device = device
        self.norm_returns = norm_returns

    def generate_episode(self, env : gym.Env):
        eps = Episode()
        obs, _ = env.reset()
        while True:
            action, log_prob = self.policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            eps.observations.append(obs)
            eps.rewards.append(reward)
            eps.actions.append(action)
            eps.log_probs.append(log_prob)
            obs = next_obs
            if terminated or truncated:
                break
        return eps
    
    def compute_returns(self, rewards):
        G = 0.0
        res = [0] * len(rewards)
        for i in reversed(range(len(rewards))):
            G = self.gamma * G + rewards[i]
            res[i] = G
        return res

    def update_policy(self, episode: Episode):
        rewards = episode.rewards
        log_probs = torch.stack(episode.log_probs, dim=0).squeeze(-1)
        returns = torch.tensor(self.compute_returns(rewards), device=self.device)
        if self.norm_returns:
            std, mean = torch.std_mean(returns)
            returns = (returns - mean) / (std + 1e-9)
        loss = (-log_probs * returns).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--norm_returns", action='store_true', dest='norm_returns')
    parser.add_argument("--num_episodes", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_sizes = (128, 128)

    agent = ReinforceAgent(obs_dim, action_dim, hidden_sizes, gamma=args.gamma, lr=args.lr, device=device, norm_returns=args.norm_returns)

    print("\n=== REINFORCE Training Configuration ===")
    print(f"Environment:          {env_name}")
    print(f"Device:               {device}")
    print(f"Observation dim:      {obs_dim}")
    print(f"Action dim:           {action_dim}")
    print(f"Hidden sizes:         {hidden_sizes}")
    print(f"Learning rate:        {args.lr}")
    print(f"Gamma (discount):     {args.gamma}")
    print(f"Normalize returns:    {args.norm_returns}")
    print(f"Num episodes:         {args.num_episodes}")
    print("========================================\n")

    episode_returns = []
    window = 100  # rolling window for avg return

    for idx in range(args.num_episodes):
        episode = agent.generate_episode(env)
        loss = agent.update_policy(episode)
        ret = episode.total_return()
        episode_returns.append(ret)

        if (idx + 1) % 5 == 0:
            start = max(0, len(episode_returns) - window)
            recent = episode_returns[start:]
            avg_return = sum(recent) / len(recent)
            print(
                f"Episode {idx + 1:4d} | "
                f"Loss: {loss:.4f} | "
                f"Return: {ret:.2f} | "
                f"Avg Return (last {len(recent)}): {avg_return:.2f}"
            )

    env.close()
    

if __name__ == "__main__":
    main()
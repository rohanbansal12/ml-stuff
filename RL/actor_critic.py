import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from dataclasses import dataclass, field
from net import ActorCriticNet
import argparse

class RolloutBuffer:
    def __init__(self, size, obs_dim, device):
        self.observations = torch.zeros((size, obs_dim), device=device)
        self.actions = torch.zeros(size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(size, device=device)
        self.dones = torch.zeros(size, device=device)
        self.values = torch.zeros(size+1, device=device)
        self.log_probs = torch.zeros(size, device=device)
        self.advantages = torch.zeros(size, device=device)
        self.returns = torch.zeros(size, device=device) 

        self.ptr = 0
        self.max_size = size
        self.device = device

    def store(self, obs, action, reward, done, value, log_prob):
        assert not self.is_full()

        self.observations[self.ptr] = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value.detach().squeeze()
        self.log_probs[self.ptr] = log_prob.detach().squeeze()

        self.ptr += 1

    def is_full(self):
        return self.ptr == self.max_size
    
    def reset(self):
        self.ptr = 0

    def compute_advantages_and_returns(self, last_value, gamma, lam):
        self.values[self.ptr] = last_value.detach()
        deltas = self.rewards[:self.ptr] + gamma * self.values[1:self.ptr+1] * (1 - self.dones[:self.ptr]) - self.values[:self.ptr]
        G = 0.0
        for i in reversed(range(self.ptr)):
            G = (gamma * lam * (1 - self.dones[i])) * G + deltas[i]
            self.advantages[i] = G
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get(self):
        return (self.observations[:self.ptr], self.actions[:self.ptr], self.log_probs[:self.ptr], 
                self.values[:self.ptr], self.advantages[:self.ptr], self.returns[:self.ptr], self.dones[:self.ptr])

class ActorCriticAgent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes,
                 gamma,
                 lam,
                 lr,
                 device,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=None,
                 norm_advantages=True):
        
        self.actor_critic = ActorCriticNet(obs_dim, action_dim, hidden_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.norm_advantages = norm_advantages
        self.device = device

    def collect_rollout(self, env : gym.Env, rollout_steps, buffer : RolloutBuffer):
        obs, _ = env.reset()
        episode_returns = []
        episode_return = 0.0
        done = False

        for t in range(rollout_steps):
            action, log_prob, value = self.actor_critic.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward

            buffer.store(obs, action, reward, done, value, log_prob)
            if done:
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs
            
        if done:
            last_value = torch.zeros(1, device=self.device)
        else:
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            _, v = self.actor_critic(obs_tensor)
            last_value = v.squeeze()

        rollout_log = {
            "episode_returns": episode_returns,
            "num_episodes": len(episode_returns),
            "steps_collected": buffer.ptr
        }
        return last_value, rollout_log
    
    def update_parameters(self, buffer : RolloutBuffer):
        obs, actions, old_log_probs, values, advantages, returns, dones = buffer.get()
        if self.norm_advantages:
            std, mean = torch.std_mean(advantages)
            advantages = (advantages - mean) / (std+1e-9)

        logits, new_values = self.actor_critic(obs)
        distributions = torch.distributions.Categorical(logits=logits)
        new_log_probs = distributions.log_prob(actions)
        entropy = distributions.entropy()


        policy_loss = -(new_log_probs * advantages).mean()
        value_loss = (new_values - returns).square().mean()
        entropy_loss = entropy.mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return  {
                "loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_loss.item()
                }
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--norm_advantages", action='store_true', dest='norm_advantages')
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--rollout_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_sizes = (128, 128)

    agent = ActorCriticAgent(obs_dim, action_dim, hidden_sizes, args.gamma, args.lam, args.lr, device, args.value_coef, args.entropy_coef, .5, args.norm_advantages)
    buffer = RolloutBuffer(args.rollout_size, obs_dim, device)

    print("\n=== Experiment Configuration ===")
    print(f"Environment:           {env_name}")
    print(f"Observation dim:       {obs_dim}")
    print(f"Action dim:            {action_dim}")
    print(f"Hidden sizes:          {hidden_sizes}")
    print(f"Learning rate:         {args.lr}")
    print(f"Gamma (discount):      {args.gamma}")
    print(f"Lambda (GAE):          {args.lam}")
    print(f"Value loss coef:       {args.value_coef}")
    print(f"Entropy coef:          {args.entropy_coef}")
    print(f"Normalize advantages:  {args.norm_advantages}")
    print(f"Rollout size:          {args.rollout_size}")
    print(f"Training iterations:   {args.num_steps}")
    print(f"Device:                {device}")
    print("================================\n")

    episode_returns = []
    window = 100

    for idx in range(args.num_steps):
        buffer.reset()
        last_value, rollout_log = agent.collect_rollout(env, args.rollout_size, buffer)
        buffer.compute_advantages_and_returns(last_value, args.gamma, args.lam)
        for _ in range(2):
            optim_logs = agent.update_parameters(buffer)

        episode_returns.extend(rollout_log['episode_returns'])

        if (idx + 1) % 5 == 0 and len(episode_returns) > 0:
            start = max(0, len(episode_returns) - window)
            recent = episode_returns[start:]
            avg_return = sum(recent) / len(recent)
            print(
                f"Rollout {idx + 1:4d} | "
                f"Loss: {optim_logs['loss']:.4f} | ",
                f"Entropy: {optim_logs['entropy']:.4f} | ",
                f"Avg Return (last {len(recent)}): {avg_return:.2f}"
            )

    env.close()

if __name__ == "__main__":
    main()
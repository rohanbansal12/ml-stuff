import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from dataclasses import dataclass, field
from net import ActorCriticNet
from util import RolloutBuffer
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

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
        self.actor_critic = torch.compile(self.actor_critic)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.norm_advantages = norm_advantages
        self.device = device
        
        self.next_obs = None
        self.env_rewards = None

    def collect_rollout(self, env : gym.vector.VectorEnv, rollout_steps, buffer : RolloutBuffer):
        if self.next_obs is None:
            self.next_obs, _ = env.reset(seed=42)
            self.env_rewards = np.zeros(env.num_envs)

        finished_episode_returns = []

        for t in range(rollout_steps):
            # 1. Predict
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self.next_obs, device=self.device, dtype=torch.float32)
                action, log_prob, value = self.actor_critic.act(obs_tensor)

            # 2. Step
            cpu_actions = action.cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = env.step(cpu_actions)
            dones = np.logical_or(terminations, truncations)

            # 3. Handle Logging (Manual accumulation)
            self.env_rewards += rewards
            for i, done in enumerate(dones):
                if done:
                    finished_episode_returns.append(self.env_rewards[i])
                    self.env_rewards[i] = 0.0

            # 4. Store (using the obs we started with)
            buffer.store(self.next_obs, action, rewards, dones, value, log_prob)
            
            # 5. Advance
            self.next_obs = next_obs
            
        # 6. Bootstrap value for GAE
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.next_obs, device=self.device, dtype=torch.float32)
            last_value = self.actor_critic.get_value(obs_tensor)
            last_value = last_value.squeeze()

        rollout_log = {
            "episode_returns": finished_episode_returns,
            "num_episodes": len(finished_episode_returns),
            "steps_collected": buffer.ptr * buffer.num_envs
        }
        return last_value, rollout_log
    
    def update_parameters(self, buffer : RolloutBuffer):
        # 1. Get Batched Data
        obs, actions, old_log_probs, values, advantages, returns, dones = buffer.get()

        # 2. Normalize Advantages
        if self.norm_advantages:
            std, mean = torch.std_mean(advantages)
            advantages = (advantages - mean) / (std + 1e-9)

        # 3. Evaluate (Gradient Step)
        logits, new_values = self.actor_critic(obs)
        new_values = new_values.squeeze()
        
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # 4. Loss Calculation
        policy_loss = -(new_log_probs * advantages).mean()
        value_loss = (new_values - returns).square().mean()
        entropy_loss = entropy.mean()

        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        # 5. Optimize
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
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--norm_advantages", action='store_true', dest='norm_advantages')
    
    # Training duration
    parser.add_argument("--total_updates", type=int, default=15000, help="Number of A2C updates")
    parser.add_argument("--rollout_size", type=int, default=80, help="Total steps per update (across all envs)")
    parser.add_argument("--log-dir", type=str, default="./runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv

    # 1. Vector Environment Setup
    env_name = "CartPole-v1"
    num_envs = 16

    def make_env():
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    env = SyncVectorEnv([make_env for _ in range(num_envs)])
    
    # Adjust rollout size to be per-env
    args.rollout_size = args.rollout_size // num_envs
    
    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n
    hidden_sizes = (64, 64)

    agent = ActorCriticAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        device=device,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=0.5,
        norm_advantages=args.norm_advantages
    )
    
    buffer = RolloutBuffer(
        steps=args.rollout_size, 
        num_envs=num_envs, 
        obs_dim=obs_dim, 
        device=device,
        cont=False
    )

    run_name = f"a2c_lr={args.lr}_gamma={args.gamma}_lam={args.lam}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)
    
    writer.add_text("hparams", str(vars(args)))


    print("\n=== A2C Parallel Experiment Configuration ===")
    print(f"Environment:           {env_name}")
    print(f"Num Envs:              {num_envs}")
    print(f"Obs/Action dim:        {obs_dim} / {action_dim}")
    print(f"Total Batch Size:      {args.rollout_size * num_envs}")
    print(f"Learning rate:         {args.lr}")
    print(f"Normalize adv:         {args.norm_advantages}")
    print("=============================================\n")

    episode_returns = []
    window = 100

    for idx in range(args.total_updates):
        buffer.reset()
        
        # 1. Collect (Parallel)
        last_value, rollout_log = agent.collect_rollout(env, args.rollout_size, buffer)
        
        # 2. GAE
        buffer.compute_advantages_and_returns(last_value, args.gamma, args.lam)
        
        # 3. Update (A2C usually does 1 epoch, PPO does many)
        optim_logs = agent.update_parameters(buffer)

        # 4. Logging
        episode_returns.extend(rollout_log['episode_returns'])

        if len(episode_returns) > 0:
            start = max(0, len(episode_returns) - window)
            recent = episode_returns[start:]
            avg_return = sum(recent) / len(recent)
        else:
            avg_return = 0.0

        if (idx + 1) % 10 == 0:
            print(
                f"Update {idx + 1:4d} | "
                f"AvgRet(last {len(recent):3d}): {avg_return:7.2f} | "
                f"Loss: {optim_logs['loss']:.4f} | "
                f"Ent: {optim_logs['entropy']:.4f} "
            )
            writer.add_scalar("Loss", optim_logs['loss'], idx+1)
            writer.add_scalar("Avg_ret", avg_return, idx+1)
            writer.add_scalar("Entropy", optim_logs['entropy'], idx+1)

    env.close()

if __name__ == "__main__":
    main()
import torch
from dataclasses import dataclass, field
import argparse
import gymnasium as gym
from net import ActorCriticNet
import numpy as np

class PPORolloutBuffer:
    def __init__(self, steps, num_envs, obs_dim, device, act_dim=0, cont=False):
        # We add num_envs to the shape initialization
        self.observations = torch.zeros((steps, num_envs, obs_dim), device=device)

        if cont:
            self.actions = torch.zeros((steps, num_envs, act_dim), device=device)
        else:
            self.actions = torch.zeros((steps, num_envs), dtype=torch.long, device=device)

        self.rewards = torch.zeros((steps, num_envs), device=device)
        self.dones = torch.zeros((steps, num_envs), device=device)
        self.values = torch.zeros((steps + 1, num_envs), device=device)
        self.log_probs = torch.zeros((steps, num_envs), device=device)
        self.advantages = torch.zeros((steps, num_envs), device=device)
        self.returns = torch.zeros((steps, num_envs), device=device) 

        self.ptr = 0
        self.max_steps = steps
        self.num_envs = num_envs
        self.device = device
        self.cont = cont

    def store(self, obs, action, reward, done, value, log_prob):
        assert not self.is_full()

        # obs is already (num_envs, obs_dim)
        self.observations[self.ptr] = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        
        if self.cont:
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        self.dones[self.ptr] = torch.as_tensor(done, device=self.device, dtype=torch.float32)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1

    def is_full(self):
        return self.ptr == self.max_steps
    
    def reset(self):
        self.ptr = 0

    def compute_advantages_and_returns(self, last_value, gamma, lam):
        # last_value shape: (num_envs,)
        self.values[self.ptr] = last_value.detach()

        rewards = self.rewards[:self.ptr]
        values_t = self.values[:self.ptr]
        values_t_plus_1 = self.values[1:self.ptr+1]
        dones = self.dones[:self.ptr]

        # Calculate deltas. Shapes match: (ptr, num_envs)
        deltas = rewards + gamma * values_t_plus_1 * (1.0 - dones) - values_t

        gamma_lam = gamma * lam
        factors = gamma_lam * (1.0 - dones)
        
        # Flip time dimension (dim 0) for recursion
        factors_rev = factors.flip(0)
        deltas_rev = deltas.flip(0)
        adv_rev = torch.zeros_like(deltas_rev)
        
        current_sum = torch.zeros(self.num_envs, device=self.device)
        
        for i in range(self.ptr):
            current_sum = deltas_rev[i] + factors_rev[i] * current_sum
            adv_rev[i] = current_sum

        self.advantages[:self.ptr] = adv_rev.flip(0) 
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def flatten_data(self):
        # Flattens (steps, num_envs, ...) -> (steps * num_envs, ...)
        # We use .reshape(-1, ...) to combine the first two dimensions
        b_obs = self.observations[:self.ptr].reshape((-1,) + self.observations.shape[2:])
        b_actions = self.actions[:self.ptr].reshape((-1,) + self.actions.shape[2:])
        b_log_probs = self.log_probs[:self.ptr].reshape(-1)
        b_advantages = self.advantages[:self.ptr].reshape(-1)
        b_returns = self.returns[:self.ptr].reshape(-1)
        b_values = self.values[:self.ptr].reshape(-1)
        b_dones = self.dones[:self.ptr].reshape(-1)
        
        return b_obs, b_actions, b_log_probs, b_values, b_advantages, b_returns, b_dones

    def get(self):
        return self.flatten_data()
    
    def iter_minibatches(self, num_minibatches):
        # 1. Get flattened data
        obs, actions, log_probs, values, advantages, returns, dones = self.flatten_data()
        
        # 2. Indices for the flattened data
        batch_size = obs.shape[0]
        idx = torch.randperm(batch_size, device=self.device)
        
        # 3. Yield chunks
        chunks = torch.chunk(idx, num_minibatches)
        for chunk in chunks:
            yield (obs[chunk], actions[chunk], log_probs[chunk], 
                   values[chunk], advantages[chunk], returns[chunk], dones[chunk])
            
@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2          # policy clip
    value_clip_eps: float = 0.2    # optional value clip
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    train_epochs: int = 10         # K epochs per batch
    num_minibatches: int = 4       # split rollout into this many minibatches
    max_grad_norm: float | None = 0.5
    target_kl: float | None = 0.01 # for early stopping per batch
    rollout_size: int = 2048       # total steps per update
    norm_advantages: bool = True

class PPOAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes, config: PPOConfig, device):
        self.config = config
        self.device = device
        self.actor_critic = ActorCriticNet(obs_dim, action_dim, hidden_sizes).to(device)
        self.actor_critic = torch.compile(self.actor_critic)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=config.lr)
        self.env_rewards = None

    def collect_rollout(self, env : gym.Env, steps_per_env, buffer : PPORolloutBuffer):
        obs, _ = env.reset(seed=42)

        if self.env_rewards is None:
            self.env_rewards = np.zeros(env.num_envs)
        finished_episode_returns = []

        for t in range(steps_per_env):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                action, log_prob, value = self.actor_critic.act(obs_tensor)

            cpu_actions = action.cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = env.step(cpu_actions)
            dones = np.logical_or(terminations, truncations)
            self.env_rewards += rewards

            for i, done in enumerate(dones):
                if done:
                    finished_episode_returns.append(self.env_rewards[i])
                    self.env_rewards[i] = 0.0

            buffer.store(obs, action, rewards, dones, value, log_prob)
            obs = next_obs
            
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            _, last_value = self.actor_critic(obs_tensor)
            last_value = last_value.squeeze()
                
        rollout_log = {
            "episode_returns": finished_episode_returns,
            "num_episodes": len(finished_episode_returns),
            "steps_collected": buffer.ptr * buffer.num_envs 
        }
        
        return last_value, rollout_log

    def update_parameters(self, buffer: PPORolloutBuffer):
        # 1. Get all data (Flattened)
        # These are now 1D tensors of size (steps * num_envs)
        obs, actions, log_probs, values, advantages, returns, dones = buffer.get()

        # 2. Normalize Advantages (Crucial Step)
        if self.config.norm_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # 3. Prepare for Mini-batching
        # We know the total batch size now
        batch_size = obs.shape[0] 
        minibatch_size = batch_size // self.config.num_minibatches
        
        # Logging accumulators
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        clipfrac_sum = 0.0
        num_updates = 0
        approx_kl_last = 0.0

        # 4. PPO Epoch Loop
        for epoch in range(self.config.train_epochs):
            # Shuffle indices for this epoch
            b_inds = torch.randperm(batch_size, device=self.device)

            # 5. Mini-batch Loop
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Slice the pre-normalized data using the random indices
                obs_mb = obs[mb_inds]
                actions_mb = actions[mb_inds]
                old_log_probs_mb = log_probs[mb_inds]
                values_old_mb = values[mb_inds]
                advantages_mb = advantages[mb_inds] # These are now CORRECTLY normalized
                returns_mb = returns[mb_inds]

                # --- The rest is identical to your original PPO logic ---
                new_log_probs, entropy, new_values = self.actor_critic.evaluate_actions(obs_mb, actions_mb)

                approx_kl = torch.mean(old_log_probs_mb - new_log_probs).item()
                approx_kl_last = approx_kl

                ratio = torch.exp(new_log_probs - old_log_probs_mb)
                unclipped = ratio * advantages_mb
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantages_mb
                
                policy_loss = -torch.min(clipped, unclipped).mean()

                # Value Loss (Optional Clipping)
                # Note: I'm assuming you fixed the value clipping issue we discussed earlier
                # by either setting large epsilon or using unclipped loss
                value_loss = 0.5 * ((new_values - returns_mb) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Logging
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()
                    total_loss_sum += loss.item()
                    policy_loss_sum += policy_loss.item()
                    value_loss_sum += value_loss.item()
                    entropy_sum += entropy_loss.item()
                    approx_kl_sum += approx_kl
                    clipfrac_sum += clipfrac
                    num_updates += 1
                
                # KL Early stop (Batch level)
                if self.config.target_kl and approx_kl > self.config.target_kl:
                    break
            
            # KL Early stop (Epoch level)
            if self.config.target_kl and approx_kl_last > self.config.target_kl:
                break

        # Averages for logging
        if num_updates > 0:
            return {
                "loss": total_loss_sum / num_updates,
                "policy_loss": policy_loss_sum / num_updates,
                "value_loss": value_loss_sum / num_updates,
                "entropy": entropy_sum / num_updates,
                "approx_kl": approx_kl_sum / num_updates,
                "clipfrac": clipfrac_sum / num_updates,
                "num_updates": num_updates,
            }
        return {}

def main():
    parser = argparse.ArgumentParser()

    # Environment / training horizon
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--total_updates", type=int, default=500, help="Number of PPO updates (outer loop)")
    parser.add_argument("--rollout_size", type=int, default=2048, help="Steps per PPO update")

    # PPO core hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_clip_eps", type=float, default=1000)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    # PPO optimization loop
    parser.add_argument("--train_epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--num_minibatches", type=int, default=4, help="Minibatches per epoch")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.01, help="Early stop threshold on approx KL (set 0 or None to disable)")

    # Advantage / network settings
    parser.add_argument("--norm_advantages", action="store_true", dest="norm_advantages")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for MLP (used twice: [h, h])")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv

    def make_env():
        env = gym.make(args.env_name, max_episode_steps=500)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    # Make environment
    num_envs = 16  # Or 32, depending on your CPU cores
    env = SyncVectorEnv([make_env for _ in range(num_envs)])
    args.rollout_size = args.rollout_size // num_envs

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n
    hidden_sizes = (args.hidden_size, args.hidden_size)

    # Build PPO config
    config = PPOConfig(
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        clip_eps=args.clip_eps,
        value_clip_eps=args.value_clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        train_epochs=args.train_epochs,
        num_minibatches=args.num_minibatches,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        rollout_size=args.rollout_size,
        norm_advantages=args.norm_advantages,
    )

    # Agent
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        config=config,
        device=device,
    )
    
    # Optional: Compile for speed
    # agent.actor_critic = torch.compile(agent.actor_critic)

    # Buffer (Updated for vector envs)
    buffer = PPORolloutBuffer(
        steps=config.rollout_size, 
        num_envs=num_envs, 
        obs_dim=obs_dim, 
        device=device
    )

    # Print experiment configuration
    print("\n=== PPO CartPole Experiment Configuration ===")
    print(f"Environment:          {args.env_name}")
    print(f"Num Envs:             {num_envs}")
    print(f"Observation dim:      {obs_dim}")
    print(f"Action dim:           {action_dim}")
    print(f"Hidden sizes:         {hidden_sizes}")
    print(f"Total updates:        {args.total_updates}")
    print(f"Steps per Env:        {config.rollout_size}")
    print(f"Total Batch Size:     {config.rollout_size * num_envs}")
    print(f"Learning rate:        {config.lr}")
    print(f"Device:               {device}")
    print("================================================\n")

    episode_returns = []
    window = 100 

    for update_idx in range(args.total_updates):
        buffer.reset()

        # 1) Collect rollout
        last_value, rollout_log = agent.collect_rollout(env, config.rollout_size, buffer)

        # 2) Compute GAE + returns in buffer
        buffer.compute_advantages_and_returns(last_value, config.gamma, config.lam)

        # 3) PPO update on this rollout
        optim_logs = agent.update_parameters(buffer)

        # 4) Track episode returns
        episode_returns.extend(rollout_log["episode_returns"])

        # 5) Logging
        if len(episode_returns) > 0:
            start = max(0, len(episode_returns) - window)
            recent = episode_returns[start:]
            avg_return = sum(recent) / len(recent)
        else:
            avg_return = 0.0
            recent = []

        if (update_idx + 1) % 5 == 0:
            print(
                f"Update {update_idx + 1:4d} | "
                f"AvgRet(last {len(recent):3d}): {avg_return:7.2f} | "
                f"Loss: {optim_logs.get('loss', 0):.4f} | "
                f"Pol: {optim_logs.get('policy_loss', 0):.4f} | "
                f"Val: {optim_logs.get('value_loss', 0):.4f} | "
                f"Ent: {optim_logs.get('entropy', 0):.4f} | "
                f"KL: {optim_logs.get('approx_kl', 0):.5f} | "
                f"ClipFrac: {optim_logs.get('clipfrac', 0):.3f} | "
                f"UpdatesThisBatch: {optim_logs.get('num_updates', 0):2d}"
            )

    env.close()


if __name__ == "__main__":
    main()
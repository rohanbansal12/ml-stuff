import torch
from dataclasses import dataclass, field
from net import PPONet, ContinuousRPONet
import gymnasium as gym
import numpy as np

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
    seed: int = 1
    rpo_alpha: float = 0.5

class RolloutBuffer:
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
    
class PPOAgent:
    def __init__(self, model: PPONet | ContinuousRPONet, obs_dim, action_dim, hidden_sizes, config: PPOConfig, device):
        self.config = config
        self.device = device
        self.actor_critic = model(obs_dim, action_dim, hidden_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=config.lr)
        self.next_obs = None

    def collect_rollout(self, env : gym.Env, steps_per_env, buffer : RolloutBuffer):
        if self.next_obs is None:
            self.next_obs, _ = env.reset(seed=self.config.seed)

        finished_episode_returns = []

        for t in range(steps_per_env):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self.next_obs, device=self.device, dtype=torch.float32)
                action, log_prob, _, value = self.actor_critic.get_action_and_value(obs_tensor)
                value = value.flatten()
                
            obs_tensor = self.next_obs.copy()
            cpu_actions = action.cpu().numpy()
            self.next_obs, rewards, terminations, truncations, infos = env.step(cpu_actions)
            dones = np.logical_or(terminations, truncations)
            buffer.store(obs_tensor, action, rewards, dones, value, log_prob)

            if "_episode" in infos:
                # indices where the environment actually finished
                indices = np.where(infos["_episode"])[0]
                for i in indices:
                    raw_return = infos["episode"]["r"][i]
                    finished_episode_returns.append(raw_return)
            
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.next_obs, device=self.device, dtype=torch.float32)
            last_value = self.actor_critic.get_value(obs_tensor).squeeze(-1)
                
        rollout_log = {
            "episode_returns": finished_episode_returns,
            "num_episodes": len(finished_episode_returns),
            "steps_collected": buffer.ptr * buffer.num_envs 
        }
        
        return last_value, rollout_log

    def update_parameters(self, buffer: RolloutBuffer):
        # 1. Get all data (Flattened)
        obs, actions, log_probs, values, advantages, returns, dones = buffer.get()

        # 2. Normalize Advantages (Crucial Step)
        if self.config.norm_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # 3. Prepare for Mini-batching
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

                _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(obs_mb, actions_mb)

                logratio = new_log_probs - old_log_probs_mb
                ratio = logratio.exp()
                approx_kl = ((ratio - 1) - logratio).mean()

                # Policy Loss
                pg_loss_unclipped = ratio * advantages_mb
                pg_loss_clipped = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantages_mb
                policy_loss = -torch.min(pg_loss_unclipped, pg_loss_clipped).mean()
                
                # Value Loss
                new_values = new_values.flatten()
                if self.config.value_clip_eps > 0:
                    v_loss_unclipped = (new_values - returns_mb) ** 2
                    clipped_values = returns_mb + torch.clamp(new_values - returns_mb, -self.config.value_clip_eps, self.config.value_clip_eps)
                    v_loss_clipped = (clipped_values - returns_mb) ** 2
                    value_loss = .5 * torch.max(v_loss_clipped, v_loss_unclipped).mean()
                else:
                    value_loss = .5 * ((new_values - returns_mb) ** 2).mean()

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
            
            # KL Early stop
            if self.config.target_kl and approx_kl > self.config.target_kl:
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
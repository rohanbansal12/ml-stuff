import torch
from dataclasses import dataclass, field

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
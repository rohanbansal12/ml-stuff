import argparse
import os
from collections import deque
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from net import ActorCriticNet, PPONet
from torch.utils.tensorboard import SummaryWriter
from util import RolloutBuffer


@dataclass
class Config:
    """PPO training configuration."""

    # Core PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2  # Set to 0 to disable value clipping
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    norm_advantages: bool = True

    # PPO-specific
    num_epochs: int = 4  # K epochs per rollout
    num_minibatches: int = 4
    target_kl: float | None = None  # Early stopping threshold (None to disable)

    # Architecture
    hidden_sizes: tuple = (64, 64)
    separate_networks: bool = True  # Separate networks tend to be more stable

    # Environment
    env_name: str = "CartPole-v1"
    num_envs: int = 4

    # Training duration
    total_timesteps: int = 500_000
    rollout_steps: int = 32  # Smaller rollouts = more frequent updates

    # Evaluation
    eval_interval: int = 10  # Evaluate every N updates
    eval_episodes: int = 10

    # Misc
    seed: int = 42
    log_dir: str = "./runs"
    compile_model: bool = False

    def to_dict(self):
        return asdict(self)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()

        # Use store_true/store_false pattern for boolean flags with proper defaults
        parser.add_argument("--lr", type=float, default=cls.lr)
        parser.add_argument("--gamma", type=float, default=cls.gamma)
        parser.add_argument("--lam", type=float, default=cls.lam)
        parser.add_argument("--clip_eps", type=float, default=cls.clip_eps)
        parser.add_argument("--value_clip_eps", type=float, default=cls.value_clip_eps)
        parser.add_argument("--value_coef", type=float, default=cls.value_coef)
        parser.add_argument("--entropy_coef", type=float, default=cls.entropy_coef)
        parser.add_argument("--max_grad_norm", type=float, default=cls.max_grad_norm)

        # Boolean flags with proper mutual exclusion
        adv_group = parser.add_mutually_exclusive_group()
        adv_group.add_argument("--norm_advantages", action="store_true", dest="norm_advantages")
        adv_group.add_argument("--no_norm_advantages", action="store_false", dest="norm_advantages")
        parser.set_defaults(norm_advantages=cls.norm_advantages)

        parser.add_argument("--num_epochs", type=int, default=cls.num_epochs)
        parser.add_argument("--num_minibatches", type=int, default=cls.num_minibatches)
        parser.add_argument("--target_kl", type=float, default=cls.target_kl)
        parser.add_argument("--hidden_sizes", type=int, nargs="+", default=list(cls.hidden_sizes))
        parser.add_argument("--separate_networks", action="store_true")
        parser.add_argument("--env_name", type=str, default=cls.env_name)
        parser.add_argument("--num_envs", type=int, default=cls.num_envs)
        parser.add_argument("--total_timesteps", type=int, default=cls.total_timesteps)
        parser.add_argument("--rollout_steps", type=int, default=cls.rollout_steps)
        parser.add_argument("--eval_interval", type=int, default=cls.eval_interval)
        parser.add_argument("--eval_episodes", type=int, default=cls.eval_episodes)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--log_dir", type=str, default=cls.log_dir)
        parser.add_argument("--compile", action="store_true", dest="compile_model")
        args = parser.parse_args()

        return cls(
            lr=args.lr,
            gamma=args.gamma,
            lam=args.lam,
            clip_eps=args.clip_eps,
            value_clip_eps=args.value_clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            norm_advantages=args.norm_advantages,
            num_epochs=args.num_epochs,
            num_minibatches=args.num_minibatches,
            target_kl=args.target_kl,
            hidden_sizes=tuple(args.hidden_sizes),
            separate_networks=args.separate_networks,
            env_name=args.env_name,
            num_envs=args.num_envs,
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            log_dir=args.log_dir,
            compile_model=args.compile_model,
        )


class Tracker:
    """Handles metric tracking and TensorBoard logging."""

    def __init__(self, log_dir: str, config: Config, window_size: int = 100):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text("config", str(config.to_dict()))

        self.window_size = window_size
        self.episode_returns = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)

        self.total_steps = 0
        self.total_updates = 0
        self.total_episodes = 0

    def log_rollout(self, episode_returns: list[float], episode_lengths: list[int], steps: int):
        """Log metrics from a rollout."""
        self.total_steps += steps

        for ret, length in zip(episode_returns, episode_lengths, strict=False):
            self.total_episodes += 1
            self.episode_returns.append(ret)
            self.episode_lengths.append(length)

            self.writer.add_scalar("train_steps/episode_return", ret, self.total_steps)
            self.writer.add_scalar("train_steps/episode_length", length, self.total_steps)

    def log_update(self, loss_dict: dict):
        """Log metrics from a parameter update."""
        self.total_updates += 1

        # Compute rolling averages
        if len(self.episode_returns) > 0:
            avg_return = sum(self.episode_returns) / len(self.episode_returns)
            avg_length = sum(self.episode_lengths) / len(self.episode_lengths)
        else:
            avg_return = 0.0
            avg_length = 0.0

        # Log by update
        self.writer.add_scalar("train/avg_return", avg_return, self.total_updates)
        self.writer.add_scalar("train/avg_length", avg_length, self.total_updates)
        self.writer.add_scalar("train/loss", loss_dict["loss"], self.total_updates)
        self.writer.add_scalar("train/policy_loss", loss_dict["policy_loss"], self.total_updates)
        self.writer.add_scalar("train/value_loss", loss_dict["value_loss"], self.total_updates)
        self.writer.add_scalar("train/entropy", loss_dict["entropy"], self.total_updates)
        self.writer.add_scalar("train/approx_kl", loss_dict["approx_kl"], self.total_updates)
        self.writer.add_scalar(
            "train/clip_fraction", loss_dict["clip_fraction"], self.total_updates
        )
        self.writer.add_scalar(
            "train/explained_variance",
            loss_dict["explained_variance"],
            self.total_updates,
        )

        # Log by steps
        self.writer.add_scalar("train_steps/avg_return", avg_return, self.total_steps)
        self.writer.add_scalar("train_steps/loss", loss_dict["loss"], self.total_steps)

        return avg_return, avg_length

    def log_eval(self, mean_return: float, std_return: float, mean_length: float):
        """Log evaluation metrics."""
        self.writer.add_scalar("eval/mean_return", mean_return, self.total_updates)
        self.writer.add_scalar("eval/std_return", std_return, self.total_updates)
        self.writer.add_scalar("eval/mean_length", mean_length, self.total_updates)
        self.writer.add_scalar("eval_steps/mean_return", mean_return, self.total_steps)

    def close(self):
        self.writer.close()


class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: Config, device: torch.device):
        self.config = config
        self.device = device

        # Choose network architecture
        if config.separate_networks:
            self.actor_critic = PPONet(obs_dim, action_dim, config.hidden_sizes).to(device)
        else:
            self.actor_critic = ActorCriticNet(obs_dim, action_dim, config.hidden_sizes).to(device)

        if config.compile_model:
            self.actor_critic = torch.compile(self.actor_critic)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=config.lr, eps=1e-5)

        # State for vectorized rollouts
        self.next_obs = None
        self.env_rewards = None
        self.env_lengths = None

    def collect_rollout(self, env: SyncVectorEnv, buffer: RolloutBuffer) -> dict:
        """Collect a rollout from the vectorized environment."""
        if self.next_obs is None:
            self.next_obs, _ = env.reset(seed=self.config.seed)
            self.env_rewards = np.zeros(env.num_envs)
            self.env_lengths = np.zeros(env.num_envs, dtype=np.int32)

        finished_returns = []
        finished_lengths = []

        for _ in range(self.config.rollout_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self.next_obs, device=self.device, dtype=torch.float32)

                if self.config.separate_networks:
                    action, log_prob, _, value = self.actor_critic.get_action_and_value(obs_tensor)
                    value = value.squeeze(-1)
                else:
                    action, log_prob, value = self.actor_critic.act(obs_tensor)

            cpu_actions = action.cpu().numpy()
            next_obs, rewards, terminations, truncations, _ = env.step(cpu_actions)
            dones = np.logical_or(terminations, truncations)

            # Track episode statistics
            self.env_rewards += rewards
            self.env_lengths += 1

            for i, done in enumerate(dones):
                if done:
                    finished_returns.append(self.env_rewards[i])
                    finished_lengths.append(self.env_lengths[i])
                    self.env_rewards[i] = 0.0
                    self.env_lengths[i] = 0

            buffer.store(self.next_obs, action, rewards, dones, value, log_prob)
            self.next_obs = next_obs

        # Bootstrap value for incomplete trajectories
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.next_obs, device=self.device, dtype=torch.float32)
            if self.config.separate_networks:
                last_value = self.actor_critic.get_value(obs_tensor).squeeze(-1)
            else:
                last_value = self.actor_critic.get_value(obs_tensor).squeeze()

        buffer.compute_advantages_and_returns(last_value, self.config.gamma, self.config.lam)

        return {
            "episode_returns": finished_returns,
            "episode_lengths": finished_lengths,
            "steps": self.config.rollout_steps * env.num_envs,
        }

    def update(self, buffer: RolloutBuffer) -> dict:
        """Perform PPO update with multiple epochs and minibatches."""
        obs, actions, old_log_probs, old_values, advantages, returns, _ = buffer.get()

        # Normalize advantages
        if self.config.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute explained variance before updates
        with torch.no_grad():
            explained_var = 1 - (returns - old_values).var() / (returns.var() + 1e-8)

        batch_size = obs.shape[0]
        minibatch_size = self.config.minibatch_size

        # Accumulators for logging
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        clip_fraction_sum = 0.0
        num_gradient_steps = 0
        final_epoch = 0

        for epoch in range(self.config.num_epochs):
            final_epoch = epoch
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Slice minibatch
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_old_values = old_values[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Forward pass
                if self.config.separate_networks:
                    _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
                        mb_obs, mb_actions
                    )
                    new_values = new_values.squeeze(-1)
                else:
                    new_log_probs, entropy, new_values = self.actor_critic.evaluate_actions(
                        mb_obs, mb_actions
                    )

                # Policy loss with clipping
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with optional clipping
                if self.config.value_clip_eps > 0:
                    # Clip value predictions around old values (not returns!)
                    clipped_values = mb_old_values + torch.clamp(
                        new_values - mb_old_values,
                        -self.config.value_clip_eps,
                        self.config.value_clip_eps,
                    )
                    value_loss_unclipped = (new_values - mb_returns).square()
                    value_loss_clipped = (clipped_values - mb_returns).square()
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (new_values - mb_returns).square().mean()

                entropy_loss = entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.config.max_grad_norm
                    )
                self.optimizer.step()

                # Logging metrics
                with torch.no_grad():
                    # Approx KL divergence (http://joschu.net/blog/kl-approx.html)
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()
                    )

                total_loss_sum += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy_loss.item()
                approx_kl_sum += approx_kl
                clip_fraction_sum += clip_fraction
                num_gradient_steps += 1

            # Early stopping based on KL divergence (check after each epoch)
            if self.config.target_kl is not None:
                # Compute average KL for this epoch
                epoch_kl = approx_kl_sum / num_gradient_steps if num_gradient_steps > 0 else 0
                if epoch_kl > self.config.target_kl:
                    break

        return {
            "loss": total_loss_sum / num_gradient_steps if num_gradient_steps > 0 else 0,
            "policy_loss": policy_loss_sum / num_gradient_steps if num_gradient_steps > 0 else 0,
            "value_loss": value_loss_sum / num_gradient_steps if num_gradient_steps > 0 else 0,
            "entropy": entropy_sum / num_gradient_steps if num_gradient_steps > 0 else 0,
            "approx_kl": approx_kl_sum / num_gradient_steps if num_gradient_steps > 0 else 0,
            "clip_fraction": clip_fraction_sum / num_gradient_steps
            if num_gradient_steps > 0
            else 0,
            "explained_variance": explained_var.item(),
            "num_epochs": final_epoch + 1,
        }

    def evaluate(self, env: gym.Env, num_episodes: int) -> tuple[list[float], list[int]]:
        """Run deterministic evaluation episodes."""
        returns = []
        lengths = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

            while True:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

                    if self.config.separate_networks:
                        logits = self.actor_critic.actor(obs_tensor.unsqueeze(0))
                    else:
                        logits, _ = self.actor_critic(obs_tensor)

                    action = logits.argmax(dim=-1).item()

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                episode_length += 1

                if terminated or truncated:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)

        return returns, lengths


def print_config(config: Config, device: torch.device, obs_dim: int, action_dim: int):
    """Print training configuration."""
    print("\n" + "=" * 50)
    print("PPO Training Configuration")
    print("=" * 50)
    for key, value in config.to_dict().items():
        print(f"  {key:20s}: {value}")
    print(f"  {'device':20s}: {device}")
    print(f"  {'obs_dim':20s}: {obs_dim}")
    print(f"  {'action_dim':20s}: {action_dim}")
    print(f"  {'batch_size':20s}: {config.batch_size}")
    print(f"  {'minibatch_size':20s}: {config.minibatch_size}")
    print(f"  {'num_updates':20s}: {config.num_updates}")
    print("=" * 50 + "\n")


def main():
    config = Config.from_args()

    # Seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Create environments
    def make_env():
        return gym.make(config.env_name)

    env = SyncVectorEnv([make_env for _ in range(config.num_envs)])
    eval_env = gym.make(config.env_name)

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n

    print_config(config, device, obs_dim, action_dim)

    # Initialize agent, buffer, tracker
    agent = PPOAgent(obs_dim, action_dim, config, device)

    buffer = RolloutBuffer(
        steps=config.rollout_steps,
        num_envs=config.num_envs,
        obs_dim=obs_dim,
        device=device,
        cont=False,
    )

    run_name = f"ppo_{config.env_name}_lr={config.lr}_clip={config.clip_eps}_separate={config.separate_networks}_seed={config.seed}"
    tracker = Tracker(os.path.join(config.log_dir, run_name), config)

    # Training loop
    for update_idx in range(config.num_updates):
        buffer.reset()

        # Collect rollout
        rollout_info = agent.collect_rollout(env, buffer)

        # Update
        loss_dict = agent.update(buffer)

        # Log
        tracker.log_rollout(
            rollout_info["episode_returns"],
            rollout_info["episode_lengths"],
            rollout_info["steps"],
        )
        avg_return, avg_length = tracker.log_update(loss_dict)

        # Print progress
        if (update_idx + 1) % 5 == 0:
            print(
                f"Update {update_idx + 1:4d}/{config.num_updates} | "
                f"Steps: {tracker.total_steps:7d} | "
                f"Avg Return: {avg_return:7.2f} | "
                f"Loss: {loss_dict['loss']:7.4f} | "
                f"KL: {loss_dict['approx_kl']:.4f} | "
                f"Clip: {loss_dict['clip_fraction']:.3f}"
            )

        # Periodic evaluation
        if (update_idx + 1) % config.eval_interval == 0:
            eval_returns, eval_lengths = agent.evaluate(eval_env, config.eval_episodes)
            mean_return = sum(eval_returns) / len(eval_returns)
            std_return = (
                sum((r - mean_return) ** 2 for r in eval_returns) / len(eval_returns)
            ) ** 0.5
            mean_length = sum(eval_lengths) / len(eval_lengths)

            tracker.log_eval(mean_return, std_return, mean_length)
            print(
                f"  [EVAL] Mean Return: {mean_return:.2f} ± {std_return:.2f} | "
                f"Mean Length: {mean_length:.1f}"
            )

    tracker.close()
    env.close()
    eval_env.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()

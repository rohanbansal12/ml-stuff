import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from dataclasses import dataclass, asdict
from collections import deque
import argparse
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import os

from net import ContinuousActorCriticNet, ContinuousRPONet
from util import RolloutBuffer


@dataclass
class Config:
    """PPO Continuous training configuration."""

    # Core PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2  # Set to 0 to disable
    value_coef: float = 0.5
    entropy_coef: float = 0.0  # Often 0 for continuous control
    max_grad_norm: float = 0.5
    norm_advantages: bool = True

    # PPO-specific
    num_epochs: int = 10
    num_minibatches: int = 32
    target_kl: float | None = None  # Early stopping threshold

    # RPO (Robust Policy Optimization)
    use_rpo: bool = False
    rpo_alpha: float = 0.5

    # Learning rate schedule
    anneal_lr: bool = True

    # Architecture
    hidden_sizes: tuple = (64, 64)

    # Environment
    env_name: str = "Pendulum-v1"
    num_envs: int = 8
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # Training duration - continuous control needs more steps
    total_timesteps: int = 3_000_000
    rollout_steps: int = 1024  # Steps per env per rollout

    # Evaluation
    eval_interval: int = 10  # Evaluate every N updates
    eval_episodes: int = 10

    # Misc
    seed: int = 42
    log_dir: str = "./runs"

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
        parser.add_argument("--lr", type=float, default=cls.lr)
        parser.add_argument("--gamma", type=float, default=cls.gamma)
        parser.add_argument("--lam", type=float, default=cls.lam)
        parser.add_argument("--clip_eps", type=float, default=cls.clip_eps)
        parser.add_argument("--value_clip_eps", type=float, default=cls.value_clip_eps)
        parser.add_argument("--value_coef", type=float, default=cls.value_coef)
        parser.add_argument("--entropy_coef", type=float, default=cls.entropy_coef)
        parser.add_argument("--max_grad_norm", type=float, default=cls.max_grad_norm)

        adv_group = parser.add_mutually_exclusive_group()
        adv_group.add_argument(
            "--norm_advantages", action="store_true", dest="norm_advantages"
        )
        adv_group.add_argument(
            "--no_norm_advantages", action="store_false", dest="norm_advantages"
        )
        parser.set_defaults(norm_advantages=cls.norm_advantages)

        parser.add_argument("--num_epochs", type=int, default=cls.num_epochs)
        parser.add_argument("--num_minibatches", type=int, default=cls.num_minibatches)
        parser.add_argument("--target_kl", type=float, default=cls.target_kl)

        parser.add_argument("--use_rpo", action="store_true")
        parser.add_argument("--rpo_alpha", type=float, default=cls.rpo_alpha)

        lr_group = parser.add_mutually_exclusive_group()
        lr_group.add_argument("--anneal_lr", action="store_true", dest="anneal_lr")
        lr_group.add_argument("--no_anneal_lr", action="store_false", dest="anneal_lr")
        parser.set_defaults(anneal_lr=cls.anneal_lr)

        parser.add_argument(
            "--hidden_sizes", type=int, nargs="+", default=list(cls.hidden_sizes)
        )
        parser.add_argument("--env_name", type=str, default=cls.env_name)
        parser.add_argument("--num_envs", type=int, default=cls.num_envs)

        norm_group = parser.add_mutually_exclusive_group()
        norm_group.add_argument(
            "--normalize_obs", action="store_true", dest="normalize_obs"
        )
        norm_group.add_argument(
            "--no_normalize_obs", action="store_false", dest="normalize_obs"
        )
        parser.set_defaults(normalize_obs=cls.normalize_obs)

        reward_group = parser.add_mutually_exclusive_group()
        reward_group.add_argument(
            "--normalize_reward", action="store_true", dest="normalize_reward"
        )
        reward_group.add_argument(
            "--no_normalize_reward", action="store_false", dest="normalize_reward"
        )
        parser.set_defaults(normalize_reward=cls.normalize_reward)

        parser.add_argument("--clip_obs", type=float, default=cls.clip_obs)
        parser.add_argument("--clip_reward", type=float, default=cls.clip_reward)
        parser.add_argument("--total_timesteps", type=int, default=cls.total_timesteps)
        parser.add_argument("--rollout_steps", type=int, default=cls.rollout_steps)
        parser.add_argument("--eval_interval", type=int, default=cls.eval_interval)
        parser.add_argument("--eval_episodes", type=int, default=cls.eval_episodes)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--log_dir", type=str, default=cls.log_dir)
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
            use_rpo=args.use_rpo,
            rpo_alpha=args.rpo_alpha,
            anneal_lr=args.anneal_lr,
            hidden_sizes=tuple(args.hidden_sizes),
            env_name=args.env_name,
            num_envs=args.num_envs,
            normalize_obs=args.normalize_obs,
            normalize_reward=args.normalize_reward,
            clip_obs=args.clip_obs,
            clip_reward=args.clip_reward,
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            log_dir=args.log_dir,
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

    def log_rollout(
        self, episode_returns: list[float], episode_lengths: list[int], steps: int
    ):
        """Log metrics from a rollout."""
        self.total_steps += steps

        for ret, length in zip(episode_returns, episode_lengths):
            self.total_episodes += 1
            self.episode_returns.append(ret)
            self.episode_lengths.append(length)

            self.writer.add_scalar("train_steps/episode_return", ret, self.total_steps)
            self.writer.add_scalar(
                "train_steps/episode_length", length, self.total_steps
            )

    def log_update(self, loss_dict: dict, lr: float):
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
        self.writer.add_scalar(
            "train/policy_loss", loss_dict["policy_loss"], self.total_updates
        )
        self.writer.add_scalar(
            "train/value_loss", loss_dict["value_loss"], self.total_updates
        )
        self.writer.add_scalar(
            "train/entropy", loss_dict["entropy"], self.total_updates
        )
        self.writer.add_scalar(
            "train/approx_kl", loss_dict["approx_kl"], self.total_updates
        )
        self.writer.add_scalar(
            "train/clip_fraction", loss_dict["clip_fraction"], self.total_updates
        )
        self.writer.add_scalar(
            "train/explained_variance",
            loss_dict["explained_variance"],
            self.total_updates,
        )
        self.writer.add_scalar("train/learning_rate", lr, self.total_updates)

        # Log by steps
        self.writer.add_scalar("train_steps/avg_return", avg_return, self.total_steps)

        return avg_return, avg_length

    def log_eval(self, mean_return: float, std_return: float, mean_length: float):
        """Log evaluation metrics."""
        self.writer.add_scalar("eval/mean_return", mean_return, self.total_updates)
        self.writer.add_scalar("eval/std_return", std_return, self.total_updates)
        self.writer.add_scalar("eval/mean_length", mean_length, self.total_updates)
        self.writer.add_scalar("eval_steps/mean_return", mean_return, self.total_steps)

    def close(self):
        self.writer.close()


class PPOContinuousAgent:
    """PPO agent for continuous action spaces."""

    def __init__(
        self, obs_dim: int, action_dim: int, config: Config, device: torch.device
    ):
        self.config = config
        self.device = device
        self.action_dim = action_dim

        # Choose network architecture
        if config.use_rpo:
            self.actor_critic = ContinuousRPONet(
                obs_dim, action_dim, config.hidden_sizes, config.rpo_alpha
            ).to(device)
        else:
            self.actor_critic = ContinuousActorCriticNet(
                obs_dim, action_dim, config.hidden_sizes
            ).to(device)

        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=config.lr, eps=1e-5
        )

        # State for vectorized rollouts
        self.next_obs = None

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, lr: float):
        """Set learning rate."""
        self.optimizer.param_groups[0]["lr"] = lr

    def collect_rollout(self, env: SyncVectorEnv, buffer: RolloutBuffer) -> dict:
        """Collect a rollout from the vectorized environment."""
        if self.next_obs is None:
            self.next_obs, _ = env.reset(seed=self.config.seed)

        finished_returns = []
        finished_lengths = []

        for _ in range(self.config.rollout_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(
                    self.next_obs, device=self.device, dtype=torch.float32
                )
                action, log_prob, _, value = self.actor_critic.get_action_and_value(
                    obs_tensor
                )
                value = value.squeeze(-1)

            cpu_actions = action.cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = env.step(cpu_actions)
            dones = np.logical_or(terminations, truncations)

            # Handle episode statistics from RecordEpisodeStatistics wrapper
            # When using individual env wrappers with SyncVectorEnv, completed episodes
            # appear in "final_info" list (one entry per env, None if not done)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        finished_returns.append(float(info["episode"]["r"]))
                        finished_lengths.append(int(info["episode"]["l"]))
            # Alternative pattern for gym.wrappers.vector.RecordEpisodeStatistics
            elif "_episode" in infos:
                for i in np.where(infos["_episode"])[0]:
                    finished_returns.append(float(infos["episode"]["r"][i]))
                    finished_lengths.append(int(infos["episode"]["l"][i]))

            buffer.store(self.next_obs, action, rewards, dones, value, log_prob)
            self.next_obs = next_obs

        # Bootstrap value
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                self.next_obs, device=self.device, dtype=torch.float32
            )
            last_value = self.actor_critic.get_value(obs_tensor).squeeze(-1)

        buffer.compute_advantages_and_returns(
            last_value, self.config.gamma, self.config.lam
        )

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

        # Explained variance before updates
        with torch.no_grad():
            explained_var = 1 - (returns - old_values).var() / (returns.var() + 1e-8)

        batch_size = obs.shape[0]
        minibatch_size = self.config.minibatch_size

        # Accumulators
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
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_old_values = old_values[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Forward pass
                _, new_log_probs, entropy, new_values = (
                    self.actor_critic.get_action_and_value(mb_obs, mb_actions)
                )
                new_values = new_values.squeeze(-1)

                # Policy loss
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if self.config.value_clip_eps > 0:
                    clipped_values = mb_old_values + torch.clamp(
                        new_values - mb_old_values,
                        -self.config.value_clip_eps,
                        self.config.value_clip_eps,
                    )
                    value_loss_unclipped = (new_values - mb_returns).square()
                    value_loss_clipped = (clipped_values - mb_returns).square()
                    value_loss = (
                        0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (new_values - mb_returns).square().mean()

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.config.max_grad_norm
                    )
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.config.clip_eps)
                        .float()
                        .mean()
                        .item()
                    )

                total_loss_sum += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy_loss.item()
                approx_kl_sum += approx_kl
                clip_fraction_sum += clip_fraction
                num_gradient_steps += 1

            # Early stopping
            if self.config.target_kl is not None:
                epoch_kl = (
                    approx_kl_sum / num_gradient_steps if num_gradient_steps > 0 else 0
                )
                if epoch_kl > self.config.target_kl:
                    break

        return {
            "loss": total_loss_sum / num_gradient_steps
            if num_gradient_steps > 0
            else 0,
            "policy_loss": policy_loss_sum / num_gradient_steps
            if num_gradient_steps > 0
            else 0,
            "value_loss": value_loss_sum / num_gradient_steps
            if num_gradient_steps > 0
            else 0,
            "entropy": entropy_sum / num_gradient_steps
            if num_gradient_steps > 0
            else 0,
            "approx_kl": approx_kl_sum / num_gradient_steps
            if num_gradient_steps > 0
            else 0,
            "clip_fraction": clip_fraction_sum / num_gradient_steps
            if num_gradient_steps > 0
            else 0,
            "explained_variance": explained_var.item(),
            "num_epochs": final_epoch + 1,
        }

    def evaluate(
        self, env: gym.Env, num_episodes: int
    ) -> tuple[list[float], list[int]]:
        """Run deterministic evaluation episodes."""
        returns = []
        lengths = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

            while True:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(
                        obs, device=self.device, dtype=torch.float32
                    )
                    # Use mean action (deterministic)
                    mean = self.actor_critic.actor_mean(obs_tensor.unsqueeze(0))
                    action = mean.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                episode_length += 1

                if terminated or truncated:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)

        return returns, lengths


def make_env(env_name: str):
    """Create environment with episode statistics tracking."""

    def _make():
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _make


def wrap_vec_env(envs: SyncVectorEnv, config: Config) -> SyncVectorEnv:
    """Apply normalization wrappers to vectorized environment."""
    envs = gym.wrappers.vector.ClipAction(envs)

    if config.normalize_obs:
        envs = gym.wrappers.vector.NormalizeObservation(envs)
        envs = gym.wrappers.vector.TransformObservation(
            envs, lambda obs: np.clip(obs, -config.clip_obs, config.clip_obs)
        )

    if config.normalize_reward:
        envs = gym.wrappers.vector.NormalizeReward(envs, gamma=config.gamma)
        envs = gym.wrappers.vector.TransformReward(
            envs,
            lambda reward: np.clip(reward, -config.clip_reward, config.clip_reward),
        )

    return envs


def print_config(config: Config, device: torch.device, obs_dim: int, action_dim: int):
    """Print training configuration."""
    print("\n" + "=" * 55)
    print("PPO Continuous Training Configuration")
    print("=" * 55)
    for key, value in config.to_dict().items():
        print(f"  {key:20s}: {value}")
    print(f"  {'device':20s}: {device}")
    print(f"  {'obs_dim':20s}: {obs_dim}")
    print(f"  {'action_dim':20s}: {action_dim}")
    print(f"  {'batch_size':20s}: {config.batch_size}")
    print(f"  {'minibatch_size':20s}: {config.minibatch_size}")
    print(f"  {'num_updates':20s}: {config.num_updates}")
    print("=" * 55 + "\n")


def main():
    config = Config.from_args()

    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Create environments
    env = SyncVectorEnv([make_env(config.env_name) for _ in range(config.num_envs)])
    env = wrap_vec_env(env, config)

    # Eval env without reward normalization (to get true returns)
    eval_env = gym.make(config.env_name)

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]

    print_config(config, device, obs_dim, action_dim)

    # Initialize
    agent = PPOContinuousAgent(obs_dim, action_dim, config, device)

    buffer = RolloutBuffer(
        steps=config.rollout_steps,
        num_envs=config.num_envs,
        obs_dim=obs_dim,
        act_dim=action_dim,
        device=device,
        cont=True,
    )

    run_name = f"ppo_cont_{config.env_name}_lr={config.lr}_seed={config.seed}"
    if config.use_rpo:
        run_name += f"_rpo={config.rpo_alpha}"
    tracker = Tracker(os.path.join(config.log_dir, run_name), config)

    # Training loop
    for update_idx in range(config.num_updates):
        # LR annealing
        if config.anneal_lr:
            frac = 1.0 - update_idx / config.num_updates
            agent.set_lr(frac * config.lr)

        buffer.reset()

        # Collect
        rollout_info = agent.collect_rollout(env, buffer)

        # Update
        loss_dict = agent.update(buffer)

        # Log
        tracker.log_rollout(
            rollout_info["episode_returns"],
            rollout_info["episode_lengths"],
            rollout_info["steps"],
        )
        avg_return, avg_length = tracker.log_update(loss_dict, agent.get_lr())

        # Print
        if (update_idx + 1) % 5 == 0:
            print(
                f"Update {update_idx + 1:4d}/{config.num_updates} | "
                f"Steps: {tracker.total_steps:8d} | "
                f"Avg Return: {avg_return:8.2f} | "
                f"Loss: {loss_dict['loss']:7.4f} | "
                f"KL: {loss_dict['approx_kl']:.4f} | "
                f"LR: {agent.get_lr():.2e}"
            )

        # Eval
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

import argparse
import os
from collections import deque
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from net import ActorCriticNet
from torch.utils.tensorboard import SummaryWriter
from util import RolloutBuffer


@dataclass
class Config:
    """Training configuration."""

    # Optimization
    lr: float = 1e-3  # A2C needs higher LR than PPO (many small updates)
    gamma: float = 0.99
    lam: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.0  # Often not needed for simple envs
    max_grad_norm: float = 0.5
    norm_advantages: bool = False

    # Architecture
    hidden_sizes: tuple = (64, 64)

    # Environment
    env_name: str = "CartPole-v1"
    num_envs: int = 16

    # Training duration
    total_timesteps: int = 500_000
    rollout_steps: int = 5  # A2C works best with short rollouts + frequent updates

    # Evaluation
    eval_interval: int = 50  # Evaluate every N updates
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
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lr", type=float, default=cls.lr)
        parser.add_argument("--gamma", type=float, default=cls.gamma)
        parser.add_argument("--lam", type=float, default=cls.lam)
        parser.add_argument("--value_coef", type=float, default=cls.value_coef)
        parser.add_argument("--entropy_coef", type=float, default=cls.entropy_coef)
        parser.add_argument("--max_grad_norm", type=float, default=cls.max_grad_norm)
        parser.add_argument("--norm_advantages", action="store_true")
        parser.add_argument("--hidden_sizes", type=int, nargs="+", default=list(cls.hidden_sizes))
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
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            norm_advantages=args.norm_advantages,
            hidden_sizes=tuple(args.hidden_sizes),
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

            # Log individual episodes by step count
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


class A2CAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: Config, device: torch.device):
        self.config = config
        self.device = device

        self.actor_critic = ActorCriticNet(obs_dim, action_dim, config.hidden_sizes).to(device)
        if config.compile_model:
            self.actor_critic = torch.compile(self.actor_critic)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=config.lr)

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
            last_value = self.actor_critic.get_value(obs_tensor).squeeze()

        buffer.compute_advantages_and_returns(last_value, self.config.gamma, self.config.lam)

        return {
            "episode_returns": finished_returns,
            "episode_lengths": finished_lengths,
            "steps": self.config.rollout_steps * env.num_envs,
        }

    def update(self, buffer: RolloutBuffer) -> dict:
        """Perform a single A2C update."""
        obs, actions, old_log_probs, values, advantages, returns, dones = buffer.get()

        # Normalize advantages
        if self.config.norm_advantages:
            std, mean = torch.std_mean(advantages)
            advantages = (advantages - mean) / (std + 1e-8)

        # Forward pass
        logits, new_values = self.actor_critic(obs)
        new_values = new_values.squeeze()

        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Losses
        policy_loss = -(new_log_probs * advantages.detach()).mean()
        value_loss = (new_values - returns).square().mean()
        entropy_loss = entropy.mean()

        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy_loss
        )

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
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
    print("A2C Training Configuration")
    print("=" * 50)
    for key, value in config.to_dict().items():
        print(f"  {key:20s}: {value}")
    print(f"  {'device':20s}: {device}")
    print(f"  {'obs_dim':20s}: {obs_dim}")
    print(f"  {'action_dim':20s}: {action_dim}")
    print(f"  {'batch_size':20s}: {config.batch_size}")
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
    agent = A2CAgent(obs_dim, action_dim, config, device)

    buffer = RolloutBuffer(
        steps=config.rollout_steps,
        num_envs=config.num_envs,
        obs_dim=obs_dim,
        device=device,
        cont=False,
    )

    run_name = f"a2c_{config.env_name}_lr={config.lr}_lam={config.lam}_seed={config.seed}"
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
        if (update_idx + 1) % 10 == 0:
            print(
                f"Update {update_idx + 1:4d}/{config.num_updates} | "
                f"Steps: {tracker.total_steps:7d} | "
                f"Episodes: {tracker.total_episodes:5d} | "
                f"Avg Return: {avg_return:7.2f} | "
                f"Loss: {loss_dict['loss']:7.4f} | "
                f"Entropy: {loss_dict['entropy']:.4f}"
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

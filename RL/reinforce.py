import argparse
import os
from collections import deque
from dataclasses import asdict, dataclass, field

import gymnasium as gym
import torch
from net import PolicyNet
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    """Training configuration."""

    lr: float = 1e-3
    gamma: float = 0.99
    norm_returns: bool = False
    num_episodes: int = 2000
    hidden_sizes: tuple = (128, 128)
    env_name: str = "CartPole-v1"
    seed: int = 42
    log_dir: str = "./runs"
    eval_interval: int = 50  # Evaluate every N episodes
    eval_episodes: int = 10  # Number of episodes per evaluation

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lr", type=float, default=cls.lr)
        parser.add_argument("--gamma", type=float, default=cls.gamma)
        parser.add_argument("--norm_returns", action="store_true")
        parser.add_argument("--num_episodes", type=int, default=cls.num_episodes)
        parser.add_argument("--hidden_sizes", type=int, nargs="+", default=list(cls.hidden_sizes))
        parser.add_argument("--env_name", type=str, default=cls.env_name)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--log_dir", type=str, default=cls.log_dir)
        parser.add_argument("--eval_interval", type=int, default=cls.eval_interval)
        parser.add_argument("--eval_episodes", type=int, default=cls.eval_episodes)
        args = parser.parse_args()
        return cls(
            lr=args.lr,
            gamma=args.gamma,
            norm_returns=args.norm_returns,
            num_episodes=args.num_episodes,
            hidden_sizes=tuple(args.hidden_sizes),
            env_name=args.env_name,
            seed=args.seed,
            log_dir=args.log_dir,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
        )


@dataclass
class Episode:
    """Container for episode data."""

    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    entropies: list = field(default_factory=list)

    def total_return(self):
        return sum(self.rewards)

    def __len__(self):
        return len(self.rewards)


class Tracker:
    """Handles metric tracking and TensorBoard logging."""

    def __init__(self, log_dir: str, config: Config, window_size: int = 100):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text("config", str(config.to_dict()))

        self.window_size = window_size
        self.episode_returns = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)

        self.total_steps = 0
        self.total_episodes = 0

    def log_episode(self, episode_return: float, episode_length: int, loss: float, entropy: float):
        """Log metrics for a training episode."""
        self.total_episodes += 1
        self.total_steps += episode_length

        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)

        avg_return = sum(self.episode_returns) / len(self.episode_returns)
        avg_length = sum(self.episode_lengths) / len(self.episode_lengths)

        # Log by episode
        self.writer.add_scalar("train/episode_return", episode_return, self.total_episodes)
        self.writer.add_scalar("train/episode_length", episode_length, self.total_episodes)
        self.writer.add_scalar("train/loss", loss, self.total_episodes)
        self.writer.add_scalar("train/entropy", entropy, self.total_episodes)
        self.writer.add_scalar("train/avg_return", avg_return, self.total_episodes)
        self.writer.add_scalar("train/avg_length", avg_length, self.total_episodes)

        # Log by steps (for cross-algorithm comparison)
        self.writer.add_scalar("train_steps/episode_return", episode_return, self.total_steps)
        self.writer.add_scalar("train_steps/avg_return", avg_return, self.total_steps)

        return avg_return, avg_length

    def log_eval(self, mean_return: float, std_return: float, mean_length: float):
        """Log evaluation metrics."""
        self.writer.add_scalar("eval/mean_return", mean_return, self.total_episodes)
        self.writer.add_scalar("eval/std_return", std_return, self.total_episodes)
        self.writer.add_scalar("eval/mean_length", mean_length, self.total_episodes)

        self.writer.add_scalar("eval_steps/mean_return", mean_return, self.total_steps)

    def close(self):
        self.writer.close()


class ReinforceAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: Config, device: torch.device):
        self.policy = PolicyNet(obs_dim, action_dim, config.hidden_sizes).to(device)
        self.gamma = config.gamma
        self.norm_returns = config.norm_returns
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.device = device

    def generate_episode(self, env: gym.Env) -> Episode:
        """Generate a single episode by rolling out the policy."""
        eps = Episode()
        obs, _ = env.reset()

        while True:
            action, log_prob, entropy = self.policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            eps.observations.append(obs)
            eps.rewards.append(reward)
            eps.actions.append(action)
            eps.log_probs.append(log_prob)
            eps.entropies.append(entropy)

            obs = next_obs
            if terminated or truncated:
                break

        return eps

    def evaluate(self, env: gym.Env, num_episodes: int) -> tuple[list[float], list[int]]:
        """Run deterministic evaluation episodes."""
        returns = []
        lengths = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

            while True:
                action = self.policy.act_deterministic(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                episode_length += 1

                if terminated or truncated:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)

        return returns, lengths

    def compute_returns(self, rewards: list[float]) -> list[float]:
        """Compute discounted returns for each timestep."""
        G = 0.0
        returns = [0.0] * len(rewards)
        for i in reversed(range(len(rewards))):
            G = self.gamma * G + rewards[i]
            returns[i] = G
        return returns

    def update_policy(self, episode: Episode) -> tuple[float, float]:
        """Perform a policy gradient update."""
        log_probs = torch.stack(episode.log_probs, dim=0).squeeze(-1)
        returns = torch.tensor(
            self.compute_returns(episode.rewards),
            device=self.device,
            dtype=torch.float32,
        ).detach()

        if self.norm_returns:
            std, mean = torch.std_mean(returns)
            returns = (returns - mean) / (std + 1e-9)

        loss = (-log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        mean_entropy = torch.stack(episode.entropies).mean().item()

        return loss.item(), mean_entropy


def print_config(config: Config, device: torch.device, obs_dim: int, action_dim: int):
    """Print training configuration."""
    print("\n" + "=" * 50)
    print("REINFORCE Training Configuration")
    print("=" * 50)
    for key, value in config.to_dict().items():
        print(f"  {key:20s}: {value}")
    print(f"  {'device':20s}: {device}")
    print(f"  {'obs_dim':20s}: {obs_dim}")
    print(f"  {'action_dim':20s}: {action_dim}")
    print("=" * 50 + "\n")


def main():
    config = Config.from_args()

    # Seed for reproducibility
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Create environments
    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print_config(config, device, obs_dim, action_dim)

    # Initialize agent and tracker
    agent = ReinforceAgent(obs_dim, action_dim, config, device)

    run_name = f"reinforce_{config.env_name}_lr={config.lr}_gamma={config.gamma}_seed={config.seed}"
    tracker = Tracker(os.path.join(config.log_dir, run_name), config)

    # Training loop
    for episode_idx in range(config.num_episodes):
        episode = agent.generate_episode(env)
        loss, entropy = agent.update_policy(episode)

        avg_return, avg_length = tracker.log_episode(
            episode_return=episode.total_return(),
            episode_length=len(episode),
            loss=loss,
            entropy=entropy,
        )

        # Print progress
        if (episode_idx + 1) % 5 == 0:
            print(
                f"Episode {episode_idx + 1:4d} | "
                f"Steps: {tracker.total_steps:7d} | "
                f"Loss: {loss:7.4f} | "
                f"Return: {episode.total_return():6.1f} | "
                f"Entropy: {entropy:.3f} | "
                f"Avg Return: {avg_return:.2f}"
            )

        # Periodic evaluation
        if (episode_idx + 1) % config.eval_interval == 0:
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

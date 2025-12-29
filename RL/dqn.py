import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from dataclasses import dataclass, asdict
from collections import deque
import argparse
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import os

from net import QNet
from util import ReplayBuffer


@dataclass
class Config:
    """DQN training configuration."""

    # Core DQN hyperparameters
    lr: float = 1e-4
    gamma: float = 0.99

    # Exploration
    eps_start: float = 1.0  # Initial exploration rate
    eps_end: float = 0.05  # Final exploration rate
    eps_decay_steps: int = 50_000  # Steps to decay epsilon over

    # Target network
    target_update_freq: int = 1000  # Steps between target network updates
    tau: float = 1.0  # 1.0 = hard update, <1.0 = soft (Polyak) update

    # Replay buffer
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 1_000  # Random actions before learning

    # Architecture
    hidden_sizes: tuple = (128, 128)

    # Environment
    env_name: str = "CartPole-v1"
    num_envs: int = 1  # DQN typically uses 1 env

    # Training duration
    total_timesteps: int = 200_000

    # Evaluation
    eval_interval: int = 5_000
    eval_episodes: int = 10

    # Misc
    seed: int = 42
    log_dir: str = "./runs"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lr", type=float, default=cls.lr)
        parser.add_argument("--gamma", type=float, default=cls.gamma)
        parser.add_argument("--eps_start", type=float, default=cls.eps_start)
        parser.add_argument("--eps_end", type=float, default=cls.eps_end)
        parser.add_argument("--eps_decay_steps", type=int, default=cls.eps_decay_steps)
        parser.add_argument(
            "--target_update_freq", type=int, default=cls.target_update_freq
        )
        parser.add_argument("--tau", type=float, default=cls.tau)
        parser.add_argument("--buffer_size", type=int, default=cls.buffer_size)
        parser.add_argument("--batch_size", type=int, default=cls.batch_size)
        parser.add_argument("--learning_starts", type=int, default=cls.learning_starts)
        parser.add_argument(
            "--hidden_sizes", type=int, nargs="+", default=list(cls.hidden_sizes)
        )
        parser.add_argument("--env_name", type=str, default=cls.env_name)
        parser.add_argument("--num_envs", type=int, default=cls.num_envs)
        parser.add_argument("--total_timesteps", type=int, default=cls.total_timesteps)
        parser.add_argument("--eval_interval", type=int, default=cls.eval_interval)
        parser.add_argument("--eval_episodes", type=int, default=cls.eval_episodes)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--log_dir", type=str, default=cls.log_dir)
        args = parser.parse_args()

        return cls(
            lr=args.lr,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_steps=args.eps_decay_steps,
            target_update_freq=args.target_update_freq,
            tau=args.tau,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            hidden_sizes=tuple(args.hidden_sizes),
            env_name=args.env_name,
            num_envs=args.num_envs,
            total_timesteps=args.total_timesteps,
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
        self.total_episodes = 0

    def log_episode(self, episode_return: float, episode_length: int):
        """Log a completed episode."""
        self.total_episodes += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)

        self.writer.add_scalar("train/episode_return", episode_return, self.total_steps)
        self.writer.add_scalar("train/episode_length", episode_length, self.total_steps)

    def log_step(self, step: int):
        """Update step counter."""
        self.total_steps = step

    def log_update(self, loss: float, epsilon: float, step: int):
        """Log metrics from a parameter update."""
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/epsilon", epsilon, step)

        if len(self.episode_returns) > 0:
            avg_return = sum(self.episode_returns) / len(self.episode_returns)
            self.writer.add_scalar("train/avg_return", avg_return, step)

    def log_eval(
        self, mean_return: float, std_return: float, mean_length: float, step: int
    ):
        """Log evaluation metrics."""
        self.writer.add_scalar("eval/mean_return", mean_return, step)
        self.writer.add_scalar("eval/std_return", std_return, step)
        self.writer.add_scalar("eval/mean_length", mean_length, step)

    def get_avg_return(self) -> float:
        if len(self.episode_returns) > 0:
            return sum(self.episode_returns) / len(self.episode_returns)
        return 0.0

    def close(self):
        self.writer.close()


class DQNAgent:
    """Deep Q-Network agent."""

    def __init__(
        self, obs_dim: int, action_dim: int, config: Config, device: torch.device
    ):
        self.config = config
        self.device = device
        self.action_dim = action_dim

        # Q-network and target network
        self.q_net = QNet(obs_dim, action_dim, config.hidden_sizes).to(device)
        self.target_net = QNet(obs_dim, action_dim, config.hidden_sizes).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)

    def get_epsilon(self, step: int) -> float:
        """Compute epsilon for epsilon-greedy exploration.

        Hint: Use linear interpolation based on current step.
        """
        if step > self.config.eps_decay_steps:
            return self.config.eps_end

        return self.config.eps_start + (step / self.config.eps_decay_steps) * (
            self.config.eps_end - self.config.eps_start
        )

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            obs: Current observation (single obs, not batched)
            epsilon: Current exploration rate

        Returns:
            action: Selected action (integer)
        """
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            obs = torch.tensor(obs, device=self.device)
            logits = self.q_net(obs)
            return logits.argmax().item()

    def update(self, buffer: ReplayBuffer) -> float:
        """Perform one DQN update step.

        Returns:
            loss: The loss value (for logging)
        """
        # Sample from replay buffer
        obs, next_obs, actions, rewards, dones = buffer.sample(
            self.config.batch_size, device=self.device
        )

        actions = actions.long().squeeze(-1)
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)

        q_preds = self.q_net(obs)
        q_preds = q_preds[torch.arange(len(actions)), actions]

        with torch.no_grad():
            q_target = self.target_net(next_obs).max(dim=-1).values
            y = rewards + self.config.gamma * q_target * (1 - dones)

        # Huber loss instead of MSE (more robust to outliers)
        loss = F.smooth_l1_loss(q_preds, y)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network.

        If tau == 1.0: Hard update (copy all weights)
        If tau < 1.0: Soft update (Polyak averaging)
            target_param = tau * param + (1 - tau) * target_param
        """
        for param, target_param in zip(
            self.q_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def evaluate(
        self, env: gym.Env, num_episodes: int
    ) -> tuple[list[float], list[int]]:
        """Run deterministic evaluation episodes (epsilon=0)."""
        returns = []
        lengths = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

            while True:
                action = self.select_action(obs, epsilon=0.0)
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
    print("DQN Training Configuration")
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
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environments
    def make_env():
        return gym.make(config.env_name)

    env = SyncVectorEnv([make_env for _ in range(config.num_envs)])
    eval_env = gym.make(config.env_name)

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n

    print_config(config, device, obs_dim, action_dim)

    # Initialize agent, buffer, tracker
    agent = DQNAgent(obs_dim, action_dim, config, device)

    # Note: ReplayBuffer expects act_dim for continuous actions
    # For discrete, we store action as single integer, so act_dim=1
    buffer = ReplayBuffer(
        buffer_size=config.buffer_size,
        num_envs=config.num_envs,
        obs_dim=obs_dim,
        act_dim=1,  # Single discrete action
        device=torch.device("cpu"),
    )

    run_name = f"dqn_{config.env_name}_lr={config.lr}_seed={config.seed}"
    tracker = Tracker(os.path.join(config.log_dir, run_name), config)

    # Initialize environment
    obs, _ = env.reset(seed=config.seed)
    env_rewards = np.zeros(config.num_envs)
    env_lengths = np.zeros(config.num_envs, dtype=np.int32)

    # Training loop
    for global_step in range(config.total_timesteps):
        tracker.log_step(global_step)

        # Get current epsilon
        epsilon = agent.get_epsilon(global_step)

        # Select action
        if global_step < config.learning_starts:
            # Random actions during warmup
            actions = np.array(
                [env.single_action_space.sample() for _ in range(config.num_envs)]
            )
        else:
            # Epsilon-greedy action selection
            # Note: For vectorized env, we need to handle each obs separately
            actions = np.array(
                [agent.select_action(obs[i], epsilon) for i in range(config.num_envs)]
            )

        # Step environment
        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        dones = np.logical_or(terminations, truncations)

        # Track episode statistics
        env_rewards += rewards
        env_lengths += 1

        for i, done in enumerate(dones):
            if done:
                tracker.log_episode(env_rewards[i], env_lengths[i])
                env_rewards[i] = 0.0
                env_lengths[i] = 0

        # Store transition (reshape actions for buffer)
        buffer.store(obs, next_obs, actions.reshape(-1, 1), rewards, terminations)
        obs = next_obs

        # Update after warmup
        if global_step >= config.learning_starts:
            loss = agent.update(buffer)

            # Update target network
            if global_step % config.target_update_freq == 0:
                agent.update_target_network()

            # Log periodically
            if global_step % 100 == 0:
                tracker.log_update(loss, epsilon, global_step)
                avg_return = tracker.get_avg_return()
                print(
                    f"Step {global_step:7d} | "
                    f"Avg Return: {avg_return:7.2f} | "
                    f"Loss: {loss:.4f} | "
                    f"Epsilon: {epsilon:.3f}"
                )

        # Periodic evaluation
        if (global_step + 1) % config.eval_interval == 0:
            eval_returns, eval_lengths = agent.evaluate(eval_env, config.eval_episodes)
            mean_return = sum(eval_returns) / len(eval_returns)
            std_return = (
                sum((r - mean_return) ** 2 for r in eval_returns) / len(eval_returns)
            ) ** 0.5
            mean_length = sum(eval_lengths) / len(eval_lengths)

            tracker.log_eval(mean_return, std_return, mean_length, global_step)
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

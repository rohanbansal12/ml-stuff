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

from net import SoftActorNet, SoftCriticNet
from util import ReplayBuffer


@dataclass
class Config:
    """SAC training configuration."""

    # Learning rates
    q_lr: float = 1e-3
    policy_lr: float = 3e-4
    alpha_lr: float = 1e-3  # Learning rate for entropy coefficient

    # SAC hyperparameters
    gamma: float = 0.99
    tau: float = 0.005  # Polyak averaging coefficient
    target_update_freq: int = 1  # Update target networks every N steps
    auto_alpha: bool = True  # Automatically tune entropy coefficient
    init_alpha: float = 0.2  # Initial entropy coefficient (if not auto-tuning)

    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000  # Random actions before this step

    # Architecture
    hidden_sizes: tuple = (256, 256)

    # Environment
    env_name: str = "Pendulum-v1"
    num_envs: int = 1

    # Training duration
    total_timesteps: int = 100_000

    # Evaluation
    eval_interval: int = 5_000  # Evaluate every N steps
    eval_episodes: int = 10

    # Misc
    seed: int = 42
    log_dir: str = "./runs"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--q_lr", type=float, default=cls.q_lr)
        parser.add_argument("--policy_lr", type=float, default=cls.policy_lr)
        parser.add_argument("--alpha_lr", type=float, default=cls.alpha_lr)
        parser.add_argument("--gamma", type=float, default=cls.gamma)
        parser.add_argument("--tau", type=float, default=cls.tau)
        parser.add_argument(
            "--target_update_freq", type=int, default=cls.target_update_freq
        )

        # Auto-alpha with mutual exclusion
        alpha_group = parser.add_mutually_exclusive_group()
        alpha_group.add_argument("--auto_alpha", action="store_true", dest="auto_alpha")
        alpha_group.add_argument(
            "--no_auto_alpha", action="store_false", dest="auto_alpha"
        )
        parser.set_defaults(auto_alpha=cls.auto_alpha)

        parser.add_argument("--init_alpha", type=float, default=cls.init_alpha)
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
            q_lr=args.q_lr,
            policy_lr=args.policy_lr,
            alpha_lr=args.alpha_lr,
            gamma=args.gamma,
            tau=args.tau,
            target_update_freq=args.target_update_freq,
            auto_alpha=args.auto_alpha,
            init_alpha=args.init_alpha,
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

    def log_update(self, loss_dict: dict, step: int):
        """Log metrics from a parameter update."""
        self.writer.add_scalar("train/q_loss", loss_dict["q_loss"], step)
        self.writer.add_scalar("train/actor_loss", loss_dict["actor_loss"], step)
        self.writer.add_scalar("train/alpha_loss", loss_dict["alpha_loss"], step)
        self.writer.add_scalar("train/alpha", loss_dict["alpha"], step)

        # Log rolling averages
        if len(self.episode_returns) > 0:
            avg_return = sum(self.episode_returns) / len(self.episode_returns)
            avg_length = sum(self.episode_lengths) / len(self.episode_lengths)
            self.writer.add_scalar("train/avg_return", avg_return, step)
            self.writer.add_scalar("train/avg_length", avg_length, step)

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


class SACAgent:
    """Soft Actor-Critic agent."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_space: tuple,
        config: Config,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.action_dim = action_dim

        # Actor network
        self.actor = SoftActorNet(
            obs_dim, action_dim, action_space, config.hidden_sizes
        ).to(device)

        # Twin Q-networks
        self.q1 = SoftCriticNet(obs_dim, action_dim, config.hidden_sizes).to(device)
        self.q2 = SoftCriticNet(obs_dim, action_dim, config.hidden_sizes).to(device)

        # Target Q-networks
        self.q1_target = SoftCriticNet(obs_dim, action_dim, config.hidden_sizes).to(
            device
        )
        self.q2_target = SoftCriticNet(obs_dim, action_dim, config.hidden_sizes).to(
            device
        )
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.policy_lr
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.q_lr,
        )

        # Entropy coefficient (alpha)
        self.auto_alpha = config.auto_alpha
        if self.auto_alpha:
            # Target entropy is -dim(A) (heuristic from SAC paper)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=config.alpha_lr
            )
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.init_alpha

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from policy."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            if deterministic:
                # Use mean action for evaluation
                mean, _ = self.actor(obs_tensor)
                action = (
                    torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
                )
            else:
                action, _ = self.actor.get_action(obs_tensor)
            return action.cpu().numpy()

    def update(self, buffer: ReplayBuffer) -> dict:
        """Perform one update step."""
        # Sample from replay buffer and transfer to training device
        obs, next_obs, actions, rewards, dones = buffer.sample(
            self.config.batch_size, device=self.device
        )

        # ----- Update Critics -----
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.get_action(next_obs)
            q1_target = self.q1_target(next_obs, next_actions)
            q2_target = self.q2_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            td_target = rewards + self.config.gamma * (1 - dones) * q_target

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        q_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ----- Update Actor -----
        actions_new, log_probs = self.actor.get_action(obs)
        q1_new = self.q1(obs, actions_new)
        q2_new = self.q2(obs, actions_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- Update Alpha -----
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = (
                -self.log_alpha.exp() * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # ----- Update Target Networks -----
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_alpha else 0.0,
            "alpha": self.alpha,
        }

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        """Polyak averaging update for target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

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
                action = self.select_action(obs, deterministic=True)
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
    print("SAC Training Configuration")
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
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Create environments
    def make_env():
        return gym.make(config.env_name)

    env = SyncVectorEnv([make_env for _ in range(config.num_envs)])
    eval_env = gym.make(config.env_name)

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]
    action_low = env.single_action_space.low
    action_high = env.single_action_space.high

    print_config(config, device, obs_dim, action_dim)

    # Initialize agent, buffer, tracker
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=(action_low, action_high),
        config=config,
        device=device,
    )

    buffer = ReplayBuffer(
        buffer_size=config.buffer_size,
        num_envs=config.num_envs,
        obs_dim=obs_dim,
        act_dim=action_dim,
        device=torch.device("cpu"),  # Store on CPU to save GPU memory
    )

    run_name = f"sac_{config.env_name}_qlr={config.q_lr}_plr={config.policy_lr}_seed={config.seed}"
    tracker = Tracker(os.path.join(config.log_dir, run_name), config)

    # Initialize environment
    obs, _ = env.reset(seed=config.seed)
    env_rewards = np.zeros(config.num_envs)
    env_lengths = np.zeros(config.num_envs, dtype=np.int32)

    # Training loop
    for global_step in range(config.total_timesteps):
        tracker.log_step(global_step)

        # Select action
        if global_step < config.learning_starts:
            # Random actions during warmup
            actions = np.array(
                [env.single_action_space.sample() for _ in range(config.num_envs)]
            )
        else:
            actions = agent.select_action(obs, deterministic=False)

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

        # Store transition
        buffer.store(obs, next_obs, actions, rewards, terminations)
        obs = next_obs

        # Update after warmup
        if global_step >= config.learning_starts:
            loss_dict = agent.update(buffer)

            # Log periodically
            if global_step % 100 == 0:
                tracker.log_update(loss_dict, global_step)
                avg_return = tracker.get_avg_return()
                print(
                    f"Step {global_step:7d} | "
                    f"Avg Return: {avg_return:7.2f} | "
                    f"Q Loss: {loss_dict['q_loss']:.4f} | "
                    f"Actor Loss: {loss_dict['actor_loss']:.4f} | "
                    f"Alpha: {loss_dict['alpha']:.4f}"
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

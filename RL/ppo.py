import torch
from dataclasses import dataclass, field
import argparse
import gymnasium as gym
from net import PPONet
from util import RolloutBuffer, PPOConfig, PPOAgent
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

def main():
    parser = argparse.ArgumentParser()

    # Environment / training horizon
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--total_timesteps", type=int, default=500000, help="Number of PPO timesteps")
    parser.add_argument("--rollout_size", type=int, default=128, help="Steps per PPO update")
    parser.add_argument("--num_envs", type=int, default=4, help='Number of envs to run in parallel')
    parser.add_argument("--seed", type=int, default=1)

    # PPO core hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal_lr", action='store_true', dest='anneal_lr', default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_clip_eps", type=float, default=1000)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    # PPO optimization loop
    parser.add_argument("--train_epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--num_minibatches", type=int, default=4, help="Minibatches per epoch")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None, help="Early stop threshold on approx KL (set 0 or None to disable)")

    # Advantage / network settings
    parser.add_argument("--norm_advantages", action="store_true", dest="norm_advantages", default=True)
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for MLP (used twice: [h, h])")
    parser.add_argument("--log-dir", type=str, default="./runs")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.rollout_size)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.num_updates = args.total_timesteps // args.batch_size

    # basic tensorboard logging
    run_name = f"ppo_lr={args.lr}_gamma={args.gamma}_lam={args.lam}_clip-eps={args.clip_eps}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)
    writer.add_text("hparams", str(vars(args)))

    # cuda setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # env setup
    def make_env():
        env = gym.make(args.env_name)
        return env

    envs = gym.vector.SyncVectorEnv([make_env for i in range(args.num_envs)])
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    hidden_sizes = (args.hidden_size, args.hidden_size)

    # Buffer (Updated for vector envs)
    buffer = RolloutBuffer(
        steps=args.rollout_size,
        num_envs=args.num_envs, 
        obs_dim=obs_dim, 
        device=device,
        cont=False
    )

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
        seed=args.seed
    )

    # Agent
    agent = PPOAgent(
        PPONet,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        config=config,
        device=device,
    )

    # Print experiment configuration
    print("\n=== PPO CartPole Experiment Configuration ===")
    print(f"Environment:          {args.env_name}")
    print(f"Num Envs:             {args.num_envs}")
    print(f"Observation dim:      {obs_dim}")
    print(f"Action dim:           {action_dim}")
    print(f"Hidden sizes:         {hidden_sizes}")
    print(f"Total updates:        {args.num_updates}")
    print(f"Steps per Env:        {args.rollout_size}")
    print(f"Total Batch Size:     {args.batch_size}")
    print(f"Learning rate:        {args.lr}")
    print(f"Device:               {device}")
    print("================================================\n")

    episode_returns = []
    window = 100 

    for update in range(1, args.num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lrnow = frac * args.lr
            agent.optimizer.param_groups[0]["lr"] = lrnow

        buffer.reset()

        # 1) Collect rollout
        last_value, rollout_log = agent.collect_rollout(envs, args.rollout_size, buffer)

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

        if update % 5 == 0:
            print(
                f"Update {update:4d} | "
                f"AvgRet(last {len(recent):3d}): {avg_return:7.2f} | "
                f"Loss: {optim_logs.get('loss', 0):.4f} | "
                f"Pol: {optim_logs.get('policy_loss', 0):.4f} | "
                f"Val: {optim_logs.get('value_loss', 0):.4f} | "
                f"Ent: {optim_logs.get('entropy', 0):.4f} | "
                f"KL: {optim_logs.get('approx_kl', 0):.5f} | "
                f"ClipFrac: {optim_logs.get('clipfrac', 0):.3f} | "
                f"UpdatesThisBatch: {optim_logs.get('num_updates', 0):2d}"
            )
            writer.add_scalar("Loss", optim_logs.get('loss', 0), update)
            writer.add_scalar("Avg_ret", avg_return, update)
            writer.add_scalar("Entropy", optim_logs.get('entropy', 0), update)
            writer.add_scalar("KL", optim_logs.get('approx_kl', 0), update)


    envs.close()


if __name__ == "__main__":
    main()
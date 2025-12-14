import torch
import argparse
import gymnasium as gym
from net import SoftActorNet, SoftCriticNet
from util import ReplayBuffer
import numpy as np
import random
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Pendulum-v1")
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Number of timesteps")
    parser.add_argument("--num_envs", type=int, default=4, help='Number of envs to run in parallel')
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for MLP (used twice: [h, h])")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target_network_frequency", type=int, default=1)
    parser.add_argument("--learning_starts", type=int, default=5e3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    action_dim = envs.single_action_space.shape[0]
    act_low, act_high = envs.single_action_space.low, envs.single_action_space.high
    hidden_sizes = (args.hidden_size, args.hidden_size)

    actor = SoftActorNet(obs_dim, action_dim, (act_low, act_high), hidden_sizes).to(device)
    q1c = SoftCriticNet(obs_dim, action_dim, hidden_sizes).to(device)
    q2c = SoftCriticNet(obs_dim, action_dim, hidden_sizes).to(device)

    q1c_target = SoftCriticNet(obs_dim, action_dim, hidden_sizes).to(device)
    q2c_target = SoftCriticNet(obs_dim, action_dim, hidden_sizes).to(device)
    q1c_target.load_state_dict(q1c.state_dict())
    q2c_target.load_state_dict(q2c.state_dict())

    q_optim = torch.optim.Adam(list(q1c.parameters()) + list(q2c.parameters()), lr=args.q_lr)
    actor_optim = torch.optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    buffer = ReplayBuffer(args.buffer_size, args.num_envs, obs_dim, action_dim)

    # learning entropy alpha weight
    target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.q_lr)

    obs, _ = envs.reset(seed=args.seed)
    episode_returns = []
    window = 100

    for global_step in range(args.total_timesteps):
        
        # sample actions (either from actor or just randomly depending on when we start learning)
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # step the env with actions
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # collect episode returns if they exist
        if "_episode" in infos:
            # indices where the environment actually finished
            indices = np.where(infos["_episode"])[0]
            for i in indices:
                raw_return = infos["episode"]["r"][i]
                episode_returns.append(raw_return)

        # store info in replay buffer
        buffer.store(obs, next_obs, actions, rewards, terminations)

        # update current obs var to next_obs
        obs = next_obs

        # perform actor, critic, alpha updates if we have reached learning start step
        if global_step > args.learning_starts:
            # get batch data from Replay Buffer
            s_obs, s_next_obs, s_actions, s_rewards, s_dones = buffer.sample(args.batch_size, device)

            # compute critic loss by generating target and same state critic predictions
            with torch.no_grad():
                next_state_actions, next_state_log_probs = actor.get_action(s_next_obs)
                q1t = q1c_target.forward(s_next_obs, next_state_actions)
                q2t = q2c_target.forward(s_next_obs, next_state_actions)
                q_target = s_rewards + args.gamma * (1 - s_dones) * (torch.min(q1t, q2t) - alpha * next_state_log_probs)

            q1 = q1c.forward(s_obs, s_actions)
            q2 = q2c.forward(s_obs, s_actions)
            qf_loss = F.mse_loss(q_target, q1) + F.mse_loss(q_target, q2)

            q_optim.zero_grad()
            qf_loss.backward()
            q_optim.step()

            # compute actor loss
            pi, log_prob_pi = actor.get_action(s_obs)
            q1_pi = q1c.forward(s_obs, pi)
            q2_pi = q2c.forward(s_obs, pi)
            actor_loss = (alpha * log_prob_pi - torch.min(q1_pi, q2_pi)).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            # compute alpha loss
            alpha_loss = (-log_alpha.exp() * (log_prob_pi + target_entropy).detach()).mean()

            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()
            alpha = log_alpha.exp().item()

            # perform polyak updates on target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(q1c.parameters(), q1c_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(q2c.parameters(), q2c_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                if len(episode_returns) > 0:
                    start = max(0, len(episode_returns) - window)
                    recent = episode_returns[start:]
                    avg_return = sum(recent) / len(recent)
                else:
                    avg_return = 0.0
                    recent = []

                print(
                    f"Step {global_step} | "
                    f"AvgRet(last {len(recent):3d}): {avg_return:7.2f} | "
                    f"QF Loss: {qf_loss.item():.4f} | "
                    f"Actor Loss: {actor_loss.item():.4f} | "
                    f"Alpha: {alpha:.4f} | "
                )

if __name__ == "__main__":
    main()
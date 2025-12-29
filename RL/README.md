# Reinforcement Learning from Scratch

A progressive exploration of modern reinforcement learning algorithms, from basic policy gradients to state-of-the-art methods. Each algorithm is implemented from scratch in PyTorch with a focus on understanding the underlying theory and practical implementation details.

---

## Overview

This repository documents a step-by-step journey through fundamental RL algorithms:

| Algorithm | Type | Action Space | Key Concepts |
|-----------|------|--------------|--------------|
| **REINFORCE** | On-policy, Policy Gradient | Discrete | Monte-Carlo returns, policy gradient theorem |
| **A2C** | On-policy, Actor-Critic | Discrete | Value baseline, GAE, variance reduction |
| **PPO** | On-policy, Actor-Critic | Discrete/Continuous | Clipped surrogate, importance sampling |
| **SAC** | Off-policy, Actor-Critic | Continuous | Maximum entropy, twin Q-networks, replay buffer |
| **DQN** | Off-policy, Value-Based | Discrete | Q-learning, replay buffer, target networks |

---

## Environments

### CartPole-v1 (Discrete Control)
- **State:** 4 continuous variables (cart position/velocity, pole angle/velocity)
- **Actions:** 2 discrete (push left/right)
- **Episode terminates:** Pole falls or cart leaves bounds
- **Max return:** 500
- **Solved:** Consistent return of 500

### Pendulum-v1 (Continuous Control)
- **State:** 3 continuous variables (cos θ, sin θ, angular velocity)
- **Actions:** 1 continuous (torque ∈ [-2, 2])
- **Return range:** -1600 (random) to 0 (optimal)
- **Solved:** Consistent return > -200

---

## Stage 1: REINFORCE (Vanilla Policy Gradient)

### Theory

The policy gradient theorem gives us the gradient of expected return:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \, G_t \right]$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the discounted return from timestep $t$.

REINFORCE estimates this expectation by sampling complete episodes and using observed returns. The loss function is:

$$L(\theta) = -\mathbb{E} \left[ \log \pi_\theta(a_t|s_t) \, G_t \right]$$

**Properties:**
- Unbiased gradient estimate
- High variance (uses full episode returns)
- No bootstrapping or value function
- Simple to implement

### Implementation

```python
# Core REINFORCE update
returns = compute_discounted_returns(rewards, gamma)
log_probs = dist.log_prob(actions)
loss = -(log_probs * returns).mean()
```

**Key hyperparameters:** `lr=1e-3`, `gamma=0.99`

### Results on CartPole-v1

| Metric | Value |
|--------|-------|
| Episodes to solve | ~250 |
| Final avg return | 450-500 |
| Variance | High |

**Observations:**
- Learning is noisy with high variance in episode returns
- Reaches 500, then often crashes and oscillates before recovering
- Sensitive to learning rate — too high causes instability, too low fails to learn
- Despite simplicity, works reasonably well on CartPole

---

## Stage 2: Advantage Actor-Critic (A2C) with GAE

### Theory

#### Baseline and Advantage

Adding a baseline $b(s_t)$ to the policy gradient reduces variance without introducing bias:

$$\nabla_\theta J = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \, (G_t - b(s_t)) \right]$$

The optimal baseline is the value function $V(s_t)$, giving us the **advantage**:

$$A_t = G_t - V(s_t)$$

which measures how much better action $a_t$ was compared to the expected value.

#### Generalized Advantage Estimation (GAE)

We have two extremes for advantage estimation:

| Method | Formula | Variance | Bias |
|--------|---------|----------|------|
| Monte-Carlo | $A_t^{MC} = G_t - V(s_t)$ | High | None |
| TD(0) | $A_t^{TD} = r_t + \gamma V(s_{t+1}) - V(s_t)$ | Low | High |

GAE interpolates using parameter $\lambda \in [0, 1]$:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

- $\lambda = 0$: Pure TD (low variance, high bias)
- $\lambda = 1$: Pure Monte-Carlo (high variance, no bias)
- $\lambda = 0.95$: Common default

#### Loss Functions

$$L_{total} = L_\pi + c_1 L_V - c_2 L_H$$

where:
- **Policy loss:** $L_\pi = -\mathbb{E}[\log \pi(a|s) \cdot A^{GAE}]$
- **Value loss:** $L_V = \mathbb{E}[(V(s) - R^{GAE})^2]$
- **Entropy bonus:** $L_H = \mathbb{E}[H(\pi(\cdot|s))]$

### Implementation

```python
# GAE computation (backwards recursion)
gae = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
    gae = delta + gamma * lam * (1 - dones[t]) * gae
    advantages[t] = gae
returns = advantages + values
```

**Key hyperparameters:** `lr=1e-3`, `gamma=0.99`, `lambda=0.95`, `rollout_steps=5`, `num_envs=16`

### Results on CartPole-v1

| Metric | Value |
|--------|-------|
| Steps to solve | ~300k-500k |
| Final avg return | 400-500 |
| Variance | Medium |

**Observations:**
- **Critical finding:** A2C requires frequent updates with small batches
  - `lr=1e-3` with 5-step rollouts: ✓ Works
  - `lr=2.5e-4` with 128-step rollouts: ✗ Fails
- The interaction between learning rate and update frequency is crucial
- More stable than REINFORCE but still shows occasional collapses
- Orthogonal initialization outperforms Xavier for this architecture

---

## Stage 3: Proximal Policy Optimization (PPO)

### Theory

#### The Problem with A2C

A2C is sample inefficient — each rollout is used for only one gradient update. Reusing data causes divergence because the policy drifts from the data distribution.

#### Importance Sampling

PPO enables multiple updates per rollout via importance sampling. Defining the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

The surrogate objective becomes:

$$L(\theta) = \mathbb{E} \left[ r_t(\theta) \, A_t \right]$$

#### Clipped Surrogate Objective

Unconstrained optimization can cause destructively large updates. PPO clips the ratio:

$$L^{CLIP}(\theta) = \mathbb{E} \left[ \min \left( r_t A_t, \, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

The clipping removes incentive for moving the ratio outside $[1-\epsilon, 1+\epsilon]$.

#### Value Clipping (Important Detail)

Value clipping should be around **old values**, not returns:

```python
# Correct implementation
clipped_values = old_values + clamp(new_values - old_values, -eps, eps)
value_loss = max((new_values - returns)², (clipped_values - returns)²)
```

### Implementation

**Key hyperparameters:** `lr=3e-4`, `clip_eps=0.2`, `num_epochs=4`, `num_minibatches=4`, `rollout_steps=32`

### Results on CartPole-v1

| Metric | Value |
|--------|-------|
| Steps to solve | ~50k-100k |
| Final avg return | 500 |
| Variance | Low |

**Observations:**
- Much more stable than A2C — often monotonic improvement
- **Separate actor-critic networks outperform shared networks** (faster convergence, more stable)
- Less sensitive to hyperparameters than A2C
- Diagnostic metrics:
  - `clip_fraction` ~0.1-0.2: Healthy
  - `approx_kl` < 0.02: Policy not changing too fast

---

## Stage 4: PPO for Continuous Control

### Continuous Action Spaces

For continuous control, the policy outputs parameters of a Gaussian:

$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$$

The log-probability becomes:

$$\log \pi(a|s) = -\frac{1}{2} \left[ \frac{(a - \mu)^2}{\sigma^2} + \log(2\pi\sigma^2) \right]$$

For multi-dimensional actions, sum log-probs across dimensions.

### The Pendulum Problem

Standard PPO often **fails catastrophically** on Pendulum-v1:
1. Policy improves initially (reaches ~-600)
2. Variance collapses as policy becomes overconfident
3. Policy degrades and gets stuck (~-1000)

### Robust Policy Optimization (RPO)

RPO fixes this by adding uniform noise to the policy mean **during training updates only**:

$$\mu_{RPO} = \mu_\theta(s) + z, \quad z \sim \text{Uniform}(-\alpha, \alpha)$$

This prevents overconfident policies and maintains exploration.

### Results on Pendulum-v1

| Method | Final Avg Return | Status |
|--------|------------------|--------|
| PPO (standard) | ~-1000 | ✗ Fails |
| PPO + RPO (α=0.5) | ~-150 | ✓ Solved |

**Observations:**
- Standard PPO fails catastrophically — improves then collapses
- RPO reliably solves Pendulum, reaching -150 to -200
- Observation and reward normalization are critical
- LR annealing improves final performance
- On-policy methods require ~3M steps vs ~100k for off-policy (SAC)

---

## Stage 5: Soft Actor-Critic (SAC)

### Theory

SAC maximizes a **maximum-entropy objective**:

$$J(\pi) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right) \right]$$

where $\mathcal{H}(\pi) = -\mathbb{E}[\log \pi(a|s)]$ is policy entropy and $\alpha$ is temperature.

#### Soft Bellman Equation

$$Q(s,a) = r(s,a) + \gamma \mathbb{E}_{s'} \left[ V(s') \right]$$

$$V(s) = \mathbb{E}_{a \sim \pi} \left[ Q(s,a) - \alpha \log \pi(a|s) \right]$$

#### Twin Q-Networks

Uses two Q-networks to reduce overestimation:

$$y = r + \gamma (1-d) \left( \min(Q_1'(s',a'), Q_2'(s',a')) - \alpha \log \pi(a'|s') \right)$$

#### Policy Loss

$$L_\pi = \mathbb{E}_{s, a \sim \pi} \left[ \alpha \log \pi(a|s) - \min(Q_1(s,a), Q_2(s,a)) \right]$$

#### Squashed Gaussian Policy

Actions bounded via tanh with Jacobian correction:

$$a = \tanh(u), \quad u \sim \mathcal{N}(\mu, \sigma)$$

$$\log \pi(a|s) = \log \mathcal{N}(u; \mu, \sigma) - \sum_i \log(1 - \tanh^2(u_i))$$

#### Automatic Entropy Tuning

Learns $\alpha$ to maintain target entropy $\bar{\mathcal{H}} \approx -\text{dim}(A)$:

$$L_\alpha = -\mathbb{E} \left[ \alpha \left( \log \pi(a|s) + \bar{\mathcal{H}} \right) \right]$$

### Implementation

**Key hyperparameters:** `q_lr=1e-3`, `policy_lr=3e-4`, `tau=0.005`, `batch_size=256`, `learning_starts=5000`

### Results on Pendulum-v1

| Metric | Value |
|--------|-------|
| Steps to solve | ~50k-80k |
| Final avg return | -150 to -200 |
| Variance | Low |

**Observations:**
- **~30x more sample efficient** than PPO on Pendulum (100k vs 3M steps)
- Automatic entropy tuning works well — alpha decreases as policy improves
- Stable learning with consistent convergence across seeds
- Correct tanh log-prob correction is essential

---

## Stage 6: Deep Q-Network (DQN)

### Theory

DQN is a **value-based** method that learns the optimal action-value function $Q^*(s, a)$ directly, rather than learning a policy. This represents a fundamentally different approach from the policy gradient methods above.

#### Q-Learning

The optimal Q-function satisfies the Bellman optimality equation:

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

We approximate $Q^*$ with a neural network $Q_\theta$ and minimize the TD error:

$$L(\theta) = \mathbb{E}\left[ \left( Q_\theta(s, a) - y \right)^2 \right]$$

where the target is:

$$y = r + \gamma \max_{a'} Q_\theta(s', a')$$

#### The Deadly Triad

Naive Q-learning with function approximation is unstable due to:
1. **Bootstrapping** — targets depend on current estimates
2. **Off-policy learning** — learning from old experiences
3. **Function approximation** — neural networks generalize unpredictably

DQN addresses these with two key innovations:

#### Experience Replay

Store transitions $(s, a, r, s', done)$ in a replay buffer and sample random minibatches. This:
- Breaks correlation between consecutive samples
- Reuses experiences for sample efficiency
- Smooths over the data distribution

#### Target Networks

Use a separate target network $Q_{\theta^-}$ for computing targets:

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

The target network is updated periodically (hard update) or slowly (Polyak averaging):

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

This stabilizes learning by keeping targets fixed during updates.

#### Epsilon-Greedy Exploration

Since we learn Q-values (not a stochastic policy), we need explicit exploration:

$$a = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_a Q_\theta(s, a) & \text{otherwise} \end{cases}$$

Epsilon decays from 1.0 to ~0.05 over training.

### Implementation

```python
# Core DQN update
q_values = q_net(obs)
q_pred = q_values[range(batch_size), actions]  # Q(s, a) for taken actions

with torch.no_grad():
    q_target = target_net(next_obs).max(dim=-1).values
    y = rewards + gamma * q_target * (1 - dones)

loss = F.smooth_l1_loss(q_pred, y)  # Huber loss
```

**Key hyperparameters:** `lr=2.5e-4`, `batch_size=128`, `buffer_size=10000`, `target_update_freq=500`, `learning_starts=10000`, `eps_decay_steps=50000`

### Double DQN (Optional Extension)

Standard DQN overestimates Q-values because the max operator selects and evaluates with the same network. Double DQN decouples selection and evaluation:

```python
# Standard DQN
q_target = target_net(next_obs).max(dim=-1).values

# Double DQN — use online net to SELECT, target net to EVALUATE
best_actions = q_net(next_obs).argmax(dim=-1)
q_target = target_net(next_obs).gather(1, best_actions.unsqueeze(-1)).squeeze(-1)
```

### Results on CartPole-v1

| Metric | Value |
|--------|-------|
| Steps to solve | ~300k-400k |
| Final avg return | 400-500 |
| Variance | Medium-High |

**Observations:**
- **Do NOT use orthogonal initialization** — unlike policy gradient methods, DQN works better with default PyTorch initialization (Kaiming/He)
- Huber loss (`smooth_l1_loss`) is more stable than MSE for Q-learning
- Long warmup period (10k steps) is critical for filling replay buffer with diverse experiences
- Target network updates every 500 steps provide good stability
- Single-seed runs show high variance; curves are smoother when averaged across seeds
- Loss can increase during training even as performance improves — this is normal as Q-values grow

### DQN vs Policy Gradient

| Aspect | DQN | Policy Gradient (PPO) |
|--------|-----|----------------------|
| What it learns | Q-values | Policy directly |
| Exploration | Epsilon-greedy (explicit) | Entropy bonus (implicit) |
| Action space | Discrete only | Discrete or continuous |
| Sample efficiency | High (off-policy) | Lower (on-policy) |
| Stability | Can be unstable | Generally more stable |
| Initialization | Default (Kaiming) | Orthogonal preferred |

---

## Comparison Summary

### Sample Efficiency

| Environment | REINFORCE | A2C | PPO | DQN | SAC |
|-------------|-----------|-----|-----|-----|-----|
| CartPole-v1 | ~150k steps | ~400k steps | ~75k steps | ~350k steps | — |
| Pendulum-v1 | — | — | ~3M steps (RPO) | — | ~75k steps |

### Algorithm Selection Guide

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Discrete actions, simple env | PPO or DQN | Both robust; PPO faster, DQN more sample efficient |
| Continuous control | SAC | Sample efficient, handles continuous naturally |
| Need on-policy | PPO + RPO | Handles continuous with RPO fix |
| Limited compute | SAC or DQN | Off-policy efficiency |
| Learning Q-values matters | DQN | Direct access to action values |

### Key Lessons Learned

1. **REINFORCE:** Simple but high-variance; baseline is critical for harder problems

2. **A2C:** Hyperparameter interactions matter
   - Learning rate × update frequency must be balanced
   - High LR + frequent updates OR low LR + large batches (but not mixed)

3. **PPO:** The robust choice for on-policy
   - Separate networks > shared networks
   - Value clipping around old values, not returns
   - Monitor `clip_fraction` and `approx_kl` for diagnostics

4. **RPO:** Simple fix for PPO's continuous control failures
   - Just add uniform noise to mean during updates
   - Prevents overconfident collapse

5. **SAC:** Dominates for continuous control
   - Off-policy = massive sample efficiency gains
   - Entropy tuning handles exploration automatically

6. **DQN:** Value-based alternative for discrete actions
   - Use default initialization, NOT orthogonal
   - Huber loss + target networks essential for stability
   - Long warmup period (10k+ steps) fills buffer with diverse data
   - Double DQN reduces overestimation if needed

---

## Hyperparameter Debugging Guide

### Diagnostic Metrics

| Metric | Healthy Range | If Unhealthy |
|--------|---------------|--------------|
| `entropy` | Gradual decrease | Collapsed → increase `entropy_coef` |
| `value_loss` | Decreasing, stable | Exploding → lower LR, check reward scale |
| `clip_fraction` (PPO) | 0.1-0.2 | Too high → lower LR |
| `approx_kl` (PPO) | < 0.02 | Spiking → enable `target_kl` |
| `alpha` (SAC) | Decreasing | Stuck high → check target entropy |
| `loss` (DQN) | Stable or slowly increasing | Exploding → add Huber loss, gradient clipping |
| `epsilon` (DQN) | Decaying to ~0.05 | — |

### Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Flat returns | LR too low | Increase LR |
| Improves then collapses | LR too high or value divergence | Lower LR, check value loss |
| High variance | Insufficient baseline | Enable `norm_advantages`, increase batch |
| Entropy collapse | Premature convergence | Increase `entropy_coef` |

---

## Repository Structure

```
├── reinforce.py          # REINFORCE implementation
├── a2c.py                # Advantage Actor-Critic with GAE
├── ppo.py                # PPO for discrete actions
├── ppo_continuous.py     # PPO for continuous actions (with RPO)
├── sac.py                # Soft Actor-Critic
├── dqn.py                # Deep Q-Network
├── net.py                # Neural network architectures
├── util.py               # Buffers (RolloutBuffer, ReplayBuffer)
└── README.md
```

---

## Usage

### Training

```bash
# REINFORCE on CartPole
python reinforce.py --lr 1e-3 --gamma 0.99

# A2C on CartPole (note: small rollouts + high LR)
python a2c.py --lr 1e-3 --rollout_steps 5 --num_envs 16

# PPO on CartPole
python ppo.py --lr 3e-4 --rollout_steps 32 --num_epochs 4

# PPO Continuous on Pendulum (standard - will likely fail)
python ppo_continuous.py --total_timesteps 3000000

# PPO Continuous with RPO (solves Pendulum)
python ppo_continuous.py --use_rpo --rpo_alpha 0.5 --total_timesteps 3000000

# SAC on Pendulum (fastest to solve)
python sac.py --total_timesteps 100000

# DQN on CartPole
python dqn.py --lr 2.5e-4 --learning_starts 10000 --total_timesteps 500000
```

### Monitoring

```bash
tensorboard --logdir ./runs
```

---

## References

- [Policy Gradient Methods for RL with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) — Sutton et al., 1999
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — Mnih et al., 2013 (DQN)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — van Hasselt et al., 2015 (Double DQN)
- [High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438) — Schulman et al., 2015
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) — Haarnoja et al., 2018
- [Robust Policy Optimization](https://arxiv.org/abs/2212.07536) — Rahman et al., 2022
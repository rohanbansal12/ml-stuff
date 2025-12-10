# Solving CartPole with Policy Gradient Methods  
A progressive exploration of modern policy-gradient reinforcement learning on the CartPole-v1 environment.  
This project begins with the simplest Monte-Carlo policy gradient (REINFORCE), adds baselines and GAE via Actor-Critic, and culminates in a full PPO implementation.

---

# 1. Overview

This repository documents a step-by-step journey through fundamental policy-gradient RL algorithms:

1. **REINFORCE (no baseline)** – simplest possible policy gradient  
2. **Actor-Critic with GAE** – lower variance, bootstrapping value estimates  
3. **Proximal Policy Optimization (PPO)** – stable clipped objective, industry standard  

The goal is to demonstrate how each algorithm improves training stability and sample efficiency on the classic **CartPole-v1** benchmark. I also found it took me some time to truly understand these algorithms. In particular, REINFORCE is quite simple for anybody familiar with basic RL, but A2C and GAE are more complex and take more mathematical understanding. PPO is the most complex algorithm but follows naturally from A2C and GAE.

---

# 2. Environment: CartPole-v1

The agent must keep a pole balanced by applying forces left or right.

- **State:** 4 continuous variables  
- **Action space:** 2 discrete actions  
- **Episode terminates:** pole falls or cart leaves bounds  
- **Max score:** 500  

The simplicity of the environment makes it ideal for learning RL fundamentals.

---

# 3. Stage 1 — REINFORCE (Vanilla Policy Gradient)

### Concept

REINFORCE uses the Monte-Carlo return `G_t` to compute an unbiased policy gradient:

$\nabla_\theta J = \mathbb{E} \big[ \nabla_\theta \log \pi_\theta(a_t|s_t) \, G_t \big]$


Properties:

- **Unbiased**, but **high variance**
- No baseline → unstable learning
- Works surprisingly well on small environments

### Implementation Details

- Network: simple MLP → logits over actions  
- Loss:  $L = -\log \pi(a_t|s_t) \, G_t$
- Return computed over full episode
- No bootstrapping

### Observations

- Learning is noisy  
- Often collapses unless LR carefully tuned  
- For simple CartPole, it works reasonably well
    - Reaches a max of around 480 in ~500 episodes and is able to maintain performance above 460 for a while.

---


# 4. Stage 2 — Advantage Actor-Critic (A2C) + GAE

After REINFORCE, we add:

- A **value network** (or shared actor-critic model)  
- **Advantage estimates** instead of Monte-Carlo returns  
- **Generalized Advantage Estimation (GAE-λ)**  

### Advantage Function

$A_t = \delta_t + (\gamma \lambda)\delta_{t+1} + \cdots$

where

$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

GAE balances low variance and low bias.

### Updated Loss

Policy loss:
$L_\pi = - \log \pi(a_t|s_t) A_t$

Value loss:
$L_V = (V(s_t) - R_t)^2$

Entropy bonus (optional):
$L_H = \beta H(\pi)$

### Benefits Over REINFORCE

- Drastically lower variance  
- Much faster convergence  
- Can bootstrap mid-episode  
- Stabilizes long-horizon credit assignment

### Observations

- CartPole is so simple that A2C isn't particularly better than REINFORCE
- Took a lot of config tweaking to get it to work well (not too slow but not unstable/crashing)

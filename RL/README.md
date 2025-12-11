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

### Mathematical Background

We can show that the gradient of our expected return objective $J(\theta)$ with respect to policy parameters $\theta$ is given by:

$\nabla_\theta J = \mathbb{E} \big[ \nabla_\theta \log \pi_\theta(a_t|s_t) \, G_t \big]$

For REINFORCE, we simply estimate this expectation by sampling full episodes and using the observed returns $G_t$. Our model produces logits over the action space, from which we can compute $\log \pi_\theta(a_t|s_t)$. We then use this to compute the policy gradient and update our parameters via gradient ascent.

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

### Mathematical Background

It is simple to observe that for any baseline function of state $b(s_t)$, the policy gradient remains unbiased:

$\nabla_\theta J = \mathbb{E} \big[ \nabla_\theta \log \pi_\theta(a_t|s_t) \, (G_t - b(s_t)) \big]$

The biggest issue with REINFORCE is its high variance due to the full episode return $G_t$, thus we want to introduce a baseline to reduce variance. The optimal baseline in terms of variance reduction is the value function $V(s_t)$, leading to the **advantage function**:

$A_t = G_t - V(s_t)$

which tells us how much better the action $a_t$ was compared to the expected value of state $s_t$. 

We do not know $V(s_t)$, so we learn it with a value network which produces estimates $V_\phi(s_t)$ parameterized by $\phi$. We again want to reduce variance in these value estimates, so we rely on temporal-difference bootstrapping (TD) to produce biased, but lower-variance estimates:

$V_\phi(s_t) \approx r_t + \gamma V_\phi(s_{t+1})$

This is the basic idea behind actor-critic methods, where we use a value network to produce a baseline for the policy gradient.

## GAE Progression

We have now shown that we have 2 extreme ways to estimate the advantage function:

$A_t^{MC} = G_t - V_\phi(s_t)$ (high variance, no bias)

$A_t^{(1)} = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ (low variance, high bias)

These represent the two extremes of a bias-variance tradeoff, so naturally we want to be able to interpolate between them. This is where **Generalized Advantage Estimation (GAE-λ)** comes in. GAE-λ computes a weighted average of n-step advantage estimates. We first compute the TD residuals (1-step advantages):

$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

and our GAE-λ advantage estimate is given by:

$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$

This allows us to tune the hyperparameter $\lambda \in [0, 1]$ to control the bias-variance tradeoff and we can easily see the extreme cases:
- $\lambda = 0$ → 1-step TD advantage (low variance, high bias)
- $\lambda = 1$ → Monte-Carlo advantage (high variance, no bias)

### Updated Loss

Policy loss - this is similar to REINFORCE but using the advantage estimate instead of the full return:
$L_\pi = - \log \pi(a_t|s_t) A_t$

Value loss - we typically use a similar GAE target for the critic updates as well to reduce variance:
$L_V = (V(s_t) - R_t)^2$

Entropy bonus (optional) - encourages exploration so the policy doesn't collapse too quickly:
$L_H = \beta H(\pi)$

### Benefits Over REINFORCE

- Drastically lower variance  
- Much faster convergence  
- Can bootstrap mid-episode  
- Stabilizes long-horizon credit assignment

### Observations

- CartPole is so simple that A2C isn't particularly better than REINFORCE
- Took a lot of config tweaking to get it to work well (not too slow but not unstable/crashing)

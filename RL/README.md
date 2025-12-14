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
    - Reaches an average return of 500 in around 250 episodes and training is quite stable for chosen hyperparameters
    - After a bit, it then crashes and seems to oscillate for a while before getting back to 500 again
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

### GAE Progression

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
    - Sensitive to learning rate, lambda/gamma, model initialization (orthogonal better than xavier)
    - Reached max average return of 500 multiple times (earliest at 8500 rollouts) although sometimes crashes and "learns again"

---

# 5. Stage 3 — Proximal Policy Optimization (PPO)

### Mathematical Background

One thing worth noting about Actor-Critic is that it is incredibly sample inefficient. For each trajectory (or partial rollout) we collect, we only do a single policy and value update. Occasionally, we may try to run multiple updates on a single batch of data, however this often leads to divergence as the policy changes too much from the data it was collected on (called **off-policy** data).

To remedy this, we will convert our gradient expectation (which should use new policy $\pi_\theta$) to use the old policy $\pi_{\theta_{old}}$ that generated the data. This is done via **importance sampling**:

$$g(\theta) = \mathbb{E}_{\pi_{\theta_{old}}} \big[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \nabla_\theta \log \pi_\theta(a_t|s_t) A_t \big] = \mathbb{E}_{\pi_{\theta_{old}}}\big[r_t(\theta)A_t \nabla_\theta \log \pi_\theta(a_t|s_t)\big] = \mathbb{E}_{\pi_{\theta_{old}}}\big[A_t \nabla_\theta r_t(\theta)\big]$$

This gradient makes it immediately clear that we can instead use an objective function:

$$L(\theta) = \mathbb{E}_{\pi_{\theta_{old}}} \big[r_t(\theta) A_t \big]$$

in which we can use the same sample multiple times because the trajectory is still generated from the old policy which makes this expectation valid.

PPO also implements a **clipped surrogate objective** to prevent large policy updates that could destabilize training. This is done by clipping the importance sampling ratio $r_t(\theta)$ within a small range around 1 (e.g., [0.8, 1.2]):

$r_t^{clipped}(\theta) = \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$

This is done because for positive advantages and large $r_t$, the gradient can push us into very greedy policies that overfit to the current batch of data and for negative advantages and small $r_t$, the gradient can push us into very conservative policies by avoiding certain actions altogether. Clipping prevents these extreme updates and keeps the policy changes more stable. 

PPO will combine the results to choose the minimum of the unclipped and clipped objectives:

$L^{PPO}(\theta) = \mathbb{E}_{\pi_{\theta_{old}}} \big[ \min(r_t(\theta) A_t, r_t^{clipped}(\theta) A_t) \big]$

Similar to A2C, we will still keep the value loss and entropy bonus:

$L_V = (V(s_t) - R_t)^2$

$L_H = \beta H(\pi)$

### Implementation Details

- Collect rollout buffer for several episodes  
- Compute advantages via GAE  
- Shuffle + minibatch SGD updates  
- Use old policy for ratio r_t  
- Target KL checked optionally for early stopping  

### Benefits Over Actor-Critic

- More stable training and often more monotonic improvements
- Generally, much better sample efficiency  
- Consistent performance across seeds
- Generally more robust to hyperparameters

### Observations

- It did seem that PPO was less sensitive to hyperparams, however model initialization still seemed to be very important
    - I also found that having an entirely separate critic network (not shared) helped a lot with performance for PPO whereas A2C seemed to work fine with a shared network
- Training was generally quite monotonic and we reached a consistent 490 reward in about 300 rollouts demonstrating good sample efficiency. It does take a while to actually hit the 500 reward exactly over many espisodes, but it gets very close quite quickly which is often more important for general RL tasks.

---

### Continuous PPO

I also implemented a continuous version of PPO to solve "Pendulum-v1". The primary change is that our model now outputs a mean and a log stdev which are used to sample from a Normal distribution. Modern implementations seem to use state-specific stddevs and tanh squashing + Jacobian correction to ensure that actions remain in bounds and our log-probs are appropriately adjusted, however I just used a naive clipping approach for simplicity. Ironically, it seems after some research that PPO continuous is not able to consistently solve Pendulum-v1 because it locks onto a suboptimal policy and then aggressively scales down action variance. The solution, however, is quite simple--a model called RPO which adds a constant factor the action mean prior to Normal sampling to ensure persistent exploration.

After implementing this, I was able to get a fairly stable and robust training setup (the full hyperparameters can be seen in the code), but the average return got to around -150 in around 90 outer updates and the model was able to maintain performance around this range through the end of training. Pendulum-v1 typically starts with a return of around -1400 and is considered "solved" at around -200 with the return always being negative.

# 6. Stage 4 — Soft Actor-Critic (SAC)

Soft Actor-Critic (SAC) is an **off-policy** actor-critic algorithm primarily designed for **continuous control**. It combines:

- A **stochastic policy** trained via the reparameterization trick  
- **Double Q-learning** to reduce overestimation bias  
- A **maximum-entropy objective** to encourage exploration  
- A **replay buffer** for high sample efficiency  
- **Target networks** updated via Polyak averaging for stability  

Compared to PPO, SAC trades on-policy stability for **off-policy efficiency and robustness** in continuous domains.


## Mathematical Background

### Maximum-Entropy Objective

Instead of maximizing expected return alone, SAC maximizes a **soft** objective that includes an entropy bonus:

$J(\pi) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \left(r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right)\right]$

where the policy entropy is defined as:

$\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a\sim\pi(\cdot|s)}[\log \pi(a|s)]$

Equivalently, when sampling $a_t \sim \pi(\cdot|s_t)$, the per-step objective becomes:

$r(s_t,a_t) - \alpha \log \pi(a_t|s_t)$

The temperature parameter $\alpha > 0$ controls the tradeoff between exploration and exploitation.

---

### Soft Value Functions

Define the **soft Q-function** as:

$Q^\pi(s,a) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k \left(r_{t+k} - \alpha \log \pi(a_{t+k}|s_{t+k})\right)\,\middle|\,s_t=s, a_t=a\right]$

Define the **soft value function** as:

$V^\pi(s) = \mathbb{E}_{a\sim\pi}\left[Q^\pi(s,a) - \alpha \log \pi(a|s)\right]$

These satisfy the **soft Bellman equation**:

$Q^\pi(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}[V^\pi(s')]$

Substituting $V^\pi$ gives:

$Q^\pi(s,a) = r(s,a) + \gamma \mathbb{E}_{s',a'\sim\pi}\left[Q^\pi(s',a') - \alpha \log \pi(a'|s')\right]$

---

### Critic Learning (Soft Policy Evaluation)

SAC approximates $Q^\pi$ using two critics $Q_{\phi_1}, Q_{\phi_2}$ and corresponding target networks $Q_{\phi_1'}, Q_{\phi_2'}$.

Given a replay transition $(s,a,r,s',d)$, we sample $a' \sim \pi_\theta(\cdot|s')$ and compute the TD target:

$y = r + \gamma (1-d) \left(\min(Q_{\phi_1'}(s',a'), Q_{\phi_2'}(s',a')) - \alpha \log \pi_\theta(a'|s')\right)$

Each critic minimizes the mean-squared TD error:

$\mathcal{L}_{Q_i} = \mathbb{E}\left[(Q_{\phi_i}(s,a) - y)^2\right], \quad i \in \{1,2\}$

Using the minimum over critics reduces overestimation bias and improves stability.

---

### Actor Learning (Soft Policy Improvement)

The optimal maximum-entropy policy satisfies:

$\pi^*(\cdot|s) \propto \exp\left(\frac{1}{\alpha} Q^*(s,\cdot)\right)$

Instead of solving this directly, SAC performs gradient descent on the policy parameters $\theta$ using the objective:

$\mathcal{L}_\pi(\theta) = \mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_\theta}\left[\alpha \log \pi_\theta(a|s) - \min(Q_{\phi_1}(s,a), Q_{\phi_2}(s,a))\right]$

The $Q$-term encourages high-value actions, while the entropy term prevents premature policy collapse.

---

### Stochastic Policy and Reparameterization

For continuous control, SAC uses a Gaussian policy with tanh squashing:

$u \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)), \qquad a = \tanh(u)$

Sampling uses the reparameterization trick so gradients can flow through the sampled action.

#### Tanh Squashing Log-Probability Correction

Since $a = \tanh(u)$ is a change of variables, the log-probability must be corrected:

$\log \pi_\theta(a|s) = \log \mathcal{N}(u;\mu_\theta(s),\sigma_\theta(s)) - \sum_j \log\left(1 - \tanh^2(u_j)\right)$

This correction is critical for correct entropy estimation and stable learning.

Actions are typically rescaled affinely to match environment bounds; this contributes only a constant Jacobian term and is often omitted from optimization.

---

### Entropy Temperature Autotuning

Rather than fixing $\alpha$, SAC automatically tunes it to maintain a target entropy:

$\mathcal{H}_{\text{target}} \approx -\text{action\_dim}$

The temperature parameter is optimized via:

$\mathcal{L}_\alpha = -\mathbb{E}\left[\log \alpha \left(\log \pi_\theta(a|s) + \mathcal{H}_{\text{target}}\right)\right]$

This keeps the policy sufficiently stochastic early in training and nearly deterministic at convergence.

---

### Target Networks and Polyak Averaging

To stabilize bootstrapped Q-learning, SAC uses slowly updated target networks:

$\phi_i' \leftarrow \tau \phi_i + (1 - \tau) \phi_i'$

with $\tau \ll 1$ (e.g., $0.005$).


## Implementation Details

- Off-policy learning using a replay buffer storing $(s,a,r,s',done)$  
- Two critics and two target critics  
- Squashed Gaussian actor with reparameterized sampling  
- Losses:
  - Critic: soft TD error  
  - Actor: entropy-regularized policy loss  
  - Alpha: entropy-matching objective  
- Target networks updated via Polyak averaging  


## Benefits Over PPO

- Much better sample efficiency due to off-policy learning  
- Stronger and more stable exploration via entropy maximization  
- Particularly effective for continuous control tasks (Pendulum, MuJoCo)  


## Observations

- Pendulum-v1 typically starts around $-1200$ return  
- SAC reliably reaches around $-200$ (considered solved) and often $-150$ or better  
- Correct tanh log-prob correction and done handling are essential for success
# Neural Ordinary Differential Equations

Connecting neural networks with differential equations for continuous-depth models.

## Suggested Order

| # | File | Concept | Difficulty |
|---|------|---------|------------|
| 1 | `ode_basics.py` | Numerical ODE solvers (Euler, RK4) | Easy |
| 2 | `neural_ode.py` | Core Neural ODE + Adjoint method | Medium |
| 3 | `augmented_ode.py` | Fixing topology limitations | Medium |
| 4 | `regularization.py` | Controlling dynamics complexity | Medium |
| 5 | `cnf.py` | Continuous Normalizing Flows | Hard |
| 6 | `latent_ode.py` | Irregular time series | Hard |

## Core Idea

**Standard ResNet:**
```
h[l+1] = h[l] + f(h[l], θ[l])
```

**Neural ODE (infinite depth limit):**
```
dh/dt = f(h(t), t, θ)
h(T) = h(0) + ∫₀ᵀ f(h(t), t, θ) dt
```

## Key Concepts

### Forward Pass
Solve the ODE numerically (Euler, RK4, adaptive methods).

### Backward Pass (Adjoint Method)
Instead of backprop through solver steps (O(L) memory):
- Define adjoint: a(t) = dL/dh(t)
- Solve adjoint ODE backwards: da/dt = -aᵀ(∂f/∂h)
- O(1) memory regardless of solver steps!

### Continuous Normalizing Flows
For generative modeling:
- Track density change: d(log p)/dt = -tr(∂f/∂z)
- Only need trace, not full Jacobian determinant
- Hutchinson estimator: tr(A) = E[vᵀAv]

## Key Papers

1. **Chen et al. (2018)** - "Neural Ordinary Differential Equations" (NeurIPS Best Paper)
2. **Grathwohl et al. (2018)** - "FFJORD: Free-form Continuous Dynamics" (CNF)
3. **Dupont et al. (2019)** - "Augmented Neural ODEs" (fixing topology)
4. **Rubanova et al. (2019)** - "Latent ODEs for Irregularly-Sampled Time Series"
5. **Finlay et al. (2020)** - "How to Train Your Neural ODE" (regularization)

## Quick Reference

```python
# ODE integration (forward)
z(T) = z(0) + ∫ f(z, t, θ) dt

# Adjoint (backward)
a(t) = dL/dz(t)
da/dt = -aᵀ (∂f/∂z)
dL/dθ = -∫ aᵀ (∂f/∂θ) dt

# CNF density
log p(z(T)) = log p(z(0)) - ∫ tr(∂f/∂z) dt

# Hutchinson trace estimator
tr(A) ≈ vᵀAv,  v ~ N(0,I)
```

## Practical Tips

1. **Start simple**: Implement Euler before RK4, fixed step before adaptive
2. **Monitor NFE**: Number of function evaluations indicates dynamics complexity
3. **Regularize**: Kinetic energy regularization prevents stiff dynamics
4. **Augment if needed**: Add dimensions if topology is limiting
5. **Use existing libraries**: `torchdiffeq` for production, implement from scratch for learning

## Dependencies

```python
# For reference implementation
pip install torchdiffeq

# But implement from scratch first for understanding!
```

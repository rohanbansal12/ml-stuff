# Bayesian Neural Networks

Learning to quantify uncertainty in neural network predictions.

## Suggested Order

| # | File | Concept | Difficulty |
|---|------|---------|------------|
| 1 | `bayesian_linreg.py` | Closed-form Bayesian inference | Easy |
| 2 | `mc_dropout.py` | Dropout as approximate inference | Easy |
| 3 | `uncertainty_types.py` | Epistemic vs aleatoric | Medium |
| 4 | `deep_ensembles.py` | Strong practical baseline | Medium |
| 5 | `bayes_by_backprop.py` | Variational inference | Hard |
| 6 | `laplace.py` | Post-hoc approximation | Hard |
| 7 | `ood_detection.py` | Application: OOD detection | Medium |

## Core Concepts

### The Bayesian View
- **Standard NN**: Learn point estimate `w*` that maximizes likelihood
- **Bayesian NN**: Learn distribution `p(w|D)` over weights
- **Prediction**: Integrate over all possible weights

### Why Uncertainty Matters
- Know when the model doesn't know
- Better calibrated predictions
- Safer deployment in high-stakes applications
- Active learning (query most uncertain examples)

### Two Types of Uncertainty
- **Epistemic**: Model uncertainty, reducible with more data
- **Aleatoric**: Data noise, irreducible

## Key Papers

1. MacKay (1992) - Laplace approximation for NNs
2. Blundell et al. (2015) - "Bayes by Backprop" / Weight uncertainty
3. Gal & Ghahramani (2016) - Dropout as Bayesian approximation
4. Lakshminarayanan et al. (2017) - Deep Ensembles
5. Kendall & Gal (2017) - Epistemic vs Aleatoric uncertainty

## Quick Reference

```python
# MC Dropout
model.train()  # Keep dropout on
preds = [model(x) for _ in range(100)]
mean, std = torch.stack(preds).mean(0), torch.stack(preds).std(0)

# Deep Ensemble
preds = [member(x) for member in ensemble]
mean, std = torch.stack(preds).mean(0), torch.stack(preds).std(0)

# Bayes by Backprop
# w ~ N(mu, softplus(rho)^2), learned via ELBO

# Laplace
# p(w|D) ≈ N(w_MAP, H^{-1}) where H = Hessian at MAP
```

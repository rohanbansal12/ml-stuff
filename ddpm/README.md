# DDPM on CIFAR-10
A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on CIFAR-10, including a U-Net backbone, noise schedules, sampling methods, and ablation experiments.

---

## 1. Overview

This project implements a DDPM (Denoising Diffusion Probabilistic Model) trained on the CIFAR-10 dataset (32x32 natural images). Everything was written from scratch in PyTorch (no reference implementations used)

Goals:
- Understand forward diffusion and reverse denoising
- Build a U-Net with optional self-attention
- Train DDPM on CIFAR-10 and generate samples
- Run ablations on architecture and sampling choices

Good intro blog post: [Lilian Weng Blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

DDPM paper: [DDPM Paper](https://arxiv.org/abs/2006.11239)

---

## 2. Diffusion Model Summary

### Forward Process ($q$)
A fixed noise schedule gradually corrupts images:
$q(x_t | x_{t-1}) = N(\sqrt{1 - \beta_t} * x_{t-1}, \beta_t * I)$

We experiment with linear and cosine beta schedules.

### Reverse Process ($p_\theta$)
The model predicts:
- $\epsilon$ (default)
- optionally: $x_0$ or $v$ (for ablations)
    - $x_0$ good for sharpness and high-res detail but unstable in high noise
    - $v$ used in modern models like DiT to balance $x_0$ for low $t$ and $\epsilon$ for high $t$

Sampling iteratively reconstructs the image:
$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t * z$

---

## 3. U-Net Architecture

The U-Net follows the DDPM architecture:
- Downsampling path with residual blocks
- Optional self-attention at 16x16 (and bottleneck)
- Sinusoidal time embeddings passed into each block
- Upsampling path with skip connections
- Output predicts noise epsilon

### Configurable components:
| Component | Default |
|----------|---------|
| Base channels | 128 |
| Attention | 16x16 resolution |
| Time embedding | sinusoidal + MLP |
| Normalization | GroupNorm |

---

## 6. Ablations

### 6.1 Self-Attention
| Config | Notes |
|--------|-------|
| No attention | baseline |
| Attention at bottleneck | marginal improvement |
| Attention at 16x16 | better global structure |

### 6.3 Loss Target
| Target | Notes |
|--------|-------|
| $\epsilon$ | stable, recommended |
| $x_0$ | sometimes sharper but unstable |
| $v$ | used by modern models (e.g., DiT) |
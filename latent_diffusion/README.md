# Latent Diffusion from Scratch

A from-scratch implementation of latent diffusion models in PyTorch, trained on CIFAR-10. This builds on the VAE and DDPM implementations to create a complete latent diffusion pipeline similar to the core architecture behind Stable Diffusion.

---

## Overview

Latent diffusion runs the diffusion process in a compressed latent space rather than pixel space. This provides significant computational savings while maintaining image quality.

| Component | Description |
|-----------|-------------|
| **VAE** | Compresses 32×32×3 images → 8×8×4 latents (24× fewer dimensions) |
| **Latent U-Net** | Smaller U-Net that denoises in latent space |
| **Diffusion** | Standard DDPM/DDIM, just operating on latents |
| **CFG** | Classifier-free guidance for sharper, class-conditional samples |

---

## Architecture

### Training Pipeline

```
Image (32×32×3) → VAE Encoder → Latent z₀ (8×8×4) → Add noise → z_t → U-Net → Predict ε
                     ↓                                                    ↓
                  (frozen)                                           MSE Loss
```

### Sampling Pipeline

```
z_T ~ N(0,I) → U-Net denoise (1000 steps) → z₀ → VAE Decoder → Image (32×32×3)
                      ↓
              (with optional CFG)
```

### Classifier-Free Guidance (CFG)

CFG dramatically improves sample sharpness by extrapolating away from unconditional predictions:

$$\epsilon_{guided} = \epsilon_{uncond} + s \cdot (\epsilon_{cond} - \epsilon_{uncond})$$

where $s$ is the guidance scale (typically 3-7).

**Training:** Randomly drop class labels 10% of the time (replace with null token)

**Sampling:** Run model twice per step—conditional and unconditional—then combine

---

## Results on CIFAR-10

| Model | Epochs | Loss | Sample Quality |
|-------|--------|------|----------------|
| Unconditional | 800 | 0.290 | Blurry but coherent |
| CFG (scale=5) | 800 | 0.281 | Sharper, clearer objects |

The CFG model shows noticeably better object boundaries and color separation. Sample quality is bounded by VAE reconstruction quality.

---

## Key Implementation Details

### Latent Scaling

VAE latents don't have unit variance. We compute a scale factor to normalize:

```python
scale = 1.0 / latents.std()  # Typically ~1.2-1.5 for our VAE
z_0 = mu * scale             # Training
z_0 = z_0 / scale            # Before decoding
```

### CFG in the Model

```python
# In forward pass, randomly drop labels during training
if self.training and self.cfg_dropout > 0:
    drop_mask = torch.rand(B) < self.cfg_dropout
    class_labels = torch.where(drop_mask, self.null_class_idx, class_labels)

c_emb = self.class_emb(class_labels)
t_emb = t_emb + c_emb  # Add class embedding to time embedding
```

### CFG in Sampling

```python
# Two forward passes per step
pred_cond = model(z_t, t, class_labels=labels, drop_class=False)
pred_uncond = model(z_t, t, class_labels=labels, drop_class=True)
pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

---

## Usage

### Prerequisites

Train a VAE first (see `../vae/`):
```bash
cd ../vae
python train_vae.py --preset fast --kl-weight 0.001 --run-name vae_for_ldm
```

### Training

**Unconditional:**
```bash
python train_latent.py --preset fast \
    --vae-checkpoint /path/to/vae/checkpoint_final.pt \
    --run-name unconditional \
    --epochs 800 \
    --batch-size 512 \
    --lr 1e-4
```

**With CFG (recommended):**
```bash
python train_latent.py --preset fast \
    --vae-checkpoint /path/to/vae/checkpoint_final.pt \
    --run-name cfg \
    --num-classes 10 \
    --guidance-scale 5.0 \
    --epochs 800 \
    --batch-size 512 \
    --lr 1e-4
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--preset` | — | Config preset: `small`, `base`, `fast` |
| `--vae-checkpoint` | required | Path to trained VAE |
| `--num-classes` | 0 | Number of classes (0=unconditional, 10 for CIFAR-10) |
| `--cfg-dropout` | 0.1 | Probability of dropping class during training |
| `--guidance-scale` | 0.0 | CFG scale at sampling (try 3-7) |
| `--epochs` | 200 | Training epochs |
| `--lr` | 3e-4 | Learning rate (1e-4 recommended) |

### Monitoring

```bash
tensorboard --logdir ./runs
```

---

## File Structure

```
├── config.py          # Dataclass configs for U-Net, diffusion, training
├── model.py           # Latent U-Net with optional class conditioning
├── diffusion.py       # Noise schedule, DDPM/DDIM samplers with CFG
├── train_latent.py    # Training script
└── README.md
```

---

## Lessons Learned

1. **Latent scaling matters** — Without normalizing latent variance, diffusion assumptions break down

2. **CFG is essential for sharp samples** — The difference between blurry and recognizable is mostly CFG, not more training

3. **VAE quality is the ceiling** — Diffusion can't produce sharper images than the VAE can reconstruct

4. **Diffusion models are slow learners** — 800+ epochs for decent results; loss decreases slowly but steadily

5. **Small latent space works** — 8×8×4 = 256 dimensions is enough for 32×32 CIFAR-10

---

## Comparison: Pixel vs Latent Diffusion

| Aspect | Pixel-Space DDPM | Latent Diffusion |
|--------|------------------|------------------|
| Input dimensions | 3,072 (32×32×3) | 256 (8×8×4) |
| U-Net size | ~50M params | ~10-20M params |
| Training speed | Slower | ~3-5× faster |
| Memory usage | Higher | Lower |
| Sample quality | Direct | Bounded by VAE |

---

## References

- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Rombach et al., 2022
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Ho & Salimans, 2022
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., 2020
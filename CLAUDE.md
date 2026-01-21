# CLAUDE.md

This repository contains machine learning experiments for self-study. Each directory implements a different ML technique from scratch, with a focus on understanding the underlying theory.

## Package Management

**Use `uv` for all package management and running scripts.**

```bash
# Install dependencies
uv sync

# Run a script
uv run python path/to/script.py

# Run from within a subdirectory
cd neural_ode && uv run python neural_ode.py

# Add a new dependency
uv add <package-name>
```

## Project Structure

| Directory | Topic |
|-----------|-------|
| `RL/` | Reinforcement learning (REINFORCE, A2C, PPO, SAC, DQN) |
| `bayesian/` | Bayesian deep learning (MC Dropout, Deep Ensembles, Bayes by Backprop, Laplace) |
| `neural_ode/` | Neural ODEs, continuous normalizing flows, latent ODEs |
| `gpt/` | GPT-style transformer language model |
| `ddpm/` | Denoising diffusion probabilistic models |
| `latent_diffusion/` | Latent diffusion models |
| `vae/` | Variational autoencoders |
| `resnet/` | ResNet implementation and ablations |
| `ViT/` | Vision Transformer |
| `simclr/` | Self-supervised contrastive learning |
| `LoRA/` | Low-rank adaptation for fine-tuning |
| `MoE/` | Mixture of Experts |
| `RLHF/` | Reinforcement learning from human feedback (DPO, IPO) |
| `quantization/` | Model quantization techniques |
| `optim/` | Custom optimizers (Adam, AdamW, etc. from scratch) |

## Conventions

- Each subdirectory typically has its own `README.md` with theory and usage
- Scripts are self-contained and can be run directly (e.g., `uv run python rl/ppo.py`)
- Plots are saved to a `plots/` subdirectory within each experiment folder
- PyTorch is the deep learning framework used throughout
- Most scripts download required datasets (MNIST, CIFAR-10, etc.) automatically to `./data/`
- Ensure that all imports are located at the top of each script for clarity

## Common Commands

```bash
# Check PyTorch and GPU availability
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run linting
uv run ruff check .

# Run tests (if any)
uv run pytest
```

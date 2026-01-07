"""
VAE Configuration - extends the existing config system.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class VAEModelConfig:
    """VAE architecture configuration."""

    channels: int = 64  # Base channel count
    channel_mults: tuple = (1, 2, 4)  # Multipliers for each resolution level
    latent_dim: int = 128  # Dimension of latent space (for vanilla VAE)
    latent_channels: int = 4  # Channels in spatial latent (for conv VAE)
    groups: int = 8  # GroupNorm groups
    num_res_blocks: int = 2  # ResBlocks per resolution level
    use_spatial_latent: bool = True  # If True: latent is (B, C, H, W), else (B, D)


@dataclass
class DiscriminatorConfig:
    """PatchGAN discriminator configuration."""

    channels: int = 64  # Base channel count
    n_layers: int = 3  # Number of conv layers
    use_spectral_norm: bool = True  # Spectral normalization for stability


@dataclass
class VAETrainConfig:
    """VAE training configuration."""

    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-4  # Reduced from 1e-3 for GAN stability
    weight_decay: float = 0.0

    # VAE-specific
    kl_weight: float = 1.0  # β in β-VAE
    kl_warmup_epochs: int = 10  # Linear warmup for KL weight
    recon_loss: Literal["mse", "l1"] = "mse"

    # Perceptual loss (LPIPS)
    use_lpips: bool = False  # Enable LPIPS perceptual loss
    lpips_weight: float = 0.1  # Weight for LPIPS loss

    # Adversarial loss (GAN)
    use_adversarial: bool = False  # Enable adversarial training
    adv_weight: float = 0.1  # Weight for adversarial loss
    adv_start_epoch: int = 0  # Epoch to start adversarial training
    disc_lr: float = 1e-4  # Discriminator learning rate
    disc_steps: int = 1  # Discriminator steps per generator step

    # Logging
    log_every: int = 50
    sample_every: int = 5
    save_every: int = 20
    num_samples: int = 16

    # Performance
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    compile_model: bool = False
    tf32: bool = True


@dataclass
class VAEConfig:
    """Master VAE configuration."""

    model: VAEModelConfig = field(default_factory=VAEModelConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    train: VAETrainConfig = field(default_factory=VAETrainConfig)

    # Experiment metadata
    run_name: str | None = None
    seed: int = 42
    data_dir: str = "./data"
    log_dir: str = "./runs"

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "VAEConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(
            model=VAEModelConfig(**data["model"]),
            discriminator=DiscriminatorConfig(**data.get("discriminator", {})),
            train=VAETrainConfig(**data["train"]),
            run_name=data.get("run_name"),
            seed=data.get("seed", 42),
            data_dir=data.get("data_dir", "./data"),
            log_dir=data.get("log_dir", "./runs"),
        )


# Presets
VAE_PRESETS = {
    "tiny": VAEConfig(
        model=VAEModelConfig(
            channels=32, channel_mults=(1, 2), num_res_blocks=1, latent_channels=2
        ),
        train=VAETrainConfig(epochs=20),
    ),
    "small": VAEConfig(
        model=VAEModelConfig(channels=64, channel_mults=(1, 2, 4), latent_channels=4),
        train=VAETrainConfig(epochs=50),
    ),
    "base": VAEConfig(
        model=VAEModelConfig(channels=128, channel_mults=(1, 2, 4, 4), latent_channels=4),
        train=VAETrainConfig(epochs=100, kl_weight=0.5),
    ),
    "fast": VAEConfig(
        model=VAEModelConfig(channels=128, channel_mults=(1, 2, 4), latent_channels=4),
        train=VAETrainConfig(
            epochs=200,
            mixed_precision="bf16",
            compile_model=True,
            batch_size=256,
            num_workers=8,
        ),
    ),
    # New preset with perceptual + adversarial loss
    "sharp": VAEConfig(
        model=VAEModelConfig(channels=128, channel_mults=(1, 2, 4), latent_channels=4),
        discriminator=DiscriminatorConfig(channels=64, n_layers=3),
        train=VAETrainConfig(
            epochs=400,
            lr=1e-4,
            kl_weight=0.001,
            use_lpips=True,
            lpips_weight=0.1,
            use_adversarial=True,
            adv_weight=0.1,
            adv_start_epoch=5,  # Let VAE warm up first
            mixed_precision="bf16",
            batch_size=512,
            num_workers=12,
        ),
    ),
}

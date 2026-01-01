"""
Latent Diffusion Configuration.

Latent diffusion runs DDPM in the VAE's latent space instead of pixel space.
This is much faster since the latent is typically 8x8x4 = 256 dims vs 32x32x3 = 3072 dims.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json
from pathlib import Path


@dataclass
class LatentUNetConfig:
    """
    U-Net config for latent space.

    Key differences from pixel-space U-Net:
    - Input/output channels = latent_channels (4) instead of RGB (3)
    - Fewer downsampling levels since latent is already 8x8
    - Can be smaller overall since latent space is more compact
    """

    in_channels: int = 4  # VAE latent channels
    out_channels: int = 4
    channels: int = 128  # Base channel count
    channel_mults: tuple = (1, 2, 2)  # Fewer levels for 8x8 input
    time_dim: int = 256  # Time embedding dimension
    groups: int = 8  # GroupNorm groups
    num_res_blocks: int = 2
    attn_resolutions: tuple = (4,)  # Attention at 4x4 resolution
    use_bottleneck_attn: bool = True
    dropout: float = 0.0

    # Classifier-free guidance settings
    num_classes: int = 0  # 0 = unconditional, >0 = class-conditional
    cfg_dropout: float = 0.1  # Probability of dropping class label during training


@dataclass
class LatentDiffusionConfig:
    """Diffusion process config (same as regular DDPM)."""

    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: Literal["linear", "cosine"] = "cosine"
    pred_type: Literal["eps", "x0", "v"] = "eps"
    cosine_s: float = 0.008


@dataclass
class LatentTrainConfig:
    """Training configuration."""

    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    ema_decay: float = 0.9999

    # Logging
    log_every: int = 50
    sample_every: int = 5
    save_every: int = 20
    num_samples: int = 16

    # Performance
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    compile_model: bool = False
    tf32: bool = True

    # Classifier-free guidance
    guidance_scale: float = 0.0  # 0 = no guidance, >0 = use CFG at sampling


@dataclass
class LatentDiffusionFullConfig:
    """Master configuration for latent diffusion."""

    model: LatentUNetConfig = field(default_factory=LatentUNetConfig)
    diffusion: LatentDiffusionConfig = field(default_factory=LatentDiffusionConfig)
    train: LatentTrainConfig = field(default_factory=LatentTrainConfig)

    # VAE settings
    vae_checkpoint: str = ""  # Path to trained VAE checkpoint
    latent_scale: float = 1.0  # Scale factor for latent normalization
    latent_shape: tuple = (4, 8, 8)  # Expected latent shape (C, H, W)

    # Experiment metadata
    run_name: Optional[str] = None
    seed: int = 42
    data_dir: str = "./data"
    log_dir: str = "./runs"

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LatentDiffusionFullConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(
            model=LatentUNetConfig(**data["model"]),
            diffusion=LatentDiffusionConfig(**data["diffusion"]),
            train=LatentTrainConfig(**data["train"]),
            vae_checkpoint=data.get("vae_checkpoint", ""),
            latent_scale=data.get("latent_scale", 1.0),
            latent_shape=tuple(data.get("latent_shape", (4, 8, 8))),
            run_name=data.get("run_name"),
            seed=data.get("seed", 42),
            data_dir=data.get("data_dir", "./data"),
            log_dir=data.get("log_dir", "./runs"),
        )


LATENT_PRESETS = {
    "small": LatentDiffusionFullConfig(
        model=LatentUNetConfig(channels=64, channel_mults=(1, 2)),
        diffusion=LatentDiffusionConfig(T=1000),
        train=LatentTrainConfig(epochs=100),
    ),
    "base": LatentDiffusionFullConfig(
        model=LatentUNetConfig(channels=128, channel_mults=(1, 2, 2)),
        diffusion=LatentDiffusionConfig(T=1000, schedule_type="cosine"),
        train=LatentTrainConfig(epochs=200),
    ),
    "fast": LatentDiffusionFullConfig(
        model=LatentUNetConfig(channels=128, channel_mults=(1, 2, 2)),
        diffusion=LatentDiffusionConfig(T=1000, schedule_type="cosine"),
        train=LatentTrainConfig(
            epochs=200,
            mixed_precision="bf16",
            compile_model=True,
            batch_size=256,
            num_workers=8,
        ),
    ),
}

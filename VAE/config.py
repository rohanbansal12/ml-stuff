"""
VAE Configuration - extends the existing config system.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json
from pathlib import Path


@dataclass
class VAEModelConfig:
    """VAE architecture configuration."""
    channels: int = 64                      # Base channel count
    channel_mults: tuple = (1, 2, 4)        # Multipliers for each resolution level
    latent_dim: int = 128                   # Dimension of latent space (for vanilla VAE)
    latent_channels: int = 4                # Channels in spatial latent (for conv VAE)
    groups: int = 8                         # GroupNorm groups
    num_res_blocks: int = 2                 # ResBlocks per resolution level
    use_spatial_latent: bool = True         # If True: latent is (B, C, H, W), else (B, D)


@dataclass
class VAETrainConfig:
    """VAE training configuration."""
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    
    # VAE-specific
    kl_weight: float = 1.0                  # β in β-VAE
    kl_warmup_epochs: int = 10              # Linear warmup for KL weight
    recon_loss: Literal["mse", "l1"] = "mse"
    
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
    train: VAETrainConfig = field(default_factory=VAETrainConfig)
    
    # Experiment metadata
    run_name: Optional[str] = None
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
            train=VAETrainConfig(**data["train"]),
            run_name=data.get("run_name"),
            seed=data.get("seed", 42),
            data_dir=data.get("data_dir", "./data"),
            log_dir=data.get("log_dir", "./runs"),
        )


VAE_PRESETS = {
    "tiny": VAEConfig(
        model=VAEModelConfig(channels=32, channel_mults=(1, 2), num_res_blocks=1, latent_channels=2),
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
        model=VAEModelConfig(channels=128, channel_mults=(1, 2, 4, 4), latent_channels=4),
        train=VAETrainConfig(
            epochs=100,
            mixed_precision="bf16",
            compile_model=True,
            batch_size=256,
            num_workers=8,
        ),
    ),
}
"""
Configuration system for DDPM training.
All hyperparameters in one place, easily serializable for reproducibility.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """U-Net architecture configuration."""

    channels: int = 64  # Base channel count
    channel_mults: tuple = (1, 2, 4)  # Multipliers for each resolution level
    time_dim: int = 128  # Time embedding dimension
    groups: int = 8  # GroupNorm groups
    num_res_blocks: int = 2  # ResBlocks per resolution level
    attn_resolutions: tuple = (
        16,
    )  # Resolutions where attention is applied (e.g., 16x16)
    use_bottleneck_attn: bool = True
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""

    T: int = 1000  # Number of diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: Literal["linear", "cosine"] = "linear"
    pred_type: Literal["eps", "x0", "v"] = "eps"
    cosine_s: float = 0.008  # Offset for cosine schedule


@dataclass
class TrainConfig:
    """Training configuration."""

    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0  # Gradient clipping (None to disable)
    ema_decay: float = 0.9999  # EMA decay (0 to disable)

    # Logging
    log_every: int = 50  # Log every N steps
    sample_every: int = 1  # Sample every N epochs
    save_every: int = 10  # Checkpoint every N epochs
    num_samples: int = 16  # Number of samples to generate

    # Performance
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)
    gradient_accumulation: int = 1  # Accumulate gradients over N steps
    tf32: bool = True  # Allow TF32 on Ampere+ GPUs


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Experiment metadata
    run_name: Optional[str] = None
    seed: int = 42
    data_dir: str = "./data"
    log_dir: str = "./runs"

    def save(self, path: Path):
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data["model"]),
            diffusion=DiffusionConfig(**data["diffusion"]),
            train=TrainConfig(**data["train"]),
            run_name=data.get("run_name"),
            seed=data.get("seed", 42),
            data_dir=data.get("data_dir", "./data"),
            log_dir=data.get("log_dir", "./runs"),
        )


# Preset configurations for quick experiments
PRESETS = {
    "tiny": Config(
        model=ModelConfig(channels=32, channel_mults=(1, 2), num_res_blocks=1),
        diffusion=DiffusionConfig(T=200),
        train=TrainConfig(epochs=20),
    ),
    "small": Config(
        model=ModelConfig(channels=64, channel_mults=(1, 2, 4), num_res_blocks=2),
        diffusion=DiffusionConfig(T=1000),
        train=TrainConfig(epochs=100),
    ),
    "base": Config(
        model=ModelConfig(channels=128, channel_mults=(1, 2, 2, 2), num_res_blocks=2),
        diffusion=DiffusionConfig(T=1000, schedule_type="cosine"),
        train=TrainConfig(epochs=200, ema_decay=0.9999),
    ),
    # Fast training preset for powerful GPUs
    "fast": Config(
        model=ModelConfig(channels=128, channel_mults=(1, 2, 2, 2), num_res_blocks=2),
        diffusion=DiffusionConfig(T=1000, schedule_type="cosine"),
        train=TrainConfig(
            epochs=200,
            ema_decay=0.9999,
            mixed_precision="bf16",
            compile_model=True,
            tf32=True,
            batch_size=256,
            num_workers=8,
        ),
    ),
}

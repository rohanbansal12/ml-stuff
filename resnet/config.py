from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class TrainConfig:
    # Model
    model: str = "resnet18"
    normalize: bool = True

    # Training
    epochs: int = 100
    batch_size: int = 512
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # LR Schedule
    lr_schedule: str = "multistep"  # "none", "multistep", "cosine"
    lr_milestones: List[int] = field(default_factory=lambda: [100, 150])
    lr_gamma: float = 0.1

    # Data
    data_dir: str = "./data"
    num_workers: int = 12

    # Logging
    run_name: Optional[str] = None
    log_dir: str = "./runs"

    # Reproducibility
    seed: Optional[int] = None

    def to_args(self) -> List[str]:
        """Convert config to command-line arguments."""
        args = [
            "--model", self.model,
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--lr", str(self.lr),
            "--momentum", str(self.momentum),
            "--weight-decay", str(self.weight_decay),
            "--lr-schedule", self.lr_schedule,
            "--lr-gamma", str(self.lr_gamma),
            "--data-dir", self.data_dir,
            "--num-workers", str(self.num_workers),
            "--log-dir", self.log_dir,
        ]

        if not self.normalize:
            args.append("--no-normalize")

        if self.lr_milestones:
            args.extend(["--lr-milestones"] + [str(m) for m in self.lr_milestones])

        if self.run_name:
            args.extend(["--run-name", self.run_name])

        if self.seed is not None:
            args.extend(["--seed", str(self.seed)])

        return args

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            return cls(**json.load(f))


# Preset configurations for common ablations
CONFIGS = {
    "baseline": TrainConfig(),

    "resnet34": TrainConfig(model="resnet34"),

    "resnet50": TrainConfig(model="resnet50"),

    "no_batchnorm": TrainConfig(normalize=False, lr=0.01),

    "cosine_schedule": TrainConfig(lr_schedule="cosine"),

    "no_schedule": TrainConfig(lr_schedule="none", epochs=100),

    "small_batch": TrainConfig(batch_size=32, lr=0.025),

    "large_batch": TrainConfig(batch_size=1024, lr=0.2),
}


def get_config(name: str) -> TrainConfig:
    """Get a preset config by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]

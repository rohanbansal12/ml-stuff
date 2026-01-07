"""
DDPM Training loop.

Features:
- Clean training loop with proper loss computation
- EMA (Exponential Moving Average) model for better sampling
- Checkpointing for resumable training
- TensorBoard logging with periodic sample generation
"""

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from config import PRESETS, Config
from data import get_dataloaders
from diffusion import DDPMSampler, NoiseSchedule, get_target, q_sample
from model import UNet, count_parameters
from torch.utils.tensorboard import SummaryWriter


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Apply shadow parameters to model (for sampling)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters after sampling."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: NoiseSchedule,
    config: Config,
    device: torch.device,
    ema: EMA | None,
    writer: SummaryWriter,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    pred_type = config.diffusion.pred_type
    accum_steps = config.train.gradient_accumulation

    # Context manager for mixed precision
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    optimizer.zero_grad()

    for step, (images, _) in enumerate(train_loader):
        x0 = images.to(device, non_blocking=True)
        batch_size = x0.size(0)

        # Sample timesteps and noise
        t = schedule.sample_timesteps(batch_size)
        noise = torch.randn_like(x0)

        # Forward diffusion
        x_t = q_sample(x0, t, schedule, noise)

        # Get target based on prediction type
        target = get_target(x0, noise, t, schedule, pred_type)

        # Forward pass with optional mixed precision
        with autocast_ctx:
            pred = model(x_t, t)
            loss = F.mse_loss(pred, target)
            loss = loss / accum_steps  # Scale for accumulation

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every accum_steps
        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                if config.train.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if config.train.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                optimizer.step()

            optimizer.zero_grad()

            # Update EMA
            if ema is not None:
                ema.update()

        total_loss += loss.item() * accum_steps  # Unscale for logging

        # Logging
        global_step = epoch * num_batches + step
        if (step + 1) % config.train.log_every == 0:
            avg_loss = total_loss / (step + 1)
            print(
                f"Epoch {epoch:03d} Step {step + 1:04d}/{num_batches} | "
                f"Loss: {loss.item() * accum_steps:.4f} (avg: {avg_loss:.4f})"
            )
            writer.add_scalar("Loss/train_step", loss.item() * accum_steps, global_step)

    return total_loss / num_batches


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    sampler: DDPMSampler,
    device: torch.device,
    num_samples: int = 16,
    fixed_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate samples from the model."""
    model.eval()
    shape = (num_samples, 3, 32, 32)

    samples = sampler.sample(model, shape, device, x_T=fixed_noise)

    # Normalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    return samples


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA | None,
    epoch: int,
    config: Config,
    loss: float,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA | None,
    device: torch.device,
) -> tuple[int, float]:
    """Load training checkpoint. Returns (epoch, loss)."""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"], checkpoint["loss"]


def main():
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")

    # Config options
    parser.add_argument(
        "--preset", type=str, choices=list(PRESETS.keys()), help="Use a preset configuration"
    )
    parser.add_argument("--config", type=str, help="Load config from JSON file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # Override common settings
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--T", type=int)
    parser.add_argument("--pred-type", type=str, choices=["eps", "x0", "v"])
    parser.add_argument("--schedule", type=str, choices=["linear", "cosine"])
    parser.add_argument("--channels", type=int)
    parser.add_argument("--no-ema", action="store_true")

    # Performance options
    parser.add_argument(
        "--mixed-precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--grad-accum", type=int, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, help="DataLoader workers")

    args = parser.parse_args()

    # Build configuration
    if args.config:
        config = Config.load(Path(args.config))
    elif args.preset:
        config = PRESETS[args.preset]
    else:
        config = Config()  # Default config

    # Apply CLI overrides
    if args.run_name:
        config.run_name = args.run_name
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.lr = args.lr
    if args.T:
        config.diffusion.T = args.T
    if args.pred_type:
        config.diffusion.pred_type = args.pred_type
    if args.schedule:
        config.diffusion.schedule_type = args.schedule
    if args.channels:
        config.model.channels = args.channels
    if args.no_ema:
        config.train.ema_decay = 0
    if args.mixed_precision:
        config.train.mixed_precision = args.mixed_precision
    if args.compile:
        config.train.compile_model = True
    if args.grad_accum:
        config.train.gradient_accumulation = args.grad_accum
    if args.num_workers:
        config.train.num_workers = args.num_workers

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Performance settings
    if config.train.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32")

    torch.backends.cudnn.benchmark = True

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Create model
    model = UNet(config.model).to(device)
    num_params = count_parameters(model)
    print(f"\nModel: {num_params:,} parameters")
    print(f"Config: channels={config.model.channels}, mults={config.model.channel_mults}")
    print(
        f"Diffusion: T={config.diffusion.T}, pred={config.diffusion.pred_type}, schedule={config.diffusion.schedule_type}"
    )

    # Optional: compile model
    if config.train.compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Mixed precision setup
    autocast_dtype = None
    scaler = None
    if config.train.mixed_precision == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler()
        print("Using FP16 mixed precision with GradScaler")
    elif config.train.mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16
        # BF16 doesn't need scaler
        print("Using BF16 mixed precision")

    # Create noise schedule and sampler
    schedule = NoiseSchedule(config.diffusion, device)
    sampler = DDPMSampler(schedule, config.diffusion.pred_type)

    # Data
    train_loader, test_loader = get_dataloaders(
        config.train.batch_size,
        config.train.num_workers,
        persistent_workers=config.train.num_workers > 0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    # EMA
    ema = EMA(model, config.train.ema_decay) if config.train.ema_decay > 0 else None

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(Path(args.resume), model, optimizer, ema, device)
        start_epoch += 1

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = config.run_name or f"ddpm_{config.diffusion.pred_type}_{timestamp}"
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    config.save(log_dir / "config.json")

    writer.add_text("config", str(config))
    writer.add_scalar("model/num_params", num_params, 0)

    # Fixed noise for consistent sample visualization
    fixed_noise = torch.randn(config.train.num_samples, 3, 32, 32, device=device)

    print(f"\nStarting training: {run_name}")
    print(f"Logging to: {log_dir}")
    print("-" * 50)

    # Training loop
    for epoch in range(start_epoch, config.train.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            schedule,
            config,
            device,
            ema,
            writer,
            epoch,
            scaler,
            autocast_dtype,
        )

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)

        print(f"\nEpoch {epoch:03d} complete | Avg Loss: {train_loss:.4f}")

        # Generate samples
        if (epoch + 1) % config.train.sample_every == 0:
            # Use EMA model for sampling if available
            if ema is not None:
                ema.apply_shadow()

            samples = generate_samples(
                model, sampler, device, config.train.num_samples, fixed_noise
            )
            grid = torchvision.utils.make_grid(samples, nrow=4)
            writer.add_image("samples", grid, epoch)

            if ema is not None:
                ema.restore()

        # Save checkpoint
        if (epoch + 1) % config.train.save_every == 0:
            save_checkpoint(
                log_dir / f"checkpoint_epoch{epoch:03d}.pt",
                model,
                optimizer,
                ema,
                epoch,
                config,
                train_loss,
            )

        print("-" * 50)

    # Final checkpoint
    save_checkpoint(
        log_dir / "checkpoint_final.pt",
        model,
        optimizer,
        ema,
        config.train.epochs - 1,
        config,
        train_loss,
    )

    writer.close()
    print(f"\nTraining complete! Logs saved to {log_dir}")


if __name__ == "__main__":
    main()

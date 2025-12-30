"""
VAE Training script.

Usage:
    python train_vae.py --preset small --epochs 50
    python train_vae.py --preset fast --kl-weight 0.5
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from pathlib import Path
from datetime import datetime
import argparse
from typing import Optional
from contextlib import nullcontext

from config import VAEConfig, VAE_PRESETS
from model import VAE, vae_loss, count_parameters
from data import get_dataloaders


def train_one_epoch(
    model: VAE,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: VAEConfig,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> dict:
    """Train for one epoch, return metrics dict."""
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = len(train_loader)

    # KL warmup: linearly increase kl_weight over first N epochs
    if epoch < config.train.kl_warmup_epochs:
        kl_weight = config.train.kl_weight * (epoch + 1) / config.train.kl_warmup_epochs
    else:
        kl_weight = config.train.kl_weight

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    for step, (images, _) in enumerate(train_loader):
        x = images.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast_ctx:
            output = model(x)
            loss, metrics = vae_loss(
                output, x, kl_weight=kl_weight, recon_type=config.train.recon_loss
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += metrics["loss"]
        total_recon += metrics["recon_loss"]
        total_kl += metrics["kl_loss"]

        # Logging
        global_step = epoch * num_batches + step
        if (step + 1) % config.train.log_every == 0:
            print(
                f"Epoch {epoch:03d} Step {step + 1:04d}/{num_batches} | "
                f"Loss: {metrics['loss']:.4f} (recon: {metrics['recon_loss']:.4f}, "
                f"kl: {metrics['kl_loss']:.4f}, β={kl_weight:.3f})"
            )
            writer.add_scalar("Loss/train_step", metrics["loss"], global_step)
            writer.add_scalar("Loss/recon_step", metrics["recon_loss"], global_step)
            writer.add_scalar("Loss/kl_step", metrics["kl_loss"], global_step)

    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon / num_batches,
        "kl_loss": total_kl / num_batches,
        "kl_weight": kl_weight,
    }


@torch.no_grad()
def generate_samples(
    model: VAE, device: torch.device, num_samples: int = 16
) -> torch.Tensor:
    """Generate samples from the prior."""
    model.eval()
    samples = model.sample(num_samples, device)
    # Normalize from [-1, 1] to [0, 1] for visualization
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    return samples


@torch.no_grad()
def reconstruct_samples(
    model: VAE, images: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct images and return (original, reconstruction) pair."""
    model.eval()
    x = images.to(device)
    output = model(x)
    recon = output.recon

    # Normalize to [0, 1]
    x = (x + 1) / 2
    recon = (recon + 1) / 2

    return x.clamp(0, 1), recon.clamp(0, 1)


def save_checkpoint(
    path: Path,
    model: VAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: VAEConfig,
    metrics: dict,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train VAE on CIFAR-10")

    parser.add_argument("--preset", type=str, choices=list(VAE_PRESETS.keys()))
    parser.add_argument("--config", type=str, help="Load config from JSON")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # Overrides
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--kl-weight", type=float, help="β for β-VAE")
    parser.add_argument("--latent-channels", type=int)
    parser.add_argument("--channels", type=int)

    # Performance
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-workers", type=int)

    args = parser.parse_args()

    # Build config
    if args.config:
        config = VAEConfig.load(Path(args.config))
    elif args.preset:
        config = VAE_PRESETS[args.preset]
    else:
        config = VAEConfig()

    # Apply overrides
    if args.run_name:
        config.run_name = args.run_name
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.lr = args.lr
    if args.kl_weight:
        config.train.kl_weight = args.kl_weight
    if args.latent_channels:
        config.model.latent_channels = args.latent_channels
    if args.channels:
        config.model.channels = args.channels
    if args.mixed_precision:
        config.train.mixed_precision = args.mixed_precision
    if args.compile:
        config.train.compile_model = True
    if args.num_workers:
        config.train.num_workers = args.num_workers

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config.train.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.seed)

    # Model
    model = VAE(config.model).to(device)
    num_params = count_parameters(model)
    print(f"\nVAE parameters: {num_params:,}")
    print(
        f"Latent shape: ({config.model.latent_channels}, 4, 4) = {config.model.latent_channels * 16} dims"
    )
    print(f"KL weight (β): {config.train.kl_weight}")

    if config.train.compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    # Mixed precision
    autocast_dtype = None
    scaler = None
    if config.train.mixed_precision == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler()
    elif config.train.mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16

    # Data
    train_loader, test_loader = get_dataloaders(
        config.train.batch_size,
        config.train.num_workers,
        persistent_workers=config.train.num_workers > 0,
    )

    # Get a fixed batch for reconstruction visualization
    fixed_images, _ = next(iter(test_loader))
    fixed_images = fixed_images[: config.train.num_samples]

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = config.run_name or f"vae_kl{config.train.kl_weight}_{timestamp}"
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    config.save(log_dir / "config.json")

    print(f"\nStarting training: {run_name}")
    print(f"Logging to: {log_dir}")
    print("-" * 50)

    # Training loop
    for epoch in range(start_epoch, config.train.epochs):
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            config,
            device,
            writer,
            epoch,
            scaler,
            autocast_dtype,
        )

        writer.add_scalar("Loss/train_epoch", metrics["loss"], epoch)
        writer.add_scalar("Loss/recon_epoch", metrics["recon_loss"], epoch)
        writer.add_scalar("Loss/kl_epoch", metrics["kl_loss"], epoch)
        writer.add_scalar("Params/kl_weight", metrics["kl_weight"], epoch)

        print(
            f"\nEpoch {epoch:03d} complete | Loss: {metrics['loss']:.4f} "
            f"(recon: {metrics['recon_loss']:.4f}, kl: {metrics['kl_loss']:.4f})"
        )

        # Visualizations
        if (epoch + 1) % config.train.sample_every == 0:
            # Random samples from prior
            samples = generate_samples(model, device, config.train.num_samples)
            grid = torchvision.utils.make_grid(samples, nrow=4)
            writer.add_image("samples/prior", grid, epoch)

            # Reconstructions
            orig, recon = reconstruct_samples(model, fixed_images, device)
            # Interleave original and reconstruction for easy comparison
            comparison = torch.stack([orig, recon], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(comparison, nrow=8)
            writer.add_image("samples/reconstruction", grid, epoch)

        # Checkpoints
        if (epoch + 1) % config.train.save_every == 0:
            save_checkpoint(
                log_dir / f"checkpoint_epoch{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                config,
                metrics,
            )

        print("-" * 50)

    # Final checkpoint
    save_checkpoint(
        log_dir / "checkpoint_final.pt",
        model,
        optimizer,
        config.train.epochs - 1,
        config,
        metrics,
    )

    writer.close()
    print(f"\nTraining complete! Logs saved to {log_dir}")


if __name__ == "__main__":
    main()

"""
VAE Training script with optional LPIPS and adversarial loss.

Usage:
    # Basic VAE training
    python train_vae.py --preset small --epochs 50

    # With perceptual loss only
    python train_vae.py --preset fast --use-lpips --lpips-weight 0.1

    # With adversarial + perceptual loss (sharpest results)
    python train_vae.py --preset sharp

    # Or manually:
    python train_vae.py --preset fast --use-lpips --use-adversarial --kl-weight 0.001
"""

import argparse
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).parent.parent))
from vae.config import VAE_PRESETS, VAEConfig
from vae.data import get_dataloaders
from vae.model import (
    LPIPS,
    VAE,
    PatchDiscriminator,
    count_parameters,
    hinge_loss_dis,
    hinge_loss_gen,
    vae_loss,
)


def train_one_epoch(
    model: VAE,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: VAEConfig,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    autocast_dtype: torch.dtype | None = None,
    lpips_model: LPIPS | None = None,
    discriminator: PatchDiscriminator | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Train for one epoch, return metrics dict."""
    model.train()
    if discriminator is not None:
        discriminator.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_lpips = 0.0
    total_gen_adv = 0.0
    total_disc = 0.0
    num_batches = len(train_loader)

    # KL warmup: linearly increase kl_weight over first N epochs
    if epoch < config.train.kl_warmup_epochs:
        kl_weight = config.train.kl_weight * (epoch + 1) / config.train.kl_warmup_epochs
    else:
        kl_weight = config.train.kl_weight

    # Check if adversarial training is active this epoch
    use_adv = (
        config.train.use_adversarial
        and discriminator is not None
        and epoch >= config.train.adv_start_epoch
    )

    # LPIPS weight
    lpips_weight = config.train.lpips_weight if config.train.use_lpips else 0.0

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    for step, (images, _) in enumerate(train_loader):
        x = images.to(device, non_blocking=True)

        # =================================================================
        # Train Discriminator (if enabled)
        # =================================================================
        disc_loss_value = 0.0
        if use_adv:
            for _ in range(config.train.disc_steps):
                disc_optimizer.zero_grad()

                with autocast_ctx:
                    # Get reconstructions (detached from VAE graph)
                    with torch.no_grad():
                        output = model(x)
                        recon = output.recon.detach()

                    # Discriminator predictions
                    real_pred = discriminator(x)
                    fake_pred = discriminator(recon)

                    # Hinge loss
                    disc_loss = hinge_loss_dis(real_pred, fake_pred)

                if scaler is not None:
                    scaler.scale(disc_loss).backward()
                    scaler.step(disc_optimizer)
                    scaler.update()
                else:
                    disc_loss.backward()
                    disc_optimizer.step()

                disc_loss_value = disc_loss.item()

        # =================================================================
        # Train VAE (Generator)
        # =================================================================
        optimizer.zero_grad()

        with autocast_ctx:
            output = model(x)

            # Base VAE loss (recon + KL + optional LPIPS)
            loss, metrics = vae_loss(
                output,
                x,
                kl_weight=kl_weight,
                recon_type=config.train.recon_loss,
                lpips_model=lpips_model,
                lpips_weight=lpips_weight,
            )

            # Add adversarial loss for generator
            gen_adv_loss_value = 0.0
            if use_adv:
                fake_pred = discriminator(output.recon)
                gen_adv_loss = hinge_loss_gen(fake_pred)
                loss = loss + config.train.adv_weight * gen_adv_loss
                gen_adv_loss_value = gen_adv_loss.item()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Accumulate metrics
        total_loss += metrics["loss"]
        total_recon += metrics["recon_loss"]
        total_kl += metrics["kl_loss"]
        if "lpips_loss" in metrics:
            total_lpips += metrics["lpips_loss"]
        total_gen_adv += gen_adv_loss_value
        total_disc += disc_loss_value

        # Logging
        global_step = epoch * num_batches + step
        if (step + 1) % config.train.log_every == 0:
            log_str = (
                f"Epoch {epoch:03d} Step {step + 1:04d}/{num_batches} | "
                f"Loss: {metrics['loss']:.4f} (recon: {metrics['recon_loss']:.4f}, "
                f"kl: {metrics['kl_loss']:.4f}"
            )
            if "lpips_loss" in metrics:
                log_str += f", lpips: {metrics['lpips_loss']:.4f}"
            if use_adv:
                log_str += f", g_adv: {gen_adv_loss_value:.4f}, d: {disc_loss_value:.4f}"
            log_str += f", β={kl_weight:.4f})"
            print(log_str)

            writer.add_scalar("Loss/train_step", metrics["loss"], global_step)
            writer.add_scalar("Loss/recon_step", metrics["recon_loss"], global_step)
            writer.add_scalar("Loss/kl_step", metrics["kl_loss"], global_step)
            if "lpips_loss" in metrics:
                writer.add_scalar("Loss/lpips_step", metrics["lpips_loss"], global_step)
            if use_adv:
                writer.add_scalar("Loss/gen_adv_step", gen_adv_loss_value, global_step)
                writer.add_scalar("Loss/disc_step", disc_loss_value, global_step)

    result = {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon / num_batches,
        "kl_loss": total_kl / num_batches,
        "kl_weight": kl_weight,
    }

    if config.train.use_lpips:
        result["lpips_loss"] = total_lpips / num_batches
    if use_adv:
        result["gen_adv_loss"] = total_gen_adv / num_batches
        result["disc_loss"] = total_disc / num_batches

    return result


@torch.no_grad()
def generate_samples(model: VAE, device: torch.device, num_samples: int = 16) -> torch.Tensor:
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
    discriminator: PatchDiscriminator | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
    }

    if discriminator is not None:
        checkpoint["discriminator_state_dict"] = discriminator.state_dict()
    if disc_optimizer is not None:
        checkpoint["disc_optimizer_state_dict"] = disc_optimizer.state_dict()

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

    # LPIPS
    parser.add_argument("--use-lpips", action="store_true", help="Enable LPIPS perceptual loss")
    parser.add_argument("--lpips-weight", type=float, help="Weight for LPIPS loss")

    # Adversarial
    parser.add_argument(
        "--use-adversarial", action="store_true", help="Enable adversarial training"
    )
    parser.add_argument("--adv-weight", type=float, help="Weight for adversarial loss")
    parser.add_argument("--adv-start-epoch", type=int, help="Epoch to start adversarial training")

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

    # Apply overrides (use `is not None` to allow 0 values)
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.kl_weight is not None:
        config.train.kl_weight = args.kl_weight
    if args.latent_channels is not None:
        config.model.latent_channels = args.latent_channels
    if args.channels is not None:
        config.model.channels = args.channels
    if args.use_lpips:
        config.train.use_lpips = True
    if args.lpips_weight is not None:
        config.train.lpips_weight = args.lpips_weight
    if args.use_adversarial:
        config.train.use_adversarial = True
    if args.adv_weight is not None:
        config.train.adv_weight = args.adv_weight
    if args.adv_start_epoch is not None:
        config.train.adv_start_epoch = args.adv_start_epoch
    if args.mixed_precision is not None:
        config.train.mixed_precision = args.mixed_precision
    if args.compile:
        config.train.compile_model = True
    if args.num_workers is not None:
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
        f"Latent shape: ({config.model.latent_channels}, 8, 8) = {config.model.latent_channels * 64} dims"
    )
    print(f"KL weight (β): {config.train.kl_weight}")

    if config.train.compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    # LPIPS model (if enabled)
    lpips_model = None
    if config.train.use_lpips:
        print(f"LPIPS enabled (weight={config.train.lpips_weight})")
        lpips_model = LPIPS().to(device)
        lpips_model.eval()

    # Discriminator (if enabled)
    discriminator = None
    disc_optimizer = None
    if config.train.use_adversarial:
        print(
            f"Adversarial training enabled (weight={config.train.adv_weight}, start_epoch={config.train.adv_start_epoch})"
        )
        discriminator = PatchDiscriminator(config.discriminator).to(device)
        disc_params = count_parameters(discriminator)
        print(f"Discriminator parameters: {disc_params:,}")

        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=config.train.disc_lr,
            betas=(0.5, 0.9),  # Standard for GAN training
        )

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
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        if discriminator is not None and "discriminator_state_dict" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        if disc_optimizer is not None and "disc_optimizer_state_dict" in checkpoint:
            disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])

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
            lpips_model=lpips_model,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
        )

        # Log epoch metrics
        writer.add_scalar("Loss/train_epoch", metrics["loss"], epoch)
        writer.add_scalar("Loss/recon_epoch", metrics["recon_loss"], epoch)
        writer.add_scalar("Loss/kl_epoch", metrics["kl_loss"], epoch)
        writer.add_scalar("Params/kl_weight", metrics["kl_weight"], epoch)

        if "lpips_loss" in metrics:
            writer.add_scalar("Loss/lpips_epoch", metrics["lpips_loss"], epoch)
        if "gen_adv_loss" in metrics:
            writer.add_scalar("Loss/gen_adv_epoch", metrics["gen_adv_loss"], epoch)
            writer.add_scalar("Loss/disc_epoch", metrics["disc_loss"], epoch)

        # Print summary
        log_str = (
            f"\nEpoch {epoch:03d} complete | Loss: {metrics['loss']:.4f} "
            f"(recon: {metrics['recon_loss']:.4f}, kl: {metrics['kl_loss']:.4f}"
        )
        if "lpips_loss" in metrics:
            log_str += f", lpips: {metrics['lpips_loss']:.4f}"
        if "gen_adv_loss" in metrics:
            log_str += f", g_adv: {metrics['gen_adv_loss']:.4f}, d: {metrics['disc_loss']:.4f}"
        log_str += ")"
        print(log_str)

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
                discriminator=discriminator,
                disc_optimizer=disc_optimizer,
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
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
    )

    writer.close()
    print(f"\nTraining complete! Logs saved to {log_dir}")


if __name__ == "__main__":
    main()

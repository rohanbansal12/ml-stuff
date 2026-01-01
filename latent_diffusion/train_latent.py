"""
Latent Diffusion Training Script.

This trains a diffusion model in the VAE's latent space instead of pixel space.
The key insight is that diffusion in a compressed latent space is much faster
than in pixel space, while the VAE decoder recovers high-quality images.

Key steps:
    1. Load pretrained VAE (frozen)
    2. Encode images to latents during training
    3. Train U-Net to denoise in latent space
    4. At sampling time: generate latent with diffusion, decode with VAE

This is the core idea behind Stable Diffusion and similar models.

Usage:
    python train_latent.py --preset base --vae-checkpoint /path/to/vae.pt
    python train_latent.py --preset fast --vae-checkpoint /path/to/vae.pt --epochs 100

    # With classifier-free guidance:
    python train_latent.py --preset fast --vae-checkpoint /path/to/vae.pt --num-classes 10 --guidance-scale 5.0
"""

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
from datetime import datetime
import argparse
import sys
from typing import Optional
from contextlib import nullcontext
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from latent_diffusion.config import LatentDiffusionFullConfig, LATENT_PRESETS
from latent_diffusion.model import LatentUNet, count_parameters
from latent_diffusion.diffusion import NoiseSchedule, DDPMSampler, q_sample, get_target

from vae.model import VAE
from vae.config import VAEConfig

import warnings

warnings.filterwarnings("ignore", message="dtype.*align")


class LatentDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that encodes images to VAE latents on-the-fly.

    This is simple but inefficient for repeated epochs since we re-encode
    every image each time. For production, use precompute_latents() instead.

    Attributes:
        base_dataset: Underlying image dataset (e.g., CIFAR-10).
        vae: Pretrained VAE encoder.
        device: Device for VAE forward pass.
        scale: Latent scaling factor for unit variance.
    """

    def __init__(
        self, base_dataset, vae: VAE, device: torch.device, scale: float = 1.0
    ):
        """
        Initialize latent dataset wrapper.

        Args:
            base_dataset: Image dataset returning (image, label) tuples.
            vae: Pretrained VAE model (will be used in eval mode).
            device: Device to run VAE encoding on.
            scale: Multiplicative scale factor for latents.
        """
        self.base_dataset = base_dataset
        self.vae = vae
        self.device = device
        self.scale = scale

    @torch.no_grad()
    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images to latents using VAE encoder.

        Uses the mean of the approximate posterior q(z|x), not a sample,
        for more stable and deterministic training.

        Args:
            images: Batch of images, shape (B, 3, H, W), range [-1, 1].

        Returns:
            Latent codes, shape (B, C, H', W'), on CPU.
        """
        images = images.to(self.device)
        mu, _ = self.vae.encode(images)
        latent = mu * self.scale
        return latent.cpu()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        # Note: Encoding one image at a time is inefficient.
        # For better performance, use precompute_latents().
        image = image.unsqueeze(0).to(self.device)
        latent = self.encode_batch(image).squeeze(0).cpu()
        return latent, label


def precompute_latents(
    dataset, vae: VAE, device: torch.device, scale: float, batch_size: int = 256
) -> torch.Tensor:
    """
    Pre-compute all latents for the entire dataset.

    This trades memory for speed: all latents are stored in RAM, but
    training doesn't require any VAE forward passes. Recommended for
    datasets that fit in memory (e.g., CIFAR-10 latents are ~50MB).

    Args:
        dataset: Image dataset returning (image, label) tuples.
        vae: Pretrained VAE model (used in eval mode, no grad).
        device: Device to run VAE encoding on.
        scale: Multiplicative scale factor for latents.
        batch_size: Batch size for encoding (larger = faster).

    Returns:
        Tensor of all latents, shape (N, C, H', W'), on CPU.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_latents = []
    vae.eval()

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Encoding latents"):
            images = images.to(device)
            mu, _ = vae.encode(images)
            latents = mu * scale
            all_latents.append(latents.cpu())

    return torch.cat(all_latents, dim=0)


class EMA:
    """
    Exponential Moving Average for model parameters.

    Maintains a shadow copy of model weights that is an exponentially
    weighted average of training weights. This often produces better
    samples than using the raw training weights.

    Math:
        shadow_{t+1} = decay · shadow_t + (1 - decay) · params_t

    Typical decay values: 0.999 to 0.9999.

    Attributes:
        model: The model whose parameters to track.
        decay: EMA decay rate (higher = slower updates).
        shadow: Dict mapping param names to EMA values.
        backup: Temporary storage when applying shadow weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        """
        Initialize EMA tracker.

        Args:
            model: Model to track.
            decay: EMA decay rate in [0, 1). Higher = more smoothing.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """
        Replace model weights with shadow weights.

        Call this before sampling/evaluation. Original weights are
        backed up and can be restored with restore().
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model weights after apply_shadow()."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """Return shadow weights for checkpointing."""
        return self.shadow

    def load_state_dict(self, state_dict):
        """Load shadow weights from checkpoint."""
        self.shadow = state_dict


def compute_latent_scale(
    vae: VAE, dataloader, device: torch.device, num_batches: int = 50
) -> float:
    """
    Compute scaling factor to normalize latent variance to ~1.

    Diffusion models assume the data distribution has unit variance.
    VAE latents may have different variance, so we compute a scale
    factor to normalize them. This is similar to what Stable Diffusion does.

    Math:
        scale = 1 / std(latents)

    After scaling: latents * scale has std ≈ 1.

    Args:
        vae: Pretrained VAE model.
        dataloader: DataLoader for computing statistics.
        device: Device to run VAE encoding on.
        num_batches: Number of batches to use for estimation.

    Returns:
        Scale factor (float) to multiply latents by.
    """
    vae.eval()
    all_latents = []

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device)
            mu, _ = vae.encode(images)
            all_latents.append(mu)

    latents = torch.cat(all_latents, dim=0)
    std = latents.std().item()
    scale = 1.0 / std

    return scale


def train_one_epoch(
    model: LatentUNet,
    vae: VAE,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: NoiseSchedule,
    config: LatentDiffusionFullConfig,
    device: torch.device,
    ema: Optional[EMA],
    writer: SummaryWriter,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> float:
    """
    Train the latent diffusion model for one epoch.

    Training loop:
        1. Encode images to latents z_0 using frozen VAE
        2. Sample timesteps t ~ Uniform(0, T)
        3. Sample noise ε ~ N(0, I)
        4. Compute noisy latent z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε
        5. Predict target (ε, z_0, or v depending on pred_type)
        6. Compute MSE loss and backprop
        7. Update EMA weights

    Args:
        model: Latent U-Net denoising model.
        vae: Frozen VAE for encoding images to latents.
        train_loader: DataLoader yielding (images, labels) batches.
        optimizer: Optimizer for model parameters.
        schedule: Noise schedule with precomputed coefficients.
        config: Full configuration object.
        device: Device to train on.
        ema: Optional EMA tracker for model weights.
        writer: TensorBoard writer for logging.
        epoch: Current epoch number (for logging).
        scaler: Optional GradScaler for mixed precision training.
        autocast_dtype: Optional dtype for automatic mixed precision.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    vae.eval()

    total_loss = 0.0
    num_batches = len(train_loader)
    pred_type = config.diffusion.pred_type
    latent_scale = config.latent_scale
    use_cfg = config.model.num_classes > 0

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) if use_cfg else None

        # Encode images to latents (no grad - VAE is frozen)
        with torch.no_grad():
            mu, _ = vae.encode(images)
            z_0 = mu * latent_scale

        # Sample timesteps and noise
        batch_size = z_0.size(0)
        t = schedule.sample_timesteps(batch_size)
        noise = torch.randn_like(z_0)

        # Forward diffusion: z_0 -> z_t
        z_t = q_sample(z_0, t=t, schedule=schedule, noise=noise)

        # Get target based on prediction type
        target = get_target(
            z_0, noise=noise, t=t, schedule=schedule, pred_type=pred_type
        )

        # Forward pass and loss
        optimizer.zero_grad()

        with autocast_ctx:
            pred = model(z_t, t, class_labels=labels)
            loss = F.mse_loss(pred, target)

        # Backward pass with optional gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            if config.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.grad_clip
                )
            optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update()

        total_loss += loss.item()

        # Logging
        global_step = epoch * num_batches + step
        if (step + 1) % config.train.log_every == 0:
            print(
                f"Epoch {epoch:03d} Step {step + 1:04d}/{num_batches} | "
                f"Loss: {loss.item():.4f}"
            )
            writer.add_scalar("Loss/train_step", loss.item(), global_step)

    return total_loss / num_batches


@torch.no_grad()
def generate_samples(
    model: LatentUNet,
    vae: VAE,
    sampler: DDPMSampler,
    config: LatentDiffusionFullConfig,
    device: torch.device,
    num_samples: int = 16,
    z_T: Optional[torch.Tensor] = None,
    class_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate image samples via latent diffusion.

    Sampling process:
        1. Sample z_T ~ N(0, I) in latent space
        2. Run reverse diffusion: z_T -> z_{T-1} -> ... -> z_0
        3. Unscale latent: z_0 = z_0 / latent_scale
        4. Decode to image: x = VAE.decode(z_0)
        5. Normalize to [0, 1] for visualization

    Args:
        model: Trained latent U-Net denoising model.
        vae: Frozen VAE for decoding latents to images.
        sampler: DDPM or DDIM sampler for reverse diffusion.
        config: Full configuration object.
        device: Device to run sampling on.
        num_samples: Number of images to generate.
        z_T: Optional starting noise. If None, sampled from N(0, I).
        class_labels: Optional class labels for conditional generation.
            If None and model is conditional, random classes are used.

    Returns:
        Generated images, shape (num_samples, 3, H, W), range [0, 1].
    """
    model.eval()
    vae.eval()

    latent_shape = (num_samples, *config.latent_shape)

    # Generate random class labels if conditional but none provided
    if config.model.num_classes > 0 and class_labels is None:
        class_labels = torch.randint(
            0, config.model.num_classes, (num_samples,), device=device
        )

    # Run reverse diffusion in latent space
    z_0 = sampler.sample(
        model=model,
        shape=latent_shape,
        device=device,
        z_T=z_T,
        class_labels=class_labels,
        progress=False,
    )

    # Unscale latent (inverse of training scaling)
    z_0 = z_0 / config.latent_scale

    # Decode to image
    images = vae.decode(z_0)

    # Normalize from [-1, 1] to [0, 1] for visualization
    images = (images + 1) / 2
    images = images.clamp(0, 1)

    return images


def save_checkpoint(
    path: Path,
    model: LatentUNet,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    epoch: int,
    config: LatentDiffusionFullConfig,
    loss: float,
):
    """
    Save training checkpoint.

    Saves model weights, optimizer state, EMA weights (if used),
    and configuration for resuming training or inference.

    Args:
        path: File path to save checkpoint to.
        model: Latent U-Net model.
        optimizer: Optimizer with current state.
        ema: Optional EMA tracker (shadow weights saved if present).
        epoch: Current epoch number.
        config: Full configuration object.
        loss: Current training loss (for logging).
    """
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


def load_vae(checkpoint_path: str, device: torch.device) -> tuple[VAE, VAEConfig]:
    """
    Load pretrained VAE from checkpoint.

    The VAE is set to eval mode and frozen (requires_grad=False)
    since we only use it for encoding/decoding, not training.

    Args:
        checkpoint_path: Path to VAE checkpoint file.
        device: Device to load model onto.

    Returns:
        Tuple of (VAE model, VAE config).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae_config = checkpoint["config"]

    vae = VAE(vae_config.model).to(device)

    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    vae.load_state_dict(state_dict=state_dict)
    vae.eval()
    vae.requires_grad_(False)  # Freeze VAE

    print(f"Loaded VAE from {checkpoint_path}")
    return vae, vae_config


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Latent Diffusion on CIFAR-10")

    parser.add_argument("--preset", type=str, choices=list(LATENT_PRESETS.keys()))
    parser.add_argument("--config", type=str, help="Load config from JSON")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # VAE settings
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument(
        "--latent-scale",
        type=float,
        help="Scale factor for latents (computed automatically if not set)",
    )

    # Overrides
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--T", type=int)
    parser.add_argument("--pred-type", type=str, choices=["eps", "x0", "v"])
    parser.add_argument("--schedule", type=str, choices=["linear", "cosine"])

    # Classifier-free guidance
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of classes (0=unconditional, 10 for CIFAR-10)",
    )
    parser.add_argument(
        "--cfg-dropout",
        type=float,
        help="Class dropout rate for CFG training (default 0.1)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="CFG scale at sampling (0=disabled, try 3-7)",
    )

    # Performance
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-workers", type=int)

    args = parser.parse_args()

    # Build config
    if args.config:
        config = LatentDiffusionFullConfig.load(Path(args.config))
    elif args.preset:
        config = LATENT_PRESETS[args.preset]
    else:
        config = LatentDiffusionFullConfig()

    # Apply overrides (use `is not None` to allow 0 values)
    config.vae_checkpoint = args.vae_checkpoint
    if args.latent_scale is not None:
        config.latent_scale = args.latent_scale
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.T is not None:
        config.diffusion.T = args.T
    if args.pred_type is not None:
        config.diffusion.pred_type = args.pred_type
    if args.schedule is not None:
        config.diffusion.schedule_type = args.schedule
    if args.num_classes is not None:
        config.model.num_classes = args.num_classes
    if args.cfg_dropout is not None:
        config.model.cfg_dropout = args.cfg_dropout
    if args.guidance_scale is not None:
        config.train.guidance_scale = args.guidance_scale
    if args.mixed_precision is not None:
        config.train.mixed_precision = args.mixed_precision
    if args.compile:
        config.train.compile_model = True
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers

    # Setup device and performance options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config.train.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.seed)

    # Load frozen VAE
    vae, vae_config = load_vae(config.vae_checkpoint, device)

    # Update latent shape from VAE config
    # For CIFAR-10 with channel_mults=(1,2,4): 32 -> 16 -> 8
    latent_h = 32 // (2 ** (len(vae_config.model.channel_mults) - 1))
    config.latent_shape = (vae_config.model.latent_channels, latent_h, latent_h)
    config.model.in_channels = vae_config.model.latent_channels
    config.model.out_channels = vae_config.model.latent_channels

    print(f"Latent shape: {config.latent_shape}")

    # Setup data
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.train.num_workers > 0,
    )

    # Compute latent scale if not provided
    if config.latent_scale == 1.0 and args.latent_scale is None:
        print("Computing latent scale factor...")
        config.latent_scale = compute_latent_scale(vae, train_loader, device)
        print(f"Latent scale: {config.latent_scale:.4f}")

    # Create denoising model
    model = LatentUNet(config.model).to(device)
    num_params = count_parameters(model)
    print(f"\nLatent U-Net parameters: {num_params:,}")

    if config.model.num_classes > 0:
        print(
            f"Class-conditional model: {config.model.num_classes} classes, CFG dropout: {config.model.cfg_dropout}"
        )
        print(f"Guidance scale at sampling: {config.train.guidance_scale}")

    if config.train.compile_model:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Setup mixed precision
    autocast_dtype = None
    scaler = None
    if config.train.mixed_precision == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler()
    elif config.train.mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16

    # Create noise schedule and sampler
    schedule = NoiseSchedule(config.diffusion, device)
    sampler = DDPMSampler(
        schedule,
        config.diffusion.pred_type,
        guidance_scale=config.train.guidance_scale,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    # Setup EMA
    ema = EMA(model, config.train.ema_decay) if config.train.ema_decay > 0 else None

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = config.run_name or f"latent_diffusion_{timestamp}"
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    config.save(log_dir / "config.json")

    # Fixed noise for consistent visualization across epochs
    fixed_z_T = torch.randn(
        config.train.num_samples, *config.latent_shape, device=device
    )

    # Fixed class labels for visualization (one of each class if conditional)
    if config.model.num_classes > 0:
        fixed_class_labels = (
            torch.arange(config.train.num_samples, device=device)
            % config.model.num_classes
        )
    else:
        fixed_class_labels = None

    print(f"\nStarting training: {run_name}")
    print(f"Logging to: {log_dir}")
    print("-" * 50)

    # Main training loop
    for epoch in range(start_epoch, config.train.epochs):
        train_loss = train_one_epoch(
            model=model,
            vae=vae,
            train_loader=train_loader,
            optimizer=optimizer,
            schedule=schedule,
            config=config,
            device=device,
            ema=ema,
            writer=writer,
            epoch=epoch,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
        )

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        print(f"\nEpoch {epoch:03d} complete | Loss: {train_loss:.4f}")

        # Generate and log samples
        if (epoch + 1) % config.train.sample_every == 0:
            # Use EMA weights for sampling if available
            if ema is not None:
                ema.apply_shadow()

            samples = generate_samples(
                model=model,
                vae=vae,
                sampler=sampler,
                config=config,
                device=device,
                num_samples=config.train.num_samples,
                z_T=fixed_z_T,
                class_labels=fixed_class_labels,
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

    # Save final checkpoint
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

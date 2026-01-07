"""
Evaluation metrics for latent diffusion models.

Implements:
- FID (Fréchet Inception Distance): Primary quality metric
- IS (Inception Score): Measures quality and diversity
- LPIPS: Perceptual similarity (for reconstruction tasks)

Usage:
    python evaluate.py --checkpoint /path/to/checkpoint.pt --vae-checkpoint /path/to/vae.pt --num-samples 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from latent_diffusion.config import LatentDiffusionFullConfig
from latent_diffusion.diffusion import DDIMSampler, DDPMSampler, NoiseSchedule
from latent_diffusion.model import LatentUNet
from vae.model import VAE


class InceptionFeatureExtractor(nn.Module):
    """
    Extract features from Inception v3 for FID/IS computation.

    For FID: Uses features from the last pooling layer (2048-dim)
    For IS: Uses the final logits (1000-dim softmax)
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        # Load pretrained Inception v3
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.eval()
        self.inception.to(device)

        # Remove final FC layer to get features
        self.inception.fc = nn.Identity()

        # For getting logits, we need the original FC
        self.fc = nn.Linear(2048, 1000).to(device)
        self.fc.load_state_dict(
            {
                "weight": inception_v3(weights=Inception_V3_Weights.DEFAULT).fc.weight,
                "bias": inception_v3(weights=Inception_V3_Weights.DEFAULT).fc.bias,
            }
        )
        self.fc.eval()

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and logits from images.

        Args:
            x: Images in range [0, 1], shape (B, 3, H, W)

        Returns:
            features: (B, 2048) for FID
            logits: (B, 1000) for IS
        """
        # Inception expects 299x299 images, normalized
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5  # Normalize to [-1, 1]

        features = self.inception(x)
        logits = self.fc(features)

        return features, logits


def compute_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Inception Distance between two sets of features.

    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2(Σ_r Σ_f)^{1/2})

    Lower is better. FID = 0 means identical distributions.

    Args:
        real_features: (N, 2048) features from real images
        fake_features: (M, 2048) features from generated images
        eps: Small constant for numerical stability

    Returns:
        FID score (float)
    """
    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Compute FID
    diff = mu_real - mu_fake

    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    # Handle numerical instability
    if not np.isfinite(covmean).all():
        print(f"Warning: sqrtm produced non-finite values, adding {eps} to diagonal")
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_fake + offset))

    # Handle imaginary components (numerical artifacts)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return float(fid)


def compute_inception_score(
    logits: np.ndarray,
    num_splits: int = 10,
) -> tuple[float, float]:
    """
    Compute Inception Score.

    IS = exp(E_x[KL(p(y|x) || p(y))])

    Higher is better. Measures both quality (confident predictions)
    and diversity (uniform marginal distribution).

    Args:
        logits: (N, 1000) logits from Inception
        num_splits: Number of splits for computing std

    Returns:
        (mean_IS, std_IS)
    """
    # Convert to probabilities
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    scores = []
    split_size = len(probs) // num_splits

    for i in range(num_splits):
        part = probs[i * split_size : (i + 1) * split_size]

        # p(y) - marginal distribution
        p_y = np.mean(part, axis=0, keepdims=True)

        # KL divergence
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        kl = np.sum(kl, axis=1)

        scores.append(np.exp(np.mean(kl)))

    return float(np.mean(scores)), float(np.std(scores))


@torch.no_grad()
def extract_features(
    dataloader: DataLoader,
    extractor: InceptionFeatureExtractor,
    device: torch.device,
    max_samples: int | None = None,
    desc: str = "Extracting features",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Inception features from a dataloader of images.

    Args:
        dataloader: DataLoader yielding (images, ...) batches
        extractor: InceptionFeatureExtractor
        device: Device to run on
        max_samples: Maximum number of samples to process
        desc: Progress bar description

    Returns:
        (features, logits) as numpy arrays
    """
    all_features = []
    all_logits = []
    num_samples = 0

    for batch in tqdm(dataloader, desc=desc):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device)

        # Ensure images are in [0, 1]
        if images.min() < 0:
            images = (images + 1) / 2

        features, logits = extractor(images)
        all_features.append(features.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

        num_samples += images.size(0)
        if max_samples and num_samples >= max_samples:
            break

    features = np.concatenate(all_features, axis=0)
    logits = np.concatenate(all_logits, axis=0)

    if max_samples:
        features = features[:max_samples]
        logits = logits[:max_samples]

    return features, logits


@torch.no_grad()
def generate_samples_batched(
    model: LatentUNet,
    vae: VAE,
    sampler: DDPMSampler,
    config: LatentDiffusionFullConfig,
    device: torch.device,
    num_samples: int,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Generate samples in batches.

    Args:
        model: Trained latent U-Net
        vae: VAE decoder
        sampler: DDPM or DDIM sampler
        config: Model config
        device: Device
        num_samples: Total samples to generate
        batch_size: Batch size for generation

    Returns:
        Generated images, shape (num_samples, 3, H, W), range [0, 1]
    """
    model.eval()
    vae.eval()

    all_samples = []
    num_generated = 0

    pbar = tqdm(total=num_samples, desc="Generating samples")

    while num_generated < num_samples:
        current_batch = min(batch_size, num_samples - num_generated)

        latent_shape = (current_batch, *config.latent_shape)

        # Generate class labels if conditional
        if config.model.num_classes > 0:
            class_labels = torch.randint(
                0, config.model.num_classes, (current_batch,), device=device
            )
        else:
            class_labels = None

        # Sample latents
        z_0 = sampler.sample(
            model=model,
            shape=latent_shape,
            device=device,
            class_labels=class_labels,
            progress=False,
        )

        # Decode
        z_0 = z_0 / config.latent_scale
        images = vae.decode(z_0)

        # Normalize to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)

        all_samples.append(images.cpu())
        num_generated += current_batch
        pbar.update(current_batch)

    pbar.close()
    return torch.cat(all_samples, dim=0)


def load_model(
    checkpoint_path: str,
    vae_checkpoint_path: str,
    device: torch.device,
) -> tuple[LatentUNet, VAE, LatentDiffusionFullConfig]:
    """Load trained latent diffusion model and VAE."""

    # Load VAE
    vae_ckpt = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
    vae_config = vae_ckpt["config"]
    vae = VAE(vae_config.model).to(device)

    state_dict = vae_ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    vae.load_state_dict(state_dict)
    vae.eval()

    # Load diffusion model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = LatentUNet(config.model).to(device)

    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Use EMA weights if available
    if "ema_state_dict" in ckpt:
        print("Using EMA weights")
        for name, param in model.named_parameters():
            if name in ckpt["ema_state_dict"]:
                param.data = ckpt["ema_state_dict"][name].to(device)

    return model, vae, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate latent diffusion model")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vae-checkpoint", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples for FID")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--use-ddim", action="store_true", help="Use DDIM sampler")
    parser.add_argument("--ddim-steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--guidance-scale", type=float, help="Override guidance scale")
    parser.add_argument("--save-samples", type=str, help="Path to save sample grid")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("\nLoading models...")
    model, vae, config = load_model(args.checkpoint, args.vae_checkpoint, device)

    if args.guidance_scale is not None:
        config.train.guidance_scale = args.guidance_scale

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Conditional: {config.model.num_classes > 0}")
    print(f"Guidance scale: {config.train.guidance_scale}")

    # Create sampler
    schedule = NoiseSchedule(config.diffusion, device)

    if args.use_ddim:
        sampler = DDIMSampler(
            schedule,
            config.diffusion.pred_type,
            num_inference_steps=args.ddim_steps,
            guidance_scale=config.train.guidance_scale,
        )
        print(f"Using DDIM with {args.ddim_steps} steps")
    else:
        sampler = DDPMSampler(
            schedule,
            config.diffusion.pred_type,
            guidance_scale=config.train.guidance_scale,
        )
        print(f"Using DDPM with {config.diffusion.T} steps")

    # Load real data
    print("\nLoading CIFAR-10...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    real_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    real_loader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Initialize feature extractor
    print("\nInitializing Inception v3...")
    extractor = InceptionFeatureExtractor(device)

    # Extract real features
    print(f"\nExtracting features from {args.num_samples} real images...")
    real_features, real_logits = extract_features(
        real_loader, extractor, device, max_samples=args.num_samples, desc="Real images"
    )

    # Generate fake samples
    print(f"\nGenerating {args.num_samples} samples...")
    fake_images = generate_samples_batched(
        model,
        vae,
        sampler,
        config,
        device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    # Save sample grid if requested
    if args.save_samples:
        import torchvision

        grid = torchvision.utils.make_grid(fake_images[:64], nrow=8)
        torchvision.utils.save_image(grid, args.save_samples)
        print(f"Saved sample grid to {args.save_samples}")

    # Extract fake features
    print("\nExtracting features from generated images...")
    fake_loader = DataLoader(fake_images, batch_size=args.batch_size)
    fake_features, fake_logits = extract_features(
        fake_loader, extractor, device, desc="Generated images"
    )

    # Compute metrics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    # FID
    fid = compute_fid(real_features, fake_features)
    print(f"\nFID: {fid:.2f}")
    print("  (Lower is better. Real data FID ≈ 0)")
    print("  Reference: CIFAR-10 FID scores typically range 10-50 for good models")

    # Inception Score
    is_mean, is_std = compute_inception_score(fake_logits)
    print(f"\nInception Score: {is_mean:.2f} ± {is_std:.2f}")
    print("  (Higher is better. Max theoretical ≈ 10 for CIFAR-10)")
    print("  Reference: Good CIFAR-10 models achieve IS 8-9")

    # Real data IS for reference
    real_is_mean, real_is_std = compute_inception_score(real_logits)
    print(f"\nReal Data IS: {real_is_mean:.2f} ± {real_is_std:.2f}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()

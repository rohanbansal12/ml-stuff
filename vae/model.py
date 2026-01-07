"""
Variational Autoencoder (VAE) for image generation.

Architecture: Encoder-Decoder with spatial latent space.
- Encoder: Image (B, 3, 32, 32) -> Latent (B, C, H, W) with mean and logvar
- Decoder: Latent (B, C, H, W) -> Reconstructed Image (B, 3, 32, 32)

Includes:
- PatchGAN discriminator for adversarial training
- LPIPS wrapper for perceptual loss

Key concepts:
1. Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)
2. KL divergence: regularizes latent to be close to N(0, I)
3. ELBO: Evidence Lower Bound = Reconstruction - KL
4. Perceptual loss: Compare VGG features instead of pixels
5. Adversarial loss: Discriminator distinguishes real vs reconstructed

References:
- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- "β-VAE" (Higgins et al., 2017) for disentanglement
- "Taming Transformers" (Esser et al., 2021) for VAE-GAN
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
from vae.config import DiscriminatorConfig, VAEModelConfig


@dataclass
class VAEOutput:
    """Container for VAE forward pass outputs."""

    recon: torch.Tensor  # Reconstructed image
    mu: torch.Tensor  # Latent mean
    logvar: torch.Tensor  # Latent log-variance
    z: torch.Tensor  # Sampled latent


class ResBlock(nn.Module):
    """Residual block without time conditioning (simpler than DDPM version)."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    """Spatial downsampling with conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling with conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    """
    Encoder: Maps image to latent distribution parameters (μ, logσ²).

    For CIFAR-10 (32x32) with channel_mults=(1,2,4):
    - Input: (B, 3, 32, 32)
    - After level 0: (B, 64, 16, 16)
    - After level 1: (B, 128, 8, 8)
    - After level 2: (B, 256, 4, 4)
    - Output μ, logvar: (B, latent_channels, 4, 4)
    """

    def __init__(self, config: VAEModelConfig, in_channels: int = 3):
        super().__init__()
        self.config = config
        ch = config.channels
        mults = config.channel_mults

        # Input projection
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        current_ch = ch

        for i, mult in enumerate(mults):
            out_ch = ch * mult
            block = nn.ModuleList(
                [
                    ResBlock(current_ch, out_ch, config.groups),
                    *[
                        ResBlock(out_ch, out_ch, config.groups)
                        for _ in range(config.num_res_blocks - 1)
                    ],
                ]
            )
            self.down_blocks.append(block)

            # Downsample except at last level
            if i < len(mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(out_ch)]))

            current_ch = out_ch

        # Output to latent parameters
        self.norm_out = nn.GroupNorm(config.groups, current_ch)
        self.conv_out = nn.Conv2d(current_ch, 2 * config.latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image (B, C, H, W)
        Returns:
            mu: Latent mean (B, latent_channels, H', W')
            logvar: Latent log-variance (B, latent_channels, H', W')
        """
        h = self.conv_in(x)

        for block in self.down_blocks:
            for layer in block:
                h = layer(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # Split into mu and logvar
        mu, logvar = torch.chunk(h, 2, dim=1)

        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder: Maps latent sample to reconstructed image.

    Mirror of encoder architecture.
    """

    def __init__(self, config: VAEModelConfig, out_channels: int = 3):
        super().__init__()
        self.config = config
        ch = config.channels
        mults = config.channel_mults

        # Compute the channel size at bottleneck
        bottleneck_ch = ch * mults[-1]

        # Input from latent
        self.conv_in = nn.Conv2d(config.latent_channels, bottleneck_ch, 3, padding=1)

        # Upsampling blocks (reverse order of encoder)
        self.up_blocks = nn.ModuleList()
        current_ch = bottleneck_ch

        for i, mult in enumerate(reversed(mults)):
            out_ch = ch * mult

            # Upsample except at first level
            if i > 0:
                self.up_blocks.append(nn.ModuleList([Upsample(current_ch)]))

            block = nn.ModuleList(
                [
                    ResBlock(current_ch, out_ch, config.groups),
                    *[
                        ResBlock(out_ch, out_ch, config.groups)
                        for _ in range(config.num_res_blocks - 1)
                    ],
                ]
            )
            self.up_blocks.append(block)
            current_ch = out_ch

        # Output projection
        self.norm_out = nn.GroupNorm(config.groups, current_ch)
        self.conv_out = nn.Conv2d(current_ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent sample (B, latent_channels, H', W')
        Returns:
            recon: Reconstructed image (B, C, H, W)
        """
        h = self.conv_in(z)

        for block in self.up_blocks:
            for layer in block:
                h = layer(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class VAE(nn.Module):
    """
    Variational Autoencoder.

    Training objective (ELBO):
        L = E_q[log p(x|z)] - β * KL(q(z|x) || p(z))
          = -reconstruction_loss - β * kl_loss

    where:
        - q(z|x) = N(μ_θ(x), σ_θ(x)) is the encoder (approximate posterior)
        - p(z) = N(0, I) is the prior
        - p(x|z) is the decoder (likelihood)
    """

    def __init__(self, config: VAEModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample z = μ + σ * ε

        This allows gradients to flow through the sampling operation.

        Args:
            mu: Mean of q(z|x), shape (B, C, H, W) or (B, D)
            logvar: Log-variance of q(z|x), same shape as mu

        Returns:
            z: Sampled latent, same shape as mu
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(mu)

        return mu + std * epsilon

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample to image."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """
        Full forward pass: encode, sample, decode.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            VAEOutput containing recon, mu, logvar, z
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return VAEOutput(recon=recon, mu=mu, logvar=logvar, z=z)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples by decoding random latents from prior p(z) = N(0, I).

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on

        Returns:
            samples: Generated images (num_samples, 3, H, W)
        """
        # Determine latent spatial size from config
        # For CIFAR-10 with 3 downsamples: 32 -> 16 -> 8 -> 4
        latent_h = 32 // (2 ** (len(self.config.channel_mults) - 1))
        latent_w = latent_h

        # Sample from prior
        z = torch.randn(num_samples, self.config.latent_channels, latent_h, latent_w, device=device)

        return self.decode(z)


# =============================================================================
# Discriminator for adversarial training
# =============================================================================


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    Outputs a grid of real/fake predictions instead of a single value.
    Each output pixel corresponds to a receptive field patch in the input.

    This is more stable than a global discriminator and provides
    per-patch gradients for the generator.
    """

    def __init__(self, config: DiscriminatorConfig, in_channels: int = 3):
        super().__init__()

        ch = config.channels
        n_layers = config.n_layers
        use_sn = config.use_spectral_norm

        # Helper to optionally apply spectral norm
        def maybe_sn(module):
            return nn.utils.spectral_norm(module) if use_sn else module

        layers = []

        # First layer (no normalization)
        layers.append(maybe_sn(nn.Conv2d(in_channels, ch, 4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Middle layers
        current_ch = ch
        for i in range(1, n_layers):
            out_ch = min(ch * (2**i), 512)
            layers.append(maybe_sn(nn.Conv2d(current_ch, out_ch, 4, stride=2, padding=1)))
            layers.append(nn.GroupNorm(8, out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_ch = out_ch

        # Final layer - output single channel (real/fake score per patch)
        layers.append(maybe_sn(nn.Conv2d(current_ch, 1, 4, stride=1, padding=1)))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, 3, H, W)

        Returns:
            Patch predictions (B, 1, H', W')
        """
        return self.model(x)


# =============================================================================
# LPIPS Perceptual Loss
# =============================================================================


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS).

    Uses pretrained VGG16 features to compute perceptual distance.
    Much better than MSE for image quality - captures structure, texture.

    Simplified implementation using VGG16 features at multiple scales.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained VGG16
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # Extract features at different layers
        # These correspond to relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])  # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])  # relu3_3
        self.slice4 = nn.Sequential(*list(vgg.children())[16:23])  # relu4_3
        self.slice5 = nn.Sequential(*list(vgg.children())[23:30])  # relu5_3

        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False

        # Learned weights for each layer (simplified: equal weights)
        self.weights = [1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5]

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet normalization."""
        # First convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Then apply ImageNet normalization
        return (x - self.mean) / self.std

    def _get_features(self, x: torch.Tensor) -> list:
        """Extract multi-scale VGG features."""
        x = self._normalize(x)

        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)

        return [h1, h2, h3, h4, h5]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance between two images.

        Args:
            x: First image (B, 3, H, W), range [-1, 1]
            y: Second image (B, 3, H, W), range [-1, 1]

        Returns:
            Perceptual distance (scalar)
        """
        feats_x = self._get_features(x)
        feats_y = self._get_features(y)

        loss = 0.0
        for i, (fx, fy) in enumerate(zip(feats_x, feats_y, strict=False)):
            # Normalize features
            fx = fx / (fx.norm(dim=1, keepdim=True) + 1e-10)
            fy = fy / (fy.norm(dim=1, keepdim=True) + 1e-10)

            # L2 distance in feature space
            loss += self.weights[i] * (fx - fy).pow(2).mean()

        return loss


# =============================================================================
# Loss functions
# =============================================================================


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between q(z|x) = N(μ, σ²) and p(z) = N(0, I).

    KL(N(μ, σ²) || N(0, I)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

    This has a nice closed-form solution for Gaussians.

    Args:
        mu: Mean of q(z|x), shape (B, ...)
        logvar: Log-variance of q(z|x), shape (B, ...)

    Returns:
        kl: KL divergence, scalar (mean over batch)
    """
    non_batch_dims = list(range(1, mu.ndim))
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=non_batch_dims).mean()


def reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, loss_type: str = "mse"
) -> torch.Tensor:
    """
    Compute reconstruction loss.

    Args:
        recon: Reconstructed image (B, C, H, W)
        target: Target image (B, C, H, W)
        loss_type: "mse" or "l1"

    Returns:
        loss: Reconstruction loss, scalar (mean over batch)
    """
    if loss_type == "mse":
        loss = F.mse_loss(recon, target, reduction="mean")
    else:
        loss = F.l1_loss(recon, target, reduction="mean")

    return loss


def hinge_loss_dis(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for discriminator.

    More stable than vanilla GAN loss. Used in BigGAN, StyleGAN, etc.

    L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]
    """
    real_loss = F.relu(1.0 - real_pred).mean()
    fake_loss = F.relu(1.0 + fake_pred).mean()
    return real_loss + fake_loss


def hinge_loss_gen(fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for generator.

    L_G = -E[D(G(z))]
    """
    return -fake_pred.mean()


def vae_loss(
    output: VAEOutput,
    target: torch.Tensor,
    kl_weight: float = 1.0,
    recon_type: str = "mse",
    lpips_model: LPIPS | None = None,
    lpips_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute total VAE loss: L = recon_loss + β * kl_loss + λ * lpips_loss

    Args:
        output: VAE forward pass output
        target: Target image
        kl_weight: β weight for KL term (β-VAE)
        recon_type: Type of reconstruction loss
        lpips_model: Optional LPIPS model for perceptual loss
        lpips_weight: Weight for perceptual loss

    Returns:
        loss: Total loss (scalar)
        metrics: Dict with individual loss components for logging
    """
    kl_loss = kl_divergence(output.mu, output.logvar)
    recon_loss = reconstruction_loss(output.recon, target, loss_type=recon_type)

    L = recon_loss + kl_weight * kl_loss

    metrics = {
        "loss": L.item(),
        "kl_loss": kl_loss.item(),
        "recon_loss": recon_loss.item(),
    }

    # Add perceptual loss if enabled
    if lpips_model is not None and lpips_weight > 0:
        lpips_loss = lpips_model(output.recon, target)
        L = L + lpips_weight * lpips_loss
        metrics["lpips_loss"] = lpips_loss.item()
        metrics["loss"] = L.item()

    return L, metrics


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

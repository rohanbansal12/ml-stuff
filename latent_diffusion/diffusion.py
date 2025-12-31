"""
Diffusion utilities for latent diffusion.

This is largely the same as regular DDPM diffusion, but operates on
VAE latent tensors instead of images. The math is identical.

Key difference: We may need to scale the latents to have unit variance
for optimal diffusion training.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from config import LatentDiffusionConfig
from tqdm import tqdm


class NoiseSchedule:
    """
    Precomputes all schedule tensors needed for diffusion.

    This class pre-calculates all the α, β, and derived quantities needed
    for the forward and reverse diffusion processes. Identical to pixel-space DDPM.

    Attributes:
        T: Number of diffusion timesteps.
        betas: Noise schedule β_t.
        alphas: 1 - β_t.
        alphas_cumprod: ᾱ_t = ∏_{s=1}^{t} α_s.
        sqrt_alphas_cumprod: √ᾱ_t.
        sqrt_one_minus_alphas_cumprod: √(1 - ᾱ_t).
        posterior_variance: β̃_t for reverse process.
    """

    def __init__(self, config: LatentDiffusionConfig, device: torch.device):
        """
        Initialize noise schedule from config.

        Args:
            config: Diffusion configuration containing T, beta_start, beta_end, etc.
            device: Device to place tensors on.
        """
        self.T = config.T
        self.device = device
        self.pred_type = config.pred_type

        if config.schedule_type == "linear":
            betas = torch.linspace(config.beta_start, config.beta_end, config.T)
        elif config.schedule_type == "cosine":
            betas = self._cosine_schedule(config.T, config.cosine_s, config.beta_end)
        else:
            raise ValueError(f"Unknown schedule: {config.schedule_type}")

        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.rsqrt(self.alphas)

        self.posterior_variance = (
            (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * self.betas
        )
        self.posterior_variance = self.posterior_variance.clamp(min=1e-20)
        self.posterior_log_variance = torch.log(self.posterior_variance)

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _cosine_schedule(self, T: int, s: float, max_beta: float) -> torch.Tensor:
        """
        Compute cosine noise schedule as proposed in "Improved DDPM".

        The cosine schedule provides better sample quality than linear,
        especially for high-resolution images.

        Math:
            f(t) = cos²((t/T + s) / (1 + s) · π/2)
            ᾱ_t = f(t) / f(0)
            β_t = 1 - ᾱ_t / ᾱ_{t-1}

        Args:
            T: Number of timesteps.
            s: Small offset to prevent β_t from being too small at t=0.
            max_beta: Maximum value for β_t (clipped).

        Returns:
            Tensor of shape (T,) containing β_t values.
        """
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1.0 - alphas
        return betas.clamp(min=1e-8, max=max_beta)

    def gather(
        self, values: torch.Tensor, t: torch.Tensor, shape: tuple
    ) -> torch.Tensor:
        """
        Gather values at timestep indices and reshape for broadcasting.

        Args:
            values: 1D tensor of precomputed values indexed by timestep.
            t: Batch of timestep indices, shape (B,).
            shape: Target shape for broadcasting (B, C, H, W).

        Returns:
            Gathered values reshaped to (B, 1, 1, 1) for broadcasting.
        """
        out = values.gather(0, t)
        return out.view(-1, *([1] * (len(shape) - 1)))

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample random timesteps uniformly from [0, T).

        Args:
            batch_size: Number of timesteps to sample.

        Returns:
            Tensor of shape (batch_size,) with random timesteps.
        """
        return torch.randint(0, self.T, (batch_size,), device=self.device)


def q_sample(
    z_0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward diffusion process: sample z_t given z_0.

    Samples from the forward diffusion posterior q(z_t | z_0), which has
    a closed-form Gaussian distribution.

    Math:
        q(z_t | z_0) = N(z_t; √ᾱ_t · z_0, (1 - ᾱ_t) · I)
        z_t = √ᾱ_t · z_0 + √(1 - ᾱ_t) · ε,  where ε ~ N(0, I)

    Args:
        z_0: Clean latent tensor, shape (B, C, H, W).
        t: Timestep indices, shape (B,).
        schedule: Noise schedule containing precomputed coefficients.
        noise: Optional pre-sampled noise. If None, sampled from N(0, I).

    Returns:
        Noisy latent z_t at timestep t, same shape as z_0.
    """
    if noise is None:
        noise = torch.randn_like(z_0)

    sqrt_alpha_bar = schedule.gather(
        values=schedule.sqrt_alphas_cumprod, t=t, shape=z_0.shape
    )
    sqrt_one_minus_alpha_bar = schedule.gather(
        values=schedule.sqrt_one_minus_alphas_cumprod, t=t, shape=z_0.shape
    )

    return sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * noise


def get_target(
    z_0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    pred_type: str,
) -> torch.Tensor:
    """
    Compute training target based on prediction parameterization.

    Different parameterizations are mathematically equivalent but have
    different training dynamics:

    Math:
        - eps:  target = ε (predict the noise)
        - x0:   target = z_0 (predict the clean latent)
        - v:    target = √ᾱ_t · ε - √(1-ᾱ_t) · z_0 (predict velocity)

    Args:
        z_0: Clean latent tensor, shape (B, C, H, W).
        noise: Noise tensor ε ~ N(0, I), same shape as z_0.
        t: Timestep indices, shape (B,).
        schedule: Noise schedule containing precomputed coefficients.
        pred_type: One of "eps", "x0", or "v".

    Returns:
        Target tensor for MSE loss, same shape as z_0.

    Raises:
        ValueError: If pred_type is not recognized.
    """
    if pred_type == "eps":
        return noise
    elif pred_type == "x0":
        return z_0
    elif pred_type == "v":
        sqrt_alpha_bar = schedule.gather(
            values=schedule.sqrt_alphas_cumprod, t=t, shape=z_0.shape
        )
        sqrt_one_minus_alpha_bar = schedule.gather(
            values=schedule.sqrt_one_minus_alphas_cumprod, t=t, shape=z_0.shape
        )
        return sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * z_0
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}")


def predict_z0_from_output(
    model_out: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    pred_type: str,
    clip_denoised: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert model output to (z0_pred, eps_pred) regardless of parameterization.

    This function inverts the forward process to recover both the predicted
    clean latent and the predicted noise from any prediction type.

    Math:
        From ε-prediction:
            z₀ = (z_t - √(1-ᾱ_t) · ε̂) / √ᾱ_t

        From x₀-prediction:
            ε̂ = (z_t - √ᾱ_t · ẑ₀) / √(1-ᾱ_t)

        From v-prediction:
            ẑ₀ = √ᾱ_t · z_t - √(1-ᾱ_t) · v̂
            ε̂ = √(1-ᾱ_t) · z_t + √ᾱ_t · v̂

    Args:
        model_out: Raw model output tensor, shape (B, C, H, W).
        z_t: Noisy latent at timestep t, same shape.
        t: Timestep indices, shape (B,).
        schedule: Noise schedule containing precomputed coefficients.
        pred_type: One of "eps", "x0", or "v".
        clip_denoised: Whether to clip z0_pred to [-1, 1]. Usually False
            for latents since they're not bounded.

    Returns:
        Tuple of (z0_pred, eps_pred), each with same shape as model_out.

    Raises:
        ValueError: If pred_type is not recognized.
    """
    sqrt_alpha_bar = schedule.gather(
        values=schedule.sqrt_alphas_cumprod, t=t, shape=z_t.shape
    )
    sqrt_one_minus_alpha_bar = schedule.gather(
        values=schedule.sqrt_one_minus_alphas_cumprod, t=t, shape=z_t.shape
    )

    if pred_type == "eps":
        eps_pred = model_out
        z0_pred = (z_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar

    elif pred_type == "x0":
        z0_pred = model_out
        eps_pred = (z_t - sqrt_alpha_bar * z0_pred) / sqrt_one_minus_alpha_bar

    elif pred_type == "v":
        v_pred = model_out
        z0_pred = sqrt_alpha_bar * z_t - sqrt_one_minus_alpha_bar * v_pred
        eps_pred = sqrt_one_minus_alpha_bar * z_t + sqrt_alpha_bar * v_pred
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}")

    if clip_denoised:
        z0_pred = z0_pred.clamp(-1.0, 1.0)
        eps_pred = (z_t - sqrt_alpha_bar * z0_pred) / sqrt_one_minus_alpha_bar

    return z0_pred, eps_pred


class DDPMSampler:
    """
    DDPM sampler for latent space reverse diffusion.

    Implements the standard DDPM reverse process, sampling from p(z_{t-1} | z_t)
    at each timestep from T-1 down to 0.

    Attributes:
        schedule: Noise schedule with precomputed coefficients.
        pred_type: Prediction parameterization ("eps", "x0", or "v").
        clip_denoised: Whether to clip predicted z_0 to [-1, 1].
    """

    def __init__(
        self, schedule: NoiseSchedule, pred_type: str, clip_denoised: bool = False
    ):
        """
        Initialize DDPM sampler.

        Args:
            schedule: Noise schedule containing precomputed coefficients.
            pred_type: One of "eps", "x0", or "v".
            clip_denoised: Whether to clip z0_pred. Usually False for latents.
        """
        self.schedule = schedule
        self.pred_type = pred_type
        self.clip_denoised = clip_denoised

    @torch.no_grad()
    def step(self, model: nn.Module, z_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Single reverse diffusion step from t to t-1.

        Samples from the reverse posterior p_θ(z_{t-1} | z_t) using the
        model's noise prediction.

        Math:
            μ_θ(z_t, t) = (1/√α_t) · (z_t - β_t · ε̂_θ / √(1-ᾱ_t))
            z_{t-1} = μ_θ + √β̃_t · ε,  where ε ~ N(0, I) for t > 0
            z_{t-1} = μ_θ               for t = 0

        Args:
            model: Denoising model that predicts noise/x0/v given (z_t, t).
            z_t: Current noisy latent, shape (B, C, H, W).
            t: Current timestep (integer).

        Returns:
            Denoised latent z_{t-1}, same shape as z_t.
        """
        batch_size = z_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=z_t.device, dtype=torch.long)

        model_out = model(z_t, t_tensor)
        z0_pred, eps_pred = predict_z0_from_output(
            model_out, z_t, t_tensor, self.schedule, self.pred_type, self.clip_denoised
        )

        shape = z_t.shape
        sqrt_recip_alpha = self.schedule.gather(
            self.schedule.sqrt_recip_alphas, t_tensor, shape
        )
        beta = self.schedule.gather(self.schedule.betas, t_tensor, shape)
        sqrt_one_minus_alpha_bar = self.schedule.gather(
            self.schedule.sqrt_one_minus_alphas_cumprod, t_tensor, shape
        )

        # Compute posterior mean
        mu = sqrt_recip_alpha * (z_t - beta * eps_pred / sqrt_one_minus_alpha_bar)

        if t > 0:
            posterior_var = self.schedule.gather(
                self.schedule.posterior_variance, t_tensor, shape
            )
            noise = torch.randn_like(z_t)
            z_prev = mu + torch.sqrt(posterior_var) * noise
        else:
            z_prev = mu

        return z_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: torch.device,
        z_T: Optional[torch.Tensor] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse diffusion from z_T ~ N(0, I) to z_0.

        Iteratively applies the reverse step from t = T-1 down to t = 0.

        Args:
            model: Denoising model that predicts noise/x0/v given (z_t, t).
            shape: Shape of latent to generate, e.g., (B, 4, 8, 8).
            device: Device to run sampling on.
            z_T: Optional starting noise. If None, sampled from N(0, I).
            progress: Whether to show tqdm progress bar.

        Returns:
            Generated clean latent z_0, shape as specified.
        """
        model.eval()

        z_t = z_T if z_T is not None else torch.randn(shape, device=device)

        timesteps = range(self.schedule.T - 1, -1, -1)
        if progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")

        for t in timesteps:
            z_t = self.step(model, z_t=z_t, t=t)

        return z_t


class DDIMSampler:
    """
    DDIM sampler for accelerated latent space sampling.

    DDIM (Denoising Diffusion Implicit Models) allows sampling with fewer
    steps than DDPM by using a non-Markovian reverse process.

    With η=0, DDIM is deterministic (same noise → same output).
    With η=1, DDIM is equivalent to DDPM.

    Attributes:
        schedule: Noise schedule with precomputed coefficients.
        pred_type: Prediction parameterization ("eps", "x0", or "v").
        eta: Stochasticity parameter. 0 = deterministic, 1 = DDPM.
        timesteps: Subsampled timestep sequence for accelerated sampling.
    """

    def __init__(
        self,
        schedule: NoiseSchedule,
        pred_type: str,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = False,
    ):
        """
        Initialize DDIM sampler.

        Args:
            schedule: Noise schedule containing precomputed coefficients.
            pred_type: One of "eps", "x0", or "v".
            num_inference_steps: Number of denoising steps (can be << T).
            eta: Stochasticity parameter in [0, 1].
            clip_denoised: Whether to clip z0_pred. Usually False for latents.
        """
        self.schedule = schedule
        self.pred_type = pred_type
        self.eta = eta
        self.clip_denoised = clip_denoised

        # Create timestep subsequence (evenly spaced)
        step_ratio = schedule.T // num_inference_steps
        self.timesteps = torch.arange(0, schedule.T, step_ratio).flip(0)

    @torch.no_grad()
    def step(
        self, model: nn.Module, z_t: torch.Tensor, t: int, t_prev: int
    ) -> torch.Tensor:
        """
        Single DDIM step from timestep t to t_prev.

        DDIM uses a non-Markovian update that allows skipping timesteps.

        Math:
            σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})
            z_{t-1} = √ᾱ_{t-1} · ẑ₀ + √(1 - ᾱ_{t-1} - σ²) · ε̂ + σ · ε

        When η=0: deterministic (σ=0, no noise added)
        When η=1: equivalent to DDPM

        Args:
            model: Denoising model that predicts noise/x0/v given (z_t, t).
            z_t: Current noisy latent, shape (B, C, H, W).
            t: Current timestep (integer).
            t_prev: Target timestep (integer), typically t - step_size.

        Returns:
            Denoised latent at timestep t_prev, same shape as z_t.
        """
        batch_size = z_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=z_t.device, dtype=torch.long)

        model_out = model(z_t, t_tensor)
        z0_pred, eps_pred = predict_z0_from_output(
            model_out=model_out,
            z_t=z_t,
            t=t_tensor,
            schedule=self.schedule,
            pred_type=self.pred_type,
            clip_denoised=self.clip_denoised,
        )

        # Get alpha values (indexed by integer, not tensor)
        alpha_bar_t = self.schedule.alphas_cumprod[t]
        alpha_bar_prev = (
            self.schedule.alphas_cumprod[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=z_t.device)
        )

        # Compute sigma for stochasticity
        sigma = self.eta * torch.sqrt(
            (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
            * (1 - alpha_bar_t / alpha_bar_prev)
        )

        # DDIM update: "predicted x0" direction + "pointing to x_t" direction
        pred_dir = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_pred
        z_prev = torch.sqrt(alpha_bar_prev) * z0_pred + pred_dir

        # Add noise if η > 0 and not at final step
        if t_prev > 0 and self.eta > 0:
            noise = torch.randn_like(z_t)
            z_prev = z_prev + sigma * noise

        return z_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: torch.device,
        z_T: Optional[torch.Tensor] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling with accelerated timestep schedule.

        Uses fewer steps than full DDPM by skipping timesteps.

        Args:
            model: Denoising model that predicts noise/x0/v given (z_t, t).
            shape: Shape of latent to generate, e.g., (B, 4, 8, 8).
            device: Device to run sampling on.
            z_T: Optional starting noise. If None, sampled from N(0, I).
            progress: Whether to show tqdm progress bar.

        Returns:
            Generated clean latent z_0, shape as specified.
        """
        model.eval()

        z_t = z_T if z_T is not None else torch.randn(shape, device=device)

        # Create pairs of (current_t, prev_t)
        timesteps = self.timesteps.tolist()
        timesteps_prev = timesteps[1:] + [-1]  # -1 indicates final step (t=0 target)

        pairs = list(zip(timesteps, timesteps_prev))
        if progress:
            pairs = tqdm(pairs, desc=f"DDIM Sampling ({len(timesteps)} steps)")

        for t, t_prev in pairs:
            z_t = self.step(model, z_t=z_t, t=t, t_prev=t_prev)

        return z_t

"""
Diffusion process: noise schedules, forward/reverse processes.

Key equations (DDPM):
- Forward: q(x_t | x_0) = N(x_t; √ᾱₜ·x_0, (1-ᾱₜ)I)
- Reverse: p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)

Prediction parameterizations:
- ε-prediction: model predicts noise ε
- x₀-prediction: model predicts clean image x₀
- v-prediction: model predicts v = √ᾱₜ·ε - √(1-ᾱₜ)·x₀
"""

import torch
import math
from typing import Optional
from config import DiffusionConfig


class NoiseSchedule:
    """
    Precomputes all the schedule tensors needed for diffusion.
    Keeps them on a specified device for efficient gathering.
    """

    def __init__(self, config: DiffusionConfig, device: torch.device):
        self.T = config.T
        self.device = device
        self.pred_type = config.pred_type

        # Compute betas based on schedule type
        if config.schedule_type == "linear":
            betas = torch.linspace(config.beta_start, config.beta_end, config.T)
        elif config.schedule_type == "cosine":
            betas = self._cosine_schedule(config.T, config.cosine_s, config.beta_end)
        else:
            raise ValueError(f"Unknown schedule: {config.schedule_type}")

        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱ_{t-1} with ᾱ_0 = 1
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]]
        )

        # Precompute commonly used terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.rsqrt(self.alphas)

        # Posterior variance: β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t
        self.posterior_variance = (
            (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * self.betas
        )
        # Clamp for numerical stability at t=0
        self.posterior_variance = self.posterior_variance.clamp(min=1e-20)
        self.posterior_log_variance = torch.log(self.posterior_variance)

        # For posterior mean computation
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

        self._log_schedule_info(config.schedule_type)

    def _cosine_schedule(self, T: int, s: float, max_beta: float) -> torch.Tensor:
        """
        Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal).
        f(t) = cos²((t/T + s) / (1+s) · π/2)
        ᾱ_t = f(t) / f(0)
        """
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        # Get alphas from cumulative product
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1.0 - alphas
        return betas.clamp(min=1e-8, max=max_beta)

    def _log_schedule_info(self, schedule_type: str):
        """Print schedule diagnostics."""
        print(f"Noise Schedule: {schedule_type}, T={self.T}")
        print(f"  β: [{self.betas[0]:.6f}, ..., {self.betas[-1]:.6f}]")
        print(
            f"  ᾱ: [{self.alphas_cumprod[0]:.4f}, ..., {self.alphas_cumprod[-1]:.6f}]"
        )
        print(f"  SNR range: [{self.snr(0):.1f}, {self.snr(self.T - 1):.4f}]")

    def snr(self, t: int) -> float:
        """Signal-to-noise ratio at timestep t."""
        return (self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])).item()

    def gather(
        self, values: torch.Tensor, t: torch.Tensor, shape: tuple
    ) -> torch.Tensor:
        """Gather values at timesteps t and reshape for broadcasting."""
        out = values.gather(0, t)
        return out.view(-1, *([1] * (len(shape) - 1)))

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps uniformly."""
        return torch.randint(0, self.T, (batch_size,), device=self.device)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward diffusion: sample x_t given x_0.
    q(x_t | x_0) = √ᾱₜ · x_0 + √(1-ᾱₜ) · ε
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alpha_bar = schedule.gather(schedule.sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alpha_bar = schedule.gather(
        schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape
    )

    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise


def get_target(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    pred_type: str,
) -> torch.Tensor:
    """
    Compute the training target based on prediction type.

    - eps: target = ε (the noise)
    - x0:  target = x_0 (the clean image)
    - v:   target = √ᾱₜ·ε - √(1-ᾱₜ)·x_0  (velocity)
    """
    if pred_type == "eps":
        return noise
    elif pred_type == "x0":
        return x0
    elif pred_type == "v":
        sqrt_alpha_bar = schedule.gather(schedule.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar = schedule.gather(
            schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )
        # v = √ᾱₜ·ε - √(1-ᾱₜ)·x₀
        return sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * x0
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}")


def predict_x0_from_output(
    model_out: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    pred_type: str,
    clip_x0: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert model output to (x0_pred, eps_pred) regardless of parameterization.

    Returns:
        x0_pred: Predicted clean image
        eps_pred: Predicted noise
    """
    shape = x_t.shape
    sqrt_alpha_bar = schedule.gather(schedule.sqrt_alphas_cumprod, t, shape)
    sqrt_one_minus_alpha_bar = schedule.gather(
        schedule.sqrt_one_minus_alphas_cumprod, t, shape
    )

    if pred_type == "eps":
        eps_pred = model_out
        # x_0 = (x_t - √(1-ᾱₜ)·ε) / √ᾱₜ
        x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar

    elif pred_type == "x0":
        x0_pred = model_out
        # ε = (x_t - √ᾱₜ·x_0) / √(1-ᾱₜ)
        eps_pred = (x_t - sqrt_alpha_bar * x0_pred) / sqrt_one_minus_alpha_bar

    elif pred_type == "v":
        v_pred = model_out
        # From v = √ᾱₜ·ε - √(1-ᾱₜ)·x₀, we can derive:
        # x_0 = √ᾱₜ·x_t - √(1-ᾱₜ)·v
        # ε = √(1-ᾱₜ)·x_t + √ᾱₜ·v
        x0_pred = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * v_pred
        eps_pred = sqrt_one_minus_alpha_bar * x_t + sqrt_alpha_bar * v_pred
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}")

    if clip_x0:
        x0_pred = x0_pred.clamp(-1.0, 1.0)
        # Recompute eps from clipped x0 for consistency
        eps_pred = (x_t - sqrt_alpha_bar * x0_pred) / sqrt_one_minus_alpha_bar

    return x0_pred, eps_pred


class DDPMSampler:
    """
    DDPM reverse process sampler.

    p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)

    where μ_θ = (1/√αₜ) · (x_t - βₜ·ε_θ / √(1-ᾱₜ))
    """

    def __init__(self, schedule: NoiseSchedule, pred_type: str, clip_x0: bool = True):
        self.schedule = schedule
        self.pred_type = pred_type
        self.clip_x0 = clip_x0

    @torch.no_grad()
    def step(self, model: torch.nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Single reverse diffusion step."""
        batch_size = x_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)

        # Get model prediction and convert to eps
        model_out = model(x_t, t_tensor)
        x0_pred, eps_pred = predict_x0_from_output(
            model_out, x_t, t_tensor, self.schedule, self.pred_type, self.clip_x0
        )

        # Compute posterior mean using the eps parameterization
        # μ_θ = (1/√αₜ)(x_t - βₜ·ε_θ/√(1-ᾱₜ))
        shape = x_t.shape
        sqrt_recip_alpha = self.schedule.gather(
            self.schedule.sqrt_recip_alphas, t_tensor, shape
        )
        beta = self.schedule.gather(self.schedule.betas, t_tensor, shape)
        sqrt_one_minus_alpha_bar = self.schedule.gather(
            self.schedule.sqrt_one_minus_alphas_cumprod, t_tensor, shape
        )

        mu = sqrt_recip_alpha * (x_t - beta * eps_pred / sqrt_one_minus_alpha_bar)

        if t > 0:
            posterior_var = self.schedule.gather(
                self.schedule.posterior_variance, t_tensor, shape
            )
            noise = torch.randn_like(x_t)
            x_prev = mu + torch.sqrt(posterior_var) * noise
        else:
            x_prev = mu

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        device: torch.device,
        x_T: Optional[torch.Tensor] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Full reverse diffusion: x_T → x_0."""
        model.eval()

        if x_T is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = x_T

        timesteps = range(self.schedule.T - 1, -1, -1)
        if progress:
            from tqdm import tqdm

            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            x_t = self.step(model, x_t, t)

        return x_t


class DDIMSampler:
    """
    DDIM sampler for accelerated sampling.
    Can use fewer steps than training T.
    """

    def __init__(
        self,
        schedule: NoiseSchedule,
        pred_type: str,
        num_inference_steps: int = 50,
        eta: float = 0.0,  # 0 = deterministic, 1 = DDPM
        clip_x0: bool = True,
    ):
        self.schedule = schedule
        self.pred_type = pred_type
        self.eta = eta
        self.clip_x0 = clip_x0

        # Create timestep subsequence
        step_ratio = schedule.T // num_inference_steps
        self.timesteps = torch.arange(0, schedule.T, step_ratio).flip(0)

    @torch.no_grad()
    def step(
        self, model: torch.nn.Module, x_t: torch.Tensor, t: int, t_prev: int
    ) -> torch.Tensor:
        """Single DDIM step from t to t_prev."""
        batch_size = x_t.size(0)
        device = x_t.device

        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Get predictions
        model_out = model(x_t, t_tensor)
        x0_pred, eps_pred = predict_x0_from_output(
            model_out, x_t, t_tensor, self.schedule, self.pred_type, self.clip_x0
        )

        # Get alpha values
        alpha_bar_t = self.schedule.alphas_cumprod[t]
        alpha_bar_prev = (
            self.schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        )

        # DDIM update
        # σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})
        sigma = self.eta * torch.sqrt(
            (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
            * (1 - alpha_bar_t / alpha_bar_prev)
        )

        # Direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_pred

        # x_{t-1} = √ᾱ_{t-1} · x̂_0 + direction + noise
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + pred_dir

        if t_prev > 0 and self.eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        device: torch.device,
        x_T: Optional[torch.Tensor] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """DDIM sampling with fewer steps."""
        model.eval()

        if x_T is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = x_T

        timesteps = self.timesteps.tolist()
        timesteps_prev = timesteps[1:] + [-1]

        pairs = list(zip(timesteps, timesteps_prev))
        if progress:
            from tqdm import tqdm

            pairs = tqdm(pairs, desc=f"DDIM Sampling ({len(timesteps)} steps)")

        for t, t_prev in pairs:
            x_t = self.step(model, x_t, t, t_prev)

        return x_t

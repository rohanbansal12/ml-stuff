import torch
import math

class DiffusionSchedule:
    def __init__(self, T, beta_start, beta_end, device, schedule_type="linear", cosine_s=0.008):
        self.T = T
        self.device = device
        self.schedule_type = schedule_type

        if schedule_type == 'linear':
            betas = torch.linspace(beta_start, beta_end, T, device=device)
        elif schedule_type == 'cosine':
            steps = T + 1
            t = torch.linspace(0, T, steps, device=device) / T
            f = torch.cos(((t + cosine_s) / (1 + cosine_s)) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            alphas = alpha_bar[1:] / alpha_bar[:-1]
            betas = 1 - alphas
            betas = betas.clamp(min=1e-8, max=beta_end) # ugly hack because with small T we have beta blowup and not enough training
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        self.betas = betas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.ones_like(self.alphas_cumprod)
        self.alphas_cumprod_prev[1:] = self.alphas_cumprod[:-1]

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas = 1 / torch.sqrt(self.alphas)
        self.posterior_variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * self.betas
        self.posterior_variance.clamp_(min=1e-20)

        print(schedule_type, T, betas[:5], betas[-5:])

    def sample_t(self, b, device):
        return torch.randint(0, self.T, (b,), device=device)
    
    def gather(self, tensor_1d, t, x_shape):
        return tensor_1d[t].view(x_shape[0], 1, 1, 1)


def q_sample(x0, t, schedule: DiffusionSchedule, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alphas_cum = schedule.alphas_cumprod.to(x0.device)[t].view(-1, 1, 1, 1)
    return torch.sqrt(alphas_cum) * x0 + torch.sqrt(1 - alphas_cum) * noise

def p_sample_step(model, x_t, t_scalar, schedule: DiffusionSchedule, pred_type):
    b = x_t.size(0)
    t = torch.ones(b, device=x_t.device, dtype=torch.long) * t_scalar
    model_out = model(x_t, t)
    shape = x_t.shape

    if pred_type == 'x0':
        sqrt_alpha_bar_t = schedule.gather(schedule.sqrt_alphas_cumprod, t, shape)
        sqrt_one_minus_alpha_bar_t = schedule.gather(schedule.sqrt_one_minus_alphas_cumprod, t, shape)
        x0_pred = model_out
        x0_pred = x0_pred.clamp(-1.0, 1.0)
        eps_theta = (x_t - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t
    elif pred_type == 'eps':
        eps_theta = model_out
    elif pred_type == 'v':
        sqrt_alpha_bar_t = schedule.gather(schedule.sqrt_alphas_cumprod, t, shape)
        sqrt_one_minus_alpha_bar_t = schedule.gather(schedule.sqrt_one_minus_alphas_cumprod, t, shape)
        v_pred = model_out
        eps_theta = sqrt_alpha_bar_t * v_pred + sqrt_one_minus_alpha_bar_t * x_t
    else:
        raise ValueError(f"Unknown prediction type: {pred_type}")

    recip_sqrt_alpha = schedule.gather(schedule.sqrt_recip_alphas, t, shape)
    betas = schedule.gather(schedule.betas, t, shape)
    post_var = schedule.gather(schedule.posterior_variance, t, shape)
    sqrt_one_minus_alphas_cumprod = schedule.gather(schedule.sqrt_one_minus_alphas_cumprod, t, shape)

    mu_theta = recip_sqrt_alpha * (x_t - betas * eps_theta / sqrt_one_minus_alphas_cumprod)

    if t_scalar > 0:
        z = torch.randn_like(x_t)
        x_prev = mu_theta + torch.sqrt(post_var) * z
    else:
        x_prev = mu_theta

    return x_prev

def p_sample_loop(model, schedule : DiffusionSchedule, pred_type, shape, device, x_t=None):
    if x_t is None:
        x_t = torch.randn(shape, device=device)
    for t in range(schedule.T-1, -1, -1):
        x_t = p_sample_step(model, x_t, t, schedule, pred_type=pred_type)
    return x_t
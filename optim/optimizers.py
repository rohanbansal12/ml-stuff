"""
Custom PyTorch Optimizer Implementations

Implements: SGD with Momentum, RMSProp, Adam, AdamW, Muon
"""

from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer


class SGDMomentum(Optimizer):
    """
    Stochastic Gradient Descent with Momentum.

    v_t = momentum * v_{t-1} + grad (or + (1 - dampening) * grad if dampening > 0)

    If nesterov:
        p = p - lr * (grad + momentum * v_t)
    else:
        p = p - lr * v_t
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = max(group["dampening"], 0.0)
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = p.grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["v"] = grad.clone()  # first step: initialize to grad directly
                else:
                    state["v"] = momentum * state["v"] + (1 - dampening) * grad

                if group["nesterov"]:
                    p.add_(grad + momentum * state["v"], alpha=-lr)
                else:
                    p.add_(state["v"], alpha=-lr)

        return loss


class RMSProp(Optimizer):
    """
    RMSProp optimizer.

    v_t = alpha * v_{t-1} + (1 - alpha) * grad^2

    If centered:
        g_t = alpha * g_{t-1} + (1 - alpha) * grad
        v_hat = v_t - g_t^2
        p = p - lr * grad / (sqrt(v_hat) + eps)
    else:
        p = p - lr * grad / (sqrt(v_t) + eps)

    If momentum > 0, applies momentum to the update.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            alpha = group["alpha"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = p.grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["v"] = torch.zeros_like(p)
                    if group["centered"]:
                        state["g"] = torch.zeros_like(p)
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                state["v"].mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group["centered"]:
                    state["g"].lerp_(grad, 1 - alpha)
                    v = state["v"] - state["g"].pow(2)
                else:
                    v = state["v"]

                update = grad / (v.sqrt() + eps)

                if momentum > 0:
                    state["momentum_buffer"] = momentum * state["momentum_buffer"] + update
                    p.add_(state["momentum_buffer"], alpha=-lr)
                else:
                    p.add_(update, alpha=-lr)

        return loss


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2

    m_hat = m_t / (1 - beta1^t)  # bias correction
    v_hat = v_t / (1 - beta2^t)

    If amsgrad:
        v_hat = max(v_hat, v_hat_{t-1})

    p = p - lr * m_hat / (sqrt(v_hat) + eps)

    Note: weight_decay here is L2 regularization (added to gradient),
    NOT decoupled weight decay. Use AdamW for decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = p.grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["v_max"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                state["m"].lerp_(grad, 1 - betas[0])
                state["v"].lerp_(grad.pow(2), 1 - betas[1])

                m_hat = state["m"] / (1 - betas[0] ** t)

                if group["amsgrad"]:
                    state["v_max"] = torch.max(state["v"], state["v_max"])
                    v_hat = state["v_max"] / (1 - betas[1] ** t)
                else:
                    v_hat = state["v"] / (1 - betas[1] ** t)

                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

        return loss


class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).

    Same as Adam but weight decay is applied directly to weights,
    not added to gradients:

    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2

    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)

    p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)

    Or equivalently:
    p = p * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["v_max"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                state["m"].lerp_(grad, 1 - betas[0])
                state["v"].lerp_(grad.pow(2), 1 - betas[1])

                m_hat = state["m"] / (1 - betas[0] ** t)

                if group["amsgrad"]:
                    state["v_max"] = torch.max(state["v"], state["v_max"])
                    v_hat = state["v_max"] / (1 - betas[1] ** t)
                else:
                    v_hat = state["v"] / (1 - betas[1] ** t)

                p.add_(m_hat / (v_hat.sqrt() + eps) + weight_decay * p, alpha=-lr)

        return loss


class Muon(Optimizer):
    """
    Muon optimizer (Momentum Orthogonalized Update).

    Applies Newton-Schulz iterations to orthogonalize the momentum,
    which approximates multiplying by the inverse square root of the
    gradient covariance matrix (similar to natural gradient / K-FAC
    but much cheaper).

    For 2D+ params (matrices):
        1. Compute momentum: m_t = momentum * m_{t-1} + grad (nesterov optional)
        2. Orthogonalize m_t using Newton-Schulz iterations
        3. p = p - lr * orthogonalized_m_t

    For 1D params (biases, norms): falls back to standard SGD with momentum.

    Newton-Schulz iteration for computing X @ (X^T @ X)^{-1/2}:
        Y_0 = X / ||X||_F
        for i in range(ns_steps):
            Y_{i+1} = Y_i @ (aI - bY_i^T @ Y_i + cY_i^T @ Y_i @ Y_i^T @ Y_i)
        where a, b, c are chosen coefficients (commonly a=3, b=3, c=1 or quintic variants)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _newton_schulz(self, G: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Newton-Schulz iteration to compute G @ (G^T @ G)^{-1/2}.

        This orthogonalizes the rows of G.

        Args:
            G: Input matrix to orthogonalize (can be reshaped from higher-dim tensor)
            steps: Number of Newton-Schulz iterations

        Returns:
            Orthogonalized matrix with same shape as G
        """
        assert G.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT

        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X

        if G.size(-2) > G.size(-1):
            X = X.mT

        return X

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                momentum = state["momentum_buffer"]
                momentum.lerp_(grad, 1 - beta)
                update = grad.lerp_(momentum, beta) if group["nesterov"] else momentum

                if p.ndim >= 2:
                    update = self._newton_schulz(update, steps=group["ns_steps"]).to(p.dtype)

                p.add_(update + weight_decay * p, alpha=-lr)

        return loss

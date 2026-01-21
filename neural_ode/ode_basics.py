"""
ODE Solvers - Foundation for Neural ODEs

Before neural ODEs, understand how to solve ODEs numerically.

An ODE: dz/dt = f(t, z)
Given initial condition z(t0) = z0, find z(t1).

Key insight for Neural ODEs:
    Instead of f being a known function, f is a neural network!
    z(t1) = z(t0) + ∫[t0 to t1] f(t, z(t); θ) dt

Numerical methods (in order of accuracy):
    1. Euler: z(t+h) = z(t) + h * f(t, z)
    2. Midpoint: Use slope at midpoint
    3. RK4: Weighted average of 4 slope estimates
    4. Adaptive: Adjust step size based on error estimate

Exercises:
    1. Implement Euler, RK4 solvers
    2. Solve simple ODE: dz/dt = -z (exponential decay)
    3. Solve 2D system: Lotka-Volterra (predator-prey)
    4. Compare accuracy vs number of steps
    5. Implement adaptive step size (Dormand-Prince / RK45)
"""

from collections.abc import Callable

import torch


def euler_step(f: Callable, z: torch.Tensor, t: float, h: float):
    """Single Euler step.

    z(t+h) = z(t) + h * f(t, z)

    Args:
        f: Dynamics function f(t, z) -> dz/dt
        z: Current state
        t: Current time
        h: Step size

    Returns:
        z at time t+h
    """
    return z + h * f(t, z)


def rk4_step(f: Callable, z: torch.Tensor, t: float, h: float):
    """Single RK4 (Runge-Kutta 4th order) step.

    Classic 4th order method:
        k1 = f(t, z)
        k2 = f(t + h/2, z + h/2 * k1)
        k3 = f(t + h/2, z + h/2 * k2)
        k4 = f(t + h, z + h * k3)
        z(t+h) = z + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    Much more accurate than Euler for same step size.

    Args:
        f: Dynamics function f(t, z) -> dz/dt
        z: Current state
        t: Current time
        h: Step size

    Returns:
        z at time t+h
    """
    k1 = f(t, z)
    k2 = f(t + h / 2, z + h / 2 * k1)
    k3 = f(t + h / 2, z + h / 2 * k2)
    k4 = f(t + h, z + h * k3)

    return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint(f: Callable, z0: torch.Tensor, t: torch.Tensor, method: str = "rk4"):
    """Integrate ODE from t[0] to t[-1], returning z at all times in t.

    Args:
        f: Dynamics function f(t, z) -> dz/dt
        z0: Initial condition, shape (batch, dim) or (dim,)
        t: Times to evaluate at, shape (n_times,), must be sorted
        method: Integration method, "rk4" or "euler"

    Returns:
        z: States at each time, shape (n_times, *z0.shape)
    """
    zs = [z0]

    for i in range(t.numel() - 1):
        h = (t[i + 1] - t[i]).item()
        t_i = t[i].item()
        if method == "rk4":
            zs.append(rk4_step(f, zs[-1], t_i, h))
        elif method == "euler":
            zs.append(euler_step(f, zs[-1], t_i, h))
        else:
            raise ValueError(f"Unknown method: {method}")

    return torch.stack(zs)


def _dopri5_step(
    f: Callable, z: torch.Tensor, t: float, h: float
):
    """Single Dormand-Prince step with error estimate.

    Returns:
        z_new: 5th order solution
        z_err: Error estimate (difference between 5th and 4th order)
        k7: Final stage (FSAL - first same as last)
    """
    # Dormand-Prince coefficients
    c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1.0
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84

    # 5th order weights (same as a7*)
    b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
    # 4th order weights (for error estimation)
    e1, e2, e3, e4, e5, e6, e7 = 5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40

    k1 = f(t, z)
    k2 = f(t + c2 * h, z + h * a21 * k1)
    k3 = f(t + c3 * h, z + h * (a31 * k1 + a32 * k2))
    k4 = f(t + c4 * h, z + h * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = f(t + c5 * h, z + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = f(t + c6 * h, z + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))

    # 5th order solution
    z_new = z + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)

    k7 = f(t + h, z_new)

    # Error estimate: difference between 5th and 4th order
    z_err = h * ((b1 - e1) * k1 + (b3 - e3) * k3 + (b4 - e4) * k4 +
                 (b5 - e5) * k5 + (b6 - e6) * k6 + (b7 - e7) * k7)

    return z_new, z_err, k7


def odeint_adaptive(
    f: Callable,
    z0: torch.Tensor,
    t: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    max_steps: int = 10000,
):
    """Adaptive step size ODE integration using Dormand-Prince (RK45).

    Uses error estimation to automatically choose step sizes.
    Smaller steps where dynamics are fast, larger where slow.

    Error control: |error| < atol + rtol * |z|

    Args:
        f: Dynamics function f(t, z) -> dz/dt
        z0: Initial condition, shape (batch, dim) or (dim,)
        t: Times to evaluate at, shape (n_times,), must be sorted
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_steps: Maximum number of steps (to prevent infinite loops)

    Returns:
        z: States at each time, shape (n_times, *z0.shape)
    """
    t_list = [ti.item() for ti in t]
    zs = [z0]
    z_current = z0
    t_current = t_list[0]

    # Safety factor and step size bounds
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0

    # Initial step size estimate
    h = (t_list[-1] - t_list[0]) / 100.0

    t_idx = 1  # Next time point to record
    n_steps = 0

    while t_idx < len(t_list) and n_steps < max_steps:
        t_target = t_list[t_idx]

        # Don't step past the target time
        h = min(h, t_target - t_current)

        # Take a trial step
        z_new, z_err, _ = _dopri5_step(f, z_current, t_current, h)

        # Compute error norm (scaled)
        scale = atol + rtol * torch.max(torch.abs(z_current), torch.abs(z_new))
        err_norm = torch.max(torch.abs(z_err) / scale).item()

        if err_norm <= 1.0:
            # Accept step
            t_current = t_current + h
            z_current = z_new
            n_steps += 1

            # Record if we've reached a target time
            if abs(t_current - t_target) < 1e-12:
                zs.append(z_current)
                t_idx += 1

            # Compute optimal step size for next iteration
            if err_norm > 1e-10:
                h_opt = h * safety * (1.0 / err_norm) ** 0.2
            else:
                h_opt = h * max_factor
            h = min(max(h_opt, h * min_factor), h * max_factor)
        else:
            # Reject step, reduce step size
            h_opt = h * safety * (1.0 / err_norm) ** 0.25
            h = max(h_opt, h * min_factor)

    if n_steps >= max_steps:
        raise RuntimeError(f"odeint_adaptive exceeded {max_steps} steps")

    return torch.stack(zs)


# ============ Test Problems ============


def exponential_decay(t: float, z: torch.Tensor):
    """dz/dt = -z, solution: z(t) = z(0) * exp(-t)"""
    return -z


def harmonic_oscillator(t: float, z: torch.Tensor):
    """2D system: position and velocity of harmonic oscillator.

    dz/dt = [v, -x] where z = [x, v]
    Solution: circular motion in phase space
    """
    res = torch.zeros_like(z)
    res[..., 0] = z[..., 1]
    res[..., 1] = -z[..., 0]
    return res


def lotka_volterra(
    t: float,
    z: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 1.5,
    delta: float = 0.075,
):
    """Predator-prey dynamics.

    z = [prey, predator]
    d(prey)/dt = alpha * prey - beta * prey * predator
    d(predator)/dt = delta * prey * predator - gamma * predator
    """
    res = torch.zeros_like(z)
    res[..., 0] = alpha * z[..., 0] - beta * z[..., 0] * z[..., 1]
    res[..., 1] = delta * z[..., 0] * z[..., 1] - gamma * z[..., 1]
    return res


def main():
    """Test ODE solvers on known systems."""
    torch.manual_seed(42)

    # Test 1 - Exponential decay
    # Compare numerical solution to analytical: z(t) = z0 * exp(-t)
    print("=" * 50)
    print("Test 1: Exponential Decay")
    print("=" * 50)

    z0 = torch.tensor([2.0])
    t = torch.linspace(0, 3, 31)

    z_numerical = odeint(exponential_decay, z0, t, method="rk4")
    z_analytical = z0 * torch.exp(-t).unsqueeze(-1)

    max_error = torch.max(torch.abs(z_numerical - z_analytical)).item()
    print(f"Initial condition: z0 = {z0.item()}")
    print(f"Time range: [0, 3] with 31 points")
    print(f"Max error (RK4 vs analytical): {max_error:.2e}")

    # Test 2 - Harmonic oscillator
    # Verify conservation of energy: x² + v² = constant
    print("\n" + "=" * 50)
    print("Test 2: Harmonic Oscillator")
    print("=" * 50)

    z0 = torch.tensor([1.0, 0.0])  # x=1, v=0 (start at max displacement)
    t = torch.linspace(0, 10, 101)

    z_traj = odeint(harmonic_oscillator, z0, t, method="rk4")
    energy = z_traj[..., 0] ** 2 + z_traj[..., 1] ** 2  # x² + v²

    initial_energy = energy[0].item()
    energy_drift = torch.max(torch.abs(energy - initial_energy)).item()
    print(f"Initial condition: x=1, v=0")
    print(f"Initial energy (x² + v²): {initial_energy:.4f}")
    print(f"Max energy drift: {energy_drift:.2e}")

    # Test 3 - Compare Euler vs RK4
    # Error vs step size (RK4 should be much better)
    print("\n" + "=" * 50)
    print("Test 3: Euler vs RK4 Accuracy")
    print("=" * 50)

    z0 = torch.tensor([1.0])
    t_end = 2.0
    n_steps_list = [10, 20, 50, 100, 200]

    print(f"Problem: exponential decay, z0=1, t=[0, {t_end}]")
    print(f"Analytical solution at t={t_end}: {torch.exp(torch.tensor(-t_end)).item():.6f}")
    print(f"\n{'Steps':<10} {'Euler Error':<15} {'RK4 Error':<15}")
    print("-" * 40)

    for n_steps in n_steps_list:
        t = torch.linspace(0, t_end, n_steps + 1)
        z_analytical = z0 * torch.exp(torch.tensor(-t_end))

        z_euler = odeint(exponential_decay, z0, t, method="euler")[-1]
        z_rk4 = odeint(exponential_decay, z0, t, method="rk4")[-1]

        euler_err = torch.abs(z_euler - z_analytical).item()
        rk4_err = torch.abs(z_rk4 - z_analytical).item()

        print(f"{n_steps:<10} {euler_err:<15.2e} {rk4_err:<15.2e}")

    # Test 4 - Lotka-Volterra
    # Visualize predator-prey cycles
    print("\n" + "=" * 50)
    print("Test 4: Lotka-Volterra Predator-Prey")
    print("=" * 50)

    z0 = torch.tensor([10.0, 5.0])  # 10 prey, 5 predators
    t = torch.linspace(0, 50, 501)

    # lotka_volterra has extra params, so wrap it
    def lv(t, z):
        return lotka_volterra(t, z)

    z_traj = odeint(lv, z0, t, method="rk4")
    prey = z_traj[:, 0]
    predator = z_traj[:, 1]

    print(f"Initial: prey={z0[0].item():.0f}, predator={z0[1].item():.0f}")
    print(f"Prey range: [{prey.min().item():.1f}, {prey.max().item():.1f}]")
    print(f"Predator range: [{predator.min().item():.1f}, {predator.max().item():.1f}]")

    # ASCII visualization - sample every 25 steps
    print("\nPopulation over time (o=prey, x=predator):")
    print("-" * 62)
    width = 50
    prey_min, prey_max = prey.min().item(), prey.max().item()
    pred_min, pred_max = predator.min().item(), predator.max().item()
    scale_max = max(prey_max, pred_max)

    for i in range(0, len(t), 25):
        prey_pos = int((prey[i].item() / scale_max) * width)
        pred_pos = int((predator[i].item() / scale_max) * width)

        line = [" "] * (width + 1)
        line[prey_pos] = "o"
        line[pred_pos] = "x" if pred_pos != prey_pos else "*"

        print(f"t={t[i].item():5.1f} |{''.join(line)}|")


if __name__ == "__main__":
    main()

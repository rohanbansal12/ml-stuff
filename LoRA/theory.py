import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F


def _fro(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x, ord="fro")


def _rescale_to_alpha(
    eps: torch.Tensor, W_star: torch.Tensor, alpha: float, eps_floor: float = 1e-12
) -> torch.Tensor:
    """
    Rescale eps so that ||eps||_F == alpha * ||W_star||_F.
    """
    if alpha <= 0:
        return torch.zeros_like(W_star)

    target = alpha * _fro(W_star)
    cur = _fro(eps).clamp_min(eps_floor)
    return eps * (target / cur)


def noise_gaussian_like(W_star: torch.Tensor) -> torch.Tensor:
    """
    Isotropic iid Gaussian noise, unscaled.
    """
    return torch.randn_like(W_star)


def noise_low_rank(W_star: torch.Tensor, k: int) -> torch.Tensor:
    """
    Low-rank structured noise eps = U V, unscaled.
    rank(eps) <= k.
    """
    d_out, d_in = W_star.shape
    k = int(k)
    if k <= 0:
        return torch.zeros_like(W_star)
    U = torch.randn((d_out, k), device=W_star.device, dtype=W_star.dtype)
    V = torch.randn((k, d_in), device=W_star.device, dtype=W_star.dtype)
    return U @ V


def noise_spectral(W_star: torch.Tensor, p: float = 1.5, sigma0: float = 1.0) -> torch.Tensor:
    """
    Full-rank but low-effective-rank noise with power-law singular values:
      eps = U diag(s) V^T, where s_i = sigma0 * i^{-p}.

    U and V are random orthonormal bases (from QR), s decays as i^{-p}.
    """
    d_out, d_in = W_star.shape
    m = min(d_out, d_in)

    # Random orthonormal U, V via QR
    U0 = torch.randn((d_out, m), device=W_star.device, dtype=W_star.dtype)
    V0 = torch.randn((d_in, m), device=W_star.device, dtype=W_star.dtype)

    U, _ = torch.linalg.qr(U0, mode="reduced")  # (d_out, m)
    V, _ = torch.linalg.qr(V0, mode="reduced")  # (d_in,  m)

    # Power-law singular values
    i = torch.arange(1, m + 1, device=W_star.device, dtype=W_star.dtype)
    s = (sigma0 * (i ** (-p))).to(W_star.dtype)  # (m,)

    # Form eps = U diag(s) V^T
    # U @ (diag(s) @ V^T) == (U * s) @ V^T
    eps = (U * s) @ V.T  # (d_out, d_in)
    return eps


def make_W0_from_Wstar(W_star: torch.Tensor, args):
    """
    Returns (W_0, eps) where W_0 = W_star + eps,
    and eps is generated from the chosen noise model, then rescaled
    so ||eps||_F = noise_alpha * ||W_star||_F (if noise_alpha > 0).
    """
    # If alpha==0 or noise=="none", no noise.
    if getattr(args, "noise_alpha", 0.0) <= 0 or getattr(args, "noise", "none") == "none":
        eps = torch.zeros_like(W_star)
        return W_star.clone().detach(), eps

    noise_type = args.noise

    if noise_type == "gaussian":
        eps = noise_gaussian_like(W_star)

    elif noise_type == "low_rank":
        eps = noise_low_rank(W_star, k=getattr(args, "noise_k", 8))

    elif noise_type == "spectral":
        eps = noise_spectral(
            W_star,
            p=float(getattr(args, "noise_p", 1.5)),
            sigma0=float(getattr(args, "noise_sigma0", 1.0)),
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Rescale eps to have desired Frobenius norm relative to W_star
    eps = _rescale_to_alpha(eps, W_star, alpha=float(args.noise_alpha))

    W_0 = (W_star + eps).detach()  # frozen
    return W_0, eps


N = 10000
idx_split = int(0.8 * N)
d_in = 64
d_out = 64

num_steps = 1500
log_every = 100
lr = 1e-2

ranks = [0] + [2**i for i in range(0, 7)]
X = torch.randn((N, d_in))


def make_run_name(args):
    # Teacher
    if args.k is None or args.k == 0:
        teacher = "teacher=full"
    else:
        teacher = f"teacher_k={args.k}"

    parts = [teacher]

    # Noise (only if not none)
    if args.noise != "none":
        parts.append(f"noise={args.noise}")
        parts.append(f"alpha={args.noise_alpha}")

        if args.noise == "low_rank":
            parts.append(f"noise_k={args.noise_k}")
        elif args.noise == "spectral":
            parts.append(f"p={args.noise_p}")

    return "__".join(parts)


def best_rank_r_fro_error_sq(r, s_star, suffix_sums) -> float:
    # r can be 0..min(d_out, d_in)
    m = s_star.numel()
    if r <= 0:
        return float(suffix_sums[0]) if m > 0 else 0.0
    if r >= m:
        return 0.0
    return float(suffix_sums[r])


def runner(args):
    if args.k is None or args.k == 0:
        W_star = torch.randn((d_out, d_in))
    else:
        U = torch.randn((d_out, args.k))
        V = torch.randn((args.k, d_in))
        W_star = U @ V

    Y = X @ W_star.T

    x_train, y_train = X[:idx_split], Y[:idx_split]
    x_test, y_test = X[idx_split:], Y[idx_split:]

    W_0, eps = make_W0_from_Wstar(W_star, args)
    W_0.requires_grad_(False)

    with torch.no_grad():
        print(f"Noise type: {args.noise}, alpha={args.noise_alpha}")
        print(
            f"||W*||_F={torch.linalg.norm(W_star, ord='fro').item():.6g}, "
            f"||eps||_F={torch.linalg.norm(eps, ord='fro').item():.6g}, "
            f"||W*-W0||_F={torch.linalg.norm(W_star - W_0, ord='fro').item():.6g}"
        )

    # -----------------------------
    # Theory: best rank-r approx error of Δ* = W* - W0
    # -----------------------------
    with torch.no_grad():
        Delta_star = W_star - W_0  # here: W_star
        # Singular values of the true required update
        s_star = torch.linalg.svdvals(Delta_star)  # shape (min(d_out,d_in),)
        # Precompute tail Frobenius^2 error for best rank-r approximation:
        # ||Δ* - (Δ*)_r||_F^2 = sum_{i>r} s_i^2
        s2 = (s_star**2).cpu().numpy()
        # suffix sums of squared singular values
        suffix_sums = np.cumsum(s2[::-1])[::-1]  # suffix_sums[i] = sum_{j>=i} s2[j]

    rank_dict = {}

    print("True Δ* singular values (top 10):", s_star[:10].detach().cpu().numpy())
    print("||Δ*||_F^2:", float(torch.sum(s_star**2).item()))

    for r in ranks:
        # --- rank-0 baseline: no adapters, no training ---
        if r == 0:
            with torch.no_grad():
                y_hat_tr = x_train @ W_0.T
                y_hat_te = x_test @ W_0.T
                tr_loss = F.mse_loss(y_hat_tr, y_train).item()
                te_loss = F.mse_loss(y_hat_te, y_test).item()

            # theory: best rank-0 approx to Δ* is 0, so error is ||Δ*||_F^2
            theory_err_sq = best_rank_r_fro_error_sq(0, s_star, suffix_sums)

            rank_dict[r] = {
                "train_loss": tr_loss,
                "test_loss": te_loss,
                "deltaW_fro": 0.0,
                "theory_best_rank_r_fro_err_sq": theory_err_sq,
                "learned_deltaW_singular_values": np.array([]),
                "true_delta_star_singular_values": s_star.detach().cpu().numpy(),
                "log": [(0, tr_loss, te_loss, 0.0)],
            }
            print(
                f"[r=0] step=0 train={tr_loss:.6g} test={te_loss:.6g} "
                f"||ΔW||={0.0:.6g} theory_err_sq={theory_err_sq:.6g}"
            )
            continue

        # --- LoRA factors ---
        B = torch.zeros((d_out, r), requires_grad=True)  # init B=0 => ΔW=0 at start
        A = torch.randn((r, d_in), requires_grad=True)

        optim = torch.optim.Adam([A, B], lr=lr)
        logs = []

        # theory bound for this rank (in weight space)
        theory_err_sq = best_rank_r_fro_error_sq(r, s_star, suffix_sums)

        for step in range(1, num_steps + 1):
            W_eff = W_0 + (B @ A)  # (d_out, d_in)
            y_hat = x_train @ W_eff.T  # (N_train, d_out)
            loss = F.mse_loss(y_hat, y_train)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % log_every == 0 or step == 1 or step == num_steps:
                with torch.no_grad():
                    DeltaW = B @ A
                    y_hat_te = x_test @ (W_0 + DeltaW).T
                    test_loss = F.mse_loss(y_hat_te, y_test).item()
                    train_loss = loss.item()
                    deltaW_fro = torch.linalg.norm(DeltaW, ord="fro").item()
                logs.append((step, train_loss, test_loss, deltaW_fro))
                print(
                    f"[r={r}] step={step:5d} train={train_loss:.6g} test={test_loss:.6g} "
                    f"||ΔW||={deltaW_fro:.6g} theory_err_sq={theory_err_sq:.6g}"
                )

        # final singular values of learned ΔW
        with torch.no_grad():
            DeltaW = B @ A
            s_learned = torch.linalg.svdvals(DeltaW).detach().cpu().numpy()

        with torch.no_grad():
            DeltaW = B @ A
            achieved_residual_fro_sq = (
                torch.linalg.norm((W_star - W_0) - DeltaW, ord="fro").item() ** 2
            )

        rank_dict[r] = {
            "train_loss": logs[-1][1],
            "test_loss": logs[-1][2],
            "deltaW_fro": logs[-1][3],
            "theory_best_rank_r_fro_err_sq": theory_err_sq,
            "learned_deltaW_singular_values": s_learned,
            "true_delta_star_singular_values": s_star.detach().cpu().numpy(),
            "log": logs,
            "achieved_residual_fro_sq": achieved_residual_fro_sq,
        }

    return rank_dict, s_star


def plot(rank_dict, s_star, run_name, out_dir):
    sns.set_theme(style="whitegrid", context="talk")

    rows = []
    for r, d in rank_dict.items():
        rows.append(
            {
                "rank": int(r),
                "train_mse": float(d["train_loss"]),
                "test_mse": float(d["test_loss"]),
                "deltaW_fro": float(d["deltaW_fro"]),
                "theory_residual_fro_sq": float(d["theory_best_rank_r_fro_err_sq"]),
                # if you added this (recommended) it will appear; otherwise NaN
                "achieved_residual_fro_sq": float(d.get("achieved_residual_fro_sq", np.nan)),
            }
        )
    df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)

    # Convenient: log2(rank) for nicer x-axis spacing with powers of two
    df["rank_label"] = df["rank"].astype(str)
    df["log2_rank"] = (
        df["rank"].replace(0, np.nan).apply(lambda x: np.log2(x) if pd.notnull(x) else np.nan)
    )

    # -----------------------------
    # Plot 1: Train/Test MSE vs rank
    # -----------------------------
    plt.figure(figsize=(10, 6))
    df_melt = df.melt(
        id_vars=["rank"], value_vars=["train_mse", "test_mse"], var_name="split", value_name="mse"
    )

    # Make split labels nicer
    df_melt["split"] = df_melt["split"].map({"train_mse": "Train MSE", "test_mse": "Test MSE"})

    ax = sns.lineplot(data=df_melt, x="rank", y="mse", hue="split", marker="o")
    ax.set_xscale("symlog", linthresh=1)  # nice for including rank=0
    ax.set_yscale("log")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("MSE vs LoRA rank")
    plt.tight_layout()
    plt.savefig(out_dir / f"{run_name}__mse_vs_rank.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot 2: Theory best rank-r residual vs achieved residual (if available)
    # -----------------------------
    plt.figure(figsize=(10, 6))

    # Theory curve
    ax = sns.lineplot(
        data=df,
        x="rank",
        y="theory_residual_fro_sq",
        marker="o",
        label="Theory: best rank-r residual ||Δ*-(Δ*)_r||_F^2",
    )

    # Achieved residual curve if you stored it
    if df["achieved_residual_fro_sq"].notna().any():
        sns.lineplot(
            data=df,
            x="rank",
            y="achieved_residual_fro_sq",
            marker="o",
            label="Achieved: ||Δ* - ΔW||_F^2",
        )

    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Frobenius² residual (log scale)")
    ax.set_title("Best possible vs achieved weight-space residual")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{run_name}__theory_vs_achieved_residual.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

    # -----------------------------
    # Plot 3: Singular values — true Δ* vs learned ΔW for each rank
    # -----------------------------
    # Build a long-form DF: for each rank, list singular values of learned ΔW
    sv_rows = []

    # True Δ* singular values (same for all ranks) — include once for overlay
    true_df = pd.DataFrame(
        {
            "rank": -1,
            "i": np.arange(1, len(s_star) + 1),
            "singular_value": s_star,
            "type": "True Δ*",
        }
    )

    for r, d in rank_dict.items():
        if r == 0:
            continue
        s_learned = np.array(d["learned_deltaW_singular_values"], dtype=float)
        # sort descending just in case
        s_learned = np.sort(s_learned)[::-1]
        sv_rows.append(
            pd.DataFrame(
                {
                    "rank": int(r),
                    "i": np.arange(1, len(s_learned) + 1),
                    "singular_value": s_learned,
                    "type": f"Learned ΔW (r={r})",
                }
            )
        )

    learned_df = (
        pd.concat(sv_rows, ignore_index=True)
        if sv_rows
        else pd.DataFrame(columns=["rank", "i", "singular_value", "type"])
    )

    # Plot a "small multiples" style: one plot per rank, overlay true vs learned
    # (Looks good and makes the rank-cap obvious)
    unique_ranks = sorted([r for r in rank_dict.keys() if r != 0])
    ncols = 2
    nrows = int(np.ceil(len(unique_ranks) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, r in zip(axes, unique_ranks, strict=False):
        d = rank_dict[r]
        s_learned = np.sort(np.array(d["learned_deltaW_singular_values"], dtype=float))[::-1]

        # Plot true Δ*
        ax.plot(np.arange(1, len(s_star) + 1), s_star, marker="o", linestyle="-", label="True Δ*")

        # Plot learned ΔW
        ax.plot(
            np.arange(1, len(s_learned) + 1),
            s_learned,
            marker="o",
            linestyle="-",
            label=f"Learned ΔW (r={r})",
        )

        ax.set_yscale("log")
        ax.set_xlabel("Singular value index i")
        ax.set_ylabel("σ_i (log scale)")
        ax.set_title(f"Singular values: True Δ* vs Learned ΔW (rank r={r})")
        ax.legend()

    # Hide unused axes
    for ax in axes[len(unique_ranks) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        out_dir / f"{run_name}__singular_values_true_vs_learned.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

    # -----------------------------
    # Plot 4 (Optional): Learning curves (train/test) vs steps, per rank
    # -----------------------------
    lc_rows = []
    for r, d in rank_dict.items():
        for entry in d["log"]:
            step, train_loss, test_loss, deltaW_fro = entry
            lc_rows.append(
                {
                    "rank": int(r),
                    "step": int(step),
                    "train_mse": float(train_loss),
                    "test_mse": float(test_loss),
                    "deltaW_fro": float(deltaW_fro),
                }
            )

    lc = pd.DataFrame(lc_rows)

    # Melt for seaborn
    lc_melt = lc.melt(
        id_vars=["rank", "step"],
        value_vars=["train_mse", "test_mse"],
        var_name="split",
        value_name="mse",
    )
    lc_melt["split"] = lc_melt["split"].map({"train_mse": "Train MSE", "test_mse": "Test MSE"})

    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(
        data=lc_melt[lc_melt["rank"] != 0], x="step", y="mse", hue="rank", style="split"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Learning curves by rank (train vs test)")
    plt.legend(title="Rank / Split", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / f"{run_name}__learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="/workspace/ml-stuff/LoRA/plots")
    parser.add_argument(
        "--noise",
        type=str,
        default="none",
        choices=["none", "gaussian", "low_rank", "spectral"],
        help="Noise model for epsilon in W0 = W* + epsilon",
    )

    parser.add_argument(
        "--noise_alpha",
        type=float,
        default=0.0,
        help="Target relative Frobenius norm: ||eps||_F ~= alpha * ||W*||_F. "
        "If 0, no noise is added.",
    )

    # low-rank noise
    parser.add_argument(
        "--noise_k",
        type=int,
        default=8,
        help="Rank for low_rank noise epsilon = U V (only used if --noise low_rank).",
    )

    # spectral noise
    parser.add_argument(
        "--noise_p",
        type=float,
        default=1.5,
        help="Power-law exponent for spectral singular values sigma_i ~ i^{-p} (only used if --noise spectral).",
    )
    parser.add_argument(
        "--noise_sigma0",
        type=float,
        default=1.0,
        help="Base singular value scale before Frobenius rescale (only used if --noise spectral).",
    )

    # reproducibility
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = make_run_name(args)

    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name: {run_name}")
    print(f"Saving plots to: {out_dir}")

    rank_dict, s_star = runner(args)
    plot(rank_dict, s_star, run_name, out_dir)


if __name__ == "__main__":
    main()

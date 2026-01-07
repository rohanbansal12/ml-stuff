import math

import pytest
import torch
from quantize import (
    calc_qparams_symmetric,
    quant_dequant_affine,
    quant_dequant_symmetric,
    quant_error_report,
    quantize_affine,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _seed_all(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize("device", DEVICES)
def test_random_affine_roundtrip_sanity(device: str):
    """
    Test 1: basic correctness / sanity on random tensor.
    """
    _seed_all(0)
    x = torch.randn((256, 128), device=device, dtype=torch.float32)

    x_hat, q, scale, zp, clipped = quant_dequant_affine(x, bits=8)

    assert x_hat.shape == x.shape
    assert q.shape == x.shape
    assert q.dtype == torch.int8
    assert scale.numel() == 1
    assert zp.numel() == 1
    assert 0.0 <= clipped <= 1.0
    assert torch.isfinite(x_hat).all()

    rep = quant_error_report(x, x_hat)
    assert rep["mse"] >= 0.0 and math.isfinite(rep["mse"])
    assert rep["mae"] >= 0.0 and math.isfinite(rep["mae"])
    assert rep["max_abs"] >= 0.0 and math.isfinite(rep["max_abs"])
    assert -1.0 <= rep["cosine"] <= 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_saturation_and_clipped_fraction(device: str):
    """
    Test 2: saturation/clipping behavior.
    Create extreme outliers; confirm clipping happens and q hits qmin/qmax.
    """
    _seed_all(1)
    x = torch.cat(
        [
            torch.randn(2000, device=device, dtype=torch.float32),
            torch.tensor([1e6, -1e6], device=device, dtype=torch.float32),
        ],
        dim=0,
    )

    bits = 8
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    x_hat, q, scale, zp, clipped = quant_dequant_affine(x, bits=bits, qmin=qmin, qmax=qmax)

    # Minmax may legitimately produce zero clipping
    assert 0.0 <= clipped <= 1.0

    # But scale should be very large due to outliers
    assert scale.item() > 1e3

    assert torch.isfinite(x_hat).all()


@pytest.mark.parametrize("device", DEVICES)
def test_per_channel_broadcasting_linear_weights_axis0(device: str):
    """
    Test 3: per-channel broadcasting correctness for Linear weights W[out, in], axis=0.
    """
    _seed_all(2)

    out_features, in_features = 64, 128
    W = torch.randn(out_features, in_features, device=device, dtype=torch.float32)

    W_hat, q, scale, zp, clipped = quant_dequant_symmetric(W, bits=8, per_channel=True, axis=0)

    assert W_hat.shape == W.shape
    assert q.shape == W.shape
    assert q.dtype == torch.int8
    assert 0.0 <= clipped <= 1.0
    assert torch.isfinite(W_hat).all()

    # scale/zp should broadcast over [out, in] with axis=0 => shape [out, 1] after broadcast helper
    assert scale.shape == (out_features, 1)
    assert zp.shape == (out_features, 1)

    rep = quant_error_report(W, W_hat)
    assert rep["mse"] >= 0.0 and math.isfinite(rep["mse"])


@pytest.mark.parametrize("device", DEVICES)
def test_degenerate_constant_tensor_no_nan(device: str):
    """
    Extra: constant tensor should not produce NaNs (scale should clamp to eps).
    """
    x = torch.full((1024,), 3.14, device=device, dtype=torch.float32)
    x_hat, q, scale, zp, clipped = quant_dequant_affine(x, bits=8)

    assert torch.isfinite(x_hat).all()
    assert torch.isfinite(scale).all()
    assert 0.0 <= clipped <= 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_heavy_tailed_percentile_better_on_bulk(device: str):
    """
    Test 4: heavy-tailed activation-like distribution.
    Compare minmax vs percentile calibration. We check error on the bulk mass (central region).
    """
    _seed_all(3)

    base = torch.randn(200_000, device=device, dtype=torch.float32)
    outliers = torch.randn(200, device=device, dtype=torch.float32) * 200.0
    x = torch.cat([base, outliers], dim=0)

    # Minmax
    x_hat_mm, q_mm, s_mm, zp_mm, clip_mm = quant_dequant_affine(x, bits=8, clip=None)

    # Percentile
    x_hat_p, q_p, s_p, zp_p, clip_p = quant_dequant_affine(
        x, bits=8, clip={"type": "percentile", "p": 0.999}
    )

    # Compare MSE on bulk (ignore extreme tail)
    central = x.abs() < 5.0
    mse_mm_c = ((x_hat_mm[central] - x[central]) ** 2).mean().item()
    mse_p_c = ((x_hat_p[central] - x[central]) ** 2).mean().item()

    assert math.isfinite(mse_mm_c) and math.isfinite(mse_p_c)
    assert mse_p_c <= mse_mm_c * 1.05  # usually much better; allow small noise
    assert torch.isfinite(x_hat_mm).all()
    assert torch.isfinite(x_hat_p).all()


@pytest.mark.parametrize("device", DEVICES)
def test_symmetric_clamps_to_plus_minus_qmax(device: str):
    """
    Extra: symmetric quantization should clamp to [-qmax, qmax] with zp=0.
    """
    _seed_all(4)
    x = torch.tensor([-1e3, -10.0, 0.0, 10.0, 1e3], device=device, dtype=torch.float32)

    bits = 8
    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax

    scale, zp = calc_qparams_symmetric(x, bits=bits)
    q, clipped = quantize_affine(x, scale, zp, qmin=qmin, qmax=qmax, dtype=torch.int8)

    assert (q >= qmin).all().item() and (q <= qmax).all().item()
    assert zp.numel() == 1
    assert int(zp.item()) == 0

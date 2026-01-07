import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantized_linear import QuantLinearW8A8, quantize_matmul1

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES_BY_DEVICE = {
    "cpu": [torch.float32],
    "cuda": [torch.float16, torch.float32],
}


def _seed_all(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-8)).item()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantize_matmul1_matches_flinear(device: str, dtype: torch.dtype):
    if dtype not in DTYPES_BY_DEVICE[device]:
        pytest.skip(f"{device} does not test dtype={dtype}")

    _seed_all(0)
    B, in_f, out_f = 64, 256, 128
    x = torch.randn(B, in_f, device=device, dtype=dtype)
    W = torch.randn(out_f, in_f, device=device, dtype=dtype)
    bias = torch.randn(out_f, device=device, dtype=dtype)

    y_ref = F.linear(x, W, bias).float()
    y_q, x_clip, w_clip = quantize_matmul1(x, W, bias)
    y_q = y_q.float()

    assert y_q.shape == y_ref.shape
    assert torch.isfinite(y_q).all().item()
    assert 0.0 <= x_clip <= 1.0
    assert 0.0 <= w_clip <= 1.0

    cos = _cos(y_q, y_ref)
    re = _rel_err(y_q, y_ref)

    # generous thresholds for random data
    assert cos > 0.995, f"cos too low: {cos}"
    assert re < 0.05, f"rel_err too high: {re}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantlinear_from_float_matches_linear_2d(device: str, dtype: torch.dtype):
    if dtype not in DTYPES_BY_DEVICE[device]:
        pytest.skip(f"{device} does not test dtype={dtype}")

    _seed_all(1)
    B, in_f, out_f = 64, 256, 128
    x = torch.randn(B, in_f, device=device, dtype=dtype)

    layer = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=dtype)
    y_ref = layer(x).float()

    q_layer = QuantLinearW8A8.from_float(layer).to(device=device)
    y_q = q_layer(x).float()

    assert y_q.shape == y_ref.shape
    assert torch.isfinite(y_q).all().item()

    cos = _cos(y_q, y_ref)
    re = _rel_err(y_q, y_ref)
    assert cos > 0.995, f"cos too low: {cos}"
    assert re < 0.05, f"rel_err too high: {re}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantlinear_supports_3d_inputs(device: str, dtype: torch.dtype):
    if dtype not in DTYPES_BY_DEVICE[device]:
        pytest.skip(f"{device} does not test dtype={dtype}")

    _seed_all(2)
    B, T, in_f, out_f = 16, 32, 128, 64
    x = torch.randn(B, T, in_f, device=device, dtype=dtype)

    layer = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=dtype)
    y_ref = layer(x).float()

    q_layer = QuantLinearW8A8.from_float(layer).to(device=device)
    y_q = q_layer(x).float()

    assert y_q.shape == y_ref.shape
    assert torch.isfinite(y_q).all().item()

    cos = _cos(y_q, y_ref)
    assert cos > 0.995, f"cos too low: {cos}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantlinear_bias_none(device: str, dtype: torch.dtype):
    if dtype not in DTYPES_BY_DEVICE[device]:
        pytest.skip(f"{device} does not test dtype={dtype}")

    _seed_all(3)
    B, in_f, out_f = 64, 256, 128
    x = torch.randn(B, in_f, device=device, dtype=dtype)

    layer = nn.Linear(in_f, out_f, bias=False).to(device=device, dtype=dtype)
    y_ref = layer(x).float()

    q_layer = QuantLinearW8A8.from_float(layer).to(device=device)
    y_q = q_layer(x).float()

    assert y_q.shape == y_ref.shape
    assert torch.isfinite(y_q).all().item()

    cos = _cos(y_q, y_ref)
    re = _rel_err(y_q, y_ref)
    assert cos > 0.995, f"cos too low: {cos}"
    assert re < 0.05, f"rel_err too high: {re}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantlinear_return_stats(device: str, dtype: torch.dtype):
    if dtype not in DTYPES_BY_DEVICE[device]:
        pytest.skip(f"{device} does not test dtype={dtype}")

    _seed_all(4)
    B, in_f, out_f = 32, 128, 64
    x = torch.randn(B, in_f, device=device, dtype=dtype)

    layer = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=dtype)
    q_layer = QuantLinearW8A8.from_float(layer).to(device=device)

    y, x_clip = q_layer(x, return_stats=True)
    assert y.shape == (B, out_f)
    assert torch.isfinite(y).all().item()
    assert isinstance(x_clip, float)
    assert 0.0 <= x_clip <= 1.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_buffers_and_shapes(device: str, dtype: torch.dtype):
    if dtype not in DTYPES_BY_DEVICE[device]:
        pytest.skip(f"{device} does not test dtype={dtype}")

    _seed_all(5)
    in_f, out_f = 32, 16
    layer = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=dtype)
    q_layer = QuantLinearW8A8.from_float(layer).to(device=device)

    assert q_layer.w_q.dtype == torch.int8
    assert q_layer.s_w.dtype == torch.float32
    assert q_layer.w_q.shape == (out_f, in_f)
    # With your broadcast helper for axis=0, s_w should be [out, 1]
    assert q_layer.s_w.shape == (out_f, 1)

    # Buffers should be on the right device after .to(device)
    assert q_layer.w_q.device.type == device
    assert q_layer.s_w.device.type == device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_state_dict_roundtrip(device: str, dtype: torch.dtype):
    """
    Save/load QuantLinearW8A8 via state_dict and ensure outputs match.
    Critical: create destination module with in/out features so buffers are allocated.
    """
    if dtype not in DTYPES_BY_DEVICE[device]:
        pytest.skip(f"{device} does not test dtype={dtype}")

    _seed_all(6)
    B, in_f, out_f = 16, 64, 32
    x = torch.randn(B, in_f, device=device, dtype=dtype)

    layer = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=dtype)
    q1 = QuantLinearW8A8.from_float(layer).to(device=device)
    y1 = q1(x).float()

    buf = io.BytesIO()
    torch.save(q1.state_dict(), buf)
    buf.seek(0)

    # Destination module: allocate buffers by passing dims
    q2 = QuantLinearW8A8(act_clip=q1.act_clip, in_features=in_f, out_features=out_f).to(
        device=device
    )
    q2.load_state_dict(torch.load(buf, map_location=device))

    # bias is a Parameter, so load_state_dict will restore it automatically
    y2 = q2(x).float()

    assert y1.shape == y2.shape
    assert torch.isfinite(y2).all().item()

    # allow tiny numeric noise; should usually be exact
    max_diff = (y1 - y2).abs().max().item()
    assert max_diff < 1e-6, f"roundtrip max_diff too high: {max_diff}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_act_clip_config_is_used(device: str, dtype: torch.dtype):
    """
    Ensure act_clip changes behavior on heavy-tailed activations.
    We don't assert 'better', only that outputs differ.
    """
    if dtype not in DTYPES_BY_DEVICE[device]:
        pytest.skip(f"{device} does not test dtype={dtype}")

    _seed_all(7)
    B, in_f, out_f = 32, 128, 64
    x = torch.randn(B, in_f, device=device, dtype=dtype)
    x[0, 0] = torch.tensor(1000.0, device=device, dtype=dtype)

    layer = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=dtype)

    q_pct = QuantLinearW8A8.from_float(layer, act_clip={"type": "percentile", "p": 0.999}).to(
        device=device
    )
    q_mm = QuantLinearW8A8.from_float(layer, act_clip={"type": "minmax"}).to(device=device)

    y_pct = q_pct(x).float()
    y_mm = q_mm(x).float()

    diff = (y_pct - y_mm).abs().mean().item()
    assert diff > 0.0
    assert torch.isfinite(y_pct).all().item()
    assert torch.isfinite(y_mm).all().item()

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict


def _reduce_minmax(
    x: torch.Tensor,
    per_channel: bool,
    axis: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns xmin, xmax. If per_channel=True, reduces across all dims except `axis`.
    Shapes:
      - per_channel=False -> scalars
      - per_channel=True  -> shape like x.shape[axis]
    """
    if not per_channel:
        return x.amin(), x.amax()

    if axis is None:
        raise ValueError("axis must be provided when per_channel=True")

    axis = axis % x.ndim
    reduce_dims = [d for d in range(x.ndim) if d != axis]
    xmin = x.amin(dim=reduce_dims)
    xmax = x.amax(dim=reduce_dims)
    return xmin, xmax


def _reduce_percentile(
    x: torch.Tensor,
    p: float,
    per_channel: bool,
    axis: Optional[int],
    symmetric: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Percentile-based range. For asymmetric: [q_low, q_high] = [1-p, p].
    For symmetric: use max(|q|) based on percentile of abs(x).
    """
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1], e.g. 0.999 for 99.9%")

    if not per_channel:
        if symmetric:
            a = torch.quantile(x.abs().flatten(), p)
            return -a, a
        lo = torch.quantile(x.flatten(), 1.0 - p)
        hi = torch.quantile(x.flatten(), p)
        return lo, hi

    if axis is None:
        raise ValueError("axis must be provided when per_channel=True")

    axis = axis % x.ndim
    # Move channel axis to front, flatten rest
    y = x.movedim(axis, 0).reshape(x.shape[axis], -1)  # [C, N]

    if symmetric:
        a = torch.quantile(y.abs(), p, dim=1)  # [C]
        return -a, a

    lo = torch.quantile(y, 1.0 - p, dim=1)  # [C]
    hi = torch.quantile(y, p, dim=1)  # [C]
    return lo, hi


def _maybe_clip_range(
    x: torch.Tensor,
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    clip: Optional[Dict[str, Union[str, float]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    clip can be:
      - None
      - {"type": "minmax"}  (no-op; included for uniformity)
      - {"type": "percentile", "p": 0.999}  (handled elsewhere; not here)
      - {"type": "range", "xmin": ..., "xmax": ...} override range
    """
    if clip is None:
        return xmin, xmax

    ctype = clip.get("type", None)
    if ctype in (None, "minmax", "percentile"):
        return xmin, xmax

    if ctype == "range":
        return (
            torch.as_tensor(clip["xmin"], device=x.device, dtype=x.dtype),
            torch.as_tensor(clip["xmax"], device=x.device, dtype=x.dtype),
        )

    raise ValueError(f"Unknown clip type: {ctype}")


def _broadcast_to_axis_param(
    param: torch.Tensor,
    x: torch.Tensor,
    per_channel: bool,
    axis: Optional[int],
) -> torch.Tensor:
    """
    Reshape param to broadcast over x. If per_channel, param is shape [C] and is
    reshaped to [1,1,...,C,...,1].
    """
    if not per_channel:
        return param

    if axis is None:
        raise ValueError("axis must be provided when per_channel=True")

    axis = axis % x.ndim
    shape = [1] * x.ndim
    shape[axis] = x.shape[axis]
    return param.reshape(shape)


def calc_qparams_affine(
    x: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
    axis: Optional[int] = None,
    qmin: Optional[int] = None,
    qmax: Optional[int] = None,
    clip: Optional[Dict[str, Union[str, float]]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Affine (asymmetric) quantization params for signed integers by default.
      q = round(x / scale) + zero_point
      x_hat = (q - zero_point) * scale

    Returns:
      scale (broadcastable to x),
      zero_point (broadcastable to x, int64)

    clip options:
      - None (min/max)
      - {"type":"percentile", "p":0.999}
      - {"type":"range", "xmin":..., "xmax":...}
    """
    if bits < 2 or bits > 16:
        raise ValueError("bits should typically be in [2,16]")

    # Default to signed integer range
    if qmin is None or qmax is None:
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1

    if clip is not None and clip.get("type") == "percentile":
        p = float(clip.get("p", 0.999))
        xmin, xmax = _reduce_percentile(x, p, per_channel, axis, symmetric=False)
    else:
        xmin, xmax = _reduce_minmax(x, per_channel, axis)

    xmin, xmax = _maybe_clip_range(x, xmin, xmax, clip)

    # Ensure xmin <= xmax
    xmin = torch.minimum(xmin, xmax)
    xmax = torch.maximum(xmax, xmin)

    # scale = (xmax - xmin) / (qmax - qmin)
    denom = float(qmax - qmin)
    scale = (xmax - xmin) / denom
    scale = torch.clamp(scale, min=eps)

    # zero_point = round(qmin - xmin / scale)
    zp = torch.round(qmin - xmin / scale).to(torch.int64)
    zp = torch.clamp(zp, qmin, qmax)

    scale = _broadcast_to_axis_param(scale, x, per_channel, axis)
    zp = _broadcast_to_axis_param(zp, x, per_channel, axis)

    return scale, zp


def calc_qparams_symmetric(
    x: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
    axis: Optional[int] = None,
    qmax: Optional[int] = None,
    clip: Optional[Dict[str, Union[str, float]]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization params (zero_point = 0).
    Typical signed symmetric int8 uses range [-127, 127] (not -128) to keep symmetry.

      q = clamp(round(x / scale), -qmax, qmax)
      x_hat = q * scale

    Returns:
      scale (broadcastable to x),
      zero_point (broadcastable to x, int64) always 0

    clip options:
      - None (use max(abs(x)))
      - {"type":"percentile", "p":0.999}  (use percentile of abs(x))
      - {"type":"range", "xmin":..., "xmax":...} (converted to symmetric via max(|xmin|,|xmax|))
    """
    if bits < 2 or bits > 16:
        raise ValueError("bits should typically be in [2,16]")

    if qmax is None:
        # Symmetric signed uses [-qmax, qmax]
        qmax = (2 ** (bits - 1)) - 1  # e.g., 127 for int8

    if clip is not None and clip.get("type") == "percentile":
        p = float(clip.get("p", 0.999))
        xmin, xmax = _reduce_percentile(x, p, per_channel, axis, symmetric=True)
        amax = torch.maximum(xmin.abs(), xmax.abs())
    else:
        xmin, xmax = _reduce_minmax(x, per_channel, axis)
        # Symmetric range based on max abs
        amax = torch.maximum(xmin.abs(), xmax.abs())

    if clip is not None and clip.get("type") == "range":
        xmin_o, xmax_o = _maybe_clip_range(x, xmin, xmax, clip)
        amax = torch.maximum(xmin_o.abs(), xmax_o.abs())

    scale = amax / float(qmax)
    scale = torch.clamp(scale, min=eps)

    zp = torch.zeros_like(scale, dtype=torch.int64)

    scale = _broadcast_to_axis_param(scale, x, per_channel, axis)
    zp = _broadcast_to_axis_param(zp, x, per_channel, axis)

    return scale, zp


def quantize_affine(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
    dtype=torch.int8,
) -> Tuple[torch.Tensor, float]:
    q = torch.round(x / scale) + zero_point.to(x.dtype)
    # count how many will clip (before clamp)
    clipped = (q < qmin) | (q > qmax)
    clipped_frac = clipped.to(torch.float32).mean().item()
    q = torch.clamp(q, qmin, qmax)
    return q.to(dtype), clipped_frac


def dequantize_affine(
    q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    qi = q.to(torch.int32)
    zpi = zero_point.to(torch.int32)
    return (qi - zpi).to(scale.dtype) * scale


def quant_dequant_affine(
    x: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
    axis: Optional[int] = None,
    qmin: Optional[int] = None,
    qmax: Optional[int] = None,
    clip: Optional[Dict[str, Union[str, float]]] = None,
    eps: float = 1e-8,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    # defaults consistent with calc_qparams_affine
    if qmin is None or qmax is None:
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1

    scale, zp = calc_qparams_affine(
        x, bits=bits, per_channel=per_channel, axis=axis,
        qmin=qmin, qmax=qmax, clip=clip, eps=eps
    )
    q, clipped_frac = quantize_affine(x, scale, zp, qmin=qmin, qmax=qmax, dtype=dtype)
    x_hat = dequantize_affine(q, scale, zp)
    return x_hat, q, scale, zp, clipped_frac


def quant_dequant_symmetric(
    x: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
    axis: Optional[int] = None,
    qmax: Optional[int] = None,
    clip: Optional[Dict[str, Union[str, float]]] = None,
    eps: float = 1e-8,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if qmax is None:
        # symmetric signed uses [-qmax, qmax], commonly qmax=127 for int8
        qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax

    scale, zp = calc_qparams_symmetric(
        x, bits=bits, per_channel=per_channel, axis=axis,
        qmax=qmax, clip=clip, eps=eps
    )
    # zp is zero in symmetric, but reuse affine code
    q, clipped_frac = quantize_affine(x, scale, zp, qmin=qmin, qmax=qmax, dtype=dtype)
    x_hat = dequantize_affine(q, scale, zp)
    return x_hat, q, scale, zp, clipped_frac

def quant_error_report(x: torch.Tensor, x_hat: torch.Tensor) -> Dict[str, float]:
    diff = x_hat - x
    cos = F.cosine_similarity(x.flatten(), x_hat.flatten(), dim=0).item()
    return {
        "mse": diff.square().mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "cosine": cos,
    }
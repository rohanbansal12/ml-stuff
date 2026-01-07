import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize import (
    calc_qparams_affine,
    calc_qparams_symmetric,
    quantize_affine,
    quantize_symmetric,
)


def quantize_matmul1(x, W, bias=None):
    s_x, zp_x = calc_qparams_affine(
        x, bits=8, per_channel=False, axis=None, clip={"type": "percentile", "p": 0.999}
    )
    x_q, x_clip = quantize_affine(x, s_x, zp_x, qmin=-128, qmax=127, dtype=torch.int8)

    s_w, zp_w = calc_qparams_symmetric(W, bits=8, per_channel=True, axis=0)
    w_q, w_clip = quantize_symmetric(W, s_w, qmax=127, dtype=torch.int8)

    # GPU-friendly "integer GEMM":
    x_centered = x_q.to(torch.float32) - zp_x.to(torch.float32)
    w_f = w_q.to(torch.float32)
    y_int = x_centered @ w_f.T

    # rescale
    y = y_int * (s_x.to(torch.float32) * s_w.to(torch.float32).T)

    if bias is not None:
        y = y + bias.to(y.dtype)

    return y, x_clip, w_clip


# expects these to exist in your module scope
# from quantize import calc_qparams_affine, calc_qparams_symmetric, quantize_affine, quantize_symmetric


class QuantLinearW8A8(nn.Module):
    def __init__(
        self,
        act_clip: dict[str, str | float] | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
        has_bias: bool | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ---- store act_clip in buffers so it roundtrips via state_dict ----
        # act_clip types supported: {"type":"percentile","p":...} or {"type":"minmax"} or None
        if act_clip is None:
            act_clip = {"type": "percentile", "p": 0.999}

        clip_type = act_clip.get("type", "percentile") if act_clip is not None else "percentile"
        if clip_type == "percentile":
            clip_type_id = 1
            p = float(act_clip.get("p", 0.999))
        elif clip_type == "minmax" or clip_type is None:
            clip_type_id = 0
            p = 0.0
        else:
            raise ValueError(f"Unsupported act_clip type for serialization: {clip_type}")

        self.register_buffer(
            "act_clip_type", torch.tensor(clip_type_id, dtype=torch.uint8), persistent=True
        )
        self.register_buffer("act_clip_p", torch.tensor(p, dtype=torch.float32), persistent=True)

        # ---- buffers for quantized weights/scales ----
        self.register_buffer("w_q", torch.empty(0, dtype=torch.int8), persistent=True)
        self.register_buffer("s_w", torch.empty(0, dtype=torch.float32), persistent=True)

        # ---- has_bias as a persistent buffer (so it roundtrips) ----
        hb = int(bool(has_bias)) if has_bias is not None else 0
        self.register_buffer("has_bias", torch.tensor(hb, dtype=torch.uint8), persistent=True)

        # ---- bias parameter (always registered if out_features known) ----
        if in_features is not None and out_features is not None:
            self._allocate_buffers(out_features, in_features)
            # Always register a bias Parameter for state_dict compatibility.
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            # bias will be allocated later (or remain None)
            self.bias = None

    @property
    def act_clip(self) -> dict[str, str | float] | None:
        """Reconstruct clip dict from buffers (state_dict-backed)."""
        t = int(self.act_clip_type.item())
        if t == 0:
            return {"type": "minmax"}
        if t == 1:
            return {"type": "percentile", "p": float(self.act_clip_p.item())}
        # Should never happen
        return {"type": "percentile", "p": 0.999}

    def _allocate_buffers(self, out_f: int, in_f: int):
        # resize existing buffers; do not replace them
        self.w_q.resize_((out_f, in_f))
        self.s_w.resize_((out_f, 1))

    @classmethod
    def from_float(cls, layer: nn.Linear, act_clip: dict[str, str | float] | None = None):
        out_f, in_f = layer.out_features, layer.in_features
        has_bias = layer.bias is not None

        # Construct with known shapes and bias so load_state_dict works later
        q = cls(act_clip=act_clip, in_features=in_f, out_features=out_f, has_bias=has_bias).to(
            layer.weight.device
        )

        # ---- quantize weights once (symmetric per-output-channel) ----
        s_w, _ = calc_qparams_symmetric(layer.weight, bits=8, per_channel=True, axis=0)
        w_q, _ = quantize_symmetric(layer.weight, s_w, qmax=127, dtype=torch.int8)

        q.w_q.copy_(w_q.contiguous())
        q.s_w.copy_(s_w.to(torch.float32).contiguous())

        # ---- bias ----
        q.has_bias.fill_(1 if has_bias else 0)
        if q.bias is None:
            q.bias = nn.Parameter(
                torch.zeros(out_f, dtype=torch.float32, device=layer.weight.device)
            )
        if has_bias:
            q.bias.data.copy_(layer.bias.detach().to(torch.float32))
        else:
            q.bias.data.zero_()

        return q

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        if self.in_features is not None and x.shape[-1] != self.in_features:
            raise RuntimeError(f"Expected last dim {self.in_features}, got {x.shape[-1]}")

        # quantize activations on the fly (affine per-tensor)
        s_x, zp_x = calc_qparams_affine(x, bits=8, per_channel=False, axis=None, clip=self.act_clip)
        x_q, x_clip = quantize_affine(x, s_x, zp_x, qmin=-128, qmax=127, dtype=torch.int8)

        # "integer" GEMM in fp32 (exact for these ranges)
        x_centered = x_q.to(torch.float32) - zp_x.to(torch.float32)
        w_f = self.w_q.to(torch.float32)
        y_int = x_centered @ w_f.T

        # rescale
        y = y_int * (s_x.to(torch.float32) * self.s_w.to(torch.float32).T)

        if bool(self.has_bias.item()) and self.bias is not None:
            y = y + self.bias.to(y.dtype)

        y = y.to(x.dtype)

        if return_stats:
            return y, x_clip
        return y


if __name__ == "__main__":
    B, in_f, out_f = 32, 128, 64
    x = torch.randn(B, in_f, device="cuda", dtype=torch.float16)
    W = torch.randn(out_f, in_f, device="cuda", dtype=torch.float16)
    bias = torch.randn(out_f, device="cuda", dtype=torch.float16)

    y_ref = torch.nn.functional.linear(x, W, bias).float()
    y_q, x_clip, w_clip = quantize_matmul1(x, W, bias)
    print("x_clip:", x_clip, "w_clip:", w_clip)
    y_q = y_q.float()

    rel_err = (y_q - y_ref).abs().mean() / (y_ref.abs().mean() + 1e-8)
    cos = torch.nn.functional.cosine_similarity(y_q.flatten(), y_ref.flatten(), dim=0)
    print("rel_err:", rel_err.item(), "cos:", cos.item())

    class ToyMLP(nn.Module):
        def __init__(self, in_f=128, h=256, out_f=64):
            super().__init__()
            self.fc1 = nn.Linear(in_f, h, bias=True)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(h, out_f, bias=True)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            return x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    # Build float model
    m_fp = ToyMLP().to(device=device, dtype=dtype).eval()

    # Clone and quantize one layer
    m_q = ToyMLP().to(device=device, dtype=dtype).eval()
    m_q.load_state_dict(m_fp.state_dict())

    m_q.fc1 = QuantLinearW8A8.from_float(m_q.fc1).to(device=device)

    # Forward comparison
    x = torch.randn(512, 128, device=device, dtype=dtype)

    with torch.no_grad():
        y_fp = m_fp(x).float()
        y_q, x_clip = m_q.fc1(x, return_stats=True)  # stats from quant layer only
        # run full model output
        y_q_full = m_q(x).float()

    cos = F.cosine_similarity(y_q_full.flatten(), y_fp.flatten(), dim=0).item()
    rel_err = ((y_q_full - y_fp).abs().mean() / (y_fp.abs().mean() + 1e-8)).item()

    print("quant layer x_clip:", x_clip)
    print("full-model cos:", cos, "rel_err:", rel_err)

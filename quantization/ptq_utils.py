import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from quantize import quantize_symmetric, calc_qparams_symmetric, quantize_affine
import triton
import triton.language as tl

def include(name: str) -> bool:
    # Target the “meaty” projections:
    # - attention: q_proj, k_proj, v_proj, out_proj
    # - MLP: fc1, fc2
    # return name.endswith("fc1") or name.endswith("fc2")
    return (
        name.endswith("self_attn.q_proj")
        or name.endswith("self_attn.k_proj")
        or name.endswith("self_attn.v_proj")
        or name.endswith("self_attn.out_proj")
        or name.endswith("fc1")
        or name.endswith("fc2")
    )

def exclude(name: str) -> bool:
    # Skip embeddings and lm_head (and anything else you want)
    return (
        ("lm_head" in name) or ("embed_tokens" in name) or ("embed_positions" in name)
    )

def get_batch_1d(tokens_1d: torch.Tensor, batch_size: int, seq_len: int, device: str):
    # tokens_1d: [N]
    # x: [B, T], y: [B, T]
    assert tokens_1d.ndim == 1
    n = tokens_1d.numel()
    ix = torch.randint(0, n - seq_len - 1, (batch_size,), device=device)
    x = torch.stack([tokens_1d[i : i + seq_len] for i in ix], dim=0)
    y = torch.stack([tokens_1d[i + 1 : i + seq_len + 1] for i in ix], dim=0)
    return x, y

class QuantLinearW8A8(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, has_bias: bool, mode: str = "W8"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("last_x_clip", torch.tensor(0.0), persistent=False)
        self.register_buffer(
            "last_x",
            torch.empty((in_features, out_features), dtype=torch.float32),
            persistent=False,
        )

        self.register_buffer(
            "w_q",
            torch.empty((out_features, in_features), dtype=torch.int8),
            persistent=True,
        )
        self.register_buffer(
            "s_w", torch.empty((out_features, 1), dtype=torch.float32), persistent=True
        )

        # activation qparams (scalar)
        self.register_buffer(
            "zp_x", torch.tensor(0, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "s_x", torch.tensor(1.0, dtype=torch.float32), persistent=True
        )

        self.register_buffer(
            "has_bias", torch.tensor(int(has_bias), dtype=torch.uint8), persistent=True
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        self.mode = mode

    @classmethod
    def from_float(cls, layer: nn.Linear, s_x: float, zp_x: int, mode: str = "W8"):
        out_f, in_f = layer.out_features, layer.in_features
        has_bias = layer.bias is not None

        q = cls(in_features=in_f, out_features=out_f, has_bias=has_bias, mode=mode).to(
            layer.weight.device
        )

        # weights: symmetric per-output-channel
        s_w, _ = calc_qparams_symmetric(layer.weight, bits=8, per_channel=True, axis=0)
        w_q, _ = quantize_symmetric(layer.weight, s_w, qmax=127, dtype=torch.int8)

        q.w_q.copy_(w_q.contiguous())
        q.s_w.copy_(s_w.to(torch.float32).contiguous())

        q.s_x.copy_(
            torch.tensor(float(s_x), device=layer.weight.device, dtype=torch.float32)
        )
        q.zp_x.copy_(
            torch.tensor(int(zp_x), device=layer.weight.device, dtype=torch.int64)
        )

        if has_bias:
            q.bias.data.copy_(layer.bias.detach().to(torch.float32))
        else:
            q.bias.data.zero_()

        return q

    def forward(self, x: torch.Tensor):
        if self.mode == "W8":
            y = (x.float() @ self.w_q.to(torch.float32).T) * self.s_w.to(
                torch.float32
            ).T

        elif self.mode == "W8A8_static":
            # quantize activations with fixed qparams
            x_q, x_clip = quantize_affine(
                x, self.s_x, self.zp_x, qmin=-127, qmax=127, dtype=torch.int8
            )
            self.last_x_clip.copy_(
                torch.as_tensor(x_clip, device=x.device, dtype=torch.float32)
            )

            x_centered = x_q.to(torch.float32) - self.zp_x.to(
                torch.float32
            )  # zp_x is scalar
            w_f = self.w_q.to(torch.float32)  # [out, in]

            # matmul: [..., in] @ [in, out] -> [..., out]
            y_int = x_centered @ w_f.T
            y = y_int * (self.s_x.to(torch.float32) * self.s_w.to(torch.float32).T)

        elif self.mode == "W8A8_dynamic":
            s_x, zp_x = calc_qparams_symmetric(
                x, bits=8, per_channel=False, axis=None, qmax=127
            )
            x_q, x_clip = quantize_symmetric(x, s_x, 127, dtype=torch.int8)

            x_centered = x_q.to(torch.float32) - zp_x.to(torch.float32)
            y_int = x_centered @ self.w_q.to(torch.float32).T
            y = y_int * (s_x.to(torch.float32) * self.s_w.to(torch.float32).T)

        if bool(self.has_bias.item()):
            y = y + self.bias.to(y.dtype)

        return y.to(x.dtype)
    
def group_weights(W: torch.Tensor, group_size: int):
    assert W.size(1) % group_size == 0
    n_groups = W.size(1) // group_size
    return W.reshape((W.size(0), n_groups, group_size))

def calc_group_scales(W_grouped: torch.Tensor, qmax=7, eps=1e-8):
    amax = W_grouped.abs().amax(dim=-1, keepdim=True)
    s_w = torch.clamp(amax / qmax, min=eps)
    return s_w

def quantize_grouped(W_grouped: torch.Tensor, s_w: torch.Tensor, qmin: int, qmax: int):
    q = torch.round(W_grouped / s_w)
    q = torch.clamp(q, min=qmin, max=qmax)
    return q.to(torch.int8)

class QuantLinearW4Grouped(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, has_bias: bool, group_size: int
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.group_size = group_size
        n_groups = in_features // group_size

        self.register_buffer(
            "w_q",
            torch.empty((out_features, n_groups, group_size), dtype=torch.int8),
            persistent=True,
        )

        self.register_buffer(
            "scales",
            torch.empty((out_features, n_groups, 1), dtype=torch.float32),
            persistent=True
        )

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    @classmethod
    def from_float(cls, layer: nn.Linear, group_size: int):
        out_f, in_f = layer.out_features, layer.in_features
        n_groups = in_f // group_size
        has_bias = layer.bias is not None
        device = layer.weight.device

        q = cls(
            in_features=in_f,
            out_features=out_f,
            has_bias=has_bias,
            group_size=group_size
        ).to(device)

        W = layer.weight.data.clone().float()
        W_g = W.view(out_f, n_groups, group_size)

        scales = W_g.abs().amax(dim=-1, keepdim=True)
        scales = scales / 7.0
        scales = scales.clamp(min=1e-5)

        w_int = (W_g / scales).round().clamp(-7, 7)

        q.w_q.copy_(w_int.to(torch.int8))
        q.scales.copy_(scales)

        if has_bias:
            q.bias.data.copy_(layer.bias.detach().to(torch.float32))
        else:
            q.bias.data.zero_()

        return q

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype

        w_recon = self.w_q.float() * self.scales
        w_recon = w_recon.reshape(self.out_features, self.in_features)

        y = x.float() @ w_recon.T

        if self.has_bias:
            y = y + self.bias.to(y.dtype)

        return y.to(orig_dtype)
    
def get_module_by_name(root, dotted_name):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        return parent[int(last)]
    else:
        return getattr(parent, last)

def set_module_by_name(root, dotted_name, new_module):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)

def capture_single_activation(model, layer_name, input_ids):
    acts = {}

    def hook_fn(module, inp, out):
        # inp is a tuple; for Linear it's (x,)
        acts["x"] = inp[0].detach()

    layer = get_module_by_name(model, layer_name)
    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_ids)

    handle.remove()
    return acts["x"]

class ActCalibrator:
    def __init__(self, percentiles=(0.99, 0.999)):
        self.ps = percentiles
        self.stats = {}  # name -> dict

    @torch.no_grad()
    def observe(self, name: str, x: torch.Tensor):
        # x: [..., in_features]
        xf = x.detach().reshape(-1, x.shape[-1])
        v = xf.float().flatten()

        mn = v.min().item()
        mx = v.max().item()

        qs = torch.quantile(v, torch.tensor(self.ps, device=v.device)).tolist()

        av = v.abs()
        aqs = torch.quantile(av, torch.tensor(self.ps, device=v.device)).tolist()
        absmax = av.max().item()

        if name not in self.stats:
            self.stats[name] = {
                "min": mn,
                "max": mx,
                "absmax": absmax,
                **{f"p{int(p * 1000)}": q for p, q in zip(self.ps, qs)},
                **{f"ap{int(p * 1000)}": q for p, q in zip(self.ps, aqs)},
                "n": v.numel(),
            }
        else:
            s = self.stats[name]
            s["min"] = min(s["min"], mn)
            s["max"] = max(s["max"], mx)
            s["absmax"] = max(s["absmax"], absmax)
            for p, q in zip(self.ps, qs):
                s[f"p{int(p * 1000)}"] = max(s[f"p{int(p * 1000)}"], q)
            for p, q in zip(self.ps, aqs):
                s[f"ap{int(p * 1000)}"] = max(s[f"ap{int(p * 1000)}"], q)
            s["n"] += v.numel()

class ActChannelCalibrator:
    def __init__(self, percentiles=(0.99, 0.999), store_on_cpu=True):
        self.ps = percentiles
        self.stats = {}
        self.store_on_cpu = store_on_cpu

    @torch.no_grad()
    def observe(self, name: str, x: torch.Tensor, ):
        xf = x.detach().reshape(-1, x.shape[-1]).float()
        axf = xf.abs()

        absmax = axf.amax(dim=0)
        aqs = torch.quantile(axf, torch.tensor(self.ps, device=axf.device), dim=0)

        if self.store_on_cpu:
            absmax = absmax.cpu()
            aqs = aqs.cpu()

        if name not in self.stats:
            self.stats[name] = {
                "absmax": absmax,
                **{f"ap{int(self.ps[i] * 1000)}": aqs[i] for i in range(len(self.ps))},
                "n": xf.shape[0]
            }
        else:
            s = self.stats[name]
            s['absmax'] = torch.max(s['absmax'], absmax)
            for i in range(len(self.ps)):
                s[f"ap{int(self.ps[i] * 1000)}"] = torch.max(s[f"ap{int(self.ps[i] * 1000)}"], aqs[i])
            s['n'] += xf.shape[0]

@torch.no_grad()
def calibrate(model, tokens_1d, batch_size, seq_len, device, num_batches=10):
    model.eval()
    calib = ActCalibrator(percentiles=(0.99, 0.999))
    handles = attach_linear_input_hooks(
        model, calib, include_filter=include, exclude_filter=exclude
    )

    for _ in range(num_batches):
        x, _ = get_batch_1d(tokens_1d, batch_size, seq_len, device)
        _ = model(input_ids=x)

    for h in handles:
        h.remove()

    return calib.stats

def attach_linear_input_hooks(
    model, calibrator, include_filter=None, exclude_filter=None
):
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if include_filter is not None and not include_filter(name):
                continue
            if exclude_filter is not None and exclude_filter(name):
                continue

            def _make_hook(nm):
                def hook(m, inp, out):
                    x = inp[0]
                    calibrator.observe(nm, x)

                return hook

            handles.append(mod.register_forward_hook(_make_hook(name)))
    return handles

@torch.no_grad()
def evaluate_simple(model, eval_data):

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in eval_data:
        logits = model(input_ids=x).logits  # [B, T, V]
        V = logits.size(-1)
        loss = F.cross_entropy(
            logits.float().view(-1, V),
            y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    val_loss = total_loss / total_tokens
    perplexity = math.exp(val_loss)
    return val_loss, perplexity

class LinearFP32GEMM(nn.Module):
    def __init__(self, layer: nn.Linear):
        super().__init__()
        # keep the original params so state_dict / tying still works
        self.weight = layer.weight
        self.bias = layer.bias
        self.in_features = layer.in_features
        self.out_features = layer.out_features

    def forward(self, x: torch.Tensor):
        y = F.linear(
            x.float(),
            self.weight.float(),
            None if self.bias is None else self.bias.float(),
        )
        return y.to(x.dtype)

def force_fp32_gemm_all_linears(model, include_filter=None, exclude_filter=None):
    to_replace = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if include_filter is not None and not include_filter(name):
                continue
            if exclude_filter is not None and exclude_filter(name):
                continue
            to_replace.append((name, mod))

    for name, mod in to_replace:
        set_module_by_name(model, name, LinearFP32GEMM(mod))

    return model

def choose_activation_range(stats, mode: str, pkey: str = "999"):
    if mode == "minmax":
        xmin, xmax = stats["min"], stats["max"]
    elif mode == "symmetric_abs":
        a = stats[f"ap{pkey}"]
        xmin, xmax = -a, a
    else:
        raise ValueError(f"unknown mode: {mode}")
    return xmin, xmax

def activation_qparams_from_range(xmin, xmax, bits=8, scheme="symmetric", eps=1e-8):
    qmax = (2 ** (bits - 1)) - 1  # 127 for int8
    if scheme == "symmetric":
        qmin = -qmax
        a = max(abs(xmin), abs(xmax))
        scale = max(a / qmax, eps)
        zp = 0
    else:
        raise ValueError(
            "For this OPT baseline, use symmetric activation ranges first."
        )
    return scale, zp, qmin, qmax

def quantize_model_w8a8(model, act_stats, mode, include, exclude):
    # gather first (avoid mutating while iterating)
    to_replace = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if not include(name) or exclude(name):
                continue
            if name in act_stats:
                to_replace.append((name, mod))

    for name, mod in to_replace:
        xmin, xmax = choose_activation_range(
            act_stats[name], mode="symmetric_abs", pkey="999"
        )
        s_x, zp_x, _, _ = activation_qparams_from_range(xmin, xmax, scheme="symmetric")
        new_mod = QuantLinearW8A8.from_float(mod, s_x=s_x, zp_x=zp_x, mode=mode)
        set_module_by_name(model, name, new_mod)

    return model

def quantize_model_w4(model, group_size, include, exclude):
    to_replace = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if not include(name) or exclude(name):
                continue
            to_replace.append((name, mod))

    for name, mod in to_replace:
        new_mod = QuantLinearW4Grouped.from_float(mod, group_size=group_size)
        set_module_by_name(model, name, new_mod)

    return model

class HessianAgg:
    def __init__(self, store_on_cpu=True, max_rows=8000, dtype=torch.float64):
        self.store_on_cpu = store_on_cpu
        self.max_rows = max_rows
        self.dtype = dtype
        self.n = 0
        self.H = None  # [d, d] on cpu (float64 by default)

    @torch.no_grad()
    def observe(self, x: torch.Tensor):
        # x: [..., d]
        X = x.detach().reshape(-1, x.shape[-1]).float()  # [N, d]
        if self.max_rows is not None and X.size(0) > self.max_rows:
            idx = torch.randint(0, X.size(0), (self.max_rows,), device=X.device)
            X = X[idx]

        X = X.to(dtype=self.dtype)
        if self.store_on_cpu:
            X = X.cpu()

        if self.H is None:
            d = X.size(1)
            self.H = torch.zeros((d, d), dtype=self.dtype, device=X.device)

        self.H += X.T @ X
        self.n += X.size(0)

    def finalize(self, damp_ratio=1e-4):
        H = self.H / max(self.n, 1)
        # damping (you said you're already doing this in build_hessian_from_X;
        # you can keep it here instead and remove elsewhere)
        diag_mean = torch.diagonal(H).mean()
        if diag_mean > 0:
            H = H + damp_ratio * diag_mean * torch.eye(H.size(0), dtype=H.dtype, device=H.device)
        return H.float()
    
def attach_one_linear_input_hook(model, layer_name: str, agg: HessianAgg):
    target = dict(model.named_modules())[layer_name]
    assert isinstance(target, torch.nn.Linear)

    def hook(mod, inp, out):
        x = inp[0]
        agg.observe(x)

    h = target.register_forward_hook(hook)
    return h

class GPTQLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
        has_perm: bool,
        group_size: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert in_features % group_size == 0
        n_groups = in_features // group_size
        self.group_size = group_size
        self.n_groups = n_groups

        self.has_perm = bool(has_perm)
        self.has_bias = bool(has_bias)

        self.register_buffer(
            "w_q",
            torch.empty((out_features, n_groups, group_size), dtype=torch.int8),
            persistent=True,
        )
        # store as [g, o] so it broadcasts nicely with y_g: [n, g, o]
        self.register_buffer(
            "s_w",
            torch.empty((n_groups, out_features), dtype=torch.float32),
            persistent=True,
        )

        # always present for state_dict compatibility
        self.register_buffer(
            "perm",
            torch.empty((in_features,), dtype=torch.long),
            persistent=True,
        )

        # always register bias param; zero it if unused
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    @classmethod
    def from_float(
        cls,
        layer: nn.Linear,
        w_q: torch.Tensor,  # [out, g, k] int8
        s_w: torch.Tensor,  # [out, g, 1] float
        group_size: int,
        perm: torch.Tensor | None,
    ):
        out_f, in_f = layer.out_features, layer.in_features
        has_bias = layer.bias is not None
        has_perm = perm is not None

        q = cls(
            in_features=in_f,
            out_features=out_f,
            has_bias=has_bias,
            has_perm=has_perm,
            group_size=group_size,
        ).to(layer.weight.device)

        q.w_q.copy_(w_q.contiguous())

        # s_w input is [out, g, 1] -> want [g, out]
        s_go = s_w.to(torch.float32).squeeze(-1).transpose(0, 1).contiguous()
        q.s_w.copy_(s_go)

        if has_bias:
            q.bias.data.copy_(layer.bias.detach().to(torch.float32))
        else:
            q.bias.data.zero_()

        if has_perm:
            q.perm.copy_(perm.to(dtype=torch.long, device=q.perm.device).contiguous())
        else:
            # identity perm for consistency
            q.perm.copy_(torch.arange(in_f, device=q.perm.device, dtype=torch.long))

        return q

    def forward(self, x: torch.Tensor):
        prefix = x.shape[:-1]
        orig_dtype = x.dtype

        x = x.flatten(start_dim=0, end_dim=-2)  # [N, in]
        if self.has_perm:
            x = x[..., self.perm]

        x = x.view((x.size(0), self.n_groups, self.group_size))  # [N, g, k]

        # int8 dot-product in fp32
        x_f = x.float()
        w_f = self.w_q.float()
        y_g = torch.einsum("n g k, o g k -> n g o", x_f, w_f)  # [N, g, o]

        # apply per-(g,o) scale: s_w is [g,o] -> [1,g,o]
        y_g = y_g * self.s_w.unsqueeze(0)

        y = y_g.sum(dim=1)  # [N, o]

        if self.has_bias:
            y = y + self.bias.to(y.dtype)

        return y.view((*prefix, self.out_features)).to(orig_dtype)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    assert q.dtype == torch.int8
    q = q.view(*q.shape[:-1], q.shape[-1] // 2, 2)
    q0 = q[..., 0] & 0x0F
    q1 = q[..., 1] & 0x0F
    packed = (q1 << 4) | q0
    return packed.to(torch.uint8)

def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.uint8

    low  = packed & 0x0F
    high = (packed >> 4) & 0x0F

    low  = low.to(torch.int8)
    high = high.to(torch.int8)

    low[low >= 8]   -= 16
    high[high >= 8] -= 16

    q = torch.stack([low, high], dim=-1)
    return q.view(*packed.shape[:-1], packed.shape[-1] * 2)

def test_int4_pack_roundtrip():
    q = torch.randint(-8, 8, (1024,), dtype=torch.int8)
    q = q[: q.numel() // 2 * 2]  # ensure even length

    packed = pack_int4(q)
    q2 = unpack_int4(packed)

    assert torch.equal(q, q2)


class AWQW4Packed(nn.Module):
    def __init__(self, in_features: int, out_features: int, has_bias: bool, group_size: int):
        super().__init__()
        assert in_features % group_size == 0
        assert group_size % 2 == 0  # must be even to pack 2x int4 per byte

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.group_size = group_size
        n_groups = in_features // group_size

        # packed bytes: last dim is halved
        self.register_buffer(
            "w_q_packed",
            torch.empty((out_features, n_groups, group_size // 2), dtype=torch.uint8),
            persistent=True,
        )

        self.register_buffer(
            "w_scales",
            torch.empty((out_features, n_groups, 1), dtype=torch.float32),
            persistent=True,
        )

        self.register_buffer(
            "x_inv_s",
            torch.empty((in_features, ), dtype=torch.float32),
            persistent=True,
        )

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    @classmethod
    def from_float(cls, layer: nn.Linear, s: torch.Tensor, group_size: int):
        out_f, in_f = layer.out_features, layer.in_features
        n_groups = in_f // group_size
        has_bias = layer.bias is not None
        device = layer.weight.device

        q = cls(in_features=in_f, out_features=out_f, has_bias=has_bias, group_size=group_size).to(device)

        W = layer.weight.detach().float()  # [out, in]
        W_g = W.view(out_f, n_groups, group_size)
        
        s = s.to(device).float().view(-1)
        inv_s_vec = (1.0 / s).contiguous()
        s_view = s.view(1, n_groups, group_size)

        # AWQ: scale weights by s, and divide activations by s in forward
        w_scaled = W_g * s_view

        scales = w_scaled.abs().amax(dim=-1, keepdim=True) / 7.0
        scales = scales.clamp(min=1e-5)

        # quantize to int4-ish values (stored in int8 container)
        w_int = torch.round(w_scaled / scales).clamp(-7, 7).to(torch.int8)   # [out, g, k]

        # pack last dim (k) -> k/2 bytes
        w_packed = pack_int4(w_int)  # expect uint8 [out, g, k/2]

        q.w_q_packed.copy_(w_packed.contiguous())
        q.w_scales.copy_(scales.contiguous())
        q.x_inv_s.copy_(inv_s_vec)

        if has_bias:
            q.bias.data.copy_(layer.bias.detach().to(torch.float32))
        else:
            q.bias.data.zero_()

        return q

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype

        w_int = unpack_int4(self.w_q_packed).to(torch.float32)  # [out, g, k]
        w_recon = (w_int * self.w_scales).reshape(self.out_features, self.in_features)  # [out, in]

        x_f = x.to(torch.float32) * self.x_inv_s
        y = x_f @ w_recon.T

        if self.has_bias:
            y = y + self.bias.to(y.dtype)

        return y.to(orig_dtype)

    # @torch.no_grad()
    # def forward_kernel(
    #     self,
    #     x: torch.Tensor,                     # [N, K]
    #     *,
    #     BLOCK_M=64,
    #     BLOCK_N=64,
    #     BLOCK_K=32,
    #     num_warps=4,
    #     num_stages=2,
    # ):
    #     # ---- shapes ----
    #     assert x.ndim == 2
    #     N, K = x.shape
    #     O = self.out_features
    #     assert K == self.in_features
    #     assert K % self.group_size == 0
    #     assert (K % BLOCK_K) == 0, "kernel uses tl.static_range(0, K, BLOCK_K)"

    #     # ---- allocate output ----
    #     y = torch.empty((N, O), device=x.device, dtype=x.dtype)

    #     # ---- pointers / tensors expected by kernel ----
    #     # W_ptr: [O, G, K/2] uint8   (packed int4)
    #     # SCALES_ptr: [O, G] fp16/fp32
    #     # INV_S_ptr: [K] fp16/fp32
    #     # BIAS_ptr: [O] fp16/fp32 or None

    #     W_packed = self.w_q_packed                            # uint8 [O, G, K/2] (your storage)
    #     scales = self.w_scales.squeeze(-1)                  # [O, G]
    #     inv_s = self.x_inv_s.to(scales.dtype)           # [K]
    #     bias = self.bias.to(scales.dtype) if self.has_bias else None

    #     # ---- compute strides (in elements, not bytes) ----
    #     # X is [N, K]
    #     stride_xn = x.stride(0)
    #     stride_xk = x.stride(1)

    #     # W is [O, G, K/2]
    #     stride_wo  = W_packed.stride(0)
    #     stride_wg  = W_packed.stride(1)
    #     stride_wk2 = W_packed.stride(2)

    #     # scales is [O, G]
    #     stride_so = scales.stride(0)
    #     stride_sg = scales.stride(1)

    #     # inv_s is [K]
    #     stride_inv = inv_s.stride(0)

    #     # Y is [N, O]
    #     stride_yn = y.stride(0)
    #     stride_yo = y.stride(1)

    #     # ---- grid ----
    #     grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(O, BLOCK_N))

    #     # ---- launch ----
    #     awq_w4_gemm_kernel[grid](
    #         x, W_packed, scales, inv_s,
    #         bias if bias is not None else x,   # dummy ptr if HAS_BIAS=0 (won't be read)
    #         y,
    #         N=N, O=O, K=K, GROUP_SIZE=self.group_size,
    #         stride_xn=stride_xn, stride_xk=stride_xk,
    #         stride_wo=stride_wo, stride_wg=stride_wg, stride_wk2=stride_wk2,
    #         stride_so=stride_so, stride_sg=stride_sg,
    #         stride_inv=stride_inv,
    #         stride_yn=stride_yn, stride_yo=stride_yo,
    #         HAS_BIAS=1 if bias is not None else 0,
    #         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    #         num_warps=num_warps,
    #         num_stages=num_stages,
    #     )
    #     return y

@triton.jit
def awq_w4_gemm_kernel(
    X_ptr, W_ptr, SCALES_ptr, INV_S_ptr, BIAS_ptr, Y_ptr,
    N: tl.constexpr, O: tl.constexpr, K: tl.constexpr, GROUP_SIZE: tl.constexpr,
    stride_xn: tl.constexpr, stride_xk: tl.constexpr,
    stride_wo: tl.constexpr, stride_wg: tl.constexpr, stride_wk2: tl.constexpr,
    stride_so: tl.constexpr, stride_sg: tl.constexpr,
    stride_inv: tl.constexpr,
    stride_yn: tl.constexpr, stride_yo: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n0 = pid_m * BLOCK_M
    o0 = pid_n * BLOCK_N
    offs_m = n0 + tl.arange(0, BLOCK_M)
    offs_o = o0 + tl.arange(0, BLOCK_N)
    mask_o = offs_o < O
    mask_m = offs_m < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # ----------------------------
    # 3) Loop over K in BLOCK_K steps
    # ----------------------------
    for k0 in tl.static_range(0, K, BLOCK_K):
    # For each k-tile:
    #   3.1) Load X tile
    #       - shape: [BLOCK_M, BLOCK_K]
    #       - mask invalid rows/cols
    #       - cast to fp32 for math
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]
        mask_k = offs_k < K

        x_ptrs = X_ptr + offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float16)
    #
    #   3.2) Apply AWQ input rescale
    #       - load INV_S for this k-range (shape [BLOCK_K])
    #       - multiply X tile by INV_S (broadcast along M)
        inv_ptrs = INV_S_ptr + offs_k * stride_inv
        inv = tl.load(inv_ptrs, mask=mask_k, other=0.0).to(tl.float16)
        x = x * inv[None, :]
    #
    #   3.3) Compute group/nibble addressing for weights
    #       - For each k in this tile:
    #           g  = k // GROUP_SIZE
    #           kk = k %  GROUP_SIZE
    #           byte_index = kk // 2     (since 2 int4 per byte)
    #           nibble_sel = kk & 1      (0 = low nibble, 1 = high nibble)
        offs_g = offs_k // GROUP_SIZE
        offs_kk = offs_k - offs_g * GROUP_SIZE
        offs_byte = offs_kk >> 1
        nibble_sel = offs_kk & 1
    #
    #   3.4) Load packed weight bytes for this tile
    #       - You want bytes corresponding to:
    #           outputs: O tile (BLOCK_N)
    #           k-range: this BLOCK_K
    #       - Use strides (stride_wo, stride_wg, stride_wk2)
    #       - Result is uint8 “packed” values
        w_ptrs = (
            W_ptr
            + offs_o[:, None] * stride_wo
            + offs_g[None, :] * stride_wg
            + offs_byte[None, :] * stride_wk2
        )
        w_mask = mask_o[:, None] & mask_k[None, :]
        w_packed = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.uint8)
    #
    #   3.5) Unpack int4 -> signed int8
    #       - Extract low/high nibble from each byte using nibble_sel
    #       - Convert 0..15 to signed range consistent with your packing:
    #           * if you used two’s complement int4: map to [-8,7]
    #           * if you used offset packing: undo the offset to [-7,7]
        low = w_packed & 0x0F
        high = (w_packed >> 4) & 0x0F
        w_u4 = tl.where(nibble_sel[None, :] == 0, low, high)
        w_s4 = tl.where(w_u4 < 8, w_u4, w_u4 - 16)
        w_i8 = w_s4.to(tl.int8)
    #
    #   3.6) Load dequant scales for this tile
    #       - scales are per-(output, group):
    #           scale[o, g]
    #       - For each k, pick its group g and broadcast across M
        scales_ptrs = SCALES_ptr + (offs_o[:, None] * stride_so) + (offs_g[None, :] * stride_sg)
        scale_ok = tl.load(
            scales_ptrs,
            mask=(mask_o[:, None] & mask_k[None, :]),
            other=0.0,
        ).to(tl.float32)
    #
    #   3.7) Dequantize weights
    #       - w_fp32 = q_int8 * scale_fp32
    #       - w_fp32 should correspond to weight values for these O and K indices
    #
        w_fp32 = (w_i8 * scale_ok).to(tl.float16)

    #   3.8) Accumulate GEMM
    #       - acc += X_tile_fp32 @ (W_tile_fp32)^T
    #       - Make sure shapes align:
    #           X: [BM, BK]
    #           W: [BN, BK]  (transpose inside dot)
    #
        acc += tl.dot(x, tl.trans(w_fp32))
    # Notes:
    #   - Keep everything fp32 inside the loop for stability
    #   - Use masks consistently to avoid OOB loads
    #   - Prefer reuse: compute g/byte/nibble once per k-tile

    # ----------------------------
    # 4) Optional bias
    # ----------------------------
    # - If HAS_BIAS:
    #     load bias for O tile [BLOCK_N]
    #     acc += bias (broadcast across M)
    if HAS_BIAS:
        bias = tl.load(BIAS_ptr + offs_o, mask=mask_o, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # ----------------------------
    # 5) Store output
    # ----------------------------
    # - Cast acc back to output dtype (fp16/bf16)
    # - Store to Y with proper strides
    # - Apply mask for tail tiles

    acc = acc.to(tl.float32)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_yn + offs_o[None, :] * stride_yo
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_o[None, :])

def compare_outputs(
    y_ref: torch.Tensor,
    y_test: torch.Tensor,
    name_ref="slow",
    name_test="triton",
):
    """
    Compare two tensors numerically.
    Intended for y_slow vs y_triton checks.
    """
    assert y_ref.shape == y_test.shape, "Shape mismatch"

    y_ref_f = y_ref.float().flatten()
    y_test_f = y_test.float().flatten()

    diff = y_test_f - y_ref_f
    abs_diff = diff.abs()

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # relative error (avoid div-by-zero)
    denom = y_ref_f.abs().clamp(min=1e-6)
    rel_err = (abs_diff / denom).mean().item()

    # cosine similarity
    cos = F.cosine_similarity(y_ref_f, y_test_f, dim=0).item()

    print(f"\n=== Output comparison ({name_test} vs {name_ref}) ===")
    print(f"Max |Δ|        : {max_abs:.6e}")
    print(f"Mean |Δ|       : {mean_abs:.6e}")
    print(f"Mean rel error : {rel_err:.6e}")
    print(f"Cosine sim     : {cos:.6f}")

    # sanity stats
    print("\nStats:")
    print(f"{name_ref}: min={y_ref_f.min():.4e}, max={y_ref_f.max():.4e}, mean={y_ref_f.mean():.4e}")
    print(f"{name_test}: min={y_test_f.min():.4e}, max={y_test_f.max():.4e}, mean={y_test_f.mean():.4e}")

    # NaN / inf check
    for name, y in [(name_ref, y_ref_f), (name_test, y_test_f)]:
        if torch.isnan(y).any():
            print(f"WARNING: {name} contains NaNs")
        if torch.isinf(y).any():
            print(f"WARNING: {name} contains Infs")
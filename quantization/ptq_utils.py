import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from quantize import quantize_symmetric, calc_qparams_symmetric, quantize_affine

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

        assert in_features % group_size == 0
        n_groups = in_features // group_size
        self.group_size = group_size
        self.n_groups = n_groups

        self.register_buffer(
            "w_q",
            torch.empty((out_features, n_groups, group_size), dtype=torch.int8),
            persistent=True,
        )
        self.register_buffer(
            "s_w",
            torch.empty((n_groups, out_features), dtype=torch.float32),
            persistent=True,
        )

        self.register_buffer(
            "has_bias", torch.tensor(int(has_bias), dtype=torch.uint8), persistent=True
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    @classmethod
    def from_float(cls, layer: nn.Linear, group_size: int):
        out_f, in_f = layer.out_features, layer.in_features
        has_bias = layer.bias is not None

        q = cls(
            in_features=in_f,
            out_features=out_f,
            has_bias=has_bias,
            group_size=group_size,
        ).to(layer.weight.device)

        # weights: symmetric per-output-channel
        w_g = group_weights(layer.weight, group_size)
        s_w = calc_group_scales(w_g, qmax=7)
        w_q = quantize_grouped(w_g, s_w, qmin=-7, qmax=7)

        q.w_q.copy_(w_q.contiguous())
        q.s_w.copy_(s_w.to(torch.float32).contiguous().squeeze(-1).T)

        if has_bias:
            q.bias.data.copy_(layer.bias.detach().to(torch.float32))
        else:
            q.bias.data.zero_()

        return q

    def forward(self, x: torch.Tensor):
        prefix = x.shape[:-1]
        orig_dtype = x.dtype

        x = x.flatten(start_dim=0, end_dim=-2)
        x = x.view((x.size(0), self.n_groups, self.group_size))

        x_f = x.float()
        w_f = self.w_q.float()
        y_g = torch.einsum("n g k, o g k -> n g o", x_f, w_f)
        y_g = y_g * self.s_w.unsqueeze(0)
        y = y_g.sum(dim=1)

        if bool(self.has_bias.item()):
            y = y + self.bias.to(y.dtype)

        return y.reshape((*prefix, self.out_features)).to(orig_dtype)
    
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
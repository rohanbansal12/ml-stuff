import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from tokenizers import Tokenizer
from quantize import quantize_affine, quantize_symmetric, calc_qparams_symmetric
from typing import Optional
import copy

import sys
sys.path.append(".")
from GPT.model import GPT
from GPT.utils import get_batch, load_shakespeare

BATCH_SIZE = 128
MAX_SEQ_LEN = 128
CALIBRATION_BATCHES = 25
EVALUATION_BATCHES = 50
SEED = 25

class ActCalibrator:
    def __init__(self, percentiles=(0.99, 0.999)):
        self.ps = percentiles
        self.stats = {}  # name -> dict

    @torch.no_grad()
    def observe(self, name: str, x: torch.Tensor):
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
                "min": mn, "max": mx, "absmax": absmax,
                **{f"p{int(p*1000)}": q for p, q in zip(self.ps, qs)},
                **{f"ap{int(p*1000)}": q for p, q in zip(self.ps, aqs)},
                "n": v.numel(),
            }
        else:
            s = self.stats[name]
            s["min"] = min(s["min"], mn)
            s["max"] = max(s["max"], mx)
            s["absmax"] = max(s["absmax"], absmax)
            for p, q in zip(self.ps, qs):
                s[f"p{int(p*1000)}"] = max(s[f"p{int(p*1000)}"], q)
            for p, q in zip(self.ps, aqs):
                s[f"ap{int(p*1000)}"] = max(s[f"ap{int(p*1000)}"], q)
            s["n"] += v.numel()

def include(name: str) -> bool:
    # only linears inside blocks (att and ffn)
    return (
        ".att.W_" in name or
        ".fc.lin" in name
    )

def exclude(name: str) -> bool:
    return ("lm_head" in name)  # skip tied head for Stage 2 baseline

def attach_linear_input_hooks(model, calibrator, include_filter=None, exclude_filter=None):
    """
    include_filter/exclude_filter: functions(name:str)->bool
    """
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
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
def calibrate(model, tokens, batch_size, seq_len, device, num_batches=50):
    model.eval()
    calib = ActCalibrator(percentiles=(0.99, 0.999))

    handles = attach_linear_input_hooks(model, calib, include_filter=include, exclude_filter=exclude)

    for _ in range(num_batches):
        x, _ = get_batch(tokens, batch_size, seq_len, device)
        _ = model(x)  # stats collected by hooks

    for h in handles:
        h.remove()

    return calib.stats

@torch.no_grad()
def collect_weight_stats(model):
    wstats = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ((".att.W_" in name) or (".fc.lin" in name)):
            W = mod.weight.detach()
            # per-output-channel absmax
            absmax = W.abs().amax(dim=1).float()  # [out]
            wstats[name] = {"absmax": absmax.cpu()}
    return wstats

@torch.no_grad()
def evaluate(model, tokens, batch_size, seq_len, device, num_batches=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for _ in range(num_batches):
        x, y = get_batch(tokens, batch_size, seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum",                   # token-sum so we can average exactly
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    val_loss = total_loss / total_tokens
    perplexity = math.exp(val_loss)
    return val_loss, perplexity

def evaluate_simple(model, eval_data):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in eval_data:
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum",                   # token-sum so we can average exactly
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    val_loss = total_loss / total_tokens
    perplexity = math.exp(val_loss)
    return val_loss, perplexity

def evaluate_with_fixed_seed(model, seed=0, eval_data=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return evaluate_simple(model, eval_data)

def choose_activation_range(stats, mode: str, pkey: str = "999"):
    if mode == "minmax":
        xmin, xmax = stats["min"], stats["max"]

    elif mode == "symmetric_abs":
        # uses abs-percentile collected during calibration, e.g. "ap999"
        a = stats[f"ap{pkey}"]
        xmin, xmax = -a, a

    elif mode == "affine_percentile":
        # requires you to have stored BOTH tails during calibration:
        # e.g. "lo999" = quantile(x, 1-p), "hi999" = quantile(x, p)
        xmin, xmax = stats[f"lo{pkey}"], stats[f"hi{pkey}"]

    else:
        raise ValueError(f"unknown mode: {mode}")

    return xmin, xmax

def activation_qparams_from_range(xmin, xmax, bits=8, scheme="affine", eps=1e-8):
    # default int8 range
    qmax = (2 ** (bits - 1)) - 1

    if scheme == "symmetric":
        qmin = -qmax
        a = max(abs(xmin), abs(xmax))
        scale = max(a / qmax, eps)
        zp = 0

    elif scheme == "affine":
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1
        scale = max((xmax - xmin) / (qmax - qmin), eps)
        zp = round(qmin - xmin / scale)
        zp = min(max(zp, qmin), qmax)

    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return scale, zp, qmin, qmax

class QuantLinearW8A8StaticAct(nn.Module):
    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        has_bias: Optional[bool] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("last_x_clip", torch.tensor(0.0), persistent=False)

        # ---- buffers for quantized weights/scales ----
        self.register_buffer("w_q", torch.empty(0, dtype=torch.int8), persistent=True)
        self.register_buffer("s_w", torch.empty(0, dtype=torch.float32), persistent=True)

        self.register_buffer("zp_x", torch.tensor(0, dtype=torch.int64), persistent=True)
        self.register_buffer("s_x", torch.tensor(1.0, dtype=torch.float32), persistent=True)

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

    def _allocate_buffers(self, out_f: int, in_f: int):
        # resize existing buffers; do not replace them
        self.w_q.resize_((out_f, in_f))
        self.s_w.resize_((out_f, 1))

    @classmethod
    def from_float(cls, layer, s_x, zp_x):
        out_f, in_f = layer.out_features, layer.in_features
        has_bias = layer.bias is not None

        q = cls(in_features=in_f, out_features=out_f, has_bias=has_bias).to(layer.weight.device)

        s_w, _ = calc_qparams_symmetric(layer.weight, bits=8, per_channel=True, axis=0)
        w_q, _ = quantize_symmetric(layer.weight, s_w, qmax=127, dtype=torch.int8)

        # ---- quantize weights once (symmetric per-output-channel) ----
        q.w_q.copy_(w_q.contiguous())
        q.s_w.copy_(s_w.to(torch.float32).contiguous())

        s_x_t = torch.as_tensor(s_x, device=layer.weight.device, dtype=torch.float32)
        zp_x_t = torch.as_tensor(zp_x, device=layer.weight.device, dtype=torch.int64)
        q.s_x.copy_(s_x_t)
        q.zp_x.copy_(zp_x_t)

        # ---- bias ----
        q.has_bias.fill_(1 if has_bias else 0)
        if q.bias is None:
            q.bias = nn.Parameter(torch.zeros(out_f, dtype=torch.float32, device=layer.weight.device))
        if has_bias:
            q.bias.data.copy_(layer.bias.detach().to(torch.float32))
        else:
            q.bias.data.zero_()

        return q


    def forward(self, x, return_stats=False):
        # "integer" GEMM in fp32 (exact for these ranges)
        x_q, x_clip = quantize_affine(x, self.s_x, self.zp_x, qmin=-127, qmax=127, dtype=torch.int8)
        self.last_x_clip.copy_(torch.as_tensor(x_clip, device=x.device, dtype=torch.float32))

        x_centered = x_q.to(torch.float32) - self.zp_x.to(torch.float32)
        w_f = self.w_q.to(torch.float32)
        y_int = x_centered @ w_f.T

        # rescale
        y = y_int * (self.s_x.to(torch.float32) * self.s_w.to(torch.float32).T)

        if bool(self.has_bias.item()) and self.bias is not None:
            y = y + self.bias.to(y.dtype)

        y = y.to(x.dtype)
        
        if return_stats:
            return y, x_clip
        return y

def set_module_by_name(root, dotted_name, new_module):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)

def quantize_model_w8a8(model, act_stats, include_filter=None, exclude_filter=None):
    to_replace = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if include_filter is not None and not include_filter(name):
                continue
            if exclude_filter is not None and exclude_filter(name):
                continue
            to_replace.append((name, mod))

    for name, mod in to_replace:
        if name in act_stats:
            xmin, xmax = choose_activation_range(act_stats[name], mode="symmetric_abs", pkey="999")
            s_x, zp_x, _, _ = activation_qparams_from_range(xmin, xmax, scheme='symmetric')
            new_mod = QuantLinearW8A8StaticAct.from_float(mod, s_x, zp_x)
            set_module_by_name(model, name, new_mod)

    return model



def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt = torch.load("/workspace/ml-stuff/ckpts/gpt/baseline_epoch=29.pt", map_location=device)
    model = GPT(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model = model.float()
    model.eval()

    tokenizer = Tokenizer.from_file("/workspace/ml-stuff/data/shakespeare_bpe.json")
    text = load_shakespeare("/workspace/ml-stuff/data/shakespeare.txt")
    encoding = tokenizer.encode(text)
    ids = encoding.ids
    tokens = torch.tensor(ids, dtype=torch.long).to(device)
    split = int(0.9 * len(tokens))
    val_tokens = tokens[split:]

    full_eval_data = []
    for batch in range(EVALUATION_BATCHES):
        x, y = get_batch(val_tokens, BATCH_SIZE, MAX_SEQ_LEN, device)
        full_eval_data.append((x, y))

    fp_baseline_loss, fp_baseline_perp = evaluate_with_fixed_seed(model, seed=SEED, eval_data=full_eval_data)
    print(f"FP: Loss:{fp_baseline_loss:.4f} | Perp:{fp_baseline_perp:.4f}")

    act_stats = calibrate(model, val_tokens, BATCH_SIZE, MAX_SEQ_LEN, device, CALIBRATION_BATCHES)
    # print("Num calibrated linears:", len(act_stats))
    # top = sorted(act_stats.items(), key=lambda kv: kv[1].get("absmax", kv[1]["max"]), reverse=True)[:10]
    # for name, s in top:
    #     print(name, {k: s[k] for k in ["absmax", "ap999", "min", "max"] if k in s})

    # weight_stats = collect_weight_stats(model)
    # print(weight_stats)

    model_quant = quantize_model_w8a8(copy.deepcopy(model), act_stats, include, exclude)
    quant_loss, quant_perp = evaluate_with_fixed_seed(model_quant, seed=SEED, eval_data=full_eval_data)
    print(f"Quant: Loss:{quant_loss:.4f} | Perp:{quant_perp:.4f}")

if __name__ == "__main__":
    main()
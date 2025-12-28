import torch
import torch.nn as nn
import copy

from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq_utils import (
    evaluate_simple,
    get_batch_1d,
    force_fp32_gemm_all_linears,
    include,
    exclude,
    ActChannelCalibrator,
    attach_linear_input_hooks,
    set_module_by_name
)
from quantize import calc_qparams_symmetric, quantize_symmetric

import sys

sys.path.append(".")
from GPT.utils import load_shakespeare  # reuse your local shakespeare loader


# -----------------------------
# Config (tune for your 4090)
# -----------------------------
MODEL_NAME = "facebook/opt-1.3b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OPT-1.3B is much heavier than your tiny GPT.
# Start small; increase BATCH_SIZE if you have headroom.
BATCH_SIZE = 8
MAX_SEQ_LEN = 128
CALIBRATION_BATCHES = 10
EVALUATION_BATCHES = 20
SEED = 25

# local text file (reuse your existing path)
TEXT_PATH = "/workspace/ml-stuff/data/shakespeare.txt"

def summarize_scale(scale_stats, k=5):
    # scale_stats: name -> tensor[1, in] or [in]
    vals = []
    for name, s in scale_stats.items():
        sf = s.float().flatten()
        vals.append((name, sf.min().item(), sf.median().item(), sf.max().item()))
    vals.sort(key=lambda t: t[3], reverse=True)
    print("Top scale max:")
    for t in vals[:k]:
        print(f"{t[0]:60s}  min={t[1]:.3e}  med={t[2]:.3e}  max={t[3]:.3e}")

@torch.no_grad()
def compute_weight_channel_stats(model, include=None, exclude=None):
    weight_stats = {}

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if include is not None and not include(name):
            continue
        if exclude is not None and exclude(name):
            continue
        w = mod.weight.detach().cpu()
        w_absmax = w.abs().amax(dim=0)
        weight_stats[name] = w_absmax.cpu()

    return weight_stats

@torch.no_grad()
def compute_act_channel_stats(
    model,
    tokens_1d,
    batch_size,
    seq_len,
    device,
    num_batches=10,
    include=None,
    exclude=None,
):
    model.eval()
    calib = ActChannelCalibrator(percentiles=(0.99, 0.999), store_on_cpu=True)
    handles = attach_linear_input_hooks(
        model, calib, include_filter=include, exclude_filter=exclude
    )
    for _ in range(num_batches):
        x, _ = get_batch_1d(tokens_1d, batch_size, seq_len, device)
        _ = model(input_ids=x)

    for h in handles:
        h.remove()

    return calib.stats

def compute_smooth_scale(act_stats, weight_stats, alpha=0.5, smin=None, smax=None, eps=1e-8):
    smooth_params = {}

    for name, act_d in act_stats.items():
        if name not in weight_stats:
            continue  # or raise

        a = act_d["ap999"]
        w = weight_stats[name]
        a = a.abs().clamp_min(eps)
        w = w.abs().clamp_min(eps)

        s = (a.pow(alpha) / w.pow(1.0 - alpha)).to(torch.float32)

        if smin is not None:
            s = s.clamp_min(smin)
        if smax is not None:
            s = s.clamp_max(smax)

        smooth_params[name] = s

    return smooth_params

def quantize_model_smooth_dynamic(model, scale_stats, include, exclude):
    # gather first (avoid mutating while iterating)
    to_replace = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if not include(name) or exclude(name):
                continue
            if name in scale_stats:
                to_replace.append((name, mod))

    sat_stats = {}
    for name, mod in to_replace:
        new_mod = SmoothQuantDynamic.from_float(mod, scale=scale_stats[name])
        set_module_by_name(model, name, new_mod)
        sat_stats[name] = new_mod.last_w_sat_frac.item()

    return model, sat_stats

class SmoothQuantDynamic(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, has_bias: bool
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("last_x_clip", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_w_sat_frac", torch.tensor(0.0), persistent=False)

        self.register_buffer(
            "w_q",
            torch.empty((out_features, in_features), dtype=torch.int8),
            persistent=True,
        )
        self.register_buffer(
            "s_w", torch.empty((out_features, 1), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "scale", torch.empty((1, in_features), dtype=torch.float32), persistent=True
        )

        self.register_buffer(
            "has_bias", torch.tensor(int(has_bias), dtype=torch.uint8), persistent=True
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    @classmethod
    def from_float(cls, layer: nn.Linear, scale: torch.Tensor):
        out_f, in_f = layer.out_features, layer.in_features
        has_bias = layer.bias is not None
        device = layer.weight.device

        q = cls(in_features=in_f, out_features=out_f, has_bias=has_bias).to(
            device
        )

        # weights: symmetric per-output-channel
        W_s = (layer.weight * scale.to(device)).float()
        s_w, _ = calc_qparams_symmetric(W_s, bits=8, per_channel=True, axis=0)
        w_q, _ = quantize_symmetric(W_s, s_w, qmax=127, dtype=torch.int8)

        q.w_q.copy_(w_q.contiguous())
        q.s_w.copy_(s_w.to(torch.float32).contiguous())
        q.scale.copy_(scale.to(torch.float32).contiguous())

        sat = sat = ((w_q == 127) | (w_q == -127)).float().mean()
        q.last_w_sat_frac.copy_(sat)

        if has_bias:
            q.bias.data.copy_(layer.bias.detach().to(torch.float32))
        else:
            q.bias.data.zero_()

        return q

    def forward(self, x):
        orig_dtype = x.dtype
        x = x / self.scale
        x_fp = x.float()
        s_x, zp_x = calc_qparams_symmetric(
                x_fp, bits=8, per_channel=False, axis=None, qmax=127
            )
        x_q, x_clip = quantize_symmetric(x_fp, s_x, 127, dtype=torch.int8)

        x_centered = x_q.float() - zp_x.float()
        y_int = x_centered @ self.w_q.to(torch.float32).T
        y = y_int * (s_x.to(torch.float32) * self.s_w.to(torch.float32).T)

        if bool(self.has_bias.item()):
            y = y + self.bias.to(y.dtype)

        return y.to(orig_dtype)

# -----------------------------
# Main
# -----------------------------
def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(DEVICE)
    print("Using device:", device)
    print("Loading:", MODEL_NAME)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # OPT tokenizer sometimes has no pad token; not needed for this script

    # Load FP model (fp16)
    fp_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    fp_model.eval()

    # Load/encode text
    text = load_shakespeare(TEXT_PATH)
    enc = tok(text, return_tensors="pt")
    tokens = enc["input_ids"][0].to(device)  # [N]
    split = int(0.9 * tokens.numel())
    val_tokens = tokens[split:]

    # Cache eval batches (fixed)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    full_eval_data = []
    for _ in range(EVALUATION_BATCHES):
        x, y = get_batch_1d(val_tokens, BATCH_SIZE, MAX_SEQ_LEN, device=device)
        full_eval_data.append((x, y))

    # Evaluate FP
    fp_loss, fp_ppl = evaluate_simple(fp_model, full_eval_data)
    print(f"FP:    Loss:{fp_loss:.6f} | Perp:{fp_ppl:.6f}")

    # Evaluate FP32 GEMM
    fp32g_model = force_fp32_gemm_all_linears(
        copy.deepcopy(fp_model), include_filter=include, exclude_filter=exclude
    )
    loss, perp = evaluate_simple(fp32g_model, full_eval_data)
    print(f"FP32:    Loss:{loss:.6f} | Perp:{perp:.6f}")


    act_stats = compute_act_channel_stats(
        fp_model, val_tokens, BATCH_SIZE, MAX_SEQ_LEN, device, 10, include, exclude
    )
    weight_stats = compute_weight_channel_stats(fp_model, include, exclude)
    alphas = [.1, .25, .5, .75, .9]

    for alpha in alphas:
        q_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        q_model.eval()
        scale_stats = compute_smooth_scale(act_stats, weight_stats, alpha=alpha, smin=1.0, smax=10)
        # include_linear = lambda x: x.endswith("fc1") or x.endswith("fc2")
        q_model, sat_stats = quantize_model_smooth_dynamic(q_model, scale_stats, include, exclude)

        # sats = sorted(sat_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        # print(alpha)
        # for name, val in sats:
        #     print(f"{name}: {val:.4f}")

        # name = list(act_stats.keys())[0]
        # print(act_stats[name]['absmax'].shape)
        # print(weight_stats[name].shape)
        # print(scale_stats[name].shape)

        q_loss, q_ppl = evaluate_simple(q_model, full_eval_data)
        print(
            f"SmoothQuant Dynamic, alpha={alpha}:    Loss:{q_loss:.6f} | Perp:{q_ppl:.6f}"
        )
        # break

if __name__ == "__main__":
    main()
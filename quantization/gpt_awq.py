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
    attach_linear_input_hooks,
    get_module_by_name,
    set_module_by_name,
    quantize_model_w4,
)

import sys

sys.path.append(".")
from GPT.utils import load_shakespeare


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "facebook/opt-1.3b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
MAX_SEQ_LEN = 128
CALIBRATION_BATCHES = 2
EVALUATION_BATCHES = 20
SEED = 25

TEXT_PATH = "/workspace/ml-stuff/data/shakespeare.txt"

MAX_X_ROWS = 1024
GROUP_SIZE = 32


@torch.no_grad()
def compute_saliency_stats(
    model,
    tokens_1d,
    batch_size,
    seq_len,
    device,
    include=None,
    exclude=None,
):
    model.eval()
    calib = SaliencyAgg(store_on_cpu=True, max_rows=MAX_X_ROWS)
    handles = attach_linear_input_hooks(
        model, calib, include_filter=include, exclude_filter=exclude
    )
    for _ in range(CALIBRATION_BATCHES):
        x, _ = get_batch_1d(tokens_1d, batch_size, seq_len, device)
        _ = model(input_ids=x)

    for h in handles:
        h.remove()

    return calib.finalize()


class AWQW4Grouped(nn.Module):
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
            persistent=True,
        )

        self.register_buffer(
            "s", torch.empty((in_features,), dtype=torch.float32), persistent=True
        )

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    @classmethod
    def from_float(cls, layer: nn.Linear, s: torch.Tensor, group_size: int):
        out_f, in_f = layer.out_features, layer.in_features
        n_groups = in_f // group_size
        has_bias = layer.bias is not None
        device = layer.weight.device

        q = cls(
            in_features=in_f,
            out_features=out_f,
            has_bias=has_bias,
            group_size=group_size,
        ).to(device)

        W = layer.weight.data.clone().float()
        W_g = W.view(out_f, n_groups, group_size)
        s_view = s.view(1, n_groups, group_size)

        w_scaled = W_g * s_view
        scales = w_scaled.abs().amax(dim=-1, keepdim=True)
        scales = scales / 7.0
        scales = scales.clamp(min=1e-5)

        w_int = (w_scaled / scales).round().clamp(-7, 7)

        q.w_q.copy_(w_int.to(torch.int8))
        q.scales.copy_(scales)
        q.s.copy_(s.float().contiguous())

        if has_bias:
            q.bias.data.copy_(layer.bias.detach().to(torch.float32))
        else:
            q.bias.data.zero_()

        return q

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype

        w_recon = self.w_q.float() * self.scales
        w_recon = w_recon.reshape(self.out_features, self.in_features)

        x = x.float() / self.s
        y = x @ w_recon.T

        if self.has_bias:
            y = y + self.bias.to(y.dtype)

        return y.to(orig_dtype)


@torch.no_grad()
def scaling_search(
    model, saliency_stats, k_ratio=0.1, group_size=32, smin=None, smax=None
):
    scale_dict = {}
    device = next(model.parameters()).device

    for name, value in saliency_stats.items():
        X = value["act_batch"].to(device).float()

        layer = get_module_by_name(model, name)
        W = layer.weight.data.clone().float()  # [Out, In]
        out_feat, in_feat = W.shape

        a_j = value["amean"].to(device).reshape(-1).float()
        k = max(1, int(k_ratio * in_feat))
        topk_indices = torch.topk(a_j, k=k).indices

        y_orig = X @ W.T

        best_mse = float("inf")
        final_s = None
        s_base = torch.ones(in_feat, device=device)

        n_groups = in_feat // group_size
        W_g = W.view(out_feat, n_groups, group_size)

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            s_val = (a_j / (a_j.mean() + 1e-8)).pow(alpha)
            if smin is not None:
                s_val = s_val.clamp_min(smin)
            if smax is not None:
                s_val = s_val.clamp_max(smax)

            s = s_base.clone()
            s[topk_indices] = s_val[topk_indices]
            s_view = s.view(1, n_groups, group_size)

            w_scaled = W_g * s_view
            scales = w_scaled.abs().amax(dim=-1, keepdim=True)
            scales = scales / 7.0
            scales = scales.clamp(min=1e-5)

            w_q_scaled = (w_scaled / scales).round().clamp(-7, 7) * scales
            w_q_hat = w_q_scaled.reshape(out_feat, in_feat)

            y_alt = (X / s) @ w_q_hat.T

            mse = (y_orig - y_alt).square().mean().item()
            if mse < best_mse:
                best_mse = mse
                final_s = s

        scale_dict[name] = final_s

    return scale_dict


class SaliencyAgg:
    def __init__(self, store_on_cpu=True, max_rows=8000, dtype=torch.float32):
        self.store_on_cpu = store_on_cpu
        self.max_rows = max_rows
        self.dtype = dtype
        self.act = {}

    @torch.no_grad()
    def observe(self, name: str, x: torch.Tensor):
        # x: [..., d]
        X = x.detach().reshape(-1, x.shape[-1]).float()  # [N, d]
        if self.max_rows is not None and X.size(0) > self.max_rows:
            idx = torch.randint(0, X.size(0), (self.max_rows,), device=X.device)
            X = X[idx]

        X = X.to(dtype=self.dtype)
        if self.store_on_cpu:
            X = X.cpu()

        if name not in self.act:
            self.act[name] = X

    @torch.no_grad()
    def finalize(self):
        stats_dict = {}
        for name in self.act:
            amean = self.act[name].abs().mean(dim=0)
            ap999 = torch.quantile(self.act[name].abs(), q=0.999, dim=0)
            stats_dict[name] = {
                "amean": amean,
                "ap999": ap999,
                "act_batch": self.act[name][:256],
            }
        return stats_dict


def quantize_model_awq(model, scale_stats, include, exclude):
    to_replace = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if not include(name) or exclude(name) or name not in scale_stats:
                continue
            to_replace.append((name, mod))

    for name, mod in to_replace:
        s = scale_stats[name]
        new_mod = AWQW4Grouped.from_float(mod, s, GROUP_SIZE)
        set_module_by_name(model, name, new_mod)

    return model


def eval_harness(tokens, full_eval_data, device, include, exclude, s=""):
    fp_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    fp_model.eval()

    q_model = quantize_model_w4(fp_model, GROUP_SIZE, include, exclude)
    loss, perp = evaluate_simple(q_model, full_eval_data)
    print(f"W4 {s}:  Loss:{loss:.6f} | Perp:{perp:.6f}")

    fp_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    fp_model.eval()
    saliency_stats = compute_saliency_stats(
        fp_model,
        tokens,
        BATCH_SIZE,
        MAX_SEQ_LEN,
        device,
        include=include,
        exclude=exclude,
    )
    scale_dict = scaling_search(
        fp_model,
        saliency_stats,
        k_ratio=0.1,
        group_size=GROUP_SIZE,
        smin=0.1,
        smax=10.0,
    )
    q_model = quantize_model_awq(fp_model, scale_dict, include, exclude)
    loss, perp = evaluate_simple(q_model, full_eval_data)
    print(f"AWQ4 {s}:  Loss:{loss:.6f} | Perp:{perp:.6f}")


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

    fp_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    fp_model.eval()

    text = load_shakespeare(TEXT_PATH)
    enc = tok(text, return_tensors="pt")
    tokens = enc["input_ids"][0].to(device)
    split = int(0.9 * tokens.numel())
    val_tokens = tokens[split:]

    # fixed eval set
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    full_eval_data = []
    for _ in range(EVALUATION_BATCHES):
        x, y = get_batch_1d(val_tokens, BATCH_SIZE, MAX_SEQ_LEN, device=device)
        full_eval_data.append((x, y))

    fp_loss, fp_ppl = evaluate_simple(fp_model, full_eval_data)
    print(f"FP:    Loss:{fp_loss:.6f} | Perp:{fp_ppl:.6f}")

    fp32g_model = force_fp32_gemm_all_linears(
        copy.deepcopy(fp_model), include_filter=include, exclude_filter=exclude
    )
    loss, perp = evaluate_simple(fp32g_model, full_eval_data)
    print(f"FP32:  Loss:{loss:.6f} | Perp:{perp:.6f}")

    include_linear = lambda x: x.endswith("fc1")
    eval_harness(val_tokens, full_eval_data, device, include_linear, exclude, "fc1")

    include_linear = lambda x: x.endswith("fc1") or x.endswith("fc2")
    eval_harness(val_tokens, full_eval_data, device, include_linear, exclude, "MLP")

    eval_harness(val_tokens, full_eval_data, device, include, exclude, "full")


if __name__ == "__main__":
    main()

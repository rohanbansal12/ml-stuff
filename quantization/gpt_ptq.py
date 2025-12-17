import torch
import torch.nn.functional as F
import copy

from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq_utils import (
    QuantLinearW4Grouped,
    evaluate_simple,
    get_module_by_name,
    capture_single_activation,
    get_batch_1d,
    force_fp32_gemm_all_linears,
    calibrate,
    quantize_model_w8a8,
    quantize_model_w4,
    include,
    exclude
)

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

    # -----------------------------
    # QuantLinearW8A8 Experiments
    # -----------------------------

    # Calibrate activation stats (on FP model)
    act_stats = calibrate(
        fp_model,
        val_tokens,
        BATCH_SIZE,
        MAX_SEQ_LEN,
        device,
        num_batches=CALIBRATION_BATCHES,
    )
    # print("Calibrated linears:", len(act_stats))

    # Get overall losses for various quantization approaches
    for mode in ["W8", "W8A8_dynamic", "W8A8_static"]:
        # Load a fresh model for quantization (avoid deepcopy on a 1.3B model)
        q_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        q_model.eval()

        # Quantize selected linears in-place
        q_model = quantize_model_w8a8(q_model, act_stats, mode, include, exclude)

        # Evaluate Quant
        q_loss, q_ppl = evaluate_simple(q_model, full_eval_data)
        print(
            f"{mode}:     Loss:{q_loss:.6f} | Perp:{q_ppl:.6f} | Delta: (Quant-FP) loss: {q_loss - fp_loss:+.6e}"
        )

    # # run one forward to populate last_x_clip
    # clip_stats = {}
    # for i in range(10):
    #     x, y = full_eval_data[i]
    #     with torch.no_grad():
    #         _ = q_model(input_ids=x).logits

    #     for name, mod in q_model.named_modules():
    #         if hasattr(mod, "last_x_clip"):
    #             v = mod.last_x_clip
    #             # only keep real scalars
    #             if torch.is_tensor(v) and v.numel() == 1:
    #                 clip_stats[name] = clip_stats.get('name', 0.0) + float(v.detach().cpu().item())

    # for name, value in clip_stats.items():
    #     clip_stats[name] = value / 10
    # top = sorted(clip_stats.items(), key=lambda kv: kv[1], reverse=True)[:10]
    # for name, v in top:
    #     print(f"{name:60s} clip_frac={v:.6f}")

    # # compare single layer outputs
    # layer_name = "model.decoder.layers.1.self_attn.out_proj"
    # layer_fp = get_module_by_name(fp_model, layer_name)
    # layer_q = get_module_by_name(q_model, layer_name)

    # layer_fp.eval()
    # layer_q.eval()

    # saved_x = {}

    # def hook_fn(mod, inp, out):
    #     saved_x["x"] = inp[0].detach()

    # h = layer_fp.register_forward_hook(hook_fn)

    # # run one real batch through the FP model to populate saved_x
    # x_tokens, _ = full_eval_data[0]
    # with torch.no_grad():
    #     _ = fp_model(input_ids=x_tokens).logits

    # h.remove()

    # x_real = saved_x["x"]
    # print("captured shape:", x_real.shape, "dtype:", x_real.dtype)

    # with torch.no_grad():
    #     y_fp = layer_fp(x_real)
    #     y_q  = layer_q(x_real)

    # diff = (y_fp - y_q).abs()
    # cos  = F.cosine_similarity(y_fp.float(), y_q.float(), dim=-1).mean().item()
    # rel  = diff.float().mean() / (y_fp.float().abs().mean() + 1e-8)

    # print(layer_name)
    # print(f"max_abs={diff.max().item():.6e}  mean_abs={diff.mean().item():.6e}  rel_mean={rel.item():.6e}  cos={cos:.6f}")
    # print("clip_frac on real x:", layer_q.last_x_clip.item())


    # -----------------------------
    # QuantLinearW4 Experiments
    # -----------------------------

    # try W4 layer with various group_sizes on rand data (ensure layer math works)
    layer_name = "model.decoder.layers.1.fc1"
    layer_fp = get_module_by_name(fp_model, layer_name)

    x_real = capture_single_activation(fp_model, layer_name, full_eval_data[0][0])
    x = x_real.reshape(-1, x_real.size(-1))

    for group_size in [32, 64, 128, 256]:
        layer_q = QuantLinearW4Grouped.from_float(layer_fp, group_size)
        W = layer_fp.weight
        W_hat = layer_q.w_q.to(layer_q.s_w.dtype) * layer_q.s_w.T.unsqueeze(-1)
        W_hat = W_hat.reshape(W.shape)
        diff = (W_hat - W).abs()
        cos  = F.cosine_similarity(W_hat.float(), W.float(), dim=-1).mean().item()
        print(f"{layer_name} W4, size={group_size} W: max_abs={diff.max().item():.6e} | mean_abs={diff.mean().item():.6e} | cos={cos:.6f}")

        y_fp = layer_fp(x)
        y_q = layer_q(x)
        diff = (y_fp - y_q).abs()
        cos  = F.cosine_similarity(y_fp.float(), y_q.float(), dim=-1).mean().item()
        rel  = diff.float().mean() / (y_fp.float().abs().mean() + 1e-8)
        print(f"{layer_name} W4, size={group_size} x: max_abs={diff.max().item():.6e} | mean_abs={diff.mean().item():.6e} | rel_mean={rel.item():.6e} | cos={cos:.6f}")

    # quantize all MLP layers and compare loss/perp
    include_linear = lambda x: x.endswith("fc1") or x.endswith("fc2")
    group_sizes = [32, 64, 128, 256]
    for group_size in group_sizes:
        q_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        q_model.eval()

        q_model = quantize_model_w4(q_model, group_size, include_linear, exclude)
        q_loss, q_ppl = evaluate_simple(q_model, full_eval_data)
        print(f"MLP W4, size={group_size}: Loss:{q_loss:.6f} | Perp:{q_ppl:.6f}")

    # quantize only fc1 MLP layers and compare loss/perp
    include_linear = lambda x: x.endswith("fc1")
    group_sizes = [32, 64, 128, 256]
    for group_size in group_sizes:
        q_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        q_model.eval()

        q_model = quantize_model_w4(q_model, group_size, include_linear, exclude)
        q_loss, q_ppl = evaluate_simple(q_model, full_eval_data)
        print(f"MLP fc1 W4, size={group_size}: Loss:{q_loss:.6f} | Perp:{q_ppl:.6f}")

    # quantize only fc2 MLP layers and compare loss/perp
    include_linear = lambda x: x.endswith("fc2")
    group_sizes = [32, 64, 128, 256]
    for group_size in group_sizes:
        q_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        q_model.eval()

        q_model = quantize_model_w4(q_model, group_size, include_linear, exclude)
        q_loss, q_ppl = evaluate_simple(q_model, full_eval_data)
        print(f"MLP fc2 W4, size={group_size}: Loss:{q_loss:.6f} | Perp:{q_ppl:.6f}")


if __name__ == "__main__":
    main()

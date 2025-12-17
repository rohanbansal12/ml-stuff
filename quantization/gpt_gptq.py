import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq_utils import (
    evaluate_simple,
    get_batch_1d,
    force_fp32_gemm_all_linears,
    include,
    exclude,
    group_weights,
    calc_group_scales,
    quantize_grouped,
    HessianAgg,
    attach_one_linear_input_hook,
    set_module_by_name,
    get_module_by_name,
    GPTQLinear
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
CALIBRATION_BATCHES = 4
EVALUATION_BATCHES = 20
SEED = 25

TEXT_PATH = "/workspace/ml-stuff/data/shakespeare.txt"

MAX_X_ROWS = 2048
DAMP_RATIO = 1e-2
GROUP_SIZE = 32
BLOCK_SIZE = 128
ACT_ORDER = False

def naive_quantize_row_w4_grouped(
    w: torch.Tensor, group_size: int = 32, qmax: int = 7, eps: float = 1e-8
):
    """
    w: [in]
    returns:
      w_hat: [in]
      w_q:  [n_groups, group_size] int8
      s_w:  [n_groups, 1] float
    """
    # 1) reshape row to [1, in] so we can reuse group_weights
    w2 = w.view(1, -1)  # [1, in]

    # 2) grouped view: [1, n_groups, group_size]
    w_g = group_weights(w2, group_size)  # [1, g, k]

    # 3) scales + quant
    s_w = calc_group_scales(w_g, qmax=qmax, eps=eps)  # [1, g, 1]
    w_q = quantize_grouped(w_g, s_w, qmin=-qmax, qmax=qmax)  # [1, g, k]

    # 4) symmetric dequant: w_hat = q * s
    w_hat_g = w_q.float() * s_w  # [1, g, k]
    w_hat = w_hat_g.view(-1)  # [in]

    # 5) drop leading batch dim for q/s
    return w_hat, w_q[0], s_w[0]


@torch.no_grad()
def quantize_gptq_row(
    w: torch.Tensor,  # [d]
    H: torch.Tensor,  # [d, d]
    bits: int = 4,
    group_size: int = 32,
    qmin: int = -7,
    qmax: int = 7,
    damp_ratio: float = 1e-4,
    eps: float = 1e-8,
    act_order: bool = False,  # optional extension
):
    """
    GPTQ row quantization, 'conceptual' version:
      1) damp H
      2) build Q = H^{-1}
      3) sequentially quantize groups and compensate the remaining weights:
            w_tail -= Q21 @ (Q11^{-1} e_blk)

    Notes:
      - This matches the math most directly, but building Q is O(d^3) and heavy.
      - Later you’ll replace "build Q" with triangular solves using a Cholesky factor.
    """
    assert w.ndim == 1
    d = w.numel()
    assert H.shape == (d, d)
    assert d % group_size == 0

    device = w.device
    w_work = w.detach().float().clone()
    H = H.detach().float().to(device)

    # optional act-order permutation
    perm = inv_perm = None
    if act_order:
        diagH = torch.diagonal(H)
        perm = torch.argsort(diagH, descending=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(d, device=device)
        w_work = w_work[perm]
        H = H[perm][:, perm]

    # ---- damp H to improve conditioning ----
    diag_mean = torch.mean(torch.diag(H))
    if diag_mean > 0:
        H = H + damp_ratio * diag_mean * torch.eye(d, device=device, dtype=H.dtype)

    # ---- build Q = H^{-1} (SPD expected after damping) ----
    # Use Cholesky: H = L L^T, then Q = H^{-1} = (L^{-T} L^{-1})
    L = torch.cholesky(H)
    Q = torch.cholesky_inverse(L)
    Q = 0.5 * (Q + Q.T)  # symmetrize for numerical tidiness

    w_hat = torch.zeros_like(w_work)
    sat_count = 0

    # scalar-per-group W4 quantization (matches your naive baseline behavior)
    def _quantize_grouped_vec(v: torch.Tensor):
        # v: [gs]
        amax = v.abs().amax()
        s = torch.clamp(amax / float(qmax), min=eps)  # scalar
        q = torch.round(v / s).clamp(qmin, qmax).to(torch.int8)
        v_hat = q.float() * s
        return v_hat, q

    n_groups = d // group_size
    for g in range(n_groups):
        i0 = g * group_size
        i1 = i0 + group_size
        idx_blk = slice(i0, i1)
        idx_tail = slice(i1, d)

        # 1) quantize this block
        w_blk = w_work[idx_blk]  # [gs]
        w_hat_blk, q_blk = _quantize_grouped_vec(w_blk)
        w_hat[idx_blk] = w_hat_blk

        e_blk = w_blk - w_hat_blk  # [gs]
        sat_count += (q_blk == qmin).sum().item() + (q_blk == qmax).sum().item()

        # 2) compensate the remaining weights using Q solves
        if i1 < d:
            Q11 = Q[idx_blk, idx_blk]  # [gs, gs]
            Q21 = Q[idx_tail, idx_blk]  # [d-i1, gs]

            # u = Q11^{-1} e_blk
            # (Q11 should be SPD; if this fails, damping is too small or H is wrong)
            Lq = torch.cholesky(Q11)
            u = torch.cholesky_solve(e_blk[:, None], Lq)[:, 0]  # [gs]

            # w_tail <- w_tail - Q21 u
            w_work[idx_tail] = w_work[idx_tail] - Q21 @ u

    if act_order:
        w_hat = w_hat[inv_perm]

    info = {"sat_frac": float(sat_count) / float(d)}
    return w_hat, info


def row_metrics(w, w_hat, eps=1e-8):
    diff = (w - w_hat).abs()
    cos = torch.nn.functional.cosine_similarity(w.float(), w_hat.float(), dim=0).item()
    rel = diff.float().mean().item() / (w.float().abs().mean().item() + eps)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_mean": rel,
        "cos": cos,
    }


def row_metrics_weighted(w, w_hat, H):
    e = (w - w_hat).float()
    Hf = H.float().to(e.device)
    # scalar quadratic form
    q = float(e @ (Hf @ e))
    # also report RMS output error on the calibration batch:
    # ||X e||^2 / N == e^T H e
    return {"eTHe": q, "rmse_out": (q**0.5)}


def quantize_grouped_vec(v: torch.Tensor, qmin: int, qmax: int, eps: float):
    # v: [gs]
    amax = v.abs().amax()
    s = torch.clamp(amax / float(qmax), min=eps)  # scalar
    q = torch.round(v / s).clamp(qmin, qmax).to(torch.int8)
    v_hat = q.float() * s
    return v_hat, q, s


@torch.no_grad()
def gptq_quantize_linear_one_layer(
    fp_linear: torch.nn.Linear,
    H_cpu_f32: torch.Tensor,
    qmin: int = -7,
    qmax: int = 7,
    eps: float = 1e-8,
    damp_ratio: float = 0.01,  # Default for OPT/Llama usually 0.01 or 0.1
    group_size=128,
    block_size=128,
    act_order=False,
):
    W = fp_linear.weight.detach().cuda().float()
    out_feat, in_feat = W.shape
    device = W.device
    
    # Move H to GPU once
    H = H_cpu_f32.detach().cuda().float()

    # 1. Handle Act Order (Permutation)
    perm = None
    if act_order:
        perm = torch.argsort(torch.diagonal(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
    
    # Cholesky + Inverse
    L = torch.linalg.cholesky(H)
    Q = torch.cholesky_inverse(L)
    Q = 0.5 * (Q + Q.T) # Symmetrize

    # 3. Storage
    n_groups = in_feat // group_size
    W_q = torch.empty((out_feat, n_groups, group_size), dtype=torch.int8, device=device)
    S_w = torch.empty((out_feat, n_groups, 1), dtype=torch.float32, device=device)

    # Error buffer for the global block update
    # We reuse this to store 'err' to avoid repeatedly slicing 'Losses'
    Losses = torch.zeros((out_feat, block_size), device=device)

    # ---------------------------------------------------------
    # Outer Loop: Iterate by BLOCK
    # ---------------------------------------------------------
    for i in range(0, in_feat, block_size):
        i_end = min(i + block_size, in_feat)
        count = i_end - i
        
        Q_block = Q[i:i_end, i:i_end]
        Q_cross = Q[i:i_end, i_end:]

        # Create a view of the current working block to avoid indexing W repeatedly
        W_block = W[:, i:i_end]

        # -----------------------------------------------------
        # Inner Loop: Sequential Column Quantization
        # -----------------------------------------------------
        for j in range(count):
            col_global = i + j
            
            # A. Calculate Scales (Only at group boundaries)
            if col_global % group_size == 0:
                # Look ahead in the main W matrix
                g_end = min(col_global + group_size, in_feat)
                # Slice and compute max (per row)
                # 1e-5 prevents division by zero
                scale = W[:, col_global:g_end].abs().amax(dim=1, keepdim=True)
                scale = (scale / qmax).clamp(min=eps)
                S_w[:, col_global // group_size, :] = scale

            # B. Quantize Single Column
            s = S_w[:, col_global // group_size, 0] # Fetch pre-calc scale
            w = W_block[:, j]                       # Current column
            
            # Quantize
            q = (w / s).round().clamp(qmin, qmax)
            w_hat = q * s
            
            # Store int8 result
            W_q[:, col_global // group_size, col_global % group_size] = q.to(torch.int8)

            # C. Calculate Error & Update Local Block
            d = Q_block[j, j]
            err = (w - w_hat) / d
            
            # Store error in Losses buffer
            Losses[:, j] = err

            # --- OPTIMIZATION STARTS HERE ---
            # Update remaining columns in this block.
            # Old way: W_work -= err @ Q
            # New way: In-place Outer Product (addr_)
            # This fuses the outer product and subtraction, avoiding allocation.
            if j < count - 1:
                 # W_block slice: [Out, Remaining]
                 # err:           [Out]
                 # Q_row:         [Remaining]
                 # Performs: W_slice = W_slice - 1.0 * (err x Q_row)
                 W_block[:, (j + 1):].addr_(err, Q_block[j, (j + 1):], alpha=-1.0)
            
            # Update the column in W_block to be the error (standard GPTQ trick)
            # This prepares W_block to be used for the global update
            W_block[:, j] = err
            # --- OPTIMIZATION ENDS HERE ---

        # -----------------------------------------------------
        # D. Global Batch Update
        # -----------------------------------------------------
        if i_end < in_feat:
            # Update the tail of the matrix
            # W_block now holds the errors (from the assignment W_block[:, j] = err)
            W[:, i_end:] -= torch.matmul(W_block, Q_cross)

    new_layer = GPTQLinear.from_float(fp_linear, W_q, S_w, group_size, perm)
    return new_layer
    
# -----------------------------
# Streaming driver: one layer at a time
# -----------------------------
@torch.no_grad()
def run_gptq_streaming_fc1_sequential(fp_model, tokens_1d, device):
    q_model = copy.deepcopy(fp_model).to(device)
    q_model.eval()

    # decide layer order once (stable, forward-ish)
    fc1_names = sorted([n for n, m in q_model.named_modules()
                        if isinstance(m, torch.nn.Linear) and n.endswith("fc1")])

    print(f"Found {len(fc1_names)} fc1 layers")

    for li, name in enumerate(fc1_names, 1):
        # 1) collect Hessian for THIS layer on the CURRENT q_model
        agg = HessianAgg(store_on_cpu=True, max_rows=MAX_X_ROWS, dtype=torch.float64)
        h = attach_one_linear_input_hook(q_model, name, agg)

        for _ in range(CALIBRATION_BATCHES):
            x, _ = get_batch_1d(tokens_1d, BATCH_SIZE, MAX_SEQ_LEN, device=device)
            _ = q_model(input_ids=x)

        h.remove()

        H = agg.finalize(damp_ratio=DAMP_RATIO)

        # 2) quantize this layer using its Hessian
        fp_linear = get_module_by_name(q_model,  name)
        new_mod = gptq_quantize_linear_one_layer(
            fp_linear, H,
            qmin=-7, qmax=7, eps=1e-8,
            group_size=GROUP_SIZE,
            block_size=BLOCK_SIZE,
            act_order=ACT_ORDER,
        )
        set_module_by_name(q_model, name, new_mod)

        # 3) cleanup
        del agg
        del H
        if li % 4 == 0 or li == len(fc1_names):
            print(f"Processed {li}/{len(fc1_names)} layers")

    return q_model



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

    q_model = run_gptq_streaming_fc1_sequential(fp_model, val_tokens, device)
    loss, perp = evaluate_simple(q_model, full_eval_data)
    print(f"GPTQ fc1:  Loss:{loss:.6f} | Perp:{perp:.6f}")

    # # testing the GPTQ logic on a single row
    # act_map = collect_fc1_activations(fp_model, val_tokens, device)
    # print(f"Collected X for {len(act_map)} fc1 layers")

    # layer_name = "model.decoder.layers.1.fc1"
    # layer = get_module_by_name(fp_model, layer_name)
    # W = layer.weight.detach().float()
    # w0 = W[0]
    # X = act_map[layer_name]
    # X64 = X.double()
    # H = (X64.T @ X64) / X64.size(0)

    # w_naive, w_q, s_w = naive_quantize_row_w4_grouped(
    #     w0, group_size=32, qmax=7, eps=1e-8
    # )
    # print(row_metrics(w0, w_naive))
    # print(row_metrics_weighted(w0, w_naive, H))
    # w_gptq, info = quantize_gptq_row(
    #     w0, H, bits=4, group_size=32, qmin=-7, qmax=7, eps=1e-8, act_order=False
    # )
    # print(row_metrics(w0, w_gptq))
    # print(row_metrics_weighted(w0, w_gptq, H))




if __name__ == "__main__":
    main()

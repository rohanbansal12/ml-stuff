import sys
import torch
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenize import tokenize_preference_example
from data.debug import debug_print_tokenized_item
from data.collate import collate_preference_batch
from engine import load_tokenizer, completion_logprobs, load_model
from typing import Sequence, Dict, Any

def test_row_extraction_invariance(
    model, tokenizer, batch, *, which="chosen", atol=1e-4
):
    device = next(model.parameters()).device
    model.eval()

    input_ids = batch[f"{which}_input_ids"].to(device)
    attn = batch[f"{which}_attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    # batch scores
    sum_batch, _, _, _ = completion_logprobs(model, input_ids, attn, prompt_lens)
    sum_batch = sum_batch.detach().cpu()

    # score each *padded row* individually (same exact tensor values)
    sum_rows = []
    for i in range(input_ids.size(0)):
        ids_i = input_ids[i:i+1]
        attn_i = attn[i:i+1]
        pl_i = prompt_lens[i:i+1]
        s_i, _, _, _ = completion_logprobs(model, ids_i, attn_i, pl_i)
        sum_rows.append(s_i[0].detach().cpu())
    sum_rows = torch.stack(sum_rows, dim=0)

    print(f"\nRow-extraction check ({which}):")
    print("batch:", sum_batch.tolist())
    print("rows :", sum_rows.tolist())
    torch.testing.assert_close(sum_batch, sum_rows, atol=atol, rtol=0.0)

def test_padding_invariance_end_to_end(
    model,
    tokenizer,
    *,
    examples: Sequence[Dict[str, Any]],
    max_len: int = 256,
    atol: float = 1e-4,
    rtol: float = 0.0,
    verbose: bool = True,
):
    """
    End-to-end test: padding should not change response-only logprob.

    Pipeline:
      1) tokenize_preference_example for each raw example (chosen+rejected)
      2) collate_preference_batch -> padded batch (left padding)
      3) compute sum_logp for chosen/rejected in batch
      4) compute sum_logp for chosen/rejected per-example without padding
      5) assert equality (within tolerance)

    Requirements:
      - tokenize_preference_example(...)
      - collate_preference_batch(...)
      - completion_logprobs(model, input_ids, attention_mask, prompt_lens)
        returning sum_logp as first output
    """
    assert tokenizer.padding_side == "left", "Tokenizer must be left-padding for this test."

    # -----------------------------
    # 1) Tokenize examples (no padding yet)
    # -----------------------------
    tokenized = [tokenize_preference_example(tokenizer, ex, max_len=max_len) for ex in examples]

    # -----------------------------
    # 2) Collate (introduces padding)
    # -----------------------------
    batch = collate_preference_batch(tokenizer, tokenized)
    test_row_extraction_invariance(model, tokenizer, batch, which="chosen")
    test_row_extraction_invariance(model, tokenizer, batch, which="rejected")

    # Move batch to model device
    device = next(model.parameters()).device
    chosen_input_ids = batch["chosen_input_ids"].to(device)
    chosen_attention_mask = batch["chosen_attention_mask"].to(device)
    rejected_input_ids = batch["rejected_input_ids"].to(device)
    rejected_attention_mask = batch["rejected_attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    # -----------------------------
    # 3) Compute batch scores (padded)
    # -----------------------------
    sum_logp_c_batch, mean_logp_c_batch, _, _ = completion_logprobs(
        model, chosen_input_ids, chosen_attention_mask, prompt_lens
    )
    sum_logp_r_batch, mean_logp_r_batch, _, _ = completion_logprobs(
        model, rejected_input_ids, rejected_attention_mask, prompt_lens
    )

    # -----------------------------
    # 4) Compute per-example scores (no padding)
    # -----------------------------
    sum_logp_c_single = []
    sum_logp_r_single = []

    with torch.no_grad():
        for i, item in enumerate(tokenized):
            # chosen
            ci = item.chosen_input_ids_1d.to(device).unsqueeze(0)           # [1, T_i]
            ca = item.chosen_attention_mask_1d.to(device).unsqueeze(0)      # [1, T_i]
            pl = torch.tensor([item.prompt_len], device=device, dtype=torch.long)

            slp_c, _, _, _ = completion_logprobs(model, ci, ca, pl)
            sum_logp_c_single.append(slp_c[0].detach().cpu())

            # rejected
            ri = item.rejected_input_ids_1d.to(device).unsqueeze(0)
            ra = item.rejected_attention_mask_1d.to(device).unsqueeze(0)

            slp_r, _, _, _ = completion_logprobs(model, ri, ra, pl)
            sum_logp_r_single.append(slp_r[0].detach().cpu())

    sum_logp_c_single = torch.stack(sum_logp_c_single, dim=0)  # [B]
    sum_logp_r_single = torch.stack(sum_logp_r_single, dim=0)  # [B]

    # Bring batch scores back to CPU for comparison
    sum_logp_c_batch_cpu = sum_logp_c_batch.detach().cpu()
    sum_logp_r_batch_cpu = sum_logp_r_batch.detach().cpu()

    # -----------------------------
    # 5) Compare + assert
    # -----------------------------
    if verbose:
        print("=" * 80)
        print("Padding invariance test")
        print(f"B={len(tokenized)} | max_len={max_len}")
        print("-" * 80)
        for i in range(len(tokenized)):
            dc = float(sum_logp_c_batch_cpu[i] - sum_logp_c_single[i])
            dr = float(sum_logp_r_batch_cpu[i] - sum_logp_r_single[i])
            print(
                f"[{i}] chosen: batch={float(sum_logp_c_batch_cpu[i]): .6f} "
                f"single={float(sum_logp_c_single[i]): .6f} diff={dc: .6e}"
            )
            print(
                f"    rejected: batch={float(sum_logp_r_batch_cpu[i]): .6f} "
                f"single={float(sum_logp_r_single[i]): .6f} diff={dr: .6e}"
            )
        print("=" * 80)

    torch.testing.assert_close(sum_logp_c_batch_cpu, sum_logp_c_single, rtol=rtol, atol=atol)
    torch.testing.assert_close(sum_logp_r_batch_cpu, sum_logp_r_single, rtol=rtol, atol=atol)

    if verbose:
        print("✅ Padding invariance PASSED (chosen + rejected).")

    return True

if __name__ == "__main__":
    tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    device = torch.device("cuda")
    model = load_model("Qwen/Qwen2.5-0.5B-Instruct", torch.float32, device)
    example = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain why the sky is blue."},
        ],
        "chosen": "The sky appears blue because shorter wavelengths of light scatter more in the atmosphere...",
        "rejected": "The sky is blue because it reflects the ocean.",
    }

    ex = tokenize_preference_example(tokenizer, example, max_len=512)
    debug_print_tokenized_item(tokenizer, ex)

    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain why the sky is blue."},
            ],
            "chosen": "The sky appears blue because shorter wavelengths scatter more strongly in the atmosphere (Rayleigh scattering).",
            "rejected": "It reflects the ocean.",
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Give three tips for learning CUDA."},
            ],
            "chosen": "1) Learn the memory hierarchy. 2) Profile kernels and optimize occupancy. 3) Minimize divergence and coalesce loads.",
            "rejected": "Use a GPU and it will be fast.",
        },
    ]

    test_padding_invariance_end_to_end(model, tokenizer, examples=examples, max_len=256, verbose=True)
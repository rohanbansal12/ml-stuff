import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenize import tokenize_preference_example
from data.collate import collate_preference_batch
from engine import load_tokenizer, completion_logprobs, load_model

tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
device = torch.device("cuda")
model = load_model("Qwen/Qwen2.5-0.5B-Instruct", torch.bfloat16, device)

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

# Tokenize
tokenized = [tokenize_preference_example(tokenizer, ex, max_len=256) for ex in examples]

print("=" * 80)
print("TOKENIZED EXAMPLES (before collation)")
print("=" * 80)
for i, item in enumerate(tokenized):
    print(f"Example {i}:")
    print(f"  prompt_len: {item.prompt_len}")
    print(f"  chosen length: {item.chosen_input_ids_1d.numel()}")
    print(f"  rejected length: {item.rejected_input_ids_1d.numel()}")
    print(f"  chosen_completion_len: {item.chosen_completion_len}")
    print(f"  rejected_completion_len: {item.rejected_completion_len}")
    print()

# Collate
batch = collate_preference_batch(tokenizer, tokenized)

print("=" * 80)
print("COLLATED BATCH")
print("=" * 80)
print(f"Batch shape: {batch['chosen_input_ids'].shape}")
print(f"Prompt lens: {batch['prompt_lens']}")
print()

# Check padding for example 0
B, T = batch['chosen_input_ids'].shape
print("Example 0 - Chosen sequence:")
print(f"  Attention mask: {batch['chosen_attention_mask'][0]}")
print(f"  First 10 input_ids: {batch['chosen_input_ids'][0, :10]}")
print(f"  Last 10 input_ids: {batch['chosen_input_ids'][0, -10:]}")
print()

# Compute scores in batch
chosen_input_ids = batch["chosen_input_ids"].to(device)
chosen_attention_mask = batch["chosen_attention_mask"].to(device)
prompt_lens = batch["prompt_lens"].to(device)

sum_logp_batch, mean_logp_batch, token_logp_batch, completion_mask_batch = completion_logprobs(
    model, chosen_input_ids, chosen_attention_mask, prompt_lens
)

print("=" * 80)
print("BATCH COMPUTATION")
print("=" * 80)
print(f"sum_logp (batch): {sum_logp_batch}")
print(f"Completion mask sum per example: {completion_mask_batch.sum(dim=1)}")
print()

# Compute score for example 0 individually
item0 = tokenized[0]
ci = item0.chosen_input_ids_1d.to(device).unsqueeze(0)
ca = item0.chosen_attention_mask_1d.to(device).unsqueeze(0)
pl = torch.tensor([item0.prompt_len], device=device, dtype=torch.long)

sum_logp_single, mean_logp_single, token_logp_single, completion_mask_single = completion_logprobs(
    model, ci, ca, pl
)

print("=" * 80)
print("SINGLE COMPUTATION (example 0)")
print("=" * 80)
print(f"sum_logp (single): {sum_logp_single}")
print(f"Completion mask sum: {completion_mask_single.sum()}")
print()

# Check if masks are different
print("=" * 80)
print("MASK COMPARISON (example 0)")
print("=" * 80)

# The batch mask is [B, T-1], we need to extract for example 0
batch_mask_0 = completion_mask_batch[0]  # [T-1]
single_mask_0 = completion_mask_single[0]  # [T_single-1]

print(f"Batch mask shape: {batch_mask_0.shape}")
print(f"Single mask shape: {single_mask_0.shape}")
print(f"Batch mask True count: {batch_mask_0.sum().item()}")
print(f"Single mask True count: {single_mask_0.sum().item()}")

# Where are they True?
batch_true_indices = batch_mask_0.nonzero(as_tuple=True)[0]
single_true_indices = single_mask_0.nonzero(as_tuple=True)[0]

print(f"Batch True indices (first 20): {batch_true_indices[:20]}")
print(f"Single True indices (first 20): {single_true_indices[:20]}")
print()

# Check the actual prompt_len
print("=" * 80)
print("PROMPT LENGTH ANALYSIS")
print("=" * 80)
print(f"Original prompt_len (tokenized): {item0.prompt_len}")
print(f"Batch prompt_lens[0]: {batch['prompt_lens'][0]}")
print(f"Single prompt_len: {pl[0]}")
print()

# Check attention pattern
print("Batch attention mask for example 0:")
attn_0 = batch['chosen_attention_mask'][0]
print(f"  Sum (total real tokens): {attn_0.sum().item()}")
print(f"  First True index: {attn_0.nonzero(as_tuple=True)[0][0].item()}")
print()

print("Single attention mask:")
print(f"  Sum (total real tokens): {ca[0].sum().item()}")
print(f"  First True index: 0 (no padding)")

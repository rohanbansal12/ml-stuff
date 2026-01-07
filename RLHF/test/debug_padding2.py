import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collate import collate_preference_batch
from data.tokenize import tokenize_preference_example
from engine import load_tokenizer

tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

examples = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain why the sky is blue."},
        ],
        "chosen": "The sky appears blue because shorter wavelengths scatter more strongly in the atmosphere (Rayleigh scattering).",
        "rejected": "It reflects the ocean.",
    }
]

# Tokenize
tokenized = [tokenize_preference_example(tokenizer, ex, max_len=256) for ex in examples]
item = tokenized[0]

print("Tokenized item:")
print(f"  prompt_len: {item.prompt_len}")
print(f"  chosen length: {item.chosen_input_ids_1d.numel()}")
print(f"  chosen_completion_len: {item.chosen_completion_len}")
print()

# Collate
batch = collate_preference_batch(tokenizer, tokenized)

print("Collated batch:")
print(f"  Batch shape: {batch['chosen_input_ids'].shape}")
print(f"  prompt_lens: {batch['prompt_lens']}")
print(f"  Attention mask: {batch['chosen_attention_mask'][0]}")
print()

# Manually compute real_index_input
attention_mask = batch["chosen_attention_mask"]
prompt_lens = batch["prompt_lens"]

real_index_input = attention_mask.cumsum(dim=1) - 1
print("real_index_input (first example, first 50 positions):")
print(real_index_input[0, :50])
print()

# Check where completion should start
completion_starts_at_real_index = prompt_lens[0].item()  # 27
print(f"Completion should start at real_index >= {completion_starts_at_real_index}")
print()

# Find where real_index_input >= 27
completion_mask_input = (real_index_input >= prompt_lens.unsqueeze(1)) & attention_mask.bool()
print("completion_mask_input (first example):")
print(completion_mask_input[0])
print(f"  First True position: {completion_mask_input[0].nonzero(as_tuple=True)[0][0].item()}")
print()

# The issue: real_index_input for position 8 (first real token) should be 0
# But cumsum gives us: 0,0,0,0,0,0,0,0,1,2,3,...
# After -1: -1,-1,-1,-1,-1,-1,-1,-1,0,1,2,...
# So position 8 has real_index=0, which is correct!

# But wait, let's check what real_index is at the prompt/completion boundary
# For the unpadded sequence: prompt has 27 tokens (indices 0-26)
# Completion starts at token 27 (real_index=27)
# For the padded sequence: 8 padding + 27 prompt = 35 total before completion
# So position 35 should have real_index=27

print("Position 35 (should be first completion token):")
print(f"  real_index_input: {real_index_input[0, 35].item()}")
print(f"  attention_mask: {attention_mask[0, 35].item()}")
print(f"  Is completion?: {completion_mask_input[0, 35].item()}")

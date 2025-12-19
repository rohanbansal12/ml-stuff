import torch
from typing import Sequence, Dict, Any


def _get(item: Any, key: str):
    # Supports dict-like or dataclass-like objects
    if isinstance(item, dict):
        return item[key]
    return getattr(item, key)

def make_response_mask_label_space(
    attention_mask: torch.Tensor,  # [B, T]
    prompt_lens: torch.Tensor,     # [B]
) -> torch.Tensor:
    """Create a mask selecting only response tokens in label space.

    Generates a boolean mask for selecting response (completion) tokens in the
    label-shifted space used for loss computation. This aligns with how
    completion_logprobs works: the response begins at label index (prompt_len - 1).

    Args:
        attention_mask: Attention mask tensor of shape [B, T] where 1 indicates
            real tokens and 0 indicates padding.
        prompt_lens: Number of prompt tokens per example, shape [B].

    Returns:
        Boolean tensor of shape [B, T-1] where True indicates response tokens
        in label space (shifted by 1 from input space).
    """
    B, T = attention_mask.shape
    Tm1 = T - 1
    attn = attention_mask[:, 1:].bool()  # [B, T-1]
    idx = torch.arange(Tm1, device=attention_mask.device).unsqueeze(0).expand(B, Tm1)
    start = (prompt_lens - 1).clamp_min(0).unsqueeze(1)  # handle prompt_len==0 defensively
    return (idx >= start) & attn

def collate_preference_batch(
    tokenizer,
    tokenized_examples: Sequence[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Collate and pad tokenized preference examples into a batch.

    Takes a list of individually tokenized preference examples (with chosen and
    rejected completions) and pads them into uniform-length batch tensors using
    left-padding (important for generation models).

    Args:
        tokenizer: Hugging Face tokenizer with pad_token_id and padding_side='left'.
        tokenized_examples: Sequence of tokenized examples, each containing:
            - chosen_input_ids_1d: 1D tensor of chosen sequence token IDs.
            - rejected_input_ids_1d: 1D tensor of rejected sequence token IDs.
            - prompt_len: Integer length of the prompt portion.

    Returns:
        Dictionary with batched tensors:
            - chosen_input_ids: Padded chosen sequences [B, T].
            - chosen_attention_mask: Attention mask for chosen [B, T].
            - rejected_input_ids: Padded rejected sequences [B, T].
            - rejected_attention_mask: Attention mask for rejected [B, T].
            - prompt_lens: Prompt lengths per example [B].
            - chosen_response_mask: Response token mask for chosen [B, T-1].
            - rejected_response_mask: Response token mask for rejected [B, T-1].

    Raises:
        AssertionError: If padding_side is not 'left' or pad_token_id is None.
    """
    assert tokenizer.padding_side == "left", "Set tokenizer.padding_side = 'left' before collation."
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer.pad_token_id must be set."

    B = len(tokenized_examples)
    assert B > 0, "Empty batch."

    # Extract 1D tensors and prompt lens
    chosen_seqs = [_get(ex, "chosen_input_ids_1d") for ex in tokenized_examples]
    rejected_seqs = [_get(ex, "rejected_input_ids_1d") for ex in tokenized_examples]
    prompt_lens_list = [int(_get(ex, "prompt_len")) for ex in tokenized_examples]

    # Ensure tensors are 1D Long
    for t in chosen_seqs + rejected_seqs:
        assert isinstance(t, torch.Tensor) and t.dim() == 1, "input_ids must be 1D torch.Tensor"
        assert t.dtype == torch.long, "input_ids should be torch.long"

    # One shared max length across both chosen/rejected so shapes match cleanly
    max_len = max(max(t.numel() for t in chosen_seqs), max(t.numel() for t in rejected_seqs))

    # Allocate padded tensors on CPU (DataLoader-friendly). Move to GPU later.
    chosen_input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    rejected_input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    chosen_attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    rejected_attention_mask = torch.zeros((B, max_len), dtype=torch.long)

    # Left-pad: write sequences aligned to the right
    for i in range(B):
        c = chosen_seqs[i]
        r = rejected_seqs[i]

        c_len = c.numel()
        r_len = r.numel()

        chosen_input_ids[i, max_len - c_len :] = c
        rejected_input_ids[i, max_len - r_len :] = r

        chosen_attention_mask[i, max_len - c_len :] = 1
        rejected_attention_mask[i, max_len - r_len :] = 1

    prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long)

    # Optional: response masks in label-space (T-1)
    chosen_response_mask = make_response_mask_label_space(chosen_attention_mask, prompt_lens)
    rejected_response_mask = make_response_mask_label_space(rejected_attention_mask, prompt_lens)

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "prompt_lens": prompt_lens,
        "chosen_response_mask": chosen_response_mask,
        "rejected_response_mask": rejected_response_mask,
    }
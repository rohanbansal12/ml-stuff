from dataclasses import dataclass
from typing import Any

import torch

from .formats import build_prompt_from_messages


def tokenize_prompt_plus_completion(
    tokenizer,
    prompt_ids_1d: torch.Tensor,  # [P]
    completion_text: str,
    *,
    max_len: int,
    truncation_side: str = "left",
) -> dict[str, torch.Tensor]:
    """Tokenize a completion and concatenate with prompt token IDs.

    Tokenizes the completion text, concatenates it with the given prompt tokens,
    and applies truncation if the combined sequence exceeds max_len.

    Args:
        tokenizer: Hugging Face tokenizer for encoding text.
        prompt_ids_1d: Pre-tokenized prompt as 1D tensor of shape [P].
        completion_text: Raw completion text to tokenize.
        max_len: Maximum total sequence length (prompt + completion).
        truncation_side: Which side to truncate from if too long.
            Either "left" (truncate prompt) or "right" (truncate completion).

    Returns:
        Dictionary containing:
            - input_ids_1d (torch.Tensor): Combined token IDs of shape [T].
            - attention_mask_1d (torch.Tensor): Attention mask of shape [T] (all 1s).
            - prompt_len (int): Prompt length after truncation.
            - completion_len (int): Completion length after truncation.
    """
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    completion_ids_1d = torch.tensor(completion_ids, dtype=torch.long, device=prompt_ids_1d.device)
    input_ids_1d = torch.cat([prompt_ids_1d, completion_ids_1d], dim=0)

    prompt_len = prompt_ids_1d.size(0)
    completion_len = len(completion_ids)

    if input_ids_1d.size(0) > max_len:
        if truncation_side == "left":
            input_ids_1d = input_ids_1d[-max_len:]
            prompt_len = max(0, max_len - completion_len)
            completion_len = min(max_len, completion_len)
        elif truncation_side == "right":
            input_ids_1d = input_ids_1d[:max_len]
            completion_len = max(0, max_len - prompt_len)
            prompt_len = min(prompt_len, max_len)
        else:
            raise ValueError(f"Unknown truncation side: {truncation_side}")

    attention_mask_1d = torch.ones_like(input_ids_1d)

    return {
        "input_ids_1d": input_ids_1d,
        "attention_mask_1d": attention_mask_1d,
        "prompt_len": prompt_len,
        "completion_len": completion_len,
    }


@dataclass
class TokenizedPreferenceExample:
    prompt_text: str
    chosen_text: str
    rejected_text: str
    prompt_len: int

    chosen_input_ids_1d: torch.Tensor
    chosen_attention_mask_1d: torch.Tensor
    rejected_input_ids_1d: torch.Tensor
    rejected_attention_mask_1d: torch.Tensor

    chosen_completion_len: int
    rejected_completion_len: int


def tokenize_preference_example(
    tokenizer,
    example: dict[str, Any],
    *,
    max_len: int,
) -> TokenizedPreferenceExample:
    """Tokenize a preference learning example with chosen and rejected completions.

    Converts a raw preference example with messages and two completions into
    tokenized sequences suitable for preference learning (e.g., DPO, RLHF).
    Uses a shared truncated prompt for both chosen and rejected completions.

    Args:
        tokenizer: Hugging Face tokenizer for encoding text.
        example: Dictionary with keys:
            - "messages": List of chat message dicts (role, content).
            - "chosen": Preferred completion text.
            - "rejected": Less preferred completion text.
        max_len: Maximum sequence length for each tokenized example.

    Returns:
        TokenizedPreferenceExample containing tokenized chosen and rejected
        sequences with shared prompt.

    Note:
        Truncation policy maintains consistency across chosen/rejected:
        - Keeps the last K prompt tokens where K allows the longer completion
          to fit within max_len.
        - Each completion keeps its last min(len(completion), max_len) tokens.
        - prompt_len is identical for both chosen and rejected by design.
    """
    messages: list[dict[str, str]] = example["messages"]
    chosen_text: str = example["chosen"]
    rejected_text: str = example["rejected"]

    prompt_text, prompt_ids_1d, _prompt_len_full = build_prompt_from_messages(tokenizer, messages)
    prompt_ids_1d = prompt_ids_1d.to(dtype=torch.long)  # keep on CPU
    P = int(prompt_ids_1d.numel())

    # Tokenize completions (IDs → tensors)
    chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=False)
    rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=False)
    chosen_comp_1d = torch.tensor(chosen_ids, dtype=torch.long, device=prompt_ids_1d.device)
    rejected_comp_1d = torch.tensor(rejected_ids, dtype=torch.long, device=prompt_ids_1d.device)

    Cc = int(chosen_comp_1d.numel())
    Cr = int(rejected_comp_1d.numel())

    # Keep last tokens of completion
    chosen_kept_c = min(Cc, max_len)
    rejected_kept_c = min(Cr, max_len)
    chosen_comp_kept = chosen_comp_1d[-chosen_kept_c:] if chosen_kept_c > 0 else chosen_comp_1d[:0]
    rejected_comp_kept = (
        rejected_comp_1d[-rejected_kept_c:] if rejected_kept_c > 0 else rejected_comp_1d[:0]
    )

    # Shared prompt budget based on longer kept completion
    max_kept_completion = max(chosen_kept_c, rejected_kept_c)
    prompt_kept_p = max(0, min(P, max_len - max_kept_completion))
    prompt_kept = prompt_ids_1d[-prompt_kept_p:] if prompt_kept_p > 0 else prompt_ids_1d[:0]
    prompt_len = int(prompt_kept.numel())

    # Concatenate
    chosen_input_ids_1d = torch.cat([prompt_kept, chosen_comp_kept], dim=0)
    rejected_input_ids_1d = torch.cat([prompt_kept, rejected_comp_kept], dim=0)

    assert chosen_input_ids_1d.numel() <= max_len
    assert rejected_input_ids_1d.numel() <= max_len

    chosen_attention_mask_1d = torch.ones_like(chosen_input_ids_1d, dtype=torch.long)
    rejected_attention_mask_1d = torch.ones_like(rejected_input_ids_1d, dtype=torch.long)

    return TokenizedPreferenceExample(
        prompt_text=prompt_text,
        chosen_text=chosen_text,
        rejected_text=rejected_text,
        prompt_len=prompt_len,
        chosen_input_ids_1d=chosen_input_ids_1d,
        chosen_attention_mask_1d=chosen_attention_mask_1d,
        rejected_input_ids_1d=rejected_input_ids_1d,
        rejected_attention_mask_1d=rejected_attention_mask_1d,
        chosen_completion_len=int(chosen_comp_kept.numel()),
        rejected_completion_len=int(rejected_comp_kept.numel()),
    )

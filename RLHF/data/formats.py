import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

@dataclass
class PreferenceExample:
    messages: List[Dict[str, str]]
    chosen: str
    rejected: str
    meta: Optional[Dict[str, Any]] = None


def build_prompt_from_messages(tokenizer, messages: List[Dict[str, str]]) -> Tuple[str, torch.Tensor, int]:
    """Build a chat-template prompt string and tokenize it.

    Applies the tokenizer's chat template to format the conversation messages
    and returns both the formatted text and tokenized IDs.

    Args:
        tokenizer: Hugging Face tokenizer with chat template support.
        messages: List of message dictionaries with 'role' and 'content' keys.
            Example: [{"role": "user", "content": "Hello"}]

    Returns:
        A tuple containing:
            - prompt_text (str): Formatted chat template string.
            - prompt_ids_1d (torch.Tensor): Token IDs of shape [P] where P is prompt length.
            - prompt_len (int): Number of tokens in the prompt (equal to P).
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids_1d = enc["input_ids"][0]
    prompt_len = prompt_ids_1d.size(0)
    return prompt_text, prompt_ids_1d, prompt_len
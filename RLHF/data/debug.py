import torch
from typing import Any

def debug_print_tokenized_item(
    tokenizer,
    tokenized_item: Any,
    *,
    tail: int = 120,
) -> None:
    """Print detailed debug information for a tokenized preference example.

    Displays comprehensive information about a tokenized preference learning example
    including the prompt text, chosen/rejected completions, token counts, decoded
    sequences, and a visual representation of which tokens are treated as prompt
    vs response tokens in label space.

    Args:
        tokenizer: Hugging Face tokenizer used for decoding.
        tokenized_item: Tokenized preference example (dict or dataclass) containing:
            - prompt_text: Original formatted prompt string.
            - chosen_text: Raw chosen completion text.
            - rejected_text: Raw rejected completion text.
            - prompt_len: Number of prompt tokens.
            - chosen_input_ids_1d: Chosen sequence token IDs.
            - rejected_input_ids_1d: Rejected sequence token IDs.
        tail: Maximum number of characters to display from decoded strings
            for readability. Default is 120.

    Note:
        The visualization uses 'P' for prompt tokens and 'R' for response tokens
        in label space, which is shifted by 1 from input space to align with
        how language model losses are computed.
    """

    # Support dict or dataclass
    def get(key):
        if isinstance(tokenized_item, dict):
            return tokenized_item[key]
        return getattr(tokenized_item, key)

    prompt_text = get("prompt_text")
    chosen_text = get("chosen_text")
    rejected_text = get("rejected_text")
    prompt_len = int(get("prompt_len"))

    chosen_ids = get("chosen_input_ids_1d")
    rejected_ids = get("rejected_input_ids_1d")

    # Decode full sequences
    chosen_decoded = tokenizer.decode(chosen_ids, skip_special_tokens=False)
    rejected_decoded = tokenizer.decode(rejected_ids, skip_special_tokens=False)

    # Basic info
    print("=" * 90)
    print("PROMPT (raw chat-template text)")
    print("-" * 90)
    print(prompt_text[-tail:])
    print()

    print("RAW COMPLETIONS")
    print("-" * 90)
    print("Chosen:")
    print(chosen_text[:tail])
    print("\nRejected:")
    print(rejected_text[:tail])
    print()

    print("TOKEN COUNTS")
    print("-" * 90)
    print(f"prompt_len: {prompt_len}")
    print(f"chosen_total_len: {chosen_ids.numel()}")
    print(f"rejected_total_len: {rejected_ids.numel()}")
    print()

    # Show decoded sequences (tail only)
    print("DECODED SEQUENCES (prompt + completion)")
    print("-" * 90)
    print("Chosen (tail):")
    print(chosen_decoded[-tail:])
    print("\nRejected (tail):")
    print(rejected_decoded[-tail:])
    print()

    # Token-level boundary visualization
    print("TOKEN-LEVEL RESPONSE MASK VISUALIZATION")
    print("-" * 90)
    print("Legend: P = prompt token, R = response token\n")

    def visualize(ids_1d, name):
        labels = ids_1d[1:]  # label-space
        Tm1 = labels.numel()

        mask = torch.zeros(Tm1, dtype=torch.bool)
        start = max(prompt_len - 1, 0)
        if start < Tm1:
            mask[start:] = True

        vis = []
        for i in range(Tm1):
            vis.append("R" if mask[i] else "P")

        # Chunk visualization for readability
        vis_str = "".join(vis)
        print(f"{name} mask:")
        print(vis_str)
        print(f"(P={vis_str.count('P')}, R={vis_str.count('R')})\n")

    visualize(chosen_ids, "Chosen")
    visualize(rejected_ids, "Rejected")

    print("=" * 90)
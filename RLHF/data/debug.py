from typing import Any

import torch
import torch.nn.functional as F


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


def _format_token_str(s: str) -> str:
    # Make whitespace visible in debug prints
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    return s


def debug_score_example(
    model,
    tokenizer,
    *,
    # Option A: full sequence directly
    input_ids_1d: torch.Tensor | None = None,  # [T]
    attention_mask_1d: torch.Tensor | None = None,  # [T]
    prompt_len: int | None = None,
    # Option B: messages + completion
    messages: list[dict[str, str]] | None = None,
    completion_text: str | None = None,
    completion_ids_1d: torch.Tensor | None = None,
    # Controls
    max_print_tokens: int = 80,
    skip_special_tokens: bool = False,
    include_prompt_tokens: bool = False,
    show_topk: int = 0,  # set >0 to show top-k alternatives per position (heavier)
) -> dict[str, Any]:
    """Debug and decompose completion log probabilities token-by-token.

    Performs a forward pass through the model and computes per-token log probabilities
    for the completion (response) portion. Prints detailed breakdown including token
    IDs, decoded text, individual and cumulative log probabilities. Essential for
    debugging RLHF/DPO scoring issues.

    Args:
        model: Language model to score with.
        tokenizer: Tokenizer for encoding/decoding.
        input_ids_1d: Full sequence token IDs [T]. Use with Option A.
        attention_mask_1d: Attention mask [T]. Use with Option A.
        prompt_len: Number of prompt tokens. Use with Option A.
        messages: Chat messages to build prompt from. Use with Option B.
        completion_text: Raw completion text. Use with Option B.
        completion_ids_1d: Pre-tokenized completion IDs. Use with Option B.
        max_print_tokens: Maximum number of tokens to print in detail table.
            Default is 80.
        skip_special_tokens: Whether to skip special tokens in decoded output.
            Default is False.
        include_prompt_tokens: If True, score and display prompt tokens too.
            Default is False (only score response tokens).
        show_topk: If > 0, show top-k alternative tokens at each position.
            Default is 0 (disabled).

    Returns:
        Dictionary containing:
            - sum_logp (float): Sum of log probabilities.
            - mean_logp (float): Mean log probability per token.
            - token_count (int): Number of scored tokens.
            - prompt_len (int): Prompt length.
            - input_ids_1d (torch.Tensor): Full sequence IDs on CPU.
            - attention_mask_1d (torch.Tensor): Attention mask on CPU.
            - token_logp_label_space (torch.Tensor): Per-token log probs [T-1].
            - response_mask_label_space (torch.Tensor): Response mask [T-1].
            - decoded_prompt (str): Decoded prompt text.
            - decoded_completion (str): Decoded completion text.
            - built_prompt_text (str or None): Chat template formatted prompt.
            - raw_completion_text (str or None): Raw completion text.

    Raises:
        ValueError: If both modes are specified, neither mode is specified, or
            required parameters are missing.

    Note:
        Label space alignment: logits[t] predicts input_ids[t+1], so the response
        begins at label index (prompt_len - 1), not prompt_len. This matches how
        language model losses are computed.

    Example:
        >>> # Option B: Using messages
        >>> result = debug_score_example(
        ...     model, tokenizer,
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     completion_text="Hi there!",
        ...     show_topk=3
        ... )
    """
    # -----------------------------
    # TODO-turned-into-code: validate input mode
    # -----------------------------
    mode_full = (
        (input_ids_1d is not None) or (attention_mask_1d is not None) or (prompt_len is not None)
    )
    mode_msgs = (
        (messages is not None) or (completion_text is not None) or (completion_ids_1d is not None)
    )

    if mode_full and mode_msgs:
        raise ValueError("Provide either full-seq inputs OR messages+completion, not both.")
    if not mode_full and not mode_msgs:
        raise ValueError(
            "Provide either (input_ids_1d, attention_mask_1d, prompt_len) OR (messages, completion_*)."
        )

    device = next(model.parameters()).device

    # -----------------------------
    # Build full sequence if using messages mode
    # -----------------------------
    built_prompt_text = None
    built_completion_text = None

    if mode_msgs:
        if messages is None:
            raise ValueError("messages must be provided in messages mode.")

        built_prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_enc = tokenizer(
            built_prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_enc["input_ids"][0].to(dtype=torch.long)

        if completion_ids_1d is None:
            if completion_text is None:
                raise ValueError(
                    "Provide either completion_text or completion_ids_1d in messages mode."
                )
            built_completion_text = completion_text
            comp_ids_list = tokenizer.encode(completion_text, add_special_tokens=False)
            completion_ids_1d = torch.tensor(comp_ids_list, dtype=torch.long)
        else:
            if not isinstance(completion_ids_1d, torch.Tensor):
                raise ValueError("completion_ids_1d must be a torch.Tensor.")
            if completion_text is not None:
                built_completion_text = completion_text  # optional, for display

        completion_ids_1d = completion_ids_1d.to(dtype=torch.long)

        # Full sequence
        input_ids_1d = torch.cat([prompt_ids, completion_ids_1d], dim=0)
        prompt_len = int(prompt_ids.numel())
        attention_mask_1d = torch.ones_like(input_ids_1d, dtype=torch.long)

    # -----------------------------
    # Full sequence mode: sanitize inputs
    # -----------------------------
    assert (
        input_ids_1d is not None and prompt_len is not None
    ), "Full-seq mode requires input_ids_1d and prompt_len."
    if attention_mask_1d is None:
        attention_mask_1d = torch.ones_like(input_ids_1d, dtype=torch.long)

    if input_ids_1d.dim() != 1:
        raise ValueError(f"input_ids_1d must be 1D, got shape {tuple(input_ids_1d.shape)}")
    if attention_mask_1d.dim() != 1:
        raise ValueError(
            f"attention_mask_1d must be 1D, got shape {tuple(attention_mask_1d.shape)}"
        )

    input_ids_1d = input_ids_1d.to(device=device, dtype=torch.long)
    attention_mask_1d = attention_mask_1d.to(device=device, dtype=torch.long)
    prompt_len = int(prompt_len)

    T = int(input_ids_1d.numel())
    if T < 2:
        raise ValueError(
            "Need at least 2 tokens to compute any next-token logprob (T must be >= 2)."
        )

    # -----------------------------
    # Forward pass (teacher forcing)
    # -----------------------------
    with torch.no_grad():
        out = model(
            input_ids=input_ids_1d.unsqueeze(0),
            attention_mask=attention_mask_1d.unsqueeze(0),
        )
        logits = out.logits[0]  # [T, V]

    # -----------------------------
    # Shift logits/labels
    # -----------------------------
    logits_shifted = logits[:-1, :]  # [T-1, V]
    labels_shifted = input_ids_1d[1:]  # [T-1]
    attn_shifted = attention_mask_1d[1:].bool()  # [T-1]

    logp_all = F.log_softmax(logits_shifted, dim=-1)  # [T-1, V]
    token_logp = logp_all.gather(dim=-1, index=labels_shifted.unsqueeze(-1)).squeeze(-1)  # [T-1]

    # -----------------------------
    # Build mask (label-space)
    # -----------------------------
    Tm1 = T - 1
    idx = torch.arange(Tm1, device=device)
    start = max(prompt_len - 1, 0)

    response_mask = (idx >= start) & attn_shifted
    if include_prompt_tokens:
        response_mask = attn_shifted.clone()

    sel = response_mask.nonzero(as_tuple=False).squeeze(-1)  # label indices

    # Token counts
    token_count = int(sel.numel())
    sum_logp = float(token_logp[sel].sum().item()) if token_count > 0 else 0.0
    mean_logp = (
        float((token_logp[sel].sum() / max(token_count, 1)).item()) if token_count > 0 else 0.0
    )

    # -----------------------------
    # Decode prompt/completion (best-effort)
    # -----------------------------
    # In full-seq mode, we approximate prompt text by decoding the first prompt_len tokens.
    prompt_dec = tokenizer.decode(
        input_ids_1d[:prompt_len], skip_special_tokens=skip_special_tokens
    )
    completion_dec = tokenizer.decode(
        input_ids_1d[prompt_len:], skip_special_tokens=skip_special_tokens
    )

    # -----------------------------
    # Pretty print
    # -----------------------------
    print("=" * 100)
    print("DEBUG SCORE EXAMPLE")
    print("-" * 100)
    print(f"total_len={T} | prompt_len={prompt_len} | completion_len={max(0, T - prompt_len)}")
    print(f"scored_token_count={token_count} | sum_logp={sum_logp:.6f} | mean_logp={mean_logp:.6f}")
    print("-" * 100)

    if built_prompt_text is not None:
        print("PROMPT TEXT (chat template) [tail]")
        print(_format_token_str(built_prompt_text[-400:]))
        print("-" * 100)

    print("DECODED PROMPT [tail]")
    print(_format_token_str(prompt_dec[-400:]))
    print("-" * 100)

    if built_completion_text is not None:
        print("RAW COMPLETION TEXT [head]")
        print(_format_token_str(built_completion_text[:400]))
        print("-" * 100)

    print("DECODED COMPLETION [tail]")
    print(_format_token_str(completion_dec[-400:]))
    print("-" * 100)

    # Choose which positions to print
    if token_count == 0:
        print("(No tokens selected for scoring under current mask settings.)")
    else:
        # Print up to max_print_tokens tokens
        show = sel[:max_print_tokens]
        print("Per-token logprob (label-space). Each row corresponds to predicting input_ids[pos].")
        print("Columns: label_i | input_pos | token_id | token_str | logp | cum_logp")
        print("-" * 100)

        cum = 0.0
        for j, i in enumerate(show.tolist()):
            input_pos = i + 1  # label i corresponds to input token position i+1
            tok_id = int(labels_shifted[i].item())
            tok_str = tokenizer.decode([tok_id], skip_special_tokens=skip_special_tokens)
            tok_str = _format_token_str(tok_str)

            lp = float(token_logp[i].item())
            cum += lp

            print(
                f"{i:7d} | {input_pos:9d} | {tok_id:7d} | {tok_str[:30]:30s} | {lp: .6f} | {cum: .6f}"
            )

            # Optional top-k alternatives
            if show_topk and show_topk > 0:
                topv, topi = torch.topk(logp_all[i], k=show_topk)
                alts = []
                for alt_id, alt_lp in zip(topi.tolist(), topv.tolist(), strict=False):
                    s = tokenizer.decode([int(alt_id)], skip_special_tokens=skip_special_tokens)
                    alts.append(f"{_format_token_str(s)}:{alt_lp:.3f}")
                print(" " * 10 + "topk: " + " | ".join(alts))

        if token_count > max_print_tokens:
            print(f"... (printed {max_print_tokens}/{token_count} scored tokens)")

    print("=" * 100)

    # -----------------------------
    # Return artifacts
    # -----------------------------
    return {
        "sum_logp": sum_logp,
        "mean_logp": mean_logp,
        "token_count": token_count,
        "prompt_len": prompt_len,
        "input_ids_1d": input_ids_1d.detach().cpu(),
        "attention_mask_1d": attention_mask_1d.detach().cpu(),
        "token_logp_label_space": token_logp.detach().cpu(),  # [T-1]
        "response_mask_label_space": response_mask.detach().cpu(),  # [T-1]
        "decoded_prompt": prompt_dec,
        "decoded_completion": completion_dec,
        "built_prompt_text": built_prompt_text,
        "raw_completion_text": built_completion_text,
    }

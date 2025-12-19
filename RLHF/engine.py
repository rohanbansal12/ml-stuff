import torch
import torch.nn.functional as F
import transformers
import datasets
import trl
import accelerate
import peft
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def env_info():
    print("=" * 60)
    print("Library Versions")
    print("=" * 60)

    print(f"PyTorch:       {torch.__version__}")
    print(f"Transformers:  {transformers.__version__}")
    print(f"Datasets:      {datasets.__version__}")
    print(f"TRL:           {trl.__version__}")
    print(f"Accelerate:    {accelerate.__version__}")
    print(f"PEFT:          {peft.__version__}")

    print("\n" + "=" * 60)
    print("GPU Information")
    print("=" * 60)

    dtype = torch.float16

    if torch.cuda.is_available():
        print("CUDA Available:     Yes")
        print(f"CUDA Version:       {torch.version.cuda}")
        print(f"Device Count:       {torch.cuda.device_count()}")
        print(f"Current Device:     {torch.cuda.current_device()}")
        print(f"Device Name:        {torch.cuda.get_device_name(0)}")

        # Check bf16 support
        supports_bf16 = torch.cuda.is_bf16_supported()
        print(f"\nBF16 Support:       {supports_bf16}")

        if supports_bf16:
            print("✓ GPU supports bfloat16 dtype")
            dtype = torch.bfloat16
        else:
            print("✗ GPU does not support bfloat16 dtype")
            print("  (Compute capability 8.0+ required for bf16)")
    else:
        print("CUDA Available:     No")
        print("Running on CPU")

    print("=" * 60)

    return dtype


def load_tokenizer(model_name, verbose=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding side to left (important for generation/RL)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if verbose:
        # Identify special tokens
        print("\nTokenizer Special Tokens:")
        print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # Check if padding token exists
    if tokenizer.pad_token is not None:
        if verbose:
            print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    else:
        tokenizer.pad_token = tokenizer.eos_token
        if verbose:
            print("  PAD token: Not set")
            print("  → Setting pad_token = eos_token")
            print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return tokenizer


def tokenizer_test(tok):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]
    chat_str = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("=" * 80)
    print("RAW CHAT TEMPLATE STRING")
    print("=" * 80)
    print(chat_str)

    encoded = tok(
        chat_str,
        return_tensors="pt",
        add_special_tokens=False,  # IMPORTANT: chat_template already adds them
    )

    input_ids = encoded["input_ids"][0]

    print("\n" + "=" * 80)
    print("TOKENIZATION INFO")
    print("=" * 80)
    print(f"Total tokens: {input_ids.shape[0]}")

    TAIL = 40
    tail_ids = input_ids[-TAIL:]

    print("\nLast token IDs:")
    print(tail_ids.tolist())

    print("\nDecoded tail:")
    print(tok.decode(tail_ids))

    # ---- 4) Sanity: where would generation begin? ----
    print("\n" + "=" * 80)
    print("GENERATION START CHECK")
    print("=" * 80)
    print("The model will start generating tokens AFTER this prompt.")
    print("Look for an assistant role marker or <|im_start|> / <|im_end|> boundary.")


def load_model(model_name, dtype, device):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype
    ).to(device)
    model.eval()

    return model


def model_test(model, tokenizer):
    print("param device:", next(model.parameters()).device)
    print("param dtype:  ", next(model.parameters()).dtype)
    print("model.training:", model.training)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    print("logits shape:", out.logits.shape)  # [B, T, vocab]
    print("vocab size:", out.logits.shape[-1])
    print("pad_token_id:", tokenizer.pad_token_id)
    print("eos_token_id:", tokenizer.eos_token_id)


@dataclass
class GenOutput:
    prompt_text: str
    prompt_len: int
    output_ids: Any  # torch.Tensor
    completion_ids: Any  # torch.Tensor
    output_text: str
    completion_text: str


def generate_one(
    model: transformers.PreTrainedModel,
    tokenizer,
    messages,
    *,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=None,
    seed=None,
    **gen_kwargs,
):
    # 0) (Optional) control randomness
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 1) Build prompt string via chat template
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2) Tokenize (IMPORTANT: add_special_tokens=False)
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    # 3) Move to model device
    enc = {k: v.to(model.device) for k, v in enc.items()}

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)
    prompt_len = input_ids.shape[1]

    # 4) Build explicit generation kwargs
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        **gen_kwargs,
    )

    if not do_sample:
        generate_kwargs["temperature"] = None
        generate_kwargs["top_p"] = None

    # Optional: repetition penalty
    if repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = repetition_penalty

    # Clean: remove Nones so HF doesn’t complain
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    # 5) Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
        )

    # 6) Slice completion tokens (this is critical later)
    completion_ids = output_ids[0, prompt_len:]

    # 7) Decode full and completion text
    output_text = tokenizer.decode(output_ids[0])
    completion_text = tokenizer.decode(completion_ids)

    return GenOutput(
        prompt_text=prompt_text,
        prompt_len=prompt_len,
        output_ids=output_ids[0],
        completion_ids=completion_ids,
        output_text=output_text,
        completion_text=completion_text,
    )


def generate_test(model, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]

    y1 = generate_one(model, tokenizer, messages, do_sample=False)
    y2 = generate_one(model, tokenizer, messages, do_sample=False)
    assert torch.equal(y1.completion_ids, y2.completion_ids)

    y1 = generate_one(model, tokenizer, messages, do_sample=True, seed=123)
    y2 = generate_one(model, tokenizer, messages, do_sample=True, seed=123)
    assert torch.equal(y1.completion_ids, y2.completion_ids)

    y1 = generate_one(model, tokenizer, messages, do_sample=True, seed=123)
    y2 = generate_one(model, tokenizer, messages, do_sample=True, seed=124)
    assert not torch.equal(y1.completion_ids, y2.completion_ids)


def completion_logprobs(model, input_ids, attention_mask, prompt_lens, require_grad = False):
    """
    input_ids:      [B, T] prompt+completion (padded)
    attention_mask: [B, T] 1 real, 0 pad
    prompt_lens:    [B] number of prompt tokens per example (int)
    """
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    position_ids = position_ids.clamp_min(0)

    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = out.logits

    logits = logits[:, :-1, :]  # [B, T-1, V]
    labels = input_ids[:, 1:]  # [B, T-1]
    attn = attention_mask[:, 1:]  # [B, T-1]

    logp_all = F.log_softmax(logits.float(), dim=-1)  # [B, T-1, V]
    token_logp = logp_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )  # [B, T-1]

    real_index_input = attention_mask.cumsum(dim=1) - 1  # [B, T]
    # completion tokens are those with real_index >= prompt_len (in input space)
    completion_mask_input = (
        real_index_input >= prompt_lens.unsqueeze(1)
    ) & attention_mask.bool()  # [B, T]

    # token_logp is label-space for tokens at input positions 1..T-1
    completion_mask = completion_mask_input[:, 1:] & attn.bool()  # [B, T-1]

    sum_logp = (token_logp * completion_mask).sum(dim=1)
    denom = completion_mask.sum(dim=1).clamp_min(1)
    mean_logp = sum_logp / denom

    return sum_logp, mean_logp, token_logp, completion_mask


def _find_single_token_id(tokenizer, candidates=None):
    """
    Find a token that the tokenizer encodes as exactly 1 token.
    Returns (token_id, token_str).
    """
    if candidates is None:
        candidates = [
            ".",
            "!",
            "?",
            ",",
            ":",
            ";",
            " the",
            " a",
            " an",
            " yes",
            " no",
            "Hello",
            " hi",
            " OK",
            "\n",
        ]

    for s in candidates:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], s

    raise RuntimeError(
        "Could not find a single-token candidate. Try expanding the candidate list."
    )


def _build_prompt(tokenizer, messages):
    # IMPORTANT: chat template already includes special tokens
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    return prompt_text, enc["input_ids"][0]  # shape [T]


def test_completion_logprobs_sanity(
    model,
    tokenizer,
    completion_logprobs_fn,
    *,
    device=None,
    verbose=True,
    run_generation_check=True,
):
    """
    Runs the sanity checks:

    Check 1: Prompt-only => 0 completion tokens selected
    Check 2: Single-token completion => exactly 1 completion token selected; sum_logp matches that token logp
    Check 3: Generation slicing consistency => modifying one completion token changes logprob

    Parameters:
      completion_logprobs_fn: your function completion_logprobs(model, input_ids, attention_mask, prompt_lens)
    """
    if device is None:
        device = model.device

    # Build a simple prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]
    prompt_text, prompt_ids_1d = _build_prompt(tokenizer, messages)
    prompt_ids_1d = prompt_ids_1d.to(device)
    prompt_len = prompt_ids_1d.numel()

    if verbose:
        print("=" * 80)
        print("Prompt text (debug):")
        print(prompt_text)
        print(f"prompt_len={prompt_len}")
        print("=" * 80)

    # ------------------------------------------------------------
    # Check 1: Prompt-only -> completion_mask should select 0 tokens
    # ------------------------------------------------------------
    input_ids = prompt_ids_1d.unsqueeze(0)  # [1, T]
    attention_mask = torch.ones_like(input_ids, device=device)
    prompt_lens = torch.tensor([prompt_len], device=device, dtype=torch.long)

    sum_logp, mean_logp, token_logp, completion_mask = completion_logprobs_fn(
        model, input_ids, attention_mask, prompt_lens
    )

    selected = int(completion_mask.sum().item())
    if verbose:
        print("[Check 1] Prompt-only selected completion tokens:", selected)

    assert selected == 0, (
        f"Check 1 failed: expected 0 completion tokens selected, got {selected}.\n"
        f"prompt_len={prompt_len}, token_logp shape={token_logp.shape}"
    )

    # ------------------------------------------------------------
    # Check 2: Single-token completion
    #   - completion_mask selects exactly 1 token
    #   - sum_logp equals that token's logprob under the model
    # ------------------------------------------------------------
    one_tok_id, one_tok_str = _find_single_token_id(tokenizer)

    completion_ids_1d = torch.tensor([one_tok_id], device=device, dtype=torch.long)
    input_ids2 = torch.cat([prompt_ids_1d, completion_ids_1d], dim=0).unsqueeze(
        0
    )  # [1, T+1]
    attention_mask2 = torch.ones_like(input_ids2, device=device)
    prompt_lens2 = torch.tensor([prompt_len], device=device, dtype=torch.long)

    sum_logp2, mean_logp2, token_logp2, completion_mask2 = completion_logprobs_fn(
        model, input_ids2, attention_mask2, prompt_lens2
    )

    selected2 = int(completion_mask2.sum().item())
    if verbose:
        print(
            f"[Check 2] Single-token completion candidate: {repr(one_tok_str)} -> id={one_tok_id}"
        )
        print("[Check 2] Selected completion tokens:", selected2)

    assert selected2 == 1, (
        f"Check 2 failed: expected exactly 1 completion token selected, got {selected2}."
    )

    # Identify the position in token_logp space that was selected
    # token_logp is aligned to labels = input_ids[:, 1:]  => length (T_total - 1)
    chosen_positions = completion_mask2[0].nonzero(as_tuple=False).squeeze(-1)
    assert chosen_positions.numel() == 1
    pos = int(chosen_positions.item())

    # The selected label token is labels[pos] which should be our one_tok_id
    labels2 = input_ids2[:, 1:]
    assert int(labels2[0, pos].item()) == one_tok_id, (
        "Check 2 failed: selected token in labels does not match our appended completion token. "
        f"labels[pos]={int(labels2[0, pos].item())}, expected={one_tok_id}"
    )

    # sum_logp2 should equal token_logp2 at that position (since only 1 token selected)
    expected = float(token_logp2[0, pos].item())
    got = float(sum_logp2[0].item())

    if verbose:
        print(f"[Check 2] sum_logp={got:.6f}, token_logp(selected)={expected:.6f}")

    assert abs(got - expected) < 1e-4, (
        f"Check 2 failed: sum_logp != selected token logp.\n"
        f"sum_logp={got}, token_logp={expected}, diff={abs(got - expected)}"
    )

    # ------------------------------------------------------------
    # Check 3: Generation slicing consistency
    #   - Generate a completion
    #   - Score it
    #   - Change one completion token => score changes
    # ------------------------------------------------------------
    if run_generation_check:
        gen_kwargs = dict(
            max_new_tokens=16,
            do_sample=False,  # deterministic for testing
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )

        prompt_len3 = input_ids.shape[1]
        completion3 = output_ids[0, prompt_len3:]
        if verbose:
            print("[Check 3] Generated completion length:", int(completion3.numel()))

        if completion3.numel() == 0:
            if verbose:
                print("[Check 3] Skipped: model generated 0 tokens (rare).")
        else:
            # Score original
            input_ids3 = output_ids.to(device)
            attention_mask3 = torch.ones_like(input_ids3, device=device)
            prompt_lens3 = torch.tensor([prompt_len3], device=device, dtype=torch.long)

            sum_logp3, _, _, _ = completion_logprobs_fn(
                model, input_ids3, attention_mask3, prompt_lens3
            )

            # Modify one completion token (first one) to a different token id
            orig_tok = int(input_ids3[0, prompt_len3].item())
            new_tok = one_tok_id
            if new_tok == orig_tok:
                # pick a different single-token if collision
                alt_candidates = ["?", "!", ".", ",", "\n", " the", " a", " no", " yes"]
                new_tok, _ = _find_single_token_id(tokenizer, candidates=alt_candidates)
                if new_tok == orig_tok:
                    new_tok = (orig_tok + 1) % tokenizer.vocab_size

            input_ids3_mod = input_ids3.clone()
            input_ids3_mod[0, prompt_len3] = new_tok

            sum_logp3_mod, _, _, _ = completion_logprobs_fn(
                model, input_ids3_mod, attention_mask3, prompt_lens3
            )

            got3 = float(sum_logp3[0].item())
            got3m = float(sum_logp3_mod[0].item())

            if verbose:
                print(f"[Check 3] sum_logp(original)={got3:.6f}")
                print(f"[Check 3] sum_logp(modified)={got3m:.6f}")
                print(f"[Check 3] delta={abs(got3 - got3m):.6f}")

            assert abs(got3 - got3m) > 1e-6, (
                "Check 3 failed: modifying a completion token did not change sum_logp. "
                "This strongly suggests an indexing/masking error."
            )

    if verbose:
        print("\n✅ All sanity checks passed.")

    return True


class RLHFEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # lock these down
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

    def build_prompt(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, torch.Tensor, int]:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.model.device)
        prompt_len = input_ids.shape[1]
        return prompt_text, input_ids[0], prompt_len

    def generate(self, messages, **gen_kwargs) -> GenOutput:
        return generate_one(self.model, self.tokenizer, messages, **gen_kwargs)

    def _completion_logprobs_from_full_sequence(
        self,
        input_ids: torch.Tensor,  # [B, T] full prompt+completion
        attention_mask: torch.Tensor,  # [B, T]
        prompt_lens: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        sum_logp, mean_logp, token_logp, mask = completion_logprobs(
            self.model, input_ids, attention_mask, prompt_lens
        )
        return sum_logp

    def logprob_of_completion(
        self,
        messages: List[Dict[str, str]],
        completion_ids: torch.Tensor,  # 1D completion token ids
    ) -> float:
        # Build prompt
        _, prompt_ids_1d, prompt_len = self.build_prompt(messages)

        assert completion_ids.dim() == 1
        assert prompt_len >= 1

        # Concatenate prompt + completion
        full_ids_1d = torch.cat(
            [prompt_ids_1d.to(self.model.device), completion_ids.to(self.model.device)],
            dim=0,
        )

        # Make batch
        input_ids = full_ids_1d.unsqueeze(0)  # [1, T]
        attention_mask = torch.ones_like(input_ids, device=self.model.device)
        prompt_lens = torch.tensor(
            [prompt_len], device=self.model.device, dtype=torch.long
        )

        sum_logp = self._completion_logprobs_from_full_sequence(
            input_ids, attention_mask, prompt_lens
        )
        return float(sum_logp.item())

    def score_pair(
        self,
        messages: List[Dict[str, str]],
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Tuple[float, float]:
        # Same prompt for both; score each completion
        logp_chosen = self.logprob_of_completion(messages, chosen_ids)
        logp_rejected = self.logprob_of_completion(messages, rejected_ids)
        return logp_chosen, logp_rejected

    def decode_completion(self, completion_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(completion_ids, skip_special_tokens=False)

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)


def main():
    device = torch.device(DEVICE)

    # basic environment/cuda checks
    dtype = env_info()

    # load tokenizer
    tok = load_tokenizer(MODEL_NAME, verbose=True)
    tokenizer_test(tok)

    # load model
    model = load_model(MODEL_NAME, dtype, device)
    model_test(model, tok)

    # test generate_one
    generate_test(model, tok)

    test_completion_logprobs_sanity(
        model=model,
        tokenizer=tok,
        completion_logprobs_fn=completion_logprobs,
        verbose=True,
        run_generation_check=True,
    )


if __name__ == "__main__":
    main()

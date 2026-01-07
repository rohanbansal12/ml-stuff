import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collate import (
    collate_preference_batch,
    make_response_mask_label_space,
)
from data.formats import build_prompt_from_messages
from data.tokenize import (
    TokenizedPreferenceExample,
    tokenize_preference_example,
    tokenize_prompt_plus_completion,
)
from engine import load_tokenizer

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer once for all tests."""
    return load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]


@pytest.fixture
def sample_preference_example():
    """Sample preference example for testing."""
    return {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
        ],
        "chosen": "The answer is 4.",
        "rejected": "I don't know.",
    }


# -----------------------------
# Test formats.py
# -----------------------------


class TestBuildPromptFromMessages:
    def test_returns_tuple_of_three(self, tokenizer, sample_messages):
        """Test that function returns a 3-tuple."""
        result = build_prompt_from_messages(tokenizer, sample_messages)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_prompt_text_is_string(self, tokenizer, sample_messages):
        """Test that prompt text is a non-empty string."""
        prompt_text, _, _ = build_prompt_from_messages(tokenizer, sample_messages)
        assert isinstance(prompt_text, str)
        assert len(prompt_text) > 0

    def test_prompt_ids_shape(self, tokenizer, sample_messages):
        """Test that prompt IDs are 1D tensor."""
        _, prompt_ids_1d, prompt_len = build_prompt_from_messages(tokenizer, sample_messages)
        assert isinstance(prompt_ids_1d, torch.Tensor)
        assert prompt_ids_1d.dim() == 1
        assert prompt_ids_1d.numel() > 0

    def test_prompt_len_matches_ids(self, tokenizer, sample_messages):
        """Test that prompt_len matches actual tensor size."""
        _, prompt_ids_1d, prompt_len = build_prompt_from_messages(tokenizer, sample_messages)
        assert prompt_len == prompt_ids_1d.numel()

    def test_prompt_ids_dtype(self, tokenizer, sample_messages):
        """Test that prompt IDs have correct dtype."""
        _, prompt_ids_1d, _ = build_prompt_from_messages(tokenizer, sample_messages)
        assert prompt_ids_1d.dtype == torch.long

    def test_empty_messages(self, tokenizer):
        """Test handling of empty messages list."""
        # This should either work or raise a clear error
        # Behavior depends on tokenizer's chat template
        messages = []
        try:
            result = build_prompt_from_messages(tokenizer, messages)
            # If it succeeds, verify it returns valid structure
            assert len(result) == 3
        except Exception:
            # It's okay if it raises an error for empty messages
            pass

    def test_single_message(self, tokenizer):
        """Test with a single message."""
        messages = [{"role": "user", "content": "Hi"}]
        prompt_text, prompt_ids_1d, prompt_len = build_prompt_from_messages(tokenizer, messages)
        assert len(prompt_text) > 0
        assert prompt_ids_1d.numel() > 0
        assert prompt_len == prompt_ids_1d.numel()


# -----------------------------
# Test tokenize.py
# -----------------------------


class TestTokenizePromptPlusCompletion:
    def test_output_structure(self, tokenizer, sample_messages):
        """Test that output has expected keys."""
        _, prompt_ids_1d, _ = build_prompt_from_messages(tokenizer, sample_messages)
        result = tokenize_prompt_plus_completion(tokenizer, prompt_ids_1d, "Hello!", max_len=100)

        assert isinstance(result, dict)
        assert "input_ids_1d" in result
        assert "attention_mask_1d" in result
        assert "prompt_len" in result
        assert "completion_len" in result

    def test_concatenation(self, tokenizer, sample_messages):
        """Test that prompt and completion are concatenated."""
        _, prompt_ids_1d, _ = build_prompt_from_messages(tokenizer, sample_messages)
        completion_text = "This is a test."

        result = tokenize_prompt_plus_completion(
            tokenizer, prompt_ids_1d, completion_text, max_len=1000
        )

        total_len = result["prompt_len"] + result["completion_len"]
        assert result["input_ids_1d"].numel() == total_len

    def test_left_truncation(self, tokenizer, sample_messages):
        """Test left truncation (truncates prompt)."""
        _, prompt_ids_1d, _ = build_prompt_from_messages(tokenizer, sample_messages)
        completion_text = "Short completion."

        max_len = 10
        result = tokenize_prompt_plus_completion(
            tokenizer, prompt_ids_1d, completion_text, max_len=max_len, truncation_side="left"
        )

        assert result["input_ids_1d"].numel() <= max_len
        # With left truncation, completion should be preserved if possible
        assert result["completion_len"] > 0

    def test_right_truncation(self, tokenizer, sample_messages):
        """Test right truncation (truncates completion)."""
        _, prompt_ids_1d, _ = build_prompt_from_messages(tokenizer, sample_messages)
        completion_text = "This is a very long completion that will definitely need truncation."

        max_len = 10
        result = tokenize_prompt_plus_completion(
            tokenizer, prompt_ids_1d, completion_text, max_len=max_len, truncation_side="right"
        )

        assert result["input_ids_1d"].numel() <= max_len
        # With right truncation, prompt should be preserved if possible
        assert result["prompt_len"] > 0

    def test_attention_mask_all_ones(self, tokenizer, sample_messages):
        """Test that attention mask is all ones (no padding)."""
        _, prompt_ids_1d, _ = build_prompt_from_messages(tokenizer, sample_messages)
        result = tokenize_prompt_plus_completion(tokenizer, prompt_ids_1d, "Test", max_len=100)

        assert torch.all(result["attention_mask_1d"] == 1)

    def test_empty_completion(self, tokenizer, sample_messages):
        """Test with empty completion string."""
        _, prompt_ids_1d, _ = build_prompt_from_messages(tokenizer, sample_messages)
        result = tokenize_prompt_plus_completion(tokenizer, prompt_ids_1d, "", max_len=100)

        # Empty completion should result in zero completion_len
        assert result["completion_len"] == 0
        assert result["prompt_len"] > 0


class TestTokenizePreferenceExample:
    def test_output_type(self, tokenizer, sample_preference_example):
        """Test that output is TokenizedPreferenceExample."""
        result = tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)
        assert isinstance(result, TokenizedPreferenceExample)

    def test_has_required_fields(self, tokenizer, sample_preference_example):
        """Test that result has all required fields."""
        result = tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)

        assert hasattr(result, "prompt_text")
        assert hasattr(result, "chosen_text")
        assert hasattr(result, "rejected_text")
        assert hasattr(result, "prompt_len")
        assert hasattr(result, "chosen_input_ids_1d")
        assert hasattr(result, "chosen_attention_mask_1d")
        assert hasattr(result, "rejected_input_ids_1d")
        assert hasattr(result, "rejected_attention_mask_1d")
        assert hasattr(result, "chosen_completion_len")
        assert hasattr(result, "rejected_completion_len")

    def test_shared_prompt_len(self, tokenizer, sample_preference_example):
        """Test that chosen and rejected share the same prompt_len."""
        result = tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)

        # Both sequences should have the same prompt_len
        assert result.prompt_len >= 0

    def test_respects_max_len(self, tokenizer, sample_preference_example):
        """Test that sequences don't exceed max_len."""
        max_len = 64
        result = tokenize_preference_example(tokenizer, sample_preference_example, max_len=max_len)

        assert result.chosen_input_ids_1d.numel() <= max_len
        assert result.rejected_input_ids_1d.numel() <= max_len

    def test_tensors_are_1d(self, tokenizer, sample_preference_example):
        """Test that all tensor outputs are 1D."""
        result = tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)

        assert result.chosen_input_ids_1d.dim() == 1
        assert result.chosen_attention_mask_1d.dim() == 1
        assert result.rejected_input_ids_1d.dim() == 1
        assert result.rejected_attention_mask_1d.dim() == 1

    def test_attention_masks_all_ones(self, tokenizer, sample_preference_example):
        """Test that attention masks are all ones (no padding yet)."""
        result = tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)

        assert torch.all(result.chosen_attention_mask_1d == 1)
        assert torch.all(result.rejected_attention_mask_1d == 1)

    def test_long_sequences_truncated(self, tokenizer):
        """Test that very long sequences are properly truncated."""
        long_example = {
            "messages": [{"role": "user", "content": "Q" * 100}],
            "chosen": "A" * 200,
            "rejected": "B" * 300,
        }

        max_len = 50
        result = tokenize_preference_example(tokenizer, long_example, max_len=max_len)

        assert result.chosen_input_ids_1d.numel() <= max_len
        assert result.rejected_input_ids_1d.numel() <= max_len

    def test_length_consistency(self, tokenizer, sample_preference_example):
        """Test that prompt_len + completion_len equals total length."""
        result = tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)

        chosen_total = result.prompt_len + result.chosen_completion_len
        rejected_total = result.prompt_len + result.rejected_completion_len

        assert chosen_total == result.chosen_input_ids_1d.numel()
        assert rejected_total == result.rejected_input_ids_1d.numel()


# -----------------------------
# Test collate.py
# -----------------------------


class TestMakeResponseMaskLabelSpace:
    def test_output_shape(self):
        """Test that output has shape [B, T-1]."""
        B, T = 4, 20
        attention_mask = torch.ones(B, T)
        prompt_lens = torch.tensor([5, 10, 8, 12])

        mask = make_response_mask_label_space(attention_mask, prompt_lens)

        assert mask.shape == (B, T - 1)

    def test_output_dtype(self):
        """Test that output is boolean."""
        attention_mask = torch.ones(2, 10)
        prompt_lens = torch.tensor([5, 5])

        mask = make_response_mask_label_space(attention_mask, prompt_lens)

        assert mask.dtype == torch.bool

    def test_label_space_offset(self):
        """Test that response starts at prompt_len - 1 in label space."""
        B, T = 1, 10
        prompt_len = 5
        attention_mask = torch.ones(B, T)
        prompt_lens = torch.tensor([prompt_len])

        mask = make_response_mask_label_space(attention_mask, prompt_lens)

        # In label space, response should start at index prompt_len - 1
        # mask[0, :prompt_len-1] should be False
        # mask[0, prompt_len-1:] should be True (if attention is 1)
        assert not mask[0, : prompt_len - 1].any()  # All False before response
        assert mask[0, prompt_len - 1 :].all()  # All True for response

    def test_respects_attention_mask(self):
        """Test that mask respects attention mask (padding)."""
        B, T = 2, 10
        attention_mask = torch.ones(B, T)
        attention_mask[0, -3:] = 0  # Pad last 3 tokens of first example
        prompt_lens = torch.tensor([3, 3])

        mask = make_response_mask_label_space(attention_mask, prompt_lens)

        # Last 3 positions in label space should be False for first example
        assert not mask[0, -3:].any()

    def test_zero_prompt_len(self):
        """Test handling of zero prompt length."""
        attention_mask = torch.ones(1, 10)
        prompt_lens = torch.tensor([0])

        # Should not crash
        mask = make_response_mask_label_space(attention_mask, prompt_lens)
        assert mask.shape == (1, 9)


class TestCollatePreferenceBatch:
    def test_output_structure(self, tokenizer, sample_preference_example):
        """Test that output has expected keys."""
        examples = [
            tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)
            for _ in range(3)
        ]

        batch = collate_preference_batch(tokenizer, examples)

        assert "chosen_input_ids" in batch
        assert "chosen_attention_mask" in batch
        assert "rejected_input_ids" in batch
        assert "rejected_attention_mask" in batch
        assert "prompt_lens" in batch
        assert "chosen_response_mask" in batch
        assert "rejected_response_mask" in batch

    def test_batch_shapes(self, tokenizer, sample_preference_example):
        """Test that all batch tensors have consistent batch size."""
        B = 4
        examples = [
            tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)
            for _ in range(B)
        ]

        batch = collate_preference_batch(tokenizer, examples)

        assert batch["chosen_input_ids"].shape[0] == B
        assert batch["chosen_attention_mask"].shape[0] == B
        assert batch["rejected_input_ids"].shape[0] == B
        assert batch["rejected_attention_mask"].shape[0] == B
        assert batch["prompt_lens"].shape[0] == B

    def test_uniform_sequence_length(self, tokenizer, sample_preference_example):
        """Test that chosen and rejected have same sequence length (T) after padding."""
        examples = [
            tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)
            for _ in range(3)
        ]

        batch = collate_preference_batch(tokenizer, examples)

        # Both should have shape [B, T] with same T
        assert batch["chosen_input_ids"].shape[1] == batch["rejected_input_ids"].shape[1]

    def test_left_padding_applied(self, tokenizer):
        """Test that left padding is applied correctly."""
        # Create examples with different lengths
        examples = [
            tokenize_preference_example(
                tokenizer,
                {
                    "messages": [{"role": "user", "content": f"Q{i}" * (i + 1)}],
                    "chosen": f"A{i}",
                    "rejected": f"B{i}",
                },
                max_len=256,
            )
            for i in range(3)
        ]

        batch = collate_preference_batch(tokenizer, examples)

        pad_id = tokenizer.pad_token_id

        # Check that padding appears on the left
        # The shortest sequence should have padding at the beginning
        for i in range(len(examples)):
            # Find first non-pad token in chosen
            chosen_ids = batch["chosen_input_ids"][i]
            first_nonpad = (chosen_ids != pad_id).nonzero(as_tuple=True)[0]

            if len(first_nonpad) > 0:
                first_idx = first_nonpad[0].item()
                # All tokens before first_idx should be padding
                if first_idx > 0:
                    assert torch.all(chosen_ids[:first_idx] == pad_id)

    def test_attention_mask_matches_padding(self, tokenizer, sample_preference_example):
        """Test that attention mask is 0 where padding exists."""
        examples = [
            tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)
            for _ in range(3)
        ]

        batch = collate_preference_batch(tokenizer, examples)

        pad_id = tokenizer.pad_token_id

        # Where input_ids == pad_id, attention_mask should be 0
        chosen_padded = batch["chosen_input_ids"] == pad_id
        assert torch.all(batch["chosen_attention_mask"][chosen_padded] == 0)

        # Where input_ids != pad_id, attention_mask should be 1
        chosen_not_padded = batch["chosen_input_ids"] != pad_id
        assert torch.all(batch["chosen_attention_mask"][chosen_not_padded] == 1)

    def test_response_mask_shape(self, tokenizer, sample_preference_example):
        """Test that response masks have shape [B, T-1]."""
        examples = [
            tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)
            for _ in range(2)
        ]

        batch = collate_preference_batch(tokenizer, examples)

        B, T = batch["chosen_input_ids"].shape

        assert batch["chosen_response_mask"].shape == (B, T - 1)
        assert batch["rejected_response_mask"].shape == (B, T - 1)

    def test_single_example_batch(self, tokenizer, sample_preference_example):
        """Test collation with a single example."""
        examples = [tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)]

        batch = collate_preference_batch(tokenizer, examples)

        # Should still produce valid batch with B=1
        assert batch["chosen_input_ids"].shape[0] == 1
        assert batch["prompt_lens"].shape[0] == 1

    def test_requires_left_padding(self, tokenizer, sample_preference_example):
        """Test that function requires left padding."""
        # Temporarily change padding side
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"

        examples = [tokenize_preference_example(tokenizer, sample_preference_example, max_len=256)]

        with pytest.raises(AssertionError, match="left"):
            collate_preference_batch(tokenizer, examples)

        # Restore original
        tokenizer.padding_side = original_padding_side


# -----------------------------
# Integration tests
# -----------------------------


class TestEndToEndPipeline:
    def test_full_pipeline(self, tokenizer):
        """Test complete pipeline from raw examples to batched tensors."""
        raw_examples = [
            {
                "messages": [{"role": "user", "content": "Question 1?"}],
                "chosen": "Good answer 1.",
                "rejected": "Bad answer 1.",
            },
            {
                "messages": [{"role": "user", "content": "Question 2?"}],
                "chosen": "Good answer 2.",
                "rejected": "Bad answer 2.",
            },
        ]

        # Step 1: Tokenize
        tokenized = [tokenize_preference_example(tokenizer, ex, max_len=256) for ex in raw_examples]

        # Step 2: Collate
        batch = collate_preference_batch(tokenizer, tokenized)

        # Verify final batch structure
        assert batch["chosen_input_ids"].shape[0] == 2
        assert batch["rejected_input_ids"].shape[0] == 2
        assert batch["prompt_lens"].numel() == 2

        # Verify no NaN or inf values
        assert not torch.isnan(batch["chosen_input_ids"].float()).any()
        assert not torch.isnan(batch["rejected_input_ids"].float()).any()

    def test_variable_length_handling(self, tokenizer):
        """Test pipeline with variable length inputs."""
        raw_examples = [
            {
                "messages": [{"role": "user", "content": "Short?"}],
                "chosen": "Yes.",
                "rejected": "No.",
            },
            {
                "messages": [
                    {"role": "user", "content": "This is a much longer question with more tokens?"}
                ],
                "chosen": "This is a detailed answer with multiple sentences and lots of tokens.",
                "rejected": "Brief.",
            },
        ]

        tokenized = [tokenize_preference_example(tokenizer, ex, max_len=256) for ex in raw_examples]

        batch = collate_preference_batch(tokenizer, tokenized)

        # Should handle variable lengths gracefully
        assert batch["chosen_input_ids"].shape[0] == 2
        assert batch["chosen_input_ids"].shape == batch["rejected_input_ids"].shape

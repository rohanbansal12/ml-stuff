import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path to import engine
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import (
    DEVICE,
    MODEL_NAME,
    GenOutput,
    RLHFEngine,
    completion_logprobs,
    generate_one,
    load_model,
    load_tokenizer,
)


@pytest.fixture(scope="module")
def device():
    """Fixture for device."""
    return torch.device(DEVICE)


@pytest.fixture(scope="module")
def dtype():
    """Fixture for dtype - returns bf16 if supported, else fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@pytest.fixture(scope="module")
def tokenizer():
    """Fixture for tokenizer - loaded once per module."""
    return load_tokenizer(MODEL_NAME)


@pytest.fixture(scope="module")
def model(dtype, device):
    """Fixture for model - loaded once per module."""
    return load_model(MODEL_NAME, dtype, device)


@pytest.fixture(scope="module")
def engine(model, tokenizer):
    """Fixture for RLHFEngine - created once per module."""
    return RLHFEngine(model, tokenizer)


@pytest.fixture
def sample_messages():
    """Fixture for sample chat messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]


class TestTokenizer:
    """Tests for tokenizer loading and configuration."""

    def test_tokenizer_loaded(self, tokenizer):
        """Test that tokenizer loads successfully."""
        assert tokenizer is not None

    def test_padding_side_left(self, tokenizer):
        """Test that padding side is set to left."""
        assert tokenizer.padding_side == "left"

    def test_truncation_side_left(self, tokenizer):
        """Test that truncation side is set to left."""
        assert tokenizer.truncation_side == "left"

    def test_special_tokens_exist(self, tokenizer):
        """Test that essential special tokens are defined."""
        # EOS and PAD are essential, BOS is optional (some models like Qwen don't use it)
        assert tokenizer.eos_token is not None
        assert tokenizer.pad_token is not None

    def test_token_ids_valid(self, tokenizer):
        """Test that token IDs are valid integers."""
        # EOS and PAD are essential, BOS is optional
        assert isinstance(tokenizer.eos_token_id, int)
        assert isinstance(tokenizer.pad_token_id, int)
        # BOS is optional - only check if it exists
        if tokenizer.bos_token is not None:
            assert isinstance(tokenizer.bos_token_id, int)

    def test_chat_template_applies(self, tokenizer, sample_messages):
        """Test that chat template can be applied."""
        result = tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tokenization_works(self, tokenizer, sample_messages):
        """Test that tokenization produces valid output."""
        chat_str = tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(chat_str, return_tensors="pt", add_special_tokens=False)
        assert "input_ids" in encoded
        assert encoded["input_ids"].shape[0] == 1
        assert encoded["input_ids"].shape[1] > 0


class TestModel:
    """Tests for model loading and configuration."""

    def test_model_loaded(self, model):
        """Test that model loads successfully."""
        assert model is not None

    def test_model_on_correct_device(self, model, device):
        """Test that model is on the expected device."""
        param_device = next(model.parameters()).device
        assert param_device.type == device.type

    def test_model_dtype(self, model, dtype):
        """Test that model has correct dtype."""
        param_dtype = next(model.parameters()).dtype
        assert param_dtype == dtype

    def test_model_in_eval_mode(self, model):
        """Test that model is in eval mode."""
        assert not model.training

    def test_model_forward_pass(self, model, tokenizer, sample_messages, device):
        """Test that model can perform a forward pass."""
        chat_str = tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(chat_str, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            output = model(**enc)

        assert hasattr(output, "logits")
        assert output.logits.ndim == 3  # [batch, seq_len, vocab]
        assert output.logits.shape[-1] > 0  # vocab size


class TestGeneration:
    """Tests for generation functionality."""

    def test_generate_one_returns_genoutput(self, model, tokenizer, sample_messages):
        """Test that generate_one returns a GenOutput object."""
        result = generate_one(model, tokenizer, sample_messages, max_new_tokens=10, do_sample=False)
        assert isinstance(result, GenOutput)

    def test_genoutput_fields(self, model, tokenizer, sample_messages):
        """Test that GenOutput has all required fields."""
        result = generate_one(model, tokenizer, sample_messages, max_new_tokens=10, do_sample=False)

        assert hasattr(result, "prompt_text")
        assert hasattr(result, "prompt_len")
        assert hasattr(result, "output_ids")
        assert hasattr(result, "completion_ids")
        assert hasattr(result, "output_text")
        assert hasattr(result, "completion_text")

    def test_deterministic_generation(self, model, tokenizer, sample_messages):
        """Test that do_sample=False produces deterministic outputs."""
        result1 = generate_one(
            model, tokenizer, sample_messages, max_new_tokens=10, do_sample=False
        )
        result2 = generate_one(
            model, tokenizer, sample_messages, max_new_tokens=10, do_sample=False
        )

        assert torch.equal(result1.completion_ids, result2.completion_ids)

    def test_seeded_generation_deterministic(self, model, tokenizer, sample_messages):
        """Test that seeded sampling produces deterministic outputs."""
        result1 = generate_one(
            model, tokenizer, sample_messages, max_new_tokens=10, do_sample=True, seed=42
        )
        result2 = generate_one(
            model, tokenizer, sample_messages, max_new_tokens=10, do_sample=True, seed=42
        )

        assert torch.equal(result1.completion_ids, result2.completion_ids)

    def test_different_seeds_different_outputs(self, model, tokenizer, sample_messages):
        """Test that different seeds produce different outputs."""
        result1 = generate_one(
            model, tokenizer, sample_messages, max_new_tokens=10, do_sample=True, seed=42
        )
        result2 = generate_one(
            model, tokenizer, sample_messages, max_new_tokens=10, do_sample=True, seed=99
        )

        # Note: there's a small chance they could be equal, but very unlikely
        assert not torch.equal(result1.completion_ids, result2.completion_ids)

    def test_completion_ids_length(self, model, tokenizer, sample_messages):
        """Test that completion_ids length is <= max_new_tokens."""
        max_new = 20
        result = generate_one(
            model, tokenizer, sample_messages, max_new_tokens=max_new, do_sample=False
        )

        assert result.completion_ids.numel() <= max_new

    def test_prompt_len_correct(self, model, tokenizer, sample_messages):
        """Test that prompt_len matches the actual prompt length."""
        result = generate_one(model, tokenizer, sample_messages, max_new_tokens=10, do_sample=False)

        # Verify that output_ids = prompt + completion
        assert result.output_ids.numel() == result.prompt_len + result.completion_ids.numel()


class TestCompletionLogprobs:
    """Tests for completion log probability computation."""

    def test_completion_logprobs_returns_correct_shapes(self, model, tokenizer, device):
        """Test that completion_logprobs returns tensors with correct shapes."""
        # Create simple input
        text = "Hello world"
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
        prompt_lens = torch.tensor([input_ids.shape[1] - 1], device=device)

        sum_logp, mean_logp, token_logp, completion_mask = completion_logprobs(
            model, input_ids, attention_mask, prompt_lens
        )

        batch_size = input_ids.shape[0]
        assert sum_logp.shape == (batch_size,)
        assert mean_logp.shape == (batch_size,)
        assert token_logp.ndim == 2
        assert completion_mask.ndim == 2

    def test_prompt_only_zero_completion_tokens(self, model, tokenizer, sample_messages, device):
        """Test that prompt-only input selects 0 completion tokens."""
        chat_str = tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(chat_str, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
        prompt_lens = torch.tensor([input_ids.shape[1]], device=device)

        _, _, _, completion_mask = completion_logprobs(
            model, input_ids, attention_mask, prompt_lens
        )

        assert completion_mask.sum().item() == 0

    def test_completion_logprobs_values_finite(self, model, tokenizer, device):
        """Test that log probabilities are finite values."""
        text = "Hello world this is a test"
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
        # Treat last 3 tokens as completion
        prompt_lens = torch.tensor([input_ids.shape[1] - 3], device=device)

        sum_logp, mean_logp, token_logp, _ = completion_logprobs(
            model, input_ids, attention_mask, prompt_lens
        )

        assert torch.isfinite(sum_logp).all()
        assert torch.isfinite(mean_logp).all()
        assert torch.isfinite(token_logp).all()


class TestRLHFEngine:
    """Tests for RLHFEngine class."""

    def test_engine_initializes(self, engine):
        """Test that RLHFEngine initializes successfully."""
        assert engine is not None
        assert engine.model is not None
        assert engine.tokenizer is not None

    def test_engine_tokenizer_config(self, engine):
        """Test that engine sets tokenizer config correctly."""
        assert engine.tokenizer.padding_side == "left"
        assert engine.tokenizer.truncation_side == "left"

    def test_build_prompt(self, engine, sample_messages):
        """Test that build_prompt returns correct types."""
        prompt_text, prompt_ids, prompt_len = engine.build_prompt(sample_messages)

        assert isinstance(prompt_text, str)
        assert isinstance(prompt_ids, torch.Tensor)
        assert isinstance(prompt_len, int)
        assert prompt_ids.ndim == 1
        assert prompt_len == prompt_ids.numel()

    def test_generate(self, engine, sample_messages):
        """Test that engine.generate works."""
        result = engine.generate(sample_messages, max_new_tokens=10, do_sample=False)

        assert isinstance(result, GenOutput)
        assert result.completion_ids.numel() > 0

    def test_logprob_of_completion(self, engine, sample_messages, device):
        """Test that logprob_of_completion returns a float."""
        # Generate a completion first
        gen_result = engine.generate(sample_messages, max_new_tokens=10, do_sample=False)
        completion_ids = gen_result.completion_ids

        # Score it
        logprob = engine.logprob_of_completion(sample_messages, completion_ids)

        assert isinstance(logprob, float)
        assert torch.isfinite(torch.tensor(logprob))

    def test_score_pair(self, engine, sample_messages, tokenizer):
        """Test that score_pair returns two floats."""
        # Create two different completions
        chosen_text = " Hello!"
        rejected_text = " Goodbye!"

        chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=False, return_tensors="pt")[0]
        rejected_ids = tokenizer.encode(
            rejected_text, add_special_tokens=False, return_tensors="pt"
        )[0]

        logp_chosen, logp_rejected = engine.score_pair(sample_messages, chosen_ids, rejected_ids)

        assert isinstance(logp_chosen, float)
        assert isinstance(logp_rejected, float)
        assert torch.isfinite(torch.tensor([logp_chosen, logp_rejected])).all()

    def test_decode_completion(self, engine, tokenizer):
        """Test that decode_completion works."""
        text = "Hello world"
        ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]

        decoded = engine.decode_completion(ids)

        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_set_seed(self, engine):
        """Test that set_seed runs without error."""
        # Just verify it doesn't crash
        engine.set_seed(42)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_generation_and_scoring_pipeline(self, engine, sample_messages):
        """Test complete pipeline: generate then score."""
        # Generate a completion
        gen_result = engine.generate(sample_messages, max_new_tokens=15, do_sample=False)

        # Score the generated completion
        logprob = engine.logprob_of_completion(sample_messages, gen_result.completion_ids)

        # Should get a finite negative log probability
        assert isinstance(logprob, float)
        assert logprob < 0  # Log probabilities are negative
        assert torch.isfinite(torch.tensor(logprob))

    def test_score_consistency(self, engine, sample_messages):
        """Test that scoring the same completion twice gives same result."""
        gen_result = engine.generate(sample_messages, max_new_tokens=10, do_sample=False)

        score1 = engine.logprob_of_completion(sample_messages, gen_result.completion_ids)
        score2 = engine.logprob_of_completion(sample_messages, gen_result.completion_ids)

        assert abs(score1 - score2) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

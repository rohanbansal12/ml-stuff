import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GPTConfig:
    d_model: int = 256
    num_heads: int = 8
    num_kv_heads: Optional[int] = None  # None = same as num_heads (MHA)
    max_seq_len: int = 256
    num_layers: int = 6
    vocab_size: int = 50257
    dropout: float = 0.1
    rope: bool = True
    rmsnorm: bool = True

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )

    @property
    def d_k(self) -> int:
        return self.d_model // self.num_heads

# =============================================================================
# Slot Allocator
# =============================================================================

class SlotAllocator:
    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.free_slots = set(range(num_slots))
        self.active_sequences = {}  # seq_id -> slot_id
    
    def allocate(self, seq_id: int) -> Optional[int]:
        """Allocate a slot for a new sequence. Returns None if full."""
        if seq_id in self.active_sequences:
            raise ValueError(f"Sequence {seq_id} already has a slot")
        if not self.free_slots:
            return None
        slot = self.free_slots.pop()
        self.active_sequences[seq_id] = slot
        return slot
    
    def release(self, seq_id: int):
        """Release a slot when sequence completes."""
        if seq_id not in self.active_sequences:
            raise ValueError(f"Sequence {seq_id} not active")
        slot = self.active_sequences.pop(seq_id)
        self.free_slots.add(slot)
    
    def get_slot(self, seq_id: int) -> Optional[int]:
        """Get slot for an active sequence. Returns None if not found."""
        return self.active_sequences.get(seq_id)
    
    def is_full(self) -> bool:
        return len(self.free_slots) == 0
    
    def num_active(self) -> int:
        return len(self.active_sequences)
    
    def num_free(self) -> int:
        return len(self.free_slots)
    
    def get_active_batch_indices(self, device: torch.device) -> torch.Tensor:
        """Get tensor of active slot indices for batched operations."""
        slots = list(self.active_sequences.values())
        return torch.tensor(slots, dtype=torch.long, device=device)
    
    def reset(self):
        """Reset all slots to free."""
        self.free_slots = set(range(self.num_slots))
        self.active_sequences.clear()
    

# =============================================================================
# KV Cache
# =============================================================================


class KVCache:
    """
    Pre-allocated KV cache for efficient inference.

    Eliminates memory allocation during generation by pre-allocating
    a fixed-size buffer and writing to specific positions.
    """

    def __init__(
        self,
        max_batch: int,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        d_k: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.d_k = d_k
        self.device = device
        self.dtype = dtype

        # Pre-allocate cache
        # Shape: (max_batch, num_layers, 2, num_kv_heads, max_seq_len, d_k)
        # The "2" dimension is for K (index 0) and V (index 1)
        self.cache = torch.zeros(
            max_batch,
            num_layers,
            2,
            num_kv_heads,
            max_seq_len,
            d_k,
            device=device,
            dtype=dtype,
        )

        # Track current sequence length for each batch slot
        self.current_lens = torch.zeros(max_batch, dtype=torch.int32, device=device)

    def update(
        self,
        batch_indices: torch.Tensor,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ):
        """
        Write new K, V to cache at current positions.

        Args:
            batch_indices: (batch,) which cache slots to update
            layer_idx: which layer's cache to update
            new_k: (batch, num_kv_heads, seq_len, d_k)
            new_v: (batch, num_kv_heads, seq_len, d_k)
        """
        batch_size = batch_indices.size(0)
        seq_len = new_k.size(2)
        start_positions = self.current_lens[batch_indices]

        for i in range(batch_size):
            slot = batch_indices[i]
            start = start_positions[i].item()
            end = start + seq_len

            assert end <= self.max_seq_len, (
                f"Sequence exceeded max length: {end} > {self.max_seq_len}"
            )

            self.cache[slot, layer_idx, 0, :, start:end, :] = new_k[i]
            self.cache[slot, layer_idx, 1, :, start:end, :] = new_v[i]

    def get(
        self, batch_indices: torch.Tensor, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve K, V from cache up to current lengths.

        Args:
            batch_indices: (batch,) which cache slots to read
            layer_idx: which layer's cache to read

        Returns:
            K, V: (batch, num_kv_heads, max_len, d_k) or (None, None) if empty
        """
        max_len = self.current_lens[batch_indices].max().item()
        if max_len == 0:
            return None, None

        K = self.cache[batch_indices][:, layer_idx, 0, :, :max_len, :]
        V = self.cache[batch_indices][:, layer_idx, 1, :, :max_len, :]
        return K, V

    def increment_lens(self, batch_indices: torch.Tensor, amount: int = 1):
        """Increment sequence lengths after processing tokens."""
        self.current_lens[batch_indices] += amount

    def get_lengths(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """Get current lengths for given batch slots."""
        return self.current_lens[batch_indices]

    def reset(self, batch_indices: torch.Tensor):
        """Reset specific cache slots for reuse."""
        self.current_lens[batch_indices] = 0

    def reset_all(self):
        """Reset entire cache."""
        self.current_lens.zero_()

    @classmethod
    def from_config(
        cls,
        config: "GPTConfig",
        max_batch: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "KVCache":
        """Create cache from model config."""
        return cls(
            max_batch=max_batch,
            max_seq_len=config.max_seq_len,
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            d_k=config.d_k,
            device=device,
            dtype=dtype,
        )


class KVCacheView:
    """
    A view into KVCache for a specific batch of sequences.

    Provides a clean interface for the model to interact with the cache
    without knowing about slot management details.
    """

    def __init__(self, cache: KVCache, batch_indices: torch.Tensor):
        self.cache = cache
        self.batch_indices = batch_indices
        # Store lengths at view creation for position offset calculation
        self._position_offset = cache.get_lengths(batch_indices).clone()

    def get(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached K, V for a layer."""
        return self.cache.get(self.batch_indices, layer_idx)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Update cache for a layer with new K, V."""
        self.cache.update(self.batch_indices, layer_idx, k, v)

    def get_position_offset(self) -> int:
        """
        Get position offset for RoPE.

        Returns the sequence position where new tokens start.
        For uniform batches, returns a single int.
        """
        # For simplicity, assume uniform lengths in batch
        # TODO: Handle variable lengths with per-sequence offsets
        return self._position_offset[0].item()

    def finalize(self, seq_len: int):
        """
        Call after forward pass to increment cached lengths.

        Args:
            seq_len: Number of tokens that were just processed
        """
        self.cache.increment_lens(self.batch_indices, seq_len)

    @property
    def current_max_len(self) -> int:
        """Current maximum sequence length in this batch."""
        return self.cache.get_lengths(self.batch_indices).max().item()


# =============================================================================
# Model Components
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm


class RoPE(nn.Module):
    """
    Rotary Position Embedding.

    Encodes position information by rotating query and key vectors,
    allowing the model to learn relative positions through the
    rotation-invariant dot product.
    """

    def __init__(self, d_k: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires head dimension to be even"

        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos/sin values for all positions."""
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        angles = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)  # (seq_len, d_k/2)
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """
        Apply rotary embeddings.

        Args:
            x: (batch, num_heads, seq_len, d_k)
            position_offset: Starting position (for KV-cache inference)

        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.size(2)

        # Get cos/sin for relevant positions
        cos = self.cos_cache[position_offset : position_offset + seq_len]
        sin = self.sin_cache[position_offset : position_offset + seq_len]

        # Reshape for broadcasting: (1, 1, seq_len, d_k/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation using the "interleaved" formula
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with support for:
    - Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
    - Rotary Position Embeddings (RoPE)
    - KV-Cache for efficient inference
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_queries_per_kv = config.num_heads // config.num_kv_heads
        self.d_k = config.d_k
        self.d_model = config.d_model
        self.layer_idx = layer_idx

        # Projections
        self.W_Q = nn.Linear(config.d_model, config.num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(config.d_model, config.num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(config.d_model, config.num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=False)

        # Causal mask
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len)
        )

        self.dropout = nn.Dropout(config.dropout)

        # RoPE
        if config.rope:
            self.rope = RoPE(self.d_k, config.max_seq_len)
        else:
            self.rope = None

    def expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match Q heads via repetition."""
        if self.num_kv_heads == self.num_heads:
            return x
        return x.repeat_interleave(self.num_queries_per_kv, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        cache_view: Optional[KVCacheView] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional KV-caching.

        Args:
            x: (batch, seq_len, d_model) input tensor
            cache_view: Optional cache view for inference

        Returns:
            (batch, seq_len, d_model) output tensor
        """
        b, s, _ = x.size()

        # Compute Q, K, V projections
        Q = self.W_Q(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(b, s, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(b, s, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Get position offset for RoPE
        position_offset = (
            cache_view.get_position_offset() if cache_view is not None else 0
        )

        # Apply RoPE to Q and K
        if self.rope is not None:
            Q = self.rope(Q, position_offset)
            K = self.rope(K, position_offset)

        # Handle KV cache
        if cache_view is not None:
            # Update cache with new K, V (rotated)
            cache_view.update(self.layer_idx, K, V)

            # Get full K, V sequence from cache
            K_cached, V_cached = cache_view.get(self.layer_idx)
            if K_cached is not None:
                K, V = K_cached, V_cached

        # Expand KV heads for attention computation
        K_exp = self.expand_kv(K)
        V_exp = self.expand_kv(V)

        # Compute attention scores
        total_kv_len = K_exp.size(2)
        scores = (Q @ K_exp.transpose(-1, -2)) * (1.0 / math.sqrt(self.d_k))

        # Apply causal mask
        # Q positions: [position_offset, position_offset + s)
        # K positions: [0, total_kv_len)
        mask = self.causal_mask[
            :, :, position_offset : position_offset + s, :total_kv_len
        ]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        if self.training:
            attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ V_exp

        # Reshape and project output
        out = out.transpose(1, 2).reshape(b, s, self.d_model)
        out = self.W_O(out)

        if self.training:
            out = self.dropout(out)

        return out


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Uses gated linear units with SiLU activation, as used in LLaMA and other
    modern architectures. More expressive than standard FFN with similar params.
    """

    def __init__(self, config: GPTConfig, hidden_mult: float = 8 / 3):
        super().__init__()

        hidden = int(config.d_model * hidden_mult)
        hidden = (
            (hidden + 255) // 256
        ) * 256  # Round to multiple of 256 for efficiency

        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()

        self.attn = MultiHeadAttention(config, layer_idx)
        self.ffn = SwiGLUFFN(config)

        norm_cls = RMSNorm if config.rmsnorm else nn.LayerNorm
        self.ln1 = norm_cls(config.d_model)
        self.ln2 = norm_cls(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        cache_view: Optional[KVCacheView] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache_view=cache_view)
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# GPT Model
# =============================================================================


class GPT(nn.Module):
    """
    GPT-style decoder-only transformer.

    Features:
    - Configurable attention: MHA, GQA, or MQA
    - Optional RoPE positional encoding
    - KV-cache support for efficient generation
    - Weight tying between token embeddings and output projection
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings (only if not using RoPE)
        if not config.rope:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.pos_emb = None

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [Block(config, layer_idx=i) for i in range(config.num_layers)]
        )

        # Final layer norm
        norm_cls = RMSNorm if config.rmsnorm else nn.LayerNorm
        self.ln_f = norm_cls(config.d_model)

        # Output projection (tied with token embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            torch.nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        cache_view: Optional[KVCacheView] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len) token indices
            cache_view: Optional KVCacheView for inference

        Returns:
            (batch, seq_len, vocab_size) logits
        """
        b, t = x.size()

        # Token embeddings
        h = self.token_emb(x)

        # Add positional embeddings if not using RoPE
        if self.pos_emb is not None:
            if cache_view is not None:
                offset = cache_view.get_position_offset()
                positions = torch.arange(offset, offset + t, device=x.device)
            else:
                positions = torch.arange(t, device=x.device)
            h = h + self.pos_emb(positions)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cache_view=cache_view)

        # Output projection
        h = self.ln_f(h)
        logits = self.lm_head(h)

        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_emb is not None:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    @staticmethod
    def top_p_top_k(logits: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
        """Apply top-p (nucleus) and top-k filtering."""
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Find cutoff
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            # Scatter mask back to original order
            mask = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
            logits = logits.masked_fill(mask, float("-inf"))

        return logits

    def _sample_token(
        self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float
    ) -> torch.Tensor:
        """Sample next token from logits."""
        logits = logits / temperature

        if top_k > 0 or top_p > 0.0:
            logits = self.top_p_top_k(logits, top_p, top_k)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with KV-caching.

        Args:
            x: (batch, seq_len) input token indices
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (0 to disable)
            top_p: Top-p nucleus filtering (0.0 to disable)
            cache: Optional pre-allocated KVCache

        Returns:
            (batch, seq_len + max_new_tokens) generated sequence
        """
        self.eval()
        b, t = x.size()
        device = x.device
        temperature = max(temperature, 1e-8)

        # Create cache if not provided
        if cache is None:
            total_len = t + max_new_tokens
            cache = KVCache(
                max_batch=b,
                max_seq_len=total_len,
                num_layers=len(self.blocks),
                num_kv_heads=self.config.num_kv_heads,
                d_k=self.config.d_k,
                device=device,
                dtype=next(self.parameters()).dtype,
            )

        batch_indices = torch.arange(b, device=device)

        # Prefill: process entire prompt
        cache_view = KVCacheView(cache, batch_indices)
        logits = self.forward(x, cache_view=cache_view)
        cache_view.finalize(t)

        # Sample first new token
        next_token = self._sample_token(logits[:, -1, :], temperature, top_k, top_p)
        x = torch.cat([x, next_token], dim=1)

        # Generate remaining tokens one at a time
        for _ in range(max_new_tokens - 1):
            cache_view = KVCacheView(cache, batch_indices)
            logits = self.forward(next_token, cache_view=cache_view)
            cache_view.finalize(1)

            next_token = self._sample_token(logits[:, -1, :], temperature, top_k, top_p)
            x = torch.cat([x, next_token], dim=1)

        return x

    @torch.no_grad()
    def generate_without_cache(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate tokens without KV-cache (slower but simpler).
        Useful for correctness testing against cached version.
        """
        self.eval()
        temperature = max(temperature, 1e-8)

        for _ in range(max_new_tokens):
            # Truncate if too long
            x_cond = (
                x
                if x.size(1) <= self.config.max_seq_len
                else x[:, -self.config.max_seq_len :]
            )

            logits = self.forward(x_cond)
            next_token = self._sample_token(logits[:, -1, :], temperature, top_k, top_p)
            x = torch.cat([x, next_token], dim=1)

        return x


# =============================================================================
# Convenience Functions
# =============================================================================


def create_model(
    d_model: int = 256,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    max_seq_len: int = 256,
    num_layers: int = 6,
    vocab_size: int = 50257,
    dropout: float = 0.1,
    rope: bool = True,
    rmsnorm: bool = True,
) -> GPT:
    """Convenience function to create a GPT model."""
    config = GPTConfig(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        vocab_size=vocab_size,
        dropout=dropout,
        rope=rope,
        rmsnorm=rmsnorm,
    )
    return GPT(config)

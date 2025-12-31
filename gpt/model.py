import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Generator


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
    attn_type: str = "standard"  # "standard", "flash", "flash2", "sdpa"

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.attn_type in ("standard", "flash", "flash2", "sdpa"), (
            "attn_type must be one of: standard, flash, flash2, sdpa"
        )

    @property
    def d_k(self) -> int:
        return self.d_model // self.num_heads


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


# =============================================================================
# Attention Implementations
# =============================================================================


class FlashAttentionNaive(nn.Module):
    """
    Naive PyTorch implementation of FlashAttention (v1 algorithm).

    This implements the tiled attention with online softmax for educational purposes.
    It won't be faster than standard attention due to Python loop overhead,
    but demonstrates the memory-efficient algorithm.
    """

    def __init__(self, d_k: int, M: Optional[int] = None, causal: bool = True):
        super().__init__()
        self.d_k = d_k
        self.M = M or 1024 * 64
        self.B_c = math.ceil(self.M / (4 * self.d_k))
        self.B_r = min(self.B_c, self.d_k)
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.causal = causal

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        b, num_heads, seq_len, d_k = Q.shape
        O = torch.zeros_like(Q)
        l = torch.zeros(b, num_heads, seq_len, 1, device=Q.device, dtype=Q.dtype)
        m = torch.full(
            (b, num_heads, seq_len, 1), float("-inf"), device=Q.device, dtype=Q.dtype
        )

        for j in range(0, seq_len, self.B_c):
            j_end = min(j + self.B_c, seq_len)
            K_j = K[:, :, j:j_end, :]
            V_j = V[:, :, j:j_end, :]

            for i in range(0, seq_len, self.B_r):
                i_end = min(i + self.B_r, seq_len)

                # Skip fully masked blocks for causal attention
                if self.causal and j > (i_end - 1):
                    continue

                Q_i = Q[:, :, i:i_end, :]
                O_i = O[:, :, i:i_end, :]
                l_i = l[:, :, i:i_end, :]
                m_i = m[:, :, i:i_end, :]

                S_ij = (Q_i @ K_j.transpose(-1, -2)) * self.scale

                # Apply causal mask within block
                if self.causal:
                    q_pos = torch.arange(i, i_end, device=Q.device).unsqueeze(1)
                    k_pos = torch.arange(j, j_end, device=Q.device).unsqueeze(0)
                    causal_mask = k_pos > q_pos
                    S_ij = S_ij.masked_fill(
                        causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )

                m_ij = S_ij.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m_i, m_ij)

                P_ij = torch.exp(S_ij - m_new)
                exp_m_i = torch.exp(m_i - m_new)
                exp_m_ij = torch.exp(m_ij - m_new)

                l_new = exp_m_i * l_i + P_ij.sum(dim=-1, keepdim=True)
                O_new = (exp_m_i * l_i * O_i + exp_m_ij * P_ij @ V_j) / l_new

                O[:, :, i:i_end, :] = O_new
                l[:, :, i:i_end, :] = l_new
                m[:, :, i:i_end, :] = m_new

        return O


class FlashAttention2Naive(nn.Module):
    """
    Naive PyTorch implementation of FlashAttention-2 algorithm.

    Key differences from v1:
    - Outer loop over Q blocks, inner loop over KV blocks (better for causal)
    - Deferred normalization until after inner loop completes
    - Returns logsumexp L for potential use in backward pass
    """

    def __init__(self, d_k: int, M: Optional[int] = None, causal: bool = True):
        super().__init__()
        self.d_k = d_k
        self.M = M or 1024 * 64
        self.B_c = math.ceil(self.M / (4 * self.d_k))
        self.B_r = min(self.B_c, self.d_k)
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.causal = causal

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, num_heads, seq_len, d_k = Q.shape
        O = torch.zeros_like(Q)
        L = torch.zeros(b, num_heads, seq_len, 1, device=Q.device, dtype=Q.dtype)

        for i in range(0, seq_len, self.B_r):
            i_end = min(i + self.B_r, seq_len)
            Br = i_end - i
            Q_i = Q[:, :, i:i_end, :]

            O_i = torch.zeros(b, num_heads, Br, d_k, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(b, num_heads, Br, 1, device=Q.device, dtype=Q.dtype)
            m_i = torch.full(
                (b, num_heads, Br, 1), float("-inf"), device=Q.device, dtype=Q.dtype
            )

            j_max = i_end if self.causal else seq_len

            for j in range(0, j_max, self.B_c):
                j_end = min(j + self.B_c, j_max)

                K_j = K[:, :, j:j_end, :]
                V_j = V[:, :, j:j_end, :]

                S_ij = (Q_i @ K_j.transpose(-1, -2)) * self.scale

                if self.causal:
                    q_pos = torch.arange(i, i_end, device=Q.device).unsqueeze(1)
                    k_pos = torch.arange(j, j_end, device=Q.device).unsqueeze(0)
                    causal_mask = k_pos > q_pos
                    S_ij = S_ij.masked_fill(
                        causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )

                m_ij = S_ij.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m_i, m_ij)

                P_ij = torch.exp(S_ij - m_new)
                exp_m = torch.exp(m_i - m_new)

                l_i = exp_m * l_i + P_ij.sum(dim=-1, keepdim=True)
                O_i = O_i * exp_m + P_ij @ V_j

                m_i = m_new

            O_i = O_i / l_i
            L_i = m_i + torch.log(l_i)

            O[:, :, i:i_end, :] = O_i
            L[:, :, i:i_end, :] = L_i

        return O, L


def attention_standard(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor,
    dropout: nn.Dropout,
    training: bool,
) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    scale = 1.0 / math.sqrt(Q.size(-1))
    scores = (Q @ K.transpose(-1, -2)) * scale
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if training:
        attn = dropout(attn)
    return attn @ V


def attention_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """PyTorch's scaled_dot_product_attention (uses FlashAttention when available)."""
    return F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=None,
        dropout_p=dropout_p if training else 0.0,
        is_causal=is_causal,
    )


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with support for:
    - Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
    - Rotary Position Embeddings (RoPE)
    - KV-Cache for efficient inference
    - Multiple attention implementations (standard, flash, flash2, sdpa)
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_queries_per_kv = config.num_heads // config.num_kv_heads
        self.d_k = config.d_k
        self.d_model = config.d_model
        self.layer_idx = layer_idx
        self.attn_type = config.attn_type
        self.dropout_p = config.dropout

        # Projections
        self.W_Q = nn.Linear(config.d_model, config.num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(config.d_model, config.num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(config.d_model, config.num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=False)

        # Causal mask (for standard attention)
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

        # Flash attention modules (naive implementations)
        if config.attn_type == "flash":
            self.flash_attn = FlashAttentionNaive(self.d_k, causal=True)
        elif config.attn_type == "flash2":
            self.flash_attn = FlashAttention2Naive(self.d_k, causal=True)
        else:
            self.flash_attn = None

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

        # Compute attention using selected implementation
        total_kv_len = K_exp.size(2)

        if self.attn_type == "standard":
            mask = self.causal_mask[
                :, :, position_offset : position_offset + s, :total_kv_len
            ]
            out = attention_standard(Q, K_exp, V_exp, mask, self.dropout, self.training)

        elif self.attn_type == "sdpa":
            # SDPA handles causal masking internally
            # Note: doesn't support arbitrary position offsets well, best for training
            out = attention_sdpa(
                Q,
                K_exp,
                V_exp,
                is_causal=(cache_view is None),
                dropout_p=self.dropout_p,
                training=self.training,
            )

        elif self.attn_type in ("flash", "flash2"):
            # Naive flash attention implementations
            # Note: these don't support KV-cache well due to causal mask assumptions
            if cache_view is not None:
                # Fall back to standard for cached inference
                mask = self.causal_mask[
                    :, :, position_offset : position_offset + s, :total_kv_len
                ]
                out = attention_standard(
                    Q, K_exp, V_exp, mask, self.dropout, self.training
                )
            else:
                result = self.flash_attn(Q, K_exp, V_exp)
                out = result[0] if isinstance(result, tuple) else result
        else:
            raise ValueError(f"Unknown attention type: {self.attn_type}")

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
# Slot Allocator
# =============================================================================


class SlotAllocator:
    """Manages allocation of cache slots to sequences."""

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

    def get_active_slots(self) -> List[int]:
        """Get list of active slot IDs."""
        return list(self.active_sequences.values())

    def get_active_seq_ids(self) -> List[int]:
        """Get list of active sequence IDs."""
        return list(self.active_sequences.keys())

    def reset(self):
        """Reset all slots to free."""
        self.free_slots = set(range(self.num_slots))
        self.active_sequences.clear()


# =============================================================================
# Sequence State
# =============================================================================


@dataclass
class SequenceState:
    """Tracks the state of an active sequence."""

    seq_id: int
    slot_id: int
    prompt_tokens: List[int]
    generated_tokens: List[int]
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    stop_token_ids: Optional[List[int]] = None

    @property
    def current_len(self) -> int:
        """Total length of prompt + generated tokens."""
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    def is_complete(self) -> bool:
        """Check if generation should stop."""
        if self.num_generated >= self.max_new_tokens:
            return True
        if self.stop_token_ids and self.generated_tokens:
            if self.generated_tokens[-1] in self.stop_token_ids:
                return True
        return False


# =============================================================================
# Generation Engine
# =============================================================================


class GenerationEngine:
    """
    Continuous batching generation engine.

    Handles dynamic batching where sequences can enter and exit the batch
    at any time, maximizing GPU utilization.

    Usage:
        engine = GenerationEngine(model, max_batch=8, max_seq_len=1024)

        # Add requests
        seq_id_1 = engine.add_request(prompt_tokens_1, max_new_tokens=100)
        seq_id_2 = engine.add_request(prompt_tokens_2, max_new_tokens=50)

        # Generate (yields tokens as they're produced)
        for seq_id, token in engine.run():
            print(f"Sequence {seq_id}: token {token}")

        # Or step manually
        while engine.has_work():
            results = engine.step()
            for seq_id, token in results.items():
                process_token(seq_id, token)
    """

    def __init__(
        self,
        model: GPT,
        max_batch: int,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.model.eval()
        self.config = model.config
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len or self.config.max_seq_len
        self.device = device or next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        # Pre-allocate KV cache
        self.cache = KVCache(
            max_batch=max_batch,
            max_seq_len=self.max_seq_len,
            num_layers=self.config.num_layers,
            num_kv_heads=self.config.num_kv_heads,
            d_k=self.config.d_k,
            device=self.device,
            dtype=self.dtype,
        )

        # Slot management
        self.slots = SlotAllocator(max_batch)

        # Sequence tracking
        self.pending_requests: List[
            Tuple[int, List[int], int, float, int, float, Optional[List[int]]]
        ] = []
        self.active_sequences: Dict[int, SequenceState] = {}
        self.completed_sequences: Dict[int, SequenceState] = {}

        # ID counter
        self._next_seq_id = 0

    def add_request(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        stop_token_ids: Optional[List[int]] = None,
    ) -> int:
        """
        Add a generation request.

        Args:
            prompt_tokens: List of token IDs for the prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (0 to disable)
            top_p: Top-p nucleus filtering (0.0 to disable)
            stop_token_ids: Token IDs that stop generation

        Returns:
            seq_id: Unique identifier for this request
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        # Add to pending queue
        self.pending_requests.append(
            (
                seq_id,
                prompt_tokens,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                stop_token_ids,
            )
        )

        return seq_id

    def _schedule_pending(self):
        """Move pending requests to active if slots available."""
        while self.pending_requests and not self.slots.is_full():
            seq_id, prompt_tokens, max_new_tokens, temp, top_k, top_p, stop_ids = (
                self.pending_requests.pop(0)
            )

            # Allocate slot
            slot_id = self.slots.allocate(seq_id)
            if slot_id is None:
                # Put it back (shouldn't happen due to is_full check)
                self.pending_requests.insert(
                    0,
                    (
                        seq_id,
                        prompt_tokens,
                        max_new_tokens,
                        temp,
                        top_k,
                        top_p,
                        stop_ids,
                    ),
                )
                break

            # Create sequence state
            self.active_sequences[seq_id] = SequenceState(
                seq_id=seq_id,
                slot_id=slot_id,
                prompt_tokens=prompt_tokens,
                generated_tokens=[],
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                stop_token_ids=stop_ids,
            )

    def _prefill_new_sequences(self) -> Dict[int, int]:
        """
        Process prompts for newly scheduled sequences.

        Returns:
            Dict mapping seq_id -> first generated token
        """
        results = {}

        # Find sequences that need prefill (no generated tokens yet)
        to_prefill = [
            seq for seq in self.active_sequences.values() if seq.num_generated == 0
        ]

        if not to_prefill:
            return results

        # Process each prefill individually (they have different lengths)
        # In production, you'd want to batch prefills of similar length
        for seq in to_prefill:
            slot_idx = torch.tensor([seq.slot_id], device=self.device)
            prompt = torch.tensor([seq.prompt_tokens], device=self.device)

            # Forward pass
            cache_view = KVCacheView(self.cache, slot_idx)
            with torch.no_grad():
                logits = self.model(prompt, cache_view=cache_view)
            cache_view.finalize(len(seq.prompt_tokens))

            # Sample token
            next_token = self._sample_token(
                logits[0, -1, :], seq.temperature, seq.top_k, seq.top_p
            )

            seq.generated_tokens.append(next_token)
            results[seq.seq_id] = next_token

        return results

    def _decode_step(self) -> Dict[int, int]:
        """
        Run one decode step for all active sequences that have started generating.

        Returns:
            Dict mapping seq_id -> generated token
        """
        results = {}

        # Find sequences in decode phase (have generated at least one token)
        decoding = [
            seq
            for seq in self.active_sequences.values()
            if seq.num_generated > 0 and not seq.is_complete()
        ]

        if not decoding:
            return results

        # Batch decode - all sequences process one token
        batch_size = len(decoding)
        slot_indices = torch.tensor(
            [seq.slot_id for seq in decoding], device=self.device
        )

        # Get last token for each sequence
        last_tokens = torch.tensor(
            [[seq.generated_tokens[-1]] for seq in decoding], device=self.device
        )

        # Forward pass
        cache_view = KVCacheView(self.cache, slot_indices)
        with torch.no_grad():
            logits = self.model(last_tokens, cache_view=cache_view)
        cache_view.finalize(1)

        # Sample tokens for each sequence
        for i, seq in enumerate(decoding):
            next_token = self._sample_token(
                logits[i, -1, :], seq.temperature, seq.top_k, seq.top_p
            )
            seq.generated_tokens.append(next_token)
            results[seq.seq_id] = next_token

        return results

    def _sample_token(
        self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float
    ) -> int:
        """Sample a single token from logits."""
        logits = logits / max(temperature, 1e-8)

        if top_k > 0 or top_p > 0.0:
            logits = GPT.top_p_top_k(logits.unsqueeze(0), top_p, top_k).squeeze(0)

        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        return token

    def _retire_completed(self) -> List[int]:
        """
        Move completed sequences out of active set.

        Returns:
            List of completed seq_ids
        """
        completed = []

        for seq_id, seq in list(self.active_sequences.items()):
            if seq.is_complete():
                # Release slot
                self.slots.release(seq_id)
                self.cache.reset(torch.tensor([seq.slot_id], device=self.device))

                # Move to completed
                self.completed_sequences[seq_id] = seq
                del self.active_sequences[seq_id]
                completed.append(seq_id)

        return completed

    def step(self) -> Dict[int, Optional[int]]:
        """
        Run one generation step.

        Returns:
            Dict mapping seq_id -> token (None if sequence completed this step)
        """
        results = {}

        # 1. Schedule pending requests into free slots
        self._schedule_pending()

        # 2. Prefill new sequences
        prefill_results = self._prefill_new_sequences()
        results.update(prefill_results)

        # 3. Decode step for active sequences
        decode_results = self._decode_step()
        results.update(decode_results)

        # 4. Retire completed sequences
        completed = self._retire_completed()
        for seq_id in completed:
            if seq_id not in results:
                results[seq_id] = None

        return results

    def has_work(self) -> bool:
        """Check if there's more work to do."""
        return bool(self.pending_requests) or bool(self.active_sequences)

    def run(self) -> Generator[Tuple[int, int], None, None]:
        """
        Run generation to completion, yielding tokens as produced.

        Yields:
            (seq_id, token) tuples as tokens are generated
        """
        while self.has_work():
            results = self.step()
            for seq_id, token in results.items():
                if token is not None:
                    yield seq_id, token

    def get_sequence(self, seq_id: int) -> Optional[SequenceState]:
        """Get sequence state by ID."""
        if seq_id in self.active_sequences:
            return self.active_sequences[seq_id]
        return self.completed_sequences.get(seq_id)

    def get_generated_tokens(self, seq_id: int) -> Optional[List[int]]:
        """Get generated tokens for a sequence."""
        seq = self.get_sequence(seq_id)
        return seq.generated_tokens if seq else None

    def get_full_sequence(self, seq_id: int) -> Optional[List[int]]:
        """Get prompt + generated tokens for a sequence."""
        seq = self.get_sequence(seq_id)
        if seq is None:
            return None
        return seq.prompt_tokens + seq.generated_tokens

    def cancel(self, seq_id: int) -> bool:
        """
        Cancel an active or pending sequence.

        Returns:
            True if cancelled, False if not found
        """
        # Check pending
        for i, (sid, *_) in enumerate(self.pending_requests):
            if sid == seq_id:
                self.pending_requests.pop(i)
                return True

        # Check active
        if seq_id in self.active_sequences:
            seq = self.active_sequences[seq_id]
            self.slots.release(seq_id)
            self.cache.reset(torch.tensor([seq.slot_id], device=self.device))
            del self.active_sequences[seq_id]
            return True

        return False

    def reset(self):
        """Reset engine state for reuse."""
        self.slots.reset()
        self.cache.reset_all()
        self.pending_requests.clear()
        self.active_sequences.clear()
        self.completed_sequences.clear()

    def stats(self) -> Dict[str, int]:
        """Get engine statistics."""
        return {
            "pending": len(self.pending_requests),
            "active": len(self.active_sequences),
            "completed": len(self.completed_sequences),
            "slots_free": self.slots.num_free(),
            "slots_active": self.slots.num_active(),
        }


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
    attn_type: str = "standard",
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
        attn_type=attn_type,
    )
    return GPT(config)

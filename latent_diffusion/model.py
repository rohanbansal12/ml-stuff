"""
U-Net for Latent Diffusion.

This is a smaller U-Net designed for the VAE's latent space (e.g., 8x8x4).
Key differences from the pixel-space U-Net:
- Fewer downsampling levels (8x8 -> 4x4 -> 2x2 is enough)
- Input/output channels match VAE latent channels
- Generally smaller since latent space is more compact

The architecture is otherwise the same as regular DDPM U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from latent_diffusion.config import LatentUNetConfig


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """Time embedding: sinusoidal -> MLP."""

    def __init__(self, time_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


class SelfAttention2d(nn.Module):
    """Self-attention for 2D feature maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        def reshape(t):
            return t.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Use PyTorch's optimized attention (FlashAttention when available)
        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(2, 3).reshape(B, C, H, W)
        out = self.proj(out)

        return x + out


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class DownBlock(nn.Module):
    """Encoder block: ResBlocks -> optional attention -> downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_blocks: int = 2,
        groups: int = 8,
        dropout: float = 0.0,
        use_attn: bool = False,
        downsample: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        for i in range(num_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.blocks.append(ResBlock(ch_in, out_channels, time_dim, groups, dropout))
            self.attns.append(
                SelfAttention2d(out_channels) if use_attn else nn.Identity()
            )

        if downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, 3, stride=2, padding=1
            )
        else:
            self.downsample = None

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips = []

        for block, attn in zip(self.blocks, self.attns):
            x = block(x, t_emb)
            x = attn(x)
            skips.append(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return skips, x


class UpBlock(nn.Module):
    """Decoder block: upsample -> concat skip -> ResBlocks -> optional attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_dim: int,
        num_blocks: int = 2,
        groups: int = 8,
        dropout: float = 0.0,
        use_attn: bool = False,
        upsample: bool = True,
    ):
        super().__init__()

        if upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
            )
        else:
            self.upsample = None

        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        for i in range(num_blocks):
            if i == 0:
                ch_in = in_channels + skip_channels
            else:
                ch_in = out_channels
            self.blocks.append(ResBlock(ch_in, out_channels, time_dim, groups, dropout))
            self.attns.append(
                SelfAttention2d(out_channels) if use_attn else nn.Identity()
            )

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor], t_emb: torch.Tensor
    ) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)

        for i, (block, attn) in enumerate(zip(self.blocks, self.attns)):
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1) if i == 0 else x
            x = block(x, t_emb)
            x = attn(x)

        return x


class LatentUNet(nn.Module):
    """
    U-Net for latent diffusion.

    Designed for small spatial inputs (e.g., 8x8) from VAE encoder.
    """

    def __init__(self, config: LatentUNetConfig):
        super().__init__()
        self.config = config

        ch = config.channels
        time_dim = config.time_dim
        groups = config.groups
        mults = config.channel_mults
        n_res = config.num_res_blocks
        dropout = config.dropout

        # Time embedding
        self.time_emb = TimeEmbedding(time_dim)

        # Input projection
        self.input_conv = nn.Conv2d(config.in_channels, ch, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch_list = [ch]

        current_ch = ch
        current_res = 8  # Assuming 8x8 latent input

        for i, mult in enumerate(mults):
            out_ch = ch * mult
            use_attn = current_res in config.attn_resolutions
            is_last = i == len(mults) - 1

            self.down_blocks.append(
                DownBlock(
                    in_channels=current_ch,
                    out_channels=out_ch,
                    time_dim=time_dim,
                    num_blocks=n_res,
                    groups=groups,
                    dropout=dropout,
                    use_attn=use_attn,
                    downsample=not is_last,
                )
            )

            ch_list.extend([out_ch] * n_res)
            current_ch = out_ch
            if not is_last:
                current_res //= 2

        # Bottleneck
        self.mid_block1 = ResBlock(current_ch, current_ch, time_dim, groups, dropout)
        self.mid_attn = (
            SelfAttention2d(current_ch) if config.use_bottleneck_attn else nn.Identity()
        )
        self.mid_block2 = ResBlock(current_ch, current_ch, time_dim, groups, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(mults)):
            out_ch = ch * mult
            skip_ch = out_ch

            if i == 0:
                in_ch = current_ch
            else:
                prev_mult = list(reversed(mults))[i - 1]
                in_ch = ch * prev_mult

            use_attn = current_res in config.attn_resolutions
            is_last = i == len(mults) - 1
            final_out_ch = ch if is_last else out_ch

            self.up_blocks.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=final_out_ch,
                    skip_channels=skip_ch,
                    time_dim=time_dim,
                    num_blocks=n_res,
                    groups=groups,
                    dropout=dropout,
                    use_attn=use_attn,
                    upsample=not (i == 0),
                )
            )

            current_ch = final_out_ch
            if not (i == 0):
                current_res *= 2

        # Output projection
        self.output_norm = nn.GroupNorm(groups, ch)
        self.output_conv = nn.Conv2d(ch, config.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_emb(t)

        # Input
        x = self.input_conv(x)

        # Encoder
        all_skips = []
        for block in self.down_blocks:
            skips, x = block(x, t_emb)
            all_skips.extend(skips)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Decoder
        skip_idx = len(all_skips)
        for block in self.up_blocks:
            n_skips = len(block.blocks)
            block_skips = all_skips[skip_idx - n_skips : skip_idx]
            skip_idx -= n_skips
            x = block(x, block_skips, t_emb)

        # Output
        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_conv(x)

        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

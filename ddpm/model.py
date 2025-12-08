import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim)
        )

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -torch.arange(half, device=device) * (math.log(10000) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        out = self.mlp(emb)
        return out
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

        self.time_proj = nn.Linear(time_dim, out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) if in_channels != out_channels else None

    def forward(self, x, time_emb):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = out + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.act(out + x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, num_blocks=1, groups=8):
        super().__init__()
        blocks = [ResBlock(in_channels, out_channels, time_dim, groups=groups)]
        for _ in range(1, num_blocks):
            blocks.append(ResBlock(out_channels, out_channels, time_dim, groups=groups))
        self.blocks = nn.ModuleList(blocks)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        h_down = self.downsample(x)
        return x, h_down
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, time_dim, num_blocks=1, groups=8):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        blocks = [ResBlock(out_channels + skip_channels, out_channels, time_dim, groups=groups)]
        for _ in range(1, num_blocks):
            blocks.append(ResBlock(out_channels, out_channels, time_dim, groups=groups))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat((x, skip), dim=1)
        for block in self.blocks:
            x = block(x, t_emb)
        return x


class UNet(nn.Module):
    def __init__(self, ch, time_dim, groups=8):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.stem = ResBlock(3, ch, time_dim, groups=groups)
        self.down_blocks = nn.ModuleList([DownBlock(ch, 2*ch, time_dim, groups=groups), DownBlock(2*ch, 4*ch, time_dim, groups=groups)])
        self.bottleneck = ResBlock(4*ch, 4*ch, time_dim, groups=groups)
        self.up_blocks = nn.ModuleList([UpBlock(4*ch, 2*ch, 4*ch, time_dim, groups=groups), UpBlock(2*ch, ch, 2*ch, time_dim, groups=groups)])
        self.proj = nn.Conv2d(ch, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x = self.stem(x, t_emb)

        skips = []
        for block in self.down_blocks:
            skip, x = block(x, t_emb)
            skips.append(skip)

        h = self.bottleneck(x, t_emb)

        for i, block in enumerate(self.up_blocks):
            h = block(h, skips[-(i+1)], t_emb)

        return self.proj(h)
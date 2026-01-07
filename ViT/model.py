import sys

import torch
import torch.nn as nn

sys.path.append(".")
from GPT.model import Block


class ViT(nn.Module):
    def __init__(self, d_model, patch_size, num_layers, num_heads, num_classes, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_patches = (32 // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros((1, 1, d_model)), requires_grad=True)
        self.pos_embed = nn.Parameter(
            torch.zeros((1, 1 + self.n_patches, d_model)), requires_grad=True
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        layers = []
        for _ in range(num_layers):
            layers.append(
                Block(
                    d_model,
                    num_heads,
                    1 + self.n_patches,
                    dropout=dropout,
                    rope=False,
                    rmsnorm=False,
                    causal=False,
                )
            )
        self.dec = nn.Sequential(*layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        b = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(-1, -2)
        x = torch.cat([self.cls_token.repeat(b, 1, 1), x], dim=1)
        x = x + self.pos_embed
        att = self.dec(x)[:, 0]
        out = self.head(self.ln_f(att))
        return out

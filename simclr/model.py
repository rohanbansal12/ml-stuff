import torch.nn as nn
import torch.nn.functional as F


class SimCLRModel(nn.Module):
    def __init__(self, encoder, feat_dim, proj_dim=128, hidden_dim=2048):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.config = dict(feat_dim=feat_dim, proj_dim=proj_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        h = self.encoder(x, feat_vec=True)
        z = self.projector(h)
        z = F.normalize(z, dim=-1)
        return h, z

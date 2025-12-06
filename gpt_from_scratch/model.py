import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, dropout=.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_0 = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones((seq_len, seq_len))).view(1, 1, seq_len, seq_len)
        causal_mask = torch.zeros((1, 1, seq_len, seq_len))
        self.register_buffer("causal_mask", causal_mask.masked_fill_(mask == 0, float("-inf")))

        self.drop = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor): # input b, s, d_model -> output b, s, d_model
        b, s, d_mod = x.size()
        d_k = d_mod // self.num_heads

        # b, s, d_model -> b, s, d_model -> b, s, h, d_k -> b, h, s, d_k
        Q = self.W_Q(x).view(b, s, self.num_heads, d_k).transpose(1, 2)
        K = self.W_K(x).view(b, s, self.num_heads, d_k).transpose(1, 2)
        V = self.W_V(x).view(b, s, self.num_heads, d_k).transpose(1, 2)

        # b, h, s, d_k -> b, h, s, s
        scores = (Q @ K.transpose(-1, -2)) * (1.0 / math.sqrt(d_k))
        scores = scores + self.causal_mask[:, :, :s, :s]
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # b, h, s, s -> b, h, s, d_k -> b, s, d_model
        att = scores @ V
        out = self.W_0(att.transpose(1, 2).view(b, s, d_mod))
        return self.drop(out)
    
class FFN(nn.Module):
    def __init__(self, d_model, dropout=.1):
        super().__init__()

        self.lin1 = nn.Linear(d_model, d_model*4)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(d_model*4, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, dropout=.1):
        super().__init__()

        self.att = MultiHeadAttention(d_model, num_heads, seq_len, dropout)
        self.fc = FFN(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.fc(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, num_layers, vocab_size, dropout=.1):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        layers = []
        for _ in range(num_layers):
            layers.append(Block(d_model, num_heads, max_seq_len, dropout=dropout))
        self.dec = nn.Sequential(*layers)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0).expand(b, t)
        embs = self.token_emb(x) + self.pos_emb(pos)
        out = self.dec(embs)
        logits = self.lm_head(out)
        return logits
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, dropout=.1, rope=False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_len = seq_len
        self.rope = rope
     
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_0 = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones((seq_len, seq_len))).view(1, 1, seq_len, seq_len)
        causal_mask = torch.zeros((1, 1, seq_len, seq_len))
        self.register_buffer("causal_mask", causal_mask.masked_fill_(mask == 0, float("-inf")))
        self.drop = nn.Dropout(dropout)

        if rope:
            assert self.d_k % 2 == 0, "RoPE requires head dimension (d_k) to be even."
            cos, sin = self.build_rope_cache()
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

    def build_rope_cache(self):
        half_dim = self.d_k // 2
        device = next(self.parameters()).device

        freq_seq = torch.arange(half_dim, device=device)
        inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

        positions = torch.arange(self.seq_len, device=device).unsqueeze(1)
        angles = positions * inv_freq.unsqueeze(0)
    
        cos = angles.cos()
        sin = angles.sin()
        
        return cos, sin
    
    @staticmethod
    def apply_rope(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    def forward(self, x : torch.Tensor): # input b, s, d_model -> output b, s, d_model
        b, s, d_mod = x.size()

        # b, s, d_model -> b, s, d_model -> b, s, h, d_k -> b, h, s, d_k
        Q = self.W_Q(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope:
            sin = self.sin[:s].unsqueeze(0).unsqueeze(0)
            cos = self.cos[:s].unsqueeze(0).unsqueeze(0)
            Q = self.apply_rope(Q, cos, sin)
            K = self.apply_rope(K, cos, sin)

        # b, h, s, d_k -> b, h, s, s
        scores = (Q @ K.transpose(-1, -2)) * (1.0 / math.sqrt(self.d_k))
        scores = scores + self.causal_mask[:, :, :s, :s]
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # b, h, s, s -> b, h, s, d_k -> b, s, d_model
        att = scores @ V
        out = self.W_0(att.transpose(1, 2).reshape(b, s, d_mod))
        return self.drop(out)
    
class FFN(nn.Module):
    def __init__(self, d_model, dropout=.1):
        super().__init__()

        self.lin1 = nn.Linear(d_model, d_model*4)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(d_model*4, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # b, s, d_model -> b, s, d_model * 4 -> b, s, d_model
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, dropout=.1, rope=False):
        super().__init__()

        self.att = MultiHeadAttention(d_model, num_heads, seq_len, dropout, rope)
        self.fc = FFN(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.fc(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, num_layers, vocab_size, dropout=.1, rope=False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        layers = []
        for _ in range(num_layers):
            layers.append(Block(d_model, num_heads, max_seq_len, dropout=dropout, rope=rope))
        self.dec = nn.Sequential(*layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0).expand(b, t)
        embs = self.token_emb(x) + self.pos_emb(pos)
        out = self.dec(embs)
        out = self.ln_f(out)
        logits = self.lm_head(out)
        return logits
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens, sample=False, temperature=1.0):
        temperature = max(temperature, 1e-4)
        self.eval()
        for _ in range(max_new_tokens):
            x_mod = x if x.size(1) <= self.max_seq_len else x[:, -self.max_seq_len:]
            logits = self(x_mod)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            if sample:
                toks = torch.multinomial(probs, num_samples=1)
            else:
                toks = torch.topk(probs, k=1, dim=-1).indices
            x = torch.cat((x, toks), dim=1)
        return x
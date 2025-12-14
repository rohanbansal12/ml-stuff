# GPT From Scratch on Shakespeare

A minimal, educational re-implementation of a GPT-style language model trained on the Shakespeare corpus.  
This project includes:  
- a custom transformer block with multi-head self-attention  
- optional pre-LN / post-LN configurations  
- custom LayerNorm and RMSNorm implementations  
- rotary (RoPE) vs absolute positional embeddings  
- a fully from-scratch Byte Pair Encoding (BPE) tokenizer  
- ablations exploring normalization, architecture choices, and embedding strategies  

---

## 1. Overview

This project implements a small GPT-style transformer trained on the Shakespeare dataset (character-level or BPE-level depending on configuration).  
The goal is to deeply understand transformer internals by re-implementing every major component:

- Multi-Head Self-Attention  
- Feedforward networks  
- Position embeddings (absolute + RoPE)  
- Custom normalization (LayerNorm, RMSNorm)  
- Autoregressive text generation  
- Training loop, batching, and sampling strategies  
- Byte-level BPE tokenizer built completely from scratch  

The code emphasizes **clarity over efficiency** — ideal for learning. Karpathy's minGPT repository was a helpful reference just to get a feel for overall structure, but all components here are implemented from first principles without copying any code.

---

## 2. Tokenization: Custom BPE From Scratch

This project includes a full GPT-2 style byte-level BPE tokenizer implementation:

### Features
- Byte → Unicode reversible mapping  
- Learned merge rules via BPE training  
- Deterministic encoding/decoding  
- Greedy merge application with ranking  
- Serialization of vocab + merges  

### Training Procedure
1. Convert raw text → byte sequence → GPT-style Unicode alphabet  
2. Count adjacent token pair frequencies  
3. Iteratively merge the most frequent pair  
4. Build a vocabulary of subword tokens  
5. Use this tokenizer for GPT training  

## 3. Model Architecture

A small decoder-only transformer following the GPT design:

### Components
- Token + positional embeddings  
- Multi-head causal self-attention  
- Feedforward MLP (2× expansion)  
- Residual connections  
- Normalization (configurable)  
- Final linear layer for logits  

### Configurable Hyperparameters
| Parameter | Default |
|-----------|----------|
| d_model | 128–256 |
| num_layers | 4–8 |
| num_heads | 4 |
| dropout | 0.1 |
| seq_len | 256 |
| vocab_size | from BPE tokenizer |

---

## Mathematical Background

### Autoregressive Language Modeling Objective

Given a sequence of tokens $x = (x_1, x_2, \dots, x_T)$, the joint probability of the sequence is factorized autoregressively:

$p(x_1, \dots, x_T) = \prod_{t=1}^T p(x_t \mid x_1, \dots, x_{t-1})$

We model each conditional distribution using a neural network parameterized by $\theta$.

Training proceeds via **maximum likelihood estimation (MLE)**, which is equivalent to minimizing the negative log-likelihood (cross-entropy loss):

$L(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{t=1}^T \log p_\theta(x_t \mid x_{<t}) \right]$

In practice, this expectation is approximated with minibatches of token sequences.

---

### Token and Positional Embeddings

Each token $x_t$ is mapped to a vector using a learned embedding matrix $E \in \mathbb{R}^{|V| \times d}$:

$e_t = E[x_t]$

Since self-attention is permutation-invariant, positional information is injected using learned or fixed positional embeddings $P_t$:

$h_t^{(0)} = e_t + P_t$

This forms the input to the first Transformer block.

---

### Self-Attention Mechanism

For each Transformer layer, we compute **queries, keys, and values**:

$Q = HW_Q,\quad K = HW_K,\quad V = HW_V$

where $H \in \mathbb{R}^{T \times d}$ is the sequence of hidden states.

The attention weights are computed via scaled dot-product attention:

$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$

Here:
- $d_k$ is the key dimension
- $M$ is a **causal mask** that prevents attending to future tokens

This enforces the autoregressive property:
$p(x_t \mid x_{<t})$ only depends on earlier tokens.

---

### Multi-Head Attention

Instead of a single attention operation, we use $h$ parallel attention heads:

$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$

The outputs are concatenated and projected:

$\text{MHA}(H) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$

Multi-head attention allows the model to attend to different representation subspaces simultaneously.

---

### Feedforward Network

Each Transformer block contains a position-wise feedforward network:

$\text{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2$

where $\sigma$ is typically ReLU or GELU. This introduces nonlinearity and increases representational capacity.

---

### Residual Connections and Normalization

Each sub-layer (attention and FFN) is wrapped with a residual connection and normalization:

$\text{LayerNorm}(x + \text{Sublayer}(x))$

Residual connections improve gradient flow, while normalization stabilizes training.

Modern GPT variants typically use **pre-normalization**, applying LayerNorm before each sub-layer.

---

### Output Projection and Softmax

The final hidden states $H^{(L)}$ are projected to vocabulary logits:

$z_t = H_t^{(L)} W_{vocab}$

The conditional probability over the next token is:

$p(x_{t+1} \mid x_{\le t}) = \text{softmax}(z_t)$

The training loss is the cross-entropy between this distribution and the true next token.

---


## 4. Attention Mechanism

Implementation of scaled dot-product attention:

scores = $(QK^\top) / \sqrt{d_k}$

scores += causal_mask

weights = softmax(scores)

output = weights @ V

### Rotary Positional Embeddings (RoPE)
Optional RoPE rotates Q/K in embedding space to encode position information multiplicatively.  
Ablation compares:

- **Absolute learned embeddings**
- **Sinusoidal embeddings**
- **RoPE applied to Q/K only**

---

## 5. Normalization: Custom Implementations

### LayerNorm (from scratch)
- Computes mean + variance across features  
- Includes learnable weight and bias  
- Implemented manually (no PyTorch LayerNorm)

### RMSNorm (residual only)
- No mean subtraction  
- Normalizes by RMS  
- Potentially more stable at scale  
- Optional scaling parameter  

### Pre-LN vs Post-LN Ablation
- **Pre-LN** improves stability for deep transformers  
- **Post-LN** matches original GPT-2 layout  

---

## 6. Training Setup

| Setting | Value |
|--------|--------|
| Dataset | Shakespeare (tiny_shakespeare.txt) |
| Context length | 256 |
| Batch size | 32–64 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Warmup | optional |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Mixed precision | optional |

---

## 7. Text Generation

Sampling loop uses:
- temperature  
- top-k filtering (optional)  
- iterative decoding  

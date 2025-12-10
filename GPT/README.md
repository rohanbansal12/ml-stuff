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

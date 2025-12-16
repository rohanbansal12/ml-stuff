# Quantization Experiments

This directory contains **from-scratch implementations and experiments** exploring post-training quantization (PTQ) for Transformer models, with a focus on *mechanistic understanding* rather than relying on black-box libraries.

The work is structured in two layers:

1. **`quantization.py`** — a self-contained, reusable toolbox for quantization primitives  
2. **Model-level experiments** — applying those primitives to a real pretrained Transformer (OPT-1.3B) and measuring loss / perplexity impact

The goal is to understand *why* naive PTQ fails, *where* error is introduced, and *what techniques are required* to make low-bit inference viable.

---

## 1. `quantization.py` — Core Quantization Utilities

This file implements the fundamental building blocks needed for PTQ experiments without relying on framework magic.

### 1.1 Quantization parameter computation

Helpers for computing scales and zero-points under different schemes:

- **Symmetric quantization**
  - Used for weights and (optionally) activations
  - Zero-point fixed at 0
- **Affine (asymmetric) quantization**
  - Used when activation distributions are skewed
  - Zero-point chosen to align real zero

Key helpers:
- `calc_qparams_symmetric(...)`
- `activation_qparams_from_range(...)`

Supports:
- Per-tensor vs per-channel scaling  
- Arbitrary bit-widths (e.g. 8-bit, 4-bit)  
- Explicit control over quantization ranges  

---

### 1.2 Quantize / dequantize operators

Low-level routines for converting between floating-point and integer representations:

- `quantize_symmetric(x, scale, qmax, dtype)`
- `quantize_affine(x, scale, zero_point, qmin, qmax, dtype)`

These functions explicitly return:
- The quantized tensor
- Clipping statistics (fraction of values clipped)

This makes **quantization error observable and measurable**, which is critical for debugging PTQ failures.

---

### 1.3 Fake-integer GEMM pattern

All experiments use the canonical PTQ computation pattern:

x_fp → quantize → int8 \
w_fp → quantize → int8 \
int32 accumulation \
rescale to fp \
(optional) add bias 

Even when run on GPU, this structure mirrors what real integer kernels do and lets us reason about:
- accumulation precision  
- scale interaction  
- where numerical error enters the pipeline  

---

## 2. Model-Level Experiments (OPT-1.3B)

We apply these primitives to a **pretrained `facebook/opt-1.3b` decoder-only Transformer** and evaluate degradation relative to the floating-point baseline.

### 2.1 Evaluation setup

- Model: **OPT-1.3B**
- Dataset: held-out slice of text
- Metrics:
  - Token-level cross-entropy loss
  - Perplexity
- Evaluation details:
  - Fixed evaluation batches (materialized once)
  - Fixed RNG seeds
  - FP16 baseline and FP32-GEMM sanity checks

This ensures observed deltas are real and not artifacts of randomness or mixed-precision behavior.

---

### 2.2 Baseline sanity checks

| Model | Loss | Perplexity |
|------|------|------------|
| FP16 baseline | 3.323711 | 27.763178 |
| FP32 GEMM | 3.323700 | 27.762894 |

**Conclusion:** FP16 vs FP32 GEMM differences are negligible (≈1e−5 loss).  
Any later changes are attributable to quantization itself.

---

## 3. Quantization Experiments & Results

### 3.1 Weight-only W8 (per-channel symmetric)

**Method**
- Quantize weights to int8
- Keep activations in floating point
- Use fake-integer GEMM

**Result**

FP:    Loss = 3.323711 \
W8:    Loss = 3.322239 \
Δloss = −1.47e−3

**Observation**
- Slight *improvement* in loss
- Interpreted as mild regularization / weight smoothing
- Expected and not concerning for small deltas

---

### 3.2 Dynamic W8A8 (per-tensor activation quantization)

**Method**
- Weights: int8, per-channel
- Activations: dynamically quantized per batch

**Result**

FP:    Loss = 3.323711 \
W8A8:  Loss = 3.422797 \
Δloss = +9.91e−2

**Observation**
- Mild but consistent degradation
- Indicates activation quantization is the dominant source of error

---

### 3.3 Static W8A8 (calibrated activation ranges)

**Method**
- Calibrate activation ranges using percentile statistics
- Freeze activation scales
- Quantize both weights and activations

**Result**

FP:    Loss = 3.323711 \
W8A8:  Loss = 6.391982 \
Δloss = +3.07

**Observation**
- Severe degradation despite very low clipping rates
- Demonstrates that:
  - Low clipping ≠ low quantization error
  - Scale mismatch dominates error, not saturation

This reproduces the **classic PTQ failure mode** motivating methods like SmoothQuant, AWQ, and GPTQ.

---
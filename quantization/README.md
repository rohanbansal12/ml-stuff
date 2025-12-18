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

## Grouped Weight-Only W4 Quantization (Naive PTQ)

After establishing strong baselines for W8A8 and observing the failure modes of naive static activation quantization, we explored **grouped weight-only W4 quantization** as a lower-bit alternative that preserves accuracy while aggressively reducing weight precision.

This section summarizes the **theory-driven experiments** and **empirical findings** from grouped W4 on OPT-1.3B.

---

### Motivation

Naive per-output-channel W4 quantization is often too coarse:
- A single large outlier in a row of the weight matrix forces a large scale.
- This wastes representational capacity for most weights in that row.

**Grouped W4** improves this by:
- Still operating *per output channel* (row of `W`)
- Further splitting the **input dimension** into small contiguous groups
- Assigning **one scale per (output channel × group)**

This reduces the impact of localized outliers without introducing activation quantization.

---

Activations remain FP16/FP32.

---

### Experiment 0 — Single-Layer Reconstruction Error

We first validated that grouped W4 behaves as expected at the **single-layer level**.

For a representative MLP layer (`fc2`):

| Group Size | Weight Cosine | Output Cosine | Rel Mean Error |
|----------:|--------------:|--------------:|---------------:|
| 32  | ~0.994 | ~0.997 | ~6–7% |
| 64  | ~0.992 | ~0.995 | ~8% |
| 128 | ~0.991 | ~0.993 | ~9–10% |
| 256 | ~0.989 | ~0.991 | ~11–12% |

**Key takeaway:**  
Smaller groups consistently reduce both weight and activation reconstruction error.

---

### Experiment 1 — Full-Model MLP W4 Quantization

We next applied grouped W4 to **all MLP layers** (`fc1` + `fc2`) in OPT-1.3B.

Baseline: FP Loss: 3.3237  | Perplexity: 27.76

Results:

| Group Size | Loss | Perplexity |
|-----------:|-----:|-----------:|
| 32  | 3.3671 | 28.99 |
| 64  | 3.3931 | 29.76 |
| 128 | 3.4048 | 30.11 |
| 256 | 3.4294 | 30.86 |

**Observation:**  
Grouped W4 degrades gracefully, with group size 32 performing best.

---

### Experiment 2 — Isolating MLP Sensitivity (fc1 vs fc2)

To understand where error originates, we quantized layers selectively.

#### Only `fc1` quantized (W4)
| Group Size | Loss |
|-----------:|-----:|
| 32  | 3.3660 |
| 64  | 3.3794 |
| 128 | 3.3825 |
| 256 | 3.4009 |

#### Only `fc2` quantized (W4)
| Group Size | Loss |
|-----------:|-----:|
| 32  | 3.3239 |
| 64  | 3.3307 |
| 128 | 3.3340 |
| 256 | 3.3369 |

**Key insight:**
- `fc1` (expansion layer) is **far more sensitive** to weight quantization
- `fc2` is surprisingly robust even at W4
- This aligns with theory: `fc1` defines the hidden basis used downstream

---

### Summary of Findings

1. **Grouped W4 works significantly better than naive per-channel W4**
2. **Smaller group sizes (≈32–64) are strongly preferred**
3. **MLP `fc1` dominates sensitivity**, not `fc2`
4. Weight-only quantization can *sometimes slightly improve loss* due to implicit regularization
5. These results validate:
   - Why modern methods (AWQ, GPTQ) focus on MLP-first strategies
   - Why attention often requires different treatment than MLPs

---

### Implications for Stage 3 (SmoothQuant)

Grouped W4 experiments reveal:
- Activation outliers are not the only problem
- Sensitivity is highly **layer- and role-dependent**
- `fc1` and attention projections are prime targets for activation smoothing

These observations directly motivate **SmoothQuant**, which aims to:
- Reduce activation outliers **before** W8A8
- Preserve per-layer signal structure without going to W4

The next stage builds on these findings.

## Stage 3 — SmoothQuant (W8A8)

**Goal:** Test whether SmoothQuant-style reparameterization can make W8A8 viable on OPT-1.3B.

### Observed Results
- **Naive static W8A8**: catastrophic loss blowup due to activation outliers.
- **Dynamic W8A8**: large improvement over static, but still worse than FP.
- **SmoothQuant (dynamic activations)**:
  - Works only in a narrow regime.
  - Best results at **α ≈ 0.75** with **`smin = 1.0`**.
  - Without `smin`, weight scales explode and cause severe loss blowups.
- **MLP-only SmoothQuant** performs noticeably better than full-model replacement.
- **Attention (Q/K/V) layers remain the most sensitive**, dominating error.
- **Symmetric percentile activation clipping (ap999)** caused major degradation and was harmful here.

**Conclusion:** SmoothQuant improves W8A8 over naive approaches but is fragile and insufficient alone; weight-only methods (AWQ/GPTQ) are the natural next step.

## Stage 4 — GPTQ (Weight-only, W4 grouped)

We implemented a from-scratch GPTQ pipeline with Hessian-based error compensation and lazy block updates.

**Results (OPT-1.3B, group size = 32):**
- **GPTQ fc1 only (24 layers):** Loss **3.3377** | Perplexity **28.15**
- **GPTQ fc2 only (24 layers):** Loss **3.3265** | Perplexity **27.84**

GPTQ consistently improves over naive W4 grouped quantization and recovers most of the fp16 performance when applied to MLP layers.

## Stage 5 - AWQ (Activation-Aware Weight Quantization)

We implemented AWQ using activation-based channel saliency and per-channel rescaling, followed by W4 grouped weight quantization (group size = 32).

### Baselines (OPT-1.3B)

- **FP:** Loss 3.3237 | Perp 27.76  
- **FP32 GEMM:** Loss 3.3237 | Perp 27.76  

### fc1 only

- **Naive W4:** Loss 3.3662 | Perp 28.97  
- **AWQ4:** Loss 3.3365 | Perp 28.12  

### MLP (fc1 + fc2)

- **Naive W4:** Loss 3.3669 | Perp 28.99  
- **AWQ4:** Loss 3.3383 | Perp 28.17  

### Summary

AWQ consistently outperforms naive W4 weight-only quantization, recovering ~0.03 loss for both fc1-only and full MLP quantization, and bringing W4 performance much closer to FP.
# Mixture-of-Experts (MoE) Routing Analysis

An empirical study of expert routing behavior in pretrained Mixture-of-Experts language models. This project analyzes how MoE models distribute computation across experts during inference and how routing patterns respond to interventions.

**Model**: Qwen/Qwen1.5-MoE-A2.7B (60 experts, top-2 routing)

---

## Overview

Mixture-of-Experts models route each token to a subset of expert networks, enabling sparse computation. This project explores:

1. **Routing diversity**: How uniformly do experts get selected?
2. **Phase differences**: Does routing differ between prefill (prompt processing) and decode (generation)?
3. **Prompt sensitivity**: Do different prompt types exhibit distinct routing patterns?
4. **Robustness**: What happens when top experts are ablated?
5. **Temperature effects**: How does scaling router logits affect expert selection?

The codebase provides tools for:
- Capturing routing logits via forward hooks
- Computing routing statistics (load, entropy, effective experts)
- Visualizing expert selection patterns
- Performing controlled interventions (ablation, temperature scaling)

---

## Project Structure

```
MoE/
├── visualization.py          # Main experiment: baseline + ablation analysis
├── alpha_sweep.py           # Temperature scaling experiment
├── routing_utils.py         # Core routing capture and manipulation utilities
├── viz_utils.py             # Plotting and metric computation functions
├── plots/
│   ├── baseline/            # Prefill vs decode routing heatmaps
│   ├── ablate_prefill/      # Expert ablation during prefill
│   ├── ablate_decode/       # Expert ablation during decode
│   └── alpha_sweep/         # Temperature scaling results
```

---

## Experiment 1: Baseline Routing Analysis (`visualization.py`)

### Goal
Understand how expert routing behaves during normal inference across different prompts and generation phases.

### Method
For each prompt type (code, math, creative writing, chat, gibberish):

1. **Prefill phase**: Run forward pass on the prompt, capture routing logits
2. **Full generation**: Generate tokens, capture routing logits for entire sequence
3. **Decode-only stats**: Subtract prefill from generation to isolate decode behavior

### Metrics Computed

For each layer and phase, we compute:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Load** | Fraction of tokens routed to each expert | Expert utilization |
| **Top-1 share** | Max load across experts | Concentration metric |
| **HHI** | Herfindahl-Hirschman Index: $\sum_e \text{load}_e^2$ | Load imbalance |
| **Entropy** | $-\sum_e p_e \log p_e$ where $p_e$ is load | Routing diversity |
| **Effective experts** | $\exp(\text{entropy})$ | "Number" of experts meaningfully used |
| **Mean $p_{\max}$** | Average max routing probability | Router confidence |
| **Mean margin** | Average gap between top-2 routing probs | Selection certainty |

### Visualizations

**Heatmaps**: Expert load per layer (rows = layers, columns = experts)
- Shows which experts dominate at different depths
- Compares prefill vs decode patterns

**Curves**: Effective experts and router confidence vs layer depth
- Reveals routing collapse or diversity trends
- Highlights phase-specific behavior

**Cross-prompt comparison**: Effective experts across all prompts
- Identifies prompt-specific routing signatures

### Key Observations

1. **Routing collapse in decode**: Decode phase exhibits lower effective experts (higher concentration) than prefill
2. **Layer-specific patterns**: Early layers show more diverse routing; deeper layers become more specialized
3. **Prompt sensitivity**: Technical prompts (code, math) trigger different routing than creative prompts
4. **Stability metrics**: High $p_{\max}$ and margin indicate confident, stable routing decisions

---

## Experiment 2: Expert Ablation (`visualization.py --mode ablate-{prefill,decode}`)

### Goal
Measure routing robustness by dynamically masking the most-used expert per layer.

### Method

**Ablate-Prefill Mode**:
1. Run baseline prefill, identify top expert per layer
2. Re-run prefill with those experts masked (logits set to $-\infty$)
3. Visualize load reallocation

**Ablate-Decode Mode**:
1. Run baseline decode, identify top experts
2. Re-run full generation with decode top experts masked
3. Compute decode-only stats (subtract ablated prefill from ablated generation)
4. Visualize decode reallocation

### Ablation Dashboard

Three-panel visualization:
1. **Baseline load**: Original expert distribution
2. **Ablated load**: Distribution after masking top expert
3. **Delta (reallocation)**: Ablated - Baseline

The delta heatmap shows:
- **Negative values** (blue): Experts that lost load (includes masked expert)
- **Positive values** (red): Experts that absorbed the redistributed load

### Key Observations

1. **Graceful degradation**: Masking top experts does not collapse routing; load redistributes to other experts
2. **Second-choice routing**: The expert with second-highest baseline load typically absorbs most redistributed tokens
3. **Layer dependence**: Deeper layers exhibit more concentrated reallocation patterns
4. **Phase differences**: Prefill routing is more robust (smoother redistribution) than decode routing

---

## Experiment 3: Temperature Scaling (`alpha_sweep.py`)

### Goal
Understand how scaling router logits (pre-softmax) affects routing diversity and model behavior.

### Method

We scale router logits by $\alpha$ before softmax:

$$p(e \mid h) = \text{softmax}(\alpha \cdot [W_e h]_{e=1}^E)$$

where:
- $\alpha > 1$: **Sharpens** routing (more concentrated, favors top experts)
- $\alpha = 1$: Baseline behavior
- $\alpha < 1$: **Smooths** routing (more uniform expert selection)

The experiment sweeps $\alpha \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\}$ and measures:
- Effective experts per layer
- Top-1 share (routing concentration)
- Entropy of load distribution

### Implementation Details

Temperature scaling is applied via input scaling to the router:

```python
with scale_router_logits_via_weight(model, alpha=2.0):
    output = model.generate(...)  # Routing is scaled by alpha
```

This patches the router's forward method to scale hidden states: `logits = W @ (alpha * h)`, which is mathematically equivalent to `softmax(alpha * logits)`.

### Visualizations

**Per-prompt curves**: Effective experts, top-1 share, and entropy vs layer depth for each $\alpha$
- Shows how temperature affects routing diversity at different depths

**Scalar summary**: Mean effective experts vs $\alpha$ across all prompts
- Reveals routing collapse trends as $\alpha$ increases

### Key Observations

1. **Routing collapse with high $\alpha$**: Increasing temperature reduces effective experts (routing becomes more concentrated)
2. **Smooth degradation**: The collapse is gradual, not a sharp phase transition
3. **Layer-specific sensitivity**: Deeper layers are more sensitive to temperature changes
4. **Prompt dependence**: Technical prompts maintain higher diversity at higher temperatures than generic chat
5. **Performance tradeoff**: While higher $\alpha$ concentrates routing, it doesn't necessarily improve generation quality (not measured here, but hypothesis)

---

## Mathematical Background

### Routing Mechanism

In top-K MoE, each token $h$ is routed to $K$ experts (typically $K=2$) based on router logits:

$$\text{logits}_e = W_e h$$

The routing probabilities are:

$$p_e = \text{softmax}(\text{logits})_e = \frac{\exp(W_e h)}{\sum_{e'} \exp(W_{e'} h)}$$

Top-K experts are selected, and their outputs are combined via weighted sum:

$$\text{output} = \sum_{e \in \text{top-K}} p_e \cdot \text{expert}_e(h)$$

### Load Balancing

Load imbalance can cause:
- **Routing collapse**: Most tokens route to a few experts (others underutilized)
- **Capacity issues**: Popular experts become bottlenecks in distributed training

The **effective experts** metric quantifies diversity:

$$N_{\text{eff}} = \exp\left(-\sum_{e=1}^E p_e \log p_e\right)$$

where $p_e$ is the fraction of tokens routed to expert $e$.

Interpretation:
- $N_{\text{eff}} = E$: Perfectly uniform routing
- $N_{\text{eff}} = 1$: All tokens route to one expert (full collapse)

### Temperature Scaling

Scaling logits by $\alpha$ before softmax is equivalent to controlling the "temperature" of the routing distribution:

$$p_e(\alpha) = \frac{\exp(\alpha \cdot W_e h)}{\sum_{e'} \exp(\alpha \cdot W_{e'} h)}$$

Properties:
- As $\alpha \to \infty$: Distribution becomes one-hot (deterministic routing to argmax)
- As $\alpha \to 0$: Distribution becomes uniform ($p_e = 1/E$ for all $e$)

This is the inverse of the standard temperature parameter in softmax ($T = 1/\alpha$).

---

## Usage

### Baseline + Ablation Experiments

```bash
# Baseline routing analysis (prefill vs decode)
python visualization.py --mode base --out_dir plots/baseline

# Ablate top prefill expert per layer
python visualization.py --mode ablate-prefill --out_dir plots/ablate_prefill

# Ablate top decode expert per layer
python visualization.py --mode ablate-decode --out_dir plots/ablate_decode

# With fine-tuned adapter
python visualization.py --mode base --adapter_path checkpoints/adapter-epoch-5
```

### Temperature Scaling Sweep

```bash
# Sweep alpha values and visualize routing collapse
python alpha_sweep.py --out_dir plots/alpha_sweep --alphas 0.5 0.75 1.0 1.5 2.0 3.0 5.0
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Experiment mode: `base`, `ablate-prefill`, `ablate-decode` | `base` |
| `--out_dir` | Output directory for plots | `plots/baseline` |
| `--adapter_path` | Optional PEFT adapter checkpoint | `None` |
| `--max_new_tokens` | Tokens to generate in decode phase | `50` |
| `--n_experts` | Number of experts in the model | `60` |
| `--alphas` | Temperature scaling values to sweep | (alpha_sweep.py only) |

---

## Code Organization

### `routing_utils.py`

Core utilities for routing analysis:

- **`attach_router_hooks()`**: Register forward hooks to capture routing logits
- **`get_router_hook()`**: Hook function that stores logits per layer
- **`summarize_routing_data()`**: Aggregate logits into counts, probabilities, and statistics
- **`ablate_experts()`**: Context manager for dynamically masking experts
- **`scale_router_logits_via_weight()`**: Context manager for temperature scaling
- **`build_model_and_tokenizer()`**: Load 4-bit quantized model with optional PEFT adapter

### `viz_utils.py`

Visualization and metric computation:

- **`compute_metrics_from_counts()`**: Convert routing counts to diversity metrics
- **`plot_prefill_vs_decode_heatmaps()`**: Side-by-side expert load heatmaps
- **`plot_prefill_vs_decode_curves()`**: Effective experts and confidence curves
- **`plot_ablation_dashboard()`**: Three-panel ablation comparison
- **`plot_alpha_sweep_curves()`**: Temperature sweep per-prompt curves
- **`plot_alpha_sweep_scalar_summary()`**: Cross-prompt temperature summary

### `visualization.py`

Main experiment script orchestrating:
- Model loading with hooks
- Prefill and generation phases
- Metric computation
- Ablation interventions
- Plot generation

### `alpha_sweep.py`

Temperature scaling experiment:
- Sweeps alpha values
- Applies temperature scaling during generation
- Compares routing statistics across temperatures
- Generates comparative visualizations


---

## Key Findings Summary

1. **Decode routing is more concentrated than prefill**: Effective experts drop significantly during generation
2. **Layer-wise specialization**: Deeper layers exhibit more focused routing patterns
3. **Robust to ablation**: Masking top experts causes graceful load redistribution
4. **Temperature-sensitive**: Increasing $\alpha$ predictably reduces routing diversity
5. **Prompt-specific signatures**: Different task types (code vs creative) show distinct routing patterns


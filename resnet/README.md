# ResNet on CIFAR-10

Training ResNet variants on CIFAR-10 with ablation studies exploring architecture choices, normalization, and learning rate schedules.

---

## Overview

| Model | Parameters | Key Characteristics |
|-------|------------|---------------------|
| **ResNet-18** | 11.2M | 4 stages, BasicBlock (2 convs per block) |
| **ResNet-34** | 21.3M | Deeper with BasicBlock |
| **ResNet-50** | 23.5M | Bottleneck blocks (1x1 → 3x3 → 1x1) |

---

## Dataset: CIFAR-10

- **Images:** 32×32 RGB
- **Classes:** 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- **Train/Test:** 50,000 / 10,000
- **Augmentation:** Random crop (padding=4), horizontal flip
- **Normalization:** mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]

**Note:** CIFAR-10 ResNets use a modified stem (3×3 conv, stride=1, no max pool) instead of the ImageNet stem (7×7 conv, stride=2, max pool) because the input is already small (32×32).

---

## The Residual Connection

### The Problem

Deep networks suffer from *degradation*: adding more layers increases training error. This isn't overfitting—it's an optimization problem. A 56-layer network should perform at least as well as a 20-layer one (extra layers could just learn identity), but SGD can't find this solution.

### The Solution

Instead of learning `H(x)`, learn the residual `F(x) = H(x) - x`:

```
output = F(x) + x
```

If identity is optimal, the network just needs `F(x) ≈ 0` (easier than `F(x) ≈ x`).

### Why It Works

**Gradient flow:** During backprop, `∂L/∂x = ∂L/∂H · (∂F/∂x + 1)`. The +1 ensures gradients always flow, even if `∂F/∂x ≈ 0`.

**Better initialization:** At init, weights are small → `F(x) ≈ 0` → network starts near identity. This is a good starting point.

---

## Architecture Details

### BasicBlock (ResNet-18/34)

```
x → [3×3 conv] → [BN] → [ReLU] → [3×3 conv] → [BN] → (+) → [ReLU] → out
 └──────────────────── identity ─────────────────────┘
```

### Bottleneck (ResNet-50+)

```
x → [1×1 conv] → [BN] → [ReLU] → [3×3 conv] → [BN] → [ReLU] → [1×1 conv] → [BN] → (+) → [ReLU]
 └────────────────────────────────── identity ──────────────────────────────────┘
```

The 1×1 convolutions reduce then expand channels (bottleneck), making the 3×3 conv cheaper.

### Downsampling

When stride > 1 or channels change, the skip connection uses a 1×1 conv projection:

```python
if stride != 1 or in_channels != out_channels:
    downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
```

---

## Ablation Results

All runs trained for 100 epochs on CIFAR-10.

### Summary Table

| Experiment | Best Val Acc | Final Val Acc | Notes |
|------------|--------------|---------------|-------|
| **Cosine Schedule** | **94.1%** | 94.0% | Best overall |
| Baseline (MultiStep) | 93.9% | 87.6% | LR drops cause instability late |
| No Schedule | 90.9% | 88.6% | Constant LR=0.1 |
| ResNet-34 | 89.9% | 84.9% | Larger model, needs more epochs |
| ResNet-50 | 88.3% | 87.0% | Bottleneck, needs more epochs |
| No BatchNorm | 87.0% | 86.7% | Significantly worse |
| Small Batch (bs=32) | 88.4% | 87.0% | Only 72 epochs logged |

### Detailed Analysis

#### Learning Rate Schedule Comparison

| Schedule | LR Pattern | Best Val Acc | Observations |
|----------|------------|--------------|--------------|
| Cosine | 0.1 → 0 smoothly | 94.1% | Smooth convergence, best results |
| MultiStep | 0.1 → 0.01 → 0.001 | 93.9% | Good but some instability after drops |
| None | Constant 0.1 | 90.9% | Underfits, never converges fully |

**Takeaway:** Cosine annealing provides the smoothest training and best final accuracy. MultiStep can cause instability when LR drops suddenly.

#### Model Depth Comparison

| Model | Params | Best Val Acc | Train Acc |
|-------|--------|--------------|-----------|
| ResNet-18 | 11.2M | 94.1% | 100% |
| ResNet-34 | 21.3M | 89.9% | 96.4% |
| ResNet-50 | 23.5M | 88.3% | 94.4% |

**Takeaway:** Deeper models underperform at 100 epochs—they need longer training to converge. ResNet-18 is sufficient for CIFAR-10 and trains fastest.

#### Batch Normalization

| Config | Best Val Acc | Final Val Acc |
|--------|--------------|---------------|
| With BN | 94.1% | 94.0% |
| Without BN | 87.0% | 86.7% |

**Takeaway:** BatchNorm provides ~7% accuracy improvement. It's essential for training deep networks effectively.

---

## Key Observations

1. **Learning rate schedule matters significantly**
   - Cosine annealing > MultiStep > No schedule
   - Smooth LR decay prevents instability

2. **Deeper isn't always better (at fixed epochs)**
   - ResNet-18 outperforms ResNet-34/50 at 100 epochs on CIFAR-10
   - Larger models need proportionally more training

3. **BatchNorm is critical**
   - ~7% accuracy gap with vs without
   - Enables higher learning rates and faster convergence

4. **CIFAR-10 specific considerations**
   - Modified stem (no aggressive downsampling) is important
   - 32×32 images don't benefit from ImageNet-style 7×7 conv stem

---

## Repository Structure

```
resnet/
├── model.py           # ResNet-18/34/50 implementations
├── train_cifar.py     # Training script with CLI args
├── utils.py           # Data loading, training/eval loops
├── config.py          # TrainConfig dataclass, preset configs
├── run_ablations.py   # Run multiple experiments
└── runs/              # TensorBoard logs
```

---

## Usage

### Single Training Run

```bash
# ResNet-18 with cosine schedule (recommended)
python train_cifar.py --model resnet18 --lr-schedule cosine --epochs 200 --seed 42

# ResNet-50 with longer training
python train_cifar.py --model resnet50 --epochs 300 --lr-schedule cosine

# Without batch normalization (for ablation)
python train_cifar.py --no-normalize --lr 0.01
```

### Ablation Studies

```bash
# List available preset configs
python run_ablations.py --list

# Run specific ablations
python run_ablations.py --config baseline cosine_schedule no_batchnorm

# Run all ablations with fixed seed
python run_ablations.py --all --seed 42

# Dry run (show commands without executing)
python run_ablations.py --all --dry-run
```

### Monitoring

```bash
tensorboard --logdir ./runs --port 6006
```

---

## Hyperparameters

### Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 128 | Standard for CIFAR |
| Learning rate | 0.1 | With SGD + momentum |
| Momentum | 0.9 | Standard |
| Weight decay | 5e-4 | L2 regularization |
| Epochs | 200 | For convergence |
| LR schedule | MultiStep [100, 150] | Or cosine |

### Recommended Config

```python
TrainConfig(
    model="resnet18",
    epochs=200,
    lr=0.1,
    lr_schedule="cosine",
    batch_size=128,
    weight_decay=5e-4,
    seed=42
)
```

---

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2015
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) — He et al., 2016 (Pre-activation ResNet)
- [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187) — He et al., 2018 (ResNet-B/C/D variants)

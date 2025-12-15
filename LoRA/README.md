## LoRA Experiments

This section studies **Low-Rank Adaptation (LoRA)** from both a theoretical and a practical perspective. The goal is twofold:

1. Build intuition for LoRA as a constrained linear algebra problem.
2. Empirically evaluate LoRA as a parameter-efficient alternative to full fine-tuning in a realistic transformer setting.

The experiments are split into two parts:
- a self-contained linear theory experiment (`theory.py`)
- a practical transformer fine-tuning experiment on SST-2 using BERT-Tiny

---

## Part I: Linear Algebra Intuition (`theory.py`)

### Setup

We consider a linear model with weights $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and study adaptation from an initial weight matrix $W_0$ to a target $W^*$.

LoRA parameterizes the update as a low-rank matrix:

$ \Delta W = W - W_0 = \frac{\alpha}{r} B A $

where:
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$

The model output is:

$ y = x (W_0 + \Delta W)^\top $

We generate synthetic data using a teacher matrix $W^*$ and train only $A, B$ while keeping $W_0$ frozen.

---

### Key Observations

1. **Best-possible rank-$r$ error**  
   The optimal rank-$r$ approximation to $\Delta^* = W^* - W_0$ is given by truncating its SVD. The residual Frobenius error is:

   $ \|\Delta^* - (\Delta^*)_r\|_F^2 = \sum_{i > r} \sigma_i^2 $

   Learned LoRA updates closely track this theoretical lower bound.

2. **Rank controls expressivity**  
   Increasing $r$ monotonically improves approximation quality in weight space.

3. **Scaling matters**  
   With fixed $\alpha$, increasing $r$ implicitly shrinks the effective update magnitude $\alpha / r$, which can cause optimization artifacts. Correct scaling is crucial for meaningful rank comparisons.

This experiment isolates LoRA’s core behavior as a **low-rank approximation problem** independent of transformers or datasets.

---

## Part II: Practical LoRA Fine-Tuning on SST-2 (Research Log)

### Experimental Setup

We evaluate LoRA in a realistic NLP setting using:

- **Model**: `prajjwal1/bert-tiny`
- **Task**: SST-2 sentiment classification
- **Dataset**: GLUE SST-2
- **Training**: 5 epochs, AdamW with linear warmup + decay
- **Adapters**: inserted into attention `query` and `value` projections

We compare three training regimes:

1. **Head-only**: freeze the encoder, train the classification head
2. **Full fine-tuning**: update all parameters
3. **LoRA**: freeze the base model, train low-rank adapters + classifier

For LoRA, we sweep ranks:

$ r \in \{1, 2, 4, 8, 16, 32\} $

---

### Correct LoRA Scaling

The effective LoRA update is:

$ \Delta W = \frac{\alpha}{r} B A $

In early runs, $\alpha$ was held fixed while increasing $r$, which caused rank-dependent optimization issues.  
We corrected this by scaling:

$ \alpha = c \cdot r \quad \Rightarrow \quad \frac{\alpha}{r} = c $

This ensures that each rank contributes updates at comparable scale and isolates the effect of **expressivity** rather than **conditioning**.

---

### Results

#### Validation Accuracy vs LoRA Rank

- Performance improves rapidly from $r=1$ to $r \approx 8$
- Beyond $r \approx 8$, gains saturate
- No systematic U-shaped degradation at higher ranks once scaling is corrected

This indicates that LoRA rank primarily controls **capacity**, not overfitting, in this regime.

---

#### Accuracy vs Trainable Parameters (Pareto Frontier)

LoRA forms a smooth tradeoff between head-only and full fine-tuning:

- Head-only: minimal parameters, clear underfitting
- Full fine-tuning: highest accuracy, highest cost
- LoRA ($r \in [8,16]$): near-optimal accuracy with <1% trainable parameters

This empirically validates LoRA’s central claim:  
**most task-specific adaptation lives in a low-rank subspace**.

---

#### Training Dynamics

- Head-only converges quickly but plateaus early
- Full fine-tuning learns fastest and reaches the highest asymptote
- LoRA shows stable, smooth learning curves with rank-dependent convergence speed

Higher ranks converge slightly faster but do not dramatically change final accuracy.

---

### Conclusions

From these experiments, we conclude:

1. LoRA behaves as expected from its linear-algebraic formulation
2. Rank controls expressivity, but **scaling controls optimization**
3. With correct scaling, LoRA does *not* exhibit inherent U-shaped performance degradation
4. Small ranks ($r \approx 8$) capture most of the benefit of full fine-tuning
5. LoRA provides a clean, practical Pareto frontier between accuracy and parameter count

Together, the theory and transformer experiments provide a coherent, end-to-end understanding of LoRA as both a mathematical object and a practical fine-tuning strategy.
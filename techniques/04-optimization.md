# Optimization & Training

Under the 10-minute training constraint, optimization efficiency matters enormously. The techniques here either improve convergence speed, final model quality, or training stability — often stacking with each other multiplicatively.

---

## Techniques

### Muon Optimizer (and Variants)

**What it is**: Muon is an optimizer designed for transformer training that applies a Newton-Schulz orthogonalization step to the gradient matrix before the update. It keeps the weight update in the "orthogonal" direction, preventing gradient collapse in deep networks.

**Variants**:
- **MuonEq-R**: Muon with equalized learning rates across parameter groups
- **Parallel Muon**: distributed Muon with parameter banking (see below)
- **Muon + WD (weight decay)**: Muon with carefully tuned weight decay (e.g., `WD=0.04`)

**Why it helps**: Standard Adam can plateau early in tiny-model regimes. Muon's orthogonalization produces updates with better spectral properties, often converging faster and to better optima in the short training budgets of Parameter Golf.

**Implementation note**: Muon involves a few Newton-Schulz iterations per step (computing `X ← (3X - X³)/2` repeatedly). The overhead is small for typical transformer widths.

**Papers**:
- [Muon: Momentum + Orthogonalization](https://kellerjordan.github.io/posts/muon/) — blog post by the competition organizer
- Related: [Shampoo](https://arxiv.org/abs/1802.09568), [SOAP](https://arxiv.org/abs/2409.11321)

---

### EMA (Exponential Moving Average) Weight Averaging

**What it is**: Maintain a running average of model weights during training:
```
ema_weights = decay * ema_weights + (1 - decay) * current_weights
```
Use `ema_weights` for evaluation and the final checkpoint.

**Why it helps**: The EMA model is smoother and more robust than the raw checkpoint at any single step. It acts like a free ensemble over recent model states, typically improving validation loss by a small but consistent margin.

**Typical decay values**: 0.999 – 0.9999. Higher decay = slower-moving average = more smoothing.

**Implementation**:
```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
        self.decay = decay

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
```

---

### SWA (Stochastic Weight Averaging)

**What it is**: Average weights from multiple checkpoints taken at regular intervals during training (typically during a cosine LR tail). Similar to EMA but with equal weighting of past checkpoints.

**Variants in Parameter Golf**:
- **SWA at step 50+**: start averaging from step 50 of a 100-step training run
- Sometimes refers to Sliding Window Attention in other contexts — be precise when reading submissions

**Why it helps**: SWA finds flatter minima in loss landscapes, which tends to generalize better. It's particularly valuable when training is short and noisy.

**Papers**:
- [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)

---

### Fused Softcapped Cross-Entropy (FusedCE)

**What it is**: A CUDA-fused implementation of cross-entropy loss that applies "soft capping" — squashing logits through `tanh(x / cap) * cap` before computing softmax.

**Why it helps**:
- **Soft capping**: prevents extreme logit values from dominating the softmax, stabilizing training, particularly useful in quantized or recurrent models where activations can blow up
- **Fusion**: combining the cap + softmax + log-likelihood into a single CUDA kernel reduces memory I/O and speeds up each step

---

### Hyperparameter Stacking

**What it is**: Combining many small hyperparameter improvements that each contribute marginally but compound to a meaningful gain. Top entries often describe "9+ hparam tweaks."

**Common stacked hparams**:
| Hyperparameter | Typical tuned value | Effect |
|----------------|-------------------|--------|
| Peak LR | Carefully scaled to model size | Faster convergence |
| Min LR ratio | e.g., 0.1× peak | Better final quality |
| LR schedule | Cosine with warmdown | Smooth decay |
| Warmup steps | 50–200 | Stability at start |
| Weight decay | 0.04–0.1 | Regularization |
| WD tapering | Reduce WD in final phase | Preserve late-training dynamics |
| Grad clip value | 1.0 or adaptive | Stability |
| Batch size | Tuned for GPU utilization | Throughput |
| Dropout | Often 0.0 in tiny models | No regularization overhead |

**Key insight**: Each tweak alone is <0.1% improvement. Together, 9 of them can mean 0.5–1% improvement in BPB.

---

### OrthoInit / Spectral Init

**What it is**: Initialize weight matrices as (scaled) orthogonal or semi-orthogonal matrices rather than standard random normal initialization.

**Why it helps**: Orthogonal initialization preserves gradient norms through the network at initialization, preventing early vanishing/exploding gradients. Particularly valuable in deep or recurrent architectures.

**Implementation**:
```python
def ortho_init_(tensor, scale=1.0):
    rows, cols = tensor.shape[0], tensor[0].numel()
    flat = tensor.new(max(rows, cols), min(rows, cols)).normal_()
    u, _, vt = torch.linalg.svd(flat, full_matrices=False)
    init = (u if rows >= cols else vt).reshape(tensor.shape)
    tensor.data.copy_(scale * init)
```

---

### Hessian-Aware SDClip / Adaptive Clipping

**What it is**: Rather than clipping gradient norms uniformly, clip gradients in a way that accounts for local curvature (Hessian). Gradients in high-curvature directions are clipped more aggressively.

**Why it helps**: Standard gradient clipping is curvature-agnostic — it can clip useful, well-conditioned gradients just as aggressively as noisy ones. Hessian-aware clipping preserves signal in flat directions and clips noise in sharp directions.

---

### Parameter Banking

**What it is**: Group multiple linear layer weight matrices into batched 3D tensors (a "bank"), enabling a single batched matrix multiplication instead of multiple individual ones.

**Why it helps**:
- **GPU utilization**: batched matmuls are better parallelized on modern GPUs, especially for small matrices
- **Memory layout**: contiguous memory for all layers in a bank reduces cache misses
- **Muon integration**: Muon's Newton-Schulz step is applied per-bank rather than per-layer, reducing overhead

---

## Training Stability in Tiny Models

Small models are more susceptible to training instability because:
- Individual layers have more weight per parameter
- Quantization noise (in QAT) is proportionally larger
- Short training runs leave less time to recover from bad early dynamics

**Stability checklist**:
1. Use orthogonal or spectral initialization
2. Warm up the learning rate over at least 50 steps
3. Apply soft logit capping (FusedCE)
4. Monitor gradient norms — clip if they spike
5. Fix BOS token handling (see SmearGate in architecture techniques)

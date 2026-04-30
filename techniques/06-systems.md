# Systems & Efficiency

Maximizing hardware utilization within the 10-minute training budget. These techniques reduce wall-clock time per step, enabling more gradient updates in the allotted time.

---

## Techniques

### Mixed Precision (bfloat16)

**What it is**: Train with bfloat16 (brain float 16) activations and gradients rather than float32. Weights are maintained in float32 for accumulation, but all forward/backward passes use bfloat16.

**Why bfloat16 over float16**:
- Same exponent range as float32 (8 bits) → no overflow/underflow issues
- Lower mantissa precision (7 bits vs. 23) → ~2× memory reduction
- Native hardware support on A100/H100 with high throughput

**Typical setup**:
```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits = model(x)
    loss = criterion(logits, targets)
```

**Memory savings**: ~2× reduction in activation memory, enabling larger batches or longer sequences within the same GPU memory.

---

### FlashAttention-3

**What it is**: A highly optimized attention kernel that computes scaled dot-product attention in O(n) memory (rather than O(n²)) by fusing the softmax and matrix multiplications into a single tiled kernel.

**FlashAttention-3 improvements over FA-2**:
- Better pipelining of warp-specialized producer/consumer operations
- Asynchronous data movement using the CUDA `memcpy_async` primitives
- Improved utilization of H100 tensor cores (FP8 support)

**Why it helps**: Attention is often the memory bottleneck for long sequences. FlashAttention-3 allows longer context windows within the same GPU memory and runs faster due to better memory access patterns.

**Usage** (via `torch.nn.functional`):
```python
# PyTorch 2.0+ uses Flash Attention automatically when available
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Papers**:
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)

---

### torch.compile

**What it is**: JIT-compiles PyTorch model code using TorchDynamo + TorchInductor, fusing operations and generating optimized CUDA kernels for the specific hardware.

**Usage**:
```python
model = torch.compile(model, mode='max-autotune')
```

**Compilation modes**:
| Mode | Compilation time | Runtime speedup |
|------|-----------------|-----------------|
| `default` | ~30s | 1.5–2× |
| `reduce-overhead` | ~60s | 2–3× |
| `max-autotune` | ~5min | 3–5× (best) |

**Why it matters for 10-min training**: With `max-autotune`, you can often get 30–50% more training steps in the same wall time — directly translating to better model quality.

**Key fusions**:
- Attention + softmax → single kernel (if not using FlashAttention)
- LayerNorm + residual add → fused kernel
- Optimizer step (Adam/Muon) → fused CUDA kernel

---

### Megakernels

**What it is**: Manually written CUDA or Triton kernels that fuse multiple operations into a single GPU kernel, avoiding round-trips to global memory between operations.

**Common examples in Parameter Golf**:
- Fused cross-entropy with softmax capping (see `FusedCE` in optimization techniques)
- Fused QAT forward pass (quantize + linear + dequantize in one kernel)
- Fused RMSNorm + linear projection

**When to write one**: When profiling shows repeated operations with small amounts of compute separated by large memory transfers. Triton is the easiest path for custom kernel writing.

```python
import triton
import triton.language as tl

@triton.jit
def fused_rms_norm_linear_kernel(x_ptr, w_ptr, out_ptr, ...):
    # Normalize x and apply linear projection in one pass
    ...
```

---

### Per-group lrzip

**What it is**: Apply the `lrzip` compression tool separately to each group of weights (e.g., each 128-weight group from per-group quantization) rather than to the entire checkpoint at once.

**Why it helps**:
- Each group has its own quantization scale → its weight values span a smaller numerical range
- Compressing each group independently exploits intra-group structure that cross-group compression obscures
- Can be combined with different compressors per layer (attention vs. MLP may compress differently)

**Practical notes**:
- `lrzip` uses LZMA + RZIP (a large-window entropy coder) for high ratios
- Overhead: the per-group metadata (scales, zero-points) needs to be stored efficiently
- Typical workflow: serialize each group → lrzip each → concatenate with offset table

---

### SWA / Sliding Window Attention (in systems context)

When used as a **systems** technique (not to be confused with Stochastic Weight Averaging):

**What it is**: Restrict each token's attention to a local window of W preceding tokens rather than the full sequence.

**Why it helps**:
- Reduces attention compute from O(n²) to O(n × W)
- Enables longer training sequences in the same compute budget
- Combined with a few global attention positions (e.g., every 64 tokens), preserves long-range information

---

## GPU Profiling Workflow

For 10-minute training, profiling is essential. Standard workflow:

```bash
# Profile a single training step
python -c "
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    loss = model(x)
    loss.backward()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
"
```

**What to look for**:
1. Operations with high CUDA time but low compute intensity → fusion candidates
2. Memory copy operations between CPU and GPU → move data to GPU earlier
3. Small matrix multiplications → candidates for parameter banking

---

## System Efficiency Summary

| Technique | Training speedup | Memory savings | Complexity |
|-----------|-----------------|----------------|------------|
| bfloat16 | 1.5–2× | 2× activations | Very low |
| FlashAttention-3 | 1.5–3× (long seqs) | O(n) vs O(n²) | Low (use library) |
| torch.compile | 2–4× | Minimal | Low (one line) |
| Custom megakernels | 2–5× per op | Varies | High |
| Per-group lrzip | N/A (post-training) | 30–60% size reduction | Medium |
| SWA (attention) | 2–8× (long seqs) | O(n×W) | Medium |

# Quantization & Compression

Quantization is the single biggest lever for fitting more model capacity into the 16 MB budget. The goal is to represent weights at lower bit-widths without destroying model quality.

---

## Techniques

### Quantization-Aware Training (QAT)

**What it is**: Train the model while simulating low-precision weights using the Straight-Through Estimator (STE). Rather than quantizing after training (which causes a big accuracy drop), the model learns to work around quantization noise from the start.

**Why it helps**: The model adapts its weight distributions to be quantization-friendly. In practice, int6 QAT loses far less quality than int6 post-training quantization.

**How it works**:
```
forward pass:  w_quantized = round(w / scale) * scale   (fake-quantized)
backward pass: gradient flows through as if weights were continuous (STE)
```

**Key implementation choices**:
- Choose bit-width (int5, int6, mixed int5/int6)
- Per-channel vs per-group scaling — per-group (e.g., group size 64 or 128) usually outperforms per-channel
- Whether to quantize embeddings separately (often kept at higher precision)

**Learning tip**: Start with the [GPTQ paper](https://arxiv.org/abs/2210.17323), then implement STE-based QAT in a nanoGPT fork. Add a `quantize_weights()` function that runs on every forward pass.

---

### GPTQ / GPTQ-lite

**What it is**: A post-training quantization algorithm that uses a small calibration dataset and per-column Hessian information to minimize the quantization error layer by layer.

**Why it helps**: More accurate than naive rounding, especially at int4/int6. The "lite" variant used in Parameter Golf uses self-generated calibration data (model samples) and a clip search to find optimal quantization ranges.

**Key steps**:
1. Run calibration data through the model to collect Hessian statistics
2. For each layer, solve for quantized weights that minimize `||W - W_q||_H` (Hessian-weighted error)
3. Optionally use grid search over clip thresholds

**Combined with QAT**: Many top entries run QAT first (to adapt the model), then apply GPTQ-lite as a final cleanup pass.

**Papers**:
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

---

### Per-group / Asymmetric Quantization

**What it is**: Rather than one scale per tensor or per row, use a separate scale (and optionally zero-point) for every group of `G` consecutive weights (e.g., G=64 or G=128).

**Why it helps**: Weight distributions vary across a row. Per-group scaling lets each group use its full dynamic range, reducing quantization error at a small metadata cost.

**Asymmetric**: Uses both a scale and a zero-point per group, allowing non-zero-centered ranges. Costs 1 extra value per group but can noticeably improve quality.

---

### Artifact Compression (zstd, lrzip, lzma)

**What it is**: After quantizing, apply a general-purpose compressor to the final checkpoint file. The 16 MB limit applies to the compressed file.

**Why it helps**: Quantized weights (especially int6 or int8) are more compressible than float32 because they have limited unique values. Adjacent weights often have similar values, so context-modeled compressors exploit this well.

**Compressor options** (roughly increasing compression ratio / time):
| Compressor | Notes |
|------------|-------|
| `zstd -22` | Fast, good ratio, widely available |
| `lzma` | Slower, slightly better ratio |
| `lrzip` | Best ratios on this type of data; can be applied per-group |
| `int8 zlib roundtrip` | Trick: convert int6 → int8, compress with zlib, exploit structure |

**Per-group lrzip**: Apply `lrzip` independently to each group's weight block. Exploits intra-group structure that cross-group compression would miss.

---

### Ternary / 1-bit Quantization

**What it is**: Extreme quantization where weights are restricted to `{-1, 0, 1}` (ternary) or `{-1, 1}` (binary).

**Why it helps**: Massively reduces storage; enables highly optimized inference kernels. Seen in non-record entries pushing 100M+ effective parameters into the budget.

**Trade-off**: Training is harder; model capacity per parameter drops significantly. Most competitive entries use int5/int6 rather than ternary for the quality/size sweet spot.

**Papers**:
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

---

### Late QAT

**What it is**: Train in full precision for most of training, then switch on quantization for only the final phase.

**Why it helps**: Preserves gradient quality and learning dynamics early in training (quantization noise can slow convergence). The model converges to a good solution first, then adapts to quantization constraints in the final phase.

**Implementation**: Add a `qat_start_step` hyperparameter; skip the `quantize_weights()` call until that step.

---

## How These Techniques Stack

A typical top-entry compression pipeline:

```
Full-precision training (full length)
  → Late QAT (final ~20% of steps, int6 per-group)
  → GPTQ-lite cleanup pass
  → per-group lrzip compression
  → final artifact ≤ 16 MB
```

The combination of QAT (quality), GPTQ (cleanup), and lrzip (compression) is more effective than any one technique alone.

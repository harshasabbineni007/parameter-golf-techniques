# How Techniques Compound

Individual techniques rarely win competitions. The top Parameter Golf entries achieve their results by systematically stacking techniques that compound multiplicatively. This document explains the stacking strategy, the order in which to add techniques, and what the top entries actually did.

---

## The Stacking Principle

Each technique provides diminishing returns when applied alone. But when applied together, they often compound:

- **Depth recurrence** doubles effective depth → model benefits more from **better optimization** (Muon)
- **Quantization** frees parameter budget → can add more **unique layers** in the recurrent architecture
- **TTT** adapts to test distribution → more effective when the **base model** is already strong
- **Compression** (lrzip) is more effective on **quantized** weights than float32 ones

The implication: add techniques in dependency order, not arbitrary order.

---

## Progress Timeline

| Phase | BPB | Techniques |
|-------|-----|-----------|
| Baseline nanoGPT | 1.2244 | Standard GPT-2 architecture, Adam, float32 |
| Hparam tuning | ~1.18 | LR schedule, batch size, warmup |
| Muon optimizer | ~1.15 | Replace Adam with Muon |
| Small vocab | ~1.13 | Vocab 1024 + SentencePiece |
| Depth recurrence | ~1.12 | 8 unique blocks × 2 loops |
| QAT int6 | ~1.11 | Per-group QAT, late application |
| + GPTQ + lrzip | ~1.10 | Post-QAT cleanup + compression |
| Gated attention | ~1.09 | SparseAttnGate / QuantGate |
| + SmearGate + hparam stack | ~1.08 | 9+ hparam tweaks, BOS fix |
| N-gram caching | ~1.07 | 7-gram backoff mixer |
| Score-First TTT | ~1.065 | Phased TTT, 5–10 steps/chunk |
| Full stack + advanced gates | ~1.061 | All of the above, tuned together |

---

## A Top-Entry Stack (Reconstructed from PR #1855)

```
Base architecture:
  ├── Depth recurrence (8 unique × 2 loops, FiLM-conditioned)
  ├── SparseAttnGate at all layers (XSA-all)
  ├── SmearGate + BOS-Fixed
  ├── 3× MLP expansion
  ├── Partial RoPE
  ├── RMSNormNoWeight
  └── Vocab 1024 (SentencePiece BPE)

Optimization:
  ├── Muon optimizer (MuonEq-R variant)
  ├── OrthoInit for all weight matrices
  ├── FusedCE (softcapped cross-entropy)
  ├── EMA decay 0.9999
  └── Hparam stack (LR, schedule, WD tapering, clip)

Quantization & Compression:
  ├── Late QAT (int6 per-group, starts at 80% of training)
  ├── GPTQ-lite cleanup pass
  └── Per-group lrzip

Systems:
  ├── torch.compile (max-autotune)
  ├── FlashAttention-3
  ├── bfloat16 mixed precision
  └── Parameter banking

Test-time:
  ├── Score-First TTT (5 steps, full fine-tune)
  ├── 7-gram backoff n-gram mixer (α=0.15)
  └── Sliding window eval (stride 64)
```

---

## Technique Interaction Matrix

Some techniques reinforce each other; others partially overlap:

| | Depth Recurrence | Muon | QAT | TTT | N-gram |
|---|---|---|---|---|---|
| **Depth Recurrence** | — | ++ | + | + | neutral |
| **Muon** | ++ | — | + | neutral | neutral |
| **QAT** | + | + | — | + | neutral |
| **TTT** | + | neutral | + | — | + |
| **N-gram** | neutral | neutral | neutral | + | — |

`++` = strong synergy, `+` = positive interaction, `neutral` = independent

**Key interactions**:
- **Depth recurrence + Muon**: Muon's orthogonalization is especially effective for deep/recurrent networks where gradient flow is a concern
- **QAT + TTT**: Pre-quant TTT (adapting before final quantization) can outperform TTT on the quantized model because adaptation gradients are cleaner
- **TTT + N-gram**: Both adapt to the test distribution — combining them with appropriate α blending outperforms either alone

---

## Diminishing Returns

Each additional technique provides smaller marginal gains. The engineering cost often doesn't justify the BPB improvement alone — but in a competition, every 0.001 BPB matters.

Rough marginal BPB improvements (applied on top of a strong baseline):
```
Depth recurrence:     -0.020 to -0.030
Muon optimizer:       -0.015 to -0.025
QAT + lrzip:          -0.010 to -0.020
Gated attention:      -0.005 to -0.010
SmearGate + hparams:  -0.003 to -0.008
TTT (score-first):    -0.010 to -0.030  ← high variance, depends on data
N-gram caching:       -0.005 to -0.015  ← depends on text repetition
Sliding window eval:  -0.002 to -0.005
```

---

## Where to Start: A Practical Guide

**Day 1**: Get a working nanoGPT baseline. Understand the BPB metric. Tune learning rate and batch size.

**Day 2–3**: Implement depth recurrence with 8 unique blocks × 2 loops. Switch to Muon. Add EMA. You should be competitive at ~1.12 BPB.

**Day 4–5**: Add QAT (int6 per-group). Add per-group lrzip compression. Tune the `qat_start_step`. Target ~1.10 BPB.

**Day 6–7**: Add Score-First TTT. This is the trickiest to implement correctly — make sure you pass the compliance check. Add 7-gram n-gram mixer with careful α tuning. Target ~1.08 BPB.

**Beyond**: Tune architecture (gated attention, SmearGate), stack hyperparameters, experiment with GPTQ cleanup. Each iteration is smaller gains but they add up.

---

## Common Mistakes

1. **Adding TTT incorrectly**: using future tokens for adaptation is the most common reason for rejection
2. **Ignoring the BOS fix**: BOS instability is subtle but causes training spikes in recurrent architectures
3. **Over-compressing before QAT**: applying lrzip to float32 weights is much less effective than lrzip on quantized int6 weights
4. **Tuning hparams on test data**: hyperparameter search should use a held-out validation split, not the competition validation set
5. **Underestimating compile time**: `torch.compile(max-autotune)` takes 3–5 minutes — on a 10-minute budget, this must be amortized carefully

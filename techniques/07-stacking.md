# How Techniques Compound

Individual techniques rarely win competitions. The strongest Parameter Golf submissions win by stacking many individually modest improvements. This document explains the stacking strategy, the order in which to add techniques, and how the late-April frontier was assembled.

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
| Baseline | 1.2244 | 9-layer SP1024 baseline |
| First big gains | ~1.15 to ~1.12 | int6/QAT, sliding eval, Muon, larger model simplifications |
| Mid frontier | ~1.08 to ~1.06 | SP8192, recurrence, parallel residuals, legal TTT, CaseOps |
| Merged late-April frontier | 1.06108 | sparse attention gate, BOS-fixed SmearGate, 9-hparam stack, phased TTT, lrzip |
| Open frontier claims | ~1.04 to ~1.03 | pre-quant TTT, CaseOps byte sidecars, SP10240, more aggressive eval adaptation |

---

## A Merged Frontier Stack (PR #1855)

```
Base architecture:
  ├── 11-layer, 512d, 8H/4KV transformer
  ├── Partial RoPE
  ├── U-Net skips and parallel residuals
  ├── Sparse attention head-output gate
  ├── SmearGate + BOS-Fixed
  └── LeakyReLU-square MLP

Optimization:
  ├── Polar-Express Newton-Schulz Muon
  ├── FusedCE (softcapped cross-entropy)
  ├── EMA
  └── 9 validated hparam overrides

Quantization & Compression:
  ├── GPTQ int6 + int7 embeddings
  ├── LQER asymmetric correction
  └── Per-group lrzip + brotli

Systems:
  ├── FlashAttention-3
  ├── bfloat16 mixed precision
  └── careful serializer layout

Test-time:
  ├── phased score-first TTT
  └── sliding-window eval under the eval budget
```

This is a real merged stack, not a hypothetical composite. It is a better learning target than older summaries that mix together techniques from unrelated eras of the competition.

---

## Open Frontier Themes

The strongest open claims in late April 2026 push below the merged 1.06108 record mainly through:

- **Pre-quant TTT**: adapt the full-precision model after a legality-preserving score pass, then quantize the adapted model
- **CaseOps + byte sidecars**: keep capitalization reversible while scoring BPB on original bytes
- **SP10240 / larger tokenizer choices**: spend more bytes on tokenization to shorten sequences
- **More aggressive eval adaptation**: longer adaptation schedules, federated averaging, or SLOT-like methods

These approaches are important to study, but they are not all merged or fully settled from a rules perspective yet.

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
QAT/GPTQ + packaging: -0.010 to -0.020
Gated attention:      -0.005 to -0.010
SmearGate + hparams:  -0.003 to -0.008
TTT (score-first):    -0.010 to -0.030  ← high variance, depends on data
Sliding window eval:  -0.002 to -0.005
Pre-quant TTT:        potentially larger than standard TTT, but contested
```

---

## Where to Start: A Practical Guide

**Day 1**: Get a working nanoGPT baseline. Understand the BPB metric. Tune learning rate and batch size.

**Day 2–3**: Implement recurrence or parallel residuals. Switch to Muon. Add EMA. This gets you into the historically important mid-pack techniques.

**Day 4–5**: Add GPTQ or late QAT. Improve artifact packaging. Tune the quantization and compression path together.

**Day 6–7**: Add score-first TTT. This is the trickiest to implement correctly, so optimize for legality and reproducibility before chasing another 0.001 BPB.

**Beyond**: Tune sparse gates, SmearGate, hparams, tokenizer choice, and possibly CaseOps or pre-quant TTT if you are comfortable living on the rules frontier.

---

## Common Mistakes

1. **Adding TTT incorrectly**: using future tokens for adaptation is the most common reason for rejection
2. **Ignoring the BOS fix**: BOS instability is subtle but causes training spikes in recurrent architectures
3. **Over-compressing before QAT**: applying lrzip to float32 weights is much less effective than lrzip on quantized int6 weights
4. **Treating open claims as confirmed history**: always separate merged records from pending PRs
5. **Underestimating compile or serialization time**: systems wins only help if the whole run still fits the real wall-clock budget

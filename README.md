# Parameter Golf — Techniques for Tiny, High-Performance LLMs

A practical reference for engineers interested in training language models under extreme size and compute constraints. All techniques here come from top submissions to the [Parameter Golf](https://github.com/KellerJordan/modded-nanogpt) competition, where models must fit within **16 MB** and train in under **10 minutes** while minimizing validation bits-per-byte (BPB) on a held-out text corpus.

The best entries have pushed from a **1.2244 BPB baseline** down to around **1.06 BPB** — a large gap for such a constrained setting. The techniques that got them there are broadly applicable to efficient LLM research.

---

## What is Parameter Golf?

Parameter Golf is a competitive benchmark where participants submit small language models subject to hard constraints:

- **Size**: final model artifact ≤ 16,000,000 bytes (including weights + compression)
- **Training time**: ≤ 10 minutes on a fixed GPU budget
- **Metric**: validation bits-per-byte (lower = better)

The competition forces practitioners to think carefully about every byte and every FLOP — producing a rich collection of techniques for model efficiency.

---

## Technique Categories

| # | Category | Key Ideas |
|---|----------|-----------|
| [1](techniques/01-quantization.md) | **Quantization & Compression** | QAT, GPTQ, per-group compression, zstd |
| [2](techniques/02-architecture.md) | **Architectural Modifications** | Depth recurrence, gated attention, custom activations |
| [3](techniques/03-tokenization.md) | **Tokenization & Data** | Small vocab, CaseOps, n-gram hash tokenizers |
| [4](techniques/04-optimization.md) | **Optimization & Training** | Muon optimizer, EMA, hyperparameter stacking |
| [5](techniques/05-test-time.md) | **Test-Time Techniques** | TTT, n-gram caching, sliding window eval |
| [6](techniques/06-systems.md) | **Systems & Efficiency** | FlashAttention-3, torch.compile, lrzip |
| [7](techniques/07-stacking.md) | **How Techniques Compound** | Stacking strategies from top entries |

---

## Quick Learning Path

If you're new to this space, work through in this order:

1. **Start with architecture**: implement [depth recurrence](techniques/02-architecture.md) in a nanoGPT fork — simple but high-impact.
2. **Add quantization**: follow the [QAT guide](techniques/01-quantization.md) with int6 weights.
3. **Tune optimization**: swap in the [Muon optimizer](techniques/04-optimization.md) and add EMA.
4. **Compress the artifact**: apply [per-group lrzip](techniques/06-systems.md) to maximize bytes used.
5. **Add test-time techniques**: implement a legal [score-first TTT](techniques/05-test-time.md) loop.

---

## Progress History

| Phase | BPB | Techniques Added |
|-------|-----|-----------------|
| Baseline | 1.2244 | Standard nanoGPT |
| Early gains | ~1.12x | QAT + hparam tuning |
| Mid-tier | ~1.09x | Depth recurrence + Muon |
| Top entries | ~1.06x | Full stack + advanced TTT + sparse gates |

---

## Contributing / Further Reading

- [Parameter Golf GitHub](https://github.com/KellerJordan/modded-nanogpt) — official leaderboard and submission PRs
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the baseline model this competition builds on
- See each technique file for paper links and implementation pointers

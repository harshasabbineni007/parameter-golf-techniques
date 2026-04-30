# Parameter Golf — Techniques for Tiny, High-Performance LLMs

A practical reference for engineers interested in training language models under extreme size and compute constraints. The techniques here are drawn from the [OpenAI Parameter Golf repository](https://github.com/openai/parameter-golf), where submissions must fit within **16 MB**, train in under **10 minutes**, and minimize validation bits-per-byte (BPB) on FineWeb.

As of **April 30, 2026**, the public repo has moved from a **1.2244 BPB baseline** to a **merged 1.06108 BPB record** in PR [#1855](https://github.com/openai/parameter-golf/pull/1855). There are also lower **open** claims, including PR [#1911](https://github.com/openai/parameter-golf/pull/1911) at **1.0354** and PR [#1972](https://github.com/openai/parameter-golf/pull/1972) at **1.03983**, but those are not merged yet and should be treated as pending rather than confirmed.

---

## What is Parameter Golf?

Parameter Golf is a competitive benchmark where participants submit small language models subject to hard constraints:

- **Size**: final model artifact ≤ 16,000,000 bytes (including weights + compression)
- **Training time**: ≤ 10 minutes on 8xH100 SXM GPUs for the official track
- **Metric**: validation bits-per-byte (lower = better), computed against original bytes

The competition forces practitioners to think carefully about every byte and every FLOP — producing a rich collection of techniques for model efficiency.

One wrinkle: the Mintlify docs leaderboard has lagged behind the GitHub repo at points, so the safest source of truth is the `openai/parameter-golf` PR history, especially for late-April frontier results.

---

## Contents

```
parameter-golf-techniques/
│
├── techniques/
│   ├── 01-quantization.md       ← QAT, GPTQ, per-group compression, brotli/lrzip/lzma
│   ├── 02-architecture.md       ← Depth recurrence, parallel residuals, sparse gating
│   ├── 03-tokenization.md       ← SP8192/SP10240, CaseOps, BigramHash, byte accounting
│   ├── 04-optimization.md       ← Muon optimizer, EMA, SWA, hyperparameter stacking
│   ├── 05-test-time.md          ← Score-first TTT, phased TTT, pre-quant TTT, n-gram caching
│   ├── 06-systems.md            ← FlashAttention-3, torch.compile, megakernels, artifact packaging
│   └── 07-stacking.md          ← How techniques compound; full merged frontier stack
│
└── resources.md                 ← Papers, repos, and further reading by category
```

### Technique Files

- [**01 · Quantization & Compression**](techniques/01-quantization.md) — QAT, GPTQ-lite, per-group asymmetric quantization, artifact compression (brotli, lrzip, lzma)
- [**02 · Architectural Modifications**](techniques/02-architecture.md) — Depth recurrence, U-Net skips, sparse attention gates, SmearGate, partial RoPE, custom activations
- [**03 · Tokenization & Data**](techniques/03-tokenization.md) — Vocabulary size tradeoffs (SP1024→SP10240), CaseOps, BigramHash, byte-accurate BPB accounting
- [**04 · Optimization & Training**](techniques/04-optimization.md) — Muon optimizer, EMA, SWA, FusedCE, hyperparameter stacking, ortho init
- [**05 · Test-Time Techniques**](techniques/05-test-time.md) — Score-first TTT, LoRA-TTT, pre-quant TTT, n-gram backoff mixer, sliding window eval
- [**06 · Systems & Efficiency**](techniques/06-systems.md) — FlashAttention-3, torch.compile, Triton megakernels, bfloat16, artifact packaging
- [**07 · How Techniques Compound**](techniques/07-stacking.md) — Stacking order, interaction matrix, full PR #1855 stack, open frontier themes

---

## Quick Learning Path

If you're new to this space, work through in this order:

1. **Start with architecture**: implement [depth recurrence](techniques/02-architecture.md) in a nanoGPT fork — simple but high-impact.
2. **Add quantization**: follow the [quantization guide](techniques/01-quantization.md) with int6 GPTQ or late QAT.
3. **Tune optimization**: swap in the [Muon optimizer](techniques/04-optimization.md) and add EMA.
4. **Compress the artifact**: apply [brotli/lrzip/lzma packaging](techniques/06-systems.md) to maximize bytes used.
5. **Add test-time techniques**: implement a legal [score-first TTT](techniques/05-test-time.md) loop, then explore phased or pre-quant variants.

---

## Progress History

| Phase | BPB | Techniques Added |
|-------|-----|-----------------|
| Baseline | 1.2244 | 9-layer SP1024 baseline |
| Early merged frontier | ~1.12 to ~1.08 | XSA, GPTQ, legal TTT, SP8192 |
| Late-April merged frontier | 1.06108 | sparse gates, BOS-fixed SmearGate, phased TTT, lrzip |
| Open claims | ~1.04 to ~1.03 | pre-quant TTT, CaseOps, SP10240, more aggressive eval adaptation |

---

## Contributing / Further Reading

- [Parameter Golf GitHub](https://github.com/openai/parameter-golf) — canonical repo for records, issues, and submission PRs
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the baseline model this competition builds on
- [Parameter Golf docs leaderboard](https://openai-parameter-golf.mintlify.app/leaderboard) — useful overview, but verify late-breaking records against GitHub PRs
- See each technique file for paper links and implementation pointers

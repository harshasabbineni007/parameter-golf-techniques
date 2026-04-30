# Resources & Further Reading

Curated references for each technique area, organized for progressive learning.

---

## Competition & Baseline

- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) — the competition repository, merged records, and open submission PRs
- [Parameter Golf leaderboard docs](https://openai-parameter-golf.mintlify.app/leaderboard) — overview page; useful, but it can lag the GitHub repo
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's minimal GPT-2 implementation; the baseline most entries build on
- [Muon optimizer blog post](https://kellerjordan.github.io/posts/muon/) — explains the optimizer used in most top entries

---

## Quantization

| Resource | What it covers |
|----------|---------------|
| [GPTQ paper](https://arxiv.org/abs/2210.17323) | Post-training quantization with Hessian-weighted error minimization |
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | Mixed int8/float16 inference |
| [BitNet](https://arxiv.org/abs/2310.11453) | 1-bit transformers |
| [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) | 1.58-bit (ternary) LLMs |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Quantized LoRA fine-tuning; relevant for LoRA-TTT |

---

## Architecture

| Resource | What it covers |
|----------|---------------|
| [Universal Transformers](https://arxiv.org/abs/1807.03819) | Depth recurrence in transformers |
| [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196) | Theoretical analysis of looped architectures |
| [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) | Rotary positional embeddings |
| [Hungry Hungry Hippos](https://arxiv.org/abs/2212.14052) | Sparse attention alternatives |
| [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) | Feature-wise Linear Modulation (loop conditioning) |

---

## Optimization

| Resource | What it covers |
|----------|---------------|
| [Adam paper](https://arxiv.org/abs/1412.6980) | Baseline optimizer |
| [Shampoo](https://arxiv.org/abs/1802.09568) | Second-order preconditioned optimizer |
| [SOAP](https://arxiv.org/abs/2409.11321) | Related to Muon; Shampoo in Adam's parameter space |
| [SWA paper](https://arxiv.org/abs/1803.05407) | Stochastic Weight Averaging finds flatter minima |
| [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) | LR schedule reference |

---

## Test-Time Techniques

| Resource | What it covers |
|----------|---------------|
| [TTT: Test-Time Training with Self-Supervision](https://arxiv.org/abs/1909.13231) | Original TTT paper |
| [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) | Modern TTT with sequence models |
| [Kneser-Ney smoothing](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing) | Background for n-gram language models |
| [LoRA paper](https://arxiv.org/abs/2106.09685) | Low-rank adaptation (used in LoRA-TTT) |

---

## Systems

| Resource | What it covers |
|----------|---------------|
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) | Fast exact attention for H100 |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Predecessor; widely deployed |
| [Triton](https://triton-lang.org/) | Python-like GPU kernel language for megakernels |
| [torch.compile docs](https://pytorch.org/docs/stable/torch.compiler.html) | PyTorch compilation guide |
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | fp16 training techniques |

---

## Tokenization

| Resource | What it covers |
|----------|---------------|
| [SentencePiece](https://github.com/google/sentencepiece) | BPE/unigram tokenizer library |
| [BPE paper](https://arxiv.org/abs/1508.07909) | Byte Pair Encoding for NLP |
| [tiktoken](https://github.com/openai/tiktoken) | OpenAI's fast BPE tokenizer |

---

## General Background

| Resource | What it covers |
|----------|---------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original transformer paper |
| [Language Models are Unsupervised Multitask Learners (GPT-2)](https://openai.com/research/language-unsupervised) | GPT-2 paper |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Visual introduction to transformers |
| [Andrej Karpathy's Neural Networks series](https://www.youtube.com/watch?v=VMj-3S1tku0) | From scratch in Python/PyTorch |

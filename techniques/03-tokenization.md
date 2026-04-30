# Tokenization & Data Handling

Tokenization choices directly affect the embedding table size (a large fraction of total parameters) and the model's ability to learn from limited data in 10 minutes.

---

## Techniques

### Small Fixed Vocabulary (e.g., 1024 tokens)

**What it is**: Replace the standard ~50,000-token vocabulary with a drastically smaller one (e.g., 1024 tokens using SentencePiece BPE or a custom scheme).

**Why it helps**: The embedding table costs `vocab_size × d_model` parameters. At d_model=512:
- Standard vocab (50k): 50,000 × 512 = **25.6M parameters** (102 MB at float32)
- Small vocab (1024): 1,024 × 512 = **0.5M parameters** (2 MB at float32)

Savings: ~97% of embedding parameters freed for model depth/width.

**Trade-off**: Smaller vocab → longer token sequences for the same text → more tokens per forward pass, higher compute cost. The sweet spot depends on your compute budget vs. parameter budget.

**How to build one**:
```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='your_corpus.txt',
    model_prefix='vocab_1024',
    vocab_size=1024,
    model_type='bpe'
)
```

---

### CaseOps (Lossless Case Transform)

**What it is**: A bijective (lossless, reversible) transform that strips case from tokens and tracks case information in a small "sidecar" structure. The main vocabulary operates on lowercased text; case is recovered at decode time.

**Why it helps**: Case-insensitive tokenization reduces effective vocabulary size (no need for separate tokens for "The", "the", "THE", etc.) while remaining fully lossless. The sidecar accounting ensures BPB measurement remains honest.

**Key property**: BPB is computed over the original text, not the case-stripped version. The sidecar overhead is counted. This makes it a legitimate compression technique rather than cheating.

---

### BigramHash / TrigramHash / Novel Tokenizers

**What it is**: Augment or replace standard BPE tokenization with hash-based n-gram representations. Rather than a learned vocabulary, encode token context as a hash of the surrounding n-gram.

**Variants**:
- **BigramHash**: encode each position as a hash of the current + previous token
- **TrigramHash**: three-token context window
- **H-net**: a custom hierarchical tokenizer
- **SP8192 / SP4096**: SentencePiece models at 8192 or 4096 vocab size

**Why it helps**: N-gram hashes add contextual information to each token's embedding at essentially zero parameter cost (just a hash table lookup or simple arithmetic). The model can distinguish "bank" as the first word vs. "bank" after "river" without enlarging the vocabulary.

**Implementation sketch**:
```python
def bigram_hash_embed(token_ids, vocab_size):
    bigrams = torch.stack([token_ids[:-1], token_ids[1:]], dim=-1)
    hashes = (bigrams[:, 0] * 31 + bigrams[:, 1]) % HASH_DIM
    return hashes  # used as additional embedding lookup indices
```

---

### Sliding Window / Long Context Attention (VarLenAttn)

**What it is**: During evaluation (and optionally training), process sequences with overlapping windows so each token benefits from a longer effective context.

**Variants**:
- **Sliding window eval with stride S**: compute loss only on tokens not in the overlap region, but attend to a full window of prior tokens
- **VarLenAttn**: variable-length attention kernels (e.g., from FlashAttention) that handle variable-length packed sequences efficiently

**Why it helps**: Language models benefit strongly from longer context — a model trained on 512-token windows can score better on test data if evaluated with 1024-token windows at stride 64. The model sees more context per token during scoring.

**Sliding window eval (stride 64)**:
```
Window 1: tokens [0,   512]  → score tokens [0,   64]
Window 2: tokens [64,  576]  → score tokens [64,  128]
...
```
Each scored token has seen up to 448 tokens of prior context rather than just what fits in a single pass.

---

## Data Handling for 10-Minute Training

### Coprime Loaders

**What it is**: A data loading strategy where multiple data-loading workers use coprime stride lengths to sample the training corpus. Ensures maximal coverage without repetition or correlated mini-batches.

**Why it helps**: In 10 minutes, you only see a fraction of a large corpus. Coprime strides ensure different workers see different parts of the data, reducing variance in gradient estimates.

### Multi-phase Global SGD

**What it is**: Divide training into phases, each with different data sampling strategies, learning rates, or dataset mixes. Transition between phases based on step count or validation loss.

**Why it helps**: Early training benefits from broad data coverage (exploration); late training benefits from focusing on hard examples or specific distributions (exploitation). Multi-phase schedules exploit this.

---

## Summary Table

| Technique | Parameter Savings | Quality Impact | Complexity |
|-----------|------------------|----------------|------------|
| Vocab 1024 | Very High | Moderate (longer seqs) | Low |
| CaseOps | Medium | Neutral (lossless) | Medium |
| BigramHash | Zero (compute only) | Small positive | Low |
| Sliding window eval | Zero | Medium positive | Low |
| Coprime loaders | Zero | Small positive | Low |

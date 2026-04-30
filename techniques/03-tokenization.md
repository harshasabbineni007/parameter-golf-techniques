# Tokenization & Data Handling

Tokenization choices directly affect the embedding table size (a large fraction of total parameters) and the model's ability to learn from limited data in 10 minutes.

---

## Techniques

### Vocabulary Size as a First-Class Tradeoff

**What it is**: Choose tokenizer size deliberately instead of defaulting to a standard 50k vocabulary. Early Parameter Golf records used very small vocabularies like 1024; later frontier entries often moved to `SP4096`, `SP8192`, and even `SP10240`.

**Why it helps**: The embedding table costs `vocab_size × d_model` parameters. At d_model=512:
- Standard vocab (50k): 50,000 × 512 = **25.6M parameters** (102 MB at float32)
- Small vocab (1024): 1,024 × 512 = **0.5M parameters** (2 MB at float32)

Smaller vocabularies save a huge number of embedding parameters, but they also make sequences longer. The frontier moved upward in vocab size once participants found they could afford larger tokenizers and still stay under 16 MB.

**Trade-off**: Smaller vocab means longer sequences and more training/eval compute. Larger vocab means more embedding bytes but fewer tokens per document. Parameter Golf's history is largely a story of finding better points on this curve.

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

**Key property**: BPB must still be computed over the original text, not the transformed stream. In practice, frontier CaseOps submissions export validation byte sidecars so the scorer uses original-byte counts instead of a guessed bytes-per-token ratio.

---

### BigramHash and Other Token-Side Context Tricks

**What it is**: Add a lightweight hashed context feature to tokens, usually with a separate embedding lookup keyed by a bigram or trigram hash.

**Variants**:
- **BigramHash**: encode each position as a hash of the current + previous token
- **TrigramHash**: three-token context window
- **SP8192 / SP4096**: SentencePiece models at 8192 or 4096 vocab size

**Why it helps**: Hash features add local context without needing a much larger vocabulary. Several strong submissions used BigramHash-style features as part of a broader stack, though it is not the main late-April frontier direction.

**Implementation sketch**:
```python
def bigram_hash_embed(token_ids, vocab_size):
    bigrams = torch.stack([token_ids[:-1], token_ids[1:]], dim=-1)
    hashes = (bigrams[:, 0] * 31 + bigrams[:, 1]) % HASH_DIM
    return hashes  # used as additional embedding lookup indices
```

---

### Byte Accounting and Validation Correctness

**What it is**: If you change the tokenizer or text transform, you also need to change how BPB is computed so it still reflects original UTF-8 bytes.

**Why it helps**: Correct accounting does not improve model quality, but it is required for a submission to be valid. Issue `#1017` in the main repo is the best reference for this.

**Common requirements**:
- Count bytes from the original text, not the transformed token stream
- Handle SentencePiece special tokens correctly
- If you use a reversible transform like CaseOps, carry the byte sidecar through evaluation

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
| Small vocab | Very High | Mixed: helps size, hurts sequence length | Low |
| SP8192/SP10240 | Moderate | Often strong frontier tradeoff | Low |
| CaseOps | Medium | Neutral (lossless) | Medium |
| BigramHash | Zero (compute only) | Small positive | Low |
| Coprime loaders | Zero | Small positive | Low |

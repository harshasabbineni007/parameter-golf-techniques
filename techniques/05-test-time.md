# Test-Time Techniques

Test-time techniques adapt the model or its predictions using information from already-scored tokens during evaluation. They're among the highest-impact techniques in Parameter Golf — and also the most rule-sensitive.

> **Critical rule**: All test-time adaptation must strictly avoid using **future or unscored tokens** when computing scores. Many submissions were rejected for subtle violations of this constraint.

---

## Techniques

### Test-Time Training (TTT)

**What it is**: During inference/evaluation, continue training (or fine-tune) the model on already-scored tokens from the validation set, then use the updated model to score the next chunk.

**Why it helps**: The validation text has a specific distribution. Adapting to it — even for a few gradient steps — lets the model specialize to the test-time domain, reducing bits-per-byte on that specific data.

**Basic TTT loop**:
```
for each chunk of validation tokens:
    1. Score chunk with current model          ← BPB contribution recorded here
    2. Train model for K steps on scored chunk ← only uses already-scored tokens
    3. Proceed to next chunk
```

**Variants**:

#### Score-First TTT (Phased TTT)
The safest and most common variant:
1. Score the current chunk *without adaptation* (under `no_grad`)
2. After scoring, adapt the model using only the just-scored chunk
3. Move to the next chunk

This guarantees no adaptation uses unscored tokens.

#### Warm-Start-A / Warm-A TTT
Initialize the TTT adaptation from a warm checkpoint rather than the base model. Reduces the number of adaptation steps needed per chunk.

#### LoRA-TTT
Rather than fine-tuning all weights, adapt only a small LoRA (low-rank adapter). Reduces adaptation compute and prevents catastrophic forgetting of the base model.

```python
# Tiny LoRA applied only during TTT
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=4):
        super().__init__()
        self.base = base_layer  # frozen during TTT
        self.lora_A = nn.Parameter(torch.zeros(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale
```

#### Pre-quant TTT
Apply TTT before the final quantization step. The model adapts in full precision (or higher precision), then is re-quantized. Avoids gradient noise from quantization during adaptation.

**Papers**:
- [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://arxiv.org/abs/1909.13231)
- [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620)

---

### N-gram Caching / Backoff N-gram Mixer

**What it is**: Build an on-the-fly frequency table from already-scored tokens, then blend its predictions with the neural model's logits.

**How it works**:
1. As you score token `t`, update counts for n-grams ending at `t` (e.g., 7-gram through unigram)
2. For token `t+1`, look up how often each vocabulary token follows the most recent n-gram
3. Blend the n-gram distribution with the model's softmax output: `P_final = α * P_ngram + (1-α) * P_model`
4. Backoff: if the 7-gram has insufficient count, fall back to 6-gram, ..., unigram

**Why it helps**: Captures local repetition patterns (repeated names, code identifiers, formulaic text) that neural models miss. The n-gram cache has zero persistent storage cost (built during evaluation).

**Implementation considerations**:
- Only update the cache with tokens that have *already been scored* — never lookahead
- Use a trie or hash map keyed on the n-gram context
- Tune the blend weight `α` — typically 0.1–0.3

**Compliance note**: N-gram caching was important in some earlier phases of the challenge, but it has also been one of the most contentious areas. It is legal only if the cache is built from previously scored tokens and never from future tokens. Many implementations were rejected or heavily scrutinized for subtle lookahead bugs.

---

### Sliding Window Evaluation (stride 64)

**What it is**: Evaluate the model using overlapping windows so each token is scored with access to more context than the model's training context length.

**How it works**:
```
Context length C = 512, Stride S = 64

Window 1: tokens [0, 512)  → score tokens [0, 64)
Window 2: tokens [64, 576) → score tokens [64, 128)
Window k: tokens [(k-1)*S, (k-1)*S + C) → score tokens [(k-1)*S, k*S)
```

Each token is scored with up to `C - S = 448` tokens of context rather than just its position within a single window.

**Why it helps**: Language models improve with longer context. For text with long-range dependencies (code, formal writing, repeated entities), this can provide meaningful BPB improvements for free.

**Cost**: `C/S = 8× ` more forward passes than stride-C evaluation. Worth it when compute is available.

---

### Entropy-Adaptive TTT Epochs

**What it is**: Dynamically adjust the number of TTT adaptation steps per chunk based on the model's prediction entropy on that chunk.

**Why it helps**: High-entropy chunks (the model is uncertain) benefit from more adaptation steps. Low-entropy chunks (the model is confident) need fewer. Allocating compute adaptively extracts more value from the fixed TTT compute budget.

**Implementation**:
```python
def adaptive_ttt_steps(logits, base_steps=5, max_steps=20):
    entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1).mean()
    scale = (entropy / BASELINE_ENTROPY).clamp(0.5, 2.0)
    return int(base_steps * scale)
```

---

## Compliance Checklist

Before submitting any test-time technique, verify:

- [ ] The model only adapts on tokens that have already been scored
- [ ] The n-gram cache is only updated after a token is scored, never before
- [ ] LoRA adapters are not initialized from or trained on validation targets before scoring
- [ ] Sliding window evaluation does not "skip" the first window's context boundary
- [ ] TTT adaptation does not carry over state from one validation sequence to another in a way that leaks future information

Violations cause disqualification. When in doubt, use Score-First TTT — it's the cleanest variant.

---

## Expected Gains

| Technique | Typical BPB Improvement | Notes |
|-----------|------------------------|-------|
| Sliding window eval (stride 64) | ~0.002–0.005 | Free compute |
| N-gram caching (7-gram backoff) | Highly variable | Historically useful, but rule-sensitive |
| Score-First TTT (5–10 steps) | ~0.010–0.030 | Depends on domain shift |
| Entropy-adaptive TTT | Small addl. | On top of base TTT |
| LoRA-TTT | ~0.010–0.025 | Cheaper than full-model adaptation |
| Pre-quant TTT | Potentially very large | Major open frontier, but under active rules debate |

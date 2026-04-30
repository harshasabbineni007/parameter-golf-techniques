# Architectural Modifications

Under a hard parameter budget, architectural choices determine how efficiently you use each byte. The modifications here either increase effective capacity without adding bytes, or improve gradient flow and optimization dynamics in small models.

---

## Techniques

### Depth Recurrence / Looped Layers

**What it is**: Reuse the same transformer block multiple times per forward pass. Instead of 16 unique layers, use 8 unique blocks called twice each — achieving 16 "effective" layers with half the parameters.

**Why it helps**: Depth (number of forward passes through non-linearities) helps expressivity more than width in many tasks. Recurrence lets you trade parameter count for forward-pass depth at no extra storage cost.

**Variants**:
- **Plain looping**: call the same `nn.Module` N times
- **FiLM conditioning**: pass a loop-index embedding through a FiLM (Feature-wise Linear Modulation) layer so each pass is aware of which iteration it's on
- **Partial recurrence**: only loop the middle layers; keep input/output layers unique

**Implementation sketch** (PyTorch):
```python
class LoopedTransformer(nn.Module):
    def __init__(self, n_unique, n_loops):
        self.blocks = nn.ModuleList([Block() for _ in range(n_unique)])
        self.n_loops = n_loops

    def forward(self, x):
        for _ in range(self.n_loops):
            for block in self.blocks:
                x = block(x)
        return x
```

**Learning tip**: Start here — it's the highest-impact architectural change and conceptually simple.

**Papers**:
- [Universal Transformers](https://arxiv.org/abs/1807.03819)
- [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196)

---

### Wider/Shallower MLPs (3× expansion)

**What it is**: Change the MLP expansion ratio from the standard 4× to 3× (i.e., `d_ff = 3 * d_model` instead of `4 * d_model`).

**Why it helps**: Under a parameter budget, reducing MLP width frees up parameters to increase depth, widen attention heads, or add more layers. In many small-model regimes, 3× is a better trade-off than 4×.

**Parameter math** (per layer, ignoring biases):
- 4× MLP: `2 × d × 4d = 8d²` parameters
- 3× MLP: `2 × d × 3d = 6d²` parameters
- Savings: 25% of MLP parameters → can be reallocated elsewhere

---

### Gated Attention / SparseAttnGate / Output Gates

**What it is**: Add a small learnable gate to attention outputs, residual branches, or selected heads so the model can suppress or amplify specific pathways.

**Variants seen in Parameter Golf**:
- **SparseAttnGate**: a narrow learned gate on head outputs
- **Attention output gates**: tiny per-head or per-window modulation layers
- **Residual/skip gates**: gates on long skip paths or parallel residual branches

**Why it helps**: In tiny models, not all heads or branches are equally useful. Gates let the model focus scarce capacity where it matters most while spending very few extra bytes.

---

### XSA / XSA-all

**What it is**: A family of attention modifications used in many mid- and late-stage submissions. In the repo, you'll often see variants described as `XSA` or `XSA-all`, meaning the mechanism is applied to some or all layers.

**Why it helps**: The exact implementation varies by PR, but the recurring theme is better attention efficiency or better parameter use under a fixed artifact budget. Treat `XSA` as a repo-specific technique family rather than a single standard architecture term.

---

### SmearGate + BOS-Fixed

**What it is**: SmearGate mixes a small amount of the previous token's hidden state into the current token through a learned gate. Later submissions discovered that this can leak information across packed-document boundaries unless BOS positions are masked correctly.

**Why it helps**: The mechanism can improve local information flow with very little parameter cost. The BOS fix matters because packed validation streams otherwise let the last token of one document influence the BOS token of the next, which is both a correctness and compliance issue.

---

### Value Residual Learning (VRL) and Other Residual Tweaks

**What it is**: A grab-bag of lightweight residual modifications inside the attention stack, such as value residuals or alternative skip paths.

**Why it helps**: These tweaks can improve gradient flow in deep or recurrent models, but they are less universal than recurrence, parallel residuals, or gating. Treat them as secondary ideas rather than the core frontier recipe.

---

### Partial RoPE (Rotary Positional Embeddings)

**What it is**: Apply Rotary Position Embeddings (RoPE) only to a subset of the head dimensions (e.g., the first 25% of dimensions) rather than all dimensions.

**Why it helps**: Full RoPE applies rotation to every dimension pair, which can over-bias the model toward positional information at the expense of semantic content. Partial RoPE is a regularizer that reserves most capacity for content representation.

**Papers**:
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

---

### Parameter Tying

**What it is**: Share weights between different parts of the model. The most common form is input-output embedding tying (often baseline), but more aggressive forms include tying weights across layers.

**Why it helps**: Reduces parameter count without reducing model capacity as measured by forward-pass computation. Tied embeddings cut `vocab_size × d_model` parameters — significant when vocab is even modestly large.

---

### LeakyReLU² / Custom Activations

**What it is**: Use squared LeakyReLU (`max(x, alpha*x)²`) or other non-standard activations instead of GELU or SiLU.

**Why it helps**: In tiny models, the choice of activation affects both optimization dynamics and expressivity. Squared activations can produce sharper gates and are easy to compute. Some Parameter Golf entries found empirical improvements from this swap.

---

### RMSNormNoWeight and Other Byte-Saving Normalization Tweaks

**What it is**: Use RMSNorm but remove the learnable scale parameter (`gamma`), reducing it to a pure normalization operation.

**Why it helps**: Each learnable scale in a standard RMSNorm costs `d_model` parameters (stored at float32 or bfloat16). Removing them saves a small but non-trivial number of bytes across all layers.

**Trade-off**: The model loses the ability to rescale each dimension post-normalization. This has shown up in some compact-model experiments, but it is not a defining ingredient of the current top Parameter Golf stacks.

---

### U-Net Style Skip Connections / Parallel Residuals

**What it is**: Add long-range skip connections (like U-Net) between early and late layers, or run multiple residual paths in parallel and combine them.

**Why it helps**: Improves gradient flow in deep/recurrent setups where standard residuals degrade over many loops. Particularly valuable in looped-layer architectures.

---

## Architectural Design Philosophy

The best tiny-model architectures balance three things:

1. **Effective depth**: more passes through non-linearities → more expressive functions
2. **Capacity concentration**: put parameters where the model needs them most (attention vs. MLP, early vs. late layers)
3. **Gradient health**: ensure gradients flow cleanly through every path, especially in recurrent/deep setups

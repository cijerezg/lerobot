# Probe documentation

The exact MolmoAct2 tensor locations, equations, and dimensions used by the probes are
documented in [MODEL_TENSORS.md](MODEL_TENSORS.md). Read that note before interpreting a
plot labelled with a layer such as `L14` or `L31`.

The important distinctions are:

| Probe family | Tensor read | Shape for this checkpoint | Meaning of `Lℓ` |
|---|---|---:|---|
| `conditions_matrix`, `domain_representations` | Token-group mean of the language block output | `[B,2560]` per group | After the complete encoder block's attention, MLP, and residual updates |
| `conditions_matrix`, `domain_representations` (`action`) | Horizon mean of the action-expert block output | `[B,768]` | After the complete expert block's self-attention, cross-attention, MLP, and residual updates |
| `attention`, `attention_budget` | Action-expert softmax weights | cross `[B,8,30,S]`; self `[B,8,30,30]` | Inside the attention sublayer, before value mixing, output projection, and the residual update |
| `embodiment_swap` | Integrated normalized action chunk | `[B,30,8]` exposed by the mixed policy | No intermediate layer is read; the intervention is measured after the full model and flow sampler |

Layer numbering is zero-based. For example, `L31` is the 32nd of 36 layer pairs. Attention
weights and post-block representations are different measurements; neither should be
described as the other. Attention weights show routing, while Jacobian probes measure
forward sensitivity.

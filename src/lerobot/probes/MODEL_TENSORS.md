# MolmoAct2 probe tensor map

This note defines exactly where the MolmoAct2 probes read the model. Layer indices in
probe artifacts are **zero-based**: `L31` is the output of the 32nd block. The dimensions
below are from the local `outputs/MolmoAct2/config.json`; code should still obtain them
from the loaded model rather than hard-code them.

## One layer pair

Let \(B\) be batch size, \(S\) the packed multimodal-prefix length, \(H=30\) the action
horizon, \(d_e=2560\) the language-transformer width, and \(d_a=768\) the action-expert
width. The released model has 36 paired layers.

```text
packed RGB/depth/text/state embeddings Z^(0) [B,S,2560]
                         |
                         v
  encoder block Lℓ: self-attention + MLP + both residual additions
                         |
                         +---- post-RoPE Kℓ,Vℓ ----+
                         v                             |
               Z^(ℓ+1) [B,S,2560]                  |
                                                       v
noise trajectory X^(0) [B,30,32] -> Linear -> [B,30,768]
                                                       |
                         action-expert block Lℓ <---+
                         self-attn -> cross-attn -> MLP
                         (each with its residual update)
                                      |
                                      v
                              X^(ℓ+1) [B,30,768]
```

The encoder block is pre-norm. In abbreviated form,

\[
\widetilde Z^\ell = Z^\ell +
  \operatorname{Attn}_\ell(\operatorname{RMSNorm}(Z^\ell)),
\qquad
Z^{\ell+1} = \widetilde Z^\ell +
  \operatorname{MLP}_\ell(\operatorname{RMSNorm}(\widetilde Z^\ell)).
\]

`capture_layer_representations` installs a forward hook on the complete decoder block.
It therefore captures \(Z^{\ell+1}\), **after attention, MLP, and their residual
additions**, but before the transformer's final `ln_f`. It does not capture the raw
attention output.

For a prompt token group \(g\), with packed positions \(I_g\), the stored vector is

\[
r^\ell_g = \frac{1}{|I_g|}\sum_{i\in I_g} Z^{\ell+1}_{1,i,:}
   \in \mathbb R^{2560}.
\]

Thus one frame contributes `[36,2560]` for every present encoder group. The groups used
by `conditions_matrix` are `img_wrist_0`, `img_external_0`, `subtask`, and the single
`<action_output>` position. Camera groups contain their image-patch positions; a missing
camera is absent rather than represented by a black image.

The action expert begins from the noisy flow trajectory. At the representation probes'
fixed first flow state, \(t=0\), the same seeded Gaussian noise is used for every frame:

\[
X^0=\operatorname{Linear}(x_0),\qquad
x_0\in\mathbb R^{B\times30\times32},\quad
X^0\in\mathbb R^{B\times30\times768}.
\]

Expert block ℓ performs action-token self-attention, cross-attention to the K/V made by
encoder block ℓ, and an MLP, with a gated residual update at each sublayer. The hook is
again on the complete block, so the stored action representation is

\[
r^\ell_{\mathrm{action}}=\frac1{30}\sum_{t=1}^{30}X^{\ell+1}_{1,t,:}
  \in\mathbb R^{768}.
\]

One frame therefore contributes `[36,768]` for the action group. This mean discards which
future timestep carried a feature; probes concerned with the horizon structure must use
the unpooled attention or action outputs instead.

## Attention tensors

Attention probes read softmax weights, not block outputs. The action expert has 8 heads
of width 96. At expert layer ℓ,

\[
A^{\ell}_{\rm self}
=\operatorname{softmax}\!\left(\frac{Q^\ell_X(K^\ell_X)^\top}{\sqrt{96}}+M_X\right)
\in\mathbb R^{B\times8\times30\times30},
\]

\[
A^{\ell}_{\rm cross}
=\operatorname{softmax}\!\left(\frac{Q^\ell_X(K^\ell_Z)^\top}{\sqrt{96}}+M_Z\right)
\in\mathbb R^{B\times8\times30\times S}.
\]

Rows are action queries; columns are action keys for self-attention and packed encoder
positions for cross-attention. Image overlays and clause-budget plots are reductions of
\(A^{\ell}_{\rm cross}\). Standard visualization capture recomputes these exact softmax
weights from detached Q/K on CPU while the actual model forward retains its normal SDPA
path. Gradient/Jacobian capture instead routes the chosen layer through the explicit
softmax in the autograd graph.

The language transformer itself has 32 query heads and 8 KV heads, all of head width
128. When explicitly requested, its attention has shape
`[B,32,S_query,S]` after repeating grouped K/V heads; the probe keeps only requested text
query rows to avoid materializing every \(S\times S\) map.

Attention weight is routing evidence, not causal importance: a large entry says where a
query read, not how much changing that key would change the action. The Jacobian probes
serve the latter question.

## Output-level action interventions

After layer 35, the action expert's final norm and linear head produce the flow velocity

\[
v_\theta(x_t,t,Z)\in\mathbb R^{B\times30\times32}.
\]

Columns beyond a sample's true action layout are masked. The configured mixed policy
exposes width 8 (individual layouts use 7 or 8 valid channels), and flow integration
produces a normalized action chunk \(a\in\mathbb R^{30\times8}\).
`embodiment_swap` probes this final integrated chunk, not an intermediate layer: it holds
the frame and flow noise fixed, changes the embodiment/control-mode text, and measures
the resulting displacement of \(a\).

## Source of truth in code

- Token grouping and post-block pooling:
  `probes/adapters/molmoact2.py::_prompt_token_groups` and
  `capture_layer_representations`.
- Attention capture: `policies/molmoact2/modeling_molmoact2.py::_patched_action_attention`.
- Probe-facing attention shapes: `probes/base.py::AttentionCaptureResult`.
- The full model architecture: `policies/molmoact2/ARCHITECTURE.md`.

# Depth short-term history — temporal attention inside the patch CNN

Status: **BUILT 2026-07-25** (encoder rewrite in
[modeling_pointmap.py](../src/lerobot/policies/depth_pointmap/modeling_pointmap.py),
tests green; capacity bumped per 2026-07-25 decision — see §3). Scope: only how
past depth frames enter the depth encoder.
Everything downstream is untouched: 192 tokens out, Fourier PE, null bank,
DepthStream, and the expert read stay as in [03_depth.md](03_depth.md). (The
read was separately rebuilt as a joint softmax on 2026-07-26 —
[depth_redesign_options.md](depth_redesign_options.md); orthogonal to this doc.)

Replaces the 2026-07-21 token-concat history path, which predates the MEM video
encoder decision (2026-07-22) and violates the token-count invariant:
$(T_h{+}1)\cdot 192 = 1152$ tokens into the stream instead of 192.

## 1. The analogy

The patch CNN is depth's analog of the ViT. History enters it the same way image
history enters the ViT — temporal attention inside the encoder, at certain
layers, past frames dropped before the output:

| | top/wrist RGB | depth |
|---|---|---|
| encoder | ViT, 27 attention layers | per-patch CNN, 3 conv blocks |
| frames ride | shared ViT weights as time slices | shared CNN weights as extra batch rows |
| history enters | temporal attention, every 4th layer | temporal attention, after each block |
| temporal keys | same patch of older frames, one union softmax with spatial keys | same patch, same pixel: past frames + current frame in one softmax |
| time signal | sinusoidal $e(\Delta t)$ at each temporal layer input | same |
| past rows | dropped after last temporal layer | dropped after last fusion, before pooling |
| output | $n$ tokens — unchanged | 192 tokens — unchanged |

## 2. Mechanism

Per-frame pipeline (unchanged): back-project → patchify → recenter → CNN blocks
(channels 5→32→64→512, resolution 40→20→10→5) → average-pool → + PE + modality
embed → token.

The 5 past frames run through the identical per-frame pipeline up to the CNN, as
extra batch rows through the **same** conv weights (weight sharing across time,
as the ViT is shared across frames).

**Temporal attention after block $k$.** Feature maps $F_k^{(s)}$ of shape
$(C_k, H_k, W_k)$ for the same patch, $s = 0$ (current) and $s = -1..-5$ (1–5 s
ago). First each frame's age is stamped on its features (sinusoidal, no
parameters, broadcast over pixels — attention is permutation-invariant, so
unlike a fixed conv it must be told which frame is which):

$$x^{(s)} = F_k^{(s)} + e(\Delta t_s), \qquad e = \text{sin/cos ladder in } \mathbb R^{C_k}$$

Then, independently at every pixel $(u,v)$ of every patch, the current frame
queries all six frames at that same pixel:

$$q = W_q\, x^{(0)}(u,v), \qquad k_s = W_k\, x^{(s)}(u,v), \qquad v_s = W_v\, x^{(s)}(u,v), \qquad s = 0, -1, \dots, -5$$

$$w_s = \frac{\exp\!\big(q \cdot k_s / \sqrt{d_h}\big)}{\sum_{s'} \exp\!\big(q \cdot k_{s'} / \sqrt{d_h}\big)}, \qquad
F_k^{(0)}(u,v) \;\leftarrow\; F_k^{(0)}(u,v) + W_o \sum_s w_s\, v_s$$

followed by a pixelwise MLP on the fused current frame (two 1×1 convs, hidden
$4C_k$, SiLU — the analog of the ViT temporal block's feed-forward):

$$F_k^{(0)}(u,v) \;\leftarrow\; F_k^{(0)}(u,v) + W_2\, \mathrm{SiLU}\big(W_1\, \mathrm{GN}(F_k^{(0)}(u,v))\big)$$

with $W_q, W_k, W_v, W_o \in \mathbb R^{C_k \times C_k}$, fresh per fusion
point, standard init, split into $C_k/64$ heads of $d_h = 64$.

- **Content-dependent frame weighting** — the point of attention over a fixed
  conv: which past frame matters (the 1 s-ago one, the 4 s-ago one, none)
  varies per sample and per location, and $w_s$ is computed from the features
  themselves.
- **The current frame sits in the key set** — the analog of the RGB union
  softmax, where temporal keys compete with spatial keys in one softmax. When
  no past frame is informative, mass parks on $s{=}0$ and the update degrades
  to a self-projection instead of force-feeding history.
- **Same patch, same pixel across time** — the CNN version of the RGB temporal
  keys (patch $i$ of older frames). No re-projection, consistent with the
  standing decision (the wrist moves; the network absorbs it). Depth-axis
  ego-motion is visible through each frame's own $\bar z$ input channel.
- Past frames evolve through the shared **spatial** blocks only; they receive
  no temporal update themselves (v1 simplification of RGB's causal
  past-to-past attention — with 3 blocks there is no depth for it to matter).
- After block 3's fusion, past rows are discarded. Pooling still emits 192 tokens.
  PE uses the current centroid when valid; a fully empty current patch falls back
  to the newest valid historical centroid. The null bank is used only when history
  is masked/missing or that patch is empty in every frame.

The softmax is over 6 keys — this is cheap attention by construction.

## 3. Parameters and cost (as built)

Capacity decision 2026-07-25: depth is worth real parameters next to the 6B
backbone, so the CNN trunk was widened (hiddens (32, 64) → (128, 256),
`cnn_hidden_channels`) and each fusion is multi-head + MLP ($12C_k^2$ per
point):

| fusion point | $C_k$ | heads | map size | params |
|---|---|---|---|---|
| after block 1 | 128 | 2 | 20×20 | 0.20M |
| after block 2 | 256 | 4 | 10×10 | 0.79M |
| after block 3 | 512 | 8 | 5×5 | 3.15M |
| fusion total | | | | 4.14M |
| CNN trunk (widened) | | | | 4.75M |
| **encoder total** | | | | **≈ 9.0M** |

Compute: the CNN runs at batch ×6; the attention is 6 keys per pixel — both
negligible. Downstream cost returns to the 192-token baseline (the 1152-token
stream input disappears). Further depth capacity belongs to the α/stream
revision (where width × depth actually multiplies), not to more history params.

## 4. Edge cases (mirror RGB exactly)

- **Episode start**: buffer clamp repeat-pads the oldest frame (already the
  case). No `is_pad` masking — same "repeat-pad v1" rule as the video encoder.
- **History dropout** (the one shared draw per sample): mask the five past keys
  → the softmax collapses onto $s{=}0$, the sample computes a history-free op —
  the same masking trick the video encoder uses.
- **Missing history window** (plain offline eval, cold RTC deque): same
  masking. Replaces today's null-slot substitution.
- **Whole-modality dropout**: the mechanism remains wired and replaces assembled
  tokens with the learned null bank, but is disabled by default
  (`dropout_prob: 0.0`).
- **Empty current patch**: recover from the newest valid history. If history is
  masked/missing or every slot is empty, use the learned per-patch null token.

## 5. Touchpoints

- `modeling_pointmap.py`: `time_embed`, history slots in `null_memory`, and
  `_history_memory` (the token concat) deleted; the temporal attention modules
  added at the encoder, created only when `history_num_samples > 0`.
- Buffer/actor plumbing (`history.depth.wrist.depth` gathering, RTC deque):
  unchanged — only consumption changes.
- Freeze whitelist / depth-lr group: no change needed — the new weights live
  under `pointmap_encoder`, which is already whitelisted.
- On acceptance: update [03_depth.md](03_depth.md) (token-count claim,
  `history_num_samples` comment) and [04_memory.md](04_memory.md) §2.3 depth
  bullet; archive this file's decision into them.

## 6. Choices (resolved 2026-07-25)

1. **Fusion points**: all three blocks — block 3 is where features are abstract
   enough to tolerate retinal drift, the analog of RGB fusing into its deepest
   layers.
2. **Key neighborhood**: same pixel only (6 keys). A 3×3 neighborhood in the
   past frames (54 keys, same params) stays the fallback if inter-frame drift
   proves to matter.
3. **Heads**: multi-head everywhere, $C_k/64$ heads of dim 64 (2/4/8).

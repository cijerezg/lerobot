# Depth integration — back-projected point-map tokens + co-evolving MoT read (design)

Status: **as built + validated end-to-end** (built 2026-06-15; GPU training + on-robot
inference validated 2026-06-20). This is the live depth path for MolmoAct2. It replaces
the gripper-frame TSDF box and the static single-encode read (all the `depth_tsdf_*`
design docs were deleted as obsolete once this landed). Companion:
`depth_pointmap_build_plan.md` — the status tracker (what/where in the code, validation
log, remaining hardware track); this doc is the *why/what*. The gate-α gradient debug
(why the gate must be read param-local and the actor depth modules whitelisted) lives in
`depth_pointmap_gate_gradient.md`.

Two halves:

- **Part A — Representation.** Turn the wrist depth image into a small set of
  **patch tokens**, each carrying (a) a learned feature of the patch's **local 3D
  shape** and (b) a positional encoding of **where the patch is in 3D**. No voxel
  grid, no metric box, no extrinsic calibration. Camera frame, metric millimeters.
- **Part B — Model integration.** Those tokens feed a **co-evolving depth stream**
  (DepthVLA-style mixture-of-transformers) that the frozen action expert reads each
  layer via an **additive gated SDPA + sink**, bit-identical at init.

---

# Part A — Representation

## A.0 Notation

| symbol | meaning |
|---|---|
| $u, v$ | pixel column / row; $u\in\{0,\dots,W-1\}$, $v\in\{0,\dots,H-1\}$ |
| $W, H$ | depth image width / height (D405 raw: $640 \times 480$) |
| $f_x, f_y, c_x, c_y$ | camera intrinsics (focal lengths, principal point), in pixels |
| $D(u,v)$ | raw depth, uint16 (D405 Z16) |
| $z(u,v)$ | metric depth in mm, $z = 0.1 \cdot D$ (D405 = 0.1 mm/level) |
| $P$ | patch side in pixels (built: 40) |
| $(a,b)$ | patch index; $a\in\{0,\dots,W/P-1\}$, $b\in\{0,\dots,H/P-1\}$ |
| $N$ | number of tokens $= (W/P)(H/P)$; at $P{=}40$: $16\times12 = 192$ |
| $\Omega_{ab}$ | set of **valid** pixels in patch $(a,b)$ |
| $\bar c_{ab}$ | patch centroid (mean 3D point over $\Omega_{ab}$), in mm |
| $d_d$ | token / feature dimension (the depth stream width; built: 512) |
| $f_{ab}$ | local-shape feature of patch $(a,b)$, $\in\mathbb R^{d_d}$ |
| $g_{ab}$ | positional encoding of patch $(a,b)$, $\in\mathbb R^{d_d}$ |
| $t_{ab}$ | final token $= f_{ab} + g_{ab}$ |

Frame: everything is in the **camera's own 3D frame**, metric **mm**. No extrinsic
$T_{G\leftarrow C}$ — the wrist mount makes the camera frame gripper-relative up to a
fixed constant the network absorbs, so the extrinsic and the metric box are both
dropped (this was the realization that retired the TSDF box).

## A.1 Depth → point map (the 4-channel image)

For every pixel, back-project to its metric 3D point:

$$
Z(u,v) = z(u,v), \qquad
X(u,v) = \frac{(u - c_x)\,Z}{f_x}, \qquad
Y(u,v) = \frac{(v - c_y)\,Z}{f_y}.
$$

Validity mask:

$$
m(u,v) = \begin{cases} 1 & z(u,v) \in [z_{\min}, z_{\max}] \\ 0 & \text{otherwise (deadzone, dropout, clipped)} \end{cases}
$$

with $z_{\min}\approx 70$ mm (D405 near limit) and $z_{\max}$ a far cutoff (§A.6).

The result is a **4-channel image** $\Phi(u,v) = [\,X,\ Y,\ Z,\ m\,]$, same $H\times W$
as the depth image.

### What the 4 channels are (the common confusion)

The channels are **NOT** "(pixel coordinates) + (depth value)". They are:

- $X, Y, Z$ — the **3D metric position** of the surface point that pixel sees, in mm.
  **$Z$ *is* the depth value.** $X$ and $Y$ are the lateral metric offsets, computed
  *from* depth and pixel location. So depth is **not** stored twice — it lives once,
  as the $Z$ channel.
- $m$ — the **validity mask** (1 = real measurement, 0 = missing). Not depth.

The pixel coordinates $(u,v)$ are **not channels** — they are the position in the
image grid (implicit in where the value sits), exactly like an RGB image doesn't
carry "row/column" channels.

**Is $X,Y$ redundant?** Yes, strictly: $X,Y$ are determined by $Z$, the pixel index
$(u,v)$, and the intrinsics. We provide them anyway, on purpose — the same reason we
add positional encodings: it hands the network the metric geometry directly so it
never has to learn the back-projection (the intrinsics) internally. Helpful
redundancy, not duplicated data.

## A.2 Patchify (non-overlapping)

Partition $\Phi$ into non-overlapping $P\times P$ patches. Patch $(a,b)$ covers
pixels $u\in[aP,(a{+}1)P)$, $v\in[bP,(b{+}1)P)$. This yields $N$ patches → $N$
tokens. $P$ sets the token count.

Do **not** bilinear-downsample the depth before patchifying: averaging depth across
an object edge invents "flying pixels" between foreground and background. Keep full
resolution; the patch handles the aggregation.

Physical patch size grows with distance (same $P$ pixels): at $f_x{=}382$, a
40-pixel patch spans $40\cdot z/f_x \approx$ 7 mm at $z{=}70$ mm, ~52 mm at
$z{=}500$ mm. This is the "fine near, coarse far" property — free, from perspective.

## A.3 Per-patch shape feature $f_{ab}$

Valid pixels in the patch: $\Omega_{ab} = \{(u,v)\in \text{patch} : m(u,v)=1\}$.

Centroid (over valid pixels only):

$$
\bar c_{ab} = \frac{1}{|\Omega_{ab}|} \sum_{(u,v)\in\Omega_{ab}} \big(X(u,v),\,Y(u,v),\,Z(u,v)\big) \in \mathbb R^3.
$$

Recenter each pixel's point to the centroid (this removes *position* but **keeps
metric scale** — a near patch has small $\Delta$, a far one large $\Delta$):

$$
\Delta(u,v) = \big(X,Y,Z\big)(u,v) - \bar c_{ab}.
$$

Build a small $P\times P$ feature image with channels $[\Delta x/s,\ \Delta y/s,\
\Delta z/s,\ m]$ ($s$ a fixed scale constant, §A.6, to keep activations O(1)), and
run a **small shared 2D CNN** over it to produce the feature:

$$
f_{ab} = \mathrm{CNN}\big([\Delta x/s,\ \Delta y/s,\ \Delta z/s,\ m]\big;\ \tfrac{\bar Z_{ab}}{s}\big) \in \mathbb R^{d_d}.
$$

The CNN is **2D in structure** (dense over the patch grid, hole-friendly via $m$, no
quantization) but **3D in content** (each pixel carries its metric $\Delta$). The
optional scalar $\bar Z_{ab}$ conditions on absolute depth so the CNN can adapt to
range-dependent noise (depth error $\propto z^2$).

> Rationale for 2D-conv-over-point-map (not a 3D voxel conv): a single depth frame
> is a heightfield (one $Z$ per pixel), so the image plane is its natural dense
> domain; voxelizing scatters it into a mostly-empty grid and adds quantization,
> hole-ambiguity, and a grid-pitch dilemma — while a vanilla 3D conv still isn't
> rotation-invariant, so its headline benefit isn't even delivered. The one real 3D
> edge (not bleeding across occlusion boundaries) is mitigated by giving the 2D conv
> the $\Delta z$ jump as a visible channel (§A.6.4).

Built: reshape to $(N, 4, P, P)=(192,4,40,40)$, one shared CNN down the batch dim
→ $(192, d_d)$ (`PatchShapeCNN` / `PatchResidualBlock2d` in `modeling_pointmap.py`).
~3 stride-2 conv blocks $40\to20\to10\to5$, then global-avg-pool → $f_{ab}$.

## A.4 Positional encoding $g_{ab}$

From the **absolute** centroid $\bar c_{ab}=(\bar X,\bar Y,\bar Z)$ (NOT recentered —
this is the "where"):

$$
g_{ab} = \mathrm{MLP}\big(\mathrm{Fourier}(\bar c_{ab})\big) \in \mathbb R^{d_d}.
$$

A wavelength $\lambda_k$ is the distance over which sinusoid $k$ completes one cycle;
the encoding stacks several from fine to coarse. Bounds are **derived** from existing
params so they track the far cutoff automatically (no separate magic numbers):

- $\lambda_{\min} = P\cdot z_{\min}/f_x$ = near token spacing $\approx 40\cdot70/382
  \approx 8$ mm (finest localization; sub-patch detail lives in $f_{ab}$, not here).
- $\lambda_{\max} = 2\, z_{\max} \approx 1600$ mm (must span the scene or distant
  points alias; lateral half-extent at the cutoff is $\approx (W/2)z_{\max}/f_x
  \approx 670$ mm, so ~1.3 m scene → 1.6 m with margin).
- **Number of wavelengths: ~8–10** (the range is ~200×, $\log_2 200 \approx 7.6$
  octaves).

Division of labor: **$f_{ab}$ = local shape (recentered)**, **$g_{ab}$ = global
position (absolute centroid)**. Fine within-patch location lives in $f$; coarse
where-is-this lives in $g$.

## A.5 Token assembly + empty patches

$$
t_{ab} = f_{ab} + g_{ab}.
$$

If $\Omega_{ab} = \varnothing$ (patch entirely invalid — all deadzone/dropout), the
centroid is undefined. Do **not** drop the token (variable count breaks batching);
substitute a **learned null token** (the `null_tokens` / modality-dropout bank).
Partially-valid patches use the centroid over $\Omega_{ab}$ and let the mask channel
inform the CNN.

The static sequence $\{t_{ab}\}_{a,b}$ (size $N$) feeds the co-evolving depth stream
(Part B).

## A.6 Details / edge cases

- **A.6.1 Holes.** Mask channel $m$ flows through the CNN. Empty patch → null token
  (§A.5). No inpainting; the network sees where data is missing and can ignore it.
- **A.6.2 Far cutoff $z_{\max}$** — the only "extent" parameter left. D405 returns
  noisy depth out to meters; most far returns are irrelevant background. Clip at
  $z_{\max}$: pixels beyond → $m=0$. One scalar soft far-plane, not a measured 3D
  box; it also bounds the PE range ($\lambda_{\max}=2z_{\max}$, §A.4). Config param,
  **default $z_{\max}=800$ mm**, tune in practice.
- **A.6.3 Coordinate scaling $s$.** Recentered $\Delta$ is O(1–40) mm; pick a fixed
  $s$ (built default `coord_scale_mm=25`) so CNN inputs are O(1). Absolute centroid
  for the PE is **not** scaled (Fourier handles the range via $\lambda$). Consistent
  with the "raw metric depth, no [0,1] normalize" rule.
- **A.6.4 Edge bleeding within a patch.** A $P\times P$ patch can straddle an object
  rim (foreground + background in one patch). The CNN sees the $\Delta z$ jump and
  can learn to separate them, but this is the one soft spot of an image-plane
  representation. Telemetry probe later (does it blur grasp edges?); not a blocker.
- **A.6.5 Single frame, single camera.** No temporal channel, no fusion, no
  extrinsic. If multi-camera or temporal history is ever needed, that is the only
  thing that pulls pose/extrinsic back in.
- **A.6.6 Independence from RGB tower resolution.** Depth tokenization is
  independent of the VLM's RGB input size (MolmoAct2 resizes RGB to 378×378). We
  patchify the full-res 640×480 depth and emit our own $N$ depth tokens; the two
  grids need not align. Depth and RGB interact via **attention** in the MoT stream
  (depth attends the wrist-cam tokens), not by per-patch gluing.
- **A.6.7 Depth↔wrist-cam positional alignment** — do nothing (option a). The RGB
  tokens come from the 378×378-resized image, so there's no naive pixel
  correspondence. Depth tokens keep their 3D PE; the wrist-cam RGB tokens keep
  theirs; the depth→wrist-cam cross-attention learns whatever association it can from
  content. Fallback if semantics aren't picked up: (b) back-project RGB patch centers
  with the same intrinsics + per-patch median depth and give the RGB keys the same
  Fourier 3D PE, so both share one metric frame — a later upgrade, not now.

## A.7 Locked defaults vs. open knobs

**Locked:** camera frame, metric mm; 4-channel point map $[X,Y,Z,m]$; non-overlap
patches; recenter-for-shape ($f$) / absolute-centroid-for-position ($g$); learned
null for empty patches; 2D point-map CNN (not 3D voxel conv); single frame / single
camera / no extrinsic. **$P{=}40 \Rightarrow N{=}192$**; far cutoff default 800 mm;
derived $\lambda$ bounds (~8–10 wavelengths); per-patch CNN (depth-only).

**Open (experiment knobs):** depth stream width $d_d$ (built 512), co-evolving
layers $M$ (built $M{=}L$); the depth↔wrist-cam fallback (b) if alignment matters.

---

# Part B — Model integration (the co-evolving stream + the read)

## B.1 The problem this fixes

A static single encode (one `forward`, the same tokens read by every action-expert
layer) commits to one abstraction level. Early action-expert layers want fine
geometric detail; late layers want semantic/relational geometry. The fix the
literature converges on (DepthVLA) is to let the depth representation **evolve
through the stack** so that at layer ℓ the action tokens attend to depth features
*as they exist at layer ℓ*.

## B.2 Reference: DepthVLA

[DepthVLA, arXiv:2510.13375] — MoT of three experts (VLM, depth, action) sharing the
**same attention layers** with distinct weights. Block-wise mask: VLM and depth
tokens attend only to themselves; action tokens attend to all streams. Not frozen;
keeps a depth-prediction aux loss. Headline ablation: depth-expert pretraining is
essential — **51.0% → 74.8%**.

Three departures for our setup:
1. **We have metric geometry** (D405 wrist depth). Our depth "expert" is **not** an
   RGB→depth estimator — its tokenizer is the **point-map encoder** of Part A (a 2D
   CNN over the back-projected point map), not a voxel transformer.
2. **We keep MolmoAct2 frozen.** A zero-init gated residual (PVI, arXiv:2603.12772)
   makes the policy bit-identical at step 0; training opens the gate.
3. **No recon aux (for now).** DepthVLA's lesson is that a co-evolving stream wants a
   geometric objective. We considered masked-depth reconstruction (A.6.2-style MAE)
   but **dropped it** — flow loss only. The α=0 "deadlock" turned out soft (§B.5), so
   flow-only training is viable; revisit recon if the stream fails to learn geometry.

## B.3 Attention coupling (LOCKED)

Per shared layer, the block mask is:

| query \ keys | VLM | wrist-cam (⊂ VLM) | depth | action |
|---|---|---|---|---|
| VLM    | ✓ (frozen causal) | ✓ | ✗ | ✗ |
| depth  | ✗ | **✓** | ✓ | ✗ |
| action | ✓ | ✓ | ✓ | ✓ |

The depth stream attends **itself + the wrist-camera tokens only** — nothing else.
Rationale: the point map is back-projected from the wrist D405, so letting depth
attend the *same physical viewpoint's* RGB tokens ties the camera's semantics to the
geometry, without inflating the attention budget. Depth does **not** attend the top
camera, language, OBS_STATE, or action.

Two consequences:
- **The depth stream is a pure function of the observation.** The wrist-cam tokens
  are part of the frozen observation prefix (fixed across denoising steps), so the
  whole depth stream is **computed once per observation and cached across all
  denoising steps**. Action tokens read the cached depth K/V each step.
- **Implementation.** We need the index range of the wrist-camera tokens inside the
  VLM sequence (`wrist_cam_token_indices` / `gather_kv_at_indices`,
  `_pointmap_wrist_meta` resolves image_patch_id / num_images / cam_index) and the
  VLM per-layer K/V so depth at layer ℓ attends VLM hidden states at layer ℓ.

## B.4 The read — additive gated SDPA + sink (DECIDED 2026-06-14, built)

The action expert reads the (co-evolving) depth state with **two SDPA calls, added**:

$$
o_i = \mathrm{SDPA}(q_i, K_{ctx,\ell}, V_{ctx,\ell}, M_{ctx})\;+\;\tanh(\alpha_\ell)\cdot \mathrm{SDPA}\big(q_i,\,[K_{d,\ell}, k_\star],\,[V_{d,\ell}, 0]\big).
$$

- At init $\tanh(\alpha_\ell)=0$ ⇒ the second term is exactly 0 ⇒ output is
  **bitwise** the context read. This is the frozen-policy guardrail, checked by
  `probes/pointmap_bit_identity.py`.
- The **sink** is one extra key column inside the *depth* softmax with zero value
  and a learned per-(layer,head) logit injected as an additive attention bias
  (constant in the query). It restores **absolute** abstaining (everything scores
  low → mass parks on the sink → read → 0) without leaving SDPA. ($k_\star$ removed
  recovers the plain depth softmax, which gives *relative* abstaining for free.)
- Both calls are static-shape ⇒ compile-friendly.

**Why NOT a single joint softmax.** The tempting alternative — action attends
$[ctx \cup depth \cup sink]$ in *one* SDPA with depth values scaled by
$\tanh(\alpha_\ell)$ — is **not bit-identical at gate 0**: in a joint softmax the
depth + sink keys absorb attention mass even when their *values* are zeroed, so at
init

$$o_i = (\text{context read}) \times \tfrac{Z_{ctx}}{Z_{ctx}+Z_{depth}+Z_\star} \neq \text{context read},$$

which fails the gate-0 probe. The additive form is the price for exact bit-identity;
the depth stream's co-evolution is unchanged either way.

Code: `gated_depth_read` (`modeling_stream.py`); patched into the action expert by
`_patch_action_expert_pointmap_read` (`modeling_molmoact2.py`), adding
$\tanh(\alpha_\ell)\cdot\texttt{gated\_depth\_read}(\dots)$ to the context read in
`cross_attn_forward`. Training threads `depth_state` through `run_layer` (one stream
block per layer); inference runs the whole stream once in `prepare_context` and
caches per-layer K/V. Stream runs float32 (autocast off) at batch B; read K/V cast
to the action dtype and expanded to the flow-timestep batch at the read site.

Implementation constraint (learned the hard way, 2026-06-20): the per-layer gate
$\tanh(\alpha_\ell)$ must be computed **inside** `run_layer` directly from the
parameter (`torch.tanh(self.depth_stream.gate[idx])`), exactly like `sink_logit` — not
precomputed once outside and captured by closure. `run_layer` is wrapped in gradient
checkpointing; a non-leaf intermediate captured by closure can lose its gradient back to
$\alpha$, which silently pins the gate at 0. Reading the parameter in-region keeps the
whole path ($\alpha \to \tanh \to$ multiply $\to$ read) inside the recomputed region.
Full debug story: `depth_pointmap_gate_gradient.md`.

## B.5 Frozen-policy preservation + the soft "deadlock"

Per-layer $\alpha_\ell$ with $\tanh(\alpha_\ell)=0$ at init scales the **additive**
depth-read term, so step 0 is bitwise the frozen policy. $\alpha_\ell$ is kept for
the per-layer amplitude ceiling ($|\cdot|<1$): depth can't exceed context, free
insurance on a frozen policy.

The α=0 "deadlock" is **soft**: $\alpha_\ell$ is trainable from step 0, since
$\partial/\partial\alpha\,[\tanh(\alpha)\cdot r] = r \neq 0$ at $\alpha=0$. Only the
stream *content* params sit at zero gradient until α moves off 0, then they
co-adapt. This is why flow-loss-only training is viable and the recon aux was
dropped (§B.2).

Because the gate is the *only* depth param with a nonzero gradient at init, the whole
stream is dead if the gate doesn't learn — so the actor's `pointmap_encoder`/
`depth_stream` params **must be trainable**. They are fresh from-scratch modules whose
names match none of `_apply_actor_freeze`'s patterns, so without an explicit
always-trainable branch they hit the unrecognized-param `else` and get frozen (this bit
us 2026-06-20; the critic already had the mirror branch). Verify uptake with
`pointmap_gate_absmax`, not just the mean (the mean can hide signed per-layer movement).

## B.6 Stream module + sizing

`DepthStream` (`modeling_stream.py`): $M$ light blocks (fresh float32, **not** in
the frozen `action_expert` state_dict). Each block = depth self-attn + cross-attn to
wrist-cam K/V + MLP. `forward` co-evolves the $N$ tokens → per-layer states;
`read_kv` projects a state into action-head space; owns $\alpha_\ell$ + per-(layer,
head) `sink_logit`. Config knobs: `stream_width` ($d_d{=}512$), `stream_num_heads`
(8), `stream_layers` (None ⇒ $M{=}L$), `stream_mlp_ratio`.

Cost is **not** a constraint: 192 depth tokens at a small width, run once per
observation (cached across denoising steps), <~1% of forward. The only per-step cost
is action→depth keys, a rounding error vs action self-attention and the frozen VLM
prefix. So the open decisions are **capacity / inductive bias, not FLOPs** ($d_d$ is
the real knob; $M{=}L$ by default, prune later via $\alpha_\ell$ telemetry).

## B.7 CUDA graphs / compile — PARKED (moot under RTC)

Graphs are force-OFF whenever `pointmap_config` is set. Originally the motivation for
moving to SDPA was to regain CUDA-graph capture (the old TSDF read was a manual
einsum→logsumexp path that broke it). Investigated 2026-06-14 and **deprioritized**:
our config runs **RTC** (`rtc_config.enabled: true`), whose denoise loop is eager and
never touches the action graph, so graphs-off costs nothing; the graph only speeds
the action loop (marginal vs the un-graphed VLM prefix); and early-training gate≈0 ⇒
depth≈0 ⇒ moot. If revisited (non-RTC inference + a measured action-loop
bottleneck): the checkpoint's `_clone_static_context` / `_copy_context_` copy a
hard-coded field list and don't know our depth K/V, so a replay would silently drop
the depth read — fix by registering depth K/V as static inputs, and add a gate-OPEN
eager-vs-graph test (the gate-0 probe can't catch depth staleness).

## B.8 Critic read (DONE 2026-06-14; GPU-validated 2026-06-20)

The critic **owns its own** depth modules (not shared with the actor) and consumes
depth as a co-evolving stream, adapted to the critic's architecture (bidirectional,
single end-of-stack read — no gate/sink, since the critic isn't frozen):
`MolmoAct2Critic` (`rl_molmoact2.py`) gets `depth_encoder` + `depth_blocks`
(`DepthStreamBlock` × `critic_llm_depth`, attending the critic's wrist-cam obs
tokens) + `depth_read_proj`; the co-evolved **final** state is appended to the
sequence `[obs | depth | value-queries]`. V(s) live, V(s′) on EMA `next_depth`.
Forks resolved: own encoder ⇒ TD grad isolated from the actor; depth modules are
critic params ⇒ ride `critic_target`'s generic EMA lerp. Trainer gotcha:
`_apply_critic_freeze`'s else-branch would freeze unknown params, so `depth_*` is
added to the always-trainable branch.

dtype note: the critic runs **bf16** while the actor's stream runs float32, so the
**same** `DepthPointmapEncoder` must be dtype-agnostic. `fourier_position_encoding`
intentionally computes its sin/cos ladder in float32 (precision over the mm range) and
the encoder casts `pe` to the projection weight dtype before `pos_proj` — without that
cast the bf16 critic crashes at `pos_proj` (`Float vs BFloat16`). This surfaced on the
first GPU train step (the bit-identity probe only exercises the float32 actor inference
path, so it could not have caught it).

## B.9 Sources

- DepthVLA — arXiv:2510.13375 (MoT depth expert; pretraining ablation; aux loss)
- PVI — arXiv:2603.12772 (zero-init residual injection into a frozen VLA)
- SpatialVLA — arXiv:2501.15830 (Ego3D PE: egocentric camera-frame back-projection,
  calibration-free — validates the no-extrinsic point-map choice)
- PointACT — arXiv:2605.21414 (hierarchical 3D, multi-scale point-action interaction)
- GST-VLA — arXiv:2603.09079 (depth → 3D primitives tokenizer)
- 3DThinkVLA — arXiv:2606.04436 (3D geometry at different feature hierarchies)
- GeoAlign — arXiv:2606.03240 (geometry queried by proprioceptive state)
- Flamingo (gated XATTN-DENSE) — per-layer tanh(α) zero-init gating precedent

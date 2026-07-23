# 03 — Depth: point-map tokens + co-evolving stream

Status: **built and validated end-to-end** (built 2026-06-15; GPU training +
on-robot inference validated 2026-06-20). This is the live and only depth path —
it replaced the earlier gripper-frame TSDF box entirely. Camera intrinsics are now
the D405 factory calibration (the 2026-06 "placeholder K" caveat is resolved).

Code: [`policies/depth_pointmap/`](../src/lerobot/policies/depth_pointmap/)
(`configuration_pointmap.py`, `modeling_pointmap.py`, `modeling_stream.py`) plus
the read patch in
[`modeling_molmoact2.py:655`](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L655).
Everything is gated on `pointmap_config` (`None` = depth-free, zero cost).

Two halves: **A** — turn one wrist depth frame into 192 metric patch tokens.
**B** — co-evolve those tokens through a DepthStream and let the frozen action
expert read them per layer through a zero-init gate.

---

## Part A — Representation

### A.1 Notation and frame

| symbol | meaning |
|---|---|
| $u, v$ | pixel column / row; image is $W \times H = 640 \times 480$ |
| $f_x, f_y, c_x, c_y$ | intrinsics in pixels (factory calibration: 394.98, 394.98, 322.56, 238.70) |
| $D(u,v)$ | raw depth, uint16 (D405 Z16); $z = 0.1 \cdot D$ mm (`depth_units_mm`) |
| $P$ | patch side = 40 px → $N = (640/40)(480/40) = 16 \times 12 = 192$ tokens |
| $d_d$ | depth stream width = 512 (`stream_width`) |

Everything is in the **camera's own 3D frame, metric millimeters**. No extrinsic:
the wrist mount makes the camera frame gripper-relative up to a constant the
network absorbs — this realization is what retired the TSDF box and its
calibration burden.

### A.2 Back-projection → 4-channel point map

`back_project` ([modeling_pointmap.py:32](../src/lerobot/policies/depth_pointmap/modeling_pointmap.py#L32)).
Per pixel:

$$Z = z(u,v), \qquad X = \frac{(u - c_x)Z}{f_x}, \qquad Y = \frac{(v - c_y)Z}{f_y}$$

$$m(u,v) = \mathbb 1\big[z(u,v) \in [z_{\min}, z_{\max}]\big], \qquad z_{\min} = 70\text{ mm},\ z_{\max} = 800\text{ mm}$$

giving $\Phi = [X, Y, Z, m] \in \mathbb R^{4 \times H \times W}$; invalid pixels
have $X{=}Y{=}Z{=}0$. The channels are the **3D metric position of the surface
point each pixel sees** plus a validity mask — not "pixel coords + depth". $X,Y$
are strictly redundant given $Z$ and the intrinsics; we provide them anyway so the
network never has to learn the back-projection internally (helpful redundancy,
same spirit as positional encodings). $z_{\max}$ is the single remaining "extent"
parameter (soft far-plane; also bounds the PE wavelengths).

### A.3 Patchify + per-patch shape feature

$\Phi$ is cut into non-overlapping $P \times P$ patches (no pre-downsampling —
bilinear-averaging depth across an object edge invents flying pixels). For patch
$(a,b)$ with valid-pixel set $\Omega_{ab}$:

$$\bar c_{ab} = \frac{1}{|\Omega_{ab}|}\sum_{(u,v) \in \Omega_{ab}} (X, Y, Z)(u,v), \qquad \Delta(u,v) = (X,Y,Z)(u,v) - \bar c_{ab}$$

Recentering removes *position* but keeps *metric scale*. The shape feature is a
small shared 2D CNN over the recentered patch:

$$f_{ab} = \mathrm{CNN}\big([\Delta_x/s,\ \Delta_y/s,\ \Delta_z/s,\ m,\ \bar Z_{ab}/s]\big) \in \mathbb R^{d_d}$$

with $s = 25$ mm (`coord_scale_mm`, keeps CNN inputs O(1)) and the constant
$\bar Z$ channel conditioning on absolute range (depth noise $\propto z^2$;
`include_centroid_depth: true`). Implementation: reshape to $(B{\cdot}192, 5, 40, 40)$
and run `PatchShapeCNN` — three stride-2 GroupNorm/SiLU residual blocks
(40→20→10→5), global-average-pool to $d_d$
([modeling_pointmap.py:103-136](../src/lerobot/policies/depth_pointmap/modeling_pointmap.py#L103-L136)).

The CNN is 2D in *structure* (dense image plane, hole-friendly via the mask, no
quantization) but 3D in *content* (every pixel carries its metric offset). A 3D
voxel conv was considered and rejected: a single depth frame is a heightfield, so
voxelizing scatters it into a mostly-empty grid, adds quantization and a grid-pitch
dilemma, and still isn't rotation-invariant. The one real 3D edge — not bleeding
across occlusion boundaries — is mitigated because the CNN sees the $\Delta_z$ jump.

### A.4 Position encoding and token assembly

The **absolute** centroid (the "where", not recentered) gets a Fourier ladder +
MLP:

$$g_{ab} = W_g\,\mathrm{Fourier}(\bar c_{ab}), \qquad \mathrm{Fourier}(x) = \big[\sin(2\pi x / \lambda_k),\ \cos(2\pi x / \lambda_k)\big]_{k=1}^{8}$$

with wavelengths geometrically spaced between bounds **derived** from existing
parameters (they track the far cutoff automatically):
$\lambda_{\min} = P\, z_{\min}/f_x \approx 7$ mm (near token spacing),
$\lambda_{\max} = 2 z_{\max} = 1600$ mm (must span the scene or distant points
alias). Final token:

$$t_{ab} = f_{ab} + g_{ab} + e_{mod}$$

($e_{mod}$ a learned modality embedding). Division of labor: $f$ = local shape
(recentered), $g$ = global position (absolute). **Empty patches** (all pixels
invalid) are replaced by a learned per-patch `null_tokens` bank — never dropped
(variable count breaks batching). The same bank substitutes the whole sample under
modality dropout (`dropout_prob: 0.25`, train only) and when depth is missing.

Depth tokenization is independent of the RGB tower's 378×378 resize; the two grids
never align pixel-wise. Depth and RGB meet through **attention** (Part B), not
per-patch gluing.

---

## Part B — Model integration

### B.1 Why co-evolving

A static single encode commits to one abstraction level, but early action-expert
layers want fine geometry and late layers want relational/semantic geometry. The
fix (DepthVLA, arXiv 2510.13375) is a mixture-of-transformers: the depth
representation evolves through the stack so layer ℓ of the action expert reads
depth *as it exists at layer ℓ*. Our departures from DepthVLA: (1) we have real
metric depth, so the tokenizer is the Part-A encoder, not an RGB→depth estimator;
(2) MolmoAct2 stays frozen — a zero-init gate makes step 0 bit-identical;
(3) no reconstruction aux loss — flow loss only (dropped 2026-06-14; viable
because the gate deadlock is soft, §B.5).

### B.2 Attention coupling (LOCKED)

| query \ keys | VLM | wrist-cam (⊂ VLM) | depth | action |
|---|---|---|---|---|
| VLM    | ✓ (frozen: causal text, bidirectional image block — [02 §2](02_base_model.md)) | ✓ | ✗ | ✗ |
| depth  | ✗ | ✓ | ✓ | ✗ |
| action | ✓ | ✓ | ✓ | ✓ |

The depth stream attends **itself + the RGB tokens of the camera that produced
the depth** (`depth_key` → `cam_index`; wrist, since the D405 is wrist-mounted) —
nothing else (no other cameras, language, state, or action). Nothing about
"wrist" is special: depth and that camera's RGB share one optical viewpoint, so
tying semantics to geometry is content-matching within a single view, no
extrinsic. Other cameras are different poses — attending them would reintroduce
the cross-view correspondence/calibration problem the point-map design removed
(and inflate the attention budget). Inside the VLM itself the cameras are
attention-symmetric (bidirectional image block); the depth stream is the only
camera-selective consumer. Consequence: the stream is a **pure function of the
observation**, so at inference it runs once per control step and its K/V are
cached across all denoising steps.

`DepthStream` ([modeling_stream.py:191](../src/lerobot/policies/depth_pointmap/modeling_stream.py#L191)):
$M$ light pre-norm blocks (fresh float32, default $M = L$, one per action-expert
layer), each

```python
def forward(self, t, wrist_k, wrist_v):          # DepthStreamBlock
    h = self.norm_self(t)
    t = t + self.self_attn(h, h, h)              # depth tokens mix
    t = t + self.cross_attn(self.norm_cross(t), wrist_k, wrist_v)   # attend wrist-cam KV at layer ℓ
    t = t + self.mlp(self.norm_mlp(t))
    return t
```

The stream owns the per-layer gate `α` and per-(layer, head) `sink_logit`.

### B.2.1 The wrist-cam bridge and the read projections, explicitly

**Slicing** (`wrist_cam_token_indices` / `gather_kv_at_indices`,
[modeling_stream.py:72-124](../src/lerobot/policies/depth_pointmap/modeling_stream.py#L72-L124)).
The VLM prefix contains image-patch tokens (`input_ids == image_patch_id`) as
`num_images` equal contiguous runs in `image_keys` order; `_pointmap_wrist_meta`
resolves (image_patch_id, num_images, cam_index). Per row $b$, with $S_b$ the
sorted patch-token positions, $T_w = T_{img}/\text{num\_images}$, and depth camera
index $c$:

$$K^w_\ell[b,i] = K^{LLM}_\ell\big[b,\ S_b[cT_w + i]\big], \qquad V^w_\ell[b,i] = V^{LLM}_\ell\big[b,\ S_b[cT_w + i]\big] \in \mathbb R^{d_{vlm}}$$

Recovered per-row (padding-robust; only assumption: equal patch-token counts
across rows). Note these are the VLM's **cached post-RoPE K/V tensors**, not
residual-stream hidden states (the critic path gathers `inputs_embeds` instead).

**Stream cross-attention.** Block $\ell$ projects them with its own
$W_k, W_v \in \mathbb R^{d_d \times d_{vlm}}$:

$$t \mathrel{+}= \mathrm{Attn}\big(Q = W_q\,\mathrm{LN}(t),\ K = W_k K^w_\ell,\ V = W_v V^w_\ell\big)$$

**Read projections** (`read_kv`). One **shared** pair
$W^r_K, W^r_V \in \mathbb R^{d_{act} \times d_d}$ ($d_{act} = H D_h$, the action
expert's head space), reused at every layer:

$$K_{d,\ell} = \mathrm{reshape}_{H \times D_h}(W^r_K D_\ell), \qquad V_{d,\ell} = \mathrm{reshape}_{H \times D_h}(W^r_V D_\ell)$$

with $D_\ell$ the depth state after block $\ell$; keys then pass the expert
block's per-head cross-attn `k_norm`. This mirrors the action expert's own single
shared `context_k_proj`/`context_v_proj`: **layer-specificity comes only from
$D_\ell$, never from the projection**.

**The read with the sink weight**, per layer $\ell$, head $h$, query $q$. The sink
is one extra column with zero key, zero value, and learned logit
$\beta_{\ell h}$ = `sink_logit[ℓ,h]` (zero-init) as an additive bias:

$$s_j = \frac{q \cdot k_{d,\ell,j}}{\sqrt{D_h}}\ (j = 1..N), \qquad s_\star = \beta_{\ell h}, \qquad
r = \frac{\sum_j e^{s_j}\, v_{d,\ell,j}}{\sum_j e^{s_j} + e^{\beta_{\ell h}}}$$

$$o = \mathrm{SDPA}(q, K_{ctx,\ell}, V_{ctx,\ell}) + \tanh(\alpha_\ell)\, r$$

The zero value means the sink contributes nothing to the numerator — it only eats
normalization mass. All $s_j \ll \beta_{\ell h}$ ⇒ $r \to 0$ (absolute
abstaining); without the sink the softmax sums to 1 over depth tokens and can only
abstain relatively. $\beta_{\ell h}$ is constant in $q$: one global abstain
threshold per (layer, head).

#### TO REVISE (flagged 2026-07-22, decide later)

- **Shared read projections**: no per-layer $W^r_K, W^r_V$ — is one $d_d \to
  d_{act}$ map enough for early-fine vs late-semantic depth states, or should the
  read be per-layer (cost: $2 L d_d d_{act}$ params)?
- **Query-independent sink**: the abstain threshold $\beta_{\ell h}$ doesn't
  depend on $q$ (zero sink key). Alternative: learn $k_\star$ so
  $s_\star = q \cdot k_\star/\sqrt{D_h} + \beta_{\ell h}$ — per-query abstaining.
- **Post-RoPE K/V as the bridge input**: the stream re-projects the VLM's cached
  keys/values (position phases baked in) rather than layer hidden states —
  convenient (already cached for the expert) but not obviously the right feature
  space for the depth cross-attention.

### B.3 The read — additive gated SDPA + sink

Decided 2026-06-14, patched into every block's cross-attention
([modeling_molmoact2.py:681-696](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L681-L696)):

```python
out = self._attention(q, k, v, attn_mask=attn_mask)        # frozen context read
if depth_kv is not None:
    out = out + depth_gate.to(out.dtype) * gated_depth_read(q, depth_kv, depth_sink)
```

$$o = \mathrm{SDPA}(q,\ K_{ctx,\ell},\ V_{ctx,\ell})\ +\ \tanh(\alpha_\ell)\cdot \mathrm{SDPA}\big(q,\ [K_{d,\ell},\ k_\star],\ [V_{d,\ell},\ 0]\big)$$

- **Bit-identity at init:** $\tanh(0) = 0$ makes the second term exactly zero, so
  step 0 is bitwise the frozen depth-free policy. Guardrail:
  `probes/pointmap_bit_identity.py`.
- **The sink** $k_\star$ is one extra key column *inside the depth softmax* with a
  zero key, zero value, and a learned per-(layer, head) logit injected as an
  additive attention bias ([modeling_stream.py:44-69](../src/lerobot/policies/depth_pointmap/modeling_stream.py#L44-L69)).
  When no depth token scores above the sink bar, the softmax mass parks on the
  sink and the read → 0: **absolute abstaining** without leaving SDPA.
- **Why not one joint softmax** over $[ctx \cup depth \cup sink]$ with scaled depth
  values: in a joint softmax the depth/sink *keys* absorb probability mass even
  when their values are zeroed, so at init
  $o = (\text{context read}) \cdot \frac{Z_{ctx}}{Z_{ctx} + Z_{d} + Z_\star} \neq \text{context read}$
  — it fails the gate-0 probe. The additive form is the price of exact
  bit-identity. Both calls are static-shape → compile-friendly.
- $|\tanh| < 1$ doubles as a per-layer amplitude ceiling: depth can never exceed
  the context read. Free insurance on a frozen policy.

Wiring: training threads `depth_state` through the layer loop in
`_compute_flow_matching_loss_joint_per_layer` (one stream block per layer, stream
in float32 with autocast off, read K/V cast to the action dtype and expanded to the
flow-timestep batch at the read site). Inference runs the whole stream inside the
patched `prepare_context` ([modeling_molmoact2.py:746-774](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L746-L774))
and caches per-layer depth K/V + gate + sink on the context; the policy hands the
encoder tokens over via `action_expert._lerobot_pointmap = (init_tokens, wrist_sel)`
once per control step.

### B.4 The soft deadlock, and the two bugs that hid it

At init, differentiate the depth term $\tanh(\alpha_\ell)\, r_\ell$:

$$\frac{\partial \mathcal L}{\partial \alpha_\ell} = \Big\langle \frac{\partial \mathcal L}{\partial o},\ r_\ell \Big\rangle (1 - \tanh^2 \alpha_\ell) \Big|_{\alpha_\ell = 0} = \Big\langle \frac{\partial \mathcal L}{\partial o},\ r_\ell \Big\rangle \neq 0,
\qquad \frac{\partial \mathcal L}{\partial \theta_{stream}} = \tanh(0) \cdot (\cdots) = 0.$$

The gate is the **only** depth parameter with nonzero gradient at init; the stream
content learns nothing until α lifts off 0, then they co-adapt. This is why
flow-loss-only training works (no recon aux needed) — and why anything that
silences the gate kills the entire stream. Two real incidents (2026-06-20):

1. **Actor freeze else-branch.** With `trainable_params` set, `_apply_actor_freeze`
   freezes any unrecognized parameter name. `pointmap_encoder.*` / `depth_stream.*`
   matched no pattern → frozen, gate included; `pointmap_gate` sat at 0.0000 while
   the flow loss fell. Fix: an explicit always-trainable branch
   ([rl_molmoact2_trainer.py:287](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L287)),
   mirroring the branch the critic already had. **Rule: every new from-scratch
   module must be whitelisted in both freeze functions.**
2. **Gradient checkpointing + closures.** The per-layer gate must be read
   **in-region** inside `run_layer` directly from the parameter
   (`torch.tanh(self.depth_stream.gate[idx])`), like `sink_logit` — a non-leaf
   captured by closure into a checkpointed region can lose its gradient path.
   (Adopted as a safety invariant; the actual 2026-06-20 culprit was bug 1.)

Telemetry: watch `pointmap_gate_absmax`, not the mean — the mean can hide signed
per-layer movement. Healthy start: gate grad ~1e-3 at step 0, absmax climbing.

### B.5 dtype boundary

The stream and encoder run float32; the critic runs bf16 and **shares the encoder
class**. `fourier_position_encoding` intentionally computes its sin/cos ladder in
float32 (precision over the mm range) and the encoder casts the PE to the
projection weight dtype before `pos_proj` — without that cast the bf16 critic
crashes at `pos_proj`. This surfaced on the first GPU train step; the bit-identity
probe only exercises the float32 actor path and cannot catch it.

### B.6 Critic depth read

The critic **owns its own** depth modules (not shared with the actor — isolates the
TD gradient) and consumes depth adapted to its architecture: bidirectional, single
end-of-stack read, no gate/sink (the critic isn't frozen).
`MolmoAct2Critic.compute_depth_tokens`
([rl_molmoact2.py:326](../src/lerobot/rl/molmoact2/rl_molmoact2.py#L326)):
encoder → its own `DepthStreamBlock × critic_llm_depth` attending the critic's
wrist-cam obs embeds → `depth_read_proj` → the final state is appended to the
critic sequence `[obs | depth | value-queries]`. V(s) uses the live modules,
V(s′) the EMA copy (depth modules are critic parameters, so they ride
`critic_target`'s generic lerp). `_apply_critic_freeze` keeps `depth_*` trainable.

### B.7 Cost and parked work

192 tokens at width 512, run once per observation: <1% of the forward; the only
per-step cost is action→depth keys — negligible. Capacity ($d_d$), not FLOPs, is
the real knob; prune $M$ later via per-layer α telemetry. **CUDA graphs are
force-OFF** when `pointmap_config` is set — parked deliberately: our config runs
RTC, whose denoise loop is eager and never touches the action graph, so graphs-off
costs nothing. If ever revisited (non-RTC + measured action-loop bottleneck): the
checkpoint's `_clone_static_context`/`_copy_context_` don't know the depth K/V
fields, so a graph replay would silently drop the depth read — register them as
static inputs and add a gate-OPEN eager-vs-graph test first.

### B.8 Sources

DepthVLA (2510.13375, MoT depth expert), PVI (2603.12772, zero-init residual into
frozen VLA), SpatialVLA (2501.15830, calibration-free egocentric back-projection),
Flamingo (per-layer tanh gate precedent); also PointACT 2605.21414, GST-VLA
2603.09079, 3DThinkVLA 2606.04436, GeoAlign 2606.03240.

### B.9 Hardware checklist for a meaningful run

- [x] Wrist mount + calibrated intrinsics (factory K in config since 2026-07-02)
- [ ] `z_max_mm` sanity on real wrist-to-object range (the 2026-06 smoke, taken
      while the D405 was still top-mounted, saw median ≈ 5.5 m vs cutoff 800 mm →
      most patches null; re-check on wrist-mounted recordings)
- [x] Raw depth end-to-end: uint16 Z16, PNG16 sidecars, no hole-fill in recordings
      (masking lives in the encoder), `depth_units_mm = 0.1`
- [ ] Train long enough for the gate to grow so depth measurably shapes behavior

# 03 — Depth: point-map tokens + co-evolving stream

Status: **built end-to-end** (Part A 2026-06-15; GPU training + on-robot
inference validated 2026-06-20; history fusion rebuilt 2026-07-25; the read
rebuilt as a joint softmax 2026-07-26 — §B.3, first training run pending). This
is the live and only depth path — it replaced the earlier gripper-frame TSDF box
entirely. Camera intrinsics are the D405 factory calibration.

Code: [`policies/depth_pointmap/`](../src/lerobot/policies/depth_pointmap/)
(`configuration_pointmap.py`, `modeling_pointmap.py`, `modeling_stream.py`) plus
the read patch in
[`modeling_molmoact2.py:655`](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L655).
Everything is gated on `pointmap_config` (`None` = depth-free, zero cost).

Two halves: **A** — turn one wrist depth frame into 192 metric patch tokens.
**B** — co-evolve those tokens through a DepthStream and let the action expert
read them per layer through a joint softmax (depth tokens as extra context
columns; rebuilt 2026-07-26, replacing the α-gated additive read that never
trained — full decision record in
[depth_redesign_options.md](depth_redesign_options.md)).

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
(channels 5→128→256→$d_d$, resolution 40→20→10→5; widened from 32/64 hiddens
2026-07-25), global-average-pool to $d_d$
([modeling_pointmap.py](../src/lerobot/policies/depth_pointmap/modeling_pointmap.py)).

### A.3.1 Short-term history (temporal attention in the CNN)

Rebuilt 2026-07-25 ([depth_history_design.md](depth_history_design.md)) to
mirror the MEM video encoder: the 5-frame depth window rides the **same** CNN
as extra batch rows, and a `TemporalFusion` after each block lets every pixel
of the current frame attend that pixel across all frames (itself included —
the abstain route) in one softmax, with sinusoidal $e(\Delta t)$ frame stamps
and a pixelwise MLP. Past rows are dropped before pooling, so **the encoder
always emits 192 tokens** — history changes token content, never token count
(the earlier token-concat build that fed $(T_h{+}1)\cdot192$ tokens to the
stream is deleted). The shared `history_images_mask` dropout draw masks the
past keys, computing exactly the missing-window op. Encoder total ≈ 9M params
(trunk 4.75M + fusion 4.14M; heads 2/4/8).
If the current patch is completely empty, its fused feature and absolute PE use
the newest valid historical centroid; it falls back to the null bank when history
is masked/missing or that patch is empty in every frame.

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
(2) no reconstruction aux loss — flow loss only (dropped 2026-06-14); the
anti-laziness lever is wrist-RGB dropout instead (§B.3.1).

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

The stream owns the per-layer depth-column bias `depth_bias` (init −2, §B.3).

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

#### TO REVISE (flagged 2026-07-22, decide later)

- **Shared read projections**: no per-layer $W^r_K, W^r_V$ — is one $d_d \to
  d_{act}$ map enough for early-fine vs late-semantic depth states, or should the
  read be per-layer (cost: $2 L d_d d_{act}$ params)?
- **Post-RoPE K/V as the bridge input**: the stream re-projects the VLM's cached
  keys/values (position phases baked in) rather than layer hidden states —
  convenient (already cached for the expert) but not obviously the right feature
  space for the depth cross-attention.

### B.3 The read — joint softmax over context + depth columns

Decided and built 2026-07-26 ([depth_redesign_options.md](depth_redesign_options.md)
§5.2; replaces the 2026-06-14 α-gated additive read — post-mortem in §B.4).
Patched into every block's cross-attention: the depth tokens are appended as
extra key/value columns of the **same** softmax the expert already runs over the
LLM context, with one learned per-layer score bias on the depth columns
(`join_depth_columns` in
[modeling_stream.py](../src/lerobot/policies/depth_pointmap/modeling_stream.py)):

```python
if depth_kv is not None:
    k, v, attn_mask = join_depth_columns(attn_mask, k=k, v=v,
                                         depth_kv=depth_kv, depth_bias=depth_bias)
out = self._attention(q, k, v, attn_mask=attn_mask)   # ONE SDPA, T_ctx + 192 columns
```

Per layer $\ell$, head $h$, query $i$, with $b_\ell$ = `depth_bias[ℓ]`:

$$o_i = \sum_{j \in ctx} w_{ij}\, v_j + \sum_{m \in depth} w_{im}\, v_{d,m}, \qquad
w = \mathrm{softmax}\Big(\big[\ q_i \cdot k_j / \sqrt{D_h}\ ;\ \ q_i \cdot k_{d,m} / \sqrt{D_h} + b_\ell\ \big]\Big)$$

- **No gate, no sink.** Depth competes with context for one normalization mass —
  the same mechanism every other modality uses. Abstention is free: mass parks on
  the context columns when depth is uninformative.
- **$b_\ell$ init −2** (soft start): each depth column's odds are multiplied by
  $e^{-2} \approx 0.135$; total depth mass also depends on the 192 depth columns
  and the number/content of valid context columns. Being additive *inside* the
  softmax, $b_\ell$ avoids the extra multiplicative $\tanh(\alpha)$ bottleneck,
  although content-gradient magnitude still scales with actual depth mass (verified:
  `test_joint_softmax_matches_eager_and_bias_gets_gradient` also proves the
  $b_\ell$ gradient flows through SDPA's `attn_mask` on this torch build).
- **No bit-identity at init**, deliberately: the old design bought exactness at
  step 0 (frozen-policy assumption) at the price of the entire depth path never
  training (§B.4). The expert is trainable and heals the small init perturbation.
- Single static-shape SDPA → compile-friendly; padding mask handled by
  concatenation (context columns keep their pad mask, depth columns are always
  valid).

Wiring: training threads `depth_state` through the layer loop in
`_compute_flow_matching_loss_joint_per_layer` (one stream block per layer, stream
in float32 with autocast off, read K/V cast to the action dtype and expanded to the
flow-timestep batch at the read site; `depth_bias[ℓ]` read in-region inside
`run_layer` — gradient-checkpointing closure rule). Inference runs the whole
stream inside the patched `prepare_context` and caches per-layer depth K/V +
`depth_bias` on the context; the policy hands the encoder tokens over via
`action_expert._lerobot_pointmap = (init_tokens, wrist_sel)` once per control
step.

### B.3.1 Anti-laziness: wrist-RGB dropout (attention masking)

Modality-laziness counterpart of the read (depth_redesign_options.md §2.3, §4.3):
with RGB explaining the demos, even a healthy depth path gets ~no gradient. At
train time, with probability `pointmap_config.rgb_dropout_prob` (default 0.15),
the depth camera's contribution is **removed from attention** for that sample:

$$\text{attention\_mask}\big[b,\ S_b[cT_w : (c{+}1)T_w]\big] = 0$$

i.e. the camera's `<im_patch>` span (same span the depth stream slices) leaves the
mask. Consequences — this is *removal*, not corruption: nothing attends those
tokens (LLM prefix and the expert's cross-attention both consume this mask), so
**no gradient reaches the vision tower through that camera**, and the model sees
absence rather than a black image it could learn as a cue. Token layout, RoPE and
`position_ids` are untouched (positions are `arange`, not a mask cumsum), so the
prompt geometry is identical to an undropped sample. Applied **after**
`_build_labels` (the answer-span math reads the mask). The video-encoder history
frames of that camera need no separate treatment: temporal fusion is per-camera,
so they only ever reach the LLM through the masked current-frame span.

Two consumers gather wrist features **directly**, bypassing attention masks — the
actor's `DepthStream` wrist bridge and the critic's `depth_blocks`. Both take a
per-row `cross_on` switch derived from the *same* mask
(`attention_mask.gather(1, wrist_sel).any(1)`), which zeroes the cross-attention
residual for dropped rows: forward contribution and gradient both vanish, and the
row's depth stream runs depth-only (self-attn + MLP). Single source of truth —
the mask — for every path.

Independent of the depth-modality dropout (`dropout_prob` 0.25) and the shared
history dropout. Training-text path only; inference is unaffected (`cross_on` is
all-True when the mask has no zeros in the span).

Telemetry (replaces `pointmap_gate*`): `depth_attn_mass_mean`/`_max` (softmax
mass on depth columns, aggregated over stream layers, captured on the first
micro-batch of logged actor updates — the per-layer breakdown is probe-only),
`depth_bias_mean`, `depth_grad_norm_preclip` (read BEFORE
`clip_grad_norm_` — the α-era metric read after it and understated ~4×), plus
`probes/depth_modality_probe.py`: the 2×2 {RGB±, depth±} MSE matrix vs GT,
pairwise action deltas, per-layer mass, and finite-difference input
sensitivities (depth vs RGB).

### B.3.2 Held-out loss (replaced the ablation deltas, 2026-08-08)

`val_loss_flow` and `val_loss_discrete_ce` (`rl/molmoact2/val_loss.py`), logged
every `log_freq` next to the train losses. `val_loss_frames` (128 = one effective
batch) frames are sampled stride-snapped and evenly across the val episodes and
**packed once at startup**, so a call is `ceil(n / batch_size)` no-grad forwards
and nothing else. Timesteps sit on the stratified quantile grid of the training
Beta and the noise is drawn once from a seeded generator, so the flow number is a
deterministic function of the weights. Pack dropout is suppressed and the policy
runs in `eval()`, which means this is the deployment regime and reads *lower*
than `train/loss_flow`; compare val to train with that offset in mind.

#### Why the ablation deltas were deleted

The retired metric ran extra no-grad legs per modality on the training
micro-batch and reported
$\Delta = \mathcal{L}_{\text{ablated}} - \mathcal{L}_{\text{present}}$
(`depth_ablation_delta`, `rgb_ablation_delta/<cam>`). It was not broken as an
estimator — measured over v6, the paired legs correlated at $0.99969$ and the
difference carried $2.5\%$ of the raw loss spread, so the seeding did its job.
It was the wrong estimand. At the optimum for the CE head,

$$\Delta_m^\star = H(a \mid x_{\setminus m}) - H(a \mid x) = I(a; x_m \mid x_{\setminus m})$$

and the conditioning set includes the parameters, which have memorised the
training episodes. On a training frame $(\text{episode}, \text{phase})$ is
recoverable from proprio plus the task string, $a$ is then a lookup, and the CMI
is zero **whether or not the visual pathway works**.

The logs show exactly that. All three deltas started positive and decayed
monotonically as memorisation completed (v6 `top`: $+0.00081 \to -0.00109$,
slope $p = 0.015$; v5 `depth`: $+0.00414 \to +0.00112$, $p = 0.019$), ending at
$0.12\%$ of the CE they sat on and $0.06\%$ of the val/train CE gap. Nor was
power the fix: at a per-measurement SNR of $0.09$, $|t| = 3$ needs ~1,100 logged
steps to confirm an effect worth $0.0004$ nats. Note also that the loss it
differenced was $97.4\%$ FAST CE and only $2.6\%$ flow, so the long-standing
"make it CE-only" note was worth nothing.

Held-out frames are the only place $H(a \mid x_{\setminus m})$ is off the
memorisation floor, which is why the replacement measures there. What survives
from the depth telemetry is `depth_token_rms_ratio` and
`depth_grad_norm_preclip`: over v6 they read $2.52 \to 3.16$ (rising) and
$2.8\times10^{-2}$, i.e. the adapter is emitting *louder* tokens that change the
answer *less* — the path is alive and the model is routing around it.

### B.4 HISTORICAL — the α-gate soft deadlock (retired 2026-07-26)

The 2026-06→07 read was $o = \mathrm{SDPA}(q, K_{ctx}, V_{ctx}) +
\tanh(\alpha_\ell)\, r_\ell$ with zero-init α and a zero-value sink column.
It never trained: across every run to 1025 steps, $\tanh\alpha$ absmax sat at
~0.007 with no trend (an Adam random walk; a real signal ⇒ α ≈ 0.5 by step
1000). Kept here as the post-mortem — the freeze-bug lessons at the end remain
load-bearing.

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
2. **Gradient checkpointing + closures.** Per-layer parameters must be read
   **in-region** inside `run_layer` directly from the parameter (today:
   `self.depth_stream.depth_bias[layer_idx]`) — a non-leaf captured by closure
   into a checkpointed region can lose its gradient path.
   (Adopted as a safety invariant; the actual 2026-06-20 culprit was bug 1.)

### B.5 dtype boundary

The stream and encoder run float32; the critic runs bf16 and **shares the encoder
class**. `fourier_position_encoding` intentionally computes its sin/cos ladder in
float32 (precision over the mm range) and the encoder casts the PE to the
projection weight dtype before `pos_proj` — without that cast the bf16 critic
crashes at `pos_proj`. This surfaced on the first GPU train step; float32-only
actor probes cannot catch it.

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

192 tokens at width 512, run once per observation; the per-denoise-step cost is
192 extra key columns in the expert's cross-attention — negligible. The stream is
~170M fresh params (36 blocks × 4.7M) — under the joint softmax its gradients are
scaled by actual depth mass without the old extra α multiplier; watch the held-out 2×2 probe for
overfitting (R3D warning) and prune $M$ via per-layer `depth_attn_mass`
telemetry if layers stay shut. **CUDA graphs are force-OFF** when
`pointmap_config` is set — parked deliberately: our config runs RTC, whose
denoise loop is eager and never touches the action graph, so graphs-off costs
nothing. If ever revisited (non-RTC + measured action-loop bottleneck): the
checkpoint's `_clone_static_context`/`_copy_context_` don't know the depth K/V
fields, so a graph replay would silently drop the depth columns — register them
as static inputs and add an eager-vs-graph test first.

### B.8 Sources

DepthVLA (2510.13375, MoT depth expert), PVI (2603.12772, zero-init residual into
frozen VLA), SpatialVLA (2501.15830, calibration-free egocentric back-projection),
Flamingo (per-layer tanh gate precedent); also PointACT 2605.21414, GST-VLA
2603.09079, 3DThinkVLA 2606.04436, GeoAlign 2606.03240.

### B.9 Hardware checklist for a meaningful run

- [x] Wrist mount + calibrated intrinsics (factory K in config since 2026-07-02)
- [x] `z_max_mm` sanity on real wrist-to-object range — verified 2026-07-26 on
      rebot-socks-annotated-v2 (all 6 eps): 88–95 % of pixels in [70, 800] mm,
      median ≈ 300 mm, 95–99 % patches non-null (the 5.5 m smoke was the old
      top mount)
- [x] Raw depth end-to-end: uint16 Z16, PNG16 sidecars, no hole-fill in recordings
      (masking lives in the encoder), `depth_units_mm = 0.1`
- [ ] First copied-visual-path training run: watch `depth_rgb_rms_ratio`,
      `depth_late_early_rms_ratio`, the component pre-clip gradient norms, and the
      2×2 probe's mse(rgb_only) − mse(rgb+depth)

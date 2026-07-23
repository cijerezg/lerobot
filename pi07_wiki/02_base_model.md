# 02 — Base model: MolmoAct2

The full parameter tables live in the in-tree reference
[`policies/molmoact2/ARCHITECTURE.md`](../src/lerobot/policies/molmoact2/ARCHITECTURE.md).
That document was written against the released `allenai/MolmoAct2-SO100_101`
checkpoint; the **current base is the multi-embodiment foundation checkpoint
`allenai/MolmoAct2`** (at `outputs/MolmoAct2`), so treat its `config.json` as the
source of truth for exact dimensions. Its SO-101 frame-conversion section is
obsolete (see "Embodiment" below). This page is the working summary plus the math
that matters for pi07.

## 1. Composition

```
MolmoAct2ForConditionalGeneration
├── model
│   ├── transformer      Qwen2.5-class decoder LLM (L layers, GQA, RoPE, SwiGLU)
│   ├── vision_backbone  SigLIP ViT → 2×2 attention pool → SwiGLU projector → LLM dim
│   ├── action_expert    DiT-style flow-matching head (one block per LLM layer)
│   └── (depth path)     pi07 addition — see 03_depth.md
└── lm_head              token logits (subtask/memory decode, discrete actions)
```

Two action paths coexist (`action_mode: both` in training):

- **Continuous** — the action expert denoises a chunk by flow matching, reading the
  VLM through per-layer cross-attention.
- **Discrete** — `lm_head` autoregressively emits `<action_start> … <action_end>`
  FAST tokens; a FAST tokenizer decodes them to a real chunk. At inference we run
  continuous; discrete CE is an auxiliary training signal (knowledge insulation
  keeps it from disturbing the continuous path).

## 2. VLM in one paragraph

Images are resized/tiled to 378×378 crops, cut into 14×14 patches, embedded by a
linear layer, and run through a SigLIP-style ViT (pre-norm LayerNorm blocks,
fp32 softmax). Two intermediate ViT layers are tapped and concatenated, a 2×2
attention pool reduces 4 patches to 1 token (196 tokens per image), and a SwiGLU
projector maps them to the LLM width. Projected image tokens are added into the
text sequence at `<image_patch>` positions. The LLM is a standard pre-norm
decoder: RMSNorm, GQA self-attention with RoPE (θ = 10⁶), SwiGLU MLP.

**Attention mask: causal text, bidirectional image block.** The mask is a causal
`tril` OR'd at prefill with `image_mask[:, :, None] & image_mask[:, None, :]`
(`_patch_batched_image_attention_bias`,
[modeling_molmoact2.py:196-206](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L196-L206)):
every image token attends every other image token, both directions, **across
cameras** (and history frames). Text stays causal; image tokens never see later
text (later-text positions aren't image positions, so the OR never unblocks
them). Consequence: no camera is privileged by the mask — top and wrist are
attention-symmetric. Camera order still matters positionally (RoPE phases,
"Image 1"/"Image 2" numbering), so the `image_keys` order must stay consistent
between training and inference, but neither camera sees more than the other.
The `token_type_ids` OR is the seam to extend if other token groups should ever
become bidirectional. During
training a patch (`_patch_training_kv_collection`,
[modeling_molmoact2.py:302](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L302))
makes every decoder layer also return its post-RoPE K/V so the action expert can
cross-attend in the same forward pass.

## 3. Action expert

A stack of DiT blocks, **one per LLM layer**, width `d_a` with `H` heads
(SO100/101 reference: `d_a = 768`, `H = 8`, head dim 96). Layer ℓ cross-attends
the LLM's layer-ℓ K/V, projected once by shared linears:

$$K_{ctx,\ell} = \mathrm{RMSNorm}(W_K\, K^{LLM}_\ell), \qquad V_{ctx,\ell} = \mathrm{RMSNorm}(W_V\, V^{LLM}_\ell)$$

with $W_K, W_V \in \mathbb{R}^{d_a \times d_{kv}}$ where $d_{kv}$ = LLM KV width
(num_kv_heads × head_dim). So every expert block reads the VLM *at its own depth* —
the same per-layer principle the DepthStream copies.

### 3.1 Block structure (AdaLN-Zero)

The flow time $t$ is embedded sinusoidally and MLP'd into a conditioning vector
$c \in \mathbb{R}^{d_a}$. Each block computes nine modulation vectors
$(\text{shift},\text{scale},\text{gate})$ for each of its three sublayers:

$$(\beta_1,\gamma_1,g_1,\beta_2,\gamma_2,g_2,\beta_3,\gamma_3,g_3) = W_{mod}\,\mathrm{SiLU}(c)$$

and with $\mathrm{mod}(x,\beta,\gamma) = x(1+\gamma) + \beta$:

$$x \mathrel{+}= g_1 \cdot \mathrm{SelfAttn}(\mathrm{mod}(\mathrm{RMSNorm}(x),\beta_1,\gamma_1))$$
$$x \mathrel{+}= g_2 \cdot \mathrm{CrossAttn}(\mathrm{mod}(\mathrm{RMSNorm}(x),\beta_2,\gamma_2);\ K_{ctx,\ell}, V_{ctx,\ell})$$
$$x \mathrel{+}= g_3 \cdot \mathrm{SwiGLU}(\mathrm{mod}(\mathrm{RMSNorm}(x),\beta_3,\gamma_3))$$

Self-attention is bidirectional over the chunk's action tokens, with RoPE and
per-head QK-RMSNorm. Cross-attention has a query projection only (K/V come
pre-projected per layer, already RoPE'd inside the LLM). $W_{mod}$ is
zero-initialized (AdaLN-Zero), residual output projections are scaled by
$(2L)^{-1/2}$. The final layer applies one more modulated RMSNorm and a
zero-initialized linear to `max_action_dim` outputs.

**pi07 modification:** the cross-attention forward of every block is patched to add
the gated depth read ([03 — Depth](03_depth.md) §4).

### 3.2 Flow matching

Exact algebra from `_prepare_flow_matching_tensors`
([modeling_molmoact2.py:1459](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L1459)):

Training. For action chunk $a$, noise $\varepsilon \sim \mathcal N(0, I)$, and
timestep $t \sim \mathrm{Beta}(\alpha{=}1.0, \beta{=}1.5)$ mapped into
$[t_{off},\, t_{off}+t_{scale}]$ (default $[0.001, 1.0]$):

$$x_t = (1-t)\,\varepsilon + t\,a, \qquad v^\*(x_t, t) = a - \varepsilon$$

$$\mathcal L_{flow} = \big\| v_\theta(x_t, t \mid \text{context}) - v^\* \big\|_2^2$$

(per-element MSE, masked over padded action dims/steps; several $t$ per sample via
`num_flow_timesteps`). Inference is plain Euler from noise
(`num_inference_steps: 5` in the live config):

$$x_0 \sim \mathcal N(0, I), \qquad x_{i+1} = x_i + \tfrac{1}{n}\, v_\theta\!\big(x_i,\ \tfrac{i}{n}\big), \qquad a \approx x_n.$$

The VLM prefix (and the depth stream) is a pure function of the observation, so it
is computed once and its K/V reused across all Euler steps.

## 4. Prompt and token layout

Sequence built by the processor
([processor_molmoact2.py](../src/lerobot/policies/molmoact2/processor_molmoact2.py)):

```
[Image 1<|image|> … Image N<|image|>]           # image tokens spliced at <image_patch>
<|im_start|>user
The task is to {task}. [The current step is {subtask}.]
The current state of the robot is <state_start><state_k>…<state_end>.
[The recent states of the robot, oldest to newest, were …]
[Images i to j are earlier frames from the {cam} camera, oldest to newest.]
[The quality is q of 5.] [The robot made {no} mistakes.]
Given these, what action should the robot take to complete the task?<|im_end|>
<|im_start|>assistant
<action_output>[discrete action tokens + <eos> in training]
```

The foundation model's setup/control clauses (embodiment / action-space routing,
`<setup_start>`/`<control_start>` tokens) were removed from pi07 on 2026-07-22 —
see [04 §1.1](04_memory.md) for the rationale and the pretraining vocabulary.

Bracketed clauses are the pi07 memory/metadata channels — each is independently
absent when its data is `None`, and with all of them off the prompt is
byte-identical to the legacy MolmoAct2 prompt (checkpoint compatibility, tested).
Full clause semantics: [04 — Memory & prompts](04_memory.md).

### 4.1 Discrete state string

Continuous state (normalized to $[-1,1]$) is quantized per dimension into
`num_state_tokens = 256` bins (`_build_discrete_state_string`,
[processor_molmoact2.py:225](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L225)):

$$k_i = \mathrm{round}\Big(\frac{s_i + 1}{2}\,(256 - 1)\Big) \in \{0,\dots,255\}$$

rendered as `<state_start><state_{k_1}>…<state_{k_D}><state_end>`. This export
supports discrete state only (`state_format='discrete'` enforced).

## 5. Anchor action encoding

`action_encoding: anchor` in the live config. Only ACTION is encoded — **states
and their norm stats stay absolute** (v2.1 decision). With current state $s$
(the anchor) and chunk $a_{1:H}$
([anchor_encoding.py](../src/lerobot/policies/molmoact2/anchor_encoding.py)):

- **anchor**: $\tilde a_k = a_k - s$ for all $k$; decode $a_k = \tilde a_k + s$.
- **delta** (available, unused): $\tilde a_1 = a_1 - s$, $\tilde a_k = a_k - a_{k-1}$;
  decode $a_k = s + \sum_{j \le k} \tilde a_j$.

`AnchorEncodeStep` runs before the normalizer (and stashes the anchor under the
`anchor_state` complementary key); `AnchorDecodeStep` runs after the unnormalizer.
Norm stats for the encoded actions come from `compute_delta_stats.py` on the
training dataset (`action_encoding_stats_path` in the config); stats files are
validated at load for encoding / chunk_size and old frame-converted files are
rejected.

## 6. Embodiment

rebot B601: 7-dim state/action (shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_yaw, wrist_roll, gripper). The pipeline operates in the **raw arm
frame** — the old SO-101 v3.0↔v2.1 joint conversion (`frame_so101.py`) was deleted
2026-07-04, and any SO-101 6-dim artifacts (checkpoints, stats) are legacy. Norm
stats come from the training dataset (`norm_tag: null`); the HF checkpoint's own
norm tags belong to its embodiments, not ours.

## 7. Backbone patches applied at load

`_load_hf_model` monkey-patches the HF `trust_remote_code` model
([modeling_molmoact2.py:913](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L913)):
batched attention-bias fix, leaf-safe image-token splicing, memory-efficient ViT
(optional gradient checkpointing), **training K/V collection** (each decoder layer
returns post-RoPE K/V), numpy bf16 cast fix, and — when `pointmap_config` is set —
`_patch_action_expert_pointmap_read` (the depth read,
[modeling_molmoact2.py:655](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L655)).

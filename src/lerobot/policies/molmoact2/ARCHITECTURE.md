# MolmoAct2 Architecture Reference

This document describes the full network architecture of the MolmoAct2
policy as implemented in this repo. The PyTorch code lives in the
HuggingFace `trust_remote_code` snapshot under
`~/.cache/huggingface/modules/transformers_modules/MolmoAct2_hyphen_SO100_101/`
(`modeling_molmoact2.py`, `configuration_molmoact2.py`, `inference.py`,
`image_processing_molmoact2.py`, `processing_molmoact2.py`), and the
LeRobot wrapper lives in
[lerobot/policies/molmoact2/](.) (`configuration_molmoact2.py`,
`modeling_molmoact2.py`, `processor_molmoact2.py`, `frame_so101.py`).

All values below are taken from the released
`allenai/MolmoAct2-SO100_101` config defaults. Anywhere the released
checkpoint's `config.json` overrides a default, treat the checkpoint
config as the source of truth.

---

## 1. Top-level composition

`MolmoAct2ForConditionalGeneration` wraps:

```
MolmoAct2ForConditionalGeneration
├── model: MolmoAct2Model
│   ├── transformer:      MolmoAct2TextModel          # Qwen2.5-7B-style decoder LLM
│   ├── vision_backbone:  MolmoAct2VisionBackbone     # SigLIP ViT + MQA pool + SwiGLU adapter
│   ├── action_expert:    ActionExpert                # flow-matching DiT-style head
│   └── action_expert_depth_gate: Linear (optional)   # off in SO100/101 checkpoint
└── lm_head: Linear(hidden=3584 → vocab=152064, bias=False)
```

Inference paths:

- **Continuous actions** — VLM forward pass collects per-layer KV; the
  action expert iterates flow-matching Euler steps cross-attending into
  those KV tensors and outputs a `(B, action_horizon, max_action_dim)`
  trajectory.
- **Discrete actions** — `lm_head` autoregressively decodes
  `<action_start> ... <action_end>` tokens, then a FAST tokenizer
  decodes them to a real-valued chunk.
- **Depth reasoning (optional)** — `lm_head` decodes
  `<depth_start> ... <depth_end>` tokens before action generation; the
  current SO100/101 checkpoint has this disabled
  (`enable_depth_reasoning=False`).

All sub-modules are described below.

---

## 2. Vision tower — `MolmoAct2VisionTransformer` (SigLIP-style ViT)

Config dataclass: `MolmoAct2VitConfig`.

| Parameter             | Value                  |
|-----------------------|------------------------|
| `hidden_size`         | 1152                   |
| `intermediate_size`   | 4304                   |
| `num_hidden_layers`   | 27 (but only layers up to `max(vit_layers)+1` are instantiated; see §3) |
| `num_attention_heads` | 16                     |
| `num_key_value_heads` | 16 (no GQA in ViT)     |
| `head_dim`            | 72                     |
| `hidden_act`          | `gelu_pytorch_tanh`    |
| `layer_norm_eps`      | 1e-6                   |
| `image_default_input_size` | (378, 378)        |
| `image_patch_size`    | 14                     |
| `image_num_pos`       | 577 (positional table; bicubic-resized to `(H/14, W/14)` at runtime) |
| `attention_dropout`   | 0.0                    |
| `residual_dropout`    | 0.0                    |
| `float32_attention`   | True (Q/K cast to fp32 for softmax) |
| `attn_implementation` | `eager` (overridable: `sdpa`, `flash_attention_2`) |
| `num_prefix_tokens`   | 0 (no CLS)             |

### 2.1 Patch + positional embed

- `patch_embedding`: `nn.Linear(14*14*3 → 1152, bias=True)` applied to
  flattened patches (input shape `(B*T, num_patches, 14*14*3)`, see
  §6.1 for how images get cropped/tiled).
- `positional_embedding`: learned table `nn.Parameter(577, 1152)`,
  reshaped to `(sqrt(577), sqrt(577), 1152)` and bicubic-interpolated
  with `antialias=True` to the per-call patch grid `(H/14, W/14)`.

### 2.2 Vision block (`MolmoAct2VisionBlock`)

Pre-norm residual block (LayerNorm, not RMSNorm):

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

- `attention_norm`: `nn.LayerNorm(1152, eps=1e-6)`
- `ffn_norm`:      `nn.LayerNorm(1152, eps=1e-6)`
- `attention`: `ViTMultiHeadDotProductAttention`
  - 16 query heads, 16 KV heads, `head_dim=72`
  - Fused projections `wq/wk/wv: Linear(1152→1152, bias=True)`,
    output `wo: Linear(1152→1152, bias=False)`
  - Optional fp32 softmax; supports `eager`, `sdpa`, `flash_attention_2`
  - No positional encoding inside the block (absolute pos added once
    after patch embed)
- `feed_forward`: `ViTMLP`
  - `w1: Linear(1152→4304, bias=True)` → `gelu_pytorch_tanh` →
    `w2: Linear(4304→1152, bias=True)`
  - Plain 2-layer MLP (no gating).

### 2.3 Forward output

`MolmoAct2VisionTransformer` returns the **list of all per-layer
hidden states** (`MolmoAct2VisionBlockCollection.forward` appends after
each block). The adapter selects two layers (next section).

---

## 3. Vision adapter — `MolmoAct2VisionBackbone`

Config dataclass: `MolmoAct2AdapterConfig`.

| Parameter             | Value                                 |
|-----------------------|---------------------------------------|
| `vit_layers`          | `(-3, -9)` — feature taps             |
| `hidden_size`         | 1152                                  |
| `num_attention_heads` | 16                                    |
| `num_key_value_heads` | 16                                    |
| `head_dim`            | 72                                    |
| `hidden_act`          | `silu`                                |
| `intermediate_size`   | 18944                                 |
| `text_hidden_size`    | 3584 (must match LLM `hidden_size`)   |
| `image_feature_dropout` | 0.0                                 |
| `pooling_attention_mask` | False                              |
| `float32_attention`   | True                                  |
| `attn_implementation` | `eager`                               |

The ViT is truncated to only run through `max(vit_layers)+1` layers
(default: layers 0..24 inclusive, since `-3 % 27 = 24`).

### 3.1 Layer feature concatenation

For each tap in `vit_layers` the corresponding hidden state is taken
and concatenated along the feature dim:

```
image_features: (B, num_crops, num_patches, 1152 * len(vit_layers)) = (..., 2304)
```

### 3.2 Spatial pooling — `image_pooling_2d`

Cross-attention over pool groups produces one token per pool window.

- Pooling grid: `pooling_size = [2, 2]` (set in the image processor),
  so 4 ViT patches → 1 pooled token. With a 27×27 ViT grid the high-res
  branch pools to about 14×14 = 196 high-res tokens per crop, plus a
  low-res branch (one resized copy of the image producing fewer
  tokens). The dataset processor budgets `196` image tokens per image
  (constant `MOLMOACT2_IMAGE_TOKENS_PER_IMAGE = 196`).
- Module: `ViTMultiHeadDotProductAttention`
  - Query: mean (or masked-mean) over the 4 patches per window —
    shape `(B*P, 1, 2304)`
  - Key/Value: the 4 source patches — shape `(B*P, 4, 2304)`
  - `wq/wk/wv: Linear(2304 → 16*72=1152, bias=True)`
  - `wo: Linear(1152 → 1152, bias=False)`
  - 16 heads of dim 72, fp32 softmax
  - Output: one 1152-d token per pooled patch group

### 3.3 Projection to LLM dim — `image_projector`

`ImageProjectorMLP`, a SwiGLU:

```
x = w2( silu(w1(x)) * w3(x) )
```

- `w1: Linear(1152 → 18944, bias=False)`
- `w3: Linear(1152 → 18944, bias=False)`
- `w2: Linear(18944 → 3584, bias=False)`
- Activation: `silu`

Output dim = `text_config.hidden_size = 3584`.

### 3.4 Token splicing

The projected image tokens are written into the text-token sequence at
positions where `input_ids == image_patch_id`. The token IDs at those
positions are mapped to zero so `wte(input_ids)` contributes nothing,
and the image features are added in (`x[image_positions] += image_features`).

---

## 4. Language model — `MolmoAct2TextModel`

Config dataclass: `MolmoAct2TextConfig`. This is a Qwen2.5-7B-style
decoder.

| Parameter                | Value                                |
|--------------------------|--------------------------------------|
| `hidden_size`            | 3584                                 |
| `num_hidden_layers`      | 48                                   |
| `num_attention_heads`    | 28                                   |
| `num_key_value_heads`    | 4 (GQA, 7 queries per KV head)       |
| `head_dim`               | 128                                  |
| `intermediate_size`      | 18944                                |
| `hidden_act`             | `silu` (SwiGLU)                      |
| `vocab_size`             | 152064 (base)                        |
| `additional_vocab_size`  | 128 (action/state/depth/control etc.)|
| `qkv_bias`               | True (bias on fused QKV)             |
| `max_position_embeddings`| 4096                                 |
| `rope_theta`             | 1e6                                  |
| `rope_scaling`           | None                                 |
| `rope_scaling_layers`    | None                                 |
| `use_qk_norm`            | False (`qk_norm_type='olmo'` if on)  |
| `layer_norm_eps`         | 1e-6                                 |
| `norm_after`             | False (pre-norm)                     |
| `embedding_dropout`      | 0.0                                  |
| `attention_dropout`      | 0.0                                  |
| `residual_dropout`       | 0.0                                  |
| `tie_word_embeddings`    | False                                |
| `initializer_range`      | 0.02                                 |

### 4.1 Embedding — `MolmoAct2Embedding`

Two parameter blocks concatenated at lookup time:

- `embedding: Parameter(152064, 3584)` — base vocab
- `new_embedding: Parameter(128, 3584)` — extra robotics/control tokens
  appended at indices `[152064, 152192)`

Embedding output → `emb_drop = Dropout(0.0)`.

`lm_head` is a separate `nn.Linear(3584 → 152064, bias=False)` (no
tying with `wte`), and the action/depth bin tokens are at fixed
offsets inside this 152064-vocab head (the extra 128 tokens of
`new_embedding` only show up on the input side; on the output side the
extras lm-head columns live in `lm_head.weight[152064:]` if added by
`resize_token_embeddings`, but the SO100/101 release keeps action/state
token IDs **below 152064**).

### 4.2 Rotary positional embedding — `MolmoAct2RotaryEmbedding`

Standard Llama-style RoPE with `rope_theta=1e6`, applied half-dim
rotation to Q and K just before attention. The class caches
`sin/cos` for the configured `max_position_embeddings` and supports
HuggingFace's `dynamic_rope_update` for advanced rope types (not used
in this checkpoint).

### 4.3 Decoder block — `MolmoAct2DecoderLayer` (pre-norm)

```
residual = x
x = attn_norm(x)
x, ... = self_attn(x, position_embeddings, attention_mask, ...)
x = residual + dropout(x)

residual = x
x = ff_norm(x)
x = mlp(x)
x = residual + dropout(x)
```

- `attn_norm`, `ff_norm`: `MolmoAct2RMSNorm(3584, eps=1e-6)`
  (computation in fp32, then cast back; learnable scale)
- `dropout`: `nn.Dropout(residual_dropout=0.0)`

There is also a `MolmoAct2PostNormDecoderLayer` subclass that applies
the norm **after** the sublayer; selected when `norm_after=True`. Not
used in the SO100/101 release.

#### 4.3.1 Self-attention — `MolmoAct2Attention`

- 28 query heads, 4 KV heads → `num_key_value_groups = 7`
- `head_dim = 128`, total Q dim `28*128 = 3584`, total KV dim per
  K or V = `4*128 = 512`
- Fused projection:
  `att_proj: Linear(3584 → 3584 + 512 + 512 = 4608, bias=True)`
- Output projection:
  `attn_out: Linear(3584 → 3584, bias=False)`
- Optional QK-RMSNorm (`use_qk_norm=False` in this checkpoint):
  - `qk_norm_type="olmo"` would normalize the full Q (3584) and K (512)
    vectors per head dim
  - `qk_norm_type="qwen3"` would normalize each head's 128-d slice
- RoPE applied to Q and K, then SDPA / eager / flash-attention.
- Past-key-value cache supported via `transformers.cache_utils.Cache`.
  The repo also monkey-patches a `_attention_forward` to additionally
  return raw K/V tensors so the action expert can cross-attend
  (see §5.4).

#### 4.3.2 MLP — `LanguageModelMLP` (SwiGLU)

- `ff_proj: Linear(3584 → 2*18944 = 37888, bias=False)`
- Split into `x, gate`; apply `silu(gate) * x`
- `ff_out: Linear(18944 → 3584, bias=False)`

### 4.4 Final normalization

After the 48 stacked decoder blocks:

- `ln_f: MolmoAct2RMSNorm(3584, eps=1e-6)`

The output goes to `lm_head` for token logits and is also the set of
hidden states used by the action expert (the action expert consumes
raw post-RoPE K/V tensors per layer, not the residual stream).

---

## 5. Action expert — `ActionExpert` (flow-matching head)

Config dataclass: `MolmoAct2ActionExpertConfig`.

| Parameter            | Value (SO100/101 checkpoint)         |
|----------------------|--------------------------------------|
| `hidden_size`        | 768                                  |
| `num_layers`         | 36 (must equal LLM `num_hidden_layers`) |
| `num_heads`          | 8, `head_dim = 768/8 = 96`           |
| `mlp_ratio`          | 4.0                                  |
| `ffn_multiple_of`    | 256 → inner = `round_up(768*4, 256) = 3072` |
| `timestep_embed_dim` | 256                                  |
| `dropout`            | 0.0                                  |
| `attn_dropout`       | 0.0                                  |
| `context_layer_norm` | True (`ActionExpertRMSNorm(768)` after KV projection) |
| `qk_norm`            | True (RMSNorm on Q/K, per-head, eps=1e-6) |
| `rope`               | True                                 |
| `causal_attn`        | False                                |
| `max_action_horizon` | 30                                   |
| `max_action_dim`     | 32                                   |

Wired to the LLM with:

- `llm_dim = text_config.hidden_size = 2560` (unused except for shape
  bookkeeping)
- `llm_kv_dim = num_key_value_heads * head_dim = 8 * 128 = 1024`

Important: the remote-code `MolmoAct2ActionExpertConfig` constructor
has defaults such as `num_heads=16`, but this repo's loaded
SO100/101 checkpoint explicitly stores `num_heads=8`. Probe outputs
therefore show 8 action-expert heads. The 16-head values in this model
belong to the vision tower / vision adapter, not to the action expert.

### 5.1 Time embedding

```
SinusoidalTimeEmbedding(dim=256)
  → Linear(256 → 768)
  → SiLU
  → Linear(768 → 768)
```

Produces a per-batch conditioning vector `c ∈ ℝ^{B, 768}`.

### 5.2 Action embedding

- `action_embed: Linear(max_action_dim=32 → 768)` applied to noisy
  trajectory `x_t ∈ ℝ^{B, H, 32}` → token sequence
  `(B, H, 768)`.

### 5.3 KV context projection

The LLM produces per-layer post-RoPE Key/Value tensors with shape
`(B, T, 8*128=1024)` (collapsed from `(B, 8_heads, T, 128)`). For each
LLM layer the action expert projects them:

- `context_k_proj: Linear(1024 → 768, bias=False)`
- `context_v_proj: Linear(1024 → 768, bias=False)`
- followed by `context_norm: ActionExpertRMSNorm(768, eps=1e-6)`
  (no affine; only normalization).
- Reshape into `(B, T, 8_heads, 96_head_dim)`.
- Optionally pass through the block's `cross_attn.k_norm` (per-head
  RMSNorm of width 96).

This means **every action-expert block has its own cross-attention KV
context, sourced from the corresponding LLM layer**.

### 5.4 ActionExpert block — `ActionExpertBlock`

Structure (DiT-style with AdaLN-Zero modulation):

```
shift_msa, scale_msa, gate_msa,
shift_mca, scale_mca, gate_mca,
shift_mlp, scale_mlp, gate_mlp   = modulation.linear(silu(c)).chunk(9, dim=1)

# 1) Self attention over the H action tokens
x = x + gate_msa.unsqueeze(1) * self_attn(
        modulate(self_norm(x), shift_msa, scale_msa),
        attn_mask=..., is_causal=False, rope_cache=...)

# 2) Cross attention to per-layer LLM KV
x = x + gate_mca.unsqueeze(1) * cross_attn(
        modulate(cross_norm(x), shift_mca, scale_mca),
        kv_k=K_l, kv_v=V_l, attn_mask=encoder_mask)

# 3) SwiGLU MLP
x = x + gate_mlp.unsqueeze(1) * mlp(
        modulate(ff_norm(x), shift_mlp, scale_mlp))
```

where `modulate(x, shift, scale) = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)`.

#### Per-block modules

- `self_norm`, `cross_norm`, `ff_norm`: `ActionExpertRMSNorm(768, eps=1e-6)`
  (no affine weight — pure norm, learnable scale is delivered by the
  modulation).
- `modulation`: `Linear(768 → 9*768)` preceded by `SiLU`.
- `self_attn = ActionExpertSelfAttention`
  - `qkv: Linear(768 → 3*768)` then split, reshape to
    `(B, H, 8_heads, 96_head_dim)`
  - Per-head Q/K RMSNorm (`ActionExpertRMSNorm(96, eps=1e-6)`,
    no affine) when `qk_norm=True`
  - RoPE on Q/K with `head_dim=96` (`ActionExpertRotaryEmbedding`,
    base 10000) — cache built once per `(seq_len, device, dtype)`
  - SDPA attention; `causal_attn=False` by default → bidirectional
    over action tokens
  - `out_proj: Linear(768 → 768)` then dropout
- `cross_attn = ActionExpertCrossAttention`
  - Query projection only: `q_proj: Linear(768 → 768)`
  - K, V come from the projected LLM KV (already in head-shape)
  - Per-head q_norm RMSNorm of width 96 (same `qk_norm_eps=1e-6`)
  - The same `k_norm` is applied to the KV (computed once per layer
    in `_prepare_kv_context`)
  - No RoPE on the cross path (LLM K already had RoPE applied)
  - SDPA attention; cross-attention mask zeros out invalid encoder
    positions
  - `out_proj: Linear(768 → 768)` then dropout
- `mlp = ActionExpertMLP` (SwiGLU)
  - `up_proj: Linear(768 → 3072)`
  - `gate_proj: Linear(768 → 3072)` — gating branch
  - `down_proj: Linear(3072 → 768)`
  - `silu(gate_proj(x)) * up_proj(x)` → `down_proj` → dropout

Initialization detail: residual projections (`self_attn.out_proj`,
`cross_attn.out_proj`, `mlp.down_proj`) are scaled by
`(2 * num_layers)^(-1/2)` = `(2 * 36)^(-1/2) ≈ 0.118`; modulation
projection is zero-initialized (DiT AdaLN-Zero convention).

### 5.5 Final layer — `ActionExpertFinalLayer`

```
shift, scale = modulation.linear(silu(c)).chunk(2, dim=1)
out = linear( modulate( norm(x), shift, scale ) )
```

- `norm: ActionExpertRMSNorm(768, eps=1e-6)`
- `modulation: Linear(768 → 2*768)` (zero-init)
- `linear: Linear(768 → max_action_dim=32)` (zero-init)

Output is the **velocity field** at the current flow time, shape
`(B, H, 32)`. Action dim padding is zeroed out (a `mask_action_dim`
vector zeroes columns beyond the robot's true `action_dim`).

### 5.6 Flow-matching sampling loop

Implemented in `MolmoAct2Model.generate_actions_from_inputs` and
`_run_action_flow_loop`:

1. Run the VLM forward (vision + text), keeping `use_cache=True` so
   each layer's K, V become the cross-attention context.
2. Sample `x_0 ~ N(0, I)` with shape `(B, H, max_action_dim=32)`.
3. For `i = 0 .. steps-1` (default `flow_matching_num_steps = 8`):
   - `t = i / steps`
   - `v = ActionExpert.forward_with_context(x_i, t, context, modulation_cache[i])`
   - `x_{i+1} = x_i + (1/steps) * v`
4. Return `x_steps` as the predicted action trajectory.

Training uses a Beta(`α=1.0`, `β=1.5`) sampler over `t ∈ [time_offset,
time_offset + time_scale]` (default `[0.001, 1.0]`), with cutoff 1.0
(`flow_matching_cutoff`, `flow_matching_time_offset`,
`flow_matching_time_scale` in both configs).

### 5.7 Optional depth gating

When `action_expert_depth_gate=True`, an extra
`Linear(llm_kv_dim=1024 → 1)` (or one such linear per layer when
`action_expert_depth_gate_per_layer=True`) computes a sigmoid scalar
from the mean-pooled non-depth token features, and multiplies the
KV tensors at the depth-token positions before they enter the
action expert. The gate bias is initialised to
`action_expert_depth_gate_init_bias = -4.0` (sigmoid → ~0.018) so the
depth tokens start "off" and are unlocked only when the gate trains
above the bias. Disabled in the SO100/101 release.

---

## 6. Token sequence layout (encoder side)

The LeRobot processor (`processor_molmoact2.py`) and the HF
`processing_molmoact2.py` together build:

```
[<setup_start> setup_tokens <setup_end>]      # optional, ~control_mode tag
[<control_start> control_tokens <control_end>] # optional, ~control_mode tag
[image_1: <image_start> ... patches ... <image_end>]
...
[image_N: <image_start> ... patches ... <image_end>]
language_instruction
[<state_start> num_state_tokens <state_end>]   # discrete state quantization
                                               # default num_state_tokens = 256
[<action_start> action_tokens <action_end>]    # only in discrete / "both" mode
[<eos>]
```

Token budgets used by `infer_molmoact2_max_sequence_length` (from
[configuration_molmoact2.py](configuration_molmoact2.py)):

| Constant | Value |
|----------|-------|
| `MOLMOACT2_IMAGE_TOKENS_PER_IMAGE` | 196 |
| `MOLMOACT2_FIXED_PROMPT_TOKEN_BUDGET` | 80 |
| `MOLMOACT2_TASK_TOKEN_BUDGET` | 32 |
| `MOLMOACT2_DISCRETE_ACTION_WRAPPER_TOKENS` | 4 |
| `MOLMOACT2_MIN_DISCRETE_ACTION_TOKENS_PER_STEP` | 6 |
| `MOLMOACT2_DISCRETE_ACTION_TOKENS_PER_DIM` | 0.95 |
| `MOLMOACT2_SEQUENCE_LENGTH_MARGIN` | 32 |
| `MOLMOACT2_SEQUENCE_LENGTH_MULTIPLE` | 64 (sequence padded to this multiple) |

### 6.1 Image processor — `MolmoActImageProcessor`

(`image_processing_molmoact2.py`)

| Parameter | Default |
|-----------|---------|
| `size`               | `{height: 378, width: 378}` |
| `image_mean`, `image_std` | ImageNet standard (used only if not 0.5/0.5) |
| `resample`           | `BILINEAR` |
| `max_crops`          | 8 |
| `overlap_margins`    | `[4, 4]` |
| `crop_mode`          | `"overlap-and-resize-c2"` |
| `patch_size`         | 14 |
| `pooling_size`       | `[2, 2]` (2×2 patch pool before the LLM) |

For each input image the processor:
1. Selects a tiling `(rows, cols)` with `rows*cols ≤ max_crops` that
   best preserves the image aspect ratio.
2. Splits the resized image into overlapping `378×378` crops, plus one
   low-res copy of the whole image.
3. Each crop is unfolded into `27*27 = 729` flat patches of
   `14*14*3 = 588` pixels (uint8 → mapped to `[-1, 1]` inside
   `MolmoAct2VisionBackbone.forward`).
4. The adapter pools 2×2 patches → roughly 14*14 = 196 high-res tokens
   per crop after pooling, hence `MOLMOACT2_IMAGE_TOKENS_PER_IMAGE`.

### 6.2 State / action / depth special tokens

Released checkpoint tokens (from `MolmoAct2Config.__init__`):

- Visual: `<image_start>`, `<image_end>`, `<image_patch>`, `<image_col>`,
  `<low_res_image_start>`, `<image_low_res>`,
  `<frame_start>`, `<frame_end>` (used for video / multi-frame).
- Output framing: `<action_start>`, `<action_end>`, `<action_output>`.
- Discrete action bins: `<action_token_start_id + k>` for `k ∈ [0, num_action_tokens)`.
- State framing: `<state_start>`, `<state_end>` plus
  `<state_token_start_id + k>` for `k ∈ [0, num_state_tokens=256)`.
- Depth framing: `<depth_start>`, `<depth_end>`, `<depth_output>` and
  `<depth_token_start_id + k>` for `k ∈ [0, num_depth_tokens)`
  (depth disabled in this checkpoint).
- Control/setup: `<setup_start>`, `<setup_end>`,
  `<control_start>`, `<control_end>`.

State input is quantized into one of `num_state_tokens=256` bins per
state dimension, and the joint sequence is wrapped by
`<state_start> ... <state_end>` (continuous state embeddings are
**not** supported by this HF export — `state_format='discrete'` is
enforced at construction time).

---

## 7. LeRobot wrapper — `MolmoAct2Policy`

(`modeling_molmoact2.py` in this directory)

Wraps the HF model with:

- Config: [`MolmoAct2Config`](configuration_molmoact2.py)
  - `chunk_size = 30`, `n_action_steps = 30` (action horizon used by
    LeRobot — note this overrides the checkpoint's
    `max_action_horizon = 30` via `_override_loaded_max_action_horizon`).
  - `action_mode ∈ {'continuous', 'discrete', 'both'}`
  - Training optimizer split:
    `optimizer_vit_lr = 5e-6`,
    `optimizer_connector_lr = 5e-6` (vision adapter),
    `optimizer_lr = 1e-5` (LLM body),
    `optimizer_action_expert_lr = 5e-5`,
    `betas = (0.9, 0.95)`, `eps = 1e-6`,
    `weight_decay = 0.0`, `grad_clip_norm = 1.0`.
  - Cosine LR schedule with `warmup=200`, decaying to
    `scheduler_decay_lr = 1e-6`.
  - Optional LoRA (`enable_lora_vlm=False` by default,
    `rank=64`, `alpha=16`, `dropout=0.05`).
  - `train_action_expert_only` freezes everything except the action
    expert.
  - `freeze_embedding=True` keeps `wte` (and shares-pointer-checked
    `lm_head`) frozen during finetuning.

- Patches applied to the HF backbone in `_load_hf_model`:
  - `_patch_batched_image_attention_bias` — replaces the upstream
    attention-bias builder with one that handles batched 4-D masks
    correctly.
  - `_patch_leaf_safe_input_embedding_update` — replaces
    `build_input_embeddings` with a clone-and-add variant so the
    image-token positions are written without leaf-tensor issues.
  - `_patch_memory_efficient_vision_backbone` — replaces
    `vision_backbone.encode_image` with a streaming variant that
    optionally runs each ViT block through gradient checkpointing.
  - `_patch_training_kv_collection` — patches each text-decoder block
    to additionally return its post-RoPE K, V so the action expert can
    cross-attend during a single full-sequence forward pass (no extra
    pass needed).
  - `_patch_numpy_dtype_cast` — fixes bf16-to-numpy conversion inside
    the HF module's normalization helpers.

- Frame conversion (SO-101): `SO101V3ToV21Step` /
  `SO101V21ToV3Step` in [`frame_so101.py`](frame_so101.py) translate
  joints 1 (`shoulder_lift`) and 2 (`elbow_flex`) between LeRobot v3.0
  and the v2.1 convention MolmoAct2 was pretrained on.

- RTC (real-time chunking) integration via `RTCProcessor` from
  `lerobot/policies/rtc/`.

---

## 8. Quick parameter-count breakdown (order of magnitude)

Approximate (ignoring biases and norms):

- ViT (first 25 blocks used): `25 × (3 × 1152² + 2 × 1152 × 4304) ≈ 0.32 G`
- Vision adapter:
  - Pool QKV: `3 × 2304 × 1152 ≈ 8 M`; out: `1152² ≈ 1.3 M`
  - Projector: `2 × 1152 × 18944 + 18944 × 3584 ≈ 112 M`
- LLM (48 blocks):
  - Embedding: `(152064 + 128) × 3584 ≈ 545 M`
  - Attention per block: `3584 × 4608 + 3584² ≈ 29.4 M`
  - MLP per block: `3584 × 37888 + 18944 × 3584 ≈ 203 M`
  - Body total: `48 × 232 M ≈ 11.1 G`
  - `lm_head`: another `545 M`
- Action expert (36 blocks):
  - Per block: self-attn `4 × 768² ≈ 2.4 M`, cross-attn `2 × 768² ≈ 1.2 M`,
    MLP `3 × 768 × 3072 ≈ 7.1 M`, modulation `768 × 9 × 768 ≈ 5.3 M`
  - Total ≈ `36 × 16 M ≈ 0.58 G`
  - Plus context projections `2 × 1024 × 768 ≈ 1.6 M`, time embed `~0.8 M`,
    action embed/final `~0.05 M`.

Total: roughly **~13 B parameters** for the released SO100/101
checkpoint (dominated by the 7 B-class Qwen2.5 LLM body).

---

## 9. Frozen tensors during default LeRobot finetuning

With the default LeRobot config (`enable_lora_vlm=False`,
`train_action_expert_only=False`, `freeze_embedding=True`):

- Frozen: `wte.embedding` and `wte.new_embedding` (the input
  embeddings); the wrapper additionally checks that this does not
  alias `lm_head` (it doesn't, since `tie_word_embeddings=False`).
- Trained: everything else, but with per-group learning rates
  (`optimizer_vit_lr`, `optimizer_connector_lr`, `optimizer_lr`,
  `optimizer_action_expert_lr`).

Under `train_action_expert_only=True` only parameters whose name
contains `"action_expert"` train — the VLM is fully frozen and put in
`eval()` mode (BatchNorm/Dropout off).

Under `enable_lora_vlm=True`, PEFT LoRA adapters are attached to the
LLM body (and optionally the action expert). LoRA defaults:
`r=64`, `α=16`, `dropout=0.05`, `bias='none'`.

# Advanced Usage Guide — MolmoAct2

This guide covers the MolmoAct2 RL pipeline end-to-end: action encodings, the offline-then-online training flow, the full configuration reference, the iterative RL loop, interventions, async inference, action post-processing, and the SO-101 joint-frame convention.

The pipeline shares its core scripts (`rl_offline.py`, `rl_learner.py`, `rl_actor_async.py`) with the $\pi_{0.5}$ implementation — only the policy class and a handful of policy-specific fields differ. If you are looking for the $\pi_{0.5}$ variant, see [advanced_usage_pi05.md](advanced_usage_pi05.md).

---

## Table of Contents

1. [Action Encodings](#action-encodings)
2. [Training Pipeline](#training-pipeline)
3. [Configuration Reference](#configuration-reference)
4. [Pretrained Merge (RETAIN-style)](#pretrained-merge-retain-style)
5. [The Iterative RL Loop](#the-iterative-rl-loop)
6. [Using a v2.1 Dataset](#using-a-v21-dataset)
7. [SO-101 Frame Convention (CRITICAL)](#so-101-frame-convention-critical)
8. [Interventions and Real-Time Control](#interventions-and-real-time-control)
9. [Async Inference Architecture](#async-inference-architecture)
10. [Action Post-Processing](#action-post-processing)
11. [Buffer Caching](#buffer-caching)
12. [Probes](#probes)
13. [Dataset Annotation (Not Implemented)](#dataset-annotation-not-implemented)

---

## Action Encodings

This is the first decision to make — it is set in the config and applies to **all** subsequent training, online and offline. Switching encodings later means recomputing statistics and retraining from base.

In the MolmoAct2 pipeline only the **action** is encoded — the observation state is always passed absolute (in v2.1 frame, after `SO101V3ToV21Step`). Normalization statistics for `observation.state` likewise remain absolute. This is different from setups that re-anchor state and action together.

### Anchor (recommended)

Each action is encoded as an offset from the initial state $\mathbf{s}_0$ of the current chunk:

$$\mathbf{d}_t = \mathbf{a}_t - \mathbf{s}_0$$

This provides **translation invariance**: the same reaching motion produces the same encoded action regardless of the arm's starting position. MolmoAct2 was originally trained on absolute actions, but in our experiments anchor generalizes substantially better.

### Absolute

Actions are raw joint positions: $\mathbf{a}_t \in \mathbb{R}^d$. No transformation applied. The dataclass default for `action_encoding`. Simplest, but does not generalize across different starting configurations.

### Delta

First-order differences:

$$\mathbf{d}_0 = \mathbf{a}_0 - \mathbf{s}_0, \quad \mathbf{d}_t = \mathbf{a}_t - \mathbf{a}_{t-1} \text{ for } t > 0$$

More compact than anchor, but accumulated errors can drift over long horizons.

### Computing Statistics

Anchor and delta encodings require precomputed normalization statistics. Generate them with:

```bash
python -m lerobot.scripts.compute_delta_stats \
    --data-dir /path/to/dataset \
    --encoding anchor
```

This iterates over all episodes, computes the per-timestep anchor (or delta) representation for every action chunk, and saves min/max/mean/std/quantile statistics to a `.pt` file. Set `policy.action_encoding_stats_path` to point to this file. The file is required only for `action_encoding: anchor | delta`.

---

## Training Pipeline

### Phase 1: Offline Training

Offline training initializes the policy from demonstrations before any robot interaction. With `skip_critic: true` (the recommended setting unless you specifically want RECAP-style RL), the critic is not trained at all and offline training is pure flow-matching + discrete CE behavior cloning.

```bash
python -m lerobot.scripts.rl_offline --config config_rl.yaml
```

This script:
- Loads the dataset from `dataset.root`
- Loads MolmoAct2 base weights from `policy.base_path` (or a finetuned checkpoint from `policy.pretrained_path` if set)
- Trains for `policy.offline_steps` steps with optional gradient accumulation
- Runs validation probes every `val_freq` steps if `val_dataset_path` is set
- Saves checkpoints every `offline_save_freq` steps under `offline_output_dir`
- Is single-process (no Accelerator / DDP) — model-agnostic, dispatches to the molmoact2 trainer based on `policy.type`

### Phase 2: Online Training

Once you have an offline checkpoint, point `policy.pretrained_path` at it and start the learner and actor on separate terminals (or machines):

**Learner** (gRPC server, runs on GPU machine):
```bash
python -m lerobot.rl.rl_learner --config config_rl.yaml
```

**Actor** (gRPC client, runs on robot machine):
```bash
python -m lerobot.rl.rl_actor_async --config config_rl.yaml
```

The actor collects transitions at the configured FPS and streams them to the learner. The learner mixes online and offline data per batch when `cfg.dataset` is configured. Updated policy weights are pushed back to the actor every `policy.actor_learner_config.policy_parameters_push_frequency` seconds (default: 120 s; see [shared_config.py](../../src/lerobot/rl/shared_config.py)).

When the distributional critic is enabled (`skip_critic: false`), the learner writes an episode video every `episode_logging_freq` episodes with the predicted critic value overlaid frame-by-frame — useful for inspecting where the critic predicts failure versus where it predicts success. With `skip_critic: true` the value overlay is skipped and the video is plain rollout. The online replay buffer is dumped to disk every `episode_save_freq` episodes so crashed runs can be resumed.

---

## Configuration Reference

The reference config is [`config_rl.yaml`](../../../config_rl.yaml). The fields below are the ones worth tuning — the rest of the YAML is either self-explanatory or rarely needs changing from defaults declared in [rl_molmoact2.py](../../src/lerobot/rl/molmoact2/rl_molmoact2.py).

### Top-Level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | str\|null | `null` | Output directory for the **online** run (checkpoints, videos, buffer dumps). Null = auto-generated timestamped dir. |
| `offline_output_dir` | str | — | Separate output dir for offline training. Keep it distinct from `output_dir`. |
| `batch_size` | int | `96` | Replay buffer sampling batch size. Multiplied by `policy.gradient_accumulation_steps` to get the effective batch. |
| `skip_critic` | bool | `true` | Skip critic training entirely. Saves VRAM and speeds up training. Set `false` only if you are running RECAP-style RL with the distributional critic. |
| `treat_main_dataset_as_golden` | bool | `true` | When `true`, every demo in `dataset.root` is treated as optimal (advantage forced to 1.0). Set to `true` for teleop demos; set to `false` only if your main dataset includes failed episodes that should be down-weighted by the critic. |
| `log_freq` | int | `20` | WandB logging frequency, in optimization steps. |
| `save_freq` | int | `100` | Online checkpoint frequency (steps). |
| `offline_save_freq` | int | `400` | Offline checkpoint frequency (steps). |

### Validation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `val_dataset_path` | str\|null | `null` | Path to a separate validation dataset. Strongly preferred over `val_split`. |
| `val_split` | float | `0.0` | Fraction of training data carved off for validation when no separate val dataset is provided. |
| `val_freq` | int | `800` | Run validation every N steps. Validation is time-consuming — don't set this too low. |
| `val_on_start` | bool | `false` | Run validation at step 0 to record a baseline. |

### Dataset

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset.root` | str | — | Local path to the annotated dataset. Must be available locally. |
| `dataset.repo_id` | str | — | HuggingFace Hub repo ID. Hub-only loading is not reliable — keep the dataset locally and rely on `dataset.root`. |
| `dataset.max_episodes` | int\|null | `null` | Cap the number of episodes loaded. Useful when the dataset is large enough to OOM during initial decode. |
| `dataset.additional_offline_dataset_paths` | list | `[]` | Extra dataset paths to merge alongside `dataset.root` — typically buffers collected by the policy in earlier RL iterations. Folding them back in is what gives the iterative loop its compounding effect. |
| `dataset.use_imagenet_stats` | bool | `false` | Use ImageNet normalization stats. Off by default for MolmoAct2 — the model's own preprocessor handles image normalization. |
| `buffer_cache_dir` | str | — | Cache directory for the offline buffer memmap (see [Buffer Caching](#buffer-caching)). |

### Policy — Identity & Checkpoint

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.type` | str | `"molmoact2_rl"` | Activates MolmoAct2RL policy. |
| `policy.task` | str | — | Task prompt (e.g., `"Pick up the red truck and put it in the bowl"`). Threaded into the VLM as text input. |
| `policy.base_path` | str | — | Path to the HF MolmoAct2 base model. Download with `hf download allenai/molmoact2-so100_101`. |
| `policy.pretrained_path` | str\|null | `null` | Path to a fine-tuned MolmoAct2 checkpoint. When set, weights are loaded from here instead of `base_path` (which still has to be set for tokenizer/processor metadata). |

### Policy — Training Mode

MolmoAct2 supports two training recipes, controlled by `action_mode` and `knowledge_insulation`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.action_mode` | str | `"both"` | `"both"` trains VLM by discrete CE on FAST tokens and the action expert by flow loss simultaneously. `"continuous"` falls back to the original "freeze VLM, train action expert only" recipe. `"discrete"` trains only the discrete head (rare). |
| `policy.inference_action_mode` | str | `"continuous"` | Which head emits actions at inference. The continuous (flow) head is faster and smoother; the discrete head is what the VLM was originally trained on. |
| `policy.knowledge_insulation` | bool | `true` | Detaches K/V going into the action expert so its gradients do not leak back into the VLM. Required when `action_mode: both`, ignored when `train_action_expert_only: true`. |
| `policy.train_action_expert_only` | bool | `false` | If VRAM is tight, set `true` to freeze the VLM and only train the action expert. Cheaper but produces worse results than KI-style "both" training. |
| `policy.discrete_action_tokenizer` | str | — | Path to the FAST tokenizer used for discrete action tokens. Download with `hf download allenai/MolmoAct2-FAST-tokenizer`. |
| `policy.freeze_embedding` | bool | `true` | Never train token embeddings. Leave at `true`. |
| `policy.gradient_checkpointing` | bool | `true` | Activation checkpointing — required on most consumer GPUs. |

### Policy — Trainable Parameters

Per-submodule freeze schedule. `null` = freeze entire module; integer `N` = train layers with index ≥ `N`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.trainable_params.vision_from_layer` | int\|null | `16` | Train ViT (depth 25, resblocks 0..24) from this layer. Setting any value also unfreezes `image_pooling_2d` and `image_projector`. |
| `policy.trainable_params.language_from_layer` | int\|null | `0` | Train the language transformer (depth 36) from this layer. Setting any value also unfreezes the final layer norm (`ln_f`) and `lm_head`. |
| `policy.trainable_params.critic_vision_from_layer` | int\|null | `4` | Critic vision branch (ignored when `skip_critic: true`). |
| `policy.trainable_params.critic_language_from_layer` | int\|null | `18` | Critic language branch (ignored when `skip_critic: true`). |

The action expert is always trained regardless of these settings.

### Policy — Losses & Optimization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.optimizer_lr` | float | `5e-5` | Action-expert learning rate. The VLM branches inherit this LR by default. |
| `policy.optimizer_weight_decay` | float | `0.1` | L2 regularization. |
| `policy.optimizer_grad_clip_norm` | float | `1.0` | Max gradient norm. |
| `policy.gradient_accumulation_steps` | int | `1` | Effective batch = `batch_size × gradient_accumulation_steps`. Raise this on smaller GPUs to keep effective batch large without OOM. |
| `policy.dtype` | str | `"bfloat16"` | Computation dtype for the VLM. |

### Policy — Reward Shaping

These are relevant only when `skip_critic: false`. When the critic is skipped, advantages are forced to 1.0 across the dataset (per `treat_main_dataset_as_golden: true`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.reward_normalization_constant` | float | `1.0` | Rewards are divided by this value before being fed to the critic. Together with `discount`, sets the effective scale of the advantage. |
| `policy.terminal_failure_reward` | float | `-10.0` | Penalty for failed / timed-out episodes. Magnitude must dominate the discounted sum of shaping rewards leading up to failure. |
| `policy.advantage_scaling` | float | `0.2` | Scales advantage before tanh squashing. |
| `policy.advantage_top_k_fraction` | float | `0.3` | Top fraction of non-override samples labeled "positive" per batch (quantile cut). Golden + intervention samples are forced positive and excluded from the threshold pool. |

### Policy — Distributional Critic

Ignored when `skip_critic: true`. The critic is an HL-Gauss distributional value head with `num_value_bins` over the support `[value_support_min, value_support_max]`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.critic_llm_depth` | int | `12` | Number of transformer layers in the critic branch. |
| `policy.num_value_bins` | int | `201` | Number of HL-Gauss bins. |
| `policy.value_support_min` | float | `-2.0` | Lower bin edge. |
| `policy.value_support_max` | float | `0.0` | Upper bin edge. |
| `policy.hl_gauss_sigma_ratio` | float | `5.0` | Gaussian smoothing sigma as a multiple of bin width. |
| `policy.critic_lr` | float | `1e-4` | Critic learning rate (separate from `optimizer_lr`). |
| `policy.critic_target_update_weight` | float | `0.005` | Polyak averaging $\tau$. |
| `policy.critic_target_update_every` | int | `4` | Target network update period (steps). |
| `policy.discount` | float | `0.97` | Temporal discount $\gamma$. |

### Policy — Buffer & Training Schedule

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.offline_steps` | int | `10000` | Offline pretraining steps. |
| `policy.online_steps` | int | `20000` | Online training steps. |
| `policy.offline_buffer_capacity` | int | `100000` | Max offline transitions. |
| `policy.online_buffer_capacity` | int | `5000` | Max online transitions. |
| `policy.online_step_before_learning` | int | `10` | Warmup transitions before the first online gradient step. |
| `policy.policy_update_freq` | int | `1` | Update actor every N critic updates. |
| `policy.critic_warmup_steps` | int | `0` | Pure critic-only updates before the actor starts learning. |
| `policy.utd_ratio` | int | `1` | Critic updates per actor update. |
| `policy.async_prefetch` | bool | `false` | Async dataloader prefetch. Set `true` once you trust the cache fingerprint. |
| `policy.storage_device` | str | `"cpu"` | Replay buffer storage device. |

### Policy — Pretrained Merge

See [Pretrained Merge (RETAIN-style)](#pretrained-merge-retain-style).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.pretrained_merge_alpha` | float | `0.0` | Mixing factor: `W ← (1 − α) · W_current + α · W_pretrained`. `0` disables. |
| `policy.pretrained_merge_every_n_steps` | int | `0` | Merge period in optimization steps. `0` disables. |
| `policy.pretrained_merge_targets` | list | `[policy, critic]` | Optimizer-group keys to merge. |

### Policy — Inference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.torch_compile` | bool | `true` | Compile the action expert with `torch.compile(mode="reduce-overhead")`. Helps meet the 30 Hz control target. |
| `policy.action_clamp_limits` | list\|null | — | Per-joint `[min, max]` limits in **degrees**, applied in v3.0 arm frame *after* unnormalization and Butterworth filtering. `null` disables clamping. |
| `policy.inference_advantage` | float\|null | `1.0` | Constant advantage value injected as prompt conditioning. Match this to training: use `null` with `skip_critic: true`, keep `1.0` with `skip_critic: false`. |
| `policy.rtc_config.enabled` | bool | `true` | Enable real-time chunking — see [Async Inference Architecture](#async-inference-architecture). |
| `policy.rtc_config.execution_horizon` | int | `5` | Steps from each chunk that are executed before requesting a new chunk. |
| `policy.rtc_config.prefix_attention_schedule` | str | `"LINEAR"` | Prefix attention modulation. |
| `policy.rtc_config.max_guidance_weight` | float | `10.0` | Max RTC guidance weight. |

### Policy — Architecture

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.chunk_size` | int | `30` | Actions emitted per forward pass. |
| `policy.n_action_steps` | int | `30` | Actions consumed per forward pass (matches `chunk_size`). |
| `policy.num_inference_steps` | int | `5` | Flow-matching denoising steps. |

### Policy — Normalization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.norm_tag` | str\|null | `"so100_so101_molmoact2"` | Loads norm stats from the HF base checkpoint. Set to `null` only when loading a fine-tuned checkpoint that already has norm stats in its `policy_preprocessor_*.safetensors`. |
| `policy.normalize_gripper` | bool | `true` | **Must match the base model.** Setting `false` against a model that was trained with `true` crushes the gripper to 1.0 via the post-norm clamp. |

### Policy — Actor-Learner Communication

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.actor_learner_config.learner_host` | str | `"0.0.0.0"` | Learner bind address. `0.0.0.0` accepts cross-machine connections. |
| `policy.actor_learner_config.learner_port` | int | `50051` | Learner gRPC port. |
| `policy.actor_learner_config.policy_parameters_push_frequency` | int | `120` | Wall-clock seconds between learner→actor weight syncs. |

### Probes

See [Probes](#probes) for the schema and per-probe details.

---

## Pretrained Merge (RETAIN-style)

Inspired by [Wang et al., 2025 (arxiv 2512.08333)](https://arxiv.org/pdf/2512.08333) (RETAIN). Periodically interpolate the live training weights with the original pretrained snapshot:

$$\mathbf{W}_\text{new} = (1 - \alpha)\,\mathbf{W}_\text{current} + \alpha\,\mathbf{W}_\text{pretrained}$$

After each merge the optimizer's per-parameter state (`exp_avg`, `exp_avg_sq`, `step`) is cleared so subsequent gradient steps explore directions free from pre-merge momentum/variance estimates.

The pretrained snapshot is captured once at training startup and stored in pinned CPU memory; merges are GPU-cheap.

**Why it helps:** finetuning a generalist VLM on a narrow task tends to (a) overfit to the demos and (b) erode the model's broader capabilities. The convex pull toward the pretrained checkpoint mitigates both — the policy retains generalist behavior while specializing on the target task, and in practice generalizes better than either the pure-pretrained or pure-finetuned model on OOD variations.

Set `pretrained_merge_alpha > 0` and `pretrained_merge_every_n_steps > 0` to enable. With `alpha = 0.2` and `every_n_steps = 6000`, every 6000 steps the live weights are pulled 20% of the way back to the pretrained snapshot. Sensible starting range: `alpha ∈ [0.1, 0.3]`, `every_n_steps ∈ [4000, 8000]`.

In `rl_offline.py`, a checkpoint is saved before and after each merge (`*_pre_merge` and `*_post_merge`) so you can diff the weights and confirm the merge did what you expected.

---

## The Iterative RL Loop

Following RECAP's recommendation, retrain from the base model each iteration and include all accumulated data to avoid policy drift.

### Cycle

1. **Offline training** on existing demonstrations.
2. **Online training** (actor + learner) to collect new experience. The learner periodically dumps the online replay buffer to disk under `output_dir`.
3. **Convert** the dumped buffer into a video-format LeRobot dataset. *(Currently no first-party script for MolmoAct2 — see [Dataset Annotation](#dataset-annotation-not-implemented).)*
4. **Merge** by adding the converted path to `dataset.additional_offline_dataset_paths`.
5. **Retrain from the MolmoAct2 base model** with all data combined — old demos, prior cycles' on-policy data, and the new buffer.

Each cycle adds more diverse experience. `additional_offline_dataset_paths` accepts multiple paths.

---

## Using a v2.1 Dataset

This pipeline assumes a **LeRobot v3.0 dataset format** (introduced in `lerobot 5.0.0`). If you have an older v2.1 dataset, migrate it before pointing the config at it:

```bash
# In-place migration of a local v2.1 dataset to v3.0 format:
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id=your/dataset \
    --root=/path/to/local/dataset \
    --push-to-hub=false

# Or for a dataset hosted on the Hub (writes the new version back to main and tags it v3.0):
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id=your/dataset
```

The migration regenerates per-episode statistics, removes the deprecated `stats.json`, and updates `codebase_version` in `info.json`. Joint values are not modified — if your v2.1 dataset was also recorded in the v2.1 SO-101 joint convention (almost all were), the joint-frame transform documented in [SO-101 Frame Convention](#so-101-frame-convention-critical) is still relevant.

---

## SO-101 Frame Convention (CRITICAL)

MolmoAct2 was pretrained on SO-100/101 data in the **v2.1 joint convention**. All datasets recorded with LeRobot v3.0 (this repo) are in the **v3.0 convention**. Two joints differ:

| Joint | Transform: v3.0 → v2.1 |
|-------|------------------------|
| 1 · shoulder_lift | `v2.1 = −v3.0 + 90` |
| 2 · elbow_flex    | `v2.1 =  v3.0 + 90` |
| 0, 3, 4, 5        | unchanged              |

**Where this is handled:** [`policies/molmoact2/frame_so101.py`](../../src/lerobot/policies/molmoact2/frame_so101.py)
- `SO101V3ToV21Step` — inserted before the normalizer in the input pipeline (converts state + action in training data and live observations).
- `SO101V21ToV3Step` — inserted after the unnormalizer in the output pipeline (converts model actions back to arm frame before sending to the robot).

**Implications for training:**
- Datasets recorded with LeRobot v3.0 (this repo) are in v3.0 convention → the transform is correct as-is.
- If you ever use a dataset recorded in v2.1 convention (e.g., raw HF Hub datasets from before PR-777), remove or bypass `SO101V3ToV21Step` — otherwise joint angles will be double-converted.
- To verify a dataset's convention: check `observation.state` mean for joint 1 (shoulder_lift). v3.0 ≈ −30° to +10°; v2.1 ≈ 90° to 130°.

**Implications for inference:**
- `policy.norm_tag: so100_so101_molmoact2` is required when loading the zero-shot base checkpoint. Set `norm_tag: null` only when loading a fine-tuned checkpoint that already has norm stats embedded.
- `policy.action_clamp_limits` are defined in **v3.0 arm frame** (clamping runs *after* `SO101V21ToV3Step`).

---

## Interventions and Real-Time Control

During online training or inference, a human operator can take control of the robot via the leader arm.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `5` | Toggle intervention (take/release control) |
| `1` | Mark episode as **success** |
| `0` | Mark episode as **failure** |
| `2` | Start next episode |

### What Happens During Intervention

1. The inference thread **pauses** (checks `shared_state.is_intervening` and sleeps).
2. The environment thread reads raw joint positions from the leader arm and executes them directly.
3. The action queue is **cleared** to prevent stale policy actions from executing.
4. Each transition is marked with `is_intervention=1.0` in the metadata.

### How Interventions Affect Training

During the actor update, transitions marked as interventions have their advantage overridden to $A = 1.0$ (maximum positive). The same override applies to golden-dataset (demonstration) transitions. With `skip_critic: true` and `treat_main_dataset_as_golden: true`, every offline sample already gets $A = 1.0$ — the intervention override matters mostly during online runs.

### Re-engagement

When the human releases control (presses `5` again), the inference thread resumes immediately with a fresh action queue, preventing jerky transitions from stale predictions.

---

## Async Inference Architecture

The async pipeline decouples neural network inference (slow, variable latency) from robot control (strict FPS).

Two daemon threads run concurrently: an **environment thread** that maintains the control loop, pops one action at a time from the `ActionQueue`, sends it to the robot, and handles episode boundaries; and an **inference thread** that pulls the latest observation, runs the policy to generate a chunk of $T$ actions, and pushes the chunk into the queue. The two threads share state through a thread-safe `SharedStateActor`. The queue absorbs the timing mismatch — chunks arrive irregularly, but the environment always has a smooth stream of actions to consume.

### Real-Time Chunking (RTC)

When `rtc_config.enabled: true`, the inference thread uses Real-Time Chunking to start the next chunk's denoising *while* the previous chunk is still being consumed. The `execution_horizon` controls how many steps from a chunk are executed before a fresh chunk is requested; the `prefix_attention_schedule` modulates how strongly the new chunk attends to the leftover actions from the previous chunk. See [`policies/rtc/`](../../src/lerobot/policies/rtc/) for the implementation.

---

## Action Post-Processing

Between the policy's raw output and the robot, three steps run in order:

### 1. Encoding Reconstruction

If `action_encoding: anchor`, add the chunk's anchor state:

$$\hat{\mathbf{a}}_t = \mathbf{d}_t + \mathbf{s}_0$$

For `delta` encoding the network outputs are cumulatively summed first.

### 2. Zero-Phase Butterworth Filter

A 2nd-order low-pass Butterworth filter (`Wn = 0.2`) is applied along the time axis of each `[T, action_dim]` chunk using `scipy.signal.filtfilt` (zero-phase). Implementation: [`apply_butterworth_filter`](../../src/lerobot/rl/inference_utils.py) in `inference_utils.py`.

Compared to the moving-average smoother used in the $\pi_{0.5}$ pipeline, the Butterworth filter has a sharper frequency cutoff and zero phase delay, so it kills high-frequency jitter without introducing the lag a causal filter would. Chunks shorter than 9 timesteps are returned unchanged (insufficient padding length for `filtfilt`).

Smoothing matters in this pipeline because the policy is continuously generating its own training data. Jittery actions produce jittery trajectories, the next iteration's online data inherits that noise, and the policy progressively learns from a degraded distribution. Low-pass filtering breaks that feedback loop and keeps the on-policy data clean.

### 3. Per-Joint Safety Clamp

If `action_clamp_limits` is set, each joint is hard-clamped to the `[min, max]` range *in v3.0 arm frame* (after `SO101V21ToV3Step`). Out-of-bounds joints emit a `[CLAMP]` warning with the raw range, so you can spot training/inference drift before it damages the robot.

The default config sets v3.0-frame limits derived from teleop joint ranges:
```yaml
action_clamp_limits: [[-105, 105], [-102, 102], [-95, 95], [-99, 99], [-159, 170], [1, 98]]
```

---

## Buffer Caching

**Highly recommended.** MolmoAct2 stores images at full resolution (480×640×3 uint8 per camera, no resize by default); a typical 100k-transition dataset is ~270 GB of pixel data. Without a memmap cache, the offline buffer decodes everything at startup and holds it in RAM — anything beyond toy datasets OOMs.

With a cache, the pre-decoded bytes live on disk and the OS pages them in only when sampled. RAM consumption stays bounded regardless of dataset size.

### Generating the cache

One-time pre-decode:

```bash
python -m lerobot.scripts.lerobot_memmap_buffer_cache \
    --repo-id your/dataset \
    --root /path/to/local/dataset \
    --cache-dir outputs/buffer_cache \
    --image-storage-dtype uint8 \
    --image-storage-size 480 640
```

Then set `buffer_cache_dir: outputs/buffer_cache` in the YAML. If you skip this step, the first training run will populate the cache for you (slow); subsequent runs are fast.

### Cache every dataset

`additional_offline_dataset_paths` (online buffers folded back from earlier RL iterations) goes through the same cache pipeline — each path is fingerprinted and looked up separately under `buffer_cache_dir`. **Run the pre-decode script once per added dataset.** A path with no matching cache silently falls back to video decode, defeating the whole point on image-heavy datasets.

---

## Probes

Probes are diagnostic scripts that run alongside training and report on policy health: action manifold structure, representation clustering, attention maps, offline inference MSE, spatial-memorization patterns, and (when the critic is active) value-distribution statistics. See [`probes/`](../../src/lerobot/probes/) for implementations and [`metrics.md`](metrics.md) for what each probe measures.

### Enable / Disable

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `probe_parameters.enable_actions` | bool | `true` | Action manifold + PCA/UMAP. |
| `probe_parameters.enable_representations` | bool | `true` | Representation clustering. |
| `probe_parameters.enable_attention` | bool | `true` | Attention map visualization. |
| `probe_parameters.enable_offline_inference` | bool | `true` | Offline inference MSE on validation data. |
| `probe_parameters.enable_spatial_memorization` | bool | `true` | Spatial memorization probe. |
| `probe_parameters.enable_action_drift_jacobian` | bool | `false` | Per-frame causal $A^\top J$ maps (needs backward — expensive). |
| `probe_parameters.enable_spatial_memorization_jacobian` | bool | `false` | Aggregated causal spatial statistics (needs backward). |
| `probe_parameters.enable_critic_values_distribution` | bool | `false` | $V(s)$ and TD-error histograms + critic gradient magnitudes (needs backward; only meaningful when `skip_critic: false`). |

### Common

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `probe_parameters.output_dir` | str | `"outputs/probe"` | Probe output directory. |
| `probe_parameters.mode` | str | `"all"` | `"collect"` (compute + save), `"plot"` (re-render from saved data), or `"all"`. |
| `probe_parameters.max_episodes` | int\|null | `5` | Episodes per probe run. |
| `probe_parameters.n_frames_per_episode` | int | `128` | Frames sampled per episode. |
| `probe_parameters.offline_inference_n_frames` | int | `5` | Frames used by the offline inference MSE probe. |
| `probe_parameters.random_seed` | int | `42` | Probe RNG seed. |
| `probe_parameters.timestep` | float | `0.5` | Single flow-matching timestep shared by all probes. |

### Actions / Representations

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `probe_parameters.ref_max_episodes` | int | `40` | Reference set size (used as anchor for PCA/UMAP fitting). |
| `probe_parameters.ref_n_frames_per_episode` | int | `256` | Frames per reference episode. |
| `probe_parameters.action_pca_dims` | int | `50` | PCA components for action embeddings. |
| `probe_parameters.repr_pca_dims` | int | `100` | PCA components for representations. |
| `probe_parameters.umap_n_neighbors` | int | `15` | UMAP neighborhood size. |
| `probe_parameters.umap_min_dist` | float | `0.1` | UMAP minimum distance. |
| `probe_parameters.umap_seed` | int | `42` | UMAP RNG seed (separate from `random_seed`). |
| `probe_parameters.sites` | str | `"prefix,suffix"` | Token sites at which to read representations. |
| `probe_parameters.ep_3d_a` | int | `0` | Episode index A for 3D action-manifold plot. |
| `probe_parameters.ep_3d_b` | int | `1` | Episode index B for 3D action-manifold plot. |
| `probe_parameters.subtask_injection` | bool | `true` | Compare ground-truth vs generated subtask representations. |

### Attention / Spatial / Jacobian

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `probe_parameters.validation_batch_size` | int | `32` | Batch size for validation-time probes. |
| `probe_parameters.attn_eval_episodes` | str\|null | `null` | Comma-separated episode indices to render attention for. `null` = sample. |
| `probe_parameters.attn_eval_subsample` | int | `2` | Spatial subsampling factor for attention overlay rendering. |
| `probe_parameters.spatial_layers` | str | `"0,9,17"` | Layers used by spatial probes. |
| `probe_parameters.spatial_n_frames` | int | `32` | Total frames for spatial aggregation (1 per unique episode). |

### Critic Values Distribution

Active only when `enable_critic_values_distribution: true`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `probe_parameters.critic_adv_frames` | int | `1000` | Frames sampled for $V(s)$ / TD-error distributions. |
| `probe_parameters.critic_grad_frames` | int | `200` | Frames sampled for $\|\nabla_\text{vision} V\|$ (forward + backward). |

---

## Dataset Annotation (Not Implemented)

The $\pi_{0.5}$ pipeline has a full subtask annotation flow (manual web UI, Gemma 4 auto-labeler, visualization). MolmoAct2 does **not** currently have an equivalent — the policy is trained directly on the task prompt without per-frame subtask labels.

This is a known gap. If you want to experiment with subtask conditioning for MolmoAct2 you would need to:
1. Adapt the pi05 annotation scripts (in [`policies/pi05_full/annotate/`](../../src/lerobot/policies/pi05_full/annotate/)) to write subtask fields compatible with MolmoAct2's input pipeline.
2. Decide how the subtask is consumed by the VLM — as a prefix to the task prompt, as a separate token stream, or via FAST tokens.
3. Wire the new field through the offline dataset loader and the actor's `build_inference_batch`.

Until that lands, the only conditioning input is `policy.task`.

# Advanced Usage Guide

This guide covers action encodings, the full training pipeline, the iterative RL loop, intervention mechanics, async inference, and the standalone inference scripts.

---

## Table of Contents

1. [Action Encodings](#action-encodings)
2. [Training Pipeline](#training-pipeline)
3. [Configuration Reference](#configuration-reference)
4. [The Iterative RL Loop](#the-iterative-rl-loop)
5. [Dataset Preparation & Annotation](#dataset-preparation--annotation)
6. [Interventions and Real-Time Control](#interventions-and-real-time-control)
7. [Async Inference Architecture](#async-inference-architecture)
8. [Inference Scripts](#inference-scripts)
9. [Buffer Caching](#buffer-caching)

---

## Action Encodings

This is the first decision to make — it is set in the config and applies to **all** subsequent training, online and offline. Switching encodings later means recomputing statistics and retraining from base.

The pipeline supports three action representations.

### Anchor (recommended)

Each action is encoded as an offset from the initial state $\mathbf{s}_0$ of the current chunk:

$$\mathbf{d}_t = \mathbf{a}_t - \mathbf{s}_0$$

This provides **translation invariance**: the same reaching motion produces the same encoded action regardless of the arm's starting position. Recommended for most use cases — it generalizes far better than absolute and is more stable than delta.

### Absolute

Actions are raw joint positions: $\mathbf{a}_t \in \mathbb{R}^d$. No transformation applied. Simplest, but does not generalize across different starting configurations.

### Delta

First-order differences:

$$\mathbf{d}_0 = \mathbf{a}_0 - \mathbf{s}_0, \quad \mathbf{d}_t = \mathbf{a}_t - \mathbf{a}_{t-1} \text{ for } t > 0$$

More compact than anchor, but accumulated errors can drift.

### Computing Statistics

Anchor and delta encodings require precomputed normalization statistics. Generate them with:

```bash
python -m lerobot.scripts.compute_delta_stats \
    --data-dir /path/to/dataset \
    --encoding anchor
```

This iterates over all episodes, computes anchor (or delta) representations for every action chunk, and saves per-timestep statistics (min, max, mean, std, quantiles) to a `.pt` file. Set `policy.action_encoding_stats_path` to point to this file.

---

## Training Pipeline

### Phase 1: Offline Training

Offline training initializes the policy and critic from demonstrations before any robot interaction. Skipping this step makes online learning highly unstable.

```bash
python -m lerobot.scripts.offline_learner_val_pi05 --config path/to/config.json
```

This script:
- Loads the dataset from `dataset.root`
- Loads pretrained $\pi_{0.5}$ weights from `policy.pi05_checkpoint`
- Trains for `offline_steps` steps (default: 10,000) with gradient accumulation
- Runs validation probes every `val_freq` steps if `val_dataset_path` is set
- Saves checkpoints every `offline_save_freq` steps to `offline_output_dir`
- Uses `accelerate` for multi-GPU support

There is also a variant without validation probes for faster iteration:

```bash
python -m lerobot.scripts.offline_learner_pi05 --config path/to/config.json
```

### Phase 2: Online Training

Update `pi05_checkpoint` in the config to point to your offline checkpoint, then run the learner and actor on separate terminals (or machines):

**Learner** (gRPC server, runs on GPU machine):
```bash
python -m lerobot.rl.learner_pi05 --config path/to/config.json
```

**Actor** (gRPC client, runs on robot machine):
```bash
python -m lerobot.rl.actor_pi05_async --config path/to/config.json
```

The actor collects transitions at 30Hz and streams them to the learner. The learner mixes online and offline data 50/50 per batch. Updated policy weights are pushed back to the actor every `policy_parameters_push_frequency` steps (default: 180).

The learner writes an episode video to `output_dir` every `episode_logging_freq` episodes with the **predicted critic value overlaid frame-by-frame** — this is the same style of clip shown at the top of the main [README](../../README.md), so you can see when the critic thinks the policy is doing well versus when it expects failure. The online replay buffer is also dumped to disk every `episode_save_freq` episodes, so a crashed run can be resumed and recent episodes inspected or fed back into the next iteration of the loop.

---

## Configuration Reference

The config file (`config-hiserl.json`) drives all scripts. The fields below are the ones worth tuning — the rest of the config is either self-explanatory or rarely needs changing from the defaults.

### Top-Level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | str\|null | `null` | Output directory for the **online** run (checkpoints, videos, buffer dumps). Null = auto-generated timestamped dir. |
| `offline_output_dir` | str | — | Separate output dir for offline training. Keep it distinct from `output_dir` so offline and online artifacts don't overwrite each other. |
| `batch_size` | int | `8` | Replay buffer sampling batch size. Multiplied by `gradient_accumulation_steps` to get the effective batch. |
| `log_freq` | int | `20` | WandB logging frequency, in optimization steps. |
| `save_freq` | int | `100` | Online checkpoint frequency (steps). |
| `offline_save_freq` | int | `400` | Offline checkpoint frequency (steps). |

### Validation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `val_dataset_path` | str\|null | `null` | Path to a separate validation dataset. Strongly preferred. You can instead split the main dataset via `val_split`, but this reduces training data and is not recommended unless you have no held-out demos. |
| `val_split` | float | `0.0` | Fraction of training data carved off for validation when no separate val dataset is provided. |
| `val_freq` | int | `400` | Run validation every N steps. |
| `val_on_start` | bool | `true` | Run validation at step 0 to record a baseline. |

### Episode Logging

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `episode_logging_freq` | int | `4` | Log video every N episodes. |
| `episode_save_freq` | int | `10` | Save episode data (buffer dump) every N episodes. |
| `video_logging_cameras` | list | `["top", "wrist"]` | Cameras to include in logged videos. |

### Dataset

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset.root` | str | — | Local path to the annotated dataset. The dataset **must** be available locally at this path. |
| `dataset.repo_id` | str | — | HuggingFace Hub repo ID. Hub-only loading is not reliable — keep the dataset locally and rely on `dataset.root`. |
| `dataset.max_episodes` | int\|null | `null` | Cap the number of episodes loaded. Useful when the dataset is large enough to blow up RAM during the initial decode/cache step; lower it until the loader fits in memory. |
| `dataset.additional_offline_dataset_paths` | list | `[]` | Extra dataset paths to merge alongside `dataset.root`. These are typically **online buffers collected by the policy in earlier RL iterations** that have since been converted to video format and annotated with subtasks. Folding them back in is what gives the iterative loop its compounding effect — each retrain sees more diverse experience without forgetting earlier coverage. Subtask indices are auto-remapped across paths to avoid collisions. |

### Policy

**Model & Loading:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.type` | str | `"pi05_rl"` | The only supported value; this field has no other effect. |
| `policy.task` | str | — | Task prompt (e.g., "Pick up the red truck and put it in the bowl"). |
| `policy.pi05_checkpoint` | str | — | Path or HF repo to pretrained weights. Use `lerobot/pi05_base` for the base model. |
| `policy.tokenizer_max_length` | int | `64` | Token budget for the prompt, which packs **task description + state + advantage**. For a 6-DoF arm, 64 is enough. For bimanual setups (more state tokens), bump to 96 or 128. Keeping this tight saves VRAM, since attention scales with sequence length. |
| `policy.max_state_dim` | int | `6` | State vector dimension. |
| `policy.num_inference_steps` | int | `5` | Diffusion denoising steps. |
| `policy.subtask_regeneration_interval` | float | `2` | How often (seconds) the policy regenerates its high-level subtask tokens during inference. Lower values track scene changes more responsively but cost more compute; higher values give more stable subtasks. Set to `0` to regenerate every call (will overwrite any operator-injected subtask in the interactive inference script — see [Inference Scripts](#inference-scripts)). |

**RL Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.use_separate_critic` | bool | `true` | Enable critic network. |
| `policy.critic_llm_depth` | int | `6` | Number of transformer layers in the critic. |
| `policy.knowledge_insulation` | bool | `true` | Advantage only affects action decoding, not perception. |
| `policy.inference_advantage` | float | `1.0` | Fixed advantage for offline training (no critic). |
| `policy.advantage_scaling` | float | `0.20` | Scales advantage before tanh squashing. |
| `policy.discount` | float | `0.97` | Temporal discount $\gamma$. |
| `policy.critic_target_update_weight` | float | `0.005` | Polyak averaging $\tau$. |
| `policy.utd_ratio` | int | `2` | Critic updates per actor update. |
| `policy.reward_normalization_constant` | float | `5.0` | All rewards (positive and negative) are divided by this value before being fed to the critic. It keeps the value targets in a numerically stable range and, together with `discount`, sets the effective scale of the advantage. If rewards are very sparse or very dense in your task, retune this — too small and the critic saturates, too large and the learning signal vanishes. |
| `policy.terminal_failure_reward` | float | `-16.0` | Penalty applied when an episode ends in failure or timeout (before normalization). The magnitude should be large enough to dominate the discounted sum of any positive shaping rewards leading up to a failure, otherwise the policy can learn to "milk" partial credit and avoid commitment to the success state. |

**Loss Weights:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.loss_weight_flow` | float | `1.0` | Flow matching (MSE) loss weight. |
| `policy.loss_weight_action_ce` | float | `1.0` | FAST action token CE loss weight. |
| `policy.loss_weight_subtask_ce` | float | `1.0` | Subtask token CE loss weight. |

**Optimization:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.critic_lr` | float | `5e-5` | Critic learning rate. |
| `policy.actor_lr` | float | `5e-5` | Actor learning rate. |
| `policy.optimizer_weight_decay` | float | `0.01` | L2 regularization. |
| `policy.grad_clip_norm` | float | `2.0` | Max gradient norm. |
| `policy.gradient_accumulation_steps` | int | `16` | **Important for low-VRAM setups.** Combined with `batch_size`, this sets the effective batch size (e.g., 8 × 16 = 128) while only `batch_size` samples ever sit on the GPU at once. Raise this on smaller GPUs to keep the effective batch large without OOM-ing. |
| `policy.policy_update_freq` | int | `1` | Update actor every N critic updates. |

**Buffer:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.online_steps` | int | `20000` | Total online training steps. |
| `policy.offline_steps` | int | `10000` | Offline pretraining steps. |
| `policy.online_buffer_capacity` | int | `5000` | Max online transitions. |
| `policy.offline_buffer_capacity` | int | `50000` | Max offline transitions. |
| `policy.online_step_before_learning` | int | `10` | Warmup transitions before training. |

**Action Encoding:**

See [Action Encodings](#action-encodings) — this is set once and propagates through the entire pipeline, so make the choice before kicking off offline training.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.action_encoding` | str | `"anchor"` | `"absolute"`, `"anchor"`, or `"delta"`. |
| `policy.action_encoding_stats_path` | str\|null | — | Path to precomputed stats `.pt` file. Required for anchor/delta. |

**Device & Precision:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.device` | str | `"cuda"` | Main device. |
| `policy.actor_device` | str | `"cuda:0"` | Actor inference device. |
| `policy.learner_device` | str | `"cuda:0"` | Learner training device. |
| `policy.storage_device` | str | `"cpu"` | Replay buffer storage device. |
| `policy.dtype` | str | `"bfloat16"` | Computation dtype. |
| `policy.use_amp` | bool | `false` | Automatic mixed precision. |
| `policy.gradient_checkpointing` | bool | `true` | Activation checkpointing to save VRAM. |

**Actor-Learner Communication:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.actor_learner_config.learner_host` | str | — | Learner IP address. |
| `policy.actor_learner_config.learner_port` | int | `50051` | Learner gRPC port. |
| `policy.actor_learner_config.policy_parameters_push_frequency` | int | `180` | Push weights every N steps. |

**RTC (Real-Time Chunking):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.rtc_config.enabled` | bool | `true` | Enable RTC for async inference. |
| `policy.rtc_config.execution_horizon` | int | `10` | Steps ahead to plan. |
| `policy.rtc_config.prefix_attention_schedule` | str | `"LINEAR"` | Prefix attention modulation. |
| `policy.rtc_config.max_guidance_weight` | float | `10.0` | Max RTC guidance weight. |

**Trainable Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.trainable_params.vision_encoder_from_layer.vision_tower` | int\|null | `5` | Train SigLIP from this layer (null = freeze all, 27 total). |
| `policy.trainable_params.vision_encoder_from_layer.multi_modal_projector` | bool | `true` | Train image-to-embedding projector. |
| `policy.trainable_params.language_from_layer` | int\|null | `0` | Train Gemma from this layer (null = freeze all, 18 total). |
| `policy.trainable_params.critic_language_from_layer` | int\|null | `1` | Train critic layers from this index. |

### Environment

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `env.type` | str | `"gym_manipulator"` | Environment class. |
| `env.name` | str | `"real_robot"` | Environment identifier. |
| `env.fps` | int | `30` | Control loop frequency. |
| `env.task` | str | — | Task description (mirrors `policy.task`). |

**Robot:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `env.robot.type` | str | — | Robot model (e.g., `"so101_follower"`). |
| `env.robot.port` | str | — | Serial port (e.g., `"/dev/ttyACM0"`). |
| `env.robot.cameras` | dict | — | Camera definitions with type, index, resolution, fps. |

**Teleop:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `env.teleop.type` | str | — | Controller type (e.g., `"so101_leader"`). |
| `env.teleop.port` | str | — | Serial port for leader arm. |

**Processor:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `env.processor.control_mode` | str | `"leader"` | `"leader"` (copy leader state) or `"policy"` (policy-controlled). |
| `env.processor.image_preprocessing.resize_size` | list | `[224, 224]` | Image resize target. |
| `env.processor.gripper.use_gripper` | bool | `true` | Enable gripper control. |
| `env.processor.reset.fixed_reset_joint_positions` | list | — | Home position joint angles (degrees). |
| `env.processor.reset.reset_time_s` | float | `10` | Duration of reset motion. |
| `env.processor.reset.control_time_s` | float | `200.0` | Max episode duration. |
| `env.processor.reset.terminate_on_success` | bool | `true` | End episode on success. |
| `env.processor.reward_classifier.pretrained_path` | str\|null | `null` | Reward classifier model path. |
| `env.processor.reward_classifier.success_threshold` | float | `0.5` | Classification threshold. |
| `env.processor.reward_classifier.success_reward` | float | `1.0` | Reward value on success. |

### Probe Parameters

See [Validation Metrics](metrics.md) for detailed descriptions of each probe. Below is the config reference.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `probe_parameters.enable_actions` | bool | `true` | Action manifold probe. |
| `probe_parameters.enable_representations` | bool | `true` | Representation clustering probe. |
| `probe_parameters.enable_attention` | bool | `true` | Attention visualization probe. |
| `probe_parameters.enable_offline_eval` | bool | `true` | Offline inference MSE probe. |
| `probe_parameters.enable_spatial_memorization` | bool | `true` | Spatial memorization probe. |
| `probe_parameters.enable_action_drift_jacobian` | bool | `true` | Jacobian causal map probe. |
| `probe_parameters.enable_spatial_memorization_jacobian` | bool | `false` | Jacobian + spatial memorization probe. |
| `probe_parameters.output_dir` | str | `"outputs/probe"` | Probe output directory. |
| `probe_parameters.mode` | str | `"all"` | `"collect"`, `"plot"`, or `"all"`. |
| `probe_parameters.max_episodes` | int | `5` | Episodes per probe run. |
| `probe_parameters.n_frames_per_episode` | int | `128` | Frames per episode. |
| `probe_parameters.timestep` | float | `0.5` | Diffusion timestep for all probes. |
| `probe_parameters.action_pca_dims` | int | `50` | PCA components for actions. |
| `probe_parameters.repr_pca_dims` | int | `100` | PCA components for representations. |
| `probe_parameters.umap_n_neighbors` | int | `15` | UMAP neighborhood size. |
| `probe_parameters.umap_min_dist` | float | `0.1` | UMAP minimum distance. |
| `probe_parameters.spatial_layers` | str | `"0,9,17"` | Layers for spatial probes. |
| `probe_parameters.spatial_n_frames` | int | `32` | Frames for spatial aggregation. |
| `probe_parameters.subtask_injection` | bool | `false` | Compare GT vs generated subtask representations. |

---

## The Iterative RL Loop

Following RECAP's recommendation, we retrain from the base model each iteration, including all accumulated data to avoid policy drift.

### Cycle

1. **Offline training** on existing demonstrations.
2. **Online training** (actor + learner) to collect new experience. The learner periodically dumps the online replay buffer to disk under `output_dir`.
3. **Convert** the dumped buffer into a video-format LeRobot dataset.
4. **Annotate** the converted dataset with subtasks (manually via web UI, or automatically with an LLM).
5. **Merge** by adding the annotated path to `dataset.additional_offline_dataset_paths`.
6. **Retrain from the base $\pi_{0.5}$ model** with all data combined — old demos, prior cycles' on-policy data, and the new annotated buffer.

Each cycle adds more diverse experience. The `additional_offline_dataset_paths` field supports multiple paths — subtask indices are automatically remapped across paths to avoid collisions when merging.

Steps 3 and 4 are covered in detail in [Dataset Preparation & Annotation](#dataset-preparation--annotation).

---

## Dataset Preparation & Annotation

The same scripts produce both the initial annotated dataset required by the quick start (≥50 episodes with subtask labels) and the per-iteration data folded in via `additional_offline_dataset_paths`.

### Convert the online buffer to a video dataset

The learner saves online transitions with raw images. Convert them to a video-format LeRobot dataset before annotating:

```bash
python -m lerobot.policies.pi05_full.annotate.online_buffer_to_video \
    --data-dir /path/to/output_dir/dataset \
    --output-dir /path/to/output_dir/dataset_video
```

`--data-dir` is the buffer dump written by the learner under its `output_dir`. Datasets recorded with `lerobot_record` are already in video format and skip this step.

### Annotate with subtasks

Two options, same on-disk result.

**Manual (web UI):**

```bash
python -m lerobot.policies.pi05_full.annotate.manual_subtask_annotate
```

Launches a Gradio app where you scrub through episodes and assign skills from a palette defined in [`skills.yaml`](../../src/lerobot/policies/pi05_full/annotate/skills.yaml). Edit that file to match your task vocabulary before annotating. An external alternative is [lerobot-data-studio](https://github.com/jackvial/lerobot-data-studio).

**LLM (Gemma 4):**

```bash
python -m lerobot.policies.pi05_full.annotate.subtask_annotate_gemma_4 \
    --data-dir /path/to/your/dataset \
    --video-key observation.images.wrist \
    --batch-size 5 \
    --output-dir /output/path
```

`--video-key` picks the camera the model sees; the wrist view usually works best for manipulation. `--batch-size` controls VRAM.

### Inspect the result

Before training on freshly annotated data, render one episode with labels burned in:

```bash
python -m lerobot.policies.pi05_full.annotate.visualize_annotations \
    --dataset /path/to/annotated_dataset \
    --output  /path/to/visualization_ep0.mp4 \
    --episode 0
```

If the segmentation looks off, fix it in the web UI or rerun Gemma 4 with a different `--video-key` before committing to a training run.

### Wire it in

Point `dataset.root` at the annotated dataset for a fresh run, or append the path to `dataset.additional_offline_dataset_paths` to merge it with existing data. Subtask indices are remapped across paths automatically.

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

During the actor update, transitions marked as interventions get their advantage overridden to $A = 1.0$ (maximum positive). This tells the policy: "this is what you should have done." The same override applies to golden dataset (demonstration) transitions.

### Re-engagement

When the human releases control (presses `5` again), the inference thread resumes immediately with a fresh action queue, preventing jerky transitions from stale predictions.

---

## Async Inference Architecture

The async pipeline decouples neural network inference (slow, variable latency) from robot control (strict 30Hz).

Two daemon threads run concurrently: an **environment thread** that maintains the 30Hz control loop, pops one action at a time from the `ActionQueue`, sends it to the robot, and handles episode boundaries; and an **inference thread** that pulls the latest observation, runs the policy to generate a chunk of $T$ actions, and pushes the chunk into the queue. The two threads share state through a thread-safe `SharedStateActor`. The queue absorbs the timing mismatch — chunks arrive irregularly, but the environment always has a smooth stream of actions to consume.

### Action Smoothing

A centered moving average (window size 5) is applied to the absolute action values after anchor/delta reconstruction:

$$\hat{a}_t = \frac{1}{5}\sum_{k=-2}^{2} a_{t+k}$$

with edge padding. Smoothing matters more here than in standard imitation learning because the policy is continuously generating its own training data. Jittery actions produce jittery trajectories, the next round of online data inherits that noise, and the policy progressively learns from a worse and worse distribution. A small amount of low-pass filtering breaks that feedback loop and keeps the on-policy data clean.

---

## Inference Scripts

Two standalone scripts run the policy without any learner or gRPC connection — useful for evaluation, deployment, and debugging a checkpoint in isolation.

**Standard async inference:**
```bash
python -m lerobot.rl.inference_pi05_async --config path/to/config.json
```

Same threading model and action queue as the actor, but no transitions are streamed anywhere. The script still writes episode videos every `episode_logging_freq` episodes and dumps the buffer every `episode_save_freq` episodes, so a session can be reviewed afterward.

**Interactive subtask injection:**
```bash
python -m lerobot.rl.inference_pi05_async_interactive --config path/to/config.json
```

Identical to the standard script, except the operator can type a subtask string into the terminal at any time. The text is tokenized and injected into the policy's subtask token cache, taking effect on the very next action chunk; the model's normal time-based subtask cache then resumes. Requires `subtask_regeneration_interval > 0` (e.g. 30) — otherwise the model regenerates subtask tokens every cycle and the override is overwritten before it takes effect.

Both scripts expect the same config used for online training and load weights from `policy.pi05_checkpoint`.

---

## Buffer Caching

For large datasets, decoding video frames at every training step is expensive. The memmap buffer cache pre-decodes everything to raw binary files:

```bash
python -m lerobot.scripts.lerobot_memmap_buffer_cache \
    --repo-id your/dataset \
    --cache-dir outputs/buffer_cache
```

This creates memory-mapped `.bin` files (one per tensor: images, actions, states, etc.) with a `metadata.json` recording shapes and dtypes. The OS virtual memory system manages paging, so 50GB+ datasets work on machines with much less free RAM.

Set `buffer_cache_dir` in the config to enable automatic cache detection via fingerprint matching.

# Advanced Usage Guide

This guide covers the full training pipeline, configuration reference, action encodings, the iterative RL loop, intervention mechanics, and async inference — everything beyond the quick start in the README.

---

## Table of Contents

1. [Training Pipeline](#training-pipeline)
2. [Configuration Reference](#configuration-reference)
3. [Action Encodings](#action-encodings)
4. [The Iterative RL Loop](#the-iterative-rl-loop)
5. [Interventions and Real-Time Control](#interventions-and-real-time-control)
6. [Async Inference Architecture](#async-inference-architecture)
7. [Buffer Caching](#buffer-caching)
8. [Frozen Parameters & Memory Optimization](#frozen-parameters--memory-optimization)

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
- Saves checkpoints every `save_freq` steps to `offline_output_dir`
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

The actor collects transitions at 30Hz and streams them to the learner via gRPC. The learner mixes online and offline data 50/50 per batch. Updated policy weights are pushed back to the actor every `policy_parameters_push_frequency` steps (default: 180).

### Inference Only

```bash
python -m lerobot.rl.inference_pi05_async --config path/to/config.json
```

Same async architecture as the actor but without gRPC communication or data collection. Useful for evaluation.

---

## Configuration Reference

The config file (`config-hiserl.json`) drives all scripts. Below is a reference for every field, organized by section.

### Top-Level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | str\|null | `null` | Output directory. Null = auto-generate timestamped dir. |
| `offline_output_dir` | str | — | Separate output dir for offline training. |
| `job_name` | str | `"default"` | Run identifier for logging. |
| `resume` | bool | `false` | Resume from existing checkpoint in `output_dir`. |
| `seed` | int | `42` | Random seed. |
| `num_workers` | int | `4` | DataLoader workers. |
| `batch_size` | int | `8` | Replay buffer sampling batch size. |
| `log_freq` | int | `20` | Log metrics every N steps. |
| `save_checkpoint` | bool | `true` | Save model checkpoints. |
| `save_freq` | int | `100` | Checkpoint frequency (steps). |
| `offline_save_freq` | int | `400` | Checkpoint frequency for offline training. |
| `use_rerun` | bool | `true` | Enable Rerun.io live visualization. |

### Validation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `val_dataset_path` | str\|null | `null` | Path to separate validation dataset. |
| `val_split` | float | `0.0` | Fraction of training data for validation (if no separate val dataset). |
| `val_freq` | int | `400` | Run validation every N steps. |
| `val_on_start` | bool | `true` | Run validation at step 0 (baseline). |

### WandB

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wandb.enable` | bool | `true` | Enable Weights & Biases logging. |
| `wandb.project` | str | — | W&B project for online training. |
| `wandb.offline_project` | str | — | W&B project for offline training. |
| `wandb.disable_artifact` | bool | `true` | Don't upload model artifacts. |

### Episode Logging

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `episode_logging_freq` | int | `4` | Log video every N episodes. |
| `episode_save_freq` | int | `10` | Save episode data every N episodes. |
| `video_logging_cameras` | list | `["top", "wrist"]` | Cameras to include in logged videos. |

### Dataset

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset.root` | str | — | Local path to the annotated dataset. |
| `dataset.repo_id` | str | — | HuggingFace Hub repo ID (fallback if local path missing). |
| `dataset.use_imagenet_stats` | bool | `false` | Use ImageNet stats for image normalization. |
| `dataset.max_episodes` | int\|null | `null` | Limit to N episodes (null = all). |
| `dataset.additional_offline_dataset_paths` | list | `[]` | Extra dataset paths to merge. Subtask indices are auto-remapped to avoid collisions. |

### Policy

**Model & Loading:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.type` | str | `"pi05_rl"` | Policy type. |
| `policy.task` | str | — | Task prompt (e.g., "Pick up the red truck and put it in the bowl"). |
| `policy.pi05_checkpoint` | str | — | Path or HF repo to pretrained weights. Use `lerobot/pi05_base` for base model. |
| `policy.tokenizer_max_length` | int | `64` | Max tokenizer length. |
| `policy.max_state_dim` | int | `6` | State vector dimension. |
| `policy.num_inference_steps` | int | `5` | Diffusion denoising steps. |

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
| `policy.reward_normalization_constant` | float | `5.0` | Divide rewards by this constant. |
| `policy.terminal_failure_reward` | float | `-16.0` | Penalty on episode failure/timeout. |

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
| `policy.gradient_accumulation_steps` | int | `16` | Accumulate gradients before stepping. |
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

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy.action_encoding` | str | `"anchor"` | `"absolute"`, `"anchor"`, or `"delta"`. See [Action Encodings](#action-encodings). |
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

## Action Encodings

The pipeline supports three action representations. The choice affects how the model sees and predicts actions.

### Absolute

Actions are raw joint positions: $\mathbf{a}_t \in \mathbb{R}^d$. No transformation applied. This is the simplest but doesn't generalize well across different starting configurations.

### Anchor

Each action is encoded as an offset from the initial state $\mathbf{s}_0$ of the current chunk:

$$\mathbf{d}_t = \mathbf{a}_t - \mathbf{s}_0$$

This provides **translation invariance**: the same reaching motion produces the same encoded action regardless of the arm's starting position. Recommended for most use cases.

### Delta

First-order differences:

$$\mathbf{d}_0 = \mathbf{a}_0 - \mathbf{s}_0, \quad \mathbf{d}_t = \mathbf{a}_t - \mathbf{a}_{t-1} \text{ for } t > 0$$

More compact than anchor but accumulated errors can drift.

### Computing Statistics

Anchor and delta encodings require precomputed normalization statistics. Generate them with:

```bash
python -m lerobot.scripts.compute_delta_stats \
    --data-dir /path/to/dataset \
    --encoding anchor
```

This iterates over all episodes, computes anchor (or delta) representations for every action chunk, and saves per-timestep statistics (min, max, mean, std, quantiles) to a `.pt` file. Set `policy.action_encoding_stats_path` to point to this file.

### Anchor Alignment During RTC

When a new action chunk is generated during real-time inference, leftover actions from the previous chunk reference the old anchor state $\mathbf{s}_0^{\text{old}}$. If the robot has moved, these need re-alignment:

$$\mathbf{d}_t^{\text{new}} = \mathbf{d}_t^{\text{old}} + (\mathbf{s}_0^{\text{old}} - \mathbf{s}_0^{\text{new}})$$

This correction is applied automatically in the action queue merging logic.

---

## The Iterative RL Loop

Following RECAP's recommendation, we retrain from the base model each iteration, including all accumulated data to avoid policy drift.

### Cycle

1. **Offline training** on existing demonstrations
2. **Online training** (actor + learner) to collect new experience
3. **Convert** online buffer to video format:
   ```bash
   python -m lerobot.policies.pi05_full.annotate.online_buffer_to_video \
       --data-dir /path/to/online_buffer \
       --output-dir /path/to/online_buffer_video
   ```
4. **Annotate** the new data with subtasks (Gemma 4 or manual)
5. **Merge** by adding the path to `dataset.additional_offline_dataset_paths`
6. **Retrain from base model** with all data combined

Each cycle adds more diverse experience. The `additional_offline_dataset_paths` field supports multiple paths — subtask indices are automatically remapped to avoid collisions when merging datasets.

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

When the human releases control (presses `5` again), the inference thread resumes immediately with a fresh action queue. The action queue is cleared to prevent jerky transitions from stale predictions.

---

## Async Inference Architecture

The async pipeline decouples neural network inference (slow, variable latency) from robot control (strict 30Hz).

### Threading Model

Two daemon threads run concurrently:

**Environment thread** (`env_interaction_worker`):
- Maintains a strict 30Hz loop using high-resolution sleep
- Pops one action per step from the `ActionQueue`
- Sends the action to the robot
- Records transitions
- Handles episode boundaries, resets, and interventions

**Inference thread** (`get_actions_worker`):
- Runs asynchronously with no timing constraints
- Pulls the latest observation from `SharedStateActor` (thread-safe)
- Runs policy inference to generate an action chunk ($T$ actions)
- Pushes the chunk to the `ActionQueue` with latency compensation
- Sleeps when the queue is full (backpressure)

### Action Queue and RTC

The `ActionQueue` bridges the timing gap:

- The policy generates chunks of $T$ actions (default: 50) at irregular intervals
- The environment consumes 1 action per step at fixed 30Hz
- RTC mode: when a new chunk arrives, the first `delay` actions are discarded (they've already been executed while inference was running). The delay is estimated using a sliding window p95 latency tracker.
- Non-RTC mode: new chunks are appended to the existing queue.

### Action Smoothing

A centered moving average (window size 5) is applied to reduce jitter:

$$\hat{a}_t = \frac{1}{5}\sum_{k=-2}^{2} a_{t+k}$$

with edge padding. Applied after anchor/delta reconstruction on the absolute action values.

### Latency Tracking

The `LatencyTracker` maintains a sliding window (last 100 inference times) and reports the **p95 percentile** (not max) to avoid single spikes causing excessive lookahead. The inference thread uses this to:
- Discard already-executed actions when merging chunks
- Sleep when the queue has enough actions buffered

### gRPC Protocol

The actor and learner communicate via gRPC (HTTP/2):

| RPC | Direction | Purpose |
|-----|-----------|---------|
| `StreamParameters` | Learner -> Actor | Push updated policy weights |
| `SendTransitions` | Actor -> Learner | Stream collected transitions |
| `SendInteractions` | Actor -> Learner | Episode statistics (reward, intervention rate) |
| `Ready` | Actor -> Learner | Handshake at startup |

Transitions are serialized with PyTorch pickle and sent in chunks to avoid gRPC message size limits. The actor retries connection up to 30 times at startup.

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

---

## Frozen Parameters & Memory Optimization

Training the full $\pi_{0.5}$ + critic requires substantial VRAM. Several strategies reduce memory usage:

1. **Selective layer freezing**: Only upper transformer layers are trained (configurable via `trainable_params`). The action expert is always fully trained.

2. **Gradient checkpointing**: When `gradient_checkpointing: true`, intermediate activations are recomputed during the backward pass instead of stored, trading compute for memory.

3. **Gradient accumulation**: With `gradient_accumulation_steps: 16` and `batch_size: 8`, the effective batch size is 128 while only 8 samples are in GPU memory at once.

4. **CPU storage**: Setting `storage_device: "cpu"` keeps the replay buffer in system RAM, freeing GPU memory for the model.

5. **bfloat16**: Using `dtype: "bfloat16"` halves the memory footprint of model weights and activations.

The default config trains SigLIP layers 5-26, all Gemma language layers, the full action expert, and critic layers 1-5. Adjust the `trainable_params` section to freeze more layers if running on smaller GPUs.

# RECAP Online Learner Architecture

This document describes the current architecture for running online RL training with RECAP using RTC (Real-Time Chunking) async inference. The system is a two-machine actor-learner split connected over gRPC.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Actor Machine (Robot + GPU)              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │               actor_pi05_async.py                     │    │
│  │                                                       │    │
│  │  ┌─────────────────┐       ┌──────────────────────┐   │    │
│  │  │  Env Thread      │       │  Inference Thread    │   │    │
│  │  │  (30 Hz loop)    │       │  (GPU, async)        │   │    │
│  │  │                  │       │                      │   │    │
│  │  │  Steps robot     │       │  Runs pi05 forward   │   │    │
│  │  │  Pops actions    │       │  Generates chunks    │   │    │
│  │  │  Builds trans.   │       │  Pulls weights       │   │    │
│  │  │  Handles teleop  │       │  Aligns anchors      │   │    │
│  │  └────────┬─────────┘       └──────────┬───────────┘   │    │
│  │           │                            │               │    │
│  │           │     SharedStateActor        │               │    │
│  │           │    ┌────────────────┐       │               │    │
│  │           └───►│  latest_obs    │◄──────┘               │    │
│  │                │  episode flags │                        │    │
│  │                │  subtask cache │                        │    │
│  │                └────────────────┘                        │    │
│  │           │                            │               │    │
│  │           ▼                            ▼               │    │
│  │        ActionQueue                parameters_queue      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  gRPC background threads:                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │ send_trans.  │ │ send_inter.  │ │ receive_policy       │  │
│  │ (client strm)│ │ (client strm)│ │ (server strm sub.)   │  │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘  │
│         │                │                     │              │
└─────────┼────────────────┼─────────────────────┼──────────────┘
          │                │                     │
          │         gRPC (network)               │
          ▼                ▼                     │
┌─────────────────────────────────────────────────────────────┐
│                    Learner Machine (GPU)                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Communication Thread (gRPC Server)                    │    │
│  │                                                       │    │
│  │  LearnerService (ThreadPoolExecutor, max_workers=3)   │    │
│  │  ┌──────────────────┐                                 │    │
│  │  │ StreamParameters │───► parameters_queue ◄── train  │    │
│  │  │ SendTransitions  │───► transition_queue  ──► train  │    │
│  │  │ SendInteractions │───► interaction_queue  ──► train │    │
│  │  │ Ready            │    (handshake)                   │    │
│  │  └──────────────────┘                                 │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Training Thread (main)                                │    │
│  │                                                       │    │
│  │  Models in GPU memory:                                │    │
│  │    - Policy (pi05 full, ~3B params)                   │    │
│  │    - Critic (6-layer Gemma transformer)               │    │
│  │    - Critic Target (EMA copy)                         │    │
│  │                                                       │    │
│  │  Buffers on storage_device (CPU):                     │    │
│  │    - Online Replay Buffer                             │    │
│  │    - Offline Replay Buffer                            │    │
│  │                                                       │    │
│  │  Training loop:                                       │    │
│  │    1. Drain transition_queue -> online buffer          │    │
│  │    2. Drain interaction_queue -> log episode stats     │    │
│  │    3. (utd_ratio - 1) x critic-only Bellman updates   │    │
│  │    4. EMA update critic_target                        │    │
│  │    5. Joint critic + actor update (flow matching)      │    │
│  │    6. Push trainable weights -> parameters_queue       │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Actor Process

**Entry point:** `python -m lerobot.rl.actor_pi05_async --config_path config-recap.json`

**Source files:**
- `lerobot/rl/actor_pi05_async.py` — process orchestration and thread spawning
- `lerobot/rl/actor_pi05_async_utils.py` — inference thread, env thread, SharedStateActor, anchor alignment
- `lerobot/rl/actor.py` — gRPC client functions (send_transitions, send_interactions, receive_policy)

### Thread Model

The actor runs 5 threads/processes total:

```
actor_cli()
│
├── 1. receive_policy       Thread    gRPC StreamParameters subscriber
├── 2. send_transitions     Thread    gRPC SendTransitions client stream
├── 3. send_interactions    Thread    gRPC SendInteractions client stream
│
└── act_with_policy_async()           Runs on main thread/process
    │
    ├── 4. environment_thread  Thread    Robot control loop (30 Hz)
    ├── 5. inference_thread    Thread    GPU inference (async)
    │
    └── main loop                       Monitors metrics, checks shutdown
```

Threads 1-3 can alternatively be `Process` instances depending on `cfg.policy.concurrency.actor`.

### Thread 4 — Environment Thread (`env_interaction_worker_actor`)

This is the real-time robot control loop. It runs at `cfg.env.fps` (typically 30 Hz) using `precise_sleep`.

**Per-tick responsibilities:**
1. Check episode boundary — wait for teleop key `2` to start a new episode
2. Detect intervention state changes — if toggled, request policy reset and flush the action queue
3. Get action:
   - If intervening: read raw joint positions from the leader arm
   - If not intervening: pop from `ActionQueue` (postprocess if absolute encoding)
   - If queue empty and not intervening: hold current joint position
4. Step the environment via `step_env_and_process_transition`
5. Read reward, done/truncated flags, intervention state from the step result
6. Build a `Transition` object:
   - `state`: current observation (images + joint state) in policy format
   - `action`: the executed action (6-DOF joint positions)
   - `reward`: scalar from reward classifier or manual signal
   - `next_state`: next observation
   - `complementary_info`: subtask tokens/masks (from SharedState cache), intervention flag, discrete penalty, subtask index (-1 for online)
7. Append transition to episode list
8. Update `SharedStateActor` with the new observation
9. On episode end (done or truncated):
   - Flush all accumulated transitions to `transitions_queue`
   - Push episode stats (reward, intervention rate) to `interactions_queue`
   - Reset counters
10. Sleep for remainder of tick

**Teleop key bindings** (read from the teleop device each tick):
- `5` — toggle intervention (take/release leader arm control)
- `1` — mark success, terminate episode
- `0` — mark failure, terminate episode
- `2` — start next episode

### Thread 5 — Inference Thread (`get_actions_worker_actor`)

Runs continuously, generating action chunks on the GPU whenever the action queue needs refilling.

**Per-iteration responsibilities:**
1. Check `update_parameters_requested` flag — if set, pull new weights from `parameters_queue` via `pull_new_policy_weights`
2. Skip if episode not active or human is intervening
3. Check if queue needs refilling: `qsize > execution_horizon + latency_delay` → sleep and continue
4. Read latest observation from `SharedStateActor`
5. Build preprocessor input:
   - Filter to `input_features` keys
   - Inject `complementary_data` with task string, empty subtask, and fixed `inference_advantage` scalar (default 1.0)
6. Get leftover actions from `ActionQueue` for RTC in-painting
7. If anchor/delta encoding: run `align_prev_actions` to correct leftover actions for stale anchor state
8. Compute `inference_delay` using p95 latency estimate
9. Run `policy.predict_action_chunk(processed_batch, inference_delay, prev_chunk_left_over, execution_horizon)` on GPU
10. Cache subtask tokens from the policy into `SharedStateActor` for the env thread
11. Reconstruct absolute actions from anchor/delta encoding
12. Apply 3-tap centered moving average for smoothing
13. Merge chunk into `ActionQueue` via `action_queue.merge()`

### SharedStateActor

Lock-protected object that bridges the env thread and inference thread:

| Field | Writer | Reader | Purpose |
|---|---|---|---|
| `latest_obs` | Env thread | Inference thread | Latest observation dict |
| `is_intervening` | Env thread | Inference thread | Human control active |
| `episode_active` | Env thread | Inference thread | Episode in progress |
| `policy_reset_requested` | Env thread | Inference thread | Request GRU state reset |
| `update_parameters_requested` | Env thread | Inference thread | Request weight pull |
| `cached_subtask_tokens` | Inference thread | Env thread | VLA-generated subtask tokens |
| `cached_subtask_masks` | Inference thread | Env thread | Subtask attention masks |
| `current_step` | Env thread | Main thread | Global interaction step |
| Metrics accumulators | Both | Main thread | Latency, wait times |

### ActionQueue

Thread-safe action buffer from `lerobot/policies/rtc/action_queue.py`. Holds two parallel queues:
- `original_queue` — raw normalized actions (used for RTC in-painting via `get_left_over`)
- `queue` — postprocessed absolute actions (used by env thread via `get`)

The `merge()` method replaces the queue contents with the new chunk, accounting for `real_delay` (actions consumed during inference).

### gRPC Transport Threads

**Thread 1 — `receive_policy`:**
- Subscribes to `LearnerService.StreamParameters(Empty)` server-streaming RPC
- Receives chunked bytes via `receive_bytes_in_chunks`
- Pushes deserialized weight bytes to `parameters_queue` (maxsize=2)
- The inference thread pops from this queue when `update_parameters_requested` is set

**Thread 2 — `send_transitions`:**
- Reads from `transitions_queue` (populated by env thread at episode end)
- Serializes transition list via `transitions_to_bytes` (PyTorch pickle + tensor data)
- Streams via `LearnerService.SendTransitions` client-streaming RPC
- Uses `send_bytes_in_chunks` to split large payloads into gRPC-friendly pieces

**Thread 3 — `send_interactions`:**
- Reads from `interactions_queue` (populated by env thread at episode end)
- Serializes episode stats dict via `python_object_to_bytes` (pickle)
- Streams via `LearnerService.SendInteractions` client-streaming RPC

### Actor Model Configuration

The actor forces `use_separate_critic = False` before constructing the policy. This means:
- Only the policy backbone (pi05 full model) is loaded — no critic or critic target
- Advantage conditioning uses a fixed scalar `inference_advantage` (default 1.0)
- The actor never computes value estimates

### Weight Loading

`pull_new_policy_weights` receives a state dict under key `"policy"` containing only trainable parameters. It loads with `strict=False`, so frozen parameter keys (which aren't sent) are silently skipped. The trainable set is determined by the learner's `trainable_params` config.

## Learner Process

**Entry point:** `python -m lerobot.rl.learner_pi05 --config_path config-recap.json`

**Source files:**
- `lerobot/rl/learner_pi05.py` — pi05-specific training loop and transition processing
- `lerobot/rl/learner.py` — shared learner utilities (gRPC server start, buffer init, checkpoint save)
- `lerobot/rl/learner_service.py` — `LearnerService` gRPC service implementation
- `lerobot/rl/pi05_train_utils.py` — shared training logic (Bellman update, flow matching, preprocessing)
- `lerobot/rl/rl_pi05.py` — `PI05RLConfig`, `PI05RLPolicy`, `Pi05TransformerCritic`

### Thread Model

The learner runs 2 threads/processes:

```
train_cli()
└── train()
    └── start_learner_threads()
        │
        ├── 1. communication_process    Thread/Process    gRPC server
        │       │
        │       └── LearnerService (grpc.server with ThreadPoolExecutor(3))
        │           ├── StreamParameters handler    (reads parameters_queue)
        │           ├── SendTransitions handler     (writes transition_queue)
        │           └── SendInteractions handler    (writes interaction_queue)
        │
        └── 2. add_actor_information_and_train()    Main thread    Training loop
```

### Thread 1 — Communication Process (`start_learner`)

Starts a gRPC server on `learner_port` (default 50051) with a `ThreadPoolExecutor(max_workers=3)`.

The server hosts `LearnerService` which implements four RPCs:

| RPC | Type | Direction | Queue |
|---|---|---|---|
| `StreamParameters` | Server-streaming | Learner → Actor | Reads from `parameters_queue` |
| `SendTransitions` | Client-streaming | Actor → Learner | Writes to `transition_queue` |
| `SendInteractions` | Client-streaming | Actor → Learner | Writes to `interaction_queue` |
| `Ready` | Unary | Actor → Learner | Handshake, returns `Empty` |

`StreamParameters` respects `seconds_between_pushes` (= `policy_parameters_push_frequency` from config) to throttle weight pushes. It pops the latest entry from `parameters_queue` (dropping stale ones) and streams it in chunks.

### Thread 2 — Training Loop (`add_actor_information_and_train`)

This is the main thread and runs a synchronous training loop.

#### Initialization

1. Create the full `PI05RLPolicy` with `use_separate_critic=True`:
   - `policy.actor` / `policy.model` — pi05 action model (~3B params)
   - `policy.critic` — `Pi05TransformerCritic` (configurable depth, default 4 layers)
   - `policy.critic_target` — EMA copy of critic
2. Enable gradient checkpointing if configured
3. Push initial weights to `parameters_queue` for the actor's first load
4. Freeze parameters based on `trainable_params` config:
   - Action expert layers (projections, time MLPs, gemma expert): always trainable
   - Vision tower: trainable from `vision_encoder_from_layer.vision_tower` onwards (default: fully frozen)
   - Multi-modal projector: trainable if `vision_encoder_from_layer.multi_modal_projector` is true (default: frozen)
   - Language model: trainable from `language_from_layer` onwards (e.g., layers 15-17 of 18)
   - Critic: norm, value_head, value_queries always trainable; layers trainable from `critic_language_from_layer` onwards
5. Share underlying memory for frozen critic/critic_target layers (VRAM optimization)
6. Create optimizers (separate param groups for actor and critic)
7. Initialize replay buffers:
   - **Online buffer**: empty, capacity from `online_buffer_capacity`
   - **Offline buffer**: loaded from annotated dataset via `ReplayBuffer.from_lerobot_dataset`, tagged with `is_golden: True`
   - Additional offline datasets loaded and merged via `load_additional_offline_datasets`
8. Create preprocessor and postprocessor via `make_pi05_full_processors_with_upgrade`

#### Main Loop (per optimization step)

```
while optimization_step < online_steps:
    │
    ├── 1. process_interaction_messages()
    │       Drain interaction_queue, log episode stats to WandB
    │
    ├── 2. process_transitions_pi05()
    │       Drain transition_queue, deserialize, add to online buffer
    │       On logging episodes: save images, run critic forward, generate video
    │
    ├── 3. Check: len(replay_buffer) >= online_step_before_learning?
    │       If not, continue (spin-wait for actor data)
    │
    ├── 4. (utd_ratio - 1) x critic-only updates:
    │       _update_critic(policy, optimizers, online_iter, offline_iter, ...)
    │       policy.update_target_networks()  (EMA)
    │
    ├── 5. Joint update (critic + actor):
    │       pi05_update_step(policy, optimizers, ...)
    │       - Sample batch (online + offline mixed)
    │       - Preprocess with action encoding transforms
    │       - Compute advantage: target_v - current_v (from critic)
    │       - Critic loss: Bellman TD error
    │       - Actor loss: flow matching + action CE + subtask CE
    │       - Gradient clipping and optimizer step
    │
    ├── 6. Log metrics (every log_freq steps)
    │
    ├── 7. Save checkpoint (every save_freq steps)
    │
    ├── 8. Save online buffer to disk (every episode_save_freq episodes)
    │
    └── 9. Push weights to actor (every policy_parameters_push_frequency seconds)
            push_actor_policy_to_queue_pi05()
            - Iterates policy.actor.named_parameters()
            - Filters to requires_grad only
            - Serializes and puts on parameters_queue
```

#### Transition Processing (`process_transitions_pi05`)

Called each loop iteration to drain `transition_queue`:

1. Pop a serialized episode (list of transitions) from the queue
2. Deserialize via `bytes_to_transitions`
3. Increment episode counter
4. If logging episode (every `episode_logging_freq`):
   - Save per-frame images from observations
   - Run critic forward pass per frame to get value estimates
   - Generate video with critic overlay via `save_video_with_critic_overlay`
5. For each transition:
   - NaN check — skip if any observation or action contains NaN
   - Action dimension mismatch fix (pad or slice to match buffer)
   - `replay_buffer.add(**transition)`

#### Weight Push (`push_actor_policy_to_queue_pi05`)

Only sends **trainable actor parameters** to minimize bandwidth:

```python
trainable_state_dict = {}
for name, param in policy.actor.named_parameters():
    if param.requires_grad:
        trainable_state_dict[name] = param

state_dicts = {"policy": move_state_dict_to_device(trainable_state_dict, device="cpu")}
```

Critic weights are never sent — the actor doesn't have a critic.

## Replay Buffers

Both buffers live in the learner process on `storage_device` (typically CPU).

### Online Replay Buffer

- **Populated by:** transitions streamed from the actor over gRPC
- **Storage:** bfloat16 tensors, lazy-initialized on first `add()` call
- **Memory optimization:** `next_states` is a shifted view into `states` (halves image memory). Episode boundaries tracked via `episode_ends` tensor to prevent cross-episode sampling.
- **Complementary info stored per transition:**
  - `discrete_penalty` — gripper penalty
  - `is_intervention` — flag for human-controlled steps
  - `subtask_index` — always -1 for online data
  - `subtask_tokens` / `subtask_masks` — VLA-generated subtask conditioning
- **Periodically saved** to disk as LeRobot dataset at `{output_dir}/online_buffer/`

### Offline Replay Buffer

- **Populated by:** pre-collected annotated dataset loaded at startup
- **Read-only** during training
- **Tagged:** all transitions carry `is_golden: True`
- **Additional datasets** can be merged via `dataset.additional_offline_dataset_paths` config

### Sampling Strategy

When both buffers exist, `batch_size` is halved. Each training step samples `batch_size // 2` from online and `batch_size // 2` from offline, then concatenates them.

## gRPC Protocol

Defined in `lerobot/transport/services.proto`:

```protobuf
service LearnerService {
  rpc StreamParameters(Empty) returns (stream Parameters);
  rpc SendTransitions(stream Transition) returns (Empty);
  rpc SendInteractions(stream InteractionMessage) returns (Empty);
  rpc Ready(Empty) returns (Empty);
}
```

All data messages use the same chunked transfer pattern:

```protobuf
message Transition {
  TransferState transfer_state = 1;   // BEGIN, MIDDLE, END
  bytes data = 2;                      // chunk of serialized payload
}
```

Payloads are PyTorch-serialized (pickle + tensor data), split into chunks that fit within gRPC message size limits, and reassembled on the receiving end.

### Data Flow Summary

| Channel | From | To | Contents | Delivery |
|---|---|---|---|---|
| `StreamParameters` | Learner | Actor | Trainable policy weights (~800 MB at bf16) | Every `policy_parameters_push_frequency` seconds (default 120s) |
| `SendTransitions` | Actor | Learner | Episode transition list (obs, action, reward, next_obs, done, complementary_info) | At episode end |
| `SendInteractions` | Actor | Learner | Episode stats dict (reward, intervention rate, step count) | At episode end |
| `Ready` | Actor | Learner | Empty handshake | At startup |

## Models

### Policy (`PI05RLPolicy`)

Subclasses `PI05FullPolicy`. The model object (`self.model` / `self.actor`) is `PI05RLPytorch`, which extends the base pi05 forward pass with advantage conditioning. Uses full pi05 features including subtask generation and FAST tokens.

On the **actor**: loaded without critic (`use_separate_critic=False`), runs in eval mode.
On the **learner**: loaded with critic, runs in train mode.

### Critic (`Pi05TransformerCritic`)

A Gemma transformer (configurable depth, default 4 layers) that receives vision features and text embeddings, appends learned query tokens, and projects through a SwiGLU MLP to produce a scalar value estimate.

Only exists on the learner. Used for:
- Computing advantages for policy training: `advantage = target_v - current_v`
- Bellman TD updates
- Logging critic values for episode videos

### Critic Target

Frozen EMA copy of the critic, updated via `policy.update_target_networks()` with weight `critic_target_update_weight` (default 0.005). Used as bootstrap target in Bellman equation. Frozen layers share memory with the critic to save VRAM.

## RTC Configuration

```json
"rtc_config": {
    "enabled": true,
    "execution_horizon": 10,
    "prefix_attention_schedule": "LINEAR",
    "max_guidance_weight": 10.0
}
```

- `execution_horizon` — target number of actions to execute from each chunk before the next chunk is ready
- Inference is triggered when `qsize <= execution_horizon + latency_delay`
- Latency estimation uses p95 of recent inference times
- In-painting uses the prefix attention schedule to blend leftover actions from the previous chunk

## Action Encoding

Three modes supported, configured via `policy.action_encoding`:

| Mode | Formula | Notes |
|---|---|---|
| `absolute` | `a_t` | Raw joint positions |
| `anchor` | `d_t = a_t - s_0` | Translation invariant, recommended |
| `delta` | `d_0 = a_0 - s_0`, `d_t = a_t - a_{t-1}` | Consecutive diffs, prone to drift |

Anchor/delta transforms are applied at training time (not stored in the buffer). At inference time, the inference thread reconstructs absolute actions from the encoded output and handles anchor realignment for leftover actions across chunk boundaries.

Normalization is per-timestep: stats have shape `[chunk_size, action_dim]`, so each position in the chunk uses its own mean/std.

## Configuration

All settings are driven by a single JSON config file (e.g., `config-recap.json`). Key sections:

| Section | Purpose |
|---|---|
| `policy.actor_learner_config.learner_host` | Learner machine IP (set to remote IP for 2-machine setup) |
| `policy.actor_learner_config.learner_port` | gRPC port (default 50051) |
| `policy.actor_learner_config.policy_parameters_push_frequency` | Seconds between weight pushes (default 120) |
| `policy.trainable_params` | Controls which layers are trained and therefore sent over the network |
| `policy.rtc_config` | RTC in-painting settings |
| `policy.action_encoding` | `"absolute"`, `"anchor"`, or `"delta"` |
| `policy.use_separate_critic` | Whether to instantiate critic (learner sets true, actor overrides to false) |
| `policy.actor_device` / `policy.learner_device` | GPU device for each process |
| `policy.storage_device` | Where replay buffers live (typically `"cpu"`) |

## Running

**Start the learner** (on GPU machine):
```bash
python -m lerobot.rl.learner_pi05 --config_path config-recap.json
```

**Start the actor** (on robot machine, separate terminal):
```bash
python -m lerobot.rl.actor_pi05_async --config_path config-recap.json
```

Set `policy.actor_learner_config.learner_host` to the learner machine's IP address.

## File Map

```
lerobot/rl/
├── rl_pi05.py                    Config (PI05RLConfig), Critic, Policy class
├── pi05_train_utils.py           Shared training logic (Bellman, flow matching, preprocessing)
├── learner_pi05.py               Learner entry point, training loop, transition processing
├── learner.py                    Shared learner utilities (gRPC server, buffer init, checkpoints)
├── learner_service.py            LearnerService gRPC service implementation
├── actor_pi05_async.py           Actor entry point, thread orchestration
├── actor_pi05_async_utils.py     Inference thread, env thread, SharedStateActor, anchor alignment
├── actor.py                      gRPC client functions, transport helpers
├── buffer.py                     ReplayBuffer implementation
├── utils.py                      preprocess_batch_for_pi05, save_video_with_critic_overlay
├── inference_pi05_async.py       Standalone inference (no training)
├── inference_utils.py            Simplified SharedState for inference mode
└── gym_manipulator.py            Environment wrapper, transition creation

lerobot/policies/rtc/
├── configuration_rtc.py          RTCConfig dataclass
├── action_queue.py               ActionQueue (thread-safe action buffer with RTC merge)
├── modeling_rtc.py               RTCProcessor (in-painting logic)
└── latency_tracker.py            Latency estimation

lerobot/transport/
├── services.proto                gRPC service and message definitions
├── services_pb2.py               Generated protobuf classes
├── services_pb2_grpc.py          Generated gRPC stubs
└── utils.py                      Serialization helpers (chunked send/receive, bytes conversion)
```

# 06 — Inference runtime

Online inference is a thin wrapper (`rl/inference_async.py`) over the RTC actor
runtime ([`rl/rtc_actor_runtime.py`](../src/lerobot/rl/rtc_actor_runtime.py)).
Three cooperating threads share `RTCSharedState` under one lock: an **env worker**
(control loop at env fps, executes actions, pushes transitions), an **inference
worker** (builds batches, runs the policy), and the gRPC actor↔learner transport.

## 1. Control flow per step

1. Env worker reads the latest action chunk, executes one step, assembles the
   transition (state, action, reward, depth sidecar, `subtask_index` stamp), and
   pushes one history entry (state, executed action, depth) to the rolling deque.
2. Inference worker, at its own cadence: takes the latest observation, merges
   history windows (`assemble_history_windows`) and the current HL context
   (subtask name + metadata defaults) into the batch, runs the policy through
   **RTC** — the flow-matching denoise loop with prefix attention over the
   still-executing chunk (`execution_horizon: 5`, LINEAR schedule,
   `max_guidance_weight: 10`). The RTC denoise loop is eager (never touches CUDA
   graphs — why graphs-off for depth costs nothing).
3. Every `subtask_regeneration_interval = 4 s`, the inference worker additionally
   runs the **HL decode** ([04 §3](04_memory.md)): generation prompt conditioned
   on `current_summary`, greedy decode, parse memory-first answer, update
   `RTCSharedState` (subtask name/index + summary). Gated on
   `subtask_max_new_tokens > 0`. Must run in the inference thread — the
   action-expert patches are not thread-safe against a concurrent worker.

Episode restart clears history, subtask, and summary; **intervention clears
nothing** (training buffers keep contiguous frames across takeover; teleop action
= executed action).

### Action bounds (both runtimes, after the Butterworth)

`utils/action_smoothing.bound_action_chunk`, called through
`rl/inference_utils.bound_policy_actions` from the RTC and non-RTC inference
workers. Bounds the decoded chunk in follower degrees relative to $s_0$, the
observed state the chunk was inferred from, in this order:

1. excursion: $a_t \leftarrow s_0 + \mathrm{clip}(a_t - s_0, -D_j, D_j)$ — `action_delta_limits`
2. absolute: $a_t \leftarrow \mathrm{clip}(a_t, lo_j, hi_j)$ — `action_clamp_limits`
3. rate: $a_t \leftarrow a_{t-1} + \mathrm{clip}(a_t - a_{t-1}, -r_j, r_j)$, $a_{-1} = s_0$ — `action_step_limits`

Invariants: the absolute clamp is a contraction so it cannot undo the excursion
bound; the rate stage runs last in tracking form, so an $s_0$ outside the workspace
walks to the box edge at $r_j$ per tick instead of jumping. The RTC leftover tensor
(normalized, fed back as guidance) is not bounded, as with the filter. Fires log one
`[BOUND]` warning per chunk with the max shift per joint. The probes must not apply
these bounds. Values are 8 wide (padding column = 0, pinned) and come from
`migration/measure_action_bounds.py` over the training roots: $D_j$ = 1.5 x the
anchor-delta q01/q99 envelope over the horizon, workspace = action q0.1/q99.9 -/+ 5 deg
capped by the driver's `joint_limits`, $r_j$ = 1.25 x the max per-tick delta in the
demos. The driver's own `joint_limits` clip and the 60 deg/s `pos_vel_velocity` cap
stay as the outer layer. The reset move (`fixed_reset_joint_positions`,
`reset_follower_position`) goes through `send_action` at 20 deg/s, so it passes the
same driver layer; `RobotEnv` reads joint names from `robot.action_features`, not from
a Feetech-style `bus.motors` (the rebot bus has none).

## 2. Depth at inference

The D405 delivers raw uint16 z16 (0.1 mm/level, spatial + hole-fill filters OFF —
masking lives in the encoder). The policy's `prepare_context` patch back-projects
and runs the DepthStream **once per control step**, caching per-layer depth K/V
across all denoising steps ([03 §B.3](03_depth.md)). `back_project` accepts the
bare unbatched `(H, W)` live frame. A one-shot log line confirms depth presence on
the inference path (guards the silent-null-bank failure mode).

## 3. Prompt at inference

`build_inference_batch`
([rl_molmoact2_trainer.py:855](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L855))
assembles the same clauses as training, with the steering defaults:

| Clause | Inference value |
|---|---|
| subtask | latest HL decode (absent until the first decode) |
| metadata | `{quality: 5, mistake: false}` — π0.7 "prompt for the best" |
| history | live deque windows (when `memory.history_keys` set) |
| advantage | `inference_advantage` — **null under skip_critic** (clause absent; a hardcoded 1.0 would be out-of-distribution against advantage-free training) |
| summary | generation prompt only, never the action prompt |

## 4. Offline eval

`probes/action_trace_probe.py` carries the headline: at anchors spaced through the
held-out episodes it scores flow sample 0 against the demonstrated chunk in
normalized space, next to the hold-still and dataset-mean constants
(`skill_vs_hold` / `skill_vs_mean` in `action_metrics.json`), and FKs the fan for
table clearance and multimodality. It absorbed `probes/offline_inference.py`,
deleted 2026-08-02: that probe's remaining job was the HL memory/subtask decode,
which went away with the summary memory (2026-08-01), and its 2-D per-joint panels
duplicated the anchors and the sample-0 seed the inspector already owned.
`probes/depth_modality_probe.py` runs the 2×2
{RGB±, depth±} matrix + per-layer depth attention mass + input sensitivities
against the real checkpoint (replaced the gate-0 bit-identity probe 2026-07-26 —
no gate exists under the joint softmax read;
`uv run python -m lerobot.probes.depth_modality_probe --config config_rl.yaml`).

## 5. Latency notes

- 30 Hz control target; `torch_compile` currently off (rule out compile noise on
  the first depth run).
- HL decode budget: 128 tokens covers "Memory: … Subtask: …" (12 s summaries are
  multi-sentence). Parked optimization: skip the memory-span decode on ticks where
  the subtask didn't change — measure first (memory-first order pays full summary
  decode before the LL gets its subtask).
- Depth stream: <1% of forward, run once per control step.

## 6. Hardware context

- **rebot B601** on the daisy-chain bus; there was an intermittent power break near
  the base (cable replacement — verify with `wiggle_rebot_cable.py` before
  trusting rollouts).
- **Motor liveness check** (2026-09-04). `motorbridge.get_state()` returns the cached
  frame, so a motor that drops off the chain mid-episode reads as a plausible frozen
  value, not as an error (15:06 episode: wrist_yaw / wrist_roll / gripper flat for 12 s
  while the policy commanded the gripper open). `RebotB601Follower._check_motor` does a
  synchronous register read (`RID_MST_ID`, 20 ms timeout) and raises `RuntimeError`
  naming the motor. `connect()` checks all seven before configure (replaces the nameless
  "register 10 not received"); `get_observation()` checks one motor per call, round-robin,
  so every joint is verified every 7 steps (~0.23 s at 30 Hz, ~1 ms per step). The raise
  propagates unchanged: env worker logs it as fatal, the supervisor sees the dead thread
  and exits the program; the unfinished episode is discarded (frozen joints are not data).
  Park does not ping, so the live joints still descend on the way out.
  Second signal, same afternoon: the 15:27 episode had all seven motors answering pings
  and sending fresh positions while wrist_yaw / wrist_roll / gripper ignored every
  command (a brownout reboot leaves a Damiao motor disabled; a latched fault does the
  same). The feedback frame's ERR nibble says so (0 disabled, 1 enabled, 8..E faults), so
  `_check_status` raises on anything but 1: `configure()` refreshes feedback 5 times after
  `enable_all` and checks once, `get_observation()` checks after every read. No extra bus
  traffic. If a healthy launch dies at configure with "reports status 0x…", the nibble
  mapping is wrong for this firmware: read the code it printed and adjust `_STATUS_ENABLED`.
- **D405** wrist-mounted on rear USB3; factory intrinsics in the config.
- **Park on disconnect.** `RebotB601Follower.disconnect()` = freeze (present pose
  re-sent as the target, cancelling any in-flight sweep) -> linear descent to
  `park_pose` (default: calibration zero = sit-down, gripper closed) at
  `park_deg_per_s` (20) -> settle: arrival re-checked every 0.5 s for up to 5 s (the
  elbow lags the ramp by ~13 deg after a long descent and closes it on its own; a fixed 1 s
  settle called that a failure on 2026-09-04) -> release torque only if every joint is
  within `park_tolerance_deg` (10). Otherwise torque stays ON and the arm holds. Every exit
  path (normal, exception, Ctrl-C via the supervisor's `finally`) reaches it through
  `_close_robot_hardware` -> `RobotEnv.close()`. Hard kill / host power loss: nothing
  runs, the Damiao motors hold their last target (no CAN timeout is set, on purpose);
  run `park_rebot.py` afterwards. The descent is a straight line in joint space and
  does not know about objects in the workspace.
- **Return to rest at episode end** (`_return_to_rest`, 2026-09-04). Key 0/1 (or any
  done/truncated) parks the follower through the same `park()` (torque stays on: it holds
  the rest pose until the next "2"), and the actuated leader rides the identical ramp
  through `park(on_step=send_feedback)`, 0.67 deg/tick at 20 deg/s (far under the 8
  raw-deg fault ceiling; keeps the 0.5 s feedback watchdog fed), then is released AT REST
  instead of dropped. Safety mode: only the leader ramps. The env worker's exit path does
  the same for an interrupted episode (Ctrl-C mid-episode, fatal exception), so the
  follower never holds mid-air waiting for Ctrl-C. Order at episode end: rest -> recorder
  `save_episode` -> debug episode log. Shutdown waits for the env worker (uncapped join,
  10 s progress line) so the hardware close cannot race the descent or the episode save;
  Ctrl-\ twice still forces an exit.
- **Leader arm** (planned shadow-takeover teleop): the Seeed leader is
  encoder-only — its joints cannot be driven, so the mirror-and-grab takeover
  design needs the HD variant or a clutch/delta scheme. `send_feedback` is the
  software hook (API verified).

## 7. Online episode dataset

Every standalone inference run records its episodes to `{output_dir}/inference_dataset`
through `rl/online_recorder.OnlineEpisodeRecorder`, in the **lerobot_record format**:

- Schema = `hw_to_dataset_features(robot.observation_features / action_features)`, the
  record script's own builder: rig camera names (`observation.images.top / wrist`), the
  rig's joint order and width (7 for rebot, so the policy's 8-wide executed action is
  sliced to the leading rig joints), video frames, `robot_type`. No reward/done/RL columns.
- Frame $t$ = the raw robot observation the step was taken from (`RobotEnv.get_raw_observation`,
  float32 joints, the camera's own uint8 frame) + the executed action (policy target, or the
  teleop target during an intervention). Written per step like `record_loop`, not from the
  online replay buffer (bf16 low-dim storage). The standalone replay buffer is gone.
- Depth: `depth_writer.write_depth` PNG16 sidecar, stride = `policy.image_stride`, phase on
  the global frame index (same grid the memmap cache reads). `save_episode` after each
  episode (video encode runs between episodes), `finalize` when the env worker exits; an
  unfinished episode at shutdown is discarded.
- `meta/online_labels.parquet` (episode_index, frame_index, index, is_intervention,
  subtask_index, subtask text): per-frame labels the recorded schema has no column for,
  a sidecar in the style of the annotation files. Subtask text rides along because the
  vocabulary may be revised.
- Consumption: add the run's `inference_dataset` as a `dataset.sources` entry; annotation
  is the same separate pass as for recordings. Hardware-free check:
  `migration/online_recorder_check.py`.
- Between-episode cost: the video encode runs in `save_episode` after the arms are at
  rest, so it only delays the next "2". `episode_logging_freq > 0` adds ~12 s per episode
  (192 synchronous PNGs + critic + video) on top; the recorder's mp4s are the durable
  videos. `streaming_encoding` (encode during the episode) is NOT enabled: a full encoder
  queue would stall the 30 Hz env loop, untested under the inference load.

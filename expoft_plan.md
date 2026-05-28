# EXPO-FT DRTC Integration Plan

## Goal

Support EXPO-FT in the `scripts/run_drtc_experiment.sh` pipeline without changing the shell runner into algorithm-specific orchestration. The runner already starts `policy_server_drtc`, launches the DRTC experiment client, and passes YAML config through. EXPO-FT should be added beside the existing RLT path in the policy/config/server/client layers.

The implementation should treat EXPO-FT as a new online adaptation layer:

- The VLA remains the expressive base policy distribution.
- A small edit policy proposes bounded local corrections around VLA action chunks.
- A critic scores base and edited candidates.
- The executed action chunk is selected by highest Q among base and edited candidates.
- Online DRTC training updates only the EXPO critic/edit modules.
- Any supervised fine-tuning of the base VLA should be a separate offline stage using the VLA's original objective.

## Current RLT Reference Points

The existing RLT path is a good integration template, but not the right abstraction to reuse directly:

- Experiment YAML fields are defined in `examples/experiments/run_drtc_experiment.py`.
- Client/server config fields are carried by `RobotClientDrtcConfig` and `RemotePolicyConfig`.
- Supported policy names are listed in `src/lerobot/async_inference/constants.py`.
- Policy classes are registered in `src/lerobot/policies/factory.py`.
- RLT server setup, context caching, online replay, and training live in `src/lerobot/async_inference/policy_server_drtc.py`.
- RLT client transition emission lives in `src/lerobot/async_inference/robot_client_drtc.py`.
- RLT tensor replay is stored in `src/lerobot/rl/rlt_buffer.py`.
- RLT modules and losses are in `src/lerobot/rl/rlt_pi05.py`, with wrapper policies in `rlt_tinypi05v2.py` and `rlt_molmoact2.py`.

RLT stores `rl_token` state and trains a frozen-VLA actor/critic head. EXPO-FT should instead store base candidates, edited/selected chunks, critic inputs, and next context for Q learning.

## Implementation Plan

### 1. Add EXPO Policy Family

Add EXPO wrapper policy types, starting with simulator-friendly Tinypi:

- `tinypi05v2_expo`
- later `molmoact2_expo`

Register them in:

- `src/lerobot/async_inference/constants.py`
- `src/lerobot/policies/factory.py`

The wrappers should expose a common interface:

- `sample_base_candidates(batch, **rtc_kwargs)`
- `edit_candidates(batch, base_candidates)`
- `score_candidates(batch, candidates)`
- `select_highest_q_candidate(batch, candidates)`
- `predict_action_chunk(batch, **rtc_kwargs)`

The VLA base sampler should be detached from Q gradients. The critic/edit modules are the online-trainable EXPO parts.

### 2. Create EXPO RL Modules

Add `src/lerobot/rl/expo.py` with:

- `ExpoEditPolicy`: bounded residual editor over action chunks.
- `ExpoCritic`: Q network over critic observation features, proprio, and action chunk.
- `ExpoCriticEnsemble`: optional clipped-min critic ensemble.
- `expo_critic_loss`.
- `expo_edit_loss`.
- `soft_update_expo_target`.
- `save_expo_checkpoint`.

Initial conservative defaults:

- Always include the unedited VLA base chunk as a candidate.
- Clamp edit deltas with `expo_edit_abs_max`.
- Prefer a small edit policy first.
- Fall back to base action when safety thresholds trip.

### 3. Add EXPO Replay Buffer

Add `src/lerobot/rl/expo_buffer.py`.

Store policy-agnostic EXPO transition data:

- critic observation features or compact image tensors,
- proprio,
- base candidate chunk or chunks,
- selected/executed chunk,
- next critic context,
- next base candidates,
- reward,
- done,
- intervention flag,
- success/failure labels,
- optional review images and episode metadata.

Do not force this into `RLTReplayBuffer`; RLT's `rl_token` schema is algorithm-specific.

### 4. Extend Config Surface

Add `expo_*` fields beside the current `rlt_*` fields in:

- `examples/experiments/run_drtc_experiment.py`
- `src/lerobot/async_inference/configs_drtc.py`
- `src/lerobot/async_inference/helpers.py`

Core fields:

- `expo_enabled`
- `expo_checkpoint`
- `expo_chunk_size`
- `expo_num_base_samples`
- `expo_num_edit_samples`
- `expo_edit_abs_max`
- `expo_bc_beta`
- `expo_smoothness_beta`
- `expo_online_collection_enabled`
- `expo_online_training_enabled`
- `expo_warmup_episodes`
- `expo_warmup_transitions`
- `expo_replay_capacity`
- `expo_batch_size`
- `expo_utd_ratio`
- `expo_critic_updates_per_edit`
- `expo_train_freq_s`
- `expo_save_freq_steps`
- `expo_output_dir`
- `expo_demo_buffer_path`
- `expo_online_buffer_path`
- `expo_discount`
- `expo_target_update_tau`
- `expo_execute_after_train_steps`
- `expo_context_cache_size`
- `expo_transition_queue_size`
- `expo_grad_clip_norm`
- `expo_q_abs_max`
- `expo_action_deviation_abs_max`
- `expo_loss_abs_max`
- `expo_safety_patience`

### 5. Add Server-Side EXPO Path

In `policy_server_drtc.py`, add EXPO logic parallel to the RLT path instead of modifying `_predict_pi05_rlt_with_context`.

The EXPO inference path should:

1. Build the preprocessed observation.
2. Sample VLA base action candidates.
3. Generate bounded edited candidates.
4. Score base plus edited candidates with the EXPO critic.
5. Select the highest-Q candidate.
6. Cache the EXPO source context for replay.
7. Publish the selected chunk with EXPO metadata.

The online trainer should:

1. Wait for replay and episode warmup.
2. Train critic with TD targets using next-context candidate max Q.
3. Train edit policy through Q selection or a conservative differentiable proxy.
4. Soft-update target critic.
5. Save EXPO checkpoints on cadence.
6. Emit `expo_*` status fields for the TUI/log stream.

### 6. Add Client/Server Transition Plumbing

Current RLT collection is split across:

- client chunk tracking in `robot_client_drtc.py`,
- server transition acceptance in `policy_server_drtc.py`,
- protobuf fields in `src/lerobot/transport/services.proto`.

For EXPO, add explicit EXPO messages rather than overloading `RLTTransitionChunk`:

- `ExpoTransitionChunk`
- `SendExpoTransitions`
- EXPO context ID fields on `ActionsDense`, or a generic context metadata extension if backward compatibility is manageable.

The client should collect executed windows the same way RLT does, including teleop interventions and terminal success/failure backfill.

### 7. Keep Base VLA Fine-Tuning Offline

Do not backprop Q gradients through the VLA in the DRTC server.

Add an optional second-stage offline script later:

- `src/lerobot/rl/train_tinypi05v2_expo_sft.py`
- `src/lerobot/rl/train_molmoact2_expo_sft.py`

These scripts should fine-tune the base VLA on reviewed successful/intervention data with the original supervised objective. That preserves EXPO-FT's stable VLA update story without putting heavyweight VLA training in the real-time server.

### 8. Add Example Configs

Start with simulation:

- `examples/experiments/configs/baseline_tinypi05v2_expo_sim.yaml`

Then add hardware:

- `examples/experiments/configs/baseline_molmoact2_expo_so101_online.yaml`

Keep `scripts/run_drtc_experiment.sh` mostly unchanged. At most, add usage comments for the new configs.

### 9. Tests And Verification

Unit tests:

- EXPO policy registration and config loading.
- EXPO replay save/load roundtrip.
- Candidate selection chooses highest Q.
- Base candidate is always included.
- Edit deltas clamp to `expo_edit_abs_max`.
- Safety fallback returns base candidate.
- Client transition windows align with shifted execution windows.
- Server transition acceptance builds the expected replay sample.

Smoke tests:

1. Collect-only EXPO run in Squint sim.
2. Offline train edit/critic from saved replay.
3. Online sim run with `expo_execute_after_train_steps` high, confirming VLA passthrough plus replay collection.
4. Online sim run with `expo_execute_after_train_steps` low, confirming EXPO candidate selection executes.
5. Hardware collect-only run with teleop intervention before enabling online execution.

## Milestone Order

1. Implement `tinypi05v2_expo` with VLA frozen, base candidate fallback, and online critic/edit training in Squint sim.
2. Add EXPO replay and DRTC transition plumbing.
3. Add tests for replay, candidate selection, safety fallback, and DRTC transition windows.
4. Run collect-only sim and train from replay.
5. Enable online EXPO execution in sim.
6. Port the EXPO wrapper to `molmoact2_expo`.
7. Add optional offline supervised VLA refresh from reviewed EXPO data.

## Practical Starting Point

The safest first implementation target is:

`tinypi05v2_expo` in Squint sim, online critic/edit training only, frozen VLA, base action fallback always available.

Once that path is stable, port the same EXPO mixin to `molmoact2_expo` and add the optional supervised VLA refresh script as a separate stage.

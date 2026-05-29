# Online RLT DRTC Ledger

This note is a ledger for the distributed trace produced by:

```bash
./scripts/run_drtc_experiment.sh --config examples/experiments/configs/baseline_molmoact2_rlt_so101_online.yaml
```

The config runs `molmoact2_rlt` on an SO101 follower with online RLT collection
and online RLT training enabled. The shell runner starts a local DRTC policy
server, starts the experiment client, and, by default, starts the TUI. The TUI
and both runtime processes share JSONL status/control side channels in `logs/`.

The goal of this document is not to draw a perfect architecture diagram. The
goal is to make the cross-thread and cross-process trace explicit enough to
answer questions like:

- When does the server train on a transition?
- Which action chunk does an executed action belong to?
- Why can a transition be collected while the RLT actor is not yet executing?
- What state is durable when the run is interrupted?

## Command-Specific Facts

These values come from
`examples/experiments/configs/baseline_molmoact2_rlt_so101_online.yaml`.

| Plane | Value |
| --- | --- |
| Policy | `policy_type=molmoact2_rlt` |
| Frozen VLA checkpoint | `outputs/molmoact2_so101_placemotor_e51_v6/checkpoints/030000/030000/pretrained_model` |
| RLT embedding | `outputs/molmoact2_rlt_embedding/rlt_embedding_step_010000.pt` |
| RLT head at startup | empty, so no actor/critic head checkpoint is loaded |
| Full DRTC chunk | `actions_per_chunk=30` |
| RLT window | `rlt_chunk_size=10` |
| Collection | `rlt_online_collection_enabled=true` |
| Training | `rlt_online_training_enabled=true` |
| Replay capacity | `50000` compact transitions |
| Training warmup | at least `1` completed critical phase and `128` transitions |
| Batch | `64`, `utd_ratio=1`, `critic_updates_per_actor=2` |
| Actor execution gate | `rlt_execute_after_train_steps=1000` |
| Replay path | `outputs/rlt_molmoact2_so101_online/rlt_online_replay.pt` |
| Review archive | `outputs/rlt_molmoact2_so101_online/rlt_review_archive.pt` |
| Periodic replay save | every `50` accepted transitions |
| Shutdown replay save | enabled by `rlt_persist_buffer_on_shutdown=true` |
| Teleop | enabled with SO101 leader, feedback enabled |
| Runtime length | `run_until_interrupt=true`, so Ctrl+C or TUI exit ends the client |

Important TUI detail: with the default TUI enabled, the policy server sees
`LEROBOT_DRTC_CONTROL_FILE` and starts online RLT training in the operator-paused
state. Press `6` in the TUI to start or pause trainer updates. Collection still
works while training is paused.

## Actors And State

| Actor / State Owner | What It Does |
| --- | --- |
| Shell runner | Kills an old server on the target port, starts `policy_server_drtc`, creates log/status/control files, then starts `run_drtc_experiment.py`. |
| TUI | Reads status JSONL from client/server and appends operator commands to the control JSONL file. |
| Client main control loop | Executes robot actions, polls teleop/control events, starts and labels RLT critical phases, triggers observation capture, and merges incoming action chunks. |
| Client observation sender thread | Captures robot observations, encodes images, and sends `TimedObservation` payloads to the server. |
| Client action receiver thread | Streams `ActionsDense` chunks from the server, decodes them, and publishes the newest chunk into the client action register. |
| Client RLT transition sender thread | Batches queued `RLTTransitionChunk` messages and sends them to the server. |
| Server gRPC receiver threads | Accept policy setup, observations, streamed RLT transitions, and optional visualization chunks. |
| Server inference producer thread | Reads the freshest observation, runs MolmoAct2 RLT inference, caches RLT source context, and publishes dense action chunks. |
| Server RLT trainer thread | Samples accepted replay transitions, trains critic and actor heads, updates target critic, saves checkpoints, and enforces safety gates. |
| Server replay buffer | In-memory `RLTReplayBuffer` protected by `_rlt_replay_lock`; periodically persisted to `rlt_online_buffer_path`. |
| Server RLT context cache | LRU cache keyed by server-owned `rlt_context_id`; connects later executed actions back to the RL token/reference/proprio observed at inference time. |

## Core Identities

| Identity | Owner | Meaning |
| --- | --- | --- |
| `control_step` | Client control loop | Logical DRTC clock. Last-write-wins registers reject stale observations/action chunks with older control steps. |
| `action_step` / `chunk_start_step` | Client execution loop and server chunk | Execution-space action index. It says where a returned chunk should be applied. |
| `source_control_step` | Server action chunk | Observation/control step that produced the action chunk. |
| `rlt_context_id` | Server | Server-owned key for a cached source context behind an action chunk. |
| `next_rlt_context_id` | Client | Link from one collectable RLT context to the next context for bootstrapping. Zero means terminal or unknown. |
| `episode_id` | Client, offset by server on load | Client critical-phase id. Server adds an offset when resuming from a persisted online buffer. |
| `policy_mode` | Server | `warmup`, `vla_passthrough`, `rlt_actor`, or `rlt_safety_passthrough`. |
| `is_intervention` | Client | True when the executed robot action was teleop or otherwise differed from the requested policy action. |
| `done`, `success`, `failure`, `reward` | Client label path | Terminal label for the buffered critical phase. |

## Planes

| Plane | Question | DRTC/RLT State |
| --- | --- | --- |
| Routing | Which process receives the event? | gRPC method, status/control JSONL file, TUI command |
| Freshness | Is this observation/action chunk still current? | `control_step`, LWW registers, reader watermarks |
| Execution | Which robot action steps actually ran? | `action_step`, `chunk_start_step`, action schedule |
| RLT correlation | Which source context produced this executed window? | `rlt_context_id`, `next_rlt_context_id` |
| Collection phase | Should executed actions become replay transitions? | rollout open, critical phase open, pending label, discard |
| Training readiness | May the trainer update heads now? | replay size, completed episodes, operator pause, batch size |
| Actor execution | May the learned head affect robot actions? | loaded head or `rlt_train_step >= rlt_execute_after_train_steps`, safety flags |
| Durability | What survives process exit? | online replay file, review archive, head checkpoints, metrics/log JSONL |

## Ledger 1: Session Startup

Question: what happens before the first robot action can be trained on?

| Step | Event | Actor | Runtime State | RLT State | Durable / Side Effect | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | User runs shell command | shell | Parses `--config` and resolves the YAML path | none | chooses log names under `logs/` | create empty client log, status JSONL, control JSONL |
| 2 | Old server on port 8080 is killed if present | shell | Port owner removed | old server, if any, receives termination | old server best-effort persists replay during shutdown/reset | start fresh server |
| 3 | Policy server process starts | shell/server | `policy_server_drtc` binds `localhost:8080` | RLT defaults only, no policy loaded yet | server log begins | wait `POLICY_SERVER_DELAY_S` |
| 4 | Experiment client starts | shell/client | `run_drtc_experiment.py` loads YAML and builds `RobotClientDrtcConfig` | config contains collection/training settings | metrics path created under `results/experiments/...` | instantiate `RobotClientDrtc` |
| 5 | Client connects robot and teleop | client | SO101 follower, cameras, and SO101 leader connect | collection status starts at `waiting_to_start_rollout` | status event if side channel enabled | call server `Ready` |
| 6 | Server handles `Ready` | server | `_reset_server()` clears registers, caches, old threads | old replay may be persisted if dirty | online replay/review archive can be saved with reason `reset` | clear shutdown event |
| 7 | Client sends `PolicySetup` | client/server | Pickled `RemotePolicyConfig` reaches `SendPolicyInstructions` | all `rlt_*` config copied to server | status `model_loading` / `model_ready` | server loads policy |
| 8 | Server loads MolmoAct2 RLT | server | frozen MolmoAct2 policy and processors are loaded | embedding checkpoint loaded; head checkpoint empty | status `rlt_configured` | start inference producer and trainer thread |
| 9 | Trainer thread starts paused under TUI | server | TUI control reader is enabled | `_rlt_training_head=paused`, operator enabled false | status reports training paused | wait for TUI key `6` |

What this ledger shows:

- `rlt_online_training_enabled=true` means the trainer thread exists, not that it
  necessarily starts updating immediately.
- With TUI enabled, operator control is a fence for trainer updates.
- Collection and inference can begin before any actor head is loaded or trained.

## Ledger 2: One DRTC Inference Chunk

Question: how does one observation become an action chunk and an RLT source
context?

| Step | Event | Actor | Freshness State | Action / RLT State | Observation / Result | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Control loop checks schedule | client main | `control_step=t`, `current_action_step=j` | action schedule may be low | if not waiting for rollout start and schedule size `<= H - s_min`, trigger inference | publish observation request |
| 2 | Observation request published | client main | `_obs_request_reg.update_if_newer(t, request)` | request has `chunk_start_step=max(j,0)` and RTC metadata | stale/equal requests cannot overwrite newer requests | observation sender wakes |
| 3 | Robot observation captured | client obs thread | reads newest observation request once | camera images, follower state, task, RTC metadata | fallback may reuse last good observation if capture fails | encode and send gRPC |
| 4 | Server receives observation | server gRPC | `_obs_reg.update_if_newer(t, timed_observation)` | no RLT context yet | server stamps receive time | inference producer wakes |
| 5 | Producer reads freshest observation | server producer | reader watermark advances to `t` | raw state captured as anchor for action encoding | older observations are skipped by the LWW register | preprocess |
| 6 | MolmoAct2 RLT path runs | server producer | same `source_control_step=t` | builds VLA reference chunk, RL token, proprio | if actor gate is closed, `policy_mode=warmup` or `vla_passthrough`; if open, `policy_mode=rlt_actor` | cache source context |
| 7 | Server caches RLT source context | server producer | cache key allocated as `rlt_context_id=c` | stores RL token, proprio, reference 10-step window, anchor, optional review JPEGs | cache insert succeeds only if collection is enabled and embedding exists | postprocess actions |
| 8 | Dense action chunk is published | server producer | `_action_reg.update_if_newer(t, ActionsDense)` | chunk carries `rlt_context_id=c`, `rlt_collectable=true`, `rlt_window_start_index` | server logs `[DRTC INFER TIMING]` | client stream receives |
| 9 | Client receives and publishes chunk | client action thread | `_action_reg.update_if_newer(source_control_step=t, chunk)` | decoded `ReceivedActionChunk` keeps RLT metadata | duplicate/reordered/stale chunks can be rejected by freshness rules | client main merges chunk |
| 10 | Client notes collectable chunk | client action thread | chunk accepted by action register | pending chunk keyed by `rlt_context_id=c` | if no critical phase is open, it goes to prebuffer; otherwise active pending map | wait for executed actions |

What this ledger shows:

- `rlt_context_id` is minted on the server before postprocessing and before the
  robot executes the actions.
- The client does not reconstruct RL tokens. It only carries context IDs and
  executed actions back to the server.
- During early online training, `policy_mode=warmup` can still be collectable.
  The replay can train from VLA-passthrough behavior before the actor is allowed
  to control the robot.

## Ledger 3: Critical Phase Collection

Question: when does the client turn executed robot actions into compact replay
transitions?

| Step | Event | Actor | Collection State | RLT Correlation | Observation / Result | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Operator presses `2` | TUI/client | command `start_rollout` appears in control JSONL | none | client starts rollout, bumps inference epoch, clears old pending state | run VLA/RLT policy normally |
| 2 | Collectable chunks arrive before critical phase | client action thread | rollout open, no critical phase open | chunks and executed actions go to prebuffer | prebuffer keeps a short window around recent actions | wait for critical start |
| 3 | Operator presses `5` | TUI/client | command `toggle_critical_intervention` | current prebuffer seeded into active critical phase | teleop intervention is enabled by the TUI path | record critical window |
| 4 | Client executes a policy action | client main | critical phase open | executed action step matches a pending chunk's step range | action is marked intervention only if robot-applied action differs from requested action | store in `_rlt_executed_actions` |
| 5 | Client executes a teleop action | client main | critical phase open, intervention true | same action-step matching | leader action overrides policy; `is_intervention=true` | store in `_rlt_executed_actions` |
| 6 | Another collectable chunk arrives | client action thread | critical phase still open | previous pending chunk gets `next_rlt_context_id=new_context_id` | source/next context link is ready for bootstrapping | `_rlt_maybe_emit_transitions` may buffer transition |
| 7 | All actions for a pending window have executed | client main or action thread | critical phase open | source context and next context known, or terminal is being forced | transition is appended to current critical-phase buffer, not yet sent to server | continue recording |
| 8 | Operator presses `5` again | TUI/client | critical intervention closes | last buffered transition is backfilled terminal if needed | TUI toggle path auto-labels success with reward 1.0 | buffered transitions are put onto `_rlt_transition_queue` |
| 9 | Operator presses `0` or sends another failure/success event | TUI/client | critical phase is ended first if still open | terminal transition gets `done=true`, `next_rlt_context_id=0` | reward is 1.0 for success or 0.0 for failure | transition sender sends to server |
| 10 | Operator discards instead | TUI/client | discard command | pending chunks/actions cleared | no transitions are queued | rollout stays open |

What this ledger shows:

- A rollout is a broad collection session. A critical phase is the labeled
  training segment inside that rollout.
- The TUI `5` key starts and ends a critical intervention segment. Ending this
  path auto-labels success. The TUI `0` path records failure, and non-TUI robot
  events can also send success/failure labels.
- The client buffers transitions until a terminal label is known. This prevents
  unlabeled critical phases from entering replay accidentally.

## Ledger 4: Server Accepts A Transition

Question: what must be true for a client transition to become a replay sample?

| Step | Event | Actor | Transition Fields | Server State Checked | Result | Durable / Side Effect |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Sender batches transitions | client sender thread | up to 16 `RLTTransitionChunk` messages | client queue may drop if full before this point | gRPC `SendRLTTransitions` starts | none |
| 2 | Server reads transition | server gRPC | `source_rlt_context_id`, `next_rlt_context_id`, executed actions, label fields | server running flag | calls `_accept_rlt_transition` | none |
| 3 | Source context lookup | server | `source_rlt_context_id=c` | `_rlt_context_cache.get(c)` | if missing, counter `rlt_transition_missing_source`, transition is dropped | no replay mutation |
| 4 | Next context lookup | server | `next_rlt_context_id=n` | cache lookup for `n` unless zero | if missing and transition is not terminal, counter `rlt_transition_missing_next`, transition is dropped | no replay mutation |
| 5 | Executed action payload decoded | server | `num_actions`, `action_dim`, bytes | shape must match payload length | invalid payload raises and is dropped by RPC handler | no replay mutation |
| 6 | Executed action converted to model space | server | raw robot action chunk | source context anchor/normalizer/action encoding | `executed_chunk` aligns with reference model space | prepare replay sample |
| 7 | Reference selected | server | `is_intervention` and `rlt_intervention_reference_mode=executed` | source reference and executed chunk | intervention samples use executed chunk as training reference; non-intervention samples use VLA reference | prepare replay sample |
| 8 | Replay sample accepted | server | reward/done/success/failure copied | `_rlt_replay_lock` | sample appended, counts incremented, completed episode set updated if terminal | status `rlt_transition_accepted` |
| 9 | Periodic persistence check | server | accepted transition count | save every 50 accepted transitions for this config | dirty replay/review archive may be saved | `rlt_online_replay.pt`, `rlt_review_archive.pt` |

What this ledger shows:

- `rlt_context_id` is the fence that proves the server still has the source
  context for the executed action window.
- Non-terminal transitions need a valid next context. Terminal transitions can
  use `next_rlt_context_id=0`.
- Replay mutation happens only on the server, after context lookup and payload
  validation.

## Ledger 5: Online Trainer And Actor Execution

Question: when does online training affect future robot actions?

| Step | Event | Actor | Training Gate | Replay / Model State | Result | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Trainer tick wakes | server trainer thread | sleeps `rlt_train_freq_s=1.0` between ticks | current replay size and completed episode count | polls control commands before/after sleep | decide state |
| 2 | TUI training paused | server trainer thread | operator enabled false | replay may still grow | training head becomes `paused` | wait for TUI `6` |
| 3 | Warmup replay not met | server trainer thread | replay size `< max(64,128)` | fewer than 128 transitions | training head `warmup_replay` | wait for more accepted transitions |
| 4 | Warmup episodes not met | server trainer thread | completed episodes `< 1` | enough transitions but no terminal label | training head `warmup_episodes` | wait for a labeled critical phase |
| 5 | Critic updates | server trainer thread | all gates pass | sample batch with success/intervention fractions | two critic updates for this config | actor update |
| 6 | Actor update | server trainer thread | same model lock | actor loss, gradient clipping, target soft update | `rlt_train_step += 1` | emit `rlt_training_step` |
| 7 | Safety check | server trainer thread | configured limits: grad clip 20, q/action/loss maxes | violations tracked | repeated unsafe updates can disable actor execution | continue or disable actor |
| 8 | Head checkpoint save | server trainer thread | `rlt_train_step % 250 == 0` | actor/critic state | saves numbered and latest head checkpoint | future sessions can load head |
| 9 | Inference gate rechecked | server producer | `rlt_train_step >= 1000` or actor checkpoint loaded, safety not disabled | same policy object under model lock | future chunks use `policy_mode=rlt_actor`; before that, `warmup`/VLA pass-through | client receives actor-refined actions |

What this ledger shows:

- The trainer and inference producer share the same policy object behind
  `_rlt_model_lock`.
- Training can improve future inference only after the actor execution gate
  opens. This config starts executing the online actor after 1000 trainer steps
  unless a head checkpoint is loaded in a later run.
- The server can keep collecting replay while the actor gate is closed.

## Ledger 6: Shutdown And Resume

Question: what survives Ctrl+C or a fresh run?

| Step | Event | Actor | Runtime State | Durable State | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | Ctrl+C or TUI exit | shell/client | client process receives stop/termination | client metrics flushed when possible | experiment process exits |
| 2 | Shell cleanup stops server it started | shell/server | server receives SIGTERM | server `finally` calls `policy_server.stop()` | `_reset_server()` runs |
| 3 | Server reset persists dirty replay | server | `rlt_persist_buffer_on_shutdown=true` | `rlt_online_replay.pt` saved with online samples only | replay survives |
| 4 | Server persists review archive | server | dirty review archive and path set | `rlt_review_archive.pt` saved append-only | review frames/labels survive |
| 5 | Trainer and producer threads join | server | shutdown event set | no new replay mutations | process exits |
| 6 | Next run loads online buffer | server | `_configure_rlt_online` sees path exists | replay samples loaded and episode offset computed | training can resume from prior online replay |
| 7 | Next run loads head only if configured | server | this YAML still has empty `rlt_head_checkpoint` | no saved head is loaded automatically | set `rlt_head_checkpoint` to `rlt_head_latest.pt` if you want to execute a prior trained head |

What this ledger shows:

- The replay file can resume automatically because `rlt_online_buffer_path` is in
  the YAML.
- The online-trained head is saved separately under `rlt_output_dir`; this YAML
  does not automatically reload it unless `rlt_head_checkpoint` is filled in.
- Logs and status JSONL are observational artifacts. The replay and head
  checkpoints are the artifacts that feed later training/evaluation.

## Failure Windows

### Missing Source Context

| Step | Event | Actor | Cause | Result |
| --- | --- | --- | --- | --- |
| 1 | Server caches context `c` | server producer | collectable action chunk produced | context enters LRU cache |
| 2 | Client delays transition for `c` | client | critical phase stays open, queue is blocked, or many newer chunks arrive | server cache may evict `c` |
| 3 | Transition for `c` arrives | server | `source_rlt_context_id=c` no longer in cache | transition is dropped; replay unchanged |

Invariant: the context cache is a correlation fence. If the server no longer has
the source context, the executed action window is not trainable.

### Training Enabled But Paused

| Step | Event | Actor | Observed State | Result |
| --- | --- | --- | --- | --- |
| 1 | TUI-enabled run starts | server | control file exists | `_rlt_training_operator_enabled=false` |
| 2 | Replay grows | server/client | transitions accepted | training state remains `paused` |
| 3 | Operator presses `6` | TUI/server | `toggle_rlt_training` command read | trainer gates can proceed |

Invariant: online collection and online training are separate planes. Replay can
grow while training is paused.

### Actor Trained But Not Executing Yet

| Step | Event | Actor | Observed State | Result |
| --- | --- | --- | --- | --- |
| 1 | Trainer increments step | server | `rlt_train_step < 1000` | future inference still uses VLA pass-through / warmup |
| 2 | Replay continues to collect | client/server | collectable contexts still emitted | more transitions train the heads |
| 3 | Step reaches 1000 | server | safety has not disabled actor | future chunks may use `policy_mode=rlt_actor` |

Invariant: training updates and robot control are deliberately decoupled by the
actor execution gate.

## Operator Invariants

- `control_step` decides freshness for observation and action registers.
- `action_step` decides which executed robot actions belong to a chunk window.
- `rlt_context_id` says "same server inference context."
- `next_rlt_context_id` says "the bootstrap context after this source window."
- `episode_id` says "same labeled critical phase," not the whole rollout.
- Replay writes happen on the server, not the client.
- A collectable chunk can be VLA pass-through; collection does not imply actor execution.
- TUI training pause blocks trainer updates but does not block replay collection.
- `rlt_online_replay.pt` resumes compact replay; `rlt_head_latest.pt` must be wired
  through `rlt_head_checkpoint` if the next run should load the trained actor.
- Discarding a critical phase drops buffered client transitions before they reach
  the server replay.

## Code Anchors

- Shell orchestration: `scripts/run_drtc_experiment.sh`
- Experiment YAML: `examples/experiments/configs/baseline_molmoact2_rlt_so101_online.yaml`
- YAML to client config: `examples/experiments/run_drtc_experiment.py`, `ExperimentConfig`, `create_client_config`, `run_experiment`
- Client runtime: `src/lerobot/async_inference/robot_client_drtc.py`
  - startup: `RobotClientDrtc.start`
  - control loop: `control_loop`
  - observation sender: `observation_sender`
  - action receiver: `action_receiver`, `_handle_actions_dense`
  - RLT collection: `_rlt_start_rollout`, `_rlt_start_critical_phase`, `_rlt_maybe_emit_transitions`, `_rlt_label_current_critical_phase`
  - transition sender: `_rlt_transition_sender`
- Server runtime: `src/lerobot/async_inference/policy_server_drtc.py`
  - policy setup: `SendPolicyInstructions`, `_configure_rlt_online`
  - observation/action registers: `SendObservations`, `_inference_producer_loop`, `_publish_dense`, `StreamActionsDense`
  - RLT context cache: `_cache_rlt_source_context`, `_accept_rlt_transition`
  - online trainer: `_rlt_trainer_loop`, `_save_rlt_head_checkpoint`
  - shutdown persistence: `_reset_server`, `stop`
- Wire fields: `src/lerobot/transport/services.proto`, `ActionsDense`, `RLTTransitionChunk`
- Freshness primitive: `src/lerobot/async_inference/lww_register.py`
- MolmoAct2 RLT wrapper: `src/lerobot/rl/rlt_molmoact2.py`
- Replay format: `src/lerobot/rl/rlt_buffer.py`
- TUI controls: `scripts/drtc_tui.py`

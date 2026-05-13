# Tinypi05v2 RLT Online Sim Findings

Date: 2026-05-13

## Scope

Starting from `scripts/run_drtc_experiment.sh` and
`examples/experiments/configs/baseline_tinypi05v2_rlt_sim.yaml`, I added a
paper-cadence online RLT-head configuration for the Squint SO101 simulator and
updated the DRTC/RLT path so the RLT actor is trained online from sim
transitions and can take over execution after warmup.

The new config is:

```bash
examples/experiments/configs/baseline_tinypi05v2_rlt_sim_paper_online.yaml
```

Run command used for the successful smoke:

```bash
./scripts/run_drtc_experiment.sh --config baseline_tinypi05v2_rlt_sim_paper_online
```

## Paper-Cadence Alignment

The config and code now map the RLT paper's online loop as follows:

| Paper detail | Local implementation |
| --- | --- |
| RLT actor outputs a `C=10` action chunk | `rlt_chunk_size: 10` |
| Rollout collects overlapping chunks at stride 2 | DRTC `s_min: 2`, and shifted RLT windows are queued from collectable chunks |
| Warm up replay with VLA/reference rollouts | `rlt_warmup_episodes: 1`, `rlt_warmup_transitions: 24`, bootstrap dataset episode 0 |
| High update-to-data ratio | `rlt_utd_ratio: 5` trainer passes per tick |
| Two critic TD updates per actor update | new `rlt_critic_updates_per_actor: 2` path |
| Critic ensemble of 2 | `rlt_num_critics: 2` |
| Reference dropout in actor training | `rlt_reference_dropout_p: 0.5` |
| Sparse success reward | terminal episode labels are backfilled to every queued transition |
| Actor refines VLA reference chunk online | server replaces only the live RTC window, not the latency/frozen prefix |

The smoke config intentionally uses `fps: 10` because that is the existing
local sim control rate. The paper's hardware runs at 50 Hz.

## Implementation Changes

- Added `baseline_tinypi05v2_rlt_sim_paper_online.yaml` while leaving the
  original sim baseline untouched.
- Plumbed `rlt_critic_updates_per_actor`, `rlt_success_sample_fraction`,
  `rlt_intervention_sample_fraction`, and `rlt_intervention_reference_mode`
  through experiment config, client config, remote policy config, and server
  status.
- Updated online training so each actor update can be preceded by multiple
  critic updates; the smoke uses 2 critic updates and 5 outer updates per tick.
- Extended the replay sampler to support success-prioritized and
  intervention-prioritized batch fractions, and to return `success` and
  `failure` tensors.
- Backfilled terminal episode labels to all queued transitions in the episode,
  so sparse reward supervision is available across the stored chunks.
- Made intervention reference handling configurable:
  - `executed`: paper-style replacement of the reference chunk with the
    executed/intervened chunk.
  - `original`: keep the VLA reference as the conditioning/reference chunk
    while training the target against the executed chunk.
- Fixed the Squint sim robot client path to record the actual action returned
  by `send_action`, after clipping/bootstrap action substitution, instead of
  recording the requested action or post-step qpos.
- Added deterministic sim reset support with `sim_reset_seed_on_terminal` for
  this controlled smoke.
- Added `ActionsDense.rlt_window_start_index` to the transport proto and
  regenerated the Python gRPC bindings.
- Fixed the live RLT/DRTC alignment bug: under RTC, the first rows of the 50
  step action chunk can be latency/frozen prefix rows. The RLT actor now
  refines `reference[:, window_start:window_start+C]`, where `window_start`
  comes from the current inference delay, and the client stores the same
  shifted window for training.

The last item was the decisive fix. Earlier runs showed the actor fitting the
stored successful samples offline, but most live edits landed in the frozen
prefix and therefore did not affect the future actions that actually executed.

## Successful Run Evidence

Run artifacts:

```bash
logs/policy_server_20260513_150604.log
logs/drtc_status_20260513_150604.jsonl
outputs/rlt_tinypi05v2_squint_marker_sim_paper_online_v7/
outputs/rlt_tinypi05v2_squint_marker_sim_paper_online_v7/videos/
```

The run completed normally after 240 seconds. It produced:

- 21 labeled episodes.
- 615 accepted online transitions.
- 2,890 online RLT training steps.
- 13 replay/archive saves.
- Checkpoints every 250 steps through `rlt_head_step_002750.pt`, plus
  `rlt_head_latest.pt`.
- No `rlt_safety_violation` or `rlt_actor_disabled_by_safety` events.

Episode labels from `logs/drtc_status_20260513_150604.jsonl`:

```text
15:06:29 episode 1  success  transitions=21
15:06:41 episode 2  failure  transitions=30
15:06:52 episode 3  failure  transitions=31
15:07:03 episode 4  failure  transitions=31
15:07:17 episode 5  failure  transitions=31
15:07:28 episode 6  failure  transitions=30
15:07:39 episode 7  failure  transitions=31
15:07:50 episode 8  failure  transitions=31
15:08:01 episode 9  failure  transitions=30
15:08:12 episode 10 failure  transitions=31
15:08:24 episode 11 failure  transitions=31
15:08:35 episode 12 failure  transitions=31
15:08:46 episode 13 failure  transitions=31
15:08:57 episode 14 failure  transitions=30
15:09:08 episode 15 failure  transitions=30
15:09:15 episode 16 success  transitions=15
15:09:26 episode 17 failure  transitions=30
15:09:37 episode 18 failure  transitions=31
15:09:49 episode 19 failure  transitions=30
15:10:00 episode 20 failure  transitions=29
15:10:11 episode 21 failure  transitions=30
```

The first success is the bootstrap/VLA warmup episode. The second success is
the important one: episode 16, saved as:

```bash
outputs/rlt_tinypi05v2_squint_marker_sim_paper_online_v7/videos/episode_0015_success.mp4
```

By that point the actor had already been enabled. Training crossed
`rlt_execute_after_train_steps: 1000` at 15:07:58, and the server diagnostics
later reported both warmup and actor execution counts:

```text
rlt_policy_mode_warmup=302
rlt_policy_mode_rlt_actor=458
```

Representative training metrics:

```text
first train step: rlt_train_step=1, replay=51, actor_bc_loss=0.5256487,
                  actor_q_mean=0.0131381, success_batch=1.0,
                  intervention_batch=1.0
train step 1000: replay=236, actor_bc_loss=0.003481,
                 actor_q_mean=0.18958
last train step:  rlt_train_step=2890, replay=615, actor_bc_loss=0.0016458,
                  actor_q_mean=0.2335522, success_batch=1.0,
                  intervention_batch=0.4375
```

This satisfies the smoke target: the online RLT head starts learning from sim
replay, is used for live action generation, and produces at least one
post-warmup success in sim.

## Iteration Notes

Earlier runs were useful but did not satisfy the target:

- `_v2`: actor eventually executed, but the single bootstrap success was
  diluted by uniform replay.
- `_v3`: success/intervention sampling and paper-style executed reference
  improved batch composition, but the actor still failed live.
- `_v4`: original-reference intervention mode, action noise, and filtering did
  not produce post-warmup success.
- `_v5`: deterministic actor and disabled action filter still failed.
- `_v6`: deterministic reset proved the actor could fit the successful samples
  offline, but live execution still failed. This exposed the RTC window
  alignment issue.
- `_v7`: shifted the RLT actor/training window past the live RTC prefix and
  produced the episode 16 actor-era success.

## Caveats

- The smoke config uses `rlt_intervention_reference_mode: original`, not the
  paper-default `executed`, because the bootstrap/intervention path in this sim
  otherwise trains on corrected references while inference conditions on the
  uncorrected VLA reference. The code supports both modes and defaults to
  `executed`.
- `sim_reset_seed_on_terminal: true` was used to make the smoke reproducible
  enough to verify learning after each code change.
- `rlt_action_std: 0.0` and `action_filter_mode: none` were used for the final
  smoke to remove exploration and filtering as confounders.
- The result is a start-of-learning smoke, not a robustness claim. There was
  one post-warmup success in the final 21-episode run.

## Verification

Focused tests passed after the code changes:

```bash
uv run --no-sync pytest \
  tests/transport/test_transport_utils.py::test_rlt_action_metadata_and_transition_chunk_roundtrip \
  tests/rl/test_rlt_online.py \
  tests/async_inference/test_robot_client_drtc.py \
  tests/async_inference/test_squint_so101_sim.py \
  tests/async_inference/test_policy_server_drtc.py
```

Result: 25 passed.

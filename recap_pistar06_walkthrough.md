# RECAP / PiStar06 Walkthrough — `jackvial/recap_baseline_120_v2`

End-to-end guide for running the **`pistar06`** (advantage-conditioned Pi0.5)
checkpoint published at [`jackvial/recap_baseline_120_v2`](https://huggingface.co/jackvial/recap_baseline_120_v2)
on a SO-101 robot. This is the PiStar06 counterpart of `recap_walkthrough.md`.

The big differences vs. the `pi05_full` walkthrough:

- **No subtask annotation step.** PiStar06 conditions on a fixed
  `Advantage: positive` text snippet that the policy injects itself — there is
  no subtask head, no subtask tokens, and no manual labelling tool.
- **No critic head, no critic-overlay video, no critic JSON.** PiStar06 was
  trained against a *frozen* SmolVLA value network during data labelling but
  ships at inference without an embedded critic.
- **Different policy module + inference script.** Use
  `lerobot.rl.inference_pistar06_async` and `config-pistar06.json`. The
  existing `inference_pi05_async` flow keeps working unchanged.

---

## Prerequisites

- [x] LeRobot repo on the `jv/carlos-recap-main` branch (or any branch where
      the `lerobot.policies.pistar06` package and
      `lerobot.rl.inference_pistar06_async` script have been merged).
- [x] SO-101 follower + leader teleop wired up exactly as in
      `recap_walkthrough.md`.
- [ ] [`jackvial/recap_baseline_120_v2`](https://huggingface.co/jackvial/recap_baseline_120_v2)
      pulled to local disk.

Pull the checkpoint if you haven't already:

```bash
uv run python -c "from huggingface_hub import snapshot_download; \
snapshot_download('jackvial/recap_baseline_120_v2', local_dir='models/recap_baseline_120_v2')"
```

That directory should contain at least:

| File | Purpose |
|---|---|
| `model.safetensors` | PiStar06 transformer weights |
| `config.json` | `PiStar06Config` (must have `"type": "pistar06"`) |
| `policy_preprocessor.json` + `state_*.safetensors` | Pre-/post-processor pipeline + normalization stats |

Sanity check the policy type:

```bash
uv run python -c "
import json
print(json.load(open('models/recap_baseline_120_v2/config.json'))['type'])
"
```

You should see `pistar06`. If it says `pi05_full` or `pi05_rl`, you've grabbed
the wrong checkpoint.

---

## Step 1: ~~Annotate Dataset with Subtasks~~ — Skipped

PiStar06 does **not** consume subtask tokens. The `manual_subtask_annotate`
Gradio tool, the bulk-annotation scripts, and the
`outputs/*_w_subtasks_*` dataset variant are all unnecessary. If you only
plan to run inference from the pre-trained Hub checkpoint you can skip
ahead to **Step 4**.

If you intend to *train* your own PiStar06 from scratch (advanced — training
lives on `jv/recap-value-network`, not this branch), you still don't need
subtask labels. You instead need an **advantage-labelled** dataset produced
by running the SmolVLA value network over your demonstrations. That flow
isn't covered here.

---

## Step 2: Compute Anchor Action Stats — Optional

`PiStar06Config` doesn't expose an `action_encoding` field, so the inference
pipeline runs in `"absolute"` action mode and does **not** need
`action_stats_anchor_*.pt`. You only need this file if you're going to do
your own offline training with anchor encoding on `jv/recap-value-network`.

If you want the file anyway:

```bash
uv run python src/lerobot/scripts/compute_delta_stats.py \
    --root outputs/so101_pickplace_success_120_v2 \
    --encoding anchor \
    --output-dir outputs/stats_so101_pickplace_success_120_v2
```

(Note the dataset path: there's no `_w_subtasks` suffix because PiStar06
doesn't need it.)

---

## Step 3: Pre-decode the Dataset — Optional, Training Only

Same caveat as Step 2: this is only useful if you're running offline training
(e.g. fine-tuning the Hub checkpoint on your own demonstrations). For
inference-only usage, skip this step.

If you do want it:

```bash
uv run python src/lerobot/scripts/lerobot_memmap_buffer_cache.py \
    --repo-id jackvial/so101_pickplace_success_120_v2 \
    --data-dir outputs/so101_pickplace_success_120_v2 \
    --cache-dir outputs/buffer_cache \
    --video-backend pyav
```

The cache layout is identical to the one described in `recap_walkthrough.md`,
just without the per-episode `subtask_index` column.

---

## Step 4: Create Your Inference Config

A starter config has been committed at `config-pistar06.json` at the repo
root. The fields you almost certainly need to edit:

| Field | Value |
|---|---|
| `policy.pretrained_path` | `"models/recap_baseline_120_v2"` (or wherever you downloaded it) |
| `env.task` | `"Pick up the orange cube and place it on the black X marker"` (your task description) |
| `env.robot.port` / `env.teleop.port` | match the serial paths on your machine |
| `env.robot.cameras.{top,side}.index_or_path` | match `/dev/videoX` for your hardware |
| `output_dir` | where rollouts + videos should be written |

Notable differences vs. `config-recap.json`:

- `policy.type` is `"pistar06"`, not `"pi05_rl"`.
- `policy.pretrained_path` replaces `policy.pi05_checkpoint`.
- The `policy.task` field is **gone** — the worker reads `cfg.env.task`
  instead, because `PiStar06Config` (which extends `PI05Config`) has no
  `task` attribute.
- All `pi05_rl`-only knobs are stripped:
  `inference_advantage`, `use_separate_critic`, `knowledge_insulation`,
  `loss_weight_*`, `subtask_regeneration_interval`, `advantage_scaling`,
  `reward_normalization_constant`, `terminal_failure_reward`, every
  `online_*`, every `actor_*` / `critic_*` / `learner_*` knob,
  `concurrency`, `trainable_params`, `discount`, `num_critics`,
  `utd_ratio`, `gradient_accumulation_steps`, `policy_update_freq`,
  `online_step_before_learning`, `actor_learner_config`, `dataset_stats`,
  `storage_device`, `online_buffer_capacity`, `async_prefetch`,
  `action_encoding`, `action_encoding_stats_path`.
- New PiStar06-specific fields are surfaced (with safe defaults that match
  the trained Hub checkpoint):
  `enable_advantage_conditioning`, `advantage_threshold`, `cfg_beta`.
- `interactive` is honored as a no-op for PiStar06 (the script logs a
  warning explaining why and continues in non-interactive mode).

If your camera config or features differ from the Hub model, `make_policy`
will raise a feature-shape mismatch on startup. Match the
`policy.input_features` / `policy.output_features` block against
`models/recap_baseline_120_v2/config.json`.

---

## Step 5: ~~Run Offline Training~~ — Not on This Branch

The `pistar06` training pipeline (`recap_train_pi_star.py`, the SmolVLA
value-network labelling, the advantage-conditioned data loader, etc.)
lives on `jv/recap-value-network`, not on `jv/carlos-recap-main`. Only the
**inference / rollout-recording** pipeline has been ported to this branch.

If you want to fine-tune the Hub checkpoint on your own data you'll need to
check out `jv/recap-value-network` and follow the training instructions
there. The Hub model can be plugged in as the starting checkpoint.

---

## Step 6: Verify the Checkpoint

Check that the downloaded checkpoint loads cleanly **without** running on
real hardware:

```bash
uv run python -c "
import lerobot.policies.pistar06
from lerobot.policies.pistar06 import PiStar06Config, PiStar06Policy
cfg = PiStar06Config.from_pretrained('models/recap_baseline_120_v2')
print('policy type =', cfg.type)
print('paligemma variant =', cfg.paligemma_variant)
print('advantage threshold =', cfg.advantage_threshold)
print('cfg_beta =', cfg.cfg_beta)
"
```

Then verify the safetensors weights count looks sane:

```bash
uv run python -c "
from safetensors.torch import safe_open
with safe_open('models/recap_baseline_120_v2/model.safetensors', framework='pt') as f:
    keys = list(f.keys())
print('num tensors:', len(keys))
print('has gemma_expert:', any('gemma_expert' in k for k in keys))
print('has action_out_proj:', any('action_out_proj' in k for k in keys))
print('has critic:', any('critic' in k for k in keys))   # should be False — PiStar06 has no critic
"
```

You should see `has critic: False`. If `True`, you've downloaded a `pi05_rl`
checkpoint by mistake; re-pull the right repo.

---

## Step 7: Inference and Rollout Recording

This is the main use-case for PiStar06 on this branch.

- [ ] Confirm `policy.pretrained_path` in `config-pistar06.json` points to
      `models/recap_baseline_120_v2` (or your local copy of it).
- [ ] Launch inference:

```bash
uv run python -m lerobot.rl.inference_pistar06_async --config_path config-pistar06.json
```

### What to expect

1. The script registers the `pistar06` policy class, loads the model onto
   your GPU, runs `_verify_loaded_weights` to confirm the checkpoint matches
   the in-memory model (you'll see `[WEIGHT CHECK] PASSED`), connects to the
   robot + cameras, and waits.
2. You'll see `[ENV] Waiting for '2' on the teleop device to start
   episode...` in the terminal.
3. Press **`2`** on the leader-arm keyboard to start the first episode. The
   robot will execute the policy at 30 Hz using RTC (asynchronous chunked
   inference).
4. A supervisor log prints every 5 seconds showing queue depth, intervention
   status, and timing metrics.

### Intervention controls

| Key | Action |
|---|---|
| `2` | Start the next episode |
| `5` | Toggle intervention — take/release control with the leader arm |
| `1` | Mark episode as **success** and end it |
| `0` | Mark episode as **failure** and end it |

Intervened timesteps are flagged as high-quality human actions in the saved
buffer (same as `pi05_rl`).

### What gets saved

- **Replay buffer as a LeRobot dataset** — flushed every `episode_save_freq`
  episodes (default 10 in `config-pistar06.json`) to
  `{output_dir}/inference_dataset/`. Contains observations, actions,
  rewards, done flags, and intervention labels.
- **Plain MP4 rollout videos** — written every `episode_logging_freq`
  episodes (default 1) under
  `{output_dir}/logging_episodes/episode_XXXXXX/episode_XXXXXX_<cam>.mp4`,
  one file per camera in `video_logging_cameras`. **No critic-value
  overlay**, no `critic_values.json`, no critic plot — PiStar06 has no
  critic head to query.
- Per-frame PNGs land in the same `episode_XXXXXX/` directory in case you
  want to re-encode externally or build your own overlay.

### Tuning the config for inference

| Field | Default in `config-pistar06.json` | Notes |
|---|---|---|
| `episode_save_freq` | `10` | How often the buffer is flushed to a LeRobot dataset |
| `episode_logging_freq` | `1` | How often a plain rollout MP4 is rendered |
| `policy.enable_advantage_conditioning` | `true` | Master switch for the `Advantage: positive` text injection. Set `false` to run vanilla Pi0.5 behaviour. |
| `policy.advantage_threshold` | `0.0` | Resolved advantage threshold from training (you almost never want to change this at inference) |
| `policy.cfg_beta` | `1.0` | Classifier-free-guidance scale. `>1.0` sharpens with CFG at inference. |
| `policy.value_network_checkpoint` | `null` | Path to the SmolVLA value-network checkpoint. **Unused at inference**, harmless to leave null. |
| `interactive` | `false` | Honored as a no-op (the script warns and continues). |
| `use_rerun` | `false` | Set `true` to stream camera feeds to a Rerun viewer in real time. |

### Iterating on rollouts

Because PiStar06 doesn't need subtask labels, the post-rollout loop is
simpler than the `pi05_full` flow:

1. Collected rollouts already live at
   `{output_dir}/inference_dataset/` as a standard LeRobot dataset.
2. To use them for further fine-tuning you'd run them through the
   PiStar06 training pipeline on `jv/recap-value-network` (which will
   compute its own per-frame advantages with the SmolVLA value network).
   You do **not** need to call `online_buffer_to_video.py` or
   `subtask_annotate_gemma_4.py`.
3. Re-pull the new checkpoint, point `policy.pretrained_path` at it, and
   collect another batch.

---

## Step 8 (Optional): Online Training

**Not supported on this branch.** PiStar06 online RL would need an
advantage-aware actor-learner pair, which doesn't currently exist —
the published actor/learner scripts (`actor_pi05_async.py`,
`learner_pi05.py`) are tied to `PI05RLConfig` and the `pi05_full`
critic. If you want online RL on PiStar06, run the inference loop above
to collect rollouts, then run offline training on
`jv/recap-value-network`.

---

## Troubleshooting

**`Could not find a choice class for 'pistar06'`**
→ The policy registry hasn't seen the `pistar06` module. Make sure your
branch contains `src/lerobot/policies/pistar06/` and that
`lerobot.rl.inference_pistar06_async` is the entrypoint you're invoking
(it has the side-effect import that registers the class).

**`Could not import GemmaTextScaledWordEmbedding`**
→ `transformers` v4.x doesn't ship that class; it was added in v5. The
ported `modeling_pistar06.py` includes a local fallback that subclasses
`nn.Embedding` and applies `embed_scale` at forward time. If you're seeing
this error, your copy of `modeling_pistar06.py` is older than the port —
re-pull the file or replace the import block with the fallback shim.

**`[WEIGHT CHECK] FAILED: Loaded weights do NOT match checkpoint!`**
→ Either `policy.pretrained_path` is unset / wrong, or you're loading a
checkpoint that was saved with a different architecture (e.g. a `pi05_rl`
checkpoint with critic-head weights named differently). Re-pull
`jackvial/recap_baseline_120_v2` and double-check the path.

**`The fields ... are not valid for PiStar06Config`**
→ You've left a `pi05_rl`-only field in your JSON. The error message lists
exactly which fields are unknown — delete them from `config-pistar06.json`.
The most common offenders are `task` (move it to `env.task`),
`pi05_checkpoint` (rename to `pretrained_path`), `inference_advantage`,
`use_separate_critic`, and the `actor_*` / `critic_*` / `online_*` blocks.

**`FATAL: RTC configuration is not populated or enabled`**
→ Your `policy.rtc_config.enabled` is `false` or missing. Set it to `true`
in `config-pistar06.json`; PiStar06 inference can only run via async RTC.

**Robot port not found**
→ `env.robot.port` and `env.teleop.port` use stable `/dev/serial/by-id/`
paths from `config-recap.json`. Replace them with the actual paths on your
machine (`ls /dev/serial/by-id/`).

**Camera index doesn't exist**
→ `env.robot.cameras.{top,side}.index_or_path` defaults to `/dev/video0`
and `/dev/video4`; adjust to whatever `v4l2-ctl --list-devices` reports.

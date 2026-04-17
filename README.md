# Implementation of a RECAP-like algorithm and other capabilities built on LeRobot

## What is implemented here:

- The full version of π0.5 with subtasks and FAST tokens with subtask annotation utilities. As of 2026/04/14 this has not been added to the `main` branch in lerobot. All credits to [@jadechoghari](https://github.com/jadechoghari).
- End-to-end implementation of RECAP-like algorithm for offline and online training.
- Asynchronous inference with RTC that can run at 30Hz with leader guided human intervention. It automatically saves a buffer and creates a video of the episode with the critic values overlaid.
- Offline eval script to compare actions across checkpoints, subtasks, or advantage labels.
- Experimental: live inference where a user can manually write subtasks for π0.5 on the fly.

The emphasis of this code is RECAP, but other capabilities were added in the process. Even if you don't intend to use RECAP, the other capabilities might be useful.

<video src="https://github.com/user-attachments/assets/9461ba1e-725b-4ee4-8d32-173fa6a86600" controls width="100%"></video>

*Inference on the SO-101 arm. The policy generalizes reasonably well from 60 demonstrations. Subtask generation is functional but shows some collision with FAST token generation (from training). Critic values track behavior accurately, though with occasional jumps.*


We are also happy to accept contributions, feature requests, or any feedback. Please reach out here or on discord @cijerezg.

## Introduction to RECAP


[RECAP](https://arxiv.org/pdf/2511.14759) is the RL algorithm developed by [Physical Intelligence](https://www.pi.website/) that was used to train the π0.6 model.


RECAP proposes an advantage-conditioned VLA, where the advantage comes from a critic that is trained along the policy (the VLA). Specifically, the advantage is computed from the critic, then binarized and passed to the policy as an extended part of the prompt.


This is very useful for two reasons:
1. The policy can now learn from its own experience because there is a grounding signal about the "goodness" of actions (i.e., the advantage value) that is completely driven by reward via the critic. Without this, the policy would make a mistake and just learn to imitate it.
2. Suboptimal demonstrations can be used for training because the critic learns a to distill what part of them is good and what part is bad, i.e., which actions have high advantage vs low.

In the following sections, we highlight the key features of this implementation, how to use it, and an overview of the code in case you want to add your own modifications.


## Key features

### Models

- The policy is π0.5 from LeRobot, and it was hard-coded. The critic shares a similar architecture with fewer layers. Both models were hard-coded throughout the codebase, so it isn't trivial to change them.

- This implementation uses the full version of π0.5, which includes subtask generation and FAST tokens with knowledge insulation.

- This implementation supports absolute actions, anchor actions (i.e., δt = at - s0), and delta actions (i.e., δt = at - at-1). Experimentally, we have found that anchor actions work best as they inherit translation invariance and aren't as prone to drift as pure delta actions.

- Actions are smoothed with a centered moving average (window size 5) applied after each chunk is generated. This reduces jerk throughout the trajectory, which is especially useful when the dataset is collected from the policy — smoother rollouts produce cleaner demonstrations.


### Offline

The file `lerobot.script.offline_learner_pi05.py` supports offline training using RECAP. This can be run instead of `lerobot-train`. 


### Online

- `lerobot.rl.learner_pi05` and `lerobot.rl.actor_pi05_async` support online training using RECAP. The training logic is the same as offline, and the difference is that the actor is collecting experience and streaming it as training happens. `actor_pi05_async` runs the policy on the real robot asynchronously using RTC at 30Hz and streams the data to the learner. The online buffer is periodically saved, so that this data can be used later on for further training. 

- `lerobot.rl.inference_pi05_async` is the equivalent of `lerobot-record` for inference. It works on the real robot and it runs RTC asynchronously at 30Hz. It keeps an online buffer with episode data that is periodically saved. Every few episodes, it also generates the video shown above. The video is also generated in the learner pipeline.


- Interventions using the actor or inference scripts are done using the leader arm. Pressing `5` gives control to the leader arm, and then pressing `5` again gives back to control the policy. Interventions might be executed at any time. Interventions are flagged in the buffer, so that the learner knows those are "good" actions.



## How to use

### Prerequisites

Before training, you need three things:

1. **Dataset** — at least 50 episodes, annotated with subtasks/skills using `lerobot/src/lerobot/policies/pi05_full/annotate/subtask_annotate.py`.
2. **π0.5 base weights** — downloaded locally from Hugging Face. Set the path in `policy.pi05_checkpoint`.
3. **Config file** — everything is driven by `config-hiserl.json`. The key fields to set before doing anything:

| Field | What it does |
|---|---|
| `dataset.root` | Path to your annotated dataset |
| `policy.pi05_checkpoint` | Path to your π0.5 base weights (or offline checkpoint) |
| `policy.task` | Natural language description of the task |
| `output_dir` | Where checkpoints, logs, and buffers are saved |
| `policy.action_encoding` | Action representation: `"absolute"`, `"anchor"`, or `"delta"`. We recommend `"anchor"` |
| `policy.action_encoding_stats_path` | Path to precomputed action stats (required for `"anchor"` and `"delta"`) |

If you use `"anchor"` or `"delta"` encoding, you need to compute the normalization stats first:

```bash
python lerobot/src/lerobot/scripts/compute_delta_stats.py \
    --data-dir /path/to/your/dataset \
    --encoding anchor
```

The output `.pt` file is what you pass to `policy.action_encoding_stats_path`.

### Dataset Annotation

Subtask annotations are required for training. There are two ways to do it.

> **Note — annotating online buffer data:** If your data comes from a saved online buffer (learner or inference), you must convert it to video format first. Run:
> ```bash
> python lerobot/src/lerobot/policies/pi05_full/annotate/online_buffer_to_video.py \
>     --data-dir /path/to/online_buffer \
>     --output-dir /path/to/online_buffer_video
> ```
> The default paths are set at the top of the script — pass them explicitly to avoid editing the file. Use the output directory as `--data-dir` for the annotation command below.
>
> **Note — Python environment:** The annotation scripts require `transformers>=5.3`, which conflicts with the custom `transformers` version used by π0.5. Use a **separate virtual environment** for annotation.

**Option 1 — Automatic (LLM-based):** Two scripts are available: `subtask_annotate.py` (Qwen2-VL / Qwen3-VL) and `gemma_subtask_annotate.py` (Gemma 4). In practice, Gemma 4 works much better than Qwen. Gemma 4 31B produces annotations that are close to manual quality, and even 4B is still decent. Qwen lags noticeably behind both.

```bash
# Gemma 4 (recommended for automatic annotation)
python lerobot/src/lerobot/policies/pi05_full/annotate/gemma_subtask_annotate.py \
    --data-dir /path/to/dataset \
    --video-key observation.images.wrist \
    --output-dir /path/to/annotated_dataset

# Qwen (alternative)
python lerobot/src/lerobot/policies/pi05_full/annotate/subtask_annotate.py \
    --data-dir /path/to/dataset \
    --video-key observation.images.wrist \
    --output-dir /path/to/annotated_dataset
```

**Option 2 — Manual (recommended):** A Gradio-based UI that lets you scrub through each episode and label skill boundaries with a slider. Much more reliable.

```bash
python lerobot/src/lerobot/policies/pi05_full/annotate/manual_subtask_annotate.py
```

Open the printed URL (usually `http://localhost:7860`) in your browser.

### Phase 1: Offline Training

Always start here. It gives the policy and critic a reasonable initialization — skipping this tends to make online training unstable.

```bash
python -m lerobot.scripts.offline_learner_pi05 --config_path config-hiserl.json
```

Train for roughly 8,000–10,000 steps (`policy.offline_steps`). Checkpoints land in `output_dir/checkpoints/`. You can override the output path with `offline_output_dir` if you want to keep offline and online runs separate.

### Phase 2: Online Training

Online training runs two separate processes: a **learner** that updates weights, and an **actor** that runs the robot.

**Start the learner:**
```bash
python -m lerobot.rl.learner_pi05 --config_path config-hiserl.json
```
Point `policy.pi05_checkpoint` to your offline checkpoint, or use `resume: true` to continue from a previous online run.

**Start the actor** (separate terminal):
```bash
python -m lerobot.rl.actor_pi05_async --config_path config-hiserl.json
```
The actor runs RTC at 30 Hz and streams experience to the learner. Make sure `policy.rtc_config.enabled` is `true`. The learner and actor communicate over gRPC — set `policy.actor_learner_config.learner_host` to the learner's IP.

### Interventions

Both the actor and inference scripts open a [Rerun](https://rerun.io/) window that streams the robot's camera feeds and critic values in real time. **We recommend watching the robot view in Rerun** — not just the physical robot — when deciding whether to intervene, since you can see the exact frames the policy is seeing along with the advantage signal.

You intervene using the leader arm. The keys are:

| Key | Action |
|---|---|
| `5` | Toggle intervention — take/release control |
| `1` | Mark episode as **success** and terminate |
| `0` | Mark episode as **failure** and terminate |
| `2` | Start next episode |

Only intervene when the policy is clearly drifting. Intervened steps are flagged as high-quality actions in the buffer, so keep them clean.

### Inference (no training)

To run the policy on the robot without training, use the inference script. It behaves like `lerobot-record`: runs RTC at 30 Hz, saves an online buffer, and opens the same Rerun window.

```bash
python -m lerobot.rl.inference_pi05_async --config-path config-hiserl.json
```

Every few episodes it also renders an annotated video with the subtasks predicted by the VLA and the critic values over time (see the example in the Key Features section above). Useful for evaluating a checkpoint before committing to more training.

### Iterative Loop

RL training is a cycle. After each online session:

1. **Convert** the raw `online_buffer` to video format:
   ```bash
   python lerobot/src/lerobot/policies/pi05_full/annotate/online_buffer_to_video.py
   ```
2. **Annotate** the new data with subtasks:
   ```bash
   python lerobot/src/lerobot/policies/pi05_full/annotate/subtask_annotate.py \
       --data-dir /path/to/online_buffer_video \
       --video-key observation.images.wrist \
       --output-dir /path/to/online_buffer_annotated
   ```
3. **Merge** by adding the annotated path to `dataset.additional_offline_dataset_paths` in your config, then run offline training again.

Rinse and repeat. Each cycle the policy sees more on-task experience and the critic gets a better signal.




## Code architecture

A map for contributors: where to go depending on what you want to change.

### `rl_pi05.py` — Config, Policy & Critic

Three things live here: the master config dataclass (`PI05RLConfig`), the critic model (`Pi05TransformerCritic`), and the full policy class (`PI05RLPolicy`). If you're adding a hyperparameter, this is where it goes. If you're touching the critic or the policy class itself, same file.

`PI05RLPolicy` is the complete model object. It subclasses `PI05FullPolicy` and swaps in `PI05RLPytorch` as its model — an extension of the base π0.5 forward pass with advantage conditioning wired in. It also instantiates the critic and critic target, handles checkpoint loading (detecting whether a checkpoint is a base π0.5 or an RL checkpoint by inspecting state dict keys), and exposes `self.actor = self.model` as an alias for compatibility with the learner infrastructure. When loading from a base checkpoint, critic weights are initialized by copying the first N layers of the actor.

The critic is a 6-layer Gemma transformer (configurable via `critic_llm_depth`). It receives vision features and text embeddings, appends 32 learned query tokens, runs them through the transformer, and projects the query outputs through a SwiGLU MLP to produce a scalar value. The query tokens are the bottleneck: they're what actually gets trained to summarize the full context into a value estimate.

### `pi05_train_utils.py` — Shared Training Logic

The heart of training. Both `learner_pi05.py` and `offline_learner_pi05.py` call into here. It contains the Bellman update, the flow matching loss, action encoding transforms (anchor/delta applied at train time, not in the buffer), and the full batch construction pipeline.

One subtlety worth knowing: for online transitions, the actor stores the exact subtask tokens it used during inference and those are passed through directly here, bypassing the preprocessor's tokenization, so the flow loss sees the same conditioning context the actor saw.

Key functions:
- **`make_pi05_preprocessor_and_postprocessor`** — builds the preprocessing pipeline and resolves normalization stats. Priority order: base checkpoint → use dataset stats directly; trained checkpoint → extract the saved normalizer state from inside it; anchor/delta encoding → additionally load `action_encoding_stats_path` and override the action stats entry. This is where to look when normalization behaves unexpectedly.
- **`remap_subtasks_for_dataset`** — when merging datasets with different subtask sets, computes the index remapping and extends the target dataset's metadata in memory. Shared subtasks are matched by name; new ones get appended.
- **`hydrate_subtasks`** — converts subtask index tensors to human-readable strings for logging and video overlays. Index `-1` maps to an empty string (used for unannotated online frames).

### `learner_pi05.py` vs `offline_learner_pi05.py`

`offline_learner_pi05.py` is `learner_pi05.py` minus the gRPC server and the online buffer. The training loop calls the same functions from `pi05_train_utils.py`. Layer freezing is controlled entirely via `policy.trainable_params` in the config — the action expert is always unfrozen, and vision/language/critic layers are gated by their respective `_from_layer` thresholds.

### RTC, Anchor Actions & Per-Timestep Normalization

RTC decouples inference from environment stepping: the policy generates action chunks on a background thread while the main thread consumes them at 30 Hz. When the queue runs low, a new chunk is generated and merged with whatever actions are left over from the previous one.

The complication with anchor encoding is that leftover actions reference a stale anchor s0. `align_prev_actions` in `actor_pi05_async_utils.py` corrects this in absolute space, then re-normalizes. What makes it non-trivial is per-timestep normalization: stats have shape `[chunk_size, action_dim]`, so each position in the chunk has its own mean/std. You can't just shift normalized values directly — the function right-aligns the leftover into a padded buffer before unnormalizing (so each value uses the stats for its original chunk position), applies the anchor delta, then re-normalizes left-aligned for the next chunk. For delta encoding with offset > 0, no correction is needed since leftover actions are consecutive diffs that don't reference s0.

Normalization by modality: actions use `QUANTILES`, joint state uses `MIN_MAX`, images use `IDENTITY`.

### Actor/Inference Async Architecture

`actor_pi05_async.py` and `inference_pi05_async.py` follow the same pattern: a main thread steps the environment at 30 Hz, a background thread runs GPU inference, and a `SharedState` object with a lock passes observations between them. The actor variant (`actor_pi05_async_utils.py`) also manages the gRPC connection to the learner — pushing transitions and pulling updated weights. The inference variant (`inference_utils.py`) drops all of that: simpler `SharedState`, no network I/O.

After each action chunk is generated, the absolute actions pass through a centered moving average (window size 5) to smooth jerk throughout the trajectory.

### `buffer.py` — Replay Buffer

A few implementation details worth knowing:

- **Lazy init**: storage tensors aren't allocated until the first `add()` call, so shapes are inferred from the first transition.
- **Memory optimization**: by default, `next_states` is not stored separately — it's a shifted view into `states`. This halves image memory but assumes `next_state[i] == state[i+1]`, which holds within an episode. Episode boundaries are tracked via `episode_ends` to avoid sampling across them.
- **Everything is stored in bfloat16**, regardless of training dtype.
- **`complementary_info`**: an arbitrary dict stored alongside each transition. This is the extension point — subtask indices, golden flags, and actor subtask tokens all go through here.

### `lerobot_memmap_buffer_cache.py` — Dataset Pre-decoding (credit: [@jackvial](https://github.com/jackvial))

Pre-decodes a LeRobot dataset into a flat memmap cache so the replay buffer can be populated almost instantly on subsequent runs instead of re-decoding video every time. 
```


### `utils.py`

Two functions worth calling out:

**`preprocess_batch_for_pi05`** — the preprocessing entry point for the offline path. Takes raw observations and actions, applies action encoding transforms, assembles `complementary_data` with the advantage value, runs the preprocessor pipeline for both current and next observations, and assembles the final forward batch with all the token fields the policy expects. If you're tracing missing keys or wrong shapes in a batch, start here.

**`save_video_with_critic_overlay`** — takes a directory of per-frame PNGs and a list of critic values, stitches the camera views side-by-side (448×448 each), and draws a critic curve in the lower half. The curve renders twice: a faint full-episode trace and a solid progressing line up to the current frame with a vertical marker. Subtask text, when available, is overlaid in the top-left corner. After the video is written, the individual PNGs are cleaned up. This is what generates the episode videos in both the learner and inference pipelines.


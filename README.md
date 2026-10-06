# LeRobot for Research

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repo is a research-oriented fork of [LeRobot](https://github.com/huggingface/lerobot). The focus is on SOTA algorithms (e.g., RECAP, advantage-weighted regression), a SOTA model (MolmoAct2) and tooling for examining the model's internals such as attention maps and clustering of internal representations.

This is an active project and we keep trying ideas, so expect things to move. [`docs/status.md`](docs/status.md) says what is switched on right now.

Results obtained using MolmoAct2 on anchor actions:

https://github.com/user-attachments/assets/fa476815-7a97-4b58-a62d-7ee1dcd91d88

A longer video with successes, recoveries, and failures is available on [YouTube](https://youtu.be/rhE2HRvlQMI).

## Key features

- Full support for the [MolmoAct2](https://allenai.org/blog/molmoact2) model: finetuning, inference, changing action space, attention maps, among others.
- End-to-end implementation of a [RECAP](https://arxiv.org/pdf/2511.14759)-like algorithm for offline and online training, plus frozen-critic advantage-weighted regression.
- A metric depth path (wrist RealSense point map into the VLM prefix), prompt-rendered short-term history, subtask generation and metadata steering (quality, mistakes, speed, precision, contact).
- Asynchronous inference with RTC that runs up to 30 Hz with leader-guided human intervention.
- A suite of validation probes to examine the model's internals: attention maps, representation clusters, critic sensitivity, steering sweeps, and a browser viewer to compare them across checkpoints.

Target hardware is the rebot B601 follower with a 102HD leader. Older SO-101 and $\pi_{0.5}$ setups are no longer maintained in this fork.

## How to install

Clone this repo

```bash
git clone https://github.com/cijerezg/lerobot.git
cd lerobot
```

and set up the environment:

```bash
uv sync --extra molmoact2 --extra async --extra training
```

## Quick start

### Things to know

- MolmoAct2 is the only policy the offline trainer accepts (`policy.type: molmoact2_rl`).
- For human intervention, only the follower-leader setup is supported.
- This repo is geared toward real robots, so there is no simulation support.
- Copy-paste commands for the rebot setup are in [`CHEAT_SHEET.md`](CHEAT_SHEET.md).

### Prerequisites

#### Dataset

The quality of the dataset is extremely important. We suggest at least 50 episodes with a consistent strategy for task execution, e.g., try to grasp objects in a consistent way as much as possible. MolmoAct2 has shown generalization capabilities, so we suggest a diverse dataset where objects are moved throughout the scene.

> [!IMPORTANT]
> This pipeline assumes a **LeRobot v3.0 dataset format** (introduced in `lerobot 5.0.0`). If your dataset is v2.1, see [Using a v2.1 Dataset](docs/guide/usage.md#using-a-v21-dataset) for the one-command migration.

**Pre-decode the dataset to a cache.** Images are stored at their original resolution, so the offline buffer would otherwise hold every decoded frame in RAM at startup. With the cache, the dataset lives on disk and only the data sampled during a training step is loaded into memory; subsequent runs load instantly.

Generate the cache once per dataset:

```bash
python -m lerobot.scripts.lerobot_memmap_buffer_cache \
    --root /path/to/local/dataset \
    --cache-dir outputs/buffer_cache \
    --image-storage-dtype uint8 \
    --image-storage-size 480 640
```

You can also pass `repo-id` instead of `root` if the dataset is on the HF hub. Then point the YAML at it with `buffer_cache_dir: outputs/buffer_cache`. More on edge cases in [the usage guide](docs/guide/usage.md#buffer-caching).

#### Config file

One YAML config drives every script in this repo. The in-package template is [`src/lerobot/rl/config_rl.yaml`](src/lerobot/rl/config_rl.yaml); the live copy we run from sits at the workspace root as `config_rl.yaml`.

To get started with training, the key fields to change are:

- `root`: the path to your dataset.
- `task`: the task prompt.
- `base_path`: a local copy of the HF MolmoAct2 model, e.g. `hf download allenai/MolmoAct2`.
- `pretrained_path`: path to your finetuned model, or null if you don't have one. `base_path` must still be set either way, since it supplies the architecture and norm stats.

> [!NOTE]
> Naming heads-up: this fork keeps the upstream LeRobot field name `pretrained_path`, but `base_path` is the actual pretrained foundation model. Read `pretrained_path` as "finetune to load on top of base."

Read the whole config before launching a run.

#### Action encoding

Decide how actions are represented before any training. The same encoding has to be used end-to-end; switching later means recomputing statistics and retraining from base.

- `absolute`: raw joint positions. Simplest, but does not generalize across starting configurations.
- `anchor` (recommended): offsets from the chunk's initial state. Translation-invariant, generalizes well.
- `delta`: first-order differences between consecutive actions. Compact, but errors can accumulate.

Set the choice via `policy.action_encoding`. `anchor` and `delta` also require precomputed normalization statistics; see [Action Encodings](docs/guide/usage.md#action-encodings) for the script that generates them.

### Training

Start with offline training so the policy has a good starting point:

```bash
uv run python -m lerobot.scripts.rl_offline --config_path path/to/config.yaml
```

> [!NOTE]
> Offline training runs validation probes that render MP4 artifacts. On Linux these can fail with `[Errno 12] Cannot allocate memory`, a virtual-memory overcommit accounting quirk with large PyTorch processes, not actual OOM. Persistent fix:
>
> ```bash
> echo 'vm.overcommit_memory = 1' | sudo tee /etc/sysctl.d/99-overcommit.conf
> sudo sysctl --system
> ```
>
> Background and tradeoffs in [docs/runbooks/system_overcommit.md](docs/runbooks/system_overcommit.md). If you'd rather not touch sysctl, set `val_on_start` to false and `val_freq` to a large number.

Once offline training has run for a while, set `pretrained_path` to the resulting checkpoint and proceed to online training. Run the learner:

```bash
uv run python -m lerobot.rl.rl_learner --config_path path/to/config.yaml
```

and on another terminal the actor:

```bash
uv run python -m lerobot.rl.rl_actor_async --config_path path/to/config.yaml
```

The learner saves buffers with the online data to disk. Those can be reused for the next round of offline or online training. Following the RECAP paper, we retrain every time from the base model and just include the additional data to avoid drift.

### Inference

> **Highly recommended before first inference: set `action_clamp_limits`.**
> Teleop the arm through its safe range, record min/max per joint, then set the limits in degrees:
>
> ```yaml
> policy:
>   action_clamp_limits:
>     - [-150, 150] # joint 1
>     - [-150, 0] # joint 2
>     - ... # one [min, max] per joint
> ```
>
> Anything outside is clamped before reaching the servos.

Once your config has a trained model, camera indices, and follower/leader ports:

```bash
uv run python -m lerobot.rl.inference_async --config_path path/to/config.yaml
```

Initial model loading takes 1 to 2 minutes.

## Documentation

Everything beyond this page lives under [`docs/`](docs/README.md):

- [`docs/guide/`](docs/README.md#guide--how-to-use-the-pipeline): how to use the pipeline. Start with [usage](docs/guide/usage.md), then [RECAP](docs/guide/recap.md), [AWR](docs/guide/awr.md) and [probes](docs/guide/probes.md).
- [`docs/design/`](docs/README.md#design--how-it-is-built): how it is built, one page per subsystem.
- [`docs/status.md`](docs/status.md): what is on and off in the current run.
- [`docs/notes/`](docs/README.md#notes--the-ideas-lab): dated design notes, rubrics and investigations, each with a status line.
- [`CHEAT_SHEET.md`](CHEAT_SHEET.md): copy-paste commands for teleop, recording, dataset prep, training, probes and hardware checks.

Credits: the original $\pi_{0.5}$ port with subtasks and FAST tokens that this pipeline grew out of is by [@jadechoghari](https://github.com/jadechoghari).

# LeRobot for Research

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


This repo is a research-oriented version of [LeRobot](https://github.com/huggingface/lerobot). The focus is on having SOTA algorithms (e.g., RECAP) and SOTA models (e.g., MolmoAct2) and tooling for examining the model's internals such as attention maps, clustering of internal representations. 

This is an active project and we expect to continually add more features and capabilities. Happy to take requests.

Results obtained using MolmoAct2 on anchor actions.

<video src="https://github.com/user-attachments/assets/fc5c1815-93b9-41ec-90cb-ec8ba84c40fd" controls width="100%"></video>


## Key features

- Full support for [MolmoAct2](https://allenai.org/blog/molmoact2) model. This includes support for finetuning, inference, changing action space, attention maps, among others.
- End-to-end implementation of [RECAP](https://arxiv.org/pdf/2511.14759)-like algorithm for offline and online training
- Asynchronous inference with RTC that runs up to 30Hz with leader-guided human intervention.
- A suite of metrics and validation tools to examine the model's internals like attention maps, clusters of representation, among others.
- The complete version of $\pi_{0.5}$ with subtasks and FAST tokens. All credits to [@jadechoghari](https://github.com/jadechoghari).



## Roadmap


MolmoAct2 is fully integrated. We have been doing preliminary tests and runs. It looks very promising, but there is a lot more to test. The next thing is a full training run with RECAP.



## How to install

Clone this repo

```bash
git clone https://github.com/cijerezg/lerobot.git
cd lerobot
```

and to set up the environment run:

```bash
uv sync --extra molmoact2 --extra async --extra training
```



## Quick start

### Things to know
- This repo supports MolmoAct2 and $\pi_{0.5}$. In our experience MolmoAct2 has been the better model, so the rest of this guide focuses on it.
- For human intervention, only the follower-leader setup is supported.
- This repo is geared toward real-robots, so there is no simulation support.


### Prerequisites

#### Dataset

The quality of the dataset is extremely important. We suggest at least 50 episodes with a consistent strategy for task execution, e.g., try to grasp object in a consistent way as much as possible. MolmoAct2 has shown generalization capabilities, so we suggest to try a diverse dataset where objects are moved throughout the scene. 

> [!IMPORTANT]
> This pipeline assumes a **LeRobot v3.0 dataset format** (introduced in `lerobot 5.0.0`). If your dataset is v2.1, see [Using a v2.1 Dataset](docs/recap/advanced_usage_molmoact2.md#using-a-v21-dataset) for the one-command migration.

**Pre-decode the dataset to a cache.** Images are stored at their original resolution, so the offline buffer would otherwise hold every decoded frame in RAM at startup — a 50k-transition dataset with two 480×640 cameras runs ~90 GB. With the cache, the dataset lives on disk and only the data sampled during a training step is loaded into memory; subsequent runs load instantly.

Generate the cache once per dataset:

```bash
python -m lerobot.scripts.lerobot_memmap_buffer_cache \
    --root /path/to/local/dataset \
    --cache-dir outputs/buffer_cache \
    --image-storage-dtype uint8 \
    --image-storage-size 480 640
```

You can also pass `repo-id` instead of `root` if the dataset is on HF hub.

Then point the YAML at it with `buffer_cache_dir: outputs/buffer_cache`. Benefits:

- Training startup is fast on subsequent runs (no re-decode).
- RAM use stays bounded regardless of dataset size — you can train on datasets much larger than your physical memory.
- The `--image-storage-size` and `--image-storage-dtype` flags are model-agnostic, so the same pipeline works for any policy that uses a different image resolution (378x378 for MolmoAct2, 224×224 for $\pi_{0.5}, etc.).

More on edge cases in [advanced usage](docs/recap/advanced_usage_molmoact2.md#buffer-caching).


#### Config file

This is the file used for all the scripts in this repo. An example of the config file can be found in the [`rl/config_rl.yaml`](src/lerobot/rl/config_rl.yaml).

To get started with training, the key fields to change are:
- `root`: this is the path to your dataset.
- `task`: this is the task prompt.
- `base_path`: this should point to a local copy of the HF MolmoAct2 model. It can be downloaded locally using: `hf download allenai/molmoact2-so100_101`.
- `pretrained_path`: path to your finetuned model, or null if you don't have one. `base_path` must still be set either way — it's the upstream base model that supplies the architecture and norm stats.

> [!NOTE]
> Naming heads-up: this fork keeps the upstream LeRobot field name `pretrained_path`, but `base_path` is the actual pretrained foundation model. Read `pretrained_path` as "finetune to load on top of base."

We suggest you take a look at the entire config file before launching a training run. 


#### Action encoding

Before launching any training, decide how actions are represented. The same encoding has to be used end-to-end — switching it later means recomputing statistics and retraining from base. Three options:

- `absolute` — raw joint positions. Simplest, but does not generalize across starting configurations.
- `anchor` (recommended) — offsets from the chunk's initial state. Translation-invariant, generalizes well.
- `delta` — first-order differences between consecutive actions. Compact, but errors can accumulate.

Set the choice via `policy.action_encoding` in the config. `anchor` and `delta` also require precomputed normalization statistics — see the [advanced usage guide](docs/recap/advanced_usage_molmoact2.md#action-encodings) for details and the script that generates them.


### Training

The same scripts work for MolmoAct2 and $\pi_{0.5}$ — the policy is selected by `policy.type` in your config.

We suggest to start with a round of offline training so that the policy has a better starting point. To run it use:

```bash
uv run python -m lerobot.scripts.rl_offline --config path/to/config.yaml
```

> [!NOTE]
> Offline training runs validation probes (attention maps, action drift, etc.) that render MP4 artifacts. On Linux these can fail with `[Errno 12] Cannot allocate memory` — a virtual-memory overcommit accounting quirk with large PyTorch processes, not actual OOM. Persistent fix:
> ```bash
> echo 'vm.overcommit_memory = 1' | sudo tee /etc/sysctl.d/99-overcommit.conf
> sudo sysctl --system
> ```
> Background and tradeoffs in [docs/engineering_notes/runbooks/system_overcommit.md](docs/engineering_notes/runbooks/system_overcommit.md). If you'd rather not touch sysctl, set val_on_start to false and a large number to val_freq.

Once the offline training has run for a while, set `pretrained_path` in the config to the resulting checkpoint and proceed to online training.

First run the learner script:

```bash
uv run python -m lerobot.rl.rl_learner --config path/to/config.yaml
```

and then on another terminal run the actor script:

```bash
uv run python -m lerobot.rl.rl_actor_async --config path/to/config.yaml
```

The learner will automatically save buffers to disk with the online data. After processing, those can be reused for the next round of offline or online training.

Following the suggestions from the RECAP paper, we suggest to retrain every time from the base model, and just include the additional data to avoid drift. 


> [!IMPORTANT]
> While these instructions might get you started, we recommend reading the [advanced usage guide](docs/recap/advanced_usage_molmoact2.md) for better results. The $\pi_{0.5}$ variant lives at [advanced_usage_pi05.md](docs/recap/advanced_usage_pi05.md).




### Inference

> **Highly recommended before first inference: set `action_clamp_limits`.**
> Teleop the arm through its safe range, record min/max per joint, then set the limits in degrees:
> ```yaml
> policy:
>   action_clamp_limits:
>     - [-150, 150]  # joint 1
>     - [-150,   0]  # joint 2
>     - ...          # one [min, max] per joint
> ```
> Anything outside isclamped before reaching the servos.

Once your config has a trained model, camera indices, and follower/leader ports, run inference with:

```bash
uv run python -m lerobot.rl.inference_async --config path/to/config.yaml
```

> **Note:** Initial model loading takes 1 to 2 minutes.


## Beyond the basics

Many _important_ details were omitted in this introduction, and as we all know the devil is in the details, especially in a research-oriented repo. We strongly recommend reading the rest of the documentation, which is structured as:

- [Advanced usage (MolmoAct2)](docs/recap/advanced_usage_molmoact2.md): a deep dive on how to use all the features in the repo with MolmoAct2.
- [RECAP implementation](docs/recap/recap_implementation.md): an overview of RECAP and details about our implementation.
- [Validation metrics](docs/recap/metrics.md): explains the metrics that we chose and how they help uncover the inner workings of the model.
- [Advanced usage ($\pi_{0.5}$)](docs/recap/advanced_usage_pi05.md): same deep dive for the $\pi_{0.5}$ policy.



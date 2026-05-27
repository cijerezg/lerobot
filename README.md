# LeRobot for Research

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


This repo is a research-oriented version of [LeRobot](https://github.com/huggingface/lerobot). The focus is on having SOTA algorithms (e.g., RECAP) and SOTA models (MolmoAct2) and tooling for examining the model's internals such as attention maps, clustering of internal representations. 

This is an active project and we expect to continually add more features and capabilities. Happy to take requests.

<video src="https://github.com/user-attachments/assets/9c5330a9-27bf-4aab-98df-ea142f40093d" controls width="100%"></video>


## Key features

- Full support for [MolmoAct2](https://allenai.org/blog/molmoact2) model. This includes support for finetuning, inference, changing action space, attention maps, among others.
- The complete version of $\pi_{0.5}$ with subtasks and FAST tokens.All credits to [@jadechoghari](https://github.com/jadechoghari).
- End-to-end implementation of [RECAP](https://arxiv.org/pdf/2511.14759)-like algorithm for offline and online training
- Asynchronous inference with RTC that runs up to 30Hz with leader guided human invervention. In addition, a live inference script where a user can write subtasks for $\pi_{0.5}$ on the fly.
- A suite of metrics and validation tools to examine the model's internals like attention maps, clusters of representation, among others.




## Roadmap


MolmoAct2 is fully integrated. I have been doing preliminary tests and runs. It looks very promising, but there is a lot more to test. The very next thing is a full run with RECAP.



## How to install

Clone this repo

```bash
git clone https://github.com/cijerezg/lerobot.git
cd lerobot
```

and then up the environment run:

```bash
uv sync --extra molmoact2 --extra async --extra training
```



## Quick start

### Things to know
- This repo supports MolmoAct2 and $\pi_{0.5}$. Based on our experience, MolmoAct2 is a better model, and for that reason we will focus on Molmo, but $\pi_{0.5}$-specific setup is documented in [docs/pi05.md](docs/pi05.md).
- For human intervention, only the follower-leader setup is supported.
- This repo is geared toward real-robots, so there is no simulation support.


### Prerequisites

#### Dataset

The quality of the dataset is extremely important. We suggest at least 50 episodes with a consistent strategy for task execution, e.g., try to grasp object in a consistent way as much as possible. Molmo has shown generalization capabilities, so we suggest to try a diverse dataset where objects are moved throughout the scene. 


#### Config file

This is the file used for all the scripts in this repo. An example of the config file can be found in the [`rl/config-hiserl.yaml`](src/lerobot/rl/config-rl.yaml).

To get started with training, the key fields to change are:
- `root`: this is the path to your dataset.
- `task`: this is is the task prompt
- `base_path`: This should to a local copy of the HF MolmoAct2 model. It can be downloaded locally using: `hf download allenai/molmoact2-so100_101`.
- `pretrained_path`: path to your finetuned model, or null if you don't have one. `base_path` must still be set either way — it's the upstream base model that supplies the architecture and norm stats. (Heads-up: this field keeps the upstream LeRobot name pretrained_path, but in this fork base_path is the actual pretrained foundation model; read `pretrained_path` as "finetune to load.").

We re

#### Action encoding

Before launching any training, decide how actions are represented. The same encoding has to be used end-to-end — switching it later means recomputing statistics and retraining from base. Three options:

- `absolute` — raw joint positions. Simplest, but does not generalize across starting configurations.
- `anchor` (recommended) — offsets from the chunk's initial state. Translation-invariant, generalizes well.
- `delta` — first-order differences between consecutive actions. Compact, but errors can accumulate.

Set the choice via `policy.action_encoding` in the config. `anchor` and `delta` also require precomputed normalization statistics — see the [advanced usage guide](docs/recap/advanced_usage.md#action-encodings) for details and the script that generates them.


### Training

The same scripts work for MolmoAct2 and $\pi_{0.5}$ — the policy is selected by `policy.type` in your config.

We suggest to start with a round of offline training so that the policy has a decent starting point. To run it use:

```bash
uv run python -m lerobot.scripts.rl_offline --config path/to/config.yaml
```

> [!NOTE]
> Offline training runs validation probes (attention maps, action drift, etc.) that render MP4 artifacts. On Linux these can fail with `[Errno 12] Cannot allocate memory` — a virtual-memory overcommit accounting quirk with large PyTorch processes, not actual OOM. Quick fix:
> ```bash
> sudo sysctl -w vm.overcommit_memory=1
> ```
> Details and the persistent fix are in [docs/engineering_notes/runbooks/system_overcommit.md](docs/engineering_notes/runbooks/system_overcommit.md). If you'd rather not touch sysctl, set the `probe_parameters.enable_*` flags to `false` in the config to skip the probes entirely.

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
> While these instructions might get you started, we recommend reading the [advanced usage guide](docs/recap/advanced_usage.md) for better results.




### Inference

> **Highly recommended before first inference: set `action_clamp_limits`.**
> Without joint clamping, a misbehaving policy can send extreme position targets and damage the hardware. Before running any policy on a physical robot, teleop the arm through its full safe range of motion and record the min/max joint positions observed. Then set `policy.action_clamp_limits` in your config as a list of `[min, max]` pairs in degrees, one per joint:
> ```yaml
> policy:
>   action_clamp_limits:
>     - [-150, 150]  # joint 1
>     - [-150,   0]  # joint 2
>     - [   0, 150]  # joint 3
>     - [-150, 150]  # joint 4
>     - [-150, 150]  # joint 5
>     - [   0,  60]  # gripper
> ```
> Replace the example values with the limits you measured for your robot. Any action outside these bounds will be silently clamped before it reaches the servos.

Once you have a trained model, your camera indices and follower and leader ports in the config file, and then you can run inference using:

```bash
uv run python -m lerobot.rl.inference_async --config path/to/config.yaml
```


## Beyond the basics

Many _important_ details were ommitted in this introduction, and as we all know the devil is in the details, especially in a research-oriented repo. For that reason, we really recommend reading the rest of the documentation, which is structure as:

- [Advanced usage](docs/recap/advanced_usage.md): a deep dive on how to use all the features in the repo.
- [RECAP implementation](docs/recap/recap_implementation.md): an overview of RECAP and details about our implementation.
- [Validation metrics](docs/recap/metrics.md): explains the metrics that we chose and how they help uncover the inner workings of the model.



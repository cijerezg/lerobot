# LeRobot for Research

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


This repo is a research-oriented version of [LeRobot](https://github.com/huggingface/lerobot). The focus is on having SOTA algorithms (e.g., RECAP) and tooling for examining the model's internals such as attention maps, clustering of internal representations. 

This is an active project and we expect to continually add more features and capabilities. Happy to take requests.

<video src="https://github.com/user-attachments/assets/9c5330a9-27bf-4aab-98df-ea142f40093d" controls width="100%"></video>


## Key features

- The complete version of $\pi_{0.5}$ with subtasks and FAST tokens.All credits to [@jadechoghari](https://github.com/jadechoghari).
- End-to-end implementation of [RECAP](https://arxiv.org/pdf/2511.14759)-like algorithm for offline and online training
- Asynchronous inference with RTC that runs up to 30Hz with leader guided human invervention. In addition, a live inference script where a user can write subtasks for $\pi_{0.5}$ on the fly.
- A suite of metrics and validation tools to examine the model's internals like attention maps, clusters of representation, among others.


## Roadmap-ish

**This section is meant to describe the work currently being done with its context**

Using this repo, I trained the policy shown in the video above to complete the task under several arrangements. However, there are still failure modes, all of which are related to grasping as it is the hardest part of the task. The goal now is to automatically identify these difficult moments and have the policy focus on them. The current idea is to use the critic. The hypothesis is that the critic variance and its gradient will be highest during the hard parts of the task because those moments determine success or failure. Weighting the flow loss with the critic gradient should then force the policy to focus on those critical steps.



## How to install

Clone this repo

```bash
git clone https://github.com/cijerezg/lerobot.git
cd lerobot
```

and then follow the same instructions in the LeRobot [$\pi_{0.5}$ page](https://huggingface.co/docs/lerobot/pi05)



## Quick start

### Things to know
- This repo only supports $\pi_{0.5}$, adding other models is non-trivial.
- For human intervention, only the follower-leader setup is supported.
- This repo is geared toward real-robots, so there is no simulation support.


### Prerequisites

#### Dataset

At least 50 episodes, annotated with subtasks. The subtask annotation can be done manually using

```bash
python -m lerobot.policies.pi05_full.annotate.manual_subtask_annotate
```

which will launch a web interface to annotate the subtasks. Another tool for manual subtask and general dataset editing is available at [lerobot-data-studio](https://github.com/jackvial/lerobot-data-studio). 


The other option for subtask annotation is to use LLM annotations. We recommend using Gemma 4, and that can be done via this command:


```bash
python -m lerobot.policies.pi05_full.annotate.subtask_annotate_gemma_4 \
    --data-dir /path/to/your/dataset \
    --video-key observation.images.wrist \
    --batch-size 5 \
    --output-dir /output/path
```

#### Config file

This is the file used for all the scripts in this repo. An example of the config file can be found in the [`rl/config-hiserl.yaml`](src/lerobot/rl/config-hiserl.yaml)

To get started with training, the key fields to change are:
- `root`: this is the path to your dataset.
- `task`: this is is the task prompt
- `pi05_checkpoint`: this is the path to your checkpoint. If starting from scratch, use `lerobot/pi05_base` which loads the weight for base $\pi_{0.5}$.


#### Action encoding

Before launching any training, decide how actions are represented. The same encoding has to be used end-to-end — switching it later means recomputing statistics and retraining from base. Three options:

- `absolute` — raw joint positions. Simplest, but does not generalize across starting configurations.
- `anchor` (recommended) — offsets from the chunk's initial state. Translation-invariant, generalizes well.
- `delta` — first-order differences between consecutive actions. Compact, but errors can accumulate.

Set the choice via `policy.action_encoding` in the config. `anchor` and `delta` also require precomputed normalization statistics — see the [advanced usage guide](docs/pi05_docs/advanced_usage.md#action-encodings) for details and the script that generates them.

### Training

We suggest to start with a round of offline training so that the policy has a decent starting point. To run it use:

```bash
python -m lerobot.scripts.offline_learner_pi05 --config path/to/config.yaml
```

Once the offline training has run for a while, update the checkpoint in the config and proceed to online training.

First run the learner script:

```bash
python -m lerobot.rl.learner_pi05 --config path/to/config.yaml
```

and then on another terminal run the actor script:

```bash
python -m lerobot.rl.actor_pi05_async --config path/to/config.yaml
```

The learner will automatically save buffers to disk with the online data. After processing, those can be reused for the next round of offline or online training.

Following the suggestions from the RECAP paper, we suggest to retrain every time from the base model, and just include the additional data to avoid drift. 


> [!IMPORTANT]
> While these instructions might get you started, we recommend reading the [advanced usage guide](docs/pi05_docs/advanced_usage.md) for better results.




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
python -m lerobot.rl.inference_pi05_async --config path/to/config.yaml
```


## Beyond the basics

Many _important_ details were ommitted in this introduction, and as we all know the devil is in the details, especially in a research-oriented repo. For that reason, we really recommend reading the rest of the documentation, which is structure as:

- [Advanced usage](docs/pi05_docs/advanced_usage.md): a deep dive on how to use all the features in the repo.
- [RECAP implementation](docs/pi05_docs/recap_implementation.md): an overview of RECAP and details about our implementation.
- [Validation metrics](docs/pi05_docs/metrics.md): explains the metrics that we chose and how they help uncover the inner workings of the model.



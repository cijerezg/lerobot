# Implementation of RECAP-like RL algorithm


This work aims to replicate [RECAP](https://arxiv.org/pdf/2511.14759) from Physical Intelligence. RECAP is the algorithm used to train $\pi_{0.6}$. 

The key idea of RECAP is an advantage-conditioned VLA. Specifically, a critic is trained along the policy, and the advantage value from the critic is fed to the policy as an extended part of the prompt. As a result, the policy learns to distinguish between high-advantage (good) and low-advantage (bad) actions.

This is very useful for two reasons:
1. The policy can now learn from its own experience because there is a grounding signal about the "goodness" of actions (i.e., the advantage value) that is completely driven by reward via the critic. Without this, the policy would make a mistake and just learn to imitate it.
2. Even demonstrations might have suboptimal trajectories, but with a carefully designed reward function, the critic can learn to distinguish them, which in turn informs the policy via the advantage.


## How does this implementation compare to RECAP


Let's first do a small recap (no pun intended).

A typical VLA is modeled as $\pi(a|o, \ell)$, where $o$ is the observation (proprioceptive + vision) and $\ell$ is a language instruction. $\pi_{0.6}$ is essentially the same and also conditioned on the advantage, namely $\pi(a|o, \ell, A)$ -- technically, they use an advantage indicator $I$ that indicates whether the advantage is above or below a threshold. In practice, the indicator is apended to the prompt containing the language instruction and the proprioceptive observations.


Now we can dive into the differences
- RECAP uses an on-policy estimator for the critic, where they compute the return for a trajectory and that is the target for the critic. In contrast, we use an off-policy estimator that uses the Bellman equation to compute the target for the critic.
- RECAP discretizes the return into bins and uses cross entropy loss. In contrast, we do not discretize the reward and use MSE loss.
- During training, RECAP drops the advantage labels randomly as proposed by classifier-free guidance. In contrast, we always provide the advantage labels to the policy.
- RECAP uses Gemma 3 4B as backbone and the action expert is enlarged to 868M parameters. In contrast, we use PaliGemma 3B as backbone and our action expert has 300M parameters, which is the $\pi_{0.5}$ model. An implicit different is that Gemma 3 processes 896x896, whereas PaliGemma 224x224.




## Usage

This guide explains how to use the Reinforcement Learning (RL) module for Pi05, focusing on the offline-to-online training workflow.

## Prerequisites

Before starting, ensure you have the following:

1.  **Dataset**: You need a dataset with at least **50 episodes** of the task.
2.  **Annotation**: The dataset must be annotated with subtasks/skills.
    *   Use the annotation tool: `lerobot/src/lerobot/policies/pi05_full/annotate/subtask_annotate.py`.
3.  **Pretrained Weights**: You must have the **Pi05 base weights** downloaded locally.
    *   These can be downloaded from Hugging Face.
    *   Set the path in your config (see below).

## Configuration

Edit `config-hiserl.json` to configure your training setup.

### Key Configuration Fields

*   **`dataset`**:
    *   `dataset.root`: Path to your primary **annotated** dataset.
    *   `dataset.additional_offline_dataset_paths`: List of paths to *other* annotated datasets (e.g., from previous online sessions).
*   **`policy`**:
    *   `policy.pi05_checkpoint`: Path to your local Pi05 base weights.
    *   `policy.task`: Natural language description of the task.
*   **`output_dir`**:
    *   Defines the root directory for all training outputs.
    *   Offline training can optionally use `offline_output_dir` to override this.

### Output Directory Structure

The training scripts will generate the following structure under your configured `output_dir`:

```
output_dir/
├── checkpoints/        # Saved model weights (Policy + Optimizer state)
│   ├── 001000/
│   ├── 002000/
│   └── last/           # Symlink to the latest checkpoint
├── logs/               # Text logs (learner_*.log, actor_*.log)
├── online_buffer/      # Raw data collected during online training
└── training_state/     # Resume state (step counts, random seeds)
```

The online_buffer is only created when running the learner_pi05.py script.

## Phase 1: Offline Training

We strongly advise starting with offline training to initialize the policy and critic. Otherwise the policy might behave erratically. 

**Command:**
```bash
python -m lerobot.scripts.offline_learner_pi05 --config_path config-hiserl.json
```

*   **Duration**: Train for **8,000 - 10,000 steps** (or more depending on dataset size).
*   **Output**: Checkpoints are saved to `output_dir/checkpoints`.


## Phase 2: Online Training

Once you have a decent offline model, you can switch to online fine-tuning with a human-in-the-loop setup.

### 1. Start the Learner (Server)

The learner updates the policy using data from the replay buffer.

**Command:**
```bash
python -m lerobot.rl.learner_pi05 --config_path config-hiserl.json
```
*   Ensure `policy.pi05_checkpoint` points to your **offline trained checkpoint** (or use the `resume` flag if continuing).


### 2. Start the Actor (Client)

The actor runs the policy on the robot asynchronously and sends the data to the learner. By default it runs RTC at 30Hz. Make sure `rtc_config` is enabled in your config.

**Command:**
```bash
python -m lerobot.rl.actor_pi05_async --config_path config-hiserl.json
```

### 3. Intervention & Leader Arm Control

During online training, you use the Leader Arm to correct the robot when it deviates from the desired trajectory.

**Leader Arm Keys:**
*   `5`: **Toggle Intervention**.
    *   **Press to Intervene**: Take control when the robot drifts or makes a mistake. Correct the trajectory manually.
    *   **Press to Release**: Return control to the policy once it is back on a safe/correct path.
*   `2`: **Start Episode**. Once you mark a reward as success or failure, press this key to start the new episode.
*   `1`: **Success**. Mark the current episode as successful and terminate.
*   `0`: **Failure**. Mark the current episode as failed and terminate.

**Strategy**: We recommend to intervene only when the policy is drifting signficantly from the desired trajectory. While in theory the advantage-conditioned policy should be able to discern good actions from actions, in practice it is still helpful to have mostly good actions.


## Iterative Workflow

RL training is an iterative process. You should alternate between online data collection and offline training.

1.  **Offline Train**: Initial boost from static dataset.
2.  **Online Train**: Collect new data (interventions/successes) with the robot.
3.  **Process Data**: Convert and annotate the online data (see below).
4.  **Merge & Loop**: Add the new data to your offline training set and repeat.






## Frozen Parameters

Due to VRAM constraints, we freeze certain parts of the parameters in the offline phase as well as online phase. This is manually done and can be modified as needed. The code is `offline_learner_pi05.py` lines 264-277, and `learner_pi05.py` lines 341-351.



## Data Handling

Online data is saved as raw states/images in `output_dir/online_buffer`. To reuse this data for offline training, you must process it.

### 1. Convert to Video
Convert the raw buffer to a standard LeRobot video dataset format.

**Script:**
`lerobot/src/lerobot/policies/pi05_full/annotate/online_buffer_to_video.py`

**Usage:**
Edit the script to point `DATA_DIR` to your `online_buffer` path and `OUTPUT_DIR` to your desired destination.
```bash
python lerobot/src/lerobot/policies/pi05_full/annotate/online_buffer_to_video.py
```

### 2. Annotate Subtasks
The new dataset needs subtask annotations just like your initial dataset.

**Script:**
`lerobot/src/lerobot/policies/pi05_full/annotate/subtask_annotate.py`

**Usage:**
```bash
python lerobot/src/lerobot/policies/pi05_full/annotate/subtask_annotate.py \
    --data-dir /path/to/online_buffer_video \
    --video-key observation.images.wrist  # or relevant camera
    --output-dir /path/to/online_buffer_annotated
```

### 3. Merge Datasets
Add the path of the *newly annotated* dataset to your `config-hiserl.json`:

```json
"dataset": {
    "root": "path/to/original_dataset",
    "additional_offline_dataset_paths": [
        "path/to/online_buffer_annotated_1",
        "path/to/online_buffer_annotated_2"
    ]
}
```

Now, when you run **Offline Training** again, it will learn from both the original and new experiences!

## Further Reading

For a detailed deep-dive into the model architecture, loss functions, and code map, please refer to the [Technical README](./TECHNICAL_README.md).

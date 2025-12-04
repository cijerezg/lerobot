# Offline Pretraining for Pi05 RL

This guide explains how to use the offline pretraining script to initialize your policy and critic before online RL training.

## Why Offline Pretraining?

Before starting online RL with the actor collecting data, it's crucial to have a well-initialized policy:
- **Better initial behavior**: Actor will collect higher-quality data from the start
- **Faster convergence**: Policy starts from a reasonable baseline
- **Safety**: Avoids random actions that could damage the robot

## Running Offline Pretraining

### Single GPU

```bash
cd /home/user/Documents/Research/RL/LeRobot
.venv/bin/python lerobot/src/lerobot/rl/offline_learner_pi05.py config-hiserl.json
```

### Multi-GPU Training (Recommended)

Use `accelerate` to leverage multiple GPUs for larger batch sizes:

```bash
cd /home/user/Documents/Research/RL/LeRobot
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  lerobot/src/lerobot/rl/offline_learner_pi05.py \
  config-hiserl.json
```

This automatically:
- Distributes training across all specified GPUs
- Increases effective batch size = `batch_size × num_processes`
- Synchronizes gradients and checkpoints across processes

### Configuration

The offline learner uses the same config as the online learner (`config-hiserl.json`). Key parameters:

- **`output_dir_offline`**: Directory for offline checkpoints (recommended to keep separate from online)
- **`dataset.repo_id`**: Offline dataset to train on (required)
- **`policy.pi05_checkpoint`**: Pretrained Pi05 checkpoint to initialize from
- **`policy.offline_steps`**: Number of optimization steps (default: 10000)
- **`save_freq`**: Checkpoint save frequency
- **`batch_size`**: Batch size for training
- **`resume`**: Set to `true` to continue from previous offline training checkpoint

### Example Configuration

```json
{
  "output_dir_offline": "outputs/offline_pretrain",
  "resume": false,
  "policy": {
    "offline_steps": 5000,
    "pi05_checkpoint": "outputs/pi05_multi-task-toys-v1.01/checkpoints/016000/pretrained_model"
  },
  "dataset": {
    "repo_id": "cijerezg/multi-task-only-ep0"
  }
}
```

## What Gets Saved

The offline learner saves checkpoints in the same format as `lerobot_train.py`:

```
outputs/<job_name>/checkpoints/005000/
├── pretrained_model/
│   ├── config.json              # Policy config
│   ├── model.safetensors        # Policy weights (including critic)
│   ├── train_config.json        # Training config
│   ├── policy_preprocessor.json # Preprocessor config
│   ├── policy_preprocessor_step_*.safetensors  # Normalization stats
│   ├── policy_postprocessor.json
│   └── policy_postprocessor_step_*.safetensors
└── training_state/
    ├── optimizer_param_groups.json
    ├── optimizer_state.safetensors
    ├── rng_state.safetensors
    └── training_step.json
```

## Loading Pretrained Checkpoint in Online Learner

The offline and online learners use **separate directories** by design:
- Offline: `output_dir_offline` 
- Online: `output_dir`

To use the pretrained checkpoint with the online learner, point to the offline checkpoint:

```json
{
  "output_dir": "outputs/online_training",
  "policy": {
    "pi05_checkpoint": "outputs/offline_pretrain/checkpoints/last/pretrained_model"
  }
}
```

Or if continuing offline training, use `resume=true`:

```json
{
  "output_dir_offline": "outputs/offline_pretrain",
  "resume": true
}
```

The learner will load:
- Policy weights (actor + critic)
- Optimizer states
- Preprocessor/postprocessor with normalization stats
- Training step count

## Monitoring Training

### WandB Logging

If enabled, you'll see:
- `loss_critic`: Critic loss
- `loss_actor`: Actor loss (every `policy_update_freq` steps)
- `advantage_mean/std`: Advantage distribution
- `critic_value_mean/std`: Critic value distribution
- `td_error_mean/std`: TD error statistics
- Histograms for advantages and critic values

### Console Output

```
[OFFLINE LEARNER] Optimization step: 100/5000
[OFFLINE LEARNER] Optimization step: 200/5000
...
[OFFLINE LEARNER] Saving checkpoint at step 1000
```

## Differences from Online Learner

The offline learner is simplified:
- ❌ No actor spawning or gRPC server
- ❌ No transition queue processing
- ❌ No online replay buffer
- ✅ Only samples from offline dataset
- ✅ Trains both policy and critic
- ✅ Saves checkpoints in compatible format

## Troubleshooting

### Out of Memory

Reduce `batch_size` or `offline_buffer_capacity`:
```json
{
  "batch_size": 32,
  "policy": {
    "offline_buffer_capacity": 10000
  }
}
```

### Dataset Not Found

Ensure `dataset.repo_id` points to a valid dataset:
```bash
# Check if dataset exists locally
ls -la /home/user/.cache/huggingface/lerobot/<repo_id>
```

### Checkpoint Loading Fails

Verify the checkpoint path exists:
```bash
ls -la outputs/<job_name>/checkpoints/last/pretrained_model/
```

# Offline π0.5 RECAP And RLT Training Checklist

Use this checklist to train `src/lerobot/scripts/offline_learner_pi05.py` on the annotated dataset `jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed`.

There are two different offline paths in this repo:

- Full RECAP / `pi05_rl` offline training uses a LeRobot dataset with images, state, actions, rewards, language/subtask metadata, and the normal `lerobot.rl.buffer.ReplayBuffer`.
- RLT-head offline training uses the compact persisted RLT replay from DRTC rollouts. It stores precomputed `rl_token`, proprio, VLA reference chunks, executed chunks, next context, reward, done, and intervention flags. It is for training only the lightweight RLT actor/critic heads on top of a frozen VLA checkpoint.

The first checklist below is for full RECAP training. The final section covers collect-rollouts-then-train-RLT-head offline.

1. Pick or copy a config, for example `recap-config-hilserl.json`.

2. Set the dataset fields:

   ```json
   "dataset": {
     "root": "outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed",
     "repo_id": "jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed",
     "use_imagenet_stats": false,
     "max_episodes": null,
     "additional_offline_dataset_paths": []
   }
   ```

3. Download or cache the dataset locally:

   ```bash
   uv run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset(repo_id='jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed', root='outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed')"
   ```

4. Precompute the memmap replay buffer cache:

   ```bash
   uv run python src/lerobot/scripts/lerobot_memmap_buffer_cache.py \
     --repo-id jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed \
     --data-dir outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed \
     --cache-dir outputs/buffer_cache \
     --video-backend pyav
   ```

   This dataset should report `7514` frames and write cache fingerprint `561a81c774cc0723`.

5. Confirm camera keys match the dataset. This dataset uses `observation.images.top` and `observation.images.side`, so the config must not reference `observation.images.wrist`.

   ```json
   "policy": {
     "input_features": {
       "observation.images.side": {
         "type": "VISUAL",
         "shape": [3, 224, 224]
       },
       "observation.images.top": {
         "type": "VISUAL",
         "shape": [3, 224, 224]
       },
       "observation.state": {
         "type": "STATE",
         "shape": [6]
       }
     }
   }
   ```

   For online rollout or inference with the same config, keep the robot camera names aligned with the dataset keys and physical viewpoints:

   ```json
   "cameras": {
     "top": {
       "type": "opencv",
       "index_or_path": "/dev/video0",
       "width": 800,
       "height": 600,
       "fps": 30,
       "fourcc": "MJPG"
     },
     "side": {
       "type": "opencv",
       "index_or_path": "/dev/video4",
       "width": 800,
       "height": 600,
       "fps": 30,
       "fourcc": "MJPG"
     }
   }
   ```

   A key mismatch such as `wrist` vs `side` can cause missing-key failures during offline replay buffer creation, or worse, swapped/missing camera inputs during online rollout.

6. Confirm the dataset has subtask annotations. The loader expects `meta/subtasks.parquet`, and training carries `subtask_index` through the replay buffer.

7. Set task text consistently in both places:

   ```json
   "policy": {
     "task": "your task description"
   },
   "env": {
     "task": "your task description"
   }
   ```

8. Set `policy.pi05_checkpoint` to the π0.5 base weights or a previous offline checkpoint:

   ```json
   "pi05_checkpoint": "models/pi05_base"
   ```

9. If using anchor actions, keep:

   ```json
   "action_encoding": "anchor"
   ```

   Then compute matching stats for this dataset:

   ```bash
   uv run python src/lerobot/scripts/compute_delta_stats.py \
     --root outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed \
     --encoding anchor \
     --chunk-size 50 \
     --output-dir outputs/stats
   ```

10. Point the config at the generated stats file:

   ```json
   "action_encoding_stats_path": "outputs/stats/action_stats_anchor_cube-subtasks-e30-base120trim-0-9-101-end-fixed.pt"
   ```

11. Choose output and logging settings:

   ```json
   "offline_output_dir": "outputs/pi05_cube_subtasks_offline",
   "offline_save_freq": 1000,
   "policy": {
     "offline_steps": 8000
   }
   ```

12. For 8 L40S GPUs, start with the same effective batch size you used on 1 GPU:

    ```json
    "batch_size": 16,
    "policy": {
      "gradient_accumulation_steps": 1,
      "actor_lr": 0.0001,
      "critic_lr": 0.0001
    }
    ```

    This gives an effective batch of `16 x 8 x 1 = 128`, close to the previous `14 x 1 x 9 = 126`.

13. Sync the prepared assets to S3 so the EC2 training machine can pull them down.

    From the local workstation after fixing the dataset, generating the memmap cache, and computing action stats:

    ```bash
    export RECAP_BUCKET=s3://lerobot-recap-024158824539

    aws s3 sync \
      outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed \
      ${RECAP_BUCKET}/assets/outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed

    aws s3 sync \
      outputs/buffer_cache/561a81c774cc0723 \
      ${RECAP_BUCKET}/assets/outputs/buffer_cache/561a81c774cc0723

    aws s3 cp \
      outputs/stats/action_stats_anchor_cube-subtasks-e30-base120trim-0-9-101-end-fixed.pt \
      ${RECAP_BUCKET}/assets/outputs/stats/action_stats_anchor_cube-subtasks-e30-base120trim-0-9-101-end-fixed.pt
    ```

    If using your `rgpu.sh` helper from `cloud-gpu-provisioning`, use it to run the same `aws s3 sync` / `aws s3 cp` commands on the appropriate source or destination host.

14. On the EC2 training machine, do the one-time auth and asset checks before launching training.

    Hugging Face auth is required because `google/paligemma-3b-pt-224` is gated. Log in with an account that has accepted access to that model:

    ```bash
    uv run huggingface-cli login
    uv run huggingface-cli whoami
    ```

    For non-interactive setup:

    ```bash
    export HF_TOKEN=hf_...
    uv run huggingface-cli login --token "$HF_TOKEN"
    ```

    Prewarm the gated model files before launching 8 ranks, so auth/download issues fail early:

    ```bash
    uv run python - <<'PY'
    from transformers import AutoConfig, AutoTokenizer

    repo = "google/paligemma-3b-pt-224"
    AutoConfig.from_pretrained(repo)
    AutoTokenizer.from_pretrained(repo)
    print("OK: loaded", repo)
    PY
    ```

    If W&B is enabled, log in or explicitly disable/offline it:

    ```bash
    uv run wandb login
    ```

    Or prefix the training command with one of:

    ```bash
    WANDB_MODE=disabled
    WANDB_MODE=offline
    ```

    Confirm the local π0.5 base checkpoint exists because the config points at `models/pi05_base`:

    ```bash
    ls models/pi05_base
    ```

15. On the EC2 training machine, from the `~/code/lerobot` repo root, restore the same paths expected by `recap-config-hilserl.json`:

    ```bash
    export RECAP_BUCKET=s3://lerobot-recap-024158824539

    mkdir -p outputs/buffer_cache outputs/stats

    aws s3 sync \
      ${RECAP_BUCKET}/assets/outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed \
      outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed

    aws s3 sync \
      ${RECAP_BUCKET}/assets/outputs/buffer_cache/561a81c774cc0723 \
      outputs/buffer_cache/561a81c774cc0723

    aws s3 cp \
      ${RECAP_BUCKET}/assets/outputs/stats/action_stats_anchor_cube-subtasks-e30-base120trim-0-9-101-end-fixed.pt \
      outputs/stats/action_stats_anchor_cube-subtasks-e30-base120trim-0-9-101-end-fixed.pt
    ```

    The restored paths should match:

    ```text
    outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed
    outputs/buffer_cache/561a81c774cc0723
    outputs/stats/action_stats_anchor_cube-subtasks-e30-base120trim-0-9-101-end-fixed.pt
    ```

16. Start a tmux session on the training machine before launching the long-running job:

    ```bash
    tmux new -s pi05-offline
    ```

    If you disconnect, reattach with:

    ```bash
    tmux attach -t pi05-offline
    ```

17. Run offline training on 8 GPUs with Accelerate:

    ```bash
    uv run accelerate launch \
      --num_processes 8 \
      --multi_gpu \
      src/lerobot/scripts/offline_learner_pi05.py \
      --config_path=recap-config-hilserl.json
    ```

    The startup logs should report `num_processes=8`, distinct `local_rank`/CUDA devices, `after_prepare ... is_ddp=True`, and a global effective batch of `128`.

18. Run offline training on a single GPU:

    ```bash
    uv run python -m lerobot.scripts.offline_learner_pi05 \
      --config_path=recap-config-hilserl.json
    ```

19. Monitor logs and checkpoints:

    ```text
    outputs/pi05_cube_subtasks_offline/logs/
    outputs/pi05_cube_subtasks_offline/checkpoints/
    ```

    Multi-GPU runs write per-rank logs like `offline_learner_<job>_rank0.log`; rank 0 includes a compact per-rank sanity table for the first few optimization steps.

20. For later online training or inference, point `policy.pi05_checkpoint` at the saved offline checkpoint, usually:

    ```text
    outputs/pi05_cube_subtasks_offline/checkpoints/<step>/pretrained_model
    ```

## Offline RLT Head Training From Collected Rollouts

Use this path when you want to:

- Run the frozen VLA on the robot.
- Collect compact RLT transitions from those rollouts.
- Train only the RLT actor/critic heads offline.
- Keep the frozen VLA backbone fixed, matching the RL Token paper's post-training setup.

This does not replace full RECAP offline training. It does not train the VLA backbone and it does not consume a normal LeRobot dataset.

1. Configure `examples/experiments/configs/baseline_pi05_rlt.yaml` for collect-only rollout recording:

   ```yaml
   policy_type: pi05_rlt
   pretrained_name_or_path: outputs/pi05_subtasks_good_dataset_4/checkpoints/last/pretrained_model
   rlt_enabled: true
   rlt_embedding_checkpoint: outputs/pi05_rlt_embedding_cube_subtasks_3/rlt_embedding_step_004200.pt
   rlt_head_checkpoint:

   rlt_online_collection_enabled: true
   rlt_online_training_enabled: false
   rlt_online_buffer_path: outputs/rlt_online/rlt_online_replay.pt
   rlt_online_buffer_save_freq_transitions: 100
   rlt_persist_buffer_on_shutdown: true
   ```

   With no `rlt_head_checkpoint` and online training disabled, the robot executes frozen-VLA pass-through actions while the client/server records compact RLT transitions.

2. Run fixed-task rollouts first.

   - [ ] Keep object poses and robot start pose low-variance.
   - [ ] Label each episode success or failure.
   - [ ] Use teleop intervention only when needed.
   - [ ] Confirm the replay file exists at `outputs/rlt_online/rlt_online_replay.pt`.
   - [ ] Keep note of the exact frozen VLA checkpoint and RLT embedding checkpoint used for collection.

3. Log in to W&B if training should upload metrics:

   ```bash
   uv run wandb login
   ```

   For offline W&B sync later:

   ```bash
   export WANDB_MODE=offline
   ```

4. Train the RLT head offline from the compact replay:

   ```bash
   uv run python -m lerobot.rl.train_pi05_rlt_head_offline \
     --policy_path=outputs/pi05_subtasks_good_dataset_4/checkpoints/last/pretrained_model \
     --replay_buffer_path=outputs/rlt_online/rlt_online_replay.pt \
     --output_dir=outputs/rlt_offline_head \
     --rlt_embedding_checkpoint=outputs/pi05_rlt_embedding_cube_subtasks_3/rlt_embedding_step_004200.pt \
     --steps=10000 \
     --batch_size=256 \
     --rlt_actor_hidden_dims='[512, 512, 512]' \
     --rlt_critic_hidden_dims='[512, 512, 512]' \
     --rlt_num_critics=4 \
     --rlt_bc_beta=0.1 \
     --rlt_jerk_beta=0.001 \
     --discount=0.985 \
     --grad_clip_norm=20.0 \
     --wandb_project=lerobot-rlt
   ```

   The script instantiates `PI05RLTPolicy` from the frozen VLA checkpoint/config for paper-aligned head construction, but the training loop updates only `rlt_actor` and `rlt_critic` from the persisted RLT replay.

5. Watch console and W&B metrics.

   Key metrics:

   - `train/actor_loss`
   - `train/critic_loss`
   - `train/kl_to_reference`
   - `train/actor_q_mean`
   - `train/pred_q_mean`
   - `train/target_q_mean`
   - `train/action_deviation_rms`
   - `train/action_deviation_abs_max`
   - `train/actor_grad_norm`
   - `train/critic_grad_norm`

   If `kl_to_reference`, action deviation, critic loss, or Q magnitudes spike, stop and retry with higher `rlt_bc_beta`, lower learning rates, or more conservative replay.

6. Checkpoints are written to:

   ```text
   outputs/rlt_offline_head/rlt_head_step_<step>.pt
   outputs/rlt_offline_head/rlt_head_latest.pt
   ```

7. Evaluate the trained RLT head by pointing the DRTC config at the checkpoint:

   ```yaml
   rlt_head_checkpoint: outputs/rlt_offline_head/rlt_head_latest.pt
   rlt_online_collection_enabled: false
   rlt_online_training_enabled: false
   ```

   Evaluate on the exact same fixed setup before adding variance.

8. Resume or extend the offline RLT run by reusing the same replay and setting:

   ```bash
   --rlt_head_checkpoint=outputs/rlt_offline_head/rlt_head_latest.pt
   ```

   Keep `rlt_chunk_size`, hidden dimensions, critic count, and token dimension compatible with the checkpoint.

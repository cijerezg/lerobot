# Offline π0.5 RECAP Training Checklist

Use this checklist to train `src/lerobot/scripts/offline_learner_pi05.py` on the annotated dataset `jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed`.

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

    On the EC2 training machine, from the `~/code/lerobot` repo root, restore the same paths expected by `recap-config-hilserl.json`:

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

14. Run offline training on 8 GPUs with Accelerate:

    ```bash
    uv run accelerate launch \
      --num_processes 8 \
      --multi_gpu \
      src/lerobot/scripts/offline_learner_pi05.py \
      --config_path=recap-config-hilserl.json
    ```

    The startup logs should report `num_processes=8`, distinct `local_rank`/CUDA devices, `after_prepare ... is_ddp=True`, and a global effective batch of `128`.

15. Run offline training on a single GPU:

    ```bash
    uv run python -m lerobot.scripts.offline_learner_pi05 \
      --config_path=recap-config-hilserl.json
    ```

16. Monitor logs and checkpoints:

    ```text
    outputs/pi05_cube_subtasks_offline/logs/
    outputs/pi05_cube_subtasks_offline/checkpoints/
    ```

    Multi-GPU runs write per-rank logs like `offline_learner_<job>_rank0.log`; rank 0 includes a compact per-rank sanity table for the first few optimization steps.

17. For later online training or inference, point `policy.pi05_checkpoint` at the saved offline checkpoint, usually:

    ```text
    outputs/pi05_cube_subtasks_offline/checkpoints/<step>/pretrained_model
    ```

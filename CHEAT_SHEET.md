# reBot Commands

Run everything from the workspace root with the root `.venv`. Never `uv run --project lerobot` (builds a separate env).

## Teleop

uv run --no-project --python .venv/bin/python lerobot/src/lerobot/scripts/lerobot_teleoperate.py \
    --robot.type=rebot_b601_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=rebot_follower_v1 \
    --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"427622270837\", width: 640, height: 480, fps: 30, use_depth: true, depth_filters: true, log_path: outputs/logs/realsense_wrist.log, log_severity: info}, top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_USB_2.0_webcam_SN0001-video-index0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=rebot_102_leader \
    --teleop.variant=102HD \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_leader_hd_v1 \
    --display_data=true

## Record

`depth_stride` must equal `policy.image_stride` in config_rl.yaml (which must divide `policy.chunk_size`).

uv run --no-project --python .venv/bin/python lerobot/src/lerobot/scripts/lerobot_record.py \
    --robot.type=rebot_b601_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=rebot_follower_v1 \
    --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"427622270837\", width: 640, height: 480, fps: 30, use_depth: true, depth_filters: true}, top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_USB_2.0_webcam_SN0001-video-index0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=rebot_102_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_leader_v1 \
    --dataset.repo_id=cijerezg/tidy-up-table-v1 \
    --dataset.single_task="Tidy up the table" \
    --dataset.fps=30 \
    --dataset.depth_stride=3 \
    --dataset.num_episodes=16 \
    --dataset.episode_time_s=600 \
    --dataset.reset_time_s=120 \
    --dataset.push_to_hub=false \
    --teleop.variant=102HD \
    --display_data=true

A crash or Ctrl-C mid-episode leaves `<root>/recovery/episode-XXXXXX.pkl` (state, action, task per frame) next to the
images. Rebuild those episodes into a new root (needs images on disk, so not with `--dataset.streaming_encoding=true`):

uv run python -m lerobot.scripts.recover_episodes --root ~/.cache/huggingface/lerobot/cijerezg/<run> --out outputs/<run>-recovered

## New dataset prep (once per dataset)

Anchor action stats (`--chunk-size` = `policy.chunk_size`):

uv run --no-project --python .venv/bin/python python -m lerobot.scripts.compute_delta_stats \
    --root outputs/rebot_dataset_dummy_v1 \
    --encoding anchor \
    --chunk-size 30

Memmap buffer cache (`--image-stride` = `policy.image_stride`, mismatch is a hard error):

uv run --no-project --python .venv/bin/python python -m lerobot.scripts.lerobot_memmap_buffer_cache \
    --repo-id cijerezg/rebot_dataset_dummy_v1 \
    --data-dir outputs/rebot_dataset_dummy_v1 \
    --cache-dir outputs/buffer_cache-rebot-dummy-v1 \
    --image-storage-dtype uint8 \
    --image-stride 3

## Offline training

uv run python -m lerobot.scripts.rl_offline --config_path=config_rl.yaml

Smoke test without Aim:

uv run python -m lerobot.scripts.rl_offline --config_path=config_rl.yaml --aim.enable=false

## Aim metrics UI

Env setup (once, or after dependency changes):

uv sync --project lerobot --extra training --extra molmoact2 --extra pi

Start the UI (http://127.0.0.1:43800), during or after training:

uv run aim up --repo ./aim

## Inference (standalone, RTC)

Loads `inference_checkpoint_path` from config_rl.yaml. `inference_send_actions_to_robot: false` = read-only preflight. Subtasks: `n` next, `b` back.

uv run python -m lerobot.rl.inference_async --config=config_rl.yaml

## Sim eval (MuJoCo nail clippers, no hardware)

`config_rl_sim.yaml` = config_rl.yaml with `env.robot: rebot_b601_sim` (layouts in `lerobot/src/lerobot/robots/rebot_b601_sim/scenes/`, README there), `sim_autostart` instead of the leader, task T085 and the auto subtask scorer. One layout, the checkpoint of the yaml; the run ends by itself after `env.robot.episodes` parks:

MUJOCO_GL=egl uv run python -m lerobot.rl.inference_async --config_path=config_rl_sim.yaml [--env.robot.scene=ep1]
uv run rerun "$(ls -dt outputs/train/$(date +%F)/*rebot-sim-nail* | head -1)/rerun.rrd"

Every layout on one checkpoint (what the chase runs on the DGX): `<run>/validation/step_<step>/sim_nail/<layout>/{rerun.rrd,subtask_events.json}` + `sim_nail/summary.md` (successes, steps reached per episode). `LAYOUTS="ep0 ep2"` and `SIM_EVAL_ARGS="--env.robot.episodes=1 --env.processor.reset.control_time_s=20"` for a short run:

lerobot/scripts/run_sim_eval.sh outputs/<run>/checkpoints/<step>/pretrained_model

Two checkpoints side by side in one rerun viewer (A top row, B bottom row, shared time axis: wrist, top, depth, joints, subtask events). Paths: checkpoint step dir, `validation/step_X`, `sim_nail`, layout dir or `.rrd`; `--layout ep2` picks the layout, `--save x.rrd` writes the merged file instead of opening the viewer:

uv run python -m lerobot.scripts.compare_sim outputs/<run_a>/checkpoints/<step> outputs/<run_b>/checkpoints/<step> [--layout ep0] [--label a --label b]

## Rollout review (cut recorded inference episodes into a training root)

One command: renders a review video per episode (missing ones only) into `migration/rollout_review_<date>/videos`, opens a browser page (http://127.0.0.1:7871). Pick an episode, play (space; 1x/2x/4x; arrows 1 s, shift 10 s), `[` at the start and `]` at the end of each stretch to remove (or type spans: `15-31, 36-43, 50-`), set the output root (default `outputs/rebot_rollouts_<date>-cut-v<N>`), Save = plan.json + build_root + verify_root + README. Unmarked episodes are kept whole; untick to leave one out. Several runs at once is fine.

uv run python -m lerobot.annotation.rebot.rollout_review outputs/train/<date>/<run> [<run> ...] [--out outputs/<root>]

## Merge annotated roots

Merges finished ReBot roots into one new root: videos and depth are hardlinked, tables re-indexed. Run verify_root after.

uv run python -m lerobot.annotation.rebot.merge_roots outputs/<root_a> outputs/<root_b> --out outputs/<merged>

## Data record

Episodes, frames and hours per root and per sampling group, from config_rl.yaml and the roots. Writes lerobot/dataset_info/DATA.md. Rerun after every data or config change; roots built but not yet in the config go in lerobot/dataset_info/pending.yaml.

uv run python lerobot/dataset_info/data_count.py

## Probe viewer

Browser UI over a run's probes (http://127.0.0.1:7870), rescans on refresh. Takes a run dir, `validation/`, or a `step_*` dir.

uv run python -m lerobot.scripts.view_probes outputs olmoact2_offline_rebot_all-v6

## Action trace in sim

The Action Inspector's flow samples played on the MuJoCo reBot, as a rerun log. In the suite it is its own probe, `enable_action_trace_sim` (needs `enable_action_trace`, which it runs after), flags from `probe_parameters.trace_sim_args` (`"--time-scale 2"` = half speed, `--execute` = physics); output `validation/step_*/action_trace_sim/action_trace_sim.rrd`. By hand, on a checkpoint and chosen anchors:

```bash
uv run python -m lerobot.probes.action_trace_probe --config config_rl_validate.yaml \
    --policy.pretrained_path outputs/<run>/checkpoints/<step>/pretrained_model \
    --dataset.root outputs/rebot_val_2026-10-14-v1 \
    --probe_parameters.output_dir outputs/probe_runs/<name> \
    --probe_parameters.trace_frames "$(cat migration/action_trace_sim_2026-10-15/grasp_anchors_6s.txt)"
# other flags: --execute, --demo-arm, --anchors, --episodes, --spawn
uv run python -m lerobot.probes.action_trace_sim --records outputs/probe_runs/<name>/action_trace/action_trace_records.npz --time-scale 2
.venv/bin/rerun outputs/probe_runs/<name>/action_trace/action_trace_sim.rrd
```

## Representation trajectories

Uses `policy.pretrained_path` in `config_rl.yaml`. Captures separate image/state/action groups, real and constant prompts, then writes `trajectories.html` plus per-panel `panels/*.js` under `outputs/probe/validation/step_<checkpoint>/representations/` (or the configured probe output directory). Keep that directory together; the page loads panels and camera previews from it.

```bash
.venv/bin/python -m lerobot.probes.representations --config config_rl.yaml
# Replot an existing cache without loading the model or datasets; window radius is in seconds:
.venv/bin/python -m lerobot.probes.representations --config config_rl.yaml \
    --probe_parameters.mode=plot --probe_parameters.repr_local_window_s=2.0
```

Pick a geometry (raw cosines are ~1 everywhere; robot- or episode-centered shows the structure) and a measure (cosine, L2 distance, norm ratio), an episode, then slide through frames: cameras, the within-episode matrix, the frame's profile against the background level, the ranked matches in other episodes (top 5 episodes per category) and the aligned cross-episode grid follow. The grid centers the selected frame and its match and compares the two episodes at offsets of ±Δ·span seconds (Δ 1–8 s, 11×11 or 15×15). Sampling is one frame every `repr_stride_s` (2 s); whole episodes are kept while they fit `repr_max_frames` (1600 → about 18 three-minute episodes); constant text doubles captures. `max_episodes` is per robot and counts training episodes; held-out (val) episodes go first under `repr_holdout_episodes` (default all). The report shows the stride and median gap. Changing the stride or frame budget needs recollection.

## Probe compare

Checkpoint compare puts one to four runs or checkpoints side by side at http://127.0.0.1:7871.
Pass run or validation folders (each opens at its last checkpoint) or explicit `step_*`
folders. Two checkpoints from the same run work as separate comparison slots:

```bash
uv run python -m lerobot.scripts.compare_probes \
    outputs/<run>/validation/step_00000400 \
    outputs/<run>/validation/step_00000800 \
    --label=step-400 --label=step-800
```

A probe's outputs are navigated by axis: one chip row per axis parsed from the filenames
(attention: Episode, Layer, Camera, View; sim: Scene, Sim episode, Camera; input swap
examples: Kind, Band). Clicking a value keeps the other axes; a dashed chip has no output at
the current combination and jumps to the nearest one. Flat probes show their figures as cards.
**Wall** fans the camera axis out per run (every camera of the selected episode and layer,
synced within a run). Each run card has its own checkpoint buttons and a through-training
sparkline; **All runs at** moves every run to a common step; **Shared outputs** filters to
panels present in every run. Missing outputs read "not recorded". Probes the suite no longer
reads (spatial memorization, Jacobians, temporal attention, critic value, future visual) are
hidden from the rail.

**Copy link** preserves probe, output, checkpoints, baseline, filter and wall mode.
Keyboard: `j`/`k` switch probes, `n`/`p` switch outputs, Space plays/pauses the stage.
Refresh reloads newly written probe output.

## Remote validation on the DGX

Run one checkpoint's probe suite on the DGX and pull results to `outputs/remote_val/<run>-<step>/`. Syncs code + config_rl.yaml; datasets/stats/cache must already be on the DGX.

lerobot/scripts/remote_validate.sh outputs/molmoact2_offline_rebot_v4/checkpoints/000400

Reattach to a detached run:

lerobot/scripts/remote_validate.sh --attach outputs/molmoact2_offline_rebot_v4/checkpoints/000400

Flags: `--dry-run`, `--keep-checkpoint`, `--force-delete`, `--out DIR`, `--config PATH`, `--host`, `--sim` (the MuJoCo nail-clipper eval, `run_sim_eval.sh` + `config_rl_sim.yaml`, instead of the probe suite; results in the same `validation/step_<step>/` as `sim_nail/`).

## Chase a run's checkpoints (overnight)

Probes the newest unprobed checkpoint on the DGX until training exits or STOP. State in `<run>/validation/.chase/`.

setsid nohup lerobot/scripts/chase_validate.sh outputs/<run> > /dev/null 2>&1 &

Sim eval instead of the probes (flags after `--` go to remote_validate.sh):

setsid nohup lerobot/scripts/chase_validate.sh outputs/<run> -- --sim > /dev/null 2>&1 &

Probes and then the sim eval in one pass per checkpoint (both land in validation/step_<step>/; the sim shows up as the "Sim Nail" probe in compare_probes):

setsid nohup lerobot/scripts/chase_validate.sh outputs/<run> -- --with-sim > /dev/null 2>&1 &

Stop after the pass in flight:

touch outputs/<run>/validation/.chase/STOP

Probe a checkpoint by hand while the chase is up:

flock outputs/remote_val/.dgx.lock lerobot/scripts/remote_validate.sh <ckpt>

## Hardware checks

Probe follower motors:

uv run python probe_rebot_motors.py

Leader USB check after any re-plug (must be 200/200; ch341 must be under Bus 003/005 in `lsusb -t`, Bus 001 drops bytes):

.venv/bin/python migration/leader_bus_capture.py burst --trials 200 --spacing 0.033 --window 0.025

Follower chain order, base first (dead tail = cable upstream of first dead motor; one dead motor = its own board):

    #  joint          send  recv
    1  shoulder_pan      1    17
    2  shoulder_lift     2    18
    3  elbow_flex        3    19
    4  wrist_flex        4    20
    5  wrist_yaw         5    21
    6  wrist_roll        6    22
    7  gripper           7    23

Leader (102HD, ttyUSB0): ids 0-6 same order, id 7 = button board.

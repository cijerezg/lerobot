# joint_onset_trace (2026-10-17)

Question: the robot never turned the wrist roll when grasping the large nail clipper. Did the model learn the
turn at all, and if so why not here? Method: action-trace probe (16 flow samples of the 1 s chunk) on training
frames where the operator starts the turn, on the robot run's reach, and across checkpoints; read out per joint
what the seeds do against what the operator did, with "hold still" as the baseline.

## Files

- `turn_onset_frames.py`: anchors. For each large-clipper grasp window, the first frame where the roll is more
  than 10 deg from its window-start value in the direction of the window's net change, plus 1 s before and after.
  Writes `turn_onset_<set>.txt` (the probe's `ep:frame` list) and `turn_onset_frames.csv` (operator direction and
  1 s change per anchor). Episode lists and roots are hard-coded at the top; edit them for another move.
- `reach_anchors.txt`: the robot run's first 8 s, one frame per second.
- `trace_clippers.py`: the probe with the deployed prompt metadata (precision 4, top pinch) and no gripper-event
  targets. The stock probe leaves those two clauses out of the prompt.
- `ckpt_sweep.sh`: runs the probe on every checkpoint of a run, four runs per checkpoint (one per data root).
  Skips outputs that exist. Checkpoint override is `--policy.pretrained_path`; the standalone probe ignores
  `inference_checkpoint_path`.
- `ckpt_sweep_readout.py`: per checkpoint, three tables: seeds with / flat / against the operator's roll
  direction and the model's turn as a fraction of the operator's; per-joint |model median - operator| at the
  chunk end next to the hold error; roll seeds and per-joint commanded change on the reach frames.
- `turn_onset_readout.py`: the per-frame version for one checkpoint.

## Run

    uv run python lerobot/adhocs/joint_onset_trace/turn_onset_frames.py
    lerobot/adhocs/joint_onset_trace/ckpt_sweep.sh                 # GPU; edit RUN/OUT/STEPS at the top
    uv run python lerobot/adhocs/joint_onset_trace/ckpt_sweep_readout.py

## Found (run rebot-stratified_sampling-20261016, robot run 2026-10-17 10-54-00)

- Training frames: the turn is learned by step 400 (seeds with the operator 57-77 %, against 2-3 %, turn rate
  0.6-1.0 of the operator's). Step 200 has not learned it.
- Robot reach frames: no checkpoint turns (flat 71-98 %); the arm reach itself is predicted at every step.
  The clipper lay about 90 deg from every training start, so this read as coverage, not skill.
- Checkpoint-to-checkpoint swings of 20 points on 24 frames x 16 seeds: too few frames to rank checkpoints.
- The gripper is the one joint worse than holding still on the onset frames from step 400 on; not followed up.

## Reuse

Any joint move tied to a subtask works the same way: change the subtask pattern, joint index and threshold in
`turn_onset_frames.py`, and the joint index in the read-outs (`ROLL`). The read-out needs frames where the
operator moves the joint decisively; on stride-sampled frames the with / flat / against split says nothing.

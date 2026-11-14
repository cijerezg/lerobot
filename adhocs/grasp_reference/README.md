# grasp_reference (2026-11-09)

Question: how do the good human demonstrations grasp each object (tool tilt, heading, tip height, arrival in the
last second, gripper opening / stall / hold angle, approach speed)? The per-object numbers behind the object table
of `src/lerobot/annotation/rubrics/correction_rubric.md`.

## Files

- `reference.py table`: every `grasp ...` segment of the annotated internal roots in `config_rl.yaml` (external
  pool and correction roots left out) with a close event inside; tool pose by FK on the twin arm
  (`annotation/corrections/arm.py`). Writes `outputs/grasp_reference/grasps.csv` and prints the per-object tables
  (median and quartiles) for grasps whose frames from 1 s before the close to the stall are all grade 4 or 5.
- `reference.py frames`: wrist | top image at the stall of 3 lid and 3 kit-flap grasps drawn from `grasps.csv`.

Definitions (close, stall, before, good) are in the script docstring.

## Run

    MUJOCO_GL=egl PYTHONPATH=lerobot/src .venv/bin/python lerobot/adhocs/grasp_reference/reference.py table
    MUJOCO_GL=egl PYTHONPATH=lerobot/src .venv/bin/python lerobot/adhocs/grasp_reference/reference.py frames

## Findings

The numbers of the 2026-11-09 run are the object table of the correction rubric. That run read the roots of the
mix40-nodepth-b512 run; rerunning reads today's config roots, so counts differ.

Changed on the move into the repo (2026-11-13), not rerun since: roots come from `config_rl.yaml` instead of a
checkpoint's `train_config.json`, the arm comes from `annotation/corrections/arm.py`, and the `calib` command
(top-camera overlay on frames of the 2026-11-09 inference runs) was dropped; `annotation/corrections/fit_top.py`
fits the top camera per episode.

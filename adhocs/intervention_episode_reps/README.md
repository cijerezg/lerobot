# intervention_episode_reps (2026-10-20)

Question: along one robot episode, what does the model represent and intend while it is off, while the operator
holds the arm, and after it is brought back to a state where it does the right thing? Is there a discontinuity
(a strategy change) in representation space, and does the model's chunk during an intervention point where the
operator went? Method: the stock representations probe and action-trace probe, run on the episode itself with the
prompt the console actually sent (subtask + precision/contact per step), then a read-out over the user's spans.

## Files

- `build_probe_root.py`: the inference episode as a probe root (`outputs/rebot_rollouts_2026-10-20-probe-v1`),
  subtask windows = the online subtasks. Copied from migration/rollout_review_2026-10-17/build_probe_root.py.
- `deployed_prompt.py`: the deployed metadata clause (quality/mistake/speed + the step's precision and contact).
- `reps_episode.py`: representations probe restricted to `--dataset.root` (the stock sampling plan would also take
  every training source in the config) with the deployed clause. Output = the stock probe's (trajectories.html,
  temporal.csv, cache/<group>.npy).
- `anchors.py` -> `anchors.txt`: 2 Hz anchors over the spans (edit SPANS), snapped to the image stride.
- `trace_episode.py`: action-trace probe on those anchors with the deployed clause. action_trace_records.npz keeps
  the state, the executed chunk (the operator's during an intervention), every flow sample and the FAST decode.
- `readout.py`: per window (WINDOWS at the top: spans, segments, reference frames) the episode-centered cosine
  matrices per token group and layer, segment-to-segment block tables, local cosine + cosine to the reference frames
  + executed speed on one time axis, key-frame strip, and from the trace records the model's commanded end-effector
  displacement vs the executed one (magnitude, direction agreement, fan spread, gripper). summary.md has the numbers.

## Run

    uv run python lerobot/adhocs/intervention_episode_reps/build_probe_root.py
    uv run python lerobot/adhocs/intervention_episode_reps/anchors.py
    migration/inference_review_2026-10-20/run_probes.sh      # the two GPU probes + the read-out
    uv run python -m lerobot.scripts.view_probes migration/inference_review_2026-10-20/reps_probe   # frame slider

## Reuse

Another episode: SRC/OUT in build_probe_root.py, SPANS in anchors.py, WINDOWS (spans, segments, reference) in
readout.py, the paths at the top of readout.py. The two wrappers take any probe root with subtask windows whose
names are in `policy.eval_subtasks`.

`subset_cache.py` crops and thins the collected cache to a window; re-running `reps_episode.py` in plot mode on the
copy gives the stock interactive trajectories.html for just that window (reps_zoom_<window>/ in the review folder).
`tabs_page.py` puts several probe pages behind tabs in one html (checkpoints.html in the review folder: one tab per
checkpoint of the robot-window + wrist-roll-demo matrix).
`compare_page.py` puts two probe pages side by side with a dropdown per pane (compare_checkpoints.html in the review
folder; the ckpt 200 entry fills in once run_ckpts.sh 200 has finished).

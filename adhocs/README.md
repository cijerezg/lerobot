# adhocs

One-off analyses that answered a real question and are worth running again. One folder per analysis, each with
a README that states the question, how to run it from the workspace root, and what it found. Scripts take the
workspace layout for granted (`config_rl.yaml`, `outputs/`), like `CHEAT_SHEET.md`. Nothing here is on the
training or probe path; the probe suite does not import it.

| folder | question | date |
|---|---|---|
| `joint_onset_trace/` | does the model know a joint move (wrist roll at the clipper grasp), and does it apply it on the robot? per-checkpoint | 2026-10-17 |
| `intervention_episode_reps/` | along one robot episode, how do representations and the model's intended chunk change across an off phase, an operator intervention and the recovery? representations + action-trace probes on the episode with the deployed prompt | 2026-10-20 |

# joint_traces

Question (2026-11-10): how bad are the action discontinuities in the rollouts recorded before the RTC window fix
(2026-11-09)?

Plots the recorded `action` and `observation.state` per joint over whole episodes, one interactive plotly HTML
(one figure per episode, one subplot per joint, shared time axis, gray = operator intervention). Episodes are
drawn at random from the rollout roots; spliced episodes are left out because their cut seam is a jump the robot
never made.

```bash
uv run python lerobot/adhocs/joint_traces/plot.py                     # 5 episodes, seed 0, pre-fix rollout roots
uv run python lerobot/adhocs/joint_traces/plot.py --seed 1 --n 5 \
    --roots outputs/rebot_2026-11-10-annotated-v1 --out outputs/analysis/joint_traces/post_fix.html

uv run python lerobot/adhocs/joint_traces/compare_repairs.py          # 3 episodes of 10-28, three candidate repairs
```

Outputs: `outputs/analysis/joint_traces/joint_traces.html`, `outputs/analysis/joint_traces/compare_repairs.html`.

## Findings (2026-11-10)

Every rollout recorded up to 2026-11-09 has a one-frame step in the commanded action at each chunk handover, every
15-17 frames (2-9 % of policy frames change speed by > 3 deg; 7-57 % of 30-frame chunks contain one). Recordings made
after the RTC window fix (roots 11-10b, 11-10c) do not (0.1-0.2 % of frames, mostly in the 1.5 s after a handback).
The root named `rebot_2026-11-10-annotated-v1` holds the 11-09 recordings and is pre-fix.

`compare_repairs.py` overlays three candidate repairs (bridge, offset blend, local low-pass) on action and state. The
offset blend was chosen; `lerobot.annotation.rebot.seam_repair_root` writes it into new `-seamfix1` roots.

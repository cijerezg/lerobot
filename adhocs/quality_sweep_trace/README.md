# quality_sweep_trace (2026-11-13)

Question: what does the quality clause do to the commanded path? One frame, one flow-noise draw, the prompt
asking for quality 1..5 (no mistake, speed 5, the frame's own precision and contact), drawn as the five
paths of the Action Inspector's 3D scene next to the demonstration.

## Files

- `run_quality_sweep_trace.py`: anchors, the five-prompt forward per anchor, and the page. The page is the
  action-trace probe's own (`_analyse`, `_figure`, `_write_dashboard_html`) with q1..q5 as the sample fan.

## Run

    uv run python lerobot/adhocs/quality_sweep_trace/run_quality_sweep_trace.py --config_path=config_rl_validate.yaml \
        --policy.pretrained_path=outputs/rebot-corr10-nodepth-b128-20261111/checkpoints/002000/pretrained_model \
        --out_dir=outputs/quality_sweep_trace/corr10-ckpt2000_2026-11-13

Flags: `--episodes=0,1,7,8` (val episodes; each needs grasp / move / release as its first three segments),
`--seed=0` (the flow-noise draw shared by the five prompts). About 15 s for 16 anchors.

Anchors: per episode four stages, approach (2 s before the first close), near grasp (0.3 s before the last
close), move (middle of the move segment), near release (0.3 s before the open). Default episodes: brown
shirt, nail clipper, cup, bit.

## Outputs (under out_dir)

- `quality_sweep_trace.html`: slider over the 16 anchors; q1 (dark) to q5 (light), black = demonstration.

## Found

Nothing recorded yet (corr10 ckpt 2000 page built 2026-11-13, not read).

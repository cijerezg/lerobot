"""Closed-loop sim eval: the policy runs in the MuJoCo nail-clipper scene on every layout (run_sim_eval.sh), and this
summary turns one pass into a probe the viewers show.

    python lerobot/scripts/probes/sim_summary.py <...>/validation/step_XXXXXXXX/sim_nail [--config config_rl_sim.yaml]

Writes summary.json, summary.md and index.json next to the layout folders: one row per layout and episode from
subtask_events.json, and one video panel per (layout, episode, view) from ``<layout>/videos/ep<N>_<view>.mp4``
(``record_inference_videos``: top camera, wrist camera, colorized wrist depth at 30 fps). The cursor is the furthest
step the scorer justified from the body poses (robots/rebot_b601_sim/nail_task.py); ``steps_done`` counts them,
``success`` is both clippers seated. The config supplies the step names. ``compare_probes`` plays the same
(layout, episode, view) video of several runs side by side.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from lerobot.probes.manifest import Metric, Panel, write_index

ap = argparse.ArgumentParser()
ap.add_argument("sim_dir", type=Path)
ap.add_argument("--config", type=Path, default=None)
args = ap.parse_args()

steps = yaml.safe_load(args.config.read_text())["policy"]["eval_subtasks"] if args.config is not None else []
rows = []
for layout in sorted(p for p in args.sim_dir.iterdir() if p.is_dir()):
    events = layout / "subtask_events.json"
    if not events.exists():
        rows.append(dict(layout=layout.name, episode=None, success=None, steps_done=None, last_event=None, note="no subtask_events.json (run failed)"))
        continue
    for ep in json.loads(events.read_text()):
        furthest = max([e["cursor"] for e in ep["events"]], default=0)
        rows.append(dict(layout=layout.name, episode=ep["episode"], success=ep["success"], steps_done=furthest,
                         furthest_step=steps[furthest - 1] if 0 < furthest <= len(steps) else None,
                         last_event=ep["events"][-1]["why"] if ep["events"] else None, t_end=ep["events"][-1]["t"] if ep["events"] else None))
done = [r for r in rows if r["success"] is not None]
summary = dict(episodes=len(done), successes=sum(r["success"] for r in done), mean_steps_done=round(sum(r["steps_done"] for r in done) / len(done), 2) if done else None,
               n_steps=len(steps), rows=rows)
(args.sim_dir / "summary.json").write_text(json.dumps(summary, indent=1))
lines = [f"# sim_nail: {summary['successes']}/{summary['episodes']} episodes succeeded, mean steps done {summary['mean_steps_done']} of {len(steps)}", "",
         "| layout | episode | success | steps done | furthest step | last event |", "|---|---|---|---|---|---|"]
lines += [f"| {r['layout']} | {r['episode']} | {r['success']} | {r['steps_done']} | {r.get('furthest_step')} | {r.get('note') or r['last_event']} |" for r in rows]
(args.sim_dir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

panels = []
for layout in sorted(p for p in args.sim_dir.iterdir() if p.is_dir()):
    for video in sorted(layout.glob("videos/ep*_*.mp4"), key=lambda p: (int(re.match(r"ep(\d+)_", p.name)[1]), p.name)):
        episode, view = re.match(r"ep(\d+)_(.+)\.mp4", video.name).groups()
        panels.append(Panel(file=video.relative_to(args.sim_dir).as_posix(), caption=f"{layout.name} episode {episode}: {view}",
                            how="The sim's rendered view at the env fps, one file per episode; depth is TURBO over the valid pixels' 2-98 percentile, invalid black.",
                            primary=view == "top" and episode == "0"))
write_index(str(args.sim_dir), sys.modules[__name__], title="Sim Nail", group="Actions",
            claim="Does the policy put the nail clippers in the case when it closes the loop in the sim?",
            metrics=[Metric("successes", "Episodes succeeded", good="high", fmt=0, primary=True, trend=True),
                     Metric("episodes", "Episodes run", fmt=0),
                     Metric("mean_steps_done", "Mean scripted steps reached", good="high", fmt=2, note=f"of {len(steps)} steps", trend=True)],
            panels=panels, summary=summary)

"""Summarise a run_sim_eval.sh pass: one row per layout and episode from subtask_events.json.

    python lerobot/scripts/probes/sim_summary.py <...>/validation/step_XXXXXXXX/sim_nail [--config config_rl_sim.yaml]

Writes summary.json and summary.md next to the layout folders. The cursor is the furthest step the scorer
justified from the body poses (robots/rebot_b601_sim/nail_task.py); `steps_done` counts them, `success`
is both clippers seated. The config supplies the step names.
"""
import argparse
import json
from pathlib import Path

import yaml

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

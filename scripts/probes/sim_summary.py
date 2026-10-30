"""Closed-loop sim eval: the policy runs in the MuJoCo scenes of the eval set (run_sim_eval.sh, one task per scene from
robots/rebot_b601_sim/tasks.py), and this summary turns one pass into a probe the viewers show.

    python lerobot/scripts/probes/sim_summary.py <...>/validation/step_XXXXXXXX/sim

Writes summary.json, summary.md and index.json next to the layout folders: one row per layout and episode from
subtask_events.json (``reward`` = cumulative value of the furthest stage the scorer reached, ``success`` = its last
stage, ``steps_done`` = the furthest subtask cursor of the scene's script), a per-layout block (tier, mean reward,
success rate) and ``reward.<layout>``, ``reward.easy`` / ``reward.hard`` (mean over the tier's layouts of the
per-layout episode mean) and ``reward.mean`` (mean of the two tiers), plus one video panel per (layout, episode, view).
Placement variants (``<layout>_v<k>`` folders, tasks.py ``layout_name``) keep their own ``reward.<layout>_v<k>`` and roll up
into ``reward.<layout>`` (mean over all their episodes); the tiers average over base layouts only.
from ``<layout>/videos/ep<N>_<view>.mp4`` (``record_inference_videos``: top camera, wrist camera, colorized wrist
depth at 30 fps). ``compare_probes`` plays the same (layout, episode, view) video of several runs side by side.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.robots.rebot_b601_sim.tasks import TASKS, TIERS, layout_name

ap = argparse.ArgumentParser()
ap.add_argument("sim_dir", type=Path)
args = ap.parse_args()

layouts = sorted(p for p in args.sim_dir.iterdir() if p.is_dir() and layout_name(p.name) in TASKS)
rows, per_layout = [], {}
for layout in layouts:
    base = layout_name(layout.name)
    task = TASKS[base]
    events = layout / "subtask_events.json"
    if not events.exists():
        rows.append(dict(layout=layout.name, base=base, tier=task.tier, episode=None, success=None, reward=None, stage=None, steps_done=None, last_event=None, note="no subtask_events.json (run failed)"))
        continue
    done = []
    for ep in json.loads(events.read_text()):
        furthest = max([e["cursor"] for e in ep["events"]], default=0)
        rows.append(dict(layout=layout.name, base=base, tier=task.tier, episode=ep["episode"], success=ep["success"], reward=ep["reward"], stage=ep["stage"], steps_done=furthest,
                         furthest_step=task.eval_subtasks[furthest - 1] if 0 < furthest <= len(task.eval_subtasks) else None,
                         last_event=ep["events"][-1]["why"] if ep["events"] else None, t_end=ep["events"][-1]["t"] if ep["events"] else None))
        done.append(ep)
    per_layout[layout.name] = dict(tier=task.tier, task=task.task, base=base, episodes=len(done), mean_reward=round(sum(ep["reward"] for ep in done) / len(done), 3) if done else None,
                                   success_rate=round(sum(ep["success"] for ep in done) / len(done), 3) if done else None)
done = [r for r in rows if r["success"] is not None]
# placement variants roll up into their base layout: the base's row is the mean over every episode of its variants
for base in sorted({v["base"] for name, v in per_layout.items() if name != v["base"]}):
    names = [name for name, v in per_layout.items() if v["base"] == base]
    eps = [r for r in done if r["layout"] in names]
    per_layout[base] = dict(tier=TASKS[base].tier, task=TASKS[base].task, base=base, variants=names, episodes=len(eps),
                            mean_reward=round(sum(r["reward"] for r in eps) / len(eps), 3) if eps else None,
                            success_rate=round(sum(r["success"] for r in eps) / len(eps), 3) if eps else None)
per_layout = dict(sorted(per_layout.items(), key=lambda kv: (kv[1]["base"], kv[0] != kv[1]["base"], kv[0])))
tier_means = {tier: [v["mean_reward"] for name, v in per_layout.items() if name == v["base"] and v["tier"] == tier and v["mean_reward"] is not None] for tier in TIERS}
reward = {name: v["mean_reward"] for name, v in per_layout.items()}
reward.update({tier: round(sum(m) / len(m), 3) if m else None for tier, m in tier_means.items()})
tiers_present = [reward[tier] for tier in TIERS if reward[tier] is not None]
reward["mean"] = round(sum(tiers_present) / len(tiers_present), 3) if tiers_present else None
summary = dict(episodes=len(done), successes=sum(r["success"] for r in done), success_rate=round(sum(r["success"] for r in done) / len(done), 3) if done else None,
               reward=reward, layouts=per_layout, rows=rows)
(args.sim_dir / "summary.json").write_text(json.dumps(summary, indent=1))
lines = [f"# sim: reward easy {reward['easy']} / hard {reward['hard']} / mean {reward['mean']}; {summary['successes']}/{summary['episodes']} episodes succeeded", "",
         "| layout | tier | episodes | mean reward | success rate |", "|---|---|---|---|---|"]
lines += [f"| {name} | {v['tier']} | {v['episodes']} | {v['mean_reward']} | {v['success_rate']} |" for name, v in per_layout.items()]
lines += ["", "| layout | episode | reward | stage | success | steps done | furthest step | last event |", "|---|---|---|---|---|---|---|---|"]
lines += [f"| {r['layout']} | {r['episode']} | {r['reward']} | {r['stage']} | {r['success']} | {r['steps_done']} | {r.get('furthest_step')} | {r.get('note') or r['last_event']} |" for r in rows]
(args.sim_dir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

panels = []
for layout in layouts:
    for video in sorted(layout.glob("videos/ep*_*.mp4"), key=lambda p: (int(re.match(r"ep(\d+)_", p.name)[1]), p.name)):
        episode, view = re.match(r"ep(\d+)_(.+)\.mp4", video.name).groups()
        panels.append(Panel(file=video.relative_to(args.sim_dir).as_posix(), caption=f"{layout.name} episode {episode}: {view}",
                            how="The sim's rendered view at the env fps, one file per episode; depth is TURBO over the valid pixels' 2-98 percentile, invalid black.",
                            primary=view == "top" and episode == "0"))
metrics = [Metric("reward.mean", "Staged reward, mean of tiers", good="high", fmt=3, primary=True, trend=True),
           Metric("reward.easy", "Staged reward, easy tier", good="high", fmt=3, trend=True),
           Metric("reward.hard", "Staged reward, hard tier", good="high", fmt=3, trend=True),
           Metric("success_rate", "Success rate", good="high", fmt=3, trend=True),
           Metric("successes", "Episodes succeeded", good="high", fmt=0),
           Metric("episodes", "Episodes run", fmt=0)]
metrics += [Metric(f"reward.{name}", f"Staged reward, {name} ({v['tier']})", good="high", fmt=3,
                   note=f"{v['task']}; mean over the placement variants {', '.join(v['variants'])}" if "variants" in v else v["task"]) for name, v in per_layout.items()]
write_index(str(args.sim_dir), sys.modules[__name__], title="Sim eval", group="Actions",
            claim="How far does the policy get on the sim tasks (easy and hard tiers) when it closes the loop?",
            metrics=metrics, panels=panels, summary=summary)

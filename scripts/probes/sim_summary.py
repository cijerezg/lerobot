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
A judge stage (run_sim_eval.sh JUDGE=1 -> ``<sim>/judge/<layout>_ep<N>.judge.metrics.json``, lerobot/scripts/probes/sim_judge.sh) adds
``reward_judge.<layout>`` / ``.<tier>`` / ``.mean`` (Gemma's reward per episode: value of the furthest stage it named, rolled up like ``reward``)
and ``judge_agreement.*`` (mean over the scored episodes of final_stage_match, stage_accuracy_running_max, spearman_progress_max, and
the judge-minus-scorer reward difference); unscored episodes carry null judge columns.
Twin layouts (tasks.py TWIN_TIERS, scorers in twin_tasks.py) add ``reward_dense`` per episode: ``reward_dense.<layout>`` /
``reward_dense.<tier>`` roll up like ``reward`` and each episode's dense record (stage timeline, carried time, drop) is written to
``<layout>/dense_ep<N>.json``. The twin tiers only appear in ``reward`` when they were run; ``reward.mean`` stays the mean of the tiers present.
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
# judge stage outputs (sim_judge.sh): {(layout, episode): metrics}; an episode without answers (n == 0) counts as unscored
judge = {}
for path in sorted((args.sim_dir / "judge").glob("*_ep*.judge.metrics.json")):
    layout_ep, ep = re.fullmatch(r"(.+)_ep(\d+)\.judge\.metrics\.json", path.name).groups()
    m = json.loads(path.read_text())
    if m.get("n", 0) > 0 and m.get("judge_reward") is not None:
        judge[(layout_ep, int(ep))] = m
JUDGE_KEYS = ("judge_reward", "judge_success", "final_stage_match", "stage_accuracy_running_max", "spearman_progress_max")
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
                         last_event=ep["events"][-1]["why"] if ep["events"] else None, t_end=ep["events"][-1]["t"] if ep["events"] else None,
                         reward_dense=ep.get("reward_dense"), carried_time_s=ep.get("carried_time_s"),
                         drop_distance_m=(ep.get("drop") or {}).get("distance_to_basket_m"),
                         **{k: judge.get((layout.name, ep["episode"]), {}).get(k) for k in JUDGE_KEYS}))
        if "reward_dense" in ep:  # twin_tasks dense record, one json per episode next to subtask_events.json
            dense = {k: ep[k] for k in ep if k not in ("events",)}
            (layout / f"dense_ep{ep['episode']}.json").write_text(json.dumps(dense, indent=1) + "\n")
        done.append(ep)
    per_layout[layout.name] = dict(tier=task.tier, task=task.task, base=base, episodes=len(done), mean_reward=round(sum(ep["reward"] for ep in done) / len(done), 3) if done else None,
                                   success_rate=round(sum(ep["success"] for ep in done) / len(done), 3) if done else None)
    if done and all("reward_dense" in ep for ep in done):
        per_layout[layout.name]["mean_reward_dense"] = round(sum(ep["reward_dense"] for ep in done) / len(done), 3)
    scored = [judge[(layout.name, ep["episode"])] for ep in done if (layout.name, ep["episode"]) in judge]
    if scored:
        per_layout[layout.name]["judged"] = len(scored)
        per_layout[layout.name]["mean_reward_judge"] = round(sum(m["judge_reward"] for m in scored) / len(scored), 3)
done = [r for r in rows if r["success"] is not None]
# placement variants roll up into their base layout: the base's row is the mean over every episode of its variants
for base in sorted({v["base"] for name, v in per_layout.items() if name != v["base"]}):
    names = [name for name, v in per_layout.items() if v["base"] == base]
    eps = [r for r in done if r["layout"] in names]
    per_layout[base] = dict(tier=TASKS[base].tier, task=TASKS[base].task, base=base, variants=names, episodes=len(eps),
                            mean_reward=round(sum(r["reward"] for r in eps) / len(eps), 3) if eps else None,
                            success_rate=round(sum(r["success"] for r in eps) / len(eps), 3) if eps else None)
    if eps and all(r["reward_dense"] is not None for r in eps):
        per_layout[base]["mean_reward_dense"] = round(sum(r["reward_dense"] for r in eps) / len(eps), 3)
    scored = [r for r in eps if r["judge_reward"] is not None]
    if scored:
        per_layout[base]["judged"] = len(scored)
        per_layout[base]["mean_reward_judge"] = round(sum(r["judge_reward"] for r in scored) / len(scored), 3)
per_layout = dict(sorted(per_layout.items(), key=lambda kv: (kv[1]["base"], kv[0] != kv[1]["base"], kv[0])))
# tiers: easy / hard always reported; the twin tiers only when one of their layouts was run (old summaries unchanged)
tiers = [tier for tier in TIERS if tier in ("easy", "hard") or any(v["tier"] == tier for v in per_layout.values())]
tier_means = {tier: [v["mean_reward"] for name, v in per_layout.items() if name == v["base"] and v["tier"] == tier and v["mean_reward"] is not None] for tier in tiers}
reward = {name: v["mean_reward"] for name, v in per_layout.items()}
reward.update({tier: round(sum(m) / len(m), 3) if m else None for tier, m in tier_means.items()})
tiers_present = [reward[tier] for tier in tiers if reward[tier] is not None]
reward["mean"] = round(sum(tiers_present) / len(tiers_present), 3) if tiers_present else None
reward_dense = {name: v["mean_reward_dense"] for name, v in per_layout.items() if "mean_reward_dense" in v}
dense_tiers = {tier: [v["mean_reward_dense"] for name, v in per_layout.items() if name == v["base"] and v["tier"] == tier and "mean_reward_dense" in v] for tier in tiers}
reward_dense.update({tier: round(sum(m) / len(m), 3) for tier, m in dense_tiers.items() if m})
reward_judge = {name: v["mean_reward_judge"] for name, v in per_layout.items() if "mean_reward_judge" in v}
judge_tiers = {tier: [v["mean_reward_judge"] for name, v in per_layout.items() if name == v["base"] and v["tier"] == tier and "mean_reward_judge" in v] for tier in tiers}
reward_judge.update({tier: round(sum(m) / len(m), 3) for tier, m in judge_tiers.items() if m})
judge_tiers_present = [reward_judge[tier] for tier in tiers if tier in reward_judge]
if judge_tiers_present:
    reward_judge["mean"] = round(sum(judge_tiers_present) / len(judge_tiers_present), 3)
scored_rows = [r for r in done if r["judge_reward"] is not None]
judge_agreement = {}
if scored_rows:
    judge_agreement = dict(episodes=len(scored_rows),
                           reward_diff=round(sum(r["judge_reward"] - r["reward"] for r in scored_rows) / len(scored_rows), 3),
                           **{k: round(sum(r[k] for r in scored_rows if r[k] is not None) / max(1, sum(r[k] is not None for r in scored_rows)), 3)
                              for k in ("final_stage_match", "stage_accuracy_running_max", "spearman_progress_max")})
summary = dict(episodes=len(done), successes=sum(r["success"] for r in done), success_rate=round(sum(r["success"] for r in done) / len(done), 3) if done else None,
               reward=reward, layouts=per_layout, rows=rows)
if reward_dense:
    summary["reward_dense"] = reward_dense
if reward_judge:
    summary["reward_judge"] = reward_judge
    summary["judge_agreement"] = judge_agreement
(args.sim_dir / "summary.json").write_text(json.dumps(summary, indent=1))
twin_line = "".join(f" / {tier} {reward[tier]}" for tier in tiers if tier not in ("easy", "hard"))
judge_line = (f"; judge reward mean {reward_judge.get('mean')} on {judge_agreement['episodes']} episodes (final stage match {judge_agreement['final_stage_match']}, "
              f"judge - scorer reward {judge_agreement['reward_diff']:+})") if reward_judge else ""
lines = [f"# sim: reward easy {reward['easy']} / hard {reward['hard']}{twin_line} / mean {reward['mean']}; {summary['successes']}/{summary['episodes']} episodes succeeded{judge_line}", "",
         "| layout | tier | episodes | mean reward | success rate |" + (" mean dense |" if reward_dense else "") + (" judged | mean judge reward |" if reward_judge else ""),
         "|---|---|---|---|---|" + ("---|" if reward_dense else "") + ("---|---|" if reward_judge else "")]
lines += [f"| {name} | {v['tier']} | {v['episodes']} | {v['mean_reward']} | {v['success_rate']} |" + (f" {v.get('mean_reward_dense')} |" if reward_dense else "")
          + (f" {v.get('judged')} | {v.get('mean_reward_judge')} |" if reward_judge else "") for name, v in per_layout.items()]
lines += ["", "| layout | episode | reward | stage | success | steps done | furthest step | last event |" + (" dense | carried s | drop m |" if reward_dense else "")
          + (" judge reward | judge success | final match | stage acc | spearman |" if reward_judge else ""),
          "|---|---|---|---|---|---|---|---|" + ("---|---|---|" if reward_dense else "") + ("---|---|---|---|---|" if reward_judge else "")]
lines += [f"| {r['layout']} | {r['episode']} | {r['reward']} | {r['stage']} | {r['success']} | {r['steps_done']} | {r.get('furthest_step')} | {r.get('note') or r['last_event']} |"
          + (f" {r.get('reward_dense')} | {r.get('carried_time_s')} | {r.get('drop_distance_m')} |" if reward_dense else "")
          + (" " + " | ".join(str(r.get(k)) for k in JUDGE_KEYS) + " |" if reward_judge else "") for r in rows]
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
metrics += [Metric(f"reward.{tier}", f"Staged reward, {tier} tier", good="high", fmt=3, trend=True) for tier in tiers if tier not in ("easy", "hard")]
metrics += [Metric(f"reward.{name}", f"Staged reward, {name} ({v['tier']})", good="high", fmt=3,
                   note=f"{v['task']}; mean over the placement variants {', '.join(v['variants'])}" if "variants" in v else v["task"]) for name, v in per_layout.items()]
metrics += [Metric(f"reward_dense.{key}", f"Dense staged reward, {key}", good="high", fmt=3, trend=key in tiers,
                   note="stage value + within-stage progress, running max over the episode (twin_tasks.py)") for key in reward_dense]
metrics += [Metric(f"reward_judge.{key}", f"Judge reward, {key}", good="high", fmt=3, trend=key in tiers or key == "mean", primary=key == "mean",
                   note="Gemma 4 31B-it over the recorded videos, one query per 5 s window: value of the furthest stage it named (sim_judge.sh)") for key in reward_judge]
metrics += [Metric(f"judge_agreement.{key}", f"Judge vs scorer: {key}", good=good, fmt=3, trend=True,
                   note="mean over the judged episodes (rewards/vlm_judge/compare_to_scorer.py)")
            for key, good in (("final_stage_match", "high"), ("stage_accuracy_running_max", "high"), ("spearman_progress_max", "high"), ("reward_diff", "none")) if judge_agreement]
write_index(str(args.sim_dir), sys.modules[__name__], title="Sim eval", group="Actions",
            claim="How far does the policy get on the sim tasks (easy and hard tiers) when it closes the loop?",
            metrics=metrics, panels=panels, summary=summary)

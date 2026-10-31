"""Agreement between a judge timeline (score_episode.py) and the sim scorer's record of the same episode
(``<layout>/subtask_events.json``: per episode ``events`` = [{t, cursor, stage, reward, why}], t in sim seconds).

    uv run python -m lerobot.rewards.vlm_judge.compare_to_scorer --timeline <run>.json \
        --events <...>/sim/bottle_basket_v1/subtask_events.json [--episode 1]

Prints: cursor accuracy per step (judge subtask -> its script cursor vs the scorer's cursor at that t; off-task
answers count as wrong and are listed), stage accuracy, per-stage lag (first t the judge's running-max stage reaches
the stage minus the scorer's t for it; negative = judge early), Spearman between judge progress and the scorer's
cumulative reward (raw and running max). The scorer's value is a step function of its events; between events it holds.
Exit code 0 always; this is a report."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .task_spec import TaskSpec


def scorer_at(events: list[dict], t: float) -> tuple[int, str | None, float]:
    cursor, stage, reward = 0, None, 0.0
    for e in events:
        if e["t"] > t + 1e-6:
            break
        cursor, stage, reward = e["cursor"], e["stage"], e["reward"]
    return cursor, stage, reward


def compare(timeline: dict, record: dict, spec: TaskSpec) -> dict:
    steps = timeline["steps"]
    events = record["events"]
    sc = [scorer_at(events, s["t"]) for s in steps]
    judge_cursor = np.array([s["cursor"] for s in steps])
    judge_stage = np.array([s["stage_id"] for s in steps])
    scorer_cursor = np.array([c for c, _, _ in sc])
    scorer_stage = np.array([spec.stage_by_name(name).id for _, name, _ in sc])
    scorer_value = np.array([r for _, _, r in sc])
    judge_p = np.array([s["progress"] for s in steps])
    judge_pmax = np.array([s["progress_max"] for s in steps])
    judge_stage_max = np.maximum.accumulate(judge_stage)

    lags = {}
    for stage in spec.stages[1:]:
        reached = [e["t"] for e in events if e["why"] == f"reached {stage.name}"]
        if not reached:
            continue
        hit = [s["t"] for s, m in zip(steps, judge_stage_max, strict=True) if m >= stage.id]
        lags[stage.name] = dict(scorer_t=reached[0], judge_t=hit[0] if hit else None, lag_s=None if not hit else round(hit[0] - reached[0], 2))
    never = [s.name for s in spec.stages[1:] if s.name not in lags and (judge_stage_max >= s.id).any()]

    def rho(a, b):
        if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return round(float(spearmanr(a, b).statistic), 3)

    return dict(
        n_steps=len(steps),
        cursor_accuracy=round(float((judge_cursor == scorer_cursor).mean()), 3),
        cursor_accuracy_within_1=round(float((np.abs(judge_cursor - scorer_cursor) <= 1).mean()), 3),
        off_task_steps=int((judge_cursor < 0).sum()),
        stage_accuracy=round(float((judge_stage == scorer_stage).mean()), 3),
        stage_accuracy_running_max=round(float((judge_stage_max == scorer_stage).mean()), 3),
        stage_lag_s=lags,
        stages_judge_only=never,
        spearman_progress=rho(judge_p, scorer_value),
        spearman_progress_max=rho(judge_pmax, scorer_value),
        scorer_final=dict(success=record["success"], reward=record["reward"], stage=record["stage"]),
        judge_final=dict(progress_max=float(judge_pmax[-1]), stage=spec.stages[int(judge_stage_max[-1])].name),
        confusion=_confusion(judge_cursor, scorer_cursor, len(spec.script)),
    )


def _confusion(judge: np.ndarray, scorer: np.ndarray, n: int) -> list[list[int]]:
    """rows = scorer cursor, cols = judge cursor (last col = off-task)."""
    m = np.zeros((n, n + 1), dtype=int)
    for j, s in zip(judge, scorer, strict=True):
        m[s, n if j < 0 else j] += 1
    return m.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--episode", type=int, default=None, help="defaults to the timeline's episode")
    ap.add_argument("--out", type=Path, default=None, help="also write the metrics json here")
    args = ap.parse_args()
    timeline = json.loads(args.timeline.read_text())
    spec = TaskSpec.load(timeline["task"])
    episode = timeline["episode"] if args.episode is None else args.episode
    record = next(r for r in json.loads(args.events.read_text()) if r["episode"] == episode)
    m = compare(timeline, record, spec)
    print(f"{spec.name} episode {episode}: {m['n_steps']} judge steps vs {len(record['events'])} scorer events")
    print(f"  cursor accuracy        {m['cursor_accuracy']:.3f}  (within 1: {m['cursor_accuracy_within_1']:.3f}, off-task answers {m['off_task_steps']})")
    print(f"  stage accuracy         {m['stage_accuracy']:.3f}  (running max: {m['stage_accuracy_running_max']:.3f})")
    print(f"  spearman progress      {m['spearman_progress']}  (running max: {m['spearman_progress_max']})")
    for name, lag in m["stage_lag_s"].items():
        print(f"  stage {name:16s} scorer t={lag['scorer_t']:6.2f}  judge t={lag['judge_t']}  lag={lag['lag_s']} s")
    if m["stages_judge_only"]:
        print(f"  stages the judge reached but the scorer never did: {m['stages_judge_only']}")
    print(f"  final: scorer {m['scorer_final']}  judge {m['judge_final']}")
    print("  confusion (rows scorer cursor, cols judge cursor + off-task):")
    for i, row in enumerate(m["confusion"]):
        print(f"    {i}: {row}")
    if args.out is not None:
        args.out.write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()

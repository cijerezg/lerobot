"""Replay recorded sim episodes through the live subtask assigner (assigner.py, Qwen3.5-9B on the Spark :8013) and compare
its cursor with the scorer's.

    # one episode
    uv run python -m lerobot.rewards.vlm_judge.assign_episode --sim-dir <sim>/twin_l1_pill_basket_far_p1 --episode 0 --out x.json
    # a whole sim run (layout -> script from tasks.py), 4 episodes in flight, resumable
    uv run python -m lerobot.rewards.vlm_judge.assign_episode --sim-run outputs/twin_eval_res600_2026-10-30/validation/step_00000000/sim \
        --episodes 0 --out-dir migration/digital_twin_2026-10-29/judge/runs/assign_res600 --workers 4

Closed loop by default, as live: the offered steps follow the assigner's OWN previous answer (cursor 0 at the reset).
``--teacher`` offers the steps around the SCORER's cursor instead (recognition alone, no drift). Every ``--every`` s of video
(0.5 = the live cadence) one query with the last ``--frames-per-camera`` frames per camera at ``--period`` s spacing.

Metrics: cursor_accuracy (assigner cursor == scorer cursor at the query t), within_1, early / late (assigner ahead / behind),
advance_lag_s per scorer cursor change (first t >= t_scorer - 5 s the assigner holds a cursor >= it, minus t_scorer; mean, max),
missed_advances (scorer advanced, assigner never followed), latency_p50_s. The scorer's cursor counts as the truth; the
two clocks agree within the 0.3 s inference lockstep."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .assigner import AssignerClient
from .batch import run_batch, sim_episodes, write_table
from .compare_to_scorer import scorer_at
from .config import ASSIGNER, JudgeConfig
from .frames import decode_sim_episode
from .prompt import Frame


def assign_sim_episode(cfg: JudgeConfig, sim_dir: Path, episode: int, out: Path, every: float, teacher: bool, with_evidence: bool,
                       max_queries: int | None = None, verbose: bool = False) -> dict:
    from lerobot.robots.rebot_b601_sim.tasks import task_for

    task = task_for(sim_dir.name)
    script = list(task.eval_subtasks)
    record = next(r for r in json.loads((sim_dir / "subtask_events.json").read_text()) if r["episode"] == episode)
    events = record["events"]
    client = AssignerClient(cfg, task.task, script, with_evidence)
    samples = decode_sim_episode(sim_dir, episode, cfg.cameras, cfg.frame_period_s)
    stride = max(1, round(every / cfg.frame_period_s))
    rows, failures, cursor = [], 0, 0
    for i in range(cfg.frames_per_camera - 1, len(samples), stride):
        if max_queries is not None and len(rows) + failures >= max_queries:
            break
        t = samples[i][0]
        truth = scorer_at(events, t)[0]
        frames = [Frame(t=tw, camera=cam, image=imgs[cam]) for tw, imgs in samples[i - cfg.frames_per_camera + 1: i + 1] for cam in cfg.cameras]
        try:
            a, meta = client.query(truth if teacher else cursor, frames)
        except Exception as e:
            failures += 1
            print(f"{sim_dir.name} ep{episode} t={t:6.1f} FAILED {type(e).__name__}: {str(e)[:120]}")
            continue
        cursor = a.cursor
        rows.append(dict(t=t, cursor=a.cursor, step=a.step, confidence=a.confidence, offered=a.offered, evidence=a.evidence,
                         scorer_cursor=truth, latency_s=meta["latency_s"]))
        if verbose:
            print(f"t={t:6.1f} scorer {truth} assigner {a.cursor} {a.step:45s} conf {a.confidence:.2f} {meta['latency_s']:.2f}s {a.evidence[:50]}")
    m = metrics(rows, events, failures)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(source=str(sim_dir), episode=episode, task=task.task, script=script, model=cfg.model, endpoint=cfg.endpoint,
                                   teacher=teacher, every_s=every, period_s=cfg.frame_period_s, frames_per_camera=cfg.frames_per_camera,
                                   created=time.strftime("%Y-%m-%d %H:%M:%S"), metrics=m, scorer=dict(success=record["success"], stage=record["stage"]),
                                   steps=rows), indent=1))
    return m


def metrics(rows: list[dict], events: list[dict], failures: int) -> dict:
    if not rows:
        return dict(n=0, failures=failures)
    a = np.array([r["cursor"] for r in rows])
    s = np.array([r["scorer_cursor"] for r in rows])
    ts = np.array([r["t"] for r in rows])
    lags, missed = [], 0
    for prev, e in zip([dict(cursor=0)] + events, events, strict=False):
        if e["cursor"] <= prev["cursor"] or e["t"] > ts[-1]:
            continue
        hit = ts[a >= e["cursor"]]
        hit = hit[hit >= e["t"] - 5.0]  # ahead by more than 5 s counts from e.t - 5 (an earlier stray jump is not following it)
        if len(hit):
            lags.append(float(hit[0] - e["t"]))
        else:
            missed += 1
    return dict(
        n=len(rows), failures=failures,
        cursor_accuracy=round(float((a == s).mean()), 3), within_1=round(float((np.abs(a - s) <= 1).mean()), 3),
        early=round(float((a > s).mean()), 3), late=round(float((a < s).mean()), 3),
        advances=len(lags) + missed, missed_advances=missed,
        advance_lag_mean_s=round(float(np.mean(lags)), 2) if lags else None, advance_lag_max_s=round(float(np.max(lags)), 2) if lags else None,
        latency_p50_s=round(float(np.median([r["latency_s"] for r in rows])), 2),
        scorer_final_cursor=int(s[-1]), assigner_final_cursor=int(a[-1]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim-dir", type=Path, help="one layout folder (<sim>/<layout>), with --episode and --out")
    src.add_argument("--sim-run", type=Path, help="a sim folder holding layout folders, with --out-dir")
    ap.add_argument("--episode", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--layouts", default="*", help="glob over layout folder names (--sim-run)")
    ap.add_argument("--episodes", type=int, nargs="+", default=None, help="episode indices per layout (--sim-run); default all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--endpoint", default=ASSIGNER["endpoint"])
    ap.add_argument("--model", default=ASSIGNER["model"])
    ap.add_argument("--period", type=float, default=ASSIGNER["frame_period_s"])
    ap.add_argument("--frames-per-camera", type=int, default=ASSIGNER["frames_per_camera"])
    ap.add_argument("--every", type=float, default=0.5, help="seconds of video between queries (0.5 = live cadence)")
    ap.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    ap.add_argument("--teacher", action="store_true", help="offer the steps around the scorer's cursor (open loop)")
    ap.add_argument("--evidence", action="store_true", help="ask for a short evidence string (more tokens)")
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()

    cfg = JudgeConfig(endpoint=args.endpoint, model=args.model, cameras=args.cameras, frames_per_camera=args.frames_per_camera,
                      frame_period_s=args.period, max_tokens=ASSIGNER["max_tokens"] + (40 if args.evidence else 0))
    if args.sim_dir is not None:
        m = assign_sim_episode(cfg, args.sim_dir, args.episode, args.out, args.every, args.teacher, args.evidence, args.max_queries, verbose=True)
        print(json.dumps(m, indent=1))
        return
    jobs = sim_episodes(args.sim_run, args.layouts, args.episodes)
    print(f"{len(jobs)} episodes, {args.workers} in flight, model {cfg.model} at {cfg.endpoint}")
    rows = run_batch(jobs, args.out_dir, "assign", lambda d, ep, out: assign_sim_episode(cfg, d, ep, out, args.every, args.teacher, args.evidence, args.max_queries), args.workers)
    write_table(rows, ["cursor_accuracy", "within_1", "early", "late", "missed_advances", "advance_lag_mean_s", "latency_p50_s"],
                args.out_dir / "summary.md", f"Assigner {cfg.model} on {args.sim_run} ({'teacher' if args.teacher else 'closed loop'}, every {args.every} s)")


if __name__ == "__main__":
    main()

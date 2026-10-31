"""Run the judge over recorded episodes and write a per-window timeline.

    uv run python -m lerobot.rewards.vlm_judge.score_episode --sim-dir <...>/sim/bottle_basket_v1 --episode 1 \
        --task bottle_basket --out migration/digital_twin_2026-10-29/judge/runs/bb_v1_ep1.json
    uv run python -m lerobot.rewards.vlm_judge.score_episode --root outputs/rebot_teleop_2026-10-29-cut-v1 --episode 0 \
        --task "basket:pill bottle" --out ... [--t-from 0 --t-to 60]
    # a whole sim run: task = sim:<layout> (sim_spec.py, from tasks.py), compared to the scorer per episode, resumable
    uv run python -m lerobot.rewards.vlm_judge.score_episode --sim-run outputs/twin_eval_res600_2026-10-30/validation/step_00000000/sim \
        --episodes 0 --out-dir migration/digital_twin_2026-10-29/judge/runs/gemma_res600 --workers 4

Frames are sampled every ``--period`` seconds (1.0); every ``--stride`` samples (default = frames_per_camera, i.e. every 5 s of
video) one query goes out with the last ``frames_per_camera`` frames of each camera, so the timeline has one row per window: t,
subtask, stage, progress, progress_max (running max), confidence, evidence, latency. The model is whatever ``--endpoint`` serves
(vllm on the Spark, or the mock server for a dry run). ``--task`` defaults to ``sim:<layout>`` for ``--sim-dir``."""

import argparse
import json
import time
from pathlib import Path

from .batch import run_batch, sim_episodes, write_table
from .client import JudgeClient
from .compare_to_scorer import compare
from .config import JudgeConfig
from .frames import decode_dataset_episode, decode_sim_episode
from .prompt import Frame
from .task_spec import TaskSpec


def score_samples(client: JudgeClient, spec: TaskSpec, samples: list, cameras: list[str], frames_per_camera: int, stride: int,
                  max_queries: int | None = None, verbose: bool = True, tag: str = "") -> tuple[list[dict], int]:
    rows, progress_max, failures = [], 0.0, 0
    for i, (t, _) in enumerate(samples):
        if (i + 1) % stride and i != len(samples) - 1:
            continue
        if max_queries is not None and len(rows) + failures >= max_queries:
            break
        window = samples[max(0, i - frames_per_camera + 1): i + 1]
        frames = [Frame(t=tw, camera=cam, image=imgs[cam]) for tw, imgs in window for cam in cameras]
        try:
            judgement, meta = client.query(spec, frames)
        except Exception as e:
            failures += 1
            print(f"{tag}t={t:7.2f}  FAILED {type(e).__name__}: {str(e)[:120]}")
            continue
        progress_max = max(progress_max, judgement.progress)
        rows.append(dict(t=t, **judgement.to_dict(), progress_max=progress_max, latency_s=meta["latency_s"]))
        if verbose:
            print(f"t={t:7.2f}  sub {judgement.subtask_id:2d} {judgement.subtask:38s} stage {judgement.stage_id} {judgement.stage:16s} "
                  f"p={judgement.progress:.2f} conf={judgement.confidence:.2f} {meta['latency_s']:.1f}s  {judgement.evidence[:60]}")
    return rows, failures


def write_timeline(out: Path, source, episode: int, spec: TaskSpec, cfg: JudgeConfig, args_like: dict, rows: list[dict], failures: int) -> dict:
    timeline = dict(source=str(source), episode=episode, task=spec.name, task_text=spec.task, model=cfg.model, endpoint=cfg.endpoint,
                    period_s=cfg.frame_period_s, frames_per_camera=cfg.frames_per_camera, cameras=cfg.cameras, **args_like,
                    created=time.strftime("%Y-%m-%d %H:%M:%S"), failures=failures, steps=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(timeline, indent=1))
    return timeline


def score_sim_episode(cfg: JudgeConfig, sim_dir: Path, episode: int, out: Path, stride: int, max_queries: int | None) -> dict:
    """Batch path: task sim:<layout>, timeline + compare_to_scorer metrics (flattened for the summary table)."""
    spec = TaskSpec.load(f"sim:{sim_dir.name}")
    samples = decode_sim_episode(sim_dir, episode, cfg.cameras, cfg.frame_period_s)
    rows, failures = score_samples(JudgeClient(cfg), spec, samples, cfg.cameras, cfg.frames_per_camera, stride, max_queries,
                                   verbose=False, tag=f"{sim_dir.name} ep{episode} ")
    timeline = write_timeline(out, sim_dir, episode, spec, cfg, {}, rows, failures)
    if not rows:
        return dict(n=0, failures=failures)
    record = next(r for r in json.loads((sim_dir / "subtask_events.json").read_text()) if r["episode"] == episode)
    m = compare(timeline, record, spec)
    lags = [v["lag_s"] for v in m["stage_lag_s"].values() if v["lag_s"] is not None]
    final_ok = m["judge_final"]["stage"] == m["scorer_final"]["stage"] or (m["scorer_final"]["stage"] is None and m["judge_final"]["stage"] == "none")
    return dict(n=m["n_steps"], failures=failures, cursor_accuracy=m["cursor_accuracy"], stage_accuracy_running_max=m["stage_accuracy_running_max"],
                spearman_progress_max=m["spearman_progress_max"], stage_lag_mean_s=round(sum(lags) / len(lags), 2) if lags else None,
                final_stage_match=float(final_ok), judge_progress_max=m["judge_final"]["progress_max"],
                scorer_reward=m["scorer_final"]["reward"], scorer_success=float(m["scorer_final"]["success"]),
                reward_dense=record.get("reward_dense"), stages_judge_only=m["stages_judge_only"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim-dir", type=Path, help="sim eval layout folder with videos/ep<N>_<cam>.mp4")
    src.add_argument("--root", type=Path, help="LeRobot dataset root")
    src.add_argument("--sim-run", type=Path, help="a sim folder holding layout folders (batch, with --out-dir)")
    ap.add_argument("--episode", type=int)
    ap.add_argument("--task", default=None, help="tasks/<name>.json stem, '<stem>:<object>' for a template, or sim:<layout>")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--layouts", default="*", help="glob over layout folder names (--sim-run)")
    ap.add_argument("--episodes", type=int, nargs="+", default=None, help="episode indices per layout (--sim-run); default all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--endpoint", default=JudgeConfig.endpoint)
    ap.add_argument("--model", default=JudgeConfig.model)
    ap.add_argument("--period", type=float, default=1.0, help="frame spacing in the strip (s)")
    ap.add_argument("--frames-per-camera", type=int, default=5)
    ap.add_argument("--stride", type=int, default=None, help="samples between queries; default = frames-per-camera, so with period 1 s and 5 frames one query covers each 5 s window (offline scoring, user 2026-10-30)")
    ap.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    ap.add_argument("--t-from", type=float, default=0.0)
    ap.add_argument("--t-to", type=float, default=float("inf"))
    ap.add_argument("--max-queries", type=int, default=None, help="stop after this many queries (smoke tests)")
    args = ap.parse_args()

    cfg = JudgeConfig(endpoint=args.endpoint, model=args.model, cameras=args.cameras, frames_per_camera=args.frames_per_camera, frame_period_s=args.period)
    stride = args.stride or args.frames_per_camera
    if args.sim_run is not None:
        jobs = sim_episodes(args.sim_run, args.layouts, args.episodes)
        print(f"{len(jobs)} episodes, {args.workers} in flight, model {cfg.model} at {cfg.endpoint}")
        rows = run_batch(jobs, args.out_dir, "judge", lambda d, ep, out: score_sim_episode(cfg, d, ep, out, stride, args.max_queries), args.workers)
        write_table(rows, ["cursor_accuracy", "stage_accuracy_running_max", "final_stage_match", "spearman_progress_max", "stage_lag_mean_s",
                           "judge_progress_max", "reward_dense", "scorer_success"],
                    args.out_dir / "summary.md", f"Judge {cfg.model} on {args.sim_run} (period {args.period} s, {args.frames_per_camera} frames/camera, stride {stride})")
        return

    spec = TaskSpec.load(args.task or f"sim:{args.sim_dir.name}")
    if args.sim_dir is not None:
        samples = decode_sim_episode(args.sim_dir, args.episode, args.cameras, args.period)
    else:
        samples = decode_dataset_episode(args.root, args.episode, args.cameras, args.period)
    samples = [s for s in samples if args.t_from <= s[0] <= args.t_to]
    print(f"{len(samples)} samples at {args.period} s, {spec.name}: {spec.task}")
    rows, failures = score_samples(JudgeClient(cfg), spec, samples, args.cameras, args.frames_per_camera, stride, args.max_queries)
    write_timeline(args.out, args.sim_dir or args.root, args.episode, spec, cfg, {}, rows, failures)
    print(f"wrote {args.out} ({len(rows)} rows, {failures} failures)")


if __name__ == "__main__":
    main()

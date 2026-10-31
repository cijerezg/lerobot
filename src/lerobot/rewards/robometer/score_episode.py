"""Robometer over one recorded episode: every ``--period`` seconds the last ``--max-frames`` top-camera frames go to
the 8012 service; the timeline keeps the last frame's progress and success (what upstream compute_reward returns) and
the whole per-frame vector of that query.

    uv run python -m lerobot.rewards.robometer.score_episode --sim-dir <...>/sim/bottle_basket_v1 --episode 1 \
        --task bottle_basket --out migration/digital_twin_2026-10-29/judge/runs/bb_v1_ep1.robometer.json
    uv run python -m lerobot.rewards.robometer.score_episode --root outputs/rebot_teleop_2026-10-29-cut-v1 --episode 0 \
        --task "basket:pill bottle" --out ...

``--task`` is a vlm_judge task spec (its task string is the prompt) or free text with ``--task-text``. The timeline
has the judge timeline's shape (t, progress, progress_max) so compare_to_scorer's Spearman applies to it too."""

import argparse
import json
import time
from pathlib import Path

from lerobot.rewards.vlm_judge.frames import decode_dataset_episode, decode_sim_episode
from lerobot.rewards.vlm_judge.task_spec import TaskSpec

from .client import RobometerClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim-dir", type=Path)
    src.add_argument("--root", type=Path)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--task", default=None, help="vlm_judge task spec name")
    ap.add_argument("--task-text", default=None, help="free task string instead of --task")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--endpoint", default="http://localhost:8012")
    ap.add_argument("--camera", default="top")
    ap.add_argument("--period", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--t-from", type=float, default=0.0)
    ap.add_argument("--t-to", type=float, default=float("inf"))
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()
    task_text = args.task_text if args.task_text is not None else TaskSpec.load(args.task).task

    client = RobometerClient(args.endpoint)
    if args.sim_dir is not None:
        samples = decode_sim_episode(args.sim_dir, args.episode, [args.camera], args.period)
    else:
        samples = decode_dataset_episode(args.root, args.episode, [args.camera], args.period)
    samples = [s for s in samples if args.t_from <= s[0] <= args.t_to]
    print(f"{len(samples)} samples at {args.period} s, task: {task_text}")

    rows, progress_max, failures = [], 0.0, 0
    for i, (t, _) in enumerate(samples):
        if args.max_queries is not None and len(rows) + failures >= args.max_queries:
            break
        frames = [imgs[args.camera] for _, imgs in samples[max(0, i - args.max_frames + 1): i + 1]]
        try:
            body, latency = client.score(task_text, frames)
        except Exception as e:
            failures += 1
            print(f"t={t:7.2f}  FAILED {type(e).__name__}: {str(e)[:120]}")
            continue
        progress, success = float(body["progress"][-1]), float(body["success"][-1])
        progress_max = max(progress_max, progress)
        rows.append(dict(t=t, progress=progress, progress_max=progress_max, success=success, n_frames=len(frames),
                         progress_per_frame=body["progress"], success_per_frame=body["success"], latency_s=latency))
        print(f"t={t:7.2f}  progress={progress:.3f} (max {progress_max:.3f})  success={success:.3f}  {latency:.2f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        source=str(args.sim_dir or args.root), episode=args.episode, task=args.task, task_text=task_text, model="lerobot/Robometer-4B",
        endpoint=args.endpoint, period_s=args.period, max_frames=args.max_frames, camera=args.camera,
        created=time.strftime("%Y-%m-%d %H:%M:%S"), failures=failures, steps=rows,
    ), indent=1))
    print(f"wrote {args.out} ({len(rows)} rows, {failures} failures)")


if __name__ == "__main__":
    main()

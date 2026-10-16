"""Review and cut recorded inference episodes (a run's inference_dataset) into a training root.

    # 1. one review mp4 per episode: top | wrist, 2x, burned-in t = REAL episode seconds, red INTERVENTION tag
    uv run python -m lerobot.annotation.rebot.rollout_review render outputs/train/<date>/<run> [<run> ...]
    # 2. mark the spans to REMOVE (or the spans to KEEP) in episode seconds: 15-31, 1:23-1:45, "50-" = to the end
    uv run python -m lerobot.annotation.rebot.rollout_review mark outputs/train/<date>/<run> --remove 15-31,36-43,50-
    uv run python -m lerobot.annotation.rebot.rollout_review mark outputs/train/<date>/<run> --keep 0-25,42-1:23,1:30-1:45 --episode 1
    # 3. build the root from the plan (build_root --plan), then verify_root
    uv run python -m lerobot.annotation.rebot.rollout_review build migration/rollout_review_<date> --out outputs/rebot_rollouts_<date>-cut-v1

Work folder (videos/, plan.json, build.log) = migration/rollout_review_<date> with <date> from the run path
outputs/train/<date>/<run>; --work overrides. Boundaries snap to multiples of 3 frames so the depth phase survives.
The kept spans of an episode are SPLICED into one episode (build_root re-encodes both cameras at a mid-episode cut);
``mark`` replaces the plan entry of that run + episode. ``task`` defaults to the string the operator typed.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot import build_root, verify_root

FPS = 30


def work_dir(run, work):
    return WORKSPACE / (work or f"migration/rollout_review_{Path(run).resolve().parent.name}")


def dataset(run):
    return Path(run).resolve() / "inference_dataset"


def episodes(root):
    return pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))], ignore_index=True)


def intervention_spans(root, episode):
    """[(start s, end s)] of operator teleop inside an episode."""
    o = pd.read_parquet(root / "meta/online_labels.parquet")
    iv = o[o.episode_index == episode].sort_values("frame_index").is_intervention.to_numpy().astype(int)
    pad = np.r_[0, iv, 0]
    return [(s / FPS, e / FPS) for s, e in zip(np.flatnonzero(np.diff(pad) == 1), np.flatnonzero(np.diff(pad) == -1))]


def video(root, e, cam):
    pre = f"videos/observation.images.{cam}"
    return root / pre / f"chunk-{int(e[pre + '/chunk_index']):03d}" / f"file-{int(e[pre + '/file_index']):03d}.mp4"


def render(runs, work):
    out = work_dir(runs[0], work) / "videos"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for run in runs:
        root, name = dataset(run), Path(run).resolve().name[:8]
        for _, e in episodes(root).iterrows():
            ep, n = int(e.episode_index), int(e.length)
            spans = intervention_spans(root, ep)
            label = f"{name} ep{ep}"
            enable = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in spans) or "0"
            fc = (
                f"[0:v]setpts=PTS-STARTPTS,drawtext=text='{label}   t=%{{pts\\:hms}}':x=10:y=10:fontsize=28:fontcolor=yellow"
                f":box=1:boxcolor=black@0.6,drawtext=text='INTERVENTION':x=10:y=50:fontsize=28:fontcolor=red:box=1"
                f":boxcolor=black@0.6:enable='{enable}'[a];[1:v]setpts=PTS-STARTPTS[b];"
                f"[a][b]hstack=inputs=2,setpts=0.5*PTS,fps={FPS}[v]"
            )
            cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
            for cam in ("top", "wrist"):
                pre = f"videos/observation.images.{cam}"
                cmd += ["-ss", f"{e[pre + '/from_timestamp']:.4f}", "-t", f"{n / FPS:.4f}", "-i", str(video(root, e, cam))]
            target = out / f"{name}_ep{ep}.mp4"
            cmd += ["-filter_complex", fc, "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                    "-pix_fmt", "yuv420p", str(target)]
            jobs.append((subprocess.Popen(cmd), target))
            print(f"{label}: {n} frames, {n / FPS:.1f} s, interventions "
                  + (", ".join(f"{a:.1f}-{b:.1f}" for a, b in spans) or "none"), flush=True)
    for p, target in jobs:
        assert p.wait() == 0, target
        print("wrote", target.relative_to(WORKSPACE), flush=True)


def seconds(text):
    """'83', '1:23', '1:23.5' -> seconds."""
    parts = text.split(":")
    return float(parts[-1]) + 60 * (int(parts[-2]) if len(parts) > 1 else 0)


def snap(s, n):
    """Episode seconds -> frame, a multiple of 3 inside [0, n - n % 3]."""
    return int(min(max(3 * round(s * FPS / 3), 0), n - n % 3))


def spans(text, n):
    """'15-31,36-43,50-' -> [(a, b)) frames, sorted, non-overlapping; an open end runs to the episode end."""
    out = []
    for item in text.split(","):
        a, b = item.split("-")
        out.append((snap(seconds(a), n), snap(seconds(b), n) if b else n - n % 3))
    out.sort()
    assert all(a < b for a, b in out) and all(out[i][1] <= out[i + 1][0] for i in range(len(out) - 1)), out
    return out


def mark(run, episode, remove, keep, why, task, work):
    assert (remove is None) != (keep is None), "give --remove or --keep"
    root = dataset(run)
    e = episodes(root).set_index("episode_index").loc[episode]
    n = int(e.length)
    if keep is not None:
        kept = spans(keep, n)
    else:
        kept, cursor = [], 0
        for a, b in spans(remove, n) + [(n - n % 3, n - n % 3)]:
            if a > cursor:
                kept.append((cursor, a))
            cursor = b
    assert kept, "nothing kept"
    cuts = [[kept[i][1], kept[i + 1][0]] for i in range(len(kept) - 1)]
    entry = dict(
        source=str(root.relative_to(WORKSPACE)),
        episode=episode,
        keep=[kept[0][0], kept[-1][1]],
        cuts=cuts,
        task=task or list(e.tasks)[0],
        run=Path(run).resolve().name,
        recorded_date=Path(run).resolve().parent.name,
        marked=f"--keep {keep}" if keep is not None else f"--remove {remove}",
        why=why,
    )
    path = work_dir(run, work) / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = json.loads(path.read_text()) if path.exists() else []
    plan = [p for p in plan if (p["source"], p["episode"]) != (entry["source"], entry["episode"])] + [entry]
    path.write_text(json.dumps(plan, indent=1))
    total = sum(b - a for a, b in kept)
    print(f"{entry['run']} ep{episode}: keep " + ", ".join(f"{a / FPS:.0f}-{b / FPS:.0f} s" for a, b in kept)
          + f" = {total} frames ({total / FPS:.1f} s) of {n}; {len(cuts)} splice(s); plan {path.relative_to(WORKSPACE)}"
          f" now {len(plan)} episode(s)", flush=True)


def build(work, out):
    work, out = WORKSPACE / work, WORKSPACE / out
    build_root.build(build_root.rows_from_plan(work), out)
    verify_root.check(out)
    plan = json.loads((work / "plan.json").read_text())
    rows = "\n".join(
        f"| {i} | {p['run']} | {p['episode']} | {p['keep']} | {p['cuts']} | {p['marked']} | {p.get('why') or ''} |"
        for i, p in enumerate(plan)
    )
    (out / "README.md").write_text(
        f"# {out.name}\n\nPolicy rollouts cut by the user's review (`lerobot.annotation.rebot.rollout_review`), NOT yet annotated."
        f" Sources untouched under outputs/train. Plan: `{plan and (work / 'plan.json').relative_to(WORKSPACE)}`;"
        f" verify_root PASS -> meta/cleanup_verification.json. `meta/tasks.parquet` = operator-typed task strings, to revise"
        f" during annotation. `meta/online_labels.parquet` keeps is_intervention + the typed `recorded_subtask`.\n\n"
        f"| ep | source run | src ep | kept (source frames) | spliced out | marked | why |\n|---|---|---|---|---|---|---|\n{rows}\n"
    )
    print("README", out / "README.md", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("runs", nargs="+")
    r.add_argument("--work")
    m = sub.add_parser("mark")
    m.add_argument("run")
    m.add_argument("--episode", type=int, default=0)
    m.add_argument("--remove")
    m.add_argument("--keep")
    m.add_argument("--why")
    m.add_argument("--task")
    m.add_argument("--work")
    b = sub.add_parser("build")
    b.add_argument("work")
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "render":
        render(args.runs, args.work)
    elif args.cmd == "mark":
        mark(args.run, args.episode, args.remove, args.keep, args.why, args.task, args.work)
    else:
        build(args.work, args.out)


if __name__ == "__main__":
    main()

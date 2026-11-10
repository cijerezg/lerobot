"""Write what the labeller reads for every frame of a correction pass.

  uv run python -m lerobot.annotation.corrections.dump <work> [--source NAME ...] [--episode N ...]

Reads ``<work>/frames.parquet`` and ``stalls.json`` (``sample``). Writes, per episode,
  <work>/<source>/ep<NNN>/episode.txt        task, stall angles, the episode's frames in order
  <work>/<source>/ep<NNN>/f<NNNNNN>/         case.txt, wrist_a.png, wrist_b.png (the wrist clip, 6 tiles each),
                                             top.png (-5, 0, +1, +2 s; on the t = 0 tile of an episode that has a
                                             fitted top camera: red circle = tool tip, magenta = tip path of the
                                             recorded chunk)
and for the pass ``<work>/jobs.txt``: one labelling job per line, an episode folder and its frame folders, at most
JOB frames each (a long episode is several jobs, to be run in order).
The labeller adds ``waypoints.json`` to the frame folder. A frame folder that has its ``case.txt`` (written last)
is skipped, so an interrupted dump resumes.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lerobot.annotation.corrections.arm import FPS, Arm, project
from lerobot.annotation.corrections.source import Episode, sources

CHUNK = 30
TIMES = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]  # seconds from the chunk start
TOP_TIMES = [-5.0, 0.0, 1.0, 2.0]
SIDE = 378
JOB = 20
DEPTH_U, DEPTH_V = [64, 192, 320, 448, 576], [60, 180, 300, 420]  # wrist-image pixels of the depth readings


def kind_text(kind: str, grade: int) -> str:
    return f"demo, grade {grade}" if kind == "demo" else kind


def tile(rgb: np.ndarray, text: str) -> np.ndarray:
    small = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    cv2.putText(small, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(small, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return small


def depth_grid(depth: np.ndarray) -> list[str]:
    """The wrist depth image as a 5 x 4 table of readings in cm: the median valid reading (under 1 m) of the
    13 x 13 patch at each pixel, "-" where the patch has none."""
    lines = [
        "wrist depth at t = 0, cm from the camera along its optical axis, at wrist-image pixels (640 x 480; inside a "
        "wrist tile u x 0.59, v x 0.79); - = no reading",
        "v \\ u " + " ".join(f"{u:5d}" for u in DEPTH_U),
    ]
    for v in DEPTH_V:
        patches = [depth[v - 6 : v + 7, u - 6 : u + 7] for u in DEPTH_U]
        valid = [p[(p > 0) & (p < 1.0)] for p in patches]
        lines.append(f"{v:5d} " + " ".join(f"{100 * np.median(p):5.1f}" if len(p) else "    -" for p in valid))
    return lines


def dump_frame(arm: Arm, e: Episode, row, stalls: str, folder: Path) -> None:
    t = row.frame
    frames = [int(np.clip(t + round(s * FPS), 0, len(e.states) - 1)) for s in TIMES]
    folder.mkdir(parents=True, exist_ok=True)

    wrist = [tile(img, f"t={(f - t) / FPS:+.1f}s") for f, img in zip(frames, e.images("wrist", frames), strict=True)]
    cv2.imwrite(str(folder / "wrist_a.png"), np.vstack([np.hstack(wrist[0:3]), np.hstack(wrist[3:6])])[:, :, ::-1])
    cv2.imwrite(str(folder / "wrist_b.png"), np.vstack([np.hstack(wrist[6:9]), np.hstack(wrist[9:12])])[:, :, ::-1])
    top_frames = [int(np.clip(t + round(s * FPS), 0, len(e.states) - 1)) for s in TOP_TIMES]
    tops = e.images("top", top_frames)
    if e.top is not None:
        now = tops[1] = tops[1].copy()
        recorded = [project(e.top, arm.gripper.fk(a)[0]) for a in e.actions[t : t + CHUNK : 3]]
        cv2.polylines(now, [np.int32(np.round(recorded))], False, (255, 0, 255), 2)
        cv2.circle(now, tuple(np.int32(np.round(project(e.top, arm.gripper.fk(e.states[t])[0])))), 8, (255, 0, 0), 2)
    tops = [tile(img, f"t={(f - t) / FPS:+.1f}s") for f, img in zip(top_frames, tops, strict=True)]
    cv2.imwrite(str(folder / "top.png"), np.hstack(tops)[:, :, ::-1])

    tip, approach, jaw = arm.gripper.fk(e.states[t])
    lines = [
        f"case {e.source.name}/ep{e.index:03d}/f{t:06d} | task: {e.task} | subtask: {row.subtask} | contact: {row.contact}"
        f" | precision: {row.precision} | frame kind: {kind_text(row.kind, row.grade)}",
        f"chunk start = frame {t} of the episode, {FPS} fps; the recorded chunk is the {CHUNK} actions from t = 0 to 1.0 s",
        f"tool at t = 0: tip {tip.round(3).tolist()} approach {approach.round(2).tolist()} jaw {jaw.round(2).tolist()} gripper {e.states[t][6]:.0f}",
        stalls,
        "",
        "time_s frame | measured: tip xyz m, approach, gripper deg | recorded action: tip xyz m, gripper deg"
        " (| operator = a human was in control at that frame)",
    ]
    for f in frames:
        tip, approach, _ = arm.gripper.fk(e.states[f])
        lines.append(
            f"{(f - t) / FPS:+.1f} {f} | {tip.round(3).tolist()} {approach.round(2).tolist()} {e.states[f][6]:.0f}"
            f" | {arm.gripper.fk(e.actions[f])[0].round(3).tolist()} {e.actions[f][6]:.0f}" + (" | operator" if e.kinds[f] == "operator" else "")
        )
    depth = e.depth_m(t)
    if depth is not None:
        lines += ["", *depth_grid(depth)]
    (folder / "case.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--episode", type=int, nargs="+")
    args = parser.parse_args()
    frames = pd.read_parquet(args.work / "frames.parquet")
    jobs = []
    for (name, index), rows in frames.groupby(["source", "episode"], sort=False):
        for part in np.array_split(rows.frame.to_numpy(), -(-len(rows) // JOB)):
            jobs.append(f"{name}/ep{index:03d} f{part[0]:06d}..f{part[-1]:06d} {len(part)} frames")
    (args.work / "jobs.txt").write_text("\n".join(jobs) + "\n")
    if args.source is not None:
        frames = frames[frames.source.isin(args.source)]
    if args.episode is not None:
        frames = frames[frames.episode.isin(args.episode)]
    stalls = json.loads((args.work / "stalls.json").read_text())
    arm = Arm()
    for name, source in sources(json.loads((args.work / "pass.json").read_text()), list(frames.source.unique())).items():
        for index, rows in frames[frames.source == name].groupby("episode"):
            e = Episode(source, index)
            folder = args.work / name / f"ep{index:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            stall_line = "gripper stalls on a held object in the pass's recordings (measured deg): " + (
                "; ".join(f"{s}: {stalls[s]}" for s in dict.fromkeys(e.subtasks) if s in stalls) or "none recorded"
            )
            lines = [
                f"task: {e.task}",
                f"{name} episode {index}: {len(e.states)} frames at {FPS} fps, {len(rows)} frames to label",
                stall_line,
                "",
                "folder | time_s | subtask | contact | frame kind",
                *(f"f{r.frame:06d} | {r.frame / FPS:.1f} | {r.subtask} | {r.contact} | {kind_text(r.kind, r.grade)}" for r in rows.itertuples()),
            ]
            (folder / "episode.txt").write_text("\n".join(lines) + "\n")
            for row in rows.itertuples():
                if not (folder / f"f{row.frame:06d}" / "case.txt").exists():
                    dump_frame(arm, e, row, stall_line, folder / f"f{row.frame:06d}")
            print("wrote", folder, len(rows), "frames")


if __name__ == "__main__":
    main()

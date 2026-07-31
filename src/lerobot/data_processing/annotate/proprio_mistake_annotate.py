#!/usr/bin/env python

"""
Failed-grasp mistake metadata from proprioception alone — no video model needed.

Writes the same four files as `metadata_annotate.py review`'s "Write to dataset"
button (meta/mistake_candidates.json, meta/metadata_review_state.json,
meta/episode_metadata.parquet, meta/mistakes.parquet, meta/metadata_info.json),
so the review UI can be launched afterwards to adjust anything by hand.

Rule
----
Take each gripper-closed interval (hysteresis on the gripper joint). The arm's
maximum L2 displacement in the six arm joints from the pose it closed in is the
discriminator: if the arm never left that pose, nothing was picked up.

Calibrated on the human-reviewed rebot_sorting_clothes_v1 and v3-1:
**22 TP / 0 FP / 3 FN** over their 25 labelled events at `--disp-max 32`. The
misses are all semantic classes — a long-travel failed grasp, a wrong-destination
carry, an unintended release — which proprio cannot separate. Displacement beats
pan travel alone: in the shirt datasets a garment is moved in several short
carries, and pan-only flags those legitimate drags as failures.

Quality 1-5 is *derived* from the mistake count, not human-scored (matches the
human scores exactly on 12/15 labelled episodes, the rest off by one).

    python proprio_mistake_annotate.py --data-dir outputs/rebot_sorting_clothes_v3-2
    python proprio_mistake_annotate.py --data-dir DS --dry-run   # print, write nothing
"""

import argparse
import glob
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

GRIPPER_DIM = 6
PAN_DIM = 0
CLOSE_ON = -60.0      # gripper above this -> closed
CLOSE_OFF = -90.0     # gripper below this -> open
# Two guards against low-displacement intervals that are not grasp attempts. Margins
# are wide: across the 22 labelled true positives the shortest is 19 frames and the
# closest to an episode edge is 477 frames, so neither guard drops a known event.
MIN_CLOSED_FRAMES = 15   # a sub-0.5 s blip mid-transit is not an attempt
BOUNDARY_MARGIN = 30     # arm parked with the gripper closed at episode start/end
WINDOW = 30           # 1 s at 30 fps, episode-aligned
SCORE_POSITIVE = 8
THRESHOLD = 4

DEFINITION = (
    "Concrete failed grasp, slip/drop, collision/knock, wrong-object or wrong-destination "
    "motion, or clearly unintended release. Slow successful positioning and clean recovery "
    "are negative."
)
LIMITATION = (
    "Recall covers the failed-grasp class only. Wrong-destination carries and unintended "
    "releases are not separable from the gripper/pan signal and were not screened. Quality "
    "1-5 is derived from the mistake count, not scored by a human."
)


def load_dataset(root: Path):
    data = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))]
    )
    state = np.stack(data["observation.state"].values).astype(np.float32)
    eps = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))]
    ).sort_values("episode_index")
    bounds = [
        (int(a), int(b), int(c))
        for a, b, c in zip(
            eps["episode_index"], eps["dataset_from_index"], eps["dataset_to_index"], strict=True
        )
    ]
    return state, bounds


def closed_intervals(grip):
    """Hysteresis threshold -> half-open (start, stop) runs where the gripper is closed."""
    runs, start, closed = [], None, False
    for i, g in enumerate(grip):
        closed = g > CLOSE_ON if not closed else g >= CLOSE_OFF
        if closed and start is None:
            start = i
        elif not closed and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(grip)))
    return [r for r in runs if r[1] - r[0] >= MIN_CLOSED_FRAMES]


def detect_events(state, bounds, disp_max, fps):
    grip, arm = state[:, GRIPPER_DIM], state[:, :GRIPPER_DIM]
    events = []
    for ep, lo, hi in bounds:
        for s, e in closed_intervals(grip[lo:hi]):
            gs, ge = lo + s, lo + e
            if gs - lo < BOUNDARY_MARGIN or hi - ge < BOUNDARY_MARGIN:
                continue
            seg = arm[gs:ge]
            disp = float(np.linalg.norm(seg - seg[0], axis=1).max())
            if disp > disp_max:
                continue
            pan = state[gs:ge, PAN_DIM]
            events.append(
                {
                    "episode_index": ep,
                    "start_seconds": round((gs - lo) / fps, 3),
                    "stop_seconds": round((ge - lo) / fps, 3),
                    "from_index": gs,
                    "to_index": ge,
                    "event_type": "failed_grasp",
                    "evidence": (
                        f"Gripper closed for {ge - gs} frames at shoulder_pan {pan[0]:.1f} deg and "
                        f"reopened after {disp:.1f} deg of total arm displacement "
                        f"({float(pan.max() - pan.min()):.1f} deg of pan): the arm never left the "
                        "pose it closed in, so nothing was picked up."
                    ),
                    "confidence": "high" if disp <= 20 else "medium-high",
                    "independently_verified": False,
                }
            )
    return events


def subtask_spans(root: Path):
    path = root / "meta" / "subtask_windows.json"
    if not path.exists():
        return []
    episodes = json.load(open(path))["episodes"]
    spans = sorted(
        (int(w["from_index"]), int(w["to_index"]), str(w["subtask"]))
        for wins in episodes.values()
        for w in wins
    )
    return spans


def label_at(spans, index):
    for a, b, s in spans:
        if a <= index < b:
            return s
    return ""


def build(root: Path, disp_max: float):
    info = json.load(open(root / "meta" / "info.json"))
    fps = float(info["fps"])
    state, bounds = load_dataset(root)
    events = detect_events(state, bounds, disp_max, fps)
    spans = subtask_spans(root)

    grid, n_positive = {}, 0
    for ep, lo, hi in bounds:
        rows = []
        for start in range(lo, hi, WINDOW):
            stop = min(start + WINDOW, hi)
            hit = next((e for e in events if e["from_index"] < stop and e["to_index"] > start), None)
            rows.append(
                {
                    "from_index": start,
                    "to_index": stop,
                    "subtask": label_at(spans, start),
                    "score": SCORE_POSITIVE if hit else 0,
                    "evidence": hit["evidence"] if hit else "",
                    "confidence": hit["confidence"] if hit else "high",
                }
            )
            n_positive += bool(hit)
        grid[str(ep)] = rows

    counts = {ep: sum(1 for e in events if e["episode_index"] == ep) for ep, _, _ in bounds}
    quality = {ep: 5 if n == 0 else (4 if n <= 2 else 3) for ep, n in counts.items()}

    candidates = {
        "model": "proprio-grasp-cycle-detector",
        "annotator": f"proprio_mistake_annotate.py --disp-max {disp_max}",
        "created_date": date.today().isoformat(),
        "interval_seconds": WINDOW / fps,
        "fps": fps,
        "grid": "episode-aligned half-open 30-frame windows; final tail may be shorter",
        "threshold": THRESHOLD,
        "positive_definition": DEFINITION,
        "review_method": (
            f"Gripper-closed intervals with max arm-joint displacement <= {disp_max} deg from the "
            "closing pose. Calibrated on rebot_sorting_clothes_v1 and v3-1 (22 TP / 0 FP / 3 FN)."
        ),
        "limitations": LIMITATION,
        "n_events": len(events),
        "n_windows": sum(len(v) for v in grid.values()),
        "n_positive_windows": n_positive,
        "events": events,
        "episodes": grid,
    }
    episode_rows = [
        {"episode_index": ep, "quality": int(quality[ep]), "from_index": lo, "to_index": hi}
        for ep, lo, hi in bounds
    ]
    mistake_rows = [
        {
            "episode_index": int(ep),
            "from_index": w["from_index"],
            "to_index": w["to_index"],
            "mistake": w["score"] >= THRESHOLD,
        }
        for ep in sorted(grid, key=int)
        for w in grid[ep]
    ]
    review_state = {
        "quality": {str(ep): int(q) for ep, q in quality.items()},
        "windows": {
            f"{r['episode_index']}:{r['from_index']}:{r['to_index']}": True
            for r in mistake_rows
            if r["mistake"]
        },
    }
    meta_info = {
        "quality_by_episode": {str(r["episode_index"]): r["quality"] for r in episode_rows},
        "n_windows": len(mistake_rows),
        "n_flagged": n_positive,
        "n_mistakes": sum(r["mistake"] for r in mistake_rows),
        "speed": "omitted",
        "mistake_candidates": {k: v for k, v in candidates.items() if k != "episodes"},
    }
    return candidates, review_state, episode_rows, mistake_rows, meta_info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--disp-max", type=float, default=32.0,
                        help="Max arm-joint displacement (deg) for a closed interval to count as a failed grasp")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir)
    candidates, review_state, episode_rows, mistake_rows, meta_info = build(root, args.disp_max)

    print(f"{root.name}: {candidates['n_events']} events, "
          f"{meta_info['n_mistakes']}/{meta_info['n_windows']} windows flagged")
    for e in candidates["events"]:
        print(f"  ep{e['episode_index']} [{e['from_index']}:{e['to_index']}] "
              f"t={e['start_seconds']:.1f}s ({e['confidence']})")
    print(f"  quality: {meta_info['quality_by_episode']}")
    if not candidates["events"]:
        print("  NOTE: no events found — check the gripper convention before trusting this.")
    if args.dry_run:
        print("  dry run — nothing written")
        return

    meta = root / "meta"
    json.dump(candidates, open(meta / "mistake_candidates.json", "w"), indent=2)
    json.dump(review_state, open(meta / "metadata_review_state.json", "w"), indent=2)
    json.dump(meta_info, open(meta / "metadata_info.json", "w"), indent=2)
    pd.DataFrame(episode_rows).to_parquet(meta / "episode_metadata.parquet", engine="pyarrow")
    pd.DataFrame(mistake_rows).to_parquet(meta / "mistakes.parquet", engine="pyarrow")
    print(f"  wrote 5 files to {meta}")


if __name__ == "__main__":
    main()

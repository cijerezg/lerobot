#!/usr/bin/env python

"""
Semantic subtask segmentation from proprioception — boundaries at the events that
actually define the phase, not on a fixed grid.

The 4 s grid quantizes every boundary to +-4 s and systematically mislabels: in
rebot_sorting_clothes_v1 ep0 the gripper closes on the sock at frame 450 but the
grid calls "move" from 360, and frames 720-1000 (arm travelling back from the
basket, gripper open) are labelled "grasp".

Phase model — one pick-and-place cycle is a gripper-closed interval:

    grasp the X          [end of previous release, close)  approach + closing
    move the X to the C  [close, arrival at C)             gripper shut, arm travels
    release the X in C   [arrival at C, opening settles)   deposit: position, then open
    return to home       [last release, episode end)       only the final park

Two boundary choices are judgement, both anchored on measured events:

* `grasp` absorbs the travel back from the previous container. Tested for a
  retreat/approach turnaround and there is none — the smoothed joint-speed minimum
  lands on the pre-grasp settle or at the segment edge, never mid-transit. Container
  to next object is one continuous sweep, so there is no boundary to split at.
* `release` starts at arrival over the container, not at the gripper opening. The
  mechanical opening is ~0.6 s (2% of frames), too tight to condition on; the deposit
  as a unit of intent is 2.6 s median (13% of frames).

`return to home` stays reserved for the final park, so the vocabulary is unchanged
(28 labels across the corpus) and no remap is needed.

Failed grasps do NOT open a cycle — a closed interval whose arm displacement stays
under --disp-max never picked anything up, so it remains inside the enclosing
"grasp the X" segment. That is exactly where the mistake span belongs.

Object identity needs vision, so it is carried over from the existing 4 s labels by
majority vote across the carry. Datasets with no prior annotation get "<object>"
placeholders for a manual pass.

    python semantic_segment.py --data-dir outputs/rebot_sorting_clothes_v1
    python semantic_segment.py --data-dir DS --write     # replace meta/subtask_windows.json
"""

import argparse
import glob
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()

GRIPPER_DIM = 6
PAN_DIM = 0
CLOSE_ON = -60.0       # gripper above this -> closed (0 = shut, -270 = wide open)
CLOSE_OFF = -90.0      # gripper below this -> open
DISP_MAX = 32.0        # max arm-joint L2 displacement for "never left the closing pose"
MIN_CLOSED_FRAMES = 15
BOUNDARY_MARGIN = 30   # arm parked with the gripper closed at episode start/end
RAMP_DELTA = 1.5       # deg/frame above which the gripper counts as actively opening
ARRIVAL_TOL = 3.0      # deg of pan within the container extremum that counts as arrived
SETTLE_DELTA = 2.0     # deg/frame below which the opening ramp counts as finished
SETTLE_MAX = 45        # cap on the opening ramp length (1.5 s)


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


def release_span(grip, close_start, close_stop):
    """The final opening ramp: the monotonic descent that ends the carry.

    `close_stop` is where the hysteresis dropped below CLOSE_OFF, partway down the
    ramp — the motion starts earlier and ends later (the joint coasts to its open
    rest value). The start is found by walking back along the descent rather than
    by a fixed "still gripping" threshold: how far the gripper shuts depends on the
    object, so a shirt held at -40 never re-enters a -20 band and would otherwise
    stretch the release back across the whole carry.
    """
    begin = close_stop
    while begin > close_start and float(grip[begin - 1] - grip[begin]) > RAMP_DELTA:
        begin -= 1
    end = close_stop
    while end + 1 < len(grip) and end - close_stop < SETTLE_MAX:
        if abs(float(grip[end + 1] - grip[end])) < SETTLE_DELTA:
            break
        end += 1
    return begin, min(end + 1, len(grip))


def arrival_at(pan, close_start, ramp_begin, extremum):
    """First frame of the carry that has reached the container, within ARRIVAL_TOL.

    The mechanical opening is only ~0.6 s, too tight to be a useful subtask label. A
    subtask is a unit of intent, so "release X in C" is the whole deposit: arrive
    over the container, then open. The arrival is a real event (the pan reaching the
    extremum that identifies the container), not an offset chosen to pad the label.
    """
    seg = pan[close_start:ramp_begin]
    reached = np.flatnonzero(np.abs(seg - extremum) <= ARRIVAL_TOL)
    return close_start + int(reached[0]) if len(reached) else ramp_begin


def container_of(pan, close_start, close_stop):
    """Signed pan extremum relative to the grasp pose — the arm's furthest committed swing.

    argmax|pan| picks the wrong side when the arm swings through negative pan on its
    way to a positive-pan container, so measure displacement from where it grasped.
    """
    seg = pan[close_start:close_stop]
    origin = float(seg[0])
    lo, hi = float(seg.min()), float(seg.max())
    return hi if abs(hi - origin) >= abs(lo - origin) else lo


def prior_labels(root: Path):
    path = root / "meta" / "subtask_windows.json"
    if not path.exists():
        return []
    episodes = json.load(open(path))["episodes"]
    return sorted(
        (int(w["from_index"]), int(w["to_index"]), str(w["subtask"]))
        for wins in episodes.values()
        for w in wins
    )


def majority_object(spans, start, stop):
    """Most common object phrase in the prior labels overlapping [start, stop)."""
    votes = Counter()
    for a, b, text in spans:
        if a < stop and b > start:
            for prefix in ("grasp the ", "move the ", "release the "):
                if text.startswith(prefix):
                    votes[text[len(prefix) :].split(" to the ")[0].split(" in the ")[0]] += min(b, stop) - max(
                        a, start
                    )
    return votes.most_common(1)[0][0] if votes else "<object>"


def segment_episode(state, lo, hi, spans, disp_max):
    grip, pan, arm = state[lo:hi, GRIPPER_DIM], state[lo:hi, PAN_DIM], state[lo:hi, :GRIPPER_DIM]
    n = hi - lo

    carries, failures = [], []
    for start, stop in closed_intervals(grip):
        if start < BOUNDARY_MARGIN or n - stop < BOUNDARY_MARGIN:
            continue
        seg = arm[start:stop]
        disp = float(np.linalg.norm(seg - seg[0], axis=1).max())
        (carries if disp > disp_max else failures).append((start, stop, disp))

    windows, cursor = [], 0
    for start, stop, _disp in carries:
        open_begin, open_end = release_span(grip, start, stop)
        obj = majority_object(spans, lo + start, lo + stop)
        extremum = container_of(pan, start, stop)
        container = "basket" if extremum > 0 else "bin"
        arrival = arrival_at(pan, start, open_begin, extremum)
        windows.append((cursor, start, f"grasp the {obj}"))
        windows.append((start, arrival, f"move the {obj} to the {container}"))
        windows.append((arrival, open_end, f"release the {obj} in the {container}"))
        cursor = open_end
    if cursor < n:
        windows.append((cursor, n, "return to home"))

    return (
        [
            {"from_index": lo + a, "to_index": lo + b, "subtask": text}
            for a, b, text in windows
            if b > a
        ],
        failures,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--disp-max", type=float, default=DISP_MAX)
    ap.add_argument("--write", action="store_true", help="replace meta/subtask_windows.json")
    args = ap.parse_args()

    root = args.data_dir
    fps = float(json.load(open(root / "meta" / "info.json"))["fps"])
    state, bounds = load_dataset(root)
    spans = prior_labels(root)

    episodes, n_fail, lengths = {}, 0, []
    for ep, lo, hi in bounds:
        windows, failures = segment_episode(state, lo, hi, spans, args.disp_max)
        episodes[str(ep)] = windows
        n_fail += len(failures)
        lengths += [w["to_index"] - w["from_index"] for w in windows]
        console.print(f"[bold]ep{ep}[/bold]  {len(windows)} segments, {len(failures)} failed closes")
        for w in windows:
            console.print(
                f"   [{w['from_index']:6d},{w['to_index']:6d}) "
                f"{(w['to_index'] - w['from_index']) / fps:5.1f}s  {w['subtask']}"
            )
        for start, stop, disp in failures:
            console.print(
                f"   [dim]      failed close [{lo + start:6d},{lo + stop:6d}) "
                f"{(stop - start) / fps:.1f}s disp={disp:.1f}deg[/dim]"
            )

    lengths = np.array(lengths)
    console.print(
        f"\n[bold]{len(lengths)} segments[/bold] over {len(bounds)} episodes, "
        f"{n_fail} failed closes; length med={np.median(lengths) / fps:.1f}s "
        f"min={lengths.min() / fps:.1f}s max={lengths.max() / fps:.1f}s"
    )

    payload = {
        "model": "proprio-semantic-segmenter",
        "annotator": f"semantic_segment.py --disp-max {args.disp_max}",
        "created_date": date.today().isoformat(),
        "segmentation": (
            "Semantic phases from the gripper-closed intervals: grasp = approach + close, "
            "move = shut and travelling, release = the opening ramp, return to home = final park. "
            "Failed closes stay inside the enclosing grasp segment."
        ),
        "interval_seconds": None,
        "top_key": "observation.images.top",
        "wrist_key": "observation.images.wrist",
        "episodes": episodes,
    }
    out = root / "meta" / ("subtask_windows.json" if args.write else "subtask_windows.semantic.json")
    if args.write and (root / "meta" / "subtask_windows.json").exists():
        (root / "meta" / "subtask_windows.json").rename(root / "meta" / "subtask_windows.grid.json")
        console.print("[yellow]previous grid windows kept as meta/subtask_windows.grid.json[/yellow]")
    json.dump(payload, open(out, "w"), indent=1)
    console.print(f"wrote {out}")


if __name__ == "__main__":
    main()

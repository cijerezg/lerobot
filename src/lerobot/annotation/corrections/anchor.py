"""What the episode itself went on to do: the grasps that worked, as the target of the corrections before them.

A labeller that places a grasp from the images lands about 2.5 cm from where the demonstrator took the object
(``grasp_test``, 2026-11-12), with or without depth. The measured tool pose at a grasp that worked has no such
error, so wherever the episode goes on to grasp the object of the frame's subtask, that pose is the anchor of the
frame: ``dump`` offers it in ``case.txt``, ``build`` records how far the label is from it.

  uv run python -m lerobot.annotation.corrections.anchor <pass> <out>

rewrites a labelled pass without a labeller: ``<out>`` is a new pass folder (for ``build``) with every
``waypoints.json`` of ``<pass>``, and where a frame has an anchor that needs no check (no closing attempt between the
frame and the grasp, and not a grasp of a demonstration stretch graded 1-3) and the label's contact pose is more than
``ANCHOR_M`` from it, the manoeuvre is replaced by: the anchor's pose, close, lift. Those frames are listed in
``<out>/anchored.csv``.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.corrections.arm import Arm
from lerobot.annotation.corrections.source import Episode, sources

CLOSE_DEG, HOLD, LIFT_M = 15.0, 60, 0.03  # closing turn, frames it stays shut, lift after it
ANCHOR_M = 0.01  # a contact pose this near the anchor's is the anchor


def closings(arm: Arm, e: Episode) -> list[tuple[int, int]]:
    """(close_frame, grasp_frame) of the grasps of the episode that worked: the gripper closes by ``CLOSE_DEG`` or
    more inside a ``grasp ...`` subtask, does not reopen for ``HOLD`` frames, the annotation has moved on to the next
    subtask by then and the tool rises ``LIFT_M`` within 3 s. close_frame = the closing starts, grasp_frame = the
    fingers stop."""
    s = e.states[:, 6]
    closing = np.flatnonzero(np.diff(s) > 0.3)
    found = []
    for run in np.split(closing, np.flatnonzero(np.diff(closing) > 3) + 1):
        if len(run) == 0 or s[run[-1] + 1] - s[run[0]] < CLOSE_DEG:
            continue
        close, stop = int(run[0]), int(run[-1] + 1)
        if stop + 90 >= len(s) or not str(e.subtasks[close]).startswith("grasp"):
            continue
        if s[stop : stop + HOLD].min() < s[stop] - 8 or e.subtasks[stop + HOLD] == e.subtasks[close]:
            continue
        tip = arm.gripper.fk(e.states[stop])[0]
        if max(arm.gripper.fk(e.states[f])[0][2] for f in range(stop, stop + 90, 5)) - tip[2] >= LIFT_M:
            found.append((close, stop))
    return found


def anchor(e: Episode, worked: list[tuple[int, int]], t: int) -> dict | None:
    """The anchor of frame ``t`` (``worked`` = ``closings`` of the episode): the next grasp that worked, when ``t``
    is in a ``grasp ...`` subtask and the subtask does not change up to it. ``frame`` = the fingers stop, ``opening``
    = the gripper as the closing starts, ``stall`` = where it stops, ``tried`` = between ``t`` and that grasp the
    gripper already closed by ``CLOSE_DEG`` or more and reopened (a failed attempt: the object may have moved since
    ``t``), ``try_frame`` = where the last such closing stopped (-1 = none), ``grade`` = the lowest grade of the
    demonstration over the closing (0 = not a graded demonstration)."""
    if not str(e.subtasks[t]).startswith("grasp"):
        return None
    for close, stop in worked:
        if close > t:
            if not (e.subtasks[t : close + 1] == e.subtasks[t]).all():
                return None
            s, last = e.states[:, 6], -1
            closing = t + np.flatnonzero(np.diff(s[t:close]) > 0.3)
            for run in np.split(closing, np.flatnonzero(np.diff(closing) > 3) + 1):
                if len(run) and s[run[-1] + 1] - s[run[0]] >= CLOSE_DEG and s[run[-1] + 1 : close + 1].min() < s[run[-1] + 1] - 8:
                    last = int(run[-1] + 1)
            return dict(
                frame=stop, close=close, opening=float(s[close]), stall=float(s[stop]), tried=last >= 0, try_frame=last,
                grade=int(e.grades[close : stop + 1].min()),
            )
    return None


def grasp(arm: Arm, e: Episode, a: dict) -> list[dict]:
    """The manoeuvre to an anchor as waypoints: its pose with the fingers open as they were, close, lift 6 cm."""
    tip, approach, _ = arm.gripper.fk(e.states[a["frame"]])
    return [
        {"like_frame": a["frame"], "gripper": round(a["opening"]), "arrive": "approach"},
        {"like_frame": a["frame"], "gripper": round(min(a["stall"] + 10, 0))},
        {"tip": (tip + [0, 0, 0.06]).round(4).tolist(), "approach": approach.round(4).tolist()},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    work, out = args.work, args.out
    out.mkdir(parents=True)
    for name in ("pass.json", "stalls.json", "frames.parquet"):
        shutil.copy(work / name, out / name)
    frames = pd.read_parquet(work / "frames.parquet")
    arm, rows = Arm(), []
    for name, source in sources(json.loads((work / "pass.json").read_text()), list(frames.source.unique())).items():
        for index, group in frames[frames.source == name].groupby("episode"):
            e = Episode(source, index)
            worked = closings(arm, e)
            for t in group.frame:
                folder = Path(name) / f"ep{index:03d}" / f"f{t:06d}"
                if not (work / folder / "waypoints.json").exists():
                    continue
                spec = json.loads((work / folder / "waypoints.json").read_text())
                a = anchor(e, worked, t)
                if "skip" not in spec and a is not None and not a["tried"] and a["grade"] not in (1, 2, 3):
                    first = spec["waypoints"][0]
                    was = arm.gripper.fk(e.states[first["like_frame"]])[0] if "like_frame" in first else np.asarray(first["tip"], dtype=float)
                    away = float(np.linalg.norm(was - arm.gripper.fk(e.states[a["frame"]])[0]))
                    if away > ANCHOR_M:
                        contact = str(e.contacts[a["close"]])
                        spec = dict(spec, contact=spec["contact"] if contact == "unknown" else contact, waypoints=grasp(arm, e, a),
                                    note=f"anchor applied (the label was {100 * away:.1f} cm from it): the grasp of frame {a['frame']}, close, lift")
                        rows.append(dict(source=name, episode=index, frame=t, was_cm=100 * away, anchor_frame=a["frame"]))
                (out / folder).mkdir(parents=True)
                (out / folder / "waypoints.json").write_text(json.dumps(spec, indent=1))
    pd.DataFrame(rows).to_csv(out / "anchored.csv", index=False)
    print(f"wrote {out}: {len(rows)} manoeuvres replaced by their anchor")


if __name__ == "__main__":
    main()

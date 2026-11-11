"""Pick the frames of a correction pass.

  uv run python -m lerobot.annotation.corrections.sample <work>

Reads ``<work>/pass.json``. A frame is taken when it is on the grid of its episode (frame % ``every`` == 0, default
30 = 1 s; grade 3 demo frames stay on the 3 s grid, ``DEMO_GRADE_3_EVERY``), is not on the grid an earlier pass
already labelled (frame % ``done_every`` == 0, default none), has 2 s of episode after it (the hindsight clip) and
  inference source   always: policy-controlled and operator-controlled frames alike; episodes flagged truncated
                     are left out whole
  rollout source     always (an annotated policy run is corrected whole)
  cut source         it is removed footage (``Source.removed``); the kind reads "policy cut" / "operator cut"
  rebot source       its frame grade is in the pass's ``demo_grades`` (default 1 and 2) and its subtask does not
                     start with one of the pass's ``demo_skip_verbs`` (default none)

Writes ``<work>/frames.parquet``, one row per frame: source, episode, frame, kind (policy / operator / demo), grade
(0 = not a graded demo), subtask, contact, precision; and ``<work>/stalls.json``: subtask -> median measured
gripper angle (deg) of the stretches where the gripper stalled on an object, over every episode of the pass.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.corrections.source import Episode, sources

AFTER = 60  # frames: the 2 s of hindsight
DEMO_GRADE_3_EVERY = 90  # frames: grade 3 demo frames are sampled every 3 s whatever the pass's grid (user 2026-11-11)


def holds(episode: Episode) -> dict:
    """subtask -> measured gripper angles of the stretches where the gripper stalled on an object (commanded more
    closed than it got, not moving)."""
    s, a = episode.states[:, 6], episode.actions[:, 6]
    idx = np.flatnonzero((a - s > 8) & (np.abs(np.gradient(s)) < 0.3))
    found = {}
    for group in np.split(idx, np.flatnonzero(np.diff(idx) > 15) + 1):
        if len(group) >= 15:
            found.setdefault(str(episode.subtasks[group[0]]), []).append(float(np.median(s[group])))
    return found


def main() -> None:
    work = Path(sys.argv[1])
    spec = json.loads((work / "pass.json").read_text())
    grades, skip_verbs = spec.get("demo_grades", [1, 2]), spec.get("demo_skip_verbs", [])
    every, done_every = spec.get("every", 30), spec.get("done_every")
    rows, stalls = [], {}
    for source in sources(spec).values():
        for index in source.bounds:
            e = Episode(source, index)
            for subtask, angles in holds(e).items():
                stalls.setdefault(subtask, []).extend(angles)
            if source.kind == "inference" and e.truncated.any():
                continue
            for t in range(0, len(e.states) - AFTER, every):
                if done_every is not None and t % done_every == 0:
                    continue
                take = dict(
                    inference=True, rollout=True, cut=e.removed[t],
                    rebot=e.grades[t] in grades and str(e.subtasks[t]).split(" ")[0] not in skip_verbs
                    and (e.grades[t] != 3 or t % DEMO_GRADE_3_EVERY == 0),
                )[source.kind]
                if take:
                    rows.append(dict(
                        source=source.name, episode=index, frame=t, grade=int(e.grades[t]),
                        kind=str(e.kinds[t]) + (" cut" if source.kind == "cut" else ""),
                        subtask=str(e.subtasks[t]), contact=str(e.contacts[t]), precision=str(e.precisions[t]),
                    ))
    frames = pd.DataFrame(rows)
    frames.to_parquet(work / "frames.parquet")
    medians = {subtask: round(float(np.median(v))) for subtask, v in stalls.items()}
    (work / "stalls.json").write_text(json.dumps(medians, indent=1))
    print(frames.groupby(["source", "kind", "grade"], sort=False).size().to_string())
    print(frames.groupby("kind").size().to_string())
    print(f"wrote {work / 'frames.parquet'}: {len(frames)} frames in {frames.groupby(['source', 'episode']).ngroups} episodes")


if __name__ == "__main__":
    main()

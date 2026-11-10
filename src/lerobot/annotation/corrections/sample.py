"""Pick the frames of a correction pass.

  uv run python -m lerobot.annotation.corrections.sample <work>

Reads ``<work>/pass.json``. A frame is taken when it is on the 3 s grid of its episode (frame % 90 == 0), has 2 s of
episode after it (the hindsight clip) and
  inference source   always: policy-controlled and operator-controlled frames alike; episodes flagged truncated
                     are left out whole
  rollout source     always (an annotated policy run is corrected whole)
  rebot source       its frame grade is 1 or 2

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

EVERY, AFTER = 90, 60  # frames: the 3 s grid, the 2 s of hindsight


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
    rows, stalls = [], {}
    for source in sources(json.loads((work / "pass.json").read_text())).values():
        for index in source.bounds:
            e = Episode(source, index)
            for subtask, angles in holds(e).items():
                stalls.setdefault(subtask, []).extend(angles)
            if e.truncated.any():
                continue
            for t in range(0, len(e.states) - AFTER, EVERY):
                if source.kind != "rebot" or e.grades[t] in (1, 2):
                    rows.append(dict(
                        source=source.name, episode=index, frame=t, kind=str(e.kinds[t]), grade=int(e.grades[t]),
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

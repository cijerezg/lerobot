"""One dataset root of a correction pass, with the labels a correction needs for every frame.

Three kinds of root:
  rollout     an annotated policy-run root: labels as a rebot root, operator takeover from
              ``meta/online_labels.parquet``
  inference   a recorded policy run: subtask, operator takeover and the truncated flag from
              ``meta/online_labels.parquet``; contact and precision from the pass's ``contact`` table by subtask
              ("unknown" for a subtask it does not list)
  rebot       an annotated demonstration root: the frame grade and subtask from the segment rows split by the
              quality spans (``load_metadata_rows``, as the trainer reads them), contact and precision from their
              tables

``Source`` arrays are indexed by the root's global frame index; ``Episode`` is one episode's slice, episode-local.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot.episode import decode, layout
from lerobot.rl.offline_dataset_utils import load_metadata_rows

DEPTH_UNITS_M = 1e-4  # the D405's depth unit


class Source:
    def __init__(self, name: str, root: str, kind: str, contact: dict, top: dict, cameras: dict):
        self.name, self.root, self.kind = name, WORKSPACE / root, kind
        # ``top``: where ``fit_top`` starts; ``cameras``: the top camera (``arm.project``) of each episode it fitted
        self.top = dict(top, pos=np.asarray(top["pos"]), mat=np.asarray(top["mat"]))
        self.cameras = {i: dict(c, pos=np.asarray(c["pos"]), mat=np.asarray(c["mat"])) for i, c in cameras.items()}
        files = sorted((self.root / "data").rglob("*.parquet"))
        data = pd.concat([pd.read_parquet(p, columns=["index", "observation.state", "action"]) for p in files]).sort_values("index")
        self.states, self.actions = np.stack(data["observation.state"]).astype(float), np.stack(data["action"]).astype(float)
        files = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        episodes = pd.concat([pd.read_parquet(p, columns=["episode_index", "tasks", "dataset_from_index", "dataset_to_index"]) for p in files])
        self.bounds = {int(e.episode_index): (int(e.dataset_from_index), int(e.dataset_to_index)) for e in episodes.itertuples()}
        self.tasks = {int(e.episode_index): str(e.tasks[0]) for e in episodes.itertuples()}
        self.depth = layout(self.root)[1]
        n = len(self.states)
        self.grades, self.truncated = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
        if kind == "inference":
            labels = pd.read_parquet(self.root / "meta/online_labels.parquet").sort_values("index")
            self.subtasks, self.truncated = labels.subtask.to_numpy().astype(str), labels.truncated.to_numpy()
            self.kinds = np.where(labels.is_intervention, "operator", "policy")
            known = [contact.get(s, ["unknown", "unknown"]) for s in self.subtasks]
            self.contacts, self.precisions = (np.array([str(k[i]) for k in known]) for i in (0, 1))
        else:
            segments, _, _, precision_rows, contact_rows = load_metadata_rows(self.root)
            self.kinds = np.full(n, "demo")
            if kind == "rollout":
                labels = pd.read_parquet(self.root / "meta/online_labels.parquet").sort_values("index")
                self.kinds = np.where(labels.is_intervention, "operator", "policy")
            self.subtasks, self.contacts, self.precisions = (np.full(n, "unknown", dtype=object) for _ in range(3))
            for row in segments:
                self.subtasks[row["from_index"] : row["to_index"]] = row["subtask"]
                self.grades[row["from_index"] : row["to_index"]] = row["quality"]
            for row in contact_rows or []:
                self.contacts[row["from_index"] : row["to_index"]] = row["contact_slug"]
            for row in precision_rows or []:
                self.precisions[row["from_index"] : row["to_index"]] = str(row["precision"])


class Episode:
    """One episode of a source: states and actions (N x 7, degrees) and the per-frame labels, episode-local."""

    def __init__(self, source: Source, index: int):
        self.source, self.index, self.task = source, index, source.tasks[index]
        self.top = source.cameras.get(index)  # None: not fitted
        self.start, end = source.bounds[index]
        self.states, self.actions = source.states[self.start : end], source.actions[self.start : end]
        self.subtasks, self.kinds, self.grades = (x[self.start : end] for x in (source.subtasks, source.kinds, source.grades))
        self.contacts, self.precisions, self.truncated = (x[self.start : end] for x in (source.contacts, source.precisions, source.truncated))

    def images(self, camera: str, frames: list[int]) -> list[np.ndarray]:
        """640 x 480 RGB frames of the top or wrist camera."""
        got = decode(dict(source=str(self.source.root), episode=self.index, key=self.source.name), camera, frames, size=(640, 480))
        return [np.asarray(got[f]) for f in frames]

    def depth_m(self, frame: int) -> np.ndarray | None:
        """Wrist depth stored within one frame of ``frame`` (it is kept for every third frame, at any phase): metres
        along the optical axis, 0 = no reading. None when the recording has no depth there."""
        for f in (frame, frame - 1, frame + 1):
            file = self.source.root / "depth" / self.source.depth / f"episode-{self.index:06d}" / f"frame-{f:06d}.png"
            if file.exists():
                return cv2.imread(str(file), cv2.IMREAD_UNCHANGED) * DEPTH_UNITS_M
        return None


def sources(spec: dict, names: list[str] | None = None) -> dict[str, Source]:
    """The sources of a pass (its ``pass.json``), by name; ``names`` keeps only those."""
    found = {}
    for s in spec["sources"]:
        if names is None or s["name"] in names:
            cameras = {int(k.split("/ep")[1]): c for k, c in spec.get("cameras", {}).items() if k.split("/ep")[0] == s["name"]}
            found[s["name"]] = Source(s["name"], s["root"], s["kind"], spec["contact"], spec["top"][s["top"]], cameras)
    return found

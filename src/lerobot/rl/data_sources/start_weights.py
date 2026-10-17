"""Per-start draw weights for a ReBot root from its segment, precision and gripper-event metadata.

The weight of a frame is 1 (base) plus a bump around the commit moment of its segment; see
``StartWeightsConfig``. Frames are indexed by the root's global dataset index, which is the replay
cache row (episode_group_views checks that the cache is the complete root).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.configs.diverse import StartWeightsConfig

PLAIN = {"move", "return"}


def _kernel(frames: np.ndarray, events: np.ndarray, before: float, after: float, fps: float, smooth: bool) -> np.ndarray:
    """0..1 bump around every event: rises over `before` s, decays over `after` s."""
    k = np.zeros(len(frames))
    for event in events:
        dt = (event - frames) / fps  # > 0 before the event
        if smooth:
            g = np.where(dt >= 0, np.exp(-0.5 * (dt / (before / 2)) ** 2) if before else 0.0,
                         np.exp(-0.5 * (dt / (after / 2)) ** 2) if after else 0.0)
        else:
            g = ((dt >= -after) & (dt <= before)).astype(float)
        k = np.maximum(k, g)
    return k


def frame_weights(root: str | Path, size: int, cfg: StartWeightsConfig) -> np.ndarray:
    """Weight of every frame of the root (length `size`), base 1."""
    meta = Path(root) / "meta"
    fps = float(json.load(open(meta / "info.json"))["fps"])
    keys = ["episode_index", "segment_index"]
    seg = pd.read_parquet(meta / "episode_metadata.parquet")[keys + ["from_index", "to_index", "subtask"]]
    seg = seg.merge(pd.read_parquet(meta / "precision.parquet")[keys + ["precision"]].drop_duplicates(keys), on=keys)
    events = pd.read_parquet(meta / "depth_gripper_events.parquet")[["event_type", "index"]]
    event_of = {kind: np.sort(events[events.event_type == kind]["index"].values) for kind in ("close", "open")}
    w = np.ones(size)
    for r in seg.itertuples():
        a, b = int(r.from_index), min(int(r.to_index), size)
        if b <= a:
            continue
        verb = r.subtask.split()[0]
        frames = np.arange(a, b)
        if verb in PLAIN:
            continue
        if verb in ("grasp", "release"):
            kind, (before, after) = ("close", cfg.grasp_window) if verb == "grasp" else ("open", cfg.release_window)
            ev = event_of[kind]; ev = ev[(ev >= a) & (ev < b)]
            peak = cfg.peak_ratio * cfg.precision_scale[int(r.precision) - 1]
            k = _kernel(frames, ev, before, after, fps, cfg.smooth)
        else:
            k = np.ones(b - a)
            if cfg.smooth:
                e = min(int(fps), (b - a) // 2)
                ramp = np.linspace(0, 1, e + 1)[1:]
                k[:e] = ramp
                k[b - a - e:] = ramp[::-1]
            peak = cfg.skill_ratio
        w[a:b] = 1 + (peak - 1) * k
    return w


def start_weights(root: str | Path, size: int, indices: torch.Tensor, cfg: StartWeightsConfig) -> torch.Tensor:
    w = frame_weights(root, size, cfg)
    return torch.as_tensor(w[indices.cpu().numpy()], dtype=torch.float32, device=indices.device)

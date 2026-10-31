"""Frames of one recorded episode every ``period`` seconds, from either source the evals leave behind:

- a sim eval layout folder (``<sim>/<layout>/videos/ep<N>_top.mp4``, ``ep<N>_wrist.mp4``; 30 fps, t = seconds since the
  scene reset, the clock of subtask_events.json),
- a LeRobot v3 dataset root (``meta/episodes`` gives the video file and the [from, to] timestamps of the episode;
  t = seconds since the episode start).

Returns ``[(t, {camera: HxWx3 uint8 RGB}), ...]``; a camera missing a frame at some t drops that sample."""

import glob
from pathlib import Path

import av
import numpy as np
import pandas as pd

CAMERA_KEYS = {"top": "observation.images.top", "wrist": "observation.images.wrist"}


def sample_video(path: Path, period: float, t_from: float = 0.0, t_to: float = float("inf")) -> list[tuple[float, np.ndarray]]:
    out = []
    next_t = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if t_from > 0:
            container.seek(int(t_from / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < t_from - 1e-6:
                continue
            if t > t_to + 1e-6:
                break
            rel = t - t_from
            if rel + 1e-6 >= next_t:
                out.append((round(rel, 3), frame.to_ndarray(format="rgb24")))
                next_t = (int(rel / period + 1e-6) + 1) * period
    return out


def _merge(per_camera: dict[str, list[tuple[float, np.ndarray]]]) -> list[tuple[float, dict[str, np.ndarray]]]:
    by_t = [dict(frames) for frames in per_camera.values()]
    times = sorted(set.intersection(*(set(d) for d in by_t)))
    cams = list(per_camera)
    return [(t, {cam: by_t[i][t] for i, cam in enumerate(cams)}) for t in times]


def decode_sim_episode(sim_dir: Path, episode: int, cameras: list[str], period: float):
    return _merge({cam: sample_video(sim_dir / "videos" / f"ep{episode}_{cam}.mp4", period) for cam in cameras})


def decode_dataset_episode(root: Path, episode: int, cameras: list[str], period: float):
    episodes = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(str(root / "meta" / "episodes" / "*" / "*.parquet"))))
    row = episodes[episodes.episode_index == episode].iloc[0]
    per_camera = {}
    for cam in cameras:
        key = CAMERA_KEYS[cam]
        path = root / "videos" / key / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}" / f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
        per_camera[cam] = sample_video(path, period, float(row[f"videos/{key}/from_timestamp"]), float(row[f"videos/{key}/to_timestamp"]))
    return _merge(per_camera)


def episode_task(root: Path, episode: int) -> str:
    episodes = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(str(root / "meta" / "episodes" / "*" / "*.parquet"))))
    return str(episodes[episodes.episode_index == episode].iloc[0]["tasks"][0])

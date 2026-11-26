"""Fingertip path of a ReBot episode and how smooth a stretch of it is (the quality screen of annotation_rubric.md).

    uv run python -m lerobot.annotation.rebot.tip_motion outputs/data/<root> <episode> --segments <labels.json>

The tip comes from forward kinematics of the recorded joints. `measure` gives, for frames [start, end):

| key | meaning |
|---|---|
| path_cm, straight_cm | length of the tip path, and of the straight line between its ends |
| directness | straight_cm / path_cm: 1 = straight, low = detour or back-and-forth |
| back_cm | largest rise of the distance to the end point: how far the tip backed away from where it ends up |
| pause_s | longest stretch below 1 cm/s |
| smoothness | spectral arc length of the speed profile (Balasubramanian et al. 2012): about -1.6 for one clean reach, more negative with every extra stop, restart or shake |
| reversals | most turns of more than 2 mm within any 1 s (3 or more = shaking) |

The labels file is the one `subtask_labeler` writes ({segments: [{start, end, subtask}]}); one row is printed per segment.
"""

import argparse
import glob
import json

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from lerobot.annotation.corrections.arm import Arm

FPS = 30


def tip(root, episode):
    """Fingertip position of every frame of the episode, cm."""
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    data = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in files)
    data = data[data.episode_index == episode].sort_values("frame_index")
    arm = Arm()
    return np.array([arm.gripper.fk(q)[0] for q in np.stack(data["observation.state"])]) * 100


def smoothness(speed):
    """Spectral arc length of a speed profile: 0 Hz to the last frequency (at most 10 Hz) holding 5% of the peak."""
    n = 2 ** (int(np.ceil(np.log2(len(speed)))) + 4)
    freq = np.arange(n) * FPS / n
    spectrum = np.abs(np.fft.fft(speed, n))
    spectrum = spectrum / spectrum.max()
    keep = freq <= 10
    freq, spectrum = freq[keep], spectrum[keep]
    last = np.flatnonzero(spectrum >= 0.05)[-1]
    freq, spectrum = freq[: last + 1], spectrum[: last + 1]
    return -np.sum(np.sqrt((np.diff(freq) / freq[-1]) ** 2 + np.diff(spectrum) ** 2))


def reversals(path):
    """Most turns of more than 2 mm in any 1 s of the path, along the main direction of that second."""
    most = 0
    for a in range(0, len(path) - 5, FPS // 2):
        piece = path[a : a + FPS] - path[a : a + FPS].mean(0)
        along = piece @ np.linalg.svd(piece, full_matrices=False)[2][0]
        count, direction, extreme = 0, 0, along[0]
        for value in along[1:]:
            if direction == 0:
                if abs(value - extreme) > 0.2:
                    direction, extreme = np.sign(value - extreme), value
            elif (value - extreme) * direction > 0:
                extreme = value
            elif (extreme - value) * direction > 0.2:
                count, direction, extreme = count + 1, -direction, value
        most = max(most, count)
    return most


def measure(positions, start, end):
    path = uniform_filter1d(positions[start:end], 5, axis=0, mode="nearest")
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    speed = step * FPS
    to_end = np.linalg.norm(path - path[-1], axis=1)
    slow = np.r_[0, (speed < 1).astype(int), 0]
    edges = np.flatnonzero(np.diff(slow))
    return {
        "path_cm": step.sum(),
        "straight_cm": to_end[0],
        "directness": to_end[0] / max(step.sum(), 1e-6),
        "back_cm": float(np.max(to_end - np.minimum.accumulate(to_end))),
        "pause_s": float((edges[1::2] - edges[::2]).max() / FPS) if len(edges) else 0.0,
        "smoothness": smoothness(speed),
        "reversals": reversals(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("episode", type=int)
    parser.add_argument("--segments", required=True)
    args = parser.parse_args()
    positions = tip(args.root, args.episode)
    segments = json.load(open(args.segments))["segments"]
    rows = [{"start": s["start"], "end": s["end"], "subtask": s["subtask"], **measure(positions, s["start"], s["end"])} for s in segments]
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

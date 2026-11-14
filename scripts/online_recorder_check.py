"""Hardware-free check of rl/online_recorder.OnlineEpisodeRecorder.

Drives the recorder with synthetic raw observations (uint8 HWC frames, uint16 depth, float
joints) and 8-wide padded actions for two episodes plus one unfinished episode, then checks
the result against the recorded corpus: identical feature schema, PNG16 sidecar on the
global-index stride grid, exact float32 joints/actions, and the online_labels sidecar.
Writes to a NEW directory under outputs/ on every run (run from the workspace root).

    .venv/bin/python lerobot/scripts/online_recorder_check.py
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.online_recorder import LABELS_FILE, OnlineEpisodeRecorder

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]


class StubRobot:
    name = "rebot_b601_follower"
    cameras = {"wrist": None, "top": None}
    observation_features = {**{f"{m}.pos": float for m in MOTORS}, "wrist": (480, 640, 3), "top": (480, 640, 3)}
    action_features = {f"{m}.pos": float for m in MOTORS}


def raw_obs(t: int) -> dict:
    joints = {f"{m}.pos": float(10 * i + t) for i, m in enumerate(MOTORS)}
    return {
        "agent_pos": np.array(list(joints.values())),
        "pixels": {cam: np.full((480, 640, 3), t % 256, np.uint8) for cam in ("wrist", "top")},
        **joints,
        "wrist.depth": torch.from_numpy(np.full((480, 640), 1000 + t, np.uint16)),
    }


run_dir = Path(tempfile.mkdtemp(prefix="online_recorder_check_", dir="outputs"))
root = run_dir / "inference_dataset"
rec = OnlineEpisodeRecorder(StubRobot(), root=root, fps=30, task="Put sock in brown basket", depth_stride=3)

lengths = [10, 7]  # 7 is not a multiple of 3: exercises the global-index phase of the depth grid
t = 0
for ep_len in lengths:
    for i in range(ep_len):
        rec.add_frame(raw_obs(t), torch.arange(8, dtype=torch.float32) + t,
                      is_intervention=(i % 4 == 0), subtask_index=i % 3, subtask=f"sub{i % 3}")
        t += 1
    rec.save_episode()
rec.add_frame(raw_obs(t), torch.zeros(8), is_intervention=False, subtask_index=-1, subtask="")  # unfinished
rec.finalize()  # saves it as episode 2, truncated

info = json.load(open(root / "meta/info.json"))
ref = json.load(open("outputs/rebot_socks_basket-annotated-v2/meta/info.json"))
schema = lambda i: {k: (v["dtype"], list(v["shape"]), v.get("names")) for k, v in i["features"].items()}
assert schema(info) == schema(ref), (schema(info), schema(ref))
assert info["robot_type"] == ref["robot_type"] and info["fps"] == 30
assert (info["total_episodes"], info["total_frames"]) == (3, 18), info

pngs = sorted(p.relative_to(root / "depth") for p in (root / "depth").rglob("*.png"))
expected = [Path(f"wrist.depth/episode-{ep:06d}/frame-{fr:06d}.png")
            for ep, start, n in [(0, 0, 10), (1, 10, 7), (2, 17, 1)] for fr in range(n) if (start + fr) % 3 == 0]
assert pngs == expected, (pngs, expected)
d = cv2.imread(str(root / "depth/wrist.depth/episode-000001/frame-000002.png"), cv2.IMREAD_UNCHANGED)
assert d.dtype == np.uint16 and d[0, 0] == 1000 + 12, (d.dtype, d[0, 0])

ds = LeRobotDataset(run_dir.name, root=root)
row = ds[12]  # global index 12 = episode 1, frame 2
assert (int(row["episode_index"]), int(row["frame_index"])) == (1, 2)
assert torch.equal(row["observation.state"], torch.tensor([10 * i + 12 for i in range(7)], dtype=torch.float32))
assert torch.equal(row["action"], torch.arange(7, dtype=torch.float32) + 12)
img = row["observation.images.wrist"]
assert img.shape == (3, 480, 640) and abs(float(img.float().mean()) * 255 - 12) < 3, (img.shape, float(img.float().mean()) * 255)

labels = pq.read_table(root / LABELS_FILE).to_pylist()
assert len(labels) == 18, len(labels)
assert labels[12] == {"episode_index": 1, "frame_index": 2, "index": 12, "is_intervention": False,
                      "subtask_index": 2, "subtask": "sub2", "truncated": False}, labels[12]
assert labels[0]["is_intervention"] is True
assert [r["truncated"] for r in labels] == [False] * 17 + [True]
assert not (root / "images").exists() or not any((root / "images").rglob("*.png")), "temp images left behind"
print("OK", root)

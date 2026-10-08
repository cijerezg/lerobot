"""Rebuild episodes a crashed recording never saved, into a NEW dataset root.

While recording, the writer keeps the images on disk and the other values (state, action, task) in
`<root>/recovery/episode-XXXXXX.pkl`, one record per frame. A crash or Ctrl-C before `save_episode`
leaves both behind. This script replays them through `add_frame` / `save_episode` into `--out`; the
crashed root is only read. Frames stop at the first one missing a record or a camera image. Depth
PNGs (`depth/<key>/episode-XXXXXX/`) are copied for the recovered frames.

    uv run python -m lerobot.scripts.recover_episodes --root <crashed root> --out <new root>
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from lerobot.utils.constants import DEFAULT_FEATURES


def read_frame_log(path: Path) -> list[dict]:
    records = []
    with open(path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except (EOFError, pickle.UnpicklingError):  # end of file, or a record cut by the crash
                return records


def recover(root: Path, out: Path) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    features = {k: v for k, v in info["features"].items() if k not in DEFAULT_FEATURES}
    image_keys = [k for k, v in features.items() if v["dtype"] in ("image", "video")]
    for ft in features.values():
        ft["shape"] = tuple(ft["shape"])

    dataset = LeRobotDataset.create(
        repo_id=out.name,
        fps=info["fps"],
        root=out,
        robot_type=info["robot_type"],
        features=features,
        use_videos=any(v["dtype"] == "video" for v in features.values()),
    )
    for log in sorted((root / "recovery").glob("episode-*.pkl")):
        old_ep = int(log.stem.split("-")[1])
        new_ep = dataset.meta.total_episodes
        records = read_frame_log(log)
        n = 0
        while n < len(records) and all(
            (root / DEFAULT_IMAGE_PATH.format(image_key=k, episode_index=old_ep, frame_index=n)).is_file()
            for k in image_keys
        ):
            n += 1
        print(f"{log.name}: {len(records)} records, {n} frames with every image -> episode {new_ep}")
        for i in range(n):
            frame = dict(records[i])
            for k in image_keys:
                path = root / DEFAULT_IMAGE_PATH.format(image_key=k, episode_index=old_ep, frame_index=i)
                frame[k] = np.asarray(Image.open(path).convert("RGB"))
            dataset.add_frame(frame)
        dataset.save_episode()

        for src in (root / "depth").glob(f"*/episode-{old_ep:06d}"):
            dst = out / "depth" / src.parent.name / f"episode-{new_ep:06d}"
            dst.mkdir(parents=True)
            for png in src.glob("frame-*.png"):
                if int(png.stem.split("-")[1]) < n:
                    shutil.copy2(png, dst / png.name)
    dataset.finalize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    recover(args.root, args.out)

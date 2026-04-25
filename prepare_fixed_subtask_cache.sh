#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed}"
REPO_ID="${REPO_ID:-jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed}"
CACHE_DIR="${CACHE_DIR:-outputs/buffer_cache}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
STALE_CACHE_FINGERPRINT="${STALE_CACHE_FINGERPRINT:-561a81c774cc0723}"
S3_PREFIX="${S3_PREFIX:-s3://YOUR_BUCKET/lerobot}"

if [[ ! -d "${ROOT}" ]]; then
  echo "dataset root does not exist: ${ROOT}" >&2
  exit 1
fi

uv run python - "${ROOT}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
info_path = root / "meta" / "info.json"
skills_path = root / "meta" / "skills.json"
data_paths = sorted((root / "data").glob("*/*.parquet"))

if not info_path.exists():
    raise FileNotFoundError(info_path)
if not skills_path.exists():
    raise FileNotFoundError(skills_path)
if not data_paths:
    raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

skills_meta = json.loads(skills_path.read_text())
skill_to_subtask_index = skills_meta["skill_to_subtask_index"]
episodes = skills_meta["episodes"]

total_counts: dict[int, int] = {}

for data_path in data_paths:
    df = pd.read_parquet(data_path)

    if "subtask_index" not in df.columns:
        subtask_index = np.empty(len(df), dtype=np.int64)

        for episode_index, episode_df in df.groupby("episode_index", sort=False):
            episode = episodes[str(int(episode_index))]
            skills = sorted(episode["skills"], key=lambda skill: float(skill["start"]))
            starts = np.array([float(skill["start"]) for skill in skills], dtype=np.float64)
            labels = np.array(
                [skill_to_subtask_index[skill["name"]] for skill in skills],
                dtype=np.int64,
            )

            timestamps = episode_df["timestamp"].astype("float64").to_numpy()
            skill_positions = np.searchsorted(starts, timestamps, side="right") - 1
            skill_positions = np.clip(skill_positions, 0, len(labels) - 1)
            subtask_index[episode_df.index.to_numpy()] = labels[skill_positions]

        df["subtask_index"] = subtask_index
        df.to_parquet(data_path, index=False)
        print(f"added subtask_index to {data_path}")
    else:
        print(f"subtask_index already present in {data_path}")

    counts = df["subtask_index"].value_counts().sort_index()
    for key, value in counts.items():
        total_counts[int(key)] = total_counts.get(int(key), 0) + int(value)

info = json.loads(info_path.read_text())
features = info.setdefault("features", {})
if "subtask_index" not in features:
    features["subtask_index"] = {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    print(f"added subtask_index to {info_path}")
else:
    print(f"subtask_index already present in {info_path}")

print("subtask_index counts:")
for key in sorted(total_counts):
    print(f"  {key}: {total_counts[key]}")
PY

echo "removing stale cache: ${CACHE_DIR}/${STALE_CACHE_FINGERPRINT}"
rm -rf "${CACHE_DIR}/${STALE_CACHE_FINGERPRINT}"

echo "regenerating memmap cache"
uv run python src/lerobot/scripts/lerobot_memmap_buffer_cache.py \
  --repo-id "${REPO_ID}" \
  --data-dir "${ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --video-backend "${VIDEO_BACKEND}"

echo
cat <<EOF
done.

# Upload fixed dataset and regenerated cache to S3 from this local machine.
aws s3 sync "${ROOT}" "${S3_PREFIX}/${ROOT}/"
aws s3 sync "${CACHE_DIR}/${STALE_CACHE_FINGERPRINT}" "${S3_PREFIX}/${CACHE_DIR}/${STALE_CACHE_FINGERPRINT}/"

# On the remote EC2 machine, sync them back down before restarting training.
aws s3 sync "${S3_PREFIX}/${ROOT}/" "${ROOT}"
aws s3 sync "${S3_PREFIX}/${CACHE_DIR}/${STALE_CACHE_FINGERPRINT}/" "${CACHE_DIR}/${STALE_CACHE_FINGERPRINT}"
EOF

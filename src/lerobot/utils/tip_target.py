"""The target of the next gripper event, per frame (docs/notes/tip_target_adapter.md).

adhocs/tip_offset/build.py writes the targets of a labelled root. This module reads them into one
row per frame: the prompt clause and the tip adapter's label both come from that row.
"""

import os

import numpy as np
import pandas as pd

# Per frame [in the wrist view (1 / 0), x, y, z, u, v]: the offset from the gripper tip to the tip at
# the next gripper event (m, tool frame) and its pixel in the 640 x 480 wrist image. The numbers are
# 0 when the first entry is 0. A NaN row = the root has no targets (no clause, no adapter loss).
TIP_TARGET_KEY = "tip_target"
# The raw wrist depth frame [480, 640] uint16 (1e-4 m) the adapter reads.
TIP_DEPTH_KEY = "tip_target_depth"
TIP_TARGET_STORE = "lerobot/annotation_data/tip_offset"


def tip_target_rows(root) -> np.ndarray | None:
    """The rows of a root [frames, 6] in frame order. None = build.py has not run on the root."""
    path = f"{TIP_TARGET_STORE}/{os.path.basename(str(root).rstrip('/'))}/targets.parquet"
    if not os.path.exists(path):
        return None
    targets = pd.read_parquet(path).sort_values("index")
    in_view = ((targets.kind != "none") & targets.in_view).to_numpy()
    rows = np.zeros((len(targets), 6), dtype=np.float32)
    rows[:, 0] = in_view
    rows[in_view, 1:] = targets[["x", "y", "z", "u", "v"]].to_numpy()[in_view]
    return rows

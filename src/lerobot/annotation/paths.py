"""Where annotation results live: outside this package, under the workspace that holds the repo.

Scripts and rubrics live in this package. Stores that later passes read (class files, proposals,
measured angles) are in ``WORKSPACE / "lerobot/annotation_data"``; everything a pass writes
(inventories, label files, sheets, review pages, reports) goes under ``WORKSPACE / "outputs/_annotation"``.
"""

import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]

# Results of the rubric v2 class pass (classes/<slug>/..., labels_<pool>.jsonl), shared by every pool.
QUALITY_V2 = WORKSPACE / os.environ.get("QUALITY_V2_WORK", "lerobot/annotation_data/quality_v2")

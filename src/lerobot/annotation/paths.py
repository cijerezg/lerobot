"""Where annotation results live: outside the lerobot repo, under the workspace that holds it.

Scripts and rubrics live in this package; everything a script writes (inventories, label files,
sheets, review pages, reports) goes under ``WORKSPACE / "migration"`` or ``WORKSPACE / "outputs"``.
"""

import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]

# Results of the rubric v2 class pass (classes/<slug>/..., labels_<pool>.jsonl), shared by every pool.
QUALITY_V2 = WORKSPACE / os.environ.get("QUALITY_V2_WORK", "migration/annotation_v2_2026-09-29")

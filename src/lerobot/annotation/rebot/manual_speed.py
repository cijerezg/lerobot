"""Add the per-segment speed of the label files to meta/speed_hybrid_v1.parquet as column ``manual_speed``
(provenance). The ``speed`` column (what the loader reads) stays the hybrid label. Run after
``lerobot.annotation.speed.rebot_speed_pass``.

    uv run python -m lerobot.annotation.rebot.manual_speed <work> outputs/<root>
"""

import json
import sys
from pathlib import Path

import pandas as pd

from lerobot.annotation.paths import WORKSPACE

if __name__ == "__main__":
    work = Path(sys.argv[1])
    meta = WORKSPACE / sys.argv[2] / "meta"
    info = json.loads((meta / "speed_hybrid_v1_info.json").read_text())
    assert "manual_speed" not in info, "already applied"
    table = pd.read_parquet(meta / "speed_hybrid_v1.parquet")
    prov = {p["episode_index"]: p for p in json.loads((meta / "provenance.json").read_text())}
    manual = []
    for row in table.itertuples():
        p = prov[row.episode_index]
        labels = json.loads((work / f"labels/{p['inventory_idx']:02d}.json").read_text())
        seg = labels["episodes"][p["part"]]["segments"][row.segment_index]
        assert seg["subtask"] == row.subtask
        manual.append(int(seg["speed"]))
    table["manual_speed"] = manual
    table.to_parquet(meta / "speed_hybrid_v1.parquet", engine="pyarrow")
    agree = float((table.manual_speed == table.speed).mean())
    info.update(
        manual_speed=dict(
            column="manual_speed",
            labels=f"{work}/labels/*.json (segments[].speed)",
            note="per-segment review, provenance only; `speed` = hybrid label",
            exact_agreement=round(agree, 3),
            shares={str(k): int(v) for k, v in sorted(pd.Series(manual).value_counts().items())},
        )
    )
    (meta / "speed_hybrid_v1_info.json").write_text(json.dumps(info, indent=2))
    hybrid = dict(table.speed.value_counts().sort_index())
    print(meta.parent.name, "hybrid", hybrid, "manual", info["manual_speed"]["shares"], "agree", round(agree, 3))

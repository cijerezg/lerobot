"""The agents' window labels of a pass (annotation_rubric.md) -> one table, per-frame labels and a review page.

    uv run python -m lerobot.annotation.rebot.window_assemble <work>

Reads every <work>/<name>/labels.json (+ window.json, signals.csv of `window_prepare`). Writes in <work>:

| file | content |
|---|---|
| table.json | one row per segment: labels, mean tip speed + level, tip path measures (`tip_motion`), quality after the fixed rules |
| mistakes.json | the mistakes, each with its window |
| frames.csv | per frame, what the prompt gets: high and mistake, both looking 1 s ahead (rubric sections 4, 5) |
| review.md | the same as tables |

Fixed quality rules on top of the agent's call: a segment a mistake overlaps is not high; a release is high only
when its move is. Speed level = terciles within the class over the segments of this folder (the frozen cuts of the
rubric, section 3, are not made yet).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.rebot.tip_motion import measure, tip

HORIZON = 30  # frames the policy predicts ahead (1 s)
WORK = Path(sys.argv[1])

rows, mistakes = [], []
for path in sorted(WORK.glob("*/labels.json")):
    name = path.parent.name
    doc = json.loads(path.read_text())
    window = json.loads((path.parent / "window.json").read_text())
    signals = pd.read_csv(path.parent / "signals.csv").set_index("frame")
    segments = sorted(doc["segments"], key=lambda s: s["start"])
    assert segments[0]["start"] == window["start"] and segments[-1]["end"] == window["end"], name
    assert all(a["end"] == b["start"] for a, b in zip(segments, segments[1:])), name
    for s in segments:
        # a mistake covers the whole motion, from its start to the event frame
        touched = [m["type"] for m in doc["mistakes"] if m["start"] < s["end"] and m["event_frame"] >= s["start"]]
        rows.append({"window": name, "root": window["root"], "episode": window["episode"], **s,
                     "speed": signals.tip_speed_cm_s.loc[s["start"]:s["end"] - 1].mean(), "mistake": ", ".join(touched)})
    mistakes += [{"window": name, **m} for m in doc["mistakes"]]

table = pd.DataFrame(rows)
tips = {key: tip(*key) for key in set(zip(table.root, table.episode))}
table = pd.concat([table, pd.DataFrame([measure(tips[r.root, r.episode], r.start, r.end) for r in table.itertuples()])], axis=1)
table["agent_quality"] = table.quality
quality = []
for i, r in table.iterrows():
    good = r.quality == "high" and not r.mistake
    if r["class"] == "release":
        good = good and i > 0 and table.window[i - 1] == r.window and table["class"][i - 1] == "move" and quality[i - 1] == "high"
    quality.append("high" if good else None)
table["quality"] = quality
# a class with fewer than 30 segments has no cuts and gets no speed label (rubric section 3)
table["level"] = table.groupby("class").speed.transform(lambda v: pd.qcut(v, 3, labels=["slow", "normal", "fast"]).astype(object) if len(v) >= 30 else None)

# per frame, a label describes the next 1 s (the action chunk): high = that second lies in high segments,
# mistake = that second reaches into the mistaken motion (so the label starts 1 s before the motion and ends at the event)
per_frame = []
for name, part in table.groupby("window"):
    start, end = int(part.start.min()), int(part.end.max())
    segment_high = np.concatenate([np.full(r.end - r.start, r.quality == "high") for r in part.itertuples()])
    for f in range(start, end):
        types = [m["type"] for m in mistakes if m["window"] == name and m["start"] - HORIZON <= f <= m["event_frame"]]
        per_frame.append({"window": name, "frame": f, "high": bool(segment_high[f - start : f - start + HORIZON + 1].all()) and not types, "mistake": ", ".join(types)})
per_frame = pd.DataFrame(per_frame)
per_frame.to_csv(WORK / "frames.csv", index=False)

lines = ["# Window labels", "", "Speed level = terciles within the class over the segments of this folder.", ""]
for name, part in table.groupby("window"):
    lines += [f"## {name} (episode {part.episode.iloc[0]})", "",
              "| frames | text | cm/s | speed | quality | direct | back cm | pause s | turns | mistake |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in part.iterrows():
        lines.append(f"| {r.start}-{r.end} | {r.subtask} | {r.speed:.1f} | {r.level} | {r.quality or ''} | {r.directness:.2f} | {r.back_cm:.1f} | "
                     f"{r.pause_s:.1f} | {r.reversals} | {r.mistake} |")
    lines.append("")
lines += ["## Mistakes", "", "| window | motion (start - event) | label on frames | type | note |", "|---|---|---|---|---|"]
lines += [f"| {m['window']} | {m['start']}-{m['event_frame']} | {m['start'] - HORIZON}-{m['event_frame']} | {m['type']} | {m['note'].replace('|', '/')} |" for m in mistakes]
(WORK / "mistakes.json").write_text(json.dumps(mistakes, indent=1))
table.to_json(WORK / "table.json", orient="records")
(WORK / "review.md").write_text("\n".join(lines) + "\n")

print(table.groupby("class").agg(n=("speed", "size"), high=("quality", lambda q: (q == "high").sum()),
                                 p10=("speed", lambda v: v.quantile(0.1)), p50=("speed", "median"), p90=("speed", lambda v: v.quantile(0.9))).round(1))
print(table.groupby("window").agg(n=("speed", "size"), high=("quality", lambda q: (q == "high").sum())))
print(per_frame.groupby("window").agg(frames=("high", "size"), high=("high", "sum"), mistake=("mistake", lambda m: (m != "").sum())))

"""One row per unit (ReBot segment, diverse atom) of the v2 scope, with its v1 labels.

uv run python -m lerobot.annotation.quality_v2.inventory   -> units.csv
"""
import json
from pathlib import Path

import pandas as pd

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
HERE = QUALITY_V2
REBOT = {
    "rebot_all": "outputs/rebot_all-annotated-v2",
    "additions": "outputs/rebot_cache_ready_2026-09-25/main_additions_train",
    "bits_book": "outputs/rebot_bits-book-annotated-v1",
    "external": "outputs/rebot_cache_ready_2026-09-25/external_rebot_train",
    "validation": "outputs/rebot_cache_ready_2026-09-25/validation_all",
}
DIVERSE = "outputs/diverse_robot_dataset_v3"
SRC = {"droid": "droid", "droid_success": "droid_success", "molmoact": "molmoact", "robochallenge": "robochallenge",
       "ur7e": "ur7e", "yam": "yam"}
PREPS = (" to the ", " into the ", " in the ", " on the ", " onto the ", " from the ", " at the ", " over the ", " off the ",
         " with the ", " inside the ", " under the ", " next to ")

rows = []
for ds, root in REBOT.items():
    m = f"{root}/meta"
    em = pd.read_parquet(f"{m}/episode_metadata.parquet")
    pr = pd.read_parquet(f"{m}/precision.parquet").set_index(["episode_index", "segment_index"]).precision
    ct = pd.read_parquet(f"{m}/contact.parquet").set_index(["episode_index", "segment_index"]).contact_slug
    mk = pd.read_parquet(f"{m}/mistakes.parquet")
    fps = json.load(open(f"{root}/meta/info.json"))["fps"]
    for s in em.itertuples():
        text = str(s.subtask); verb = text.split(" ")[0]
        rest = text[len(verb):].strip(); obj, tgt = rest, ""
        for p in PREPS:
            if p in " " + rest:
                i = (" " + rest).index(p); obj, tgt = (" " + rest)[:i].strip(), (" " + rest)[i:].strip(); break
        k = (s.episode_index, s.segment_index)
        nm = int(((mk.episode_index == s.episode_index) & (mk.from_index < s.to_index) & (mk.to_index > s.from_index)).sum())
        rows.append(dict(dataset=ds, half="rebot", episode=str(s.episode_index), unit=f"seg{s.segment_index}", subtask=text,
                         verb=verb, object=obj, target=tgt, fps=fps, frames=s.to_index - s.from_index,
                         seconds=(s.to_index - s.from_index) / fps, v1_quality=s.quality, precision=pr.get(k), contact=ct.get(k),
                         v1_mistakes=nm, split="train" if ds != "validation" else "validation"))

for part in ("corpus", "fmb"):
    p = Path(DIVERSE) / part
    pr = {(a["episode_id"], a["parent_interval_index"], a["atom_index"]): a.get("precision") for a in map(json.loads, open(p / "precision_atoms.jsonl"))}
    ct = {(a["episode_id"], a["parent_interval_index"], a["atom_index"]): a.get("contact_slug") for a in map(json.loads, open(p / "contact_atoms.jsonl"))}
    for a in map(json.loads, open(p / "subtask_atoms.jsonl")):
        k = (a["episode_id"], a["parent_interval_index"], a["atom_index"])
        tgt = " ".join(x for x in (a.get("preposition"), a.get("container")) if x)
        rows.append(dict(dataset="fmb" if part == "fmb" else SRC[a["source"]], half="diverse", episode=a["episode_id"],
                         unit=f"p{a['parent_interval_index']}a{a['atom_index']}", subtask=a["subtask"], verb=a["verb"],
                         object=a["object"] or "", target=tgt, fps=a["native_rate_hz"],
                         frames=a["end_timestep_exclusive"] - a["start_timestep"], seconds=a["duration_s"],
                         v1_quality=a["quality"], precision=pr.get(k), contact=ct.get(k), v1_mistakes=len(a["mistake_events"]),
                         split=a.get("split", "train")))

df = pd.DataFrame(rows)
df.to_csv(HERE / "units.csv", index=False)
print(df.groupby("dataset", sort=False).agg(episodes=("episode", "nunique"), units=("unit", "size"),
                                            hours=("seconds", lambda x: round(x.sum() / 3600, 2)),
                                            subtasks=("subtask", "nunique"), objects=("object", "nunique")).to_string())
print("total", len(df), round(df.seconds.sum() / 3600, 2), "h")

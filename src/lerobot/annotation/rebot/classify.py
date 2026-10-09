"""Side-by-side classes of a labelled pass: one row per staging-root segment, in the units format that
``quality_v2.material`` reads. Sheet classes are ``<pool>/<action>/<family>``.

<work>/pass.json gives the pool, the dataset name, the staging root and three tables: "class_rules"
([subtask prefix, action, family], the longest matching prefix wins: "grasp the hand" must not take "grasp the handle"), "reuse" (classes graded against an existing class file,
its FINAL.md and references) and "nearest" (the comparison class of a class that gets its own strategy pass).

    uv run python -m lerobot.annotation.rebot.classify <work>      -> <work>/units_classed.csv, class_map.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE

if __name__ == "__main__":
    work = Path(sys.argv[1])
    config = json.loads((work / "pass.json").read_text())
    pool, name, reuse, nearest = config["pool"], config["dataset"], config["reuse"], config["nearest"]
    v2 = f"{QUALITY_V2.relative_to(WORKSPACE)}/classes"
    root = WORKSPACE / config["staging"]
    em = pd.read_parquet(root / "meta/episode_metadata.parquet")
    mk = pd.read_parquet(root / "meta/mistakes.parquet")
    labels = {}
    for p in json.loads((root / "meta/provenance.json").read_text()):
        d = json.loads((work / f"labels/{p['inventory_idx']:02d}.json").read_text())
        labels[p["episode_index"]] = d["episodes"][p["part"]]
    rows = []
    for s in em.itertuples():
        action, family = max((len(prefix), a, f) for prefix, a, f in config["class_rules"] if s.subtask.startswith(prefix))[1:]
        seg = labels[s.episode_index]["segments"][s.segment_index]
        assert seg["subtask"] == s.subtask
        overlap = (mk.episode_index == s.episode_index) & (mk.from_index < s.to_index) & (mk.to_index > s.from_index)
        sheet = f"{pool}/{action}/{family}"
        row = dict(dataset=name, half="rebot", episode=s.episode_index, unit=f"seg{s.segment_index}", subtask=s.subtask)
        row.update(verb=s.subtask.split()[0], object="", target="", fps=30.0, frames=s.to_index - s.from_index)
        row.update(seconds=(s.to_index - s.from_index) / 30, v1_quality=0, precision=seg["precision"], contact=seg["contact"])
        row.update(v1_mistakes=int(overlap.sum()), split="train", pool=pool, verb_group=action, family=family, target_type="")
        row.update(family_unsure=False, action=action, **{"class": sheet}, sheet_class=sheet)
        rows.append(row)
    u = pd.DataFrame(rows)
    u.to_csv(work / "units_classed.csv", index=False)
    cmap = {}
    for sheet, g in u.groupby("sheet_class"):
        key = sheet.split("/", 1)[1]
        owner = reuse[key] if key in reuse else sheet.replace("/", "__")
        cmap[sheet] = dict(slug=sheet.replace("/", "__"), units=len(g), reuse=reuse.get(key), nearest=nearest.get(key))
        cmap[sheet]["final"] = f"{v2}/{owner}/strategy/FINAL.md"
        print(f"{len(g):4d} {sheet:40s} reuse={reuse.get(key)} nearest={nearest.get(key)}")
    (work / "class_map.json").write_text(json.dumps(cmap, indent=1))

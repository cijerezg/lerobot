"""New ReBot root with some episodes' task strings replaced. The source is read only. Never overwrites.

    uv run python -m lerobot.annotation.rebot.retask_root outputs/<root> --out outputs/<new_root> --tasks <map.json>

``map.json`` is ``{"<episode_index>": "<new task>"}``. Every file is hardlinked; the files that carry the task
(meta/tasks.parquet, the episodes table, every data file, provenance.json) are unlinked and rewritten. Tasks are
re-indexed in order of first use, so a string no episode keeps any more is dropped. Each changed provenance row keeps
the old string under ``retask_from``. Run verify_root on the result.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def rewrite(path):
    """Break the hardlink so writing the new file leaves the source untouched."""
    path.unlink()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", required=True, type=Path)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    assert not out.exists(), out
    new_task = {int(e): t for e, t in json.loads(args.tasks.read_text()).items()}

    for dirpath, _, files in os.walk(root):
        dst = out / Path(dirpath).relative_to(root)
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = Path(dirpath) / f
            if src.is_symlink():
                os.symlink(os.path.realpath(src), dst / f)
            else:
                os.link(src, dst / f)

    info = json.loads((root / "meta/info.json").read_text())
    ep_file = out / "meta/episodes/chunk-000/file-000.parquet"
    eps = pd.read_parquet(ep_file)
    assert list(eps.episode_index) == list(range(len(eps))) and set(new_task) <= set(eps.episode_index)
    assert all(len(t) == 1 for t in eps.tasks), "one task per episode"
    old_task = {int(e): t[0] for e, t in zip(eps.episode_index, eps.tasks, strict=True)}
    task = {e: new_task.get(e, t) for e, t in old_task.items()}
    tasks = list(dict.fromkeys(task[e] for e in sorted(task)))
    task_id = {t: i for i, t in enumerate(tasks)}

    eps["tasks"] = [[task[int(e)]] for e in eps.episode_index]
    for stat in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        eps[f"stats/task_index/{stat}"] = [[float(task_id[task[int(e)]])] if stat in ("mean", "q01", "q10", "q50", "q90", "q99")
                                            else [task_id[task[int(e)]]] for e in eps.episode_index]
    eps["stats/task_index/std"] = [[0.0]] * len(eps)
    eps.to_parquet(rewrite(ep_file), index=False)
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(rewrite(out / "meta/tasks.parquet"))

    for _, e in eps.iterrows():
        path = out / info["data_path"].format(chunk_index=e["data/chunk_index"], file_index=e["data/file_index"])
        d = pd.read_parquet(path)
        assert set(d.episode_index) == {e.episode_index}, path
        d["task_index"] = task_id[task[int(e.episode_index)]]
        d.to_parquet(rewrite(path), index=False)

    provenance = json.loads((root / "meta/provenance.json").read_text())
    for p in provenance:
        e = p["episode_index"]
        if e in new_task and new_task[e] != old_task[e]:
            p.update(task=new_task[e], retask_from=old_task[e])
    (rewrite(out / "meta/provenance.json")).write_text(json.dumps(provenance, indent=2))
    changed = sum(new_task[e] != old_task[e] for e in new_task)
    print(f"{out}: {changed} episodes retasked, {len(tasks)} tasks")


if __name__ == "__main__":
    main()

"""Merge finished ReBot roots (build_root + finalize + speed + gripper events) into one new root. Sources are read only.
Never overwrites.

    uv run python -m lerobot.annotation.rebot.merge_roots outputs/<root_a> outputs/<root_b> [...] --out outputs/<root>

Episodes keep their order (root by root). Videos, depth PNGs and gripper QA images are hardlinked (one data / video file
per episode, renumbered); the data files are rewritten with episode, index and task offsets. Every meta table with an
``episode_index`` column is concatenated with the episode offset and the global-index columns shifted by the frame
offset; tasks and subtasks are re-indexed over the union; subtask_windows.json, provenance.json and the speed trace
follow the same offsets; references/ is the union (a name in two roots must be identical). The ``*_info.json`` files keep
the first root's fields and list every root's file under ``merged_from``. Run verify_root on the result.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import load_stats, write_stats

GLOBAL = ["index", "from_index", "to_index", "raw_from_index", "raw_to_index", "commit_index"]
SKIP = {"info.json", "stats.json", "cleanup_verification.json", "provenance.json", "subtask_windows.json"}


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def read_index(root, name, column):
    t = pd.read_parquet(root / f"meta/{name}.parquet")
    return list(t.index) if column in t.columns else list(t[column])


def episode_table(root):
    return pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    roots, out = [r.resolve() for r in args.roots], args.out.resolve()
    assert len(roots) >= 2 and not out.exists(), out
    infos = [json.loads((r / "meta/info.json").read_text()) for r in roots]
    assert all(i["features"] == infos[0]["features"] and i["fps"] == infos[0]["fps"] for i in infos), "features differ"
    vkeys = [k for k, v in infos[0]["features"].items() if v["dtype"] == "video"]
    meta_files = [sorted(p.name for p in (r / "meta").iterdir() if p.is_file()) for r in roots]
    for r, names in zip(roots, meta_files, strict=True):
        assert set(names) - {"online_labels.parquet"} == set(meta_files[0]) - {"online_labels.parquet"}, f"{r.name}: meta files differ"
    meta_names = sorted(set().union(*meta_files))

    tasks = list(dict.fromkeys(t for r in roots for t in pd.read_parquet(r / "meta/tasks.parquet").index))
    subtasks = sorted({s for r in roots for s in pd.read_parquet(r / "meta/subtasks.parquet").index})
    task_id, subtask_id = {t: i for i, t in enumerate(tasks)}, {s: i for i, s in enumerate(subtasks)}

    episodes, tables, windows, provenance, traces = [], {}, {}, [], []
    ep_off = frame_off = 0
    for r, info in zip(roots, infos, strict=True):
        task_map = {i: task_id[t] for t, i in pd.read_parquet(r / "meta/tasks.parquet")["task_index"].items()}
        eps = episode_table(r)
        assert list(eps.episode_index) == list(range(len(eps))) and len(eps) == info["total_episodes"]
        assert eps.groupby("data/file_index").size().eq(1).all(), f"{r.name}: expected one data file per episode"
        for _, e in eps.iterrows():
            new = ep_off + int(e.episode_index)
            d = pd.read_parquet(r / info["data_path"].format(chunk_index=e["data/chunk_index"], file_index=e["data/file_index"]))
            d["episode_index"] += ep_off
            d["index"] += frame_off
            d["task_index"] = d["task_index"].map(task_map)
            dst = out / infos[0]["data_path"].format(chunk_index=0, file_index=new)
            dst.parent.mkdir(parents=True, exist_ok=True)
            d.to_parquet(dst, index=False)
            for k in vkeys:
                src = info["video_path"].format(video_key=k, chunk_index=e[f"videos/{k}/chunk_index"], file_index=e[f"videos/{k}/file_index"])
                link(r / src, out / infos[0]["video_path"].format(video_key=k, chunk_index=0, file_index=new))
            for dpath in sorted((r / "depth").glob(f"*/episode-{int(e.episode_index):06d}")):
                for png in dpath.iterdir():
                    link(png, out / "depth" / dpath.parent.name / f"episode-{new:06d}" / png.name)
            qa = r / f"meta/depth_gripper_event_qa/episode_{int(e.episode_index):04d}.png"
            if qa.exists():
                link(qa, out / f"meta/depth_gripper_event_qa/episode_{new:04d}.png")
        eps["episode_index"] += ep_off
        eps["data/chunk_index"], eps["data/file_index"] = 0, eps["episode_index"]
        for k in vkeys:
            eps[f"videos/{k}/chunk_index"], eps[f"videos/{k}/file_index"] = 0, eps["episode_index"]
        eps["meta/episodes/chunk_index"], eps["meta/episodes/file_index"] = 0, 0
        eps[["dataset_from_index", "dataset_to_index"]] += frame_off
        for stat in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
            eps[f"stats/episode_index/{stat}"] = eps[f"stats/episode_index/{stat}"].map(lambda v: v + ep_off)
            eps[f"stats/index/{stat}"] = eps[f"stats/index/{stat}"].map(lambda v: v + frame_off)
            eps[f"stats/task_index/{stat}"] = eps[f"stats/task_index/{stat}"].map(lambda v: np.vectorize(task_map.get)(v.astype(int)).astype(v.dtype))
        episodes.append(eps)

        for name in meta_names:
            if name.endswith(".parquet") and name not in ("tasks.parquet", "subtasks.parquet"):
                path = r / "meta" / name
                if not path.exists():
                    continue
                t = pd.read_parquet(path)
                assert "episode_index" in t.columns, name
                t["episode_index"] += ep_off
                for c in GLOBAL:
                    if c in t.columns:
                        t[c] += frame_off
                tables.setdefault(name, []).append(t)
        sw = json.loads((r / "meta/subtask_windows.json").read_text())
        for e, rows in sw["episodes"].items():
            windows[str(int(e) + ep_off)] = [{**w, "from_index": w["from_index"] + frame_off, "to_index": w["to_index"] + frame_off} for w in rows]
        provenance += [{**p, "episode_index": p["episode_index"] + ep_off, "merged_from": str(r)} for p in json.loads((r / "meta/provenance.json").read_text())]
        traces.append(np.load(r / "meta/speed_hybrid_v1_trace.npz"))
        ep_off += info["total_episodes"]
        frame_off += info["total_frames"]

    meta = out / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    pd.concat(episodes, ignore_index=True).to_parquet(meta / "episodes/chunk-000/file-000.parquet", index=False)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_parquet(meta / name, index=False)
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(meta / "tasks.parquet")
    pd.DataFrame({"subtask_index": range(len(subtasks))}, index=pd.Index(subtasks, name="subtask")).to_parquet(meta / "subtasks.parquet")

    # the trace has one entry per state transition: n - 1 per root, joined by one invalid transition across the boundary
    gap = {k: np.full(1, np.nan if np.issubdtype(t.dtype, np.floating) else 0, dtype=t.dtype) for k in traces[0].files if (t := traces[0][k]).ndim}
    merged = {k: np.concatenate([x for t in traces for x in (gap[k], t[k])][1:]) for k in gap}
    np.savez_compressed(meta / "speed_hybrid_v1_trace.npz", **merged, native_rate_hz=traces[0]["native_rate_hz"])
    assert all(len(v) == frame_off - 1 for v in merged.values())

    first_sw = json.loads((roots[0] / "meta/subtask_windows.json").read_text())
    first_sw["annotator"] = [json.loads((r / "meta/subtask_windows.json").read_text())["annotator"] for r in roots]
    (meta / "subtask_windows.json").write_text(json.dumps({**first_sw, "episodes": windows}, indent=2))
    (meta / "provenance.json").write_text(json.dumps(provenance, indent=2))
    for name in meta_files[0]:
        if name.endswith(".json") and name not in SKIP:
            parts = [json.loads((r / "meta" / name).read_text()) for r in roots]
            if all(p == parts[0] for p in parts):
                (meta / name).write_text(json.dumps(parts[0], indent=2))
            else:
                (meta / name).write_text(json.dumps({**parts[0], "merged_from": {str(r): p for r, p in zip(roots, parts, strict=True)}}, indent=2))
    write_stats(aggregate_stats([load_stats(r) for r in roots]), out)
    info = {**infos[0], "total_episodes": ep_off, "total_frames": frame_off, "total_tasks": len(tasks), "splits": {"train": f"0:{ep_off}"}}
    (meta / "info.json").write_text(json.dumps(info, indent=4))

    for r in roots:
        for ref in (r / "references").glob("*.json"):
            dst = out / "references" / ref.name
            if dst.exists():
                assert dst.read_bytes() == ref.read_bytes(), f"references/{ref.name} differs between roots"
            else:
                link(ref, dst)
    rows = "\n".join(f"| {i} | {r} | {inf['total_episodes']} | {inf['total_frames']} |" for i, (r, inf) in enumerate(zip(roots, infos, strict=True)))
    (out / "README.md").write_text(
        f"# {out.name}\n\nMerge of finished roots (`lerobot.annotation.rebot.merge_roots`), episodes in this order. Sources untouched; "
        f"per-episode origin in meta/provenance.json (`merged_from`).\n\n| # | root | episodes | frames |\n|---|---|---|---|\n{rows}\n\n"
        f"Total: {ep_off} episodes, {frame_off} frames.\n"
    )
    print(f"{out}: {ep_off} episodes, {frame_off} frames, {len(tasks)} tasks, {len(subtasks)} subtasks")


if __name__ == "__main__":
    main()

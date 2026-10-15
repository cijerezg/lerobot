"""Append or replace validation episodes, preserving source labels and media under index remapping.

Usage: python -m lerobot.annotation.rebot.replace_validation_episode --base BASE
    --extra EXTRA --exclude 2 [--extra-episodes 3 5] [--split train|val] --out DEST
--exclude drops base episodes; --extra-episodes keeps only those extra episodes (default all);
--split names the split written to provenance, split_status and info (default val).
The sources are read-only. RGB/depth files are hardlinked; tabular indices are rebuilt.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lerobot.datasets.aggregate import merge_video_feature_info_for_aggregate
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

CAMERAS = {"observation.images.top": "observation.images.external_0", "observation.images.wrist": "observation.images.wrist_0"}
GLOBAL_COLUMNS = {"index", "from_index", "to_index", "raw_from_index", "raw_to_index", "commit_index"}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda v: np.asarray(v).tolist()) + "\n")


def table(root, folder):
    return pd.concat([pd.read_parquet(p) for p in sorted((root / folder).rglob("*.parquet"))], ignore_index=True)


def canonical(key):
    for old, new in CAMERAS.items():
        # Already canonical wrist_0 must not acquire another suffix.
        if key == old or key.startswith("stats/" + old + "/") or key.startswith("videos/" + old + "/"):
            return key.replace(old, new, 1)
    return key


def numeric_array(value):
    if isinstance(value, (np.ndarray, list, tuple)):
        return np.asarray([numeric_array(v) for v in value])
    return value


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src.resolve(), dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    parser.add_argument("--extra-episodes", type=int, nargs="*", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = [args.base.resolve(), args.extra.resolve()]
    out = args.out.resolve()
    work = out.with_name(out.name + ".building")
    assert not out.exists() and not work.exists(), "Refusing to overwrite a dataset or unfinished build"
    infos = {r: read(r / "meta/info.json") for r in roots}
    eps = {r: table(r, "meta/episodes") for r in roots}
    assert set(args.exclude) <= set(eps[roots[0]].episode_index)
    assert args.extra_episodes is None or set(args.extra_episodes) <= set(eps[roots[1]].episode_index)
    assert infos[roots[0]]["fps"] == infos[roots[1]]["fps"]
    assert infos[roots[0]]["robot_type"] == infos[roots[1]]["robot_type"]
    features = {r: {canonical(k): v for k, v in infos[r]["features"].items()} for r in roots}
    assert set(features[roots[0]]) == set(features[roots[1]])
    for k, v in features[roots[0]].items():
        assert (v["dtype"], v["shape"]) == (features[roots[1]][k]["dtype"], features[roots[1]][k]["shape"]), k
    selections = [(r, row) for r in roots for _, row in eps[r].iterrows()
                  if (r != roots[0] or int(row.episode_index) not in args.exclude)
                  and (r != roots[1] or args.extra_episodes is None or int(row.episode_index) in args.extra_episodes)]
    work.mkdir(parents=True)
    (work / "meta").mkdir()
    vocab = {}
    for name, column in [("tasks", "task_index"), ("subtasks", "subtask_index")]:
        joined = pd.read_parquet(roots[0] / f"meta/{name}.parquet").copy()
        for label in pd.read_parquet(roots[1] / f"meta/{name}.parquet").index:
            if label not in joined.index:
                joined.loc[label, column] = int(joined[column].max()) + 1 if len(joined) else 0
        joined[column] = joined[column].astype("int64")
        joined.to_parquet(work / f"meta/{name}.parquet")
        vocab[name] = joined[column]
    source_tasks = {r: pd.read_parquet(r / "meta/tasks.parquet").task_index for r in roots}
    source_subtasks = {r: pd.read_parquet(r / "meta/subtasks.parquet").subtask_index for r in roots}
    task_maps = {r: {int(idx): int(vocab["tasks"].loc[label]) for label, idx in source_tasks[r].items()} for r in roots}
    subtask_maps = {r: {int(idx): int(vocab["subtasks"].loc[label]) for label, idx in source_subtasks[r].items()} for r in roots}
    manifest, rows, datasets, video_stats = [], [], [], []
    offset = 0
    for new_ep, (r, original) in enumerate(selections):
        old_ep, old_start, length = int(original.episode_index), int(original.dataset_from_index), int(original.length)
        data = pd.read_parquet(r / infos[r]["data_path"].format(chunk_index=int(original["data/chunk_index"]), file_index=int(original["data/file_index"])))
        data = data[data.episode_index.eq(old_ep)].copy()
        assert len(data) == length and data["index"].tolist() == list(range(old_start, old_start + length))
        data["episode_index"] = new_ep
        data["index"] += offset - old_start
        data["task_index"] = data.task_index.map(task_maps[r]).astype("int64")
        dst = work / f"data/chunk-000/file-{new_ep:03d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(dst, index=False)
        datasets.append(data)
        row = {canonical(k): v for k, v in original.items()}
        row.update(episode_index=new_ep, dataset_from_index=offset, dataset_to_index=offset + length)
        row.update({"data/chunk_index": 0, "data/file_index": new_ep,
                    "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        episode_video_stats = {}
        for old_key, feature in infos[r]["features"].items():
            if feature["dtype"] != "video":
                continue
            new_key = canonical(old_key)
            prefix = "videos/" + old_key
            src = r / infos[r]["video_path"].format(video_key=old_key, chunk_index=int(original[prefix + "/chunk_index"]), file_index=int(original[prefix + "/file_index"]))
            link(src, work / f"videos/{new_key}/chunk-000/file-{new_ep:03d}.mp4")
            row[f"videos/{new_key}/chunk_index"] = 0
            row[f"videos/{new_key}/file_index"] = new_ep
            # Aggregate common mandatory video statistics, not optional quantiles absent in older sources.
            episode_video_stats[new_key] = {s: numeric_array(row[f"stats/{new_key}/{s}"]) for s in ("min", "max", "mean", "std", "count")}
        video_stats.append(episode_video_stats)
        for camera in (r / "depth").iterdir():
            src_ep = camera / f"episode-{old_ep:06d}"
            if src_ep.exists():
                name = "wrist_0.depth" if camera.name == "wrist.depth" else camera.name
                for p in src_ep.rglob("*"):
                    if p.is_file():
                        link(p, work / "depth" / name / f"episode-{new_ep:06d}" / p.relative_to(src_ep))
        changed_stats = compute_episode_stats({k: data[k].to_numpy() for k in ("index", "episode_index", "task_index")}, {k: features[r][k] for k in ("index", "episode_index", "task_index")})
        for key, values in changed_stats.items():
            for stat, value in values.items():
                row[f"stats/{key}/{stat}"] = value.tolist()
        rows.append(row)
        manifest.append(dict(source=str(r), source_episode=old_ep, episode_index=new_ep, old_from_index=old_start,
                             new_from_index=offset, length=length, camera_map={k: canonical(k) for k in infos[r]["features"] if canonical(k) != k}))
        offset += length
    # Retain all source per-episode stats, including optional quantiles (null where unavailable).
    dst = work / "meta/episodes/chunk-000/file-000.parquet"
    dst.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(dst, index=False)
    metadata_files = sorted(set(p.name for r in roots for p in (r / "meta").glob("*.parquet")) - {"tasks.parquet", "subtasks.parquet"})
    row_counts = {}
    for filename in metadata_files:
        parts = []
        for m in manifest:
            r = Path(m["source"])
            src = r / "meta" / filename
            if not src.exists():
                continue
            t = pd.read_parquet(src)
            assert "episode_index" in t, filename
            t = t[t.episode_index.eq(m["source_episode"])].copy()
            t["episode_index"] = m["episode_index"]
            for col in GLOBAL_COLUMNS.intersection(t.columns):
                # Preserve negative sentinels.
                mask = t[col].notna() & t[col].ge(0)
                t.loc[mask, col] += m["new_from_index"] - m["old_from_index"]
            for col, mapping in [("task_index", task_maps[r]), ("subtask_index", subtask_maps[r])]:
                if col in t:
                    t[col] = t[col].map(mapping).astype(t[col].dtype)
            parts.append(t)
        if not parts:  # the table exists in a source but none of the chosen episodes has rows (online labels of a teleop root)
            parts = [pd.read_parquet(next(r / "meta" / filename for r in roots if (r / "meta" / filename).exists())).iloc[:0]]
        result = pd.concat(parts, ignore_index=True, sort=False)
        result.to_parquet(work / "meta" / filename, index=False)
        row_counts[filename] = len(result)
    windows = dict(model="assistant-vision-review", created_date="2026-10-11", top_key="observation.images.external_0",
                   wrist_key="observation.images.wrist_0", interval_seconds=None, episodes={},
                   annotation_sources=[str(r / "meta/subtask_windows.json") for r in roots])
    provenance = []
    for m in manifest:
        r = Path(m["source"])
        windows["episodes"][str(m["episode_index"])] = [dict(w, from_index=w["from_index"] + m["new_from_index"] - m["old_from_index"], to_index=w["to_index"] + m["new_from_index"] - m["old_from_index"])
            for w in read(r / "meta/subtask_windows.json")["episodes"][str(m["source_episode"])]]
        p = next(copy.deepcopy(p) for p in read(r / "meta/provenance.json") if p["episode_index"] == m["source_episode"])
        p.update(episode_index=m["episode_index"], split=args.split, immediate_source=str(r), immediate_source_episode=m["source_episode"])
        provenance.append(p)
    write(work / "meta/subtask_windows.json", windows)
    write(work / "meta/provenance.json", provenance)
    # Trace arrays describe transitions f -> f+1. Never connect episodes or fabricate absent labels.
    traces = {r: dict(np.load(r / "meta/speed_hybrid_v1_trace.npz")) for r in roots}
    assert all(set(z) == set(traces[roots[0]]) for z in traces.values())
    merged = {}
    for key, array in traces[roots[0]].items():
        if array.ndim == 0:
            assert all(np.array_equal(z[key], array) for z in traces.values())
            merged[key] = array.copy()
            continue
        default = np.nan if np.issubdtype(array.dtype, np.floating) else 0
        merged[key] = np.full((offset - 1, *array.shape[1:]), default, dtype=array.dtype)
        for m in manifest:
            src = traces[Path(m["source"])][key]
            start, n = m["old_from_index"], m["length"] - 1
            if start + n <= len(src):
                merged[key][m["new_from_index"]:m["new_from_index"] + n] = src[start:start+n]
    np.savez_compressed(work / "meta/speed_hybrid_v1_trace.npz", **merged)
    # Copy schemas/rubric definitions with fresh provenance, not stale coverage or QA summaries.
    for filename in ("contact_info.json", "precision_info.json", "speed_info.json", "speed_hybrid_v1_info.json", "depth_gripper_event_labels_info.json", "metadata_info.json"):
        sources = {str(r): read(r / "meta" / filename) for r in roots if (r / "meta" / filename).exists()}
        common = copy.deepcopy(next(iter(sources.values())))
        for key in ("generated", "created_date", "rows", "shares", "manual_speed", "segments", "mistakes", "quality_spans", "segment_quality", "annotator", "method", "command", "creation_date", "event_counts", "row_counts", "target_statistics", "qa_plots", "date", "new_root", "labels_dir", "counts"):
            common.pop(key, None)
        common.update(created_date="2026-10-11", source_metadata=sources, dataset_row_counts=row_counts,
                      annotation_coverage="Preserved per source; the legacy bit-kit episode from the earlier probe recording has console subtasks and depth events but lacks manual quality/speed/precision/contact annotations.")
        if filename in ("precision_info.json", "contact_info.json"):
            channel = filename.removesuffix("_info.json")
            channel_table = pd.read_parquet(work / "meta" / (channel + ".parquet"))
            common.update(counts={"rows": len(channel_table), "per_value_all": {str(k): int(v) for k, v in channel_table[channel].value_counts().sort_index().items()}})
        if filename == "speed_hybrid_v1_info.json":
            common.update(method="hybrid_motion_duration_v1", rows=row_counts["speed_hybrid_v1.parquet"], trace="speed_hybrid_v1_trace.npz", trace_note="Internal source transitions preserved; episode joins and absent bit-kit labels masked invalid.")
        if filename == "depth_gripper_event_labels_info.json":
            events = pd.read_parquet(work / "meta/depth_gripper_events.parquet")
            common.update(event_counts={str(k): int(v) for k, v in events.event_type.value_counts().items()}, row_counts={"event_rows": len(events), "frame_label_rows": row_counts["depth_gripper_event_labels.parquet"], "source_frames": offset})
        if filename == "metadata_info.json":
            common.update(segments=row_counts["episode_metadata.parquet"], mistakes=row_counts["mistakes.parquet"], quality_spans=row_counts["quality_spans.parquet"], method="Source annotations retained under episode/frame remapping; see source_metadata for each review method.")
        write(work / "meta" / filename, common)
    info = copy.deepcopy(infos[roots[0]])
    info["features"] = merge_video_feature_info_for_aggregate([SimpleNamespace(features=features[r]) for r in roots])
    for key, feature in info["features"].items():
        if feature["dtype"] == "video" and features[roots[0]][key]["info"].get("video.codec") != features[roots[1]][key]["info"].get("video.codec"):
            feature["info"]["video.codec"] = None
    info.update(total_episodes=len(manifest), total_frames=offset, total_tasks=len(vocab["tasks"]), total_subtasks=len(vocab["subtasks"]), splits={args.split: f"0:{len(manifest)}"})
    write(work / "meta/info.json", info)
    full = pd.concat(datasets, ignore_index=True)
    low = {k: np.stack(full[k]) if v["shape"] != [1] else full[k].to_numpy() for k, v in info["features"].items() if v["dtype"] not in ("video", "image")}
    stats = aggregate_stats(video_stats)
    stats.update(compute_episode_stats(low, {k: info["features"][k] for k in low}))
    write(work / "meta/stats.json", stats)
    write(work / "meta/split_status.json", {"status": "assigned", "split": args.split})
    write(work / "REPLACEMENT.json", dict(base=str(roots[0]), extra=str(roots[1]), excluded_base_episodes=args.exclude, extra_episodes=args.extra_episodes, split=args.split,
          episodes=manifest, total_episodes=len(manifest), total_frames=offset, annotation_row_counts=row_counts,
          note="Original sources unchanged; preserved annotations remapped. Source row_id and uid remain unchanged as provenance identifiers."))
    assert full["index"].tolist() == list(range(offset))
    work.rename(out)
    print(f"Built {out}: {len(manifest)} episodes / {offset} frames")


if __name__ == "__main__":
    main()

"""Build a new ReBot root from kept ranges of source episodes. Sources are read only. Never overwrites.

Every row (one kept range of one source episode) becomes one episode. A kept range [a, b) with a % 3 == 0 and
(b - a) % 3 == 0 shifts the video from_timestamp by a / fps and renames the depth PNGs to the new episode-local
frame, so the depth phase (every third frame) survives. ``cuts`` (idle or no-progress stretches inside the
range, multiples of 3 from a) are removed: frames map through ``episode.remap``; an episode with a cut inside
the range is SPLICED (both camera videos re-encoded from the source frames with the source encoder settings:
av1, g 2, crf 30, preset 12); a cut at the end only shortens the range (videos stay hardlinks). Rollout
``online_labels.parquet`` rows are carried over (sliced, re-indexed).

Two inputs:

    # labelled pass: <work>/inventory.json + <work>/labels/<IDX>.json (segments, mistakes); writes the
    # subtask tables too. --split picks train, val or all (the staging root of the class pass). An episode
    # without its own "cuts" takes them from <work>/idle_cuts.json (idle_scan) once that file exists.
    uv run python -m lerobot.annotation.rebot.build_root <work> --out outputs/<root> --split train
    # cut plan only: <work>/plan.json = [{source, episode, keep, cuts, task, ...}]; no annotation tables.
    uv run python -m lerobot.annotation.rebot.build_root <work> --out outputs/<root> --plan
"""

import argparse
import copy
import json
import os
from datetime import date
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot.episode import kept, layout, records, remap
from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats


def plain(v):
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def json_write(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False))


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def encode_kept(src_video, t0, k, out, fps):
    """Re-encode the source frames k (episode-local, video starts at t0) into a new mp4, source encoder settings."""
    want, idx = set(int(f) for f in k), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    options = VideoEncoderConfig(preset=12).get_codec_options(None, as_strings=True)
    with av.open(str(src_video)) as c, av.open(str(out), "w") as o:
        s = c.streams.video[0]
        st = o.add_stream("libsvtav1", fps, options=options)
        st.pix_fmt = "yuv420p"
        st.width, st.height = s.width, s.height
        c.seek(int(max(0, t0 + int(k[0]) / fps - 0.5) / s.time_base), stream=s)
        for frame in c.decode(s):
            i = round((float(frame.pts * s.time_base) - t0) * fps)
            if i in want:
                assert i == k[idx], (out, i, k[idx])
                idx += 1
                for packet in st.encode(av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")):
                    o.mux(packet)
            if i >= k[-1]:
                break
        for packet in st.encode():
            o.mux(packet)
    assert idx == len(k), (out, idx, len(k))


def rows_from_labels(work, split):
    """One row per kept range of the labelled pass, in inventory order."""
    work = Path(work)
    idle = json.loads((work / "idle_cuts.json").read_text())["cuts"] if (work / "idle_cuts.json").exists() else []
    out = []
    for r in records(work):
        d = json.loads((work / f"labels/{r['idx']:02d}.json").read_text())
        assert d["key"] == r["key"] and d["frames"] == r["frames"], r["key"]
        for j, e in enumerate(d["episodes"]):
            if split == "all" or e["split"] == split:
                provenance = dict(
                    source_key=r["key"],
                    inventory_idx=r["idx"],
                    part=j,
                    source_digest=r["digest"],
                    split=split,
                    kind=r["kind"],
                    flags=d.get("flags", []) + e.get("flags", []),
                )
                out.append(
                    dict(
                        source=r["source"],
                        episode=r["episode"],
                        key=r["key"],
                        frames=r["frames"],
                        keep=e["keep"],
                        cuts=e.get("cuts") or [c["cut"] for c in idle if c["idx"] == r["idx"] and c["part"] == j],
                        task=e["task"],
                        segments=e["segments"],
                        mistakes=e["mistakes"],
                        provenance=provenance,
                    )
                )
    return out


def rows_from_plan(work):
    """One row per entry of <work>/plan.json. Keys other than source, episode, keep, cuts, task go to provenance."""
    out = []
    for p in json.loads((Path(work) / "plan.json").read_text()):
        core = {k: p[k] for k in ("source", "episode", "keep", "cuts", "task")}
        extra = {k: v for k, v in p.items() if k not in core}
        out.append(dict(core, key=f"{p['source']} ep{p['episode']}", provenance=extra))
    return out


def build(rows, destination, split="train", annotator=None):
    """Rows with ``segments`` also get the subtask tables (subtasks, episode_metadata, mistakes, subtask_windows)."""
    destination = Path(destination)
    assert not destination.exists(), f"Refusing to overwrite {destination}"
    destination.mkdir(parents=True)
    meta = destination / "meta"
    meta.mkdir()
    labelled = "segments" in rows[0]
    keys, depth = layout(WORKSPACE / rows[0]["source"])
    info = json.loads((WORKSPACE / rows[0]["source"] / "meta/info.json").read_text())
    fps = info["fps"]
    features = copy.deepcopy(info["features"])
    columns = [k for k, v in features.items() if v["dtype"] != "video"]
    tasks = sorted({r["task"] for r in rows})
    task_index = {v: k for k, v in enumerate(tasks)}
    ep_rows, seg_rows, mistake_rows, prov, online, windows, all_stats, all_data = [], [], [], [], [], {}, [], []
    video_files, offset = {}, 0
    for ep, row in enumerate(rows):
        src, sep, (a, b), cuts = WORKSPACE / row["source"], row["episode"], row["keep"], row["cuts"]
        sdf = pd.concat([pd.read_parquet(p) for p in sorted((src / "data").rglob("*.parquet"))], ignore_index=True)
        df = sdf[sdf.episode_index == sep].sort_values("frame_index").reset_index(drop=True)
        assert (df.frame_index.values == np.arange(len(df))).all() and 0 <= a < b <= len(df), (row["key"], a, b)
        assert "frames" not in row or len(df) == row["frames"], row["key"]
        k = kept((a, b), cuts)
        spliced = bool(len(k)) and k[-1] - k[0] + 1 != len(k)
        # depth phase survives
        assert a % 3 == 0 and len(k) % 3 == 0 and all((x - a) % 3 == 0 for c in cuts for x in c), (row["key"], a, b)
        df = df.iloc[k].copy().reset_index(drop=True)[columns]
        n = len(df)
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["timestamp"] = (np.arange(n) / fps).astype(np.float32)
        df["episode_index"] = np.int64(ep)
        df["index"] = np.arange(offset, offset + n, dtype=np.int64)
        df["task_index"] = np.int64(task_index[row["task"]])
        if labelled:
            windows[str(ep)] = []
            for i, s in enumerate(row["segments"]):
                f0, f1 = offset + remap(k, s["from_index"]), offset + remap(k, s["to_index"])
                assert f0 < f1, (row["key"], s["subtask"], "segment wholly inside a cut")
                seg_rows.append(
                    dict(
                        episode_index=ep,
                        segment_index=i,
                        from_index=f0,
                        to_index=f1,
                        subtask=s["subtask"],
                        quality=0,
                        vision_reviewed=True,
                        anchor_baseline=False,
                        note=s["what_happens"],
                    )
                )
                windows[str(ep)].append(dict(from_index=f0, to_index=f1, subtask=s["subtask"]))
            assert seg_rows[-1]["to_index"] == offset + n and windows[str(ep)][0]["from_index"] == offset
            for m in row["mistakes"]:
                m0, m1 = offset + remap(k, m["from_index"]), offset + remap(k, m["to_index"])
                if m0 == m1:
                    continue
                mistake_rows.append(
                    dict(
                        episode_index=ep,
                        from_index=m0,
                        to_index=m1,
                        mistake=True,
                        mistake_type=m["type"],
                        note=m["what_happens"],
                    )
                )
        olp = src / "meta/online_labels.parquet"
        o = pd.read_parquet(olp) if olp.exists() else None
        o = o[o.episode_index == sep] if o is not None else None
        if o is not None and len(o):  # a teleop episode of a mixed root has no rows
            o = o.sort_values("frame_index").iloc[k]
            online.append(
                pd.DataFrame(
                    dict(
                        episode_index=np.int64(ep),
                        frame_index=np.arange(n, dtype=np.int64),
                        index=df["index"].values,
                        is_intervention=o.is_intervention.values,
                        recorded_subtask=next((o[c].values for c in ("subtask", "recorded_subtask") if c in o), [""] * n),
                    )
                )
            )
        path = destination / f"data/chunk-000/file-{ep:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        em = pd.concat([pd.read_parquet(p) for p in sorted((src / "meta/episodes").rglob("*.parquet"))])
        original = em[em.episode_index == sep].iloc[0]
        er = {c: v for c, v in original.items() if not c.startswith("stats/")}
        er.update(episode_index=ep, tasks=[row["task"]], length=n, dataset_from_index=offset, dataset_to_index=offset + n)
        er.update({"data/chunk_index": 0, "data/file_index": ep, "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        for key_ in keys:
            prefix = f"videos/{key_}"
            chunk, file = int(original[prefix + "/chunk_index"]), int(original[prefix + "/file_index"])
            old = src / prefix / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
            t0 = float(original[prefix + "/from_timestamp"])
            key = f"{old}#spliced-ep{ep}" if spliced else str(old)
            if key not in video_files:
                file_id = sum(f"/{key_}/" in v for v in video_files)
                new = destination / prefix / f"chunk-000/file-{file_id:03d}.mp4"
                if spliced:
                    encode_kept(old, t0, k, new, fps)
                else:
                    link(old, new)
                video_files[key] = file_id
            er[prefix + "/chunk_index"] = 0
            er[prefix + "/file_index"] = video_files[key]
            if spliced:
                er[prefix + "/from_timestamp"], er[prefix + "/to_timestamp"] = 0.0, n / fps
            else:
                er[prefix + "/from_timestamp"] = t0 + int(k[0]) / fps
                er[prefix + "/to_timestamp"] = t0 + (int(k[-1]) + 1) / fps
        keep_set = set(int(f) for f in k)
        for p in sorted((src / f"depth/{depth}/episode-{sep:06d}").glob("*.png")):
            f = int(p.stem.split("-")[1])
            if f in keep_set:
                link(p, destination / f"depth/{depth}/episode-{ep:06d}" / f"frame-{remap(k, f):06d}.png")
        numeric = {c: (np.stack(df[c]) if c in ("action", "observation.state") else df[c].to_numpy()) for c in df.columns}
        stats = compute_episode_stats(numeric, {c: features[c] for c in columns})
        for key in keys:
            stats[key] = {
                c.rsplit("/", 1)[1]: np.asarray(plain(original[c]), dtype=np.float64)
                for c in original.index
                if c.startswith(f"stats/{key}/")
            }
        for key, values in stats.items():
            for name, value in values.items():
                er[f"stats/{key}/{name}"] = plain(value)
        all_stats.append(stats)
        all_data.append(df)
        ep_rows.append(er)
        prov.append(
            dict(
                episode_index=ep,
                source=str(src),
                source_episode=sep,
                keep=[a, b],
                cuts=[list(c) for c in cuts],
                spliced=spliced,
                task=row["task"],
                **row["provenance"],
            )
        )
        offset += n
        print(split, ep, row["key"], [a, b], "cuts", cuts, n, "frames", "SPLICED" if spliced else "", flush=True)
    (meta / "episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame(ep_rows).to_parquet(meta / "episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(meta / "tasks.parquet")
    if labelled:
        subtasks = sorted({s["subtask"] for r in rows for s in r["segments"]})
        pd.DataFrame({"subtask_index": range(len(subtasks))}, index=pd.Index(subtasks, name="subtask")).to_parquet(
            meta / "subtasks.parquet"
        )
        pd.DataFrame(seg_rows).to_parquet(meta / "episode_metadata.parquet", index=False)
        mistake_columns = ["episode_index", "from_index", "to_index", "mistake", "mistake_type", "note"]
        mistake_types = {"episode_index": "int64", "from_index": "int64", "to_index": "int64", "mistake": "bool"}
        pd.DataFrame(mistake_rows, columns=mistake_columns).astype(mistake_types).to_parquet(
            meta / "mistakes.parquet", index=False
        )
        json_write(
            meta / "subtask_windows.json",
            dict(
                model="assistant-vision-review",
                annotator=annotator,
                created_date=date.today().isoformat(),
                interval_seconds=None,
                top_key=keys[0],
                wrist_key=keys[1],
                episodes=windows,
            ),
        )
        json_write(meta / "split_status.json", dict(status="assigned", split=split))
    if online:
        pd.concat(online, ignore_index=True).to_parquet(meta / "online_labels.parquet", index=False)
    full = pd.concat(all_data, ignore_index=True)
    stats = aggregate_stats(all_stats)
    whole = {c: (np.stack(full[c]) if c in ("action", "observation.state") else full[c].to_numpy()) for c in full.columns}
    stats.update(compute_episode_stats(whole, {c: features[c] for c in columns}))
    json_write(meta / "stats.json", stats)
    info.update(
        features=features,
        total_episodes=len(rows),
        total_frames=offset,
        total_tasks=len(tasks),
        splits={"train" if split == "all" else split: f"0:{len(rows)}"},
    )
    json_write(meta / "info.json", info)
    json_write(meta / "provenance.json", prov)
    print("DONE", destination, len(rows), "episodes", offset, "frames")
    return destination


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    if args.plan:
        build(rows_from_plan(args.work), WORKSPACE / args.out)
    else:
        build(rows_from_labels(args.work, args.split), WORKSPACE / args.out, args.split, f"{args.work}/labels/*.json")

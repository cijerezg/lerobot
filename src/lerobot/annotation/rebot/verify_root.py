"""Acceptance checks on a root built by build_root, read from its meta/provenance.json.

Every root: kept-frame action/state equal to the source; online_labels aligned; videos hardlinked with a shifted
from_timestamp, or re-encoded when spliced and matched to the source frame on both sides of every splice; depth
PNGs hardlinked, renumbered through the cut map, one every third frame; the actual reader on three frames per
episode. A root that carries the annotation tables (meta/quality_spans.parquet) also gets: dense subtask
coverage, speed / precision / contact row-aligned, spans and windows inside their episode, every mistake inside
a grade 1-2 span and a quality <= 2 segment, ReplayBuffer materialization.

    uv run python -m lerobot.annotation.rebot.verify_root outputs/<root>     -> <root>/meta/cleanup_verification.json
"""

import argparse
import json
import os
from pathlib import Path

from lerobot.annotation.paths import WORKSPACE

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_DATASETS_CACHE"] = str(WORKSPACE / "outputs/_annotation/loader_cache")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lerobot.annotation.rebot.episode import CAMS, decode, kept, remap  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.rl.buffer import ReplayBuffer  # noqa: E402
from lerobot.rl.offline_dataset_utils import _subtask_indices_from_windows, load_metadata_rows  # noqa: E402
from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png  # noqa: E402


def check_annotations(root, n, eps, dataset):
    """Table checks of a finished annotated root; returns the counts for the report."""
    meta = root / "meta"
    subtasks = pd.read_parquet(meta / "subtasks.parquet")
    seg, mist, speed, prec, cont = load_metadata_rows(root)  # seg = segment pieces of constant frame grade
    assert prec is not None and cont is not None
    segments = pd.read_parquet(meta / "episode_metadata.parquet").to_dict("records")
    coverage = np.zeros(n, np.int8)
    expected = []
    for s in seg:
        coverage[s["from_index"] : s["to_index"]] += 1
        assert 1 <= s["quality"] <= 5 and s["note"]
        expected.append((s["from_index"], s["to_index"], int(subtasks.loc[s["subtask"], "subtask_index"])))
    assert (coverage == 1).all()
    keys = ["episode_index", "segment_index", "from_index", "to_index"]
    # the loader rebuilds its precision rows from the windows, so alignment is checked on the table itself
    precision = pd.read_parquet(meta / "precision.parquet").to_dict("records")
    for table in (speed, precision, cont):
        assert len(table) == len(segments)
        assert all(all(a[k] == b[k] for k in keys) for a, b in zip(segments, table, strict=True))
    assert all(1 <= r["speed"] <= 5 for r in speed) and all(1 <= r["precision"] <= 5 for r in precision + prec)
    spans = pd.read_parquet(meta / "quality_spans.parquet")
    windows = pd.read_parquet(meta / "precision_windows.parquet")
    eb = eps.set_index("episode_index")
    for t in (spans, windows):
        lo, hi = eb.loc[t.episode_index].dataset_from_index.values, eb.loc[t.episode_index].dataset_to_index.values
        assert ((t.from_index >= lo) & (t.to_index <= hi) & (t.from_index < t.to_index)).all()
    for m in mist:  # rubric 3.2 / 9: every mistake inside a span of grade 1 or 2; its host segment quality <= 2
        low = spans[
            (spans.episode_index == m["episode_index"])
            & (spans.quality <= 2)
            & (spans.raw_from_index <= m["from_index"])
            & (spans.raw_to_index >= m["to_index"])
        ]
        host = [
            s
            for s in seg
            if s["episode_index"] == m["episode_index"] and s["from_index"] <= m["from_index"] < s["to_index"]
        ]
        assert len(low) and len(host) == 1 and host[0]["quality"] <= 2, m
    buf = ReplayBuffer(capacity=n, device="cpu", storage_device="cpu")
    buf.size = n
    buf.complementary_info = {}
    buf.complementary_info_keys = []
    buf.materialize_metadata(seg, mist, speed, prec, cont)
    q = buf.complementary_info["metadata_quality"].float().numpy()
    mk = buf.complementary_info["metadata_mistake"].float().numpy()
    assert (q >= 1).all() and (q <= 5).all()
    derived = _subtask_indices_from_windows(dataset, n).numpy().reshape(-1)
    assert all((derived[a:b] == i).all() for a, b, i in expected)
    return dict(
        segments=len(segments),
        quality_pieces=len(seg),
        mistakes=len(mist),
        mistake_frames=int(mk.sum()),
        quality_spans=len(spans),
        precision_windows=len(windows),
        subtasks=len(subtasks),
        frame_quality={str(k): int((q == k).sum()) for k in range(1, 6)},
    )


def check(root, report=None):
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    prov = json.loads((meta / "provenance.json").read_text())
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
    eps = pd.concat([pd.read_parquet(p) for p in sorted((meta / "episodes").rglob("*.parquet"))], ignore_index=True)
    tasks = pd.read_parquet(meta / "tasks.parquet")
    n = len(data)
    assert n == info["total_frames"] and len(eps) == len(prov) == info["total_episodes"]
    assert np.array_equal(data["index"], np.arange(n))
    assert (eps.dataset_from_index % 3 == 0).all() and (eps.length % 3 == 0).all()
    online = pd.read_parquet(meta / "online_labels.parquet") if (meta / "online_labels.parquet").exists() else None
    dataset = LeRobotDataset("cijerezg/" + root.name, root=root, video_backend="pyav", download_videos=False)
    assert len(dataset) == n
    annotated = (meta / "quality_spans.parquet").exists()
    counts = check_annotations(root, n, eps, dataset) if annotated else {}
    samples = depth_files = splice_checks = 0
    for p, (_, e) in zip(prov, eps.iterrows(), strict=True):
        ep, source, sep, (a, b) = p["episode_index"], Path(p["source"]), p["source_episode"], p["keep"]
        kf = kept(p["keep"], p["cuts"])
        orig = pd.concat([pd.read_parquet(f) for f in sorted((source / "data").rglob("*.parquet"))])
        orig = orig[orig.episode_index == sep].sort_values("frame_index").reset_index(drop=True)
        orig = orig.iloc[kf].reset_index(drop=True)
        dst = data[data.episode_index == ep].reset_index(drop=True)
        assert len(dst) == len(kf) == e.length
        for k in ("action", "observation.state"):
            assert np.array_equal(np.stack(orig[k]), np.stack(dst[k]))
        assert np.array_equal(dst.frame_index.values, np.arange(len(dst)))
        assert (dst.task_index == tasks.loc[p["task"], "task_index"]).all() and list(e.tasks) == [p["task"]]
        if online is not None and (source / "meta/online_labels.parquet").exists():
            so = pd.read_parquet(source / "meta/online_labels.parquet")
            so = so[so.episode_index == sep].sort_values("frame_index").iloc[kf]
            o = online[online.episode_index == ep]
            assert np.array_equal(o.is_intervention.values, so.is_intervention.values)
            assert list(o.recorded_subtask) == list(so.subtask)
        sepi = pd.concat([pd.read_parquet(f) for f in (source / "meta/episodes").rglob("*.parquet")])
        sepi = sepi[sepi.episode_index == sep].iloc[0]
        for cam in CAMS:
            pre = f"videos/observation.images.{cam}"
            va = root / pre / f"chunk-{int(e[pre + '/chunk_index']):03d}" / f"file-{int(e[pre + '/file_index']):03d}.mp4"
            vb = source / pre / f"chunk-{int(sepi[pre + '/chunk_index']):03d}" / f"file-{int(sepi[pre + '/file_index']):03d}.mp4"
            if p["spliced"]:
                assert not os.path.samefile(va, vb) and e[pre + "/from_timestamp"] == 0
            else:
                assert os.path.samefile(va, vb)
                assert abs(e[pre + "/from_timestamp"] - (sepi[pre + "/from_timestamp"] + kf[0] / 30)) < 1e-6
        if p["spliced"]:  # the frames on both sides of each splice are the source frames (lossy re-encode)
            rec = dict(source=p["source"], episode=sep, key=f"{root.name} ep{ep}")
            probe = sorted({remap(kf, c0) - 1 for c0, c1 in p["cuts"] if c0 > a} | {remap(kf, c1) for c0, c1 in p["cuts"] if c1 < b})
            for local in probe:
                row = dataset[int(e.dataset_from_index) + local]
                for cam in CAMS:
                    f = int(kf[local])
                    src = np.asarray(decode(rec, cam, [f], (640, 480))[f], np.float32)
                    got = row[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255
                    err = np.abs(got - src).mean()
                    assert err < 6, (ep, cam, f, err)
                    splice_checks += 1
        pngs = sorted((root / f"depth/wrist.depth/episode-{ep:06d}").glob("*.png"))
        # one PNG every third frame, at the phase the episode was recorded with (load_depth_png tolerates it)
        local = [int(f.stem.split("-")[1]) for f in pngs]
        assert local[0] < 3 and local == list(range(local[0], len(dst), 3))
        for f in pngs:
            original = source / f"depth/wrist.depth/episode-{sep:06d}" / f"frame-{kf[int(f.stem.split('-')[1])]:06d}.png"
            assert os.path.samefile(f, original)
            depth_files += 1
        for local in (0, len(dst) // 2, len(dst) - 3):
            row = dataset[int(e.dataset_from_index) + local]
            assert row["task"] == p["task"]
            for cam in CAMS:
                assert tuple(row[f"observation.images.{cam}"].shape) == (3, 480, 640)
            d = load_depth_png(root, "wrist.depth", ep, local - local % 3)
            assert d.dtype == np.uint16 and d.shape == (480, 640)
            samples += 1
        print("PASS episode", ep, p.get("source_key", source.parent.name), p["keep"], "cuts", p["cuts"], flush=True)
    result = dict(
        dataset=str(root.relative_to(WORKSPACE)),
        episodes=len(eps),
        frames=n,
        tasks=len(tasks),
        reader_samples=samples,
        linked_depth_files=depth_files,
        splice_frame_checks=splice_checks,
        idle_cuts=[dict(episode_index=p["episode_index"], cuts=p["cuts"], spliced=p["spliced"]) for p in prov if p["cuts"]],
        annotated=annotated,
        **counts,
    )
    if (meta / "split_status.json").exists():
        result["split_status"] = json.loads((meta / "split_status.json").read_text())
    (report or meta / "cleanup_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    check((WORKSPACE / args.root).resolve(), args.report)

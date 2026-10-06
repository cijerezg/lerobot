"""Second-pass strips: 8 frames evenly across a segment, top over wrist, gripper value burned in.

For rows whose fixed tiles missed the contact (flags image_unreadable / mixed_class).
Usage (workspace root):
    uv run python -m lerobot.annotation.precision_contact.render_strips <row_id> [<row_id> ...] --out review/second_pass
Writes <out>/<row_id>.jpg (':' -> '_').
"""

import argparse, glob, sys
from collections import defaultdict
from pathlib import Path

import cv2, numpy as np, pandas as pd

HERE = Path("migration/precision_contact_annotation_2026-09-25")  # results folder of the 09-25 pass
sys.argv, _argv = sys.argv[:1], sys.argv
from lerobot.annotation.precision_contact import render_rows as rr  # noqa: E402
from lerobot.annotation.precision_contact import sample_contacts as sc  # noqa: E402
sys.argv = _argv

N, W, H = 8, 256, 192


def gripper(root):
    data = pd.concat([pd.read_parquet(f, columns=["index", "observation.state"])
                      for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))]).sort_values("index")
    return np.stack(data["observation.state"].values)[:, 6]


def strip(r, g):
    f0, f1 = r["from_index"], r["to_index"]
    frames = [int(round(f0 + (f1 - 1 - f0) * k / (N - 1))) for k in range(N)]
    rows = []
    for cam in ("top", "wrist"):
        vp, t0 = r["_video"][cam]
        if vp is None:
            rows.append([np.zeros((H, W, 3), np.uint8)] * N); continue
        times = [t0 + (f - r["_off"]) / r["_fps"] for f in frames]
        imgs, st = sc.grab_many(vp, times, r["_fps"]); lut = dict(zip(st, imgs))
        rows.append([cv2.resize(lut[t], (W, H)) for t in times])
    cols = []
    for k, f in enumerate(frames):
        lab = np.zeros((22, W, 3), np.uint8)
        rr.put(lab, f"f{f} +{(f - f0) / r['_fps']:.1f}s grip {g[f]:.0f}", (3, 15), 0.45)
        cols.append(np.vstack([lab, rows[0][k], rows[1][k]]))
    head = np.full((26, W * N, 3), (40, 40, 90), np.uint8)
    rr.put(head, f"{r['row_id']} | {r['subtask']} | f{f0}-{f1} ({r['len_s']}s) | top over wrist", (4, 18), 0.55)
    return np.vstack([head, np.hstack(cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("row_ids", nargs="+")
    ap.add_argument("--out", default=str(HERE / "review/second_pass"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    by_root = defaultdict(set)
    for rid in a.row_ids: by_root[rid.split(":")[0]].add(rid)
    paths = dict(rr.ROOTS)
    for short, ids in by_root.items():
        rows, _ = rr.load_root(short, paths[short]); g = gripper(paths[short])
        for r in rows:
            if r["row_id"] in ids:
                p = out / f"{r['row_id'].replace(':', '_')}.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(strip(r, g), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                print(p)


if __name__ == "__main__":
    main()

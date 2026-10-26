"""Extract the first top-camera frame of every object-grasp segment in the internal ReBot roots, overlay a
table-aligned grid, and write a manifest. Run from the workspace root:

    uv run python lerobot/adhocs/object_coverage/extract_frames.py

Outputs under lerobot/adhocs/object_coverage/frames/<object>/ and manifest.csv.
"""
import re, subprocess, csv
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT = Path("lerobot/adhocs/object_coverage")
ROOTS = """outputs/rebot_all-annotated-v14-tasks2
outputs/rebot_main_additions-annotated-v3
outputs/rebot_bits_kitfold_2026-10-06-annotated-v2-tasks1
outputs/rebot_additions_2026-10-04-annotated-v12-tasks1
outputs/rebot_teleop_2026-10-07-annotated-v1-tasks1
outputs/rebot_teleop_2026-10-08-annotated-v1-tasks1
outputs/rebot_oldval_train-2026-10-14-v1
outputs/rebot_2026-10-14-annotated-v4
outputs/rebot_2026-10-16-annotated-v1
outputs/rebot_2026-10-17-merged-v1
outputs/rebot_2026-10-21-annotated-v1""".split()

# object key -> (regex on the clause, one-per-episode)
OBJECTS = {
    "nail_clipper": (r"^grasp the (small )?nail clipper$", False),
    "large_clipper": (r"^grasp the large nail clipper$", False),
    "headphone_case": (r"^(grasp|close) the (headphone case lid|lid of the case with the headphones)$", True),
    "bit_kit": (r"^(grasp the outer edge of the flap|fold the flap|lift the flap|push the flap)", True),
    "bit": (r"^grasp .*\bbit\b(?! kit)", False),
}

# table corners in the 640x480 top image: TL (robot side left), TR, BR, BL
CORNERS = np.array([[128, 42], [632, 38], [635, 388], [95, 385]], dtype=float)
COLS, ROWS = 6, 4
COL_NAMES = "ABCDEF"


def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A))
    return vt[-1].reshape(3, 3)


# table coordinates (u, v) in [0, COLS] x [0, ROWS] -> pixel
H = homography(np.array([[0, 0], [COLS, 0], [COLS, ROWS], [0, ROWS]], float), CORNERS)


def to_px(u, v):
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def draw_grid(img):
    d = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default(size=16)
    for i in range(COLS + 1):
        d.line([to_px(i, 0), to_px(i, ROWS)], fill=(255, 255, 0, 200), width=2)
    for j in range(ROWS + 1):
        d.line([to_px(0, j), to_px(COLS, j)], fill=(255, 255, 0, 200), width=2)
    for i in range(COLS):
        for j in range(ROWS):
            x, y = to_px(i + 0.5, j + 0.5)
            lab = f"{COL_NAMES[i]}{j + 1}"
            d.rectangle([x - 14, y - 10, x + 14, y + 10], fill=(0, 0, 0, 150))
            d.text((x - 11, y - 9), lab, fill=(255, 255, 0, 255), font=font)
    # compass: 0 deg = along the long table edge (image horizontal), 90 = toward the robot (image up)
    cx, cy = 50, 440
    d.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], outline=(255, 255, 255, 220), width=2)
    d.line([cx - 30, cy, cx + 30, cy], fill=(255, 255, 255, 220), width=2)
    d.line([cx, cy - 30, cx, cy + 30], fill=(255, 255, 255, 220), width=2)
    d.text((cx + 32, cy - 8), "0", fill=(255, 255, 255, 255), font=font)
    d.text((cx - 5, cy - 48), "90", fill=(255, 255, 255, 255), font=font)
    d.text((cx + 4, cy + 32), "robot is up", fill=(255, 255, 255, 255), font=ImageFont.load_default(size=12))
    return img


def main():
    rows = []
    for root in ROOTS:
        r = Path(root)
        seg = pd.read_parquet(r / "meta/episode_metadata.parquet")
        eps = pd.concat(pd.read_parquet(p) for p in sorted((r / "meta/episodes").glob("*/*.parquet")))
        eps = eps.set_index("episode_index")
        fps = 30
        for key, (pat, one_per_ep) in OBJECTS.items():
            m = seg[seg.subtask.str.match(pat)]
            if one_per_ep:
                if key == "headphone_case":
                    g = m[m.subtask.str.startswith("grasp")]
                    rest = m[~m.episode_index.isin(g.episode_index)]
                    m = pd.concat([g, rest.groupby("episode_index").head(1)])
                else:
                    m = m.groupby("episode_index").head(1)
            for _, s in m.iterrows():
                e = eps.loc[s.episode_index]
                frame = int(s.from_index - e["dataset_from_index"])  # segment indices are dataset-absolute
                t = e["videos/observation.images.top/from_timestamp"] + frame / fps
                vid = r / "videos/observation.images.top" / f"chunk-{e['videos/observation.images.top/chunk_index']:03d}" / f"file-{e['videos/observation.images.top/file_index']:03d}.mp4"
                name = f"{r.name}__ep{s.episode_index:03d}_seg{s.segment_index:02d}.png"
                out = OUT / "frames" / key / name
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    raw = OUT / "frames" / key / ("raw_" + name)
                    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", str(vid),
                                    "-frames:v", "1", "-update", "1", "-y", str(raw)], check=True)
                    draw_grid(Image.open(raw).convert("RGB")).save(out)
                    raw.unlink()
                rows.append(dict(object=key, root=r.name, episode=int(s.episode_index), segment=int(s.segment_index),
                                 frame=frame, clause=s.subtask, image=str(out.relative_to(OUT))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "manifest.csv", index=False)
    print(df.groupby("object").size())


if __name__ == "__main__":
    main()

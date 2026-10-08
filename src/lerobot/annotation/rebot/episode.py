"""One ReBot episode on disk: low-dim data, operator flags, decoded frames, contact sheets, kept-frame maps.

A record is a dict with ``source`` (dataset root), ``episode`` (index in that root), ``key`` (the name
burned into sheets) and ``frames``. A pass keeps its records in ``<work>/inventory.json``; ``record`` makes
one for any root. Frames are SOURCE-EPISODE-LOCAL everywhere. Gripper (state dim 6): 0 = shut, negative =
open (about -270 wide open).
"""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CAMS = ["top", "wrist"]


def layout(root):
    """(video keys [top, wrist], wrist depth folder) of a root: own roots name them top / wrist, cache-ready roots
    external_0 / wrist_0."""
    features = json.loads((Path(root) / "meta/info.json").read_text())["features"]
    if "observation.images.top" in features:
        return ["observation.images.top", "observation.images.wrist"], "wrist.depth"
    return ["observation.images.external_0", "observation.images.wrist_0"], "wrist_0.depth"


def records(work):
    """The pass inventory: one record per source episode."""
    return json.loads((Path(work) / "inventory.json").read_text())


def record(root, episode):
    """A record for one episode of any dataset root (no inventory needed)."""
    root = Path(root)
    rec = dict(source=str(root), episode=int(episode), key=f"{root.parent.name}/{root.name} ep{episode}")
    rec["frames"] = len(_episode(rec))
    return rec


def episodes(root):
    """Episode indices of a dataset root."""
    files = sorted((Path(root) / "data").rglob("*.parquet"))
    return sorted(pd.concat([pd.read_parquet(p, columns=["episode_index"]) for p in files]).episode_index.unique())


def kept(keep, cuts):
    """Source frames an episode keeps: [a, b) minus the idle cuts."""
    a, b = keep
    m = np.ones(b - a, bool)
    for c0, c1 in cuts:
        m[c0 - a : c1 - a] = False
    return np.arange(a, b)[m]


def remap(k, f):
    """Source frame -> new episode-local frame = kept frames before f. A frame inside a cut lands on the
    splice point, so an interval [x, y) maps to [remap(x), remap(y)) and an interval wholly inside a cut
    becomes empty."""
    return int(np.searchsorted(k, f))


def _episode(rec):
    root = Path(rec["source"])
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    return data[data.episode_index == rec["episode"]].sort_values("frame_index")


def states(rec):
    """(state[N,7], action[N,7]) in degrees."""
    d = _episode(rec)
    return np.stack(d["observation.state"]), np.stack(d.action)


def interventions(rec):
    """bool[N] operator-teleop frames of a rollout (None for teleop recordings)."""
    p = Path(rec["source"]) / "meta/online_labels.parquet"
    if not p.exists():
        return None
    o = pd.read_parquet(p)
    o = o[o.episode_index == rec["episode"]]
    return o.sort_values("frame_index").is_intervention.to_numpy() if len(o) else None  # teleop episode of a mixed root


def decode(rec, camera, want, size=(256, 192)):
    """{frame: PIL image} for the wanted frames of one camera."""
    root = Path(rec["source"])
    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    row = eps[eps.episode_index == rec["episode"]].iloc[0]
    if f"videos/observation.images.{camera}/chunk_index" not in row:  # cache-ready roots name the cameras external_0 / wrist_0
        camera = {"top": "external_0", "wrist": "wrist_0"}[camera]
    prefix = f"videos/observation.images.{camera}"
    chunk, file = int(row[prefix + "/chunk_index"]), int(row[prefix + "/file_index"])
    path = root / prefix / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
    t0 = float(row[prefix + "/from_timestamp"])
    want = set(map(int, want))
    got = {}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0, t0 + min(want) / 30 - 0.5) / s.time_base), stream=s)
        for frame in c.decode(s):
            i = round((float(frame.pts * s.time_base) - t0) * 30)
            if i in want:
                got[i] = frame.to_image().resize(size)
            if i >= max(want):
                break
    assert want == got.keys(), (rec["key"], camera, sorted(want - got.keys()))
    return got


def sheet(rec, want, out, cols=6, size=(256, 192)):
    """Top over wrist per tile; frame index, seconds, gripper state (and TELEOP on operator frames) burned in."""
    want = sorted(set(int(w) for w in want if 0 <= w < rec["frames"]))
    s, _ = states(rec)
    iv = interventions(rec)
    imgs = {cam: decode(rec, cam, want, size) for cam in CAMS}
    w, h = size[0], 2 * size[1] + 24
    canvas = Image.new("RGB", (cols * w, 30 + int(np.ceil(len(want) / cols)) * h), "#151515")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), rec["key"], fill="cyan")
    for k, i in enumerate(want):
        x, y = (k % cols) * w, 30 + (k // cols) * h
        tag = " TELEOP" if iv is not None and iv[i] else ""
        label = f"f{i} {i / 30:.1f}s grip {s[i, 6]:.0f}{tag}"
        draw.text((x + 3, y + 3), label, fill="orange" if tag else "white")
        for j, cam in enumerate(CAMS):
            canvas.paste(imgs[cam][i], (x, y + 24 + j * size[1]))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88)
    return out

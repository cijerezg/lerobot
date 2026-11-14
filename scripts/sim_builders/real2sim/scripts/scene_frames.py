"""Extract external (top) and wrist frames of one episode at chosen indices, with the matching
observation.state, and build a contact sheet.
usage: scene_frames.py ROOT EPISODE OUT [--n 8 | --indices i j k ...] [--sheet name.jpg]"""
import argparse, glob, json, pathlib, av, cv2, numpy as np, pandas as pd

TOP = "observation.images.top"; WRIST = "observation.images.wrist"


def episode_row(root, ep):
    ep_meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
    return ep_meta[ep_meta.episode_index == ep].iloc[0]


def episode_states(root, ep):
    row = episode_row(root, ep)
    df = pd.read_parquet(f"{root}/data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet")
    df = df[df.episode_index == ep].sort_values("frame_index")
    return np.stack(df["observation.state"].to_numpy()), int(row["length"])


def decode_frames(root, ep, key, indices):
    """Decode the episode's video once and return {index: BGR frame}. One video file per episode in these roots."""
    row = episode_row(root, ep)
    path = f"{root}/videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
    fps = 30; start = int(round(row[f"videos/{key}/from_timestamp"] * fps))
    want = {start + int(i) for i in indices}; out = {}
    with av.open(path) as c:
        for n, fr in enumerate(c.decode(video=0)):
            if n in want: out[n - start] = fr.to_ndarray(format="bgr24")
            if n >= max(want): break
    return out


def sheet(frames, labels, cols=4, scale=0.5):
    tiles = []
    for im, lab in zip(frames, labels):
        t = cv2.resize(im, None, fx=scale, fy=scale); cv2.putText(t, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(t, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1); tiles.append(t)
    while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("episode", type=int); p.add_argument("out"); p.add_argument("--n", type=int, default=8); p.add_argument("--indices", type=int, nargs="*"); p.add_argument("--sheet", default="sheet.jpg"); a = p.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    states, length = episode_states(a.root, a.episode)
    idx = a.indices if a.indices else [int(round(x)) for x in np.linspace(0, length - 1, a.n)]
    top = decode_frames(a.root, a.episode, TOP, idx); wrist = decode_frames(a.root, a.episode, WRIST, idx)
    rec = {}
    for i in idx:
        cv2.imwrite(str(out / f"top_{i:05d}.jpg"), top[i], [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(str(out / f"wrist_{i:05d}.jpg"), wrist[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec[i] = states[i].round(2).tolist()
    (out / "states.json").write_text(json.dumps(rec, indent=1))
    s = np.vstack([sheet([top[i] for i in idx], [f"top f{i} t={i/30:.1f}s" for i in idx]), sheet([wrist[i] for i in idx], [f"wrist f{i}" for i in idx])])
    cv2.imwrite(str(out / a.sheet), s, [cv2.IMWRITE_JPEG_QUALITY, 90]); print("wrote", out / a.sheet, "length", length, "indices", idx)
    for i in idx: print(i, rec[i])

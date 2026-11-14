"""Real robot masks for the external (top) camera by colour: the table is orange wood, the arm is white/black/lime.
non_wood(frame): pixels whose HSV hue is outside the wood band or with low saturation.
static mask S: non-wood in >= STATIC_FRAC of the sampled frames (objects, floor, base, cables).
robot mask R_t = non_wood(frame_t) & ~S, cleaned (open + small components removed).
usage: scene_masks.py ROOT EPISODE WORK [--every 10] [--qc i j k]
writes WORK/frames_every.npz (frames, indices, states), WORK/static_mask.png, WORK/qc/mask_qc_*.jpg"""
import argparse, pathlib, sys, av, cv2, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scene_frames import episode_row, episode_states, TOP

STATIC_FRAC = 0.85
# region of interest in the top image (pixels): the table top plus the strip above its far edge where the robot base stands
ROI_POLY = np.array([(96, 388), (128, 48), (250, 0), (420, 0), (532, 48), (522, 388)], np.int32)


def roi_mask(h=480, w=640):
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [ROI_POLY], 1); return m.astype(bool)

WOOD_H = (3, 24)   # OpenCV hue (0..180): orange wood incl. shadows
WOOD_S_MIN = 28


def non_wood(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    wood = (h >= WOOD_H[0]) & (h <= WOOD_H[1]) & (s >= WOOD_S_MIN) & (v > 20)
    return ~wood


def clean(mask, min_area=150, k=3):
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def robot_mask(bgr, static):
    return clean(non_wood(bgr) & ~static & roi_mask(*bgr.shape[:2]))


def decode_every(root, ep, every):
    row = episode_row(root, ep); key = TOP
    path = f"{root}/videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
    start = int(round(row[f"videos/{key}/from_timestamp"] * 30)); n_ep = int(row["length"]); frames, idx = [], []
    with av.open(path) as c:
        for n, fr in enumerate(c.decode(video=0)):
            i = n - start
            if i < 0: continue
            if i >= n_ep: break
            if i % every == 0: frames.append(fr.to_ndarray(format="bgr24")); idx.append(i)
    return np.stack(frames), np.array(idx)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("episode", type=int); p.add_argument("work"); p.add_argument("--every", type=int, default=10); p.add_argument("--qc", type=int, nargs="*", default=None); a = p.parse_args()
    work = pathlib.Path(a.work); (work / "qc").mkdir(parents=True, exist_ok=True)
    frames, idx = decode_every(a.root, a.episode, a.every); states, _ = episode_states(a.root, a.episode)
    np.savez(work / "frames_every.npz", frames=frames, indices=idx, states=states[idx])
    nw = np.stack([non_wood(f) for f in frames]); static = nw.mean(0) >= STATIC_FRAC
    static = cv2.dilate(static.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # margin around static objects
    cv2.imwrite(str(work / "static_mask.png"), static.astype(np.uint8) * 255)
    print("frames", frames.shape, "static fraction %.3f" % static.mean())
    qc = a.qc if a.qc else [idx[len(idx) // 6], idx[len(idx) // 2], idx[5 * len(idx) // 6]]
    for i in qc:
        f = frames[list(idx).index(i)]; m = robot_mask(f, static); ov = f.copy(); ov[m] = (0.4 * ov[m] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        nwv = cv2.cvtColor((non_wood(f) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR); sv = cv2.cvtColor(static.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        for im, t in ((f, f"frame {i}"), (nwv, "non-wood"), (sv, "static"), (ov, "robot mask")): cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(work / "qc" / f"mask_qc_{i:05d}.jpg"), np.vstack([np.hstack([f, nwv]), np.hstack([sv, ov])]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("qc frames", qc)

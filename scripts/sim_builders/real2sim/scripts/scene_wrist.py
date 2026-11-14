"""Wrist (D405) helpers: recorded intrinsics, depth sidecar loader, back-projection, RANSAC table plane.
Intrinsics source: config_rl.yaml `rebot_depth_intrinsics` (factory depth intrinsics printed by realsense_pointcloud.py;
RGB comes from the left imager, same intrinsics). Sidecar: depth/wrist.depth/episode-XXXXXX/frame-XXXXXX.png, uint16,
0.1 mm per level, every 3rd frame (10 Hz), spatial + hole-fill filters applied at recording (lerobot_memmap_buffer_cache.py, depth_writer.py).
usage: scene_wrist.py sheet ROOT EP OUT.jpg [frames...]  |  scene_wrist.py stats ROOT EP [--every 30]"""
import pathlib, sys, cv2, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

FX, FY, CX, CY = 394.9832, 394.9832, 322.5604, 238.6966
UNITS_M = 1e-4          # metres per level
DEPTH_KEY = "wrist.depth"; W, H = 640, 480
D405_MIN, D405_MAX = 0.07, 0.50   # metres, used for the sim depth clip


def depth_frames(root, ep):
    d = pathlib.Path(root) / "depth" / DEPTH_KEY / f"episode-{ep:06d}"
    return sorted(int(p.stem.split("-")[1]) for p in d.glob("frame-*.png"))


def load_depth(root, ep, frame):
    """Raw uint16 levels (0 = invalid); frame must be on the sidecar grid (nearest one otherwise)."""
    avail = depth_frames(root, ep); f = min(avail, key=lambda k: abs(k - frame))
    return cv2.imread(f"{root}/depth/{DEPTH_KEY}/episode-{ep:06d}/frame-{f:06d}.png", cv2.IMREAD_UNCHANGED), f


def backproject(depth_m, fx=FX, fy=FY, cx=CX, cy=CY):
    """(H, W) metres -> (H, W, 3) camera-frame points (x right, y down, z forward); invalid -> nan."""
    v, u = np.mgrid[0:depth_m.shape[0], 0:depth_m.shape[1]]; z = depth_m.astype(np.float64); z = np.where(z > 0, z, np.nan)
    return np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)


def ransac_plane(pts, n_iter=300, thresh=0.004, rng=None):
    """pts (N, 3). Returns unit normal n (pointing toward the camera, i.e. n.z < 0), d with n.p + d = 0, inlier mask."""
    rng = rng or np.random.default_rng(0); best = (None, None, np.zeros(len(pts), bool))
    for _ in range(n_iter):
        i = rng.choice(len(pts), 3, replace=False); p = pts[i]; n = np.cross(p[1] - p[0], p[2] - p[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; d = -n @ p[0]; inl = np.abs(pts @ n + d) < thresh
        if inl.sum() > best[2].sum(): best = (n, d, inl)
    n, d, inl = best
    # refine by least squares on the inliers
    c = pts[inl].mean(0); _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False); n = vt[2]; n = n if n[2] < 0 else -n; d = -n @ c
    inl = np.abs(pts @ n + d) < thresh; return n, d, inl


def colourmap(depth_m, vmin=0.05, vmax=0.5):
    x = np.clip((depth_m - vmin) / (vmax - vmin), 0, 1); im = cv2.applyColorMap((255 * (1 - x)).astype(np.uint8), cv2.COLORMAP_TURBO); im[depth_m <= 0] = 0; return im


if __name__ == "__main__":
    from scene_frames import decode_frames, WRIST
    cmd, root, ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if cmd == "sheet":
        out = sys.argv[4]; frames = [int(x) for x in sys.argv[5:]] or [0, 333, 669, 1002, 1335, 1671, 2004, 2337]
        rgb = decode_frames(root, ep, WRIST, frames); tiles = []
        for f in frames:
            d, fa = load_depth(root, ep, f); dm = d * UNITS_M; valid = d > 0
            a = rgb[f].copy(); b = colourmap(dm)
            for im, t in ((a, f"wrist rgb f{f}"), (b, f"depth f{fa} med {np.median(dm[valid]):.3f} m valid {valid.mean():.2f}")): cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            tiles.append(np.hstack([a, b]))
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]; cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("wrote", out)
    elif cmd == "stats":
        every = int(sys.argv[5]) if len(sys.argv) > 5 else 30
        for f in depth_frames(root, ep)[::every // 3]:
            d, _ = load_depth(root, ep, f); dm = d * UNITS_M; valid = (d > 0) & (dm < 1.0); pts = backproject(np.where(valid, dm, 0))[valid]
            if len(pts) < 2000: print(f, "valid %.2f" % valid.mean(), "too few"); continue
            sub = pts[np.random.default_rng(0).choice(len(pts), min(4000, len(pts)), replace=False)]; n, dd, inl = ransac_plane(sub)
            print(f"f{f:5d} valid {valid.mean():.2f} med {np.median(dm[valid]):.3f} m  plane inliers {inl.mean():.2f} dist {dd:.3f} normal {n.round(3)}")

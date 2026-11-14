"""Scan segment -> ~N sharp colour frames for VGGT + a gripper mask (pixels with depth 0 in most frames of the whole scan).
usage: cut_frames.py scan_dir work_dir --range F0 F1 [--n 140]
writes work_dir/frames/NNNNNN.jpg (full 848x480 colour), work_dir/frames.json, scan_dir/../gripper_mask.png (once)."""
import argparse, json, pathlib, shutil, cv2, numpy as np
p = argparse.ArgumentParser(); p.add_argument("scan"); p.add_argument("work"); p.add_argument("--range", nargs=2, type=int, required=True); p.add_argument("--n", type=int, default=140); a = p.parse_args()
S = pathlib.Path(a.scan); W = pathlib.Path(a.work); F = W / "frames"; F.mkdir(parents=True, exist_ok=True)
mask_png = S.parent / "gripper_mask.png"
if not mask_png.exists():
    ids = range(0, len(list((S / "depth").glob("*.png"))), 5); zero = np.zeros((480, 848), np.float32)
    for i in ids: zero += cv2.imread(str(S / "depth" / f"{i:06d}.png"), -1) == 0
    zero /= len(ids); m = (zero > 0.85).astype(np.uint8) * 255; m[:400] = 0  # the gripper only occupies the bottom rows
    m = cv2.dilate(m, np.ones((15, 15), np.uint8)); cv2.imwrite(str(mask_png), m); print("gripper mask px", int((m > 0).sum()))
f0, f1 = a.range; cand = []
for i in range(f0, f1 + 1):
    im = cv2.imread(str(S / "color" / f"{i:06d}.jpg")); cand.append((i, cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
cand = np.array(cand); edges = np.linspace(f0, f1 + 1, a.n + 1); kept = []
for lo, hi in zip(edges[:-1], edges[1:]):
    c = cand[(cand[:, 0] >= lo) & (cand[:, 0] < hi)]
    if len(c): kept.append(int(c[np.argmax(c[:, 1])][0]))
for i in kept: shutil.copy(S / "color" / f"{i:06d}.jpg", F / f"{i:06d}.jpg")
json.dump({"range": [f0, f1], "frames": kept, "sharp_median": float(np.median(cand[:, 1]))}, open(W / "frames.json", "w"))
print(f"{f1 - f0 + 1} frames in range, kept {len(kept)}, sharpness median {np.median(cand[:, 1]):.0f}")

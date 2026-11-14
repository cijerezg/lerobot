"""Real-frame colour statistics for the appearance fit: desk wood (top f0), wall (wrist f0), floor (wrist f2400).
Hand-picked object-free patches (640x480 px), luminance tails trimmed (keep 25th..95th pct) so the same trim can be applied to the
sim masks. Writes twin/appearance/fit_real.json (mean/std RGB 0..255 per region). Run from the LeRobot root:
uv run python lerobot/scripts/sim_builders/digital_twin/scripts/appear_fit.py"""
import json, pathlib, cv2, numpy as np
ROOT = pathlib.Path("lerobot/scripts/sim_builders/digital_twin"); OUT = ROOT / "twin/appearance"
PATCHES = {  # (y0, y1, x0, x1)
    "desk": ("twin/renders/real_top_f0.jpg", [(280, 430, 460, 515), (90, 160, 405, 465), (350, 440, 45, 105), (405, 440, 380, 470)]),
    "wall": ("room/renders/real_wrist_f0.png", [(20, 200, 60, 300), (20, 200, 420, 620)]),
    "floor": ("room/renders/real_wrist_f2400.png", [(260, 440, 545, 635)]),
    "desk_wrist_ref": ("room/renders/real_wrist_f0.png", [(285, 330, 50, 140), (440, 475, 380, 450)]),  # reference only: wrist-cam white balance
}
def trim(px):
    lum = px @ np.array([0.114, 0.587, 0.299]); lo, hi = np.percentile(lum, [25, 95]); return px[(lum >= lo) & (lum <= hi)]
def stats(px):
    return dict(mean_rgb=px[:, ::-1].mean(0).round(2).tolist(), std_rgb=px[:, ::-1].std(0).round(2).tolist(), n=int(len(px)))
if __name__ == "__main__":
    res = {}; vis = []
    for name, (img, boxes) in PATCHES.items():
        im = cv2.imread(str(ROOT / img)); px = np.concatenate([im[y0:y1, x0:x1].reshape(-1, 3) for y0, y1, x0, x1 in boxes]).astype(float)
        res[name] = stats(trim(px)); res[name]["std_within_rgb"] = np.mean([trim(im[y0:y1, x0:x1].reshape(-1, 3).astype(float))[:, ::-1].std(0) for y0, y1, x0, x1 in boxes], 0).round(2).tolist()
        res[name]["raw_mean_rgb"] = px[:, ::-1].mean(0).round(2).tolist(); res[name]["patches"] = boxes; res[name]["image"] = img
        v = im.copy()
        for y0, y1, x0, x1 in boxes: cv2.rectangle(v, (x0, y0), (x1, y1), (0, 255, 0), 2)
        vis.append(v)
    cv2.imwrite(str(OUT / "fit_patches.jpg"), np.hstack(vis), [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(res, open(OUT / "fit_real.json", "w"), indent=1); print(json.dumps(res, indent=1))

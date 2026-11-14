"""Find the Letter sheet in the table-level layer and derive metres per unit.
Known-aspect rectangle search: paper pixels = bright low-saturation table pixels (M); the object footprint is the
hole (H) in the table layer nearest the cloud origin. For each orientation (long side along the sheet's edge
direction or across it), rotate the masks so the rectangle is axis-aligned and maximise
    J = |M inside R| - |wood inside R|      (wood = table pixels that are neither paper nor hole)
over centre and size with integral images, subject to the hole centre lying inside R.
usage: sheet_scale.py work_dir [long_mm short_mm]"""
import sys, pathlib, numpy as np, cv2, open3d as o3d
W = pathlib.Path(sys.argv[1]); Lmm, Smm = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (279.4, 215.9); ratio = Lmm / Smm
pc = o3d.io.read_point_cloud(str(W / "cloud_table.ply")); P = np.asarray(pc.points); C = np.asarray(pc.colors)
ext = np.ptp(P[:, :2], axis=0).max(); res = 1200
on = np.abs(P[:, 2]) < ext * 0.004; lo = P[:, :2].min(0); s = (res - 1) / ext
img = np.zeros((res, res, 3), np.uint8); uv = ((P[on][:, :2] - lo) * s).astype(int)
img[res - 1 - uv[:, 1], uv[:, 0]] = (C[on][:, ::-1] * 255).astype(np.uint8)
filled = cv2.morphologyEx((img.sum(2) > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
holes = (1 - filled).astype(np.uint8); n, lab, stats, cen = cv2.connectedComponentsWithStats(holes)
border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])); origin_px = np.array([(0 - lo[0]) * s, res - 1 - (0 - lo[1]) * s])
cands = [i for i in range(1, n) if i not in border and stats[i, cv2.CC_STAT_AREA] > 2000]
hole = min(cands, key=lambda i: np.linalg.norm(cen[i] - origin_px)); H = (lab == hole).astype(np.uint8); hx, hy, hw, hh = stats[hole, :4]; hc = cen[hole]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
M = ((hsv[..., 1] < 25) & (hsv[..., 2] > 190) & (filled > 0)).astype(np.uint8); M = cv2.morphologyEx(M, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
Wd = ((filled > 0) & (M == 0) & (H == 0) & (hsv[..., 1] > 35) & (hsv[..., 2] > 120)).astype(np.uint8)  # the bright saturated table top only
band = cv2.dilate(H, np.ones((int(0.8 * max(hw, hh)) | 1,) * 2, np.uint8)); ys, xs = np.nonzero(M & band)
(_, _), _, ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
best = None
for th in (ang, ang + 90):
    Rm = cv2.getRotationMatrix2D((res / 2, res / 2), th, 1.0)  # rotates image by +th (counter-clockwise in image coords)
    Mr = cv2.warpAffine(M, Rm, (res, res), flags=cv2.INTER_NEAREST); Wr = cv2.warpAffine(Wd, Rm, (res, res), flags=cv2.INTER_NEAREST)
    hcr = Rm[:, :2] @ hc + Rm[:, 2]
    IM = cv2.integral(Mr).astype(np.int64); IW = cv2.integral(Wr).astype(np.int64)
    def rect_sum(I, x0, y0, x1, y1): return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
    for L in range(int(0.5 * max(hw, hh)), int(0.9 * res), 3):
        S_ = int(round(L / ratio)); step = max(2, L // 60)
        xs0 = np.arange(int(hcr[0] - L + 4), int(hcr[0]) - 4, step); ys0 = np.arange(int(hcr[1] - S_ + 4), int(hcr[1]) - 4, step)
        xs0 = xs0[(xs0 >= 0) & (xs0 + L < res)]; ys0 = ys0[(ys0 >= 0) & (ys0 + S_ < res)]
        if len(xs0) == 0 or len(ys0) == 0: continue
        X0, Y0 = np.meshgrid(xs0, ys0); J = rect_sum(IM, X0, Y0, X0 + L, Y0 + S_) - rect_sum(IW, X0, Y0, X0 + L, Y0 + S_)
        k = np.argmax(J)
        if best is None or J.flat[k] > best[0]: best = (int(J.flat[k]), th, L, S_, int(X0.flat[k]), int(Y0.flat[k]), Rm)
J, th, L, S_, x0, y0, Rm = best; Rinv = cv2.invertAffineTransform(Rm)
corners_r = np.array([[x0, y0], [x0 + L, y0], [x0 + L, y0 + S_], [x0, y0 + S_]], np.float32); box = corners_r @ Rinv[:, :2].T + Rinv[:, 2]
scale = Lmm / 1000 / (L / s); print("hole bbox px", hw, hh, "; best rect long %d short %d px at angle %.1f, J=%d" % (L, S_, th, J))
print("m per unit %.4f (Letter %.1f x %.1f mm assumed; aspect fixed, no independent ratio check)" % (scale, Lmm, Smm))
# diagnostic: paper coverage inside the rectangle, and how much paper lies just outside (should be ~0)
Mr = cv2.warpAffine(M, Rm, (res, res), flags=cv2.INTER_NEAREST); Hr = cv2.warpAffine(H, Rm, (res, res), flags=cv2.INTER_NEAREST)
inside = Mr[y0:y0 + S_, x0:x0 + L].mean() + Hr[y0:y0 + S_, x0:x0 + L].mean(); ring = Mr[max(y0 - 12, 0):y0 + S_ + 12, max(x0 - 12, 0):x0 + L + 12].sum() - Mr[y0:y0 + S_, x0:x0 + L].sum()
print("coverage inside (paper+hole) %.2f ; paper pixels in a 12 px ring outside: %d" % (inside, ring))
box_w = np.stack([box[:, 0] / s + lo[0], (res - 1 - box[:, 1]) / s + lo[1]], 1)
np.savez(W / "sheet.npz", box=box_w, scale=scale, centre=box_w.mean(0), coverage=inside, ring=ring)
dbg = img.copy(); dbg[M > 0] = (255, 0, 255); dbg[H > 0] = (40, 40, 40); cv2.polylines(dbg, [box.astype(int).reshape(-1, 1, 2)], True, (0, 0, 255), 2)
bx0, by0 = int(max(box[:, 0].min() - 60, 0)), int(max(box[:, 1].min() - 60, 0)); cv2.imwrite(str(W / "sheet_fit.png"), dbg[by0:int(box[:, 1].max()) + 60, bx0:int(box[:, 0].max()) + 60])

"""Measure the nail clippers from cloud_table.ply: points inside the sheet rectangle, above the table, not white
(the sheet and the foam insert are white; the clippers are dark/metallic), clustered with DBSCAN; report each cluster's
footprint (PCA length/width) and max height in mm, write clippers_top.png (top-down crop, cluster ids) and clippers.npz.
usage: nail_clippers.py work_dir [sheet_npz_name] [z_min_mm] [max_brightness]  (defaults sheet.npz, 4, 0.6)"""
import sys, pathlib, numpy as np, cv2, open3d as o3d
W = pathlib.Path(sys.argv[1]); sh = np.load(W / (sys.argv[2] if len(sys.argv) > 2 else "sheet.npz"))
zmin = (float(sys.argv[3]) if len(sys.argv) > 3 else 4.0) / 1000; bmax = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
scale = float(sh["scale"]); print("scale m/unit %.4f" % scale)
pc = o3d.io.read_point_cloud(str(W / "cloud_table.ply")); P = np.asarray(pc.points) * scale; C = np.asarray(pc.colors)
box = sh["box"] * scale; c = box.mean(0); e0 = box[1] - box[0]; e1 = box[3] - box[0]
u0 = e0 / np.linalg.norm(e0); u1 = e1 / np.linalg.norm(e1); h0 = np.linalg.norm(e0) / 2; h1 = np.linalg.norm(e1) / 2
rel = P[:, :2] - c; inb = (np.abs(rel @ u0) < h0 - 0.005) & (np.abs(rel @ u1) < h1 - 0.005)
bright = C.mean(1)
tab = inb & (bright > 0.7) & (np.abs(P[:, 2]) < 0.02)
print("sheet-level noise (bright points in rect): z std %.1f mm, 1/99 pct %.1f / %.1f mm" % (P[tab, 2].std() * 1e3, *np.percentile(P[tab, 2], [1, 99]) * 1e3))
up = inb & (P[:, 2] > zmin) & (P[:, 2] < 0.08) & (bright < bmax)
Q = P[up]; Cq = C[up]; print("candidate points (in rect, z>%.0f mm, brightness<%.2f)" % (zmin * 1e3, bmax), len(Q))
q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q)); lab = np.array(q.cluster_dbscan(eps=0.004, min_points=20))
ids, cnt = np.unique(lab[lab >= 0], return_counts=True); order = np.argsort(-cnt); print("clusters", len(ids))
rows = []
for k in order[:8]:
    i = ids[k]; S = Q[lab == i]; Cs = Cq[lab == i]
    xy = S[:, :2] - S[:, :2].mean(0); u, sv, vt = np.linalg.svd(xy, full_matrices=False); proj = xy @ vt.T
    L, Wd = np.ptp(proj[:, 0]), np.ptp(proj[:, 1]); L98 = np.diff(np.percentile(proj[:, 0], [1, 99]))[0]; W98 = np.diff(np.percentile(proj[:, 1], [1, 99]))[0]
    hmax = S[:, 2].max(); h99 = np.percentile(S[:, 2], 99)
    rows.append(dict(id=int(i), n=len(S), centre_mm=S[:, :2].mean(0) * 1e3, len_mm=L * 1e3, wid_mm=Wd * 1e3, len98_mm=L98 * 1e3, wid98_mm=W98 * 1e3, hmax_mm=hmax * 1e3, h99_mm=h99 * 1e3, brightness=Cs.mean(), axis=vt[0]))
    print("cluster %d: n=%d centre xy (mm) %s len %.1f wid %.1f (1-99pct %.1f x %.1f) hmax %.1f h99 %.1f mm brightness %.2f axis %s"
          % (i, len(S), (S[:, :2].mean(0) * 1e3).round(0), L * 1e3, Wd * 1e3, L98 * 1e3, W98 * 1e3, hmax * 1e3, h99 * 1e3, Cs.mean(), vt[0].round(2)))
np.savez(W / "clippers.npz", points=Q, colors=Cq, labels=lab, rows=np.array(rows, dtype=object), scale=scale)
# top-down crop: all in-rect points (coloured) with the dark clusters' ids
A = P[inb & (P[:, 2] > -0.005)]; Ca = C[inb & (P[:, 2] > -0.005)]
res = 1200; lo = A[:, :2].min(0); s = (res - 1) / np.ptp(A[:, :2], axis=0).max()
img = np.full((res, res, 3), 255, np.uint8); o = np.argsort(A[:, 2]); uv = ((A[o][:, :2] - lo) * s).astype(int)
img[res - 1 - uv[:, 1], uv[:, 0]] = (Ca[o][:, ::-1] * 255).astype(np.uint8)
for r in rows:
    p = ((r["centre_mm"] / 1e3 - lo) * s).astype(int); cv2.putText(img, str(r["id"]), (p[0] + 12, res - 1 - p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imwrite(str(W / "clippers_top.png"), img)
# tight crop around the dark clusters only
if len(Q): 
    lo2 = Q[:, :2].min(0) - 0.01; hi2 = Q[:, :2].max(0) + 0.01; s2 = 999 / (hi2 - lo2).max()
    im2 = np.full((1000, 1000, 3), 255, np.uint8); sel = (A[:, 0] > lo2[0]) & (A[:, 0] < hi2[0]) & (A[:, 1] > lo2[1]) & (A[:, 1] < hi2[1])
    B = A[sel]; Cb = Ca[sel]; o = np.argsort(B[:, 2]); uv = ((B[o][:, :2] - lo2) * s2).astype(int); uv = np.clip(uv, 0, 999)
    im2[999 - uv[:, 1], uv[:, 0]] = (Cb[o][:, ::-1] * 255).astype(np.uint8); cv2.imwrite(str(W / "clippers_crop.png"), im2)

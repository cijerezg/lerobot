"""Cross-check only (does not write sheet.npz): fit a rectangle to ALL bright low-saturation table-level pixels
of cloud_table.ply, not just the band around the object hole. usage: nail_sheet_check.py work_dir"""
import sys, pathlib, numpy as np, cv2, open3d as o3d
W = pathlib.Path(sys.argv[1]); Lmm, Smm = 279.4, 215.9
pc = o3d.io.read_point_cloud(str(W / "cloud_table.ply")); P = np.asarray(pc.points); C = np.asarray(pc.colors)
ext = np.ptp(P[:, :2], axis=0).max(); res = 1200
on = np.abs(P[:, 2]) < ext * 0.004; lo = P[:, :2].min(0); s = (res - 1) / ext
img = np.zeros((res, res, 3), np.uint8); uv = ((P[on][:, :2] - lo) * s).astype(int)
img[res - 1 - uv[:, 1], uv[:, 0]] = (C[on][:, ::-1] * 255).astype(np.uint8)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
sheet = ((hsv[..., 1] < 25) & (hsv[..., 2] > 190)).astype(np.uint8)
sheet = cv2.morphologyEx(sheet, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, stats, cen = cv2.connectedComponentsWithStats(sheet)
# the sheet = the component nearest the origin (table-frame origin is near the sheet centre)
o = ((0 - lo) * s); o = np.array([o[0], res - 1 - o[1]])
areas = stats[1:, cv2.CC_STAT_AREA]; big = [i + 1 for i in np.argsort(-areas)[:5]]
for i in big: print("component", i, "area", stats[i, cv2.CC_STAT_AREA], "centre px", cen[i].round(0), "dist to origin px %.0f" % np.linalg.norm(cen[i] - o))
i = min(big, key=lambda i: np.linalg.norm(cen[i] - o))
m = (lab == i).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
# fill the object holes so minAreaRect sees the outer boundary
cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cnt = max(cnts, key=cv2.contourArea)
rect = cv2.minAreaRect(cnt); (cx, cy), (w, h), ang = rect; L, S = max(w, h), min(w, h)
print("rect sides px %.1f %.1f ratio %.3f (letter 1.294)" % (L, S, L / S))
sc1 = Lmm / 1000 / (L / s); sc2 = Smm / 1000 / (S / s)
print("m per unit: from long %.4f, from short %.4f, disagreement %.1f%%" % (sc1, sc2, 100 * (sc2 / sc1 - 1)))
print("fill ratio of component inside rect %.2f" % (cv2.contourArea(cnt) / (w * h)))
dbg = img.copy(); dbg[m > 0] = (255, 0, 255); box = cv2.boxPoints(rect)
cv2.polylines(dbg, [box.astype(int).reshape(-1, 1, 2)], True, (0, 0, 255), 2)
x0, y0 = max(int(cx - L), 0), max(int(cy - L), 0); cv2.imwrite(str(W / "sheet_check.png"), dbg[y0:int(cy + L), x0:int(cx + L)])
box_w = np.stack([box[:, 0] / s + lo[0], (res - 1 - box[:, 1]) / s + lo[1]], 1)
np.savez(W / "sheet_check.npz", box=box_w, scale=sc1, scale_short=sc2, centre=box_w.mean(0), ratio=L / S)

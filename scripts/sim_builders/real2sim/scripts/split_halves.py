"""Clamshell case: keep clusters standing on the sheet, find the hinge seam along the major axis, split into base and lid.
Writes base.ply, lid.ply, hinge.json (axis point + direction, metres, table frame) and split_view.jpg. usage: split_halves.py work_dir"""
import sys, json, pathlib, numpy as np, cv2, open3d as o3d
W = pathlib.Path(sys.argv[1]); m = o3d.io.read_triangle_mesh(str(W / "mesh_object.ply")); sh = np.load(W / "sheet.npz"); box = sh["box"] * float(sh["scale"])
V = np.asarray(m.vertices); Tt = np.asarray(m.triangles); cl, cnt, _ = m.cluster_connected_triangles(); cl = np.asarray(cl); cnt = np.asarray(cnt)
# sheet box as oriented rectangle test
c = box.mean(0); e0 = box[1] - box[0]; e1 = box[3] - box[0]; L0, L1 = np.linalg.norm(e0), np.linalg.norm(e1); e0 /= L0; e1 /= L1
def in_sheet(xy, shrink=0.9):
    r = xy - c; return (np.abs(r @ e0) < shrink * L0 / 2) & (np.abs(r @ e1) < shrink * L1 / 2)
keep = np.zeros(len(cl), bool)
for k in np.argsort(-cnt)[:20]:
    tri = Tt[cl == k]; cen = V[tri].mean(1); foot = cen[cen[:, 2] < 0.03]
    if len(foot) > 50 and in_sheet(foot[:, :2]).mean() > 0.4 and cnt[k] > 0.01 * cnt.max(): keep[cl == k] = True
    print("cluster %d tris %d on_sheet %.2f zmax %.3f -> %s" % (k, cnt[k], in_sheet(foot[:, :2]).mean() if len(foot) else 0, cen[:, 2].max(), "keep" if keep[cl == k].any() else "drop"))
m.remove_triangles_by_mask(~keep); m.remove_unreferenced_vertices(); V = np.asarray(m.vertices); Tt = np.asarray(m.triangles)
# major axis of the footprint, seam = density minimum near the middle
xy = V[:, :2]; mu = xy.mean(0); u, s, vt = np.linalg.svd(xy - mu, full_matrices=False); ax = vt[0]; perp = vt[1]
t = (xy - mu) @ ax; lo_t, hi_t = np.percentile(t, [1, 99]); bins = np.linspace(lo_t, hi_t, 120); pp = (xy - mu) @ perp; idx = np.clip(np.digitize(t, bins) - 1, 0, len(bins) - 2)
width = np.array([np.ptp(pp[idx == i]) if (idx == i).sum() > 20 else np.inf for i in range(len(bins) - 1)])  # silhouette width per slice
mid = (bins[:-1] + bins[1:]) / 2; win = np.abs(mid) < 0.35 * (hi_t - lo_t) / 2; seam_t = mid[win][np.argmin(width[win])]
print("waist width %.3f m at t=%.3f (slice widths min/max %.3f %.3f)" % (width[win].min(), seam_t, width[np.isfinite(width)].min(), width[np.isfinite(width)].max()))
tri_t = t[Tt].mean(1); side_a = tri_t < seam_t
# the half with more volume above 3 cm holds the headphones = base; the other is the lid
ha = (V[Tt][side_a][:, :, 2] > 0.03).mean(); hb = (V[Tt][~side_a][:, :, 2] > 0.03).mean()
base_mask = side_a if ha >= hb else ~side_a
def part(mask):
    p = o3d.geometry.TriangleMesh(m); p.remove_triangles_by_mask(~mask); p.remove_unreferenced_vertices(); return p
base, lid = part(base_mask), part(~base_mask)
seam_pt = mu + ax * seam_t; lid_dir = ax if (np.asarray(lid.vertices)[:, :2].mean(0) - seam_pt) @ ax > 0 else -ax
# refine the hinge from the two halves' wall rings (z 15..35 mm, strap loop and headphones excluded): the fabric hinge is the
# perpendicular bisector of the two ring centres, and the pivot sits midway between the two outer edges so that a pi flip
# lands the lid outline on the base outline (the SVD axis of the whole footprint is biased by the loop and the seam fabric)
def ring(p):
    v = np.asarray(p.vertices); v = v[(v[:, 2] > 0.015) & (v[:, 2] < 0.035)][:, :2].astype(np.float32); hull = cv2.convexHull(v)[:, 0, :]; M = cv2.moments(hull)
    return np.array([M["m10"], M["m01"]]) / M["m00"], hull
cb, hb = ring(base); cl, hl = ring(lid); u = cl - cb; u /= np.linalg.norm(u); perp_r = np.array([-u[1], u[0]])
far_l = (hl @ u).max(); far_b = (hb @ u).min(); pivot = ((far_l + far_b) / 2 - cb @ u) * u + cb
mir = hl - 2 * np.outer((hl - pivot) @ u, u)  # lid hull after the flip
print("hinge refine: axis turned %.1f deg, pivot moved %.1f mm along the case; base/lid outer edges at %.0f/%.0f mm from the pivot" % (np.degrees(np.arccos(abs(perp_r @ perp))), np.linalg.norm(pivot - seam_pt) * 1000, (far_b - pivot @ u) * 1000, (far_l - pivot @ u) * 1000))
Mm = cv2.moments(mir.astype(np.float32)); cm = np.array([Mm["m10"], Mm["m01"]]) / Mm["m00"]
print("flipped lid ring vs base ring: centroid offset %.1f mm along the case, %.1f mm across; hull-to-hull chamfer %.1f mm" % (((cm - cb) @ u) * 1000, ((cm - cb) @ perp_r) * 1000, np.mean([np.linalg.norm(hb - q, axis=1).min() for q in mir]) * 1000))
seam_pt, perp, ax, lid_dir = pivot, perp_r, u, u
# re-split at the refined pivot line: the seam fabric strip between the waist and the pivot belongs to the base, so no lid
# triangle sits behind the hinge (a strip behind the pivot sweeps below the table while the lid rotates and shoves the base)
base_mask = ((V[Tt].mean(1)[:, :2] - pivot) @ u) < 0; base, lid = part(base_mask), part(~base_mask)
# pull tab on the lid's outer edge (the thing the robot pinches): cloud points above the lid rim near the far edge
cloud = o3d.io.read_point_cloud(str(W / "cloud_table.ply")); P = np.asarray(cloud.points) * float(sh["scale"]); tp = (P[:, :2] - pivot) @ u; spp = (P[:, :2] - pivot) @ perp
lv = np.asarray(lid.vertices); rim = float(np.percentile(lv[((lv[:, :2] - pivot) @ u) > far_l - pivot @ u - 0.02, 2], 95))
cand = (P[:, 2] > rim + 0.004) & (P[:, 2] < 0.12) & (tp > far_l - pivot @ u - 0.04) & (tp < far_l - pivot @ u + 0.03) & (np.abs(spp - cl @ perp + pivot @ perp) < 0.12)
hist, eb = np.histogram(spp[cand], bins=np.arange(-0.13, 0.13, 0.005)); k = np.argmax(hist); lo = k
while lo > 0 and hist[lo - 1] > 0.25 * hist[k]: lo -= 1
hi = k
while hi < len(hist) - 1 and hist[hi + 1] > 0.25 * hist[k]: hi += 1
tab_pts = cand & (spp >= eb[lo]) & (spp < eb[hi + 1]); tab = None; up = tab_pts & (P[:, 2] > rim + 0.010)  # width from the clear part above the rim noise
if tab_pts.sum() > 2000:
    s_lo, s_hi = np.percentile(spp[up], [5, 95])
    tab = {"centre_ts_m": [float(np.median(tp[tab_pts])), float((s_lo + s_hi) / 2)], "width_m": float(s_hi - s_lo), "thickness_m": 0.010,
           "top_z_m": float(np.percentile(P[tab_pts, 2], 98)), "rim_z_m": rim, "n_points": int(tab_pts.sum())}
    print("lid tab: %d cloud points, centre t %.0f s %.0f mm, width %.0f mm, top z %.0f mm, lid rim %.0f mm" % (tab_pts.sum(), tab["centre_ts_m"][0] * 1000, tab["centre_ts_m"][1] * 1000, tab["width_m"] * 1000, tab["top_z_m"] * 1000, rim * 1000))
else: print("lid tab: not found (%d points)" % tab_pts.sum())
# TSDF erodes the thin fabric rim (2 mm voxels, 8 mm truncation): lift each half's top wall band to the raw-cloud rim height,
# per 10 deg bin about the wall-ring centre, in the same radial band (outer 12 mm), capped at 8 mm; the top 12 mm of the wall stretch
def raise_rim(p, side):
    V = np.asarray(p.vertices); Mt = np.c_[(V[:, :2] - pivot) @ u, (V[:, :2] - pivot) @ perp, V[:, 2]]; ring = Mt[(Mt[:, 2] > 0.015) & (Mt[:, 2] < 0.035)][:, :2]; c = ring.mean(0)
    Pc = np.c_[tp, spp, P[:, 2]]; Pc = Pc[side(Pc[:, 0]) & (Pc[:, 2] > 0.012) & (Pc[:, 2] < 0.065) & (np.abs(Pc[:, 1]) < 0.13)]
    def polar(X): rel = X[:, :2] - c; return np.arctan2(rel[:, 1], rel[:, 0]), np.linalg.norm(rel, axis=1)
    am, rm = polar(Mt); ac, rc = polar(Pc); dz = np.zeros(len(V)); step = np.pi / 18; log = []
    for a0 in np.arange(-np.pi, np.pi, step):
        mb = (am >= a0) & (am < a0 + step) & (Mt[:, 2] > 0.012); mc = (ac >= a0) & (ac < a0 + step)
        if mb.sum() < 30 or mc.sum() < 50: continue
        r99 = np.percentile(rm[mb], 99); ob = mb & (rm > r99 - 0.008); oc = mc & (rc > r99 - 0.010) & (rc < r99 + 0.006)
        if oc.sum() < 20: continue
        zm = np.percentile(Mt[ob, 2], 95); d = float(np.clip(np.percentile(Pc[oc, 2], 90) - zm, 0, 0.008)); band = mb & (rm > r99 - 0.012) & (Mt[:, 2] > zm - 0.012)
        dz[band] = d * np.clip((Mt[band, 2] - (zm - 0.012)) / 0.012, 0, 1); log.append((int(np.degrees(a0)), int(zm * 1000), int(d * 1000)))
    V[:, 2] += dz; p.vertices = o3d.utility.Vector3dVector(V); print("rim raise (deg: mesh rim mm, +lift mm):", " ".join("%d:%d+%d" % l for l in log))
    sm = np.asarray(o3d.geometry.TriangleMesh(p).filter_smooth_taubin(number_of_iterations=10).vertices); V[dz > 0] = sm[dz > 0]; p.vertices = o3d.utility.Vector3dVector(V)  # the TSDF rim is ragged: smooth the lifted band only
raise_rim(base, lambda t: (t < -0.005) & (t > -0.2)); raise_rim(lid, lambda t: (t > 0.005) & (t < 0.2))
o3d.io.write_triangle_mesh(str(W / "base.ply"), base); o3d.io.write_triangle_mesh(str(W / "lid.ply"), lid)
hinge = {"point_m": [float(seam_pt[0]), float(seam_pt[1]), 0.0], "axis": [float(perp[0]), float(perp[1]), 0.0], "lid_side_dir": [float(lid_dir[0]), float(lid_dir[1]), 0.0],
         "base_bbox_m": [np.asarray(base.vertices).min(0).tolist(), np.asarray(base.vertices).max(0).tolist()], "lid_bbox_m": [np.asarray(lid.vertices).min(0).tolist(), np.asarray(lid.vertices).max(0).tolist()], "lid_tab": tab}
json.dump(hinge, open(W / "hinge.json", "w"), indent=1); print(json.dumps(hinge, indent=1))
# QC: top view, base red tint, lid blue tint, seam line green
res = 800; allv = V; lo = allv[:, :2].min(0) - 0.02; sc = (res - 1) / (np.ptp(allv[:, :2], axis=0).max() + 0.04); img = np.full((res, res, 3), 255, np.uint8)
for p, col in ((base, (80, 80, 255)), (lid, (255, 120, 80))):
    pv = np.asarray(p.vertices); o = np.argsort(pv[:, 2]); uv = ((pv[o][:, :2] - lo) * sc).astype(int); img[res - 1 - uv[:, 1], uv[:, 0]] = col
a, b = seam_pt - perp * 0.3, seam_pt + perp * 0.3; pa, pb = ((a - lo) * sc).astype(int), ((b - lo) * sc).astype(int)
cv2.line(img, (pa[0], res - 1 - pa[1]), (pb[0], res - 1 - pb[1]), (0, 180, 0), 2); cv2.imwrite(str(W / "split_view.jpg"), img)

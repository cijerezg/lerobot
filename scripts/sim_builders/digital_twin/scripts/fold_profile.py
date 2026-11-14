"""Step 1 of the tri-fold kit: PCA-align the fused kit mesh (long axis -> x), build a 2 mm height map, project the ridge height onto x
and find the two hinge valleys between the three raised panels. Writes work/bits_kit_fold/profile.json + profile.png + heightmap.npz.
usage: fold_profile.py (MUJOCO_GL unused)"""
import json, pathlib, numpy as np, open3d as o3d, cv2
W = pathlib.Path("assets/sim_sources/digital_twin/work/bits_kit"); O = pathlib.Path("assets/sim_sources/digital_twin/work/bits_kit_fold"); O.mkdir(exist_ok=True)
m = o3d.io.read_triangle_mesh(str(W / "mesh_object.ply")); V = np.asarray(m.vertices); C = np.asarray(m.vertex_colors)
pc = o3d.io.read_point_cloud(str(W / "cloud_table.ply")); P = np.asarray(pc.points)
# PCA on the xy footprint of the mesh (z >= 3 mm to skip the table-level fan), long axis -> +x
top = V[V[:, 2] > 0.003]; c = top[:, :2].mean(0); U, S, Vt = np.linalg.svd(top[:, :2] - c, full_matrices=False); ax = Vt[0]
if ax[0] < 0: ax = -ax
R = np.array([[ax[0], ax[1]], [-ax[1], ax[0]]]); yaw = float(np.degrees(np.arctan2(ax[1], ax[0]))); print("long axis yaw %.2f deg, centre %s" % (yaw, c.round(4)))
def align(X): Y = X.copy(); Y[:, :2] = (X[:, :2] - c) @ R.T; return Y
Va = align(V); Pa = align(P)
# 2 mm height map from the mesh vertices (max z per cell) and from the cloud (95th pct), in the aligned frame
res = 0.002; x0, x1 = Va[:, 0].min() - 0.004, Va[:, 0].max() + 0.004; y0, y1 = Va[:, 1].min() - 0.004, Va[:, 1].max() + 0.004
nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
def hmap(X, q):
    ix = ((X[:, 0] - x0) / res).astype(int); iy = ((X[:, 1] - y0) / res).astype(int); ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    H = np.full((ny, nx), np.nan); key = iy[ok] * nx + ix[ok]; z = X[ok, 2]; order = np.argsort(key); key, z = key[order], z[order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    for s, e in zip(starts, np.r_[starts[1:], len(key)]): H.flat[key[s]] = np.percentile(z[s:e], q)
    return H
Hm = hmap(Va, 100); Hc = hmap(Pa[Pa[:, 2] > -0.005], 95)
# footprint: cells where the mesh is above 3 mm; interior y band = central 60 % of the footprint width so the rim (blue nylon, raised) does not dominate
fp = Hm > 0.003; cols = np.nonzero(fp.any(0))[0]; rows = np.nonzero(fp.any(1))[0]; xs = x0 + (np.arange(nx) + 0.5) * res; ys = y0 + (np.arange(ny) + 0.5) * res
yc0, yc1 = ys[rows[0]], ys[rows[-1]]; print("footprint x %.1f..%.1f mm (%.1f), y %.1f..%.1f mm (%.1f)" % (xs[cols[0]] * 1e3, xs[cols[-1]] * 1e3, (xs[cols[-1]] - xs[cols[0]]) * 1e3, yc0 * 1e3, yc1 * 1e3, (yc1 - yc0) * 1e3))
inner = (ys > yc0 + 0.2 * (yc1 - yc0)) & (ys < yc1 - 0.2 * (yc1 - yc0))
prof = {}
for name, H in (("mesh_max", Hm), ("mesh_p90", None), ("cloud_p95", Hc)):
    if H is None: continue
    Hi = np.where(inner[:, None], H, np.nan); prof[name + "_median"] = np.nanmedian(Hi, 0); prof[name + "_p90"] = np.nanpercentile(Hi, 90, axis=0); prof[name + "_mean"] = np.nanmean(Hi, 0)
# smooth the median mesh profile 10 mm, find the two deepest valleys with raised regions on both sides, at least 80 mm from the ends
pm = prof["mesh_max_median"].copy(); pm[np.isnan(pm)] = 0; k = 5; sm = np.convolve(pm, np.ones(k) / k, mode="same")
lo, hi = cols[0] + 40, cols[-1] - 40  # 80 mm from each end
cand = [i for i in range(lo + 1, hi) if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1]]
# valley prominence: min over the lower of the two side maxima within 60 mm minus the valley value
def prom(i):
    L = sm[max(cols[0], i - 40):i].max(); Rr = sm[i + 1:min(cols[-1], i + 40)].max(); return min(L, Rr) - sm[i]
cand = sorted(cand, key=prom, reverse=True); picks = []
for i in cand:
    if all(abs(i - j) > 40 for j in picks): picks.append(i)
    if len(picks) == 2: break
picks = sorted(picks); fx = [float(xs[i]) for i in picks]; print("fold lines x = %.1f, %.1f mm (valley heights %.1f, %.1f mm; prominence %.1f, %.1f mm)" % (fx[0] * 1e3, fx[1] * 1e3, sm[picks[0]] * 1e3, sm[picks[1]] * 1e3, prom(picks[0]) * 1e3, prom(picks[1]) * 1e3))
edges = [float(xs[cols[0]]), fx[0], fx[1], float(xs[cols[-1]])]; widths = [1e3 * (edges[i + 1] - edges[i]) for i in range(3)]; print("panel widths (edge to fold, fold to fold, fold to edge) mm:", np.round(widths, 1))
# per-panel heights: slab = 20th pct of the interior height map (fabric floor between contents), contents = 90th pct, rim = max near the y edges
stats = []
for i in range(3):
    sel = (xs > edges[i] + 0.012) & (xs < edges[i + 1] - 0.012); Hi = Hm[:, sel]; Hin = Hi[inner]; Hin = Hin[~np.isnan(Hin)]
    stats.append(dict(x_mm=[edges[i] * 1e3, edges[i + 1] * 1e3], width_mm=widths[i], floor_p20_mm=float(np.percentile(Hin, 20) * 1e3), median_mm=float(np.median(Hin) * 1e3), p90_mm=float(np.percentile(Hin, 90) * 1e3), max_mm=float(Hin.max() * 1e3)))
    print("panel %d: floor p20 %.1f, median %.1f, p90 %.1f, max %.1f mm" % (i, stats[-1]["floor_p20_mm"], stats[-1]["median_mm"], stats[-1]["p90_mm"], stats[-1]["max_mm"]))
# plot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw=dict(height_ratios=[1.2, 1]))
axs[0].imshow(np.nan_to_num(Hm) * 1e3, origin="lower", extent=[x0 * 1e3, x1 * 1e3, y0 * 1e3, y1 * 1e3], cmap="viridis", vmin=0, vmax=30); axs[0].set_title("mesh height map (mm), aligned frame"); [axs[0].axvline(f * 1e3, color="r") for f in fx]
axs[0].axhline((yc0 + 0.2 * (yc1 - yc0)) * 1e3, color="w", ls=":"); axs[0].axhline((yc1 - 0.2 * (yc1 - yc0)) * 1e3, color="w", ls=":")
for kname in ("mesh_max_median", "mesh_max_p90", "cloud_p95_median"): axs[1].plot(xs * 1e3, prof[kname] * 1e3, label=kname)
axs[1].plot(xs * 1e3, sm * 1e3, "k", lw=2, label="mesh median smoothed"); [axs[1].axvline(f * 1e3, color="r") for f in fx]; axs[1].legend(); axs[1].set_xlabel("x mm"); axs[1].set_ylabel("height mm"); axs[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(O / "profile.png", dpi=110)
np.savez(O / "heightmap.npz", Hm=Hm, Hc=Hc, xs=xs, ys=ys, x0=x0, y0=y0, res=res)
json.dump(dict(centre_table_xy=c.tolist(), yaw_deg=yaw, R_rows=R.tolist(), fold_x_m=fx, edges_x_m=edges, panel_widths_mm=widths, footprint_y_m=[yc0, yc1], panels=stats,
               profile_x_m=xs.tolist(), profile_mesh_median_m=np.nan_to_num(prof["mesh_max_median"]).tolist()), open(O / "profile.json", "w"), indent=1)

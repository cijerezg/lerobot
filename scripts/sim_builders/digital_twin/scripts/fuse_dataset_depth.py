"""Fuse the recorded wrist depth of one episode into a coloured point cloud in the sim table frame.

Every depth frame: sim qpos from the recorded state, wrist camera pose from MuJoCo, depth unprojected (0.07-0.45 m, robot
pixels masked), world points + wrist RGB colour. Voxel-downsampled cloud -> PLY + npz; table plane statistics on free table
cells, object slice clusters with footprints, top-down (colour + height) and oblique renders, stats.json.

usage: fuse_dataset_depth.py ROOT EP OUT --tag base [--dpos mm mm mm] [--drot deg deg deg] [--shift 0] [--stride_px 2] [--voxel 0.002]
"""
import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dataset_depth_lib as L  # noqa: E402

TABLE_XY = (0.3467, 0.0244, 0.3708, 0.5053)   # sim table box centre + half sizes (scene.xml)


def voxel(points, colours, size):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colours[:, ::-1].astype(np.float64) / 255.0)   # BGR -> RGB
    return pc.voxel_down_sample(size)


def fuse(root, ep, sim, states, frames, row, shift, stride_px, voxel_size, rows_max=380, sim_mask=False, raw_every=16):
    """rows_max: image rows below it are dropped (the real finger tips never rise above row ~400; the sim finger silhouette
    misses the real fingers by up to 4 cm, so sim_mask=True alone leaks fingers into the cloud)."""
    chunks, raw, raw_frame = [], [], []
    P, C = [], []
    row_cut = np.zeros((L.H, L.W), bool)
    row_cut[rows_max:] = True
    for k, (f, bgr) in enumerate(L.iter_rgb(root, ep, row, frames)):
        s = states[int(np.clip(f + shift, 0, len(states) - 1))]
        mask = row_cut | sim.robot_mask(s) if sim_mask else row_cut
        pts_c, uv = L.unproject(L.load_depth(root, ep, f), stride=stride_px, exclude=mask)
        if len(pts_c) == 0:
            continue
        t, R = sim.camera_pose(s)
        pts_w = (pts_c @ R.T + t).astype(np.float32)
        P.append(pts_w)
        C.append(bgr[uv[:, 1], uv[:, 0]])
        raw.append(pts_w[::raw_every])
        raw_frame.append(np.full(len(raw[-1]), f, np.int32))
        if len(P) == 100 or k == len(frames) - 1:
            chunks.append(voxel(np.concatenate(P), np.concatenate(C), voxel_size))
            P, C = [], []
            print(f"  fused {k + 1}/{len(frames)} frames, {sum(len(c.points) for c in chunks)} voxel points", flush=True)
    pc = chunks[0]
    for c in chunks[1:]:
        pc += c
    pc = pc.voxel_down_sample(voxel_size)
    return pc, np.concatenate(raw), np.concatenate(raw_frame)


def table_stats(pts, pc_pts, cell=0.02, plane=None):
    """RANSAC plane on the raw subsample near z 0 (or a given plane); free table cells = 2 cm xy cells with no points > 8 mm above it."""
    pts = pts.astype(np.float64)
    if plane is None:
        near = pts[np.abs(pts[:, 2]) < 0.05]
        rng = np.random.default_rng(0)
        sub = near[rng.choice(len(near), min(60000, len(near)), replace=False)]
        n, d, _ = L.ransac_plane(sub, n_iter=400, thresh=0.005, rng=rng)
    else:
        n, d = plane
    h = pts @ n + d                       # signed height above the fitted plane
    tilt = float(np.degrees(np.arccos(n[2])))
    z_centre = -(n[0] * TABLE_XY[0] + n[1] * TABLE_XY[1] + d) / n[2]
    cx, cy = TABLE_XY[0] + TABLE_XY[2], TABLE_XY[1]
    ij = np.floor((pts[:, :2] - [TABLE_XY[0] - TABLE_XY[2], TABLE_XY[1] - TABLE_XY[3]]) / cell).astype(int)
    on_table = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < int(2 * TABLE_XY[2] / cell)) & (ij[:, 1] < int(2 * TABLE_XY[3] / cell))
    key = ij[:, 0] * 1000 + ij[:, 1]
    keys, inv, counts = np.unique(key[on_table], return_inverse=True, return_counts=True)
    hh = h[on_table]
    p95 = np.full(len(keys), -np.inf)
    np.maximum.at(p95, inv, hh)            # max height per cell
    # p90-p10 thickness per cell via sorting
    order = np.lexsort((hh, inv))
    hs, invs = hh[order], inv[order]
    starts = np.searchsorted(invs, np.arange(len(keys)))
    ends = np.append(starts[1:], len(hs))
    thick, p10c, p90c = np.zeros(len(keys)), np.zeros(len(keys)), np.zeros(len(keys))
    for i, (a, b) in enumerate(zip(starts, ends)):
        seg = hs[a:b]
        p10c[i], p90c[i] = np.percentile(seg, [10, 90])
        thick[i] = p90c[i] - p10c[i]
    free = (counts >= 30) & (p95 < 0.008)
    free_pts = free[inv]
    res = np.abs(hh[free_pts])
    return dict(normal=n.round(5).tolist(), tilt_deg=round(tilt, 3), plane_z_at_table_centre_mm=round(1e3 * z_centre, 2),
                cells_total=int(len(keys)), cells_free=int(free.sum()), free_points=int(free_pts.sum()),
                residual_median_mm=round(1e3 * float(np.median(res)), 2), residual_p90_mm=round(1e3 * float(np.percentile(res, 90)), 2),
                cell_thickness_p90_p10_median_mm=round(1e3 * float(np.median(thick[free])), 2), cell_thickness_p90_p10_p90_mm=round(1e3 * float(np.percentile(thick[free], 90)), 2)), (n, d), dict(keys=keys, free=free, thick=thick, counts=counts, cell=cell)


def objects(pc, plane, h_lo=0.006, h_hi=0.30, eps=0.01, min_points=20):
    n, d = plane
    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors)
    h = pts @ n + d
    ij = pts[:, :2]
    on = (ij[:, 0] > TABLE_XY[0] - TABLE_XY[2]) & (ij[:, 0] < TABLE_XY[0] + TABLE_XY[2]) & (ij[:, 1] > TABLE_XY[1] - TABLE_XY[3]) & (ij[:, 1] < TABLE_XY[1] + TABLE_XY[3])
    sel = (h > h_lo) & (h < h_hi) & on
    sub = o3d.geometry.PointCloud()
    sub.points = o3d.utility.Vector3dVector(pts[sel])
    sub.colors = o3d.utility.Vector3dVector(cols[sel])
    labels = np.asarray(sub.cluster_dbscan(eps=eps, min_points=min_points))
    rows = []
    for lab in range(labels.max() + 1):
        m = labels == lab
        if m.sum() < 200:
            continue
        q = pts[sel][m]
        hq = h[sel][m]
        xy = q[:, :2] - q[:, :2].mean(0)
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        proj = xy @ vt.T
        ext = np.percentile(proj, 98, axis=0) - np.percentile(proj, 2, axis=0)
        rows.append(dict(cluster=int(lab), points=int(m.sum()), centre_xy_mm=(1e3 * q[:, :2].mean(0)).round(0).tolist(),
                         footprint_mm=(1e3 * ext).round(0).tolist(), height_p95_mm=round(1e3 * float(np.percentile(hq, 95)), 1),
                         height_max_mm=round(1e3 * float(hq.max()), 1), colour_rgb=(255 * cols[sel][m].mean(0)).round(0).tolist()))
    rows.sort(key=lambda r: -r["points"])
    return rows, sub, labels


def render(pc, plane, rows, out, tag):
    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors)
    n, d = plane
    h = pts @ n + d
    on = (np.abs(pts[:, 0] - TABLE_XY[0]) < TABLE_XY[2] + 0.05) & (np.abs(pts[:, 1] - TABLE_XY[1]) < TABLE_XY[3] + 0.05) & (h > -0.03) & (h < 0.35)
    p, c, hh = pts[on], cols[on], h[on]
    order = np.argsort(hh)                 # draw high points last
    fig, ax = plt.subplots(1, 2, figsize=(18, 7.5))
    for a, colour, title in ((ax[0], c[order], "wrist RGB"), (ax[1], None, "height above fitted table (mm)")):
        if colour is None:
            sc = a.scatter(p[order, 1], p[order, 0], c=1e3 * hh[order], s=0.4, cmap="turbo", vmin=-10, vmax=120, linewidths=0)
            fig.colorbar(sc, ax=a, fraction=0.03)
        else:
            a.scatter(p[order, 1], p[order, 0], c=colour, s=0.4, linewidths=0)
        a.add_patch(plt.Rectangle((TABLE_XY[1] - TABLE_XY[3], TABLE_XY[0] - TABLE_XY[2]), 2 * TABLE_XY[3], 2 * TABLE_XY[2], fill=False, ec="k", lw=0.8))
        a.plot(0, 0, "k^", ms=8)
        for r in rows[:25]:
            a.annotate(str(r["cluster"]), (r["centre_xy_mm"][1] / 1e3, r["centre_xy_mm"][0] / 1e3), color="w" if colour is not None else "k", fontsize=7, ha="center")
        a.set_aspect("equal")
        a.set_xlabel("table y (m), robot base at the triangle, looking from above")
        a.set_ylabel("table x (m)")
        a.set_xlim(TABLE_XY[1] + TABLE_XY[3] + 0.05, TABLE_XY[1] - TABLE_XY[3] - 0.05)   # y left = robot's left when x points away
        a.set_ylim(TABLE_XY[0] - TABLE_XY[2] - 0.05, TABLE_XY[0] + TABLE_XY[2] + 0.05)
        a.set_title(f"{tag}: {title}")
    fig.tight_layout()
    fig.savefig(out / f"topdown_{tag}.png", dpi=120)
    plt.close(fig)
    rng = np.random.default_rng(0)
    i = rng.choice(len(p), min(250000, len(p)), replace=False)
    fig = plt.figure(figsize=(14, 9))
    a = fig.add_subplot(111, projection="3d")
    a.scatter(p[i, 0], p[i, 1], hh[i], c=c[i], s=0.3, linewidths=0)
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_zlabel("height (m)")
    a.set_box_aspect((2 * TABLE_XY[2] + 0.1, 2 * TABLE_XY[3] + 0.1, 0.25))
    a.set_zlim(-0.03, 0.22)
    a.view_init(elev=35, azim=-150)
    a.set_title(f"{tag}: oblique view of the fused cloud")
    fig.tight_layout()
    fig.savefig(out / f"oblique_{tag}.png", dpi=110)
    plt.close(fig)


def side_profile(raw, plane, out, tag):
    """Height histogram of the raw table points and a thin x slice: shows the smear directly."""
    n, d = plane
    h = raw @ n + d
    on = (np.abs(raw[:, 0] - TABLE_XY[0]) < TABLE_XY[2]) & (np.abs(raw[:, 1] - TABLE_XY[1]) < TABLE_XY[3])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].hist(1e3 * h[on & (np.abs(h) < 0.05)], bins=200, color="k")
    ax[0].set_xlabel("height above fitted table (mm), all raw points within 5 cm")
    ax[0].set_yscale("log")
    sl = on & (np.abs(raw[:, 0] - 0.40) < 0.01) & (h > -0.03) & (h < 0.25)
    ax[1].scatter(raw[sl, 1], 1e3 * h[sl], s=0.3, c="k", linewidths=0)
    ax[1].set_xlabel("y (m), slice |x - 0.40| < 1 cm")
    ax[1].set_ylabel("height (mm)")
    ax[1].set_ylim(-30, 250)
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(out / f"profile_{tag}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("ep", type=int)
    ap.add_argument("out")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--dpos", type=float, nargs=3, default=(0, 0, 0), help="camera position offset in gripper_end frame, mm")
    ap.add_argument("--drot", type=float, nargs=3, default=(0, 0, 0), help="camera rotation offset (rotvec in camera frame), deg")
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--stride_px", type=int, default=2)
    ap.add_argument("--voxel", type=float, default=0.002)
    ap.add_argument("--frames", type=int, nargs=2, default=None, help="only depth frames in [lo, hi)")
    ap.add_argument("--exclude_held", action="store_true", help="drop frames with the gripper closed after its first opening (object in hand)")
    ap.add_argument("--rows_max", type=int, default=380)
    ap.add_argument("--sim_mask", action="store_true")
    ap.add_argument("--h_lo", type=float, default=0.006, help="object slice starts this far above the plane (m)")
    ap.add_argument("--eps", type=float, default=0.01, help="DBSCAN radius (m)")
    ap.add_argument("--plane_from", default=None, help="tag whose fitted table plane is used instead of fitting one (windows with little free table)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    states, row = L.episode_states(a.root, a.ep)
    frames = np.array(L.depth_frames(a.root, a.ep))
    if a.frames is not None:
        frames = frames[(frames >= a.frames[0]) & (frames < a.frames[1])]
    if a.exclude_held:
        closed = states[:, 6] > -120
        first_open = int(np.argmax(~closed))
        held = closed & (np.arange(len(states)) >= first_open)
        frames = frames[~held[frames]]
    frames = [int(f) for f in frames]
    cloud_file = out / f"fused_{a.tag}.npz"
    if cloud_file.exists():                 # statistics / renders only, the fusion is cached
        z = np.load(cloud_file)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(z["points"].astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(z["colours"].astype(np.float64))
        raw, raw_frame = z["raw"], z["raw_frame"]
    else:
        sim = L.Sim()
        sim.set_camera_offset(np.asarray(a.dpos) / 1e3, np.deg2rad(a.drot))
        pc, raw, raw_frame = fuse(a.root, a.ep, sim, states, frames, row, a.shift, a.stride_px, a.voxel, a.rows_max, a.sim_mask)
        np.savez_compressed(cloud_file, points=np.asarray(pc.points).astype(np.float32), colours=np.asarray(pc.colors).astype(np.float32), raw=raw, raw_frame=raw_frame)
        o3d.io.write_point_cloud(str(out / f"fused_{a.tag}.ply"), pc)
    pts = np.asarray(pc.points)
    stats = dict(tag=a.tag, dpos_mm=list(a.dpos), drot_deg=list(a.drot), shift=a.shift, frames=len(frames), frame_window=a.frames,
                 exclude_held=a.exclude_held, rows_max=a.rows_max, sim_mask=a.sim_mask, voxel_mm=1e3 * a.voxel,
                 voxel_points=int(len(pts)), raw_subsample_points=int(len(raw)),
                 extent_min_m=pts.min(0).round(3).tolist(), extent_max_m=pts.max(0).round(3).tolist())
    given = None
    if a.plane_from is not None:
        z = np.load(out / f"plane_{a.plane_from}.npz")
        given = (z["n"], float(z["d"]))
    tstats, plane, cells = table_stats(raw, pts, plane=given)
    tstats["plane_from"] = a.plane_from
    stats["table"] = tstats
    rows, sub, labels = objects(pc, plane, h_lo=a.h_lo, eps=a.eps)
    stats["object_slice"] = dict(h_lo_mm=1e3 * a.h_lo, eps_mm=1e3 * a.eps)
    stats["objects"] = rows
    np.savez(out / f"plane_{a.tag}.npz", n=plane[0], d=plane[1])
    json.dump(stats, open(out / f"stats_{a.tag}.json", "w"), indent=1)
    render(pc, plane, rows, out, a.tag)
    side_profile(raw, plane, out, a.tag)
    print(json.dumps({k: v for k, v in stats.items() if k != "objects"}, indent=1))
    print("objects (cluster, points, centre xy mm, footprint mm, height p95 / max mm):")
    for r in rows[:30]:
        print(f"  {r['cluster']:3d} {r['points']:6d} {r['centre_xy_mm']} {r['footprint_mm']} {r['height_p95_mm']:.0f} / {r['height_max_mm']:.0f}")

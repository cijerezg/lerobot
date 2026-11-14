"""vggt.npz -> table frame (z up, origin = sheet-ish centre), filtered cloud PLY + ortho renders. usage: table_frame.py work_dir"""
import sys, pathlib, numpy as np, cv2, open3d as o3d
W = pathlib.Path(sys.argv[1]); d = np.load(W / "vggt.npz")
pts = d["world_points"].reshape(-1, 3); conf = d["depth_conf"].reshape(-1); col = d["images"].reshape(-1, 3)
keep = conf > np.percentile(conf, 40); pts, col = pts[keep], col[keep]
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64))); pc.colors = o3d.utility.Vector3dVector(col / 255.0)
pc = pc.voxel_down_sample(voxel_size=np.ptp(pts, axis=0).max() / 800)
plane, inl = pc.segment_plane(distance_threshold=np.ptp(pts, axis=0).max() / 400, ransac_n=3, num_iterations=2000)
n = np.array(plane[:3]); n /= np.linalg.norm(n)
P = np.asarray(pc.points); C = np.asarray(pc.colors)
# camera centres are above the table: make normal point towards them
cams = -np.einsum("nij,nj->ni", d["extrinsics"][:, :3, :3].transpose(0, 2, 1), d["extrinsics"][:, :3, 3])
if np.mean(cams @ n + plane[3]) < 0: n, plane = -n, [-x for x in plane]
# rotation taking n -> z
z = n; x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x); y = np.cross(z, x); R = np.stack([x, y, z])
Pt = (P @ R.T); Pt[:, 2] += plane[3] * 1.0 if False else 0
h = P @ n + plane[3]; Pt[:, 2] = h
inl_pts = Pt[inl]; ctr = np.median(inl_pts[:, :2], axis=0); Pt[:, :2] -= ctr
print("table inliers", len(inl), "of", len(P), "; table extent xy", np.ptp(inl_pts[:, 0]), np.ptp(inl_pts[:, 1]))
np.savez(W / "table_frame.npz", R=R, n=n, d=plane[3], ctr=ctr)
o3d.io.write_point_cloud(str(W / "cloud_table.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pt)).__class__(points=o3d.utility.Vector3dVector(Pt), colors=o3d.utility.Vector3dVector(C)) if False else _pc) if False else None
pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pt)); pc2.colors = o3d.utility.Vector3dVector(C); o3d.io.write_point_cloud(str(W / "cloud_table.ply"), pc2)
def ortho(ax0, ax1, sel, name, res=900, flip=False):
    Q = Pt[sel]; Cq = C[sel]; lo = Q[:, [ax0, ax1]].min(0); hi = Q[:, [ax0, ax1]].max(0); s = (res - 1) / (hi - lo).max()
    img = np.full((res, res, 3), 255, np.uint8); order = np.argsort(Q[:, 3 - ax0 - ax1])
    uv = ((Q[order][:, [ax0, ax1]] - lo) * s).astype(int); img[res - 1 - uv[:, 1], uv[:, 0]] = (Cq[order][:, ::-1] * 255).astype(np.uint8)
    cv2.imwrite(str(W / name), img); return lo, s
lo, s = ortho(0, 1, np.ones(len(Pt), bool), "ortho_top.png")
np.savez(W / "ortho_top_map.npz", lo=lo, s=s)
ortho(0, 2, np.abs(Pt[:, 1]) < np.ptp(inl_pts[:, 1]) * 0.08, "ortho_side_x.png")
ortho(1, 2, np.abs(Pt[:, 0]) < np.ptp(inl_pts[:, 0]) * 0.08, "ortho_side_y.png")
print("height range above table (unitless):", np.percentile(Pt[:, 2], [1, 50, 99]))

"""Metric TSDF fusion of the D405 depth with VGGT poses (scaled by scale.json), table frame from RANSAC on the metric cloud,
object = largest point cluster above the table. No sheet. Optional pose refinement: open3d RGB-D odometry on consecutive pairs
+ VGGT-nearest loop pairs, pose graph optimisation (init = VGGT relative poses).
usage: fuse_d405.py scan_dir work_dir [--voxel_mm 2] [--pad_m 0.03] [--refine none|pg|f2m] [--poses poses.npz] [--src d405|vggt] [--zmax 0.45] [--pick K] [--min_h 0.006] [--tag name]
writes work_dir/{poses.npz, cloud_table.ply, mesh_scene<tag>.ply, mesh_object<tag>.ply, mesh_views<tag>.jpg, fuse<tag>.json}"""
import argparse, json, pathlib, cv2, numpy as np, open3d as o3d
p = argparse.ArgumentParser(); p.add_argument("scan"); p.add_argument("work"); p.add_argument("--voxel_mm", type=float, default=2.0); p.add_argument("--pad_m", type=float, default=0.03)
p.add_argument("--refine", default="none", help="none | pg | f2m (pose graph, then frame-to-model ICP)"); p.add_argument("--src", default="d405"); p.add_argument("--pick", type=int, default=-1); p.add_argument("--min_h", type=float, default=0.006)
p.add_argument("--tag", default=""); p.add_argument("--max_res", type=int, default=400); p.add_argument("--zmax", type=float, default=0.45); p.add_argument("--poses", default=None, help="reuse poses npz (skips refinement)"); a = p.parse_args()
S = pathlib.Path(a.scan); W = pathlib.Path(a.work); d = np.load(W / "vggt.npz"); K = json.load(open(S / "intrinsic.json")); s = json.load(open(W / "scale.json"))["scale_m_per_unit"]
gm = cv2.imread(str(S.parent / "gripper_mask.png"), 0) > 0; N = len(d["frames"]); stems = [pathlib.Path(str(f)).stem for f in d["frames"]]
intr = o3d.camera.PinholeCameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
def load(i):
    z = cv2.imread(str(S / "depth" / f"{stems[i]}.png"), -1).astype(np.float32) * K["depth_scale"]; z[(z < 0.07) | (z > a.zmax) | gm] = 0
    c = cv2.imread(str(S / "color" / f"{stems[i]}.jpg"))[:, :, ::-1]
    return o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(c)), o3d.geometry.Image(z), depth_scale=1.0, depth_trunc=a.zmax, convert_rgb_to_intensity=False)
E = np.tile(np.eye(4), (N, 1, 1)); E[:, :3] = d["extrinsics"]; E[:, :3, 3] *= s  # world->cam, metres
if a.poses: E = np.load(a.poses)["E"]; a.refine = str(np.load(a.poses)["refined"])
elif a.refine != "none":
    opt = o3d.pipelines.odometry.OdometryOption(depth_diff_max=0.02, depth_min=0.07, depth_max=a.zmax); jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    rgbd = [load(i) for i in range(N)]; cams = np.array([np.linalg.inv(E[i])[:3, 3] for i in range(N)]); pg = o3d.pipelines.registration.PoseGraph()
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(E[0]))); pairs = [(i, i + 1) for i in range(N - 1)] + [(i, i + 2) for i in range(N - 2)]
    for i in range(N):  # loop pairs: VGGT-nearest camera not within 5 frames
        dd = np.linalg.norm(cams - cams[i], axis=1); dd[max(0, i - 5):i + 6] = 1e9; j = int(np.argmin(dd))
        if dd[j] < 0.05 and (min(i, j), max(i, j)) not in pairs: pairs.append((min(i, j), max(i, j)))
    nok = 0; odo = {}
    for i, j in pairs:
        init = E[j] @ np.linalg.inv(E[i])  # cam_i -> cam_j
        ok, T, info = o3d.pipelines.odometry.compute_rgbd_odometry(rgbd[i], rgbd[j], intr, init, jac, opt)
        if not ok: continue
        nok += 1; odo[(i, j)] = T
        if j == i + 1: pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(pg.nodes[i].pose @ np.linalg.inv(T)))
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, j, T, info, uncertain=(j != i + 1)))
        while len(pg.nodes) <= j: pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(E[len(pg.nodes)])))  # odometry failed: fall back to VGGT pose
    while len(pg.nodes) < N: pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(E[len(pg.nodes)])))
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    o3d.pipelines.registration.global_optimization(pg, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(), o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                   o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=0.01, edge_prune_threshold=0.25, reference_node=0))
    E = np.array([np.linalg.inv(n.pose) for n in pg.nodes]); print(f"refine: {nok}/{len(pairs)} odometry pairs ok ({len(pairs) - 2 * N + 3} loop pairs)")
if a.refine == "f2m" and not a.poses:  # frame-to-model: point-to-plane ICP of each frame against the accumulated cloud, init = previous refined pose x pose-graph relative motion
    Epg = E.copy(); E = E.copy(); model = None; est = o3d.pipelines.registration.TransformationEstimationPointToPlane(); fits = []
    for i in range(N):
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(load(i), intr).voxel_down_sample(0.003); pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.01, 30))
        if model is not None:
            T = np.linalg.inv(E[i - 1] if i else E[0]) @ np.linalg.inv(Epg[i] @ np.linalg.inv(Epg[i - 1])) if False else np.linalg.inv(Epg[i] @ np.linalg.inv(Epg[i - 1]) @ E[i - 1])  # cam_i -> world init
            for md in (0.02, 0.008, 0.004):
                r = o3d.pipelines.registration.registration_icp(pc, model, md, T, est, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)); T = r.transformation
            fits.append((r.fitness, r.inlier_rmse)); E[i] = np.linalg.inv(T)
        pw = o3d.geometry.PointCloud(pc).transform(np.linalg.inv(E[i])); model = pw if model is None else (model + pw).voxel_down_sample(0.003)
        if i % 10 == 0: model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.01, 30))
    fits = np.array(fits); print(f"f2m: fitness median {np.median(fits[:, 0]):.2f} min {fits[:, 0].min():.2f}, rmse median {1e3 * np.median(fits[:, 1]):.2f} mm")
if not a.poses: np.savez(W / f"poses{a.tag}.npz", E=E, refined=a.refine)
# metric cloud for the table plane (every frame, pixel stride 4)
P, C = [], []
for i in range(N):
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(load(i), intr, E[i]); pc = pc.uniform_down_sample(16); P.append(np.asarray(pc.points)); C.append(np.asarray(pc.colors))
P = np.concatenate(P); C = np.concatenate(C); pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.colors = o3d.utility.Vector3dVector(C); pc = pc.voxel_down_sample(0.002)
plane, inl = pc.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=3000); n = np.array(plane[:3]); n /= np.linalg.norm(n); dd = plane[3]
cams = np.array([np.linalg.inv(E[i])[:3, 3] for i in range(N)])
if np.mean(cams @ n + dd) < 0: n, dd = -n, -dd
z = n; x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x); y = np.cross(z, x); R = np.stack([x, y, z]); Pq = np.asarray(pc.points); Cq = np.asarray(pc.colors)
Pt = Pq @ R.T; Pt[:, 2] = Pq @ n + dd; ctr = np.median(Pt[inl][:, :2], axis=0); Pt[:, :2] -= ctr
res = np.abs(Pq[inl] @ n + dd); print(f"table: {len(inl)} of {len(Pq)} pts, residual median {1e3 * np.median(res):.2f} mm p90 {1e3 * np.percentile(res, 90):.2f} mm")
pt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pt)); pt.colors = o3d.utility.Vector3dVector(Cq); o3d.io.write_point_cloud(str(W / f"cloud_table{a.tag}.ply"), pt)
# object = largest cluster above the table
up = np.nonzero((Pt[:, 2] > a.min_h) & (Pt[:, 2] < 0.3))[0]; sub = pt.select_by_index(up); lab = np.array(sub.cluster_dbscan(eps=0.008, min_points=15)); Pu = Pt[up]
labs, cnt = np.unique(lab[lab >= 0], return_counts=True); order = np.argsort(-cnt); cands = []
for r, k in enumerate(order[:6]):
    q = Pu[lab == labs[k]]; cands.append(dict(rank=r, n=int(cnt[k]), centre=q[:, :2].mean(0).round(3).tolist(), size_mm=(1e3 * np.ptp(q, axis=0)).round(0).tolist(), zmax_mm=round(1e3 * q[:, 2].max(), 1)))
    print("cluster", cands[-1])
k = order[a.pick if a.pick >= 0 else 0]; obj = Pu[lab == labs[k]]; cen_t = np.r_[obj[:, :2].mean(0), 0]; ext = np.ptp(obj, axis=0)
# TSDF volume (uniform) centred on the object, in world metres
Rt = np.vstack([R[:2], n]); Tm = np.eye(4); Tm[:3, :3] = Rt; Tm[:3, 3] = np.r_[-ctr, dd]  # world -> table frame
cen_w = np.linalg.inv(Tm) @ np.r_[cen_t[0], cen_t[1], ext[2] / 2, 1]; L = (ext.max() + 2 * a.pad_m) * 1.1; vox = a.voxel_mm / 1000; res_v = min(int(np.ceil(L / vox)), a.max_res); vox = L / res_v
print(f"volume side {L:.3f} m, {res_v} voxels, voxel {1e3 * vox:.2f} mm, centre (table) {cen_t[:2].round(3)}")
vol = o3d.pipelines.integration.UniformTSDFVolume(length=L, resolution=res_v, sdf_trunc=4 * vox, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, origin=cen_w[:3] - L / 2)
if a.src == "d405":
    for i in range(N): vol.integrate(load(i), intr, E[i])
else:
    Hv, Wv = d["depth"].shape[1:3]; gmv = cv2.resize(gm.astype(np.uint8), (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0
    for i in range(N):
        dep = d["depth"][i, ..., 0].astype(np.float32) * s; dep[gmv | (dep > a.zmax)] = 0; Kv = d["intrinsics"][i]
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(d["images"][i])), o3d.geometry.Image(dep), depth_scale=1.0, depth_trunc=a.zmax, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, o3d.camera.PinholeCameraIntrinsic(Wv, Hv, Kv[0, 0], Kv[1, 1], Kv[0, 2], Kv[1, 2]), E[i])
mesh = vol.extract_triangle_mesh(); mesh.transform(Tm); mesh.compute_vertex_normals(); o3d.io.write_triangle_mesh(str(W / f"mesh_scene{a.tag}.ply"), mesh)
V = np.asarray(mesh.vertices); half = ext[:2] / 2 + a.pad_m; keep = (np.abs(V[:, 0] - cen_t[0]) < half[0]) & (np.abs(V[:, 1] - cen_t[1]) < half[1]) & (V[:, 2] > 0.003)
ob = mesh.select_by_index(np.nonzero(keep)[0]); cl, cnt2, _ = ob.cluster_connected_triangles(); cl = np.asarray(cl); cnt2 = np.asarray(cnt2); big = np.nonzero(cnt2 > 0.02 * cnt2.max())[0]
ob.remove_triangles_by_mask(~np.isin(cl, big)); ob.remove_unreferenced_vertices(); o3d.io.write_triangle_mesh(str(W / f"mesh_object{a.tag}.ply"), ob); Vo = np.asarray(ob.vertices)
info = dict(src=a.src, refined=a.refine, n_frames=N, table_residual_mm=[float(1e3 * np.median(res)), float(1e3 * np.percentile(res, 90))], clusters=cands, picked=int(a.pick if a.pick >= 0 else 0),
            cloud_size_mm=(1e3 * ext).round(1).tolist(), mesh_size_mm=(1e3 * np.ptp(Vo, axis=0)).round(1).tolist() if len(Vo) else None, mesh_verts=len(Vo), mesh_clusters=len(big), voxel_mm=1e3 * vox)
json.dump(info, open(W / f"fuse{a.tag}.json", "w"), indent=1); print("object mesh verts", len(Vo), "size mm", info["mesh_size_mm"], "clusters", len(big))
def view(Vv, Cc, ax0, ax1, res=600):
    lo = Vv[:, [ax0, ax1]].min(0); sc = (res - 1) / max(np.ptp(Vv[:, [ax0, ax1]], axis=0).max(), 1e-6); im = np.full((res, res, 3), 255, np.uint8)
    o = np.argsort(Vv[:, 3 - ax0 - ax1]); uv = ((Vv[o][:, [ax0, ax1]] - lo) * sc).astype(int); im[res - 1 - uv[:, 1], uv[:, 0]] = (Cc[o][:, ::-1] * 255).astype(np.uint8); return im
if len(Vo):
    Co = np.asarray(ob.vertex_colors); ang = np.deg2rad(35); Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]); tilt = np.deg2rad(55)
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]); Vq = Vo @ Rz.T @ Rx.T
    cv2.imwrite(str(W / f"mesh_views{a.tag}.jpg"), np.hstack([view(Vo, Co, 0, 1), view(Vo, Co, 0, 2), view(Vo, Co, 1, 2), view(Vq, Co, 0, 2)]))

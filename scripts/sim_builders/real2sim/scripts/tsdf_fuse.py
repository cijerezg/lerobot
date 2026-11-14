"""Fuse VGGT depths into a TSDF mesh, cut out the object above the table around the sheet, save in metres.
Outputs in work_dir: mesh_scene.ply (table frame, metres), mesh_object.ply (object only, z=0 at table), mesh_views.jpg
usage: tsdf_fuse.py work_dir [--voxel_mm 2] [--conf_pct 30] [--pad_m 0.03]"""
import argparse, pathlib, numpy as np, cv2, open3d as o3d
p = argparse.ArgumentParser(); p.add_argument("work"); p.add_argument("--voxel_mm", type=float, default=2.0)
p.add_argument("--conf_pct", type=float, default=30); p.add_argument("--pad_m", type=float, default=0.03); a = p.parse_args()
W = pathlib.Path(a.work); d = np.load(W / "vggt.npz"); tf = np.load(W / "table_frame.npz"); sh = np.load(W / "sheet.npz")
scale = float(sh["scale"]); R, n, dd, ctr = tf["R"], tf["n"], float(tf["d"]), tf["ctr"]
# table frame: X = R @ p ; X_z := p.n + d ; X_xy -= ctr ; then metres
def to_table(p):
    X = p @ R.T; X[..., 2] = p @ n + dd; X[..., :2] -= ctr; return X * scale
vox = a.voxel_mm / 1000 / scale  # voxel in vggt units
# ScalableTSDFVolume returns nothing in open3d 0.20 here; use a uniform cube centred on the sheet (table-frame -> vggt world)
sc_t = np.r_[sh["centre"] + ctr, -dd]; sc_w = np.vstack([R[:2], n]).T @ sc_t
L = (np.ptp(sh["box"], axis=0).max() + 2 * a.pad_m / scale) * 1.15; res_v = int(np.ceil(L / vox)); res_v = min(res_v, 420); vox = L / res_v
print("uniform volume: side %.3f m, %d voxels per side, voxel %.2f mm" % (L * scale, res_v, vox * scale * 1000))
vol = o3d.pipelines.integration.UniformTSDFVolume(length=L, resolution=res_v, sdf_trunc=4 * vox, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, origin=sc_w - L / 2)
thr = np.percentile(d["depth_conf"], a.conf_pct); H, Wd = d["depth"].shape[1:3]
for i in range(len(d["frames"])):
    dep = d["depth"][i, ..., 0].astype(np.float32).copy(); dep[d["depth_conf"][i] < thr] = 0
    K = d["intrinsics"][i]; intr = o3d.camera.PinholeCameraIntrinsic(Wd, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    E = np.eye(4); E[:3] = d["extrinsics"][i]  # world->cam, vggt units
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(d["images"][i])), o3d.geometry.Image(dep), depth_scale=1.0, depth_trunc=3.0 / scale, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, E)
mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
Tm = np.eye(4); Tm[:3, :3] = np.diag([scale] * 3) @ np.vstack([R[:2], n]); Tm[:3, 3] = np.array([-ctr[0] * scale, -ctr[1] * scale, dd * scale])
mesh.transform(Tm); print("scene bbox m", np.asarray(mesh.vertices).min(0).round(3), np.asarray(mesh.vertices).max(0).round(3)); o3d.io.write_triangle_mesh(str(W / "mesh_scene.ply"), mesh)
V = np.asarray(mesh.vertices); box = sh["box"] * scale; c = box.mean(0); half = np.abs((box - c)).max(0) + a.pad_m
# note: sheet box is axis-aligned only approximately; crop by a padded axis-aligned bbox of the box corners
keep = (np.abs(V[:, 0] - c[0]) < half[0]) & (np.abs(V[:, 1] - c[1]) < half[1]) & (V[:, 2] > 0.004)
obj = mesh.select_by_index(np.nonzero(keep)[0]); tri_cl, cnt, _ = obj.cluster_connected_triangles()
tri_cl = np.asarray(tri_cl); cnt = np.asarray(cnt); big = np.nonzero(cnt > 0.02 * cnt.max())[0]
obj.remove_triangles_by_mask(~np.isin(tri_cl, big)); obj.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(str(W / "mesh_object.ply"), obj)
Vo = np.asarray(obj.vertices); print("scene verts", len(V), "object verts", len(Vo), "object bbox m", Vo.min(0).round(3), Vo.max(0).round(3), "clusters kept", len(big))
# QC renders: vertex splats in 4 views
def view(Vv, Cc, ax0, ax1, res=700):
    lo = Vv[:, [ax0, ax1]].min(0); sc = (res - 1) / np.ptp(Vv[:, [ax0, ax1]], axis=0).max(); im = np.full((res, res, 3), 255, np.uint8)
    o = np.argsort(Vv[:, 3 - ax0 - ax1]); uv = ((Vv[o][:, [ax0, ax1]] - lo) * sc).astype(int); im[res - 1 - uv[:, 1], uv[:, 0]] = (Cc[o][:, ::-1] * 255).astype(np.uint8)
    return cv2.dilate(im, np.ones((2, 2), np.uint8)) if False else cv2.erode(im, np.ones((2, 2), np.uint8))
Co = np.asarray(obj.vertex_colors); ang = np.deg2rad(35); Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
tilt = np.deg2rad(55); Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]); Vq = Vo @ Rz.T @ Rx.T
cv2.imwrite(str(W / "mesh_views.jpg"), np.hstack([view(Vo, Co, 0, 1), view(Vo, Co, 0, 2), view(Vo, Co, 1, 2), view(Vq, Co, 0, 2)]))

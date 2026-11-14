"""Recover the world->table transform of the kit fusion (same recipe as fuse_d405.py: RANSAC plane on the metric cloud, x = cross(y_cam, n),
origin = median of the inlier xy) and reproject candidate fold lines (aligned-frame x, from profile.json) into scan frames to check them
against the visible seams. Writes work/bits_kit_fold/fold_candidates.jpg and table_frame.json."""
import json, pathlib, numpy as np, cv2, open3d as o3d
D = pathlib.Path("assets/sim_sources/digital_twin"); S = D / "scans/obj_orbit"; W = D / "work/bits_kit"; O = D / "work/bits_kit_fold"
K = json.load(open(S / "intrinsic.json")); E = np.load(W / "poses.npz")["E"]; fr = json.load(open(W / "frames.json"))["frames"]; gm = cv2.imread(str(S.parent / "gripper_mask.png"), 0) > 0
intr = o3d.camera.PinholeCameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
def load(i):
    z = cv2.imread(str(S / "depth" / f"{fr[i]:06d}.png"), -1).astype(np.float32) * K["depth_scale"]; z[(z < 0.07) | (z > 0.45) | gm] = 0
    c = cv2.imread(str(S / "color" / f"{fr[i]:06d}.jpg"))[:, :, ::-1]
    return o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(c)), o3d.geometry.Image(z), depth_scale=1.0, depth_trunc=0.45, convert_rgb_to_intensity=False)
P = []
for i in range(len(fr)):
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(load(i), intr, E[i]).uniform_down_sample(16); P.append(np.asarray(pc.points))
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(P))).voxel_down_sample(0.002)
plane, inl = pc.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=3000); n = np.array(plane[:3]); n /= np.linalg.norm(n); dd = plane[3]
cams = np.array([np.linalg.inv(E[i])[:3, 3] for i in range(len(fr))])
if np.mean(cams @ n + dd) < 0: n, dd = -n, -dd
x = np.cross([0, 1, 0], n); x /= np.linalg.norm(x); y = np.cross(n, x); R = np.stack([x, y, n]); Pq = np.asarray(pc.points); Pt = Pq @ R.T; Pt[:, 2] = Pq @ n + dd; ctr = np.median(Pt[inl][:, :2], axis=0)
Tm = np.eye(4); Tm[:3, :3] = R; Tm[:3, 3] = np.r_[-ctr, dd]; Tinv = np.linalg.inv(Tm)
# sanity: the saved table cloud vs this frame (object cluster centre from fuse.json was (0.083, 0.002))
# sanity: ICP of this cloud onto the saved cloud_table.ply (same recipe); the residual transform is folded into Tm
saved = o3d.io.read_point_cloud(str(W / "cloud_table.ply")).voxel_down_sample(0.004); mine = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pt)).voxel_down_sample(0.004)
reg = o3d.pipelines.registration.registration_icp(mine, saved, 0.02, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())
print("ICP mine->saved: fitness %.3f rmse %.4f, translation %s mm, rot deg %.2f" % (reg.fitness, reg.inlier_rmse, (1e3 * reg.transformation[:3, 3]).round(1), np.degrees(np.arccos(np.clip((np.trace(reg.transformation[:3, :3]) - 1) / 2, -1, 1)))))
Tm = reg.transformation @ Tm; Tinv = np.linalg.inv(Tm)
json.dump(dict(T_world_to_table=Tm.tolist(), plane=list(map(float, plane)), icp_fitness=reg.fitness, icp_rmse=reg.inlier_rmse), open(O / "table_frame.json", "w"), indent=1)
pr = json.load(open(O / "profile.json")); c = np.array(pr["centre_table_xy"]); Ra = np.array(pr["R_rows"])
def table_pt(xa, ya, z): return np.r_[Ra.T @ np.array([xa, ya]) + c, z]
Kc = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
def proj(i, Pt_):
    Pw = (Tinv @ np.r_[Pt_, 1])[:3]; Pc = E[i, :3, :3] @ Pw + E[i, :3, 3]
    if Pc[2] <= 0.05: return None
    u = Kc @ (Pc / Pc[2]); return int(round(u[0])), int(round(u[1]))
cands = [(-0.097, (0, 0, 255), "-97"), (0.085, (0, 255, 0), "85"), (0.135, (255, 0, 255), "135")]
tiles = []
for want in (120, 160, 250, 400, 560, 660):
    i = int(np.argmin(np.abs(np.array(fr) - want))); im = cv2.imread(str(S / "color" / f"{fr[i]:06d}.jpg"))
    for xx, col, lab in cands + [(-0.255, (255, 255, 0), ""), (0.229, (255, 255, 0), "")]:
        pts = [proj(i, table_pt(xx, yy, 0.010)) for yy in np.linspace(-0.125, 0.125, 12)]; pts = [p for p in pts if p]
        for a, b in zip(pts[:-1], pts[1:]): cv2.line(im, a, b, col, 2)
        if pts and lab: cv2.putText(im, lab, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(im, str(fr[i]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2); tiles.append(im)
cv2.imwrite(str(O / "fold_candidates.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("wrote", O / "fold_candidates.jpg")

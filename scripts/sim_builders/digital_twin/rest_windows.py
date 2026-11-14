"""obj_<name> scan (static camera, objects moved by hand) -> resting windows + fused object clouds.

Per part (the camera moved once, at --split): table plane from the --empty frames; a frame is "still" when its depth differs
from the previous one by < 6 mm on > 90 % of valid pixels outside the gripper mount (fixed near pixels); runs of >= 10 still frames are
windows (a moving hand breaks stillness). Per window: median depth -> points above the plane by > 6 mm -> largest clusters -> windows/wNN.ply (table frame, m)
+ wNN.jpg thumb + windows.json.
usage: rest_windows.py scans/obj_all --empty 0 120 --split 2900 --empty2 3700 4100
"""
import argparse, glob, json, pathlib, cv2, numpy as np, open3d as o3d
ap = argparse.ArgumentParser(); ap.add_argument("scan"); ap.add_argument("--empty", nargs=2, type=int, required=True)
ap.add_argument("--split", type=int, default=None); ap.add_argument("--empty2", nargs=2, type=int, default=None); a = ap.parse_args()
S = pathlib.Path(a.scan); K = json.load(open(S / "intrinsic.json")); scale = K["depth_scale"]
files = sorted(glob.glob(str(S / "depth/*.png"))); N = len(files); (S / "windows").mkdir(exist_ok=True)
u, v = np.meshgrid(np.arange(K["width"]), np.arange(K["height"]))
def load(i): return cv2.imread(files[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * scale
def unproject(z): return np.stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z], -1)
def plane_from(lo, hi):
    z = np.median(np.stack([load(i) for i in range(lo, hi, 5)]), 0); P = unproject(z)[(z > 0.07) & (z < 1.0)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64))).voxel_down_sample(0.004)
    pl, inl = pc.segment_plane(0.003, 3, 1000); n = np.array(pl[:3]); n /= np.linalg.norm(n)
    if n[2] > 0: n, pl = -n, [-x for x in pl]   # normal towards the camera (camera looks down +z)
    x = np.cross([0, 1, 0], n); x /= np.linalg.norm(x); y = np.cross(n, x); R = np.stack([x, y, n])
    print("plane inliers %d/%d, camera height %.3f m, tilt %.1f deg" % (len(inl), len(pc.points), abs(pl[3]), np.degrees(np.arccos(-n[2]))))
    return np.array(pl), R
parts = [(0, a.split or N, a.empty)] + ([(a.split, N, a.empty2)] if a.split else [])
windows = []
for lo, hi, emp in parts:
    pl, R = plane_from(*emp); prev = load(lo); still = np.zeros(hi - lo, bool)
    mount = np.median(np.stack([load(i) for i in range(emp[0], emp[1], 5)]), 0); mount = (mount > 0.07) & (mount < 0.15)   # gripper in view
    for i in range(lo + 1, hi):
        z = load(i); ok = (z > 0.07) & (prev > 0.07) & (z < 1.0) & ~mount
        still[i - lo] = (np.abs(z - prev)[ok] < 0.006).mean() > 0.90; prev = z
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0])); runs = [(lo + s, lo + e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= 10]
    for s, e in runs:
        z = np.median(np.stack([load(i) for i in range(s, e, max(1, (e - s) // 20))]), 0); P = unproject(z); h = -(P @ pl[:3] + pl[3])
        m = (z > 0.07) & (z < 1.0) & (h > 0.006) & (h < 0.30) & ~mount
        if m.sum() < 500: continue
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(((P[m] @ R.T) - np.r_[0, 0, 0]).astype(np.float64)))
        pc.points = o3d.utility.Vector3dVector(np.asarray(pc.points) - np.r_[0, 0, np.asarray(pc.points)[:, 2].min() - h[m].min()])
        col = cv2.cvtColor(cv2.imread(files[(s + e) // 2].replace("depth", "color").replace(".png", ".jpg")), cv2.COLOR_BGR2RGB)
        pc.colors = o3d.utility.Vector3dVector(col[m] / 255.0); pc = pc.voxel_down_sample(0.0015)
        lab = np.asarray(pc.cluster_dbscan(0.012, 30)); keep = np.isin(lab, [k for k in np.unique(lab) if k >= 0 and (lab == k).sum() > 300])
        pc = pc.select_by_index(np.flatnonzero(keep)); Q = np.asarray(pc.points)
        if len(Q) < 300: continue
        k = len(windows); o3d.io.write_point_cloud(str(S / f"windows/w{k:02d}.ply"), pc)
        th = cv2.imread(files[(s + e) // 2].replace("depth", "color").replace(".png", ".jpg")); ov = th.copy(); ov[m] = (0.5 * ov[m] + [0, 0, 127]).astype(np.uint8)
        cv2.putText(ov, f"w{k:02d} {s}-{e} ({s/15:.0f}-{e/15:.0f}s) {len(Q)} pts {np.ptp(Q, 0)[0]*1000:.0f}x{np.ptp(Q, 0)[1]*1000:.0f}x{Q[:,2].max()*1000:.0f} mm", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(S / f"windows/w{k:02d}.jpg"), cv2.resize(ov, (636, 360)))
        windows.append({"id": k, "start": int(s), "end": int(e), "points": int(len(Q)), "size_mm": (np.ptp(Q, 0) * 1000).round(0).tolist(), "part": int(lo > 0)})
        print(windows[-1])
json.dump(windows, open(S / "windows/windows.json", "w"), indent=1); print(len(windows), "windows")

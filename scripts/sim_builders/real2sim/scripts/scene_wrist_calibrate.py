"""Solve the D405 wrist camera pose in the gripper_end body frame.
(a) plane method: table plane (normal, distance) from the real depth per frame vs the plane predicted from the sim
    gripper_end pose and a candidate camera pose; least squares over 6 pose parameters.
(b) finger method: IoU between the real dark-finger mask and the sim finger silhouette from the candidate camera.
(c) combined: plane cost + finger cost, Nelder-Mead from (a).
Camera frame convention inside this file: OpenCV (x right, y down, z forward). The MuJoCo quaternion is derived at the end.
usage: scene_wrist_calibrate.py ROOT EP WORK [--plane_frames ...] [--finger_frames ...]"""
import argparse, json, pathlib, sys, cv2, numpy as np, mujoco
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scene_sim as S, scene_wrist as Wr
from scene_frames import episode_states, decode_frames, WRIST

PLANE_FRAMES = [300, 360, 420, 480, 510, 600, 660, 1290, 1830, 1890, 1950, 2010, 2160, 2220, 2280]
FINGER_FRAMES = [0, 300, 510, 810, 1020, 1335, 1890, 2220]
SIG_N, SIG_D = 0.02, 0.003   # plane normal (unit vector components) and distance (m) noise scales


def finger_mask(bgr):
    """Dark finger silhouettes: low value pixels in the lower 55 % of the image whose component touches the bottom edge."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); dark = (hsv[..., 2] < 70).astype(np.uint8); dark[: int(0.45 * bgr.shape[0])] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros(n, bool)
    for j in range(1, n): keep[j] = stats[j, cv2.CC_STAT_AREA] > 800 and stats[j, cv2.CC_STAT_TOP] + stats[j, cv2.CC_STAT_HEIGHT] >= bgr.shape[0] - 2
    return keep[lab]


def pose_from_params(p):
    return p[:3], Rot.from_rotvec(p[3:6]).as_matrix()


class Fit:
    def __init__(self, root, ep, work):
        self.root, self.ep, self.work = root, ep, pathlib.Path(work); self.states, _ = episode_states(root, ep)
        self.model, _ = S.build_model(wrist_cam=True); self.data = mujoco.MjData(self.model); self.V = S.Views(self.model); self.gid = self.model.body("gripper_end").id

    def gripper_pose(self, f):
        S.set_state(self.model, self.data, self.states[f]); return self.data.xmat[self.gid].reshape(3, 3).copy(), self.data.xpos[self.gid].copy()

    # ---- plane method
    def measure_planes(self, frames, qc=()):
        out = []
        for f in frames:
            d, fa = Wr.load_depth(self.root, self.ep, f); dm = d * Wr.UNITS_M; valid = (d > 0) & (dm < 1.0); P = Wr.backproject(np.where(valid, dm, 0)); pts = P[valid]
            sub = pts[np.random.default_rng(0).choice(len(pts), min(6000, len(pts)), replace=False)]; n, dd, inl = Wr.ransac_plane(sub)
            res = np.abs(pts @ n + dd); inl_full = res < 0.004; out.append(dict(frame=f, depth_frame=fa, n=n, d=dd, inlier_frac=float(inl_full.mean()), rms_mm=float(1e3 * np.sqrt((res[inl_full] ** 2).mean()))))
            if f in qc:
                im = Wr.colourmap(dm); m = np.zeros(valid.shape, bool); m[valid] = inl_full; im[~m & valid] = (0.3 * im[~m & valid]).astype(np.uint8); cv2.putText(im, f"f{fa} plane inliers {inl_full.mean():.2f} d {dd:.3f} m n {np.round(n, 2)}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imwrite(str(self.work / "wrist" / f"plane_qc_{fa:05d}.jpg"), im)
        return out

    def plane_residuals(self, p, planes, poses):
        """p = (pos 3, rotvec 3[, distance bias]); the optional bias absorbs a constant sim-vs-real height offset."""
        t_gc, R_gc = pose_from_params(p); b = p[6] if len(p) > 6 else 0.0; r = []
        for pl, (R_wg, t_wg) in zip(planes, poses):
            R_wc = R_wg @ R_gc; t_wc = t_wg + R_wg @ t_gc; n_pred = R_wc.T @ np.array([0, 0, 1.0]); d_pred = t_wc[2]
            r.append(np.concatenate([(n_pred - pl["n"]) / SIG_N, [(d_pred - pl["d"] - b) / SIG_D]]))
        return np.concatenate(r)

    def plane_table(self, p, planes, poses):
        t_gc, R_gc = pose_from_params(p); b = p[6] if len(p) > 6 else 0.0; rows = []
        for pl, (R_wg, t_wg) in zip(planes, poses):
            R_wc = R_wg @ R_gc; t_wc = t_wg + R_wg @ t_gc; n_pred = R_wc.T @ np.array([0, 0, 1.0]); d_pred = t_wc[2] - b
            rows.append((pl["frame"], float(np.degrees(np.arccos(np.clip(n_pred @ pl["n"], -1, 1)))), float(1e3 * (d_pred - pl["d"])), pl["d"], pl["inlier_frac"], pl["rms_mm"]))
        return rows

    # ---- finger method
    def finger_sim_masks(self, p, frames):
        t_gc, R_gc = pose_from_params(p); S.set_body_camera(self.model, "wrist", t_gc, R_gc); out = []
        for f in frames:
            S.set_state(self.model, self.data, self.states[f]); out.append(self.V.body_mask(self.data, ["gripper_left", "gripper_right"], cam="wrist"))
        return out

    def finger_ious(self, p, frames, real, band=False):
        """IoU of the finger silhouettes; band=True scores only the rows the real fingers occupy (from 10 px above the real
        tip row down), so sim fingers that protrude higher (longer meshes than the real pads) are not penalised."""
        out = []
        for a, b in zip(self.finger_sim_masks(p, frames), real):
            if band:
                rows = np.nonzero(b.any(1))[0]; top = max(int(rows.min()) - 10, 0) if len(rows) else 0; a = a.copy(); a[:top] = False
            out.append((a & b).sum() / max((a | b).sum(), 1))
        return np.array(out)

    def tip_rows(self, p, frames, real):
        """Topmost image row of the sim finger silhouette vs the real one (negative = sim fingers protrude higher)."""
        out = []
        for a, b in zip(self.finger_sim_masks(p, frames), real):
            ra = np.nonzero(a.any(1))[0]; rb = np.nonzero(b.any(1))[0]; out.append((int(ra.min()) if len(ra) else -1, int(rb.min()) if len(rb) else -1))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("ep", type=int); ap.add_argument("work"); ap.add_argument("--plane_frames", type=int, nargs="*", default=PLANE_FRAMES); ap.add_argument("--finger_frames", type=int, nargs="*", default=FINGER_FRAMES); a = ap.parse_args()
    work = pathlib.Path(a.work); (work / "wrist").mkdir(parents=True, exist_ok=True); F = Fit(a.root, a.ep, a.work)
    planes = F.measure_planes(a.plane_frames, qc=a.plane_frames[:3]); poses = [F.gripper_pose(f) for f in a.plane_frames]
    print("plane measurements (frame, inlier frac, rms mm, d m, n):"); [print(f"  f{pl['frame']:5d} {pl['inlier_frac']:.2f} {pl['rms_mm']:.1f} {pl['d']:.3f} {np.round(pl['n'], 3)}") for pl in planes]
    # two starts: camera looks along gripper +x (toward the fingertips/table), image down = -z_g or +z_g
    starts = []
    for sgn in (+1, -1):
        R0 = np.column_stack([-sgn * np.array([0, 1.0, 0]), -sgn * np.array([0, 0, 1.0]), np.array([1.0, 0, 0])]); starts.append(np.concatenate([[-0.09, 0.0, 0.05 * sgn], Rot.from_matrix(R0).as_rotvec()]))
    best = None
    for k, p0 in enumerate(starts):
        r = least_squares(F.plane_residuals, p0, args=(planes, poses)); cost = float(np.mean(r.fun ** 2)); print(f"plane LSQ start {k}: mean sq residual {cost:.3f} pos {r.x[:3].round(4)} rotvec {r.x[3:6].round(4)}")
        if best is None or cost < best[0]: best = (cost, r.x)
    p_plane = best[1]; rows = F.plane_table(p_plane, planes, poses)
    print("plane fit residuals: frame | normal angle deg | distance error mm | measured d | inliers | plane rms mm"); [print("  f%5d %6.2f %7.1f %.3f %.2f %.1f" % row) for row in rows]
    # finger masks
    rgb = decode_frames(a.root, a.ep, WRIST, a.finger_frames); real_f = [finger_mask(rgb[f]) for f in a.finger_frames]
    for f in a.finger_frames[:3]:
        ov = rgb[f].copy(); m = finger_mask(rgb[f]); ov[m] = (0.4 * ov[m] + 0.6 * np.array([0, 255, 0])).astype(np.uint8); cv2.putText(ov, f"finger mask f{f} gripper {F.states[f][6]:.0f} deg", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2); cv2.imwrite(str(work / "wrist" / f"finger_mask_qc_{f:05d}.jpg"), ov)
    iou_plane = F.finger_ious(p_plane, a.finger_frames, real_f); print("finger IoU at the plane solution", iou_plane.round(3), "mean %.3f" % iou_plane.mean())
    plane_cost = lambda p: float(np.mean(F.plane_residuals(p, planes, poses) ** 2))
    # the plane method leaves the translation across the view direction free: coarse grid over the translation with the
    # plane rotation fixed, seeded by the best finger IoU (IoU = 0 gives Nelder-Mead nothing to follow)
    grid_frames = a.finger_frames[::2]; grid_real = real_f[::2]; best_g = (-1, None)
    for x in np.arange(-0.14, -0.03, 0.01):
        for y in np.arange(-0.04, 0.041, 0.01):
            for z in np.arange(0.01, 0.121, 0.01):
                pg = np.concatenate([[x, y, z], p_plane[3:6]]); v = F.finger_ious(pg, grid_frames, grid_real).mean()
                if v > best_g[0]: best_g = (v, pg)
    print(f"translation grid: best finger IoU {best_g[0]:.3f} at pos {best_g[1][:3].round(3)} (plane cost there {plane_cost(best_g[1]):.2f})"); p_seed = best_g[1]
    f_only = minimize(lambda p: 1 - F.finger_ious(p, a.finger_frames, real_f).mean(), p_seed, method="Nelder-Mead", options=dict(maxiter=500, maxfev=500, xatol=1e-4, fatol=1e-5, initial_simplex=np.vstack([p_seed] + [p_seed + np.eye(6)[i] * [0.01, 0.01, 0.01, 0.03, 0.03, 0.03][i] for i in range(6)])))
    iou_f = F.finger_ious(f_only.x, a.finger_frames, real_f); print(f"finger-only: mean IoU {iou_f.mean():.3f} pos {f_only.x[:3].round(4)} rotvec {f_only.x[3:6].round(4)}; plane cost there {plane_cost(f_only.x):.2f}")
    # H1: 6 pose parameters + a constant plane-distance bias (would mean the sim arm sits 'b' higher over the table than the real one)
    p7 = np.concatenate([f_only.x, [0.0]]); p7[6] = float(np.mean([r[2] for r in F.plane_table(f_only.x, planes, poses)]) / 1e3)
    comb_b = minimize(lambda p: plane_cost(p) + 20 * (1 - F.finger_ious(p, a.finger_frames, real_f).mean()), p7, method="Nelder-Mead", options=dict(maxiter=1200, maxfev=1200, xatol=1e-4, fatol=1e-5, initial_simplex=np.vstack([p7] + [p7 + np.eye(7)[i] * [0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.01][i] for i in range(7)])))
    p_b = comb_b.x; iou_b = F.finger_ious(p_b, a.finger_frames, real_f)
    print(f"H1 combined with bias: b = {1e3 * p_b[6]:.1f} mm, plane cost {plane_cost(p_b):.2f}, finger IoU {iou_b.round(3)} mean {iou_b.mean():.3f}; pos {p_b[:3].round(4)} rotvec {p_b[3:6].round(4)}")
    # H2 (primary): no bias; fingers scored on the band they occupy (lateral alignment), plane normals + distances taken as is
    cost_band = lambda p: plane_cost(p) + 20 * (1 - F.finger_ious(p, a.finger_frames, real_f, band=True).mean())
    best = None
    for p0 in (f_only.x, np.concatenate([[p_plane[0], p_seed[1], p_seed[2]], p_plane[3:6]])):
        r = minimize(cost_band, p0, method="Nelder-Mead", options=dict(maxiter=1200, maxfev=1200, xatol=1e-4, fatol=1e-5, initial_simplex=np.vstack([p0] + [p0 + np.eye(6)[i] * [0.01, 0.01, 0.01, 0.03, 0.03, 0.03][i] for i in range(6)])))
        print(f"  H2 start pos {p0[:3].round(3)}: cost {r.fun:.3f}")
        if best is None or r.fun < best.fun: best = r
    p_c = best.x; iou_c = F.finger_ious(p_c, a.finger_frames, real_f); iou_cb = F.finger_ious(p_c, a.finger_frames, real_f, band=True); rows_c = F.plane_table(p_c, planes, poses); tips = F.tip_rows(p_c, a.finger_frames, real_f)
    print(f"H2 combined (no bias): plane cost {plane_cost(p_c):.2f} (plane-only {plane_cost(p_plane):.2f}), finger IoU full {iou_c.round(3)} mean {iou_c.mean():.3f}, band {iou_cb.round(3)} mean {iou_cb.mean():.3f}; pos {p_c[:3].round(4)} rotvec {p_c[3:6].round(4)}")
    print("  finger tip rows sim/real:", tips)
    print("H2 plane residuals: frame | normal deg | distance mm"); [print("  f%5d %6.2f %7.1f" % row[:3]) for row in rows_c]
    def pack(p, name):
        t, R = pose_from_params(p); q = S.cv_rot_to_mj_quat(R); return dict(name=name, pos=t.tolist(), rotvec_cv=p[3:6].tolist(), R_cv_cols_in_gripper=R.T.tolist(), quat_mj_wxyz=q.tolist(), fovy=S.WRIST_FOVY)
    sims = F.finger_sim_masks(p_c, a.finger_frames)
    for f, rm, sm, v in zip(a.finger_frames, real_f, sims, iou_c):
        ov = rgb[f].copy(); ov[rm] = (0.6 * ov[rm] + 0.4 * np.array([255, 200, 0])).astype(np.uint8); cs, _ = cv2.findContours(sm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cs, -1, (0, 255, 0), 2)
        cv2.putText(ov, f"f{f} finger IoU {v:.2f} (band {iou_cb[a.finger_frames.index(f)]:.2f})", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2); cv2.imwrite(str(work / "wrist" / f"finger_overlay_{f:05d}.jpg"), ov)
    out = dict(intrinsics=dict(fx=Wr.FX, fy=Wr.FY, cx=Wr.CX, cy=Wr.CY, source="config_rl.yaml rebot_depth_intrinsics (factory D405 depth intrinsics, 640x480)"), mount_body="gripper_end",
               plane=dict(**pack(p_plane, "plane"), frames=a.plane_frames, residuals=rows, cost=plane_cost(p_plane), finger_iou=iou_plane.tolist()),
               finger=dict(**pack(f_only.x, "finger"), frames=a.finger_frames, iou=iou_f.tolist(), plane_cost=plane_cost(f_only.x)),
               combined_bias_H1=dict(**pack(p_b, "combined_bias"), distance_bias_m=float(p_b[6]), plane_cost=plane_cost(p_b), finger_iou=iou_b.tolist()),
               combined=dict(**pack(p_c, "combined"), plane_residuals=rows_c, plane_cost=plane_cost(p_c), finger_iou=iou_c.tolist(), finger_iou_band=iou_cb.tolist(), tip_rows_sim_real=tips))
    json.dump(out, open(work / "wrist_camera.json", "w"), indent=1); print("wrote", work / "wrist_camera.json")

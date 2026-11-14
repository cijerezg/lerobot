"""Calibrate the external camera (pos 3, rotvec 3, fovy 1) from the robot: maximise the mean IoU between the
rendered robot silhouette (MuJoCo segmentation) and the colour-threshold robot mask over N frames of one episode.
usage: scene_calibrate.py WORK [--n_frames 14] [--maxiter 600] [--starts K]
reads WORK/frames_every.npz + WORK/static_mask.png (from scene_masks.py); writes WORK/camera.json, WORK/calib_frames.npz,
WORK/qc/calib_overlay_*.jpg (real frame, rendered silhouette edge: red = initial guess, green = optimised)."""
import argparse, json, pathlib, sys, time, cv2, numpy as np, mujoco
from scipy.optimize import minimize
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scene_sim as S, scene_masks as M


def farthest_points(X, n, first=0):
    chosen = [first]; d = np.linalg.norm(X - X[first], axis=1)
    for _ in range(n - 1):
        j = int(np.argmax(d)); chosen.append(j); d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return chosen


class Objective:
    def __init__(self, model, states, real_masks, valid):
        self.model = model; self.data = mujoco.MjData(model); self.V = S.Views(model); self.states = states; self.real = real_masks; self.valid = valid; self.n_eval = 0

    def sim_masks(self, p):
        S.set_camera(self.model, p[:3], p[3:6], p[6]); out = []
        for st in self.states:
            S.set_state(self.model, self.data, st); out.append(self.V.robot_mask(self.data) & self.valid)
        return out

    def ious(self, p):
        return np.array([(a & b).sum() / max((a | b).sum(), 1) for a, b in zip(self.sim_masks(p), self.real)])

    def __call__(self, p):
        self.n_eval += 1; return 1.0 - self.ious(p).mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("work"); ap.add_argument("--n_frames", type=int, default=14); ap.add_argument("--maxiter", type=int, default=600); ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--init", default=None, help="camera.json of a previous pass: start there (plus --starts-1 perturbed starts)")
    ap.add_argument("--gate", type=int, default=0, help="keep only real-mask components within this many px of the --init silhouette (removes the clipper case and shadow blobs)")
    ap.add_argument("--static_frac", type=float, default=M.STATIC_FRAC); ap.add_argument("--tag", default=""); a = ap.parse_args()
    work = pathlib.Path(a.work); d = np.load(work / "frames_every.npz"); frames, idx, states = d["frames"], d["indices"], d["states"]
    static = cv2.imread(str(work / "static_mask.png"), 0) > 0
    if a.static_frac != M.STATIC_FRAC:
        nw = np.stack([M.non_wood(f) for f in frames]); static = cv2.dilate((nw.mean(0) >= a.static_frac).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool); print("static fraction %.3f at frac %.2f" % (static.mean(), a.static_frac))
    valid = ~static & M.roi_mask()
    sel = farthest_points(states[:, :5], a.n_frames, first=0); sel = sorted(sel)
    cal_frames, cal_idx, cal_states = frames[sel], idx[sel], states[sel]
    real = [M.robot_mask(f, static) for f in cal_frames]
    print("calibration frames", cal_idx.tolist()); print("real mask px", [int(m.sum()) for m in real])
    model, _ = S.build_model(); obj = Objective(model, cal_states, real, valid)
    # hand guess: camera above the table on the +x side of the robot looking down, image up = -x (robot side)
    base = np.array([0.35, 0.0, 1.0, 0.0, 0.0, np.pi / 2, 45.0])
    if a.init:
        c = json.load(open(a.init)); base = np.array(c["pos"] + c["rotvec"] + [c["fovy"]])
    if a.gate:
        k = np.ones((2 * a.gate + 1, 2 * a.gate + 1), np.uint8); gated = []
        for rm, sm in zip(real, obj.sim_masks(base)):
            near = cv2.dilate(sm.astype(np.uint8), k).astype(bool); n, lab = cv2.connectedComponents(rm.astype(np.uint8), connectivity=8)
            keep = np.zeros(n, bool)
            for j in range(1, n): keep[j] = (near & (lab == j)).any()
            gated.append(keep[lab])
        print("gated real mask px", [int(m.sum()) for m in gated]); real = gated; obj.real = real
    t0 = time.time(); print("guess ious", obj.ious(base).round(3), "mean %.3f" % obj.ious(base).mean(), "eval %.2fs" % ((time.time() - t0) / 2))
    starts = [base.copy()]
    rng = np.random.default_rng(0)
    for k in range(a.starts - 1):
        s = base.copy(); s[0] += rng.uniform(-0.15, 0.15); s[1] += rng.uniform(-0.1, 0.1); s[2] = rng.uniform(0.75, 1.3); s[6] = rng.uniform(38, 62)
        tilt = S.Rot.from_rotvec(rng.uniform(-0.2, 0.2, 3)) * S.Rot.from_rotvec(base[3:6]); s[3:6] = tilt.as_rotvec(); starts.append(s)
    best = None
    for k, s in enumerate(starts):
        t0 = time.time(); r = minimize(obj, s, method="Nelder-Mead", options=dict(maxiter=a.maxiter, maxfev=a.maxiter, xatol=1e-4, fatol=1e-5, initial_simplex=np.vstack([s] + [s + np.eye(7)[i] * [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 4.0][i] for i in range(7)])))
        print(f"start {k}: mean IoU {1 - r.fun:.4f} after {r.nfev} evals ({time.time() - t0:.0f}s) pos {r.x[:3].round(3)} rotvec {r.x[3:6].round(3)} fovy {r.x[6]:.2f}")
        if best is None or r.fun < best.fun: best = r
    # polish the best
    r = minimize(obj, best.x, method="Nelder-Mead", options=dict(maxiter=a.maxiter, maxfev=a.maxiter, xatol=1e-5, fatol=1e-6)); best = r if r.fun < best.fun else best
    p = best.x; iou0 = obj.ious(base); iou1 = obj.ious(p)
    print("final mean IoU %.4f" % iou1.mean()); print("per-frame", list(zip(cal_idx.tolist(), iou0.round(3).tolist(), iou1.round(3).tolist())))
    R = S.Rot.from_rotvec(p[3:6]); lookat = S.backproject_z0(np.array([[320.0, 240.0]]), p[:3], p[3:6], p[6])[0]
    cam = dict(pos=p[:3].tolist(), rotvec=p[3:6].tolist(), quat_wxyz=S.quat_wxyz(p[3:6]).tolist(), fovy=float(p[6]), lookat_z0=lookat.tolist(), image=[640, 480],
               f_px=float(240 / np.tan(np.deg2rad(p[6]) / 2)), mean_iou=float(iou1.mean()), guess_mean_iou=float(iou0.mean()), frames=cal_idx.tolist(), iou_guess=iou0.tolist(), iou_final=iou1.tolist(), guess=base.tolist())
    (work / f"camera{a.tag}.json").write_text(json.dumps(cam, indent=1)); np.savez(work / "calib_frames.npz", frames=cal_frames, indices=cal_idx, states=cal_states, real_masks=np.stack(real))
    sim0 = obj.sim_masks(base); sim1 = obj.sim_masks(p)
    for f, i, rm, m0, m1 in zip(cal_frames, cal_idx, real, sim0, sim1):
        ov = f.copy(); ov[rm] = (0.6 * ov[rm] + 0.4 * np.array([255, 200, 0])).astype(np.uint8)
        for m, col in ((m0, (0, 0, 255)), (m1, (0, 255, 0))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cs, -1, col, 1)
        cv2.putText(ov, f"f{i} IoU {iou0[list(cal_idx).index(i)]:.2f} -> {iou1[list(cal_idx).index(i)]:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(work / "qc" / f"calib_overlay{a.tag}_{i:05d}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", work / f"camera{a.tag}.json")

"""Check a calibrated external camera on another episode: robot-silhouette IoU on N farthest-pose frames, overlays,
and optionally a local refit to see how far the camera would move.
usage: scene_check_episode.py ROOT EPISODE CAMERA_JSON OUT_DIR [--n 8] [--gate 45] [--static_frac 0.6] [--refit]"""
import argparse, json, pathlib, sys, cv2, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scene_sim as S, scene_masks as M
from scene_calibrate import farthest_points, Objective

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("episode", type=int); ap.add_argument("camera"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--gate", type=int, default=45); ap.add_argument("--static_frac", type=float, default=0.6); ap.add_argument("--refit", action="store_true"); a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); tag = pathlib.Path(a.root).name + f"_ep{a.episode}"
frames, idx = M.decode_every(a.root, a.episode, 10); states, _ = M.episode_states(a.root, a.episode); states = states[idx]
nw = np.stack([M.non_wood(f) for f in frames]); static = cv2.dilate((nw.mean(0) >= a.static_frac).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool); valid = ~static & M.roi_mask()
sel = sorted(farthest_points(states[:, :5], a.n, first=0)); cf, ci, cs = frames[sel], idx[sel], states[sel]
real = [M.robot_mask(f, static) for f in cf]
cam = json.load(open(a.camera)); p0 = np.array(cam["pos"] + cam["rotvec"] + [cam["fovy"]])
model, _ = S.build_model(); obj = Objective(model, cs, real, valid)
k = np.ones((2 * a.gate + 1, 2 * a.gate + 1), np.uint8); gated = []
for rm, sm in zip(real, obj.sim_masks(p0)):
    near = cv2.dilate(sm.astype(np.uint8), k).astype(bool); n, lab = cv2.connectedComponents(rm.astype(np.uint8), connectivity=8); keep = np.zeros(n, bool)
    for j in range(1, n): keep[j] = (near & (lab == j)).any()
    gated.append(keep[lab])
obj.real = gated; iou = obj.ious(p0); print(tag, "frames", ci.tolist()); print("IoU with the given camera", iou.round(3).tolist(), "mean %.3f" % iou.mean())
res = dict(root=a.root, episode=a.episode, frames=ci.tolist(), iou=iou.tolist(), mean_iou=float(iou.mean()))
if a.refit:
    r = minimize(obj, p0, method="Nelder-Mead", options=dict(maxiter=600, maxfev=600, xatol=1e-4, fatol=1e-5)); iou1 = obj.ious(r.x)
    print("refit: mean IoU %.3f -> %.3f; pos moved %.3f m, rot %.2f deg, fovy %.1f -> %.1f" % (iou.mean(), iou1.mean(), np.linalg.norm(r.x[:3] - p0[:3]), np.rad2deg((S.Rot.from_rotvec(r.x[3:6]) * S.Rot.from_rotvec(p0[3:6]).inv()).magnitude()), p0[6], r.x[6]))
    res.update(refit=dict(pos=r.x[:3].tolist(), rotvec=r.x[3:6].tolist(), fovy=float(r.x[6]), mean_iou=float(iou1.mean()), iou=iou1.tolist()))
sims = obj.sim_masks(p0)
for f, i, rm, sm, v in zip(cf, ci, gated, sims, iou):
    ov = f.copy(); ov[rm] = (0.6 * ov[rm] + 0.4 * np.array([255, 200, 0])).astype(np.uint8); cs_, _ = cv2.findContours(sm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cs_, -1, (0, 255, 0), 1)
    cv2.putText(ov, f"{tag} f{i} IoU {v:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); cv2.imwrite(str(out / f"check_{tag}_{i:05d}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
json.dump(res, open(out / f"check_{tag}.json", "w"), indent=1)

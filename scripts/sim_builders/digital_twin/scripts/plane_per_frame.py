"""Per-depth-frame table plane in the wrist camera, carried into the sim world frame by the recorded state.

For every depth frame: RANSAC plane on the in-range points (camera frame). Then, for a state time shift and a wrist camera
pose offset, the plane goes to the world frame through the sim forward kinematics; tilt against world z and the signed
height of the measured plane under the camera footprint (table = z 0) are the per-frame errors. Reports the time shift
that minimises the spread, refines a 6-dof camera offset (Nelder-Mead on the median errors), writes planes.npz,
plane_per_frame.json and plane_per_frame.png.

usage: plane_per_frame.py ROOT EP OUT [--stride 4]
"""
import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dataset_depth_lib as L  # noqa: E402


def measure_planes(root, ep, frames, stride):
    out = []
    rng = np.random.default_rng(0)
    for k, f in enumerate(frames):
        pts, _ = L.unproject(L.load_depth(root, ep, f), stride=stride)
        if len(pts) < 500:
            out.append((f, np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, len(pts)))
            continue
        sub = pts[rng.choice(len(pts), min(6000, len(pts)), replace=False)]
        n, d, inl = L.ransac_plane(sub, rng=rng)
        res = np.abs(pts @ n + d)
        inl_all = res < 0.004
        out.append((f, *n, d, float(inl_all.mean()), float(1e3 * np.sqrt((res[inl_all] ** 2).mean())), len(pts)))
        if k % 200 == 0:
            print(f"  plane {k}/{len(frames)} f{f} inliers {inl_all.mean():.2f} rms {1e3 * np.sqrt((res[inl_all] ** 2).mean()):.1f} mm d {d:.3f}", flush=True)
    a = np.array(out, float)
    return dict(frame=a[:, 0].astype(int), n=a[:, 1:4], d=a[:, 4], inlier_frac=a[:, 5], rms_mm=a[:, 6], npts=a[:, 7])


def world_errors(sim, states, planes, shift=0, dpos=(0, 0, 0), drot=(0, 0, 0)):
    """tilt (deg) and signed height (m) of each measured plane in world; positive height = measured table above z 0."""
    sim.set_camera_offset(dpos, drot)
    tilt, height = np.full(len(planes["frame"]), np.nan), np.full(len(planes["frame"]), np.nan)
    for k, f in enumerate(planes["frame"]):
        if not np.isfinite(planes["d"][k]):
            continue
        s = int(np.clip(f + shift, 0, len(states) - 1))
        t, R = sim.camera_pose(states[s])
        n_w = R @ planes["n"][k]
        d_w = planes["d"][k] - n_w @ t
        if n_w[2] < 0:
            n_w, d_w = -n_w, -d_w
        tilt[k] = np.degrees(np.arccos(np.clip(n_w[2], -1, 1)))
        # plane z under the camera footprint (x_c, y_c): n.x x + n.y y + n.z z + d = 0
        height[k] = -(n_w[0] * t[0] + n_w[1] * t[1] + d_w) / n_w[2]
    return tilt, height


def summary(tilt, height, good):
    h = 1e3 * height[good]
    t = tilt[good]
    return dict(n=int(good.sum()), height_median_mm=float(np.median(h)), height_mad_mm=float(1.4826 * np.median(np.abs(h - np.median(h)))),
                height_p10_mm=float(np.percentile(h, 10)), height_p90_mm=float(np.percentile(h, 90)), tilt_median_deg=float(np.median(t)), tilt_p90_deg=float(np.percentile(t, 90)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("ep", type=int)
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    states, _ = L.episode_states(a.root, a.ep)
    frames = L.depth_frames(a.root, a.ep)
    cache = out / "planes.npz"
    if cache.exists():
        planes = dict(np.load(cache))
    else:
        planes = measure_planes(a.root, a.ep, frames, a.stride)
        np.savez(cache, **planes)
    sim = L.Sim()
    # frames whose dominant plane is the table: enough in-range points, a well supported plane
    good0 = (planes["npts"] > 3000) & (planes["inlier_frac"] > 0.5)
    print(f"{len(frames)} depth frames, {good0.sum()} with a dominant plane (>3000 in-range pts, inlier frac > 0.5)")
    report = dict(frames=len(frames), dominant_plane_frames=int(good0.sum()))
    # time shift of the state against the depth frame
    shifts = {}
    for sh in range(-4, 5):
        tilt, height = world_errors(sim, states, planes, shift=sh)
        good = good0 & np.isfinite(height) & (np.abs(height - np.nanmedian(height[good0])) < 0.03)
        shifts[sh] = summary(tilt, height, good)
        print(f"shift {sh:+d}: height med {shifts[sh]['height_median_mm']:6.1f} mad {shifts[sh]['height_mad_mm']:5.1f} p10/p90 {shifts[sh]['height_p10_mm']:6.1f}/{shifts[sh]['height_p90_mm']:6.1f} mm, tilt med {shifts[sh]['tilt_median_deg']:.2f} p90 {shifts[sh]['tilt_p90_deg']:.2f} deg (n {shifts[sh]['n']})")
    # a shift counts only if it clearly tightens the spread (>= 10 % lower MAD than no shift)
    best_shift = min(shifts, key=lambda s: shifts[s]["height_mad_mm"])
    if shifts[best_shift]["height_mad_mm"] > 0.9 * shifts[0]["height_mad_mm"]:
        best_shift = 0
    report["time_shift"] = {str(k): v for k, v in shifts.items()}
    report["best_shift"] = best_shift
    tilt0, height0 = world_errors(sim, states, planes, shift=best_shift)
    good = good0 & np.isfinite(height0) & (np.abs(height0 - np.nanmedian(height0[good0])) < 0.03)
    report["baseline"] = summary(tilt0, height0, good)
    # camera offset refinement: median |height| + tilt spread (tilt median is a rotation error, height median a translation error)
    def cost(p):
        tilt, height = world_errors(sim, states, planes, shift=best_shift, dpos=p[:3], drot=p[3:])
        h = height[good]
        t = tilt[good]
        return float(np.median(np.abs(h)) * 1e3 + 1.4826 * np.median(np.abs(h - np.median(h))) * 1e3 + 5 * np.median(t))

    p0 = np.zeros(6)
    r = minimize(cost, p0, method="Nelder-Mead", options=dict(maxiter=600, maxfev=600, xatol=1e-4, fatol=1e-3,
                 initial_simplex=np.vstack([p0] + [p0 + np.eye(6)[i] * [0.01, 0.01, 0.01, 0.03, 0.03, 0.03][i] for i in range(6)])))
    tilt1, height1 = world_errors(sim, states, planes, shift=best_shift, dpos=r.x[:3], drot=r.x[3:])
    report["refined"] = dict(dpos_mm=(1e3 * r.x[:3]).round(2).tolist(), drot_deg=np.degrees(r.x[3:]).round(3).tolist(), cost0=cost(p0), cost=float(r.fun), **summary(tilt1, height1, good))
    print("baseline ", report["baseline"])
    print("refined  ", report["refined"])
    # height error vs joints: does the residual follow the arm pose (kinematics) or stay flat (hand-eye)?
    names = ["pan", "lift", "elbow", "wrist_flex", "wrist_yaw", "wrist_roll"]
    corr = {}
    S = states[np.clip(planes["frame"] + best_shift, 0, len(states) - 1)]
    for j, nm in enumerate(names):
        corr[nm] = float(np.corrcoef(S[good, j], 1e3 * height1[good])[0, 1])
    report["height_vs_joint_corr_refined"] = corr
    print("corr(height err, joint):", {k: round(v, 2) for k, v in corr.items()})
    sim.set_camera_offset()
    json.dump(report, open(out / "plane_per_frame.json", "w"), indent=1)
    np.savez(out / "plane_errors.npz", frame=planes["frame"], good=good, tilt0=tilt0, height0=height0, tilt1=tilt1, height1=height1)
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(planes["frame"][good], 1e3 * height0[good], ".", ms=2, label="calibrated camera")
    ax[0].plot(planes["frame"][good], 1e3 * height1[good], ".", ms=2, label="refined offset")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel("table height under camera (mm)")
    ax[0].legend()
    ax[1].plot(planes["frame"][good], tilt0[good], ".", ms=2)
    ax[1].plot(planes["frame"][good], tilt1[good], ".", ms=2)
    ax[1].set_ylabel("plane tilt (deg)")
    ax[2].plot(planes["frame"], planes["inlier_frac"], ".", ms=2)
    ax[2].set_ylabel("plane inlier frac")
    ax[2].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out / "plane_per_frame.png", dpi=110)
    print("wrote", out / "plane_per_frame.json")

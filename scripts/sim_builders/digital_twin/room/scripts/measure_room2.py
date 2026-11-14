import sys, pathlib, cv2, numpy as np, mujoco, pandas as pd
scene, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, "lerobot/src"); from lerobot.probes.action_trace_sim import to_qpos
FX, FY, CX, CY = 394.9832, 394.9832, 322.5604, 238.6966
m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m); cam = m.camera("wrist").id
df = pd.read_parquet(f"{root}/data/chunk-000/file-000.parquet", columns=["observation.state"]); states = np.stack(df["observation.state"].to_numpy())
def world(fr):
    d.qpos[:8] = to_qpos(states[fr]); mujoco.mj_forward(m, d); c, R = d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy()
    dep = cv2.imread(f"{root}/depth/wrist.depth/episode-000000/frame-{fr:06d}.png", cv2.IMREAD_UNCHANGED); z = dep.astype(np.float64) * 1e-4; z[dep == 0] = np.nan
    v, u = np.mgrid[0:480, 0:640]; P = np.stack([(u - CX) / FX * z, -(v - CY) / FY * z, -z], -1); W = P @ R.T + c; ok = np.isfinite(z); return W[ok], z[ok]
# desk far edge from frame 0: table-plane points histogram in x
W, z = world(0); tab = (np.abs(W[:, 2]) < 0.012) & (W[:, 0] < 0.77)
h, e = np.histogram(W[tab, 0], bins=np.arange(0.3, 0.78, 0.01)); print("frame0 table-plane x histogram (0.3..0.77, 1 cm):"); print(" ".join(f"{a:.2f}:{n}" for a, n in zip(e[:-1], h)))
# wall plane fit on frame 0 and 4200: points x>0.7, z in 0.05..0.6 -> least squares x = a*y + b*z + c
for fr in [0, 4200]:
    W, z = world(fr); sel = (W[:, 0] > 0.7) & (W[:, 2] > 0.03) & (W[:, 2] < 0.6) & (z < 0.9); A = np.c_[W[sel, 1], W[sel, 2], np.ones(sel.sum())]; coef, *_ = np.linalg.lstsq(A, W[sel, 0], rcond=None); res = W[sel, 0] - A @ coef
    print(f"frame {fr} wall fit x = {coef[0]:+.3f} y {coef[1]:+.3f} z + {coef[2]:.3f}; n {sel.sum()} rms {res.std()*1e3:.1f} mm; y range {W[sel,1].min():.2f}..{W[sel,1].max():.2f} z range {W[sel,2].min():.2f}..{W[sel,2].max():.2f}")
# floor plane RANSAC on frames 2400, 300
rng = np.random.default_rng(0)
for fr in [2400, 300, 2700]:
    W, z = world(fr); sel = (W[:, 2] < -0.35) & (W[:, 2] > -1.1) & (z < 1.6); P = W[sel]
    best = None
    for _ in range(300):
        i = rng.choice(len(P), 3, replace=False); n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; dd = -n @ P[i[0]]; inl = np.abs(P @ n + dd) < 0.02
        if best is None or inl.sum() > best[0]: best = (inl.sum(), n, dd, inl)
    k, n, dd, inl = best; n = n * np.sign(n[2]); Q = P[inl]; print(f"frame {fr} floor RANSAC: n {n.round(3)} z at (x=0.5,y=-1) = {(-(n[0]*0.5 + n[1]*-1 + dd*np.sign(n[2])*1)/n[2]) if False else np.median(Q[:,2]):.3f} (median z of {k} inliers of {len(P)}), tilt {np.degrees(np.arccos(abs(n[2]))):.1f} deg, x range {Q[:,0].min():.2f}..{Q[:,0].max():.2f} y range {Q[:,1].min():.2f}..{Q[:,1].max():.2f}")
    # side: beyond -y: wall or furniture? points with y < -0.6, z > -0.3
    side = (W[:, 1] < -0.6) & (W[:, 2] > -0.3) & (W[:, 2] < 0.8) & (z < 2.0); print(f"   side (-y) upright pts {side.sum()}: y pct 5/50/95 {np.percentile(W[side,1],[5,50,95]).round(2) if side.sum() else None}, x pct {np.percentile(W[side,0],[5,50,95]).round(2) if side.sum() else None}")

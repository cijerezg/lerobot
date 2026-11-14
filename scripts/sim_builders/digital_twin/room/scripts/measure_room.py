"""Measure wall / floor / table edge from the 10-29 wrist depth frames through the sim wrist camera, and overlay the sim
table box edges on the real top frame 0. usage: measure_room.py SCENE_XML ROOT OUT_DIR FRAMES..."""
import sys, pathlib, cv2, numpy as np, mujoco, pandas as pd
scene, root, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]); frames = [int(f) for f in sys.argv[4:]]
out.mkdir(parents=True, exist_ok=True)
FX, FY, CX, CY = 394.9832, 394.9832, 322.5604, 238.6966
GRIPPER_DEG_OPEN, GRIPPER_SLIDE_OPEN = None, None
sys.path.insert(0, "lerobot/src"); from lerobot.probes.action_trace_sim import to_qpos
m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m)
df = pd.read_parquet(f"{root}/data/chunk-000/file-000.parquet", columns=["observation.state", "frame_index", "episode_index"])
df = df[df.episode_index == 0]
states = np.stack(df["observation.state"].to_numpy())
print("states", states.shape, "qpos0", d.qpos[:8].round(3), "park state f0", states[0].round(1))
cam = m.camera("wrist").id
def depth_pts(frame):
    avail = sorted(int(p.stem.split("-")[1]) for p in (pathlib.Path(root) / "depth/wrist.depth/episode-000000").glob("frame-*.png"))
    f = min(avail, key=lambda k: abs(k - frame)); dep = cv2.imread(f"{root}/depth/wrist.depth/episode-000000/frame-{f:06d}.png", cv2.IMREAD_UNCHANGED)
    z = dep.astype(np.float64) * 1e-4; z[dep == 0] = np.nan
    v, u = np.mgrid[0:480, 0:640]; P = np.stack([(u - CX) / FX * z, (v - CY) / FY * z, z], -1); return P, f, z
for fr in frames:
    d.qpos[:8] = to_qpos(states[fr]); mujoco.mj_forward(m, d)
    c, R = d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy()
    P, f, z = depth_pts(fr); Pm = np.stack([P[..., 0], -P[..., 1], -P[..., 2]], -1); W = Pm @ R.T + c
    ok = np.isfinite(W[..., 0])
    print(f"\nframe {fr} (depth f{f}): cam pos {c.round(3)} fwd {(-R[:, 2]).round(2)} valid px {ok.sum()} depth range {np.nanpercentile(z, [1, 50, 99]).round(3)}")
    X, Y, Z = W[..., 0][ok], W[..., 1][ok], W[..., 2][ok]
    tab = np.abs(Z) < 0.01; print(f"  table pts {tab.sum()}: x {np.percentile(X[tab], [1, 50, 99]).round(3) if tab.sum() else None} y {np.percentile(Y[tab], [1, 50, 99]).round(3) if tab.sum() else None}")
    wall = (Z > 0.05) & (Z < 0.7) & (X > 0.5); print(f"  wall cand pts {wall.sum()}: x pct 5/25/50/75/95 {np.percentile(X[wall], [5, 25, 50, 75, 95]).round(3) if wall.sum() else None}")
    fl = Z < -0.3; print(f"  floor cand pts {fl.sum()}: z pct 5/25/50/75/95 {np.percentile(Z[fl], [5, 25, 50, 75, 95]).round(3) if fl.sum() else None}; x pct {np.percentile(X[fl], [5, 50, 95]).round(3) if fl.sum() else None}; y pct {np.percentile(Y[fl], [5, 50, 95]).round(3) if fl.sum() else None}")
    # colour-coded world x / z images for a visual check
    xi = np.full((480, 640), np.nan); xi[ok] = X; zi = np.full((480, 640), np.nan); zi[ok] = Z
    xim = cv2.applyColorMap((255 * np.clip(np.nan_to_num(xi, nan=0) / 1.0, 0, 1)).astype(np.uint8), cv2.COLORMAP_JET); xim[~ok] = 0
    zim = cv2.applyColorMap((255 * np.clip((np.nan_to_num(zi, nan=-1) + 0.8) / 1.2, 0, 1)).astype(np.uint8), cv2.COLORMAP_JET); zim[~ok] = 0
    cv2.imwrite(str(out / f"depth_world_f{fr}.jpg"), np.hstack([xim, zim]))
# top frame overlay
d.qpos[:] = 0; mujoco.mj_forward(m, d)
ext = m.camera("ext").id; c, R = d.cam_xpos[ext].copy(), d.cam_xmat[ext].reshape(3, 3).copy(); fov = m.cam_fovy[ext]; fpx = 240 / np.tan(np.deg2rad(fov) / 2)
def proj(p):
    q = R.T @ (np.asarray(p) - c); return np.array([320 + fpx * q[0] / -q[2], 240 - fpx * q[1] / -q[2]])
g = m.geom("table"); cx_, cy_, _ = m.geom_pos[g.id]; hx, hy, _ = m.geom_size[g.id]
top = cv2.imread(str(out.parent / "real/top_f0.png"))
corners = [(cx_ - hx, cy_ - hy, 0), (cx_ + hx, cy_ - hy, 0), (cx_ + hx, cy_ + hy, 0), (cx_ - hx, cy_ + hy, 0)]
pts = np.array([proj(p) for p in corners]).astype(int); cv2.polylines(top, [pts], True, (0, 255, 0), 1)
for wx, col in [(0.82, (0, 0, 255)), (0.90, (255, 0, 0))]:
    for zz in [-0.75, 0.0, 0.3]:
        a, b = proj((wx, -0.8, zz)), proj((wx, 0.8, zz)); cv2.line(top, tuple(a.astype(int)), tuple(b.astype(int)), col, 1)
for p in corners: print("corner", np.round(p, 3), "->", proj(p).round(1))
print("edge x=0.7175 at y=0 ->", proj((0.7175, 0, 0)).round(1), "wall base x=0.82 z=-0.75 ->", proj((0.82, 0, -0.75)).round(1), "wall x=0.82 z=0 ->", proj((0.82, 0, 0)).round(1), "wall x=0.82 z=0.5 ->", proj((0.82, 0, 0.5)).round(1))
cv2.imwrite(str(out / "top_f0_overlay.png"), top)
r = mujoco.Renderer(m, 480, 640); r.update_scene(d, "ext"); sim = r.render()[:, :, ::-1]
cv2.imwrite(str(out / "top_f0_blend.png"), cv2.addWeighted(cv2.imread(str(out.parent / "real/top_f0.png")), 0.5, np.ascontiguousarray(sim), 0.5, 0))

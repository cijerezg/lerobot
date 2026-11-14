"""Sharp wood-grain patch from a real wrist frame: the frame is rectified onto the table plane z=0 through the sim wrist camera
posed at the frame's state (the D405 at ~0.2 m gives ~3 px/mm against the top camera's 0.47), restricted to pixels whose sim
segmentation is the table, and the largest clean axis-aligned rectangle is kept as a detail layer: the patch divided by its own
low-pass (shade and the arm's soft shadow removed), in world orientation and metres, for photo_scene.py --detail.
usage: MUJOCO_GL=egl uv run python scripts/table_detail.py ROOT EP FRAME SCENE_XML OUT_NPZ [--px_per_m 2000]"""
import argparse, pathlib, sys, cv2, mujoco, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from scene_frames import episode_states, decode_frames, WRIST
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import to_qpos
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("ep", type=int); ap.add_argument("frame", type=int); ap.add_argument("scene"); ap.add_argument("out"); ap.add_argument("--px_per_m", type=int, default=2000); ap.add_argument("--sigma_mm", type=float, default=7.5, help="low-pass removed from the patch: wide enough to take the arm's soft shadow, narrow enough to keep the grain")
ap.add_argument("--no_lit_only", action="store_true", help="also use shadowed table (bigger patch, but the soft shadow edge tiles into lines)"); a = ap.parse_args()
states, _ = episode_states(a.root, a.ep); real = decode_frames(a.root, a.ep, WRIST, [a.frame])[a.frame]  # BGR
model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); data.qpos[:8] = to_qpos(states[a.frame].astype(float)); mujoco.mj_forward(model, data)
H, W = 480, 640; cid = model.camera("wrist").id; pos = data.cam_xpos[cid]; R = data.cam_xmat[cid].reshape(3, 3); f = (H / 2) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
r = mujoco.Renderer(model, H, W); r.enable_segmentation_rendering(); r.update_scene(data, "wrist"); seg = r.render()[:, :, 0]
table = cv2.erode(np.isin(seg, [model.geom(n).id for n in ("table", "table_top") if n in [model.geom(i).name for i in range(model.ngeom)]]).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
# world footprint of the table pixels -> texel grid
d_cam = np.stack([(np.arange(W)[None, :] - W / 2) / f * np.ones((H, 1)), -(np.arange(H)[:, None] - H / 2) / f * np.ones((1, W)), -np.ones((H, W))], -1)
d_w = d_cam @ R.T; t = -pos[2] / d_w[..., 2]; P = pos + t[..., None] * d_w
# the real fingers reach past the sim ones: also drop dark pixels (fingers are black, the arm's shadow on the wood is not)
ok = table & (t > 0) & (cv2.erode((real.mean(2) > 70).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
# and only the lit part of the table: the arm's soft shadow edge would tile into lines. Lit = large-scale brightness above
# 0.9 x the table's 90th percentile.
bright = cv2.GaussianBlur(real.mean(2).astype(np.float32), (0, 0), 15); lit = bright > 0.9 * np.percentile(bright[ok], 90)
if not a.no_lit_only: ok &= cv2.erode(lit.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
x0, x1 = P[ok, 0].min(), P[ok, 0].max(); y0, y1 = P[ok, 1].min(), P[ok, 1].max(); nx, ny = int((x1 - x0) * a.px_per_m), int((y1 - y0) * a.px_per_m)
xs = x0 + (np.arange(nx) + 0.5) / a.px_per_m; ys = y1 - (np.arange(ny) + 0.5) / a.px_per_m  # rows along -y like photo_scene.py
X, Y = np.meshgrid(xs, ys); Pg = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3); c = (Pg - pos) @ R; z = -c[:, 2]
U = (W / 2 + f * c[:, 0] / z).reshape(ny, nx).astype(np.float32); V = (H / 2 - f * c[:, 1] / z).reshape(ny, nx).astype(np.float32)
patch = cv2.remap(real, U, V, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0); good = cv2.remap(ok.astype(np.uint8), U, V, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
# largest all-good axis-aligned rectangle (histogram stack)
best = (0, 0, 0, 0, 0); h = np.zeros(nx, int)
for j in range(ny):
    h = np.where(good[j], h + 1, 0); stack = []
    for i in range(nx + 1):
        cur = h[i] if i < nx else 0; start = i
        while stack and stack[-1][1] >= cur:
            s0, hh = stack.pop(); area = hh * (i - s0)
            if area > best[0]: best = (area, s0, i, j - hh + 1, j + 1); 
            start = s0
        stack.append((start, cur))
_, i0, i1, j0, j1 = best; sub = patch[j0:j1, i0:i1].astype(np.float32); low = cv2.GaussianBlur(sub, (0, 0), a.sigma_mm * a.px_per_m / 1e3); detail = sub / np.maximum(low, 1)
np.savez(a.out, detail=detail, px_per_m=a.px_per_m); cv2.imwrite(a.out.replace(".npz", "_patch.jpg"), sub.astype(np.uint8)); cv2.imwrite(a.out.replace(".npz", "_detail.jpg"), np.clip(detail * 128, 0, 255).astype(np.uint8))
print(f"frame {a.frame}: table pixels {ok.mean() * 100:.0f}% of the image, footprint {(x1 - x0) * 1e3:.0f} x {(y1 - y0) * 1e3:.0f} mm, patch {(i1 - i0) / a.px_per_m * 1e3:.0f} x {(j1 - j0) / a.px_per_m * 1e3:.0f} mm at {a.px_per_m / 1e3:.0f} px/mm -> {a.out}")

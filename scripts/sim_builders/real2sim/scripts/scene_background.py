"""Clean real background for the top camera (SIMPLER-style composite).
1. Per pixel, the median over the episode's static-layout frames (before the real robot rearranges the clutter) in which the
   sim robot (posed from the recorded state, silhouette dilated 45 px: the real fingers reach ~4 cm past the sim ones, plus
   the arm's shadow) does not cover the pixel.
2. Table pixels free in at least --hover_frac (50 %) of those frames under a per-body mask (base + shoulder links 20 px,
   wrist + gripper 45 px; frames whose 15 px window is mostly non-wood rejected, the real arm's lime cover overshoots the sim
   mask) are refilled: per pixel, the frame at the --fill_quantile (70) of the 15 px box-blurred brightness among its free
   frames, pixel copied from that frame (neighbours pick the same frame: no speckle). Brighter than the median: the arm's soft
   shadow, the cable and the fingers are dark minorities at a pixel, so the lit frames win.
3. The rest of the table (the strip around the base the wide mask never frees, the real parked arm's silhouette, the cone
   the gripper hovers over, the object footprints) has no lit sample: it takes the smooth lit shade (normalised Gaussian over
   the lit wood, --shade_sigma px) times a real grain, the largest clean rectangle of lit wood as the ratio to its 8 px
   low-pass, mirror-tiled over the image (seamless, same strip direction). The sim base
   silhouette with --base_keep px beside it down to its bottom row (the real base, lime bar and mount are wider than the sim
   mesh; the composite always renders the sim base) and the clutter with its edges (free-body footprints + 15 px) keep their
   pixels. (Before 2026-10-18 the strip was frame 0 lifted by a seam-band ratio: the parked arm's shadow and gripper stayed.)
4. The start footprints of the free bodies (+ --object_dilate px, default all of them: the composite renders every free body)
   are grain-filled the same way, so a moved object is not also seen lying at its real start spot (a real object wider than
   its primitive + margin leaves a sliver: its dark pixels past the table hull are inpainted; inpainting every ring pixel
   past the hull smeared the table edges).
Writes OUT, OUT_inpaint_mask (the footprints), OUT_fill_mask (128 refilled, 255 grain-filled).
usage: MUJOCO_GL=egl uv run python scripts/scene_background.py WORK/frames_every.npz SCENE_XML OUT_PNG [--inpaint body ...]"""
import argparse, pathlib, sys, cv2, mujoco, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from scene_masks import non_wood
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import to_qpos
ap = argparse.ArgumentParser(); ap.add_argument("frames"); ap.add_argument("scene"); ap.add_argument("out"); ap.add_argument("--robot_dilate", type=int, default=45); ap.add_argument("--near_dilate", type=int, default=20, help="base + link1..3 silhouette margin for the refill"); ap.add_argument("--base_keep", type=int, default=25, help="real pixels kept beside the sim base silhouette down to its bottom row (the real base, lime bar and mount are wider than the sim mesh)"); ap.add_argument("--object_dilate", type=int, default=30); ap.add_argument("--inpaint", nargs="*", default=None, help="free bodies whose start footprint is grain-filled (default: all, every free body is rendered by the composite)"); ap.add_argument("--static_until", default="auto", help="cached-frame index limit for the median: 'auto' = up to the first frame whose table pixels (outside both robot masks) differ from frame 0 by > 10 levels on average (tidy-up episodes move the clutter; a whole-episode median shows objects where they ended up), an int, or 'all'")
ap.add_argument("--fill_quantile", type=float, default=70.0); ap.add_argument("--hover_frac", type=float, default=0.5, help="pixels free in fewer frames than this fraction take the grain fill"); ap.add_argument("--shade_sigma", type=float, default=50.0); a = ap.parse_args()
d = np.load(a.frames); frames, states = d["frames"], d["states"]  # BGR (scene_masks.py writes cv2 frames)
model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); r = mujoco.Renderer(model, 480, 640); r.enable_segmentation_rendering()
NEAR = ["base_link", "link1", "link2", "link3"]; FAR = ["link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right"]
near = np.isin(np.arange(model.nbody), [model.body(b).id for b in NEAR]); far = np.isin(np.arange(model.nbody), [model.body(b).id for b in FAR]); base = np.isin(np.arange(model.nbody), [model.body("base_link").id])
free = np.array([model.body_jntnum[b] > 0 and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE for b in range(model.nbody)])
K = lambda d: np.ones((2 * d + 1,) * 2, np.uint8)
def seg(): r.update_scene(data, "ext"); return r.render()[:, :, 0]
def seg_mask(sel, s):
    m = ((s >= 0) & sel[model.geom_bodyid[np.clip(s, 0, None)]]).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, K(1)) > 0  # free-body silhouette edges come back as thin lines of robot geom ids
covered = np.zeros(frames.shape[:3], bool); covered_body = np.zeros(frames.shape[:3], bool)
for i, st in enumerate(states):
    data.qpos[:8] = to_qpos(np.asarray(st, float)); mujoco.mj_forward(model, data); s = seg(); n_, f_ = seg_mask(near, s).astype(np.uint8), seg_mask(far, s).astype(np.uint8)
    covered[i] = cv2.dilate(n_ | f_, K(a.robot_dilate)) > 0; covered_body[i] = (cv2.dilate(n_, K(a.near_dilate)) | cv2.dilate(f_, K(a.robot_dilate))) > 0
data.qpos[:8] = to_qpos(np.asarray(states[0], float)); mujoco.mj_forward(model, data); seg0 = seg(); table = seg0 == model.geom("table").id  # frame-0 pose (parked)
arm_geoms = (near | far)[model.geom_bodyid] & ~base[model.geom_bodyid]; saved = model.geom_group[arm_geoms].copy(); model.geom_group[arm_geoms] = 5
opt = mujoco.MjvOption(); opt.geomgroup[:] = 1; opt.geomgroup[5] = 0; r.update_scene(data, "ext", opt); base0 = seg_mask(base, r.render()[:, :, 0]); model.geom_group[arm_geoms] = saved  # the whole base, arm hidden
base_bottom = np.flatnonzero(base0.any(1)).max(); print(f"sim base silhouette rows {np.flatnonzero(base0.any(1)).min()}..{base_bottom}, cols {np.flatnonzero(base0.any(0)).min()}..{np.flatnonzero(base0.any(0)).max()}")  # the real base, lime bar and mount are wider than the sim mesh but end at its bottom row; the parked gripper hangs below it
base_keep = (cv2.dilate(base0.astype(np.uint8), K(a.base_keep)) > 0) & (np.arange(table.shape[0])[:, None] <= base_bottom - 2) | (cv2.dilate(base0.astype(np.uint8), K(4)) > 0)
hull = np.zeros(table.shape, np.uint8); cv2.fillConvexPoly(hull, cv2.convexHull(np.argwhere(table)[:, ::-1].astype(np.int32)), 1)
table_full = table | ((seg_mask(near | far, seg0) | seg_mask(free, seg0)) & (hull > 0))  # the table under the parked arm and the objects too (a robot-hidden render z-fights)
clutter = cv2.dilate(seg_mask(free, seg0).astype(np.uint8), K(a.object_dilate)) > 0  # every free body (real pixels + their shadows): not lit wood
if a.inpaint is not None: free = np.isin(np.arange(model.nbody), [model.body(n).id for n in a.inpaint])
objects = cv2.dilate(seg_mask(free, seg0).astype(np.uint8), K(a.object_dilate))
# frames usable: before the real robot rearranges the clutter (the sim scene is the frame-0 layout)
n_use = len(frames)
if a.static_until == "auto":
    diffs = []
    for i in range(len(frames)):
        ok = table & ~covered[i] & ~covered[0]
        diffs.append(float(np.abs(frames[i].astype(np.float32) - frames[0].astype(np.float32)).mean(2)[ok].mean()) if ok.sum() > 1000 else 0.0)
    moved = np.flatnonzero(np.array(diffs) > 10.0); n_use = int(moved[0]) if len(moved) else len(frames)
    print(f"static layout until cached frame {n_use} of {len(frames)} (table change vs frame 0: {np.round(diffs[:n_use + 2], 1).tolist()[-4:]} ...)")
elif a.static_until != "all": n_use = int(a.static_until)
if n_use < 10: n_use = 10
cov = covered[:n_use]; F = frames[:n_use].astype(np.float32)
med = np.nanmedian(np.where(cov[..., None], np.nan, F), axis=0); have = ~np.isnan(med[:, :, 0])
median = frames[0].astype(np.float32); median[have] = med[have]
# 2. refill with the per-body mask and the brightness quantile
nonwood = np.stack([cv2.blur(non_wood(f).astype(np.float32), (15, 15)) for f in frames[:n_use]]) > 0.5
valid = ~covered_body[:n_use] & ~nonwood & ~(objects > 0)[None]; count = valid.sum(0)
scope = table_full & ~(objects > 0) & ~base_keep & ~clutter  # clutter edges keep the median (a brightness quantile fringes dark objects)
hole = (scope & (count < a.hover_frac * n_use)) | (objects > 0) & table_full & ~base_keep; refill = scope & ~hole  # the arm hovers there most of the time: no lit sample, and the quantile would pick its lagging real silhouette
gray = F.mean(3); vf = valid.astype(np.float32)
B = np.stack([cv2.blur(g * v, (15, 15)) / np.maximum(cv2.blur(v, (15, 15)), 1e-3) for g, v in zip(gray, vf)])[:, refill]
order = np.argsort(np.where(valid[:, refill], B, -np.inf), axis=0)  # invalid frames sort first
rank = (n_use - count[refill]) + np.round(a.fill_quantile / 100 * (count[refill] - 1)).astype(int)
pick = np.take_along_axis(order, rank[None], 0)[0]; median[refill] = F[:, refill][pick, np.arange(refill.sum())]
print(f"refilled {refill.sum()} table px (q{a.fill_quantile:.0f}, samples per px median {int(np.median(count[refill]))}), grain-filled {hole.sum()} (free < {a.hover_frac:.0%} of {n_use} frames; never free {(table_full & ~have & ~base_keep).sum()})")
# 3. smooth lit shade (normalised Gaussian over the lit wood); holes take real grain mirrored along the row, scaled to it
wood = ~(cv2.dilate(non_wood(np.clip(median, 0, 255).astype(np.uint8)).astype(np.uint8), K(6)) > 0)  # the real base beyond the ring, cables: not wood
lit = (table_full & wood & ~clutter & ~base_keep & ~(objects > 0) & ~hole).astype(np.float32)
img = median.copy(); low = cv2.GaussianBlur(img, (0, 0), 8)
shade = cv2.GaussianBlur(low * lit[:, :, None], (0, 0), a.shade_sigma) / np.maximum(cv2.GaussianBlur(lit, (0, 0), a.shade_sigma), 1e-4)[:, :, None]
low_c = cv2.GaussianBlur(img * lit[:, :, None], (0, 0), 8) / np.maximum(cv2.GaussianBlur(lit, (0, 0), 8), 1e-4)[:, :, None]  # low-pass over clean wood only
# grain = the largest clean rectangle of lit wood as the ratio to its low-pass, mirror-tiled over the image (seamless), on the shade
def largest_rect(m):  # histogram stack, returns (y0, y1, x0, x1) of the largest all-true rectangle
    best = (0, (0, 0, 0, 0)); h = np.zeros(m.shape[1], int)
    for y in range(m.shape[0]):
        h = np.where(m[y], h + 1, 0); stack = []
        for x in range(m.shape[1] + 1):
            cur = h[x] if x < m.shape[1] else 0
            while stack and h[stack[-1]] >= cur:
                top = stack.pop(); w = x - (stack[-1] + 1 if stack else 0); area = h[top] * w
                if area > best[0]: best = (area, (y - h[top] + 1, y + 1, stack[-1] + 1 if stack else 0, x))
            stack.append(x)
    return best[1]
y0, y1, x0, x1 = largest_rect(lit > 0); patch = img[y0:y1, x0:x1] / np.maximum(low_c[y0:y1, x0:x1], 1)
cyc = np.concatenate([patch, patch[::-1]], 0); cyc = np.concatenate([cyc, cyc[:, ::-1]], 1)
yy = (np.arange(img.shape[0]) - y0) % cyc.shape[0]; xx = (np.arange(img.shape[1]) - x0) % cyc.shape[1]; grain = cyc[yy][:, xx]
img[hole] = shade[hole] * grain[hole]
print(f"grain patch {y1 - y0} x {x1 - x0} px at ({y0}, {x0}), mirror-tiled over {hole.sum()} hole px")
median = np.clip(img, 0, 255).astype(np.uint8)
# dark object pixels past the table hull (a black case over the table edge, wider than its primitive): inpainted from the basket and floor
outside = (objects > 0) & ~table_full & ~base_keep & (cv2.cvtColor(median, cv2.COLOR_BGR2HSV)[..., 2] < 60)
outside = cv2.morphologyEx(outside.astype(np.uint8), cv2.MORPH_OPEN, K(2)); median = cv2.inpaint(median, outside, 5, cv2.INPAINT_TELEA)
print(f"dark object pixels past the hull inpainted: {int(outside.sum())}")
cv2.imwrite(a.out, median); cv2.imwrite(a.out.replace(".png", "_inpaint_mask.png"), (objects > 0).astype(np.uint8) * 255)
fm = np.zeros(table.shape, np.uint8); fm[refill] = 128; fm[hole] = 255; cv2.imwrite(a.out.replace(".png", "_fill_mask.png"), fm)
print(f"{len(frames)} frames; wide-mask-free frames per pixel: min {int((~cov).sum(0).min())}, median {int(np.median((~cov).sum(0)))}; {(~have).mean() * 100:.2f}% of pixels never free; inpainted {objects.mean() * 100:.1f}% -> {a.out}")

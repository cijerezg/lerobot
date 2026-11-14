"""Build the brown seagrass basket asset and the bottle_basket scene.
Textures: real wrist-camera crops of the basket (outputs/rebot_teleop_2026-10-21-cut-v1 ep 0 frames 6000/6030).
Dimensions: wrist depth back-projection (measure_basket.py): rim 0.30 x 0.25 m outside, 0.21 m tall, rim rope ~0.025 m.
Run from the workspace root:  uv run python lerobot/scripts/sim_builders/easy_envs/bottle_basket/build_basket.py
Writes scenes/assets/photoreal/basket/*, scenes/bottle_basket/{scene.xml,objects.json,background_top.png}.
"""
import json, shutil
from pathlib import Path
import cv2, numpy as np, trimesh

WS = Path(__file__).resolve().parents[5]
REF = Path(__file__).resolve().parents[1] / "basket_reference"
SCENES = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes"
A = SCENES / "assets/photoreal/basket"; A.mkdir(parents=True, exist_ok=True)
OUTDIR = SCENES / "bottle_basket"; OUTDIR.mkdir(exist_ok=True)

# ---------------- dimensions (m) ----------------
H = 0.20            # wall height to the rim centre; rim rope radius 0.012 -> top at 0.212 (measured 0.21-0.216)
HX0, HY0 = 0.140, 0.115   # half extents at the base (walls flare ~1 cm per side)
HX1, HY1 = 0.150, 0.125   # half extents at the rim: 0.30 x 0.25 outside (measured 0.29-0.30 x 0.24-0.26)
WALL = 0.012        # weave thickness
FLOOR = 0.012       # floor top height inside
RIM_R = 0.012
POWER = 4           # superellipse corner rounding
BASKET_POS = (0.41, 0.20, 0.0)
WEAVE_GAIN = (0.25, 0.28, 0.20)   # material rgba on the outer weave: real / rendered basket wall in the top camera (verify_bottle_basket.py prints it)
WEAVE_RGBA = "%.3f %.3f %.3f 1" % WEAVE_GAIN; INNER_RGBA = "%.3f %.3f %.3f 1" % tuple(.75 * g for g in WEAVE_GAIN)
BOTTLE_POS = (0.38, -0.08, 0.045)
BOTTLE_R, BOTTLE_H = 0.0275, 0.09
EXT_CAM = np.array([0.60900, -0.01395, 0.84855])

# ---------------- textures from the real frames ----------------
def crop(img, quad, size):
    dst = np.float32([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]])
    Hm = cv2.getPerspectiveTransform(np.float32(quad), dst)
    return cv2.warpPerspective(img, Hm, size, flags=cv2.INTER_CUBIC)

def flatten_light(im, sigma=25):
    """Remove broad lighting (keep the weave's own light/dark strands) so strips tile without seams."""
    f = im.astype(np.float32); blur = cv2.GaussianBlur(f, (0, 0), sigma)
    return np.clip(f * (f.reshape(-1, 3).mean(0) / np.maximum(blur, 1)), 0, 255).astype(np.uint8)

f6030 = cv2.imread(str(REF / "wrist_frame_6030.png"))
# outer wall seen frontally at close range: ep 0 frame 1020 (34.0 s). Quad aligned with the rope rows (bottom edge of the
# wall from (80,214) to (380,117)); its metric size comes from the D405 depth sidecar at the quad corners.
f1020 = cv2.imread(str(REF / "wrist_frame_1020.png"))
dep = cv2.imread(str(WS / "outputs/rebot_teleop_2026-10-21-cut-v1/depth/wrist.depth/episode-000000/frame-001020.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) * 1e-4
quad = [[52, 110], [352, 13], [377, 108], [77, 205]]  # inset from the wall/table edge
FX, CX, CY = 394.9832, 322.5604, 238.6966
def p3(u, v):
    z = np.median(dep[v - 4:v + 5, u - 4:u + 5]); return np.array([(u - CX) / FX * z, (v - CY) / FX * z, z])
c3 = [p3(*q) for q in quad]
patch_w_m = float((np.linalg.norm(c3[1] - c3[0]) + np.linalg.norm(c3[2] - c3[3])) / 2); patch_h_m = float((np.linalg.norm(c3[3] - c3[0]) + np.linalg.norm(c3[2] - c3[1])) / 2)
PW, PH = 480, int(round(480 * patch_h_m / patch_w_m))
patch = flatten_light(crop(f1020, quad, (PW, PH)), sigma=40)
mean_bgr = patch.reshape(-1, 3).mean(0)
json.dump({"weave": {"source": "outputs/rebot_teleop_2026-10-21-cut-v1 wrist ep0 frame 1020 (outer wall, frontal), metric size from the depth sidecar",
                     "quad": quad, "patch_m": [patch_w_m, patch_h_m],
                     "mean_bgr": mean_bgr.tolist()},
           "rim": {"source": "wrist ep0 frame 6030 near rim", "quad": [[130, 282], [600, 278], [600, 326], [130, 330]]}},
          open(A / "texture_sources.json", "w"), indent=1)

# ---------------- meshes ----------------
def outline(hx, hy, n, power=POWER):
    t = np.linspace(0, 2 * np.pi, n + 1)   # closed: first == last (uv seam)
    c, s = np.cos(t), np.sin(t)
    return np.c_[hx * np.sign(c) * np.abs(c) ** (2 / power), hy * np.sign(s) * np.abs(s) ** (2 / power)], t / (2 * np.pi)

# weave texture: mirror-tile the real patch at its true scale over one perimeter (u) x wall height (v)
p_out, _ = outline((HX0 + HX1) / 2, (HY0 + HY1) / 2, 720); PERIM = float(np.sum(np.linalg.norm(np.diff(p_out, axis=0), axis=1)))
def periodic(p, fade, axis):
    """Crossfade the end of the patch into its start so plain tiling is seamless."""
    p = np.moveaxis(p, axis, 1).astype(np.float32); W = p.shape[1]; core = p[:, :W - fade].copy(); wrap = p[:, W - fade:]
    a = np.linspace(0, 1, fade, dtype=np.float32)[None, :, None]; core[:, :fade] = a * core[:, :fade] + (1 - a) * wrap
    return np.moveaxis(core, 1, axis)
tile = periodic(periodic(patch, 64, 1), 24, 0); th, tw = tile.shape[:2]
tw_m, th_m = patch_w_m * tw / PW, patch_h_m * th / PH
big = np.tile(tile, (int(np.ceil(H / th_m)), int(np.ceil(PERIM / tw_m)), 1))
weave = big[:int(round(H / th_m * th)), :int(round(PERIM / tw_m * tw))][::-1].astype(np.uint8)  # v=0 at the floor (OBJ vt origin bottom)
cv2.imwrite(str(A / "basket_weave.png"), weave)
print("weave patch %.3f x %.3f m -> texture %dx%d for perimeter %.3f m" % (patch_w_m, patch_h_m, weave.shape[1], weave.shape[0], PERIM))
# rim rope: the near rim seen from above (frame 6030), mirrored to wrap the tube
rope = flatten_light(crop(f6030, [[130, 282], [600, 278], [600, 326], [130, 330]], (768, 96)), sigma=15)
rim_tex = np.tile(np.vstack([rope, rope[::-1]]), (1, 3, 1))                     # 2304 x 192
cv2.imwrite(str(A / "basket_rim.png"), rim_tex)
def extrude(rows, n=320, inward=False, relief=0.0, vrange=(0.0, 1.0)):
    """rows: list of (z, hx, hy). uv: u = perimeter fraction, v = height fraction. relief: weave bumps in m."""
    v, uv, f = [], [], []
    zs = [r[0] for r in rows]; lo, hi = min(zs), max(zs)
    for i, (z, hx, hy) in enumerate(rows):
        p, tf = outline(hx, hy, n)
        nrm = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)
        bump = relief * np.cos(2 * np.pi * z / 0.014) * (0.6 + 0.4 * np.cos(2 * np.pi * 24 * tf))
        p = p + nrm * bump[:, None] * (-1 if inward else 1)
        for j in range(n + 1):
            v.append([p[j, 0], p[j, 1], z]); uv.append([tf[j], vrange[0] + (vrange[1] - vrange[0]) * (z - lo) / max(hi - lo, 1e-9)])
    for i in range(len(rows) - 1):
        for j in range(n):
            a = i * (n + 1) + j; b = a + 1; c = a + n + 1; d = c + 1
            f.extend([(a, c, b), (b, c, d)] if inward else [(a, b, c), (b, d, c)])
    return np.array(v), np.array(f), np.array(uv)

def cap(z, hx, hy, n=320, up=True, uvscale=1.0):
    p, tf = outline(hx, hy, n)
    v = [[0, 0, z]] + [[x, y, z] for x, y in p]; uv = [[.5, .5]] + [[.5 + x / (2 * hx) * uvscale, .5 + y / (2 * hy) * uvscale] for x, y in p]
    f = [(0, j + 1, j + 2) if up else (0, j + 2, j + 1) for j in range(n)]
    return np.array(v), np.array(f), np.array(uv)

def tube_ring(z, hx, hy, r, n=320, m=20):
    p, tf = outline(hx, hy, n)
    tang = np.gradient(p, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.c_[tang[:, 1], -tang[:, 0]]      # outward in-plane normal
    v, uv, f = [], [], []
    for j in range(n + 1):
        for k in range(m + 1):
            a = 2 * np.pi * k / m
            v.append([p[j, 0] + r * np.cos(a) * nrm[j, 0], p[j, 1] + r * np.cos(a) * nrm[j, 1], z + r * np.sin(a)])
            uv.append([tf[j], k / m])
    for j in range(n):
        for k in range(m):
            a = j * (m + 1) + k; b = a + 1; c = a + m + 1; d = c + 1
            f.extend([(a, b, c), (b, d, c)])
    return np.array(v), np.array(f), np.array(uv)

def write_obj(path, v, f, uv):
    m = trimesh.Trimesh(v, f, process=False); nrm = m.vertex_normals
    with open(path, "w") as fh:
        fh.writelines("v %.6f %.6f %.6f\n" % tuple(x) for x in v)
        fh.writelines("vt %.6f %.6f\n" % tuple(x) for x in uv)
        fh.writelines("vn %.6f %.6f %.6f\n" % tuple(x) for x in nrm)
        fh.writelines("f " + " ".join(f"{j + 1}/{j + 1}/{j + 1}" for j in face) + "\n" for face in f)
    return len(f)

def merge(parts):
    v, f, uv = [], [], []; off = 0
    for pv, pf, puv in parts:
        v.append(pv); f.append(pf + off); uv.append(puv); off += len(pv)
    return np.vstack(v), np.vstack(f), np.vstack(uv)

rows_out = [(0.0, HX0, HY0), (0.05, HX0 + (HX1 - HX0) * .25, HY0 + (HY1 - HY0) * .25), (0.10, (HX0 + HX1) / 2, (HY0 + HY1) / 2),
            (0.15, HX0 + (HX1 - HX0) * .75, HY0 + (HY1 - HY0) * .75), (H, HX1, HY1)]
rows_in = [(z, hx - WALL, hy - WALL) for z, hx, hy in rows_out]; rows_in[0] = (FLOOR, rows_in[0][1], rows_in[0][2])
parts = {
    "basket_wall_outer": [extrude(rows_out, relief=0.0025), cap(0.0, HX0, HY0, up=False)],
    "basket_wall_inner": [extrude(rows_in, inward=True, relief=0.0025)],
    "basket_floor": [cap(FLOOR, rows_in[0][1], rows_in[0][2], up=True, uvscale=.9)],
    "basket_rim": [tube_ring(H, HX1 - WALL / 2, HY1 - WALL / 2, RIM_R)],
}
tri = {name: write_obj(A / f"{name}.obj", *merge(p)) for name, p in parts.items()}
print("meshes", tri)

# ---------------- scene.xml ----------------
# the yellow-label sleeve of the ep5 standing bottle faces the ext camera from its ep0 pose; keep that facing here
phi_ep0 = np.arctan2(EXT_CAM[1] - (-0.2671), EXT_CAM[0] - 0.4118)
phi_new = np.arctan2(EXT_CAM[1] - BOTTLE_POS[1], EXT_CAM[0] - BOTTLE_POS[0])
yaw = phi_new - phi_ep0; bq = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
scene = f'''<?xml version='1.0' encoding='utf-8'?>
<mujoco model="scene_bottle_basket">
  <compiler angle="radian" assetdir=".." autolimits="true" />
  <include file="../rebot_b601_dm_photo.xml" />
  <visual>
    <headlight diffuse=".15 .15 .15" ambient=".37 .37 .37" specular=".10 .10 .10" />
    <rgba haze="0.4548 0.3618 0.2712 1.0000" />
    <global offwidth="1280" offheight="960" />
    <quality shadowsize="4096" />
  </visual>
  <asset>
    <texture name="table_top_tex" type="2d" file="table_top.png" />
    <material name="table_top" texture="table_top_tex" texrepeat="1 1" texuniform="false" specular="0" shininess="0" />
    <texture type="skybox" builtin="flat" rgb1="0.5685 0.4643 0.3526" rgb2="0.5685 0.4643 0.3526" width="32" height="32" />
    <material name="wood" rgba="0.5291 0.3455 0.2161 1.0000" specular="0.15" shininess="0.2" />
    <material name="floor_mat" rgba="0.1213 0.1025 0.0542 1.0000" />
    <!-- the bottle: ep0's photoreal standing bottle (lerobot/scripts/sim_builders/photoreal_objects, nail_ep5) -->
    <texture name="detail_yellow_label" type="2d" file="assets/photoreal/objects/yellow_label.png" />
    <material name="clutter_white" rgba=".80 .81 .79 1" specular=".23" shininess=".35" />
    <material name="clutter_cap" rgba=".88 .89 .87 1" specular=".28" shininess=".45" />
    <material name="clutter_yellow_label" texture="detail_yellow_label" specular=".15" shininess=".25" />
    <mesh name="nail_ep5_bottle_yellow_cap_standing_00" file="assets/photoreal/objects/nail_ep5_bottle_yellow_cap_standing_00.obj" inertia="shell" />
    <mesh name="nail_ep5_bottle_yellow_cap_standing_01" file="assets/photoreal/objects/nail_ep5_bottle_yellow_cap_standing_01.obj" inertia="shell" />
    <mesh name="nail_ep5_bottle_yellow_cap_standing_02" file="assets/photoreal/objects/nail_ep5_bottle_yellow_cap_standing_02.obj" inertia="shell" />
    <!-- the basket: real seagrass weave and rim crops (assets/photoreal/basket/texture_sources.json) -->
    <texture name="basket_weave_tex" type="2d" file="assets/photoreal/basket/basket_weave.png" />
    <texture name="basket_rim_tex" type="2d" file="assets/photoreal/basket/basket_rim.png" />
    <material name="basket_weave" texture="basket_weave_tex" rgba="{WEAVE_RGBA}" specular=".12" shininess=".2" />
    <material name="basket_weave_inner" texture="basket_weave_tex" rgba="{INNER_RGBA}" specular=".08" shininess=".15" />
    <material name="basket_floor" texture="basket_weave_tex" rgba=".55 .53 .50 1" specular=".05" shininess=".1" />
    <material name="basket_rim" texture="basket_rim_tex" rgba=".40 .42 .32 1" specular=".15" shininess=".25" />
    <mesh name="basket_wall_outer" file="assets/photoreal/basket/basket_wall_outer.obj" inertia="shell" />
    <mesh name="basket_wall_inner" file="assets/photoreal/basket/basket_wall_inner.obj" inertia="shell" />
    <mesh name="basket_floor" file="assets/photoreal/basket/basket_floor.obj" inertia="shell" />
    <mesh name="basket_rim" file="assets/photoreal/basket/basket_rim.obj" inertia="shell" />
  </asset>
  <default>
    <default class="basket_col">
      <geom type="box" group="3" condim="4" friction="0.9 0.005 0.0001" rgba="0.35 0.25 0.15 1" />
    </default>
    <default class="basket_vis">
      <geom type="mesh" group="2" contype="0" conaffinity="0" mass="0" />
    </default>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.72" material="floor_mat" />
    <geom name="table" type="box" size="0.3708 0.5053 0.015" pos="0.3467 0.0244 -0.015" material="wood" contype="0" conaffinity="0" />
    <geom name="table_top" type="plane" size="0.3708 0.5053 0.001" pos="0.3467 0.0244 0.0005" material="table_top" contype="0" conaffinity="0" />
    <geom name="table_contact" type="plane" size="0.3708 0.5053 0.001" pos="0.3467 0.0244 0" group="3" />
    <camera name="ext" pos="0.60900 -0.01395 0.84855" quat="0.716093 0.064734 0.080219 0.690351" fovy="60.730" />
    <body name="bottle" pos="{BOTTLE_POS[0]:.4f} {BOTTLE_POS[1]:.4f} {BOTTLE_POS[2]:.4f}" quat="{bq[0]:.5f} 0 0 {bq[3]:.5f}">
      <freejoint />
      <geom group="3" name="bottle" type="cylinder" size="{BOTTLE_R:.4f} {BOTTLE_H / 2:.4f}" mass="0.12" rgba="0.7201 0.5728 0.4158 1.0000" condim="4" friction="0.6 0.005 0.0001" />
      <geom name="bottle_body_visual" type="mesh" mesh="nail_ep5_bottle_yellow_cap_standing_00" material="clutter_white" group="2" contype="0" conaffinity="0" mass="0" />
      <geom name="bottle_cap_visual" type="mesh" mesh="nail_ep5_bottle_yellow_cap_standing_01" material="clutter_cap" group="2" contype="0" conaffinity="0" mass="0" />
      <geom name="bottle_label_visual" type="mesh" mesh="nail_ep5_bottle_yellow_cap_standing_02" material="clutter_yellow_label" group="2" contype="0" conaffinity="0" mass="0" />
    </body>
    <!-- fixed (welded) body: it cannot slide when the bottle is dropped in; hollow collider = floor + four walls -->
    <body name="basket" pos="{BASKET_POS[0]:.4f} {BASKET_POS[1]:.4f} {BASKET_POS[2]:.4f}" quat="1 0 0 0">
      <geom class="basket_col" name="basket_floor" size="{HX0 - WALL:.4f} {HY0 - WALL:.4f} {FLOOR / 2:.4f}" pos="0 0 {FLOOR / 2:.4f}" />
      <geom class="basket_col" name="basket_wall_front" size="{WALL / 2:.4f} {(HY0 + HY1) / 2:.4f} {(H + RIM_R) / 2:.4f}" pos="{(HX0 + HX1) / 2 - WALL / 2:.4f} 0 {(H + RIM_R) / 2:.4f}" />
      <geom class="basket_col" name="basket_wall_back" size="{WALL / 2:.4f} {(HY0 + HY1) / 2:.4f} {(H + RIM_R) / 2:.4f}" pos="{-((HX0 + HX1) / 2 - WALL / 2):.4f} 0 {(H + RIM_R) / 2:.4f}" />
      <geom class="basket_col" name="basket_wall_right" size="{(HX0 + HX1) / 2:.4f} {WALL / 2:.4f} {(H + RIM_R) / 2:.4f}" pos="0 {(HY0 + HY1) / 2 - WALL / 2:.4f} {(H + RIM_R) / 2:.4f}" />
      <geom class="basket_col" name="basket_wall_left" size="{(HX0 + HX1) / 2:.4f} {WALL / 2:.4f} {(H + RIM_R) / 2:.4f}" pos="0 {-((HY0 + HY1) / 2 - WALL / 2):.4f} {(H + RIM_R) / 2:.4f}" />
      <geom class="basket_vis" name="basket_wall_outer_visual" mesh="basket_wall_outer" material="basket_weave" />
      <geom class="basket_vis" name="basket_wall_inner_visual" mesh="basket_wall_inner" material="basket_weave_inner" />
      <geom class="basket_vis" name="basket_floor_visual" mesh="basket_floor" material="basket_floor" />
      <geom class="basket_vis" name="basket_rim_visual" mesh="basket_rim" material="basket_rim" />
    </body>
    <light directional="true" name="detail_key" pos=".3 -.3 2" dir="-.1 .15 -1" diffuse=".40 .40 .40" specular=".8 .8 .8" castshadow="false" />
    <light directional="true" name="detail_fill" pos="1.5 1 1.5" dir="-.6 -.5 -.6" diffuse=".16 .16 .16" specular=".45 .45 .45" castshadow="false" />
    <light name="soft_shadow_0" directional="true" pos=".3 -.3 2" dir="-0.17 0.07999999999999999 -1" diffuse=".045 .045 .045" specular="0 0 0" castshadow="true" />
    <light name="soft_shadow_1" directional="true" pos=".3 -.3 2" dir="-0.17 0.22 -1" diffuse=".045 .045 .045" specular="0 0 0" castshadow="true" />
    <light name="soft_shadow_2" directional="true" pos=".3 -.3 2" dir="-0.03 0.07999999999999999 -1" diffuse=".045 .045 .045" specular="0 0 0" castshadow="true" />
    <light name="soft_shadow_3" directional="true" pos=".3 -.3 2" dir="-0.03 0.22 -1" diffuse=".045 .045 .045" specular="0 0 0" castshadow="true" />
  </worldbody>
</mujoco>
'''
(OUTDIR / "scene.xml").write_text(scene)
shutil.copyfile(SCENES / "clippers_only/background_top.png", OUTDIR / "background_top.png")
objects = {
    "task": "Pick the bottle up and put it in the brown basket.",
    "objects": {
        "bottle": {"xy": list(BOTTLE_POS[:2]), "z": BOTTLE_POS[2], "yaw_deg": float(np.degrees(yaw)), "kind": "cylinder standing", "radius_m": BOTTLE_R, "height_m": BOTTLE_H,
                   "free": True, "visual": "ep0 bottle_yellow_cap_standing (nail_ep5 photoreal meshes, real yellow label crop); yaw keeps the label facing the ext camera"},
        "basket": {"xy": list(BASKET_POS[:2]), "z": 0.0, "yaw_deg": 0.0, "kind": "open box, fixed body", "outer_rim_m": [2 * HX1, 2 * HY1], "outer_base_m": [2 * HX0, 2 * HY0],
                   "height_m": H + RIM_R, "wall_m": WALL, "floor_top_m": FLOOR, "inner_m": [2 * (HX0 + HX1) / 2 - 2 * WALL, 2 * (HY0 + HY1) / 2 - 2 * WALL, H + RIM_R - FLOOR],
                   "free": False, "collision": "floor box + four wall boxes (group 3), hollow",
                   "visual": "assets/photoreal/basket: woven walls (relief + real seagrass crops), rolled rim, floor"},
    },
    "separation_m": float(np.hypot(BASKET_POS[0] - BOTTLE_POS[0], BASKET_POS[1] - BOTTLE_POS[1])),
    "radius_from_base_m": {"bottle": float(np.hypot(*BOTTLE_POS[:2])), "basket_centre": float(np.hypot(*BASKET_POS[:2]))},
    "basket_measurement": "lerobot/scripts/sim_builders/easy_envs/basket_reference/basket_depth_measurement.json (wrist D405 depth of the 2026-10-21 teleop + sim FK camera pose)",
    "camera": "ext/wrist as clippers_only (same table, background_top.png, lights)",
    "source": "lerobot/scripts/sim_builders/easy_envs/bottle_basket/build_basket.py",
}
(OUTDIR / "objects.json").write_text(json.dumps(objects, indent=1))
print("wrote", OUTDIR / "scene.xml", "bottle yaw deg", np.degrees(yaw).round(1), "separation", objects["separation_m"])

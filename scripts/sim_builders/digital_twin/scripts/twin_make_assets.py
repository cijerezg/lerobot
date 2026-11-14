"""Simple generated assets for the twin (v2): twin/assets/roar_bottle (lathe mesh + label texture warped off the real wrist frame
300 of rebot_teleop_2026-10-29-cut-v1 ep 0), twin/assets/navy_bottle (lathe mesh, plain navy; the far-left dark bottle), twin/assets/bits
(hex prism mesh for the loose screwdriver bits) and the brightened kit texture twin/assets/bits_kit/bits_kit_colors_gain.png.
Run from the LeRobot root: uv run python lerobot/scripts/sim_builders/digital_twin/scripts/twin_make_assets.py"""
import json, pathlib, numpy as np, cv2

ROOT = pathlib.Path("lerobot/scripts/sim_builders/digital_twin"); SRC = pathlib.Path("assets/sim_sources/digital_twin"); TWIN = ROOT / "twin"; A = TWIN / "assets"
KIT_GAIN = (4.0, 2.9, 3.2)  # per channel B, G, R: real blue rim mean BGR (142, 62, 31) on top frame f0 (HSV mask) / scan texture blue px (35.5, 21.3, 9.8); a scalar x4 left a cyan cast

def lathe_obj(path, profile, n=48):
    """profile = [(z, r), ...] bottom -> top; closed by a bottom and a top disc. u = angle / 2pi (seam at +x, CCW seen from +z), v = arc length fraction."""
    zs, rs = np.array([p[0] for p in profile]), np.array([p[1] for p in profile])
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(zs), np.diff(rs)))]); vfrac = arc / arc[-1]
    V, VT, F = [], [], []
    for k, (z, r) in enumerate(profile):
        for i in range(n + 1):
            a = 2 * np.pi * i / n; V.append((r * np.cos(a), r * np.sin(a), z)); VT.append((i / n, vfrac[k]))
    rows = len(profile)
    for k in range(rows - 1):
        for i in range(n):
            a, b, c, d = k * (n + 1) + i, k * (n + 1) + i + 1, (k + 1) * (n + 1) + i + 1, (k + 1) * (n + 1) + i
            F += [(a, b, c), (a, c, d)]
    for z, vv, flip in [(zs[0], 0.0, True), (zs[-1], 1.0, False)]:
        c = len(V); V.append((0, 0, z)); VT.append((0.5, vv)); row = 0 if flip else rows - 1
        for i in range(n):
            a, b = row * (n + 1) + i, row * (n + 1) + i + 1; F.append((c, b, a) if flip else (c, a, b))
    with open(path, "w") as f:
        for v in V: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in VT: f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for a, b, c in F: f.write(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n")
    return arc[-1]

# ---- ROAR bottle: 65 mm body, 210 mm incl. the teal cap
R = A / "roar_bottle"; R.mkdir(parents=True, exist_ok=True)
profile = [(0.0, 0.030), (0.004, 0.0325), (0.135, 0.0325), (0.150, 0.030), (0.165, 0.022), (0.185, 0.020)]
lathe_obj(R / "roar_bottle_body.obj", profile)
# label: the real wrist frame shows the front half of the body; quad = body silhouette (top-left, top-right, bottom-right, bottom-left), hand-picked
wr = cv2.imread(str(ROOT / "room/renders/real_wrist_f300.png"))
quad = dict(tl=(367, 138), tr=(480, 138), br=(435, 385), bl=(347, 385))
WV, H = 256, 512  # visible half -> 256 columns; full wrap 512
tex = np.zeros((H, 2 * WV, 3), np.uint8)
rows = np.arange(H); fy = rows / (H - 1)                                      # 0 = image top = bottle shoulder (v = 1)
cx = (1 - fy) * (quad["tl"][0] + quad["tr"][0]) / 2 + fy * (quad["bl"][0] + quad["br"][0]) / 2
hw = (1 - fy) * (quad["tr"][0] - quad["tl"][0]) / 2 + fy * (quad["br"][0] - quad["bl"][0]) / 2
sy = (1 - fy) * quad["tl"][1] + fy * quad["bl"][1]
theta = -np.pi / 2 + np.pi * (np.arange(WV) + 0.5) / WV                       # cylinder angle of each flat column
mapx = (cx[:, None] + np.sin(theta)[None, :] * hw[:, None]).astype(np.float32); mapy = np.repeat(sy[:, None], WV, 1).astype(np.float32)
vis = cv2.remap(wr, mapx, mapy, cv2.INTER_LINEAR)
orange = np.median(vis.reshape(-1, 3), 0)                                     # fill colour for the unseen half
tex[:, :WV] = vis; tex[:, WV:] = orange
# soften the two seams with a 16-px blend into the fill colour
for e in range(16):
    w = (e + 1) / 17.0; tex[:, WV - 16 + e] = (w * orange + (1 - w) * tex[:, WV - 16 + e]).astype(np.uint8); tex[:, 15 - e] = (w * orange + (1 - w) * tex[:, 15 - e]).astype(np.uint8)
tex = np.roll(tex, WV // 2, axis=1)                                            # label centre faces +x (u = 0.25 -> angle 90 deg would be +y; roll so centre at u = 0)
cv2.imwrite(str(R / "roar_bottle_tex.png"), tex)
teal = wr[90:105, 415:455].reshape(-1, 3).mean(0)[::-1] / 255.0 * 1.15
(R / "roar_bottle.xml").write_text(f"""<mujoco>
  <asset>
    <texture name="roar_bottle_tex" type="2d" file="roar_bottle_tex.png"/>
    <material name="roar_bottle_mat" texture="roar_bottle_tex" specular="0.3" shininess="0.4"/>
    <mesh name="roar_bottle_body" file="roar_bottle_body.obj"/>
  </asset>
  <default>
    <default class="roar_bottle_col"><geom group="3" condim="4" friction="0.6 0.005 0.0001"/></default>
    <default class="roar_bottle_vis"><geom group="2" contype="0" conaffinity="0" mass="0"/></default>
  </default>
  <worldbody>
    <body name="roar_bottle" pos="0 0 0">
      <freejoint/>
      <geom class="roar_bottle_vis" type="mesh" mesh="roar_bottle_body" material="roar_bottle_mat"/>
      <geom class="roar_bottle_vis" type="cylinder" size="0.0175 0.0125" pos="0 0 0.1975" rgba="{teal[0]:.3f} {teal[1]:.3f} {teal[2]:.3f} 1"/>
      <geom class="roar_bottle_col" type="cylinder" size="0.0325 0.0725" pos="0 0 0.0725" mass="0.45"/>
      <geom class="roar_bottle_col" type="cylinder" size="0.022 0.020" pos="0 0 0.165" mass="0.03"/>
      <geom class="roar_bottle_col" type="cylinder" size="0.0175 0.0125" pos="0 0 0.1975" mass="0.02"/>
    </body>
  </worldbody>
</mujoco>
""")
(R / "roar_bottle_meta.json").write_text(json.dumps({"size_m": [0.065, 0.065, 0.210], "mass_kg": 0.5, "profile_z_r": profile, "label_source": "room/renders/real_wrist_f300.png", "label_quad_px": quad, "unseen_half_fill_bgr": orange.tolist(), "cap_rgb": teal.round(3).tolist(), "hulls": "3 cylinders (body, neck, cap)"}, indent=1))
print("roar bottle: fill colour bgr", orange, "cap rgb", teal.round(3))

# ---- navy bottle: 70 x 200 mm, blue screw cap (the far-left dark bottle of wrist frame 2400)
N = A / "navy_bottle"; N.mkdir(parents=True, exist_ok=True)
prof_n = [(0.0, 0.032), (0.004, 0.035), (0.140, 0.035), (0.160, 0.028), (0.172, 0.024)]
lathe_obj(N / "navy_bottle_body.obj", prof_n)
(N / "navy_bottle.xml").write_text("""<mujoco>
  <asset>
    <mesh name="navy_bottle_body" file="navy_bottle_body.obj"/>
  </asset>
  <default>
    <default class="navy_bottle_col"><geom group="3" condim="4" friction="0.6 0.005 0.0001"/></default>
    <default class="navy_bottle_vis"><geom group="2" contype="0" conaffinity="0" mass="0"/></default>
  </default>
  <worldbody>
    <body name="navy_bottle" pos="0 0 0">
      <freejoint/>
      <geom class="navy_bottle_vis" type="mesh" mesh="navy_bottle_body" rgba="0.10 0.13 0.26 1"/>
      <geom class="navy_bottle_vis" type="cylinder" size="0.0245 0.014" pos="0 0 0.186" rgba="0.33 0.38 0.55 1"/>
      <geom class="navy_bottle_col" type="cylinder" size="0.035 0.075" pos="0 0 0.075" mass="0.42"/>
      <geom class="navy_bottle_col" type="cylinder" size="0.027 0.011" pos="0 0 0.161" mass="0.04"/>
      <geom class="navy_bottle_col" type="cylinder" size="0.0245 0.014" pos="0 0 0.186" mass="0.04"/>
    </body>
  </worldbody>
</mujoco>
""")
(N / "navy_bottle_meta.json").write_text(json.dumps({"size_m": [0.070, 0.070, 0.200], "mass_kg": 0.5, "profile_z_r": prof_n, "source": "built from wrist frame 2400 (navy bottle, blue screw cap); the downloaded pump bottle is a different object"}, indent=1))

# ---- hex bit: 25 mm long along x, 6 mm across flats (flats facing +-z)
B = A / "bits"; B.mkdir(parents=True, exist_ok=True)
Rc = 0.003 / np.cos(np.pi / 6); L = 0.0125
with open(B / "bit_hex.obj", "w") as f:
    for x in (-L, L):
        for k in range(6):
            a = np.pi / 3 * k; f.write(f"v {x:.6f} {Rc*np.cos(a):.6f} {Rc*np.sin(a):.6f}\n")
    for k in range(6):
        a, b, c, d = k + 1, (k + 1) % 6 + 1, (k + 1) % 6 + 7, k + 7; f.write(f"f {a} {d} {c}\nf {a} {c} {b}\n")
    f.write("f 1 3 2\nf 1 4 3\nf 1 5 4\nf 1 6 5\nf 7 8 9\nf 7 9 10\nf 7 10 11\nf 7 11 12\n")

# ---- kit texture gain (copy of assets/bits_kit/bits_kit_colors.png; the original is untouched)
K = A / "bits_kit"; K.mkdir(parents=True, exist_ok=True)
src = cv2.imread(str(SRC / "assets/bits_kit/bits_kit_colors.png"))
out = np.clip(src.astype(np.float32) * np.array(KIT_GAIN, np.float32), 0, 255).astype(np.uint8)
cv2.imwrite(str(K / "bits_kit_colors_gain.png"), out)
hs = cv2.cvtColor(src, cv2.COLOR_BGR2HSV); msk = (hs[..., 0] > 95) & (hs[..., 0] < 130) & (hs[..., 1] > 80) & (hs[..., 2] > 30)  # blue px of the SOURCE texture
print(f"kit texture gain BGR {KIT_GAIN}: source blue px mean BGR {src[msk].mean(0).round(1)} -> {out[msk].mean(0).round(1)} (real rim 142 62 31), saturated px {(out == 255).any(-1).mean()*100:.1f} %")

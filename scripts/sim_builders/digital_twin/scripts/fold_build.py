"""Tri-fold bits kit: split the rigid scan asset into centre + two hinged flaps and write assets/bits_kit_fold/.
Inputs: work/bits_kit_fold/profile.json + heightmap.npz (PCA-aligned frame, fold_profile.py), assets/bits_kit/bits_kit_visual.obj + colors.png
(table frame minus origin_table_m from bits_kit_meta.json). Fold lines = the two height valleys (checked against the seams in the scan
frames, see FOLD_REPORT.md). Colliders: one slab box per panel + content boxes from the detrended height map. Hinges at the fold lines at
slab-top height; the flap slabs are inset one slab thickness from the fold (the fabric hinge strip) so a flap can swing past vertical.
Open rest angle is held by a dead-band tendon spring (zero torque above REST_LO, pushes up below it); gravity holds a closed flap.
usage: fold_build.py  (from the LeRobot root)"""
import json, pathlib, shutil, numpy as np, cv2
D = pathlib.Path("assets/sim_sources/digital_twin"); W = D / "work/bits_kit_fold"; A = D / "assets/bits_kit_fold"; A.mkdir(parents=True, exist_ok=True)
pr = json.load(open(W / "profile.json")); c = np.array(pr["centre_table_xy"]); R = np.array(pr["R_rows"]); meta_rigid = json.load(open(D / "assets/bits_kit/bits_kit_meta.json"))
hm = np.load(W / "heightmap.npz"); Hm = hm["Hm"]; xs = hm["xs"]; ys = hm["ys"]; res = float(hm["res"])
# ---- panels (aligned frame, metres). Folds = valley minima of the smoothed median profile; outer edges = where the median height passes
# 6 mm on the skirted left end (-245) and the sharp lifted edge on the right (229). Width 240 mm (height-map extent at 6 mm: 228-242).
FOLDS = [-0.100, 0.083]; EDGES = [-0.245, FOLDS[0], FOLDS[1], 0.229]; HALF_W = 0.120; YC = -0.0015
SLAB = 0.007; STRIP = 0.030; STRIP_T = 0.0025; STRIP_MASS = 0.005; GAP = 0.002; MASS = 0.35; REST_A = np.radians(6.0); REST_B = np.radians(6.0); K_SPRING = 1.0; DAMPING = 0.03; FRICTIONLOSS = 0.01
BIT_R = 0.00225; BIT_LEN = 0.022; BIT_PITCH = 0.0085; BIT_ROW_PITCH = 0.026; BIT_LIFT = 0.002  # visual bit rows (v3), see body_geoms
widths = np.diff(EDGES); names = ["left", "centre", "right"]; print("panel widths mm:", (1e3 * widths).round(1))
# body origins in the aligned frame: centre = centre-panel footprint centre at table level; flaps = hinge point (fold line, slab top)
# fabric hinge = strip body between the fold line (hinge A, centre-to-strip) and the flap (hinge B, strip-to-flap), both at slab-top height
origin = {"centre": np.array([(FOLDS[0] + FOLDS[1]) / 2, YC, 0.0]), "left_strip": np.array([FOLDS[0], YC, SLAB]), "right_strip": np.array([FOLDS[1], YC, SLAB]),
          "left": np.array([FOLDS[0] - STRIP, YC, SLAB]), "right": np.array([FOLDS[1] + STRIP, YC, SLAB])}
SIGN = {"left": -1, "right": 1}  # outward direction of the flap along x
# slab x ranges (aligned frame): flaps inset by GAP at the fold
slab_x = {"left": (EDGES[0], FOLDS[0] - STRIP - GAP), "centre": (FOLDS[0], FOLDS[1]), "right": (FOLDS[1] + STRIP + GAP, EDGES[3])}
strip_x = {"left_strip": (FOLDS[0] - STRIP, FOLDS[0]), "right_strip": (FOLDS[1], FOLDS[1] + STRIP)}
# ---- content boxes: read by hand off the gridded height map (work/bits_kit_fold/content_boxes.jpg; floor ~8-10 mm, bits 15-17, driver 21-25,
# the card flap is lifted ~10 mm at its edge in the scan). Heights are above the slab. Only bit columns, driver, extension bar, pockets, card.
H = np.nan_to_num(Hm).astype(np.float32); miss = np.isnan(Hm).astype(np.uint8)
H8 = np.clip(H * 1e3 * 5, 0, 255).astype(np.uint8); H = cv2.medianBlur(cv2.inpaint(H8, miss, 3, cv2.INPAINT_TELEA), 5).astype(np.float32) / 5e3
mm = lambda x0, x1, y0, y1, h, what: dict(x=[x0 / 1e3, x1 / 1e3], y=[YC + y0 / 1e3, YC + y1 / 1e3], h=h / 1e3, what=what)
contents = {"left": [mm(-238, -180, -105, 5, 8, "bit column A (4-5 rows)"), mm(-175, -115, -105, 5, 8, "bit column B (4-5 rows)"), mm(-235, -115, 20, 105, 4, "flat fabric pocket")],
            "centre": [mm(-95, -30, -105, -15, 8, "bit column (4 rows)"), mm(0, 35, 30, 100, 10, "driver handle"), mm(45, 62, -100, 45, 6, "extension bar / pen driver"), mm(-95, -15, 20, 100, 4, "flat fabric pocket")],
            "right": [mm(115, 220, -100, 100, 4, "card pocket with product card")]}
for n in names: print(n, "content boxes:", ", ".join("%.0fx%.0fx%.0f mm %s" % (1e3 * (b["x"][1] - b["x"][0]), 1e3 * (b["y"][1] - b["y"][0]), 1e3 * b["h"], b["what"]) for b in contents[n]))
# debug image: detrended height with boxes
fig = np.clip(np.nan_to_num(H) * 1e3 / 30, 0, 1); img = cv2.applyColorMap((fig * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS); img = cv2.flip(img, 0); sc = 3; img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
def px(x, y): return int((x - xs[0]) / res * sc), int(img.shape[0] - (y - ys[0]) / res * sc)
for n in names:
    for b in contents[n]: cv2.rectangle(img, px(b["x"][0], b["y"][1]), px(b["x"][1], b["y"][0]), (0, 0, 255), 2)
    cv2.rectangle(img, px(slab_x[n][0], YC + HALF_W), px(slab_x[n][1], YC - HALF_W), (255, 255, 255), 1)
for f in FOLDS: cv2.line(img, px(f, -0.14), px(f, 0.14), (0, 255, 255), 2)
cv2.imwrite(str(W / "content_boxes.jpg"), img); print("wrote", W / "content_boxes.jpg")
# ---- visual mesh split (per-face uv OBJ from the rigid asset): faces by centroid x, trimmed to the panel footprint + 3 mm, table fan dropped
src = (D / "assets/bits_kit/bits_kit_visual.obj").read_text().splitlines()
V = np.array([l.split()[1:] for l in src if l.startswith("v ")], float); VT = np.array([l.split()[1:] for l in src if l.startswith("vt ")], float); nF = len(V) // 3
Vt = V + np.array(meta_rigid["origin_table_m"]); Va = Vt.copy(); Va[:, :2] = (Vt[:, :2] - c) @ R.T
cen = Va.reshape(nF, 3, 3).mean(1); keep = (cen[:, 0] > EDGES[0] - 0.003) & (cen[:, 0] < EDGES[3] + 0.003) & (np.abs(cen[:, 1] - YC) < HALF_W + 0.003) & (cen[:, 2] > 0.0025)
bounds = [FOLDS[0] - STRIP, FOLDS[0], FOLDS[1], FOLDS[1] + STRIP]; part_names = ["left", "left_strip", "centre", "right_strip", "right"]; panel_of = np.digitize(cen[:, 0], bounds); faces_kept = {}
for i, n in enumerate(part_names):
    sel = np.nonzero(keep & (panel_of == i))[0]; faces_kept[n] = len(sel); idx = (sel[:, None] * 3 + np.arange(3)).reshape(-1); Vb = Va[idx] - origin[n]; uv = VT[idx]
    with open(A / f"bits_kit_{n}_visual.obj", "w") as fh:
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in Vb); fh.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in uv)
        fh.writelines(f"f {3 * k + 1}/{3 * k + 1} {3 * k + 2}/{3 * k + 2} {3 * k + 3}/{3 * k + 3}\n" for k in range(len(sel)))
shutil.copy(D / "assets/bits_kit/bits_kit_colors.png", A / "bits_kit_fold_colors.png"); print("faces kept per panel:", faces_kept, "of", nF, "(dropped %d skirt/table faces)" % (nF - keep.sum()))
# ---- geoms (body frames). Per-panel mass by area, split inside the panel by box volume. Hinge axes: +y lifts the left flap, -y the right.
def box(n, x0, x1, y0, y1, z0, z1):
    o = origin[n]; return dict(pos=[(x0 + x1) / 2 - o[0], (y0 + y1) / 2 - o[1], (z0 + z1) / 2 - o[2]], size=[(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2], vol=(x1 - x0) * (y1 - y0) * (z1 - z0))
geoms = {}; masses = {}
for i, n in enumerate(names):
    x0, x1 = slab_x[n]; g = [("slab", box(n, x0, x1, YC - HALF_W, YC + HALF_W, 0.0, SLAB))]
    g += [(f"content{k}", box(n, b["x"][0], b["x"][1], b["y"][0], b["y"][1], SLAB, SLAB + b["h"])) for k, b in enumerate(contents[n])]
    vol = sum(d["vol"] for _, d in g); pm = MASS * widths[i] / widths.sum() - (STRIP_MASS if n != "centre" else 0); masses[n] = pm
    for _, d in g: d["mass"] = pm * d["vol"] / vol
    geoms[n] = g
for n in ("left_strip", "right_strip"):
    x0, x1 = strip_x[n]; d = box(n, x0, x1, YC - HALF_W, YC + HALF_W, SLAB - STRIP_T, SLAB); d["mass"] = STRIP_MASS; geoms[n] = [("strip", d)]; masses[n] = STRIP_MASS
# spring dead-band edges: gravity torque about hinge B (flap geoms, flap tilted REST_A + REST_B) and about hinge A (flap + strip, strip at REST_A),
# each balanced by K_SPRING * (lo - rest). u = outward distance from the pivot, z = height above it; lifting by phi: u' = u cos phi - z sin phi.
def lever(u, z, phi): return u * np.cos(phi) - z * np.sin(phi)
def torques(n):
    sgn = SIGN[n]; tB = sum(d["mass"] * 9.81 * lever(sgn * d["pos"][0], d["pos"][2], REST_A + REST_B) for _, d in geoms[n])
    tA = sum(d["mass"] * 9.81 * (STRIP * np.cos(REST_A) + lever(sgn * d["pos"][0], d["pos"][2], REST_A + REST_B)) for _, d in geoms[n])
    st = geoms[n + "_strip"][0][1]; tA += st["mass"] * 9.81 * lever(sgn * st["pos"][0], st["pos"][2], REST_A); return tA, tB
spring_lo = {}
for n in ("left", "right"):
    tA, tB = torques(n); spring_lo[n] = float(REST_A + tA / K_SPRING); spring_lo[n + "_flap"] = float(REST_B + tB / K_SPRING)
    print("%s: flap %.3f kg, gravity torque about A %.3f Nm, about B %.3f Nm -> dead-band edges A %.3f, B %.3f rad" % (n, masses[n], tA, tB, spring_lo[n], spring_lo[n + "_flap"]))
# ---- XML
f5 = lambda v: " ".join(f"{x:.5f}" for x in v)
L = ["<mujoco>", "  <compiler angle=\"radian\"/>", "  <asset>", '    <texture name="bits_kit_fold_tex" type="2d" file="bits_kit_fold_colors.png"/>', '    <material name="bits_kit_fold_mat" texture="bits_kit_fold_tex" specular="0.05" shininess="0.1"/>',
     '    <material name="bits_kit_fold_cloth" rgba="0.06 0.07 0.12 1" specular="0.05" shininess="0.1"/>', '    <material name="bits_kit_fold_rim" rgba="0.10 0.20 0.60 1" specular="0.05" shininess="0.1"/>',
     '    <material name="bits_kit_fold_bit" rgba="0.62 0.63 0.66 1" specular="0.6" shininess="0.5"/>', '    <material name="bits_kit_fold_holder" rgba="0.08 0.08 0.10 1" specular="0.05" shininess="0.1"/>'] + [f'    <mesh name="bits_kit_{n}_visual" file="bits_kit_{n}_visual.obj"/>' for n in part_names]
L += ["  </asset>", "  <default>", '    <default class="bits_kit_fold_col"><geom type="box" group="3" condim="4" friction="0.8 0.005 0.0001" solref="0.02 1"/></default>',
      '    <default class="bits_kit_fold_vis"><geom group="2" contype="0" conaffinity="0" mass="0"/></default>', "  </default>", "  <worldbody>"]
def body_geoms(n, ind):
    out = [f'{ind}<geom class="bits_kit_fold_vis" type="mesh" mesh="bits_kit_{n}_visual" material="bits_kit_fold_mat"/>']
    s = geoms[n][0][1]; out.append(f'{ind}<geom class="bits_kit_fold_vis" type="box" pos="{f5(s["pos"])}" size="{f5([s["size"][0], s["size"][1], s["size"][2] * 0.9])}" material="{"bits_kit_fold_rim" if n.endswith("strip") else "bits_kit_fold_cloth"}"/>')
    out += [f'{ind}<geom name="bits_kit_{n}_{k}" class="bits_kit_fold_col" pos="{f5(d["pos"])}" size="{f5(d["size"])}" mass="{d["mass"]:.5f}"/>' for k, d in geoms[n]]
    # v3 (2026-10-30): visual bit rows on the bit-column content boxes (the scan texture there is saturated white). Rows of BIT_PITCH-spaced
    # cylinders lying along the column (y), BIT_ROW_PITCH apart, tops BIT_LIFT above the box top, one dark elastic holder strip across each row.
    for k, d in geoms[n]:
        if not k.startswith("content") or "bit column" not in contents[n][int(k[7:])]["what"]: continue
        (cx, cy, cz), (sx, sy, sz) = d["pos"], d["size"]; top = cz + sz; n_rows = int(2 * sy // BIT_ROW_PITCH); n_bits = int((2 * sx - BIT_LEN / 4) // BIT_PITCH) + 1
        for r in range(n_rows):
            ry = cy + (r - (n_rows - 1) / 2) * (2 * sy / n_rows)
            out.append(f'{ind}<geom class="bits_kit_fold_vis" type="box" pos="{f5([cx, ry, top + BIT_LIFT - 0.0002])}" size="{f5([sx, 0.004, 0.0005])}" material="bits_kit_fold_holder"/>')
            out += [f'{ind}<geom class="bits_kit_fold_vis" type="cylinder" pos="{f5([cx + (j - (n_bits - 1) / 2) * BIT_PITCH, ry, top + BIT_LIFT - BIT_R])}" size="{BIT_R:.5f} {BIT_LEN / 2:.5f}" quat="0.707107 0.707107 0 0" material="bits_kit_fold_bit"/>' for j in range(n_bits)]
    return out
L += ['    <body name="bits_kit_centre" pos="0 0 0.001">', "      <freejoint/>"] + body_geoms("centre", "      ")
for n, axis in (("left", "0 1 0"), ("right", "0 -1 0")):
    relA = origin[n + "_strip"] - origin["centre"]; relB = origin[n] - origin[n + "_strip"]
    L += [f'      <body name="bits_kit_{n}_strip" pos="{f5(relA)}">', f'        <joint name="bits_kit_hinge_{n}" type="hinge" axis="{axis}" range="-0.1 3.4" damping="{DAMPING}" frictionloss="{FRICTIONLOSS}" armature="1e-5"/>'] + body_geoms(n + "_strip", "        ")
    L += [f'        <body name="bits_kit_{n}" pos="{f5(relB)}">', f'          <joint name="bits_kit_hinge_{n}_flap" type="hinge" axis="{axis}" range="-0.5 3.4" damping="{DAMPING}" frictionloss="{FRICTIONLOSS}" armature="1e-5"/>'] + body_geoms(n, "          ") + ["        </body>", "      </body>"]
L += ["    </body>", "  </worldbody>", "  <tendon>"]  # flap vs centre are grandparent-grandchild: collided by the normal pipeline (only direct parent-child is filtered)
L += [f'    <fixed name="bits_kit_spring_{n}" springlength="{spring_lo[n]:.4f} 4" stiffness="{K_SPRING}"><joint joint="bits_kit_hinge_{n}" coef="1"/></fixed>' for n in ("left", "left_flap", "right", "right_flap")]
L += ["  </tendon>", "</mujoco>"]
(A / "bits_kit_fold.xml").write_text("\n".join(L) + "\n")
meta = dict(frame="aligned PCA frame of fold_profile.py: x = long axis, z = 0 at the table; centre body origin = centre-panel footprint centre", aligned_centre_table_xy=c.tolist(), aligned_R_rows=R.tolist(),
            fold_x_m=FOLDS, edges_x_m=EDGES, panel_widths_mm=(1e3 * widths).round(1).tolist(), kit_width_mm=2e3 * HALF_W, slab_thickness_mm=1e3 * SLAB, hinge_height_mm=1e3 * SLAB,
            hinge_strip=dict(width_mm=1e3 * STRIP, thickness_mm=1e3 * STRIP_T, mass_kg=STRIP_MASS, flap_slab_gap_mm=1e3 * GAP, note="strip body between hinge A (fold line, centre-to-strip) and hinge B (strip-to-flap); flap slabs start at fold + strip + gap"),
            body_origins_aligned_m={k: v.tolist() for k, v in origin.items()}, masses_kg={**{k: round(v, 4) for k, v in masses.items()}, "total": MASS},
            content_boxes={n: [dict(x_mm=(1e3 * np.array(b["x"])).round(1).tolist(), y_mm=(1e3 * np.array(b["y"])).round(1).tolist(), h_mm=round(1e3 * b["h"], 1), what=b["what"]) for b in contents[n]] for n in names},
            joints=dict(left=dict(A="bits_kit_hinge_left", B="bits_kit_hinge_left_flap", axis=[0, 1, 0]), right=dict(A="bits_kit_hinge_right", B="bits_kit_hinge_right_flap", axis=[0, -1, 0]), range_rad=dict(A=[-0.1, 3.4], B=[-0.5, 3.4]),
                        sign="increasing angle lifts the flap toward the centre on both joints; flap tilt = A + B; closed = A + B ~ pi with the flap flat on the stack",
                        damping=DAMPING, frictionloss=FRICTIONLOSS, armature=1e-5),
            spring=dict(kind="fixed tendon dead-band spring on each of the four hinges: torque K*(lo - q) for q < lo, zero above", stiffness_Nm_per_rad=K_SPRING, deadband_lo_rad={k: round(v, 4) for k, v in spring_lo.items()}, design_rest_deg=dict(A=6.0, B=6.0, flap_tilt=12.0)),
            closing_order="right (card flap, inner) first, left (two bit columns, outer) on top", faces_per_body=faces_kept, texture="bits_kit_fold_colors.png (scan colours), flat cloth colour on the underside boxes",
            bit_rows=dict(kind="visual only (group 2, no contact, no mass): cylinders lying along y on every bit-column content box, rows along y, one holder strip per row", bit_diameter_mm=2e3 * BIT_R, bit_length_mm=1e3 * BIT_LEN, pitch_mm=1e3 * BIT_PITCH, row_pitch_mm=1e3 * BIT_ROW_PITCH, top_above_box_mm=1e3 * BIT_LIFT, materials=dict(bit="bits_kit_fold_bit rgba 0.62 0.63 0.66 specular 0.6 shininess 0.5", holder="bits_kit_fold_holder rgba 0.08 0.08 0.10")))
json.dump(meta, open(A / "bits_kit_fold_meta.json", "w"), indent=1); print("wrote", A / "bits_kit_fold.xml")

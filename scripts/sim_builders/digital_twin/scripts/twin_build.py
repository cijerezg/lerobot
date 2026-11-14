"""Assemble twin/scene_twin.xml: the room (room/room.xml) + robot + aligned ext camera, with every desk object as a free body at the
poses of twin/layout.json. Package objects (clipper case + two clippers, bottle, basket, hinged headphone case) are pulled from the
package scenes by element name; the scanned bits kit and the real-label pill bottle are copied into twin/assets/<name>/ (the kit with
the x4 texture of twin_make_assets.py), the generated ROAR / navy bottles and the hex bit live there already; everything is referenced
relative to the package scenes dir (one assetdir per model, and the robot include needs it to be the package scenes dir). v2 adds six
loose bits, the clear bits box, the bit holder and four basket contents as primitive free bodies. v4 adds the package ep3 tape roll and
paper cup (photoreal objects, pulled by body name) at free desk spots, and paints the kit's product card white in the texture.
Run from the LeRobot root: uv run python lerobot/scripts/sim_builders/digital_twin/scripts/twin_build.py"""
import json, pathlib, shutil, math, copy, numpy as np, cv2
import xml.etree.ElementTree as ET

ROOT = pathlib.Path("lerobot/scripts/sim_builders/digital_twin"); TWIN = ROOT / "twin"; SCENES = pathlib.Path("lerobot/src/lerobot/robots/rebot_b601_sim/scenes")
SRC = pathlib.Path("assets/sim_sources/digital_twin")                           # scanned / downloaded object assets (outside the repo)
ASSETDIR = "../../../../src/lerobot/robots/rebot_b601_sim/scenes"               # from twin/ to the package scenes dir
TWIN_FROM_SCENES = "../../../../../scripts/sim_builders/digital_twin/twin/assets"  # from the package scenes dir back to twin/assets
layout = json.loads((TWIN / "layout.json").read_text()); obj = layout["objects"]

def quat_yaw(deg):
    h = math.radians(deg) / 2; return f"{math.cos(h):.6f} 0 0 {math.sin(h):.6f}"
def ang(name): return obj[name]["axis_angle_deg"]
xy = {k: v["table_xy"] for k, v in obj.items()}

# ---- poses (table frame). yaw conventions, see TWIN_REPORT.md
poses = {}
poses["bits_kit"] = (xy["bits_kit"], 0.003, ang("bits_kit") + 180.0)                      # asset long side = +x; +180 puts the pocket end to the image right
hp = ang("headphone_case"); poses["headphone_case"] = (xy["headphone_case"], 0.022, 242.18 + (hp - 180.0))  # close_lid: lid lies at -x of the base for body yaw 242.18
poses["roar_bottle"] = (xy["roar_bottle"], 0.002, 290.0); poses["pill_bottle"] = (xy["pill_bottle"], 0.002, 110.0); poses["navy_bottle"] = (xy["navy_bottle"], 0.002, 0.0)  # labels face +y / -x, toward the wrist camera at f300 (pill label at body +x; ROAR label at body -x)
for i in range(1, 7): poses[f"bit_{i}"] = (xy[f"bit_{i}"], 0.003 + 0.001, ang(f"bit_{i}"))       # hex across flats 6 mm, flat face down
poses["bits_box"] = (xy["bits_box"], 0.0125 + 0.002, ang("bits_box")); poses["bit_holder"] = (xy["bit_holder"], 0.0075 + 0.002, ang("bit_holder"))
poses["basket"] = (xy["basket"], 0.002, ang("basket"))                                   # basket long side (0.29) = body x
ca = ang("clipper_case"); poses["clipper_case"] = (xy["clipper_case"], 0.002, ca - 90.0 + 180.0)  # case long side = body y (clippers_only yaw 87.5 -> long side along x); +180 puts the dark hinge half to the image left
poses["clipper_large"] = (xy["clipper_large"], 0.0071, ang("clipper_large"))
cx, cy = xy["clipper_case"]; ux, uy = math.cos(math.radians(ca)), math.sin(math.radians(ca))
poses["clipper_small"] = ([cx + 0.02 * ux, cy + 0.02 * uy], 0.017, ca)                   # in the slot nearer the image right, along the long side
# basket contents: offsets in the basket frame (floor top z = 0.012, inner 0.266 x 0.216), placed from the top frame (upright bottle at the
# image bottom-right of the basket = +x +y, lying can at the back = -x, cards in the middle-left)
bx, by = xy["basket"]; bya = math.radians(ang("basket")); cb, sb = math.cos(bya), math.sin(bya)
def in_basket(dx, dy, z, yaw): return ([bx + cb * dx - sb * dy, by + sb * dx + cb * dy], 0.012 + z + 0.002, ang("basket") + yaw)
poses["basket_bottle"] = in_basket(0.06, 0.06, 0.0, 0.0); poses["basket_can"] = in_basket(-0.08, 0.02, 0.030, 0.0)
poses["basket_card"] = in_basket(0.0, -0.055, 0.003, 30.0); poses["basket_card_white"] = in_basket(0.0, -0.055, 0.006 + 0.001 + 0.001, 10.0)  # white card on top of the navy one
# v4: added objects (not in the real frame), package ep3 tape roll (75 x 30 mm, lying flat, collider half-height 0.015) and paper cup (80 x 110, half-height 0.055)
poses["tape_roll"] = (xy["tape_roll"], 0.015 + 0.002, 0.0); poses["paper_cup"] = (xy["paper_cup"], 0.055 + 0.002, 0.0)

# ---- source scenes
def load(name): return ET.parse(SCENES / name / "scene.xml").getroot()
room_scene = ET.parse(ROOT / "room" / "clippers_only_room" / "scene.xml").getroot()
clip, bb, cl, ep3 = load("clippers_only"), load("bottle_basket"), load("close_lid"), load("ep3")
def pick_assets(root, keep):
    return [copy.deepcopy(e) for e in root.find("asset") if keep(e.get("name", ""), e)]
def pick_defaults(root, names):
    return [copy.deepcopy(e) for e in root.find("default") if e.get("class") in names]
def body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name: return copy.deepcopy(b)
    raise KeyError(name)

assets, defaults, bodies = [], [], []
assets += pick_assets(room_scene, lambda n, e: n.startswith("table_top") or e.get("type") == "skybox")
assets += pick_assets(clip, lambda n, e: "clipper" in n or n.startswith("case_"))
defaults += pick_defaults(clip, {"clipper_case_col"})
assets += pick_assets(bb, lambda n, e: not (n.startswith("table_top") or n in {"wood", "floor_mat"} or e.get("type") == "skybox"))
defaults += pick_defaults(bb, {"basket_col", "basket_vis"})
assets += pick_assets(cl, lambda n, e: n.startswith("hinged_nail_ep2") or n in {"clutter_fabric", "clutter_lining", "clutter_black", "clutter_rubber", "clutter_gray", "clutter_zipper", "detail_fabric", "detail_lining"})
assets += pick_assets(ep3, lambda n, e: n.startswith("nail_ep3_tape_roll") or n.startswith("nail_ep3_paper_cup") or n in {"detail_cup_print", "clutter_tape", "clutter_core", "clutter_inner", "clutter_cap", "clutter_white", "clutter_cup_print"})

# twin assets: (body name, asset name, source dir, dest dir name) -> copy if needed + re-point files
KIT_TEX = "bits_kit_fold_colors_gain.png"; TA = TWIN / "assets"
# ---- kit texture (v3, 2026-10-30): per-channel gain BGR (4.0, 2.9, 3.2) (twin_make_assets.py: real rim 142 62 31 / scan rim 35.5 21.3 9.8) on the
# saturated (blue) texels only, a flat x1.2 on the neutral ones (black fabric -> ~30 30 30, product card), blended by the source saturation;
# the bit-column faces (content boxes "bit column", mesh UVs, texture row = (1 - v) * H, see paint()) are painted the dark cloth
# colour because the scan is saturated white there (the bits themselves are visual geoms in bits_kit_fold.xml).
KIT_SRC = SRC / "assets" / "bits_kit_fold"; kmeta = json.loads((KIT_SRC / "bits_kit_fold_meta.json").read_text())
ksrc = cv2.imread(str(KIT_SRC / "bits_kit_fold_colors.png")); KH, KW = ksrc.shape[:2]
ksat = cv2.cvtColor(ksrc, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32); kw = np.clip((ksat - 40.0) / 80.0, 0, 1)[..., None]
kout = np.clip(ksrc.astype(np.float32) * (1.2 + kw * (np.array([4.0, 2.9, 3.2], np.float32) - 1.2)), 0, 255)
kbits = np.zeros((KH, KW), np.uint8)
def paint(mask, tris, val):
    """the kit texture is a per-face atlas (every face owns a 2 x 2 texel block, median UV triangle area 0.5 px): fill each face's UV bbox.
    MuJoCo samples OBJ textures at image row (1 - v) * H (checked 2026-10-30 with a red/green row test: rows 453-623 land on the LEFT flap),
    so the v3 paint (cv2.fillPoly at row v * H, dotted and on the wrong panel) is replaced -- v4"""
    lo = np.floor(tris.min(1) * [KW, KH]).astype(int); hi = np.floor(tris.max(1) * [KW, KH]).astype(int)
    for (x0, v0), (x1, v1) in zip(lo, hi): mask[KH - 1 - v1:KH - v0, x0:x1 + 1] = val
for panel, ztop in (("left", 0.001), ("centre", 0.008)):   # ztop = slab top + 1 mm in the body frame
    lines = (KIT_SRC / f"bits_kit_{panel}_visual.obj").read_text().splitlines()
    V = np.array([l.split()[1:4] for l in lines if l.startswith("v ")], float); VT = np.array([l.split()[1:3] for l in lines if l.startswith("vt ")], float)
    cen = V.reshape(-1, 3, 3).mean(1); o = kmeta["body_origins_aligned_m"][panel]
    for bx in kmeta["content_boxes"][panel]:
        if "bit column" not in bx["what"]: continue
        (x0, x1), (y0, y1) = np.array(bx["x_mm"]) / 1e3 - o[0], np.array(bx["y_mm"]) / 1e3 - o[1]
        sel = (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 1] > y0) & (cen[:, 1] < y1) & (cen[:, 2] > ztop)
        paint(kbits, VT.reshape(-1, 3, 2)[sel], 1)
kout[kbits.astype(bool)] = (30, 30, 32)
# v4: the right flap's product card (content box "card pocket with product card", every face of the flap mesh in the box's xy footprint) is painted
# uniform white BGR 235 with a 2 mm border at 205; the flap's other neutral-bright texels (saturated scan, source HSV S < 70 and V > 110) get the cloth colour
lines = (KIT_SRC / "bits_kit_right_visual.obj").read_text().splitlines()
V = np.array([l.split()[1:4] for l in lines if l.startswith("v ")], float); VT = np.array([l.split()[1:3] for l in lines if l.startswith("vt ")], float)
cen = V.reshape(-1, 3, 3).mean(1); o = kmeta["body_origins_aligned_m"]["right"]; bx = next(b for b in kmeta["content_boxes"]["right"] if "card" in b["what"])
(x0, x1), (y0, y1) = np.array(bx["x_mm"]) / 1e3 - o[0], np.array(bx["y_mm"]) / 1e3 - o[1]
sel = (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 1] > y0) & (cen[:, 1] < y1) & (cen[:, 2] > -0.003)   # whole footprint: the scanned pocket surface dips to slab level (z 0) between streaks, so no top-only threshold
edge = np.minimum(np.minimum(cen[:, 0] - x0, x1 - cen[:, 0]), np.minimum(cen[:, 1] - y0, y1 - cen[:, 1])) < 0.002
kflap = np.zeros((KH, KW), np.uint8); kcard = np.zeros((KH, KW), np.uint8)
paint(kflap, VT.reshape(-1, 3, 2), 1); paint(kcard, VT.reshape(-1, 3, 2)[sel & ~edge], 1); paint(kcard, VT.reshape(-1, 3, 2)[sel & edge], 2)
khsv = cv2.cvtColor(ksrc, cv2.COLOR_BGR2HSV); kstray = kflap.astype(bool) & (kcard == 0) & (khsv[..., 1] < 70) & (khsv[..., 2] > 110)
kout[kstray] = (30, 30, 32); kout[kcard == 1] = (235, 235, 235); kout[kcard == 2] = (205, 205, 205)
print(f"kit card: {int((kcard == 1).sum())} white + {int((kcard == 2).sum())} border texels ({int(sel.sum())} faces); right flap stray bright texels -> cloth: {int(kstray.sum())}; flap texels > 150 outside the card now: {int(((kout.max(-1) > 150) & kflap.astype(bool) & (kcard == 0)).sum())}")
(TA / "bits_kit_fold").mkdir(parents=True, exist_ok=True); cv2.imwrite(str(TA / "bits_kit_fold" / KIT_TEX), kout.astype(np.uint8))
print(f"kit texture: bit-column texels {int(kbits.sum())} painted cloth; texels > 150: {(kout.max(-1) > 150).mean() * 100:.1f} % (source x gain would give {(np.clip(ksrc * np.array([4.0, 2.9, 3.2]), 0, 255).max(-1) > 150).mean() * 100:.1f} %)")
tendons = []
for bname, aname, src, dname in [("bits_kit", "bits_kit_fold", SRC / "assets" / "bits_kit_fold", "bits_kit_fold"), ("pill_bottle", "pill_bottle_real", SRC / "assets_dl" / "pill_bottle_real", "pill_bottle_real"),
                                 ("roar_bottle", "roar_bottle", TA / "roar_bottle", "roar_bottle"), ("navy_bottle", "navy_bottle", TA / "navy_bottle", "navy_bottle")]:
    dst = TA / dname; dst.mkdir(parents=True, exist_ok=True)
    ax = ET.parse(src / f"{aname}.xml").getroot()
    for e in ax.find("asset"):
        if e.get("file"):
            fn = KIT_TEX if (aname == "bits_kit_fold" and e.tag == "texture") else e.get("file")
            if src != dst and fn == e.get("file"): shutil.copy2(src / fn, dst / fn)
            e.set("file", f"{TWIN_FROM_SCENES}/{dname}/{fn}")
        assets.append(e)
    if src != dst:
        for mtl in src.glob("*.mtl"): shutil.copy2(mtl, dst / mtl.name)
        shutil.copy2(src / f"{aname}_meta.json", dst / f"{aname}_meta.json")
    defaults += list(ax.find("default")); b = body(ax, "bits_kit_centre" if aname == "bits_kit_fold" else aname); b.set("name", bname); bodies.append(b)
    if ax.find("tendon") is not None: tendons += list(ax.find("tendon"))

# v2 primitives
assets.append(ET.Element("mesh", name="bit_hex", file=f"{TWIN_FROM_SCENES}/bits/bit_hex.obj"))
def prim(name, geoms):
    b = ET.Element("body", name=name)
    for g in geoms: ET.SubElement(b, "geom", **g)
    return b
BIT_RGBA = {1: "0.55 0.55 0.58 1", 2: "0.55 0.55 0.58 1", 3: "0.80 0.15 0.12 1", 4: "0.90 0.80 0.20 1", 5: "0.55 0.55 0.58 1", 6: "0.55 0.55 0.58 1"}
for i in range(1, 7): bodies.append(prim(f"bit_{i}", [dict(type="mesh", mesh="bit_hex", mass="0.006", rgba=BIT_RGBA[i], condim="4", friction="0.6 0.005 0.0001")]))
CLEAR = "0.9 0.9 0.95 0.35"; VIS = dict(contype="0", conaffinity="0", mass="0", group="2")
bodies.append(prim("bits_box", [dict(type="box", size="0.06 0.04 0.0125", mass="0.08", rgba=CLEAR, group="3", condim="4", friction="0.6 0.005 0.0001"),
                                dict(type="box", size="0.055 0.035 0.004", pos="0 0 -0.006", rgba="0.5 0.5 0.52 1", **VIS),
                                dict(type="box", size="0.012 0.008 0.003", pos="-0.035 0.018 0.001", rgba="0.8 0.15 0.12 1", **VIS),
                                dict(type="box", size="0.012 0.008 0.003", pos="0.03 0.018 0.001", rgba="0.8 0.15 0.12 1", **VIS),
                                dict(type="box", size="0.012 0.008 0.003", pos="-0.03 -0.012 0.001", rgba="0.9 0.8 0.2 1", **VIS),
                                dict(type="box", size="0.015 0.008 0.003", pos="0.02 -0.012 0.001", rgba="0.25 0.45 0.8 1", **VIS),
                                dict(type="box", size="0.010 0.006 0.003", pos="0.0 0.012 0.001", rgba="0.15 0.15 0.15 1", **VIS)]))
bodies.append(prim("bit_holder", [dict(type="box", size="0.03 0.0125 0.0075", mass="0.04", rgba="0.75 0.76 0.77 1", condim="4", friction="0.6 0.005 0.0001")]))
# basket contents (all free bodies)
bodies.append(prim("basket_bottle", [dict(type="cylinder", size="0.032 0.1", pos="0 0 0.1", mass="0.5", rgba="0.08 0.10 0.20 1", condim="4", friction="0.6 0.005 0.0001"),
                                     dict(type="cylinder", size="0.0325 0.035", pos="0 0 0.095", rgba="0.92 0.92 0.90 1", **VIS),
                                     dict(type="cylinder", size="0.0327 0.012", pos="0 0 0.072", rgba="0.25 0.55 0.35 1", **VIS),
                                     dict(type="cylinder", size="0.024 0.01", pos="0 0 0.21", rgba="0.08 0.10 0.20 1", **VIS)]))
bodies.append(prim("basket_can", [dict(type="capsule", size="0.030 0.045", quat="0.707107 0.707107 0 0", mass="0.35", rgba="0.78 0.78 0.80 0", condim="4", friction="0.6 0.005 0.0001"),  # capsule collider (cylinder-on-box rests 6 mm deep in MuJoCo)
                                  dict(type="cylinder", size="0.030 0.065", quat="0.707107 0.707107 0 0", rgba="0.78 0.78 0.80 1", **VIS),
                                  dict(type="cylinder", size="0.0302 0.03", quat="0.707107 0.707107 0 0", rgba="0.75 0.15 0.15 1", **VIS)]))
bodies.append(prim("basket_card", [dict(type="box", size="0.045 0.03 0.003", mass="0.03", rgba="0.10 0.12 0.28 1", condim="4", friction="0.6 0.005 0.0001")]))
bodies.append(prim("basket_card_white", [dict(type="box", size="0.04 0.028 0.001", mass="0.01", rgba="0.95 0.95 0.93 1", condim="4", friction="0.6 0.005 0.0001")]))

bodies += [body(clip, "clipper_case"), body(clip, "clipper_large"), body(clip, "clipper_small"), body(bb, "basket"), body(cl, "headphone_case")]
for bn, mass in (("tape_roll", "0.03"), ("paper_cup", "0.30")):   # v4: ep3 photoreal objects; the group-5 photo-projection geom is dropped, mass set on the collider
    b = body(ep3, bn)
    for g in [g for g in b.findall("geom") if g.get("group") == "5"]: b.remove(g)
    b.find(f"geom[@name='{bn}']").set("mass", mass); bodies.append(b)
for b in bodies:
    (x, y), z, yaw = poses[b.get("name")]; b.set("pos", f"{x:.4f} {y:.4f} {z:.4f}"); b.set("quat", quat_yaw(yaw))
    if b.find("freejoint") is None: b.insert(0, ET.Element("freejoint"))
    ro = layout.get("rest_overrides", {}).get(b.get("name"))
    if ro: b.set("pos", " ".join(f"{v:.4f}" for v in ro["pos"])); b.set("quat", " ".join(f"{v:.5f}" for v in ro["quat"]))

# ---- assemble
seen = set(); uniq = []
for e in assets:
    k = (e.tag, e.get("name")) if e.get("name") else (e.tag, e.get("type"), e.get("rgb1"))
    if k not in seen: seen.add(k); uniq.append(e)
mj = ET.Element("mujoco", model="scene_twin")
ET.SubElement(mj, "compiler", angle="radian", assetdir=ASSETDIR, autolimits="true")
ET.SubElement(mj, "include", file=f"{ASSETDIR}/rebot_b601_dm_photo.xml")
ET.SubElement(mj, "include", file="appearance/lights_visual.xml")   # appearance pass 2026-10-30 (twin/appearance/APPEARANCE_REPORT.md)
ET.SubElement(mj, "include", file="appearance/materials.xml")       # room_floor_mat, referenced by room/room.xml
a = ET.SubElement(mj, "asset"); a.extend(uniq)
for e in a:
    if e.tag == "texture" and e.get("name") == "table_top_tex": e.set("file", "../../../../../scripts/sim_builders/digital_twin/twin/appearance/table_top_fit.png")
    if e.tag == "material" and e.get("name") == "table_top": e.set("specular", "0.2"); e.set("shininess", "0.3")   # v3: mild desk sheen (was 0 / 0; 0.25 moved the top-view desk mean +3.3, over the 3-level budget)
dft = ET.SubElement(mj, "default"); dft.extend(defaults)
wb = ET.SubElement(mj, "worldbody")
ET.SubElement(wb, "include", file="../room/room.xml")
rwb = room_scene.find("worldbody")
wb.append(copy.deepcopy(next(g for g in rwb.findall("geom") if g.get("name") == "table_contact")))
wb.append(copy.deepcopy(rwb.find("camera")))
wb.extend(bodies)
ET.SubElement(wb, "include", file="appearance/lights.xml")
if tendons: ET.SubElement(mj, "tendon").extend(tendons)
ET.indent(mj, space="  "); (TWIN / "scene_twin.xml").write_text('<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(mj, encoding="unicode") + "\n")
layout["assets"] = {"bits_kit": "assets/bits_kit_fold (tri-fold scan + visual bit rows), texture twin/assets/bits_kit_fold/bits_kit_fold_colors_gain.png: x(4.0, 2.9, 3.2) BGR on blue texels, x1.2 on neutral, bit columns painted cloth", "pill_bottle": "assets_dl/pill_bottle_real (50 x 100 mm, Trader Joe's Vitamin D label)",
                    "roar_bottle": "twin/assets/roar_bottle (generated, label from wrist f300)", "navy_bottle": "twin/assets/navy_bottle (generated navy cylinder 70 x 200, blue cap; pump bottle rejected)",
                    "bit_1..6": "twin/assets/bits/bit_hex.obj 25 x 6 mm, 6 g", "bits_box": "box 120 x 80 x 25 mm clear + grey tray + coloured bits (visual), 80 g", "bit_holder": "box 60 x 25 x 15 mm light grey, 40 g",
                    "basket_bottle": "cylinder 64 x 200 mm navy + white/green label band, 0.5 kg", "basket_can": "cylinder 60 x 130 mm silver lying along basket y (capsule collider), 0.35 kg", "basket_card": "box 90 x 60 x 6 mm navy, 30 g", "basket_card_white": "box 80 x 56 x 2 mm white on the navy card, 10 g",
                    "basket, clipper_case, clipper_large, clipper_small, headphone_case": "package scenes, as v1",
                    "tape_roll": "package ep3 tape_roll (photoreal: beige tape ring 75 x 30 mm lying flat, hollow core), 0.03 kg, added v4 (not in the real frame)",
                    "paper_cup": "package ep3 paper_cup (photoreal: tapered cup 80 x 110 mm, cup_print.png), 0.30 kg, added v4 (not in the real frame)"}
layout["poses"] = {k: {"pos": [round(v[0][0], 4), round(v[0][1], 4), v[1]], "yaw_deg": round(v[2] % 360.0, 1)} for k, v in poses.items()}
for k, ro in layout.get("rest_overrides", {}).items(): layout["poses"][k]["start_pose_from_rest_override"] = ro["pos"]
(TWIN / "layout.json").write_text(json.dumps(layout, indent=1))
print("wrote", TWIN / "scene_twin.xml", "bodies:", [b.get("name") for b in bodies]); print({k: v for k, v in layout["poses"].items()})

"""Build scenes/close_lid: the open headphone case (ep2 hinged meshes) with a real hinge joint on the lid.

Reads: clippers_only/scene.xml (visual/lighting/cameras/table), ep2/scene.xml (which material each hinged part uses),
lerobot/scripts/sim_builders/photoreal_headphone_hinge/hinge_metadata.json (pivot, axis, lid/hinge part lists).
Writes: scenes/close_lid/scene.xml, scenes/close_lid/objects.json, copies clippers_only/background_top.png.

Frames. The hinged meshes are stored in the ep2 body frame = the builder's canonical frame rotated by ANGLE about z.
Canonical: base shell centred at the origin, hinge line along +x at y = HY + GAP/2, lid shell mirrored beyond it (+y),
shell bottom at z = -HZ (table), rim at z = 0.58 HZ. Body quat is chosen so canonical +y (base -> lid) points to
world -x (lid toward the robot, as in the 2026-10-21 recordings) and the hinge line runs along world y.
"""
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

WS = Path("/home/user/Documents/Research/RL/LeRobot")
SCENES = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes"
OUT = SCENES / "close_lid"
META = json.load(open(WS / "lerobot/scripts/sim_builders/photoreal_headphone_hinge/hinge_metadata.json"))["nail_ep2"]
EP = "nail_ep2"

HINGE_X, CASE_Y = 0.34, 0.0174  # world: hinge line x, lateral centre (same lateral spot as the clippers_only case)
HX, HY, HZ = META["size"]  # canonical half extents of one shell: along hinge, across hinge, height
GAP = 0.007
ANGLE = META["angle"]
PIVOT_LOCAL = np.array(META["pivot_local"])
AXIS_LOCAL = np.array(META["axis_local"])
LID_PARTS = [int(n[-2:]) for n in META["lid_geoms"]]
HINGE_PARTS = [int(n[-2:]) for n in META["hinge_geoms"]]
BASE_PARTS = [i for i in range(META["parts"]) if i not in LID_PARTS + HINGE_PARTS]
CLOSED = float(np.pi)


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_z(a):
    return f"{np.cos(a / 2):.6f} 0 0 {np.sin(a / 2):.6f}"


def fmt(v):
    return " ".join(f"{x:.5f}" for x in v)


R_LOCAL = rz(ANGLE)  # canonical -> body-local
PSI = np.pi / 2 - ANGLE  # body-local -> world yaw: canonical +y -> world -x, canonical +x -> world +y
ORIGIN = np.array([HINGE_X + (HY + GAP / 2), CASE_Y, HZ])  # base shell centre: hinge is HY+GAP/2 toward the robot

# materials per part, from the ep2 layout
materials = {}
for g in ET.parse(SCENES / "ep2/scene.xml").getroot().iter("geom"):
    if g.get("mesh", "").startswith(f"hinged_{EP}_"):
        materials[g.get("mesh")] = g.get("material")

src = ET.parse(SCENES / "clippers_only/scene.xml").getroot()
src_asset = src.find("asset")
src_world = src.find("worldbody")

root = ET.Element("mujoco", model="scene_close_lid")
ET.SubElement(root, "compiler", angle="radian", assetdir="..", autolimits="true")
ET.SubElement(root, "include", file="../rebot_b601_dm_photo.xml")
root.append(src.find("visual"))
asset = ET.SubElement(root, "asset")
keep_tex = {"table_top_tex", "detail_fabric", "detail_lining"}
keep_mat = {"table_top", "wood", "floor_mat", "clutter_fabric", "clutter_lining", "clutter_black", "clutter_zipper", "clutter_gray", "clutter_rubber"}
for e in src_asset:
    if e.tag == "texture" and (e.get("name") in keep_tex or e.get("type") == "skybox"):
        asset.append(e)
    if e.tag == "material" and e.get("name") in keep_mat:
        asset.append(e)
for i in range(META["parts"]):
    ET.SubElement(asset, "mesh", name=f"hinged_{EP}_{i:02d}", file=f"assets/photoreal/headphones/hinged_{EP}_{i:02d}.obj", inertia="shell")

world = ET.SubElement(root, "worldbody")
for e in src_world:
    if e.tag in ("geom", "camera") or (e.tag == "light"):
        world.append(e)
# keep order: floor/table/camera first, lights last (as in clippers_only); the case body goes before the lights
lights = [e for e in world if e.tag == "light"]
for e in lights:
    world.remove(e)

world.append(ET.Comment(
    " headphone_case: the loaded base half (headphones inside), fixed to the table so a push on the lid cannot slide it. "
    "headphone_lid: the empty half on joint lid_hinge. lid_hinge = 0 is the recorded flat-open pose (lid resting on the table "
    f"toward the robot); positive = closing; CLOSED = {CLOSED:.4f} rad (pi): the lid rim meets the base rim exactly, both "
    "shells being mirror images about the hinge line at rim height. Score 'closed' as lid_hinge > 2.9 rad. "
    "Gravity holds the lid open below ~1.83 rad (lid centre of mass passes over the hinge there) and shut above it; "
    "the upper joint limit is the stop at closed. "))
case = ET.SubElement(world, "body", name="headphone_case", pos=fmt(ORIGIN), quat=quat_z(PSI))
# base collision: one solid block up to the rim plus a cap over the earcups (the fingers must not sink into the base)
def box(parent, name, c_can, half, mass, **kw):
    c = R_LOCAL @ np.asarray(c_can)
    if kw.pop("lid", False):
        c = c - PIVOT_LOCAL
    ET.SubElement(parent, "geom", name=name, type="box", group="3", size=fmt(half), pos=fmt(c), quat=quat_z(ANGLE), mass=f"{mass:.4f}", condim="4", friction="0.8 0.005 0.0001", **kw)

box(case, "headphone_case_block_a", (0, 0, -0.0042), (0.088, 0.062, 0.0158), 0.07)  # two overlapping blocks: rounded-corner footprint
box(case, "headphone_case_block_b", (0, 0, -0.0042), (0.080, 0.070, 0.0158), 0.05)
box(case, "headphone_case_earcups", (0, -0.0015, 0.0151), (0.062, 0.031, 0.0035), 0.03)
for i in BASE_PARTS + HINGE_PARTS:
    n = f"hinged_{EP}_{i:02d}"
    ET.SubElement(case, "geom", name=n, type="mesh", mesh=n, material=materials[n], group="2", contype="0", conaffinity="0", mass="0")

lid = ET.SubElement(case, "body", name="headphone_lid", pos=fmt(PIVOT_LOCAL))
ET.SubElement(lid, "joint", name="lid_hinge", type="hinge", axis=fmt(AXIS_LOCAL), range=f"0 {CLOSED:.5f}", damping="0.02", frictionloss="0.002", solreflimit="0.004 1", solimplimit="0.95 0.99 0.001 0.5 2")
CY = 2 * HY + GAP  # canonical lid centre y
# lid collision: tapered-shell approximation, floor at the bottom (76 % footprint) and four rim walls (outer face at the shell wall)
box(lid, "headphone_lid_floor", (0, CY, -0.0170), (0.070, 0.057, 0.0025), 0.024, lid=True)
box(lid, "headphone_lid_wall_xp", (HX - 0.0035, CY, -0.0002), (0.0035, 0.062, 0.0118), 0.009, lid=True)
box(lid, "headphone_lid_wall_xn", (-(HX - 0.0035), CY, -0.0002), (0.0035, 0.062, 0.0118), 0.009, lid=True)
box(lid, "headphone_lid_wall_far", (0, CY + HY - 0.0035, -0.0002), (0.078, 0.0035, 0.0118), 0.009, lid=True)
box(lid, "headphone_lid_wall_hinge", (0, CY - HY + 0.0035, -0.0002), (0.078, 0.0035, 0.0118), 0.009, lid=True)
for i in LID_PARTS:
    n = f"hinged_{EP}_{i:02d}"
    ET.SubElement(lid, "geom", name=n, type="mesh", mesh=n, material=materials[n], group="2", contype="0", conaffinity="0", mass="0", pos=fmt(-PIVOT_LOCAL))

for e in lights:
    world.append(e)

OUT.mkdir(exist_ok=True)
tree = ET.ElementTree(root)
ET.indent(tree)
tree.write(OUT / "scene.xml", encoding="utf-8", xml_declaration=True)
shutil.copyfile(SCENES / "clippers_only/background_top.png", OUT / "background_top.png")

R_WORLD = rz(PSI) @ R_LOCAL  # canonical -> world
hinge_world = ORIGIN + rz(PSI) @ PIVOT_LOCAL
lid_centre_world = ORIGIN + R_WORLD @ np.array([0, CY, 0])
objects = {
    "source": "built by lerobot/scripts/sim_builders/easy_envs/close_lid/build_close_lid.py from clippers_only (table, cameras, lights, background) "
              "and the ep2 hinged headphone meshes; placement chosen like the clippers_only case, lid toward the robot as in the 2026-10-21 close-lid recordings",
    "objects": {
        "headphone_case": {"xy": [round(float(ORIGIN[0]), 4), round(float(ORIGIN[1]), 4)], "z": HZ, "body_yaw_deg": round(float(np.degrees(PSI) % 360), 2),
                           "fixed": True, "half_extents_m": [HX, HY, HZ], "note": "loaded base half, world-fixed body"},
        "headphone_lid": {"xy": [round(float(lid_centre_world[0]), 4), round(float(lid_centre_world[1]), 4)], "joint": "lid_hinge",
                          "hinge_point_world": [round(float(v), 4) for v in hinge_world],
                          "hinge_axis_world": [round(float(v), 4) for v in (rz(PSI) @ AXIS_LOCAL)],
                          "open_rad": 0.0, "closed_rad": round(CLOSED, 5), "half_extents_m": [HX, HY, HZ]},
    },
}
(OUT / "objects.json").write_text(json.dumps(objects, indent=1))
print("origin", ORIGIN, "psi", PSI, "hinge_world", hinge_world, "axis_world", rz(PSI) @ AXIS_LOCAL, "lid centre", lid_centre_world)

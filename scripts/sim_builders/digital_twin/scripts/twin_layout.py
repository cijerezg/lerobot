"""Pixel picks on the real top frame (ep 0 f0 of rebot_teleop_2026-10-29-cut-v1) -> table-frame poses, written to twin/layout.json.
The picks are unprojected through the CALIBRATED ext camera (package clippers_only scene: same intrinsics, yaw and tilt as the
real webcam) after subtracting the constant pixel offset measured on the robot base plate (the webcam moved since the 10-16
calibration). The aligned camera of the room scene is only used for rendering. Run from the LeRobot root:
  MUJOCO_GL=egl uv run python lerobot/scripts/sim_builders/digital_twin/scripts/twin_layout.py"""
import json, pathlib, numpy as np, mujoco

ROOT = pathlib.Path("lerobot/scripts/sim_builders/digital_twin"); OUT = ROOT / "twin" / "layout.json"
CALIB = "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/clippers_only/scene.xml"
# real px = calibrated-render px + OFFSET (robot base plate green bar: real (344,127) vs calibrated render (335,75); desk far-right corner real (598,73) vs (588,29))
OFFSET = np.array([9.0, 50.0])

# picks in the real 640x480 frame: footprint centre (u, v, feature height h in m) and two points on the long axis (same height)
PICKS = {
    "bits_kit":       dict(centre=(315, 238, 0.02), axis=[(245, 159), (435, 230)], note="open bit organizer, centre of the blue tray; axis = long side"),
    "headphone_case": dict(centre=(200, 370, 0.03), axis=[(200, 370), (155, 300)], note="base half (with the headphones) centre; axis = base -> lid half centre"),
    "roar_bottle":    dict(centre=(105, 218, 0.0),  axis=None, note="ROAR drink bottle, base ellipse centre (standing)"),
    "pill_bottle":    dict(centre=(98, 256, 0.0),   axis=None, note="small white pill bottle, base centre"),
    "navy_bottle":    dict(centre=(116, 140, 0.0),  axis=None, note="dark navy bottle with a blue screw cap at the far-left corner, base centre (v2: built navy cylinder 70 x 200)"),
    # v2 clutter (top frame f0): four loose bits are visible near the robot base; bit_5/bit_6 are added beside them to reach the briefed six
    "bit_1":          dict(centre=(475, 126, 0.003), axis=[(470, 121), (481, 131)], note="loose bit, blue-grey (rendered grey)"),
    "bit_2":          dict(centre=(502, 122, 0.003), axis=[(495, 120), (509, 124)], note="loose bit, black (rendered grey)"),
    "bit_3":          dict(centre=(489, 141, 0.003), axis=[(483, 144), (496, 138)], note="loose bit, red"),
    "bit_4":          dict(centre=(521, 135, 0.003), axis=[(518, 129), (524, 141)], note="loose bit, cream (rendered yellow)"),
    "bit_5":          dict(centre=(462, 136, 0.003), axis=[(457, 133), (467, 139)], note="added grey bit (not in the frame)"),
    "bit_6":          dict(centre=(512, 112, 0.003), axis=[(507, 108), (517, 116)], note="added grey bit (not in the frame)"),
    "bits_box":       dict(centre=(548, 178, 0.0125), axis=[(520, 174), (580, 182)], note="small clear bits box right of the robot base, lid centre; axis = long side"),
    "bit_holder":     dict(centre=(322, 350, 0.0075), axis=[(335, 333), (308, 368)], note="silver bit holder between the large clipper and the case; axis = long side"),
    "basket":         dict(centre=(577, 311, 0.0),  axis=[(527, 250), (548, 385)], note="footprint centre estimated from the rim (0.21 m tall, real nadir (340,326)); axis = bottom edge of the left wall"),
    "clipper_case":   dict(centre=(411, 365, 0.015), axis=[(372, 345), (436, 338)], note="white tin centre; axis = long side"),
    "clipper_large":  dict(centre=(288, 367, 0.005), axis=[(276, 380), (300, 357)], note="clipper lying left of the case"),
}
# v4: added objects with no pixel pick (not in the real frame); table xy given directly, free desk spots 0.25-0.45 m from the base
ADDED = {"tape_roll": dict(table_xy=(0.16, -0.39), note="added, not in the real frame: package ep3 tape roll, flat, between the kit's far end and the bottles"),
         "paper_cup": dict(table_xy=(0.20, 0.25), note="added, not in the real frame: package ep3 paper cup, between the kit and the loose bits")}
# desk-edge picks for the sanity check (real px, on the table)
DESK = dict(back_left=(118, 100), back_right=(598, 73), front_left=(38, 432), front_mid=(320, 418), left_mid=(60, 250))

m = mujoco.MjModel.from_xml_path(CALIB); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
cid = m.camera("ext").id; f = 0.5 * 480 / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2); R = d.cam_xmat[cid].reshape(3, 3); t = d.cam_xpos[cid]

def unproject(u, v, h=0.0):
    """real pixel -> table point on the plane z = h."""
    u, v = np.array([u, v], float) - OFFSET
    ray = R @ np.array([(u - 320) / f, -(v - 240) / f, -1.0]); s = (h - t[2]) / ray[2]; p = t + s * ray; return p[:2]

def axis_angle(a, b, h):
    pa, pb = unproject(*a, h), unproject(*b, h); return float(np.degrees(np.arctan2(pb[1] - pa[1], pb[0] - pa[0])))

layout = {"source_frame": "outputs/rebot_teleop_2026-10-29-cut-v1 ep0 frame 0, observation.images.top (640x480)",
          "camera_used_for_unprojection": {"scene": CALIB, "pos": t.round(5).tolist(), "fovy": float(m.cam_fovy[cid]), "note": "calibrated webcam pose; the room scene's aligned ext camera is for rendering only"},
          "pixel_offset_real_minus_calibrated": OFFSET.tolist(), "table_frame": "robot base at the origin, z = 0 desk top, x toward the far edge / top camera, y to the image right", "objects": {}}
for name, p in PICKS.items():
    u, v, h = p["centre"]; xy = unproject(u, v, h); a = axis_angle(*p["axis"], h) if p["axis"] else None
    layout["objects"][name] = {"pick_px": [u, v], "pick_height_m": h, "axis_px": p["axis"], "table_xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)], "axis_angle_deg": None if a is None else round(a, 1), "note": p["note"]}
    print(f"{name:15s} px ({u},{v}) h {h:.3f} -> table ({xy[0]:+.3f}, {xy[1]:+.3f})" + (f" axis {a:+.1f} deg" if a is not None else ""))
for name, p in ADDED.items():
    layout["objects"][name] = {"pick_px": None, "pick_height_m": None, "axis_px": None, "table_xy": list(p["table_xy"]), "axis_angle_deg": None, "note": p["note"]}
layout["desk_edge_check"] = {k: [round(float(c), 3) for c in unproject(*v)] for k, v in DESK.items()}
print("desk edge picks ->", layout["desk_edge_check"])
OUT.write_text(json.dumps(layout, indent=1)); print("wrote", OUT)

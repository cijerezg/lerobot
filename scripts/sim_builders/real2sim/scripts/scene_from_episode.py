"""Write the MuJoCo scene for one real frame (robot + table + clipper case + clippers [+ clutter]) and render it from
the calibrated external camera next to the real frame.
usage: scene_from_episode.py ROOT EPISODE FRAME CAMERA_JSON PIXELS_JSON OUT_DIR [--later f1 f2 ...] [--no_clutter]
Object pixel centres/axes come from PIXELS_JSON (read by eye); they are back-projected through the camera onto the
table plane z = 0 (or z = object height for tall objects). Yaw = world direction of the back-projected axis.
Outputs: OUT_DIR/scene_nail.xml (self-contained: absolute mesh dir, local copy of the robot model with widened ranges),
OUT_DIR/objects.json (poses in metres / degrees), OUT_DIR/composite_{frame}.jpg (real | sim | blend), composites.jpg."""
import argparse, json, pathlib, re, sys, cv2, numpy as np, mujoco, trimesh
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scene_sim as S
from scene_frames import episode_states, decode_frames, TOP

ASSET = pathlib.Path(__file__).resolve().parents[1] / "assets/clipper_case/clipper_case.xml"
CLUTTER_RGBA = {"bottle_yellow_cap_standing": "0.95 0.95 0.92 1", "bottle_white_lying": "0.95 0.95 0.92 1", "spray_bottle_lying": "0.08 0.08 0.1 1",
                "blue_pouch": "0.1 0.1 0.12 1", "headphone_case_closed": "0.08 0.08 0.08 1", "bits_box": "0.75 0.75 0.78 0.9", "mic_knob": "0.08 0.08 0.08 1"}
CLUTTER_DIAM_M = {"bottle_yellow_cap_standing": 0.055, "bottle_white_lying": 0.05, "spray_bottle_lying": 0.055, "mic_knob": 0.055}
# nominal masses (kg); kitchen-scale values pending (user 2026-10-17). Clutter = free bodies since 2026-10-17 night (was fixed to the world).
CLUTTER_MASS_KG = {"bottle_yellow_cap_standing": 0.12, "bottle_white_lying": 0.08, "spray_bottle_lying": 0.25, "blue_pouch": 0.15, "headphone_case_closed": 0.25, "bits_box": 0.15, "mic_knob": 0.05}
FRICTION = 'condim="4" friction="0.6 0.005 0.0001"'


def free_body(name, pos, quat, geom):
    return f'    <body name="{name}" pos="{pos}" quat="{quat}">\n      <freejoint/>\n      {geom}\n    </body>\n'


def yaw_quat(yaw):
    return f"{np.cos(yaw / 2):.5f} 0 0 {np.sin(yaw / 2):.5f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("episode", type=int); ap.add_argument("frame", type=int); ap.add_argument("camera"); ap.add_argument("pixels"); ap.add_argument("out")
    ap.add_argument("--later", type=int, nargs="*", default=[]); ap.add_argument("--no_clutter", action="store_true"); ap.add_argument("--wrist_solution", default="combined", help="key in wrist_camera.json"); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cam = json.load(open(a.camera)); pos, rv, fovy = np.array(cam["pos"]), np.array(cam["rotvec"]), cam["fovy"]
    px = json.load(open(a.pixels)); bp = lambda uv, z=0.0: S.backproject_z0(np.array(uv, float), pos, rv, fovy, z0=z)
    # scale at the table plane for pixel -> metre sizes
    m_per_px = np.linalg.norm(bp([321, 240]) - bp([320, 240]))
    # table: back-project the corners; the table is axis aligned with the robot (x toward the camera side, y to the image right)
    tc = {k: bp(v) for k, v in px["table_corners"].items()}
    x_far = 0.5 * (tc["far_left"][0] + tc["far_right"][0]); x_near = 0.5 * (tc["near_left"][0] + tc["near_right"][0])
    y_left = 0.5 * (tc["far_left"][1] + tc["near_left"][1]); y_right = 0.5 * (tc["far_right"][1] + tc["near_right"][1])
    table = dict(x=[float(x_far), float(x_near)], y=[float(y_left), float(y_right)], corners_world={k: v.round(4).tolist() for k, v in tc.items()})
    objects = {}
    def pose_from(name, z=0.0):
        o = px["objects"][name]; c = bp(o["center"], z); ax = bp(o["axis"], z) if "axis" in o else None
        yaw = float(np.arctan2(ax[1][1] - ax[0][1], ax[1][0] - ax[0][0])) if ax is not None else 0.0
        length = float(np.linalg.norm(ax[1] - ax[0])) if ax is not None else None
        objects[name] = dict(xy=[float(c[0]), float(c[1])], yaw_deg=float(np.rad2deg(yaw)), axis_length_m=length, center_px=o["center"]); return c, yaw, length
    # clipper case: the long axis is body +y and the black end is at +y (object_materials.py class map), so the pixel axis is read
    # from the foam end to the black end and the body yaw puts +y on it (no 180 deg ambiguity)
    mesh_axis = np.pi / 2
    c_case, yaw_case, _ = pose_from("clipper_case", z=0.012); q_case = yaw_case - mesh_axis
    c_l, yaw_l, _ = pose_from("clipper_large", z=0.005); c_s, yaw_s, _ = pose_from("clipper_small", z=0.005)
    objects["clipper_case"]["body_yaw_deg"] = float(np.rad2deg(q_case))
    # asset bodies with absolute mesh paths and the measured poses
    src = ASSET.read_text(); src = re.sub(r'file="([^"/]+)"', lambda m: f'file="{ASSET.parent / m.group(1)}"', src)
    src = re.sub(r'<body name="clipper_case" pos="[^"]*"', f'<body name="clipper_case" pos="{c_case[0]:.4f} {c_case[1]:.4f} 0.0" quat="{yaw_quat(q_case)}"', src)  # z 0: the skirt box rests on the table
    src = re.sub(r'<body name="clipper_large" pos="[^"]*" quat="[^"]*"', f'<body name="clipper_large" pos="{c_l[0]:.4f} {c_l[1]:.4f} 0.0051" quat="{yaw_quat(yaw_l)}"', src)
    src = re.sub(r'<body name="clipper_small" pos="[^"]*" quat="[^"]*"', f'<body name="clipper_small" pos="{c_s[0]:.4f} {c_s[1]:.4f} 0.0046" quat="{yaw_quat(yaw_s)}"', src)
    asset_block = re.search(r"<asset>(.*?)</asset>", src, re.S).group(1); default_block = re.search(r"\n  <default>\n(.*?)\n  </default>", src, re.S).group(1) + "\n"; body_block = re.search(r"<worldbody>(.*?)</worldbody>", src, re.S).group(1)
    # clutter as primitives (sizes from pixels at the table scale, heights guessed)
    clutter = ""
    if not a.no_clutter:
        for name, o in px["objects"].items():
            if name.startswith("clipper"): continue
            h = o.get("height_m", 0.05); c, yaw, length = pose_from(name, z=h); rgba = o.get("rgba", CLUTTER_RGBA.get(name, "0.6 0.6 0.6 1"))
            CLUTTER_MASS_KG[name] = o.get("mass_kg", CLUTTER_MASS_KG.get(name, 0.1)); diam = o.get("diameter_m", CLUTTER_DIAM_M.get(name, 0.05))
            if "axis" in o and "diameter_px" in o:  # lying cylinder: axis along local x after a 90 deg roll about y
                r = diam / 2; q = S.Rot.from_euler("ZY", [yaw, np.pi / 2]).as_quat(); q = f"{q[3]:.5f} {q[0]:.5f} {q[1]:.5f} {q[2]:.5f}"
                clutter += free_body(name, f"{c[0]:.4f} {c[1]:.4f} {r:.4f}", q, f'<geom name="{name}" type="cylinder" size="{r:.4f} {length / 2:.4f}" mass="{CLUTTER_MASS_KG[name]}" rgba="{rgba}" {FRICTION}/>'); objects[name].update(kind="cylinder lying", radius_m=r, length_m=length)
            elif "diameter_px" in o:
                r = diam / 2
                clutter += free_body(name, f"{c[0]:.4f} {c[1]:.4f} {h / 2:.4f}", "1 0 0 0", f'<geom name="{name}" type="cylinder" size="{r:.4f} {h / 2:.4f}" mass="{CLUTTER_MASS_KG[name]}" rgba="{rgba}" {FRICTION}/>'); objects[name].update(kind="cylinder standing", radius_m=r, height_m=h)
            else:
                sx, sy = [0.5 * s * m_per_px * (pos[2] - h) / pos[2] for s in o["size_px"]]  # pixel size at height h -> metres
                typ = "ellipsoid" if o.get("kind") == "ellipsoid" or name.startswith("headphone") else "box"; zc = h / 2
                size = f"{sx:.4f} {sy:.4f} {h / 2:.4f}"
                clutter += free_body(name, f"{c[0]:.4f} {c[1]:.4f} {zc:.4f}", yaw_quat(yaw), f'<geom name="{name}" type="{typ}" size="{size}" mass="{CLUTTER_MASS_KG[name]}" rgba="{rgba}" {FRICTION}/>'); objects[name].update(kind=typ, half_size_m=[sx, sy, h / 2])
    # robot model copy with widened ranges, compiler line removed (the scene sets an absolute meshdir)
    robot_src = S.MODEL.read_text()
    for joint, (lo, hi) in S.SOFT_LIMITS_DEG.items():
        m = re.search(rf'<joint name="{joint}"[^>]*range="([-0-9.e]+) ([-0-9.e]+)"', robot_src); lo = min(np.deg2rad(lo), float(m.group(1))); hi = max(np.deg2rad(hi), float(m.group(2))); widened = f'{lo:.4f} {hi:.4f}"'
        robot_src = re.sub(rf'(<joint name="{joint}"[^>]*range=")[^"]*"', lambda m: m.group(1) + widened, robot_src); robot_src = re.sub(rf'(<position [^>]*name="{joint}"[^>]*ctrlrange=")[^"]*"', lambda m: m.group(1) + widened, robot_src)
    robot_src = re.sub(r"<compiler[^>]*/>", "", robot_src)
    wrist_json = out / "wrist_camera.json"   # from scene_wrist_calibrate.py: D405 pose in the gripper_end frame
    if wrist_json.exists():
        wc = json.load(open(wrist_json))[a.wrist_solution]; robot_src = S.add_wrist_camera(robot_src, wc["pos"], wc["quat_mj_wxyz"], wc["fovy"]); print("wrist camera", a.wrist_solution, "pos", np.round(wc["pos"], 4), "quat", np.round(wc["quat_mj_wxyz"], 4), "fovy %.2f" % wc["fovy"])
    (out / "rebot_b601_dm_scene_nail.xml").write_text(robot_src)
    wood = px.get("table_rgb", [200, 168, 145]); wood = " ".join(f"{c / 255:.3f}" for c in wood)
    tx, ty = 0.5 * (table["x"][0] + table["x"][1]), 0.5 * (table["y"][0] + table["y"][1]); hx, hy = 0.5 * abs(table["x"][1] - table["x"][0]), 0.5 * abs(table["y"][1] - table["y"][0])
    q = S.quat_wxyz(rv)
    scene = f'''<?xml version='1.0' encoding='utf-8'?>
<mujoco model="scene_nail">
  <!-- generated by scene_from_episode.py from {a.root} episode {a.episode} frame {a.frame}; camera {pathlib.Path(a.camera).name} -->
  <compiler angle="radian" meshdir="{S.MJCF / 'assets'}" autolimits="true"/>
  <include file="rebot_b601_dm_scene_nail.xml"/>
  <visual>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.6 0.6 0.6 1"/>
    <global offwidth="640" offheight="480"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.75 0.77 0.78" rgb2="0.75 0.77 0.78" width="32" height="32"/>
    <material name="wood" rgba="{wood} 1" specular="0.15" shininess="0.2"/>
    <material name="floor_mat" rgba="0.16 0.17 0.12 1"/>
{asset_block}  </asset>
  <default>
{default_block}  </default>
  <worldbody>
    <light pos="0.3 0 2.0" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1" castshadow="true"/>
    <light pos="1.5 1.0 1.5" dir="-0.6 -0.5 -0.6" directional="true" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.72" material="floor_mat"/>
    <geom name="table" type="box" size="{hx:.4f} {hy:.4f} 0.015" pos="{tx:.4f} {ty:.4f} -0.015" material="wood" contype="0" conaffinity="0"/>
    <geom name="table_contact" type="plane" size="{hx:.4f} {hy:.4f} 0.001" pos="{tx:.4f} {ty:.4f} 0" group="3"/>  <!-- collision on a plane: cylinders on the box sank 7-9 mm (box-convex contacts), plane-cylinder has its own collider -->
    <camera name="ext" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" fovy="{fovy:.3f}"/>
{clutter}{body_block}  </worldbody>
</mujoco>
'''
    (out / "scene_nail.xml").write_text(scene)
    json.dump(dict(table=table, m_per_px_at_table=float(m_per_px), objects=objects, camera=cam, source=dict(root=a.root, episode=a.episode, frame=a.frame)), open(out / "objects.json", "w"), indent=1)
    print("table x %.3f..%.3f y %.3f..%.3f (m); %.1f px/m at z=0" % (*table["x"], *table["y"], 1 / m_per_px))
    for k, v in objects.items(): print(f"{k:28s} xy ({v['xy'][0]:+.3f}, {v['xy'][1]:+.3f}) yaw {v['yaw_deg']:+6.1f} deg" + (f" axis {v['axis_length_m']:.3f} m" if v["axis_length_m"] else ""))
    # render composites
    model = mujoco.MjModel.from_xml_path(str(out / "scene_nail.xml")); data = mujoco.MjData(model); V = S.Views(model)
    states, _ = episode_states(a.root, a.episode); frames_idx = [a.frame] + list(a.later); real = decode_frames(a.root, a.episode, TOP, frames_idx)
    rows = []
    for i in frames_idx:
        S.set_state(model, data, states[i]); sim = np.ascontiguousarray(V.rgb(data)[:, :, ::-1]); r = real[i]; blend = cv2.addWeighted(r, 0.5, sim, 0.5, 0)
        mask = V.robot_mask(data); cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); ov = r.copy(); cv2.drawContours(ov, cs, -1, (0, 255, 0), 1)
        tiles = [ov, sim, blend]
        for t, lab in zip(tiles, (f"real f{i} + sim robot edge", "sim", "50/50 blend")): cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3); cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        row = np.hstack(tiles); cv2.imwrite(str(out / f"composite_{i:05d}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 92]); rows.append(row)
    cv2.imwrite(str(out / "composites.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("wrote", out / "scene_nail.xml", "and", len(rows), "composites")


if __name__ == "__main__":
    main()

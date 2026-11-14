"""Fit the robot's three MJCF materials to the real top frames: per material, mean RGB of the real pixels inside the
(eroded) sim robot mask over the mean of the sim pixels, applied as a gain on the material rgba (clipped to 1), alum
specular lowered. Writes a copy of the robot MJCF; SCENE_IN and its robot include stay untouched.
usage: MUJOCO_GL=egl uv run python scripts/robot_materials.py SCENE_IN FRAMES_DIR ROBOT_OUT_XML"""
import argparse, json, pathlib, re, cv2, mujoco, numpy as np
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import to_qpos
ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("frames"); ap.add_argument("out"); ap.add_argument("--affine", type=float, nargs=6, help="top camera color_gain (RGB) + color_offset of the sim yaml: real = gain * render + offset, so the materials are fitted on (real - offset) / gain"); a = ap.parse_args()
model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); r = mujoco.Renderer(model, 480, 640)
states = json.load(open(pathlib.Path(a.frames) / "states.json"))
robot_bodies = [model.body(b).id for b in ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right"]]
acc = {}
for f, st in states.items():
    data.qpos[:8] = to_qpos(np.asarray(st, float)); mujoco.mj_forward(model, data)
    r.disable_segmentation_rendering(); r.update_scene(data, "ext"); sim = r.render()
    r.enable_segmentation_rendering(); r.update_scene(data, "ext"); seg = r.render()[:, :, 0]; r.disable_segmentation_rendering()
    real = cv2.imread(str(pathlib.Path(a.frames) / f"top_{int(f):05d}.jpg"))[:, :, ::-1]
    gid = np.clip(seg, 0, None); rob = (seg >= 0) & np.isin(model.geom_bodyid[gid], robot_bodies)
    rob = cv2.erode(rob.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # edges carry table pixels (IoU 0.68 fit)
    for mid in np.unique(model.geom_matid[gid][rob]):
        m = rob & (model.geom_matid[gid] == mid); acc.setdefault(model.material(mid).name, [[], []])
        acc[model.material(mid).name][0].append(real[m]); acc[model.material(mid).name][1].append(sim[m])
robot_src = (pathlib.Path(a.scene).parent / re.search(r'<include file="([^"]+)"', pathlib.Path(a.scene).read_text()).group(1)).read_text()
for name, (re_, si) in acc.items():
    target = np.concatenate(re_).astype(float).mean(0)
    if a.affine: target = (target - np.array(a.affine[3:])) / np.array(a.affine[:3])
    gain = target / np.concatenate(si).astype(float).mean(0)
    m = re.search(rf'<material name="{name}" rgba="([\d.]+) ([\d.]+) ([\d.]+) 1"[^/]*/>', robot_src); rgba = np.clip(np.array(list(map(float, m.groups()))) * gain, 0, 1)
    spec = ' specular="0.2" shininess="0.2"' if name == "alum" else ' specular="0.4" shininess="0.3"'
    robot_src = robot_src.replace(m.group(0), f'<material name="{name}" rgba="{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} 1"{spec} />')
    print(f"{name}: {len(np.concatenate(re_))} px, gain {np.round(gain, 2)} -> rgba {np.round(rgba, 3)}")
pathlib.Path(a.out).write_text(robot_src); print("wrote", a.out)

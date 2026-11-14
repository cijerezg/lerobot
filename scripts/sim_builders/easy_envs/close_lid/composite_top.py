"""Top-view composite of close_lid the way the sim robot's top camera does it (config_rl_sim.yaml top settings), open and closed."""
import cv2, mujoco, numpy as np
from pathlib import Path
WS = Path("/home/user/Documents/Research/RL/LeRobot"); S = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid"; OUT = Path(__file__).resolve().parents[1]
GAIN, OFF, STATIC_GAIN = np.array([1.222, 1.278, 1.165]), np.array([8.0, 8.5, 28.1]), np.array([1.054, 1.231, 1.403])
model = mujoco.MjModel.from_xml_path(str(S / "scene.xml")); data = mujoco.MjData(model); model.opt.timestep = 1 / 510
bg = cv2.cvtColor(cv2.imread(str(S / "background_top.png")), cv2.COLOR_BGR2RGB)
static = np.isin(np.arange(model.ngeom), [model.geom(n).id for n in ("table", "table_top")]); dynamic = model.geom_bodyid != 0
r = mujoco.Renderer(model, 480, 640)
def composite():
    r.update_scene(data, "ext"); rgb = np.clip(r.render() * GAIN + OFF, 0, 255).astype(np.uint8)
    r.enable_segmentation_rendering(); r.update_scene(data, "ext"); geom = r.render()[:, :, 0]; r.disable_segmentation_rendering()
    render = dynamic | static; mask = ((geom >= 0) & render[np.clip(geom, 0, None)]).astype(np.uint8)
    st = (geom >= 0) & static[np.clip(geom, 0, None)]; rgb = rgb.copy(); rgb[st] = np.clip(rgb[st] * STATIC_GAIN, 0, 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32); mask = cv2.GaussianBlur(mask, (0, 0), 2)
    return (mask[:, :, None] * rgb + (1 - mask[:, :, None]) * bg).astype(np.uint8)
jid = model.joint("lid_hinge").id; qadr = model.jnt_qposadr[jid]
mujoco.mj_forward(model, data); cv2.imwrite(str(OUT / "close_lid_top_open_composite.png"), cv2.cvtColor(composite(), cv2.COLOR_RGB2BGR))
data.qpos[qadr] = np.pi; mujoco.mj_forward(model, data); cv2.imwrite(str(OUT / "close_lid_top_closed_composite.png"), cv2.cvtColor(composite(), cv2.COLOR_RGB2BGR))
data.qpos[qadr] = 1.3; mujoco.mj_forward(model, data); cv2.imwrite(str(OUT / "close_lid/close_lid_top_midclose_composite.png"), cv2.cvtColor(composite(), cv2.COLOR_RGB2BGR))
r.close(); print("composites written")

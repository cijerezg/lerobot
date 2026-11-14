"""Side views with collision group 3 visible, open and closed, plus contact-mask check gripper vs lid boxes."""
import mujoco, numpy as np
from PIL import Image
from pathlib import Path
WS = Path("/home/user/Documents/Research/RL/LeRobot"); S = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml"; OUT = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(S)); data = mujoco.MjData(model)
jid = model.joint("lid_hinge").id; qadr = model.jnt_qposadr[jid]
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = [0.36, 0.0174, 0.03]; cam.distance = 0.55; cam.azimuth = 150; cam.elevation = -25
opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[2] = 1; opt.geomgroup[3] = 1
r = mujoco.Renderer(model, 480, 640)
tiles = []
for q in (0.0, 1.3, np.pi):
    data.qpos[qadr] = q; mujoco.mj_forward(model, data)
    r.update_scene(data, cam, opt); tiles.append(r.render())
    opt.geomgroup[2] = 0; r.update_scene(data, cam, opt); tiles.append(r.render()); opt.geomgroup[2] = 1
sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, 6, 2)])
Image.fromarray(sheet).save(OUT / "debug_collision_views.png"); r.close()
g = [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body("gripper_left").id and model.geom_contype[i] | model.geom_conaffinity[i]]
lidbox = model.geom("headphone_lid_floor").id
print("gripper_left collision geoms:", len(g), "contype/conaffinity", model.geom_contype[g[0]], model.geom_conaffinity[g[0]], "| lid floor contype/conaffinity", model.geom_contype[lidbox], model.geom_conaffinity[lidbox])
print("pair collides:", bool((model.geom_contype[g[0]] & model.geom_conaffinity[lidbox]) or (model.geom_contype[lidbox] & model.geom_conaffinity[g[0]])))

"""Compile scenes/close_lid, render ext/wrist, torque-close the lid, check it reaches the closed angle and stays."""
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

WS = Path("/home/user/Documents/Research/RL/LeRobot")
SCENE = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml"
OUT = Path(__file__).resolve().parents[1]
CLOSED = np.pi

spec = mujoco.MjSpec.from_file(str(SCENE))
model = spec.compile()
model.opt.timestep = 1.0 / (30 * 17)  # the sim robot: fps 30 * substeps 17 (config_rl_sim.yaml)
data = mujoco.MjData(model)
jid = model.joint("lid_hinge").id
qadr, dadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
lid = model.body("headphone_lid").id
base = model.body("headphone_case").id
print("bodies: headphone_case id", base, "parent", model.body_parentid[base], "jntnum", model.body_jntnum[base], "| headphone_lid id", lid, "parent", model.body_parentid[lid])
print("lid mass %.4f kg, inertia diag %s, local com %s" % (model.body_mass[lid], model.body_inertia[lid].round(7), model.body_ipos[lid].round(4)))
print("base mass %.4f kg (fixed body)" % model.body_mass[base])
print("joint range", model.jnt_range[jid], "axis(local)", model.jnt_axis[jid], "damping", model.dof_damping[dadr], "frictionloss", model.dof_frictionloss[dadr])
mujoco.mj_forward(model, data)
hinge_w = data.xpos[lid]
axis_w = data.xmat[lid].reshape(3, 3) @ model.jnt_axis[jid]
print("hinge point world", hinge_w.round(4), "axis world", axis_w.round(4))
names = [model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i] in (lid, base)]
print("geoms:", names)
I_hinge = None


def render(cam, path, w=640, h=480):
    r = mujoco.Renderer(model, h, w)
    r.update_scene(data, cam)
    Image.fromarray(r.render()).save(path)
    r.close()


# 1. open state settles
for _ in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
q_open = float(data.qpos[qadr])
print("after 1 s free: lid_hinge = %.5f rad, qvel = %.5f" % (q_open, data.qvel[dadr]))
contacts = {(model.geom(c.geom1).name, model.geom(c.geom2).name) for c in data.contact[: data.ncon]}
print("contacts open:", contacts)
render("ext", OUT / "close_lid_top_open.png")
render("wrist", OUT / "close_lid_wrist.png")

# gravity torque on the hinge in the open pose, for the push estimate
mujoco.mj_forward(model, data)
print("gravity bias torque on hinge (open) %.4f N m" % data.qfrc_bias[dadr])

# 2. torque push: 0.08 N m (= 0.5 N at the lid's far edge, 0.15 m from the hinge) until the lid passes the balance
#    point (1.9 rad, centre of mass over the hinge), then release: the rest of the fall is gravity only
log = []
t = 0.0
peak_vel = 0.0
pushing = True
t_release = None
while t < 3.0:
    if pushing and data.qpos[qadr] > 1.9:
        pushing = False
        t_release = round(t, 3)
    data.qfrc_applied[dadr] = 0.08 if pushing else 0.0
    mujoco.mj_step(model, data)
    t += model.opt.timestep
    if not pushing:
        peak_vel = max(peak_vel, float(data.qvel[dadr]))
    if int(round(t / model.opt.timestep)) % int(round(0.1 / model.opt.timestep)) == 0:
        log.append((round(t, 2), round(float(data.qpos[qadr]), 4), round(float(data.qvel[dadr]), 3)))
print("t, q, qvel:", log)
q_end = float(data.qpos[qadr])
print("released at t = %s s; closed: q = %.4f rad (%.1f deg), closed target %.4f, residual %.4f rad; peak gravity-only closing speed %.2f rad/s" % (t_release, q_end, np.degrees(q_end), CLOSED, CLOSED - q_end, peak_vel))
# 3. stays shut another 2 s with nothing applied
for _ in range(int(2.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
print("after 2 s more: q = %.4f rad, qvel = %.5f" % (data.qpos[qadr], data.qvel[dadr]))
render("ext", OUT / "close_lid_top_closed.png")
# lid extent closed: world z of the lid geoms' AABB tops
tops = [float(data.geom_xpos[i][2] + model.geom_size[i][2]) for i in range(model.ngeom) if model.geom_bodyid[i] == lid and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX]
print("closed lid box tops (world z): max %.4f m" % max(tops))
result = {"t_release": t_release, "q_open": q_open, "q_closed": q_end, "closed_target": CLOSED, "peak_closing_speed_rad_s": peak_vel, "lid_mass": float(model.body_mass[lid]), "log": log}
(OUT / "close_lid/verify_result.json").write_text(json.dumps(result, indent=1))
assert 2.9 < q_end < CLOSED + 0.03 and abs(data.qvel[dadr]) < 2e-3, "lid did not close and settle"
print("PASS")

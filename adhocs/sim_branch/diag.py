"""Debugging only (reads the sim's ground truth): run the stage files from a frame and print, through the last stage,
the hinge angle, the base half's lift and what each finger presses on.

  MUJOCO_GL=egl PYTHONPATH=lerobot/src .venv/bin/python lerobot/adhocs/sim_branch/diag.py twin_l3_close_lid_p2 450 <plan_a.py> ...
"""

import importlib.util
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from simlab import Gripper, observe, restore, send

from lerobot.robots.rebot_b601_sim.easy_tasks import LID, LID_HINGE

layout, t = sys.argv[1], int(sys.argv[2])
every = 5
robot, _, _ = restore(layout, t)
m, d = robot.model, robot.data
gripper = Gripper(m, list(robot.config.joint_limits.values()))
if "PAD_TORSION" in os.environ:  # what-if: the pads' torsional friction coefficient (the scene has 0.02 m)
    for pad in ("gripper_left_pad", "gripper_right_pad"):
        m.geom_friction[m.geom(pad).id, 1] = float(os.environ["PAD_TORSION"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
lid, case = m.body(LID).id, m.body("headphone_case").id
hinge = int(m.jnt_qposadr[m.joint(LID_HINGE).id])
fingers = {gripper.left: "L", gripper.right: "R"}
paths = [Path(p) for p in sys.argv[3:]]
for path in paths:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions = np.asarray(module.plan(observe(robot), gripper), dtype=float)
    z0 = d.xpos[case][2]
    for i, a in enumerate(actions):
        send(robot, a)
        if path is not paths[-1] or (i % every and i != len(actions) - 1):
            continue
        press, force = {}, np.zeros(6)
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for finger, other, geom in ((b1, b2, c.geom2), (b2, b1, c.geom1)):
                if finger in fingers and other in (lid, case):
                    mujoco.mj_contactForce(m, d, k, force)
                    key = fingers[finger] + ":" + m.geom(geom).name.replace("headphone_", "")
                    press[key] = press.get(key, 0.0) + force[0]
        tip, approach, _ = gripper.pose(d)
        want = gripper.fk(a)[0]
        up = d.xmat[case].reshape(3, 3)[:, 2]
        print(
            f"{i:4d} hinge {np.degrees(d.qpos[hinge]):6.1f}  base dz {1e3 * (d.xpos[case][2] - z0):5.1f} mm tilt {np.degrees(np.arccos(min(1.0, abs(up[2])))):4.1f}"
            f"  grip cmd {a[6]:6.1f} read {robot._present_deg()[6]:6.1f}  tip lag {1e3 * np.linalg.norm(tip - want):4.1f} mm  "
            + " ".join(f"{k} {v:.1f}N" for k, v in sorted(press.items()))
        )
robot.disconnect()

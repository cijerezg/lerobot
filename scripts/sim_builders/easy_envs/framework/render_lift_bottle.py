"""Compile scenes/lift_bottle, step 2 s, report the bottle pose and render `ext` at 480x640 to lift_bottle_top.png."""
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

from lerobot.robots.rebot_b601_sim.rebot_b601_sim import scene_path

m = mujoco.MjModel.from_xml_path(str(scene_path("lift_bottle")))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
b = m.body("bottle").id
p0 = d.xpos[b].copy()
while d.time < 2.0:
    mujoco.mj_step(m, d)
p1 = d.xpos[b]
print(f"bottle start {np.round(p0, 4)} after 2 s {np.round(p1, 4)} moved {np.linalg.norm(p1 - p0) * 1e3:.2f} mm, up-axis z {d.xmat[b].reshape(3, 3)[2, 2]:.4f}")
r = mujoco.Renderer(m, 480, 640)
r.update_scene(d, camera="ext")
img = r.render()
r.close()
out = Path(sys.argv[1])
cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("wrote", out)

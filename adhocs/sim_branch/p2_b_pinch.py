"""twin_l3_close_lid_p2, stage b: from the hover, lower the open jaws over the lid's side wall and pinch it.

The wall's rim is read from the wrist depth of the hover frame: in the wrist image the side wall runs up the image at
u about 296 (the thin bright line), and the rim is the nearest surface on a scan line across it."""

import numpy as np
from motion import path
from simlab import deproject

ROWS = range(255, 280, 5)  # the straight part of the wall, toward the lid's far end
COLS = slice(280, 313)
INSERT_M = 0.012  # finger tips this far below the rim
OPEN_DEG = -70.0
PINCH_DEG = -8.0


def rim_point(obs: dict) -> np.ndarray:
    depth, wrist = obs["wrist_depth_m"], obs["calib"]["wrist"]
    points = []
    for v in ROWS:
        u = COLS.start + int(np.argmin(depth[v, COLS]))
        points.append(deproject(wrist, u, v, float(depth[v, u])))
    return np.mean(points, axis=0)


def plan(obs: dict, gripper) -> np.ndarray:
    rim = rim_point(obs)
    q = obs["state_deg"]
    _, approach, jaw = gripper.fk(q)
    grasp = rim - [0.0, 0.0, INSERT_M]
    lower = path(gripper, q, [(grasp, approach, jaw, OPEN_DEG)])
    pinch = path(gripper, lower[-1], [(grasp, approach, jaw, PINCH_DEG)])
    return np.vstack([lower, pinch, np.repeat(pinch[-1:], 5, axis=0)])

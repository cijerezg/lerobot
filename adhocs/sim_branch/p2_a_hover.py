"""twin_l3_close_lid_p2, stage a: from wherever the arm is, go and hover over the lid's side wall, jaws across it.

Targets are pixel picks in the top image of frame 0 of the same episode (arm parked, case unoccluded; the case has not
moved since). The lid is the empty half, the one nearer the robot. Heights are a guess (rim about 4 cm): the wrist
camera never saw the case in this episode, so the hover stays 15 cm up and the next stage reads the wrist depth."""

import numpy as np
from motion import path
from simlab import ray_to_height

BASE_CENTRE_PX = (225.5, 251.5)  # half with the earbuds
LID_CENTRE_PX = (267.0, 193.5)  # empty half
SIDE_WALL_PX = (307.0, 204.0)  # the lid's right-hand side wall in the image, straight part, toward the far end
RIM_Z = 0.04
HOVER_Z = 0.15
OPEN_DEG = -70.0


def targets(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(wall point at RIM_Z, unit hinge->far direction, unit hinge direction)."""
    top = obs["calib"]["top"]
    u = (ray_to_height(top, *LID_CENTRE_PX, RIM_Z) - ray_to_height(top, *BASE_CENTRE_PX, RIM_Z)) * [1, 1, 0]
    u /= np.linalg.norm(u)
    return ray_to_height(top, *SIDE_WALL_PX, RIM_Z), u, np.array([-u[1], u[0], 0.0])


def plan(obs: dict, gripper) -> np.ndarray:
    wall, u, hinge = targets(obs)
    q = obs["state_deg"]
    tip, approach, jaw = gripper.fk(q)
    down = np.array([0.0, 0.0, -1.0])
    lift = tip + [0.0, 0.0, 0.07]  # clear the table and the case under the forearm before travelling
    hover = np.array([wall[0], wall[1], HOVER_Z])
    return path(gripper, q, [(lift, approach, jaw, OPEN_DEG), (hover, down, hinge, OPEN_DEG)])

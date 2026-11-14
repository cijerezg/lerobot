"""twin_l3_close_lid_p2, stage c: with the lid's side wall pinched, swing the lid over the hinge, let go past the
vertical and press it shut.

The jaw axis is parallel to the hinge, so the wall has to pivot between the finger pads while the tip rides the arc.
The sim's pads hold about 0.05 N m of pivot torque per newton of pinch, and the base half only resists 0.1 N m before
it comes up with the lid: a hard pinch (11 N, first attempt) or a 5 N one (second) carries the whole case flat. So the
jaws are backed off to RELAX_DEG looser than where the hard pinch stalled, about 1 N, and the tool leans with the lid
by up to LEAN_DEG. The hinge line is a pixel pick in the frame-0 top image (the seam between the two halves), at rim
height."""

import numpy as np
from motion import path
from p2_a_hover import targets
from p2_b_pinch import INSERT_M, OPEN_DEG, PINCH_DEG
from simlab import ray_to_height

HINGE_PX = (245.0, 222.0)
RELEASE_DEG = 117.0
RELAX_DEG = 1.5
LEAN_DEG = 0.0
UP = np.array([0.0, 0.0, 1.0])


def plan(obs: dict, gripper) -> np.ndarray:
    _, u, _ = targets(obs)
    q = obs["state_deg"]
    tip, approach, jaw = gripper.fk(q)
    rim = tip + INSERT_M * UP
    hinge = ray_to_height(obs["calib"]["top"], *HINGE_PX, rim[2])
    r = (rim - hinge) @ u
    centre = rim - r * u  # the point of the hinge line abreast of the pinch

    def on_arc(deg: float) -> np.ndarray:
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return centre + r * (c * u + s * UP) - INSERT_M * (c * UP - s * u)

    def lean(deg: float) -> np.ndarray:  # the tool axis turned about the hinge the way the lid turns
        a = np.deg2rad(LEAN_DEG * deg / RELEASE_DEG)
        return np.cos(a) * approach + np.sin(a) * u

    light = q[6] - RELAX_DEG
    relax = np.repeat(np.append(q[:6], light)[None], 8, axis=0)
    angles = np.arange(9.0, RELEASE_DEG + 1, 9.0)
    swing = path(gripper, relax[-1], [(on_arc(a), lean(a), jaw, light) for a in angles])
    release = path(gripper, swing[-1], [(on_arc(RELEASE_DEG), lean(RELEASE_DEG), jaw, OPEN_DEG)])
    worst = max(np.linalg.norm(gripper.fk(swing[-1])[0] - on_arc(RELEASE_DEG)), 0.0)
    print(f"hinge {hinge.round(3)} radius {r:.3f} m, tip error at release {1e3 * worst:.1f} mm, light pinch {light:.1f} deg")
    over = centre - 0.6 * r * u
    press = path(gripper, release[-1], [
        (on_arc(RELEASE_DEG) + 0.04 * UP, approach, jaw, PINCH_DEG),
        (over + 0.05 * UP, approach, jaw, PINCH_DEG),
        (over + 0.005 * UP, approach, jaw, PINCH_DEG),
    ])
    return np.vstack([relax, swing, release, press, np.repeat(press[-1:], 10, axis=0)])

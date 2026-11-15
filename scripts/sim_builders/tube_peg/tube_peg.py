"""Tube and peg holder for the twin desk (task "put the tube in the peg", 2026-11-15).

Tube (user, 2026-11-15): outer radius 5 mm, wall 1.2 mm (inner radius 3.8 mm), soft in reality, rigid here, always standing.
A hollow tube has no convex hull, so the collision is split by contact bits (collide if contype1 & conaffinity2 or the reverse;
the robot, the desk and every other object are 1 / 1):
- bore: a ring of N_SEG thin boxes, inner faces tangent to the inner radius (the bore keeps its 3.8 mm), 2 / 2: touches only the peg;
- core: N_SEG / 2 full-diameter slabs (a solid N_SEG-gon prism), 4 / 1: touches the fingers, the desk and the plate, not the peg.
  Fingers on the 1.2 mm ring boxes pressed 0.6 mm into them at 18 N and the tube crept out of the grasp (verify_tube_peg.py);
- the peg is a capsule (rounded top), PEG_R = inner radius - 1 mm, 2 / 2, plus a copy 8 / 1 that only the fingers touch.
The ring boxes in group 2 are the visual. TUBE_LEN, the plate size, the peg height and the masses are estimates from the wrist images.
"""

import numpy as np

TUBE_R_OUT = 0.0050
TUBE_WALL = 0.0012
TUBE_R_IN = TUBE_R_OUT - TUBE_WALL
TUBE_LEN = 0.010
TUBE_MASS = 0.0005
N_SEG = 16
PEG_R = TUBE_R_IN - 0.0010
PEG_TOP = 0.022  # peg top above the plate bottom
PLATE = (0.025, 0.020, 0.002)  # half sizes
HOLDER_MASS = 0.030
CONTACT = 'condim="4" solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2"'


def tube_body(name: str = "tube") -> str:
    """Body origin at the bottom centre of the standing tube, axis = body z."""
    half_t = TUBE_WALL / 2
    half_w = TUBE_R_OUT * np.tan(np.pi / N_SEG)
    lines = [f'    <body name="{name}" pos="0 0 0" quat="1 0 0 0">', "      <freejoint />"]
    for i in range(N_SEG):
        a = 2 * np.pi * i / N_SEG
        pose = (f'pos="{(TUBE_R_IN + half_t) * np.cos(a):.6f} {(TUBE_R_IN + half_t) * np.sin(a):.6f} {TUBE_LEN / 2:.6f}" '
                f'quat="{np.cos(a / 2):.6f} 0 0 {np.sin(a / 2):.6f}" size="{half_t:.6f} {half_w:.6f} {TUBE_LEN / 2:.6f}"')
        lines.append(f'      <geom name="{name}_bore_{i}" type="box" {pose} mass="0" group="3" contype="2" conaffinity="2" friction="0.4 0.005 0.0001" {CONTACT} />')
        lines.append(f'      <geom name="{name}_vis_{i}" type="box" {pose} mass="0" group="2" contype="0" conaffinity="0" rgba="0.90 0.92 0.93 1" />')
    for i in range(N_SEG // 2):
        a = 2 * np.pi * i / N_SEG
        lines.append(f'      <geom name="{name}_core_{i}" type="box" pos="0 0 {TUBE_LEN / 2:.6f}" quat="{np.cos(a / 2):.6f} 0 0 {np.sin(a / 2):.6f}" '
                     f'size="{TUBE_R_OUT:.6f} {half_w:.6f} {TUBE_LEN / 2:.6f}" mass="{2 * TUBE_MASS / N_SEG:.7f}" group="3" contype="4" conaffinity="1" friction="0.8 0.005 0.0001" {CONTACT} />')
    return "\n".join(lines) + "\n    </body>\n"


def holder_body(name: str = "tube_holder") -> str:
    """Body origin at the plate's bottom centre, peg along body z at the plate centre."""
    hx, hy, hz = PLATE
    return (f'    <body name="{name}" pos="0 0 0" quat="1 0 0 0">\n      <freejoint />\n'
            f'      <geom name="{name}_plate" type="box" size="{hx} {hy} {hz}" pos="0 0 {hz}" mass="{HOLDER_MASS * 0.9:.4f}" friction="0.8 0.005 0.0001" {CONTACT} rgba="0.80 0.88 0.92 0.55" />\n'
            f'      <geom name="{name}_peg" type="capsule" size="{PEG_R:.4f}" fromto="0 0 {2 * hz} 0 0 {PEG_TOP - PEG_R:.4f}" mass="{HOLDER_MASS * 0.1:.4f}" contype="2" conaffinity="2" friction="0.4 0.005 0.0001" {CONTACT} rgba="0.80 0.88 0.92 0.75" />\n'
            f'      <geom name="{name}_peg_fingers" type="capsule" size="{PEG_R:.4f}" fromto="0 0 {2 * hz} 0 0 {PEG_TOP - PEG_R:.4f}" mass="0" group="3" contype="8" conaffinity="1" friction="0.4 0.005 0.0001" {CONTACT} rgba="0.80 0.88 0.92 0.75" />\n'
            f"    </body>\n")

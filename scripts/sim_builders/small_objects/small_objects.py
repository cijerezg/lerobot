"""The small objects of the "tidy up the table" recordings (2026-11-25 / 26) for the twin desk: die + dice cup, wall plug +
compartment box, precision screwdriver + its case, hex nut + the printed holder with the pin ("peg").

Sizes are estimates from the top and wrist frames (the gripper pad, 10 x 14 mm, is the ruler), not measurements:
- die 16 mm; cup 74 mm across at the rim, 90 mm tall, dark outside, tan inside;
- plug (wall anchor) 6 x 30 mm, yellow / red / blue; box 120 x 80 x 26 mm clear plastic, back row (screws) and front row
  (red, blue, yellow plugs, one colour per compartment, one plug already in each), lid open behind it, orange latch in front;
- screwdriver 105 mm (steel handle 5 x 60 mm, dark shaft); case 76 x 140 x 12 mm blue tray, yellow label over the far end, two
  screwdrivers already in it, clear lid open to the right;
- nut 8 mm across the flats, 4 mm thick, 4.4 mm hole (M5-like; the real ones look like M4, ~7 mm); holder = black plate with a
  2 mm steel pin 12 mm tall and the open hoop next to it. The nut goes over the pin.

Collision is primitives. Thin lying objects (plug, screwdriver) are boxes so they do not roll. Containers are hollow (floor +
walls). Each place has a marker box ``<container>_region[_<colour>]`` (group 5, no contact): the scorer's footprint and rim
(twin_tasks.PickPlaceLadder ``region``). The nut and the pin split the contact bits as the tube and its peg do
(../tube_peg/tube_peg.py): hole ring 2 / 2 touches only the pin, solid core 4 / 1 touches fingers, desk and plate, the pin has a
fingers-only copy 8 / 1. Body origins: die, plug, nut at the centre; screwdriver at the handle centre (shaft toward +x);
containers at the bottom centre; the holder at the plate bottom under the pin.
"""

import numpy as np

CONTACT = 'condim="4" solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2"'
COL = f'group="3" friction="0.8 0.005 0.0001" {CONTACT}'
SOLID = COL.replace('group="3"', 'group="2"')  # a collision geom that is also what the cameras see
VIS = 'mass="0" group="2" contype="0" conaffinity="0"'
REGION = 'mass="0" group="5" contype="0" conaffinity="0" rgba="0 1 0 0.2"'

DIE = 0.016
DIE_MASS = 0.004
CUP_R_TOP, CUP_R_BOTTOM, CUP_H, CUP_WALL, CUP_FLOOR = 0.037, 0.031, 0.090, 0.003, 0.004
CUP_MASS = 0.080
CUP_SEG = 24
PLUG_LEN, PLUG_D = 0.030, 0.006
PLUG_MASS = 0.001
PLUG_RGBA = {"yellow": "0.95 0.80 0.10 1", "red": "0.85 0.15 0.12 1", "blue": "0.10 0.50 0.80 1"}
BOX = (0.120, 0.080, 0.026)  # plug box outer x, y, z
BOX_WALL = 0.0015
BOX_ROW_Y = 0.004  # the divider between the rows (the front row is the deeper one)
BOX_MASS = 0.060
BOX_COLOURS = ("red", "blue", "yellow")  # front row, -x to +x
DRIVER_HANDLE, DRIVER_SHAFT, DRIVER_D, DRIVER_SHAFT_D = 0.060, 0.045, 0.005, 0.0024
DRIVER_MASS = 0.015
CASE = (0.076, 0.140, 0.012)  # screwdriver case outer x, y, z
CASE_WALL = 0.002
CASE_MASS = 0.120
CASE_SLOT_X = (-0.030, -0.018)  # the two screwdrivers already in the case
NUT_FLATS, NUT_H, NUT_HOLE_R = 0.008, 0.004, 0.0022
NUT_MASS = 0.001
PIN_R = 0.0010
PIN_TOP = 0.015  # above the plate bottom
HOLDER_PLATE = (0.025, 0.022, 0.0015)  # half sizes
HOLDER_PLATE_X = 0.017  # plate centre from the pin
HOOP_X, HOOP_R, HOOP_H, HOOP_WALL = 0.035, 0.021, 0.020, 0.0016
HOOP_SEG = 20
HOLDER_MASS = 0.015

REST_Z = dict(die=DIE / 2, plug=PLUG_D / 2, screwdriver=DRIVER_D / 2, nut=NUT_H / 2)  # body origin above the surface it lies on
PLACE = dict(die="dice_cup", plug="plug_box", screwdriver="screwdriver_case", nut="nut_holder")  # body name prefixes: object -> where it goes


def kind_of(body):
    """die | plug | screwdriver | nut for an object body (die, die_2, plug_red, ...), None for a place."""
    return next((kind for kind, place in PLACE.items() if body.startswith(kind) and not body.startswith(place)), None)


def _quat(*turns):
    """Product of axis rotations ("x" | "y" | "z", radians), applied right to left."""
    q = np.array([1.0, 0, 0, 0])
    for axis, a in turns:
        w, v = np.cos(a / 2), np.sin(a / 2) * np.eye(3)["xyz".index(axis)]
        q = np.concatenate([[q[0] * w - q[1:] @ v], q[0] * v + w * q[1:] + np.cross(q[1:], v)])
    return " ".join(f"{x:.6f}" for x in q)


def _v(*xs):
    return " ".join(f"{x:.5f}" for x in xs)


def _body(name, geoms):
    return "\n".join([f'    <body name="{name}" pos="0 0 0" quat="1 0 0 0">', "      <freejoint />"] + [f"      <geom {g} />" for g in geoms]) + "\n    </body>\n"


def die_body(name="die"):
    h = DIE / 2
    geoms = [f'name="{name}_col" type="box" size="{_v(h, h, h)}" mass="{DIE_MASS}" {COL}',
             f'type="box" size="{_v(h, h, h)}" rgba="0.95 0.95 0.92 1" {VIS}']
    step = 0.0042
    spots = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)], 4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
             5: [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)], 6: [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0)]}
    faces = [(2, 1, 1), (2, -1, 6), (0, 1, 2), (0, -1, 5), (1, 1, 3), (1, -1, 4)]  # axis, side, number (opposite faces add to 7)
    turn = {0: ("y", np.pi / 2), 1: ("x", np.pi / 2), 2: ("z", 0.0)}
    for axis, side, number in faces:
        u, w = [k for k in range(3) if k != axis]
        for a, b in spots[number]:
            p = np.zeros(3); p[axis] = side * h; p[u] = a * step; p[w] = b * step
            geoms.append(f'type="cylinder" size="0.0015 0.0002" pos="{_v(*p)}" quat="{_quat(turn[axis])}" rgba="0.05 0.05 0.05 1" {VIS}')
    return _body(name, geoms)


def cup_body(name="dice_cup"):
    """Standing cup, open at the top; the wall is a ring of tilted boxes (wider at the rim)."""
    tilt = np.arctan2(CUP_R_TOP - CUP_R_BOTTOM, CUP_H)
    r_mid = (CUP_R_TOP + CUP_R_BOTTOM) / 2 - CUP_WALL / 2
    half_w = CUP_R_TOP * np.tan(np.pi / CUP_SEG)
    half_h = CUP_H / 2 / np.cos(tilt)
    geoms = [f'name="{name}_floor" type="cylinder" size="{_v(CUP_R_BOTTOM, CUP_FLOOR / 2)}" pos="0 0 {CUP_FLOOR / 2}" mass="{CUP_MASS / 4}" rgba="0.13 0.12 0.12 1" {SOLID}',
             f'type="cylinder" size="{_v(CUP_R_BOTTOM - CUP_WALL, 0.0003)}" pos="0 0 {CUP_FLOOR + 0.0003}" rgba="0.60 0.36 0.24 1" {VIS}']
    for i in range(CUP_SEG):
        a = 2 * np.pi * i / CUP_SEG
        quat = _quat(("z", a), ("y", tilt))
        geoms.append(f'name="{name}_wall_{i}" type="box" size="{_v(CUP_WALL / 2, half_w, half_h)}" pos="{_v(r_mid * np.cos(a), r_mid * np.sin(a), CUP_H / 2)}" quat="{quat}" '
                     f'mass="{0.75 * CUP_MASS / CUP_SEG:.5f}" rgba="0.13 0.12 0.12 1" {SOLID}')
        r_in = r_mid - CUP_WALL / 2 - 0.0003
        geoms.append(f'type="box" size="{_v(0.0003, half_w, half_h - 0.002)}" pos="{_v(r_in * np.cos(a), r_in * np.sin(a), CUP_H / 2)}" quat="{quat}" rgba="0.60 0.36 0.24 1" {VIS}')
    inside = (CUP_R_BOTTOM - CUP_WALL) / np.sqrt(2)  # the square inside the floor circle
    geoms.append(f'name="{name}_region" type="box" size="{_v(inside, inside, (CUP_H - CUP_FLOOR) / 2)}" pos="0 0 {(CUP_H + CUP_FLOOR) / 2}" {REGION}')
    return _body(name, geoms)


def plug_body(name, colour):
    """Lying wall plug, long axis = body x."""
    rgba = PLUG_RGBA[colour]
    h, r = PLUG_LEN / 2, PLUG_D / 2
    geoms = [f'name="{name}_col" type="box" size="{_v(h, r, r)}" mass="{PLUG_MASS}" {COL}',
             f'type="cylinder" size="{_v(r * 0.8, h)}" quat="{_quat(("y", np.pi / 2))}" rgba="{rgba}" {VIS}']
    geoms += [f'type="cylinder" size="{_v(r, 0.0015)}" pos="{_v(x, 0, 0)}" quat="{_quat(("y", np.pi / 2))}" rgba="{rgba}" {VIS}' for x in (-0.013, -0.006, 0.001, 0.008)]
    return _body(name, geoms)


def _box_compartments():
    """Inside (lo, hi) in x and y of the three front compartments of the plug box, by colour."""
    x0 = -BOX[0] / 2 + BOX_WALL
    pitch = (BOX[0] - 2 * BOX_WALL) / 3
    y_lo, y_hi = -BOX[1] / 2 + BOX_WALL, BOX_ROW_Y - BOX_WALL / 2
    return {colour: ((x0 + i * pitch + BOX_WALL / 2, y_lo), (x0 + (i + 1) * pitch - BOX_WALL / 2, y_hi)) for i, colour in enumerate(BOX_COLOURS)}


def plug_box_body(name="plug_box"):
    """x = the long side, the front row (plugs) at -y, the open lid behind at +y."""
    X, Y, Z = BOX
    w = BOX_WALL
    clear = 'rgba="0.85 0.90 0.92 0.35"'
    walls = [("floor", (X / 2, Y / 2, w / 2), (0, 0, w / 2)), ("front", (X / 2, w / 2, Z / 2), (0, -Y / 2 + w / 2, Z / 2)), ("back", (X / 2, w / 2, Z / 2), (0, Y / 2 - w / 2, Z / 2)),
             ("left", (w / 2, Y / 2, Z / 2), (-X / 2 + w / 2, 0, Z / 2)), ("right", (w / 2, Y / 2, Z / 2), (X / 2 - w / 2, 0, Z / 2)), ("row", (X / 2, w / 2, Z / 2), (0, BOX_ROW_Y, Z / 2))]
    walls += [(f"divider_{i}", (w / 2, Y / 2, Z / 2), (x, 0, Z / 2)) for i, x in enumerate((-(X - 2 * w) / 6, (X - 2 * w) / 6))]
    geoms = [f'name="{name}_{tag}" type="box" size="{_v(*size)}" pos="{_v(*pos)}" mass="{BOX_MASS / 10:.4f}" {clear} {SOLID}' for tag, size, pos in walls]
    lid_tilt = np.radians(15)  # the open lid rests behind the box, its far edge up
    geoms.append(f'name="{name}_lid" type="box" size="{_v(X / 2, Y / 2, w / 2)}" pos="{_v(0, Y / 2 + Y / 2 * np.cos(lid_tilt), Z + Y / 2 * np.sin(lid_tilt))}" quat="{_quat(("x", lid_tilt))}" '
                 f'mass="{BOX_MASS / 5:.4f}" {clear} {SOLID}')
    geoms.append(f'name="{name}_latch" type="box" size="0.010 0.004 0.006" pos="{_v(0, -Y / 2 - 0.004, 0.010)}" mass="0.002" rgba="0.95 0.35 0.10 1" {SOLID}')
    for colour, ((x_lo, y_lo), (x_hi, y_hi)) in _box_compartments().items():
        geoms.append(f'name="{name}_region_{colour}" type="box" size="{_v((x_hi - x_lo) / 2, (y_hi - y_lo) / 2, (Z - w) / 2)}" pos="{_v((x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (Z + w) / 2)}" {REGION}')
        # the plug already there, along the back of its compartment
        geoms.append(f'name="{name}_plug_{colour}" type="box" size="{_v(PLUG_LEN / 2, PLUG_D / 2, PLUG_D / 2)}" pos="{_v((x_lo + x_hi) / 2, y_hi - 0.006, w + PLUG_D / 2)}" mass="{PLUG_MASS}" rgba="{PLUG_RGBA[colour]}" {SOLID}')
    rng = np.random.default_rng(0)
    for _ in range(14):  # screws in the back row: visual only
        x, y, yaw = rng.uniform(-X / 2 + 0.012, X / 2 - 0.012), rng.uniform(BOX_ROW_Y + 0.008, Y / 2 - 0.008), rng.uniform(0, np.pi)
        geoms.append(f'type="cylinder" size="0.0012 0.009" pos="{_v(x, y, w + 0.002)}" quat="{_quat(("z", yaw), ("y", np.pi / 2))}" rgba="0.70 0.70 0.72 1" {VIS}')
    return _body(name, geoms)


def screwdriver_body(name="screwdriver"):
    """Lying precision screwdriver: origin at the handle centre, shaft toward +x."""
    hh, hs, r, rs = DRIVER_HANDLE / 2, DRIVER_SHAFT / 2, DRIVER_D / 2, DRIVER_SHAFT_D / 2
    along = _quat(("y", np.pi / 2))
    geoms = [f'name="{name}_col" type="box" size="{_v(hh, r, r)}" mass="{DRIVER_MASS * 0.85:.4f}" {COL}',
             f'name="{name}_shaft_col" type="box" size="{_v(hs, rs, rs)}" pos="{_v(hh + hs, 0, 0)}" mass="{DRIVER_MASS * 0.15:.4f}" {COL}',
             f'type="cylinder" size="{_v(r, hh)}" quat="{along}" rgba="0.78 0.78 0.80 1" {VIS}',
             f'type="cylinder" size="{_v(r * 1.1, 0.003)}" pos="{_v(-hh + 0.003, 0, 0)}" quat="{along}" rgba="0.60 0.60 0.63 1" {VIS}',
             f'type="cylinder" size="{_v(rs, hs)}" pos="{_v(hh + hs, 0, 0)}" quat="{along}" rgba="0.15 0.15 0.17 1" {VIS}']
    return _body(name, geoms)


def case_body(name="screwdriver_case"):
    """y = the long side, the label over the far end (+y), the open lid flat on the table at +x."""
    X, Y, Z = CASE
    w = CASE_WALL
    blue = 'rgba="0.10 0.25 0.50 1"'
    walls = [("floor", (X / 2, Y / 2, w / 2), (0, 0, w / 2)), ("near", (X / 2, w / 2, Z / 2), (0, -Y / 2 + w / 2, Z / 2)), ("far", (X / 2, w / 2, Z / 2), (0, Y / 2 - w / 2, Z / 2)),
             ("left", (w / 2, Y / 2, Z / 2), (-X / 2 + w / 2, 0, Z / 2)), ("right", (w / 2, Y / 2, Z / 2), (X / 2 - w / 2, 0, Z / 2))]
    geoms = [f'name="{name}_{tag}" type="box" size="{_v(*size)}" pos="{_v(*pos)}" mass="{CASE_MASS / 6:.4f}" {blue} {SOLID}' for tag, size, pos in walls]
    geoms.append(f'name="{name}_lid" type="box" size="{_v(X / 2, Y / 2, 0.001)}" pos="{_v(X + 0.002, 0, 0.001)}" mass="{CASE_MASS / 6:.4f}" rgba="0.85 0.90 0.92 0.30" {SOLID}')
    geoms.append(f'type="box" size="{_v(X / 2 - w, 0.024, 0.0004)}" pos="{_v(0, Y / 2 - w - 0.024, Z - 0.001)}" rgba="0.95 0.75 0.10 1" {VIS}')  # the label: the shafts go under it
    geoms.append(f'name="{name}_region" type="box" size="{_v(X / 2 - w, Y / 2 - w, (Z - w) / 2)}" pos="0 0 {(Z + w) / 2}" {REGION}')
    hh, hs, r, rs = DRIVER_HANDLE / 2, DRIVER_SHAFT / 2, DRIVER_D / 2, DRIVER_SHAFT_D / 2
    for i, x in enumerate(CASE_SLOT_X):
        y = -Y / 2 + w + 0.004 + hh
        geoms.append(f'name="{name}_driver_{i}" type="box" size="{_v(r, hh, r)}" pos="{_v(x, y, w + r)}" mass="0.010" rgba="0.78 0.78 0.80 1" {SOLID}')
        geoms.append(f'type="cylinder" size="{_v(rs, hs)}" pos="{_v(x, y + hh + hs, w + r)}" quat="{_quat(("x", np.pi / 2))}" rgba="0.15 0.15 0.17 1" {VIS}')
    return _body(name, geoms)


def nut_body(name="nut"):
    """Hex nut lying flat, axis = body z."""
    half = NUT_FLATS / 2
    half_w = half * np.tan(np.pi / 6)
    half_t = (half - NUT_HOLE_R) / 2
    geoms = []
    for i in range(6):
        a = np.pi * i / 3
        pose = f'pos="{_v((NUT_HOLE_R + half_t) * np.cos(a), (NUT_HOLE_R + half_t) * np.sin(a), 0)}" quat="{_quat(("z", a))}" size="{_v(half_t, half_w, NUT_H / 2)}"'
        geoms.append(f'name="{name}_bore_{i}" type="box" {pose} mass="0" group="3" contype="2" conaffinity="2" friction="0.4 0.005 0.0001" {CONTACT}')
        geoms.append(f'type="box" {pose} rgba="0.10 0.10 0.11 1" {VIS}')
    for i in range(3):
        geoms.append(f'name="{name}_core_{i}" type="box" quat="{_quat(("z", np.pi * i / 3))}" size="{_v(half, half_w, NUT_H / 2)}" mass="{NUT_MASS / 3:.6f}" contype="4" conaffinity="1" {COL}')
    return _body(name, geoms)


def nut_holder_body(name="nut_holder"):
    """Printed holder: plate, the pin (origin under it, along body z) and the open hoop at +x."""
    hx, hy, hz = HOLDER_PLATE
    black = 'rgba="0.08 0.08 0.09 1"'
    geoms = [f'name="{name}_plate" type="box" size="{_v(hx, hy, hz)}" pos="{_v(HOLDER_PLATE_X, 0, hz)}" mass="{HOLDER_MASS * 0.7:.4f}" {black} {SOLID}',
             f'name="{name}_peg" type="capsule" size="{PIN_R}" fromto="0 0 {2 * hz} 0 0 {PIN_TOP - PIN_R}" mass="0.0005" contype="2" conaffinity="2" friction="0.4 0.005 0.0001" {CONTACT} rgba="0.75 0.75 0.78 1"',
             f'name="{name}_peg_fingers" type="capsule" size="{PIN_R}" fromto="0 0 {2 * hz} 0 0 {PIN_TOP - PIN_R}" mass="0" group="3" contype="8" conaffinity="1" friction="0.4 0.005 0.0001" {CONTACT}']
    half_w = HOOP_R * np.tan(np.pi / HOOP_SEG)
    for i in range(HOOP_SEG):
        a = 2 * np.pi * i / HOOP_SEG
        r = HOOP_R - HOOP_WALL / 2
        geoms.append(f'name="{name}_hoop_{i}" type="box" size="{_v(HOOP_WALL / 2, half_w, HOOP_H / 2)}" pos="{_v(HOOP_X + r * np.cos(a), r * np.sin(a), HOOP_H / 2)}" quat="{_quat(("z", a))}" '
                     f'mass="{HOLDER_MASS * 0.3 / HOOP_SEG:.6f}" {black} {SOLID}')
    geoms.append(f'type="box" size="0.004 0.0025 0.0002" pos="{_v(0.010, -0.008, 2 * hz)}" rgba="0.35 0.33 0.30 1" {VIS}')  # the slot in the plate
    return _body(name, geoms)

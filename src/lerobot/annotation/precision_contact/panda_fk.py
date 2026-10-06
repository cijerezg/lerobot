"""Franka Panda forward kinematics from the published modified-DH table. No URDF needed."""
import numpy as np

# (a, d, alpha) per joint, Craig convention: T_i = RotX(alpha) TransX(a) RotZ(theta) TransZ(d)
DH = [(0, 0.333, 0), (0, 0, -np.pi/2), (0, 0.316, np.pi/2), (0.0825, 0, np.pi/2),
      (-0.0825, 0.384, -np.pi/2), (0, 0, np.pi/2), (0.088, 0, np.pi/2)]
FLANGE_D = 0.107
HAND_TCP_D = 0.1034
HAND_YAW = -np.pi / 4

def _t(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st*ca, ct*ca, -sa, -d*sa], [st*sa, ct*sa, ca, d*ca], [0, 0, 0, 1]])

def fk(q, tcp='hand'):
    """q: (7,) -> 4x4 base->frame. tcp: 'flange' | 'hand' (flange + 0.1034 m along z, yaw -45 deg)."""
    T = np.eye(4)
    for (a, d, al), th in zip(DH, q):
        T = T @ _t(a, d, al, th)
    T = T @ _t(0, FLANGE_D, 0, 0)
    if tcp == 'hand':
        T = T @ _t(0, HAND_TCP_D, 0, HAND_YAW)
    return T

def fk_batch(Q, tcp='hand'):
    return np.stack([fk(q, tcp) for q in Q])

def quat_to_rot(q, order):
    """order 'wxyz' or 'xyzw' -> 3x3."""
    if order == 'wxyz': w, x, y, z = q
    else: x, y, z, w = q
    n = np.sqrt(w*w+x*x+y*y+z*z); w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

LIMITS = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698],
                   [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973]])

"""Joint-space action sequences from tool-frame waypoints (used by the plan files)."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def frame(approach: np.ndarray, jaw: np.ndarray) -> np.ndarray:
    """Rotation matrix with columns (jaw, approach x jaw, approach)."""
    approach = approach / np.linalg.norm(approach)
    jaw = jaw - (jaw @ approach) * approach
    jaw = jaw / np.linalg.norm(jaw)
    return np.stack([jaw, np.cross(approach, jaw), approach], axis=1)


def path(gripper, q_deg: np.ndarray, waypoints: list[tuple], speed_deg_s: float = 40.0, ramp: int = 12, fps: int = 30) -> np.ndarray:
    """(N, 7) joint targets from q_deg (7,) through waypoints (tip, approach, jaw, gripper_deg), stopping at the last.

    Each segment is a straight tip line with a slerped tool orientation, solved by IK seeded step to step. The whole
    path is timed as one move: the fastest arm joint ramps up over `ramp` steps, cruises at speed_deg_s (the policy's
    own sustained speed is 35-38 deg/s) and ramps down. The jaw is symmetric: the nearer sign is taken."""
    q = np.asarray(q_deg, dtype=float)
    fine = [q]
    for tip1, approach1, jaw1, grip1 in waypoints:
        tip0, approach0, jaw0 = gripper.fk(fine[-1])
        r0, r1 = frame(approach0, jaw0), frame(np.asarray(approach1, float), np.asarray(jaw1, float))
        flipped = r1 * np.array([-1.0, -1.0, 1.0])
        if Rotation.from_matrix(r0.T @ flipped).magnitude() < Rotation.from_matrix(r0.T @ r1).magnitude():
            r1 = flipped
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([r0, r1])))
        grip0 = fine[-1][6]
        for s in np.linspace(0.0, 1.0, 41)[1:]:
            r = slerp(s).as_matrix()
            arm = gripper.ik(fine[-1], tip0 + s * (np.asarray(tip1) - tip0), r[:, 2], r[:, 0], iters=30)
            fine.append(np.append(arm, grip0 + s * (grip1 - grip0)))
    fine = np.asarray(fine)
    step = np.abs(np.diff(fine[:, :6], axis=0)).max(axis=1)
    step = np.maximum(step, np.abs(np.diff(fine[:, 6])) / 5)  # the gripper may turn 5x faster than an arm joint
    length = np.concatenate([[0.0], np.cumsum(step)])
    cruise = speed_deg_s / fps
    n = max(2 * ramp, int(np.ceil(length[-1] / cruise)) + ramp)
    k = np.arange(n) + 0.5
    speed = np.minimum(1.0, np.minimum(k, n - k) / ramp)
    at = np.cumsum(speed) / speed.sum() * length[-1]
    return np.column_stack([np.interp(at, length, fine[:, j]) for j in range(7)])

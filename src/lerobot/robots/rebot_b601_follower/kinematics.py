"""Forward kinematics for the reBot B601-DM, backed by the vendored fixend URDF.

Maps dataset joint vectors (degrees, dims 0..5 = shoulder_pan … wrist_roll) to link
placements in the arm's base frame. The mapping to URDF joint1..joint6 is the identity —
see ``urdf/README.md`` for the limit-vs-observed-range check that establishes it.

Clearance is computed from per-link convex hulls rather than link origins: a link's
lowest point is generally not at either of its endpoints (the gripper body hangs well
past ``end_link``'s origin), so an origin-only check misses exactly the collisions that
matter near a table.

Requires ``pinocchio``, which arrives with the ``placo-dep`` extra. Imported lazily so
importing the robot package stays dependency-free.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

URDF_DIR = Path(__file__).parent / "urdf"
URDF_PATH = URDF_DIR / "reBot-DevArm_fixend.urdf"
HULLS_PATH = URDF_DIR / "link_hulls.npz"

# URDF link frames, base outward. Index 0 is the fixed base, index -1 the end effector.
LINK_NAMES = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "end_link")
EE_LINK = "end_link"
ARM_DOF = 6  # dims 0..5 of the 7-dim action; dim 6 is the gripper and has no URDF joint


class RebotKinematics:
    """FK for the 6 arm joints. Joint angles are always in degrees, dataset order."""

    def __init__(self, urdf_path: str | Path = URDF_PATH, hulls_path: str | Path = HULLS_PATH):
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        if self.model.nq != ARM_DOF:
            raise RuntimeError(f"{urdf_path} has nq={self.model.nq}, expected {ARM_DOF}.")
        self.data = self.model.createData()
        self._frame_ids = [self.model.getFrameId(name) for name in LINK_NAMES]
        self._hulls_path = Path(hulls_path)

    @cached_property
    def hulls(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """``link -> (vertices (V,3), faces (F,3))`` in link frame."""
        with np.load(self._hulls_path) as npz:
            return {name: (npz[f"{name}.v"], npz[f"{name}.f"]) for name in LINK_NAMES}

    def frames(self, q_deg: np.ndarray) -> np.ndarray:
        """Link placements for a trajectory of joint vectors.

        Args:
            q_deg: ``(T, >=6)`` or ``(>=6,)`` joint angles in degrees. Columns beyond the
                first 6 (i.e. the gripper) are ignored.

        Returns:
            ``(T, n_links, 4, 4)`` homogeneous transforms, base frame.
        """
        q_deg = np.atleast_2d(np.asarray(q_deg, dtype=np.float64))[:, :ARM_DOF]
        out = np.zeros((len(q_deg), len(LINK_NAMES), 4, 4))
        out[:, :, 3, 3] = 1.0
        for t, q in enumerate(np.deg2rad(q_deg)):
            self._pin.forwardKinematics(self.model, self.data, q)
            self._pin.updateFramePlacements(self.model, self.data)
            for i, fid in enumerate(self._frame_ids):
                placement = self.data.oMf[fid]
                out[t, i, :3, :3] = placement.rotation
                out[t, i, :3, 3] = placement.translation
        return out

    def link_origins(self, q_deg: np.ndarray) -> np.ndarray:
        """``(T, n_links, 3)`` link-frame origins — the arm skeleton."""
        return self.frames(q_deg)[:, :, :3, 3]

    def ee_path(self, q_deg: np.ndarray) -> np.ndarray:
        """``(T, 3)`` end-effector positions."""
        return self.frames(q_deg)[:, LINK_NAMES.index(EE_LINK), :3, 3]

    def ee_rotations(self, q_deg: np.ndarray) -> np.ndarray:
        """``(T, 3, 3)`` end-effector rotation matrices in the base frame."""
        return self.frames(q_deg)[:, LINK_NAMES.index(EE_LINK), :3, :3]

    def hull_points(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        """Transform every link hull into the base frame.

        Args:
            frames: ``(T, n_links, 4, 4)`` from :meth:`frames`.

        Returns:
            ``link -> (T, V, 3)`` world-frame hull vertices.
        """
        return {
            name: np.einsum("tij,vj->tvi", frames[:, i, :3, :3], self.hulls[name][0])
            + frames[:, i, None, :3, 3]
            for i, name in enumerate(LINK_NAMES)
        }

    def min_heights_by_link(self, frames: np.ndarray) -> np.ndarray:
        """``(T, n_links)`` minimum hull height for each link placement.

        The base column is retained so callers can explicitly select safety groups
        (whole moving arm, distal tool, forearm) without recomputing FK or hulls.
        """
        points = self.hull_points(frames)
        return np.stack([points[name][:, :, 2].min(axis=1) for name in LINK_NAMES], axis=1)

    def min_height_from_frames(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Whole-moving-arm minimum from precomputed frames."""
        per_link = self.min_heights_by_link(frames)[:, 1:]
        return per_link.min(axis=1), per_link.argmin(axis=1) + 1

    def min_height(self, q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lowest point of the moving arm at each timestep.

        ``base_link`` is excluded: its hull bottoms out at exactly z=0 (that plane *is*
        the mounting surface), so including it pins the minimum to 0 for every pose.

        Returns:
            ``(z_min (T,), link_index (T,))`` — the height in metres and which link owns
            it, so a violation can be attributed to the elbow rather than the gripper.
        """
        return self.min_height_from_frames(self.frames(q_deg))

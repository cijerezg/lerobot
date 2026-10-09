"""Pure-torch serial-chain forward kinematics built from a kinematics asset.

An asset is a directory under ``lerobot/model/kinematics_assets/<robot>/`` with a ``chain.json``
that gives either a URDF (file + ordered joint names + tool link) or a Craig modified-DH table,
plus the fixed tool transform to the fingertip midpoint and the gripper map. Every joint is
revolute. The hand frame returned by :meth:`SerialChain.fk` uses the common axes: z approach,
y closing, origin at the fingertip midpoint.

Internally every joint ``i`` is ``T_i = Fixed_i @ Rot(axis_i, q_i)``; the chain is
``T_tool = prod_i T_i @ Tool``. Angles are radians inside; :attr:`SerialChain.joint_units`
says what the corpus stores and :meth:`fk` converts at the boundary.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET  # nosec B405: parses the vendored URDF assets only
from dataclasses import dataclass
from pathlib import Path

import torch

ASSETS_DIR = Path(__file__).resolve().parent.parent / "kinematics_assets"


def rpy_matrix(rpy: list[float]) -> torch.Tensor:
    """URDF roll-pitch-yaw (fixed-axis x, y, z) to a 3x3 float64 rotation: ``Rz(y) Ry(p) Rx(r)``."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=torch.float64)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=torch.float64)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float64)
    return rz @ ry @ rx


def homogeneous(xyz: list[float], rpy: list[float]) -> torch.Tensor:
    t = torch.eye(4, dtype=torch.float64)
    t[:3, :3] = rpy_matrix(rpy)
    t[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return t


def axis_angle_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues: ``axis`` (3,) unit, ``theta`` (B,) -> (B, 3, 3)."""
    k = axis / axis.norm()
    kx = torch.zeros(3, 3, dtype=theta.dtype, device=theta.device)
    kx[0, 1], kx[0, 2], kx[1, 0], kx[1, 2], kx[2, 0], kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device)
    c = torch.cos(theta)[:, None, None]
    s = torch.sin(theta)[:, None, None]
    return eye + s * kx + (1 - c) * (kx @ kx)


def _rotation_homogeneous(r: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) -> (B, 4, 4) with zero translation, built out of place so vmap can batch it."""
    b = r.shape[0]
    zeros = torch.zeros(b, 3, 1, dtype=r.dtype, device=r.device)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=r.dtype, device=r.device).expand(b, 1, 4)
    return torch.cat([torch.cat([r, zeros], 2), bottom], 1)


def rotation_log(r: torch.Tensor) -> torch.Tensor:
    """Rotation vector (B, 3) of (B, 3, 3), differentiable at the identity.

    ``v = (R - R^T)^vee = 2 sin(theta) k``; the scale ``theta / (2 sin theta)`` is a function of
    ``c = cos theta`` alone. Near the identity the series ``1/2 + (1 - c)/6`` replaces it, and the
    exact branch is evaluated on a clamped ``c`` so its gradient stays finite under ``torch.where``.
    """
    c = ((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    v = torch.stack([r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], -1)
    small = c > 1 - 1e-6
    c_safe = c.clamp(max=1 - 1e-6)
    exact = torch.acos(c_safe) / (2 * torch.sqrt(1 - c_safe**2))
    scale = torch.where(small, 0.5 + (1 - c) / 6, exact)
    return v * scale[:, None]


@dataclass
class GripperMap:
    raw_closed: float
    raw_open: float
    stroke_m: float

    def aperture(self, raw: torch.Tensor) -> torch.Tensor:
        """Raw corpus gripper value -> opening in meters, clipped to ``[0, stroke]``."""
        frac = (raw - self.raw_closed) / (self.raw_open - self.raw_closed)
        return (frac * self.stroke_m).clamp(0.0, self.stroke_m)


@dataclass
class SerialChain:
    name: str
    joint_names: list[str]
    joint_units: str  # "deg" or "rad": what the corpus stores for this robot
    fixed: torch.Tensor  # (n, 4, 4) float64, parent joint frame -> this joint frame at q = 0
    axes: torch.Tensor  # (n, 3) float64, rotation axis in the joint frame
    tool: torch.Tensor  # (4, 4) float64, last joint frame -> hand frame (fingertip midpoint)
    gripper: GripperMap

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def to_rad(self, q: torch.Tensor) -> torch.Tensor:
        return torch.deg2rad(q) if self.joint_units == "deg" else q

    def from_rad(self, q: torch.Tensor) -> torch.Tensor:
        return torch.rad2deg(q) if self.joint_units == "deg" else q

    def frames(self, q: torch.Tensor) -> torch.Tensor:
        """Joint frames after each joint: ``q`` (B, >= dof) in corpus units -> (B, dof + 1, 4, 4).

        Index ``i`` is the frame of joint ``i`` after its rotation; index ``dof`` is the hand frame.
        """
        q = torch.atleast_2d(q)
        theta = self.to_rad(q[:, : self.dof].to(torch.float64))
        fixed = self.fixed.to(theta.device)
        axes = self.axes.to(theta.device)
        t = torch.eye(4, dtype=torch.float64, device=theta.device).expand(theta.shape[0], 4, 4)
        out = []
        for i in range(self.dof):
            t = t @ fixed[i] @ _rotation_homogeneous(axis_angle_matrix(axes[i], theta[:, i]))
            out.append(t)
        out.append(t @ self.tool.to(theta.device))
        return torch.stack(out, 1)

    def fk(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``q`` (B, dof + 1) corpus units (last column = raw gripper) -> ``(p (B,3), R (B,3,3), g (B,))``."""
        q = torch.atleast_2d(q)
        hand = self.frames(q)[:, -1]
        g = (
            self.gripper.aperture(q[:, self.dof].to(torch.float64))
            if q.shape[1] > self.dof
            else torch.zeros(q.shape[0], dtype=torch.float64)
        )
        return hand[:, :3, 3], hand[:, :3, :3], g

    def pose(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Arm joints only: ``q`` (B, dof) -> ``(p, R)``. Differentiable; used by the IK Jacobian."""
        hand = self.frames(q)[:, -1]
        return hand[:, :3, 3], hand[:, :3, :3]


def _floats(element: ET.Element | None, attr: str, default: str) -> list[float]:
    text = element.get(attr) if element is not None else None
    return [float(v) for v in (text if text is not None else default).split()]


def _link(joint: ET.Element, tag: str) -> str:
    element = joint.find(tag)
    link = element.get("link") if element is not None else None
    if link is None:
        raise ValueError(f"joint {joint.get('name')} has no <{tag} link=...>")
    return link


def _urdf_chain(spec: dict, asset_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    root = ET.parse(asset_dir / spec["file"]).getroot()  # nosec B314: vendored asset, not user input
    joints = {j.get("name"): j for j in root.iter("joint")}
    fixed, axes = [], []
    for name in spec["joints"]:
        j = joints[name]
        fixed.append(
            homogeneous(_floats(j.find("origin"), "xyz", "0 0 0"), _floats(j.find("origin"), "rpy", "0 0 0"))
        )
        axes.append(torch.tensor(_floats(j.find("axis"), "xyz", "0 0 1"), dtype=torch.float64))
    return torch.stack(fixed), torch.stack(axes)


def _tool_link_offset(spec: dict, asset_dir: Path) -> torch.Tensor:
    """Fixed joints between the last revolute joint and ``tool_link`` folded into one transform."""
    root = ET.parse(asset_dir / spec["file"]).getroot()  # nosec B314
    joints = list(root.iter("joint"))
    by_child = {_link(j, "child"): j for j in joints}
    last = next(j for j in joints if j.get("name") == spec["joints"][-1])
    chain: list[torch.Tensor] = []
    link = spec["tool_link"]
    while link != _link(last, "child"):
        j = by_child[link]
        chain.insert(
            0,
            homogeneous(_floats(j.find("origin"), "xyz", "0 0 0"), _floats(j.find("origin"), "rpy", "0 0 0")),
        )
        link = _link(j, "parent")
    t = torch.eye(4, dtype=torch.float64)
    for h in chain:
        t = t @ h
    return t


def _dh_chain(spec: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Craig modified DH ``T_i = RotX(alpha) TransX(a) RotZ(theta) TransZ(d)`` as fixed + z-rotation.

    ``TransZ(d)`` commutes with ``RotZ(theta)``, so the fixed part is ``RotX(alpha) TransX(a) TransZ(d)``
    = ``Trans(a, -d sin alpha, d cos alpha) RotX(alpha)``. Returns the flange offset as the third item.
    """
    fixed, axes = [], []
    for a, d, alpha in spec["rows_a_d_alpha"]:
        fixed.append(homogeneous([a, -d * math.sin(alpha), d * math.cos(alpha)], [alpha, 0.0, 0.0]))
        axes.append(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    flange = homogeneous([0.0, 0.0, spec["flange_d"]], [0.0, 0.0, 0.0])
    return torch.stack(fixed), torch.stack(axes), flange


def load_chain(asset_dir: str | Path) -> SerialChain:
    asset_dir = Path(asset_dir)
    spec = json.loads((asset_dir / "chain.json").read_text())
    tool = homogeneous(spec["tool"]["xyz"], spec["tool"]["rpy"])
    if "urdf" in spec:
        fixed, axes = _urdf_chain(spec["urdf"], asset_dir)
        tool = _tool_link_offset(spec["urdf"], asset_dir) @ tool
    else:
        fixed, axes, flange = _dh_chain(spec["dh"])
        tool = flange @ tool
    g = spec["gripper"]
    return SerialChain(
        name=spec["name"],
        joint_names=list(spec["joint_names"]),
        joint_units=spec["joint_units"],
        fixed=fixed,
        axes=axes,
        tool=tool,
        gripper=GripperMap(g["raw_closed"], g["raw_open"], g["stroke_m"]),
    )

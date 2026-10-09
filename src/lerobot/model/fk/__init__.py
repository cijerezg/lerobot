"""Per-robot differentiable FK and IK on the kinematics assets. Registry keyed by corpus ``robot_type``."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from lerobot.model.fk.chain import ASSETS_DIR, SerialChain, load_chain
from lerobot.model.fk.ik import IKResult, solve_ik

__all__ = ["ASSETS_DIR", "IKResult", "SerialChain", "asset_dirs", "get_chain", "load_chain", "solve_ik"]


@cache
def asset_dirs() -> dict[str, Path]:
    """``robot_type`` (every alias listed in the asset's chain.json) -> asset directory."""
    out: dict[str, Path] = {}
    for spec_path in sorted(ASSETS_DIR.glob("*/chain.json")):
        spec = json.loads(spec_path.read_text())
        for key in [spec["name"], *spec["robot_types"]]:
            out[key] = spec_path.parent
    return out


@cache
def get_chain(robot_type: str) -> SerialChain:
    return load_chain(asset_dirs()[robot_type])

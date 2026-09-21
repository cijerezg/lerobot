"""Metadata contract for canonical, independently cacheable ReBot additions."""
from __future__ import annotations

import json
from pathlib import Path


def prepared_rebot_contract(root) -> dict | None:
    if root is None:
        return None
    path = Path(root) / 'meta' / 'cache_ready.json'
    if not path.is_file():
        return None
    contract = json.loads(path.read_text())
    if contract.get('schema_version') != 1 or contract.get('fps') != 30:
        raise ValueError(f'Unsupported prepared ReBot contract: {path}')
    roles = contract['camera_roles']
    if not roles or len(set(roles)) != len(roles) or set(roles) - {'external_0', 'external_1', 'wrist_0'}:
        raise ValueError(f'Invalid prepared camera roles: {path}')
    if contract.get('depth_key') is not None:
        if contract['depth_key'] != 'wrist_0.depth' or contract.get('depth_units_mm_per_level') != 0.1:
            raise ValueError(f'Unsupported prepared depth storage: {path}')
        if 'wrist_0' not in roles or contract.get('depth_intrinsics') is None:
            raise ValueError(f'Prepared depth requires wrist RGB and intrinsics: {path}')
    return contract


def prepared_episode_depth(root, episode_index: int) -> dict | None:
    contract = prepared_rebot_contract(root)
    if contract is None:
        return None
    if "episode_depth" in contract:
        return contract["episode_depth"][str(int(episode_index))]
    return {"present": contract["depth_key"] is not None, "intrinsics": contract.get("depth_intrinsics")}


def episode_has_depth(root, episode_index: int) -> bool:
    entry = prepared_episode_depth(root, episode_index)
    return entry is None or bool(entry["present"])

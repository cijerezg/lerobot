#!/usr/bin/env python3
"""Load and apply reviewed commit-frame overrides for pilot material."""

import json
import re
from pathlib import Path

UID_RE = re.compile(r".+__ep\d+_(?:p\d+a\d+|seg\d+)$")
FIELDS = {"frame", "kind", "found", "reason"}
KINDS = {"action_end", "end"}


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_commit_overrides(path):
    """Load and structurally validate an auditable UID-keyed override file."""
    path = Path(path)
    with path.open() as handle:
        rows = json.load(handle, object_pairs_hook=_object_without_duplicates)
    if not isinstance(rows, dict):
        raise ValueError(f"{path}: override root must be an object keyed by canonical uid")
    for uid, row in rows.items():
        if not UID_RE.fullmatch(uid):
            raise ValueError(f"{path}: non-canonical override uid {uid!r}")
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError(f"{path}: {uid}: expected fields {sorted(FIELDS)}")
        if type(row["frame"]) is not int:
            raise ValueError(f"{path}: {uid}: frame must be an integer")
        if row["kind"] not in KINDS:
            raise ValueError(f"{path}: {uid}: kind must be one of {sorted(KINDS)}")
        if type(row["found"]) is not bool:
            raise ValueError(f"{path}: {uid}: found must be boolean")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError(f"{path}: {uid}: reason must be a non-empty string")
    return rows


def apply_commit_override(uid, f0, f1, default, overrides):
    """Return a reviewed override or the untouched default commit tuple.

    Unit intervals are half-open. A found commit must name a real frame within
    [f0, f1). A deliberately not-found action may use f1 as its safe alignment
    fallback, matching the existing end-commit convention.
    """
    row = overrides.get(uid)
    if row is None:
        return default
    frame = row["frame"]
    if not f0 <= frame <= f1:
        raise ValueError(f"{uid}: override frame {frame} outside unit bounds [{f0}, {f1}]")
    if row["found"] and frame == f1:
        raise ValueError(f"{uid}: found override cannot use exclusive unit end {f1}")
    return frame, row["kind"], row["found"]

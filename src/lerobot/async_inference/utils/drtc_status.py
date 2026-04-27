"""Structured status events for the DRTC experiment TUI."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_STATUS_ENV = "LEROBOT_DRTC_STATUS_FILE"
_CONTROL_ENV = "LEROBOT_DRTC_CONTROL_FILE"
_LOCK = threading.Lock()


def emit_status(source: str, event: str, **fields: Any) -> None:
    """Append one JSON status event when the DRTC status side-channel is enabled."""
    path_str = os.environ.get(_STATUS_ENV)
    if not path_str:
        return

    payload = {
        "ts": time.time(),
        "source": source,
        "event": event,
        **fields,
    }

    try:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, separators=(",", ":"), default=str)
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Status updates are informational and must never affect robot control.
        return


def emit_control(command: str, **fields: Any) -> None:
    """Append one TUI control command when the control side-channel is enabled."""
    path_str = os.environ.get(_CONTROL_ENV)
    if not path_str:
        return

    payload = {
        "ts": time.time(),
        "command": command,
        **fields,
    }

    try:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, separators=(",", ":"), default=str)
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        return


class DrtcControlReader:
    """Incremental reader for TUI control commands."""

    def __init__(self) -> None:
        self._path_str = os.environ.get(_CONTROL_ENV)
        self._offset = 0
        self._partial = ""

    @property
    def enabled(self) -> bool:
        return bool(self._path_str)

    def read_commands(self) -> list[str]:
        if not self._path_str:
            return []

        path = Path(self._path_str)
        if not path.exists():
            return []

        try:
            size = path.stat().st_size
            if size < self._offset:
                self._offset = 0
                self._partial = ""

            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self._offset)
                data = f.read()
                self._offset = f.tell()
        except Exception:
            return []

        if not data:
            return []

        data = self._partial + data
        if data.endswith("\n"):
            self._partial = ""
            lines = data.splitlines()
        else:
            lines = data.splitlines()
            self._partial = lines.pop() if lines else data

        commands: list[str] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                command = payload.get("command")
                if isinstance(command, str):
                    commands.append(command)
        return commands

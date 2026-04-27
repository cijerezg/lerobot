"""Structured status events for the DRTC experiment TUI."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_STATUS_ENV = "LEROBOT_DRTC_STATUS_FILE"
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

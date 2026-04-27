#!/usr/bin/env python3
"""Two-tab terminal UI for DRTC experiment status and logs."""

from __future__ import annotations

import argparse
import curses
import io
import json
import os
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TailedTextFile:
    path: Path
    label: str
    offset: int = 0
    partial: str = ""

    def read_new_lines(self) -> list[str]:
        if not self.path.exists():
            return []

        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.partial = ""

        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            try:
                f.seek(self.offset)
            except io.UnsupportedOperation:
                data = f.read()
                return data.splitlines()
            data = f.read()
            self.offset = f.tell()

        if not data:
            return []

        data = self.partial + data
        if data.endswith("\n"):
            self.partial = ""
            return data.splitlines()

        lines = data.splitlines()
        if not lines:
            self.partial = data
            return []
        self.partial = lines.pop()
        return lines


@dataclass
class TuiState:
    client: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    status_events: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=1000))

    def apply_status_event(self, event: dict[str, Any]) -> None:
        source = str(event.get("source", "unknown"))
        target = self.server if source == "policy_server" else self.client
        target.update(event)

        event_name = str(event.get("event", "status"))
        detail = _status_event_detail(event)
        stamp = _format_time(float(event.get("ts", time.time())))
        self.status_events.append(f"{stamp} {source}: {event_name}{detail}")


def _format_time(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _status_event_detail(event: dict[str, Any]) -> str:
    fields: list[str] = []
    for key in (
        "phase",
        "label",
        "rlt_replay_size",
        "rlt_training_head",
        "rlt_train_step",
        "episode_id",
    ):
        if key in event and event[key] not in (None, ""):
            fields.append(f"{key}={event[key]}")
    if not fields:
        return ""
    return " | " + " ".join(fields)


def _phase_label(phase: Any) -> str:
    mapping = {
        "recording": "Recording",
        "recording_with_intervention": "Recording with intervention",
        "reset": "Episode complete/reset",
        "waiting_to_start_episode": "Episode complete/reset",
    }
    return mapping.get(str(phase or "reset"), str(phase or "Episode complete/reset"))


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return True

    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            fields = stat_path.read_text(encoding="utf-8").split()
            if len(fields) >= 3 and fields[2] == "Z":
                return False
        except OSError:
            return False

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _update_from_files(
    state: TuiState,
    status_tail: TailedTextFile,
    log_tails: list[TailedTextFile],
) -> None:
    for line in status_tail.read_new_lines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            state.apply_status_event(event)

    for tail in log_tails:
        for line in tail.read_new_lines():
            state.log_lines.append(f"[{tail.label}] {line}")


def _add_line(stdscr: curses.window, y: int, x: int, text: str, width: int, attr: int = 0) -> None:
    height, screen_width = stdscr.getmaxyx()
    if y < 0 or y >= height or x >= screen_width:
        return
    max_width = max(0, min(width, screen_width - x))
    if max_width <= 0:
        return
    try:
        stdscr.addnstr(y, x, text.ljust(max_width), max_width, attr)
    except curses.error:
        return


def _draw_tabs(stdscr: curses.window, active_tab: int, title: str) -> int:
    _, width = stdscr.getmaxyx()
    tab_text = "[Left/Right: switch tabs] [q: quit and stop experiment]"
    _add_line(stdscr, 0, 0, title, width, curses.A_BOLD)
    _add_line(stdscr, 1, 0, tab_text, width, curses.A_REVERSE)
    marker = "Active: Main" if active_tab == 0 else "Active: Logs"
    _add_line(stdscr, 2, 0, marker, width)
    return 4


def _draw_main(stdscr: curses.window, state: TuiState, start_y: int, closing: bool) -> None:
    height, width = stdscr.getmaxyx()
    client = state.client
    server = state.server
    y = start_y

    phase = _phase_label(client.get("phase"))
    last_label = client.get("last_label") or client.get("label") or "none"
    replay_size = server.get("rlt_replay_size", "n/a")
    replay_capacity = server.get("rlt_replay_capacity", "n/a")
    train_head = server.get("rlt_training_head", "unknown")
    episodes_recorded = max(
        int(client.get("episodes_recorded") or 0),
        int(server.get("rlt_completed_episodes") or 0),
    )

    rows = [
        ("Phase", phase),
        ("Episode ID", str(client.get("episode_id", "n/a"))),
        ("Last reset label", str(last_label)),
        ("Episodes recorded", str(episodes_recorded)),
        ("Current episode transitions", str(client.get("current_episode_transitions", "n/a"))),
        ("Replay buffer", f"{replay_size}/{replay_capacity}"),
        ("Online training enabled", _yes_no(server.get("rlt_online_training_enabled"))),
        ("Training state", str(train_head)),
        ("Actor head training", _yes_no(server.get("rlt_actor_training"))),
        ("Critic head training", _yes_no(server.get("rlt_critic_training"))),
        ("Train step", str(server.get("rlt_train_step", "n/a"))),
    ]

    for label, value in rows:
        if y >= height - 1:
            return
        _add_line(stdscr, y, 0, f"{label:28} {value}", width)
        y += 1

    y += 1
    if closing:
        _add_line(stdscr, y, 0, "Experiment process exited; closing TUI...", width, curses.A_BOLD)
        y += 2

    robot_controls = [
        "2: start episode",
        "5: toggle intervention",
        "1: mark success/pass",
        "0: mark failure/terminate",
    ]
    tui_controls = [
        "Left/Right: switch tabs",
        "q: quit and stop experiment",
    ]
    _add_line(stdscr, y, 0, "Robot Controls", width, curses.A_BOLD)
    y += 1
    _add_line(stdscr, y, 0, "  " + "   ".join(robot_controls), width)
    y += 2
    if y >= height:
        return
    _add_line(stdscr, y, 0, "TUI Controls", width, curses.A_BOLD)
    y += 1
    _add_line(stdscr, y, 0, "  " + "   ".join(tui_controls), width)
    y += 2
    if y >= height:
        return

    _add_line(stdscr, y, 0, "Recent status events", width, curses.A_BOLD)
    y += 1
    for line in list(state.status_events)[-(height - y - 1) :]:
        _add_line(stdscr, y, 0, line, width)
        y += 1
        if y >= height:
            return


def _draw_logs(stdscr: curses.window, state: TuiState, start_y: int) -> None:
    height, width = stdscr.getmaxyx()
    available = max(0, height - start_y)
    lines = list(state.log_lines)[-available:]
    y = start_y
    for line in lines:
        _add_line(stdscr, y, 0, line, width)
        y += 1


def _run_curses(
    stdscr: curses.window,
    *,
    status_file: Path,
    client_log_file: Path,
    server_log_file: Path,
    watch_pid: int | None,
) -> int:
    with suppress(curses.error):
        curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.nodelay(True)
    stdscr.timeout(200)

    state = TuiState()
    status_tail = TailedTextFile(status_file, "status")
    log_tails = [
        TailedTextFile(client_log_file, "client"),
        TailedTextFile(server_log_file, "server"),
    ]
    active_tab = 0
    dead_since: float | None = None

    while True:
        _update_from_files(state, status_tail, log_tails)
        alive = _pid_alive(watch_pid)
        if not alive and dead_since is None:
            dead_since = time.time()
        if dead_since is not None and (time.time() - dead_since) > 2.0:
            return 0

        stdscr.erase()
        start_y = _draw_tabs(stdscr, active_tab, "DRTC Experiment")
        if active_tab == 0:
            _draw_main(stdscr, state, start_y, closing=dead_since is not None)
        else:
            _draw_logs(stdscr, state, start_y)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            return 130
        if key == curses.KEY_LEFT:
            active_tab = 0
        elif key == curses.KEY_RIGHT:
            active_tab = 1


def _run_smoke(status_file: Path, client_log_file: Path, server_log_file: Path) -> int:
    state = TuiState()
    status_tail = TailedTextFile(status_file, "status")
    log_tails = [
        TailedTextFile(client_log_file, "client"),
        TailedTextFile(server_log_file, "server"),
    ]
    _update_from_files(state, status_tail, log_tails)

    phase = _phase_label(state.client.get("phase"))
    replay_size = state.server.get("rlt_replay_size", "n/a")
    replay_capacity = state.server.get("rlt_replay_capacity", "n/a")
    train_head = state.server.get("rlt_training_head", "unknown")
    print(f"phase={phase}")
    print(f"replay_buffer={replay_size}/{replay_capacity}")
    print(f"training_state={train_head}")
    print(f"log_lines={len(state.log_lines)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DRTC experiment TUI")
    parser.add_argument("--status-file", required=True, type=Path)
    parser.add_argument("--client-log-file", required=True, type=Path)
    parser.add_argument("--server-log-file", required=True, type=Path)
    parser.add_argument("--watch-pid", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Parse inputs once and print a summary")
    args = parser.parse_args()

    if args.smoke:
        return _run_smoke(args.status_file, args.client_log_file, args.server_log_file)

    return int(curses.wrapper(
        _run_curses,
        status_file=args.status_file,
        client_log_file=args.client_log_file,
        server_log_file=args.server_log_file,
        watch_pid=args.watch_pid,
    ))


if __name__ == "__main__":
    raise SystemExit(main())

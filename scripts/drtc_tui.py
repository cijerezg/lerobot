#!/usr/bin/env python3
"""Textual terminal UI for DRTC experiment status and logs."""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROBOT_KEY_COMMANDS = {
    "2": "start_episode",
    "5": "toggle_intervention",
    "1": "success",
    "0": "failure",
}


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
    status_events: deque[str] = field(default_factory=lambda: deque(maxlen=16))
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=1000))
    actor_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=180))
    critic_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=180))

    def apply_status_event(self, event: dict[str, Any]) -> None:
        source = str(event.get("source", "unknown"))
        target = self.server if source == "policy_server" else self.client
        target.update(event)

        actor_loss = _to_float(event.get("rlt_actor_loss"))
        critic_loss = _to_float(event.get("rlt_critic_loss"))
        if actor_loss is not None:
            self.actor_loss_history.append(actor_loss)
        if critic_loss is not None:
            self.critic_loss_history.append(critic_loss)

        event_name = str(event.get("event", "status"))
        detail = _status_event_detail(event)
        stamp = _format_time(float(event.get("ts", time.time())))
        self.status_events.append(f"{stamp} {source}: {event_name}{detail}")


def _format_time(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_loss(value: Any) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.6f}"


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
        "model_loading": "Model loading",
        "recording": "Recording",
        "recording_with_intervention": "Recording with intervention",
        "reset": "Episode complete/reset",
        "waiting_to_start_episode": "Episode complete/reset",
        "waiting_to_start_next_episode": "Waiting to start next episode/reset (press 2)",
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
) -> list[str]:
    for line in status_tail.read_new_lines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            state.apply_status_event(event)

    new_log_lines: list[str] = []
    for tail in log_tails:
        for line in tail.read_new_lines():
            labelled = f"[{tail.label}] {line}"
            state.log_lines.append(labelled)
            new_log_lines.append(labelled)
    return new_log_lines


def _write_control_command(control_file: Path | None, command: str, state: TuiState) -> None:
    label = command.replace("_", " ")
    if control_file is None:
        state.status_events.append(f"{_format_time(time.time())} tui: control disabled ({label})")
        return

    payload = {
        "ts": time.time(),
        "source": "drtc_tui",
        "command": command,
    }
    try:
        control_file.parent.mkdir(parents=True, exist_ok=True)
        with control_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError as e:
        state.status_events.append(f"{_format_time(time.time())} tui: failed to send {label}: {e}")
        return
    state.status_events.append(f"{_format_time(payload['ts'])} tui: sent {label}")


def _sparkline(values: deque[float], *, width: int = 64) -> str:
    if not values:
        return "no samples yet"

    ticks = "▁▂▃▄▅▆▇█"
    selected = list(values)[-width:]
    low = min(selected)
    high = max(selected)
    if high == low:
        return ticks[0] * len(selected)
    scale = (len(ticks) - 1) / (high - low)
    return "".join(ticks[int((value - low) * scale)] for value in selected)


def _history_summary(values: deque[float]) -> str:
    if not values:
        return "latest=n/a min=n/a max=n/a"
    selected = list(values)
    return f"latest={selected[-1]:.6f} min={min(selected):.6f} max={max(selected):.6f}"


def _build_smoke_state(status_file: Path, client_log_file: Path, server_log_file: Path) -> TuiState:
    state = TuiState()
    status_tail = TailedTextFile(status_file, "status")
    log_tails = [
        TailedTextFile(client_log_file, "client"),
        TailedTextFile(server_log_file, "server"),
    ]
    _update_from_files(state, status_tail, log_tails)
    return state


def _run_smoke(status_file: Path, client_log_file: Path, server_log_file: Path) -> int:
    state = _build_smoke_state(status_file, client_log_file, server_log_file)
    phase = _phase_label(state.client.get("phase"))
    replay_size = state.server.get("rlt_replay_size", "n/a")
    replay_capacity = state.server.get("rlt_replay_capacity", "n/a")
    train_head = state.server.get("rlt_training_head", "unknown")
    print(f"phase={phase}")
    print(f"replay_buffer={replay_size}/{replay_capacity}")
    print(f"training_state={train_head}")
    print(f"actor_loss={_format_loss(state.server.get('rlt_actor_loss'))}")
    print(f"critic_loss={_format_loss(state.server.get('rlt_critic_loss'))}")
    print(f"actor_loss_samples={len(state.actor_loss_history)}")
    print(f"critic_loss_samples={len(state.critic_loss_history)}")
    print(f"log_lines={len(state.log_lines)}")
    return 0


def _run_textual(
    *,
    status_file: Path,
    control_file: Path | None,
    client_log_file: Path,
    server_log_file: Path,
    watch_pid: int | None,
) -> int:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Grid, Vertical
        from textual.widgets import Footer, Header, RichLog, Static, TabbedContent, TabPane
    except ImportError as e:
        raise SystemExit(
            "Textual is required for the DRTC TUI. Install the async extra or run with --no-tui."
        ) from e

    class DrtcTuiApp(App[int]):
        CSS = """
        Screen {
            layout: vertical;
        }

        #main_tab {
            layout: vertical;
        }

        #dashboard {
            grid-size: 2 2;
            grid-gutter: 1 2;
            height: 16;
            padding: 1 1;
        }

        .card {
            border: round $primary;
            padding: 0 1;
            height: 100%;
        }

        #loss_panel {
            border: round $secondary;
            padding: 0 1;
            height: 9;
            margin: 0 1 1 1;
        }

        #recent_panel {
            border: round $accent;
            padding: 0 1;
            height: 1fr;
            margin: 0 1 1 1;
        }

        #logs {
            height: 1fr;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("left", "show_main", "Main"),
            ("right", "show_logs", "Logs"),
            ("q", "quit_app", "Quit"),
            ("2", "robot_start", "Start episode"),
            ("5", "robot_intervention", "Toggle intervention"),
            ("1", "robot_success", "Success"),
            ("0", "robot_failure", "Failure"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.state = TuiState()
            self.status_tail = TailedTextFile(status_file, "status")
            self.log_tails = [
                TailedTextFile(client_log_file, "client"),
                TailedTextFile(server_log_file, "server"),
            ]
            self.dead_since: float | None = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with TabbedContent(id="tabs"):
                with TabPane("Main", id="main"):
                    with Vertical(id="main_tab"):
                        with Grid(id="dashboard"):
                            yield Static(id="phase_card", classes="card")
                            yield Static(id="episode_card", classes="card")
                            yield Static(id="training_card", classes="card")
                            yield Static(id="controls_card", classes="card")
                        yield Static(id="loss_panel")
                        yield Static(id="recent_panel")
                with TabPane("Logs", id="logs_tab"):
                    yield RichLog(id="logs", wrap=True, markup=False, highlight=False)
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(0.2, self.refresh_from_files)
            self.refresh_dashboard()

        def refresh_from_files(self) -> None:
            for line in _update_from_files(self.state, self.status_tail, self.log_tails):
                self.query_one("#logs", RichLog).write(line)

            alive = _pid_alive(watch_pid)
            if not alive and self.dead_since is None:
                self.dead_since = time.time()
            if self.dead_since is not None and (time.time() - self.dead_since) > 2.0:
                self.exit(0)
                return

            self.refresh_dashboard()

        def refresh_dashboard(self) -> None:
            client = self.state.client
            server = self.state.server
            phase = _phase_label(client.get("phase"))
            last_label = client.get("last_label") or client.get("label") or "none"
            episodes_recorded = max(
                int(client.get("episodes_recorded") or 0),
                int(server.get("rlt_completed_episodes") or 0),
            )
            replay_size = server.get("rlt_replay_size", "n/a")
            replay_capacity = server.get("rlt_replay_capacity", "n/a")
            train_head = server.get("rlt_training_head", "unknown")

            closing = "\nExperiment process exited; closing TUI..." if self.dead_since is not None else ""
            self.query_one("#phase_card", Static).update(
                "[b]Phase[/b]\n"
                f"{phase}\n\n"
                f"Intervention: {_yes_no(client.get('intervention'))}"
                f"{closing}"
            )
            self.query_one("#episode_card", Static).update(
                "[b]Episode[/b]\n"
                f"ID: {client.get('episode_id', 'n/a')}\n"
                f"Recorded: {episodes_recorded}\n"
                f"Last label: {last_label}\n"
                f"Current transitions: {client.get('current_episode_transitions', 'n/a')}"
            )
            self.query_one("#training_card", Static).update(
                "[b]RLT Training[/b]\n"
                f"Replay: {replay_size}/{replay_capacity}\n"
                f"Train step: {server.get('rlt_train_step', 'n/a')}\n"
                f"State: {train_head}\n"
                f"Actor active: {_yes_no(server.get('rlt_actor_training'))}\n"
                f"Critic active: {_yes_no(server.get('rlt_critic_training'))}"
            )
            self.query_one("#controls_card", Static).update(
                "[b]Controls[/b]\n"
                "2: start episode\n"
                "5: toggle intervention\n"
                "1: mark success/pass\n"
                "0: mark failure/terminate\n\n"
                "Left/Right: tabs   q: quit"
            )
            self.query_one("#loss_panel", Static).update(
                "[b]RLT Loss[/b]\n"
                f"Actor  {_history_summary(self.state.actor_loss_history)}\n"
                f"{_sparkline(self.state.actor_loss_history)}\n"
                f"Critic {_history_summary(self.state.critic_loss_history)}\n"
                f"{_sparkline(self.state.critic_loss_history)}"
            )
            recent = "\n".join(self.state.status_events) or "No status events yet"
            self.query_one("#recent_panel", Static).update("[b]Recent Status[/b]\n" + recent)

        def action_show_main(self) -> None:
            self.query_one("#tabs", TabbedContent).active = "main"

        def action_show_logs(self) -> None:
            self.query_one("#tabs", TabbedContent).active = "logs_tab"

        def action_quit_app(self) -> None:
            self.exit(130)

        def _send_robot_command(self, command: str) -> None:
            _write_control_command(control_file, command, self.state)
            self.refresh_dashboard()

        def action_robot_start(self) -> None:
            self._send_robot_command("start_episode")

        def action_robot_intervention(self) -> None:
            self._send_robot_command("toggle_intervention")

        def action_robot_success(self) -> None:
            self._send_robot_command("success")

        def action_robot_failure(self) -> None:
            self._send_robot_command("failure")

    return int(DrtcTuiApp().run())


def main() -> int:
    parser = argparse.ArgumentParser(description="DRTC experiment TUI")
    parser.add_argument("--status-file", required=True, type=Path)
    parser.add_argument("--control-file", type=Path, default=None)
    parser.add_argument("--client-log-file", required=True, type=Path)
    parser.add_argument("--server-log-file", required=True, type=Path)
    parser.add_argument("--watch-pid", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Parse inputs once and print a summary")
    args = parser.parse_args()

    if args.smoke:
        return _run_smoke(args.status_file, args.client_log_file, args.server_log_file)

    return _run_textual(
        status_file=args.status_file,
        control_file=args.control_file,
        client_log_file=args.client_log_file,
        server_log_file=args.server_log_file,
        watch_pid=args.watch_pid,
    )


if __name__ == "__main__":
    raise SystemExit(main())

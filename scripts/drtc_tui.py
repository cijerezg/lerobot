#!/usr/bin/env python3
"""Textual terminal UI for DRTC experiment status and logs."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

ROBOT_KEY_COMMANDS = {
    "2": "start_rollout",
    "3": "start_critical_phase",
    "4": "end_critical_phase",
    "5": "toggle_critical_intervention",
    "1": "success",
    "0": "failure",
    "9": "discard_episode",
    "8": "end_rollout",
}
LOSS_HISTORY_SAMPLES = 240
LOSS_CHART_WIDTH = 56
LOSS_CHART_HEIGHT = 3
LOSS_AXIS_WIDTH = 10
TRAJECTORY_CHUNKS = 10
TRAJECTORY_EXECUTED_ACTIONS = 500
TRAJECTORY_CHART_WIDTH = 72
TRAJECTORY_CHART_HEIGHT = 3
TRAJECTORY_MAX_DIMS = 6
TRAJECTORY_QUEUE_SIZE = 1000
TRAJECTORY_CHUNK_MARKERS = "0123456789"


@dataclass
class TrajectoryChunk:
    source_step: int
    actions: list[list[float]]
    frozen_len: int
    timestamp: float
    rtc_params: dict[str, Any] | None = None
    prefix_weights: list[float] | None = None


@dataclass
class ExecutedAction:
    step: int
    action: list[float]
    timestamp: float


@dataclass
class RolloutRow:
    rollout: int
    rollout_start_ts: float
    critical_phase_id: int | None = None
    server_episode_id: int | None = None
    critical_start_s: float | None = None
    critical_end_s: float | None = None
    label: str = "open"
    discard: bool = False


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
    actor_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=LOSS_HISTORY_SAMPLES))
    critic_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=LOSS_HISTORY_SAMPLES))
    trajectory_chunks: deque[TrajectoryChunk] = field(default_factory=lambda: deque(maxlen=TRAJECTORY_CHUNKS))
    executed_actions: deque[ExecutedAction] = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_EXECUTED_ACTIONS)
    )
    rollouts: dict[int, RolloutRow] = field(default_factory=dict)
    rollout_order: deque[int] = field(default_factory=lambda: deque(maxlen=200))
    trajectory_status: str = "disabled"
    trajectory_error: str = ""

    def apply_status_event(self, event: dict[str, Any]) -> None:
        source = str(event.get("source", "unknown"))
        target = self.server if source == "policy_server" else self.client
        target.update(event)
        self._apply_rollout_status_event(source, event)

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

    def _get_or_create_rollout_row(self, rollout_id: int, timestamp: float) -> RolloutRow:
        row = self.rollouts.get(rollout_id)
        if row is None:
            row = RolloutRow(rollout=rollout_id, rollout_start_ts=timestamp)
            self.rollouts[rollout_id] = row
            self.rollout_order.append(rollout_id)
        return row

    def _apply_rollout_status_event(self, source: str, event: dict[str, Any]) -> None:
        event_name = str(event.get("event", "status"))
        timestamp = _to_float(event.get("ts")) or time.time()

        if source == "policy_server":
            server_episode_id = _to_int(event.get("episode_id"))
            client_episode_id = _to_int(event.get("client_episode_id"))
            if server_episode_id is not None and client_episode_id is not None:
                for row in self.rollouts.values():
                    if row.critical_phase_id == client_episode_id:
                        row.server_episode_id = server_episode_id
            return

        rollout_id = _to_int(event.get("rollout_id"))
        if rollout_id is None:
            return

        if event_name == "rlt_rollout_started":
            rollout_start_ts = _to_float(event.get("rollout_start_ts")) or timestamp
            self._get_or_create_rollout_row(rollout_id, rollout_start_ts).rollout_start_ts = rollout_start_ts
            return

        row = self._get_or_create_rollout_row(rollout_id, timestamp)
        critical_phase_id = _to_int(event.get("critical_phase_id") or event.get("episode_id"))
        if critical_phase_id is not None:
            row.critical_phase_id = critical_phase_id

        if event_name in {"rlt_critical_phase_started", "rlt_critical_intervention_started"}:
            critical_start_s = _to_float(event.get("critical_start_s"))
            if critical_start_s is None:
                critical_start_ts = _to_float(event.get("critical_start_ts")) or timestamp
                critical_start_s = max(0.0, critical_start_ts - row.rollout_start_ts)
            row.critical_start_s = critical_start_s
            row.critical_end_s = None
            row.label = "open"
            row.discard = False
            return

        if event_name in {"rlt_critical_phase_ended", "rlt_critical_phase_labeled", "rlt_critical_phase_discarded"}:
            critical_end_s = _to_float(event.get("critical_end_s"))
            if critical_end_s is None:
                critical_end_ts = _to_float(event.get("critical_end_ts")) or timestamp
                critical_end_s = max(0.0, critical_end_ts - row.rollout_start_ts)
            row.critical_end_s = critical_end_s

        if event_name == "rlt_critical_phase_labeled":
            row.label = str(event.get("label") or "open")
            row.discard = False
        elif event_name == "rlt_critical_phase_discarded":
            row.label = "discarded"
            row.discard = True

    def apply_trajectory_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "trajectory_status":
            self.trajectory_status = str(event.get("status", "unknown"))
            self.trajectory_error = str(event.get("error", ""))
            return

        if event_type == "action_chunk":
            chunk = _parse_trajectory_chunk(event)
            if chunk is not None:
                self.trajectory_chunks.append(chunk)
            return

        if event_type == "executed_action":
            action = _parse_executed_action(event)
            if action is not None:
                self.executed_actions.append(action)


def _format_time(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _format_datetime(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _plain_text(value: Any) -> str:
    return str(value).replace("[", "(").replace("]", ")")


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
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
        "rollout_id",
        "critical_phase_id",
        "critical_start_s",
        "critical_end_s",
        "seeded_prebuffer_chunks",
        "rlt_replay_size",
        "rlt_training_head",
        "rlt_train_step",
        "episode_id",
        "buffered_transitions_dropped",
    ):
        if key in event and event[key] not in (None, ""):
            fields.append(f"{key}={event[key]}")
    if not fields:
        return ""
    return " | " + " ".join(fields)


def _phase_label(phase: Any) -> str:
    mapping = {
        "model_loading": "Model loading",
        "rollout_running": "VLA rollout running",
        "critical_recording": "Critical phase recording",
        "critical_recording_with_intervention": "Critical phase recording with intervention",
        "critical_pending_label": "Critical phase ended; label or discard",
        "recording": "Critical phase recording",
        "recording_with_intervention": "Critical phase recording with intervention",
        "reset": "Episode complete/reset",
        "waiting_to_start_episode": "Episode complete/reset",
        "waiting_to_start_next_episode": "Waiting to start next episode/reset (press 2)",
        "waiting_to_start_rollout": "Waiting to start VLA rollout/reset (press 2)",
    }
    return mapping.get(str(phase or "reset"), str(phase or "Episode complete/reset"))


def _coerce_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None

    parsed: list[float] = []
    for item in value:
        numeric = _to_float(item)
        if numeric is None:
            return None
        parsed.append(numeric)
    return parsed


def _parse_trajectory_chunk(event: dict[str, Any]) -> TrajectoryChunk | None:
    source_step = _to_int(event.get("source_step"))
    if source_step is None:
        return None

    raw_actions = event.get("actions")
    if not isinstance(raw_actions, list):
        return None

    actions: list[list[float]] = []
    for raw_action in raw_actions:
        action = _coerce_float_list(raw_action)
        if action:
            actions.append(action)
    if not actions:
        return None

    frozen_len = _to_int(event.get("frozen_len")) or 0
    timestamp = _to_float(event.get("timestamp"))
    if timestamp is None:
        timestamp = time.time()
    rtc_params = event.get("rtc_params") if isinstance(event.get("rtc_params"), dict) else None
    prefix_weights = _coerce_float_list(event.get("prefix_weights"))
    return TrajectoryChunk(
        source_step=source_step,
        actions=actions,
        frozen_len=frozen_len,
        timestamp=timestamp,
        rtc_params=rtc_params,
        prefix_weights=prefix_weights,
    )


def _parse_executed_action(event: dict[str, Any]) -> ExecutedAction | None:
    step = _to_int(event.get("step"))
    action = _coerce_float_list(event.get("action"))
    if step is None or not action:
        return None
    timestamp = _to_float(event.get("timestamp"))
    if timestamp is None:
        timestamp = time.time()
    return ExecutedAction(step=step, action=action, timestamp=timestamp)


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _display_rollout_label(label: str) -> str:
    if label == "failure":
        return "fail"
    return label


def _review_label(label: str | None) -> str:
    if label in {"success", "failure", "open"}:
        return label
    if label == "fail":
        return "failure"
    return "open"


def _ordered_rollout_rows(state: TuiState) -> list[RolloutRow]:
    rows: list[RolloutRow] = []
    for rollout_id in state.rollout_order:
        row = state.rollouts.get(rollout_id)
        if row is not None:
            rows.append(row)
    return rows


def _latest_reviewable_rollout(state: TuiState) -> RolloutRow | None:
    for row in reversed(_ordered_rollout_rows(state)):
        if row.critical_phase_id is not None:
            return row
    return None


def _review_sidecar_path_from_state(state: TuiState) -> Path | None:
    replay_path = state.server.get("replay_path") or state.server.get("rlt_replay_path")
    if not replay_path:
        return None
    return Path(str(replay_path)).with_suffix(".review.json")


def _write_rollout_review_edit(
    state: TuiState,
    *,
    label: str | None = None,
    discard: bool | None = None,
) -> None:
    row = _latest_reviewable_rollout(state)
    if row is None:
        state.status_events.append(f"{_format_time(time.time())} tui: no rollout row to edit")
        return

    sidecar_path = _review_sidecar_path_from_state(state)
    if sidecar_path is None:
        state.status_events.append(f"{_format_time(time.time())} tui: no replay path for review sidecar")
        return

    episode_id = row.server_episode_id or row.critical_phase_id
    if episode_id is None:
        state.status_events.append(f"{_format_time(time.time())} tui: selected rollout has no episode id")
        return

    sidecar: dict[str, Any] = {"version": 1, "episodes": {}}
    if sidecar_path.exists():
        try:
            existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                sidecar = existing
        except (OSError, json.JSONDecodeError):
            sidecar = {"version": 1, "episodes": {}}

    sidecar["version"] = 1
    episodes = sidecar.get("episodes")
    if not isinstance(episodes, dict):
        episodes = {}
        sidecar["episodes"] = episodes

    entry = episodes.get(str(episode_id))
    if not isinstance(entry, dict):
        entry = {}
    current_label = _review_label(str(row.label))
    next_label = _review_label(label or entry.get("label") or current_label)
    next_discard = bool(discard if discard is not None else entry.get("deleted", row.discard))
    episodes[str(episode_id)] = {"label": next_label, "deleted": next_discard}

    try:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as e:
        state.status_events.append(f"{_format_time(time.time())} tui: failed to write review sidecar: {e}")
        return

    row.label = next_label
    row.discard = next_discard
    state.status_events.append(
        f"{_format_time(time.time())} tui: saved review episode={episode_id} "
        f"label={_display_rollout_label(next_label)} discard={next_discard}"
    )


def _format_rollouts_panel(state: TuiState) -> str:
    rows = _ordered_rollout_rows(state)
    lines = [
        "[b]Rollouts[/b]",
        "s: mark latest success   f: mark latest fail   d: toggle latest discard",
        "",
        (
            f"{'rollout':>7}  {'rollout_start':19}  {'critical_start':>14}  "
            f"{'critical_end':>12}  {'label':>7}  {'discard':>7}"
        ),
        "-" * 80,
    ]
    if not rows:
        lines.append("No rollout rows yet")
        return "\n".join(lines)

    for row in rows[-80:]:
        lines.append(
            f"{row.rollout:>7}  "
            f"{_format_datetime(row.rollout_start_ts):19}  "
            f"{_format_seconds(row.critical_start_s):>14}  "
            f"{_format_seconds(row.critical_end_s):>12}  "
            f"{_display_rollout_label(row.label):>7}  "
            f"{str(bool(row.discard)).lower():>7}"
        )
    return "\n".join(lines)


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


def _put_latest(queue: Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    try:
        queue.put_nowait(event)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(event)
        except Full:
            pass


def _drain_queue(queue: Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while True:
        try:
            events.append(queue.get_nowait())
        except Empty:
            return events


async def _trajectory_listener_loop(
    ws_url: str,
    event_queue: Queue[dict[str, Any]],
    stop_event: threading.Event,
) -> None:
    try:
        import websockets
    except ImportError:
        _put_latest(
            event_queue,
            {
                "type": "trajectory_status",
                "status": "disabled",
                "error": "websockets package is not installed",
            },
        )
        return

    while not stop_event.is_set():
        _put_latest(event_queue, {"type": "trajectory_status", "status": "connecting", "error": ""})
        try:
            async with websockets.connect(ws_url) as websocket:
                _put_latest(event_queue, {"type": "trajectory_status", "status": "connected", "error": ""})
                while not stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        continue

                    try:
                        event = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        _put_latest(event_queue, event)
        except Exception as e:
            if not stop_event.is_set():
                _put_latest(
                    event_queue,
                    {
                        "type": "trajectory_status",
                        "status": "disconnected",
                        "error": str(e),
                    },
                )
                await asyncio.sleep(1.0)

    _put_latest(event_queue, {"type": "trajectory_status", "status": "stopped", "error": ""})


def _run_trajectory_listener(
    ws_url: str,
    event_queue: Queue[dict[str, Any]],
    stop_event: threading.Event,
) -> None:
    asyncio.run(_trajectory_listener_loop(ws_url, event_queue, stop_event))


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


def _line_chart(
    values: deque[float],
    *,
    width: int = LOSS_CHART_WIDTH,
    height: int = LOSS_CHART_HEIGHT,
) -> list[str]:
    if not values:
        return [" " * (LOSS_AXIS_WIDTH + 2) + "no samples yet"]

    selected = list(values)[-width:]
    low = min(selected)
    high = max(selected)
    rows = [[" "] * len(selected) for _ in range(height)]

    if high == low:
        row_indices = [height // 2] * len(selected)
    else:
        span = high - low
        row_indices = [
            height - 1 - int(round((value - low) / span * (height - 1)))
            for value in selected
        ]

    for column, row in enumerate(row_indices):
        rows[row][column] = "•"

    lines: list[str] = []
    for row, cells in enumerate(rows):
        if height == 1:
            axis_value = selected[-1]
        else:
            axis_value = high - ((high - low) * row / (height - 1))
        lines.append(f"{axis_value:>{LOSS_AXIS_WIDTH}.4g} ┤{''.join(cells).rstrip()}")
    return lines


def _history_summary(values: deque[float]) -> str:
    if not values:
        return "samples=0 latest=n/a min=n/a max=n/a"
    selected = list(values)
    return (
        f"samples={len(selected)} latest={selected[-1]:.6f} "
        f"min={min(selected):.6f} max={max(selected):.6f}"
    )


def _format_loss_series(label: str, values: deque[float]) -> str:
    chart = "\n".join(_line_chart(values))
    return f"[b]{label}[/b] {_history_summary(values)}\n{chart}"


def _format_loss_chart(actor_values: deque[float], critic_values: deque[float]) -> str:
    return (
        "[b]RLT Loss Chart[/b]\n"
        f"{_format_loss_series('Actor', actor_values)}\n"
        f"{_format_loss_series('Critic', critic_values)}"
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _resample_values(values: list[float], width: int) -> list[float]:
    if width <= 0:
        return []
    if len(values) <= width:
        return values
    if width == 1:
        return [values[-1]]

    sampled: list[float] = []
    last_index = len(values) - 1
    for column in range(width):
        source_index = round(column * last_index / (width - 1))
        sampled.append(values[source_index])
    return sampled


def _barline(values: list[float], *, width: int = TRAJECTORY_CHART_WIDTH) -> str:
    if not values:
        return "n/a"

    ticks = "▁▂▃▄▅▆▇█"
    selected = _resample_values(values, width)
    low = min(selected)
    high = max(selected)
    if high == low:
        return ticks[0] * len(selected)

    scale = (len(ticks) - 1) / (high - low)
    return "".join(ticks[int((value - low) * scale)] for value in selected)


def _format_rtc_params(chunk: TrajectoryChunk | None) -> str:
    if chunk is None:
        return "RTC: no chunks yet"
    if not chunk.rtc_params:
        return f"RTC: frozen_len={chunk.frozen_len} params=n/a"

    params = chunk.rtc_params
    h_value = params.get("H", len(chunk.actions))
    delay = params.get("d", chunk.frozen_len)
    overlap_end = params.get("overlap_end", h_value)
    schedule = params.get("schedule", "unknown")
    sigma_d = params.get("sigma_d", "auto")
    max_guidance_weight = params.get("max_guidance_weight", "auto")
    return (
        "RTC: "
        f"H={h_value} d={delay} overlap_end={overlap_end} "
        f"schedule={schedule} sigma_d={sigma_d} max_beta={max_guidance_weight}"
    )


def _format_prefix_weights(chunk: TrajectoryChunk | None) -> str:
    if chunk is None or not chunk.prefix_weights:
        return "Prefix weights: n/a"
    return f"Prefix weights: {_barline(chunk.prefix_weights)}"


def _trajectory_dim_count(chunks: deque[TrajectoryChunk], executed_actions: deque[ExecutedAction]) -> int:
    for chunk in reversed(chunks):
        if chunk.actions:
            return min(len(chunk.actions[0]), TRAJECTORY_MAX_DIMS)
    for executed in reversed(executed_actions):
        if executed.action:
            return min(len(executed.action), TRAJECTORY_MAX_DIMS)
    return 0


def _trajectory_step_window(
    chunks: deque[TrajectoryChunk],
    executed_actions: deque[ExecutedAction],
) -> tuple[int, int] | None:
    starts: list[int] = []
    ends: list[int] = []
    for chunk in chunks:
        starts.append(chunk.source_step)
        ends.append(chunk.source_step + len(chunk.actions) - 1)
    for executed in executed_actions:
        starts.append(executed.step)
        ends.append(executed.step)

    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _step_to_column(step: int, start_step: int, end_step: int, width: int) -> int:
    if end_step <= start_step:
        return 0
    return int(round((step - start_step) / (end_step - start_step) * (width - 1)))


def _value_to_row(value: float, low: float, high: float, height: int) -> int:
    if high == low:
        return height // 2
    row = height - 1 - int(round((value - low) / (high - low) * (height - 1)))
    return max(0, min(height - 1, row))


def _format_trajectory_dimension(
    chunks: deque[TrajectoryChunk],
    executed_actions: deque[ExecutedAction],
    dim: int,
    *,
    width: int = TRAJECTORY_CHART_WIDTH,
    height: int = TRAJECTORY_CHART_HEIGHT,
) -> str:
    window = _trajectory_step_window(chunks, executed_actions)
    if window is None:
        return f"joint {dim}: no trajectory samples"

    start_step, end_step = window
    values: list[float] = []
    for chunk in chunks:
        for action in chunk.actions:
            if dim < len(action):
                values.append(action[dim])
    for executed in executed_actions:
        if dim < len(executed.action):
            values.append(executed.action[dim])
    if not values:
        return f"joint {dim}: no trajectory samples"

    low = min(values)
    high = max(values)
    rows = [[" "] * width for _ in range(height)]

    for chunk_index, chunk in enumerate(chunks):
        marker = TRAJECTORY_CHUNK_MARKERS[chunk_index % len(TRAJECTORY_CHUNK_MARKERS)]
        for offset, action in enumerate(chunk.actions):
            if dim >= len(action):
                continue
            step = chunk.source_step + offset
            column = _step_to_column(step, start_step, end_step, width)
            row = _value_to_row(action[dim], low, high, height)
            rows[row][column] = marker

    for executed in executed_actions:
        if dim >= len(executed.action):
            continue
        column = _step_to_column(executed.step, start_step, end_step, width)
        row = _value_to_row(executed.action[dim], low, high, height)
        rows[row][column] = "*"

    lines = [f"joint {dim} latest={values[-1]:.4g} min={low:.4g} max={high:.4g}"]
    for row, cells in enumerate(rows):
        axis_value = high - ((high - low) * row / (height - 1)) if height > 1 else values[-1]
        lines.append(f"{axis_value:>9.4g} ┤{''.join(cells).rstrip()}")
    return "\n".join(lines)


def _format_trajectory_legend(chunks: deque[TrajectoryChunk]) -> str:
    if not chunks:
        return "Chunks: none"

    parts: list[str] = []
    for index, chunk in enumerate(chunks):
        marker = TRAJECTORY_CHUNK_MARKERS[index % len(TRAJECTORY_CHUNK_MARKERS)]
        end_step = chunk.source_step + len(chunk.actions) - 1
        parts.append(f"{marker}:src={chunk.source_step} steps={chunk.source_step}-{end_step}")
    return "Chunks: " + "  ".join(parts[-TRAJECTORY_CHUNKS:])


def _format_trajectory_summary(state: TuiState) -> str:
    latest = state.trajectory_chunks[-1] if state.trajectory_chunks else None
    dim_count = _trajectory_dim_count(state.trajectory_chunks, state.executed_actions)
    latest_age = "n/a" if latest is None else f"{max(0.0, time.time() - latest.timestamp):.1f}s"
    error = f" error={_plain_text(state.trajectory_error)}" if state.trajectory_error else ""
    return (
        f"WebSocket: {state.trajectory_status}{error}\n"
        f"Chunks={len(state.trajectory_chunks)} executed={len(state.executed_actions)} "
        f"dims={dim_count or 'n/a'} latest_age={latest_age}"
    )


def _format_trajectory_panel(state: TuiState) -> str:
    dim_count = _trajectory_dim_count(state.trajectory_chunks, state.executed_actions)
    latest = state.trajectory_chunks[-1] if state.trajectory_chunks else None
    lines = [
        "[b]Trajectory[/b]",
        _format_trajectory_summary(state),
        _format_rtc_params(latest),
        _format_prefix_weights(latest),
        _format_trajectory_legend(state.trajectory_chunks),
    ]
    if dim_count == 0:
        lines.append("\nWaiting for action_chunk or executed_action messages...")
        return "\n".join(lines)

    lines.append("\nPer-joint trajectories. chunk markers=0-9, executed=*")
    for dim in range(dim_count):
        lines.append(_format_trajectory_dimension(state.trajectory_chunks, state.executed_actions, dim))
    return "\n".join(lines)


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
    trajectory_ws_url: str | None,
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
            height: 12;
            margin: 0 1 1 1;
        }

        #recent_panel {
            border: round $accent;
            padding: 0 1;
            height: 1fr;
            margin: 0 1 1 1;
        }

        #trajectory_panel {
            height: 1fr;
            padding: 1 1;
        }

        #rollouts_panel {
            height: 1fr;
            padding: 1 1;
        }

        #logs {
            height: 1fr;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("left", "show_main", "Main"),
            ("right", "show_logs", "Logs"),
            ("t", "show_trajectory", "Trajectory"),
            ("r", "show_rollouts", "Rollouts"),
            ("q", "quit_app", "Quit"),
            ("2", "robot_start_rollout", "Start rollout"),
            ("3", "robot_start_critical", "Start critical"),
            ("4", "robot_end_critical", "End critical"),
            ("5", "robot_intervention", "Critical intervention"),
            ("1", "robot_success", "Critical success"),
            ("0", "robot_failure", "Critical failure"),
            ("9", "robot_discard", "Discard critical"),
            ("8", "robot_end_rollout", "End rollout"),
            ("s", "rollout_mark_success", "Review success"),
            ("f", "rollout_mark_failure", "Review fail"),
            ("d", "rollout_toggle_discard", "Review discard"),
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
            self.trajectory_events: Queue[dict[str, Any]] | None = None
            self.trajectory_stop_event: threading.Event | None = None
            self.trajectory_thread: threading.Thread | None = None
            if trajectory_ws_url:
                self.state.trajectory_status = "connecting"
                self.trajectory_events = Queue(maxsize=TRAJECTORY_QUEUE_SIZE)
                self.trajectory_stop_event = threading.Event()
                self.trajectory_thread = threading.Thread(
                    target=_run_trajectory_listener,
                    args=(trajectory_ws_url, self.trajectory_events, self.trajectory_stop_event),
                    name="drtc_tui_trajectory_listener",
                    daemon=True,
                )
                self.trajectory_thread.start()

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
                with TabPane("Trajectory", id="trajectory"):
                    yield Static(id="trajectory_panel")
                with TabPane("Rollouts", id="rollouts"):
                    yield Static(id="rollouts_panel")
                with TabPane("Logs", id="logs_tab"):
                    yield RichLog(id="logs", wrap=True, markup=False, highlight=False)
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(0.2, self.refresh_from_files)
            self.refresh_dashboard()

        def refresh_from_files(self) -> None:
            for line in _update_from_files(self.state, self.status_tail, self.log_tails):
                self.query_one("#logs", RichLog).write(line)

            if self.trajectory_events is not None:
                for event in _drain_queue(self.trajectory_events):
                    self.state.apply_trajectory_event(event)

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
            critical_recorded = max(
                int(client.get("critical_phases_recorded") or client.get("episodes_recorded") or 0),
                int(server.get("rlt_completed_episodes") or 0),
            )
            critical_discarded = int(
                client.get("critical_phases_discarded") or client.get("episodes_discarded") or 0
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
                "[b]Rollout / Critical[/b]\n"
                f"Rollout: {client.get('rollout_id', 'n/a')} "
                f"({_yes_no(client.get('rollout_open'))})\n"
                f"Critical: {client.get('critical_phase_id', client.get('episode_id', 'n/a'))} "
                f"({_yes_no(client.get('critical_phase_open', client.get('episode_open')))})\n"
                f"Pending label: {_yes_no(client.get('critical_pending_label'))}\n"
                f"Recorded: {critical_recorded}  Discarded: {critical_discarded}\n"
                f"Last label: {last_label}\n"
                "Current transitions: "
                f"{client.get('current_critical_transitions', client.get('current_episode_transitions', 'n/a'))}"
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
                "2: start VLA rollout\n"
                "5: start/end critical + intervention\n"
                "0: critical failure/keep\n"
                "9: discard critical\n"
                "8: end VLA rollout\n"
                "r: rollouts table\n\n"
                "Left/Right: tabs   q: quit"
            )
            self.query_one("#loss_panel", Static).update(
                _format_loss_chart(self.state.actor_loss_history, self.state.critic_loss_history)
            )
            recent = "\n".join(self.state.status_events) or "No status events yet"
            self.query_one("#recent_panel", Static).update("[b]Recent Status[/b]\n" + recent)
            self.query_one("#trajectory_panel", Static).update(_format_trajectory_panel(self.state))
            self.query_one("#rollouts_panel", Static).update(_format_rollouts_panel(self.state))

        def action_show_main(self) -> None:
            self.query_one("#tabs", TabbedContent).active = "main"

        def action_show_logs(self) -> None:
            self.query_one("#tabs", TabbedContent).active = "logs_tab"

        def action_show_trajectory(self) -> None:
            self.query_one("#tabs", TabbedContent).active = "trajectory"

        def action_show_rollouts(self) -> None:
            self.query_one("#tabs", TabbedContent).active = "rollouts"

        def action_quit_app(self) -> None:
            self.exit(130)

        def on_unmount(self) -> None:
            if self.trajectory_stop_event is not None:
                self.trajectory_stop_event.set()

        def _send_robot_command(self, command: str) -> None:
            _write_control_command(control_file, command, self.state)
            self.refresh_dashboard()

        def action_robot_start_rollout(self) -> None:
            self._send_robot_command("start_rollout")

        def action_robot_start_critical(self) -> None:
            self._send_robot_command("start_critical_phase")

        def action_robot_end_critical(self) -> None:
            self._send_robot_command("end_critical_phase")

        def action_robot_intervention(self) -> None:
            self._send_robot_command("toggle_critical_intervention")

        def action_robot_success(self) -> None:
            self._send_robot_command("success")

        def action_robot_failure(self) -> None:
            self._send_robot_command("failure")

        def action_robot_discard(self) -> None:
            self._send_robot_command("discard_episode")

        def action_robot_end_rollout(self) -> None:
            self._send_robot_command("end_rollout")

        def action_rollout_mark_success(self) -> None:
            _write_rollout_review_edit(self.state, label="success")
            self.refresh_dashboard()

        def action_rollout_mark_failure(self) -> None:
            _write_rollout_review_edit(self.state, label="failure")
            self.refresh_dashboard()

        def action_rollout_toggle_discard(self) -> None:
            row = _latest_reviewable_rollout(self.state)
            _write_rollout_review_edit(self.state, discard=not bool(row.discard) if row is not None else None)
            self.refresh_dashboard()

    return int(DrtcTuiApp().run())


def main() -> int:
    parser = argparse.ArgumentParser(description="DRTC experiment TUI")
    parser.add_argument("--status-file", required=True, type=Path)
    parser.add_argument("--control-file", type=Path, default=None)
    parser.add_argument("--client-log-file", required=True, type=Path)
    parser.add_argument("--server-log-file", required=True, type=Path)
    parser.add_argument("--watch-pid", type=int, default=None)
    parser.add_argument(
        "--trajectory-ws-url",
        default=None,
        help="Optional trajectory visualization WebSocket URL, for example ws://localhost:8089",
    )
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
        trajectory_ws_url=args.trajectory_ws_url,
    )


if __name__ == "__main__":
    raise SystemExit(main())

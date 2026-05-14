#!/usr/bin/env python

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import logging
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from contextlib import suppress
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

from lerobot.async_inference.utils.trajectory_viz import encode_image_for_viz
from lerobot.robots.squint_so101 import SquintSO101Robot, SquintSO101RobotConfig
from lerobot.robots.squint_so101.squint_so101 import ACTION_KEYS
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "examples/experiments/configs/baseline_tinypi05v2_rlt_sim_paper_online.yaml"
DEFAULT_WATCH_PATHS = ("src/lerobot/robots/squint_so101",)


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


class CameraVizServer:
    def __init__(self, *, http_port: int, ws_port: int):
        self.http_port = int(http_port)
        self.ws_port = int(ws_port)
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=8)
        self._control_queue: Queue[dict[str, Any]] = Queue(maxsize=128)
        self._clients: set[Any] = set()
        self._shutdown = threading.Event()
        self._http_server: HTTPServer | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for target, name in (
            (self._run_http_server, "squint_camera_viz_http"),
            (self._run_ws_server_thread, "squint_camera_viz_ws"),
        ):
            thread = threading.Thread(target=target, daemon=True, name=name)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._shutdown.set()
        if self._http_server is not None:
            self._http_server.shutdown()

    def drain_controls(self) -> list[dict[str, Any]]:
        controls: list[dict[str, Any]] = []
        while True:
            try:
                controls.append(self._control_queue.get_nowait())
            except Empty:
                return controls

    def on_observation(
        self,
        observation: dict[str, Any],
        *,
        step: int,
        action: dict[str, float],
        camera_state: dict[str, Any] | None = None,
        camera_status: str | None = None,
    ) -> None:
        cameras: list[dict[str, Any]] = []
        for name, value in observation.items():
            if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3:
                continue
            if value.shape[-1] != 3:
                continue
            encoded = encode_image_for_viz(value, max_width=480, jpeg_quality=70)
            if encoded is None:
                continue
            cameras.append({"name": str(name), **encoded})
        if not cameras:
            return

        payload = {
            "type": "frame",
            "timestamp": time.time(),
            "step": int(step),
            "cameras": sorted(cameras, key=lambda item: item["name"]),
            "action": action,
        }
        if camera_state is not None:
            payload["camera_state"] = camera_state
        if camera_status is not None:
            payload["camera_status"] = camera_status
        try:
            self._queue.put_nowait(payload)
        except Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(payload)
            except Empty:
                pass

    def _run_http_server(self) -> None:
        html_dir = Path(__file__).parent

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(html_dir), **kwargs)

            def log_message(self, format, *args):  # noqa: A002
                pass

            def do_GET(self):
                if self.path in {"/", "/index.html"}:
                    self.path = "/squint_sim_teleop.html"
                return super().do_GET()

        self._http_server = ReusableHTTPServer(("0.0.0.0", self.http_port), Handler)
        self._http_server.serve_forever()

    def _run_ws_server_thread(self) -> None:
        asyncio.run(self._run_ws_server())

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            async for message in websocket:
                self._on_client_message(message)
        finally:
            self._clients.discard(websocket)

    async def _broadcast_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(0.01)
            try:
                payload = self._queue.get_nowait()
            except Empty:
                continue
            if not self._clients:
                continue
            message = json.dumps(payload)
            await asyncio.gather(
                *[client.send(message) for client in self._clients],
                return_exceptions=True,
            )

    async def _run_ws_server(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed; camera viewer disabled")
            return

        async with websockets.serve(self._handler, "0.0.0.0", self.ws_port):
            await self._broadcast_loop()

    def _on_client_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or payload.get("type") != "camera_control":
            return
        try:
            self._control_queue.put_nowait(payload)
        except Full:
            with suppress(Empty):
                self._control_queue.get_nowait()
            with suppress(Full):
                self._control_queue.put_nowait(payload)


class TerminalKeyboardJointTeleop:
    def __init__(self, *, joint_step: float, gripper_step: float):
        self.joint_step = float(joint_step)
        self.gripper_step = float(gripper_step)
        self._fd: int | None = None
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> TerminalKeyboardJointTeleop:
        if not sys.stdin.isatty():
            raise RuntimeError("keyboard teleop requires an interactive TTY")
        self._fd = sys.stdin.fileno()
        while True:
            try:
                self._old_settings = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                break
            except termios.error as exc:
                if exc.args and exc.args[0] == errno.EINTR:
                    continue
                raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    @staticmethod
    def help_text() -> str:
        return (
            "Keyboard teleop: a/d pan, w/s shoulder, j/l elbow, i/k wrist flex, "
            "u/o wrist roll, z/x gripper, r reset, p print action, q quit"
        )

    def apply(self, action: dict[str, float]) -> tuple[dict[str, float], str | None]:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                return action, None
            ch = sys.stdin.read(1)
            if ch == "q":
                return action, "quit"
            if ch == "r":
                return action, "reset"
            if ch == "p":
                return action, "print"
            self._apply_key(action, ch)

    def _apply_key(self, action: dict[str, float], ch: str) -> None:
        bindings: dict[str, tuple[str, float]] = {
            "a": ("shoulder_pan.pos", -self.joint_step),
            "d": ("shoulder_pan.pos", self.joint_step),
            "w": ("shoulder_lift.pos", self.joint_step),
            "s": ("shoulder_lift.pos", -self.joint_step),
            "j": ("elbow_flex.pos", -self.joint_step),
            "l": ("elbow_flex.pos", self.joint_step),
            "i": ("wrist_flex.pos", self.joint_step),
            "k": ("wrist_flex.pos", -self.joint_step),
            "u": ("wrist_roll.pos", -self.joint_step),
            "o": ("wrist_roll.pos", self.joint_step),
            "z": ("gripper.pos", -self.gripper_step),
            "x": ("gripper.pos", self.gripper_step),
        }
        binding = bindings.get(ch)
        if binding is None:
            return
        key, delta = binding
        action[key] = float(action.get(key, 0.0) + delta)


def _load_robot_config(args: argparse.Namespace) -> SquintSO101RobotConfig:
    if args.config:
        from examples.experiments.run_drtc_experiment import create_robot_config, load_experiments_from_yaml

        configs = load_experiments_from_yaml(Path(args.config))
        if not configs:
            raise ValueError(f"No experiments found in {args.config}")
        robot_config = create_robot_config(configs[0])
        if not isinstance(robot_config, SquintSO101RobotConfig):
            raise TypeError(f"Config {args.config} does not select robot_type=squint_so101")
    else:
        robot_config = SquintSO101RobotConfig()

    if args.env_id:
        robot_config.env_id = args.env_id
    if args.dataset_root:
        robot_config.dataset_root = args.dataset_root
    if args.marker_xy_offset is not None:
        robot_config.marker_xy_offset = args.marker_xy_offset
    if args.marker_yaw_degrees is not None:
        robot_config.marker_yaw_degrees = args.marker_yaw_degrees
    if args.sensor_width is not None:
        robot_config.sensor_width = args.sensor_width
    if args.sensor_height is not None:
        robot_config.sensor_height = args.sensor_height
    if args.max_episode_steps is not None:
        robot_config.max_episode_steps = args.max_episode_steps
    robot_config.video_dir = args.video_dir
    robot_config.video_every_episodes = args.video_every_episodes
    return robot_config


def _action_from_observation(observation: dict[str, Any]) -> dict[str, float]:
    return {key: float(observation[key]) for key in ACTION_KEYS if key in observation}


def _clip_action(robot: SquintSO101Robot, action: dict[str, float]) -> dict[str, float]:
    low = getattr(robot, "_unit_low", None)
    high = getattr(robot, "_unit_high", None)
    if low is None or high is None:
        return action
    clipped = dict(action)
    for index, key in enumerate(ACTION_KEYS):
        if key not in clipped:
            continue
        clipped[key] = float(np.clip(clipped[key], float(low[index]), float(high[index])))
    return clipped


def _canonical_camera_name(name: Any) -> str | None:
    name_l = str(name or "").lower()
    if name_l in {"top", "render", "render_camera"}:
        return "top"
    if name_l in {"side", "base", "base_camera"}:
        return "side"
    return None


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return fallback.astype(np.float64)
    return vector / norm


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def _camera_control_defaults(robot: SquintSO101Robot) -> dict[str, dict[str, list[float]]]:
    env = getattr(robot, "_env", None)
    unwrapped = getattr(env, "unwrapped", env)
    side_settings = getattr(unwrapped, "base_camera_settings", None) or {}

    top_eye = [0.5, 0.3, 0.35]
    top_target = [0.3, 0.0, 0.1]
    if getattr(unwrapped, "target_type", None) == "marker":
        with suppress(Exception):
            from lerobot.robots.squint_so101.sim.envs import place

            top_eye = list(place.MARKER_TOP_CAMERA_POS)
            top_target = list(place.MARKER_TOP_CAMERA_TARGET)

    return {
        "top": {
            "eye": [float(value) for value in top_eye],
            "target": [float(value) for value in top_target],
        },
        "side": {
            "eye": [float(value) for value in side_settings.get("pos", [0.6, 0.3, 0.3])],
            "target": [float(value) for value in side_settings.get("target", [0.3, 0.0, 0.05])],
        },
    }


def _ensure_camera_control_state(robot: SquintSO101Robot) -> dict[str, dict[str, list[float]]]:
    env = getattr(robot, "_env", None)
    unwrapped = getattr(env, "unwrapped", env)
    if not hasattr(unwrapped, "_teleop_camera_defaults"):
        defaults = _camera_control_defaults(robot)
        setattr(unwrapped, "_teleop_camera_defaults", defaults)
        setattr(unwrapped, "_teleop_camera_state", json.loads(json.dumps(defaults)))
    return getattr(unwrapped, "_teleop_camera_state")


def _set_camera_pose(
    robot: SquintSO101Robot,
    camera: str,
    eye: list[float] | np.ndarray,
    target: list[float] | np.ndarray,
) -> None:
    env = getattr(robot, "_env", None)
    unwrapped = getattr(env, "unwrapped", env)
    eye_values = [float(value) for value in eye]
    target_values = [float(value) for value in target]
    state = _ensure_camera_control_state(robot)
    state[camera] = {"eye": eye_values, "target": target_values}

    if camera == "side":
        if hasattr(unwrapped, "base_camera_settings"):
            unwrapped.base_camera_settings = {"pos": eye_values, "target": target_values}
        if hasattr(unwrapped, "camera_mount") and hasattr(unwrapped, "sample_camera_poses"):
            unwrapped.camera_mount.set_pose(unwrapped.sample_camera_poses(n=unwrapped.num_envs))
            if getattr(unwrapped, "gpu_sim_enabled", False):
                unwrapped.scene._gpu_apply_all()
        return

    with suppress(Exception):
        if "render_camera" not in unwrapped.scene.human_render_cameras:
            robot._render_rgb()
        from mani_skill.utils import sapien_utils

        camera_obj = unwrapped.scene.human_render_cameras["render_camera"].camera
        camera_obj.set_local_pose(sapien_utils.look_at(eye_values, target_values).sp)


def _save_camera_poses(robot: SquintSO101Robot, selected_camera: str | None = None) -> Path:
    env = getattr(robot, "_env", None)
    unwrapped = getattr(env, "unwrapped", env)
    state = _ensure_camera_control_state(robot)
    payload = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env_id": getattr(robot, "env_id", None),
        "selected_camera": selected_camera,
        "cameras": {
            name: {
                "eye": [float(value) for value in values["eye"]],
                "target": [float(value) for value in values["target"]],
            }
            for name, values in state.items()
        },
    }
    path = Path("outputs/camera_alignment/saved_camera_poses.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    setattr(unwrapped, "_teleop_camera_status", f"Saved camera poses to {path}")
    print(f"Saved camera poses to {path}", flush=True)
    return path


def _apply_camera_control(robot: SquintSO101Robot, payload: dict[str, Any]) -> bool:
    mode = str(payload.get("mode") or "orbit").lower()
    camera = _canonical_camera_name(payload.get("camera"))
    if mode == "save":
        _save_camera_poses(robot, selected_camera=camera)
        return True
    if camera is None:
        return False

    state = _ensure_camera_control_state(robot)
    defaults = getattr(getattr(robot._env, "unwrapped", robot._env), "_teleop_camera_defaults", {})

    if mode == "reset":
        default = defaults.get(camera)
        if not default:
            return False
        _set_camera_pose(robot, camera, default["eye"], default["target"])
        return True

    if mode == "set":
        eye = payload.get("eye")
        target = payload.get("target")
        if not isinstance(eye, list) or not isinstance(target, list) or len(eye) != 3 or len(target) != 3:
            return False
        _set_camera_pose(robot, camera, eye, target)
        return True

    current = state.get(camera)
    if current is None:
        return False
    eye = np.asarray(current["eye"], dtype=np.float64)
    target = np.asarray(current["target"], dtype=np.float64)
    offset = eye - target
    distance = max(float(np.linalg.norm(offset)), 1e-4)
    forward = _normalize(target - eye, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = _normalize(np.cross(forward, world_up), np.array([1.0, 0.0, 0.0], dtype=np.float64))
    up = _normalize(np.cross(right, forward), world_up)

    dx = float(payload.get("dx") or 0.0)
    dy = float(payload.get("dy") or 0.0)
    if mode == "orbit":
        yaw = -dx * 0.006
        pitch = -dy * 0.006
        offset = _rotate_about_axis(offset, world_up, yaw)
        right = _normalize(np.cross(_normalize(-offset, forward), world_up), right)
        offset = _rotate_about_axis(offset, right, pitch)
        eye = target + offset
    elif mode == "pan":
        scale = distance * 0.0016
        delta = (-right * dx + up * dy) * scale
        eye = eye + delta
        target = target + delta
    elif mode == "dolly":
        factor = float(np.exp(np.clip(dy, -240.0, 240.0) * 0.0015))
        eye = target + offset * factor
    elif mode == "move_xy":
        step = float(payload.get("step") or 0.01)
        delta = np.array([dx * step, dy * step, 0.0], dtype=np.float64)
        eye = eye + delta
        target = target + delta
    else:
        return False

    eye[2] = max(float(eye[2]), 0.015)
    target[2] = max(float(target[2]), 0.0)
    _set_camera_pose(robot, camera, eye, target)
    return True


def _camera_control_state(robot: SquintSO101Robot) -> dict[str, Any]:
    state = _ensure_camera_control_state(robot)
    return {
        name: {
            "eye": [round(float(value), 4) for value in values["eye"]],
            "target": [round(float(value), 4) for value in values["target"]],
        }
        for name, values in state.items()
    }


def _camera_control_status(robot: SquintSO101Robot) -> str | None:
    env = getattr(robot, "_env", None)
    unwrapped = getattr(env, "unwrapped", env)
    return getattr(unwrapped, "_teleop_camera_status", None)


def _make_leader(args: argparse.Namespace):
    if args.teleop_mode not in {"so100_leader", "so101_leader"}:
        return None
    if not args.teleop_port:
        raise ValueError("--teleop-port is required for SO leader teleop")
    config = SOLeaderTeleopConfig(
        port=args.teleop_port,
        id=args.teleop_id,
        use_degrees=args.leader_use_degrees,
    )
    teleop = make_teleoperator_from_config(config)
    teleop.connect()
    return teleop


def run_worker(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    robot_config = _load_robot_config(args)
    robot = SquintSO101Robot(robot_config)
    server = CameraVizServer(http_port=args.http_port, ws_port=args.ws_port)
    leader = None
    running = True

    def _handle_signal(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    server.start()
    viewer_url = f"http://localhost:{args.http_port}/?ws_port={args.ws_port}"
    print(f"Squint sim teleop viewer: {viewer_url}", flush=True)
    print(f"Environment: {robot_config.env_id or 'inferred from dataset/task'}", flush=True)

    robot.connect()
    _ensure_camera_control_state(robot)
    observation = robot.get_observation()
    action = _action_from_observation(observation)
    leader = _make_leader(args)

    keyboard_cm = (
        TerminalKeyboardJointTeleop(joint_step=args.joint_step, gripper_step=args.gripper_step)
        if args.teleop_mode == "keyboard"
        else None
    )

    try:
        if keyboard_cm is None:
            return _run_loop(args, robot, server, action, leader, lambda: running)
        with keyboard_cm as keyboard:
            print(keyboard.help_text(), flush=True)

            def _keyboard_action(current: dict[str, float]) -> tuple[dict[str, float], str | None]:
                return keyboard.apply(current)

            return _run_loop(args, robot, server, action, leader, lambda: running, keyboard_action=_keyboard_action)
    finally:
        if leader is not None:
            leader.disconnect()
        robot.disconnect()
        server.stop()


def _run_loop(
    args: argparse.Namespace,
    robot: SquintSO101Robot,
    server: CameraVizServer,
    action: dict[str, float],
    leader: Any,
    keep_running,
    keyboard_action=None,
) -> int:
    period = 1.0 / float(args.fps)
    step = 0
    last_print = 0.0
    while keep_running():
        tick = time.perf_counter()
        command: str | None = None
        for control in server.drain_controls():
            _apply_camera_control(robot, control)
        if keyboard_action is not None:
            action, command = keyboard_action(action)
            if command == "quit":
                break
            if command == "reset":
                robot._reset_episode(seed=robot.config.seed)
                action = _action_from_observation(robot.get_observation())
                print("Reset simulator episode", flush=True)
            elif command == "print":
                print(action, flush=True)
        elif leader is not None:
            action = {key: float(value) for key, value in leader.get_action().items() if key in ACTION_KEYS}

        action = _clip_action(robot, action)
        robot.send_action(action)
        observation = robot.get_observation()
        action.update(_action_from_observation(observation) if args.follow_applied_action else {})
        server.on_observation(
            observation,
            step=step,
            action=action,
            camera_state=_camera_control_state(robot),
            camera_status=_camera_control_status(robot),
        )

        now = time.time()
        if now - last_print > 2.0:
            print(f"step={step} action={json.dumps(action, sort_keys=True)}", flush=True)
            last_print = now
        step += 1
        elapsed = time.perf_counter() - tick
        if elapsed < period:
            time.sleep(period - elapsed)
    return 0


def _snapshot(paths: list[Path]) -> dict[Path, int]:
    mtimes: dict[Path, int] = {}
    for root in paths:
        files = [root] if root.is_file() else [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
        for path in files:
            with suppress(FileNotFoundError):
                mtimes[path] = path.stat().st_mtime_ns
    return mtimes


def _changed(before: dict[Path, int], after: dict[Path, int]) -> list[Path]:
    changed = [path for path, mtime in after.items() if before.get(path) != mtime]
    changed.extend(path for path in before if path not in after)
    return sorted(set(changed))


def run_supervisor(args: argparse.Namespace, argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    watch_paths = [Path(path).resolve() for path in (args.watch_path or DEFAULT_WATCH_PATHS)]
    snapshot = _snapshot(watch_paths)
    worker_argv = [sys.executable, "-m", "lerobot.robots.squint_so101.sim.teleop_dev", "--worker"]
    worker_argv.extend(arg for arg in argv if arg not in {"--no-reload"})
    worker_argv.extend(["--no-reload"])

    child: subprocess.Popen | None = None
    running = True

    def _stop_child() -> None:
        nonlocal child
        if child is None:
            return
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        child = None

    def _handle_signal(_signum, _frame) -> None:
        nonlocal running
        running = False
        _stop_child()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while running:
            if child is None or child.poll() is not None:
                print("Starting Squint sim teleop worker...", flush=True)
                child = subprocess.Popen(worker_argv)
                time.sleep(1.0)

            time.sleep(args.reload_poll_s)
            current = _snapshot(watch_paths)
            changed = _changed(snapshot, current)
            if changed:
                print("Reloading Squint sim teleop worker after changes:", flush=True)
                for path in changed[:8]:
                    print(f"  {path}", flush=True)
                if len(changed) > 8:
                    print(f"  ... {len(changed) - 8} more", flush=True)
                snapshot = current
                _stop_child()
        return 0
    finally:
        _stop_child()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast Squint SO101 sim teleop dev loop")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Experiment YAML to reuse for sim settings")
    parser.add_argument("--env-id", default=None, help="Override sim env id, e.g. SO101PlaceCubeMarker-v1")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root used for task/action stats")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--http-port", type=int, default=8098)
    parser.add_argument("--ws-port", type=int, default=8099)
    parser.add_argument("--teleop-mode", choices=["keyboard", "so100_leader", "so101_leader"], default="keyboard")
    parser.add_argument("--teleop-port", default="")
    parser.add_argument("--teleop-id", default=None)
    parser.add_argument("--leader-use-degrees", action="store_true")
    parser.add_argument("--joint-step", type=float, default=2.0)
    parser.add_argument("--gripper-step", type=float, default=4.0)
    parser.add_argument("--marker-xy-offset", type=float, nargs=2, default=None)
    parser.add_argument("--marker-yaw-degrees", type=float, default=None)
    parser.add_argument("--sensor-width", type=int, default=None)
    parser.add_argument("--sensor-height", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--video-dir", default="outputs/squint_sim_teleop/videos")
    parser.add_argument("--video-every-episodes", type=int, default=0)
    parser.add_argument("--follow-applied-action", action="store_true")
    parser.add_argument("--watch-path", action="append", default=None)
    parser.add_argument("--reload-poll-s", type=float, default=0.75)
    parser.add_argument("--no-reload", action="store_true", help="Run a single worker without file watching")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker or args.no_reload:
        return run_worker(args)
    return run_supervisor(args, argv)


if __name__ == "__main__":
    raise SystemExit(main())

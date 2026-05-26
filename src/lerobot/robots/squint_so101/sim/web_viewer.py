from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

from lerobot.async_inference.utils.trajectory_viz import encode_image_for_viz


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


class CameraVizServer:
    def __init__(self, *, http_port: int, ws_port: int, max_width: int, jpeg_quality: int):
        self.http_port = int(http_port)
        self.ws_port = int(ws_port)
        self.max_width = int(max_width)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=8)
        self._control_queue: Queue[dict[str, Any]] = Queue(maxsize=128)
        self._clients: set[Any] = set()
        self._shutdown = threading.Event()
        self._http_server: HTTPServer | None = None

    def start(self) -> None:
        for target, name in (
            (self._run_http_server, "squint_camera_viz_http"),
            (self._run_ws_server_thread, "squint_camera_viz_ws"),
        ):
            thread = threading.Thread(target=target, daemon=True, name=name)
            thread.start()

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
    ) -> dict[str, float]:
        total_start = time.perf_counter()
        timing: dict[str, float] = {
            "viz_clients": float(len(self._clients)),
        }
        cameras: list[dict[str, Any]] = []
        encode_total_ms = 0.0
        for name, value in observation.items():
            if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3:
                continue
            if value.shape[-1] != 3:
                continue
            encode_start = time.perf_counter()
            encoded = encode_image_for_viz(
                value,
                max_width=self.max_width,
                jpeg_quality=self.jpeg_quality,
            )
            encode_ms = (time.perf_counter() - encode_start) * 1000.0
            encode_total_ms += encode_ms
            timing[f"viz_encode_{name}_ms"] = encode_ms
            if encoded is not None:
                cameras.append({"name": str(name), **encoded})

        timing["viz_encode_total_ms"] = encode_total_ms
        timing["viz_camera_count"] = float(len(cameras))
        if not cameras:
            timing["viz_total_ms"] = (time.perf_counter() - total_start) * 1000.0
            return timing

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
        timing["viz_total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return timing

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
            return

        async with websockets.serve(self._handler, "0.0.0.0", self.ws_port):
            await self._broadcast_loop()

    def _on_client_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or payload.get("type") not in {"camera_control", "sim_control"}:
            return
        try:
            self._control_queue.put_nowait(payload)
        except Full:
            try:
                self._control_queue.get_nowait()
                self._control_queue.put_nowait(payload)
            except Empty:
                pass

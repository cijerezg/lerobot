#!/usr/bin/env python
"""FLUX 3 Action policy server for the fork's sim rig (runs under .venv-flux3 = upstream LeRobot main).

Loads a complete flux3 policy (the reBot base package) or a LoRA checkpoint (pretrained_model dir with
adapter_config.json; its adapter config names the base package) plus the checkpoint's saved processors, and
serves the fork's ``flux3_remote`` policy (lerobot/rl/flux3) over a Unix socket:

    ("hello", {...})                    -> {"chunk_size", "action_dim"}
    ("reset", None)                     -> "ok"            (episode boundary: policy + both processors)
    ("predict", {images: {dataset key: HxWx3 uint8}, state: float32[D], task: str, anchor: float32[D] | None})
                                        -> ("ok", float32[chunk_size, D])  absolute joint commands (degrees)

``anchor`` is the command the robot executes next (the fork's RTC leftover head); in command-delta mode the
chunk integrates from it. Without it the history step anchors on the measured state (its own first-tick rule).

    .venv-flux3/bin/python lerobot/scripts/flux3_server.py --checkpoint <dir> [--socket /tmp/flux3.sock] [--compile]
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.flux3 import Flux3Policy
from lerobot.policies.flux3.processor_flux3 import ObservationHistoryNormalizerProcessorStep
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE

logger = logging.getLogger("flux3_server")
logger.setLevel(logging.INFO)
logger.propagate = False  # lerobot configures the root logger on import; keep our lines regardless
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_handler)


def load(checkpoint: Path, device: torch.device, compile_model: bool, attn_mode: str | None):
    if (checkpoint / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(str(checkpoint))
        logger.info("LoRA checkpoint; base package %s", peft_config.base_model_name_or_path)
        base = Flux3Policy.from_pretrained(peft_config.base_model_name_or_path)
        policy = PeftModel.from_pretrained(base, str(checkpoint), config=peft_config)
        config = base.config
    else:
        policy = Flux3Policy.from_pretrained(str(checkpoint))
        config = policy.config
    config.compile_model = compile_model
    if attn_mode is not None:
        config.attn_mode = attn_mode
    config.device = str(device)
    policy = policy.to(device).eval()
    # The saved pipelines carry the device they were exported on (the base package was built on CPU).
    pre, post = make_pre_post_processors(
        config, pretrained_path=str(checkpoint), preprocessor_overrides={"device_processor": {"device": device.type}}
    )
    history = next(step for step in pre.steps if isinstance(step, ObservationHistoryNormalizerProcessorStep))
    return policy, config, pre, post, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--socket", default="/tmp/flux3.sock")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compile", action="store_true", help="policy.compile_model (lazy torch.compile of the DiT)")
    ap.add_argument("--attn-mode", default=None, choices=["torch", "flash", "cudnn"], help="policy.attn_mode override")
    args = ap.parse_args()
    device = torch.device(args.device)
    policy, config, pre, post, history = load(args.checkpoint.resolve(), device, args.compile, args.attn_mode)
    action_dim, chunk_size = config.action_dim, config.chunk_size
    logger.info(
        "serving %s: chunk %d x %d, cameras %s, %s actions (absolute dims %s), %d steps guidance %s/%s, attn %s, compile %s",
        args.checkpoint, chunk_size, action_dim, config.camera_order, config.action_representation,
        config.delta_absolute_dims, config.num_inference_steps, config.guidance_scale, config.guidance_scale_action,
        config.attn_mode, config.compile_model,
    )

    @torch.no_grad()
    def predict(request: dict) -> np.ndarray:
        obs = {key: np.ascontiguousarray(img) for key, img in request["images"].items()}
        obs[OBS_STATE] = np.asarray(request["state"], dtype=np.float32)
        batch = prepare_observation_for_inference(obs, device, task=request.get("task") or "")
        if request.get("anchor") is not None:
            history.last_command = torch.as_tensor(np.asarray(request["anchor"], dtype=np.float32), device=device)[None]
        prepared = pre(batch)
        chunk = policy.predict_action_chunk(prepared)  # (1, chunk, D) normalized
        commands = post(chunk)  # (1, chunk, D) absolute
        return commands[0].detach().cpu().float().numpy()

    def reset() -> None:
        policy.reset()
        pre.reset()
        post.reset()

    # Warm-up on a rig-shaped observation so the first episode does not pay for lazy init.
    warm = {
        "images": {key: np.zeros((480, 640, 3), dtype=np.uint8) for key in config.camera_order},
        "state": np.zeros(action_dim, dtype=np.float32),
        "task": "warm up",
        "anchor": None,
    }
    for i in range(2):
        t0 = time.perf_counter()
        predict(warm)
        torch.cuda.synchronize() if device.type == "cuda" else None
        logger.info("warm-up %d: %.2f s", i + 1, time.perf_counter() - t0)
    reset()

    if os.path.exists(args.socket):
        os.remove(args.socket)
    with Listener(args.socket, family="AF_UNIX") as listener:
        logger.info("listening on %s", args.socket)
        while True:
            conn = listener.accept()
            logger.info("client connected")
            n, total = 0, 0.0
            try:
                while True:
                    kind, payload = conn.recv()
                    if kind == "hello":
                        conn.send({"chunk_size": chunk_size, "action_dim": action_dim})
                    elif kind == "reset":
                        reset()
                        conn.send("ok")
                    elif kind == "predict":
                        t0 = time.perf_counter()
                        try:
                            out = predict(payload)
                        except Exception as e:  # the client raises; the server keeps serving
                            logger.exception("predict failed")
                            conn.send(("error", repr(e)))
                            continue
                        dt = time.perf_counter() - t0
                        n, total = n + 1, total + dt
                        if n % 20 == 1:
                            logger.info("predict %d: %.3f s (mean %.3f s)", n, dt, total / n)
                        conn.send(("ok", out))
                    else:
                        conn.send(("error", f"unknown request {kind!r}"))
            except (EOFError, ConnectionResetError):
                logger.info("client disconnected after %d predictions (mean %.3f s)", n, total / max(n, 1))
            finally:
                conn.close()


if __name__ == "__main__":
    main()

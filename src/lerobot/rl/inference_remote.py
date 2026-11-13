#!/usr/bin/env python
"""
RTC inference with the policy in another process: one server holds the model, many inference runs call it.

Built for the sim eval (lerobot/scripts/run_sim_eval.sh, 2026-11-12): every layout used to load its own copy
of the checkpoint (86 s alone, 11 GB GPU + ~20 GB RAM each), and several copies sharing the GPU slowed every
call. Here the checkpoint is loaded once and the layouts run as light CPU processes.

    # server: loads the checkpoint, then listens on the socket (the socket file appearing = ready)
    python -m lerobot.rl.inference_remote --serve=true --policy_socket=/tmp/policy.sock \
        --config_path=config_rl_sim.yaml --inference_checkpoint_path=<ckpt>/pretrained_model --output_dir=<fresh>
    # client: inference_async with the policy calls sent to the server; every other flag as inference_async
    python -m lerobot.rl.inference_remote --policy_socket=/tmp/policy.sock --config_path=config_rl_sim.yaml ...

The client is the unchanged RTC runtime: it builds the processors, the batch, the RTC leftover and decodes the
chunk as before, on the CPU. Only ``predict_action_chunk`` crosses the socket, as the preprocessed batch.
The server runs the calls that queued up during its last forward as one batch (``max_batch``; calls batch only
with calls of the same shapes, delay and guided length). Each connection has its own noise generator seeded with
the client's ``cfg.seed``, and a batched row draws from its own generator.
``policy.per_episode_seed`` (a generator per rollout inside the policy) is not carried over.
"""
import logging
import os
import queue
import signal
import time
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener
from threading import Thread

import numpy as np
import torch
from torch import nn

import lerobot.rl.gym_manipulator  # noqa: F401 - registers robot/camera/teleop config choices
import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 - registers MolmoAct2RLConfig
import lerobot.rl.pi05.rl_pi05  # noqa: F401 - registers PI05RLConfig
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.rl.inference_async import _configure_inference_checkpoint
from lerobot.rl.rl_trainer import Trainer
from lerobot.rl.rtc_actor_runtime import _build_inference_policy, act_with_policy_rtc_inference
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RemoteInferenceConfig(TrainRLServerPipelineConfig):
    policy_socket: str | None = None  # unix socket path the server listens on
    serve: bool = False  # True = the server; False = an inference run that calls it
    max_batch: int = 16  # server: most waiting calls to run as one forward


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    return value


def _to_tensor(value, device):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    if isinstance(value, dict):
        return {key: _to_tensor(item, device) for key, item in value.items()}
    return value


def _signature(value):
    """What two calls must share to run as one batch: array shapes, list lengths and every other value."""
    if isinstance(value, np.ndarray):
        return value.shape if value.ndim > 0 else value.item()
    if isinstance(value, dict):
        return tuple((key, _signature(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_signature(item) for item in value)
    if isinstance(value, list):
        return len(value)
    return value


def _concatenate(values):
    """Calls of one signature as one batch: arrays and lists joined along the batch axis, the rest shared."""
    first = values[0]
    if isinstance(first, np.ndarray) and first.ndim > 0:
        return np.concatenate(values)
    if isinstance(first, dict):
        return {key: _concatenate([value[key] for value in values]) for key in first}
    if isinstance(first, list):
        return [item for value in values for item in value]
    return first


class RemotePolicy(nn.Module):
    """What the RTC runtime needs of a policy, with the forward on the server."""

    def __init__(self, config, socket: str, seed: int):
        super().__init__()
        self.config = config
        self._conn = Client(socket, family="AF_UNIX")
        self._conn.send(seed)
        # nn.Module needs one parameter for .to(device) / device queries in the runtime.
        self._device_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        if kwargs.get("prev_chunk_left_over") is not None:
            # The runtime passes the leftover without a batch axis; the server joins calls along it.
            kwargs["prev_chunk_left_over"] = kwargs["prev_chunk_left_over"][None]
        self._conn.send((_to_numpy(batch), _to_numpy(kwargs)))
        return torch.from_numpy(self._conn.recv()).to(self._device_anchor.device)


class RemotePolicyTrainer:
    """The config's trainer, except that the policy it makes is the remote one."""

    def __init__(self, trainer: Trainer, socket: str):
        self._trainer = trainer
        self._socket = socket

    def make_policy(self, cfg) -> nn.Module:
        return RemotePolicy(cfg.policy, self._socket, cfg.seed)

    def __getattr__(self, name):
        return getattr(self._trainer, name)


def _read_requests(conn, requests: queue.Queue, device) -> None:
    generator = torch.Generator(device=device).manual_seed(conn.recv())
    while True:
        try:
            batch, kwargs = conn.recv()
        except EOFError:
            return
        requests.put((conn, generator, batch, kwargs))


def _accept(listener: Listener, requests: queue.Queue, device) -> None:
    while True:
        Thread(target=_read_requests, args=(listener.accept(), requests, device), daemon=True).start()


def serve(cfg: RemoteInferenceConfig) -> None:
    # The device and backend setup of act_with_policy_rtc_inference.
    set_seed(cfg.seed)
    device = get_safe_torch_device(getattr(cfg.policy, "actor_device", None) or cfg.policy.device, log=True)
    cfg.policy.device = str(device)
    cfg.policy.rtc_config.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    policy, _, _ = _build_inference_policy(cfg, Trainer.for_config(cfg), device, should_init_critic=False)

    requests: queue.Queue = queue.Queue()
    listener = Listener(cfg.policy_socket, family="AF_UNIX")
    Thread(target=_accept, args=(listener, requests, device), daemon=True).start()
    logger.info("[POLICY_SERVER] Listening on %s.", cfg.policy_socket)
    calls = 0
    logged = 0
    forwards = 0
    busy = 0.0
    while True:
        # Everything that queued up during the last forward runs as one batch per signature.
        waiting = [requests.get()]
        while len(waiting) < cfg.max_batch and not requests.empty():
            waiting.append(requests.get())
        groups: dict = {}
        for request in waiting:
            groups.setdefault(_signature(request[2:]), []).append(request)
        start = time.perf_counter()
        for group in groups.values():
            conns, generators, batches, kwargs = zip(*group, strict=True)
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                chunks = policy.predict_action_chunk(
                    _to_tensor(_concatenate(batches), device),
                    generator=generators[0] if len(group) == 1 else list(generators),
                    **_to_tensor(_concatenate(kwargs), device),
                )
            for conn, chunk in zip(conns, chunks.cpu().numpy(), strict=True):
                conn.send(chunk[None])
        busy += time.perf_counter() - start
        calls += len(waiting)
        forwards += len(groups)
        if calls - logged >= 100:
            logger.info(
                "[POLICY_SERVER] %d calls served; the last %d: %.3f s per call, %.1f calls per forward, %d waiting.",
                calls, calls - logged, busy / (calls - logged), (calls - logged) / forwards, requests.qsize(),
            )
            logged = calls
            forwards = 0
            busy = 0.0


@parser.wrap()
def remote_cli(cfg: RemoteInferenceConfig):
    cfg.validate()

    log_dir = os.path.join(cfg.output_dir or "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    name = "policy_server" if cfg.serve else f"actor_{cfg.job_name}"
    init_logging(log_file=os.path.join(log_dir, f"{name}.log"), display_pid=False)
    _configure_inference_checkpoint(cfg)
    if cfg.serve:
        serve(cfg)
        return

    logger.info("[INFERENCE] RTC inference against the policy server at %s starting.", cfg.policy_socket)
    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    def _sigint(sig, frame):
        logger.info("[INFERENCE] SIGINT - shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _sigint)

    # The model is on the server; this process only preprocesses and decodes.
    cfg.policy.device = "cpu"
    cfg.policy.actor_device = "cpu"
    try:
        act_with_policy_rtc_inference(
            cfg=cfg,
            trainer=RemotePolicyTrainer(Trainer.for_config(cfg), cfg.policy_socket),
            shutdown_event=shutdown_event,
        )
    finally:
        shutdown_event.set()
        logger.info("[INFERENCE] inference_remote finished.")


if __name__ == "__main__":
    remote_cli()

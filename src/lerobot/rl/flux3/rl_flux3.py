"""FLUX 3 Action behind the fork's RTC runtime.

The policy itself runs in a separate process under upstream LeRobot main (``.venv-flux3``, torch 2.10 +
NATTEN): ``lerobot/scripts/flux3_server.py`` loads the checkpoint and its saved processors and answers
``predict`` / ``reset`` over a Unix socket. This module is the fork side: ``Flux3RemoteConfig`` is the
``policy:`` block of ``config_rl_sim_flux3.yaml`` and ``Flux3RemotePolicy`` is what ``rtc_actor_runtime``
calls (``predict_action_chunk`` / ``reset`` / ``config.rtc_config``). Observations go out as uint8 HWC
images + the raw joint state; the answer is one absolute command chunk (chunk_size, 7) in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.connection import Client

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("flux3_remote")
@dataclass
class Flux3RemoteConfig(PreTrainedConfig):
    task: str = ""
    push_to_hub: bool = False
    # Unix socket of flux3_server.py (lerobot/scripts/flux3_server.py --socket ...).
    server_socket: str = "/tmp/flux3.sock"
    # Rig camera key (after env.features_map) -> dataset camera key the checkpoint was trained on.
    server_camera_map: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.external_0": "observation.images.top",
            "observation.images.wrist_0": "observation.images.wrist",
        }
    )
    chunk_size: int = 32
    n_action_steps: int = 32
    rtc_config: RTCConfig = field(default_factory=lambda: RTCConfig(execution_horizon=16))
    action_encoding: str = "absolute"
    # Operator / scorer subtask script (the sim registry passes these per layout; the policy ignores them).
    eval_subtasks: list[str] = field(default_factory=list)
    eval_home_subtask: str = "return to home"
    eval_subtask_precisions: list[int] | None = None
    eval_subtask_contacts: list[str] | None = None
    eval_home_precision: int = 1
    eval_home_contact: str = "na"
    # inference_utils.bound_policy_actions: per-joint envelope of the demos (config_rl.yaml values, 7 wide).
    action_lag_limits: list[float] | None = None
    action_delta_limits: list[float] | None = None
    action_clamp_limits: list[list[float]] | None = None
    action_step_limits: list[float] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def get_optimizer_preset(self) -> AdamWConfig:
        # TrainPipelineConfig.validate() asks for one even at inference; nothing trains through the fork.
        return AdamWConfig(lr=1e-4)

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            raise ValueError("flux3_remote needs observation.state in input_features")
        for key in self.server_camera_map:
            if key not in self.input_features:
                raise ValueError(f"server_camera_map key {key!r} is not an input feature")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None


class Flux3RemotePolicy(nn.Module):
    """Thin client; the runtime's postprocessor is None, so the chunk returned here goes to the robot."""

    def __init__(self, config: Flux3RemoteConfig):
        super().__init__()
        self.config = config
        self._conn = Client(config.server_socket, family="AF_UNIX")
        self._conn.send(("hello", {"action_dim": self.action_dim}))
        info = self._conn.recv()
        if info.get("chunk_size") != config.chunk_size or info.get("action_dim") != self.action_dim:
            raise RuntimeError(f"flux3 server serves {info}, config expects chunk {config.chunk_size} x {self.action_dim}")
        # nn.Module needs one parameter for .to(device) / device queries in the runtime.
        self._device_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def action_dim(self) -> int:
        return int(self.config.output_features["action"].shape[0])

    def reset(self) -> None:
        self._conn.send(("reset", None))
        self._conn.recv()

    @staticmethod
    def _image_hwc_uint8(img: Tensor) -> np.ndarray:
        img = img.detach().cpu()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        if img.dtype != torch.uint8:
            img = (img.float().clamp(0, 1) * 255).round().to(torch.uint8) if img.max() <= 1.0 else img.round().clamp(0, 255).to(torch.uint8)
        return np.ascontiguousarray(img.numpy())

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict,
        inference_delay: int = 0,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
        **_,
    ) -> Tensor:
        state = batch[OBS_STATE].detach().cpu().float().reshape(-1)[: self.action_dim].numpy()
        # Command-delta anchor = the command the robot executes next (the leftover's head), else the state.
        anchor = None
        if prev_chunk_left_over is not None and prev_chunk_left_over.numel() > 0:
            anchor = prev_chunk_left_over[0, : self.action_dim].detach().cpu().float().numpy()
        request = {
            "images": {dst: self._image_hwc_uint8(batch[src]) for src, dst in self.config.server_camera_map.items()},
            "state": state.astype(np.float32),
            "task": batch.get("task", self.config.task),
            "anchor": None if anchor is None else anchor.astype(np.float32),
        }
        self._conn.send(("predict", request))
        kind, payload = self._conn.recv()
        if kind == "error":
            raise RuntimeError(f"flux3 server: {payload}")
        chunk = torch.from_numpy(np.asarray(payload, dtype=np.float32))  # (chunk_size, action_dim)
        return chunk.unsqueeze(0).to(self._device_anchor.device)

    def forward(self, batch):  # pragma: no cover - training lives upstream
        raise NotImplementedError("flux3_remote is inference only")

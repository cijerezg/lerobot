"""
DRTC Policy Server

This implementation follows the DRTC algorithm with:
- 2-thread architecture (observation receiver + main inference loop)
- SPSC last-write-wins registers for observation/actions handoff

Threading model (2 threads):
- Main thread: inference loop, runs policy, sends actions
- Observation receiver thread: receives observations from clients via gRPC

Example:
```shell
python -m lerobot.async_inference.policy_server_drtc \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --obs_queue_timeout=2
```
"""

import logging
import os
import pickle  # nosec
import threading
import time
from collections import OrderedDict
from concurrent import futures
from contextlib import suppress
from typing import Any

import draccus
import grpc
import numpy as np
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.utils.constants import OBS_STATE

from .configs_drtc import PolicyServerDrtcConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    Observation,
    RemotePolicyConfig,
    TimedObservation,
    get_logger,
    raw_observation_to_observation,
)
from .lww_register import LWWRegister
from .rtc_guidance import AsyncRTCConfig, AsyncRTCProcessor
from .utils.compression import decode_images_from_transport
from .utils.metrics import DiagnosticMetrics, EvActionChunk, Metrics
from .utils.simulation import SpikeDelaySimulator
from .utils.trajectory_viz import TrajectoryVizServer
from .utils.viz_utils import compute_prefix_weights_for_viz

_INITIAL_K = -(2**63)


def _infer_model_action_horizon(policy_config: Any) -> tuple[str, int] | None:
    """Infer the maximum action horizon from a loaded policy config."""
    if policy_config is None:
        return None

    for field_name in ("chunk_size", "n_action_steps", "horizon"):
        value = getattr(policy_config, field_name, None)
        if isinstance(value, int) and value > 0:
            return field_name, value

    return None


class ActionChunkCache:
    """LRU cache for raw action chunks, keyed by source control step (t).

    Used for RTC inpainting: the server caches raw (pre-postprocess) action chunks
    so the client can reference them by source control step + index range instead
    of sending post-processed actions (which have different dimensions).

    For action_encoding in {"anchor", "delta"} we additionally cache the
    anchor (chunk-start joint state) used to generate each chunk, so the
    server can re-align cached deltas to the *new* anchor at prefix
    reconstruction time (see `align_prev_actions`).
    """

    def __init__(self, max_size: int = 10):
        """Initialize the cache.

        Args:
            max_size: Maximum number of chunks to cache (oldest evicted first).
        """
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._anchors: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._max_size = max_size

    def put(
        self,
        src_step: int,
        raw_actions: torch.Tensor,
        anchor: torch.Tensor | None = None,
    ) -> None:
        """Store a raw action chunk (and optional anchor) keyed by source step.

        Args:
            src_step: The source step (observation timestep) for this chunk.
            raw_actions: Raw action tensor of shape (B, T, A) or (T, A).
            anchor: Optional pre-preprocess joint state used as the anchor when
                this chunk was generated. Required for cross-chunk RTC alignment
                under `anchor` / `delta` action encodings.
        """
        # If already exists, remove it first so it goes to the end (most recent)
        if src_step in self._cache:
            del self._cache[src_step]
            self._anchors.pop(src_step, None)

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            evicted_step, _ = self._cache.popitem(last=False)
            self._anchors.pop(evicted_step, None)

        # Store a detached clone to avoid holding onto computation graph
        self._cache[src_step] = raw_actions.detach().clone()
        if anchor is not None:
            self._anchors[src_step] = anchor.detach().clone()

    def get(self, src_step: int) -> torch.Tensor | None:
        """Retrieve a cached chunk by source step.

        Args:
            src_step: The source step to look up.

        Returns:
            The cached tensor or None if not found.
        """
        return self._cache.get(src_step)

    def get_anchor(self, src_step: int) -> torch.Tensor | None:
        """Retrieve the anchor (chunk-start state) used for `src_step`'s chunk."""
        return self._anchors.get(src_step)

    def clear(self) -> None:
        """Clear all cached chunks."""
        self._cache.clear()
        self._anchors.clear()

class PolicyServerDrtc(services_pb2_grpc.AsyncInferenceServicer):
    """DRTC policy server.

    This implementation follows the 2-thread model from the paper:
    - Main thread: runs the inference loop
    - Observation receiver thread: receives observations from clients via gRPC

    Thread communication uses SPSC last-write-wins registers (keyed by timesteps).
    """

    prefix = "policy_server_drtc"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerDrtcConfig):
        """Initialize the policy server.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.shutdown_event = threading.Event()

        # Diagnostic metrics (console only; avg/max timings).
        diag = DiagnosticMetrics(
            fps=config.fps,
            window_s=config.metrics_diagnostic_window_s,
            interval_s=config.metrics_diagnostic_interval_s,
            enabled=config.metrics_diagnostic_enabled,
            verbose=config.metrics_diagnostic_verbose,
            prefix="DIAG_SERVER",
        )
        diag.start()
        self._metrics = Metrics(experiment=None, diagnostic=diag)

        # SPSC LWW registers
        # - Receiver thread -> inference producer: latest observation (by control_step)
        # - Inference producer -> StreamActionsDense: latest dense actions (by control_step)
        self._obs_reg: LWWRegister[TimedObservation | None] = LWWRegister(
            initial_control_step=_INITIAL_K, initial_value=None
        )
        self._action_reg: LWWRegister[services_pb2.ActionsDense | None] = LWWRegister(
            initial_control_step=_INITIAL_K, initial_value=None
        )

        self._policy_ready = threading.Event()
        self._producer_thread: threading.Thread | None = None

        # Policy components (set by SendPolicyInstructions)
        self.device: str | None = None
        self.policy_type: str | None = None
        self.lerobot_features: dict[str, Any] | None = None
        self.actions_per_chunk: int | None = None
        self.policy: Any = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

        # pi05_rl-specific: task / advantage / robot_type to populate
        # `complementary_data` before the pi05_full preprocessor runs. Mirrors
        # what `lerobot.rl.inference_utils.get_actions_worker` injects.
        self._pi05_task_str: str | None = None
        self._pi05_advantage: float = 1.0
        self._pi05_robot_type: str = ""

        # Cross-chunk RTC anchor alignment (anchor / delta encodings):
        # the postprocessor's NormalizerProcessorStep is needed to round-trip
        # cached prefix slices through unnormalize -> shift -> renormalize so
        # they reference the *current* anchor instead of the stale one used
        # when the chunk was generated.
        self._action_normalizer: Any = None
        self._action_encoding: str = "absolute"

        # One-shot debug print of the first few inference chunks. Used to
        # verify the per-chunk anchor reconstruction (anchor / unnormalized
        # delta / sum row 0) matches the standalone reference. Reset to 0
        # by `_reset_server`. Tunable via env LEROBOT_DRTC_DEBUG_CHUNKS.
        try:
            self._debug_chunks_remaining = int(
                os.environ.get("LEROBOT_DRTC_DEBUG_CHUNKS", "5")
            )
        except ValueError:
            self._debug_chunks_remaining = 5

        # Client-driven RTC (optional)
        self._rtc_cfg: AsyncRTCConfig | None = None

        # Action chunk cache for RTC (stores raw actions before postprocessing).
        # Placeholder; resized to match actions_per_chunk in SendPolicyInstructions.
        self._action_cache = ActionChunkCache(max_size=10)

        # Spike delay simulator for experiments
        self._delay_simulator = SpikeDelaySimulator(config=config.mock_spike_config)

        # Trajectory visualization server (HTTP + WebSocket)
        self._trajectory_viz_server: TrajectoryVizServer | None = None
        self._trajectory_viz_thread: threading.Thread | None = None
        if config.trajectory_viz_enabled:
            self._trajectory_viz_server = TrajectoryVizServer(
                ws_port=config.trajectory_viz_ws_port,
                http_port=config.trajectory_viz_http_port,
            )
            self._trajectory_viz_thread = threading.Thread(
                target=self._trajectory_viz_server.start,
                name="trajectory_viz_server",
                daemon=True,
            )
            self._trajectory_viz_thread.start()
            print(
                "Trajectory visualization server started on "
                f"http://0.0.0.0:{config.trajectory_viz_http_port} "
                f"(WebSocket: ws://0.0.0.0:{config.trajectory_viz_ws_port})"
            )

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    @staticmethod
    def _format_timing_parts(timings: dict[str, Any]) -> str:
        parts: list[str] = []
        for key, value in timings.items():
            if isinstance(value, bool):
                parts.append(f"{key}={value}")
            elif isinstance(value, int | float):
                parts.append(f"{key}={float(value):.1f}ms")
            else:
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def _reset_server(self) -> None:
        """Reset server state when a new client connects.

        Joins the old producer thread before reassigning registers so the
        thread doesn't leak (it holds a reader bound to the old register).
        """
        self.shutdown_event.set()
        self._policy_ready.clear()

        # Wait for the old producer thread to observe shutdown and exit
        # before replacing registers, so it doesn't loop forever on the
        # old register after shutdown_event is cleared.
        if self._producer_thread is not None and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=5.0)
            if self._producer_thread.is_alive():
                self.logger.warning(
                    "Producer thread did not exit within 5s during reset; "
                    "a new thread will be started anyway."
                )
        self._producer_thread = None

        # Reset registers (avoid leaking prior session values)
        self._obs_reg = LWWRegister(initial_control_step=_INITIAL_K, initial_value=None)
        self._action_reg = LWWRegister(initial_control_step=_INITIAL_K, initial_value=None)
        self._action_cache.clear()

        try:
            self._debug_chunks_remaining = int(
                os.environ.get("LEROBOT_DRTC_DEBUG_CHUNKS", "5")
            )
        except ValueError:
            self._debug_chunks_remaining = 5

    # -------------------------------------------------------------------------
    # gRPC Service Methods (called by receiver thread)
    # -------------------------------------------------------------------------

    def Ready(self, request, context):  # noqa: N802
        """Handle client ready signal. Resets server state for new session."""
        self._metrics.diagnostic.counter("client_ready", 1)
        self._reset_server()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendTrajectoryChunk(self, request, context):  # noqa: N802
        """Receive trajectory chunk from robot client for visualization."""
        if self._trajectory_viz_server is None:
            return services_pb2.Empty()

        # Decode the packed float32 actions
        num_actions = request.num_actions
        action_dim = request.action_dim
        if num_actions > 0 and action_dim > 0:
            actions_flat = np.frombuffer(request.actions_f32, dtype=np.float32)
            actions = actions_flat.reshape(num_actions, action_dim).tolist()
        else:
            actions = []

        # Create EvActionChunk event and forward to viz server
        event = EvActionChunk(
            src_control_step=request.source_step,  # proto field is source_step
            actions=actions,
            frozen_len=request.frozen_len,
            timestamp=request.timestamp,
        )
        self._trajectory_viz_server.on_chunk(event)

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive and load policy from client instructions."""
        if not self.running:
            return services_pb2.Empty()

        t_total_start = time.perf_counter()

        # Deserialize policy configuration
        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        # Resize RTC chunk cache to match the client's chunk size so we always
        # keep enough history for the full action horizon.
        self._action_cache = ActionChunkCache(max_size=self.actions_per_chunk)

        # Skip loading real policy in mock mode
        if self.config.mock_policy:
            self._metrics.diagnostic.counter("mock_policy_mode", 1)
            self._policy_ready.set()
            return services_pb2.Empty()

        # Load policy
        policy_class = get_policy_class(self.policy_type)

        self.logger.info(
            "Loading policy weights | type=%s | path=%s | device=%s",
            self.policy_type,
            policy_specs.pretrained_name_or_path,
            self.device,
        )
        t_load_start = time.perf_counter()
        if self.policy_type == "pi05_rl":
            # PI05FullPolicy.from_pretrained's state-dict remapper only prepends
            # `model.` and does not strip the `actor.` / `critic.` prefixes used
            # by PI05RLPolicy checkpoints. Going through it loads
            # `models/pi05_base` via __init__'s `pi05_checkpoint` and then
            # silently drops every `actor.*` key on the second load, leaving the
            # model with base pi05 weights instead of the RL fine-tune. Mirror
            # `inference_pi05_async.py`: load the config, point
            # `pi05_checkpoint` at the RL checkpoint itself, and let
            # `PI05RLPolicy.__init__`'s actor/critic split loader handle the
            # safetensors directly.
            from lerobot.configs.policies import PreTrainedConfig

            cfg_obj = PreTrainedConfig.from_pretrained(policy_specs.pretrained_name_or_path)
            cfg_obj.pi05_checkpoint = policy_specs.pretrained_name_or_path
            # DRTC inference only uses the actor; constructing the critic doubles
            # the large PI05 backbone footprint and can exhaust VRAM.
            cfg_obj.use_separate_critic = False
            self.policy = policy_class(cfg_obj)
        elif self.policy_type == "pi05_rlt":
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.rl.rlt_pi05 import PI05RLTConfig

            base_cfg = PreTrainedConfig.from_pretrained(policy_specs.pretrained_name_or_path)
            cfg_obj = PI05RLTConfig.from_base_config(
                base_cfg,
                device=self.device,
                rlt_enabled=bool(getattr(policy_specs, "rlt_enabled", False)),
                rlt_embedding_checkpoint=getattr(policy_specs, "rlt_embedding_checkpoint", None),
                rlt_head_checkpoint=getattr(policy_specs, "rlt_head_checkpoint", None),
                rlt_chunk_size=int(getattr(policy_specs, "rlt_chunk_size", 10)),
                rlt_token_dim=int(getattr(policy_specs, "rlt_token_dim", 2048)),
                rlt_bc_beta=float(getattr(policy_specs, "rlt_bc_beta", 1.0)),
                rlt_reference_dropout_p=float(getattr(policy_specs, "rlt_reference_dropout_p", 0.5)),
                subtask_generation_enabled=False,
                pi05_checkpoint=policy_specs.pretrained_name_or_path,
                rtc_config=None,
            )
            if getattr(policy_specs, "num_flow_matching_steps", None) is not None:
                cfg_obj.num_inference_steps = int(policy_specs.num_flow_matching_steps)
            self.policy = policy_class.from_pretrained(
                policy_specs.pretrained_name_or_path,
                config=cfg_obj,
                strict=False,
            )
        else:
            self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        t_load_done = time.perf_counter()
        self.logger.info(
            "Loaded policy weights in %.1fs | moving to %s ...",
            t_load_done - t_load_start,
            self.device,
        )

        t_to_start = time.perf_counter()
        self.policy.to(self.device)
        t_to_done = time.perf_counter()
        self.logger.info("Moved policy to %s in %.1fs", self.device, t_to_done - t_to_start)

        inferred_horizon = _infer_model_action_horizon(getattr(self.policy, "config", None))
        if inferred_horizon is not None:
            horizon_field, model_horizon = inferred_horizon
            if self.actions_per_chunk > model_horizon:
                raise ValueError(
                    "Requested actions_per_chunk "
                    f"({self.actions_per_chunk}) exceeds model-supported horizon "
                    f"({model_horizon}, from policy config field '{horizon_field}') "
                    f"for checkpoint '{policy_specs.pretrained_name_or_path}'. "
                    f"Set actions_per_chunk <= {model_horizon}."
                )

        # Load preprocessor and postprocessor
        device_override = {"device": self.device}
        self.logger.info("Building pre/post processors ...")
        t_pp_start = time.perf_counter()
        if self.policy_type in ("pi05_rl", "pi05_rlt"):
            # pi05_rl uses a custom processor pipeline ("runtime upgrade" path) that
            # mirrors the standalone `inference_pi05_async.py` setup. We build a
            # minimal cfg shim so `make_pi05_full_processors_with_upgrade` can read
            # the fields it needs from `policy.config` without requiring the full
            # `TrainRLServerPipelineConfig` to be sent over the wire.
            from types import SimpleNamespace

            from lerobot.rl.pi05_train_utils import make_pi05_full_processors_with_upgrade

            shim_cfg = SimpleNamespace(policy=self.policy.config)
            self.preprocessor, self.postprocessor = make_pi05_full_processors_with_upgrade(
                cfg=shim_cfg, dataset=None, is_main_process=True
            )

            # Force eval mode and disable AMP, mirroring inference_pi05_async.py.
            with suppress(Exception):
                self.policy.config.use_amp = False
            self.policy = self.policy.eval()

            # Cache pi05-only fields needed to populate `complementary_data`
            # before each call to `self.preprocessor(...)`.
            self._pi05_task_str = str(getattr(self.policy.config, "task", "") or "")
            self._pi05_advantage = float(getattr(self.policy.config, "inference_advantage", 1.0))
            self._pi05_robot_type = ""

            # Allow the client to override `inference_advantage` per-experiment
            # (e.g. positive vs negative A/B). Mirror the override onto
            # `policy.config` so any downstream code that re-reads it stays
            # consistent.
            adv_override = getattr(policy_specs, "inference_advantage", None)
            if adv_override is not None:
                old_adv = self._pi05_advantage
                self._pi05_advantage = float(adv_override)
                with suppress(Exception):
                    self.policy.config.inference_advantage = self._pi05_advantage
                self.logger.info(
                    "pi05_rl inference_advantage overridden by client: %.4f -> %.4f",
                    old_adv,
                    self._pi05_advantage,
                )
            else:
                self.logger.info(
                    "%s inference_advantage from policy config: %.4f",
                    self.policy_type,
                    self._pi05_advantage,
                )
            if self.policy_type == "pi05_rlt":
                self.logger.info(
                    "pi05_rlt configured | rlt_enabled=%s | embedding=%s | head=%s | "
                    "subtask_generation_enabled=%s",
                    getattr(self.policy.config, "rlt_enabled", False),
                    getattr(self.policy.config, "rlt_embedding_checkpoint", None),
                    getattr(self.policy.config, "rlt_head_checkpoint", None),
                    getattr(self.policy.config, "subtask_generation_enabled", False),
                )
        else:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=policy_specs.pretrained_name_or_path,
                preprocessor_overrides={
                    "device_processor": device_override,
                    "rename_observations_processor": {"rename_map": policy_specs.rename_map},
                },
                postprocessor_overrides={"device_processor": device_override},
            )
        t_pp_done = time.perf_counter()
        self.logger.info("Built pre/post processors in %.1fs", t_pp_done - t_pp_start)

        subtask_interval = getattr(policy_specs, "subtask_regeneration_interval", None)
        if subtask_interval is not None:
            cfg_obj = getattr(self.policy, "config", None)
            if cfg_obj is not None and hasattr(cfg_obj, "subtask_regeneration_interval"):
                old_interval = getattr(cfg_obj, "subtask_regeneration_interval")
                cfg_obj.subtask_regeneration_interval = float(subtask_interval)
                self.logger.info(
                    "subtask_regeneration_interval overridden by client: %s -> %.3f",
                    old_interval,
                    cfg_obj.subtask_regeneration_interval,
                )
            else:
                self._metrics.diagnostic.counter("subtask_regeneration_interval_override_ignored", 1)

        subtask_generation_enabled = getattr(policy_specs, "subtask_generation_enabled", None)
        if subtask_generation_enabled is not None:
            cfg_obj = getattr(self.policy, "config", None)
            if cfg_obj is not None and hasattr(cfg_obj, "subtask_generation_enabled"):
                old_enabled = getattr(cfg_obj, "subtask_generation_enabled")
                cfg_obj.subtask_generation_enabled = bool(subtask_generation_enabled)
                self.logger.info(
                    "subtask_generation_enabled overridden by client: %s -> %s",
                    old_enabled,
                    cfg_obj.subtask_generation_enabled,
                )
            else:
                self._metrics.diagnostic.counter("subtask_generation_enabled_override_ignored", 1)

        # Cache action encoding + locate the preprocessor's NormalizerProcessorStep
        # so we can renormalize aligned RTC prefix slices. The standalone
        # `inference_utils.py:386-403` plucks the same step from
        # `policy.preprocessor.steps` and uses its `_normalize_action`.
        self._action_encoding = str(
            getattr(getattr(self.policy, "config", None), "action_encoding", "absolute") or "absolute"
        )
        self._action_normalizer = None
        if self._action_encoding in ("anchor", "delta"):
            try:
                from lerobot.processor import NormalizerProcessorStep

                self._action_normalizer = next(
                    s for s in self.preprocessor.steps if isinstance(s, NormalizerProcessorStep)
                )
                self.logger.info(
                    "RTC anchor alignment enabled (action_encoding=%s) | normalizer found",
                    self._action_encoding,
                )
            except (StopIteration, ImportError) as e:
                self.logger.warning(
                    "action_encoding=%s but could not locate NormalizerProcessorStep "
                    "in preprocessor.steps (%s); RTC prefix anchor alignment will be skipped.",
                    self._action_encoding,
                    e,
                )
        self._metrics.diagnostic.timing_s("policy_load_ms", t_load_done - t_load_start)
        self._metrics.diagnostic.timing_s("policy_to_ms", t_to_done - t_to_start)
        self._metrics.diagnostic.timing_s("policy_processors_ms", t_pp_done - t_pp_start)
        self._metrics.diagnostic.timing_s("policy_total_ms", time.perf_counter() - t_total_start)

        # Apply num_flow_matching_steps override if provided by client
        # (Alex Soare optimization: Beta should scale with n)
        num_flow_steps = getattr(policy_specs, "num_flow_matching_steps", None)
        if num_flow_steps is not None:
            cfg_obj = getattr(self.policy, "config", None)
            if cfg_obj is not None:
                # PI0/PI05 use num_inference_steps, SmolVLA uses num_steps
                if hasattr(cfg_obj, "num_inference_steps"):
                    cfg_obj.num_inference_steps = num_flow_steps
                elif hasattr(cfg_obj, "num_steps"):
                    cfg_obj.num_steps = num_flow_steps
                else:
                    self._metrics.diagnostic.counter("num_flow_steps_override_ignored", 1)

        # Enable per-chunk model phase timings. The PI05/PI05-RL model code
        # synchronizes CUDA around phase boundaries when this flag is set, so
        # the console breakdown reflects actual GPU time instead of launch time.
        model_value = getattr(self.policy, "model", None)
        if model_value is not None:
            with suppress(Exception):
                model_value._profile_inference = True
            self.logger.info("DRTC per-chunk inference timing enabled")

        # Optional: enable RTC via client instructions (server-side inpainting)
        if getattr(policy_specs, "rtc_enabled", False) and self.policy_type != "pi05_rlt":
            # Handle optional max_guidance_weight (None = use num_flow_matching_steps, Alex Soare opt)
            max_gw_raw = getattr(policy_specs, "rtc_max_guidance_weight", None)
            max_gw = float(max_gw_raw) if max_gw_raw is not None else None

            self._rtc_cfg = AsyncRTCConfig(
                enabled=True,
                prefix_attention_schedule=str(getattr(policy_specs, "rtc_prefix_attention_schedule", "linear")),
                max_guidance_weight=max_gw,
                sigma_d=float(getattr(policy_specs, "rtc_sigma_d", 1.0)),
                full_trajectory_alignment=bool(getattr(policy_specs, "rtc_full_trajectory_alignment", False)),
            )
            # NOTE: We do NOT pass self.postprocessor to RTC guidance because:
            # - RTC operates INSIDE the model's denoising loop in raw action space (e.g. 32 dims)
            # - The postprocessor (NormalizeProcessor) expects executable action space (e.g. 6 dims)
            # - These dimensions are incompatible; the model's action head converts at the end
            # - For now, RTC guidance compares in raw model space (prev must match model dims)
            rtc = AsyncRTCProcessor(self._rtc_cfg, postprocess=None)

            # Flow policies expect `policy.rtc_processor` and `policy.model.rtc_processor`.
            self.policy.rtc_processor = rtc
            model_value = getattr(self.policy, "model", None)
            if model_value is not None:
                model_value.rtc_processor = rtc

            # Satisfy policy-side `_rtc_enabled()` checks without importing RTCConfig.
            cfg_obj = getattr(self.policy, "config", None)
            if cfg_obj is not None:
                with suppress(Exception):
                    cfg_obj.rtc_config = type("RTCConfigShim", (), {"enabled": True})()

        # Apply spike configuration from client (for experiments)
        spikes = getattr(policy_specs, "spikes", [])
        if spikes:
            self._delay_simulator = SpikeDelaySimulator.from_dicts(spikes)
            self._metrics.diagnostic.counter("spike_events_configured", len(spikes))

        # Warmup: run dummy inference passes to trigger CUDA kernel compilation
        # and memory allocation so the first real measurement isn't inflated.
        if self.config.warmup_passes > 0:
            self._warmup_model(num_passes=self.config.warmup_passes)

        self._policy_ready.set()
        self.logger.info(
            "Policy READY | type=%s | total_load=%.1fs | accepting observations",
            self.policy_type,
            time.perf_counter() - t_total_start,
        )

        # Start producer thread (if needed) to generate actions outside the RPC path (lower jitter).
        if self._producer_thread is None or not self._producer_thread.is_alive():
            self._producer_thread = threading.Thread(
                target=self._inference_producer_loop,
                name="policy_server_drtc_inference_producer",
                daemon=True,
            )
            self._producer_thread.start()

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from client and enqueue for inference.

        This method is called by the gRPC receiver thread.
        """
        t_total_start = time.perf_counter()

        # Receive observation bytes (stamp receive_time AFTER full payload
        # arrives so that client-to-server latency captures the actual
        # network transfer of the chunked image payload, not just the
        # gRPC handler dispatch time).
        t_recv_start = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )
        t_recv_done = time.perf_counter()
        receive_time = time.time()

        # Deserialize
        t_deser_start = time.perf_counter()
        timed_observation = pickle.loads(received_bytes)  # nosec
        t_deser_done = time.perf_counter()

        # Decode images
        t_decode_start = time.perf_counter()
        decoded_observation, _ = decode_images_from_transport(timed_observation.get_observation())
        timed_observation.observation = decoded_observation
        t_decode_done = time.perf_counter()

        # Stamp the server receive time for granular latency decomposition
        timed_observation.server_received_ts = receive_time

        obs_control_step = timed_observation.get_control_step()
        obs_timestamp = timed_observation.get_timestamp()

        # Diagnostics
        # Provide a stable `step` field for compact diagnostics.
        self._metrics.diagnostic.set_context(step=obs_control_step, last_obs_step=obs_control_step, chunk_size=self.actions_per_chunk)
        self._metrics.diagnostic.timing_s("obs_recv_ms", t_recv_done - t_recv_start)
        self._metrics.diagnostic.timing_s("deser_ms", t_deser_done - t_deser_start)
        self._metrics.diagnostic.timing_s("obs_decode_ms", t_decode_done - t_decode_start)
        self._metrics.diagnostic.timing_s("obs_one_way_latency_ms", receive_time - obs_timestamp)
        self._metrics.diagnostic.timing_s("obs_total_ms", time.perf_counter() - t_total_start)

        # Publish newest observation (monotone w.r.t. control_step)
        self._obs_reg.update_if_newer(obs_control_step, timed_observation)

        return services_pb2.Empty()

    def StreamActionsDense(self, request, context):  # noqa: N802
        """Server-streaming dense actions RPC (streaming-only action transport)."""
        if not self._policy_ready.is_set():
            return
        reader = self._action_reg.reader(initial_watermark=_INITIAL_K)
        while self.running and context.is_active():
            state, _, is_new = reader.read_if_newer()
            dense = state.value
            if not is_new or dense is None:
                time.sleep(0.01)
                continue
            yield dense

    # -------------------------------------------------------------------------
    # Inference Pipeline
    # -------------------------------------------------------------------------

    def _publish_dense(self, dense: services_pb2.ActionsDense) -> None:
        control_step = int(dense.source_control_step)
        self._action_reg.update_if_newer(control_step, dense)

    def _warmup_model(self, num_passes: int = 2) -> None:
        """Run dummy inference passes to warm up CUDA kernels and memory allocations.

        The first forward pass through a PyTorch model on GPU triggers JIT compilation
        of CUDA kernels and cuDNN workspace allocation, adding hundreds of milliseconds
        to inference time. Running a few dummy passes here ensures this overhead is paid
        during startup, not during the first real measurement.

        Args:
            num_passes: Number of dummy inference passes to run.
        """
        if self.preprocessor is None or self.postprocessor is None:
            self.logger.warning("Cannot warmup: pre/post processors not initialized")
            return
        if self.policy is None:
            self.logger.warning("Cannot warmup: policy not loaded")
            return

        self.logger.info(f"Warming up model with {num_passes} dummy inference pass(es)...")
        t_warmup_start = time.perf_counter()

        try:
            # Build a dummy observation matching the format produced by
            # raw_observation_to_observation(): {OBS_STATE: (1, state_dim), image_keys: (1, C, H, W), task: str}
            dummy_obs: dict[str, Any] = {}

            # State: derive dimensionality from lerobot_features
            if self.lerobot_features:
                state_features = self.lerobot_features.get("observation.state", [])
                state_dim = len(state_features) if isinstance(state_features, (list, tuple)) else 6
            else:
                state_dim = 6
            dummy_obs["observation.state"] = torch.zeros(1, state_dim)

            # Images: use policy's image_features to get (C, H, W) shapes
            for key, feat in self.policy_image_features.items():
                c, h, w = feat.shape
                # After prepare_image + unsqueeze: float32 in [0, 1], shape (1, C, H, W)
                dummy_obs[key] = torch.zeros(1, c, h, w, dtype=torch.float32)

            # Task string (VLA models require this)
            dummy_obs["task"] = "warmup"

            for i in range(num_passes):
                t_pass_start = time.perf_counter()

                # Preprocess (inject pi05_rl complementary_data when applicable)
                pass_obs = self._inject_pi05_complementary_data(dict(dummy_obs))
                obs = self.preprocessor(pass_obs)

                # Inference -- call policy directly (not _get_action_chunk)
                # to avoid recording warmup timings in diagnostic metrics.
                with torch.no_grad():
                    action_tensor = self.policy.predict_action_chunk(obs)

                # Postprocess (same path as real inference)
                if action_tensor.ndim != 3:
                    action_tensor = action_tensor.unsqueeze(0)
                action_tensor = action_tensor[:, : self.actions_per_chunk, :]
                b, t_dim, a = action_tensor.shape
                flat = action_tensor.reshape(b * t_dim, a)
                flat = self.postprocessor(flat)

                t_pass_done = time.perf_counter()
                self.logger.info(
                    f"  Warmup pass {i + 1}/{num_passes}: {(t_pass_done - t_pass_start) * 1000:.1f}ms"
                )

            t_warmup_done = time.perf_counter()
            warmup_total_ms = (t_warmup_done - t_warmup_start) * 1000
            self.logger.info(f"Model warmup complete ({warmup_total_ms:.0f}ms total)")
            self._metrics.diagnostic.timing_ms("warmup_total_ms", warmup_total_ms)

        except Exception as e:
            self.logger.error(f"Warmup failed (non-fatal, first inference may be slow): {e}")
            self._metrics.diagnostic.counter("warmup_failed", 1)

    def _inference_producer_loop(self) -> None:
        """Continuously produce the latest action chunk from the latest observation (low jitter)."""
        reader = self._obs_reg.reader(initial_watermark=_INITIAL_K)
        consecutive_errors = 0

        while self.running:
            if not self._policy_ready.is_set():
                time.sleep(0.01)
                continue

            state, _, is_new = reader.read_if_newer()
            obs = state.value
            if not is_new or obs is None:
                time.sleep(0.01)
                continue

            try:
                t_total_start = time.perf_counter()

                # Apply simulated delay (for experiments)
                self._delay_simulator.apply_delay()

                t_infer_start = time.perf_counter()

                # Use mock policy or real policy
                if self.config.mock_policy:
                    dense = self._mock_predict_action_chunk_dense(obs)
                else:
                    dense = self._predict_action_chunk_dense(obs)
                t_infer_done = time.perf_counter()

                # Stamp server-side timestamps for granular latency decomposition
                dense.server_obs_received_ts = float(getattr(obs, "server_received_ts", 0.0))
                dense.server_action_sent_ts = time.time()

                self._publish_dense(dense)
                # Provide a stable `step` field for compact diagnostics.
                self._metrics.diagnostic.set_context(
                    step=int(obs.get_control_step()),
                    last_infer_src_step=int(obs.get_control_step()),
                    chunk_size=self.actions_per_chunk,
                )
                self._metrics.diagnostic.timing_s("infer_total_ms", t_infer_done - t_infer_start)
                self._metrics.diagnostic.timing_s("producer_loop_total_ms", time.perf_counter() - t_total_start)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                self.logger.error("Error in inference producer loop: %s", e, exc_info=True)
                self._metrics.diagnostic.counter("inference_producer_error", 1)
                # Exponential backoff: 0.1s, 0.2s, 0.4s, ... capped at 2s
                backoff = min(0.1 * (2 ** (consecutive_errors - 1)), 2.0)
                time.sleep(backoff)

    def _mock_predict_action_chunk_dense(self, observation_t: TimedObservation) -> services_pb2.ActionsDense:
        """Generate mock actions for simulation experiments (no real model)."""
        action_dim = self.config.mock_action_dim
        actions_per_chunk = self.actions_per_chunk or 50

        # Generate random actions
        actions_np = np.random.randn(actions_per_chunk, action_dim).astype(np.float32) * 0.1
        payload = np.asarray(actions_np, dtype=np.float32, order="C")

        dense = services_pb2.ActionsDense(
            timestamp=float(observation_t.get_timestamp()),
            source_control_step=int(observation_t.get_control_step()),
            chunk_start_step=int(observation_t.chunk_start_step),
            dt=float(self.config.environment_dt),
            num_actions=int(payload.shape[0]),
            action_dim=int(payload.shape[1]),
            actions_f32=payload.tobytes(order="C"),
        )
        return dense

    def _inject_pi05_complementary_data(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Add `complementary_data` (task/subtask/advantage) and `robot_type` for pi05_rl.

        Mirrors the per-step injection in `lerobot.rl.inference_utils.get_actions_worker`
        which is required by the `pi05_full` preprocessor's
        `Pi05FullPrepareStateTokenizerProcessorStep`. Other policy types do not
        require this and the dict is returned unchanged.
        """
        if self.policy_type not in ("pi05_rl", "pi05_rlt"):
            return observation
        observation["robot_type"] = self._pi05_robot_type
        complementary_data = {
            "task": [self._pi05_task_str or ""],
            "subtask": [""],
        }
        if self.policy_type == "pi05_rl":
            complementary_data["advantage"] = torch.tensor([[self._pi05_advantage]], dtype=torch.float32)
        observation["complementary_data"] = complementary_data
        return observation

    def _align_prefix_slice(
        self,
        slice_norm: torch.Tensor,
        src_step: int,
        anchor_now: torch.Tensor | None,
        start_idx: int,
    ) -> torch.Tensor:
        """Re-anchor a single cached prefix slice to the *current* observation state.

        Generalization of `lerobot.rl.actor_pi05_async_utils.align_prev_actions`
        for arbitrary `(start_idx, end_idx)` slices of a cached chunk. Each slice
        is treated as having `offset = start_idx` (i.e. its first row was at
        position `start_idx` in the source chunk's per-timestep stats).

        Returns the (re-normalized, in model space) slice, same shape as input.
        Falls back to returning `slice_norm` unchanged when alignment is not
        possible (e.g. absolute encoding, normalizer unavailable, missing
        anchor in cache).
        """
        if self._action_encoding not in ("anchor", "delta"):
            return slice_norm
        if anchor_now is None or self._action_normalizer is None:
            return slice_norm
        if self.actions_per_chunk is None:
            return slice_norm

        anchor_old = self._action_cache.get_anchor(int(src_step))
        if anchor_old is None:
            return slice_norm

        # delta with offset > 0: only d_0 references s_0; consecutive diffs need no fix.
        if self._action_encoding == "delta" and start_idx > 0:
            return slice_norm

        n_seg, action_dim = slice_norm.shape
        chunk_size = int(self.actions_per_chunk)
        if start_idx >= chunk_size:
            return slice_norm
        n_seg_eff = min(n_seg, chunk_size - start_idx)
        if n_seg_eff <= 0:
            return slice_norm
        end_idx = start_idx + n_seg_eff

        device = slice_norm.device
        dtype = slice_norm.dtype

        # Right-align so per-timestep postprocessor stats align with original positions
        right_padded = torch.zeros(chunk_size, action_dim, device=device, dtype=dtype)
        right_padded[start_idx:end_idx] = slice_norm[:n_seg_eff]

        try:
            d_abs = self.postprocessor(right_padded)
        except Exception as e:
            self.logger.warning("RTC alignment: postprocessor failed (%s); skipping", e)
            return slice_norm

        # Apply shift in absolute action space, only on the executable joint dims
        a_old = anchor_old.squeeze(0) if anchor_old.dim() > 1 else anchor_old
        a_now = anchor_now.squeeze(0) if anchor_now.dim() > 1 else anchor_now
        a_old = a_old.to(device=d_abs.device, dtype=d_abs.dtype)
        a_now = a_now.to(device=d_abs.device, dtype=d_abs.dtype)
        delta_s = a_old - a_now
        a_out = d_abs.shape[-1]
        correct_dim = min(delta_s.shape[-1], a_out)
        delta_s_use = delta_s[:correct_dim]

        if self._action_encoding == "anchor":
            d_abs[start_idx:end_idx, :correct_dim] = (
                d_abs[start_idx:end_idx, :correct_dim] + delta_s_use
            )
        else:  # delta with start_idx == 0
            d_abs[0, :correct_dim] = d_abs[0, :correct_dim] + delta_s_use

        # Left-align for renorm: the model receives prev_chunk_left_over at positions 0..n_seg-1
        left_padded = torch.zeros(chunk_size, a_out, device=d_abs.device, dtype=d_abs.dtype)
        left_padded[:n_seg_eff] = d_abs[start_idx:end_idx]

        try:
            renorm = self._action_normalizer._normalize_action(left_padded, inverse=False)
        except Exception as e:
            self.logger.warning("RTC alignment: renormalize failed (%s); skipping", e)
            return slice_norm

        # Match input shape: cached chunk uses model dim (e.g. 32-dim padded); renorm is
        # whatever the normalizer returns. Pad/truncate to keep the original action_dim.
        out = torch.zeros(n_seg_eff, action_dim, device=device, dtype=dtype)
        copy_dim = min(action_dim, renorm.shape[-1])
        out[:, :copy_dim] = renorm[:n_seg_eff, :copy_dim].to(device=device, dtype=dtype)
        # If the input had more rows than chunk_size could absorb, copy the remainder
        # through unchanged (no per-timestep stats available beyond chunk_size).
        if n_seg_eff < n_seg:
            tail = slice_norm[n_seg_eff:]
            out = torch.cat([out, tail], dim=0)
        return out

    def _predict_action_chunk_dense(self, observation_t: TimedObservation) -> services_pb2.ActionsDense:
        """Run inference on an observation and return dense packed actions (lower jitter)."""
        if self.actions_per_chunk is None:
            raise RuntimeError("actions_per_chunk is not set; did SendPolicyInstructions run?")
        if self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("pre/post processors not initialized; did SendPolicyInstructions run?")

        def _sync() -> None:
            if (
                isinstance(self.device, str)
                and self.device.startswith("cuda")
                and torch.cuda.is_available()
            ):
                torch.cuda.synchronize()

        def _now() -> float:
            _sync()
            return time.perf_counter()

        server_timings: dict[str, float] = {}
        t_profile_start = _now()
        t_last = t_profile_start

        def _mark(name: str) -> None:
            nonlocal t_last
            now = _now()
            server_timings[name] = (now - t_last) * 1000.0
            t_last = now

        # Optional RTC metadata (client-provided hard-mask prefix + estimated delay).
        rtc_meta = None
        raw_obs_any = observation_t.get_observation()
        if isinstance(raw_obs_any, dict):
            rtc_meta = raw_obs_any.get("__rtc__")

        # Remove RTC metadata before policy preprocessing (avoid surprising processors).
        if rtc_meta is not None and isinstance(raw_obs_any, dict):
            raw_obs = dict(raw_obs_any)
            raw_obs.pop("__rtc__", None)
        else:
            raw_obs = raw_obs_any
        _mark("obs_meta")

        # 1. Prepare observation
        observation: Observation = raw_observation_to_observation(
            raw_obs,
            self.lerobot_features,
            self.policy_image_features,
        )
        _mark("raw_obs_to_observation")

        # Capture pre-preprocess raw joint state as the chunk-start anchor
        # required by `anchor` / `delta` action encodings. The preprocessor
        # normalizes OBS_STATE in-place, so we must snapshot it first.
        anchor_state: torch.Tensor | None = None
        if isinstance(observation, dict) and OBS_STATE in observation:
            obs_state_raw = observation[OBS_STATE]
            if isinstance(obs_state_raw, torch.Tensor):
                anchor_state = obs_state_raw.detach().clone()
        _mark("anchor_snapshot")

        # 2. Preprocess (inject pi05_rl complementary_data when applicable)
        observation = self._inject_pi05_complementary_data(observation)
        observation = self.preprocessor(observation)
        _mark("preprocess")

        # 3. Inference (avoid autograd / reduce variance)
        # NOTE: Do NOT use `torch.inference_mode()` here: RTC guidance needs to temporarily
        # enable gradients for the inpainting correction term, and inference_mode cannot be
        # overridden. `torch.no_grad()` keeps the normal path efficient while still allowing
        # nested `torch.enable_grad()` for RTC.
        src_control_step = int(observation_t.get_control_step())

        with torch.no_grad():
            rtc_kwargs: dict[str, Any] = {}
            rtc_prefix_len = 0
            if rtc_meta is not None and self._rtc_cfg is not None and self._rtc_cfg.enabled:
                try:
                    d = int(rtc_meta.get("latency_steps", 0))
                    action_schedule_spans = rtc_meta.get("action_schedule_spans")

                    # overlap_end from client: where fresh region starts (H - max(s_min, d))
                    H = self.actions_per_chunk
                    overlap_end = int(rtc_meta.get("overlap_end") or (H - d))
                    self._metrics.diagnostic.counter("rtc_meta_seen", 1)

                    # Reconstruct prefix tensor from multiple cached chunks
                    if action_schedule_spans:
                        slices: list[torch.Tensor] = []
                        for control_src_step, start_idx, end_idx in action_schedule_spans:
                            cached_chunk = self._action_cache.get(int(control_src_step))
                            if cached_chunk is None:
                                self._metrics.diagnostic.counter("rtc_cache_miss", 1)
                            else:
                                self._metrics.diagnostic.counter("rtc_cache_hit", 1)
                            if cached_chunk is not None:
                                # Extract slice from cached chunk (B, T, A) or (T, A)
                                if cached_chunk.ndim == 2:
                                    raw_slice = cached_chunk[start_idx:end_idx, :]
                                else:
                                    # Squeeze batch dim for concatenation
                                    raw_slice = cached_chunk[0, start_idx:end_idx, :]
                                # Re-anchor under anchor / delta encodings so the
                                # cached deltas reference the *current* observation
                                # state instead of the stale anchor used at
                                # generation time. (No-op for absolute encoding.)
                                aligned_slice = self._align_prefix_slice(
                                    slice_norm=raw_slice,
                                    src_step=int(control_src_step),
                                    anchor_now=anchor_state,
                                    start_idx=int(start_idx),
                                )
                                slices.append(aligned_slice)
                                if (
                                    self._action_encoding in ("anchor", "delta")
                                    and aligned_slice is not raw_slice
                                ):
                                    self._metrics.diagnostic.counter("rtc_prefix_aligned", 1)

                        if slices:
                            # Concatenate all slices along time dimension -> (T_total, A)
                            prefix_tensor = torch.cat(slices, dim=0)
                            prefix_tensor = prefix_tensor.unsqueeze(0)  # (1, T_total, A)
                            T_prefix = prefix_tensor.shape[1]
                            rtc_prefix_len = int(T_prefix)

                            # Clamp overlap_end to what we actually have in the prefix
                            # This allows graceful degradation when cache is incomplete
                            effective_overlap_end = min(overlap_end, T_prefix)

                            # Zero-pad to max_action_dim if model uses padded action space
                            max_action_dim = getattr(self.policy.config, "max_action_dim", None)
                            if max_action_dim is not None and prefix_tensor.shape[-1] < max_action_dim:
                                b, t, a = prefix_tensor.shape
                                padded = torch.zeros(
                                    b, t, max_action_dim,
                                    device=prefix_tensor.device,
                                    dtype=prefix_tensor.dtype,
                                )
                                padded[:, :, :a] = prefix_tensor
                                prefix_tensor = padded

                            rtc_kwargs = {
                                "inference_delay": d,
                                "prev_chunk_left_over": prefix_tensor.to(device=self.device),
                                "overlap_end": effective_overlap_end,  # Clamped for RTC guidance
                                "overlap_end_intended": overlap_end,  # Original for visualization
                            }
                            self._metrics.diagnostic.counter("rtc_applied", 1)
                        else:
                            self._metrics.diagnostic.counter("rtc_not_applied_no_slices", 1)
                    else:
                        self._metrics.diagnostic.counter("rtc_not_applied_empty_prefix", 1)
                except Exception:
                    self._metrics.diagnostic.counter("rtc_meta_error", 1)
                    rtc_kwargs = {}
            _mark("rtc_prefix")

            action_tensor = self._get_action_chunk(observation, **rtc_kwargs)
            _mark("policy_predict")

        # Ensure (B, T, A)
        if action_tensor.ndim != 3:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor[:, : self.actions_per_chunk, :]

        b, t, a = action_tensor.shape

        # Cache raw action chunk BEFORE postprocessing (for future RTC inpainting).
        # Key by control_step so RTC action_schedule_spans spans can look up the
        # right chunk. We also stash the anchor used to generate this chunk so
        # cross-chunk RTC prefix slices can be re-anchored to the *current*
        # observation state (`align_prev_actions`).
        if src_control_step >= 0:
            self._action_cache.put(src_control_step, action_tensor, anchor=anchor_state)
        _mark("raw_action_cache")

        # 4. Vectorized postprocess: (B, T, A_in) -> (B*T, A_in) -> (B, T, A_out)
        flat = action_tensor.reshape(b * t, a)
        flat = self.postprocessor(flat)
        if not isinstance(flat, torch.Tensor):
            raise TypeError(f"postprocessor must return torch.Tensor, got {type(flat)}")
        a_out = flat.shape[-1]
        action_tensor = flat.reshape(b, t, a_out)
        _mark("postprocess")

        # 5. Anchor / delta action-encoding reconstruction.
        # When the policy is trained with action_encoding="anchor" the model
        # emits per-step deltas relative to the chunk-start joint state; with
        # "delta" the deltas are sequential (cumulative). Without adding the
        # anchor back the robot drives to the unnormalized delta space (e.g.
        # "stretches out and stays"). This mirrors `inference_utils.py`
        # `get_actions_worker` PHASE 5 (`unnormalized + anchor` for "anchor",
        # `cumsum + anchor` for "delta").
        action_encoding = getattr(getattr(self.policy, "config", None), "action_encoding", "absolute")
        debug_anchor_dump = (
            self._debug_chunks_remaining > 0
            and action_encoding in ("anchor", "delta")
        )
        if action_encoding in ("anchor", "delta") and anchor_state is None:
            # If we got here, the per-chunk reconstruction silently no-ops and
            # the robot will be commanded to the unnormalized delta directly,
            # which will look like the arm collapsing to the floor / "stretching
            # out" pose. Surface it loudly so we don't chase it as a model bug.
            self.logger.warning(
                "[DRTC ANCHOR DEBUG] action_encoding=%s but anchor_state is None for "
                "src_step=%d -- per-chunk anchor add-back will be skipped (this is "
                "almost certainly the cause of any 'pushing into the ground' behavior).",
                action_encoding,
                src_control_step,
            )
        if action_encoding in ("anchor", "delta") and anchor_state is not None:
            anchor = anchor_state.to(device=action_tensor.device, dtype=action_tensor.dtype)
            # anchor shape may be (1, A_state) or (A_state,); broadcast to (1, 1, A_out)
            if anchor.dim() == 2:
                anchor = anchor.squeeze(0)
            anchor_dim = anchor.shape[-1]
            if anchor_dim != a_out:
                # Action and state may be padded differently; truncate or pad anchor
                # to match the postprocessed action dim conservatively.
                if anchor_dim > a_out:
                    anchor = anchor[:a_out]
                else:
                    pad = torch.zeros(a_out - anchor_dim, device=anchor.device, dtype=anchor.dtype)
                    anchor = torch.cat([anchor, pad], dim=0)
            anchor_b = anchor.view(1, 1, a_out)
            if debug_anchor_dump:
                # One-shot diagnostic: prints chunk_size, encoding, anchor row,
                # first unnormalized delta, and the reconstructed first action so
                # we can compare against `inference_utils.py` PHASE 5 by eye.
                _delta0 = action_tensor[0, 0].detach().to("cpu").float().tolist()
                _anchor0 = anchor.detach().to("cpu").float().tolist()
                _sum0 = (action_tensor[0, 0] + anchor).detach().to("cpu").float().tolist()
                _cfg = getattr(self.policy, "config", None)
                self.logger.info(
                    "[DRTC ANCHOR DEBUG] src_step=%d chunk=%dx%d "
                    "policy.chunk_size=%s action_encoding=%s a_in=%d a_out=%d anchor_dim=%d | "
                    "delta[0]=%s | anchor=%s | recon[0]=%s",
                    src_control_step,
                    t,
                    a_out,
                    getattr(_cfg, "chunk_size", "?"),
                    action_encoding,
                    a,
                    a_out,
                    anchor_dim,
                    [f"{x:+.4f}" for x in _delta0],
                    [f"{x:+.4f}" for x in _anchor0],
                    [f"{x:+.4f}" for x in _sum0],
                )
                self._debug_chunks_remaining -= 1
            if action_encoding == "anchor":
                action_tensor = action_tensor + anchor_b
            else:  # "delta"
                action_tensor = torch.cumsum(action_tensor, dim=1) + anchor_b
        _mark("anchor_reconstruct")

        # Drop batch dim and move to CPU once
        actions_cpu = action_tensor.squeeze(0).detach().to("cpu")
        actions_np = actions_cpu.to(torch.float32).numpy()

        payload = np.asarray(actions_np, dtype=np.float32, order="C")
        _mark("cpu_numpy_payload")

        # Emit action chunk to trajectory visualization (if enabled)
        if self._trajectory_viz_server is not None:
            # Build RTC params dict for visualization
            rtc_params_viz: dict[str, Any] | None = None
            prefix_weights_viz: list[float] | None = None

            if self._rtc_cfg is not None and self._rtc_cfg.enabled and rtc_kwargs:
                d_viz = rtc_kwargs.get("inference_delay", 0)
                # Use intended overlap_end for visualization (not clamped to prefix length)
                overlap_end_viz = rtc_kwargs.get("overlap_end_intended", rtc_kwargs.get("overlap_end", self.actions_per_chunk))
                H_viz = self.actions_per_chunk

                rtc_params_viz = {
                    "d": d_viz,
                    "H": H_viz,
                    "overlap_end": overlap_end_viz,
                    "sigma_d": self._rtc_cfg.sigma_d,
                    "schedule": self._rtc_cfg.prefix_attention_schedule,
                    "max_guidance_weight": self._rtc_cfg.max_guidance_weight,
                    "full_trajectory_alignment": self._rtc_cfg.full_trajectory_alignment,
                }
                prefix_weights_viz = compute_prefix_weights_for_viz(
                    d_viz, overlap_end_viz, H_viz, self._rtc_cfg.prefix_attention_schedule
                )

            # Create and emit the event
            actions_list = actions_np.tolist()
            event = EvActionChunk(
                src_control_step=src_control_step,
                actions=actions_list,
                frozen_len=rtc_kwargs.get("inference_delay", 0) if rtc_kwargs else 0,
                timestamp=time.time(),
                rtc_params=rtc_params_viz,
                prefix_weights=prefix_weights_viz,
            )
            self._trajectory_viz_server.on_chunk(event)
        _mark("trajectory_viz")

        dense_kwargs: dict[str, Any] = dict(
            timestamp=float(observation_t.get_timestamp()),
            source_control_step=int(observation_t.get_control_step()),
            chunk_start_step=int(observation_t.chunk_start_step),
            dt=float(self.config.environment_dt),
            num_actions=int(payload.shape[0]),
            action_dim=int(payload.shape[1]),
            actions_f32=payload.tobytes(order="C"),
        )
        dense = services_pb2.ActionsDense(**dense_kwargs)
        _mark("dense_proto")
        server_timings["total"] = (_now() - t_profile_start) * 1000.0

        model_value = getattr(self.policy, "model", None)
        model_outer = getattr(model_value, "_phase_timings_outer", {}) if model_value is not None else {}
        model_inner = getattr(model_value, "_phase_timings", {}) if model_value is not None else {}
        rtc_status = "applied" if rtc_kwargs else ("meta" if rtc_meta is not None else "none")
        self.logger.info(
            "[DRTC INFER TIMING] src_step=%d chunk_start=%d rtc=%s prefix_len=%d "
            "server={%s} model_outer={%s} model_inner={%s}",
            src_control_step,
            int(observation_t.chunk_start_step),
            rtc_status,
            rtc_prefix_len,
            self._format_timing_parts(server_timings),
            self._format_timing_parts(model_outer),
            self._format_timing_parts(model_inner),
        )
        return dense

    def _get_action_chunk(self, observation: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """Get action chunk from the policy."""
        t0 = time.perf_counter()
        chunk = self.policy.predict_action_chunk(observation, **kwargs)
        t1 = time.perf_counter()
        self._metrics.diagnostic.timing_s("policy_predict_ms", t1 - t0)

        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # Add batch dimension: (chunk_size, action_dim) -> (1, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def stop(self) -> None:
        """Stop the server."""
        self._reset_server()
        self._metrics.diagnostic.stop()


@draccus.wrap()
def serve_drtc(cfg: PolicyServerDrtcConfig) -> None:
    """Start the DRTC PolicyServer."""
    # Create server instance
    policy_server = PolicyServerDrtc(cfg)

    # Setup gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    bound_port = server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    if bound_port == 0:
        raise RuntimeError(
            f"Failed to bind gRPC server to {cfg.host}:{cfg.port}. "
            "Is the port already in use, or are you binding to an unavailable interface?"
        )

    server_started = False
    try:
        server.start()
        server_started = True
        print(f"PolicyServerDrtc listening on {cfg.host}:{bound_port}")
        logging.getLogger("policy_server_drtc").info("gRPC server bound to %s:%s", cfg.host, bound_port)
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received; shutting down")
    except Exception:
        policy_server.logger.error("Policy server crashed", exc_info=True)
        raise
    finally:
        # Best-effort cleanup to avoid dangling threads on failures.
        try:
            policy_server.stop()
        except Exception:
            policy_server.logger.error("Error while stopping policy server", exc_info=True)
        if server_started:
            server.stop(grace=5)
    print("Server terminated")


if __name__ == "__main__":
    serve_drtc()

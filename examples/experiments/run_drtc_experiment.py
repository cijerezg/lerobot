#!/usr/bin/env python3
"""
DRTC Experiment Runner

This script runs experiments on a REAL ROBOT to validate the DRTC algorithm. It assumes the policy server is already running.

Experiment parameters are defined in YAML config files that live in
examples/experiments/configs/.

Usage:
    python examples/experiments/run_async_inference_experiment.py --config mixture_of_faults
    python examples/experiments/run_async_inference_experiment.py --config spike --output_dir results/experiments
    python examples/experiments/run_async_inference_experiment.py --config path/to/custom.yaml
"""

import argparse
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from lerobot.async_inference.configs_drtc import RobotClientDrtcConfig
from lerobot.async_inference.robot_client_drtc import RobotClientDrtc
from lerobot.async_inference.utils.simulation import (
    DisconnectConfig,
    DisconnectEvent,
    DropConfig,
    DropEvent,
    DuplicateConfig,
    DuplicateEvent,
    ReorderConfig,
    ReorderEvent,
)
from lerobot.cameras.opencv import OpenCVCameraConfig

# `lerobot` consolidated SO100Follower/SO101Follower into a single `so_follower`
# package; SO100FollowerConfig and SO101FollowerConfig are kept as TypeAlias.
from lerobot.robots.so_follower import SO100FollowerConfig, SO101FollowerConfig

logger = logging.getLogger(__name__)


# Default to loopback because `scripts/run_drtc_experiment.sh` spawns the policy
# server locally (and `PolicyServerDrtcConfig.host` defaults to "localhost").
# Override with `--server_address host:port` for a remote/distributed setup.
DEFAULT_SERVER_ADDRESS = "localhost:8080"
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "so101_follower_2026_01_03"
DEFAULT_CAMERA1_PATH = "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:2:1.0-video-index0"
DEFAULT_CAMERA2_PATH = "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0"
DEFAULT_CAMERA_WIDTH = 800
DEFAULT_CAMERA_HEIGHT = 600
DEFAULT_CAMERA_FPS = 30
DEFAULT_CAMERA_FOURCC = "MJPG"
DEFAULT_CAMERA_USE_THREADED_ASYNC_READ = True
DEFAULT_CAMERA_ALLOW_STALE_FRAMES = True
DEFAULT_MODEL_PATH = "jackvial/so101_smolvla_pickplaceorangecube_e100"
DEFAULT_TASK = "Pick up the orange cube and place it on the black X marker with the white background"

CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    estimator: str
    cooldown: bool
    # Hardware
    robot_type: str = "so101"
    gpu: str = ""
    client_host: str = ""
    server_host: str = ""
    robot_port: str = DEFAULT_ROBOT_PORT
    robot_id: str = DEFAULT_ROBOT_ID
    camera1_path: str = DEFAULT_CAMERA1_PATH
    camera2_path: str = DEFAULT_CAMERA2_PATH
    # Names used as keys in the robot's `cameras` dict. Different policies expect
    # different camera key names (e.g. smolvla expects "camera1"/"camera2", while
    # pi05_rl expects "wrist"/"top"), so make the names YAML-configurable.
    camera1_name: str = "camera1"
    camera2_name: str = "camera2"
    camera_width: int = DEFAULT_CAMERA_WIDTH
    camera_height: int = DEFAULT_CAMERA_HEIGHT
    camera_fps: int = DEFAULT_CAMERA_FPS
    camera_fourcc: str | None = DEFAULT_CAMERA_FOURCC
    camera_use_threaded_async_read: bool = DEFAULT_CAMERA_USE_THREADED_ASYNC_READ
    camera_allow_stale_frames: bool = DEFAULT_CAMERA_ALLOW_STALE_FRAMES
    # Policy
    policy_type: str = "smolvla"
    pretrained_name_or_path: str = DEFAULT_MODEL_PATH
    # pi05_rl-only: scalar advantage value injected into the prompt at inference
    # time. The pi05_full preprocessor scales by `advantage_scaling`, applies
    # tanh, and bins it into "Advantage: positive" / "Advantage: negative" in
    # the language prompt. None = use the value baked into the loaded policy
    # config (`policy.config.inference_advantage`, default 1.0). Use 0.0 to
    # force the "negative" label as a diagnostic. Ignored for non-pi05_rl
    # policies.
    inference_advantage: float | None = None
    # pi05_rl/pi05_full-only: optional override for how often generated
    # subtask tokens are refreshed. None = use the loaded policy config.
    subtask_regeneration_interval: float | None = None
    # pi05_rl/pi05_full-only: when False, skip runtime subtask generation and
    # condition action sampling on the main task prompt only.
    subtask_generation_enabled: bool = True
    # pi05_rlt-only: lightweight RLT modules on top of frozen PI0.5.
    rlt_enabled: bool = False
    rlt_embedding_checkpoint: str | None = None
    rlt_head_checkpoint: str | None = None
    rlt_chunk_size: int = 10
    rlt_token_dim: int = 2048
    rlt_bc_beta: float = 1.0
    rlt_reference_dropout_p: float = 0.5
    rlt_online_collection_enabled: bool = False
    rlt_online_training_enabled: bool = False
    rlt_warmup_episodes: int = 1
    rlt_warmup_transitions: int = 128
    rlt_replay_capacity: int = 10000
    rlt_batch_size: int = 64
    rlt_utd_ratio: int = 1
    rlt_train_freq_s: float = 1.0
    rlt_save_freq_steps: int = 500
    rlt_output_dir: str = "outputs/rlt_online"
    rlt_demo_buffer_path: str | None = None
    rlt_online_buffer_path: str | None = None
    rlt_online_buffer_save_freq_transitions: int = 100
    rlt_persist_buffer_on_shutdown: bool = True
    # Review-only image capture (off the inference hot path; consumed by the
    # lerobot-data-studio offline viewer).
    rlt_review_capture_enabled: bool = False
    rlt_review_jpeg_quality: int = 80
    rlt_review_archive_path: str | None = None
    rlt_actor_lr: float = 3e-4
    rlt_critic_lr: float = 3e-4
    rlt_discount: float = 0.99
    rlt_target_update_tau: float = 0.005
    rlt_execute_after_train_steps: int = 1000000
    rlt_context_cache_size: int = 256
    rlt_transition_queue_size: int = 256
    # DRTC parameters
    latency_k: float = 2.0
    epsilon: int = 2
    s_min: int = 15
    latency_alpha: float = 0.125
    latency_beta: float = 0.25
    # Timing
    duration_s: float = 60.0
    run_until_interrupt: bool = False
    fps: int = 60
    actions_per_chunk: int = 50
    # Flow matching / RTC
    num_flow_matching_steps: int | None = 8
    rtc_enabled: bool = True
    rtc_max_guidance_weight: float | None = None
    rtc_prefix_attention_schedule: str = "linear"
    rtc_sigma_d: float = 0.2
    rtc_full_trajectory_alignment: bool = False
    # Butterworth filter
    action_filter_mode: str = "butterworth"
    action_filter_butterworth_cutoff: float = 3.0
    action_filter_butterworth_order: int = 2
    action_filter_gain: float = 1.4
    action_filter_past_buffer_size: int = 10
    # Drop/spike/duplicate/reorder/disconnect injection
    drop_obs_config: DropConfig | None = None
    drop_action_config: DropConfig | None = None
    dup_obs_config: DuplicateConfig | None = None
    dup_action_config: DuplicateConfig | None = None
    reorder_obs_config: ReorderConfig | None = None
    reorder_action_config: ReorderConfig | None = None
    disconnect_config: DisconnectConfig | None = None
    spikes: list[dict] = field(default_factory=list)
    # Diagnostics
    full_diagnostics: bool = False
    trajectory_viz_enabled: bool = False
    # Teleop intervention (mirrors inference_pi05_async behaviour, minimal scope).
    # When enabled, the client connects an additional leader arm, swaps in the
    # leader's joint positions whenever IS_INTERVENTION is true, flushes the
    # action schedule on disengage, and sends follower-pose feedback to the
    # leader otherwise.
    teleop_enabled: bool = False
    teleop_type: str = ""
    teleop_port: str = ""
    teleop_id: str = ""
    teleop_send_feedback: bool = True


# ---- YAML config loading ----

# Scalar fields that map 1:1 from YAML keys to ExperimentConfig constructor args.
_SCALAR_FIELDS = frozenset({
    "name", "estimator", "cooldown",
    "robot_type", "gpu", "client_host", "server_host",
    "robot_port", "robot_id",
    "camera1_path", "camera2_path",
    "camera1_name", "camera2_name",
    "camera_width", "camera_height", "camera_fps", "camera_fourcc",
    "camera_use_threaded_async_read", "camera_allow_stale_frames",
    "policy_type", "pretrained_name_or_path", "inference_advantage",
    "subtask_regeneration_interval", "subtask_generation_enabled",
    "rlt_enabled", "rlt_embedding_checkpoint", "rlt_head_checkpoint",
    "rlt_chunk_size", "rlt_token_dim", "rlt_bc_beta", "rlt_reference_dropout_p",
    "rlt_online_collection_enabled", "rlt_online_training_enabled",
    "rlt_warmup_episodes", "rlt_warmup_transitions", "rlt_replay_capacity",
    "rlt_batch_size", "rlt_utd_ratio", "rlt_train_freq_s", "rlt_save_freq_steps",
    "rlt_output_dir", "rlt_demo_buffer_path", "rlt_online_buffer_path",
    "rlt_online_buffer_save_freq_transitions", "rlt_persist_buffer_on_shutdown",
    "rlt_review_capture_enabled", "rlt_review_jpeg_quality", "rlt_review_archive_path",
    "rlt_actor_lr", "rlt_critic_lr", "rlt_discount",
    "rlt_target_update_tau", "rlt_execute_after_train_steps",
    "rlt_context_cache_size", "rlt_transition_queue_size",
    "latency_k", "epsilon", "s_min", "latency_alpha", "latency_beta",
    "duration_s", "run_until_interrupt", "fps", "actions_per_chunk",
    "num_flow_matching_steps", "rtc_enabled", "rtc_max_guidance_weight",
    "rtc_prefix_attention_schedule", "rtc_sigma_d", "rtc_full_trajectory_alignment",
    "action_filter_mode", "action_filter_butterworth_cutoff",
    "action_filter_butterworth_order", "action_filter_gain",
    "action_filter_past_buffer_size",
    "full_diagnostics",
    "trajectory_viz_enabled",
    "teleop_enabled", "teleop_type", "teleop_port", "teleop_id", "teleop_send_feedback",
})


def _parse_experiment_dict(d: dict) -> ExperimentConfig:
    """Convert a raw YAML dict into an ExperimentConfig."""
    kwargs: dict = {k: v for k, v in d.items() if k in _SCALAR_FIELDS}

    # Fault-injection lists -> typed config objects
    if "drop_obs" in d:
        kwargs["drop_obs_config"] = DropConfig(drops=[DropEvent(**e) for e in d["drop_obs"]])
    if "drop_action" in d:
        kwargs["drop_action_config"] = DropConfig(drops=[DropEvent(**e) for e in d["drop_action"]])
    if "dup_obs" in d:
        kwargs["dup_obs_config"] = DuplicateConfig(duplicates=[DuplicateEvent(**e) for e in d["dup_obs"]])
    if "dup_action" in d:
        kwargs["dup_action_config"] = DuplicateConfig(duplicates=[DuplicateEvent(**e) for e in d["dup_action"]])
    if "reorder_obs" in d:
        kwargs["reorder_obs_config"] = ReorderConfig(reorders=[ReorderEvent(**e) for e in d["reorder_obs"]])
    if "reorder_action" in d:
        kwargs["reorder_action_config"] = ReorderConfig(reorders=[ReorderEvent(**e) for e in d["reorder_action"]])
    if "disconnect" in d:
        kwargs["disconnect_config"] = DisconnectConfig(disconnects=[DisconnectEvent(**e) for e in d["disconnect"]])
    if "spikes" in d:
        kwargs["spikes"] = d["spikes"]

    return ExperimentConfig(**kwargs)


def load_experiments_from_yaml(path: Path) -> list[ExperimentConfig]:
    """Load one or more ExperimentConfig from a YAML file.

    Supports two formats:

    **Single experiment** -- top-level dict IS the experiment::

        name: my_experiment
        estimator: jk
        cooldown: true

    **Multi-experiment** -- has an ``experiments`` key (and optional ``defaults``)::

        defaults:
          estimator: jk
          cooldown: true
        experiments:
          - name: run_1
          - name: run_2
            estimator: max_last_10
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

    if "experiments" in raw:
        defaults = raw.get("defaults", {})
        configs = []
        for exp_dict in raw["experiments"]:
            merged = {**defaults, **exp_dict}
            configs.append(_parse_experiment_dict(merged))
        return configs

    return [_parse_experiment_dict(raw)]


def resolve_config_path(config_arg: str) -> Path:
    """Resolve a ``--config`` argument to a YAML file path.

    Accepts:
      - A relative or absolute path to a ``.yaml`` file.
      - A bare name (e.g. ``spike``), which resolves to
        ``examples/experiments/configs/<name>.yaml``.
    """
    path = Path(config_arg)
    if path.exists():
        return path

    # Try appending .yaml
    if not config_arg.endswith(".yaml"):
        with_ext = Path(config_arg + ".yaml")
        if with_ext.exists():
            return with_ext

    # Try the bundled configs directory
    in_configs = CONFIGS_DIR / config_arg
    if in_configs.exists():
        return in_configs
    if not config_arg.endswith(".yaml"):
        in_configs_yaml = CONFIGS_DIR / (config_arg + ".yaml")
        if in_configs_yaml.exists():
            return in_configs_yaml

    raise FileNotFoundError(
        f"Config not found: {config_arg} (also tried {CONFIGS_DIR / config_arg})"
    )


def create_robot_config(config: ExperimentConfig) -> SO100FollowerConfig | SO101FollowerConfig:
    camera_fourcc = config.camera_fourcc.strip() if isinstance(config.camera_fourcc, str) else config.camera_fourcc
    if camera_fourcc == "":
        camera_fourcc = None

    camera_cfg = {
        config.camera2_name: OpenCVCameraConfig(
            index_or_path=config.camera2_path,
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps,
            fourcc=camera_fourcc,
            use_threaded_async_read=config.camera_use_threaded_async_read,
            allow_stale_frames=config.camera_allow_stale_frames,
        ),
        config.camera1_name: OpenCVCameraConfig(
            index_or_path=config.camera1_path,
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps,
            fourcc=camera_fourcc,
            use_threaded_async_read=config.camera_use_threaded_async_read,
            allow_stale_frames=config.camera_allow_stale_frames,
        ),
    }
    robot_type_normalized = config.robot_type.strip().lower()
    if robot_type_normalized in {"so101", "so101_follower"}:
        return SO101FollowerConfig(port=config.robot_port, id=config.robot_id, cameras=camera_cfg)
    if robot_type_normalized in {"so100", "so100_follower"}:
        return SO100FollowerConfig(port=config.robot_port, id=config.robot_id, cameras=camera_cfg)

    raise ValueError(
        f"Unsupported robot_type '{config.robot_type}'. "
        "Supported values: so101, so101_follower, so100, so100_follower."
    )


def create_teleop_config(config: ExperimentConfig):
    """Build a TeleoperatorConfig from the experiment's teleop_* fields, or None.

    Note: SOLeaderTeleopConfig is registered for both "so100_leader" and
    "so101_leader" and uses a single underlying SOLeader implementation. The
    `type` property is derived from the draccus registry, so we only validate
    that the requested type is one of the supported leader variants and rely on
    the factory to dispatch correctly.
    """
    if not config.teleop_enabled:
        return None
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

    teleop_type_norm = config.teleop_type.strip().lower()
    if teleop_type_norm not in {"so101_leader", "so100_leader"}:
        raise ValueError(
            f"Unsupported teleop_type {config.teleop_type!r}. "
            "Supported values: so101_leader, so100_leader."
        )
    if not config.teleop_port:
        raise ValueError("teleop_port must be set when teleop_enabled=True")
    return SOLeaderTeleopConfig(
        port=config.teleop_port,
        id=config.teleop_id or None,
    )


def create_client_config(
    config: ExperimentConfig,
    metrics_path: Path,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    trajectory_viz_ws_url: str | None = None,
) -> RobotClientDrtcConfig:
    """Create a client config for a single experiment."""
    robot_cfg = create_robot_config(config)
    teleop_cfg = create_teleop_config(config)
    client_kwargs = dict(
        robot=robot_cfg,
        server_address=server_address,
        robot_type=config.robot_type,
        gpu=config.gpu,
        client_host=config.client_host,
        server_host=config.server_host,
        policy_device="cuda",
        policy_type=config.policy_type,
        pretrained_name_or_path=config.pretrained_name_or_path,
        inference_advantage=config.inference_advantage,
        subtask_regeneration_interval=config.subtask_regeneration_interval,
        subtask_generation_enabled=config.subtask_generation_enabled,
        rlt_enabled=config.rlt_enabled,
        rlt_embedding_checkpoint=config.rlt_embedding_checkpoint,
        rlt_head_checkpoint=config.rlt_head_checkpoint,
        rlt_chunk_size=config.rlt_chunk_size,
        rlt_token_dim=config.rlt_token_dim,
        rlt_bc_beta=config.rlt_bc_beta,
        rlt_reference_dropout_p=config.rlt_reference_dropout_p,
        rlt_online_collection_enabled=config.rlt_online_collection_enabled,
        rlt_online_training_enabled=config.rlt_online_training_enabled,
        rlt_warmup_episodes=config.rlt_warmup_episodes,
        rlt_warmup_transitions=config.rlt_warmup_transitions,
        rlt_replay_capacity=config.rlt_replay_capacity,
        rlt_batch_size=config.rlt_batch_size,
        rlt_utd_ratio=config.rlt_utd_ratio,
        rlt_train_freq_s=config.rlt_train_freq_s,
        rlt_save_freq_steps=config.rlt_save_freq_steps,
        rlt_output_dir=config.rlt_output_dir,
        rlt_demo_buffer_path=config.rlt_demo_buffer_path,
        rlt_online_buffer_path=config.rlt_online_buffer_path,
        rlt_online_buffer_save_freq_transitions=config.rlt_online_buffer_save_freq_transitions,
        rlt_persist_buffer_on_shutdown=config.rlt_persist_buffer_on_shutdown,
        rlt_review_capture_enabled=config.rlt_review_capture_enabled,
        rlt_review_jpeg_quality=config.rlt_review_jpeg_quality,
        rlt_review_archive_path=config.rlt_review_archive_path,
        rlt_actor_lr=config.rlt_actor_lr,
        rlt_critic_lr=config.rlt_critic_lr,
        rlt_discount=config.rlt_discount,
        rlt_target_update_tau=config.rlt_target_update_tau,
        rlt_execute_after_train_steps=config.rlt_execute_after_train_steps,
        rlt_context_cache_size=config.rlt_context_cache_size,
        rlt_transition_queue_size=config.rlt_transition_queue_size,
        actions_per_chunk=config.actions_per_chunk,
        fps=config.fps,
        s_min=config.s_min,
        latency_estimator_type=config.estimator,
        cooldown_enabled=config.cooldown,
        latency_k=config.latency_k,
        epsilon=config.epsilon,
        latency_alpha=config.latency_alpha,
        latency_beta=config.latency_beta,
        # Flow matching / RTC
        num_flow_matching_steps=config.num_flow_matching_steps,
        rtc_enabled=config.rtc_enabled,
        rtc_max_guidance_weight=config.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=config.rtc_prefix_attention_schedule,
        rtc_sigma_d=config.rtc_sigma_d,
        rtc_full_trajectory_alignment=config.rtc_full_trajectory_alignment,
        # Butterworth filter
        action_filter_mode=config.action_filter_mode,
        action_filter_butterworth_cutoff=config.action_filter_butterworth_cutoff,
        action_filter_butterworth_order=config.action_filter_butterworth_order,
        action_filter_gain=config.action_filter_gain,
        action_filter_past_buffer_size=config.action_filter_past_buffer_size,
        # Diagnostics and robustness
        metrics_diagnostic_enabled=True,
        metrics_diagnostic_interval_s=2.0,
        metrics_diagnostic_window_s=10.0,
        metrics_diagnostic_verbose=config.full_diagnostics,
        control_use_deadline_clock=True,
        obs_fallback_on_failure=True,
        obs_fallback_max_age_s=2.0,
        trajectory_viz_enabled=config.trajectory_viz_enabled,
        # Drop/spike/duplicate/reorder/disconnect injection
        drop_obs_config=config.drop_obs_config,
        drop_action_config=config.drop_action_config,
        dup_obs_config=config.dup_obs_config,
        dup_action_config=config.dup_action_config,
        reorder_obs_config=config.reorder_obs_config,
        reorder_action_config=config.reorder_action_config,
        disconnect_config=config.disconnect_config,
        spikes=config.spikes,
        metrics_path=str(metrics_path),
        # Teleop intervention
        teleop_enabled=config.teleop_enabled,
        teleop=teleop_cfg,
        teleop_send_feedback=config.teleop_send_feedback,
    )
    if trajectory_viz_ws_url:
        client_kwargs["trajectory_viz_ws_url"] = trajectory_viz_ws_url
    return RobotClientDrtcConfig(**client_kwargs)


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    trajectory_viz_ws_url: str | None = None,
    task: str = DEFAULT_TASK,
    experiment_name: str | None = None,
) -> dict:
    """Run a single standalone experiment (creates and tears down client)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        # Use the provided name verbatim; append timestamp if folder already exists.
        exp_dir = output_dir / experiment_name
        if exp_dir.exists():
            exp_dir = output_dir / f"{experiment_name}_{timestamp}"
        exp_name = exp_dir.name
    else:
        exp_name = f"{config.name}_{timestamp}"
        exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / f"{exp_name}.csv"

    logger.info(f"Running experiment: {config.name}")
    logger.info(f"  Estimator: {config.estimator}, Cooldown: {config.cooldown}, Full diagnostics: {config.full_diagnostics}")
    if config.drop_obs_config:
        logger.info(f"  Drop obs: {config.drop_obs_config}")
    if config.drop_action_config:
        logger.info(f"  Drop action: {config.drop_action_config}")
    if config.dup_obs_config:
        logger.info(f"  Dup obs: {config.dup_obs_config}")
    if config.dup_action_config:
        logger.info(f"  Dup action: {config.dup_action_config}")
    if config.reorder_obs_config:
        logger.info(f"  Reorder obs: {config.reorder_obs_config}")
    if config.reorder_action_config:
        logger.info(f"  Reorder action: {config.reorder_action_config}")
    if config.disconnect_config:
        logger.info(f"  Disconnect: {config.disconnect_config}")
    if config.spikes:
        logger.info(f"  Spikes: {config.spikes}")

    client_cfg = create_client_config(
        config,
        metrics_path,
        server_address=server_address,
        trajectory_viz_ws_url=trajectory_viz_ws_url,
    )
    client = RobotClientDrtc(client_cfg)

    def stop_after_duration():
        if config.duration_s <= 0:
            return
        time.sleep(config.duration_s)
        client.signal_stop()

    def signal_handler(sig, frame):
        client.signal_stop()

    timer_thread: threading.Thread | None = None
    if not config.run_until_interrupt:
        timer_thread = threading.Thread(target=stop_after_duration, daemon=True)
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        logger.info("Starting client...")
        if client.start():
            logger.info("Client started successfully")
            obs_thread = threading.Thread(target=client.observation_sender, daemon=True)
            action_thread = threading.Thread(target=client.action_receiver, daemon=True)
            obs_thread.start()
            action_thread.start()
            if timer_thread is not None:
                timer_thread.start()
                logger.info(f"Running for {config.duration_s}s...")
            else:
                logger.info("Running until user interrupt (Ctrl+C)...")
            try:
                client.control_loop(task=task)
            except Exception as e:
                logger.exception(f"Control loop error: {e}")
            # Wait for the timer thread to finish (it calls signal_stop which flushes)
            if timer_thread is not None:
                timer_thread.join(timeout=5.0)
            # Ensure metrics are flushed from the main thread in case signal_stop
            # hasn't finished or was never called (e.g. control loop exited early).
            if client._metrics.experiment is not None and client.config.metrics_path:
                try:
                    client._metrics.experiment.flush(client.config.metrics_path)
                except Exception:
                    pass
            success = metrics_path.exists()
            logger.info(f"Experiment finished. Metrics saved: {success}")
            if success:
                exp_dir = metrics_path.parent
                logger.info(f"Metrics file: {metrics_path}")
                logger.info("To plot:")
                logger.info(f"  uv run python examples/experiments/plot_results.py --input {exp_dir}")
            return {"success": success, "metrics_path": str(metrics_path)}
        else:
            logger.error("Client failed to start!")
            return {"success": False, "error": "Client failed to start"}
    except Exception as e:
        logger.exception(f"Exception during experiment: {e}")
        return {"success": False, "error": str(e)}
    finally:
        # Only disconnect at the very end for standalone experiments
        try:
            client.stop()
        except Exception:
            pass
        finally:
            signal.signal(signal.SIGINT, original_handler)


def main():
    parser = argparse.ArgumentParser(
        description="DRTC Experiment Runner",
        epilog=(
            "Config files live in examples/experiments/configs/. "
            "Pass a bare name (e.g. spike) or a path to a .yaml file."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file, or a bare config name from examples/experiments/configs/",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help=(
            "Optional custom run name (single-experiment configs only). "
            "Overrides the name from the YAML file."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="results/experiments")
    parser.add_argument("--server_address", type=str, default=DEFAULT_SERVER_ADDRESS)
    parser.add_argument(
        "--trajectory_viz_ws_url",
        type=str,
        default=None,
        help=(
            "Optional WebSocket URL for trajectory visualization. "
            "Used when trajectory visualization is enabled in the experiment config."
        ),
    )
    parser.add_argument("--pause_between_s", type=float, default=10.0)
    parser.add_argument(
        "--run_until_interrupt",
        action="store_true",
        help="Override YAML duration_s and run the experiment until Ctrl+C.",
    )
    parser.add_argument(
        "--inference_advantage",
        type=float,
        default=None,
        help=(
            "pi05_rl-only override for the advantage scalar injected into the "
            "policy prompt at inference time. Overrides the value in the YAML "
            "config (and ultimately in the loaded policy config). Use 1.0 to "
            "force 'positive' in the prompt, 0.0 to force 'negative'."
        ),
    )

    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    configs = load_experiments_from_yaml(config_path)
    logger.info(f"Loaded {len(configs)} experiment(s) from {config_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, config in enumerate(configs):
        if len(configs) > 1:
            logger.info(f"{'='*50}")
            logger.info(f"[{i+1}/{len(configs)}] {config.name}")
            logger.info(f"{'='*50}")

        if args.inference_advantage is not None:
            logger.info(
                "Overriding inference_advantage from CLI: %s -> %s",
                config.inference_advantage,
                args.inference_advantage,
            )
            config.inference_advantage = args.inference_advantage
        if args.run_until_interrupt:
            logger.info("Overriding YAML duration: running until Ctrl+C")
            config.run_until_interrupt = True

        experiment_name = (args.experiment_name or "").strip() or None
        result = run_experiment(
            config,
            output_dir,
            server_address=args.server_address,
            trajectory_viz_ws_url=args.trajectory_viz_ws_url,
            task=DEFAULT_TASK,
            experiment_name=experiment_name if len(configs) == 1 else None,
        )
        results.append(result)

        if i < len(configs) - 1:
            logger.info(f"Pausing {args.pause_between_s}s before next experiment...")
            time.sleep(args.pause_between_s)

    if len(configs) > 1:
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"All experiments complete: {success_count}/{len(results)} succeeded")


if __name__ == "__main__":
    main()

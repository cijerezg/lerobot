# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration classes for the DRTC async inference implementation.

These configurations follow the DRTC algorithm
with proper SPSC mailboxes, Jacobson-Karels latency estimation,
cool-down mechanism, and freshest-observation-wins merging.
"""

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from .constants import DEFAULT_FPS, DEFAULT_OBS_QUEUE_TIMEOUT
from .utils.simulation import DisconnectConfig, DropConfig, DuplicateConfig, ReorderConfig, SpikeDelayConfig

# =============================================================================
# Robot Client Configuration
# =============================================================================


@dataclass
class RobotClientDrtcConfig:
    """Configuration for the DRTC robot client.

    This configuration follows the DRTC algorithm
    with proper SPSC mailboxes, Jacobson-Karels latency estimation,
    cool-down mechanism, and freshest-observation-wins merging.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use (e.g., 'act', 'smolvla')"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Actions per chunk (should be <= policy's max action horizon)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk (H in the paper)"})

    # Teleop intervention (mirrors inference_pi05_async behaviour, minimal scope).
    # When enabled, the client constructs an additional teleop device (e.g. a
    # leader arm). Per tick it checks IS_INTERVENTION; while engaged the leader's
    # joint positions override the policy chunk and are sent to the follower.
    # On disengage the action schedule is flushed so the next inference round
    # re-anchors at the new pose.
    teleop_enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable teleop intervention. When True, requires `teleop` to be set."
        },
    )
    teleop: TeleoperatorConfig | None = field(
        default=None,
        metadata={
            "help": "Teleop device configuration (e.g. SOLeaderTeleopConfig). "
            "Required when teleop_enabled=True."
        },
    )
    teleop_send_feedback: bool = field(
        default=True,
        metadata={
            "help": "When not intervening, push the latest follower pose back to "
            "the leader via teleop_device.send_feedback(...) so the leader gently "
            "tracks the follower for a smooth handover."
        },
    )

    # Hardware metadata (for experiment reports)
    robot_type: str = field(default="", metadata={"help": "Robot type identifier (e.g. so101)"})
    gpu: str = field(default="", metadata={"help": "GPU used for inference (e.g. RTX 4070 TI SUPER)"})
    client_host: str = field(default="", metadata={"help": "Description of the client host (e.g. local server)"})
    server_host: str = field(default="", metadata={"help": "Description of the server host (e.g. local server)"})

    # Task instruction for the robot
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration (for policy inference on server)
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})

    # Control frequency
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Control loop frequency in Hz"})

    # DRTC parameters
    s_min: int = field(
        default=14,
        metadata={
            "help": "Minimum execution horizon in action steps (s_min from RTC paper). "
            "Trigger inference when schedule_size <= s_min. "
            "Effective execution horizon is max(s_min, latency_steps)."
        },
    )
    epsilon: int = field(
        default=1,
        metadata={
            "help": "Cooldown buffer in action steps. "
            "After triggering inference, cooldown is set to latency_steps + epsilon. "
            "Small values (1-2) prevent over-triggering without adding significant delay."
        },
    )
    latency_estimator_type: str = field(
        default="jk",
        metadata={"help": "Latency estimator type: 'jk' (Jacobson-Karels), 'max_last_10', or 'fixed'"},
    )
    latency_alpha: float = field(
        default=0.125, metadata={"help": "Jacobson-Karels smoothing factor for RTT mean"}
    )
    latency_beta: float = field(
        default=0.25, metadata={"help": "Jacobson-Karels smoothing factor for RTT deviation"}
    )
    latency_k: float = field(
        default=1.5, metadata={"help": "Jacobson-Karels scaling factor for deviation (K)"}
    )
    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size after stopping"}
    )

    # RTC (client-driven, server-side inpainting; flow policies only)
    rtc_enabled: bool = field(
        default=True,
        metadata={"help": "Enable RTC-style inpainting on the policy server (flow policies only)"},
    )
    rtc_max_guidance_weight: float | None = field(
        default=None,
        metadata={
            "help": "RTC max guidance weight (clamp). If None, uses num_flow_matching_steps "
            "(Alex Soare optimization: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html)"
        },
    )
    rtc_prefix_attention_schedule: str = field(
        default="linear",
        metadata={"help": "RTC prefix attention schedule: zeros|ones|linear|exp"},
    )
    rtc_sigma_d: float = field(
        default=0.2,
        metadata={
            "help": "RTC prior variance σ_d. Lower values (e.g., 0.2) give stronger guidance "
            "and smoother transitions. 1.0 = original RTC behavior. "
            "(Alex Soare optimization: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html)"
        },
    )
    rtc_full_trajectory_alignment: bool = field(
        default=False,
        metadata={
            "help": "Skip gradient computation in RTC and use error directly. "
            "Faster and smoother when distance between chunks is small."
        },
    )
    num_flow_matching_steps: int | None = field(
        default=8,
        metadata={
            "help": "Override for number of flow matching denoising steps. "
            "If None, uses the policy's default (e.g., 10 for PI0/SmolVLA). "
            "Higher values = smoother but slower inference. "
            "(Alex Soare optimization: Beta should scale with n)"
        },
    )

    # pi05_rl-only: scalar advantage value injected into the policy prompt at
    # inference time. The pi05_full preprocessor reads this from
    # `complementary_data["advantage"]`, scales it by `advantage_scaling`,
    # passes it through `tanh`, bins it, and renders it as
    # "Advantage: positive" or "Advantage: negative" inside the language prompt.
    # `None` means "use the value baked into the loaded policy config"
    # (`policy.config.inference_advantage`, default 1.0). Use 0.0 for a
    # diagnostic A/B that forces the "negative" label, or 1.0 to force
    # "positive". Has no effect for non-pi05_rl policies.
    inference_advantage: float | None = field(
        default=None,
        metadata={
            "help": "Override for pi05_rl inference advantage scalar (controls "
            "the positive/negative label in the prompt). None = use the value "
            "from the loaded policy's config."
        },
    )
    subtask_regeneration_interval: float | None = field(
        default=None,
        metadata={
            "help": "Override for PI05 subtask-token cache refresh interval in seconds. "
            "None = use the loaded policy's config; 0 = regenerate every chunk."
        },
    )
    subtask_generation_enabled: bool = field(
        default=True,
        metadata={
            "help": "Enable runtime PI05 subtask generation. If False, action sampling "
            "conditions on the main task prompt without an explicit subtask segment."
        },
    )
    rlt_enabled: bool = field(
        default=False,
        metadata={"help": "Enable the PI05 RLT actor head when using policy_type='pi05_rlt'."},
    )
    rlt_embedding_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Path to an offline-trained RLT embedding checkpoint for policy_type='pi05_rlt'."},
    )
    rlt_head_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Path to an online-trained RLT actor/critic checkpoint for policy_type='pi05_rlt'."},
    )
    rlt_chunk_size: int = field(
        default=10,
        metadata={"help": "Number of leading action steps refined by the RLT actor head."},
    )
    rlt_token_dim: int | None = field(
        default=None,
        metadata={
            "help": (
                "Dimensionality of the compact RLT token. If None, falls back to the "
                "policy-type default (2048 for pi05_rlt, vlm_width for tinypi05_rlt)."
            )
        },
    )
    rlt_actor_hidden_dims: list[int] | None = field(
        default=None,
        metadata={"help": "Hidden layer sizes for the RLT actor MLP. None preserves the policy default."},
    )
    rlt_critic_hidden_dims: list[int] | None = field(
        default=None,
        metadata={"help": "Hidden layer sizes for the RLT critic MLP. None preserves the policy default."},
    )
    rlt_actor_residual_scale: float = field(
        default=0.25,
        metadata={"help": "Maximum residual action scale added by the RLT actor head."},
    )
    rlt_actor_mode: str = field(
        default="gaussian",
        metadata={
            "help": (
                "RLT actor parameterization. 'gaussian' (paper Eq. 4) directly "
                "predicts mu_theta(x, ã); 'residual' adds ã + scale*tanh(MLP(...))."
            )
        },
    )
    rlt_action_std: float = field(
        default=0.05,
        metadata={
            "help": (
                "Fixed Gaussian exploration std applied during online data "
                "collection. Set 0 to disable noise."
            )
        },
    )
    rlt_num_critics: int = field(
        default=1,
        metadata={"help": "Number of RLT critics. Values >1 use clipped-min critic targets."},
    )
    rlt_bc_beta: float = field(
        default=1.0,
        metadata={"help": "BC/reference-action regularization coefficient for RLT training."},
    )
    rlt_bc_action_weights: list[float] | None = field(
        default=None,
        metadata={"help": "Optional per-action BC weights. Length must match RLT action dimension."},
    )
    rlt_jerk_beta: float = field(
        default=0.0,
        metadata={"help": "Optional smoothness penalty on second differences of actor action chunks."},
    )
    rlt_reference_dropout_p: float = field(
        default=0.5,
        metadata={"help": "Reference action dropout probability for RLT actor training."},
    )
    rlt_online_collection_enabled: bool = field(
        default=False,
        metadata={"help": "Collect compact online RLT transitions on the DRTC client."},
    )
    rlt_online_training_enabled: bool = field(
        default=False,
        metadata={"help": "Train PI05 RLT actor/critic heads online on the DRTC server."},
    )
    rlt_warmup_episodes: int = field(
        default=1,
        metadata={"help": "Minimum completed episodes before online RLT training can update heads."},
    )
    rlt_warmup_transitions: int = field(
        default=128,
        metadata={"help": "Minimum replay transitions before online RLT training can update heads."},
    )
    rlt_replay_capacity: int = field(
        default=10000,
        metadata={"help": "Maximum number of compact RLT transitions kept in server replay."},
    )
    rlt_batch_size: int = field(
        default=64,
        metadata={"help": "RLT online training batch size."},
    )
    rlt_utd_ratio: int = field(
        default=1,
        metadata={"help": "RLT online updates per trainer tick."},
    )
    rlt_train_freq_s: float = field(
        default=1.0,
        metadata={"help": "Seconds between online RLT trainer ticks."},
    )
    rlt_save_freq_steps: int = field(
        default=500,
        metadata={"help": "Save an RLT head checkpoint every N online training steps."},
    )
    rlt_output_dir: str = field(
        default="outputs/rlt_online",
        metadata={"help": "Directory for online RLT head checkpoints."},
    )
    rlt_demo_buffer_path: str | None = field(
        default=None,
        metadata={"help": "Optional persisted replay buffer used to seed online RLT training."},
    )
    rlt_online_buffer_path: str | None = field(
        default=None,
        metadata={"help": "Optional persisted online replay buffer to load and update during collection."},
    )
    rlt_online_buffer_save_freq_transitions: int = field(
        default=100,
        metadata={"help": "Save the online RLT buffer every N accepted transitions. 0 disables periodic saves."},
    )
    rlt_persist_buffer_on_shutdown: bool = field(
        default=True,
        metadata={"help": "Persist the online RLT buffer when the policy server resets or stops."},
    )
    rlt_actor_lr: float = field(default=3e-4, metadata={"help": "Online RLT actor learning rate."})
    rlt_critic_lr: float = field(default=3e-4, metadata={"help": "Online RLT critic learning rate."})
    rlt_discount: float = field(default=0.99, metadata={"help": "Online RLT critic discount factor."})
    rlt_target_update_tau: float = field(
        default=0.005,
        metadata={"help": "Soft-update coefficient for the RLT target critic."},
    )
    rlt_execute_after_train_steps: int = field(
        default=1000000,
        metadata={"help": "Do not execute the online RLT actor until this many training steps have completed."},
    )
    rlt_context_cache_size: int = field(
        default=256,
        metadata={"help": "Maximum server-side RLT source contexts retained for transition resolution."},
    )
    rlt_transition_queue_size: int = field(
        default=256,
        metadata={"help": "Maximum queued client-side RLT transitions before dropping."},
    )
    rlt_grad_clip_norm: float | None = field(
        default=None,
        metadata={"help": "Optional gradient norm clipping value for online RLT actor and critic updates."},
    )
    rlt_q_abs_max: float | None = field(
        default=None,
        metadata={"help": "Disable actor execution if observed absolute Q values exceed this threshold."},
    )
    rlt_action_deviation_abs_max: float | None = field(
        default=None,
        metadata={"help": "Disable actor execution if actor-reference absolute deviation exceeds this threshold."},
    )
    rlt_loss_abs_max: float | None = field(
        default=None,
        metadata={"help": "Disable actor execution if actor or critic loss magnitude exceeds this threshold."},
    )
    rlt_safety_patience: int = field(
        default=3,
        metadata={"help": "Number of consecutive unsafe RLT updates before actor execution is disabled."},
    )

    # Diagnostic metrics (console output; avg/max only)
    metrics_diagnostic_enabled: bool = field(
        default=True,
        metadata={"help": "Enable periodic diagnostic metrics printed to console (avg/max timings)"},
    )
    metrics_diagnostic_interval_s: float = field(
        default=2.0, metadata={"help": "How often to print diagnostic metrics (seconds)"}
    )
    metrics_diagnostic_window_s: float = field(
        default=10.0, metadata={"help": "Rolling window for diagnostic metrics (seconds)"}
    )
    metrics_diagnostic_verbose: bool = field(
        default=False,
        metadata={"help": "If True, include full timing/counter details in diagnostic console output"},
    )

    # Trajectory visualization (sends data to policy server via gRPC)
    trajectory_viz_enabled: bool = field(
        default=False,
        metadata={"help": "Enable sending trajectory data to policy server for visualization"},
    )
    trajectory_viz_ws_url: str = field(
        default="ws://localhost:8089",
        metadata={"help": "WebSocket URL for trajectory visualization server (for executed actions)"},
    )

    # Control-loop clocking (optional)
    control_use_deadline_clock: bool = field(
        default=True,
        metadata={"help": "Use a deadline-based control clock (reduces jitter under overruns)"},
    )

    # Observation sender robustness
    obs_fallback_on_failure: bool = field(
        default=True,
        metadata={
            "help": "If robot observation capture fails, reuse the last good observation to avoid stalling"
        },
    )
    obs_fallback_max_age_s: float = field(
        default=2.0,
        metadata={"help": "Max age (seconds) of the last good observation that may be reused on failure"},
    )

    # Simulation mode (for experiments)
    use_mock_robot: bool = field(
        default=False,
        metadata={"help": "Use mock robot instead of real hardware (for experiments without a physical robot)"},
    )
    cooldown_enabled: bool = field(
        default=True,
        metadata={"help": "Enable cooldown mechanism (set False for classic async baseline)"},
    )
    inference_reset_mode: str = field(
        default="cooldown",
        metadata={
            "help": "Mode for resetting inference readiness: "
            "'cooldown' (default) decrements each tick and allows recovery from drops; "
            "'merge_reset' resets only when actions are merged (RTC-style, stalls on drops)"
        },
    )

    # Drop injection (for experiments)
    drop_obs_config: DropConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for observation drop injection. "
            "Example: DropConfig(random_drop_p=0.05) or DropConfig(burst_period_s=20, burst_duration_s=1)"
        },
    )
    drop_action_config: DropConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for action chunk drop injection. "
            "Example: DropConfig(random_drop_p=0.05) or DropConfig(burst_period_s=20, burst_duration_s=1)"
        },
    )

    # Duplicate injection (for experiments)
    dup_obs_config: DuplicateConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for observation duplicate injection. "
            "Example: DuplicateConfig(duplicates=[DuplicateEvent(start_s=5, duration_s=1)])"
        },
    )
    dup_action_config: DuplicateConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for action chunk duplicate injection. "
            "Example: DuplicateConfig(duplicates=[DuplicateEvent(start_s=5, duration_s=1)])"
        },
    )

    # Reorder injection (for experiments)
    reorder_obs_config: ReorderConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for observation reorder injection (pairwise hold-and-swap). "
            "Example: ReorderConfig(reorders=[ReorderEvent(start_s=5, duration_s=2)])"
        },
    )
    reorder_action_config: ReorderConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for action chunk reorder injection (pairwise hold-and-swap). "
            "Example: ReorderConfig(reorders=[ReorderEvent(start_s=5, duration_s=2)])"
        },
    )

    # Disconnect injection (for experiments)
    disconnect_config: DisconnectConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for network disconnect injection (blocks obs and action threads). "
            "Example: DisconnectConfig(disconnects=[DisconnectEvent(start_s=5, duration_s=3)])"
        },
    )

    # Spike injection (for experiments, passed to server)
    # List of dicts: [{"start_s": 5.0, "delay_ms": 2000}, ...]
    spikes: list[dict] = field(
        default_factory=list,
        metadata={
            "help": "Explicit spike events as list of dicts. "
            "Example: [{'start_s': 5, 'delay_ms': 2000}, {'start_s': 15, 'delay_ms': 1000}]"
        },
    )

    # Experiment metrics (disk output; CSV export)
    metrics_path: str | None = field(
        default=None,
        metadata={"help": "Path to write experiment metrics CSV (None = disabled)"},
    )
    # Action smoothing to reduce policy jitter / servo hunting
    # Modes: "none", "adaptive_lowpass", "hold_stable", "butterworth"
    action_filter_mode: str = field(
        default="none",
        metadata={
            "help": "Action filtering mode: "
            "'none' = no filtering, "
            "'adaptive_lowpass' = IIR filter with adaptive alpha based on delta magnitude, "
            "'hold_stable' = hold previous action when delta is below threshold (eliminates jitter), "
            "'butterworth' = proper low-pass filter with configurable cutoff frequency"
        },
    )
    action_filter_alpha_min: float = field(
        default=0.1,
        metadata={
            "help": "Low-pass filter alpha for small deltas (heavy smoothing). "
            "Used when action delta is below deadband threshold. Range: (0, 1]. "
            "Lower = more smoothing. 0.1 gives strong attenuation of high-freq jitter."
        },
    )
    action_filter_alpha_max: float = field(
        default=0.5,
        metadata={
            "help": "Low-pass filter alpha for large deltas (faster response). "
            "Used when action delta exceeds deadband threshold. Range: (0, 1]"
        },
    )
    action_filter_deadband: float = field(
        default=0.05,
        metadata={
            "help": "Deadband threshold in action units (radians for joints). "
            "For 'adaptive_lowpass': deltas below this get alpha_min, above get alpha_max. "
            "For 'hold_stable': deltas below this are ignored entirely. "
            "Default 0.05 rad ≈ 3 degrees."
        },
    )
    action_filter_butterworth_cutoff: float = field(
        default=10.0,
        metadata={
            "help": "Butterworth filter cutoff frequency in Hz. "
            "Frequencies above this are attenuated. Should be < fps/2 (Nyquist). "
            "Recommended: 10-12 Hz for 60 Hz control rate."
        },
    )
    action_filter_butterworth_order: int = field(
        default=2,
        metadata={
            "help": "Butterworth filter order (1-4). "
            "Higher = sharper frequency rolloff but more phase lag."
        },
    )
    action_filter_gain: float = field(
        default=1.0,
        metadata={
            "help": "Gain multiplier applied after filtering to compensate amplitude attenuation. "
            "Values > 1.0 boost the filtered signal."
        },
    )
    action_filter_past_buffer_size: int = field(
        default=5,
        metadata={
            "help": "Number of past executed actions to keep in filter buffer. "
            "Used by 'median' and 'butterworth' modes for history."
        },
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step in seconds."""
        return 1.0 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")
        if self.s_min <= 0:
            raise ValueError(f"s_min must be positive, got {self.s_min}")
        if self.s_min >= self.actions_per_chunk:
            raise ValueError(
                f"s_min must be < actions_per_chunk, got {self.s_min} >= {self.actions_per_chunk}"
            )
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if self.latency_estimator_type not in ("jk", "max_last_10", "fixed"):
            raise ValueError(
                f"latency_estimator_type must be 'jk', 'max_last_10', or 'fixed', got {self.latency_estimator_type}"
            )
        if self.inference_reset_mode not in ("cooldown", "merge_reset"):
            raise ValueError(
                f"inference_reset_mode must be 'cooldown' or 'merge_reset', got {self.inference_reset_mode}"
            )
        if self.metrics_diagnostic_interval_s <= 0:
            raise ValueError(
                f"metrics_diagnostic_interval_s must be positive, got {self.metrics_diagnostic_interval_s}"
            )
        if self.metrics_diagnostic_window_s <= 0:
            raise ValueError(
                f"metrics_diagnostic_window_s must be positive, got {self.metrics_diagnostic_window_s}"
            )
        if self.obs_fallback_max_age_s <= 0:
            raise ValueError(f"obs_fallback_max_age_s must be positive, got {self.obs_fallback_max_age_s}")
        if self.rtc_max_guidance_weight is not None and self.rtc_max_guidance_weight <= 0:
            raise ValueError(f"rtc_max_guidance_weight must be positive or None, got {self.rtc_max_guidance_weight}")
        if self.rtc_sigma_d <= 0:
            raise ValueError(f"rtc_sigma_d must be positive, got {self.rtc_sigma_d}")
        if self.num_flow_matching_steps is not None and self.num_flow_matching_steps <= 0:
            raise ValueError(f"num_flow_matching_steps must be positive or None, got {self.num_flow_matching_steps}")
        if self.subtask_regeneration_interval is not None and self.subtask_regeneration_interval < 0:
            raise ValueError(
                "subtask_regeneration_interval must be non-negative or None, "
                f"got {self.subtask_regeneration_interval}"
            )
        if self.rlt_chunk_size <= 0:
            raise ValueError(f"rlt_chunk_size must be positive, got {self.rlt_chunk_size}")
        if self.rlt_token_dim is not None and self.rlt_token_dim <= 0:
            raise ValueError(f"rlt_token_dim must be positive or None, got {self.rlt_token_dim}")
        for name, dims in (
            ("rlt_actor_hidden_dims", self.rlt_actor_hidden_dims),
            ("rlt_critic_hidden_dims", self.rlt_critic_hidden_dims),
        ):
            if dims is not None and (not dims or any(dim <= 0 for dim in dims)):
                raise ValueError(f"{name} must contain positive dimensions, got {dims}")
        if self.rlt_actor_residual_scale <= 0:
            raise ValueError(
                f"rlt_actor_residual_scale must be positive, got {self.rlt_actor_residual_scale}"
            )
        if self.rlt_actor_mode not in ("gaussian", "residual"):
            raise ValueError(
                f"rlt_actor_mode must be 'gaussian' or 'residual', got {self.rlt_actor_mode!r}"
            )
        if self.rlt_action_std < 0:
            raise ValueError(
                f"rlt_action_std must be non-negative, got {self.rlt_action_std}"
            )
        if self.rlt_num_critics <= 0:
            raise ValueError(f"rlt_num_critics must be positive, got {self.rlt_num_critics}")
        if self.rlt_bc_beta < 0:
            raise ValueError(f"rlt_bc_beta must be non-negative, got {self.rlt_bc_beta}")
        if self.rlt_bc_action_weights is not None and any(weight < 0 for weight in self.rlt_bc_action_weights):
            raise ValueError(
                f"rlt_bc_action_weights must be non-negative, got {self.rlt_bc_action_weights}"
            )
        if self.rlt_jerk_beta < 0:
            raise ValueError(f"rlt_jerk_beta must be non-negative, got {self.rlt_jerk_beta}")
        if not 0 <= self.rlt_reference_dropout_p <= 1:
            raise ValueError(
                "rlt_reference_dropout_p must be in [0, 1], "
                f"got {self.rlt_reference_dropout_p}"
            )
        if self.rlt_warmup_episodes < 0:
            raise ValueError(f"rlt_warmup_episodes must be non-negative, got {self.rlt_warmup_episodes}")
        if self.rlt_warmup_transitions < 0:
            raise ValueError(f"rlt_warmup_transitions must be non-negative, got {self.rlt_warmup_transitions}")
        if self.rlt_replay_capacity <= 0:
            raise ValueError(f"rlt_replay_capacity must be positive, got {self.rlt_replay_capacity}")
        if self.rlt_batch_size <= 0:
            raise ValueError(f"rlt_batch_size must be positive, got {self.rlt_batch_size}")
        if self.rlt_utd_ratio <= 0:
            raise ValueError(f"rlt_utd_ratio must be positive, got {self.rlt_utd_ratio}")
        if self.rlt_train_freq_s <= 0:
            raise ValueError(f"rlt_train_freq_s must be positive, got {self.rlt_train_freq_s}")
        if self.rlt_save_freq_steps <= 0:
            raise ValueError(f"rlt_save_freq_steps must be positive, got {self.rlt_save_freq_steps}")
        if self.rlt_online_buffer_save_freq_transitions < 0:
            raise ValueError(
                "rlt_online_buffer_save_freq_transitions must be non-negative, "
                f"got {self.rlt_online_buffer_save_freq_transitions}"
            )
        if self.rlt_actor_lr <= 0:
            raise ValueError(f"rlt_actor_lr must be positive, got {self.rlt_actor_lr}")
        if self.rlt_critic_lr <= 0:
            raise ValueError(f"rlt_critic_lr must be positive, got {self.rlt_critic_lr}")
        if not 0 <= self.rlt_discount <= 1:
            raise ValueError(f"rlt_discount must be in [0, 1], got {self.rlt_discount}")
        if not 0 < self.rlt_target_update_tau <= 1:
            raise ValueError(
                f"rlt_target_update_tau must be in (0, 1], got {self.rlt_target_update_tau}"
            )
        if self.rlt_execute_after_train_steps < 0:
            raise ValueError(
                "rlt_execute_after_train_steps must be non-negative, "
                f"got {self.rlt_execute_after_train_steps}"
            )
        if self.rlt_context_cache_size <= 0:
            raise ValueError(f"rlt_context_cache_size must be positive, got {self.rlt_context_cache_size}")
        if self.rlt_transition_queue_size <= 0:
            raise ValueError(
                f"rlt_transition_queue_size must be positive, got {self.rlt_transition_queue_size}"
            )
        if self.rlt_grad_clip_norm is not None and self.rlt_grad_clip_norm <= 0:
            raise ValueError(f"rlt_grad_clip_norm must be positive or None, got {self.rlt_grad_clip_norm}")
        for name, value in (
            ("rlt_q_abs_max", self.rlt_q_abs_max),
            ("rlt_action_deviation_abs_max", self.rlt_action_deviation_abs_max),
            ("rlt_loss_abs_max", self.rlt_loss_abs_max),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None, got {value}")
        if self.rlt_safety_patience <= 0:
            raise ValueError(f"rlt_safety_patience must be positive, got {self.rlt_safety_patience}")
        if self.action_filter_mode not in ("none", "adaptive_lowpass", "hold_stable", "butterworth"):
            raise ValueError(
                f"action_filter_mode must be 'none', 'adaptive_lowpass', 'hold_stable', or 'butterworth', "
                f"got {self.action_filter_mode}"
            )
        if self.action_filter_alpha_min <= 0 or self.action_filter_alpha_min > 1:
            raise ValueError(f"action_filter_alpha_min must be in (0, 1], got {self.action_filter_alpha_min}")
        if self.action_filter_alpha_max <= 0 or self.action_filter_alpha_max > 1:
            raise ValueError(f"action_filter_alpha_max must be in (0, 1], got {self.action_filter_alpha_max}")
        if self.action_filter_deadband < 0:
            raise ValueError(f"action_filter_deadband must be non-negative, got {self.action_filter_deadband}")
        if self.action_filter_butterworth_cutoff <= 0:
            raise ValueError(f"action_filter_butterworth_cutoff must be positive, got {self.action_filter_butterworth_cutoff}")
        if self.action_filter_butterworth_cutoff >= self.fps / 2:
            raise ValueError(
                f"action_filter_butterworth_cutoff must be < fps/2 (Nyquist), "
                f"got {self.action_filter_butterworth_cutoff} >= {self.fps / 2}"
            )
        if self.action_filter_butterworth_order < 1 or self.action_filter_butterworth_order > 4:
            raise ValueError(f"action_filter_butterworth_order must be 1-4, got {self.action_filter_butterworth_order}")
        if self.action_filter_gain <= 0:
            raise ValueError(f"action_filter_gain must be positive, got {self.action_filter_gain}")
        if self.action_filter_past_buffer_size < 1:
            raise ValueError(f"action_filter_past_buffer_size must be >= 1, got {self.action_filter_past_buffer_size}")
        if self.teleop_enabled and self.teleop is None:
            raise ValueError("teleop_enabled=True requires `teleop` to be set")


# =============================================================================
# Policy Server Configuration
# =============================================================================


@dataclass
class PolicyServerDrtcConfig:
    """Configuration for the DRTC PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    following the 2-thread model from the DRTC paper.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second (control frequency)"})

    # Diagnostic metrics (console output; avg/max only)
    metrics_diagnostic_enabled: bool = field(
        default=True,
        metadata={"help": "Enable periodic diagnostic metrics printed to console (avg/max timings)"},
    )
    metrics_diagnostic_interval_s: float = field(
        default=2.0, metadata={"help": "How often to print diagnostic metrics (seconds)"}
    )
    metrics_diagnostic_window_s: float = field(
        default=10.0, metadata={"help": "Rolling window for diagnostic metrics (seconds)"}
    )
    metrics_diagnostic_verbose: bool = field(
        default=False,
        metadata={"help": "If True, include full timing/counter details in diagnostic console output"},
    )

    # Observation queue timeout
    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT,
        metadata={"help": "Timeout for observation queue in seconds"},
    )

    # Mock policy configuration (for simulation experiments)
    mock_policy: bool = field(
        default=False,
        metadata={"help": "Use mock policy instead of real model (for experiments)"},
    )
    mock_spike_config: SpikeDelayConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for mock inference latency spikes. "
            "Example: SpikeDelayConfig.from_dicts([{'start_s': 5, 'delay_ms': 2000}])"
        },
    )
    mock_action_dim: int = field(
        default=6,
        metadata={"help": "Action dimension for mock policy output"},
    )

    # Model warmup (CUDA kernel compilation + memory allocation on first pass)
    warmup_passes: int = field(
        default=2,
        metadata={
            "help": "Number of dummy inference passes to run after loading the model. "
            "This eliminates CUDA cold-start latency from the first real measurement. "
            "Set to 0 to disable warmup."
        },
    )

    # Trajectory visualization (receives data from robot client via gRPC)
    trajectory_viz_enabled: bool = field(
        default=False,
        metadata={"help": "Enable trajectory visualization server (HTTP + WebSocket)"},
    )
    trajectory_viz_http_port: int = field(
        default=8088,
        metadata={"help": "HTTP port for trajectory visualization web page"},
    )
    trajectory_viz_ws_port: int = field(
        default=8089,
        metadata={"help": "WebSocket port for trajectory data streaming"},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step in seconds."""
        return 1.0 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")
        if self.metrics_diagnostic_interval_s <= 0:
            raise ValueError(
                f"metrics_diagnostic_interval_s must be positive, got {self.metrics_diagnostic_interval_s}"
            )
        if self.metrics_diagnostic_window_s <= 0:
            raise ValueError(
                f"metrics_diagnostic_window_s must be positive, got {self.metrics_diagnostic_window_s}"
            )

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import builtins
import datetime as dt
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.hub import HubMixin
from lerobot.utils.sample_weighting import SampleWeightingConfig

from . import parser
from .default import DatasetConfig, EvalConfig, PeftConfig, WandBConfig
from .policies import PreTrainedConfig
from .rewards import RewardModelConfig

TRAIN_CONFIG_NAME = "train_config.json"


def _migrate_legacy_rabc_fields(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return migrated payload for legacy RA-BC fields, or None when no migration is needed."""
    legacy_fields = (
        "use_rabc",
        "rabc_progress_path",
        "rabc_kappa",
        "rabc_epsilon",
        "rabc_head_mode",
    )
    if not any(key in config for key in legacy_fields):
        return None

    migrated_config = dict(config)
    use_rabc = bool(migrated_config.pop("use_rabc", False))
    rabc_progress_path = migrated_config.pop("rabc_progress_path", None)
    rabc_kappa = migrated_config.pop("rabc_kappa", None)
    rabc_epsilon = migrated_config.pop("rabc_epsilon", None)
    rabc_head_mode = migrated_config.pop("rabc_head_mode", None)

    # New configs may already define sample_weighting explicitly. In that case,
    # legacy fields are ignored after being stripped from the payload.
    if migrated_config.get("sample_weighting") is None and use_rabc:
        sample_weighting: dict[str, Any] = {"type": "rabc"}
        if rabc_progress_path is not None:
            sample_weighting["progress_path"] = rabc_progress_path
        if rabc_kappa is not None:
            sample_weighting["kappa"] = rabc_kappa
        if rabc_epsilon is not None:
            sample_weighting["epsilon"] = rabc_epsilon
        if rabc_head_mode is not None:
            sample_weighting["head_mode"] = rabc_head_mode
        migrated_config["sample_weighting"] = sample_weighting

    return migrated_config


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    reward_model: RewardModelConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Set to True to use deterministic cuDNN algorithms for reproducibility.
    # This disables cudnn.benchmark and may reduce training speed by ~10-20 percent.
    cudnn_deterministic: bool = False
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200
    tolerance_s: float = 1e-4
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # Sample weighting configuration (e.g., for RA-BC training)
    sample_weighting: SampleWeightingConfig | None = None

    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    @property
    def is_reward_model_training(self) -> bool:
        """True when the config targets a reward model rather than a policy."""
        return self.reward_model is not None

    @property
    def trainable_config(self) -> PreTrainedConfig | RewardModelConfig:
        """Return whichever config (policy or reward_model) is active."""
        if self.is_reward_model_training:
            return self.reward_model  # type: ignore[return-value]
        return self.policy  # type: ignore[return-value]

    def validate(self) -> None:
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        reward_model_path = parser.get_path_arg("reward_model")

        if reward_model_path:
            cli_overrides = parser.get_cli_overrides("reward_model")
            self.reward_model = RewardModelConfig.from_pretrained(
                reward_model_path, cli_overrides=cli_overrides
            )
            self.reward_model.pretrained_path = str(Path(reward_model_path))
        elif policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=yaml_overrides + cli_overrides
            )
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )

            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )

            policy_dir = Path(config_path).parent
            if self.policy is not None:
                self.policy.pretrained_path = policy_dir
            if self.reward_model is not None:
                self.reward_model.pretrained_path = str(policy_dir)
            self.checkpoint_path = policy_dir.parent

        if self.policy is None and self.reward_model is None:
            raise ValueError(
                "Neither policy nor reward_model is configured. "
                "Please specify one with `--policy.path` or `--reward_model.path`."
            )

        active_cfg = self.trainable_config
        if not self.job_name:
            if self.env is None:
                self.job_name = f"{active_cfg.type}"
            else:
                self.job_name = f"{self.env.type}_{active_cfg.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = active_cfg.get_optimizer_preset()
            self.scheduler = active_cfg.get_scheduler_preset()

        if hasattr(active_cfg, "push_to_hub") and active_cfg.push_to_hub and not active_cfg.repo_id:
            raise ValueError("'repo_id' argument missing. Please specify it to push the model to the hub.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Keys for draccus pretrained-path loading."""
        return ["policy", "reward_model"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # because of the third-party library draccus uses Any as the return type

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        # Legacy RA-BC migration only applies to framework-saved checkpoints (always JSON).
        # Hand-written YAML/TOML configs are expected to use the current sample_weighting schema.
        if config_file is not None and config_file.endswith(".json"):
            with open(config_file) as f:
                config = json.load(f)
            migrated_config = _migrate_legacy_rabc_fields(config)
            if migrated_config is not None:
                with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
                    json.dump(migrated_config, f)
                    config_file = f.name

        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)


@dataclass
class ProbeConfig:
    """Parameters for the diagnostic probes under ``lerobot.probes``.

    Every probe that runs a model forward does so in the deployment regime — the
    subtask and metadata clauses, short-term history, and wrist depth a rollout
    carries (``probes.utils.probe_frame_inputs``) — and every sampler snaps its
    anchors onto the ``policy.image_stride`` grid, because that is where stored
    image rows and depth sidecar PNGs exist.
    """

    # Enable / disable individual probes
    enable_actions: bool = True
    enable_representations: bool = True
    enable_attention: bool = True
    enable_spatial_memorization: bool = True
    enable_action_drift_jacobian: bool = False  # subtask-conditioned action-output/token Jacobians
    enable_spatial_memorization_jacobian: bool = False  # aggregated causal spatial stats (needs backward)
    enable_critic_values_distribution: bool = False  # critic V/TD-error distributions + gradient magnitudes (needs backward)
    enable_mem_history_influence: bool = False  # MEM: how much history (full/image/state) shifts the action chunk
    enable_mem_temporal_attention: bool = False  # MEM: temporal-read distributions + spatial examples
    enable_action_trace: bool = False  # interactive action inspector: 3D, wrist/gripper, safety, multimodality
    enable_metadata_steering: bool = False  # quality/mistake clause: steering range + usefulness
    enable_depth_modality: bool = False  # point-map depth: 2x2 modality grid, per-layer mass, b_l
    enable_attention_budget: bool = False  # how the action tokens' attention budget shifts over frames
    enable_subtask_sweep: bool = False  # does the subtask clause move the action chunk (memory chain hop 2)
    enable_objective: bool = False  # flow + FAST loss on val against a matched training sample

    # Common
    output_dir: str = "outputs/probe"
    mode: str = "all"  # "collect" | "plot" | "all"
    max_episodes: int | None = 5
    n_frames_per_episode: int = 128
    random_seed: int = 42
    timestep: float = 0.5  # single diffusion timestep used by all probes
    # Action sensitivity: sample real frames within every episode/subtask and
    # estimate each grouped full-horizon Jacobian norm with this many VJPs.
    action_sensitivity_frames_per_subtask: int = 6
    action_sensitivity_projections: int = 4

    # Actions / representations
    ref_max_episodes: int = 40
    ref_n_frames_per_episode: int = 256
    action_pca_dims: int = 50  # action-manifold PCA — the space every action metric is measured in
    # Reference frames nearest in state space that a prediction is allowed to match
    # against ("is this motion performed *from here*"). Too wide and it degenerates to
    # the global nearest neighbour; too narrow and the neighbourhood is noise.
    action_nn_state_k: int = 256
    # Where the fitted reference manifold is cached. None = action_manifold.pt one level
    # above the probe's output, i.e. shared by every validation step of a run. Point
    # several runs at one path to hold the coordinate system fixed across checkpoints; a
    # refit is logged loudly because it invalidates comparison with earlier runs.
    action_manifold_cache: str | None = None
    repr_pca_dims: int = 100
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_seed: int = 42
    ep_3d_a: int = 0
    ep_3d_b: int = 1

    # Attention / spatial / Jacobian
    attn_eval_episodes: str | None = None
    attn_eval_subsample: int = 2
    spatial_layers: str = "0,9,17"
    spatial_n_frames: int = 32  # total frames (1 per unique episode)

    # Depth modality (2x2 conditions x FD sensitivity per frame, so keep it small)
    depth_modality_n_frames: int = 6

    # Attention budget. Reuses spatial_layers and n_frames_per_episode (one capture
    # per frame covers every layer). budget_fd_sensitivity adds a causal series at
    # two extra forwards per frame — mass says "read", FD says "load-bearing".
    budget_fd_sensitivity: bool = False

    # Objective. One forward per frame per split, so the cost is
    # n_frames_per_episode x episodes x 2. objective_max_episodes is split across the
    # training sources, so it is the total train-side episode budget, not per source.
    objective_n_frames_per_episode: int = 24
    objective_max_episodes: int | None = 8
    objective_exemplars_per_band: int = 3  # frames shown at each of p5 / p50 / p95

    # Subtask sweep: n_frames x (max_labels + n_seeds) forwards, so keep all three small.
    subtask_sweep_n_frames: int = 8
    subtask_sweep_max_labels: int = 16
    subtask_sweep_n_seeds: int = 3

    # Critic values distribution
    critic_adv_frames: int = 1000  # frames sampled for V(s) / TD-error distribution
    critic_grad_frames: int = 200  # frames sampled for ||dV/dvision|| (forward+backward)

    # Action trace (URDF forward kinematics; open-loop pre-flight)
    trace_episodes: str | None = None  # comma-separated episode indices; None = all
    trace_anchor_stride_s: float = 2.0  # seconds between anchor frames
    trace_max_anchors_per_episode: int = 30
    trace_n_samples: int = 5  # independent flow draws per anchor — the fan
    trace_table_z: float = 0.0  # table plane height (m). 0 = the arm's own mounting plane.
    trace_clearance_warn_m: float = 0.01  # samples dipping below this are drawn red


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional
    offline_output_dir: str | None = None
    offline_save_freq: int | None = None
    buffer_cache_dir: str | None = None
    # "fallback" decodes video when a source cache is missing; "require"
    # fails before training so a collection cannot accidentally decode one
    # uncached dataset for hours.
    cache_policy: str = "fallback"
    use_rerun: bool = True
    video_logging_cameras: list[str] | None = None  # derived from policy.image_features in validate() when unset
    episode_logging_freq: int = 4
    episode_save_freq: int = 10
    probe_parameters: ProbeConfig = field(default_factory=ProbeConfig)

    # Validation
    val_dataset_path: str | None = None
    val_split: float = 0.0
    val_freq: int = 1000
    val_on_start: bool = False
    skip_critic: bool = False             # skip all critic training (forward+backward)

    def validate(self) -> None:
        super().validate()
        if self.cache_policy not in {"fallback", "require"}:
            raise ValueError(
                f"cache_policy must be 'fallback' or 'require', got {self.cache_policy!r}."
            )
        if self.video_logging_cameras is None and self.policy is not None:
            self.video_logging_cameras = [k.split(".")[-1] for k in self.policy.image_features]

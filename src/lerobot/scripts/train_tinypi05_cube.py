#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.policies.tinypi05.configuration_tinypi05 import TinyPI05Config
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins


_PI05_SWALLOWED_LOAD_WARNING = "Warning: Could not remap state dict keys:"


class _TeeStream:
    """File-like wrapper that writes to two underlying streams."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        self._primary.write(data)
        self._secondary.write(data)
        return len(data)

    def flush(self):
        self._primary.flush()
        self._secondary.flush()

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()


@contextlib.contextmanager
def _strict_pi05_state_dict_load():
    """Escalate `pi05.modeling_pi05.PI05Policy.from_pretrained` mismatches to a hard error.

    The upstream loader catches every exception raised by
    ``model.load_state_dict(remapped_state_dict, strict=True)`` and only prints
    ``Warning: Could not remap state dict keys: <err>`` before returning a model
    with random-init weights in the affected layers.  When fine-tuning that's a
    silent footgun: training continues against partially-loaded weights and the
    user only finds out when eval results are nonsense.  We tee stdout, look
    for the marker, and ``SystemExit`` so training never starts with a corrupt
    load.
    """
    from lerobot.policies.pi05 import modeling_pi05

    original = modeling_pi05.PI05Policy.from_pretrained

    @classmethod
    def wrapped(cls, pretrained_name_or_path, *args, **kwargs):
        buf = io.StringIO()
        tee = _TeeStream(sys.stdout, buf)
        with contextlib.redirect_stdout(tee):
            model = original.__func__(cls, pretrained_name_or_path, *args, **kwargs)
        captured = buf.getvalue()
        if _PI05_SWALLOWED_LOAD_WARNING in captured:
            tail = captured.split(_PI05_SWALLOWED_LOAD_WARNING, 1)[1].strip()
            raise SystemExit(
                "\n"
                f"Refusing to fine-tune from {pretrained_name_or_path}: the "
                f"checkpoint did not load cleanly into the reconstructed "
                f"TinyPI05 model. The PI05 loader swallowed the following "
                f"strict-load error and would have continued with random-init "
                f"weights in the affected layers:\n\n"
                f"{_PI05_SWALLOWED_LOAD_WARNING} {tail}\n\n"
                "Common causes:\n"
                "  * `pretrained_vision_model` / `pretrained_language_embeddings` in "
                "the checkpoint config do not match what was used to build the model "
                "(they double as architecture switches: vision layer count, presence "
                "of the SigLIP attention-pooling head, embed_tokens vocab size).\n"
                "  * Architecture preset / overrides differ from the original run.\n"
            )
        return model

    modeling_pi05.PI05Policy.from_pretrained = wrapped
    try:
        yield
    finally:
        modeling_pi05.PI05Policy.from_pretrained = original


DEFAULT_DATASET = "jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed"
DEFAULT_DATASET_ROOT = "outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed"


def _optional_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _architecture_overrides(args: argparse.Namespace) -> dict:
    fields = [
        "vlm_width",
        "vlm_depth",
        "vlm_mlp_dim",
        "vlm_num_heads",
        "vlm_num_kv_heads",
        "vlm_head_dim",
        "expert_width",
        "expert_depth",
        "expert_mlp_dim",
        "expert_num_heads",
        "expert_num_kv_heads",
        "expert_head_dim",
        "vision_hidden_size",
        "vision_intermediate_size",
        "vision_num_hidden_layers",
        "vision_num_attention_heads",
        "vision_patch_size",
    ]
    return {field: getattr(args, field) for field in fields if getattr(args, field) is not None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyPI05 from scratch on the cube subtasks dataset.")
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--job-name", default="tinypi05_cube")
    parser.add_argument("--steps", type=_optional_positive_int, default=30_000)
    parser.add_argument("--batch-size", type=_optional_positive_int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=_optional_positive_int, default=5_000)
    parser.add_argument("--log-freq", type=_optional_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--architecture-preset", default="small_500m")
    parser.add_argument("--image-size", type=_optional_positive_int, default=224)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--optimizer-lr", type=float, default=2.5e-5)
    parser.add_argument(
        "--optimizer-lr-vision",
        type=float,
        default=2.5e-6,
        help=(
            "Learning rate for the (pretrained) vision tower. Lower than --optimizer-lr "
            "so SigLIP weights are gently fine-tuned instead of clobbered. "
            "Pass a value <= 0 to disable per-group LRs and use --optimizer-lr for everything."
        ),
    )
    parser.add_argument(
        "--pretrained-vision-model",
        default="google/siglip-base-patch16-224",
        help=(
            "HF id of a SigLIP checkpoint to load into the vision_tower slot. "
            "Pass 'none' (or empty) to keep the legacy random-init vision tower."
        ),
    )
    parser.add_argument(
        "--pretrained-language-embeddings",
        default=None,
        help=(
            "HF id of a causal-LM checkpoint whose `embed_tokens` matrix should "
            "bootstrap the random VLM (e.g. 'google/gemma-3-270m'). Requires the "
            "source hidden_size to match vlm_width; when set without an explicit "
            "--architecture-preset, the script auto-selects 'gemma3_270m_emb'. "
            "The source's tokenizer is also auto-paired."
        ),
    )
    parser.add_argument(
        "--optimizer-lr-language-embeddings",
        type=float,
        default=2.5e-6,
        help=(
            "Learning rate for the pretrained `embed_tokens` matrix. Lower than "
            "--optimizer-lr so the embedding priors are gently fine-tuned. "
            "Pass <= 0 to use --optimizer-lr for the embeddings as well."
        ),
    )
    parser.add_argument("--scheduler-warmup-steps", type=int, default=1_000)
    parser.add_argument("--scheduler-decay-steps", type=int, default=30_000)
    parser.add_argument("--scheduler-decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument(
        "--use-critical-section-weights",
        action="store_true",
        help=(
            "Apply per-timestep loss weights from a `meta/critical_sections.json` "
            "annotation file. Annotated timesteps (in episode-relative seconds) get "
            "an elevated MSE weight; everything else stays at 1.0."
        ),
    )
    parser.add_argument(
        "--critical-sections-path",
        type=str,
        default=None,
        help=(
            "Override the path to `critical_sections.json`. When omitted, defaults "
            "to `<dataset-root>/meta/critical_sections.json`."
        ),
    )
    parser.add_argument(
        "--critical-section-default-weight",
        type=float,
        default=5.0,
        help="Default weight applied to timesteps inside an annotated section without an explicit weight.",
    )
    parser.add_argument(
        "--only-critical-annotated-episodes",
        action="store_true",
        help=(
            "Restrict the dataloader sampler to only episodes that have at least one "
            "critical section. Requires --use-critical-section-weights and a valid "
            "annotations file."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Resume a previous run. Pass either a `pretrained_model` directory "
            "(e.g. `outputs/train/<run>/checkpoints/005000/pretrained_model`) "
            "or its parent step directory. The saved train config is loaded so "
            "the policy/dataset/optimizer settings exactly match the original run; "
            "only operational flags (--steps, --save-freq, --log-freq, --num-workers, "
            "--wandb, --wandb-project) are overridden from the CLI."
        ),
    )
    parser.add_argument(
        "--finetune-from",
        type=Path,
        default=None,
        help=(
            "Fine-tune from a previous checkpoint. Pass either a `pretrained_model` "
            "directory (e.g. `outputs/train/<run>/checkpoints/005000/pretrained_model`) "
            "or its parent step directory. The policy *architecture* (preset, dtype, "
            "freeze flags, vision/embedding presence) is loaded from the checkpoint and "
            "its weights are loaded into the model, but everything else is fresh: new "
            "dataset, new batch size, new output dir, new optimizer/scheduler (driven by "
            "--optimizer-lr / --scheduler-* CLI flags), and `cfg.resume=False`. "
            "Architecture override flags (--architecture-preset, --vlm-*, --expert-*, "
            "--vision-*, --image-size, --dtype, --no-gradient-checkpointing, "
            "--pretrained-vision-model, --pretrained-language-embeddings) are ignored "
            "because the architecture is fixed by the checkpoint. Mutually exclusive "
            "with --resume-from."
        ),
    )

    for field in [
        "vlm-width",
        "vlm-depth",
        "vlm-mlp-dim",
        "vlm-num-heads",
        "vlm-num-kv-heads",
        "vlm-head-dim",
        "expert-width",
        "expert-depth",
        "expert-mlp-dim",
        "expert-num-heads",
        "expert-num-kv-heads",
        "expert-head-dim",
        "vision-hidden-size",
        "vision-intermediate-size",
        "vision-num-hidden-layers",
        "vision-num-attention-heads",
        "vision-patch-size",
    ]:
        parser.add_argument(f"--{field}", type=_optional_positive_int, default=None)

    return parser.parse_args()


def _resolve_resume_paths(resume_from: Path) -> tuple[Path, Path, Path]:
    """Return (pretrained_model_dir, step_dir, run_dir) from a user-supplied path.

    Accepts either the `pretrained_model` directory (containing `train_config.json`)
    or the parent step directory (containing a `pretrained_model/` subdir).
    """
    candidate = resume_from.expanduser().resolve()
    if (candidate / TRAIN_CONFIG_NAME).is_file():
        pretrained_model_dir = candidate
    elif (candidate / "pretrained_model" / TRAIN_CONFIG_NAME).is_file():
        pretrained_model_dir = candidate / "pretrained_model"
    else:
        raise FileNotFoundError(
            f"Could not find {TRAIN_CONFIG_NAME} under {candidate}. Pass either a "
            "`pretrained_model` directory or its parent step directory."
        )

    step_dir = pretrained_model_dir.parent
    # step_dir layout: <run_dir>/checkpoints/<step>
    if step_dir.parent.name != "checkpoints":
        raise ValueError(
            f"Unexpected checkpoint layout: {step_dir} is not inside a `checkpoints/` dir."
        )
    run_dir = step_dir.parent.parent
    return pretrained_model_dir, step_dir, run_dir


def _resume_training(args: argparse.Namespace) -> None:
    pretrained_model_dir, _step_dir, run_dir = _resolve_resume_paths(args.resume_from)

    cfg = TrainPipelineConfig.from_pretrained(pretrained_model_dir)
    cfg.resume = True
    cfg.output_dir = run_dir

    cfg.steps = args.steps
    cfg.save_freq = args.save_freq
    cfg.log_freq = args.log_freq
    cfg.num_workers = args.num_workers
    # Preserve the original wandb run_id (saved in cfg.wandb) so the resumed run
    # continues the same wandb run, but let the CLI flip enable/project on/off.
    cfg.wandb.enable = args.wandb
    cfg.wandb.project = args.wandb_project

    # `TrainPipelineConfig.validate()` reads `--config_path` from sys.argv to
    # populate `policy.pretrained_path` and `checkpoint_path` when resuming.
    config_path_arg = f"--config_path={pretrained_model_dir / TRAIN_CONFIG_NAME}"
    if config_path_arg not in sys.argv:
        sys.argv.append(config_path_arg)

    train(cfg)


def _finetune_training(args: argparse.Namespace) -> None:
    pretrained_model_dir, _step_dir, _run_dir = _resolve_resume_paths(args.finetune_from)

    # Load the policy config (architecture, dtype, freeze flags, etc.) from the
    # checkpoint so the model we instantiate matches the saved weights exactly.
    # Dispatch through the base class so the `type` field in config.json picks
    # the right subclass via the draccus choice registry.
    policy = PreTrainedConfig.from_pretrained(pretrained_model_dir)
    if not isinstance(policy, TinyPI05Config):
        raise SystemExit(
            f"Expected a tinypi05 checkpoint, got policy type {type(policy).__name__}."
        )

    # NOTE: keep `pretrained_vision_model` and `pretrained_language_embeddings`
    # as they appear in the checkpoint's config -- they do double duty as
    # architecture switches at construction time (e.g. SigLIP-base adds 2 extra
    # vision layers + an attention-pooling `head` module that the raw preset
    # does NOT have, and gemma-3-270m has vocab=262144 vs the PaliGemma default
    # 257152).  If we nulled them out, the rebuilt model would have the wrong
    # number of layers / vocab and the checkpoint state_dict load would fail
    # with "Unexpected key" errors and embed_tokens shape mismatches.  The HF
    # download here is wasteful (the checkpoint weights overwrite both) but
    # required for the shapes to match.

    # Tell make_policy() to load the checkpoint weights into the freshly
    # constructed model.
    policy.pretrained_path = pretrained_model_dir

    # Override LR / scheduler from CLI so the user controls the fine-tune
    # schedule (the saved values are appropriate for the original run, not for
    # a fresh fine-tune which usually wants its own warmup / decay horizon).
    optimizer_lr_vision: float | None = args.optimizer_lr_vision
    if optimizer_lr_vision is not None and optimizer_lr_vision <= 0:
        optimizer_lr_vision = None
    optimizer_lr_language_embeddings: float | None = args.optimizer_lr_language_embeddings
    if optimizer_lr_language_embeddings is not None and optimizer_lr_language_embeddings <= 0:
        optimizer_lr_language_embeddings = None

    policy.optimizer_lr = args.optimizer_lr
    policy.optimizer_lr_vision = optimizer_lr_vision
    policy.optimizer_lr_language_embeddings = optimizer_lr_language_embeddings
    policy.scheduler_warmup_steps = args.scheduler_warmup_steps
    policy.scheduler_decay_steps = args.scheduler_decay_steps
    policy.scheduler_decay_lr = args.scheduler_decay_lr
    policy.push_to_hub = False

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            use_imagenet_stats=False,
        ),
        policy=policy,
        output_dir=args.output_dir,
        job_name=args.job_name,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        wandb=WandBConfig(enable=args.wandb, project=args.wandb_project),
        use_critical_section_weights=args.use_critical_section_weights,
        critical_sections_path=args.critical_sections_path,
        critical_section_default_weight=args.critical_section_default_weight,
        only_critical_annotated_episodes=args.only_critical_annotated_episodes,
    )
    with _strict_pi05_state_dict_load():
        train(cfg)


def main() -> None:
    register_third_party_plugins()
    args = parse_args()

    if args.resume_from is not None and args.finetune_from is not None:
        raise SystemExit("--resume-from and --finetune-from are mutually exclusive.")

    if args.resume_from is not None:
        _resume_training(args)
        return

    if args.finetune_from is not None:
        _finetune_training(args)
        return

    pretrained_vision_model: str | None = args.pretrained_vision_model
    if pretrained_vision_model is not None:
        normalized = pretrained_vision_model.strip().lower()
        if normalized in {"", "none", "null"}:
            pretrained_vision_model = None

    optimizer_lr_vision: str | float | None = args.optimizer_lr_vision
    if optimizer_lr_vision is not None and optimizer_lr_vision <= 0:
        optimizer_lr_vision = None

    pretrained_language_embeddings: str | None = args.pretrained_language_embeddings
    if pretrained_language_embeddings is not None:
        normalized = pretrained_language_embeddings.strip().lower()
        if normalized in {"", "none", "null"}:
            pretrained_language_embeddings = None

    optimizer_lr_language_embeddings: float | None = args.optimizer_lr_language_embeddings
    if optimizer_lr_language_embeddings is not None and optimizer_lr_language_embeddings <= 0:
        optimizer_lr_language_embeddings = None

    architecture_preset = args.architecture_preset
    # Auto-pair gemma-3-270m embeddings with the width-compatible preset when
    # the user did not explicitly override --architecture-preset.
    if (
        pretrained_language_embeddings == "google/gemma-3-270m"
        and architecture_preset == "small_500m"
    ):
        architecture_preset = "gemma3_270m_emb"

    policy = TinyPI05Config(
        architecture_preset=architecture_preset,
        image_resolution=(args.image_size, args.image_size),
        dtype=args.dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        freeze_vision_encoder=False,
        train_expert_only=False,
        pretrained_vision_model=pretrained_vision_model,
        pretrained_language_embeddings=pretrained_language_embeddings,
        optimizer_lr=args.optimizer_lr,
        optimizer_lr_vision=optimizer_lr_vision,
        optimizer_lr_language_embeddings=optimizer_lr_language_embeddings,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_decay_steps=args.scheduler_decay_steps,
        scheduler_decay_lr=args.scheduler_decay_lr,
        push_to_hub=False,
        **_architecture_overrides(args),
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            use_imagenet_stats=False,
        ),
        policy=policy,
        output_dir=args.output_dir,
        job_name=args.job_name,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        wandb=WandBConfig(enable=args.wandb, project=args.wandb_project),
        use_critical_section_weights=args.use_critical_section_weights,
        critical_sections_path=args.critical_sections_path,
        critical_section_default_weight=args.critical_section_default_weight,
        only_critical_annotated_episodes=args.only_critical_annotated_episodes,
    )
    train(cfg)


if __name__ == "__main__":
    main()

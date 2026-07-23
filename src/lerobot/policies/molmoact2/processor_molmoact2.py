from __future__ import annotations

import json
import os
import random
import re
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import require_package

from .configuration_molmoact2 import MolmoAct2Config, infer_molmoact2_max_sequence_length
from .anchor_encoding import (
    AnchorDecodeStep,
    AnchorEncodeStep,
    policy_action_with_anchor_to_transition,
)

ACTION_OUTPUT_TOKEN = "<action_output>"  # nosec B105
ACTION_START_TOKEN = "<action_start>"  # nosec B105
ACTION_END_TOKEN = "<action_end>"  # nosec B105
ACTION_TOKEN_PREFIX = "<action_"  # nosec B105
STATE_START_TOKEN = "<state_start>"  # nosec B105
STATE_END_TOKEN = "<state_end>"  # nosec B105
STATE_TOKEN_PREFIX = "<state_"  # nosec B105

_QUESTION_TRAILING_SENTENCE_PUNCTUATION = ".,!?;:,\u2026"
_QUESTION_TRAILING_CLOSERS = "\"'\u201d\u2019)]}"
_QUESTION_SURROUNDING_DELIMITERS = "\"'`\u201c\u201d\u2018\u2019[](){}"
_QUESTION_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")


def _resolve_checkpoint_location(
    checkpoint_path: str,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> str:
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        raise ValueError("MolmoAct2 policy requires `base_path`.")
    local_path = Path(checkpoint_path).expanduser()
    if local_path.exists():
        return str(local_path)
    return snapshot_download(
        repo_id=checkpoint_path,
        repo_type="model",
        revision=revision,
        force_download=force_download,
        token=_hf_token(),
    )


def _load_hf_norm_stats_for_tag(
    checkpoint_path: str,
    *,
    revision: str | None,
    force_download: bool,
    norm_tag: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    norm_tag = str(norm_tag or "").strip()
    if not norm_tag:
        raise ValueError("MolmoAct2 HF checkpoint inference requires `policy.norm_tag` for normalization.")

    checkpoint_location = Path(
        _resolve_checkpoint_location(
            checkpoint_path,
            revision=revision,
            force_download=force_download,
        )
    )
    config_path = checkpoint_location / "config.json"
    norm_stats_filename = "norm_stats.json"
    if config_path.exists():
        with suppress(OSError, json.JSONDecodeError):
            norm_stats_filename = str(
                json.loads(config_path.read_text()).get("norm_stats_filename") or norm_stats_filename
            )

    stats_path = checkpoint_location / norm_stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(
            f"MolmoAct2 HF checkpoint is missing {norm_stats_filename!r}; cannot resolve norm_tag={norm_tag!r}."
        )
    payload = json.loads(stats_path.read_text())
    metadata_by_tag = payload.get("metadata_by_tag")
    if not isinstance(metadata_by_tag, dict):
        raise ValueError(f"MolmoAct2 norm stats file {stats_path} has no metadata_by_tag mapping.")
    metadata = metadata_by_tag.get(norm_tag)
    if metadata is None:
        available = sorted(str(tag) for tag in metadata_by_tag)
        raise ValueError(f"Unknown MolmoAct2 norm_tag={norm_tag!r}. Available tags: {available}.")
    if not isinstance(metadata, dict):
        raise ValueError(f"MolmoAct2 norm_tag={norm_tag!r} metadata must be a mapping.")

    def numeric_stats(raw_stats: dict[str, Any]) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for key, value in raw_stats.items():
            if key == "names":
                continue
            if isinstance(value, (list, tuple)) and any(isinstance(item, str) for item in value):
                continue
            stats[key] = deepcopy(value)
        return stats

    action_stats = metadata.get("action_stats")
    state_stats = metadata.get("state_stats")
    if not isinstance(action_stats, dict) or not isinstance(state_stats, dict):
        raise ValueError(f"MolmoAct2 norm_tag={norm_tag!r} must define action_stats and state_stats.")
    return {ACTION: numeric_stats(action_stats), OBS_STATE: numeric_stats(state_stats)}, metadata


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _normalize_image(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    while arr.ndim > 3 and int(arr.shape[0]) == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in {3, 4}:
        raise ValueError(f"Unsupported image shape for MolmoAct2: {arr.shape}.")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype in (np.float16, np.float32, np.float64):
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _normalize_question_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(_QUESTION_SURROUNDING_DELIMITERS).strip()
        for pattern in _QUESTION_PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    chunks = [chunk.strip() for chunk in re.split(r"[.!?]+", normalized) if chunk.strip()]
    if len(chunks) > 1:
        normalized = "; ".join(chunks)
    return normalized.lower()


def _build_discrete_state_string(state: np.ndarray, num_state_tokens: int) -> str:
    if num_state_tokens <= 0:
        raise ValueError(f"num_state_tokens must be > 0, got {num_state_tokens}.")
    arr = np.asarray(state, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_state_tokens - 1)
    token_ids = np.clip(np.rint(scaled).astype(np.int64), 0, int(num_state_tokens) - 1).reshape(-1)
    return f"{STATE_START_TOKEN}{''.join(f'{STATE_TOKEN_PREFIX}{int(token_id)}>' for token_id in token_ids)}{STATE_END_TOKEN}"


def _build_robot_text(
    *,
    task: str,
    discrete_state_string: str,
    num_images: int,
    current_subtask: str | None = None,
    metadata: dict[str, Any] | None = None,
    history_states: list[str] | None = None,
    history_image_spans: list[tuple[str, int, int]] | None = None,
) -> str:
    """Memory clauses: None disables a clause entirely (byte-identical legacy prompt).

    history_image_spans: (camera, first, last) 1-based image numbers of past frames
    appended after the current camera images (pretraining layout keeps cameras first)."""
    state_clause = (
        f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    )
    subtask_clause = f" The current step is {current_subtask}." if current_subtask else ""
    history_clause = (
        f" The recent states of the robot, oldest to newest, were {' '.join(history_states)}."
        if history_states
        else ""
    )
    history_clause += "".join(
        f" Images {first} to {last} are earlier frames from the {cam} camera, oldest to newest."
        for cam, first, last in history_image_spans or []
    )
    metadata_clause = ""
    if metadata is not None:
        if "quality" in metadata:
            metadata_clause += f" The quality is {int(metadata['quality'])} of 5."
        if "mistake" in metadata:
            metadata_clause += (
                " The robot made a mistake." if metadata["mistake"] else " The robot made no mistakes."
            )
        if "speed" in metadata:
            metadata_clause += f" The speed is {metadata['speed']}."
    prompt = (
        f"The task is to {task}.{subtask_clause}{state_clause}{history_clause}"
        f"{metadata_clause} "
        f"Given these, what action should the robot take to complete the task?"
    )
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(f"Image {idx + 1}<|image|>" for idx in range(num_images))
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{ACTION_OUTPUT_TOKEN}"


def _build_subtask_generation_text(
    *,
    task: str,
    discrete_state_string: str,
    num_images: int,
    summary: str | None = None,
) -> str:
    """Generation prompt (two-prompt design): same visual/state context as the action
    prompt, but the question asks for the next step. `summary` is the MEM language
    memory m_t conditioning the decision (None = clause off, "" = empty memory).
    The assistant's answer is memory-first: the updated memory, then the subtask
    (see `parse_generation_answer`); at training time the caller appends it (+eos)
    and puts CE labels on it."""
    state_clause = (
        f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    )
    memory_clause = "" if summary is None else f" Memory: {summary or EMPTY_MEMORY_TEXT}"
    prompt = (
        f"The task is to {task}.{memory_clause}{state_clause} "
        f"Given these, what step should the robot perform next?"
    )
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(f"Image {idx + 1}<|image|>" for idx in range(num_images))
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


EMPTY_MEMORY_TEXT = "none yet."


def build_generation_answer(subtask: str, summary: str | None) -> str:
    """Assistant answer for the generation prompt, memory-first: the updated memory is
    decoded before the subtask so subtask selection conditions on the fresh summary
    (None = subtask-only training)."""
    if summary is None:
        return subtask
    return f"{generation_answer_memory_prefix(summary)} Subtask: {subtask}"


def generation_answer_memory_prefix(summary: str | None) -> str | None:
    """The "Memory: …" prefix of build_generation_answer; None for subtask-only answers.
    Used to split the answer-span labels into memory vs subtask tokens."""
    if summary is None:
        return None
    return f"Memory: {summary or EMPTY_MEMORY_TEXT}"


def parse_generation_answer(text: str) -> tuple[str, str | None]:
    """Inverse of build_generation_answer for rollout decodes: returns
    (subtask_text, summary), summary None when the decode carried no memory span
    and "" for an empty memory."""
    if "Subtask:" not in text:
        return text.strip(), None
    memory, _, subtask = text.partition("Subtask:")
    memory = memory.strip()
    memory = memory.removeprefix("Memory:").strip()
    return subtask.strip().rstrip("."), "" if memory == EMPTY_MEMORY_TEXT else memory


def snap_to_subtask_vocab(text: str, names: list[str]) -> int:
    """Map generated text to the standardized subtask vocabulary: normalized exact
    match, else closest fuzzy match, else -1."""
    import difflib

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    normalized = {norm(name): i for i, name in enumerate(names)}
    key = norm(text)
    if key in normalized:
        return normalized[key]
    close = difflib.get_close_matches(key, list(normalized), n=1, cutoff=0.6)
    return normalized[close[0]] if close else -1


def _as_text_list(value: Any, batch_size: int) -> list[str]:
    if value is None:
        return [""] * batch_size
    if isinstance(value, str):
        return [value] * batch_size
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [str(value.item())] * batch_size
        flat = value.detach().cpu().reshape(-1).tolist()
        texts = [str(item) for item in flat]
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [str(value.item())] * batch_size
        texts = [str(item) for item in value.reshape(-1).tolist()]
    elif isinstance(value, (list, tuple)):
        texts = [str(item) for item in value]
    else:
        texts = [str(value)]
    if len(texts) == batch_size:
        return texts
    if len(texts) == 1:
        return texts * batch_size
    raise ValueError(f"Expected {batch_size} task strings, got {len(texts)}.")


def _tokenize_discrete_action(action: np.ndarray, processor: Any) -> list[int]:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 1:
        arr = arr[None, None, :]
    tokens_out = processor(arr)
    if isinstance(tokens_out, dict):
        tokens_out = tokens_out.get("input_ids", next(iter(tokens_out.values())))
    if isinstance(tokens_out, np.ndarray):
        tokens_out = tokens_out.tolist()
    if torch.is_tensor(tokens_out):
        tokens_out = tokens_out.detach().cpu().tolist()
    if not isinstance(tokens_out, list):
        raise TypeError(f"Unexpected discrete action tokenizer output type: {type(tokens_out)}")
    if tokens_out and isinstance(tokens_out[0], (list, tuple, np.ndarray)):
        tokens_out = tokens_out[0]
    return [int(token_id) for token_id in tokens_out]


def _build_discrete_action_string(action: np.ndarray, processor: Any) -> str:
    token_ids = _tokenize_discrete_action(action, processor)
    pieces = "".join(f"{ACTION_TOKEN_PREFIX}{int(token_id)}>" for token_id in token_ids)
    return f"{ACTION_START_TOKEN}{pieces}{ACTION_END_TOKEN}"


def _single_token_id(tokenizer: Any, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"MolmoAct2 token {token!r} must encode to one token, got {token_ids}.")
    return int(token_ids[0])


def _flatten_feature_names(raw_names: Any) -> list[str] | None:
    if raw_names is None:
        return None
    if isinstance(raw_names, dict):
        names: list[str] = []
        for value in raw_names.values():
            if isinstance(value, (list, tuple)):
                names.extend(str(item) for item in value)
            elif value is not None:
                names.append(str(value))
        return names or None
    if isinstance(raw_names, (list, tuple)):
        names = [str(item) for item in raw_names]
        return names or None
    return [str(raw_names)]


def _feature_dim(stats: dict[str, Any] | None) -> int | None:
    if not isinstance(stats, dict):
        return None
    for key in ("mean", "std", "min", "max", "q01", "q99", "q10", "q90", "mask"):
        value = stats.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            return int(value.shape[-1]) if value.ndim > 0 else None
        arr = np.asarray(value)
        return int(arr.shape[-1]) if arr.ndim > 0 else None
    return None


def _feature_names_from_meta(dataset_meta: Any | None, feature_key: str) -> list[str] | None:
    if dataset_meta is None:
        return None

    root = getattr(dataset_meta, "root", None)
    candidate_roots = []
    if root is not None:
        repo_id = str(getattr(dataset_meta, "repo_id", "") or "").strip()
        if repo_id:
            candidate_roots.append(Path(root) / repo_id)
        candidate_roots.append(Path(root))
    for candidate_root in candidate_roots:
        info_path = candidate_root / "meta" / "info.json"
        if info_path.exists():
            try:
                with info_path.open("r", encoding="utf-8") as f:
                    info = json.load(f)
                names = _flatten_feature_names((info.get("features") or {}).get(feature_key, {}).get("names"))
                if names:
                    return names
            except (OSError, json.JSONDecodeError, AttributeError):
                pass

    for container in (
        getattr(getattr(dataset_meta, "info", None), "features", None),
        getattr(dataset_meta, "features", None),
    ):
        if not isinstance(container, dict):
            continue
        feature = container.get(feature_key)
        if not isinstance(feature, dict):
            continue
        names = _flatten_feature_names(feature.get("names"))
        if names:
            return names
    return None


def _add_gripper_masks_to_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
    dataset_meta: Any | None,
    *,
    normalize_gripper: bool,
    dataset_feature_names: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]] | None:
    if not dataset_stats:
        return dataset_stats

    stats = deepcopy(dataset_stats)
    for key in (ACTION, OBS_STATE):
        feature_stats = stats.get(key)
        if not isinstance(feature_stats, dict):
            continue
        dim = _feature_dim(feature_stats)
        if dim is None:
            continue

        if normalize_gripper:
            feature_stats["mask"] = [True] * dim
            continue

        names = _flatten_feature_names((dataset_feature_names or {}).get(key))
        if names is None:
            names = _feature_names_from_meta(dataset_meta, key)
        if names is None:
            names = _flatten_feature_names(feature_stats.get("names"))
        if names is None:
            continue
        if len(names) != dim:
            continue
        feature_stats["mask"] = ["gripper" not in name.lower() for name in names]
    return stats


class _MolmoAct2MaskedNormalizationMixin:
    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: Any, *, inverse: bool = False
    ) -> Tensor:
        transformed = super()._apply_transform(tensor, key, feature_type, inverse=inverse)
        stats = getattr(self, "_tensor_stats", {}).get(key, {})

        mask = stats.get("mask") if isinstance(stats, dict) else None
        if mask is None:
            return transformed
        mask = mask.to(device=tensor.device, dtype=torch.bool)
        if mask.ndim != 1 or tensor.shape[-1] != mask.shape[0]:
            return transformed
        while mask.ndim < tensor.ndim:
            mask = mask.unsqueeze(0)
        return torch.where(mask, transformed, tensor)


@ProcessorStepRegistry.register(name="molmoact2_masked_normalizer")
@dataclass
class MolmoAct2MaskedNormalizerProcessorStep(_MolmoAct2MaskedNormalizationMixin, NormalizerProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """The base normalizer only touches OBSERVATION/ACTION; the short-term-memory
        history window rides in COMPLEMENTARY_DATA (batch_to_transition routes any
        "history.*" key there), so without this it would reach the prompt as raw
        (un-normalized) joint values instead of the [-1, 1] range _build_discrete_state_string
        expects — normalize it here with the exact same OBS_STATE stats/mask."""
        transition = super().__call__(transition)
        history_key = f"history.{OBS_STATE}"
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if isinstance(complementary, dict) and history_key in complementary and OBS_STATE in self.features:
            transition = transition.copy()
            complementary = dict(complementary)
            tensor = torch.as_tensor(complementary[history_key])
            complementary[history_key] = self._apply_transform(
                tensor, OBS_STATE, self.features[OBS_STATE].type, inverse=False
            )
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition


@ProcessorStepRegistry.register(name="molmoact2_masked_unnormalizer")
@dataclass
class MolmoAct2MaskedUnnormalizerProcessorStep(_MolmoAct2MaskedNormalizationMixin, UnnormalizerProcessorStep):
    pass


def _masked_clamp(tensor: Tensor, mask: list[bool] | None) -> Tensor:
    """Clamp to [-1, 1] only on dimensions where mask is True (normalized dims).
    Dims where mask is False are in raw units (e.g. raw degrees) and must not be clamped."""
    t = torch.as_tensor(tensor)
    if mask is None:
        return t.clamp(-1.0, 1.0)
    m = torch.tensor(mask, dtype=torch.bool, device=t.device)
    while m.ndim < t.ndim:
        m = m.unsqueeze(0)
    return torch.where(m, t.clamp(-1.0, 1.0), t)


@ProcessorStepRegistry.register(name="molmoact2_clamp_normalized")
@dataclass
class MolmoAct2ClampNormalizedProcessorStep(ProcessorStep):
    """Clamp q01/q99-normalized state and action to the range used by the old trainer.
    action_mask / state_mask mark which dims are actually normalized; unmasked (raw-unit)
    dims are skipped so they are not corrupted by the [-1, 1] clamp."""

    action_mask: list[bool] | None = None
    state_mask: list[bool] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict) and OBS_STATE in observation:
            observation = observation.copy()
            observation[OBS_STATE] = _masked_clamp(observation[OBS_STATE], self.state_mask)
            transition[TransitionKey.OBSERVATION] = observation
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = _masked_clamp(action, self.action_mask)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="molmoact2_pack_inputs")
@dataclass
class MolmoAct2PackInputsProcessorStep(ProcessorStep):
    base_path: str
    base_revision: str | None = None
    base_force_download: bool = False
    trust_remote_code: bool = True
    action_mode: str = "both"
    discrete_action_tokenizer: str = "allenai/MolmoAct2-FAST-Tokenizer"
    image_keys: list[str] = field(default_factory=list)
    normalize_language: bool = True
    # Accepted and ignored: present in processor configs saved before the setup/control
    # prompt clauses were removed (2026-07-22), and saved configs load as raw kwargs.
    setup_type: str = ""
    control_mode: str = ""
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    num_state_tokens: int = 256
    max_sequence_length: int | None = None
    chunk_size: int = 30
    max_action_dim: int = 32
    env_action_dim: int | None = None
    # Memory prompt clauses: index → name vocabulary (from subtasks.parquet) and
    # per-component training dropout (π0.7 recipe; applied only when actions are
    # present, i.e. training text — inference prompts are deterministic).
    subtask_names: list[str] = field(default_factory=list)
    subtask_dropout: float = 0.3
    metadata_dropout: float = 0.15
    # MEM summary memory: text table indexed by the buffer's summary_*_index
    # columns (synced from summaries.parquet); dropout removes BOTH the prompt's
    # memory clause and the answer's memory span (subtask-only sample).
    summary_texts: list[str] = field(default_factory=list)
    summary_dropout: float = 0.3
    # Short-term memory: recent-state clause built from the buffer/RTC lookback windows
    # (ReplayBuffer.sample() / assemble_history_windows, complementary key
    # "history.{OBS_STATE}"). Absent key = clause off (byte-identical legacy prompt);
    # dropout only applies to training text, same convention as the other clauses.
    history_dropout: float = 0.3
    # Runtime toggle (not persisted): "action" builds the action prompt;
    # "subtask_generation" builds the generation prompt/labels instead. Callers
    # flip it around a pipeline call so generation gets the SAME normalization.
    prompt_mode: str = "action"

    def __post_init__(self) -> None:
        require_package("transformers", extra="molmoact2")
        from transformers import AutoProcessor

        checkpoint_location = _resolve_checkpoint_location(
            self.base_path,
            revision=self.base_revision,
            force_download=bool(self.base_force_download),
        )
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_location,
            trust_remote_code=self.trust_remote_code,
            use_fast=False,
            token=_hf_token(),
        )
        self.action_processor = None
        if self.action_mode in {"discrete", "both"}:
            self.action_processor = AutoProcessor.from_pretrained(
                self.discrete_action_tokenizer,
                trust_remote_code=self.trust_remote_code,
                token=_hf_token(),
            )
        self._action_start_id = _single_token_id(self.processor.tokenizer, ACTION_START_TOKEN)
        self._action_end_id = _single_token_id(self.processor.tokenizer, ACTION_END_TOKEN)
        self._eos_token = self.processor.tokenizer.eos_token or ""
        self._eos_token_id = self.processor.tokenizer.eos_token_id

    def get_config(self) -> dict[str, Any]:
        return {
            "base_path": self.base_path,
            "base_revision": self.base_revision,
            "base_force_download": self.base_force_download,
            "trust_remote_code": self.trust_remote_code,
            "action_mode": self.action_mode,
            "discrete_action_tokenizer": self.discrete_action_tokenizer,
            "image_keys": list(self.image_keys),
            "normalize_language": self.normalize_language,
            "num_state_tokens": self.num_state_tokens,
            "max_sequence_length": self.max_sequence_length,
            "chunk_size": self.chunk_size,
            "max_action_dim": self.max_action_dim,
            "env_action_dim": self.env_action_dim,
            "subtask_names": list(self.subtask_names),
            "subtask_dropout": self.subtask_dropout,
            "metadata_dropout": self.metadata_dropout,
            "summary_texts": list(self.summary_texts),
            "summary_dropout": self.summary_dropout,
            "history_dropout": self.history_dropout,
        }

    def _resolve_max_sequence_length(
        self,
        *,
        num_images: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        include_discrete_action: bool,
        history_num_samples: int = 0,
    ) -> int:
        if self.max_sequence_length is not None:
            return int(self.max_sequence_length)
        return infer_molmoact2_max_sequence_length(
            num_images=num_images,
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            include_discrete_action=include_discrete_action,
            history_num_samples=history_num_samples,
        )

    def _fix_attention_mask(self, inputs) -> None:
        """Recompute attention_mask from the pad id: the HF processor's insert_bos assumes
        LEFT padding, so with the tokenizer's right padding it marks trailing pad tokens
        as valid — the model would attend padding and the answer-span label math breaks."""
        pad_id = self.processor.tokenizer.pad_token_id
        inputs["attention_mask"] = (inputs["input_ids"] != pad_id).to(inputs["attention_mask"].dtype)

    def _batch_size(self, observation: dict[str, Any], action: Tensor | None) -> int:
        if action is not None:
            return int(action.shape[0])
        state = observation.get(OBS_STATE)
        if torch.is_tensor(state) or isinstance(state, np.ndarray):
            return int(state.shape[0]) if getattr(state, "ndim", 0) > 1 else 1
        for key in self._resolve_image_keys(observation):
            value = observation[key]
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                return int(value.shape[0]) if getattr(value, "ndim", 0) == 4 else 1
        return 1

    def _resolve_image_keys(self, observation: dict[str, Any]) -> list[str]:
        if self.image_keys:
            missing = [key for key in self.image_keys if key not in observation]
            if missing:
                raise ValueError(f"MolmoAct2 image_keys missing from observation: {missing}.")
            return list(self.image_keys)
        keys = [key for key in observation if str(key).startswith(f"{OBS_IMAGES}.")]
        if not keys:
            keys = [key for key in observation if str(key).startswith("observation.image")]
        if not keys:
            raise ValueError("MolmoAct2 requires at least one image observation.")
        return sorted(keys)

    def _extract_images(self, observation: dict[str, Any], batch_size: int) -> list[list[np.ndarray]]:
        images_by_example: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
        for key in self._resolve_image_keys(observation):
            value = observation[key]
            for batch_idx in range(batch_size):
                item = value
                if (torch.is_tensor(value) or isinstance(value, np.ndarray)) and getattr(
                    value, "ndim", 0
                ) >= 4:
                    item = value[batch_idx]
                images_by_example[batch_idx].append(_normalize_image(item))
        return images_by_example

    def _extract_state(self, observation: dict[str, Any], batch_size: int) -> Tensor:
        if OBS_STATE not in observation:
            raise ValueError("MolmoAct2 requires observation.state for discrete state prompting.")
        state = torch.as_tensor(observation[OBS_STATE], dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if int(state.shape[0]) != batch_size:
            raise ValueError(f"State batch size {state.shape[0]} does not match batch size {batch_size}.")
        return state

    def _pad_action(self, action: Tensor, action_is_pad: Any | None) -> tuple[Tensor, Tensor, Tensor]:
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.ndim != 3:
            raise ValueError(f"MolmoAct2 expected action shape [B, T, D], got {tuple(action.shape)}.")
        if action.shape[-1] > self.max_action_dim:
            raise ValueError(
                f"Action dim {action.shape[-1]} exceeds MolmoAct2 max_action_dim={self.max_action_dim}."
            )
        padded = torch.zeros(
            (*action.shape[:-1], self.max_action_dim),
            device=action.device,
            dtype=torch.float32,
        )
        padded[..., : action.shape[-1]] = action.to(dtype=torch.float32)
        action_dim_is_pad = torch.ones(
            (action.shape[0], self.max_action_dim), device=action.device, dtype=torch.bool
        )
        action_dim_is_pad[:, : action.shape[-1]] = False
        if action_is_pad is None:
            action_horizon_is_pad = torch.zeros(action.shape[:2], device=action.device, dtype=torch.bool)
        else:
            action_horizon_is_pad = torch.as_tensor(action_is_pad, device=action.device, dtype=torch.bool)
            if action_horizon_is_pad.ndim == 1:
                action_horizon_is_pad = action_horizon_is_pad.unsqueeze(0)
            if tuple(action_horizon_is_pad.shape) != tuple(action.shape[:2]):
                raise ValueError(
                    "action_is_pad must match action horizon shape: "
                    f"got {tuple(action_horizon_is_pad.shape)} for action {tuple(action.shape)}."
                )
        return padded, action_horizon_is_pad, action_dim_is_pad

    def _build_labels(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        labels = torch.full_like(input_ids, -100)
        for batch_idx in range(input_ids.shape[0]):
            valid = attention_mask[batch_idx].to(dtype=torch.bool)
            row = input_ids[batch_idx]
            starts = (row == self._action_start_id).nonzero(as_tuple=False).flatten().tolist()
            ends = (row == self._action_end_id).nonzero(as_tuple=False).flatten().tolist()
            end_ptr = 0
            for start in starts:
                while end_ptr < len(ends) and ends[end_ptr] < start:
                    end_ptr += 1
                if end_ptr >= len(ends):
                    raise ValueError(
                        "Found <action_start> without matching <action_end> in MolmoAct2 labels."
                    )
                end = int(ends[end_ptr])
                label_end = end + 1
                if (
                    self._eos_token_id is not None
                    and label_end < int(row.shape[0])
                    and int(row[label_end]) == int(self._eos_token_id)
                ):
                    label_end += 1
                labels[batch_idx, start:label_end] = row[start:label_end]
                end_ptr += 1
            if not starts:
                raise ValueError("No discrete action span found in MolmoAct2 training text.")
            labels[batch_idx] = torch.where(
                valid, labels[batch_idx], torch.full_like(labels[batch_idx], -100)
            )
        return labels

    def _extract_tasks(self, observation: dict, complementary: dict, batch_size: int) -> list[str]:
        task_source = complementary.get("task")
        if task_source is None:
            task_source = observation.get("task")
        if task_source is None:
            task_source = observation.get("observation.language")
        if task_source is None:
            task_source = complementary.get("language_instruction")
        tasks = _as_text_list(task_source, batch_size)
        if self.normalize_language:
            tasks = [_normalize_question_text(task) for task in tasks]
        return tasks

    def _pack_subtask_generation(self, transition: EnvTransition) -> EnvTransition:
        """Pack the subtask-generation prompt (two-prompt design). Runs INSIDE the
        pipeline (prompt_mode toggle) so states arrive normalized like the action path.

        When samples carry a subtask name (training), the text is prompt + name +
        eos and complementary "labels" holds CE targets on the answer span only
        (padding-side agnostic: the last answer_len non-pad positions);
        "subtask_valid" marks samples that had a name. Without names (rollout),
        prompts only — feed to generate_subtask_tokens.
        """
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        batch_size = self._batch_size(observation, None)
        state = self._extract_state(observation, batch_size)
        images_by_example = self._extract_images(observation, batch_size)
        tasks = self._extract_tasks(observation, complementary, batch_size)
        subtask_texts = self._extract_subtask_texts(complementary, batch_size)
        prev_summaries, target_summaries = self._extract_summaries(complementary, batch_size)

        state_np = state.detach().cpu().numpy()
        prompts: list[str] = []
        fulls: list[str] = []
        mids: list[str] = []  # prompt + memory prefix; splits the answer span into memory/subtask
        flat_images: list[np.ndarray] = []
        for batch_idx in range(batch_size):
            images = images_by_example[batch_idx]
            flat_images.extend(images)
            name = subtask_texts[batch_idx]
            prev_summary = prev_summaries[batch_idx]
            target_summary = target_summaries[batch_idx]
            if name is not None and random.random() < self.summary_dropout:
                prev_summary = None
                target_summary = None
            prompt = _build_subtask_generation_text(
                task=tasks[batch_idx],
                discrete_state_string=_build_discrete_state_string(
                    state_np[batch_idx], self.num_state_tokens
                ),
                num_images=len(images),
                summary=prev_summary,
            )
            prompts.append(prompt)
            if name:
                answer = build_generation_answer(name, target_summary)
                fulls.append(f"{prompt}{answer}{self._eos_token}")
                memory_prefix = generation_answer_memory_prefix(target_summary)
                mids.append(prompt if memory_prefix is None else f"{prompt}{memory_prefix}")
            else:
                fulls.append(prompt)
                mids.append(prompt)

        valid = torch.tensor([name is not None for name in subtask_texts])
        build_labels = bool(valid.any())
        inputs = self.processor(
            text=fulls if build_labels else prompts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
        )
        self._fix_attention_mask(inputs)
        if build_labels:
            pad_id = self.processor.tokenizer.pad_token_id
            prompt_inputs = self.processor(
                text=prompts, images=flat_images, return_tensors="pt", padding=True
            )
            full_lengths = inputs["attention_mask"].sum(dim=1)
            prompt_lengths = (prompt_inputs["input_ids"] != pad_id).sum(dim=1)
            mid_lengths = (
                prompt_lengths
                if mids == prompts
                else (
                    self.processor(text=mids, images=flat_images, return_tensors="pt", padding=True)[
                        "input_ids"
                    ]
                    != pad_id
                ).sum(dim=1)
            )
            labels = torch.full_like(inputs["input_ids"], -100)
            summary_mask = torch.zeros_like(labels, dtype=torch.bool)
            for batch_idx in range(batch_size):
                answer_len = int(full_lengths[batch_idx] - prompt_lengths[batch_idx])
                if not valid[batch_idx] or answer_len <= 0:
                    continue
                nonpad = inputs["attention_mask"][batch_idx].nonzero().reshape(-1)
                answer_span = nonpad[-answer_len:]
                labels[batch_idx, answer_span] = inputs["input_ids"][batch_idx, answer_span]
                memory_len = int(mid_lengths[batch_idx] - prompt_lengths[batch_idx])
                if memory_len > 0:
                    summary_mask[batch_idx, answer_span[:memory_len]] = True
            inputs["labels"] = labels
            inputs["summary_label_mask"] = summary_mask
        complementary.update(dict(inputs))
        complementary["subtask_valid"] = valid
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def _extract_subtask_texts(self, complementary: dict, batch_size: int) -> list[str | None]:
        """Current-subtask text per sample: the "subtask" strings when present (rollout
        path), else subtask_index rendered through the vocabulary (offline batches)."""
        texts = complementary.get("subtask")
        if texts is not None:
            return [t if t else None for t in _as_text_list(texts, batch_size)]
        indices = complementary.get("subtask_index")
        if indices is None or not self.subtask_names:
            return [None] * batch_size
        flat = torch.as_tensor(indices).detach().cpu().reshape(-1).long().tolist()
        if len(flat) == 1:
            flat = flat * batch_size
        return [self.subtask_names[i] if 0 <= i < len(self.subtask_names) else None for i in flat]

    def _extract_summaries(
        self, complementary: dict, batch_size: int
    ) -> tuple[list[str | None], list[str | None]]:
        """(conditioning, target) summary text per sample. Rollout path: "summary"
        strings (conditioning only, no target). Offline path: summary_prev_index /
        summary_target_index rendered through summary_texts; -1 = empty memory ("").
        All-None when summaries aren't wired (subtask-only training)."""
        texts = complementary.get("summary")
        if texts is not None:
            return list(texts), [None] * batch_size
        prev = complementary.get("summary_prev_index")
        if prev is None or not self.summary_texts:
            return [None] * batch_size, [None] * batch_size

        def render(indices) -> list[str]:
            flat = torch.as_tensor(indices).detach().cpu().reshape(-1).long().tolist()
            return ["" if i < 0 else self.summary_texts[i] for i in flat]

        return render(prev), render(complementary["summary_target_index"])

    def _extract_history_states(self, complementary: dict, batch_size: int) -> list[list[str] | None]:
        """Discrete-state strings for the short-term history window (oldest → newest
        per sample), read from complementary key "history.{OBS_STATE}" (shape
        (B, T_h, D), the ReplayBuffer.sample()/assemble_history_windows lookback).
        None per sample when history wasn't gathered (memory.history_keys empty or
        doesn't include OBS_STATE) — same clause-off convention as subtask/metadata."""
        history = complementary.get(f"history.{OBS_STATE}")
        if history is None:
            return [None] * batch_size
        arr = torch.as_tensor(history, dtype=torch.float32).detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[None]
        return [
            [_build_discrete_state_string(arr[b, t], self.num_state_tokens) for t in range(arr.shape[1])]
            for b in range(batch_size)
        ]

    def _extract_history_images(
        self, complementary: dict, batch_size: int
    ) -> list[list[tuple[str, list[np.ndarray]]] | None]:
        """Past camera frames for the short-term memory window, read from complementary
        keys "history.{OBS_IMAGES}.{cam}" (B, T_h, C, H, W; uint8 cache rows or
        policy-format floats — _normalize_image reconciles both). Per sample: a list of
        (camera, frames oldest → newest), or None when no image history was gathered."""
        prefix = f"history.{OBS_IMAGES}."
        keys = sorted(k for k in complementary if k.startswith(prefix) and not k.endswith("_is_pad"))
        if not keys:
            return [None] * batch_size
        out = []
        for batch_idx in range(batch_size):
            entry = []
            for key in keys:
                frames = complementary[key]
                item = frames[batch_idx] if getattr(frames, "ndim", 0) >= 5 else frames
                entry.append((key.removeprefix(prefix), [_normalize_image(f) for f in item]))
            out.append(entry)
        return out

    @staticmethod
    def _extract_metadata(complementary: dict, batch_size: int) -> list[dict | None]:
        metadata = complementary.get("metadata")
        if metadata is not None:
            if isinstance(metadata, dict):
                return [metadata] * batch_size
            return list(metadata)
        # Offline batches: per-frame metadata columns from materialize_metadata.
        # Speed is optional (omitted for single-operator data); the clause renders partially.
        quality = complementary.get("metadata_quality")
        if quality is None:
            return [None] * batch_size
        quality = torch.as_tensor(quality).detach().cpu().float().reshape(-1)
        mistake = torch.as_tensor(complementary["metadata_mistake"]).detach().cpu().float().reshape(-1)
        speed = complementary.get("metadata_speed")
        if speed is not None:
            speed = torch.as_tensor(speed).detach().cpu().float().reshape(-1)
        return [
            {"quality": int(quality[i]), "mistake": bool(mistake[i] > 0.5)}
            | ({"speed": int(speed[i])} if speed is not None else {})
            for i in range(batch_size)
        ]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.prompt_mode == "subtask_generation":
            return self._pack_subtask_generation(transition)
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        if not isinstance(observation, dict):
            raise ValueError("MolmoAct2 expected an observation dictionary.")
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        raw_action = transition.get(TransitionKey.ACTION)
        action = torch.as_tensor(raw_action, dtype=torch.float32) if raw_action is not None else None
        batch_size = self._batch_size(observation, action)
        state = self._extract_state(observation, batch_size)
        images_by_example = self._extract_images(observation, batch_size)

        tasks = self._extract_tasks(observation, complementary, batch_size)
        complementary["task"] = tasks

        action_padded = None
        action_horizon_is_pad = None
        action_dim_is_pad = torch.ones((batch_size, self.max_action_dim), dtype=torch.bool)
        real_action_dim = int(self.env_action_dim or 0)
        if action is not None:
            action_is_pad = complementary.get("action_is_pad")
            if action_is_pad is None:
                action_is_pad = complementary.get("action_horizon_is_pad")
            action_padded, action_horizon_is_pad, action_dim_is_pad = self._pad_action(action, action_is_pad)
            real_action_dim = int(action.shape[-1])
        elif real_action_dim > 0:
            action_dim_is_pad[:, :real_action_dim] = False

        subtask_texts = self._extract_subtask_texts(complementary, batch_size)
        metadata_list = self._extract_metadata(complementary, batch_size)
        history_states_list = self._extract_history_states(complementary, batch_size)
        history_images_list = self._extract_history_images(complementary, batch_size)

        prompt_texts: list[str] = []
        full_texts: list[str] = []
        flat_images: list[np.ndarray] = []
        state_np = state.detach().cpu().numpy()
        build_action_labels = action is not None and self.action_mode in {"discrete", "both"}
        max_num_images = 0
        for batch_idx in range(batch_size):
            images = images_by_example[batch_idx]
            discrete_state = _build_discrete_state_string(state_np[batch_idx], self.num_state_tokens)
            current_subtask = subtask_texts[batch_idx]
            metadata = metadata_list[batch_idx]
            history_states = history_states_list[batch_idx]
            history_images = history_images_list[batch_idx]
            if build_action_labels:  # training text: per-component dropout (π0.7 recipe)
                if random.random() < self.subtask_dropout:
                    current_subtask = None
                if random.random() < self.metadata_dropout:
                    metadata = None
                if random.random() < self.history_dropout:
                    # One flip drops the whole short-term block: states and frames
                    # describe the same window, dropping them independently would
                    # let the model exploit whichever survived.
                    history_states = None
                    history_images = None
            history_image_spans = None
            if history_images:
                images = list(images)
                history_image_spans = []
                for cam, frames in history_images:
                    history_image_spans.append((cam, len(images) + 1, len(images) + len(frames)))
                    images.extend(frames)
            flat_images.extend(images)
            max_num_images = max(max_num_images, len(images))
            prompt = _build_robot_text(
                task=tasks[batch_idx],
                discrete_state_string=discrete_state,
                num_images=len(images),
                history_states=history_states,
                current_subtask=current_subtask,
                metadata=metadata,
                history_image_spans=history_image_spans,
            )
            prompt_texts.append(prompt)
            if build_action_labels:
                if self.action_processor is None:
                    raise ValueError("Discrete MolmoAct2 training requires an action tokenizer.")
                answer = _build_discrete_action_string(
                    action[batch_idx].detach().cpu().numpy(), self.action_processor
                )
                full_texts.append(f"{prompt}{answer}{self._eos_token}")
            else:
                full_texts.append(prompt)

        text = full_texts if build_action_labels else prompt_texts
        inputs = self.processor(text=text, images=flat_images, return_tensors="pt", padding=True)
        self._fix_attention_mask(inputs)
        if action is None:
            action_horizon = self.chunk_size
        elif action.ndim == 2:
            action_horizon = 1
        else:
            action_horizon = int(action.shape[1])
        max_sequence_length = self._resolve_max_sequence_length(
            num_images=max_num_images,
            state_dim=int(state.shape[-1]),
            action_dim=max(real_action_dim, 1),
            action_horizon=action_horizon,
            include_discrete_action=build_action_labels,
            history_num_samples=max((len(h) for h in history_states_list if h), default=0),
        )
        if int(inputs["input_ids"].shape[1]) > max_sequence_length:
            raise ValueError(
                f"MolmoAct2 sequence length {int(inputs['input_ids'].shape[1])} exceeds "
                f"max_sequence_length={max_sequence_length}."
            )

        if build_action_labels:
            inputs["labels"] = self._build_labels(inputs["input_ids"], inputs["attention_mask"])

        complementary.update(dict(inputs))
        complementary["action_dim_is_pad"] = action_dim_is_pad
        if action_horizon_is_pad is not None:
            complementary["action_horizon_is_pad"] = action_horizon_is_pad

        if action_padded is not None:
            transition[TransitionKey.ACTION] = action_padded
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="molmoact2_clamp_action")
@dataclass
class MolmoAct2ClampActionProcessorStep(ProcessorStep):
    """Clamp model action output to [-1, 1] before unnormalization.
    action_mask marks which dims are normalized; unmasked (raw-unit) dims are skipped."""

    action_mask: list[bool] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = _masked_clamp(action, self.action_mask)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_molmoact2_pre_post_processors(
    config: MolmoAct2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
    action_stats_override: dict[str, torch.Tensor] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    env_action_dim = None
    if config.output_features and ACTION in config.output_features:
        env_action_dim = int(config.output_features[ACTION].shape[0])

    hf_metadata: dict[str, Any] = {}
    if dataset_stats is None and str(config.norm_tag or "").strip():
        dataset_stats, hf_metadata = _load_hf_norm_stats_for_tag(
            config.base_path,
            revision=config.base_revision,
            force_download=bool(config.base_force_download),
            norm_tag=config.norm_tag,
        )

    if action_stats_override is not None:
        if dataset_stats is None:
            dataset_stats = {}
        else:
            dataset_stats = deepcopy(dataset_stats)
        dataset_stats[ACTION] = action_stats_override

    image_keys = list(config.image_keys)
    if not image_keys and isinstance(hf_metadata.get("camera_keys"), list):
        image_keys = [str(key) for key in hf_metadata["camera_keys"]]
    chunk_size = int(hf_metadata.get("action_horizon") or config.chunk_size)

    masked_dataset_stats = _add_gripper_masks_to_stats(
        dataset_stats,
        dataset_meta,
        normalize_gripper=config.normalize_gripper,
        dataset_feature_names=config.dataset_feature_names,
    )

    def _mask_list(key: str) -> list[bool] | None:
        stats = (masked_dataset_stats or {}).get(key, {})
        m = stats.get("mask") if isinstance(stats, dict) else None
        if m is None:
            return None
        return [bool(v) for v in (m.tolist() if hasattr(m, "tolist") else m)]

    action_mask = _mask_list(ACTION)
    state_mask = _mask_list(OBS_STATE)

    action_encoding = getattr(config, "action_encoding", "absolute")
    use_anchor = action_encoding in ("anchor", "delta")

    # MemoryConfig lives on the RL wrapper config (MolmoAct2RLConfig.memory), not on the
    # bare MolmoAct2Config used for BC/eval, hence the getattr default.
    memory_cfg = getattr(config, "memory", None)
    history_dropout = memory_cfg.history_dropout if memory_cfg is not None else 0.3

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        *([AnchorEncodeStep(encoding=action_encoding)] if use_anchor else []),
        MolmoAct2MaskedNormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=masked_dataset_stats,
        ),
        MolmoAct2ClampNormalizedProcessorStep(action_mask=action_mask, state_mask=state_mask),
        MolmoAct2PackInputsProcessorStep(
            base_path=config.base_path,
            base_revision=config.base_revision,
            base_force_download=config.base_force_download,
            trust_remote_code=config.trust_remote_code,
            action_mode=config.action_mode,
            discrete_action_tokenizer=config.discrete_action_tokenizer,
            image_keys=image_keys,
            normalize_language=config.normalize_language,
            num_state_tokens=config.num_state_tokens,
            max_sequence_length=config.max_sequence_length,
            chunk_size=chunk_size,
            max_action_dim=config.expected_max_action_dim,
            env_action_dim=env_action_dim,
            history_dropout=history_dropout,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        MolmoAct2ClampActionProcessorStep(action_mask=action_mask),
        MolmoAct2MaskedUnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=masked_dataset_stats,
        ),
        *([AnchorDecodeStep(encoding=action_encoding)] if use_anchor else []),
        DeviceProcessorStep(device="cpu"),
    ]

    post_to_transition = (
        policy_action_with_anchor_to_transition if use_anchor else policy_action_to_transition
    )

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=post_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

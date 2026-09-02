from __future__ import annotations

import json
import os
import random
import re
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.datasets.embodiment import (
    EMBODIMENT_NAMES,
    canonical_embodiment,
    embodiment_article,
    embodiment_name,
)
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
from lerobot.utils.action_metrics import ACTION_HOLD_KEY
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import require_package

from .action_layout import require_prefix_valid_mask, trim_to_native, valid_dim_mask
from .anchor_encoding import (
    ANCHOR_KEY,
    AnchorDecodeStep,
    AnchorEncodeStep,
    policy_action_with_anchor_to_transition,
)
from .configuration_molmoact2 import MolmoAct2Config, infer_molmoact2_max_sequence_length

_FAST_BIN_PROBE_LIMIT = 2048  # FAST bin values are c - min_token; nothing legitimate exceeds this

ACTION_OUTPUT_TOKEN = "<action_output>"  # nosec B105
ACTION_START_TOKEN = "<action_start>"  # nosec B105
ACTION_END_TOKEN = "<action_end>"  # nosec B105
ACTION_TOKEN_PREFIX = "<action_"  # nosec B105
STATE_START_TOKEN = "<state_start>"  # nosec B105
STATE_END_TOKEN = "<state_end>"  # nosec B105
STATE_TOKEN_PREFIX = "<state_"  # nosec B105
# Placeholder for one past proprio state (04_memory.md §2.4): an unused reserved
# token whose input embedding gets the linearly projected state ADDED on top,
# mirroring the <im_patch> + image-features scatter.
STATE_HISTORY_TOKEN = "<extra_0>"  # nosec B105
# Point-map depth placeholders: one position per pooled depth token. The model replaces
# this reserved token's arbitrary embedding with depth_marker + projected depth features,
# matching <im_patch> + projected RGB features at the VLM-prefix seam.
DEPTH_TOKEN = "<extra_1>"  # nosec B105

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
    embodiment: str | None = None,
    current_subtask: str | None = None,
    metadata: dict[str, Any] | None = None,
    num_history_states: int = 0,
    num_depth_tokens: int = 0,
) -> str:
    """Memory clauses: None/0 disables a clause entirely (byte-identical legacy prompt).

    num_history_states: past proprio states rendered as continuous placeholder
    positions (one STATE_HISTORY_TOKEN per timestep, oldest to newest; the model
    scatters projected states onto them — 04_memory.md §2.4). Image history is
    NOT in the prompt: it enters through the MEM video encoder.

    num_depth_tokens: point-map depth tokens rendered as DEPTH_TOKEN placeholders,
    row-major like the encoder's patch grid. The clause sits immediately after the
    task so the depth span is as close as the prompt layout allows to the image
    patches it describes.

    embodiment: which robot recorded this sample. It leads the prompt because it
    conditions how everything after it is read — the state tokens are a 6-DOF arm's
    or a 7-DOF arm's, and every state vector pads to max_state_dim either way, so
    the clause is the only thing distinguishing them. None omits it (byte-identical
    legacy prompt). Unlike the subtask and metadata clauses it gets NO training
    dropout: those describe things unavailable at inference, while the embodiment is
    always known, so dropping it would only teach the model to ignore it."""
    embodiment_clause = (
        f"The robot is {embodiment_article(embodiment)} {embodiment}. " if embodiment else ""
    )
    state_clause = (
        f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    )
    subtask_clause = f" The current step is {current_subtask}." if current_subtask else ""
    depth_clause = (
        f" The depth of the scene is {DEPTH_TOKEN * num_depth_tokens}." if num_depth_tokens > 0 else ""
    )
    history_clause = (
        f" The recent states of the robot, oldest to newest, are "
        f"{STATE_HISTORY_TOKEN * num_history_states}."
        if num_history_states > 0
        else ""
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
        f"{embodiment_clause}The task is to {task}."
        f"{depth_clause}{subtask_clause}{state_clause}{history_clause}"
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
    embodiment: str | None = None,
) -> str:
    """Generation prompt (two-prompt design): same visual/state context as the action
    prompt, but the question asks for the next step. The assistant's answer is the
    subtask alone; at training time the caller appends it (+eos) and puts CE labels
    on it."""
    embodiment_clause = (
        f"The robot is {embodiment_article(embodiment)} {embodiment}. " if embodiment else ""
    )
    state_clause = (
        f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    )
    prompt = (
        f"{embodiment_clause}The task is to {task}.{state_clause} "
        f"Given these, what step should the robot perform next?"
    )
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(f"Image {idx + 1}<|image|>" for idx in range(num_images))
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


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


def _encodable_bin_map(processor: Any) -> np.ndarray:
    """Nearest encodable bin for every bin value, probed once and cached on the processor.

    The FAST BPE has no UNK token, so a bin whose byte-level form is missing from the
    vocabulary encodes to *nothing*: the coefficient is deleted, every later value slides
    down one slot, and because the layout is frequency-major the decoder then reads each
    joint's coefficients out of its neighbour's cell. One deleted DC value corrupts the
    whole chunk. The shipped MolmoAct2 tokenizer is missing seven bins below 33 (its
    initial_alphabet passed raw codepoints where the model consumes byte-level ones) and
    everything past its fitted range.

    Snapping moves a coefficient by one step at most, i.e. 1 / (scale * sqrt(horizon))
    in normalized action units, well under the quantization error already present.
    """
    cached = getattr(processor, "_encodable_bin_map", None)
    if cached is not None:
        return cached
    bpe = processor.bpe_tokenizer
    bins = np.arange(_FAST_BIN_PROBE_LIMIT)
    encodable = np.array([bpe.decode(bpe(chr(b))["input_ids"]) == chr(b) for b in bins])
    if not encodable.any():
        raise RuntimeError("FAST action tokenizer cannot encode any bin value.")
    good = np.flatnonzero(encodable)
    nearest = good[np.abs(bins[:, None] - good[None, :]).argmin(axis=1)]
    processor._encodable_bin_map = nearest
    return nearest


def _tokenize_discrete_action(action: np.ndarray, processor: Any) -> list[int]:
    """FAST encode: DCT over time, quantize, snap to the alphabet, BPE.

    Mirrors UniversalActionProcessor.__call__ but snaps unencodable bins first — see
    _encodable_bin_map. The coefficient count is asserted because a short span is
    otherwise silent here and only surfaces as a scrambled training target.
    """
    from scipy.fft import dct

    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected one action chunk, got a batch of {arr.shape[0]}.")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Action must be (horizon, dim), got {arr.shape}.")

    horizon, dim = arr.shape
    coefficients = np.around(dct(arr, axis=0, norm="ortho") * processor.scale)
    bins = np.maximum(coefficients.reshape(-1) - processor.min_token, 0).astype(np.int64)
    bins = _encodable_bin_map(processor)[np.clip(bins, 0, _FAST_BIN_PROBE_LIMIT - 1)]
    token_ids = processor.bpe_tokenizer("".join(map(chr, bins)))["input_ids"]

    decoded = len(processor.bpe_tokenizer.decode(token_ids))
    if decoded != horizon * dim:
        raise RuntimeError(
            f"FAST encode produced {decoded} coefficients for a {horizon}x{dim} chunk. "
            "The tokenizer alphabet is dropping values the snap did not cover."
        )
    return [int(token_id) for token_id in token_ids]


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


EMBODIMENT_STATS_FORMAT = "embodiment_stats_v2"
EMBODIMENT_STATS_LAYOUT = "native_source_order_v1"


def load_embodiment_stats(path: str, *, encoding: str, chunk_size: int) -> dict[str, Any]:
    """Read a compute_embodiment_stats.py artifact, checking it describes this pipeline.

    The stats must be for the SAME action encoding and horizon the pipeline runs: anchor
    quantiles say nothing about absolute actions, and a 30-step artifact cannot normalize
    a 50-step chunk. Both mismatches are silent at runtime, so they are errors here.
    """
    payload = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != EMBODIMENT_STATS_FORMAT:
        raise ValueError(
            f"{path} is not a {EMBODIMENT_STATS_FORMAT} artifact "
            f"(got format={payload.get('format') if isinstance(payload, dict) else type(payload).__name__!r}). "
            "Regenerate it with lerobot/src/lerobot/scripts/compute_embodiment_stats.py."
        )
    if str(payload.get("encoding")) != str(encoding):
        raise ValueError(
            f"{path} holds {payload.get('encoding')!r} stats but the policy runs "
            f"action_encoding={encoding!r}."
        )
    if payload.get("layout") != EMBODIMENT_STATS_LAYOUT:
        raise ValueError(
            f"{path} has layout={payload.get('layout')!r}, expected {EMBODIMENT_STATS_LAYOUT!r}. "
            "Regenerate it so state/action statistics are computed in source order."
        )
    if int(payload.get("action_width", 0)) < 1 or int(payload.get("state_width", 0)) < 1:
        raise ValueError(f"{path} must declare positive action_width and state_width values.")
    if int(payload.get("chunk_size", -1)) != int(chunk_size):
        raise ValueError(
            f"{path} holds chunk_size={payload.get('chunk_size')} but the policy runs "
            f"chunk_size={chunk_size}."
        )
    names = list(payload.get("embodiments") or [])
    if not names:
        raise ValueError(f"{path} declares no embodiments.")
    return payload


def _stacked_stats_from_artifact(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Artifact -> the stats mapping the normalizer steps consume.

    Tensors keep their leading embodiment axis; the mixin gathers a row per sample.
    """
    stats: dict[str, dict[str, Any]] = {}
    for key, entries in (payload.get("stats") or {}).items():
        stats[key] = dict(entries)
    return stats


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

        # A per-embodiment artifact ships its own [E, D] mask marking each robot's real
        # dims. Replacing it with a flat row here would un-mask every robot's padding.
        existing = feature_stats.get("mask")
        if torch.is_tensor(existing) and existing.ndim > 1:
            continue
        if isinstance(existing, list) and existing and isinstance(existing[0], (list, tuple)):
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


def _align_mask(mask: Tensor, tensor: Tensor) -> Tensor:
    """Broadcast a [D] or [B, D] mask against `tensor`, aligning on the last axis.

    A per-embodiment mask is per-sample ([B, D]), so its leading axis must stay on the
    batch axis and the horizon axis gets inserted between them — unsqueezing at the
    front like the shared [D] case would silently align the batch against the horizon.
    """
    if mask.ndim == 1:
        while mask.ndim < tensor.ndim:
            mask = mask.unsqueeze(0)
        return mask
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(1)
    return mask


def _dimension_mask(
    complementary: dict[str, Any], key: str, tensor: Tensor, batch_size: int
) -> Tensor:
    raw = complementary.get(key)
    if raw is None:
        return valid_dim_mask(batch_size, int(tensor.shape[-1]), int(tensor.shape[-1]), device=tensor.device)
    mask = torch.as_tensor(raw, dtype=torch.bool, device=tensor.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    if tuple(mask.shape) != (batch_size, int(tensor.shape[-1])):
        raise ValueError(
            f"{key} must have shape {(batch_size, int(tensor.shape[-1]))}, got {tuple(mask.shape)}."
        )
    return mask


def _pad_lowdim(tensor: Tensor, mask: Tensor, width: int | None, key: str) -> tuple[Tensor, Tensor]:
    if width is None or int(tensor.shape[-1]) == width:
        return tensor, mask
    if int(tensor.shape[-1]) > width:
        raise ValueError(f"{key} width {tensor.shape[-1]} exceeds configured canonical width {width}.")
    padded = torch.zeros((*tensor.shape[:-1], width), dtype=tensor.dtype, device=tensor.device)
    padded[..., : tensor.shape[-1]] = tensor
    padded_mask = torch.ones((mask.shape[0], width), dtype=torch.bool, device=mask.device)
    padded_mask[:, : mask.shape[-1]] = mask
    return padded, padded_mask


@ProcessorStepRegistry.register(name="molmoact2_unified_layout")
@dataclass
class MolmoAct2UnifiedLayoutProcessorStep(ProcessorStep):
    """Give state, state history, and actions a shared width, not a shared order.

    Values pass through in source order -- the gripper stays wherever its robot
    records it. What is unified is the width: each vector is right-padded to the
    configured canonical width and the fill is recorded in a prefix-valid
    ``*_dim_is_pad`` mask. State and action masks remain separate because their
    native widths need not be identical.
    """

    state_dim: int | None = None
    action_dim: int | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        reference = action
        if reference is None and isinstance(observation, dict):
            reference = observation.get(OBS_STATE)
        if reference is None:
            return transition
        reference = torch.as_tensor(reference)
        batch_size = int(reference.shape[0])

        state_mask = None
        if isinstance(observation, dict) and OBS_STATE in observation:
            observation = observation.copy()
            state = torch.as_tensor(observation[OBS_STATE])
            state_mask = _dimension_mask(complementary, "state_dim_is_pad", state, batch_size)
            state, state_mask = _pad_lowdim(state, state_mask, self.state_dim, OBS_STATE)
            state_mask = require_prefix_valid_mask(state_mask, "state_dim_is_pad")
            observation[OBS_STATE] = state
            complementary["state_dim_is_pad"] = state_mask
            transition[TransitionKey.OBSERVATION] = observation

        history_key = f"history.{OBS_STATE}"
        if history_key in complementary:
            history = torch.as_tensor(complementary[history_key])
            if state_mask is None:
                state_mask = _dimension_mask(complementary, "state_dim_is_pad", history, batch_size)
                history, state_mask = _pad_lowdim(
                    history, state_mask, self.state_dim, history_key
                )
                state_mask = require_prefix_valid_mask(state_mask, "state_dim_is_pad")
                complementary["state_dim_is_pad"] = state_mask
            elif self.state_dim is not None and int(history.shape[-1]) != self.state_dim:
                history, _ = _pad_lowdim(history, state_mask, self.state_dim, history_key)
            if int(history.shape[-1]) != int(state_mask.shape[-1]):
                raise ValueError(
                    f"{history_key} width {history.shape[-1]} does not match state mask "
                    f"width {state_mask.shape[-1]}."
                )
            complementary[history_key] = history

        if action is not None:
            action = torch.as_tensor(action)
            action_mask = _dimension_mask(complementary, "action_dim_is_pad", action, batch_size)
            action, action_mask = _pad_lowdim(action, action_mask, self.action_dim, ACTION)
            action_mask = require_prefix_valid_mask(action_mask, "action_dim_is_pad")
            transition[TransitionKey.ACTION] = action
            complementary["action_dim_is_pad"] = action_mask

        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="molmoact2_restore_action_layout")
@dataclass
class MolmoAct2RestoreActionLayoutProcessorStep(ProcessorStep):
    """Cut the policy's padded output back to the deployed robot's own width.

    The policy emits values in the robot's source order, so this is a slice and not
    an inverse permutation.
    """

    native_action_dim: int

    def __post_init__(self) -> None:
        if self.native_action_dim < 1:
            raise ValueError(f"native_action_dim must be positive, got {self.native_action_dim}.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = trim_to_native(
                torch.as_tensor(action), native_dim=self.native_action_dim
            )
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


class _MolmoAct2MaskedNormalizationMixin:
    """Masked normalization, optionally with one stats row per embodiment.

    When ``embodiment_names`` is non-empty, every tensor in ``_tensor_stats`` carries a
    leading embodiment axis (``[E, D]`` for state, ``[E, T, D]`` for an action chunk)
    and the row for each sample is gathered from the ``embodiment_index`` the replay
    buffer carries. Keeping the stacked stats inside ``_tensor_stats`` rather than a
    parallel attribute is what makes them survive ``state_dict``/``load_state_dict``,
    so a checkpoint reloads the per-embodiment stats it trained with.

    Empty ``embodiment_names`` is the legacy path: one shared stats row, untouched.
    """

    # Set for the duration of one pipeline call by _with_embodiment_indices; the base
    # class's _apply_transform signature has no room for per-sample context.
    _active_embodiment_indices: Any = None

    @contextmanager
    def _with_embodiment_indices(self, transition: Any):
        """Expose this transition's embodiment_index to _apply_transform."""
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) if transition else None
        indices = complementary.get("embodiment_index") if isinstance(complementary, dict) else None
        previous = self._active_embodiment_indices
        self._active_embodiment_indices = indices
        try:
            yield
        finally:
            self._active_embodiment_indices = previous

    def _embodiment_rows(self, tensor: Tensor) -> Tensor | None:
        """Per-sample stats rows, or None on the legacy shared-stats path."""
        if not getattr(self, "embodiment_names", None):
            return None
        batch_size = int(tensor.shape[0]) if tensor.ndim > 1 else 1
        indices = self._active_embodiment_indices
        if indices is None:
            default = int(getattr(self, "default_embodiment_index", -1))
            if default < 0:
                raise ValueError(
                    "Per-embodiment normalization is active but this batch carries no "
                    "'embodiment_index'. Materialize it (materialize_dataset_labels), pass it "
                    "through the postprocessor payload, or set default_embodiment_index for a "
                    "single-robot deployment."
                )
            rows = torch.full((batch_size,), default, dtype=torch.long)
        else:
            rows = torch.as_tensor(indices).detach().reshape(-1).long()
            if rows.numel() == 1 and batch_size > 1:
                rows = rows.expand(batch_size)
        if int(rows.min()) < 0:
            unknown = sorted({int(v) for v in rows.tolist() if v < 0})
            raise ValueError(
                f"Per-embodiment normalization received unknown embodiment index/indices {unknown}. "
                "Normalizing an unlabelled robot with another robot's stats is exactly the error "
                "this path exists to prevent; label the source or add its robot to "
                "lerobot/datasets/embodiment.py."
            )
        if int(rows.max()) >= len(self.embodiment_names):
            raise ValueError(
                f"embodiment_index {int(rows.max())} is outside this checkpoint's vocabulary of "
                f"{len(self.embodiment_names)} embodiments ({self.embodiment_names}). The stats "
                "artifact and the buffer labels disagree."
            )
        return rows.to(tensor.device)

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: Any, *, inverse: bool = False
    ) -> Tensor:
        stats = getattr(self, "_tensor_stats", {}).get(key, {})
        rows = self._embodiment_rows(tensor) if isinstance(stats, dict) else None

        if rows is not None and stats:
            # Gather this batch's rows onto the tensor's device/dtype BEFORE calling the
            # base transform: it re-runs self.to() whenever the stats device or dtype
            # differs from the input, which would rebuild _tensor_stats from self.stats
            # and undo the swap below.
            gathered = {
                name: value[rows].to(device=tensor.device, dtype=tensor.dtype)
                if isinstance(value, Tensor) and value.shape[0] == len(self.embodiment_names)
                else value
                for name, value in stats.items()
            }
            saved = self._tensor_stats[key]
            self._tensor_stats[key] = gathered
            try:
                transformed = super()._apply_transform(tensor, key, feature_type, inverse=inverse)
            finally:
                self._tensor_stats[key] = saved
            stats = gathered
        else:
            transformed = super()._apply_transform(tensor, key, feature_type, inverse=inverse)

        mask = stats.get("mask") if isinstance(stats, dict) else None
        if mask is None:
            return transformed
        mask = mask.to(device=tensor.device, dtype=torch.bool)
        if tensor.shape[-1] != mask.shape[-1]:
            return transformed
        return torch.where(_align_mask(mask, tensor), transformed, tensor)


@ProcessorStepRegistry.register(name="molmoact2_masked_normalizer")
@dataclass
class MolmoAct2MaskedNormalizerProcessorStep(_MolmoAct2MaskedNormalizationMixin, NormalizerProcessorStep):
    # Non-empty => _tensor_stats carries a leading embodiment axis (see the mixin).
    embodiment_names: list[str] = field(default_factory=list)
    default_embodiment_index: int = -1

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        with self._with_embodiment_indices(transition):
            return self._call_inner(transition)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["embodiment_names"] = list(self.embodiment_names)
        config["default_embodiment_index"] = int(self.default_embodiment_index)
        return config

    def _call_inner(self, transition: EnvTransition) -> EnvTransition:
        """The base normalizer only touches OBSERVATION/ACTION; the short-term-memory
        history window rides in COMPLEMENTARY_DATA (batch_to_transition routes any
        "history.*" key there), so without this it would reach the prompt as raw
        (un-normalized) joint values instead of the [-1, 1] range _build_discrete_state_string
        expects — normalize it here with the exact same OBS_STATE stats/mask."""
        action = transition.get(TransitionKey.ACTION)
        complementary_before = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        hold = None
        if action is not None and ACTION in self.features:
            action_tensor = torch.as_tensor(action)
            if isinstance(complementary_before, dict) and ANCHOR_KEY in complementary_before:
                # A physical hold is zero at every target in anchor-encoded space.
                hold = torch.zeros_like(action_tensor)
            else:
                observation = transition.get(TransitionKey.OBSERVATION)
                if isinstance(observation, dict) and OBS_STATE in observation:
                    state = torch.as_tensor(observation[OBS_STATE]).to(action_tensor)
                    action_dim = int(action_tensor.shape[-1])
                    state = state[..., :action_dim]
                    if action_tensor.ndim == state.ndim + 1:
                        hold = state.unsqueeze(-2).expand_as(action_tensor)
                    elif action_tensor.ndim == state.ndim:
                        hold = state

        transition = super().__call__(transition)
        history_key = f"history.{OBS_STATE}"
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        has_history = (
            isinstance(complementary, dict)
            and history_key in complementary
            and OBS_STATE in self.features
        )
        if has_history or hold is not None:
            transition = transition.copy()
            complementary = dict(complementary or {})
            if has_history:
                tensor = torch.as_tensor(complementary[history_key])
                complementary[history_key] = self._apply_transform(
                    tensor, OBS_STATE, self.features[OBS_STATE].type, inverse=False
                )
            if hold is not None:
                complementary[ACTION_HOLD_KEY] = self._apply_transform(
                    hold, ACTION, self.features[ACTION].type, inverse=False
                )
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition


@ProcessorStepRegistry.register(name="molmoact2_masked_unnormalizer")
@dataclass
class MolmoAct2MaskedUnnormalizerProcessorStep(_MolmoAct2MaskedNormalizationMixin, UnnormalizerProcessorStep):
    """Inverse of the normalizer, and it must gather the SAME per-embodiment row.

    The postprocessor runs on a PolicyAction, so the embodiment index only reaches here
    if the caller puts it in the payload (policy_action_with_anchor_to_transition
    forwards it) or default_embodiment_index is set for a single-robot deployment.
    Unnormalizing with the wrong row returns actions at the wrong scale, silently.
    """

    embodiment_names: list[str] = field(default_factory=list)
    default_embodiment_index: int = -1

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        with self._with_embodiment_indices(transition):
            return super().__call__(transition)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["embodiment_names"] = list(self.embodiment_names)
        config["default_embodiment_index"] = int(self.default_embodiment_index)
        return config


def _masked_clamp(tensor: Tensor, mask: list[bool] | Tensor | None) -> Tensor:
    """Clamp to [-1, 1] only on dimensions where mask is True (normalized dims).
    Dims where mask is False are in raw units (e.g. raw degrees) and must not be clamped.
    A per-embodiment mask arrives already gathered, as [B, D]."""
    t = torch.as_tensor(tensor)
    if mask is None:
        return t.clamp(-1.0, 1.0)
    m = mask if isinstance(mask, Tensor) else torch.tensor(mask, dtype=torch.bool)
    m = m.to(device=t.device, dtype=torch.bool)
    if t.shape[-1] != m.shape[-1]:
        return t
    return torch.where(_align_mask(m, t), t.clamp(-1.0, 1.0), t)


@ProcessorStepRegistry.register(name="molmoact2_clamp_normalized")
@dataclass
class MolmoAct2ClampNormalizedProcessorStep(ProcessorStep):
    """Clamp q01/q99-normalized state and action to the range used by the old trainer.
    action_mask / state_mask mark which dims are actually normalized; unmasked (raw-unit)
    dims are skipped so they are not corrupted by the [-1, 1] clamp."""

    action_mask: list[bool] | None = None
    state_mask: list[bool] | None = None
    # Per-embodiment masks, one row per entry of embodiment_names. Robots differ in DOF,
    # so the padded dims that must escape the [-1, 1] clamp differ per sample too — a
    # single shared mask would clamp a 7-DOF robot's padding as if it were real.
    action_masks: list[list[bool]] | None = None
    state_masks: list[list[bool]] | None = None

    def _rows(self, transition: EnvTransition, tensor: Any) -> Tensor | None:
        if not self.action_masks and not self.state_masks:
            return None
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        indices = complementary.get("embodiment_index") if isinstance(complementary, dict) else None
        if indices is None:
            return None
        rows = torch.as_tensor(indices).detach().cpu().reshape(-1).long()
        batch_size = int(torch.as_tensor(tensor).shape[0])
        if rows.numel() == 1 and batch_size > 1:
            rows = rows.expand(batch_size)
        return rows

    @staticmethod
    def _gather(
        masks: list[list[bool]] | None, rows: Tensor | None, fallback: list[bool] | None
    ) -> list[bool] | Tensor | None:
        """Per-sample mask rows, falling back to the shared mask when unavailable."""
        if masks is None or rows is None:
            return fallback
        table = torch.tensor(masks, dtype=torch.bool)
        return table[rows.clamp(min=0)]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict) and OBS_STATE in observation:
            observation = observation.copy()
            rows = self._rows(transition, observation[OBS_STATE])
            observation[OBS_STATE] = _masked_clamp(
                observation[OBS_STATE], self._gather(self.state_masks, rows, self.state_mask)
            )
            transition[TransitionKey.OBSERVATION] = observation
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            rows = self._rows(transition, action)
            transition[TransitionKey.ACTION] = _masked_clamp(
                action, self._gather(self.action_masks, rows, self.action_mask)
            )
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if isinstance(complementary, dict) and ACTION_HOLD_KEY in complementary:
            hold = complementary[ACTION_HOLD_KEY]
            rows = self._rows(transition, hold)
            complementary = dict(complementary)
            complementary[ACTION_HOLD_KEY] = _masked_clamp(
                hold, self._gather(self.action_masks, rows, self.action_mask)
            )
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
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
    # setup_type remains parse-only compatibility.
    setup_type: str = ""
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    num_state_tokens: int = 256
    max_sequence_length: int | None = None
    chunk_size: int = 30
    max_action_dim: int = 32
    env_action_dim: int | None = None
    # Memory prompt clauses: index → name vocabulary (from subtasks.parquet) and
    # optional per-component training dropout (applied only when actions are
    # present, i.e. training text — inference prompts are deterministic). Zero is
    # the default so training matches deployment unless the launch explicitly asks
    # for clause deletion.
    subtask_names: list[str] = field(default_factory=list)
    subtask_dropout: float = 0.0
    metadata_dropout: float = 0.0
    # Embodiment clause vocabulary: index → prompt name, pinned into the saved config
    # so a checkpoint renders through the vocabulary it trained with even after
    # lerobot/datasets/embodiment.py grows new robots. No dropout knob on purpose —
    # the embodiment is known at inference, so hiding it only teaches the model to
    # ignore it (see _build_robot_text).
    embodiment_names: list[str] = field(default_factory=lambda: list(EMBODIMENT_NAMES))
    # Short-term memory (04_memory.md §2.4): state history becomes continuous
    # placeholder tokens, image history rides to the MEM video encoder as tensors
    # (complementary keys "history.{OBS_STATE}" / "history.{OBS_IMAGES}.{cam}").
    # Absent keys = clause/tensors off (byte-identical legacy prompt); one dropout
    # flip removes the WHOLE block (states + frames) for training text only.
    # history_stride_seconds parameterizes the e(t) time stamps.
    history_dropout: float = 0.0
    history_stride_seconds: float = 1.0
    # Anti-laziness RGB dropout (depth_redesign_options.md §4.3): mask ONE camera's
    # <im_patch> span out of the attention mask (the depth camera's), so the sample
    # is solvable only through depth — nothing attends the span, no gradient flows
    # through its vision path. Training text only, independent draw per sample.
    # rgb_dropout_key is the bare camera name; empty or rgb_dropout 0 disables.
    rgb_dropout: float = 0.0
    rgb_dropout_key: str = ""
    # Point-map depth token count (patch grid of the depth image). 0 = depth-free:
    # no DEPTH_TOKEN placeholders, byte-identical legacy prompt. The count is fixed
    # per config — the encoder emits N tokens whether or not depth is present or
    # dropped out (it swaps in its learned null bank), so the prompt never varies.
    num_depth_tokens: int = 0
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
        self._state_history_id = _single_token_id(self.processor.tokenizer, STATE_HISTORY_TOKEN)
        self._depth_token_id = _single_token_id(self.processor.tokenizer, DEPTH_TOKEN)
        self._image_patch_id = _single_token_id(self.processor.tokenizer, "<im_patch>")
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
            "embodiment_names": list(self.embodiment_names),
            "subtask_dropout": self.subtask_dropout,
            "metadata_dropout": self.metadata_dropout,
            "history_dropout": self.history_dropout,
            "history_stride_seconds": self.history_stride_seconds,
            "rgb_dropout": self.rgb_dropout,
            "rgb_dropout_key": self.rgb_dropout_key,
            "num_depth_tokens": self.num_depth_tokens,
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
            num_depth_tokens=self.num_depth_tokens,
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

    def _pad_action(
        self,
        action: Tensor,
        action_is_pad: Any | None,
        action_dim_is_pad: Any | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        padded_dim_is_pad = torch.ones(
            (action.shape[0], self.max_action_dim), device=action.device, dtype=torch.bool
        )
        if action_dim_is_pad is None:
            padded_dim_is_pad[:, : action.shape[-1]] = False
        else:
            native_mask = torch.as_tensor(action_dim_is_pad, device=action.device, dtype=torch.bool)
            if native_mask.ndim == 1:
                native_mask = native_mask.unsqueeze(0)
            if native_mask.shape[0] == 1 and action.shape[0] > 1:
                native_mask = native_mask.expand(action.shape[0], -1)
            if tuple(native_mask.shape) != (int(action.shape[0]), int(action.shape[-1])):
                raise ValueError(
                    "action_dim_is_pad must match the unpadded action dimensions: "
                    f"got {tuple(native_mask.shape)} for action {tuple(action.shape)}."
                )
            padded_dim_is_pad[:, : action.shape[-1]] = native_mask
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
        return padded, action_horizon_is_pad, padded_dim_is_pad

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
        embodiment_texts = self._extract_embodiment_texts(complementary, batch_size)
        history_stack = self._extract_history_image_stack(
            complementary, self._resolve_image_keys(observation), batch_size
        )

        state_np = state.detach().cpu().numpy()
        prompts: list[str] = []
        fulls: list[str] = []
        flat_images: list[np.ndarray] = []
        history_on = torch.ones(batch_size, dtype=torch.bool)
        for batch_idx in range(batch_size):
            images = images_by_example[batch_idx]
            flat_images.extend(images)
            name = subtask_texts[batch_idx]
            if name is not None and random.random() < self.history_dropout:
                history_on[batch_idx] = False  # training text only; the encoder sees no history
            prompt = _build_subtask_generation_text(
                task=tasks[batch_idx],
                discrete_state_string=_build_discrete_state_string(
                    state_np[batch_idx], self.num_state_tokens
                ),
                num_images=len(images),
                embodiment=embodiment_texts[batch_idx],
            )
            prompts.append(prompt)
            fulls.append(f"{prompt}{name}{self._eos_token}" if name else prompt)

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
            labels = torch.full_like(inputs["input_ids"], -100)
            for batch_idx in range(batch_size):
                answer_len = int(full_lengths[batch_idx] - prompt_lengths[batch_idx])
                if not valid[batch_idx] or answer_len <= 0:
                    continue
                nonpad = inputs["attention_mask"][batch_idx].nonzero().reshape(-1)
                answer_span = nonpad[-answer_len:]
                labels[batch_idx, answer_span] = inputs["input_ids"][batch_idx, answer_span]
            inputs["labels"] = labels
        complementary.update(dict(inputs))
        if history_stack is not None:
            frames, times = history_stack
            complementary["history_images"] = frames
            complementary["history_image_times"] = times
            complementary["history_images_mask"] = history_on
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

    def _extract_embodiment_texts(self, complementary: dict, batch_size: int) -> list[str | None]:
        """Embodiment name per sample: an explicit "embodiment" string (rollout path,
        where cfg.env.robot.type is the only source), else embodiment_index rendered
        through the vocabulary (offline batches). Unknown (-1) omits the clause."""
        texts = complementary.get("embodiment")
        if texts is not None:
            return [
                canonical_embodiment(text) for text in _as_text_list(texts, batch_size)
            ]
        indices = complementary.get("embodiment_index")
        if indices is None:
            return [None] * batch_size
        flat = torch.as_tensor(indices).detach().cpu().reshape(-1).long().tolist()
        if len(flat) == 1:
            flat = flat * batch_size
        return [embodiment_name(index, self.embodiment_names) for index in flat]

    def _extract_history_states(self, complementary: dict, batch_size: int) -> Tensor | None:
        """Normalized past states for the short-term history window, read from
        complementary key "history.{OBS_STATE}" (the ReplayBuffer.sample()/
        assemble_history_windows lookback, already normalized like the current
        state upstream). Returns (B, T_h, D) float32 oldest → newest, or None when
        history wasn't gathered. Consumed as continuous state tokens (§2.4): one
        STATE_HISTORY_TOKEN position per timestep, projected by the model."""
        history = complementary.get(f"history.{OBS_STATE}")
        if history is None:
            return None
        arr = torch.as_tensor(history, dtype=torch.float32).detach().cpu()
        if arr.ndim == 2:
            arr = arr[None]
        if int(arr.shape[0]) != batch_size:
            raise ValueError(
                f"history.{OBS_STATE} batch size {int(arr.shape[0])} does not match {batch_size}."
            )
        return arr

    def _extract_history_image_stack(
        self, complementary: dict, image_keys: list[str], batch_size: int
    ) -> tuple[Tensor, Tensor] | None:
        """Past camera frames for the MEM video encoder, read from complementary keys
        "history.{OBS_IMAGES}.{cam}" (B, T_h, C, H, W; uint8 cache rows or policy-format
        floats — _normalize_image reconciles both). Frames go through the SAME
        image-processor transform as the prompt images (crop_mode "resize": one crop,
        729 patches) and ride to the model as a separate tensor, never as prompt images.

        Returns (frames, times): frames (B, cams, T_h, num_patches, patch_dim) bfloat16
        (the ViT consumes bf16; halves transport vs float32) with cams ordered like the
        prompt images, times (T_h,) seconds-before-now oldest → newest (for the
        sinusoidal e(t)); or None when no image history."""
        prefix = f"history.{OBS_IMAGES}."
        keys = [k for k in complementary if k.startswith(prefix) and not k.endswith("_is_pad")]
        if not keys:
            return None
        expected = [key for key in image_keys if f"history.{key}" in keys]
        if len(expected) != len(image_keys) or len(keys) != len(image_keys):
            raise ValueError(
                f"MEM video encoder needs image history for exactly the prompt cameras "
                f"{image_keys}, got history keys {sorted(keys)}."
            )
        flat_frames: list[np.ndarray] = []
        num_slots = None
        for batch_idx in range(batch_size):
            for key in image_keys:
                frames = complementary[f"history.{key}"]
                item = frames[batch_idx] if getattr(frames, "ndim", 0) >= 5 else frames
                if num_slots is None:
                    num_slots = len(item)
                flat_frames.extend(_normalize_image(f) for f in item)
        pixel_values = self.processor.image_processor(images=flat_frames)["pixel_values"]
        pixel_values = torch.as_tensor(np.asarray(pixel_values)).to(torch.bfloat16)
        if int(pixel_values.shape[0]) != batch_size * len(image_keys) * num_slots:
            raise ValueError(
                "MEM video encoder requires crop_mode 'resize' (one crop per frame); "
                f"got {int(pixel_values.shape[0])} crops for "
                f"{batch_size * len(image_keys) * num_slots} history frames."
            )
        frames = pixel_values.view(batch_size, len(image_keys), num_slots, *pixel_values.shape[1:])
        times = torch.tensor(
            [self.history_stride_seconds * (num_slots - j) for j in range(num_slots)],
            dtype=torch.float32,
        )
        return frames, times

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
            action_padded, action_horizon_is_pad, action_dim_is_pad = self._pad_action(
                action, action_is_pad, complementary.get("action_dim_is_pad")
            )
            if ACTION_HOLD_KEY in complementary:
                hold = torch.as_tensor(complementary[ACTION_HOLD_KEY], dtype=torch.float32)
                if tuple(hold.shape) != tuple(action.shape):
                    raise ValueError(
                        f"{ACTION_HOLD_KEY} must match action before padding: "
                        f"got {tuple(hold.shape)} and {tuple(action.shape)}."
                    )
                complementary[ACTION_HOLD_KEY] = self._pad_action(
                    hold, action_is_pad, complementary.get("action_dim_is_pad")
                )[0]
            real_action_dim = int((~action_dim_is_pad).sum(dim=-1).max())
        elif real_action_dim > 0:
            action_dim_is_pad[:, :real_action_dim] = False

        subtask_texts = self._extract_subtask_texts(complementary, batch_size)
        embodiment_texts = self._extract_embodiment_texts(complementary, batch_size)
        metadata_list = self._extract_metadata(complementary, batch_size)
        history_states = self._extract_history_states(complementary, batch_size)
        image_keys = self._resolve_image_keys(observation)
        history_stack = self._extract_history_image_stack(complementary, image_keys, batch_size)

        prompt_texts: list[str] = []
        full_texts: list[str] = []
        flat_images: list[np.ndarray] = []
        state_np = state.detach().cpu().numpy()
        build_action_labels = action is not None and self.action_mode in {"discrete", "both"}
        max_num_images = 0
        history_on = torch.ones(batch_size, dtype=torch.bool)
        rgb_drop_cam = None
        rgb_dropped = torch.zeros(batch_size, dtype=torch.bool)
        if build_action_labels and self.rgb_dropout > 0 and self.rgb_dropout_key:
            drop_key = f"{OBS_IMAGES}.{self.rgb_dropout_key}"
            resolved_keys = self._resolve_image_keys(observation)
            if drop_key in resolved_keys:
                rgb_drop_cam = resolved_keys.index(drop_key)
        for batch_idx in range(batch_size):
            images = images_by_example[batch_idx]
            discrete_state = _build_discrete_state_string(state_np[batch_idx], self.num_state_tokens)
            current_subtask = subtask_texts[batch_idx]
            metadata = metadata_list[batch_idx]
            if build_action_labels:  # training text: per-component dropout (π0.7 recipe)
                if random.random() < self.subtask_dropout:
                    current_subtask = None
                if random.random() < self.metadata_dropout:
                    metadata = None
                if random.random() < self.history_dropout:
                    # One flip drops the whole short-term block: state placeholders
                    # AND the sample's history keys in the video encoder, whose
                    # temporal step is then left with the query's own timestep only.
                    # No history is visible; this is NOT bit-identical to the
                    # pretrained single-frame op (that invariant was dropped with the
                    # separable rebuild, 2026-08-03).
                    history_on[batch_idx] = False
                if rgb_drop_cam is not None and random.random() < self.rgb_dropout:
                    # Anti-laziness RGB dropout: the camera's <im_patch> span is
                    # masked out of attention below — token layout unchanged, no
                    # consumer attends it, no gradient through its vision path.
                    rgb_dropped[batch_idx] = True
            flat_images.extend(images)
            max_num_images = max(max_num_images, len(images))
            prompt = _build_robot_text(
                task=tasks[batch_idx],
                discrete_state_string=discrete_state,
                num_images=len(images),
                embodiment=embodiment_texts[batch_idx],
                num_history_states=(
                    int(history_states.shape[1])
                    if history_states is not None and history_on[batch_idx]
                    else 0
                ),
                num_depth_tokens=self.num_depth_tokens,
                current_subtask=current_subtask,
                metadata=metadata,
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
            history_num_samples=int(history_states.shape[1]) if history_states is not None else 0,
        )
        if int(inputs["input_ids"].shape[1]) > max_sequence_length:
            raise ValueError(
                f"MolmoAct2 sequence length {int(inputs['input_ids'].shape[1])} exceeds "
                f"max_sequence_length={max_sequence_length}."
            )

        if build_action_labels:
            inputs["labels"] = self._build_labels(inputs["input_ids"], inputs["attention_mask"])

        if rgb_drop_cam is not None and bool(rgb_dropped.any()):
            # RGB dropout = attention masking (depth_redesign_options.md §4.3): the
            # dropped camera's <im_patch> span leaves the attention mask, so no
            # consumer attends those tokens and no gradient reaches the vision tower
            # through them. Positions/RoPE are unaffected (position_ids are arange,
            # not mask-cumsum). The depth stream's wrist bridge bypasses this mask and
            # is killed model-side by deriving its per-row switch from this same mask.
            # Applied AFTER _build_labels (answer-span math reads the mask).
            from lerobot.policies.depth_pointmap.modeling_stream import mask_camera_patch_span

            rows = rgb_dropped.nonzero(as_tuple=True)[0]
            inputs["attention_mask"] = mask_camera_patch_span(
                inputs["attention_mask"],
                inputs["input_ids"],
                image_patch_id=self._image_patch_id,
                num_images=max_num_images,
                cam_index=rgb_drop_cam,
                rows=rows,
            )

        complementary.update(dict(inputs))
        complementary["action_dim_is_pad"] = action_dim_is_pad
        if history_states is not None:
            complementary["history_state_values"] = history_states
            complementary["state_history_token_id"] = torch.tensor(self._state_history_id)
        if self.num_depth_tokens > 0:
            complementary["depth_token_id"] = torch.tensor(self._depth_token_id)
        if history_stack is not None:
            frames, times = history_stack
            complementary["history_images"] = frames
            complementary["history_image_times"] = times
            complementary["history_images_mask"] = history_on
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
    canonical_action_dim = env_action_dim
    canonical_state_dim = None
    if config.input_features and OBS_STATE in config.input_features:
        canonical_state_dim = int(config.input_features[OBS_STATE].shape[0])

    hf_metadata: dict[str, Any] = {}
    if dataset_stats is None and str(config.norm_tag or "").strip():
        dataset_stats, hf_metadata = _load_hf_norm_stats_for_tag(
            config.base_path,
            revision=config.base_revision,
            force_download=bool(config.base_force_download),
            norm_tag=config.norm_tag,
        )

    if action_stats_override is not None:
        dataset_stats = {} if dataset_stats is None else deepcopy(dataset_stats)
        dataset_stats[ACTION] = action_stats_override

    image_keys = list(config.image_keys)
    if not image_keys and isinstance(hf_metadata.get("camera_keys"), list):
        image_keys = [str(key) for key in hf_metadata["camera_keys"]]
    chunk_size = int(hf_metadata.get("action_horizon") or config.chunk_size)

    action_encoding = getattr(config, "action_encoding", "absolute")
    use_anchor = action_encoding in ("anchor", "delta")

    # Per-embodiment stats replace the single pooled row entirely: pooling a Franka's
    # joint ranges with an ARX5's makes q01/q99 span the union, so each robot's own
    # motion is squashed into a sliver of [-1, 1] before the state discretization.
    embodiment_names: list[str] = []
    embodiment_masks: dict[str, list[list[bool]]] = {}
    embodiment_stats_path = getattr(config, "embodiment_stats_path", None)
    if embodiment_stats_path:
        artifact = load_embodiment_stats(
            embodiment_stats_path,
            encoding=action_encoding,
            chunk_size=int(hf_metadata.get("action_horizon") or config.chunk_size),
        )
        embodiment_names = list(artifact["embodiments"])
        canonical_action_dim = int(artifact["action_width"])
        canonical_state_dim = int(artifact["state_width"])
        dataset_stats = _stacked_stats_from_artifact(artifact)
        for key, entries in artifact["stats"].items():
            embodiment_masks[key] = [[bool(v) for v in row] for row in entries["mask"].tolist()]
        if action_stats_override is not None:
            raise ValueError(
                "embodiment_stats_path and action_encoding_stats_path both set. The "
                "per-embodiment artifact already carries the encoded action stats; the "
                "single-row override would silently replace every embodiment's row with one."
            )

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
        flat = m.tolist() if hasattr(m, "tolist") else m
        if flat and isinstance(flat[0], list):  # per-embodiment table, not a shared row
            return None
        return [bool(v) for v in flat]

    action_mask = _mask_list(ACTION)
    state_mask = _mask_list(OBS_STATE)
    default_embodiment_index = -1
    if embodiment_names:
        configured = canonical_embodiment(getattr(config, "embodiment", None))
        if configured is not None and configured in embodiment_names:
            default_embodiment_index = embodiment_names.index(configured)

    # MemoryConfig lives on the RL wrapper config (MolmoAct2RLConfig.memory), not on the
    # bare MolmoAct2Config used for BC/eval, hence the getattr default.
    memory_cfg = getattr(config, "memory", None)
    history_dropout = memory_cfg.history_dropout if memory_cfg is not None else 0.0
    # Anti-laziness RGB dropout rides pointmap_config (depth_redesign_options.md §4.3):
    # the blacked camera is the depth camera; depth-free configs get 0 (no-op).
    pointmap_cfg = getattr(config, "pointmap_config", None)
    rgb_dropout = pointmap_cfg.rgb_dropout_prob if pointmap_cfg is not None else 0.0
    rgb_dropout_key = pointmap_cfg.depth_key if pointmap_cfg is not None else ""
    # DEPTH_TOKEN placeholder count = the 2D attention-pooler's output grid.
    num_depth_tokens = pointmap_cfg.num_pooled_tokens if pointmap_cfg is not None else 0

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        MolmoAct2UnifiedLayoutProcessorStep(
            state_dim=canonical_state_dim,
            action_dim=canonical_action_dim,
        ),
        *([AnchorEncodeStep(encoding=action_encoding)] if use_anchor else []),
        MolmoAct2MaskedNormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=masked_dataset_stats,
            embodiment_names=embodiment_names,
            default_embodiment_index=default_embodiment_index,
        ),
        MolmoAct2ClampNormalizedProcessorStep(
            action_mask=action_mask,
            state_mask=state_mask,
            action_masks=embodiment_masks.get(ACTION),
            state_masks=embodiment_masks.get(OBS_STATE),
        ),
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
            subtask_dropout=config.subtask_dropout,
            metadata_dropout=config.metadata_dropout,
            history_dropout=history_dropout,
            history_stride_seconds=config.history_stride_seconds,
            rgb_dropout=rgb_dropout,
            rgb_dropout_key=rgb_dropout_key,
            num_depth_tokens=num_depth_tokens,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        MolmoAct2ClampActionProcessorStep(action_mask=action_mask),
        MolmoAct2MaskedUnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=masked_dataset_stats,
            embodiment_names=embodiment_names,
            default_embodiment_index=default_embodiment_index,
        ),
        *([AnchorDecodeStep(encoding=action_encoding)] if use_anchor else []),
        *(
            [
                MolmoAct2RestoreActionLayoutProcessorStep(native_action_dim=env_action_dim)
            ]
            if env_action_dim is not None
            else []
        ),
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

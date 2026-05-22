"""SO-101 v3.0 ↔ v2.1 joint-frame transform for MolmoAct2.

MolmoAct2 was pretrained on SO-100/101 data in the pre-PR-777 calibration
("v2.1 frame"). LeRobot now records and commands the arm in the post-PR-777
calibration ("v3.0 frame"). Two joints differ:

    shoulder_lift (idx 1): sign flip + 90° offset
    elbow_flex    (idx 2): +90° offset

Official conversion: docs/source/backwardcomp.mdx.

This module is the single boundary that crosses the two frames. The processor
pipeline inserts ``SO101V3ToV21Step`` before the v2.1 normalizer (training and
inference) and ``SO101V21ToV3Step`` after the v2.1 unnormalizer (inference).
The model and its losses keep operating in v2.1 frame; norm stats stay v2.1.

Delete this file (and the three insertion points in processor_molmoact2.py)
when MolmoAct2 moves to anchor delta actions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# v2.1 = SIGNS * v3.0 + OFFSETS  (per docs/source/backwardcomp.mdx)
SIGNS = torch.tensor([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
OFFSETS = torch.tensor([0.0, 90.0, 90.0, 0.0, 0.0, 0.0])


def arm_to_model(x: torch.Tensor) -> torch.Tensor:
    """v3.0 arm frame → v2.1 model frame. Broadcasts over leading dims; last dim is 6."""
    return SIGNS.to(x) * x + OFFSETS.to(x)


def model_to_arm(x: torch.Tensor) -> torch.Tensor:
    """v2.1 model frame → v3.0 arm frame. Inverse of ``arm_to_model``."""
    return (x - OFFSETS.to(x)) * SIGNS.to(x)


# Stats keys that hold per-joint location (apply affine directly).
_LOCATION_KEYS = ("mean", "q50")
# Stats keys that hold per-joint bounds in (low, high) pairs (apply affine, then
# elementwise min/max so a sign flip swaps the endpoints).
_BOUND_PAIRS = (("min", "max"), ("q01", "q99"), ("q10", "q90"))
# Stats keys whose value is invariant under v3 ↔ v2.1 (|SIGNS|=1, additive
# offset doesn't change scale; count is a scalar).
_INVARIANT_KEYS = ("std", "count")


def _affine_per_joint(values: torch.Tensor) -> torch.Tensor:
    """Apply v2.1 = SIGNS * v3 + OFFSETS along the last dim. Last dim must be 6."""
    return SIGNS.to(values) * values + OFFSETS.to(values)


def _convert_feature_stats(feature_stats: dict[str, Any]) -> dict[str, Any]:
    """v3.0 → v2.1 for a single 6-dim feature's stats dict (action or state)."""
    out = deepcopy(feature_stats)

    def _as_tensor(key: str) -> torch.Tensor | None:
        if key not in out:
            return None
        return torch.as_tensor(out[key], dtype=torch.float32)

    for key in _LOCATION_KEYS:
        tensor = _as_tensor(key)
        if tensor is not None and tensor.shape[-1] == SIGNS.shape[0]:
            out[key] = _affine_per_joint(tensor).tolist()

    for low_key, high_key in _BOUND_PAIRS:
        low = _as_tensor(low_key)
        high = _as_tensor(high_key)
        if low is None or high is None:
            continue
        if low.shape[-1] != SIGNS.shape[0] or high.shape[-1] != SIGNS.shape[0]:
            continue
        low_v21 = _affine_per_joint(low)
        high_v21 = _affine_per_joint(high)
        out[low_key] = torch.minimum(low_v21, high_v21).tolist()
        out[high_key] = torch.maximum(low_v21, high_v21).tolist()

    # _INVARIANT_KEYS need no change; left as-is on `out`.
    return out


def stats_v3_to_v21(
    stats: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Convert action and observation.state stats from v3.0 to v2.1.

    The pipeline already remaps live tensors via ``SO101V3ToV21Step`` before the
    normalizer. Dataset-derived stats, however, are computed on v3.0 raw data
    and would otherwise be mis-aligned with the v2.1 tensors the normalizer
    sees. Apply this once before passing dataset stats into the processor.

    Image stats and any non-joint features pass through unchanged.
    """
    if not stats:
        return stats
    converted = dict(stats)
    for key in (ACTION, OBS_STATE):
        feature_stats = converted.get(key)
        if isinstance(feature_stats, dict):
            converted[key] = _convert_feature_stats(feature_stats)
    return converted


@ProcessorStepRegistry.register(name="so101_v3_to_v21")
@dataclass
class SO101V3ToV21Step(ProcessorStep):
    """Convert observation.state and (if present) action from v3.0 → v2.1.

    Insert before the v2.1 normalizer in the input pipeline.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict) and OBS_STATE in observation:
            observation = observation.copy()
            observation[OBS_STATE] = arm_to_model(torch.as_tensor(observation[OBS_STATE]))
            transition[TransitionKey.OBSERVATION] = observation
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = arm_to_model(torch.as_tensor(action))
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="so101_v21_to_v3")
@dataclass
class SO101V21ToV3Step(ProcessorStep):
    """Convert model action output from v2.1 → v3.0.

    Insert after the v2.1 unnormalizer in the output pipeline.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = model_to_arm(torch.as_tensor(action))
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

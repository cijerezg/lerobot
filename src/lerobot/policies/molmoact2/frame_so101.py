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

from dataclasses import dataclass

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

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

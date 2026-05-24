"""Anchor/delta action encoding for MolmoAct2.

These steps live INSIDE the molmoact2 pipeline so that encoding (a - s) and
decoding (Delta + s, cumsum(Delta) + s) happen in v2.1 frame, after
SO101V3ToV21Step and before SO101V21ToV3Step. The frame steps therefore see
absolute actions only; encoded deltas are frame-coherent with the v2.1
normalizer stats produced by compute_delta_stats.py --frame-conversion
so101_v3_to_v21.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.converters import create_transition
from lerobot.types import EnvTransition, PolicyAction, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

ENCODINGS = ("anchor", "delta")

ANCHOR_KEY = "anchor_state_v21"


def _encode(action: torch.Tensor, anchor: torch.Tensor, encoding: str) -> torch.Tensor:
    if encoding == "anchor":
        return action - anchor[..., None, :]
    d0 = action[..., 0, :] - anchor
    if action.shape[-2] > 1:
        d_rest = torch.diff(action, dim=-2)
        return torch.cat([d0[..., None, :], d_rest], dim=-2)
    return d0[..., None, :]


def _decode(action: torch.Tensor, anchor: torch.Tensor, encoding: str) -> torch.Tensor:
    if encoding == "anchor":
        return action + anchor[..., None, :]
    return torch.cumsum(action, dim=-2) + anchor[..., None, :]


@ProcessorStepRegistry.register(name="anchor_encode")
@dataclass
class AnchorEncodeStep(ProcessorStep):
    """Replace ACTION with the encoded target and stash the anchor.

    Insert AFTER SO101V3ToV21Step, BEFORE the normalizer.
    """

    encoding: str

    def __post_init__(self) -> None:
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding={self.encoding!r} not in {ENCODINGS}")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict) or OBS_STATE not in observation:
            return transition
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        action = torch.as_tensor(action)
        anchor = torch.as_tensor(observation[OBS_STATE])[..., : action.shape[-1]]
        encoded = _encode(action, anchor.to(action), self.encoding)

        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp[ANCHOR_KEY] = anchor

        new = transition.copy()
        new[TransitionKey.ACTION] = encoded
        new[TransitionKey.COMPLEMENTARY_DATA] = comp

        return new

    def get_config(self) -> dict[str, Any]:
        return {"encoding": self.encoding}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="anchor_decode")
@dataclass
class AnchorDecodeStep(ProcessorStep):
    """Reconstruct absolute v2.1 action from the decoded delta.

    Insert AFTER the unnormalizer, BEFORE SO101V21ToV3Step. Requires the
    caller to populate ``complementary_data[ANCHOR_KEY]`` (v2.1 frame).
    """

    encoding: str

    def __post_init__(self) -> None:
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding={self.encoding!r} not in {ENCODINGS}")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        anchor = comp.get(ANCHOR_KEY)
        if anchor is None:
            raise RuntimeError(
                f"AnchorDecodeStep requires complementary_data[{ANCHOR_KEY!r}]"
            )

        action = torch.as_tensor(action)
        anchor = torch.as_tensor(anchor).to(action)
        decoded = _decode(action, anchor, self.encoding)

        new = transition.copy()
        new[TransitionKey.ACTION] = decoded
        return new

    def get_config(self) -> dict[str, Any]:
        return {"encoding": self.encoding}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def policy_action_with_anchor_to_transition(payload: Any) -> EnvTransition:
    """Postprocessor ``to_transition`` adapter for the anchor/delta path.

    Accepts either a bare PolicyAction tensor (no decode; back-compat) or a
    dict ``{"action": tensor, ANCHOR_KEY: tensor}`` carrying the anchor so
    AnchorDecodeStep can reconstruct the absolute action.
    """
    if isinstance(payload, PolicyAction):
        return create_transition(action=payload)
    if isinstance(payload, dict):
        action = payload.get(ACTION)
        if action is None:
            action = payload.get("action")
        if not isinstance(action, PolicyAction):
            raise ValueError(
                f"postprocessor payload dict missing PolicyAction under {ACTION!r}/'action'"
            )
        anchor = payload.get(ANCHOR_KEY)
        comp = {ANCHOR_KEY: anchor} if anchor is not None else None
        return create_transition(action=action, complementary_data=comp)
    raise ValueError(
        f"postprocessor payload must be PolicyAction or dict, got {type(payload).__name__}"
    )

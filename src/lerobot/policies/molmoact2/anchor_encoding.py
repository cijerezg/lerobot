"""Anchor/delta action encoding for MolmoAct2.

These steps live INSIDE the molmoact2 pipeline: encoding (a - s) right after
batching, decoding (Delta + s, cumsum(Delta) + s) right after the unnormalizer.
Everything operates in the raw arm frame; encoded deltas are coherent with the
normalizer stats produced by compute_delta_stats.py on the same dataset.
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

ANCHOR_KEY = "anchor_state"
# Rides the postprocessor payload beside the anchor so the unnormalizer can gather the
# same per-embodiment stats row the normalizer used. Without it a mixed-robot policy
# unnormalizes every action with whichever row happens to be the default.
EMBODIMENT_INDEX_KEY = "embodiment_index"


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

    Insert BEFORE the normalizer.
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

        # Anchor/delta encoding subtracts state from action elementwise, which is only
        # meaningful when both live in the same space. Every packed source is joint-space
        # (see action_layout), so there is no mode to check -- a future Cartesian corpus
        # would have to re-establish that invariant before reaching this step.
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        action = torch.as_tensor(action)
        anchor = torch.as_tensor(observation[OBS_STATE])[..., : action.shape[-1]]
        encoded = _encode(action, anchor.to(action), self.encoding)

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
    """Reconstruct the absolute action from the decoded delta.

    Insert AFTER the unnormalizer. Requires the caller to populate
    ``complementary_data[ANCHOR_KEY]``.
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
        comp = {}
        anchor = payload.get(ANCHOR_KEY)
        if anchor is not None:
            comp[ANCHOR_KEY] = anchor
        embodiment_index = payload.get(EMBODIMENT_INDEX_KEY)
        if embodiment_index is not None:
            comp[EMBODIMENT_INDEX_KEY] = embodiment_index
        # Per-row normalization selects its stats row with a CONFIGURABLE column
        # (stats_index_key, "action_layout_id" for a mixture keyed by action
        # convention), so forwarding only the two names known here strands the
        # unnormalizer with no row. Carry whatever the caller passed.
        nested = payload.get(TransitionKey.COMPLEMENTARY_DATA)
        if isinstance(nested, dict):
            comp.update(nested)
        return create_transition(action=action, complementary_data=comp or None)
    raise ValueError(
        f"postprocessor payload must be PolicyAction or dict, got {type(payload).__name__}"
    )

"""Hand features must not change the native replay-cache contract."""

from types import SimpleNamespace

import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.depth_pointmap.configuration_pointmap import DepthPointmapConfig
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.rl.data_sources.diverse_actor_buffer import DiverseSampleSpec
from lerobot.rl.data_sources.diverse_actor_cache import cache_fingerprint
from lerobot.rl.data_sources.diverse_integration import sample_spec_from_config


def policy_config(hand_block, depth):
    return MolmoAct2Config(
        device="cpu",
        hand_block=hand_block,
        pointmap_config=DepthPointmapConfig(depth_key="wrist_0") if depth else None,
        input_features={
            **{
                f"observation.images.{role}": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 480, 640)
                )
                for role in ("external_0", "external_1", "wrist_0")
            },
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
    )


@pytest.mark.parametrize("depth", [False, True])
def test_hand_block_reuses_native_cache_contract(tmp_path, depth):
    joint = policy_config(False, depth)
    hand = policy_config(True, depth)
    assert hand.input_features["observation.state"].shape == (18,)
    assert hand.output_features["action"].shape == (15,)

    joint_spec = sample_spec_from_config(SimpleNamespace(policy=joint))
    hand_spec = sample_spec_from_config(SimpleNamespace(policy=hand))
    assert hand_spec.max_width == 8
    assert hand_spec.fingerprint() == joint_spec.fingerprint()
    assert hand_spec.fingerprint() == DiverseSampleSpec(load_depth=depth).fingerprint()

    for part in ("corpus", "fmb"):
        (tmp_path / part).mkdir()
        for name in ("episodes.jsonl", "actor_anchors_5hz.jsonl"):
            (tmp_path / part / name).write_text("{}\n")
    assert cache_fingerprint(tmp_path, hand_spec, anchors=2, episodes=2) == cache_fingerprint(
        tmp_path, joint_spec, anchors=2, episodes=2
    )

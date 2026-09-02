"""Camera roles, absent views and RGB temporal order (integration plan phase D).

Gate D: a batch mixing two- and three-camera samples must preprocess; every present
camera must carry one current frame and three correctly timestamped history frames;
and an absent role must contribute zero attended tokens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="molmoact2 processor imports policy deps")

from lerobot.datasets.diverse_actor_selection import (  # noqa: E402
    PACKED_CURRENT_SLOT,
    open_federated_corpus,
    packed_history_slots,
    select_actor_anchors,
)
from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    MolmoAct2PackInputsProcessorStep,
)
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.rl.data_sources.diverse_actor_buffer import (  # noqa: E402
    DiverseActorBuffer,
    DiverseSampleSpec,
)
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[3] / "outputs/diverse_robot_dataset"
BASE_PATH = Path(__file__).resolve().parents[3] / "outputs/MolmoAct2"
ROLE_KEYS = [f"{OBS_IMAGES}.external_0", f"{OBS_IMAGES}.external_1", f"{OBS_IMAGES}.wrist_0"]


@pytest.fixture(scope="module")
def selection():
    if not (DATA_ROOT / "corpus" / "episodes.jsonl").is_file():
        pytest.skip("federated corpora not present")
    return select_actor_anchors(open_federated_corpus(DATA_ROOT))


def _pack_step(**overrides):
    if not (BASE_PATH / "processor_config.json").is_file():
        pytest.skip("MolmoAct2 base checkpoint not present")
    kwargs = {
        "base_path": str(BASE_PATH),
        "action_mode": "continuous",
        "image_keys": list(ROLE_KEYS),
        "chunk_size": 30,
        "max_action_dim": 8,
    }
    kwargs.update(overrides)
    return MolmoAct2PackInputsProcessorStep(**kwargs)


def _transition(batch_size: int, present: list[list[bool]], history_slots: int = 3):
    observation = {OBS_STATE: torch.zeros(batch_size, 8)}
    complementary = {f"history.{OBS_STATE}": torch.zeros(batch_size, history_slots, 8)}
    for index, key in enumerate(ROLE_KEYS):
        observation[key] = torch.zeros(batch_size, 3, 64, 64, dtype=torch.uint8)
        complementary[f"history.{key}"] = torch.zeros(
            batch_size, history_slots, 3, 64, 64, dtype=torch.uint8
        )
        complementary[f"camera_is_present.{key}"] = torch.tensor(
            [row[index] for row in present], dtype=torch.bool
        )
    return create_transition(
        observation=observation,
        action=torch.zeros(batch_size, 30, 8),
        complementary_data=complementary,
    )


# ── Presence lookup ──────────────────────────────────────────────────────────


def test_presence_is_read_by_key_not_by_column_order() -> None:
    presence = MolmoAct2PackInputsProcessorStep._extract_camera_presence(
        {
            f"camera_is_present.{ROLE_KEYS[0]}": torch.tensor([True, True]),
            f"camera_is_present.{ROLE_KEYS[1]}": torch.tensor([False, True]),
        },
        # deliberately reversed relative to how the batch was written
        list(reversed(ROLE_KEYS)),
        2,
    )
    # reversed order: [wrist_0, external_1, external_0]
    assert presence[:, 2].tolist() == [True, True]
    assert presence[:, 1].tolist() == [False, True]
    assert presence[:, 0].tolist() == [True, True]  # no column -> present


def test_a_batch_without_presence_columns_is_unchanged() -> None:
    presence = MolmoAct2PackInputsProcessorStep._extract_camera_presence({}, ROLE_KEYS, 3)
    assert presence.all()


# ── Gate D: absent roles contribute no attended tokens ───────────────────────


def test_a_mixed_two_and_three_camera_batch_preprocesses() -> None:
    step = _pack_step()
    out = step(_transition(2, [[True, True, True], [True, False, True]]))
    complementary = out[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary["input_ids"].shape[0] == 2
    # The token layout stays uniform; only the mask differs.
    assert complementary["input_ids"].shape == complementary["attention_mask"].shape


def test_an_absent_camera_has_zero_attended_patch_tokens() -> None:
    from lerobot.policies.depth_pointmap.modeling_stream import wrist_cam_token_indices

    step = _pack_step()
    out = step(_transition(2, [[True, True, True], [True, False, True]]))
    complementary = out[TransitionKey.COMPLEMENTARY_DATA]
    mask, ids = complementary["attention_mask"], complementary["input_ids"]
    for cam_index in range(3):
        span = wrist_cam_token_indices(
            ids, image_patch_id=step._image_patch_id, num_images=3, cam_index=cam_index
        )
        assert mask[0, span[0]].sum() == span.shape[1], "row 0 records every camera"
        expected = 0 if cam_index == 1 else span.shape[1]
        assert mask[1, span[1]].sum() == expected


def test_the_absent_camera_also_loses_its_temporal_history() -> None:
    """A zero placeholder must not be read as a past frame either."""
    step = _pack_step()
    out = step(_transition(2, [[True, True, True], [True, False, True]]))
    history_mask = out[TransitionKey.COMPLEMENTARY_DATA]["history_images_mask"]
    assert history_mask.shape == (2, 3)
    assert history_mask[0].all()
    assert history_mask[1].tolist() == [True, False, True]


def test_rgb_history_reaches_the_video_encoder_with_the_configured_ages() -> None:
    step = _pack_step(history_times_seconds=[6.0, 4.0, 2.0])
    out = step(_transition(2, [[True, True, True], [True, True, True]]))
    complementary = out[TransitionKey.COMPLEMENTARY_DATA]
    frames = complementary["history_images"]
    # (B, cameras, T_h, patches, patch_dim), cameras in prompt order.
    assert frames.ndim == 5
    assert frames.shape[0] == 2
    assert frames.shape[1] == len(ROLE_KEYS)
    assert frames.shape[2] == 3
    assert complementary["history_image_times"].tolist() == [6.0, 4.0, 2.0]


# ── Role mapping and temporal order against the corpus ───────────────────────


def test_roles_carry_the_frames_the_corpus_recorded_for_them(selection) -> None:
    """Recognizable frames: each role must equal that source's own named camera."""
    spec = DiverseSampleSpec()
    buffer = DiverseActorBuffer(selection, spec)
    index = next(
        i
        for i, row in enumerate(selection.rows)
        if row["source"] == "droid" and len(row["camera_roles"]) == 3
    )
    row = selection.rows[index]
    assert row["camera_roles"] == {
        "external_0": "left_external",
        "external_1": "right_external",
        "wrist_0": "wrist",
    }
    sample = buffer.load_sample(index)
    episode = selection.corpus.common.episode(row["episode_id"])
    slots = [*spec.history_slots, PACKED_CURRENT_SLOT]
    packed = np.asarray(row["history_frames"], dtype=np.int64)

    for role, camera in row["camera_roles"].items():
        reference = np.asarray(episode.frames(camera, packed[slots]))
        geometry = buffer.geometry_for(reference.shape[1:3])
        from lerobot.rl.data_sources.diverse_actor_buffer import _resize_frames

        expected = _resize_frames(reference, geometry, nearest=False).transpose(0, 3, 1, 2)
        assert np.array_equal(sample["images"][role], expected), f"{role} holds the wrong camera"


def test_history_frames_are_oldest_to_newest(selection) -> None:
    spec = DiverseSampleSpec(load_images=False)
    buffer = DiverseActorBuffer(selection, spec)
    index = next(i for i, row in enumerate(selection.rows) if row["source"] == "robochallenge")
    sample = buffer.load_sample(index)
    ages = sample["timestamp"] - sample["history_timestamps"]
    assert list(np.round(ages, 3)) == [6.0, 4.0, 2.0]
    assert packed_history_slots(list(spec.history_ages_s)) == [0, 2, 4]

"""Tests for the source-native diverse actor buffer (integration plan phase B).

Gate B is the anchor-encoding test: for fixed rows from every source, all 30 encoded
points must match a direct reference, and (encoded + anchor state) must reconstruct
every absolute action including the gripper.

Geometry tests run everywhere. Corpus-backed tests skip when the data is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.datasets.diverse_actor_selection import (
    PACKED_CURRENT_SLOT,
    open_federated_corpus,
    select_actor_anchors,
)
from lerobot.policies.molmoact2.anchor_encoding import ANCHOR_KEY, AnchorEncodeStep
from lerobot.processor.converters import create_transition
from lerobot.rl.buffer import concatenate_variable_dim_batch_transitions
from lerobot.rl.data_sources.diverse_actor_buffer import (
    SOURCE_IDS,
    DiverseActorBuffer,
    DiverseSampleSpec,
    ResizeGeometry,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

DATA_ROOT = Path(__file__).resolve().parents[3] / "outputs/diverse_robot_dataset"


@pytest.fixture(scope="module")
def selection():
    if not (DATA_ROOT / "corpus" / "episodes.jsonl").is_file():
        pytest.skip("federated corpora not present")
    return select_actor_anchors(open_federated_corpus(DATA_ROOT))


@pytest.fixture(scope="module")
def buffer(selection):
    return DiverseActorBuffer(
        selection, DiverseSampleSpec(load_images=False, load_depth=True), seed=0
    )


def _one_row_per_source(selection) -> list[int]:
    """The first anchor of each source, in corpus order -- deterministic across runs."""
    picked: dict[str, int] = {}
    for index, row in enumerate(selection.rows):
        picked.setdefault(row["source"], index)
    return [picked[source] for source in sorted(picked)]


# ── Resize geometry ──────────────────────────────────────────────────────────


def test_geometry_preserves_aspect_and_centres_the_pad() -> None:
    wide = ResizeGeometry.fit((720, 1280), (480, 640))  # 16:9 into 4:3
    assert wide.scale == 0.5
    assert wide.valid_box == (60, 0, 360, 640)
    square = ResizeGeometry.fit((256, 256), (480, 640))
    assert square.valid_box == (0, 80, 480, 480)
    assert ResizeGeometry.fit((480, 640), (480, 640)).is_identity


def test_pixel_valid_mask_is_exactly_the_sensor_region() -> None:
    geometry = ResizeGeometry.fit((720, 1280), (480, 640))
    mask = geometry.pixel_valid_mask()
    assert mask.sum() == 360 * 640
    assert not mask[:60].any() and not mask[420:].any()


def test_intrinsics_follow_the_same_resize_and_pad() -> None:
    """A pixel and its back-projection must agree about where the image moved."""
    geometry = ResizeGeometry.fit((480, 640), (240, 640))
    fx, fy, cx, cy = geometry.transform_intrinsics([400.0, 400.0, 320.0, 240.0])
    assert (fx, fy) == (200.0, 200.0)
    # cx scales with the image; cy scales and then shifts by the centring pad.
    assert cx == 320.0 * geometry.scale + geometry.left
    assert cy == 240.0 * geometry.scale + geometry.top


# ── Sample contract ──────────────────────────────────────────────────────────


def test_batch_shapes_and_masks(buffer) -> None:
    batch = buffer.sample(8, action_chunk_size=30)
    assert batch[ACTION].shape == (8, 30, 8)
    assert batch["state"][OBS_STATE].shape == (8, 8)
    assert batch["state"][f"history.{OBS_STATE}"].shape == (8, 3, 8)
    info = batch["complementary_info"]
    for position in range(8):
        native = int((~info["action_dim_is_pad"][position]).sum())
        assert native in (7, 8)
        # Padding is a suffix, and state and action agree on the width.
        assert not info["action_dim_is_pad"][position, :native].any()
        assert info["action_dim_is_pad"][position, native:].all()
        assert torch.equal(info["action_dim_is_pad"][position], info["state_dim_is_pad"][position])
        # sum, not max: a native-8 sample has an empty padding slice.
        assert batch[ACTION][position, :, native:].abs().sum() == 0


def test_a_different_chunk_size_is_refused_rather_than_trimmed(buffer) -> None:
    with pytest.raises(ValueError, match="packs"):
        buffer.sample(2, action_chunk_size=50)


def test_identity_columns_point_back_at_the_sampled_rows(buffer, selection) -> None:
    indices = _one_row_per_source(selection)
    batch = buffer.collate(indices)
    info = batch["complementary_info"]
    assert info["diverse_row_index"].tolist() == indices
    for position, index in enumerate(indices):
        row = selection.rows[index]
        assert info["source_id"][position].item() == SOURCE_IDS[row["source"]]
        assert info["anchor_index"][position].item() == int(row["anchor_index"])
        assert info["action_layout_id"][position].item() == int(row["action_layout_id"])


def test_every_source_is_representable_in_one_batch(buffer, selection) -> None:
    batch = buffer.collate(_one_row_per_source(selection))
    info = batch["complementary_info"]
    assert set(info["source_id"].tolist()) == {SOURCE_IDS[s] for s in ("droid", "droid_success", "fmb", "robochallenge", "ur7e")}
    # Mixed widths survive collation rather than being unified by truncation.
    widths = (~info["action_dim_is_pad"]).sum(dim=1).tolist()
    assert set(widths) == {7, 8}


# ── Gate B: the chunk and its anchor encoding ────────────────────────────────


def test_the_whole_thirty_point_chunk_survives_loading(buffer, selection) -> None:
    indices = _one_row_per_source(selection)
    batch = buffer.collate(indices)
    for position, index in enumerate(indices):
        row = selection.rows[index]
        reference = np.asarray(selection.corpus.actor_sample(row, cameras=False)["action"])
        native = reference.shape[1]
        assert reference.shape[0] == 30
        loaded = batch[ACTION][position, :, :native].numpy()
        # Every one of the 30 points, not just the first.
        assert np.allclose(loaded, reference, atol=1e-6), f"{row['episode_id']} chunk changed"
        assert not np.allclose(loaded[0], loaded[-1]) or np.allclose(reference[0], reference[-1])


def test_anchor_encoding_matches_a_direct_reference_for_every_source(buffer, selection) -> None:
    """Gate B. Encode the batch with the real pipeline step and check all 30 points."""
    indices = _one_row_per_source(selection)
    batch = buffer.collate(indices)
    encoded = AnchorEncodeStep(encoding="anchor")(
        create_transition(
            observation={OBS_STATE: batch["state"][OBS_STATE]},
            action=batch[ACTION],
        )
    )
    encoded_action = encoded[TransitionKey.ACTION]
    anchor = encoded[TransitionKey.COMPLEMENTARY_DATA][ANCHOR_KEY]

    for position, index in enumerate(indices):
        row = selection.rows[index]
        sample = selection.corpus.actor_sample(row, cameras=False)
        reference_action = np.asarray(sample["action"], dtype=np.float64)
        reference_state = np.asarray(sample["observation.state"], dtype=np.float64)[
            PACKED_CURRENT_SLOT
        ]
        native = reference_action.shape[1]

        # 1. the anchor is the current state, not the first action and not a history slot
        assert np.allclose(anchor[position, :native].numpy(), reference_state, atol=1e-5)

        # 2. every encoded point equals action[k] - state, computed independently here
        expected = reference_action - reference_state[None, :]
        actual = encoded_action[position, :, :native].numpy().astype(np.float64)
        assert actual.shape == (30, native)
        assert np.allclose(actual, expected, atol=1e-4), f"{row['source']} encoding differs"

        # 3. reconstruction, including the gripper (the last native dimension)
        rebuilt = actual + reference_state[None, :]
        assert np.allclose(rebuilt, reference_action, atol=1e-4)
        assert np.allclose(rebuilt[:, -1], reference_action[:, -1], atol=1e-4)

        # 4. padded dimensions stay exactly zero through the encoding
        assert encoded_action[position, :, native:].abs().sum() == 0


def test_encoding_is_not_secretly_a_first_action_delta(buffer, selection) -> None:
    """The classic failure mode: anchoring on action[0] instead of the current state."""
    indices = _one_row_per_source(selection)
    batch = buffer.collate(indices)
    encoded = AnchorEncodeStep(encoding="anchor")(
        create_transition(
            observation={OBS_STATE: batch["state"][OBS_STATE]}, action=batch[ACTION]
        )
    )[TransitionKey.ACTION]
    # If the anchor were action[0], the first encoded row would be identically zero.
    assert encoded[:, 0, :].abs().max() > 0


# ── Mixing with a ReBot-shaped batch ─────────────────────────────────────────


def test_a_diverse_batch_concatenates_with_a_seven_dimensional_peer(buffer, selection) -> None:
    diverse = buffer.collate(_one_row_per_source(selection))
    size = diverse[ACTION].shape[0]
    rebot = {
        "state": {OBS_STATE: torch.zeros(2, 7)},
        ACTION: torch.zeros(2, 30, 7),
        "reward": torch.zeros(2),
        "next_state": {OBS_STATE: torch.zeros(2, 7)},
        "done": torch.zeros(2),
        "truncated": torch.zeros(2),
        "complementary_info": {},
    }
    diverse = {key: value for key, value in diverse.items()}
    diverse["state"] = {OBS_STATE: diverse["state"][OBS_STATE]}
    diverse["next_state"] = {OBS_STATE: diverse["next_state"][OBS_STATE]}
    merged = concatenate_variable_dim_batch_transitions(diverse, rebot)
    assert merged[ACTION].shape == (size + 2, 30, 8)
    mask = merged["complementary_info"]["action_dim_is_pad"]
    assert mask.shape == (size + 2, 8)
    # The ReBot rows arrive 7-wide and are marked padded in dimension 8.
    assert mask[-2:, 7].all()

"""Tests for the diverse actor cache and the ReBot role adapter (plan phases C/D).

Gate C is the equivalence test: a cached anchor and an uncached one must agree on
state, every action point, every present RGB frame, depth, masks, timestamps, and
metadata. The cache is built at a small image size so the sparse bank files stay small.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.datasets.diverse_actor_selection import (
    open_federated_corpus,
    select_actor_anchors,
)
from lerobot.rl.buffer import concatenate_variable_dim_batch_transitions
from lerobot.rl.data_sources.diverse_actor_buffer import DiverseActorBuffer, DiverseSampleSpec
from lerobot.rl.data_sources.diverse_actor_cache import (
    DiverseFrameCache,
    build_cache,
    cache_fingerprint,
    find_cache,
    plan_cache,
    source_depth_intrinsics,
    training_depth_intrinsics,
)
from lerobot.rl.data_sources.rebot_role_adapter import (
    RebotCacheProbeError,
    probe_rebot_cache,
    rebot_role_renames,
)
from lerobot.utils.constants import ACTION, OBS_STATE

DATA_ROOT = Path(__file__).resolve().parents[3] / "outputs/diverse_robot_dataset"
# Small enough that the preallocated (sparse) frame banks cost nothing on disk.
TEST_SPEC = DiverseSampleSpec(image_size=(48, 64), depth_size=(48, 64))


@pytest.fixture(scope="module")
def selection():
    if not (DATA_ROOT / "corpus" / "episodes.jsonl").is_file():
        pytest.skip("federated corpora not present")
    return select_actor_anchors(open_federated_corpus(DATA_ROOT))


@pytest.fixture(scope="module")
def built(selection, tmp_path_factory):
    """A partial cache over the smallest episode of two contrasting sources."""
    grouped = selection.rows_by_episode()
    wanted = []
    for source, embodiment in (("fmb", "Franka"), ("robochallenge", "UR5")):
        candidates = [
            (len(rows), episode_id)
            for episode_id, rows in grouped.items()
            if (rows[0]["source"], rows[0]["embodiment"]) == (source, embodiment)
        ]
        wanted.append(min(candidates)[1])
    cache_dir = tmp_path_factory.mktemp("diverse_cache")
    path = build_cache(DATA_ROOT, cache_dir, TEST_SPEC, selection=selection, episodes=wanted)
    return path, wanted


# ── Plan ─────────────────────────────────────────────────────────────────────


def test_frames_are_banked_once_not_per_anchor(selection) -> None:
    """The whole point: overlapping histories share bank rows."""
    plans, bank_sizes, _ = plan_cache(selection, TEST_SPEC)
    slots = TEST_SPEC.num_history + 1
    materialized = sum(len(plan.row_indices) * slots for plan in plans if "external_0" in plan.roles)
    assert bank_sizes["external_0"] < materialized / 2
    # An episode's bank rows are its distinct referenced frames, ascending and unique.
    plan = next(plan for plan in plans if len(plan.row_indices) > 20)
    assert len(plan.referenced) == len(set(plan.referenced.tolist()))
    assert list(plan.referenced) == sorted(plan.referenced)


def test_bank_offsets_are_deterministic(selection) -> None:
    first = plan_cache(selection, TEST_SPEC)
    second = plan_cache(selection, TEST_SPEC)
    assert [plan.offsets for plan in first[0]] == [plan.offsets for plan in second[0]]
    assert first[1] == second[1]


# ── Fingerprint ──────────────────────────────────────────────────────────────


def test_fingerprint_moves_with_the_sample_contract(selection) -> None:
    base = cache_fingerprint(DATA_ROOT, TEST_SPEC, anchors=100, episodes=4)
    other_history = DiverseSampleSpec(
        image_size=TEST_SPEC.image_size, depth_size=TEST_SPEC.depth_size, history_ages_s=(6.0, 4.0)
    )
    assert cache_fingerprint(DATA_ROOT, other_history, anchors=100, episodes=4) != base
    assert cache_fingerprint(DATA_ROOT, TEST_SPEC, anchors=101, episodes=4) != base
    other_size = DiverseSampleSpec(image_size=(96, 128), depth_size=TEST_SPEC.depth_size)
    assert cache_fingerprint(DATA_ROOT, other_size, anchors=100, episodes=4) != base


def test_a_partial_cache_is_not_offered_to_training(built) -> None:
    path, _ = built
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["partial"] is True
    args = {"anchors": metadata["selection"]["anchors"], "episodes": metadata["selection"]["episodes"]}
    assert find_cache(DATA_ROOT, path.parent, TEST_SPEC, require_complete=True, **args) is None
    assert find_cache(DATA_ROOT, path.parent, TEST_SPEC, require_complete=False, **args) == path


def test_build_is_atomic(built) -> None:
    path, _ = built
    # Nothing half-built is left behind under a findable name.
    assert not list(path.parent.glob(".building-*"))
    assert (path / "metadata.json").is_file()


# ── Gate C: cached and uncached agree ────────────────────────────────────────


def test_cached_and_uncached_anchors_agree_on_everything(selection, built) -> None:
    path, episodes = built
    cache = DiverseFrameCache(path, TEST_SPEC)
    live = DiverseActorBuffer(selection, TEST_SPEC)
    cached = DiverseActorBuffer(selection, TEST_SPEC, cache=cache)

    first_of_episode: dict[str, int] = {}
    for index in cache.built_row_indices:
        first_of_episode.setdefault(selection.rows[int(index)]["episode_id"], int(index))
    picks = sorted(first_of_episode.values())
    assert len(picks) == len(episodes)

    left = live.collate(picks)
    right = cached.collate(picks)
    assert torch.equal(left[ACTION], right[ACTION])
    for key, value in left["state"].items():
        assert torch.equal(value, right["state"][key]), f"state {key} differs"
    for key, value in left["complementary_info"].items():
        assert torch.equal(value, right["complementary_info"][key]), f"complementary {key} differs"


def test_reading_an_anchor_the_partial_cache_lacks_is_an_error(selection, built) -> None:
    path, _ = built
    cache = DiverseFrameCache(path, TEST_SPEC)
    held = set(cache.built_row_indices.tolist())
    missing = next(index for index in range(len(selection.rows)) if index not in held)
    with pytest.raises(KeyError, match="not in this cache"):
        cache.sample(missing)


def test_depth_survives_the_cache_losslessly(selection, built) -> None:
    path, _ = built
    cache = DiverseFrameCache(path, TEST_SPEC)
    fmb = [
        int(index)
        for index in cache.built_row_indices
        if selection.rows[int(index)]["source"] == "fmb"
    ]
    assert fmb
    sample = cache.sample(fmb[0])
    assert sample["depth"].dtype == np.uint16  # never rounded through a float
    assert sample["depth_valid"].dtype == np.bool_
    # The column is float32; 0.1 is not exactly representable and the 1.5e-9 relative
    # error on a scale factor is far below the sensor quantum.
    assert sample["depth_units_mm_per_level"] == pytest.approx(0.1)
    assert sample["depth_scale_provenance"] == "user_authorized_same_D405_model_assumption"
    # Presence and pixel validity are different facts: depth is present here, and only
    # some of its pixels are valid.
    assert 0 < sample["depth_valid"].mean() < 1


# ── Depth intrinsics ─────────────────────────────────────────────────────────


def test_depth_intrinsics_follow_the_stated_chain() -> None:
    """256x256 is not 4:3, so the FMB step is a non-uniform scale by construction."""
    fx, fy, cx, cy = source_depth_intrinsics((256, 256))
    assert fx != fy
    assert fx == pytest.approx(394.9832 * 256 / 640)
    assert fy == pytest.approx(394.9832 * 256 / 480)
    # The training resize is uniform, so it cannot restore the ratio it inherited.
    train_fx, train_fy, _, _ = training_depth_intrinsics((256, 256), (480, 640))
    assert train_fx / train_fy == pytest.approx(fx / fy)


def test_cache_records_the_intrinsics_assumption(built) -> None:
    path, _ = built
    metadata = json.loads((path / "metadata.json").read_text())
    assert "non_uniform_resize" in metadata["depth_intrinsics_assumption"]
    assert metadata["depth_base_intrinsics"] == [394.9832, 394.9832, 322.5604, 238.6966]
    assert metadata["depth_units_mm_per_level"] == 0.1
    assert len(metadata["depth_training_intrinsics"]) == 4


# ── ReBot cache probe ────────────────────────────────────────────────────────


def _write_rebot_metadata(root: Path, **overrides) -> Path:
    metadata = {
        "num_transitions": 100,
        "image_keys": ["observation.images.wrist", "observation.images.top"],
        "depth_keys": ["wrist.depth"],
        "image_storage_dtype": "uint8",
        "image_storage_size": None,
        "image_stride": 3,
        "fingerprint": "deadbeef",
        "shapes": {"observation.images.top": [3, 480, 640], "depth.wrist.depth": [480, 640]},
        "dtypes": {"dones": "bool"},
    }
    metadata.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root


def test_a_healthy_rebot_cache_is_reused(tmp_path) -> None:
    report = probe_rebot_cache(
        _write_rebot_metadata(tmp_path / "cache"),
        history_offsets_frames=[180, 120, 60],
        measure_reach=False,
    )
    assert report.usable, report.problems
    assert report.image_stride == 3


def test_a_stride_that_cannot_address_minus_six_seconds_fails(tmp_path) -> None:
    report = probe_rebot_cache(
        _write_rebot_metadata(tmp_path / "cache", image_stride=7),
        history_offsets_frames=[180, 120, 60],
        measure_reach=False,
    )
    assert not report.usable
    assert any("not multiples of image_stride" in problem for problem in report.problems)


def test_a_missing_depth_column_fails(tmp_path) -> None:
    report = probe_rebot_cache(
        _write_rebot_metadata(tmp_path / "cache", depth_keys=[]),
        history_offsets_frames=[180, 120, 60],
        measure_reach=False,
    )
    assert not report.usable
    assert any("depth" in problem for problem in report.problems)


def test_a_missing_camera_fails(tmp_path) -> None:
    report = probe_rebot_cache(
        _write_rebot_metadata(tmp_path / "cache", image_keys=["observation.images.top"]),
        history_offsets_frames=[180],
        measure_reach=False,
    )
    assert not report.usable
    assert any("missing RGB columns" in problem for problem in report.problems)


def test_role_renames_cover_rgb_and_depth() -> None:
    renames = rebot_role_renames(
        ["observation.images.top", "observation.images.wrist"], ["wrist.depth"]
    )
    assert renames["observation.images.top"] == "observation.images.external_0"
    assert renames["observation.images.wrist"] == "observation.images.wrist_0"
    assert renames["depth.wrist.depth"] == "depth.wrist_0.depth"


def test_an_unmapped_rebot_camera_refuses_to_guess() -> None:
    with pytest.raises(RebotCacheProbeError, match="unmapped"):
        rebot_role_renames(["observation.images.overhead"], [])


# ── Mixed batch ──────────────────────────────────────────────────────────────


def test_a_seven_wide_peer_concatenates_with_state_history_present(selection, built) -> None:
    """ReplayBuffer puts state history in "state"; padding has to reach it there."""
    path, _ = built
    cached = DiverseActorBuffer(selection, TEST_SPEC, cache=DiverseFrameCache(path, TEST_SPEC))
    diverse = cached.collate(sorted(int(i) for i in DiverseFrameCache(path, TEST_SPEC).built_row_indices)[:2])
    size = diverse[ACTION].shape[0]
    peer = {
        "state": {
            OBS_STATE: torch.zeros(2, 7, dtype=torch.bfloat16),
            f"history.{OBS_STATE}": torch.zeros(2, 3, 7, dtype=torch.bfloat16),
        },
        ACTION: torch.zeros(2, 30, 7, dtype=torch.bfloat16),
        "reward": torch.zeros(2),
        "next_state": {OBS_STATE: torch.zeros(2, 7, dtype=torch.bfloat16)},
        "done": torch.zeros(2),
        "truncated": torch.zeros(2),
        "complementary_info": {},
    }
    trimmed = dict(diverse)
    trimmed["state"] = {
        OBS_STATE: diverse["state"][OBS_STATE],
        f"history.{OBS_STATE}": diverse["state"][f"history.{OBS_STATE}"],
    }
    trimmed["next_state"] = {OBS_STATE: diverse["next_state"][OBS_STATE]}
    merged = concatenate_variable_dim_batch_transitions(trimmed, peer)
    assert merged["state"][f"history.{OBS_STATE}"].shape == (size + 2, 3, 8)
    # bfloat16 beside float32 promotes up, never down.
    assert merged[ACTION].dtype == torch.float32

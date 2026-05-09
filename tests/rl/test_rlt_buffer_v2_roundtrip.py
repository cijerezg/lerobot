"""Tests for the v2 RLT replay buffer (review-only image / episode metadata)."""

import json

import pytest
import torch

from lerobot.rl.rlt_buffer import (
    RLT_REPLAY_BUFFER_VERSION,
    RLTReplayBuffer,
    RLTReplaySample,
)


def _v2_sample(
    *,
    offset: float = 0.0,
    episode_id: int | None = None,
    done: bool = False,
    success: bool = False,
    failure: bool = False,
    is_intervention: bool = False,
    images_jpeg: dict[str, bytes] | None = None,
    inference_ts: float | None = None,
    chunk_start_step: int | None = None,
) -> RLTReplaySample:
    return RLTReplaySample(
        rl_token=torch.ones(4) + offset,
        proprio=torch.ones(2) + offset,
        reference_chunk=torch.zeros(3, 2) + offset,
        executed_chunk=torch.ones(3, 2) + offset,
        next_rl_token=torch.ones(4) * 2 + offset,
        next_proprio=torch.ones(2) * 2 + offset,
        next_reference_chunk=torch.zeros(3, 2) + offset,
        reward=1.0 if success else (0.0 if failure else 0.0),
        done=done,
        is_intervention=is_intervention,
        images_jpeg=images_jpeg,
        inference_ts=inference_ts,
        episode_id=episode_id,
        success=success,
        failure=failure,
        chunk_start_step=chunk_start_step,
    )


def _write_review_sidecar(replay_path, episodes: dict[str, dict[str, object]]) -> None:
    sidecar = {
        "version": 1,
        "buffer_path": str(replay_path),
        "episodes": episodes,
    }
    replay_path.with_suffix(".review.json").write_text(json.dumps(sidecar), encoding="utf-8")


def test_v2_save_load_round_trip_preserves_review_fields(tmp_path):
    replay = RLTReplayBuffer(capacity=4)
    replay.add(
        _v2_sample(
            offset=0.0,
            episode_id=7,
            done=False,
            inference_ts=1234.5,
            images_jpeg={"observation.images.front": b"\xff\xd8\xff\xd9"},  # SOI/EOI markers
            chunk_start_step=10,
        )
    )
    replay.add(
        _v2_sample(
            offset=1.0,
            episode_id=7,
            done=True,
            success=True,
            inference_ts=1235.0,
            images_jpeg={
                "observation.images.front": b"\xff\xd8\xff\xd9",
                "observation.images.wrist": b"\xff\xd8\xff\xda",
            },
            chunk_start_step=20,
        )
    )

    path = tmp_path / "rlt_v2.pt"
    replay.save(path)

    loaded = RLTReplayBuffer.load(path, capacity=4)
    samples = loaded.samples()
    assert len(samples) == 2

    s0, s1 = samples
    assert s0.episode_id == 7
    assert s0.done is False
    assert s0.success is False
    assert s0.failure is False
    assert s0.inference_ts == 1234.5
    assert s0.chunk_start_step == 10
    assert s0.images_jpeg == {"observation.images.front": b"\xff\xd8\xff\xd9"}

    assert s1.episode_id == 7
    assert s1.done is True
    assert s1.success is True
    assert s1.failure is False
    assert s1.inference_ts == 1235.0
    assert s1.chunk_start_step == 20
    assert set(s1.images_jpeg.keys()) == {
        "observation.images.front",
        "observation.images.wrist",
    }


def test_v2_state_dict_announces_version_2():
    replay = RLTReplayBuffer(capacity=2)
    replay.add(_v2_sample(episode_id=0))
    state = replay.state_dict()
    assert state["version"] == RLT_REPLAY_BUFFER_VERSION == 2


def test_loading_v1_buffer_defaults_review_fields_to_none(tmp_path):
    """A v1-shaped buffer (no review keys) must round-trip cleanly with None defaults."""
    # Synthesize a v1 state_dict by stripping the review keys from a v2 dump.
    v2_replay = RLTReplayBuffer(capacity=2)
    v2_replay.add(_v2_sample(episode_id=0, inference_ts=42.0))
    state_dict = v2_replay.state_dict()
    # Strip every v2-only key from the persisted samples.
    v1_keys_to_drop = {
        "images_jpeg",
        "inference_ts",
        "episode_id",
        "success",
        "failure",
        "chunk_start_step",
    }
    for sample in state_dict["samples"]:
        for key in v1_keys_to_drop:
            sample.pop(key, None)
    state_dict["version"] = 1

    path = tmp_path / "rlt_v1.pt"
    torch.save(state_dict, path)

    loaded = RLTReplayBuffer.load(path, capacity=2)
    samples = loaded.samples()
    assert len(samples) == 1
    s = samples[0]
    assert s.images_jpeg is None
    assert s.inference_ts is None
    assert s.episode_id is None
    assert s.success is None
    assert s.failure is None
    assert s.chunk_start_step is None
    # And v1 fields are still intact.
    assert s.rl_token.shape == (4,)
    assert bool(s.done) is False


def test_review_sidecar_absent_leaves_samples_unchanged(tmp_path):
    replay = RLTReplayBuffer(capacity=3)
    replay.add(_v2_sample(offset=0.0, episode_id=1, success=False, failure=False))
    replay.add(_v2_sample(offset=1.0, episode_id=2, done=True, success=True, failure=False))
    path = tmp_path / "rlt_online_replay.pt"
    replay.save(path)

    loaded = RLTReplayBuffer.load(path, capacity=3, apply_review_sidecar=True)
    samples = loaded.samples()

    assert len(samples) == 2
    assert [sample.episode_id for sample in samples] == [1, 2]
    assert [sample.success for sample in samples] == [False, True]
    assert [sample.failure for sample in samples] == [False, False]
    assert [sample.done for sample in samples] == [False, True]
    assert [sample.reward for sample in samples] == [0.0, 1.0]


def test_review_sidecar_deleted_episode_removes_samples(tmp_path):
    replay = RLTReplayBuffer(capacity=4)
    replay.add(_v2_sample(offset=0.0, episode_id=12))
    replay.add(_v2_sample(offset=1.0, episode_id=12, done=True, failure=True))
    replay.add(_v2_sample(offset=2.0, episode_id=14))
    path = tmp_path / "rlt_online_replay.pt"
    replay.save(path)
    _write_review_sidecar(
        path,
        {
            "12": {"label": "failure", "deleted": True},
            "999": {"label": "success", "deleted": False},
        },
    )

    loaded = RLTReplayBuffer.load(path, capacity=4, apply_review_sidecar=True)

    assert len(loaded) == 1
    assert [sample.episode_id for sample in loaded.samples()] == [14]


@pytest.mark.parametrize(
    ("label", "expected_success", "expected_failure", "expected_reward"),
    [
        ("success", True, False, 1.0),
        ("failure", False, True, 0.0),
    ],
)
def test_review_sidecar_open_episode_changed_to_outcome_updates_samples(
    tmp_path, label, expected_success, expected_failure, expected_reward
):
    replay = RLTReplayBuffer(capacity=2)
    replay.add(_v2_sample(offset=0.0, episode_id=14, done=False, success=False, failure=False))
    replay.add(_v2_sample(offset=1.0, episode_id=14, done=False, success=False, failure=False))
    path = tmp_path / "rlt_online_replay.pt"
    replay.save(path)
    _write_review_sidecar(path, {"14": {"label": label, "deleted": False}})

    loaded = RLTReplayBuffer.load(path, capacity=2, apply_review_sidecar=True)
    samples = loaded.samples()

    assert [sample.success for sample in samples] == [expected_success, expected_success]
    assert [sample.failure for sample in samples] == [expected_failure, expected_failure]
    assert [sample.done for sample in samples] == [False, True]
    assert [sample.reward for sample in samples] == [0.0, expected_reward]


def test_review_sidecar_outcome_changed_to_open_clears_flags(tmp_path):
    replay = RLTReplayBuffer(capacity=2)
    replay.add(_v2_sample(offset=0.0, episode_id=3, done=False, success=True, failure=False))
    replay.add(_v2_sample(offset=1.0, episode_id=3, done=True, success=True, failure=False))
    path = tmp_path / "rlt_online_replay.pt"
    replay.save(path)
    _write_review_sidecar(path, {"3": {"label": "open", "deleted": False}})

    loaded = RLTReplayBuffer.load(path, capacity=2, apply_review_sidecar=True)
    samples = loaded.samples()

    assert [sample.success for sample in samples] == [False, False]
    assert [sample.failure for sample in samples] == [False, False]
    assert [sample.done for sample in samples] == [False, False]
    assert [sample.reward for sample in samples] == [0.0, 0.0]


def test_review_sidecar_malformed_entries_are_ignored_without_crashing(tmp_path, caplog):
    replay = RLTReplayBuffer(capacity=3)
    replay.add(_v2_sample(offset=0.0, episode_id=1, done=True, success=True, failure=False))
    replay.add(_v2_sample(offset=1.0, episode_id=2, done=True, success=False, failure=True))
    path = tmp_path / "rlt_online_replay.pt"
    replay.save(path)
    _write_review_sidecar(
        path,
        {
            "1": {"label": "not-a-label", "deleted": False},
            "2": {"label": "success", "deleted": "false"},
            "999": {"label": "success", "deleted": False},
        },
    )

    with caplog.at_level("WARNING"):
        loaded = RLTReplayBuffer.load(path, capacity=3, apply_review_sidecar=True)

    samples = loaded.samples()
    assert len(samples) == 2
    assert samples[0].success is True
    assert samples[0].failure is False
    assert samples[1].success is False
    assert samples[1].failure is True
    assert "Ignoring malformed RLT review sidecar entry" in caplog.text


def test_review_sidecar_ignores_buffers_without_episode_ids(tmp_path):
    replay = RLTReplayBuffer(capacity=2)
    replay.add(_v2_sample(offset=0.0, episode_id=None, success=False, failure=False))
    path = tmp_path / "rlt_online_replay.pt"
    replay.save(path)
    _write_review_sidecar(path, {"0": {"label": "success", "deleted": False}})

    loaded = RLTReplayBuffer.load(path, capacity=2, apply_review_sidecar=True)
    sample = loaded.samples()[0]

    assert sample.episode_id is None
    assert sample.success is False
    assert sample.failure is False


def test_sample_batch_unaffected_by_review_fields(tmp_path):
    """Training batches must not depend on review fields (they're not part of the loss)."""
    replay = RLTReplayBuffer(capacity=4)
    replay.add(_v2_sample(offset=0.0, images_jpeg={"a": b"x"}))
    replay.add(_v2_sample(offset=1.0, images_jpeg=None))
    batch = replay.sample(2)
    assert batch["rl_token"].shape == (2, 4)
    assert batch["proprio"].shape == (2, 2)
    assert batch["reference_chunk"].shape == (2, 3, 2)
    assert batch["executed_chunk"].shape == (2, 3, 2)
    # No review keys should leak into the training batch.
    assert "images_jpeg" not in batch
    assert "inference_ts" not in batch
    assert "episode_id" not in batch

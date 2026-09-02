"""Tests for FMB source-native views and no-copy corpus federation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lerobot.datasets.fmb_corpus import FederatedDiverseCorpus, FMBCorpus
from lerobot.utils.gripper_event_targets import depth_gripper_event_labels_from_closed_mask


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _synthetic_fmb(root: Path) -> None:
    episode_id = "episode_000000_unit"
    episode_root = root / "episodes" / episode_id
    episode_root.mkdir(parents=True)
    frames = 80
    timestamps = np.arange(frames, dtype=np.float64) / 10.0
    joints = np.stack([timestamps + offset for offset in range(7)], axis=1)
    np.save(episode_root / "timestamp_s.npy", timestamps)
    # `q` is what state and action both read; the other two are retained by the
    # converter but deliberately unused (see FMBCorpusEpisode.state).
    np.save(episode_root / "q.npy", joints)
    gripper = np.zeros(frames, dtype=np.int64)
    gripper[65:75] = 1
    source_actions = np.zeros((frames, 7), dtype=np.float64)
    source_actions[:, -1] = gripper
    np.save(episode_root / "actions.npy", source_actions)
    np.save(episode_root / "tcp_pose.npy", np.zeros((frames, 7), dtype=np.float64))
    np.save(episode_root / "gripper_pose.npy", gripper)
    event_labels, _ = depth_gripper_event_labels_from_closed_mask(gripper == 1, fps=10.0)
    for key, values in event_labels.items():
        np.save(episode_root / f"{key}.npy", values)
    for camera in ("side_1", "side_2", "wrist_1"):
        np.save(episode_root / f"{camera}_rgb.npy", np.zeros((frames, 2, 2, 3), dtype=np.uint8))
    depth = np.full((frames, 2, 2), 2500, dtype=np.uint16)
    depth[0, 0, 0], depth[0, 0, 1] = 0, 65_535
    np.save(episode_root / "wrist_1_depth_z16.npy", depth)

    interval = {
        "start_timestep": 0,
        "end_timestep_exclusive": frames,
        "start_s_nominal": 0.0,
        "end_s_nominal_exclusive": 8.0,
        "duration_s_nominal": 8.0,
        "normalized_description": "grasp the object",
        "classification": "clean_complete",
        "quality": 4,
        "quality_provenance": "human_reviewed_rebot_rubric",
        "subtask_outcome": "unknown",
        "critic_eligible": True,
        "critic_rejection_reason": None,
        "pause_events": [],
        "mistake_events": [],
        "recovery_events": [],
    }
    episode = {
        "episode_id": episode_id,
        "nominal_fps": 10.0,
        "frame_count": frames,
        "split": "train",
        "primitive_intervals": [interval],
    }
    _write_jsonl(root / "episodes.jsonl", [episode])
    _write_jsonl(
        root / "actor_anchors_5hz.jsonl",
        [
            {
                "episode_id": episode_id,
                "split": "train",
                "anchor_timestep": 60,
                "anchor_s_nominal": 6.0,
                "history_frames": [0, 10, 20, 30, 40, 50, 60],
                "primitive": "grasp",
            }
        ],
    )
    _write_jsonl(
        root / "critic_intervals.jsonl",
        [{"episode_id": episode_id, "split": "train", **interval}],
    )


def test_fmb_actor_and_critic_views_keep_native_arrays_and_depth_contract(tmp_path: Path) -> None:
    _synthetic_fmb(tmp_path)
    corpus = FMBCorpus(tmp_path)

    actor_row = corpus.actor_anchors()[0]
    assert actor_row["views_copy_source_arrays"] is False
    assert actor_row["subtask"] == "grasp the object"
    assert actor_row["quality"] == 4
    assert actor_row["mistake"] is False
    actor = corpus.actor_sample(actor_row)
    # State and action are both [7 measured Franka joints, gripper] -- copy_state, so
    # they share a width and the action is a future slice of the same trajectory.
    assert actor["observation.state"].shape == (7, 8)
    assert actor["action"].shape == (30, 8)
    assert actor["action_source"] == "copy_state"
    assert actor["embodiment"] == "Franka"
    assert actor["observation.state_semantics"] == [
        "measured_franka_joint_positions",
        "gripper_pose",
    ]
    assert set(actor["observation.images"]) == {"side_1", "side_2", "wrist_1"}
    assert actor["observation.depth.wrist_1_z16"].dtype == np.uint16
    assert not actor["observation.depth.wrist_1_valid_mask"][0, 0, :2].any()
    assert actor["observation.depth.units_mm_per_level"] == 0.1
    assert actor["observation.depth.scale_provenance"] == "user_authorized_same_D405_model_assumption"
    assert actor["depth_gripper_close_target"] == np.float32(2.0**-0.5)
    assert actor["depth_gripper_open_target"] == np.float32(2.0**-1.5)
    assert actor["depth_gripper_close_target"].dtype == np.float32
    assert actor["depth_gripper_open_target"].dtype == np.float32

    critic_row = corpus.critic_intervals()[0]
    critic = corpus.critic_sample(critic_row, action_points=16, observation_frames=3)
    assert critic["action"].shape == (80, 8)
    assert critic["native_action_samples"] == 80
    assert critic["action.subsampled"].shape == (16, 8)
    assert critic["action.subsampled_mask"].all()
    assert critic["observation.depth.wrist_1_z16"].shape == (3, 2, 2)


def test_federation_merges_indexes_without_copying_source_arrays(tmp_path: Path) -> None:
    common_root, fmb_root = tmp_path / "common", tmp_path / "fmb"
    common_root.mkdir()
    (common_root / "episodes").mkdir()
    for name in ("episodes.jsonl", "actor_anchors_5hz.jsonl", "critic_intervals.jsonl"):
        (common_root / name).write_text("", encoding="utf-8")
    fmb_root.mkdir()
    _synthetic_fmb(fmb_root)

    corpus = FederatedDiverseCorpus(common_root, fmb_root)
    actor_rows = corpus.actor_anchors(source="fmb")
    critic_rows = corpus.critic_intervals(source="fmb")
    assert len(actor_rows) == 1 and actor_rows[0]["corpus_key"] == "fmb"
    assert len(critic_rows) == 1 and critic_rows[0]["corpus_key"] == "fmb"
    assert corpus.actor_sample(actor_rows[0], cameras=False)["views_copy_source_arrays"] is False

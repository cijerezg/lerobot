from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.diverse_pilot import (
    ACTION_OFFSETS_SECONDS,
    AdmissionError,
    SourceSpec,
    anchor_timestamps,
    annotation_window_eligibility,
    audit_source,
    fetch_metadata,
    generate_review_proxies,
    load_staged_lerobot_episode,
    loader_view,
    pack_numerical_episode,
    packed_feature_schema,
    resolve_lerobot_payload,
    safe_cleanup_payload,
    sample_action_chunk,
    validate_packed_v3_dataset,
    write_json,
    write_packed_v3_dataset,
)


def _spec(**overrides) -> SourceSpec:
    values = {
        "name": "test",
        "repo_id": "owner/test",
        "revision": "abc",
        "source_format": "lerobot_v3",
        "pilot_episodes": 2,
        "robot_type": "robot",
        "license": "apache-2.0",
        "real_robot_evidence": "real robot",
        "state_fields": ("observation.state",),
        "action_fields": ("action",),
        "gripper_field": "action[1]",
        "state_semantics": "native",
        "action_semantics": "native command",
        "gripper_semantics": "native",
        "joint_names": ("joint", "gripper"),
        "metadata_patterns": ("README.md", "meta/**"),
        "notes": "",
    }
    values.update(overrides)
    return SourceSpec(**values)


def test_metadata_fetch_rejects_payload_patterns(tmp_path: Path):
    spec = _spec(metadata_patterns=("meta/**", "videos/**"))
    with pytest.raises(ValueError, match="payload prefix"):
        fetch_metadata(spec, tmp_path)


def test_audit_blocks_source_without_commanded_joint_action(tmp_path: Path):
    spec = _spec(source_format="split_tar_archives", action_fields=(), license=None)
    audit = audit_source(spec, tmp_path)
    assert not audit["admission"]["passed"]
    assert "No usable joint-action field" in " ".join(audit["admission"]["failures"])


def test_audit_admits_explicit_copy_state_action_exception(tmp_path: Path):
    spec = _spec(
        source_format="split_tar_archives",
        action_fields=(),
        action_source="copy_state",
        license=None,
        usage_basis="public repository usage grant",
        action_semantics="measured state copied exactly into action",
    )
    audit = audit_source(spec, tmp_path)
    assert audit["admission"]["passed"]
    assert not audit["action"]["native_commanded_joint_action"]
    assert audit["action"]["copied_from_state"]
    assert audit["action"]["training_action_available"]
    assert audit["action"]["source"] == "copy_state"


def test_copy_state_loader_uses_exact_state_vector_as_action(tmp_path: Path):
    spec = _spec(action_fields=(), action_source="copy_state")
    metadata_root = tmp_path / "metadata"
    episode_meta = metadata_root / "meta/episodes/chunk-000/file-000.parquet"
    episode_meta.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "tasks": [["copy measured joints"]],
                "length": [3],
                "data/chunk_index": [0],
                "data/file_index": [0],
                "dataset_from_index": [0],
                "dataset_to_index": [3],
            }
        ),
        episode_meta,
    )
    staging_root = tmp_path / "staging"
    data_path = staging_root / "owner__test/data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    state = np.asarray([[1.0, 0.08], [1.5, 0.04], [2.0, 0.0]], dtype=np.float32)
    pq.write_table(
        pa.table(
            {
                "timestamp": np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
                "episode_index": np.zeros(3, dtype=np.int64),
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            }
        ),
        data_path,
    )
    manifest = {
        "repo_id": "owner/test",
        "episodes": [
            {"episode_index": 0, "files": {"data": "data/chunk-000/file-000.parquet"}}
        ],
    }
    episode = load_staged_lerobot_episode(spec, manifest, metadata_root, staging_root, 0)
    np.testing.assert_array_equal(episode.states, state)
    np.testing.assert_array_equal(episode.actions, state)
    assert not np.shares_memory(episode.actions, episode.states)


def test_resolver_refuses_failed_audit(tmp_path: Path):
    with pytest.raises(AdmissionError):
        resolve_lerobot_payload(
            _spec(),
            {"admission": {"passed": False, "failures": ["missing action"]}},
            tmp_path,
            [0],
        )


def test_15hz_interpolation_preserves_all_components_and_gripper():
    timestamps = np.arange(0, 10, 1 / 15, dtype=np.float64)
    values = np.stack([2 * timestamps, -timestamps, 0.25 + 0.05 * timestamps], axis=1).astype(np.float32)
    chunk = sample_action_chunk(timestamps, values, 6.0, native_rate_hz=15)
    expected_t = 6.0 + ACTION_OFFSETS_SECONDS
    np.testing.assert_allclose(chunk.values[:, 0], 2 * expected_t, atol=2e-6)
    np.testing.assert_allclose(chunk.values[:, 1], -expected_t, atol=2e-6)
    np.testing.assert_allclose(chunk.values[:, 2], 0.25 + 0.05 * expected_t, atol=2e-6)
    assert chunk.values.shape == (30, 3)
    assert chunk.interpolated_mask.any()


def test_30hz_uses_native_samples_exactly():
    timestamps = np.arange(0, 10, 1 / 30, dtype=np.float64)
    values = np.arange(len(timestamps) * 2, dtype=np.float32).reshape(-1, 2)
    chunk = sample_action_chunk(timestamps, values, 6.0, native_rate_hz=30)
    start = 180
    np.testing.assert_array_equal(chunk.values, values[start : start + 30])
    assert not chunk.interpolated_mask.any()


def test_duplicate_or_reversed_clock_is_rejected():
    timestamps = np.asarray([0.0, 0.1, 0.1, 0.2, 8.0])
    with pytest.raises(ValueError, match="duplicates"):
        anchor_timestamps(timestamps, 2.0)


def test_boundary_policy_keeps_only_complete_samples():
    timestamps = np.arange(0, 10 + 1 / 30, 1 / 30)
    anchors = anchor_timestamps(timestamps, 2.0)
    np.testing.assert_allclose(anchors, [6.0, 8.0])


def test_full_window_rejects_anchor_that_touches_bad_motion():
    annotations = {
        "segments": [
            {
                "start_s": 0.0,
                "end_s": 8.0,
                "retention": "keep",
                "retention_reason": "useful_motion",
            },
            {
                "start_s": 8.0,
                "end_s": 9.0,
                "retention": "reject",
                "retention_reason": "erratic",
            },
            {
                "start_s": 9.0,
                "end_s": 20.0,
                "retention": "keep",
                "retention_reason": "recovery",
            },
        ]
    }
    assert annotation_window_eligibility(annotations, 1.0, 7.999)["eligible"]
    assert not annotation_window_eligibility(annotations, 1.0, 8.0)["eligible"]
    decision = annotation_window_eligibility(annotations, 2.0, 8.5)
    assert not decision["eligible"]
    assert decision["rejection_reasons"] == ["erratic"]


def test_static_history_does_not_reject_later_active_target():
    annotations = {
        "segments": [
            {
                "start_s": 0.0,
                "end_s": 3.0,
                "retention": "keep",
                "retention_reason": "useful_motion",
            },
            {
                "start_s": 3.0,
                "end_s": 4.0,
                "retention": "reject",
                "retention_reason": "static",
            },
            {
                "start_s": 4.0,
                "end_s": 20.0,
                "retention": "keep",
                "retention_reason": "useful_motion",
            },
        ]
    }
    history_only = annotation_window_eligibility(annotations, 2.0, 8.0, action_start_s=6.0)
    assert history_only["eligible"]
    future_static = annotation_window_eligibility(annotations, 0.0, 5.0, action_start_s=3.5)
    assert not future_static["eligible"]
    assert future_static["rejection_reasons"] == ["static"]


def test_packed_episode_and_loader_contract():
    timestamps = np.arange(0, 11, 1 / 30, dtype=np.float64)
    state = np.stack([timestamps, timestamps + 1], axis=1).astype(np.float32)
    action = np.stack([timestamps, timestamps + 2, timestamps * 0 + 0.7], axis=1).astype(np.float32)
    samples = pack_numerical_episode(
        timestamps,
        state,
        timestamps,
        action,
        native_action_rate_hz=30,
        stride_s=2,
    )
    sample = samples[0]
    loaded = loader_view(
        {
            "observation.state_history": sample.observation_values[:6],
            "observation.state": sample.observation_values[6],
            "action": sample.action.values,
        }
    )
    assert loaded["observation.state"].shape == (7, 2)
    assert loaded["action"].shape == (30, 3)
    np.testing.assert_array_equal(loaded["action"][:, -1], np.full(30, 0.7, dtype=np.float32))


def test_packed_schema_keeps_matrix_dimensions():
    schema = packed_feature_schema(8, 8)
    assert schema["observation.state_history"]["shape"] == (6, 8)
    assert schema["action"]["shape"] == (30, 8)
    assert schema["source.action_source_timestamps"]["shape"] == (30, 2)
    assert schema["source.action_source_values"]["shape"] == (30, 2, 8)

    physical_double = packed_feature_schema(8, 8, state_dtype="float64", action_dtype="float64")
    assert physical_double["observation.state"]["dtype"] == "float64"
    assert physical_double["observation.state_history"]["dtype"] == "float64"
    assert physical_double["action"]["dtype"] == "float64"
    assert physical_double["source.action_source_values"]["dtype"] == "float64"


def test_contact_sheet_uses_episode_trimmed_proxy(tmp_path: Path, monkeypatch):
    staging_root = tmp_path / "staging"
    source = staging_root / "owner__test/videos/camera/chunk-000/file-000.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source shard")
    output_root = tmp_path / "review"
    calls = []

    def record_call(command, *, check):
        assert check
        calls.append(command)

    monkeypatch.setattr("lerobot.datasets.diverse_pilot.subprocess.run", record_call)
    manifest = {
        "repo_id": "owner/test",
        "episodes": [
            {
                "episode_index": 4,
                "files": {
                    "videos": {
                        "camera": {
                            "path": "videos/camera/chunk-000/file-000.mp4",
                            "from_timestamp": 10.0,
                            "to_timestamp": 20.0,
                        }
                    }
                },
            }
        ],
    }
    generate_review_proxies(manifest, staging_root, output_root)
    assert len(calls) == 2
    proxy = output_root / "episode_000004_camera.mp4"
    assert calls[1][calls[1].index("-i") + 1] == str(proxy)
    assert "-ss" not in calls[1]
    assert "-t" not in calls[1]


def test_cleanup_only_removes_manifest_listed_staging_files(tmp_path: Path):
    root = tmp_path / "owner__test"
    selected = root / "data/chunk-000/file-000.parquet"
    unselected = root / "data/chunk-000/file-001.parquet"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"selected")
    unselected.write_bytes(b"keep")
    manifest = {"repo_id": "owner/test", "files": ["data/chunk-000/file-000.parquet"]}
    removed = safe_cleanup_payload(manifest, tmp_path)
    assert removed == [selected]
    assert not selected.exists()
    assert unselected.exists()


def test_synthetic_packed_v3_writer_and_validator_round_trip(tmp_path: Path):
    spec = _spec()
    metadata_root = tmp_path / "metadata"
    episode_meta = metadata_root / "meta/episodes/chunk-000/file-000.parquet"
    episode_meta.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "tasks": [["synthetic native joint command"]],
                "length": [301],
                "data/chunk_index": [0],
                "data/file_index": [0],
                "dataset_from_index": [0],
                "dataset_to_index": [301],
            }
        ),
        episode_meta,
    )
    write_json(
        metadata_root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "fps": 30,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [2], "names": None},
                "action": {"dtype": "float32", "shape": [2], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        },
    )
    staging_root = tmp_path / "staging"
    data_path = staging_root / "owner__test/data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    timestamps = np.arange(301, dtype=np.float64) / 30
    state = np.stack([timestamps, timestamps + 1], axis=1).astype(np.float32)
    action = np.stack([timestamps * 2, timestamps * 0 + 0.37], axis=1).astype(np.float32)
    pq.write_table(
        pa.table(
            {
                "timestamp": timestamps.astype(np.float32),
                "episode_index": np.zeros(301, dtype=np.int64),
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            }
        ),
        data_path,
    )
    manifest = {
        "repo_id": "owner/test",
        "revision": "abc",
        "files": ["data/chunk-000/file-000.parquet"],
        "episodes": [
            {
                "episode_index": 0,
                "files": {"data": "data/chunk-000/file-000.parquet", "videos": {}},
            }
        ],
    }
    annotations_root = tmp_path / "annotations"
    write_json(
        annotations_root / "episode_000000.annotations.json",
        {
            "review_status": "validated",
            "segments": [
                {
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "retention": "keep",
                    "retention_reason": "useful_motion",
                    "subtask": "synthetic move",
                    "quality": 5,
                    "mistake_events": [],
                }
            ],
        },
    )
    output = tmp_path / "packed"
    report = write_packed_v3_dataset(
        spec,
        {"admission": {"passed": True}},
        manifest,
        metadata_root,
        staging_root,
        annotations_root,
        output,
        output_repo_id="local/synthetic-packed",
        stride_s=2,
        min_chunks_per_episode=2,
    )
    assert report["total_packed_samples"] == 2
    assert report["episodes"][0]["candidate_anchors"] == 2
    assert report["episodes"][0]["rejected_anchors"] == 0
    validation = validate_packed_v3_dataset(output, "local/synthetic-packed")
    assert validation["passed"]
    assert validation["rows"] == 2
    assert validation["action_dimension"] == 2

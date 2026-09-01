from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = ROOT / "examples/dataset/diverse_robot_dataset/prepare_fmb.py"
REVIEW_PATH = ROOT / "examples/dataset/diverse_robot_dataset/finalize_fmb_review.py"
ANNOTATE_PATH = ROOT / "examples/dataset/diverse_robot_dataset/annotate_fmb_critic.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_fmb = load_module("prepare_fmb", PREPARE_PATH)
finalize_fmb_review = load_module("finalize_fmb_review", REVIEW_PATH)
annotate_fmb_critic = load_module("annotate_fmb_critic", ANNOTATE_PATH)


def test_diverse_selection_covers_all_geometries_before_repeats() -> None:
    files = []
    for shape in map(str, range(1, 10)):
        for size in "SML":
            for length in "SL":
                for index, (angle, distractor) in enumerate(
                    (("horizontal", "n"), ("horizontal", "y"), ("vertical", "n"), ("vertical", "y"))
                ):
                    path = f"single_object_manipulation_dataset/{shape}_{size}_{length}_1_{angle}_{distractor}_{index}.npy"
                    files.append(
                        {
                            "path": path,
                            "bytes": 1,
                            "sha256": hashlib.sha256(path.encode()).hexdigest(),
                            "object": {
                                "shape": shape,
                                "size": size,
                                "length": length,
                                "color": "1",
                                "angle": angle,
                                "distractor": distractor,
                                "trajectory": str(index),
                            },
                        }
                    )

    selected = prepare_fmb.choose_diverse(files, 100, set())
    geometry_counts = {}
    for item in selected:
        key = prepare_fmb.geometry_key(item)
        geometry_counts[key] = geometry_counts.get(key, 0) + 1

    assert len(selected) == 100
    assert len(geometry_counts) == 54
    assert sorted(geometry_counts.values()) == [1] * 8 + [2] * 46
    assert len({item["path"] for item in selected}) == 100


def test_actor_accounting_matches_audited_window_contract() -> None:
    result = prepare_fmb.actor_accounting(["grasp"] * 80)

    assert result["original_2s"]["episode_level_candidate_anchors"] == 1
    assert result["proposed_5hz"]["episode_level_candidate_anchors"] == 5
    assert result["proposed_10hz"]["episode_level_candidate_anchors"] == 10
    assert result["proposed_10hz"]["primitive_conditioned_future_candidate_anchors"] == 10


def test_critic_intervals_reference_every_native_action() -> None:
    labels = ["grasp"] * 3 + ["insert"] * 4
    intervals = prepare_fmb.primitive_intervals(labels)

    assert [(item["start_timestep"], item["end_timestep_exclusive"]) for item in intervals] == [
        (0, 3),
        (3, 7),
    ]
    assert sum(item["native_action_samples"] for item in intervals) == len(labels)


def test_review_rejects_ineligible_interval_without_reason() -> None:
    candidate = {"primitive": "grasp", "start_timestep": 0, "end_timestep_exclusive": 3}
    decision = {
        **candidate,
        "classification": "unclear",
        "pause_assessment": "unknown",
        "retry_assessment": "unknown",
        "interruption_assessment": "unknown",
        "mistake_assessment": "unknown",
        "recovery_assessment": "unknown",
        "critic_eligible": False,
        "critic_rejection_reason": None,
        "quality": None,
        "quality_provenance": "none",
    }

    try:
        finalize_fmb_review.validate_decision(candidate, decision)
    except ValueError as exc:
        assert "explicit reason" in str(exc)
    else:
        raise AssertionError("Expected missing rejection reason to fail validation")


def synthetic_source(frame_count: int) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for camera in prepare_fmb.CAMERAS:
        source[f"obs/{camera}"] = np.zeros((frame_count, 256, 256, 3), dtype=np.uint8)
        source[f"obs/{camera}_depth"] = np.full(
            (frame_count, 256, 256), 2500, dtype=np.uint16
        )
    source.update(
        {
            "obs/tcp_pose": np.zeros((frame_count, 7), dtype=np.float64),
            "obs/tcp_vel": np.zeros((frame_count, 6), dtype=np.float64),
            "obs/tcp_force": np.zeros((frame_count, 3), dtype=np.float64),
            "obs/tcp_torque": np.zeros((frame_count, 3), dtype=np.float64),
            "obs/q": np.zeros((frame_count, 7), dtype=np.float64),
            "obs/dq": np.zeros((frame_count, 7), dtype=np.float64),
            "obs/jacobian": np.zeros((frame_count, 6, 7), dtype=np.float64),
            "obs/gripper_pose": np.zeros(frame_count, dtype=np.float64),
            "actions": np.zeros((frame_count, 7), dtype=np.float64),
            "primitive": np.asarray(["grasp"] * frame_count),
        }
    )
    source["obs/wrist_1_depth"][0, 0, 0] = 0
    source["obs/wrist_1_depth"][0, 0, 1] = 65535
    return source


def test_converter_retains_only_approved_modalities_and_validates(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    source_path = "single_object_manipulation_dataset/1_S_S_1_horizontal_n_0.npy"
    raw_path = raw_root / source_path
    raw_path.parent.mkdir(parents=True)
    np.save(raw_path, synthetic_source(2), allow_pickle=True)
    file_spec = {
        "path": source_path,
        "bytes": raw_path.stat().st_size,
        "sha256": prepare_fmb.sha256(raw_path),
        "object": prepare_fmb.parse_source_path(source_path),
        "split": "train",
        "pilot_reviewed": False,
    }
    manifest = {
        "source_repo_id": prepare_fmb.SOURCE_REPO_ID,
        "source_revision": prepare_fmb.SOURCE_REVISION,
        "files": [file_spec],
        "conversion_policy": {
            "keep_rgb": list(prepare_fmb.KEEP_RGB),
            "keep_depth": list(prepare_fmb.KEEP_DEPTH),
            "omit_rgb": list(prepare_fmb.OMIT_RGB),
            "omit_depth": list(prepare_fmb.OMIT_DEPTH),
            "depth_units_mm_per_level": prepare_fmb.DEPTH_MM_PER_LEVEL,
            "depth_scale_provenance": prepare_fmb.DEPTH_SCALE_PROVENANCE,
        },
    }
    pilot_audit = tmp_path / "pilot_audit.json"
    pilot_audit.write_text(
        json.dumps({"acceptance_gate": {"status": "passed", "full_converter_allowed": True}})
    )
    output_root = tmp_path / "corpus"
    output_root.mkdir()
    prepare_fmb.write_json(output_root / "source_manifest.json", manifest)

    converted = prepare_fmb.convert(manifest, raw_root, output_root, pilot_audit, None)
    prepare_fmb.write_json(output_root / "corpus.json", converted["corpus"])
    report = prepare_fmb.validate(manifest, raw_root, output_root)

    assert report["status"] == "passed"
    episode_dir = next((output_root / "episodes").iterdir())
    names = {path.name for path in episode_dir.iterdir()}
    assert "side_1_rgb.npy" in names
    assert "side_2_rgb.npy" in names
    assert "wrist_1_rgb.npy" in names
    assert "wrist_1_depth_z16.npy" in names
    assert "wrist_2_rgb.npy" not in names
    assert "side_1_depth.npy" not in names
    critic = prepare_fmb.read_jsonl(output_root / "critic_intervals.jsonl")
    assert critic[0]["critic_eligible"] is False
    assert critic[0]["critic_rejection_reason"] == "pending_episode_visual_review"


def test_rebot_mistake_event_validates_inside_native_subtask() -> None:
    interval = {"start_timestep": 10, "end_timestep_exclusive": 30}
    event = {
        "start_timestep": 15,
        "end_timestep_exclusive": 20,
        "mistake": True,
        "mistake_type": "failed_close",
        "note": "Visible close on empty space.",
    }
    annotate_fmb_critic.validate_mistake(interval, event)
    assert event["mistake"] is True


def test_rebot_mistake_span_must_stay_inside_native_subtask() -> None:
    interval = {"start_timestep": 10, "end_timestep_exclusive": 30}
    event = {
        "start_timestep": 25,
        "end_timestep_exclusive": 31,
        "mistake": True,
        "mistake_type": "slip",
        "note": "Object visibly escapes the gripper.",
    }

    try:
        annotate_fmb_critic.validate_mistake(interval, event)
    except ValueError as exc:
        assert "crosses" in str(exc)
    else:
        raise AssertionError("Expected a cross-boundary mistake event to fail validation")

#!/usr/bin/env python

"""Convert selected RoboChallenge episodes to LeRobot v3.

RoboChallenge does not expose a native commanded joint action. For this one
source, the measured joint-position and gripper-width vector is copied exactly
into the action fields at the same timestamp. Sources with native actions use
their native fields and do not pass through this converter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.diverse_pilot import SourceSpec, resolve_lerobot_payload, write_json
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = tuple(f"joint_{index}" for index in range(6))
SOURCE_REPO_ID = "RoboChallenge/Table30v2"
SOURCE_REVISION = "c58ad2cc76ce722ea54de51f9fae03012a698f47"
KNOWN_CAMERAS = {
    "observation.images.global": "cam_global_rgb.mp4",
    "observation.images.wrist": "cam_arm_rgb.mp4",
    "observation.images.side": "cam_side_rgb.mp4",
}


def _episode_cameras(episode_root: Path) -> dict[str, str]:
    cameras = {
        key: filename
        for key, filename in KNOWN_CAMERAS.items()
        if (episode_root / "videos" / filename).is_file()
    }
    if not cameras:
        raise FileNotFoundError(f"No supported RGB cameras in {episode_root}")
    return cameras


def _load_states(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _joint_gripper_vector(record: dict) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(record["joint_positions"], dtype=np.float32)
    gripper = np.asarray([record["gripper_width"]], dtype=np.float32)
    if joints.shape != (6,):
        raise ValueError(f"Expected six single-arm joints, got {joints.shape}")
    if not np.isfinite(joints).all() or not np.isfinite(gripper).all():
        raise ValueError("Joint or gripper state contains NaN or infinity")
    return joints, gripper


def _open_videos(
    episode_root: Path, expected_frames: int, cameras: dict[str, str]
) -> tuple[dict[str, cv2.VideoCapture], int, int]:
    captures = {}
    dimensions = set()
    try:
        for key, filename in cameras.items():
            capture = cv2.VideoCapture(str(episode_root / "videos" / filename))
            if not capture.isOpened():
                raise FileNotFoundError(episode_root / "videos" / filename)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count != expected_frames:
                raise ValueError(
                    f"{episode_root.name} {key} has {frame_count} frames; expected {expected_frames}"
                )
            dimensions.add(
                (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            )
            captures[key] = capture
        if len(dimensions) != 1:
            raise ValueError(f"Camera dimensions disagree in {episode_root}: {sorted(dimensions)}")
        height, width = dimensions.pop()
        return captures, height, width
    except Exception:
        for capture in captures.values():
            capture.release()
        raise


def _features(height: int, width: int, cameras: dict[str, str]) -> dict:
    features = {
        key: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
        for key in cameras
    }
    features.update(
        {
            "observation.joint_position": {
                "dtype": "float32",
                "shape": (6,),
                "names": list(JOINT_NAMES),
            },
            "observation.gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_width"],
            },
            "action.joint_position": {
                "dtype": "float32",
                "shape": (6,),
                "names": list(JOINT_NAMES),
            },
            "action.gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_width"],
            },
            "source.timestamp": {"dtype": "float64", "shape": (1,), "names": None},
        }
    )
    return features


def write_conversion_metadata(
    raw_task_root: Path,
    dataset_root: Path,
    repo_id: str,
    robot_type: str,
    episode_indices: list[int],
    source_revision: str = SOURCE_REVISION,
) -> None:
    conversion = {
        "source_repo_id": SOURCE_REPO_ID,
        "source_revision": source_revision,
        "converted_repo_id": repo_id,
        "source_task": raw_task_root.name,
        "action_source": "copy_state",
        "action_semantics": "Exact same-timestamp copy of measured joint positions and gripper width.",
        "episodes": [
            {"episode_index": index, "source_episode_index": source_index}
            for index, source_index in enumerate(episode_indices)
        ],
    }
    write_json(dataset_root / "meta" / "robochallenge_conversion.json", conversion)
    adapter_spec = SourceSpec(
        name="robochallenge_raw",
        repo_id=repo_id,
        revision=source_revision,
        source_format="lerobot_v3",
        pilot_episodes=len(episode_indices),
        robot_type=robot_type,
        license=None,
        real_robot_evidence="RoboChallenge real-robot task archive.",
        state_fields=("observation.joint_position", "observation.gripper_position"),
        action_fields=(),
        gripper_field="observation.gripper_position",
        state_semantics="Measured joint positions and gripper width.",
        action_semantics="Exact same-timestamp copy of the measured state vector.",
        gripper_semantics="Measured gripper width copied unchanged.",
        joint_names=(*JOINT_NAMES, "gripper_width"),
        metadata_patterns=(),
        action_source="copy_state",
        usage_basis="Public repository access under the Hugging Face Terms of Service.",
    )
    adapter_audit = {
        "admission": {"passed": True, "failures": []},
        "repository": {"resolved_revision": source_revision},
    }
    manifest = resolve_lerobot_payload(
        adapter_spec,
        adapter_audit,
        dataset_root,
        list(range(len(episode_indices))),
    )
    manifest["source_format"] = "lerobot_v3"
    manifest["source_repo_id"] = SOURCE_REPO_ID
    for episode, source_episode_index in zip(manifest["episodes"], episode_indices, strict=True):
        episode["source_episode_index"] = source_episode_index
    write_json(dataset_root / "meta" / "local_acquisition_manifest.json", manifest)


def convert_task(
    raw_task_root: Path,
    dataset_root: Path,
    repo_id: str,
    episode_indices: list[int],
    source_revision: str = SOURCE_REVISION,
) -> None:
    if dataset_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset root: {dataset_root}")
    task_info = json.loads((raw_task_root / "meta" / "task_info.json").read_text(encoding="utf-8"))
    fps = int(task_info["video_info"]["fps"])
    prompt = task_info["task_desc"]["prompt"]
    robot_type = task_info["task_desc"]["task_tag"][-1]

    first_episode = raw_task_root / "data" / f"episode_{episode_indices[0]:06d}"
    cameras = _episode_cameras(first_episode)
    first_states = _load_states(first_episode / "states" / "states.jsonl")
    probe_captures, height, width = _open_videos(first_episode, len(first_states), cameras)
    for capture in probe_captures.values():
        capture.release()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=robot_type,
        fps=fps,
        features=_features(height, width, cameras),
        use_videos=True,
        image_writer_threads=4,
        camera_encoder=VideoEncoderConfig(vcodec="h264", crf=23, preset="veryfast", g=fps),
    )
    try:
        for episode_index in episode_indices:
            episode_root = raw_task_root / "data" / f"episode_{episode_index:06d}"
            states = _load_states(episode_root / "states" / "states.jsonl")
            episode_cameras = _episode_cameras(episode_root)
            if episode_cameras != cameras:
                raise ValueError(f"Camera set changes in {episode_root}")
            captures, episode_height, episode_width = _open_videos(
                episode_root, len(states), cameras
            )
            if (episode_height, episode_width) != (height, width):
                raise ValueError(f"Camera dimensions change in {episode_root}")
            try:
                timestamps = np.asarray([record["timestamp"] for record in states], dtype=np.float64)
                if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
                    raise ValueError(f"Invalid timestamp clock in {episode_root}")
                for record, source_timestamp in zip(states, timestamps, strict=True):
                    joints, gripper = _joint_gripper_vector(record)
                    frame = {
                        "observation.joint_position": joints,
                        "observation.gripper_position": gripper,
                        "action.joint_position": joints.copy(),
                        "action.gripper_position": gripper.copy(),
                        "source.timestamp": np.asarray([source_timestamp], dtype=np.float64),
                        "task": prompt,
                    }
                    for key, capture in captures.items():
                        ok, image = capture.read()
                        if not ok:
                            raise ValueError(f"Failed to read {key} for {episode_root.name}")
                        frame[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    dataset.add_frame(frame)
                dataset.save_episode()
            finally:
                for capture in captures.values():
                    capture.release()
        dataset.finalize()
        write_conversion_metadata(
            raw_task_root,
            dataset_root,
            repo_id,
            robot_type,
            episode_indices,
            source_revision,
        )
    except Exception:
        dataset.finalize()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-task-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    args = parser.parse_args()
    convert_task(
        args.raw_task_root,
        args.dataset_root,
        args.repo_id,
        args.episodes,
        source_revision=args.source_revision,
    )


if __name__ == "__main__":
    main()

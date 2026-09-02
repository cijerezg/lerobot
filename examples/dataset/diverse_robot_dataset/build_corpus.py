#!/usr/bin/env python

"""Build the unified source-native corpus behind the actor and critic views.

The validated packed components hold one sparse anchor every two seconds: seven
observations and a one-second future. That representation cannot answer a
different anchor stride and cannot show a critic a whole subtask, because the
continuous action timeline between anchors was never stored.

This builder writes the missing layer. For every visually accepted episode it
stores the continuous native timeline once - timestamps, state, actions, the
per-camera video at the native rate, the reviewed annotations, and the source
provenance - so the actor and critic views are two readings of one corpus rather
than two frozen datasets. Nothing here is a new recording: it re-expresses
episodes that were already acquired, reviewed, and validated.

Subcommands:
  ingest    convert one source's accepted episodes into corpus episodes
  views     write the actor anchor view and the critic interval view
  validate  check arrays, video alignment, annotations, splits, and accounting
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "lerobot/src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "lerobot/src"))

from lerobot.datasets.diverse_pilot import (  # noqa: E402
    SourceSpec,
    annotation_window_eligibility,
    load_source_specs,
    load_staged_lerobot_episode,
)

CORPUS_FORMAT = "diverse_robot_source_native_actor_critic_corpus_v1"
BUILDER_VERSION = 1
DATASET_ROOT = REPO_ROOT / "outputs/diverse_robot_dataset"
BUILD_ROOT = REPO_ROOT / "outputs/diverse_robot_dataset_build"
CORPUS_ROOT = DATASET_ROOT / "corpus"
CONFIG_DIR = Path(__file__).parent

HISTORY_S = 6.0
FUTURE_POINTS = 30
FUTURE_FPS = 30.0
FUTURE_END_S = (FUTURE_POINTS - 1) / FUTURE_FPS
VIDEO_CRF = 18
VIDEO_PRESET = "fast"
READ_CHUNK = 8 * 1024 * 1024

# Reject reasons that end a critic subtask rather than pausing inside it.
PAUSE_REASONS = {"static", "no_op", "stationary"}
# A publisher keep-range boundary is not behaviour: it says the clip stops there, not that
# something interrupted the robot. Quality and off-task rejections are reviewer judgements.
SOURCE_RANGE_REASONS = {"source_excluded"}
QUALITY_REASONS = {"low_quality"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    source: str
    component: str
    embodiment: str
    dataset_root: Path
    staging_root: Path
    metadata_root: Path
    spec: SourceSpec
    video_origin: str
    raw_task_root: Path | None = None
    # A later review round has no packed component of its own: its manifest, selection, and
    # annotations live beside the round's review material instead.
    manifest_path: Path | None = None
    selection_path: Path | None = None
    annotations_root: Path | None = None
    review_round: str = "round1"

    @property
    def manifest(self) -> dict[str, Any]:
        return read_json(self.manifest_path or self.dataset_root / "meta/acquisition_manifest.json")

    @property
    def selection(self) -> dict[str, Any]:
        return read_json(self.selection_path or self.dataset_root / "meta/source_selection.json")

    def annotations_path(self, episode_index: int) -> Path:
        root = self.annotations_root or self.dataset_root / "meta/annotations"
        return root / f"episode_{episode_index:06d}.annotations.json"


def _spec_from_config(config_name: str, spec_name: str) -> SourceSpec:
    for spec in load_source_specs(CONFIG_DIR / config_name):
        if spec.name == spec_name:
            return spec
    raise KeyError(f"{spec_name!r} is absent from {config_name}")


def _robochallenge_spec(dataset_root: Path, embodiment: str) -> SourceSpec:
    """RoboChallenge has no source config file; its spec lives in the packed component."""
    packed = read_json(dataset_root / "meta/packed_extension.json")["source_spec"]
    return SourceSpec(
        name=f"robochallenge_{embodiment.lower()}",
        repo_id=packed["repo_id"],
        revision=packed["revision"],
        source_format="lerobot_v3",
        pilot_episodes=0,
        robot_type=packed["robot_type"],
        license="cc-by-4.0",
        real_robot_evidence="RoboChallenge Table30v2 real teleoperated tabletop recordings.",
        state_fields=tuple(packed["state_fields"]),
        action_fields=tuple(packed["action_fields"]),
        gripper_field=packed["gripper_field"],
        state_semantics="Six measured joint positions followed by the measured gripper width.",
        action_semantics=(
            "RoboChallenge exposes no commanded action; the measured state is copied at the same "
            "timestamp under the audited copy_state exception."
        ),
        gripper_semantics="Measured gripper width in metres, copied into the action vector.",
        joint_names=(),
        metadata_patterns=(),
        action_source=packed["action_source"],
    )


def discover_components(source: str) -> list[Component]:
    source_root = BUILD_ROOT / source
    index = read_json(source_root / "index.json")
    components: list[Component] = []
    for entry in index["components"]:
        if entry.get("status") != "validated":
            continue
        dataset_root = source_root / entry["dataset_root"]
        embodiment = str(entry["embodiment"])
        if source == "robochallenge":
            task = str(entry["task"])
            spec = _robochallenge_spec(dataset_root, embodiment)
            # The converted staging directory name is authoritative in the acquisition
            # manifest; two task slugs do not match the task name character for character.
            staged_repo_id = read_json(dataset_root / "meta/acquisition_manifest.json")["repo_id"]
            components.append(
                Component(
                    source=source,
                    component=task,
                    embodiment=embodiment,
                    dataset_root=dataset_root,
                    staging_root=source_root / "staging",
                    metadata_root=source_root / "staging" / staged_repo_id.replace("/", "__"),
                    spec=spec,
                    video_origin="robochallenge_raw",
                    raw_task_root=source_root / "staging/raw" / task / task,
                )
            )
            continue
        spec_name = {"droid": "droid_failure", "droid_success": "droid_success", "ur7e": "ur7e_stack"}[source]
        config_name = {"droid": "droid_sources.json", "droid_success": "droid_sources.json"}.get(
            source, "ur7e_sources.json"
        )
        spec = _spec_from_config(config_name, spec_name)
        metadata_name = {"ur7e": "ur7e_stack"}.get(source, spec_name)
        components.append(
            Component(
                source=source,
                component=str(entry.get("component") or entry.get("task")),
                embodiment=embodiment,
                dataset_root=dataset_root,
                staging_root=source_root / "staging",
                metadata_root=source_root / "metadata" / metadata_name,
                spec=spec,
                video_origin="staged_v3",
            )
        )
    return components


# --------------------------------------------------------------------------------------
# Episode ingest
# --------------------------------------------------------------------------------------


def episode_id(component: Component, episode_index: int) -> str:
    return f"{component.source}__{component.component}__ep{episode_index:06d}"


def probe_video(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    """Probe one video. Counting frames decodes the whole file, so only ask when needed."""
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    entries = "stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt"
    if count_frames:
        command.append("-count_frames")
        entries += ",nb_read_frames"
    command += ["-show_entries", entries, "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    if frames == 0 and not count_frames:
        return probe_video(path, count_frames=True)
    numerator, _, denominator = str(stream["r_frame_rate"]).partition("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator or 1),
        "frames": frames,
        "codec": str(stream["codec_name"]),
        "pix_fmt": str(stream["pix_fmt"]),
        "frame_count_source": "decoded" if count_frames else "container",
    }


def encode_episode_video(
    source_path: Path, destination: Path, *, start_s: float, frames: int, fps: float
) -> dict[str, Any]:
    """Re-encode one episode's segment with a one-second GOP for fast anchor decoding."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    gop = max(1, int(round(fps)))
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.6f}"]
    command += [
        "-i",
        str(source_path),
        "-frames:v",
        str(frames),
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-crf",
        str(VIDEO_CRF),
        "-preset",
        VIDEO_PRESET,
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-x264-params",
        "scenecut=0:open_gop=0",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)
    probe = probe_video(destination)
    if probe["frames"] != frames:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"{destination.name}: encoded {probe['frames']} frames, expected {frames} "
            f"from {source_path} at {start_s:.6f}s"
        )
    return {
        "ffmpeg_command": command,
        "encoded": probe,
        "gop_frames": gop,
        "crf": VIDEO_CRF,
        "preset": VIDEO_PRESET,
        "source_start_s": start_s,
    }


def _camera_sources(
    component: Component, episode_index: int, episode_manifest: dict[str, Any], frames: int
) -> dict[str, dict[str, Any]]:
    """Resolve one physical camera stream per episode with its trim window."""
    sources: dict[str, dict[str, Any]] = {}
    raw_videos = (
        component.raw_task_root / "data" / f"episode_{episode_index:06d}" / "videos"
        if component.raw_task_root is not None
        else None
    )
    # Three RoboChallenge tasks had their raw extraction cleaned up after conversion. Their
    # converted staging shard is the remaining copy, so fall back to it rather than refusing.
    if component.video_origin == "robochallenge_raw" and raw_videos is not None and raw_videos.is_dir():
        mapping = {
            "observation.images.global": "cam_global_rgb.mp4",
            "observation.images.wrist": "cam_arm_rgb.mp4",
            "observation.images.side": "cam_side_rgb.mp4",
        }
        for camera in episode_manifest["files"]["videos"]:
            path = raw_videos / mapping[camera]
            if not path.is_file():
                raise FileNotFoundError(f"Missing raw camera file: {path}")
            probe = probe_video(path)
            if probe["frames"] != frames:
                raise ValueError(f"{path}: raw video holds {probe['frames']} frames, timeline holds {frames}")
            sources[camera.rsplit(".", 1)[-1]] = {
                "path": path,
                "start_s": 0.0,
                "origin": "robochallenge_raw_episode_file",
                "source_relative": str(path.relative_to(BUILD_ROOT)),
            }
        return sources
    staged_root = component.staging_root / component.manifest["repo_id"].replace("/", "__")
    for camera, video in episode_manifest["files"]["videos"].items():
        path = staged_root / video["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing staged camera shard: {path}")
        sources[camera.rsplit(".", 1)[-1]] = {
            "path": path,
            "start_s": float(video["from_timestamp"]),
            "origin": "staged_lerobot_v3_shard",
            "source_relative": str(path.relative_to(BUILD_ROOT)),
        }
    return sources


def _accepted_episodes(component: Component) -> list[int]:
    selection = component.selection
    if "accepted_episode_indices" in selection:
        return sorted(int(index) for index in selection["accepted_episode_indices"])
    return sorted(int(index) for index in selection["accepted"])


def _accepted_episode_detail(selection: dict[str, Any], episode_index: int) -> dict[str, Any]:
    """Return optional legacy per-episode metadata from an accepted mapping.

    Current RoboChallenge selections store ``accepted`` as a list and keep the
    review metadata at the selection/annotation level. Older selections may use
    an episode-indexed mapping instead.
    """
    accepted = selection.get("accepted", {})
    if not isinstance(accepted, dict):
        return {}
    detail = accepted.get(str(episode_index), {})
    return detail if isinstance(detail, dict) else {}


def assign_splits(episode_indices: list[int]) -> dict[int, str]:
    """Episode-level 80/10/10 with at least one validation and test episode per component."""
    ordered = sorted(episode_indices)
    splits = {}
    for position, index in enumerate(ordered):
        remainder = position % 10
        splits[index] = "validation" if remainder == 8 else "test" if remainder == 9 else "train"
    if len(ordered) >= 4:
        if "validation" not in splits.values():
            splits[ordered[-2]] = "validation"
        if "test" not in splits.values():
            splits[ordered[-1]] = "test"
    return splits


def ingest_episode(
    component: Component,
    episode_index: int,
    split: str,
    corpus_root: Path,
    *,
    hash_cache: dict[Path, str],
    overwrite: bool,
) -> dict[str, Any]:
    identifier = episode_id(component, episode_index)
    destination = corpus_root / "episodes" / identifier
    record_path = destination / "episode.json"
    if record_path.is_file() and not overwrite:
        return read_json(record_path)

    manifest = component.manifest
    # RoboChallenge staging re-indexes the accepted episodes; annotations and this corpus
    # stay keyed on the source episode index, so map to the staged index before reading.
    staged_index = episode_index
    for item in manifest["episodes"]:
        if int(item.get("source_episode_index", item["episode_index"])) == episode_index:
            staged_index = int(item["episode_index"])
            break
    arrays = load_staged_lerobot_episode(
        component.spec, manifest, component.metadata_root, component.staging_root, staged_index
    )
    timestamps = np.asarray(arrays.timestamps, dtype=np.float64)
    relative = timestamps - timestamps[0]
    frames = int(len(relative))
    duration_s = float(relative[-1])
    intervals = np.diff(relative)
    measured_rate_hz = float(1.0 / np.median(intervals))
    # Use the source's declared rate, not the measured one. Float32 timestamps quantize a
    # 30 Hz stream to 30.0000286 Hz, and the packer's exact-native branch keys off an exact
    # 30.0, so a measured rate would silently interpolate the audited copy_state actions.
    native_rate_hz = declared_rate_hz(component, measured_rate_hz)
    if abs(native_rate_hz - measured_rate_hz) / native_rate_hz > 1e-3:
        raise ValueError(
            f"{episode_id(component, episode_index)}: declared {native_rate_hz} Hz disagrees "
            f"with the measured {measured_rate_hz} Hz"
        )

    annotations = read_json(component.annotations_path(episode_index))
    if annotations.get("review_status") != "validated":
        raise ValueError(f"{identifier}: annotations are not validated")

    episode_manifest = next(
        item for item in manifest["episodes"] if int(item["episode_index"]) == staged_index
    )
    camera_sources = _camera_sources(component, episode_index, episode_manifest, frames)

    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "timestamp_s.npy", relative)
    np.save(destination / "source_timestamp_s.npy", timestamps)
    np.save(destination / "state.npy", np.asarray(arrays.states, dtype=np.float64))
    np.save(destination / "action.npy", np.asarray(arrays.actions, dtype=np.float64))

    cameras = []
    for name, source in sorted(camera_sources.items()):
        video_path = destination / "videos" / f"{name}.mp4"
        encoding = encode_episode_video(
            source["path"],
            video_path,
            start_s=float(source["start_s"]),
            frames=frames,
            fps=native_rate_hz,
        )
        source_path = source["path"]
        if source_path not in hash_cache:
            hash_cache[source_path] = sha256(source_path)
        cameras.append(
            {
                "name": name,
                "path": f"videos/{name}.mp4",
                "frames": encoding["encoded"]["frames"],
                "width": encoding["encoded"]["width"],
                "height": encoding["encoded"]["height"],
                "fps": encoding["encoded"]["fps"],
                "codec": encoding["encoded"]["codec"],
                "sha256": sha256(video_path),
                "bytes": video_path.stat().st_size,
                "source": {
                    "origin": source["origin"],
                    "path": source["source_relative"],
                    "sha256": hash_cache[source_path],
                    "start_s": float(source["start_s"]),
                },
                "encoding": {key: encoding[key] for key in ("ffmpeg_command", "gop_frames", "crf", "preset")},
            }
        )

    selection = component.selection
    accepted_detail = _accepted_episode_detail(selection, episode_index)
    keep_subtasks = [
        str(segment["subtask"]) for segment in annotations["segments"] if segment.get("retention") == "keep"
    ]
    task = (
        annotations.get("task") or accepted_detail.get("task") or (keep_subtasks[0] if keep_subtasks else "")
    )
    record = {
        "episode_id": identifier,
        "corpus_format": CORPUS_FORMAT,
        "builder_version": BUILDER_VERSION,
        "source": component.source,
        "component": component.component,
        "embodiment": component.embodiment,
        "split": split,
        "task": task,
        "source_repo_id": component.spec.repo_id,
        "source_revision": manifest.get("revision", component.spec.revision),
        "source_episode_index": episode_index,
        "staged_episode_index": staged_index,
        "review_round": component.review_round,
        "frames": frames,
        "duration_s": duration_s,
        "native_rate_hz": native_rate_hz,
        "measured_rate_hz": measured_rate_hz,
        "rate_provenance": "declared_by_the_source_dataset_info",
        "timestamp_provenance": "source native per-episode timestamps, rebased to a zero start",
        "state_dimension": int(arrays.states.shape[1]),
        "action_dimension": int(arrays.actions.shape[1]),
        "state_semantics": component.spec.state_semantics,
        "action_semantics": component.spec.action_semantics,
        "gripper_semantics": component.spec.gripper_semantics,
        "action_source": component.spec.action_source,
        "state_fields": list(component.spec.state_fields),
        "action_fields": list(component.spec.action_fields),
        "robot_type": component.spec.robot_type,
        "joint_names": list(component.spec.joint_names),
        "arrays": {
            "timestamp_s": {"path": "timestamp_s.npy", "shape": [frames]},
            "source_timestamp_s": {"path": "source_timestamp_s.npy", "shape": [frames]},
            "state": {"path": "state.npy", "shape": [frames, int(arrays.states.shape[1])]},
            "action": {"path": "action.npy", "shape": [frames, int(arrays.actions.shape[1])]},
        },
        "cameras": cameras,
        "annotations": annotations,
        "review": {
            "outcome": annotations.get("outcome", accepted_detail.get("outcome", "unknown")),
            "notes": annotations.get("reviewer_notes", accepted_detail.get("notes", "")),
            "uuid": accepted_detail.get("uuid"),
            "scene_family": accepted_detail.get("scene_family"),
            "quality_provenance": annotations.get("quality_provenance")
            or (
                "human_reviewed"
                if component.source != "robochallenge"
                else "source_derived_automatic"
            ),
            "quality_mistake_review_provenance": annotations.get(
                "quality_mistake_review_provenance"
            ),
            # Who made the accept/reject call. The annotation wins over the selection so a
            # per-episode record cannot be silently upgraded by a task-level default, and a
            # model screen stays distinguishable from the human-reviewed DROID annotations.
            "review_provenance": (
                annotations.get("review_provenance")
                or selection.get("review_provenance")
                or "visual_review_of_production_component"
            ),
            "reviewer_model": annotations.get("reviewer_model") or selection.get("reviewer_model"),
            "review_date": annotations.get("review_date") or selection.get("review_date"),
            "review_prompt": annotations.get("review_prompt") or selection.get("review_prompt"),
            "selection_rule": selection.get("selection_rule"),
        },
        "source_tasks": list(arrays.tasks),
    }
    write_json(record_path, record)
    return record


def ingest(source: str, corpus_root: Path, *, components: list[str] | None, overwrite: bool) -> None:
    hash_cache: dict[Path, str] = {}
    records = []
    for component in discover_components(source):
        if components and component.component not in components:
            continue
        accepted = _accepted_episodes(component)
        splits = assign_splits(accepted)
        for episode_index in accepted:
            record = ingest_episode(
                component,
                episode_index,
                splits[episode_index],
                corpus_root,
                hash_cache=hash_cache,
                overwrite=overwrite,
            )
            records.append(record)
            print(
                f"{record['episode_id']:52s} frames={record['frames']:5d} "
                f"{record['duration_s']:7.2f}s cameras={len(record['cameras'])} "
                f"split={record['split']}",
                flush=True,
            )
    refresh_episode_index(corpus_root)
    print(f"ingested {len(records)} episodes from {source}", flush=True)


def declared_rate_hz(component: Component, fallback: float) -> float:
    """The source dataset's declared fps, which is the rate the packed components used."""
    info_path = component.metadata_root / "meta/info.json"
    if info_path.is_file():
        declared = read_json(info_path).get("fps")
        if declared:
            return float(declared)
    return fallback


def refresh_metadata(corpus_root: Path) -> int:
    """Re-derive declared rates for episodes ingested before the rate provenance was fixed."""
    declared: dict[tuple[str, str], float] = {}
    for source in ("robochallenge", "droid", "droid_success", "ur7e"):
        if not (BUILD_ROOT / source / "index.json").is_file():
            continue
        for component in discover_components(source):
            rate = declared_rate_hz(component, 0.0)
            if rate:
                declared[(component.source, component.component)] = rate
    updated = 0
    for record_path in sorted((corpus_root / "episodes").glob("*/episode.json")):
        record = read_json(record_path)
        rate = declared.get((record["source"], record["component"]))
        if not rate or record.get("rate_provenance") == "declared_by_the_source_dataset_info":
            continue
        record["measured_rate_hz"] = record["native_rate_hz"]
        record["native_rate_hz"] = rate
        record["rate_provenance"] = "declared_by_the_source_dataset_info"
        write_json(record_path, record)
        updated += 1
    refresh_episode_index(corpus_root)
    return updated


def round_component(source: str, task: str, review_round: int) -> Component:
    """A later review round for an existing task: new episodes, same source contract."""
    if source != "robochallenge":
        raise ValueError(f"Review rounds are implemented for robochallenge, not {source!r}")
    base = next(item for item in discover_components(source) if item.component == task)
    source_root = BUILD_ROOT / source
    review_root = source_root / "review" / task / f"round{review_round}"
    slug = f"robochallenge-{base.embodiment.lower()}-{task.replace('_', '-')}"
    staged_root = source_root / "staging" / f"local__{slug}-round{review_round}-source"
    manifest_path = staged_root / "meta/local_acquisition_manifest.json"
    for path in (review_root / "selection.json", manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Round {review_round} is not ready: {path} is missing")
    return Component(
        source=base.source,
        component=base.component,
        embodiment=base.embodiment,
        dataset_root=base.dataset_root,
        staging_root=base.staging_root,
        metadata_root=staged_root,
        spec=base.spec,
        video_origin=base.video_origin,
        raw_task_root=base.raw_task_root,
        manifest_path=manifest_path,
        selection_path=review_root / "selection.json",
        annotations_root=review_root,
        review_round=f"round{review_round}",
    )


def ingest_round(source: str, task: str, review_round: int, corpus_root: Path, *, overwrite: bool) -> None:
    component = round_component(source, task, review_round)
    accepted = _accepted_episodes(component)
    splits = assign_splits(accepted)
    hash_cache: dict[Path, str] = {}
    for episode_index in accepted:
        record = ingest_episode(
            component,
            episode_index,
            splits[episode_index],
            corpus_root,
            hash_cache=hash_cache,
            overwrite=overwrite,
        )
        print(
            f"{record['episode_id']:52s} frames={record['frames']:5d} "
            f"{record['duration_s']:7.2f}s split={record['split']} {record['review_round']}",
            flush=True,
        )
    refresh_episode_index(corpus_root)
    print(f"ingested {len(accepted)} round-{review_round} episodes from {source}/{task}", flush=True)


def refresh_episode_index(corpus_root: Path) -> list[dict[str, Any]]:
    rows = []
    for record_path in sorted((corpus_root / "episodes").glob("*/episode.json")):
        record = read_json(record_path)
        rows.append(
            {
                "episode_id": record["episode_id"],
                "source": record["source"],
                "component": record["component"],
                "embodiment": record["embodiment"],
                "split": record["split"],
                "frames": record["frames"],
                "duration_s": record["duration_s"],
                "native_rate_hz": record["native_rate_hz"],
                "action_source": record["action_source"],
                "cameras": [camera["name"] for camera in record["cameras"]],
                "task": record["task"],
                "source_episode_index": record["source_episode_index"],
                "directory": f"episodes/{record['episode_id']}",
            }
        )
    write_jsonl(corpus_root / "episodes.jsonl", rows)
    return rows


# --------------------------------------------------------------------------------------
# Actor view
# --------------------------------------------------------------------------------------


def _nearest_frames(timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(timestamps, targets, side="left"), 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    choose_left = np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets)
    return np.where(choose_left, left, right).astype(np.int64)


def actor_anchors(record: dict[str, Any], corpus_root: Path, stride_s: float) -> list[dict[str, Any]]:
    """Candidate anchors over one episode, with the reviewed eligibility decision attached."""
    directory = corpus_root / "episodes" / record["episode_id"]
    timestamps = np.load(directory / "timestamp_s.npy")
    annotations = record["annotations"]
    last_anchor = float(timestamps[-1]) - FUTURE_END_S
    if last_anchor < HISTORY_S:
        return []
    count = int(math.floor((last_anchor - HISTORY_S) / stride_s + 1e-9)) + 1
    anchors = HISTORY_S + np.arange(count, dtype=np.float64) * stride_s
    history_offsets = np.asarray([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
    # Boundary-relative fields let an experiment change the subtask-conditioning policy
    # without rebuilding the corpus: keep the anchor, decide later whether its future may
    # leave the subtask it started in.
    intervals = _merge_subtask_intervals(annotations)
    rows = []
    for index, anchor in enumerate(anchors):
        anchor = float(anchor)
        eligibility = annotation_window_eligibility(
            annotations,
            anchor - HISTORY_S,
            anchor + FUTURE_END_S,
            action_start_s=anchor,
        )
        segment = _segment_at(annotations, anchor)
        observation_frames = _nearest_frames(timestamps, anchor + history_offsets)
        timing_error = timestamps[observation_frames] - (anchor + history_offsets)
        containing = next(
            (
                (position, item)
                for position, item in enumerate(intervals)
                if item["start_s"] <= anchor < item["end_s"]
            ),
            (None, None),
        )
        interval_index, interval = containing
        future_end = anchor + FUTURE_END_S
        rows.append(
            {
                "episode_id": record["episode_id"],
                "source": record["source"],
                "component": record["component"],
                "embodiment": record["embodiment"],
                "split": record["split"],
                "anchor_index": index,
                "anchor_s": anchor,
                "anchor_frame": int(observation_frames[-1]),
                "history_frames": [int(value) for value in observation_frames],
                "max_observation_timing_error_s": float(np.abs(timing_error).max()),
                "future_end_s": anchor + FUTURE_END_S,
                "future_points": FUTURE_POINTS,
                "future_rate_hz": FUTURE_FPS,
                "native_rate_hz": record["native_rate_hz"],
                "action_source": record["action_source"],
                "retained": bool(eligibility["eligible"]),
                "rejection_reasons": eligibility["rejection_reasons"],
                "subtask_interval_index": interval_index,
                "time_since_subtask_start_s": (
                    None if interval is None else anchor - float(interval["start_s"])
                ),
                "time_to_subtask_end_s": (None if interval is None else float(interval["end_s"]) - anchor),
                "future_inside_subtask": (
                    None if interval is None else bool(future_end <= float(interval["end_s"]) + 1e-9)
                ),
                "subtask": segment.get("subtask"),
                "quality": segment.get("quality"),
                "quality_provenance": record["review"]["quality_provenance"],
                "mistake": bool(segment.get("mistake_events")),
                "retention_reason": segment.get("retention_reason"),
                "source_arrays_reference": f"episodes/{record['episode_id']}",
                "views_copy_source_arrays": False,
            }
        )
    return rows


def _segment_at(annotations: dict[str, Any], timestamp_s: float) -> dict[str, Any]:
    segments = annotations["segments"]
    for position, segment in enumerate(segments):
        start, end = float(segment["start_s"]), float(segment["end_s"])
        if start <= timestamp_s < end or (position == len(segments) - 1 and math.isclose(timestamp_s, end)):
            return segment
    return {}


# --------------------------------------------------------------------------------------
# Critic view
# --------------------------------------------------------------------------------------


def _merge_subtask_intervals(annotations: dict[str, Any]) -> list[dict[str, Any]]:
    """Group reviewed segments into complete subtask intervals.

    A short static gap inside one subtask is a pause and stays inside the interval,
    which is what a critic needs to see. Any other rejected span is an interruption:
    it closes the interval, and the interval it closed is truncated rather than complete.
    """
    intervals: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_pause: dict[str, Any] | None = None
    for segment in sorted(annotations["segments"], key=lambda item: float(item["start_s"])):
        start, end = float(segment["start_s"]), float(segment["end_s"])
        if segment["retention"] == "keep":
            subtask = str(segment["subtask"])
            if current is not None and current["subtask"] == subtask:
                if pending_pause is not None:
                    current["pause_events"].append(pending_pause)
                current["end_s"] = end
                current["segments"].append(segment)
            else:
                if current is not None:
                    current["end_boundary"] = "subtask_transition"
                    intervals.append(current)
                current = {
                    "subtask": subtask,
                    "start_s": start,
                    "end_s": end,
                    "segments": [segment],
                    "pause_events": [],
                    "interruption_events": [],
                    "start_boundary": "subtask_transition" if intervals or start > 0 else "episode_start",
                    "end_boundary": "episode_end",
                }
            pending_pause = None
            continue
        reason = str(segment["retention_reason"])
        if reason in PAUSE_REASONS:
            pending_pause = {"start_s": start, "end_s": end, "reason": reason, "type": "pause"}
            continue
        if current is not None:
            current["end_boundary"] = f"interruption:{reason}"
            current["interruption_events"].append(
                {"start_s": start, "end_s": end, "reason": reason, "type": "interruption"}
            )
            intervals.append(current)
            current = None
        pending_pause = None
    if current is not None:
        intervals.append(current)
    return intervals


def critic_intervals(record: dict[str, Any], corpus_root: Path) -> list[dict[str, Any]]:
    directory = corpus_root / "episodes" / record["episode_id"]
    timestamps = np.load(directory / "timestamp_s.npy")
    annotations = record["annotations"]
    episode_outcome = record["review"].get("outcome") or "unknown"
    merged = _merge_subtask_intervals(annotations)
    keep_ends = [
        float(segment["end_s"]) for segment in annotations["segments"] if segment.get("retention") == "keep"
    ]
    last_keep_end = max(keep_ends) if keep_ends else 0.0
    single_subtask_episode = len({item["subtask"] for item in merged}) == 1 and len(merged) == 1
    rows = []
    for index, interval in enumerate(merged):
        start_frame = int(np.searchsorted(timestamps, interval["start_s"], side="left"))
        end_frame = int(np.searchsorted(timestamps, interval["end_s"], side="left"))
        end_frame = min(max(end_frame, start_frame + 1), len(timestamps))
        duration_s = float(interval["end_s"] - interval["start_s"])
        samples = end_frame - start_frame
        qualities = [int(segment["quality"]) for segment in interval["segments"]]
        mistake_events = [
            dict(event, type="mistake")
            for segment in interval["segments"]
            for event in segment.get("mistake_events", [])
        ]
        recovery_events = [
            {
                "start_s": float(segment["start_s"]),
                "end_s": float(segment["end_s"]),
                "type": "recovery",
                "basis": str(segment["retention_reason"]),
            }
            for segment in interval["segments"]
            if str(segment.get("retention_reason")) == "recovery" or int(segment["quality"]) == 2
        ]
        events = interval["interruption_events"]
        reasons = {str(event["reason"]) for event in events}
        # An excluded span after the last kept motion is the end of the publisher's clip,
        # not an interruption of this subtask, so it does not disqualify the interval.
        trailing_clip_boundary = bool(events) and all(
            float(event["start_s"]) >= last_keep_end - 1e-9 and reason in SOURCE_RANGE_REASONS
            for event, reason in zip(events, [str(item["reason"]) for item in events], strict=True)
        )
        interrupted = bool(events) and not trailing_clip_boundary
        classification = (
            "interrupted"
            if interrupted
            else "complete_to_clip_boundary"
            if trailing_clip_boundary
            else "complete_with_pause"
            if interval["pause_events"]
            else "clean_and_complete"
        )
        rejection: str | None = None
        if interrupted:
            rejection = (
                "truncated_by_source_range"
                if reasons <= SOURCE_RANGE_REASONS
                else "truncated_by_low_quality_span"
                if reasons <= QUALITY_REASONS
                else "truncated_by_reviewed_interruption"
            )
        elif samples < 2 or duration_s < 1.0:
            rejection = "interval_shorter_than_one_second"
        subtask_outcome = (
            episode_outcome if single_subtask_episode and episode_outcome != "unknown" else "unknown"
        )
        rows.append(
            {
                "episode_id": record["episode_id"],
                "source": record["source"],
                "component": record["component"],
                "embodiment": record["embodiment"],
                "split": record["split"],
                "interval_index": index,
                "normalized_description": interval["subtask"],
                "primitive": None,
                "boundary_provenance": "human_reviewed",
                "start_boundary": interval["start_boundary"],
                "end_boundary": (
                    "source_keep_range_end" if trailing_clip_boundary else interval["end_boundary"]
                ),
                "end_boundary_provenance": (
                    "source_native_keep_range" if trailing_clip_boundary else "human_reviewed"
                ),
                "start_s": float(interval["start_s"]),
                "end_s_exclusive": float(interval["end_s"]),
                "duration_s": duration_s,
                "start_timestep": start_frame,
                "end_timestep_exclusive": end_frame,
                "native_action_samples": samples,
                "native_rate_hz": record["native_rate_hz"],
                "action_source": record["action_source"],
                "classification": classification,
                "critic_eligible": rejection is None,
                "critic_rejection_reason": rejection,
                "pause_events": interval["pause_events"],
                "interruption_events": interval["interruption_events"],
                "mistake_events": mistake_events,
                "recovery_events": recovery_events,
                "mistake_assessment": "human_reviewed",
                "recovery_assessment": "human_reviewed",
                "interruption_assessment": "human_reviewed",
                "pause_assessment": "human_reviewed",
                "quality": min(qualities) if qualities else None,
                "quality_values": sorted(set(qualities)),
                "quality_provenance": record["review"]["quality_provenance"],
                "subtask_outcome": subtask_outcome,
                "episode_outcome": episode_outcome,
                "outcome_provenance": "episode_level_visual_review",
                "action_sequence_reference": {
                    "path": f"episodes/{record['episode_id']}/action.npy",
                    "start_timestep": start_frame,
                    "end_timestep_exclusive": end_frame,
                    "contains_every_native_action": True,
                },
                "timestamp_reference": {
                    "path": f"episodes/{record['episode_id']}/timestamp_s.npy",
                    "start_timestep": start_frame,
                    "end_timestep_exclusive": end_frame,
                },
                "state_and_observation_reference": {
                    "episode_directory": f"episodes/{record['episode_id']}",
                    "same_timestep_range": True,
                },
            }
        )
    return rows


def build_views(corpus_root: Path, stride_s: float) -> dict[str, Any]:
    records = [read_json(path) for path in sorted((corpus_root / "episodes").glob("*/episode.json"))]
    stride_name = f"{1.0 / stride_s:.0f}hz".replace(".", "_")
    actor_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    for record in records:
        actor_rows.extend(actor_anchors(record, corpus_root, stride_s))
        critic_rows.extend(critic_intervals(record, corpus_root))
    retained = [row for row in actor_rows if row["retained"]]
    actor_path = corpus_root / f"actor_anchors_{stride_name}.jsonl"
    write_jsonl(actor_path, actor_rows)
    write_jsonl(corpus_root / "critic_intervals.jsonl", critic_rows)
    summary = {
        "actor_view": {
            "path": actor_path.name,
            "stride_s": stride_s,
            "candidate_anchors": len(actor_rows),
            "retained_anchors": len(retained),
            "retention": len(retained) / len(actor_rows) if actor_rows else 0.0,
            "rejection_reasons": _count_reasons(actor_rows),
            "by_split": _count_by(retained, "split"),
            "by_source": _count_by(retained, "source"),
            "retained_with_future_inside_subtask": sum(1 for row in retained if row["future_inside_subtask"]),
        },
        "critic_view": {
            "path": "critic_intervals.jsonl",
            "candidate_intervals": len(critic_rows),
            "eligible_intervals": sum(1 for row in critic_rows if row["critic_eligible"]),
            "eligible_duration_s": sum(row["duration_s"] for row in critic_rows if row["critic_eligible"]),
            "eligible_native_actions": sum(
                row["native_action_samples"] for row in critic_rows if row["critic_eligible"]
            ),
            "classifications": _count_by(critic_rows, "classification"),
            "rejection_reasons": {
                reason: sum(1 for row in critic_rows if row["critic_rejection_reason"] == reason)
                for reason in sorted(
                    {row["critic_rejection_reason"] for row in critic_rows if row["critic_rejection_reason"]}
                )
            },
            "by_split": _count_by([row for row in critic_rows if row["critic_eligible"]], "split"),
            "by_source": _count_by([row for row in critic_rows if row["critic_eligible"]], "source"),
        },
        "episodes": len(records),
        "views_copy_source_sensor_or_action_arrays": False,
    }
    write_json(corpus_root / "views_summary.json", summary)
    return summary


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row[field])] = counts.get(str(row[field]), 0) + 1
    return dict(sorted(counts.items()))


def _count_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["rejection_reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def _decode(path: Path, timestamps: list[float], fps: float) -> np.ndarray:
    from lerobot.datasets.video_utils import decode_video_frames

    frames = decode_video_frames(path, timestamps, tolerance_s=0.51 / fps, backend="pyav", return_uint8=True)
    return frames.permute(0, 2, 3, 1).numpy().astype(np.int16)


def check_video_alignment(record: dict[str, Any], corpus_root: Path, samples: int) -> list[dict[str, Any]]:
    """Confirm corpus frame k really is source frame k, not a seek that slipped by one."""
    directory = corpus_root / "episodes" / record["episode_id"]
    frames = int(record["frames"])
    fps = float(record["native_rate_hz"])
    offsets = [-2, -1, 0, 1, 2]
    positions = [int(round(fraction * frames)) for fraction in np.linspace(0.2, 0.8, samples)]
    positions = [index for index in positions if 2 < index < frames - 3]
    results = []
    for camera in record["cameras"]:
        corpus_path = directory / camera["path"]
        source_path = BUILD_ROOT / camera["source"]["path"]
        if not source_path.is_file() or not positions:
            results.append({"camera": camera["name"], "verdict": "source_absent"})
            continue
        start_s = float(camera["source"]["start_s"])
        camera_results = []
        for position in positions:
            # Decode each comparison window on its own: five neighbouring timestamps cost one
            # local seek, where a single call spanning the episode decodes most of the source.
            corpus_frame = _decode(corpus_path, [position / fps], fps)[0]
            candidates = _decode(
                source_path, [start_s + (position + offset) / fps for offset in offsets], fps
            )
            errors = np.abs(candidates - corpus_frame[None]).mean(axis=(1, 2, 3))
            spread = float(errors.max() - errors.min())
            aligned_error = float(errors[offsets.index(0)])
            best = offsets[int(np.argmin(errors))]
            margin = aligned_error - float(errors.min())
            # Re-encoding leaves a noise floor, and a still moment makes neighbouring source
            # frames indistinguishable. Only a margin above both is evidence of a shift.
            tie_threshold = max(0.1 * spread, 0.25 * float(errors.min()), 0.05)
            verdict = (
                "aligned"
                if best == 0
                else "tie_inconclusive"
                if margin <= tie_threshold
                else "offset_candidate"
            )
            camera_results.append(
                {
                    "camera": camera["name"],
                    "frame": position,
                    "best_offset": best,
                    "mean_abs_error": aligned_error,
                    "offset_spread": spread,
                    "margin": margin,
                    "tie_threshold": tie_threshold,
                    "verdict": verdict,
                }
            )
        # A real off-by-one shifts the whole stream, so it must appear at more than one
        # sampled frame. A lone dissenting frame is a tie, not evidence of a shift.
        votes: dict[int, int] = {}
        for item in camera_results:
            if item["verdict"] == "offset_candidate":
                votes[item["best_offset"]] = votes.get(item["best_offset"], 0) + 1
        shifted = {offset for offset, count in votes.items() if count >= 2}
        for item in camera_results:
            if item["verdict"] == "offset_candidate":
                item["verdict"] = "misaligned" if item["best_offset"] in shifted else "tie_inconclusive"
        results.extend(camera_results)
    return results


def validate(corpus_root: Path, *, alignment_samples: int, check_hashes: bool) -> dict[str, Any]:
    records = [read_json(path) for path in sorted((corpus_root / "episodes").glob("*/episode.json"))]
    failures: list[str] = []
    alignment: list[dict[str, Any]] = []
    accounting: dict[str, dict[str, Any]] = {}

    for record in records:
        identifier = record["episode_id"]
        directory = corpus_root / "episodes" / identifier
        frames = int(record["frames"])
        timestamps = np.load(directory / "timestamp_s.npy")
        state = np.load(directory / "state.npy")
        action = np.load(directory / "action.npy")
        if len(timestamps) != frames or len(state) != frames or len(action) != frames:
            failures.append(f"{identifier}: array lengths disagree with the declared frame count")
        if np.any(np.diff(timestamps) <= 0):
            failures.append(f"{identifier}: timestamps are not strictly increasing")
        if not (np.isfinite(state).all() and np.isfinite(action).all()):
            failures.append(f"{identifier}: state or action contains NaN or infinity")
        if record["action_source"] == "copy_state" and not np.array_equal(state, action):
            failures.append(f"{identifier}: copy_state action differs from the measured state")

        annotations = record["annotations"]
        if annotations.get("review_status") != "validated":
            failures.append(f"{identifier}: annotations are not validated")
        cursor = 0.0
        for segment in sorted(annotations["segments"], key=lambda item: float(item["start_s"])):
            if abs(float(segment["start_s"]) - cursor) > 1e-6:
                failures.append(f"{identifier}: annotation segments are not contiguous")
            cursor = float(segment["end_s"])
        if cursor < float(record["duration_s"]) - 1e-3:
            failures.append(f"{identifier}: annotations stop before the episode ends")
        if record["split"] not in {"train", "validation", "test"}:
            failures.append(f"{identifier}: invalid split {record['split']!r}")

        for camera in record["cameras"]:
            video_path = directory / camera["path"]
            if not video_path.is_file():
                failures.append(f"{identifier}: missing {camera['path']}")
                continue
            probe = probe_video(video_path)
            if probe["frames"] != frames:
                failures.append(
                    f"{identifier}/{camera['name']}: {probe['frames']} video frames against "
                    f"{frames} timeline samples"
                )
            if check_hashes and sha256(video_path) != camera["sha256"]:
                failures.append(f"{identifier}/{camera['name']}: video hash changed since ingest")

        if alignment_samples:
            for result in check_video_alignment(record, corpus_root, alignment_samples):
                result["episode_id"] = identifier
                alignment.append(result)
                if result["verdict"] == "misaligned":
                    failures.append(
                        f"{identifier}/{result['camera']}: frame {result.get('frame')} matches "
                        f"source offset {result.get('best_offset')}"
                    )

        bucket = accounting.setdefault(
            record["source"],
            {
                "accepted_episodes": 0,
                "accepted_span_duration_s": 0.0,
                "unique_synchronized_timesteps": 0,
                "native_action_samples": 0,
                "rgb_images_by_camera": {},
                "depth_maps_by_camera": {},
                "embodiments": {},
                "split_episode_counts": {},
                "counts_provenance": "observed_directly_from_converted_corpus_arrays",
            },
        )
        bucket["accepted_episodes"] += 1
        bucket["accepted_span_duration_s"] += float(record["duration_s"])
        bucket["unique_synchronized_timesteps"] += frames
        bucket["native_action_samples"] += frames
        for camera in record["cameras"]:
            key = camera["name"]
            bucket["rgb_images_by_camera"][key] = bucket["rgb_images_by_camera"].get(key, 0) + frames
        bucket["embodiments"][record["embodiment"]] = bucket["embodiments"].get(record["embodiment"], 0) + 1
        bucket["split_episode_counts"][record["split"]] = (
            bucket["split_episode_counts"].get(record["split"], 0) + 1
        )

    report = {
        "corpus_format": CORPUS_FORMAT,
        "builder_version": BUILDER_VERSION,
        "episodes": len(records),
        "per_source_accounting": accounting,
        "alignment_checks": {
            "samples_per_camera": alignment_samples,
            "checked": len(alignment),
            "verdicts": {
                verdict: sum(1 for item in alignment if item["verdict"] == verdict)
                for verdict in sorted({item["verdict"] for item in alignment})
            },
        },
        "video_hashes_checked": check_hashes,
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    write_json(corpus_root / "validation_report.json", report)
    write_jsonl(corpus_root / "alignment_checks.jsonl", alignment)
    return report


# --------------------------------------------------------------------------------------
# Packed actor components
# --------------------------------------------------------------------------------------

PACK_BATCH_ANCHORS = 24


def _decode_corpus_frames(
    directory: Path, camera: dict[str, Any], anchors: list[Any], fps: float
) -> np.ndarray:
    """Decode the seven history frames for a batch of anchors from one corpus video."""
    from lerobot.datasets.video_utils import decode_video_frames

    query = [float(timestamp) for anchor in anchors for timestamp in anchor.observation_timestamps]
    frames = decode_video_frames(
        directory / camera["path"], query, tolerance_s=0.51 / fps, backend="pyav", return_uint8=True
    )
    array = frames.permute(0, 2, 3, 1).numpy()
    return array.reshape(len(anchors), 7, camera["height"], camera["width"], 3)


def pack_component(
    corpus_root: Path,
    records: list[dict[str, Any]],
    dataset_root: Path,
    *,
    repo_id: str,
    stride_s: float,
    min_chunks_per_episode: int,
) -> dict[str, Any]:
    from lerobot.datasets.diverse_pilot import (
        annotation_at,
        pack_numerical_episode,
        packed_feature_schema,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if dataset_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing packed root: {dataset_root}")
    first = records[0]
    cameras = first["cameras"]
    state_dimension = int(first["state_dimension"])
    action_dimension = int(first["action_dimension"])
    features = packed_feature_schema(
        state_dimension, action_dimension, state_dtype="float64", action_dtype="float64"
    )
    labels = ("tm6", "tm5", "tm4", "tm3", "tm2", "tm1", "t")
    for camera in cameras:
        for label in labels:
            features[f"observation.images.{camera['name']}.{label}"] = {
                "dtype": "video",
                "shape": (camera["height"], camera["width"], 3),
                "names": None,
            }
    features.update(
        {
            key: {"dtype": "string", "shape": (1,), "names": None}
            for key in (
                "source.repo_id",
                "source.revision",
                "source.robot_type",
                "source.state_semantics",
                "source.action_semantics",
                "source.gripper_semantics",
                "source.action_source",
                "source.corpus_episode_id",
                "source.split",
                "annotation.subtask",
                "annotation.retention_reason",
            )
        }
    )
    features["annotation.quality"] = {"dtype": "int64", "shape": (1,), "names": None}
    features["annotation.mistake"] = {"dtype": "bool", "shape": (1,), "names": None}

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=str(first["robot_type"]),
        fps=1,
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )

    episodes_report = []
    for record in records:
        directory = corpus_root / "episodes" / record["episode_id"]
        timestamps = np.load(directory / "timestamp_s.npy")
        state = np.load(directory / "state.npy")
        action = np.load(directory / "action.npy")
        annotations = record["annotations"]
        candidates = pack_numerical_episode(
            timestamps,
            state,
            timestamps,
            action,
            native_action_rate_hz=float(record["native_rate_hz"]),
            stride_s=stride_s,
        )
        retained = []
        rejected: dict[str, int] = {}
        for sample in candidates:
            decision = annotation_window_eligibility(
                annotations,
                sample.anchor_timestamp - HISTORY_S,
                sample.anchor_timestamp + FUTURE_END_S,
                action_start_s=sample.anchor_timestamp,
            )
            if decision["eligible"]:
                retained.append(sample)
                continue
            for reason in decision["rejection_reasons"]:
                rejected[reason] = rejected.get(reason, 0) + 1
        if len(retained) < min_chunks_per_episode:
            raise ValueError(
                f"{record['episode_id']}: only {len(retained)} eligible anchors at stride {stride_s}"
            )
        for start in range(0, len(retained), PACK_BATCH_ANCHORS):
            batch = retained[start : start + PACK_BATCH_ANCHORS]
            decoded = {
                camera["name"]: _decode_corpus_frames(
                    directory, camera, batch, float(record["native_rate_hz"])
                )
                for camera in cameras
            }
            for position, sample in enumerate(batch):
                annotation = annotation_at(annotations, sample.anchor_timestamp)
                frame = {
                    "observation.state": sample.observation_values[-1],
                    "observation.state_history": sample.observation_values[:-1],
                    "action": sample.action.values,
                    "source.episode_index": np.asarray([record["source_episode_index"]], dtype=np.int64),
                    "source.anchor_timestamp": np.asarray([sample.anchor_timestamp], dtype=np.float64),
                    "source.observation_timestamps": sample.observation_timestamps,
                    "source.observation_frame_indices": sample.observation_frame_indices,
                    "source.observation_timing_error": sample.observation_timing_error,
                    "source.action_timestamps": sample.action.target_timestamps,
                    "source.action_source_timestamps": sample.action.source_timestamps,
                    "source.action_source_indices": sample.action.source_indices,
                    "source.action_source_values": sample.action.source_values,
                    "source.action_interpolated_mask": sample.action.interpolated_mask,
                    "source.repo_id": record["source_repo_id"],
                    "source.revision": record["source_revision"],
                    "source.robot_type": record["robot_type"],
                    "source.state_semantics": record["state_semantics"],
                    "source.action_semantics": record["action_semantics"],
                    "source.gripper_semantics": record["gripper_semantics"],
                    "source.action_source": record["action_source"],
                    "source.corpus_episode_id": record["episode_id"],
                    "source.split": record["split"],
                    "annotation.subtask": annotation["subtask"],
                    "annotation.retention_reason": annotation["retention_reason"],
                    "annotation.quality": np.asarray([annotation["quality"]], dtype=np.int64),
                    "annotation.mistake": np.asarray([annotation["mistake"]], dtype=bool),
                    "task": record["task"] or f"corpus episode {record['episode_id']}",
                }
                for camera in cameras:
                    for offset, label in enumerate(labels):
                        frame[f"observation.images.{camera['name']}.{label}"] = decoded[camera["name"]][
                            position, offset
                        ]
                dataset.add_frame(frame)
        dataset.save_episode()
        episodes_report.append(
            {
                "corpus_episode_id": record["episode_id"],
                "source_episode_index": record["source_episode_index"],
                "split": record["split"],
                "source_frames": int(len(timestamps)),
                "candidate_anchors": len(candidates),
                "packed_rows": len(retained),
                "rejection_reasons": dict(sorted(rejected.items())),
            }
        )
        print(
            f"  packed {record['episode_id']:52s} rows={len(retained):5d} candidates={len(candidates):5d}",
            flush=True,
        )

    report = {
        "repo_id": repo_id,
        "stride_s": stride_s,
        "episodes": episodes_report,
        "packed_rows": sum(item["packed_rows"] for item in episodes_report),
        "candidate_anchors": sum(item["candidate_anchors"] for item in episodes_report),
        "generated_from": "source_native_corpus",
        "corpus_format": CORPUS_FORMAT,
    }
    write_json(dataset_root / "meta/corpus_extraction_report.json", report)
    return report


def pack(
    corpus_root: Path,
    *,
    source: str | None,
    component: str | None,
    stride_s: float,
    min_chunks_per_episode: int,
) -> None:
    records = [read_json(path) for path in sorted((corpus_root / "episodes").glob("*/episode.json"))]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if source and record["source"] != source:
            continue
        if component and record["component"] != component:
            continue
        groups.setdefault((record["source"], record["component"]), []).append(record)
    stride_name = f"{1.0 / stride_s:.0f}hz"
    reports = []
    for (group_source, group_component), group in sorted(groups.items()):
        dataset_root = corpus_root / f"packed_{stride_name}" / group_source / group_component
        if dataset_root.exists():
            print(f"skipping existing {dataset_root}", flush=True)
            continue
        print(f"packing {group_source}/{group_component} ({len(group)} episodes)", flush=True)
        reports.append(
            pack_component(
                corpus_root,
                sorted(group, key=lambda item: item["source_episode_index"]),
                dataset_root,
                repo_id=f"local/diverse-corpus-{group_source}-{group_component}-{stride_name}",
                stride_s=stride_s,
                min_chunks_per_episode=min_chunks_per_episode,
            )
        )
    total = sum(item["packed_rows"] for item in reports)
    print(f"packed {total} rows across {len(reports)} components", flush=True)


# --------------------------------------------------------------------------------------
# Unified accounting
# --------------------------------------------------------------------------------------

FMB_CORPUS = DATASET_ROOT / "fmb"


def _blank_bucket() -> dict[str, Any]:
    return {
        "accepted_episodes": 0,
        "accepted_span_duration_s": 0.0,
        "unique_synchronized_timesteps": 0,
        "native_action_samples": 0,
        "rgb_images_by_camera": {},
        "depth_maps_by_camera": {},
        "candidate_actor_anchors": 0,
        "retained_actor_rows": 0,
        "actor_rejection_reasons": {},
        "candidate_critic_intervals": 0,
        "critic_eligible_intervals": 0,
        "critic_eligible_duration_s": 0.0,
        "critic_eligible_native_actions": 0,
        "critic_rejection_reasons": {},
        "split_episode_counts": {},
        "embodiments": {},
    }


def _accumulate(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    frames = int(record["frames"])
    bucket["accepted_episodes"] += 1
    bucket["accepted_span_duration_s"] += float(record["duration_s"])
    bucket["unique_synchronized_timesteps"] += frames
    bucket["native_action_samples"] += frames
    for camera in record["cameras"]:
        key = camera["name"]
        bucket["rgb_images_by_camera"][key] = bucket["rgb_images_by_camera"].get(key, 0) + frames
    bucket["split_episode_counts"][record["split"]] = (
        bucket["split_episode_counts"].get(record["split"], 0) + 1
    )
    bucket["embodiments"][record["embodiment"]] = bucket["embodiments"].get(record["embodiment"], 0) + 1


def _finish(bucket: dict[str, Any]) -> dict[str, Any]:
    timesteps = bucket["unique_synchronized_timesteps"]
    bucket["row_to_unique_timestep_ratio"] = bucket["retained_actor_rows"] / timesteps if timesteps else 0.0
    bucket["accepted_span_duration_s"] = round(bucket["accepted_span_duration_s"], 3)
    bucket["critic_eligible_duration_s"] = round(bucket["critic_eligible_duration_s"], 3)
    return bucket


def ledger(corpus_root: Path, actor_view: str) -> dict[str, Any]:
    records = [read_json(path) for path in sorted((corpus_root / "episodes").glob("*/episode.json"))]
    actor_rows = read_jsonl(corpus_root / actor_view)
    critic_rows = read_jsonl(corpus_root / "critic_intervals.jsonl")

    components: dict[tuple[str, str], dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for record in records:
        key = (record["source"], record["component"])
        _accumulate(components.setdefault(key, _blank_bucket()), record)
        _accumulate(sources.setdefault(record["source"], _blank_bucket()), record)

    episode_component = {record["episode_id"]: (record["source"], record["component"]) for record in records}
    for row in actor_rows:
        key = episode_component[row["episode_id"]]
        for bucket in (components[key], sources[key[0]]):
            bucket["candidate_actor_anchors"] += 1
            if row["retained"]:
                bucket["retained_actor_rows"] += 1
            for reason in row["rejection_reasons"]:
                bucket["actor_rejection_reasons"][reason] = (
                    bucket["actor_rejection_reasons"].get(reason, 0) + 1
                )
    for row in critic_rows:
        key = episode_component[row["episode_id"]]
        for bucket in (components[key], sources[key[0]]):
            bucket["candidate_critic_intervals"] += 1
            if row["critic_eligible"]:
                bucket["critic_eligible_intervals"] += 1
                bucket["critic_eligible_duration_s"] += float(row["duration_s"])
                bucket["critic_eligible_native_actions"] += int(row["native_action_samples"])
            else:
                reason = str(row["critic_rejection_reason"])
                bucket["critic_rejection_reasons"][reason] = (
                    bucket["critic_rejection_reasons"].get(reason, 0) + 1
                )

    totals = _blank_bucket()
    for bucket in sources.values():
        for field in (
            "accepted_episodes",
            "accepted_span_duration_s",
            "unique_synchronized_timesteps",
            "native_action_samples",
            "candidate_actor_anchors",
            "retained_actor_rows",
            "candidate_critic_intervals",
            "critic_eligible_intervals",
            "critic_eligible_duration_s",
            "critic_eligible_native_actions",
        ):
            totals[field] += bucket[field]
        for field in (
            "rgb_images_by_camera",
            "split_episode_counts",
            "embodiments",
            "actor_rejection_reasons",
            "critic_rejection_reasons",
        ):
            for key_name, value in bucket[field].items():
                totals[field][key_name] = totals[field].get(key_name, 0) + value

    external: dict[str, Any] = {}
    if (FMB_CORPUS / "corpus.json").is_file():
        fmb = read_json(FMB_CORPUS / "corpus.json")["accounting"]
        derived = fmb["derived_training_rows"]
        stored_views = derived.get("stored_actor_anchor_views", {})
        actor_5hz = stored_views.get("5hz", {}).get(
            "rows",
            derived["actor_anchor_accounting"]["proposed_5hz"]["episode_level_candidate_anchors"],
        )
        actor_10hz = stored_views.get("10hz", {}).get("rows", derived["stored_actor_anchor_view"]["rows"])
        external["fmb"] = {
            "path": str(FMB_CORPUS.relative_to(DATASET_ROOT)),
            "accepted_episodes": fmb["accepted_source_episodes"],
            "unique_synchronized_timesteps": fmb["unique_synchronized_timesteps_observed"],
            "native_action_samples": fmb["production_retained_sensor_counts"]["native_action_samples"],
            "accepted_span_duration_s": fmb["accepted_continuous_span_duration_s_nominal"],
            "rgb_images_by_camera": fmb["production_retained_sensor_counts"]["rgb_images_by_camera"],
            "depth_maps_by_camera": fmb["production_retained_sensor_counts"]["depth_maps_by_camera"],
            "retained_actor_rows_5hz": actor_5hz,
            "retained_actor_rows_10hz": actor_10hz,
            "candidate_critic_intervals": derived["candidate_critic_intervals"],
            "critic_eligible_intervals": derived["critic_eligible_intervals"],
            "split_episode_counts": fmb["split_episode_counts"],
            "note": (
                "FMB remains in its source-native store with wrist depth. The federated "
                "loader joins its indexes with the common corpus without copying source arrays."
            ),
        }

    report = {
        "format": "diverse_robot_unified_coverage_ledger_v1",
        "corpus_root": str(corpus_root.relative_to(DATASET_ROOT)),
        "actor_view": actor_view,
        "components": {
            f"{source}/{component}": _finish(bucket)
            for (source, component), bucket in sorted(components.items())
        },
        "sources": {name: _finish(bucket) for name, bucket in sorted(sources.items())},
        "totals": _finish(totals),
        "external_corpora": external,
        "combined_with_external": {
            "actor_stride_hz": 5.0,
            "accepted_episodes": totals["accepted_episodes"]
            + sum(item["accepted_episodes"] for item in external.values()),
            "unique_synchronized_timesteps": totals["unique_synchronized_timesteps"]
            + sum(item["unique_synchronized_timesteps"] for item in external.values()),
            "retained_actor_rows": totals["retained_actor_rows"]
            + sum(item.get("retained_actor_rows_5hz", 0) for item in external.values()),
            "critic_eligible_intervals": totals["critic_eligible_intervals"]
            + sum(item.get("critic_eligible_intervals", 0) for item in external.values()),
        },
        "federation": {
            "loader": "lerobot.datasets.fmb_corpus.FederatedDiverseCorpus",
            "common_root": str(corpus_root.relative_to(DATASET_ROOT)),
            "fmb_root": str(FMB_CORPUS.relative_to(DATASET_ROOT)),
            "views_copy_source_arrays": False,
        },
        "accounting_provenance": {
            "episode_camera_and_timestep_counts": "observed_directly_from_converted_corpus_arrays",
            "durations": "derived_exactly_from_source_native_timestamps",
            "actor_and_critic_counts": "derived_exactly_from_the_stored_view_files",
            "external_corpora": "read_from_that_corpus_own_validated_accounting",
        },
    }
    write_json(BUILD_ROOT / "provenance" / "unified_coverage_ledger.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument(
        "--source", required=True, choices=["robochallenge", "droid", "droid_success", "ur7e"]
    )
    ingest_parser.add_argument("--component", action="append")
    ingest_parser.add_argument("--overwrite", action="store_true")

    round_parser = subparsers.add_parser("ingest-round")
    round_parser.add_argument("--source", default="robochallenge")
    round_parser.add_argument("--task", required=True)
    round_parser.add_argument("--round", type=int, default=2)
    round_parser.add_argument("--overwrite", action="store_true")

    subparsers.add_parser("index")
    subparsers.add_parser("refresh-metadata")

    views_parser = subparsers.add_parser("views")
    views_parser.add_argument("--stride-hz", type=float, default=5.0)

    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--source")
    pack_parser.add_argument("--component")
    pack_parser.add_argument("--stride-hz", type=float, default=5.0)
    pack_parser.add_argument("--min-chunks-per-episode", type=int, default=3)

    ledger_parser = subparsers.add_parser("ledger")
    ledger_parser.add_argument("--actor-view", default="actor_anchors_5hz.jsonl")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--alignment-samples", type=int, default=3)
    validate_parser.add_argument("--check-hashes", action="store_true")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.source, args.corpus_root, components=args.component, overwrite=args.overwrite)
    elif args.command == "ingest-round":
        ingest_round(args.source, args.task, args.round, args.corpus_root, overwrite=args.overwrite)
    elif args.command == "refresh-metadata":
        print(f"{refresh_metadata(args.corpus_root)} episode records updated")
    elif args.command == "index":
        rows = refresh_episode_index(args.corpus_root)
        print(f"{len(rows)} episodes indexed")
    elif args.command == "views":
        summary = build_views(args.corpus_root, 1.0 / args.stride_hz)
        print(json.dumps(summary, indent=2))
    elif args.command == "ledger":
        report = ledger(args.corpus_root, args.actor_view)
        print(json.dumps({"totals": report["totals"], "sources": list(report["sources"])}, indent=2))
    elif args.command == "pack":
        pack(
            args.corpus_root,
            source=args.source,
            component=args.component,
            stride_s=1.0 / args.stride_hz,
            min_chunks_per_episode=args.min_chunks_per_episode,
        )
    elif args.command == "validate":
        report = validate(
            args.corpus_root,
            alignment_samples=args.alignment_samples,
            check_hashes=args.check_hashes,
        )
        print(json.dumps(report, indent=2))
        if report["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()

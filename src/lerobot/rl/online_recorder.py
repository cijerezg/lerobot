"""Record online inference episodes in the lerobot_record format.

One LeRobotDataset per run at ``{output_dir}/inference_dataset`` with the schema a teleop
recording of the same rig gets (``hw_to_dataset_features`` on the robot's own feature
dicts: rig camera names, native joint order and width, video frames, ``robot_type``), and
depth in the PNG16 sidecar (``depth_writer.write_depth``) on the policy's ``image_stride``
grid with the global-index phase. The training loaders (``dataset.sources`` +
``lerobot_memmap_buffer_cache``) consume it like any recorded session; annotation is the
same separate pass.

Frames are written as they happen from the env's raw robot observation (float32 joints,
the camera's own uint8 frame), not from the online replay buffer, whose bf16 low-dim
storage would round the joint angles.

Per-frame labels the recorded schema has no column for go to ``meta/online_labels.parquet``
(episode_index, frame_index, index, is_intervention, subtask_index, subtask): a sidecar in
the style of the annotation files, invisible to the loaders. The subtask text rides next to
its index because the subtask vocabulary may be revised.
"""

import logging
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.datasets.depth_writer import write_depth
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

logger = logging.getLogger(__name__)

LABELS_FILE = "meta/online_labels.parquet"


class OnlineEpisodeRecorder:
    def __init__(self, robot, *, root, fps: int, task: str, depth_stride: int) -> None:
        self.robot = robot
        self.root = Path(root)
        self.fps = fps
        self.task = task
        self.depth_stride = depth_stride
        self.features = {
            **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True),
            **hw_to_dataset_features(robot.action_features, ACTION),
        }
        # Rig joint order and width; the env maps action[i] onto the same list.
        self.joint_names = list(robot.action_features)
        self.dataset: LeRobotDataset | None = None
        self._labels: list[dict] = []
        self._episode_labels: list[dict] = []

    def _open(self) -> None:
        self.dataset = LeRobotDataset.create(
            self.root.parent.name,
            self.fps,
            root=self.root,
            robot_type=self.robot.name,
            features=self.features,
            use_videos=True,
            image_writer_threads=4 * len(self.robot.cameras),
        )
        logger.info("[RECORDER] Recording online episodes to %s", self.root)

    def add_frame(
        self, raw_obs: dict, action, *, is_intervention: bool, subtask_index: int, subtask: str
    ) -> None:
        """One frame = the robot observation the step was taken from + the executed action.

        ``raw_obs`` is ``RobotEnv``'s unprocessed observation ({"agent_pos", "pixels": {cam: HWC
        uint8}, "{joint}.pos", "{cam}.depth"}); ``action`` is the executed target in the policy's
        padded layout width, of which the leading rig joints are kept.
        """
        if self.dataset is None:
            self._open()
        values = {**raw_obs["pixels"], **{k: v for k, v in raw_obs.items() if k not in ("pixels", "agent_pos")}}
        action = torch.as_tensor(action).detach().float().cpu().numpy().reshape(-1)
        frame = {
            **build_dataset_frame(self.features, values, prefix=OBS_STR),
            **build_dataset_frame(self.features, dict(zip(self.joint_names, action)), prefix=ACTION),
            "task": self.task,
        }
        self.dataset.add_frame(frame)
        depth = {
            k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in values.items() if k.endswith(".depth")
        }
        write_depth(self.dataset, depth, self.depth_stride)
        episode_buffer = self.dataset.writer.episode_buffer
        frame_index = episode_buffer["size"] - 1
        self._episode_labels.append({
            "episode_index": episode_buffer["episode_index"],
            "frame_index": frame_index,
            "index": self.dataset.meta.total_frames + frame_index,
            "is_intervention": bool(is_intervention),
            "subtask_index": int(subtask_index),
            "subtask": subtask,
        })

    def save_episode(self) -> None:
        if self.dataset is None or not self.dataset.has_pending_frames():
            return
        episode_buffer = self.dataset.writer.episode_buffer
        episode_index, num_frames = episode_buffer["episode_index"], episode_buffer["size"]
        self.dataset.save_episode()
        self._labels.extend(self._episode_labels)
        self._episode_labels = []
        pq.write_table(pa.Table.from_pylist(self._labels), self.root / LABELS_FILE)
        logger.info("[RECORDER] Episode %d saved (%d frames).", episode_index, num_frames)

    def finalize(self) -> None:
        if self.dataset is None:
            return
        writer = self.dataset.writer
        if self.dataset.has_pending_frames():
            episode_index, num_frames = writer.episode_buffer["episode_index"], writer.episode_buffer["size"]
            logger.warning("[RECORDER] Discarding %d frames of the unfinished episode.", num_frames)
            writer.clear_episode_buffer()  # waits for the image writer; deletes only image-dtype dirs
            writer.cleanup_interrupted_episode(episode_index)  # the video keys' temporary PNGs
            self._episode_labels = []
        self.dataset.finalize()
        images_dir = self.root / "images"
        if images_dir.exists() and not any(images_dir.rglob("*.png")):
            shutil.rmtree(images_dir)
        logger.info(
            "[RECORDER] Dataset closed: %d episodes, %d frames at %s",
            self.dataset.meta.total_episodes, self.dataset.meta.total_frames, self.root,
        )

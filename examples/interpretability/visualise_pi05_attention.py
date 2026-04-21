#!/usr/bin/env python
"""Run a Pi0.5 / `pi05_rl` policy on dataset frames and emit attention-overlay videos.

Mirrors `examples/visualise_original_data_attention.py` from
https://github.com/villekuosmanen/physical-AI-interpretability but for the
PaliGemma + Gemma-300m architecture used by `pi05_full` / `pi05_rl`.

Example:
    python -m examples.interpretability.visualise_pi05_attention \\
        --policy-path outputs/pi05_full_offline_recap_20260418_3/checkpoints/002000/pretrained_model \\
        --dataset-repo-id jackvial/so101_pickplace_success_120_v2_w_subtasks_v2 \\
        --episode-id 0 \\
        --output-dir output/pi05_attention/ep0
"""

from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_policy, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins

import lerobot.rl.rl_pi05  # noqa: F401  - registers PI05RLConfig ("pi05_rl") via import side-effects

from examples.interpretability._video import encode_video_ffmpeg
from examples.interpretability.pi05_attention_mapper import PI05PolicyWithAttention


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Pi0.5 attention-overlay videos for a dataset episode."
    )
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the pretrained Pi0.5 policy checkpoint.")
    parser.add_argument("--dataset-repo-id", type=str, required=True,
                        help="Hugging Face repo id of the LeRobot dataset.")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Optional local path to a LeRobotDataset (skips HF "
                             "download). Use this when the dataset only exists "
                             "on disk.")
    parser.add_argument("--episode-id", type=int, required=True,
                        help="Episode index to analyse.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for MP4s. Defaults to "
                             "./output/pi05_attention/ep<episode-id>/.")
    parser.add_argument("--target-layer", type=int, default=-1,
                        help="Action-expert self-attn layer to hook (-1 = last).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (cuda or cpu).")
    parser.add_argument("--capture-strategy", type=str, default="mean",
                        choices=["mean", "last_action"],
                        help="How to aggregate attention across the 50 action positions.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional cap on the number of frames processed.")
    parser.add_argument("--rename", type=str, nargs="*", default=None,
                        help="Rename dataset keys to model keys, e.g. "
                             "`--rename observation.images.side:observation.images.wrist`. "
                             "Useful when the recorded dataset uses different camera "
                             "names than the policy was trained with.")
    return parser.parse_args()


def _parse_rename(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {}
    mapping: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(
                f"Invalid --rename entry {item!r}; expected `from:to`."
            )
        src, dst = item.split(":", 1)
        mapping[src.strip()] = dst.strip()
    return mapping


def _build_policy_and_preprocessor(
    policy_path: str, dataset, device: torch.device
):
    """Load policy + preprocessor, special-casing pi05_rl's runtime upgrade path."""
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    if hasattr(policy_cfg, "device"):
        policy_cfg.device = device.type

    policy_type = getattr(policy_cfg, "type", "")
    if policy_type == "pi05_rl":
        # Mirror lerobot.async_inference.policy_server_drtc: PI05RLPolicy's
        # actor/critic split loader is invoked directly via __init__; going
        # through make_policy/from_pretrained would silently load only the
        # base pi05 weights.
        policy_cfg.pi05_checkpoint = policy_path
        policy_class = get_policy_class(policy_type)
        policy = policy_class(policy_cfg)
        policy = policy.to(device)
        policy.eval()

        from lerobot.rl.pi05_train_utils import make_pi05_full_processors_with_upgrade

        shim_cfg = SimpleNamespace(policy=policy_cfg)
        preprocessor, _postprocessor = make_pi05_full_processors_with_upgrade(
            cfg=shim_cfg, dataset=dataset, is_main_process=True
        )
    else:
        policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
        policy = policy.to(device)
        policy.eval()
        preprocessor, _postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": device.type}},
        )

    return policy, preprocessor, policy_cfg


def _frame_to_observation(
    frame: dict[str, Any],
    image_keys: list[str],
    rename_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a `LeRobotDataset` frame to the raw observation dict expected
    by the wrapper's preprocessor.

    Images are kept as `(C, H, W)` float tensors in `[0, 1]` (the
    `AddBatchDimensionObservationStep` will add the batch dim).
    """
    observation: dict[str, Any] = {}
    rename_map = rename_map or {}
    inverse_rename = {dst: src for src, dst in rename_map.items()}

    for key in image_keys:
        source_key = inverse_rename.get(key, key)
        if source_key not in frame:
            continue
        value = frame[source_key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        while value.dim() > 3:
            value = value.squeeze(0)
        if value.dim() != 3:
            raise ValueError(
                f"Expected 3D image tensor for {key}, got shape {value.shape}"
            )
        if value.shape[0] not in (1, 3) and value.shape[-1] in (1, 3):
            value = value.permute(2, 0, 1)
        if value.dtype != torch.float32:
            value = value.to(torch.float32)
        if value.max() > 1.0:
            value = value / 255.0
        observation[key] = value

    if OBS_STATE in frame:
        state_value = frame[OBS_STATE]
        if not isinstance(state_value, torch.Tensor):
            state_value = torch.as_tensor(state_value)
        if state_value.dim() == 0:
            state_value = state_value.unsqueeze(0)
        observation[OBS_STATE] = state_value.to(torch.float32)

    if "task" in frame:
        task_str = frame["task"]
        if isinstance(task_str, torch.Tensor):
            task_str = task_str.item() if task_str.numel() == 1 else str(task_str)
        observation["task"] = str(task_str)

    return observation


def main() -> int:
    register_third_party_plugins()
    args = _parse_args()

    rename_map = _parse_rename(args.rename)
    device = torch.device(args.device)
    output_dir = args.output_dir or f"./output/pi05_attention/ep{args.episode_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_repo_id}"
          + (f" (root={args.dataset_root})" if args.dataset_root else ""))
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    print(f"Loaded dataset with {dataset.num_episodes} episodes (fps={dataset.fps}).")
    if args.episode_id >= dataset.num_episodes:
        raise ValueError(
            f"--episode-id {args.episode_id} out of range (dataset has "
            f"{dataset.num_episodes} episodes)."
        )

    print(f"Loading policy: {args.policy_path}")
    policy, preprocessor, policy_cfg = _build_policy_and_preprocessor(
        args.policy_path, dataset, device
    )
    print(f"Policy type: {getattr(policy_cfg, 'type', '?')}; chunk_size="
          f"{getattr(policy_cfg, 'chunk_size', '?')}; num_inference_steps="
          f"{getattr(policy_cfg, 'num_inference_steps', '?')}.")

    wrapped = PI05PolicyWithAttention(
        policy=policy,
        preprocessor=preprocessor,
        target_layer_idx=args.target_layer,
        capture_strategy=args.capture_strategy,
    )
    print(
        f"Hooking expert layer {args.target_layer} | image_keys="
        f"{wrapped.image_keys} | tokens_per_image={wrapped.tokens_per_image}"
    )

    print(f"Filtering frames for episode {args.episode_id}...")
    episode_frames = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == args.episode_id
    )
    n = len(episode_frames)
    if n == 0:
        raise ValueError(f"Episode {args.episode_id} is empty.")
    if args.max_frames is not None:
        n = min(n, args.max_frames)
    print(f"Processing {n} frames.")

    per_cam_buffers: list[list[np.ndarray]] = [[] for _ in wrapped.image_keys]
    combined_buffer: list[np.ndarray] = []

    for i in tqdm(range(n), desc="Processing frames"):
        frame_idx = episode_frames[i]["index"].item()
        frame = dataset[frame_idx]

        observation = _frame_to_observation(frame, wrapped.image_keys, rename_map)
        _action, attn_maps = wrapped.select_action(observation)
        vis_per_cam = wrapped.visualize_attention(
            attention_maps=attn_maps, observation=observation
        )

        valid_frames: list[np.ndarray] = []
        for cam_idx, vis in enumerate(vis_per_cam):
            if vis is None:
                continue
            per_cam_buffers[cam_idx].append(vis)
            valid_frames.append(vis)

        if (
            len(valid_frames) == len(wrapped.image_keys)
            and all(f.shape[0] == valid_frames[0].shape[0] for f in valid_frames)
        ):
            combined_buffer.append(np.hstack(valid_frames))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    fps = int(dataset.fps)
    for cam_idx, key in enumerate(wrapped.image_keys):
        cam_name = key.split(".")[-1]
        out_path = os.path.join(
            output_dir, f"attention_ep{args.episode_id}_{cam_name}_{timestamp}.mp4"
        )
        encode_video_ffmpeg(per_cam_buffers[cam_idx], out_path, fps)

    if combined_buffer:
        out_path = os.path.join(
            output_dir, f"attention_ep{args.episode_id}_combined_{timestamp}.mp4"
        )
        encode_video_ffmpeg(combined_buffer, out_path, fps)

    print(f"Done. Outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Convert a LeRobot dataset into RL replay buffer frame format and save to disk.

Decodes video frames, resizes images, and writes every field (states, actions,
rewards, dones, complementary info) as flat numpy memmap files alongside a
metadata manifest. The resulting cache can be loaded via
ReplayBuffer.from_cache() to populate the replay buffer almost instantly.
Because the image data stays memory-mapped, it is paged in on demand by the OS
rather than loaded all at once, keeping RAM usage low even for large datasets.

How memory mapping works
------------------------
Each tensor (images, states, actions, …) is written to a separate .bin file
using np.memmap with mode="w+". This creates a file on disk whose bytes are the
raw array data in row-major order — no headers, no compression, just contiguous
dtype elements. Writing through a memmap means the OS can flush pages to disk
incrementally, so the full dataset never needs to live in RAM at once.

On the read side (ReplayBuffer.from_cache), the same .bin files are reopened
with np.memmap(mode="r") and wrapped in a torch tensor via torch.from_numpy().
The tensor shares the memmap's memory: no data is copied until the program
actually accesses a page. The OS virtual-memory subsystem loads pages on demand
and can evict them under memory pressure, which is why a 50 GB image cache can
be used on a machine with far less free RAM.

Small arrays (actions, rewards, dones, non-image state) are .clone()'d into
regular RAM after loading because they're cheap to hold and benefit from faster
random access. Image arrays stay memmap-backed so only the pages touched during
each training batch are resident.

A metadata.json file is written alongside the .bin files recording shapes,
dtypes, and a dataset fingerprint so the loader can reconstruct tensors with the
correct dimensions and verify cache validity.

Usage
-----
Step 1: Generate the cache from a dataset (one-time cost):

    uv run python src/lerobot/scripts/lerobot_memmap_buffer_cache.py \
        --repo-id jackvial/so101_pickplace_success_120_v2_with_subtasks \
        --data-dir outputs/so101_pickplace_success_120_v2_w_subtasks \
        --cache-dir outputs/buffer_cache \
        --video-backend pyav

    This decodes every video frame, resizes images to 224x224, and writes
    the result as .bin memmap files under outputs/buffer_cache/<fingerprint>/.

Step 2: Use the cache in training. The offline learner picks it up
automatically via the ``buffer_cache_dir`` config field, which defaults
to "outputs/buffer_cache". When ``ReplayBuffer.from_lerobot_dataset()``
is called with a ``cache_dir``, it checks for a matching fingerprint and
loads the memmap cache instead of re-decoding video. No code changes are
needed — just make sure the cache directory exists and contains the output
from step 1.

    You can also load a cache directly in code:

        from lerobot.rl.buffer import ReplayBuffer
        buf = ReplayBuffer.from_cache("outputs/buffer_cache/<fingerprint>", device="cuda:0")
        batch = buf.sample(batch_size=32, action_chunk_size=50)

Options:
    --repo-id        HuggingFace dataset repo ID (used if --data-dir is not set)
    --data-dir       Local path to the dataset root directory
    --cache-dir      Where to write the memmap cache (required)
    --video-backend  Video decoder: "pyav" (default) or "torchcodec"
    --num-workers    DataLoader workers for parallel decoding (default: min(4, cpu_count))
    --inject-golden  Tag every frame with is_golden=True (default: True)
"""

import argparse
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F_vision
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, REWARD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)


def dataset_fingerprint(dataset: LeRobotDataset) -> str:
    key = f"{dataset.root}|{dataset.meta.total_frames}|{dataset.meta.total_episodes}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def sanitize_key(key: str) -> str:
    return key.replace("/", "_")


def main():
    parser = argparse.ArgumentParser(description="Pre-decode dataset video frames into memmap cache")
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--video-backend", type=str, default="pyav", choices=["pyav", "torchcodec"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--inject-golden", action="store_true", default=True,
                        help="Inject is_golden=True for all frames (matches offline training)")
    args = parser.parse_args()

    if args.data_dir is None and args.repo_id is None:
        parser.error("Must provide --data-dir or --repo-id")

    logger.info("Loading dataset...")
    if args.data_dir:
        dataset = LeRobotDataset(
            repo_id=args.repo_id or "local/dataset",
            root=args.data_dir,
            video_backend=args.video_backend,
        )
    else:
        dataset = LeRobotDataset(repo_id=args.repo_id, video_backend=args.video_backend)

    num_frames = len(dataset)
    fingerprint = dataset_fingerprint(dataset)
    cache_path = Path(args.cache_dir) / fingerprint
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {num_frames} frames, fingerprint={fingerprint}")
    logger.info(f"Cache directory: {cache_path}")

    sample = dataset[0]

    state_keys = [k for k in sample if k.startswith("observation.")]
    complementary_info_keys = [k for k in sample if k.startswith("complementary_info.")]
    if "subtask_index" in sample:
        complementary_info_keys.append("subtask_index")

    has_done_key = DONE in sample
    if not has_done_key:
        logger.info("'next.done' not in dataset, inferring from episode boundaries")

    image_keys = [k for k in state_keys if "images" in k]
    non_image_state_keys = [k for k in state_keys if "images" not in k]

    logger.info(f"Image keys: {image_keys}")
    logger.info(f"Non-image state keys: {non_image_state_keys}")
    logger.info(f"Complementary info keys: {complementary_info_keys}")

    # Determine shapes after resize
    shapes = {}
    dtypes_np = {}

    for key in image_keys:
        shapes[key] = (3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        dtypes_np[key] = np.uint16  # bf16 stored as uint16

    for key in non_image_state_keys:
        val = sample[key]
        shapes[key] = tuple(val.shape)
        dtypes_np[key] = np.uint16

    action_val = sample[ACTION]
    if action_val.ndim == 2:
        action_val = action_val[0]
    shapes["actions"] = tuple(action_val.shape)
    dtypes_np["actions"] = np.uint16

    shapes["rewards"] = ()
    dtypes_np["rewards"] = np.uint16

    shapes["dones"] = ()
    dtypes_np["dones"] = np.bool_

    shapes["truncateds"] = ()
    dtypes_np["truncateds"] = np.bool_

    shapes["episode_ends"] = ()
    dtypes_np["episode_ends"] = np.bool_

    for key in complementary_info_keys:
        val = sample[key]
        if isinstance(val, torch.Tensor):
            shapes[f"complementary_info.{key}"] = tuple(val.shape)
            dtypes_np[f"complementary_info.{key}"] = np.uint16
        else:
            shapes[f"complementary_info.{key}"] = ()
            dtypes_np[f"complementary_info.{key}"] = np.uint16

    if args.inject_golden:
        shapes["complementary_info.is_golden"] = ()
        dtypes_np["complementary_info.is_golden"] = np.uint16

    # N = num_frames - 1 because transitions are (current, next) pairs,
    # but we also need the last frame's state for the final transition's next_state.
    # The buffer stores N transitions where N = num_frames (each frame becomes a transition).
    # Actually looking at the buffer code: it yields num_frames transitions total
    # (the last one has next_state = current_state when done).
    N = num_frames

    logger.info(f"Allocating memmap files for {N} transitions...")

    memmaps = {}
    for key, shape in shapes.items():
        safe_key = sanitize_key(key)
        full_shape = (N, *shape) if shape else (N,)
        fp = np.memmap(
            str(cache_path / f"{safe_key}.bin"),
            dtype=dtypes_np[key],
            mode="w+",
            shape=full_shape,
        )
        memmaps[safe_key] = fp

    # Set up data loader
    num_workers = args.num_workers or min(4, multiprocessing.cpu_count() or 1)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    def to_bf16_uint16(tensor: torch.Tensor) -> np.ndarray:
        return tensor.to(torch.bfloat16).view(torch.uint16).numpy()

    logger.info("Decoding frames...")
    iterator = iter(loader)
    prev_sample = next(iterator)
    idx = 0

    for current_sample in tqdm(iterator, total=num_frames - 1, desc="Decoding"):
        is_last = False
        _write_transition(
            memmaps, idx, prev_sample, current_sample, is_last,
            state_keys, image_keys, non_image_state_keys,
            complementary_info_keys, has_done_key, args.inject_golden,
            to_bf16_uint16,
        )
        prev_sample = current_sample
        idx += 1

    # Last transition
    _write_transition(
        memmaps, idx, prev_sample, None, True,
        state_keys, image_keys, non_image_state_keys,
        complementary_info_keys, has_done_key, args.inject_golden,
        to_bf16_uint16,
    )
    idx += 1

    # Flush all memmaps
    for mm in memmaps.values():
        mm.flush()
        del mm
    memmaps.clear()

    # Write metadata
    metadata = {
        "fingerprint": fingerprint,
        "num_transitions": idx,
        "dataset_root": str(dataset.root),
        "total_frames": dataset.meta.total_frames,
        "total_episodes": dataset.meta.total_episodes,
        "image_size": list(IMAGE_SIZE),
        "state_keys": state_keys,
        "image_keys": image_keys,
        "non_image_state_keys": non_image_state_keys,
        "complementary_info_keys": list(complementary_info_keys),
        "inject_golden": args.inject_golden,
        "shapes": {sanitize_key(k): list(v) if isinstance(v, tuple) else v for k, v in shapes.items()},
        "dtypes": {sanitize_key(k): np.dtype(v).str for k, v in dtypes_np.items()},
    }
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done! Wrote {idx} transitions to {cache_path}")
    logger.info(f"Cache size: {sum(f.stat().st_size for f in cache_path.iterdir()) / 1e9:.1f} GB")


def _write_transition(
    memmaps, idx, current_sample, next_sample, is_last,
    state_keys, image_keys, non_image_state_keys,
    complementary_info_keys, has_done_key, inject_golden,
    to_bf16_uint16,
):
    expected_h, expected_w = IMAGE_SIZE

    # State images
    for key in image_keys:
        tensor = current_sample[key]
        if tensor.shape[-2:] != (expected_h, expected_w):
            tensor = F_vision.resize(tensor, (expected_h, expected_w))
            tensor = tensor.clamp(0.0, 1.0)
        memmaps[sanitize_key(key)][idx] = to_bf16_uint16(tensor)

    # Non-image state
    for key in non_image_state_keys:
        memmaps[sanitize_key(key)][idx] = to_bf16_uint16(current_sample[key])

    # Action
    action = current_sample[ACTION]
    if action.ndim == 2:
        action = action[0]
    memmaps["actions"][idx] = to_bf16_uint16(action)

    # Done
    if has_done_key:
        done = bool(current_sample[DONE].item())
    else:
        done = False
        if is_last or next_sample is not None and next_sample["episode_index"] != current_sample["episode_index"]:
            done = True

    memmaps["dones"][idx] = done
    memmaps["truncateds"][idx] = done
    memmaps["episode_ends"][idx] = done

    # Reward
    if REWARD in current_sample:
        reward = current_sample[REWARD].item()
    else:
        reward = 1.0 if done else 0.0
    memmaps["rewards"][idx] = to_bf16_uint16(torch.tensor(reward, dtype=torch.float32))

    # Complementary info
    for key in complementary_info_keys:
        safe_key = sanitize_key(f"complementary_info.{key}")
        if safe_key not in memmaps:
            continue
        val = current_sample[key]
        if isinstance(val, torch.Tensor):
            memmaps[safe_key][idx] = to_bf16_uint16(val)
        else:
            memmaps[safe_key][idx] = to_bf16_uint16(torch.tensor(val, dtype=torch.float32))

    if inject_golden:
        safe_key = "complementary_info.is_golden"
        memmaps[safe_key][idx] = to_bf16_uint16(torch.tensor(True, dtype=torch.float32))


if __name__ == "__main__":
    main()

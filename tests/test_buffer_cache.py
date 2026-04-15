"""
Smoke-test for the memory-mapped ReplayBuffer cache (predecode → from_cache round-trip).

Run:
    uv run pytest tests/test_buffer_cache.py -v
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.rl.buffer import ReplayBuffer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_SIZE = (224, 224)


def _bf16_to_uint16(tensor: torch.Tensor) -> np.ndarray:
    return tensor.to(torch.bfloat16).view(torch.uint16).numpy()


def _sanitize(key: str) -> str:
    return key.replace("/", "_")


def _write_synthetic_cache(
    cache_dir: Path,
    *,
    num_transitions: int = 100,
    num_episodes: int = 5,
    image_keys: list[str] | None = None,
    non_image_state_keys: list[str] | None = None,
    action_dim: int = 6,
    inject_golden: bool = True,
) -> dict:
    """
    Write a minimal memmap cache that mirrors what predecode_dataset.py produces.
    Returns the raw tensors so callers can verify round-trip fidelity.
    """
    if image_keys is None:
        image_keys = ["observation.images.top"]
    if non_image_state_keys is None:
        non_image_state_keys = ["observation.state"]

    state_keys = image_keys + non_image_state_keys
    n = num_transitions
    ep_len = n // num_episodes

    shapes: dict[str, tuple] = {}
    dtypes_np: dict[str, np.dtype] = {}

    for key in image_keys:
        shapes[key] = (3, *IMAGE_SIZE)
        dtypes_np[key] = np.uint16

    for key in non_image_state_keys:
        shapes[key] = (14,)
        dtypes_np[key] = np.uint16

    shapes["actions"] = (action_dim,)
    dtypes_np["actions"] = np.uint16

    shapes["rewards"] = ()
    dtypes_np["rewards"] = np.uint16

    shapes["dones"] = ()
    dtypes_np["dones"] = np.bool_

    shapes["truncateds"] = ()
    dtypes_np["truncateds"] = np.bool_

    shapes["episode_ends"] = ()
    dtypes_np["episode_ends"] = np.bool_

    comp_keys: list[str] = []
    if inject_golden:
        shapes["complementary_info.is_golden"] = ()
        dtypes_np["complementary_info.is_golden"] = np.uint16
        comp_keys.append("is_golden")

    cache_dir.mkdir(parents=True, exist_ok=True)

    memmaps: dict[str, np.memmap] = {}
    for key, shape in shapes.items():
        safe = _sanitize(key)
        full_shape = (n, *shape) if shape else (n,)
        memmaps[safe] = np.memmap(
            str(cache_dir / f"{safe}.bin"),
            dtype=dtypes_np[key],
            mode="w+",
            shape=full_shape,
        )

    torch.manual_seed(42)
    raw: dict[str, torch.Tensor] = {}

    for key in image_keys:
        imgs = torch.rand(n, 3, *IMAGE_SIZE).clamp(0.0, 1.0)
        memmaps[_sanitize(key)][:] = _bf16_to_uint16(imgs)
        raw[key] = imgs.to(torch.bfloat16)

    for key in non_image_state_keys:
        states = torch.randn(n, 14)
        memmaps[_sanitize(key)][:] = _bf16_to_uint16(states)
        raw[key] = states.to(torch.bfloat16)

    actions = torch.randn(n, action_dim)
    memmaps["actions"][:] = _bf16_to_uint16(actions)
    raw["actions"] = actions.to(torch.bfloat16)

    rewards = torch.zeros(n)
    dones = torch.zeros(n, dtype=torch.bool)
    for ep in range(num_episodes):
        end_idx = min((ep + 1) * ep_len - 1, n - 1)
        dones[end_idx] = True
        rewards[end_idx] = 1.0

    memmaps["rewards"][:] = _bf16_to_uint16(rewards)
    memmaps["dones"][:] = dones.numpy()
    memmaps["truncateds"][:] = dones.numpy()
    memmaps["episode_ends"][:] = dones.numpy()

    raw["rewards"] = rewards.to(torch.bfloat16)
    raw["dones"] = dones
    raw["episode_ends"] = dones

    if inject_golden:
        golden = torch.ones(n)
        memmaps["complementary_info.is_golden"][:] = _bf16_to_uint16(golden)
        raw["complementary_info.is_golden"] = golden.to(torch.bfloat16)

    for mm in memmaps.values():
        mm.flush()

    metadata = {
        "fingerprint": "test_fingerprint_1234",
        "num_transitions": n,
        "dataset_root": "/tmp/fake_dataset",
        "total_frames": n,
        "total_episodes": num_episodes,
        "image_size": list(IMAGE_SIZE),
        "state_keys": state_keys,
        "image_keys": image_keys,
        "non_image_state_keys": non_image_state_keys,
        "complementary_info_keys": comp_keys,
        "inject_golden": inject_golden,
        "shapes": {
            _sanitize(k): list(v) if isinstance(v, tuple) else v
            for k, v in shapes.items()
        },
        "dtypes": {
            _sanitize(k): np.dtype(v).str for k, v in dtypes_np.items()
        },
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return raw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def cache_dir():
    d = Path(tempfile.mkdtemp(prefix="buffer_cache_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestFromCache:
    """Verify ReplayBuffer.from_cache loads memmap data correctly."""

    def test_loads_without_error(self, cache_dir):
        _write_synthetic_cache(cache_dir, num_transitions=50)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        assert buf.initialized
        assert buf.size == 50
        assert len(buf) == 50

    def test_state_keys_loaded(self, cache_dir):
        _write_synthetic_cache(cache_dir, num_transitions=50)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        assert "observation.images.top" in buf.states
        assert "observation.state" in buf.states

    def test_image_data_fidelity(self, cache_dir):
        raw = _write_synthetic_cache(cache_dir, num_transitions=50)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        loaded = buf.states["observation.images.top"]
        expected = raw["observation.images.top"]
        assert loaded.shape == expected.shape
        assert loaded.dtype == torch.bfloat16
        assert torch.equal(loaded, expected)

    def test_state_data_fidelity(self, cache_dir):
        raw = _write_synthetic_cache(cache_dir, num_transitions=50)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        loaded = buf.states["observation.state"]
        expected = raw["observation.state"]
        assert torch.equal(loaded, expected)

    def test_actions_fidelity(self, cache_dir):
        raw = _write_synthetic_cache(cache_dir, num_transitions=50)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        assert torch.equal(buf.actions, raw["actions"])

    def test_dones_and_rewards(self, cache_dir):
        raw = _write_synthetic_cache(cache_dir, num_transitions=50)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        assert buf.dones.dtype == torch.bool
        assert torch.equal(buf.dones, raw["dones"])
        assert torch.equal(buf.rewards, raw["rewards"])

    def test_complementary_info(self, cache_dir):
        _write_synthetic_cache(cache_dir, num_transitions=50, inject_golden=True)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        assert buf.has_complementary_info
        assert "is_golden" in buf.complementary_info
        assert buf.complementary_info["is_golden"].shape == (50,)

    def test_no_complementary_info(self, cache_dir):
        _write_synthetic_cache(cache_dir, num_transitions=50, inject_golden=False)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        assert not buf.has_complementary_info

    def test_sample_after_load(self, cache_dir):
        _write_synthetic_cache(cache_dir, num_transitions=200, num_episodes=10)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu", use_drq=False)
        batch = buf.sample(batch_size=8, action_chunk_size=1)
        assert batch["action"].shape[0] == 8
        assert "observation.images.top" in batch["state"]
        assert "observation.state" in batch["state"]

    def test_optimize_memory_next_states(self, cache_dir):
        _write_synthetic_cache(cache_dir, num_transitions=50)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        assert buf.next_states is buf.states

    def test_missing_metadata_raises(self, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="No metadata.json"):
            ReplayBuffer.from_cache(cache_dir, device="cpu")

    def test_position_wraps(self, cache_dir):
        _write_synthetic_cache(cache_dir, num_transitions=100)
        buf = ReplayBuffer.from_cache(cache_dir, device="cpu")
        assert buf.position == 0  # 100 % 100 == 0
        assert buf.size == 100

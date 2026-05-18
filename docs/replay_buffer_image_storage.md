# Replay Buffer Image Storage Changes

This note documents the replay-buffer image storage changes made for the pi05 / MolmoAct2 split.

## Summary

The replay buffer now separates **how images are stored** from **how each policy preprocesses images**.

Previously, the offline dataset path was effectively pi05-specific: images were resized to `224x224` inside `ReplayBuffer.from_lerobot_dataset(...)` and stored as `bfloat16`. That worked for pi05, but it was a poor fit for MolmoAct2, which wants raw-resolution images.

The buffer now has explicit image storage options:

```python
image_storage_dtype: str = "bfloat16"  # "bfloat16" or "uint8"
image_storage_size: tuple[int, int] | None = None  # None means raw resolution
```

For the RL policy configs touched here, the default is now:

```python
image_storage_dtype = "uint8"
image_storage_size = None
```

That means actor-fed online data and offline dataset data can keep raw camera resolution while using one byte per channel instead of bf16's two bytes per channel.

## What Changed

### 1. `ReplayBuffer` has image storage policy

File: `src/lerobot/rl/buffer.py`

`ReplayBuffer.__init__` now accepts:

```python
image_storage_dtype="bfloat16"
image_storage_size=None
```

Images are handled specially at `ReplayBuffer.add(...)` time:

- If `image_storage_dtype="uint8"`:
  - `uint8` inputs are stored as-is.
  - floating inputs in `[0, 1]` are converted to `[0, 255]` and stored as `torch.uint8`.
  - byte-scale floating inputs are clamped/rounded into `uint8`.
- If `image_storage_dtype="bfloat16"`:
  - images keep the old normalized tensor style and are stored as `torch.bfloat16`.
- If `image_storage_size is None`:
  - the input resolution is preserved.
- If `image_storage_size=(H, W)`:
  - images are resized before storage.

Non-image state, actions, rewards, and complementary info still use the old bf16-oriented storage path.

### 2. The hardcoded offline `224x224` resize was removed

File: `src/lerobot/rl/buffer.py`

`ReplayBuffer.from_lerobot_dataset(...)` no longer manually resizes images to `224x224`. Instead, it streams transitions into `ReplayBuffer.add(...)`, and `add(...)` applies the configured image storage policy.

This is the key design change: the dataset loader no longer decides policy-specific image resolution.

### 3. Offline memmap caches support raw `uint8`

File: `src/lerobot/scripts/lerobot_memmap_buffer_cache.py`

The cache builder now accepts:

```bash
--image-storage-dtype bfloat16|uint8
--image-storage-size HEIGHT WIDTH
--image-storage-size raw
```

Examples:

```bash
# Old-style pi05-ish cache: 224x224 bf16
python src/lerobot/scripts/lerobot_memmap_buffer_cache.py \
  --data-dir /path/to/dataset \
  --cache-dir outputs/buffer_cache \
  --image-storage-dtype bfloat16 \
  --image-storage-size 224 224
```

```bash
# Raw uint8 cache for MolmoAct2 / raw-image replay
python src/lerobot/scripts/lerobot_memmap_buffer_cache.py \
  --data-dir /path/to/dataset \
  --cache-dir outputs/buffer_cache \
  --image-storage-dtype uint8 \
  --image-storage-size raw \
  --flush-every 512 \
  --drop-cache
```

A raw `480x640` RGB image stored as `uint8` costs:

```text
480 * 640 * 3 * 1 byte = 921,600 bytes ~= 0.92 MB per frame per camera
```

The old raw bf16 equivalent would be about `1.84 MB`, so `uint8` halves the raw-image cost.

### 4. Cache fingerprints include image storage settings

File: `src/lerobot/rl/buffer.py`

The cache fingerprint now includes:

- dataset root
- total frames
- total episodes
- selected state keys
- image storage dtype
- image storage size
- cache schema version

This prevents accidentally loading a stale `224/bf16` cache when the run expects `raw/uint8`.

Backward compatibility remains for old `224/bf16` caches using the previous fingerprint format.

### 5. `ReplayBuffer.from_cache(...)` loads image dtype honestly

File: `src/lerobot/rl/buffer.py`

Old cache images were stored as `np.uint16` and viewed as `torch.bfloat16`.

New raw image caches can store images as `np.uint8`; `from_cache(...)` now keeps those tensors as `torch.uint8` instead of viewing them as bf16.

Image arrays remain memmap-backed for offline caches, so the whole image dataset does not need to live in RAM.

## Wiring Status

Yes, the changes are wired into the main online/offline RL buffer construction paths.

### Online buffer

Files:

- `src/lerobot/rl/rl_learner.py`
- `src/lerobot/rl/learner.py`

The empty actor-fed online replay buffer now receives:

```python
image_storage_dtype=getattr(cfg.policy, "image_storage_dtype", "bfloat16")
image_storage_size=getattr(cfg.policy, "image_storage_size", None)
```

So for policies with the new defaults, online actor images are stored as raw `uint8`.

### Offline buffer

Files:

- `src/lerobot/rl/rl_learner.py`
- `src/lerobot/rl/learner.py`
- `src/lerobot/rl/learner_pi05.py`
- `src/lerobot/scripts/offline_learner_pi05.py`
- `src/lerobot/scripts/rl_offline.py`

Offline `ReplayBuffer.from_lerobot_dataset(...)` calls now pass through:

```python
image_storage_dtype=getattr(cfg.policy, "image_storage_dtype", "bfloat16")
image_storage_size=getattr(cfg.policy, "image_storage_size", (224, 224))
```

The fallback preserves old behavior for policy configs that do not define the new fields. The RL configs updated in this change do define them, so they use raw `uint8` by default.

### Policy config defaults

Files:

- `src/lerobot/rl/rl_molmoact2.py`
- `src/lerobot/rl/rl_pi05.py`

Both now include:

```python
image_storage_dtype: str = "uint8"
image_storage_size: tuple[int, int] | None = None
```

That means the RL path defaults to raw-resolution `uint8` image storage unless overridden.

## Intended Usage Now

### Online actor data

No special caller changes are needed. The actor/learner path still calls:

```python
replay_buffer.add(...)
```

The buffer decides image storage format internally.

If the policy config has:

```python
image_storage_dtype = "uint8"
image_storage_size = None
```

then actor images are stored raw as `uint8` in the online replay buffer.

### Offline dataset without prebuilt cache

No special caller changes are needed. The learner still calls:

```python
ReplayBuffer.from_lerobot_dataset(...)
```

If no cache is found, the dataset is decoded and transitions are inserted through `ReplayBuffer.add(...)`, which applies the configured storage policy.

### Offline dataset with prebuilt cache

Build the cache once with matching image storage settings:

```bash
python src/lerobot/scripts/lerobot_memmap_buffer_cache.py \
  --data-dir /path/to/dataset \
  --cache-dir outputs/buffer_cache \
  --image-storage-dtype uint8 \
  --image-storage-size raw
```

For very large raw caches, prefer the streamed writer controls shown above. `--drop-cache` syncs periodically and asks Linux to evict already-written output pages, which keeps the cache build from growing RAM through dirty page cache. `--flush-every` controls how often that happens; smaller values reduce memory pressure and can slow the build.

Then train with `cfg.buffer_cache_dir` pointing at `outputs/buffer_cache`.

`ReplayBuffer.from_lerobot_dataset(...)` will look for a cache whose fingerprint matches the dataset and image storage settings. If found, it loads the memmap cache. If not found, it falls back to decoding the dataset directly.

## Policy Preprocessing Responsibility

The buffer no longer guarantees that images are already resized for a model.

That is intentional.

- pi05 preprocessing already handles `uint8` images and resizes/pads to the configured pi05 image resolution before model use.
- MolmoAct2 preprocessing normalizes images into `uint8` numpy-style image inputs for its processor, so raw `uint8` storage is natural.

In other words:

```text
ReplayBuffer: stores observations efficiently.
Policy preprocessing: converts observations into model inputs.
```

## Expected Benefits

For raw `480x640` RGB images:

- raw bf16: about `1.84 MB/frame/camera`
- raw uint8: about `0.92 MB/frame/camera`
- old `224x224` bf16: about `0.30 MB/frame/camera`

So raw `uint8` is still larger than `224/bf16`, but it is roughly half the size of raw bf16 and preserves full image resolution.

For offline datasets, using memmap means those raw images live on disk and are paged in by the OS when sampled, instead of being fully loaded into process RAM.

## Important Caveats

- Random replay sampling from memmap is random disk IO. NVMe should be much better than HDD or network storage.
- `uint8` images sampled from the buffer remain `torch.uint8` unless DrQ augmentation is enabled. Model preprocessors must handle `uint8` correctly.
- DrQ augmentation converts image batches to float `[0, 1]` before applying augmentation.
- The cache builder now streams output files instead of writing through output memmaps. Use `--drop-cache` for huge raw caches when RAM pressure matters.
- Online data is still an in-memory circular replay buffer. It is smaller now because images can be `uint8`, but it is not yet disk-backed.
- Disk-backed online replay was intentionally not implemented in this pass. The safer first step is fixed offline memmap plus uint8 online RAM storage.

## Files Touched

Main implementation:

- `src/lerobot/rl/buffer.py`
- `src/lerobot/scripts/lerobot_memmap_buffer_cache.py`

Wiring:

- `src/lerobot/rl/rl_learner.py`
- `src/lerobot/rl/learner.py`
- `src/lerobot/rl/learner_pi05.py`
- `src/lerobot/scripts/offline_learner_pi05.py`
- `src/lerobot/scripts/rl_offline.py`
- `src/lerobot/rl/offline_dataset_utils.py`
- `src/lerobot/rl/pi05_train_utils.py`

Config defaults:

- `src/lerobot/rl/rl_molmoact2.py`
- `src/lerobot/rl/rl_pi05.py`

Focused tests added but not run here:

- `tests/test_buffer_cache.py`
- `tests/utils/test_replay_buffer.py`

## Verification Done

A syntax-only compile check was run on the edited Python files with `python3 -m py_compile`.

The pytest suite was not run, per instruction.

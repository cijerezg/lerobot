# Fixing LeRobot Data Studio Subtask Export

## Summary

The missing `subtask_index` column is very likely coming from `../lerobot-data-studio`.

`lerobot-data-studio` currently preserves subtask annotations as metadata files, but it does not materialize those annotations into the frame data that training consumes:

- It writes `meta/skills.json`.
- It writes `meta/subtasks.parquet`.
- It does not write `subtask_index` into `data/chunk-*/file-*.parquet`.
- It does not declare `subtask_index` in `meta/info.json`.

That produces a dataset that looks annotated in Data Studio, but the training buffer sees no complementary subtask signal.

## Where The Bug Is

Relevant files in `../lerobot-data-studio`:

- `src/lerobot_data_studio/backend/background_tasks.py`
- `src/lerobot_data_studio/backend/subtask_annotations.py`
- `src/lerobot_data_studio/backend/trimmed_dataset_export.py`
- `src/lerobot_data_studio/backend/idle_trim.py`

The export flow calls:

```python
export_subtask_annotations(
    source_dataset=dataset,
    destination_dataset=filtered_dataset,
    episode_mapping=episode_mapping,
    keep_time_ranges=keep_time_ranges or None,
)
```

`export_subtask_annotations()` rebases the annotation ranges and writes only:

```python
write_skills_json(destination_dataset, payload)
write_subtasks_parquet(destination_dataset, payload.skill_to_subtask_index)
```

The frame parquet and feature schema are created earlier. For the trimmed copied-video path, `trimmed_dataset_export.py` creates the destination metadata with:

```python
destination_meta = LeRobotDatasetMetadata.create(
    repo_id=new_repo_id,
    fps=source.meta.fps,
    features=source.meta.features,
    ...
)
```

Then `_copy_trimmed_data()` writes copied frame rows. Since `source.meta.features` does not include `subtask_index`, the destination `meta/info.json` does not include it. Since `_copy_trimmed_data()` only copies existing frame columns, the destination parquet does not include it either.

There is a similar issue in the `idle_trim.py` path: `_frame_for_add()` intentionally drops keys not declared in `source.features`. So even if `dataset.__getitem__()` had extra annotation info, it would be stripped unless `subtask_index` is a declared feature.

## Correct Fix

Fix the export path so annotated datasets always contain three consistent pieces:

1. `meta/skills.json`
2. `meta/subtasks.parquet`
3. Per-frame `subtask_index` in `data/chunk-*/file-*.parquet`, declared in `meta/info.json`

The best place is immediately after `export_subtask_annotations()` in `background_tasks.py`, before `filtered_dataset.push_to_hub(...)`.

## Implementation Plan

Add a helper to `src/lerobot_data_studio/backend/subtask_annotations.py`:

```python
SUBTASK_INDEX_FEATURE = {
    "dtype": "int64",
    "shape": [1],
    "names": None,
}
```

Add a function such as:

```python
def materialize_subtask_index_feature(
    dataset: LeRobotDataset,
    payload: SubtaskAnnotationsResponse,
) -> None:
    ...
```

That function should:

1. Return early if `payload` is `None` or has no episodes.
2. Read `dataset.root / "meta" / "info.json"`.
3. Add `features["subtask_index"] = SUBTASK_INDEX_FEATURE` if missing.
4. Update `dataset.meta.features["subtask_index"]` as well, if available.
5. Iterate all `dataset.root / "data" / "*/*.parquet"` files.
6. For each row, assign `subtask_index` by matching row `episode_index` and relative `timestamp` against `payload.episodes[str(episode_index)].skills`.
7. Write the parquet back with the new `subtask_index` column.

Use episode-relative timestamps. The exported `skills.json` ranges are already rebased by `export_subtask_annotations()`, so row timestamps in the destination dataset should be compared directly to each skill's `start` and `end`.

Recommended assignment behavior:

- Use `start <= timestamp < end` for normal frames.
- Clamp exact trailing edge frames to the last skill in the episode.
- If a tiny floating point gap leaves a frame unassigned, fill from the nearest previous or next skill and log a warning.

## Wire It Into Export

In `background_tasks.py`, change the current call from:

```python
export_subtask_annotations(
    source_dataset=dataset,
    destination_dataset=filtered_dataset,
    episode_mapping=episode_mapping,
    keep_time_ranges=keep_time_ranges or None,
)
```

to:

```python
subtask_payload = export_subtask_annotations(
    source_dataset=dataset,
    destination_dataset=filtered_dataset,
    episode_mapping=episode_mapping,
    keep_time_ranges=keep_time_ranges or None,
)
materialize_subtask_index_feature(filtered_dataset, subtask_payload)
```

Import the new helper:

```python
from .subtask_annotations import (
    export_subtask_annotations,
    materialize_subtask_index_feature,
    sync_subtask_metadata_from_repo,
)
```

## Tests To Add

Add coverage in `tests/test_subtask_annotations.py`:

- Create a fake destination dataset with a real `meta/info.json`.
- Create a small `data/chunk-000/file-000.parquet` with `episode_index`, `timestamp`, and frame columns.
- Call `export_subtask_annotations(...)`.
- Call `materialize_subtask_index_feature(...)`.
- Assert:
  - `subtask_index` exists in the data parquet.
  - `subtask_index` values match expected skill ranges.
  - `meta/info.json` contains `features.subtask_index`.

Add or extend coverage in `tests/test_trimmed_dataset_export.py`:

- Export a trimmed dataset with source subtask annotations.
- Materialize subtasks after export.
- Assert the trimmed/rebased timestamps produce the expected per-frame `subtask_index` values.

## Validation Commands

After patching `../lerobot-data-studio`, create a new exported dataset and run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("outputs/YOUR_EXPORTED_DATASET")
info = json.loads((root / "meta/info.json").read_text())
df = pd.read_parquet(root / "data/chunk-000/file-000.parquet")

print("has info feature:", "subtask_index" in info["features"])
print("has parquet column:", "subtask_index" in df.columns)
print(df["subtask_index"].value_counts().sort_index())
print(pd.read_parquet(root / "meta/subtasks.parquet"))
PY
```

Then regenerate the memmap cache:

```bash
rm -rf outputs/buffer_cache/<fingerprint>

uv run python src/lerobot/scripts/lerobot_memmap_buffer_cache.py \
  --repo-id YOUR_REPO_ID \
  --data-dir outputs/YOUR_EXPORTED_DATASET \
  --cache-dir outputs/buffer_cache \
  --video-backend pyav
```

The important log line should be:

```text
Complementary info keys: ['subtask_index']
```

If it still says:

```text
Complementary info keys: []
```

then either the parquet column is still missing, or `meta/info.json` still does not declare it.

## One-Time Repair For Existing Datasets

For already-exported datasets, use the same logic as `prepare_fixed_subtask_cache.sh` in this repo:

```bash
./prepare_fixed_subtask_cache.sh
```

That repairs the fixed dataset locally and regenerates the memmap cache. The long-term fix should still go into `../lerobot-data-studio` so future exports are correct without manual repair.

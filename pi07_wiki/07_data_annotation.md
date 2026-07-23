# 07 — Data & annotation (reference)

## Datasets

- `outputs/rebot_socks_v1` — raw: 6 eps, 29,192 frames @ 30 fps, top + wrist RGB, wrist depth sidecar.
- `outputs/rebot-socks-annotated-v2` — **training-ready**: `subtask_index` per frame, 13-label canonical vocab, 81 hand-written summaries (12 s grid), quality/mistake metadata, depth hardlinked.
- Everything SO-101-era is legacy 6-dim; don't mix.

Episode GT for eval sanity: ep0 white; ep1 white,black; ep2 white×2,black; ep3 +blue-heeled; ep4 +blue-striped, mistake ~162→185 s; ep5 interleaved, mistake ~176→205 s.

## Chain (scripts in `data_processing/annotate/`)

Order matters: `summary_annotate.py` (12 s grid) → `subtask_annotate_grid.py` (4 s grid, conditions on summaries; labels atomic + progress-free) → `metadata_annotate.py annotate` + `review` (suspicion 0–10, `--threshold 4`; review UI writes the meta files). Validation sets: `--reuse-map <train-root>`. AV1 videos → ffmpeg, not cv2.

Key invariants:
- 4 s subtask grid = `subtask_regeneration_interval`; 12 s summary grid = 3:1.
- Post-annotation validation required (the empty-label → "Subtask: ;" bug class).
- v2 provenance: Gemma subtasks hand-corrected (91/242 changed, originals kept as `gemma_subtask`); summaries fully manual.

## Annotator verdicts (2026-07-18)

Gemma 4 31B: summaries unusable (counting echo loop); subtasks ~62% (misses releases, colors in clutter, fumbles→"move"). Upgrades untried: GLM-4.6V FP8 (DGX), Molmo2-8B.

## Per-new-dataset prep

annotate → norm stats + `compute_delta_stats.py --encoding anchor --chunk-size 50` → memmap cache `--image-stride 5` (must match config) → verify first steps: `loss_subtask_ce` nonzero, `subtask_index`/`summary_*_index`/`metadata_*` in batch, depth one-shot log fires.

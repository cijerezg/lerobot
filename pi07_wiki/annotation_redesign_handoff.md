# Annotation redesign — handoff (2026-08-02)

## Goal

Replace episode-level quality + fixed-grid subtasks with **per-subtask-segment**
labels, so the steering metadata varies at the scale the policy actually sees
(chunk = 1.0 s). Motivation: only **12.1%** of training frames carried quality=5, the
label every rollout prompts, and quality 1-2 never occurred at all.

Three channels, all on semantic spans:
1. **Subtasks** — semantic phases, not a 4 s grid. DONE (see below).
2. **Quality 1-5** — per segment. Mistake ⇒ bad quality, but quality degrades without
   mistakes too (nested, not orthogonal).
3. **Mistakes** — semantic spans: begins where alignment failed, ends after the missed
   grasp. Retrospective judgement.

Summaries: keep the existing data, stop generating for new datasets. Generation is
already off (`subtask_max_new_tokens: 0`, `subtask_loss_weight: 0.0`).

## State: corpus consolidated and verified

Merged **by task**, 9 sessions → 3 train sources + 1 val. Originals untouched.

| dataset | eps | frames |
|---|---|---|
| `outputs/rebot_socks_basket-v1` (v1 eps0-8 + socks-annotated-v2) | 15 | 68,635 |
| `outputs/rebot_shirts_bin-v1` (v2 + v2-2) | 9 | 30,822 |
| `outputs/rebot_two_container-v1` (v3 + v3-1 + v3-2 + v3-3 eps1-4) | 14 | 66,283 |
| `outputs/rebot_val-v1` (v1 ep9 + v2-1 + v3-3 ep0) | 3 | 14,021 |

Train 165,740 / 38 eps. Sampling spread at B=32: **3.64x → 1.04x**. Depth is
hardlinked. Scripts: `migration/consolidate_stage_{a,b,c,c2}.py`,
`migration/verify_consolidation.py` (8 checks, all pass: state/action bit-identical,
depth filename sets + inodes, timestamps, video). Provenance is by **content
fingerprint**, in `outputs/_staging/provenance.json`.

## Segmentation: DONE

`lerobot/src/lerobot/data_processing/annotate/semantic_segment.py`, already run on all
four merged sets → `meta/subtask_windows.semantic.json`.

Phase model (one cycle = one gripper-closed interval), vocabulary unchanged (28 labels):

    grasp the X          [end of prev release, close)   approach + closing
    move the X to the C  [close, arrival at C)          shut, travelling
    release the X in C   [arrival at C, opening settles) deposit
    return to home       [last release, episode end)    final park only

Two documented judgement calls: `grasp` absorbs container→next-object transit (tested
for a turnaround — there is none, it's one continuous sweep); `release` starts at
arrival over the container, not at the gripper opening (mechanical open is 0.6 s, too
tight to condition on). Failed closes stay *inside* the enclosing grasp segment.

**767 segments** — grasp 47.3% / move 30.5% / release 13.4% / return 8.8%.
242 grasp + 242 move + 242 release + 41 return.

## Next: the rubric, then labeling

Write the rubric first, get it reviewed, THEN label. All labeling is done by the
assistant for consistency.

Measured anchors (use these, don't invent thresholds):

- **failed closes inside grasp segments**: 0→193, 1→36, 2→9, 3+→4. So ~20% of grasp
  segments contain a failure — a well-populated bad class.
- **path efficiency** (net joint displacement / total path length) is the directness
  proxy. Medians: grasp 0.57, move 0.69, release 0.66, return 0.83.
- **duration** medians: grasp 11.1 s, move 7.9 s, release 2.6 s, return 9.3 s.
- **`pause_frac` is dead** — median 0.00 everywhere. Teleop is continuous. Don't use it.
- 69 failed closes total; approach→reopen span median **1.7 s** (p90 3.3 s), which is
  the natural mistake-span length.

Calibrate so quality=5 lands around 40% of segments, not 12%. Precision matters most
at the bottom of the scale (failure / no failure), least at the 4-vs-5 boundary.

**114 segments need a vision pass for object identity** — the 5 v3-3 episodes, which
were never annotated (87 in two_container, 27 in val). They carry `<object>`
placeholders. Everywhere else object identity was carried over from the old 4 s labels
by majority vote, so no vision needed.

## Gotchas

- **`rebot_sorting_clothes_v3-4` is DEAD** — its parquet was destroyed 2026-08-02
  (state/action gone; video+depth survive but are useless). Excluded from the corpus.
  Its role as two-container val episode is filled by v3-3 ep0.
- **Never modify a dataset in place.** Write to a new dir; read a backup before
  trusting it. LeRobot v3 layouts have basename collisions (`data/chunk-000/file-000.parquet`
  vs `meta/episodes/chunk-000/file-000.parquet`) — that collision is what killed v3-4.
- `aggregate_datasets` carries **no** depth and **no** `meta/*` annotation file.
- 7 episodes (~20k train frames) have 2nd-generation wrist video at 42.6-45.2 dB from
  the split re-encode. `top` is byte-identical everywhere.
- Downstream after annotation: norm stats + `compute_delta_stats.py --encoding anchor`,
  memmap cache, then `config_rl.yaml` (sources, weights, `val_dataset_path`).
  Use **chunk_size 30 / image_stride 3** — `07_data_annotation.md` still says 50/5 and
  is stale.

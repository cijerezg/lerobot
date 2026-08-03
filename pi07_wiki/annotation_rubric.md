# Annotation rubric — quality + mistakes on semantic segments (2026-08-02)

Companion to `annotation_redesign_handoff.md`. Segmentation is done (767 segments,
`meta/subtask_windows.semantic.json`); this defines the two remaining channels.

## What the label means

The policy reads exactly this at frame $t$:

    The task is to {task}. The current step is {subtask}. ... The quality is {q} of 5.
    The robot made {a mistake | no mistakes}. Given these, what action should the robot take?

Every rollout prompts `quality=5, mistake=False`. So the operative definition is not
"how did this look" but:

> **quality 5 = the action stream in this segment is what we want the policy to emit
> when it is told the execution is perfect.** Everything else is a graded statement of
> how far the actions deviate from that.

Both channels are retrospective — the full episode is visible, so a grasp is judged by
whether it *did* succeed, not by whether it looked plausible at the time.

## Units

| channel | unit | storage |
|---|---|---|
| quality 1-5 | one value per semantic segment, constant across it | `meta/episode_metadata.parquet`, one row per **segment** |
| mistake | span, strictly inside one segment | `meta/mistakes.parquet`, one row per **event** |

Quality is **constant across the segment** — decided 2026-08-02 with the cost known.
A failure-containing grasp segment runs 15-26 s against a ~1.7 s failure, so ~13% of
all frames are clean recovery carrying quality 1-2, and the policy never sees recovery
under the quality=5 prompt every rollout gets. Accepted because it keeps quality 1:1
with a subtask label, and because it lands the distribution on target rather than off
it: failure-containing grasp segments are 14.9% of frames against a 20% budget for
quality 1-2.

`materialize_metadata` and `frame_metadata_lookup` both broadcast by row
`from_index`/`to_index` and never assume one row per episode, so per-segment quality
rows need **no code change** — only the two docstrings that say "per episode" go stale.

## Quality 1-5

Nested, not orthogonal: a mistake forces 1-2, but 3 is reachable with no discrete failure.

| q | definition |
|---|---|
| **5** | Direct. Goes to the target and does the thing. No failure, no visible correction. |
| **4** | Succeeds with one minor correction — a single hesitation, a small overshoot pulled back, a wrist adjustment before closing. Still clearly purposeful throughout. |
| **3** | Succeeds, but laboured: repeated corrections, a wandering search over the object, the object shoved/dragged into a better pose before the grasp. **No discrete failure event.** |
| **2** | Contains exactly one failure event, recovered inside the same segment. |
| **1** | Two or more failure events, or one failure not recovered inside the segment (object left displaced, delivered to the wrong container, unintended release). |

### Per-stratum anchors (measured, $n=767$)

Guides for the 5-vs-4-vs-3 calls, not overrides — vision decides, these catch drift.
`eff` = net joint displacement / path length; `rev` = smoothed pan direction reversals.

Pooling by phase alone is **wrong**, for two structural reasons measured in the corpus:

- `grasp` absorbs the container→next-object transit by design, so the first grasp of an
  episode has no transit in it: clean `eff` median **0.78** against **0.57** for later
  grasps. It also unparks, which costs reversals: `rev` median 3 against 1. A pooled
  `eff ≥ 0.60, rev ≤ 2` would hand every first grasp a 5 on `eff` (p10 = 0.70) and then
  take it away again on `rev` (p25 = 3).
- The bin sits at negative pan, outside the top camera view, and shirts go in over
  several carries — so release-into-bin is structurally slower and less direct than
  release-into-basket (`eff` 0.56 / 3.5 s vs 0.71 / 2.3 s).

`move` and `return` are stable across datasets and containers and stay pooled.

| stratum | n | 5 | 4 | 3 | duration flag |
|---|---|---|---|---|---|
| grasp / first in episode | 41 | eff ≥ 0.78 | eff 0.72-0.78 | eff < 0.72 | > 17 s |
| grasp / later | 201 | eff ≥ 0.57, rev ≤ 2 | eff 0.42-0.57, or rev 3-4 | eff < 0.42 | > 14 s |
| move | 242 | eff ≥ 0.69 | eff 0.62-0.69, or a visible pause/wobble | eff < 0.62, or the object visibly shifts in the gripper but is retained | > 10 s |
| release / basket | 148 | ≤ 2.3 s | 2.3-3.4 s, or lands on the rim and settles in | > 3.4 s, repositioning over the container before opening | — |
| release / bin | 94 | ≤ 3.5 s | 3.5-6.5 s | > 6.5 s | — |
| return | 41 | eff ≥ 0.83, direct park | eff 0.62-0.83, one detour | eff < 0.62, wandering before parking | > 27 s |

The `5` and `4` columns are the p50 and p20 of each stratum's **failure-free**
segments, so the anchors alone put ~50% of clean segments at 5. With 80% of segments
clean that is ~40% at quality 5 — the calibration target, reached by construction
rather than by nudging.

The duration flag is not a grade. It marks a segment as **worth a hard second look for
an unflagged failure**: failure-containing grasps run 15.4 s (one failure) and 25.7 s
(two) against 9.8 s clean, so a long segment with no proprio flag is where a slip,
knock or drop is hiding.

**Release is graded on duration, not `eff`.** A deposit is ~75 frames in which the arm
is nearly stationary, so net/path is a ratio of two small numbers: it correlates −0.46
with duration, its IQR widens from 0.17 to 0.29 from the shortest to the longest
quartile, and its quartile medians are not even monotonic (0.75 / 0.62 / 0.68 / 0.51).
Duration says the same thing without the noise — a long release *is* repositioning over
the container. `move` and `grasp` are long enough for `eff` to be stable.

`pause_frac` is dead (median 0.00 everywhere) — teleop is continuous. Not used.

## Mistakes

A mistake is a **discrete visible failure event**, not a stretch of clumsiness.
Clumsy-but-successful is quality 3 with `mistake=False`. This is the boundary that
keeps the boolean channel meaningful.

| type | event |
|---|---|
| `failed_close` | gripper closes on nothing, or loses the object during the close |
| `slip` | object escapes the gripper after a successful close, mid-carry |
| `drop` | unintended release outside a container |
| `knock` | arm displaces an object or a container without grasping it |
| `wrong_target` | wrong object grasped, or delivered to the wrong container |

`mistake_type` is stored for analysis; the prompt only ever sees the boolean.

### Span

- **Start** — the first frame of the *committed* attempt that fails: the final descent /
  closing approach, taken from where the gripper path stops tracking the object
  (visible misalignment). Not the whole approach.
- **End** — ~0.3 s after the failure is observable: gripper fully reopened on nothing,
  object clear of the fingers, knocked object at rest. **Recovery is not part of the span.**
- For a flagged failed close the span is derived from the detector's closed interval
  $[c_a, c_b)$ as $[c_a - 12,\ c_b + 15)$ — 0.4 s of committed descent before the
  gripper shuts, 0.5 s after it reopens so the span covers the reveal that nothing was
  picked up. Over the 75 flagged intervals (median 48 frames) that gives a median span
  of **2.5 s**, p90 3.7 s. Vision moves the start earlier when the sheet shows
  misalignment before the descent.
- Measured natural length: approach→reopen median **1.7 s**, p90 3.3 s. A span outside
  ~1-4 s wants justifying in `note`.
- One row per event. Multiple events in one segment ⇒ multiple rows ⇒ quality 1.
- Spans never cross a segment boundary; clip to the segment the failure event lands in.

### Not mistakes

Repositioning an object for a better grip; dragging a shirt to the bin in several
carries (the design of the shirt task); a second pass over the workspace; a slow but
monotonic approach.

## Calibration targets

Precision matters most at the failure / no-failure boundary, least at 4-vs-5.

| | target |
|---|---|
| quality 5 | ~40% of **frames** (currently 12.1%) |
| quality 4 / 3 | ~25% / ~15% |
| quality 2 / 1 | ~15% / ~5% |
| `mistake=True` | 2-4% of frames |

The mistake budget is bottom-up, not a quota: 69 failed closes are already located by
proprio (`disp_max=32`, 22 TP / 0 FP / 3 FN on the reviewed subsets), spread over 46 of
242 grasp segments (19%). At 1.7 s median that is ~2.4% of frames before the vision
pass adds slips, drops, knocks and wrong-targets — the classes proprio cannot see.

## Procedure

Labeling is done by the assistant for consistency, from contact sheets (top over wrist,
frame indices burned in) generated per segment.

| tier | coverage | density |
|---|---|---|
| grasp, 242 segments | every segment | 8-12 columns; +1 dense sheet around each proprio-flagged failed close |
| move + release, 484 segments | every segment | batched, one sheet per episode per phase; individual sheet only when features fall outside the 5-band |
| return, 41 segments | every segment | 4 columns |

Proprio is the **triage**, never the verdict: it seeds candidate failed closes and flags
off-band segments, vision confirms and sets the span.

Every row carries a one-line `note` recording what was seen. That is the audit trail —
a quality 2 with no note is not reviewable.

**114 segments carry `<object>` placeholders** (the 5 v3-3 episodes: two_container
eps 10-13, val ep 2) and get object identity from the same grasp sheets. Verify object
identity opportunistically elsewhere too: `majority_object` carryover is already wrong at
least once — two_container ep9 `[46569,47630)` is labelled "white sock" over a blue sock.

## Output

- `meta/episode_metadata.parquet` — `episode_index, segment_index, from_index, to_index, subtask, quality, note`
- `meta/mistakes.parquet` — `episode_index, from_index, to_index, mistake, mistake_type, note`
- `meta/metadata_info.json` — rubric version, annotator, definition/limitation strings

Written to a **new dataset directory**, never in place.

## Downstream

Norm stats → `compute_delta_stats.py --encoding anchor` → memmap cache → `config_rl.yaml`.
`chunk_size 30 / image_stride 3`.

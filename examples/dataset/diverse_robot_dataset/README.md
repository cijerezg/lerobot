# Diverse real-robot production dataset

Production outputs live under `outputs/diverse_robot_dataset/`. Pilot outputs
are evidence and test fixtures only; they are not part of the training set.

## Unified source-native corpus

`build_corpus.py` writes the layer the packed components never had: the continuous
native timeline of every accepted episode. The packed v2 components stored one anchor
every two seconds, so no denser stride and no whole-subtask action sequence could be
recovered from them without going back to staging. The corpus stores each episode once -
timestamps, state, native actions, per-camera video at the native rate, the reviewed
annotations, and the provenance - and the actor and critic views are indexes into it.

```bash
# One source at a time; each accepted episode is read from its validated component's
# annotations and acquisition manifest, so nothing is re-reviewed.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/build_corpus.py ingest --source droid_success

# Actor anchors at 5 Hz and critic subtask intervals, written as indexes, not copies.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/build_corpus.py views --stride-hz 5

# Arrays, annotations, splits, video frame counts, and source frame alignment.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/build_corpus.py validate --alignment-samples 3

# Per-source and per-component accounting, including the referenced FMB corpus.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/build_corpus.py ledger

# Optional: dense packed LeRobot v3 components in the existing training schema.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/build_corpus.py pack --stride-hz 5
```

Episode video is re-encoded at CRF 18 with a one-second GOP so that anchor decoding at any
stride stays cheap; the encoder command, the source path, and the source SHA-256 are
recorded per camera. `validate` decodes sampled frames from both the corpus and the source
and requires the corpus frame to match the source frame at the same index, which is what
catches a seek that slipped by one. A shift moves the whole stream, so a camera is called
misaligned only when at least two sampled frames agree on the same non-zero offset and each
clears a tie threshold set by the encode noise floor and the local motion; a single frame's
argmin during a still moment is recorded as `tie_inconclusive`, not a failure. As of
2026-08-31 the corpus passes with zero failures across 204 episodes and 1,734 frame checks.

Rates come from the source's declared fps, never from the stored timestamps: float32
quantization reports a 30 Hz stream as 30.0000286 Hz, which would route RoboChallenge and
UR7e through the interpolation branch and turn their audited same-timestamp `copy_state`
actions into interpolated ones. `refresh-metadata` re-derives that field for episodes
ingested before the rule existed.

Read the corpus with `lerobot.datasets.diverse_corpus`:

```python
from lerobot.datasets.diverse_corpus import DiverseCorpus

corpus = DiverseCorpus("outputs/diverse_robot_dataset/corpus")
actor = corpus.actor_sample(corpus.actor_anchors(split="train")[0])
critic = corpus.critic_sample(corpus.critic_intervals(split="train")[0], action_points=32)
```

The critic sample returns every native action in the reviewed subtask. `action_points`
subsamples at load time and always keeps the first and last action, prefers gripper
transitions and reviewed mistake or recovery events, and returns a validity mask; the
stored sequence is never truncated.

### A further review round on an already-built task

The RoboChallenge task staging holds roughly a thousand episodes per task and only ten per
task were accepted, so more episodes cost no download. `robochallenge_round2.py` nominates
candidates from the episodes no earlier round looked at, renders the same review surfaces,
and after your acceptance writes the annotations, converts the episodes, and ingests them.

```bash
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/robochallenge_round2.py --task press_the_button

# after reading review/press_the_button/round2/episode_*.global.jpg
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/robochallenge_round2.py \
  --task press_the_button --accepted <ten episode indices>
```

The validated two-second components under each source's `datasets/` are left untouched.

## FMB production corpus

FMB passed the admission gate and is now a validated production component. All
100 production trajectories and all 521 unchanged source-native primitive
intervals were visually reviewed. Every interval is complete, uninterrupted,
and critic-eligible. Quality and mistake supervision follows the ReBot contract
in `pi07_wiki/annotation_rubric.md`, using
`outputs/rebot_val-annotated-v3` and
`outputs/rebot_shirts_bin-annotated-v2` as golden schema examples: quality is
one integer constant over a complete primitive interval, while a mistake is a
boolean only over the frames of a discrete visible failure event. Formal outcome
and recovery remain unused and unknown.

The pinned production manifest is `fmb_production.json`. It selects 100
distinct trajectories, retains the 12 reviewed pilot trajectories, covers all 54
shape-size-length geometries before repeats, balances horizontal/vertical and
distractor/no-distractor conditions 50/50, and assigns 80/10/10
train/validation/test splits at source-episode level.

```bash
# Reproduce the pilot review gate after rendering all four RGB-depth pairs.
.venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/finalize_fmb_review.py

# Validate the production-wide ReBot-compatible quality/mistake artifact.
.venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/annotate_fmb_critic.py \
  --review-root outputs/diverse_robot_dataset/fmb/production/review \
  --labels lerobot/examples/dataset/diverse_robot_dataset/fmb_production_quality_mistakes.json \
  validate

# Recreate the pinned selection from immutable Hub metadata if needed.
.venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/prepare_fmb.py select

# Acquisition is the networked step. Conversion and validation are local.
.venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/prepare_fmb.py acquire
.venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/prepare_fmb.py \
  --review-root outputs/diverse_robot_dataset/fmb/production/review convert
.venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/prepare_fmb.py views
.venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/prepare_fmb.py \
  --review-root outputs/diverse_robot_dataset/fmb/production/review validate
```

The validated corpus lives at
`outputs/diverse_robot_dataset/fmb/production/corpus/`. Its `episodes/`
directories are the single source-native collection. Each episode stores the
complete native `actions`, primitive labels, original timestep indices,
nominal-grid timestamps, low-dimensional observations, side-1/side-2/wrist-1
RGB, and wrist-1 Z16 depth. `actor_anchors_5hz.jsonl`,
`actor_anchors_10hz.jsonl`, and `critic_intervals.jsonl` only reference those
episode arrays; they do not copy sensor frames or actions into separate actor
and critic datasets.

Measured validated production accounting:

- 100 episodes, 21,892 unique synchronized timesteps/native actions, and
  2,189.2 seconds on the documented nominal 10 Hz grid;
- 87,568 RGB images and 87,568 depth maps observed in raw source files;
- 65,676 retained RGB images across side 1, side 2, and wrist 1, plus 21,892
  retained wrist-1 depth maps;
- 793/662 candidate episode-level/primitive-conditioned actor anchors at the
  original two-second stride, 7,471/6,320 at 5 Hz, and 14,892/12,590 at 10 Hz;
- 521 source-native primitive intervals with every native action referenced;
  all 521 intervals and 21,892 timesteps are visually reviewed,
  critic-eligible, and labelled quality 5 under the ReBot rubric;
- zero visible mistake events and zero mistake-positive timesteps; the all-5
  distribution was measured rather than quota-forced or inferred from expert
  status; and
- 80/10/10 source episodes in train/validation/test, with no actor or critic
  records crossing splits.

The review used retained-camera contact sheets for every episode plus a dense
24-sample second look for 48 primitive-relative motion-efficiency or duration
outliers across 38 episodes. Those metrics selected review candidates only and
never assigned labels. The flagged executions were geometry-dependent or slow
but direct; no correction, retry, failed close, slip, drop, knock, or wrong
target was visible. The production label artifact is
`fmb_production_quality_mistakes.json`, and the 100 applied review files are in
`outputs/diverse_robot_dataset/fmb/production/review/`.

Depth remains source-native D405 Z16 `uint16`. Zero and 65,535 are invalid.
Metric interpretation is 0.1 mm per valid level with provenance
`user_authorized_same_D405_model_assumption`; it is not presented as measured
FMB calibration. External depth, wrist-2 RGB, and wrist-2 depth are absent from
converted episode directories. Validation checks exact raw-to-retained-array
equality, hashes, schemas, synchronization, depth metadata/counts, native action
coverage, primitive coverage, derived-row counts, split isolation, and dropped
modalities.

Raw staging is 27 GiB and the retained corpus is 15 GiB. The validation report
permits removal only because every selected source hash and exact retained-array
comparison passed, but raw staging has not been removed. The pilot
`converted_validation/` output is a test fixture and is not an additional
training corpus.

Use the common and FMB stores as one logical corpus without copying either:

```python
from lerobot.datasets.fmb_corpus import FederatedDiverseCorpus

corpus = FederatedDiverseCorpus(
    "outputs/diverse_robot_dataset/corpus",
    "outputs/diverse_robot_dataset/fmb/production/corpus",
)
actor_rows = corpus.actor_anchors(split="train")
critic_rows = corpus.critic_intervals(split="train")
```

The refreshed unified ledger is
`outputs/diverse_robot_dataset/unified_coverage_ledger.json`. At the comparable
5 Hz stride it records 304 episodes, 247,067 unique synchronized timesteps,
40,507 retained actor rows, and 843 critic-eligible intervals. FMB also retains
its separate 14,892-row 10 Hz actor index. All indexes reference the original
source-native episode arrays and duplicate zero sensor/action arrays.

## RoboChallenge

RoboChallenge uses the component workflow below. Its production manifest selects 10 single-arm
tasks and targets 10 visually accepted source episodes per task. Every accepted
episode contributes all clean active anchors at a two-second stride. The
measured six-joint plus gripper-width state is copied unchanged into action for
this source-specific exception.

Reviewers should use `robochallenge/review/<task>/episode_XXXXXX.mp4`. Each file
shows the global and wrist cameras side by side. The `t`, `tm1`, ..., `tm6`
directories inside a finalized LeRobot dataset are model-facing temporal
features, not separate episodes and not the review interface.

Run one task component end to end. Acquisition is the only networked step:

```bash
# Steps 1-2: download at the pinned revision, verify bytes and SHA-256,
# scan the concatenated tar, then extract into bounded staging.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/acquire_robochallenge.py \
  --task item_classification \
  --downloads-root outputs/diverse_robot_dataset/robochallenge/staging/downloads \
  --raw-root outputs/diverse_robot_dataset/robochallenge/staging/raw

# Steps 3-4: nominate 15 candidates and render review proxies.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/run_robochallenge_component.py \
  --task item_classification review

# Review the proxies, then steps 5-8: annotations, conversion, packed
# extraction, validation, component metadata, and the index update.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/run_robochallenge_component.py \
  --task item_classification finalize --accepted 0 152 201 220 232 561 807 849 872 932
```

Visual acceptance is deliberately a human step between `review` and `finalize`.
Nomination balances candidates across physical robot IDs with a hard per-robot
quota; the realized balance is recorded under `robot_balance` in
`candidates.json`. `finalize` refuses to attach metadata or touch the index
unless the validation report passes.

Validated annotation JSON files feed the existing packed-v3 converter and
extractor. Final components are written under
`outputs/diverse_robot_dataset/robochallenge/datasets/<embodiment>/<task>/`.

## DROID Failure and DROID Success

DROID is already LeRobot v3 on the Hub, so the RoboChallenge scripts do not
apply: there is no split tar to verify and no `states.jsonl`. Use
`prepare_droid.py` and `run_droid_component.py`, whose source spec is
`droid_sources.json`. This source exposes native commanded joint actions, so the
RoboChallenge `copy_state` exception must not be extended to it, and its 15 Hz
rate means the 30 Hz interpolation path runs for the first time: a validated
component must report a non-zero `interpolated_action_points`.

Acquisition is two-stage. Every episode's low-dimensional record lives in one of
18 shared data shards (77 MB each, ~766 episodes each), so the complete
low-dimensional collection costs 1.4 GB and is downloaded once. Video is the
expensive half: 1,356 shards of ~203 MB, and the three cameras do not share shard
boundaries, so one candidate costs about 2.4 video files. A 24-candidate
component is therefore roughly 10-12 GB.

```bash
# Once: audit at the pinned revision, then fetch every low-dimensional shard and
# summarize all 13,747 source episodes against their commanded-action traces.
uv run --no-project --python .venv/bin/python -m lerobot.scripts.lerobot_diverse_pilot \
  --config lerobot/examples/dataset/diverse_robot_dataset/droid_sources.json \
  --metadata-root outputs/diverse_robot_dataset/droid/metadata \
  --output-root outputs/diverse_robot_dataset/droid audit

uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/prepare_droid.py scan \
  --metadata-root outputs/diverse_robot_dataset/droid/metadata/droid_failure \
  --staging-root outputs/diverse_robot_dataset/droid/staging \
  --output outputs/diverse_robot_dataset/droid/review/episode_scan.json

# Per component: nominate 24, download only their payload, render review sheets.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/run_droid_component.py --lab AUTOLab review

# Read the overview sheets, shortlist, then render two-second-stride detail sheets.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/run_droid_component.py --lab AUTOLab \
  detail --episodes 398 496 522

# Write one episode_XXXXXX.review.json per accepted episode, then finalize.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/run_droid_component.py --lab AUTOLab \
  finalize --accepted 18 98 398 401 496 522 578 1417 1470 1514
```

A component is one lab, which is the scene family. DROID has no `robot_id`, so
the hard balance quota runs over `(lab + external camera serial pair, operator)`
instead. `episode_XXXXXX.left.jpg` and `.right.jpg` are the primary still-review
surface: twelve labelled tiles spanning the whole episode at 400 px, sized so a
reviewer sees them at full resolution. Detail sheets are the same layout at the
two-second anchor stride, so a visible event maps straight onto an annotation
span in source time.

Annotation is the deliverable for this source, not a formality. The reviewer
writes `episode_XXXXXX.review.json` with the task, outcome, notes, and contiguous
spans carrying `retention`, `retention_reason`, `subtask`, `quality`, and
`mistake_events`. `prepare_droid.py annotations` intersects those spans with the
commanded-action activity screen and with the publisher's `keep_ranges`, so the
reviewer decides intent and outcome while the automatic screen only subdivides an
already-approved span. A window that leaves a source keep interval is rejected as
`source_excluded` unless the reviewer sets `override_source_keep`.

Both DROID splits use the same driver and the same lifecycle; select with
`--source droid_failure` (default) or `--source droid_success`. They share a
robot, a rate, a shard layout, and a schema. Two things differ.

`droid_success` carries language: three pipe-separated phrasings per episode.
The scan records the first phrasing as `source_task`, nomination reports it, and
the review sheet header prints it. It is a nomination and cross-check signal, not
a substitute for proxy review; `droid_failure` has none, which is why task
identity there had to come from the sheets alone.

`droid_success` is selected for **recovery**, not for additional clean successes.
Plan Section 1.6 found recovery to be the one Section 1.3 gap the failure split
left mostly open, at 18 chunks from two components. A success episode containing a
failed attempt has by definition recovered. The scan therefore counts
`gripper_close_events` — open-to-closed transitions in the commanded gripper
channel, with hysteresis at 0.6/0.3 — and `nominate --min-close-events 2` keeps
only episodes that attempted a grasp at least twice. It is a cheap
low-dimensional signal for a retry; whether a retry is a genuine recovery is
still decided by looking.

```bash
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/run_droid_component.py \
  --source droid_success --lab TRI review
```

`coverage_ledger.py` takes a repeatable `--component-root`, so one ledger can span
sources whose components have different shapes:

```bash
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/coverage_ledger.py \
  --component-root outputs/diverse_robot_dataset/robochallenge \
  --component-root outputs/diverse_robot_dataset/droid \
  --component-root outputs/diverse_robot_dataset/droid_success \
  --output outputs/diverse_robot_dataset/coverage_ledger.json
```

# Diverse real-robot production dataset

Production outputs live under `outputs/diverse_robot_dataset/`. Pilot outputs
are evidence and test fixtures only; they are not part of the training set.

RoboChallenge is built first. Its production manifest selects 10 single-arm
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

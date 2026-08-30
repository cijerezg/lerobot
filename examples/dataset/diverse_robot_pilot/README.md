# Diverse real-robot acquisition pilot

This directory is the executable companion to `DIVERSE_ROBOT_DATASET_PLAN.md`.
The tooling is deliberately admission-gated: it inspects Hub repository and
LeRobot metadata first, writes source audits, and refuses to resolve or download
episode payloads when the source usage basis, real-robot provenance, native
state, or usable action mapping is not established. Native commanded actions
are preserved when present. RoboChallenge alone uses the explicitly configured
`copy_state` exception described below.

No command in the audit or nomination phase downloads `data/`, `videos/`, or
`images/`. Set `HF_TOKEN` only if Hub authentication or license acceptance is
required.

```bash
cd /home/user/Documents/Research/RL/LeRobot

uv run python lerobot/src/lerobot/scripts/lerobot_diverse_pilot.py \
  --config lerobot/examples/dataset/diverse_robot_pilot/config.json \
  --metadata-root outputs/diverse_robot_pilot/metadata \
  --output-root outputs/diverse_robot_pilot/generated \
  audit

uv run python lerobot/src/lerobot/scripts/lerobot_diverse_pilot.py \
  --config lerobot/examples/dataset/diverse_robot_pilot/config.json \
  --metadata-root outputs/diverse_robot_pilot/metadata \
  --output-root outputs/diverse_robot_pilot/generated \
  nominate
```

After reviewing `generated/candidate_manifest.json`, resolve exact files for an
admitted source without downloading them:

```bash
uv run python lerobot/src/lerobot/scripts/lerobot_diverse_pilot.py \
  --config lerobot/examples/dataset/diverse_robot_pilot/config.json \
  --metadata-root outputs/diverse_robot_pilot/metadata \
  --output-root outputs/diverse_robot_pilot/generated \
  plan-download --source droid_success --episodes 9339 9298
```

## Operator phase

The committed plans currently resolve to approximately 415 MiB for UR7e,
668 MiB for DROID Success, and 669 MiB for DROID Failure (about 1.71 GiB total).
The downloader checks free space with a 20% reserve and fetches only exact paths
listed in each plan.

If a command returns 401/403, visit the dataset page, accept any displayed terms,
then run `hf auth login` or export a read-only `HF_TOKEN`. The metadata audit on
the pinned revisions reported all four repositories public and ungated; this may
change independently of the code.

```bash
set -euo pipefail

STAGE="/home/user/Documents/Research/RL/LeRobot/outputs/diverse_robot_pilot/staging"
PILOT=outputs/diverse_robot_pilot/generated

for SOURCE in ur7e_stack droid_success droid_failure; do
  uv run python lerobot/src/lerobot/scripts/lerobot_diverse_pilot.py \
    --config lerobot/examples/dataset/diverse_robot_pilot/config.json \
    --metadata-root outputs/diverse_robot_pilot/metadata \
    --output-root "$PILOT" \
    download --plan "$PILOT/download_plans/$SOURCE.json" --staging-root "$STAGE"

  uv run python lerobot/src/lerobot/scripts/lerobot_diverse_pilot.py \
    --config lerobot/examples/dataset/diverse_robot_pilot/config.json \
    --metadata-root outputs/diverse_robot_pilot/metadata \
    --output-root "$PILOT" \
    proxies --plan "$PILOT/download_plans/$SOURCE.json" \
    --staging-root "$STAGE" --proxy-root "$PILOT/review/$SOURCE"
done
```

Review the candidate behavior in every camera proxy and contact sheet. Reject
near-duplicate episodes and regenerate the exact plan with `plan-download` if
needed. Fill every generated `*.annotations.json`: set `review_status` to
`validated` and cover the complete `[0, episode_duration_s]` interval with
contiguous keep/reject spans. Kept spans require semantic labels:

```json
{
  "start_s": 0.0,
  "end_s": 8.5,
  "retention": "keep",
  "retention_reason": "informative_mistake",
  "subtask": "reach for the cup",
  "quality": 4,
  "mistake_events": [
    {"id": "m1", "type": "wrong_target", "start_s": 3.1, "end_s": 3.8}
  ]
}
```

Rejected spans need only `start_s`, `end_s`, `retention: "reject"`, and a
short `retention_reason` such as `static`, `erratic`, or `low_quality`.
An anchor is retained only when its complete history/action window lies within
approved behavior.

Then write and validate the two-second multi-chunk dataset. This example is for
DROID Success; repeat per source and use a distinct output repo id/root.

```bash
ROOT="outputs/diverse_robot_pilot/datasets/droid-success-filtered"
uv run python lerobot/src/lerobot/scripts/lerobot_diverse_pilot.py \
  --config lerobot/examples/dataset/diverse_robot_pilot/config.json \
  --metadata-root outputs/diverse_robot_pilot/metadata \
  --output-root "$PILOT" \
  extract --source droid_success \
  --plan "$PILOT/download_plans/droid_success.json" \
  --staging-root "$STAGE" --annotations-root "$PILOT/review/droid_success" \
  --dataset-root "$ROOT" --repo-id local/diverse-droid-success-filtered

uv run python lerobot/src/lerobot/scripts/lerobot_diverse_pilot.py \
  --config lerobot/examples/dataset/diverse_robot_pilot/config.json \
  --metadata-root outputs/diverse_robot_pilot/metadata \
  --output-root "$PILOT" \
  validate --dataset-root "$ROOT" --repo-id local/diverse-droid-success-filtered
```

Check `meta/extraction_report.json` for at least three retained chunks per
ordinary episode, review the rejection reasons and representative frames, and
then run validation. Use `--min-chunks-per-episode 1` only for a reviewed rare
event that justifies the exception. Cleanup is intentionally separate and
deletes only manifest-listed staged files:

```bash
uv run python lerobot/src/lerobot/scripts/lerobot_diverse_pilot.py \
  --config lerobot/examples/dataset/diverse_robot_pilot/config.json \
  --metadata-root outputs/diverse_robot_pilot/metadata --output-root "$PILOT" \
  cleanup --plan "$PILOT/download_plans/droid_success.json" \
  --staging-root "$STAGE" --dataset-root outputs/diverse_robot_pilot/datasets/droid-success-filtered \
  --confirm-validated-output
```

RoboChallenge does not expose a native commanded joint action. The approved
source-specific exception copies each same-timestamp measured six-joint plus
gripper-width vector exactly into both state and action. Do not use the source
reference converter's end-effector action mapping, and do not apply this
exception to UR7e, DROID, or another source with native actions.

Convert only reviewed raw episodes into a temporary local v3 source:

```bash
uv run python lerobot/examples/dataset/diverse_robot_pilot/convert_robochallenge.py \
  --raw-task-root outputs/diverse_robot_pilot/staging/robochallenge/shred_paper \
  --dataset-root outputs/diverse_robot_pilot/staging/local__robochallenge-ur5-shred-paper-source \
  --repo-id local/robochallenge-ur5-shred-paper-source --episodes 484 731
```

The converter writes `meta/local_acquisition_manifest.json`, including the raw
episode mapping. Pass the converted root explicitly to the standard extractor:

```bash
uv run python lerobot/src/lerobot/scripts/lerobot_diverse_pilot.py \
  --config lerobot/examples/dataset/diverse_robot_pilot/config.json \
  --metadata-root outputs/diverse_robot_pilot/metadata \
  --output-root outputs/diverse_robot_pilot/generated \
  extract --source robochallenge_raw \
  --source-metadata-root outputs/diverse_robot_pilot/staging/local__robochallenge-ur5-shred-paper-source \
  --plan outputs/diverse_robot_pilot/staging/local__robochallenge-ur5-shred-paper-source/meta/local_acquisition_manifest.json \
  --staging-root outputs/diverse_robot_pilot/staging \
  --annotations-root outputs/diverse_robot_pilot/generated/review/robochallenge_shred_paper \
  --dataset-root outputs/diverse_robot_pilot/datasets/robochallenge-ur5-shred-paper-filtered \
  --repo-id local/diverse-robochallenge-ur5-shred-paper-filtered
```

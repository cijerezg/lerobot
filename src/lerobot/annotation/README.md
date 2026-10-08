# Annotation

Everything that defines or produces a label lives in this folder.

## The rule

| What | Where |
|---|---|
| Every annotation script | here |
| Every rubric, label spec and vocabulary | here (`rubrics/`, `vocab.py`) |
| Results: inventories, label files, sheets, strips, review pages, reports, logs | outside the repo: `migration/<pass>_<date>/` or `outputs/_annotation/` |
| Labels of record | ReBot: dataset `meta/` tables; diverse: JSONL sidecars in each `corpus/` and `fmb/` store under `outputs/` |

- No annotation script or rubric is written in `migration/`. A script written there for a quick look moves
  here before the pass ends, or is deleted.
- No result is committed to this repo.
- Before writing a function, look for it in the shared modules below. If two passes need the same thing, it
  goes in a shared module and both call it. A pass script holds only what is specific to that pass (its source
  roots, its cut plan, its exemptions).
- Scripts take the results folder as an argument or an environment variable. They never write next to
  themselves. `paths.WORKSPACE` is the folder that holds `lerobot/`, `migration/` and `outputs/`.
- Commands run from the workspace root: `uv run python -m lerobot.annotation.<folder>.<script> ...`.

## Authority and entry points

This directory (`lerobot/src/lerobot/annotation/`, import path `lerobot.annotation`) is the centralized home of
annotation code and instructions. Update the relevant rubric here when a decision changes; do not create competing
rubrics, handoff briefs or scripts in results folders. Results folders contain evidence, decisions and progress, not
new policy. This README is the entry point for a new session.

Read `rubrics/annotation_principles.md` first. It overrides older conflicting rules. Then read the channel rubrics:
`quality_mistake_rubric_v2.md` governs frame-local quality, mistake events and precision windows;
`precision_rubric.md` supplies precision levels; `contact_strategy_rubric.md` supplies contact definitions;
`subtask_atoms_rubric.md` supplies diverse segmentation and grammar. Historical examples and legacy tool constraints
never override the current contract. Existing annotations are hypotheses to check, not ground truth.

For a **diverse-only sampled audit**, use [Diverse dataset: sampled audit](#diverse-dataset-sampled-audit) below.
The ReBot pass commands and `rebot/AGENT_BRIEF.md` are not the diverse execution procedure.

## Layout

| Path | What it is |
|---|---|
| `rubrics/annotation_principles.md` | The one test every label must pass. Read first. |
| `rubrics/quality_mistake_rubric_v2.md` | Quality spans, mistakes, precision windows. |
| `rubrics/precision_rubric.md` | Precision per step (1 coarse to 5 fine). |
| `rubrics/contact_strategy_rubric.md` | Contact strategy per step; the table `vocab.py` is tested against. |
| `rubrics/subtask_atoms_rubric.md` | Atom grammar, naming and quality rules for the diverse corpus. |
| `rubrics/depth_gripper_event_labels.md` | Gripper-event targets of the depth auxiliary loss. |
| `vocab.py` | Closed vocabularies: contact elements (code, phrase) and mistake types. Imported by training code. |
| `paths.py` | `WORKSPACE` and `QUALITY_V2` (the results folder of the class pass). |
| `rebot/` | A ReBot pass, start to finish (order below). `AGENT_BRIEF.md` is the brief the annotating and grading agents follow; review feedback is folded into it and into the rubrics. |
| `quality_v2/` | The rubric v2 class pass, shared by every pool: class rules, side-by-side material, dense strips, grading batches, second read, label compiler, checks. |
| `precision_contact/` | Frame grabbing, approach angle, row sheets and strips (used by `quality_v2`), and the text-rule priors for precision and contact. |
| `atoms/` | Subtask-atom pipeline over the diverse corpus: propose, sheets, verdict, agent_check, assemble, manual editor, validator; its reviewer briefs. |
| `speed/` | Speed label: `hybrid_motion_speed` (the adopted method), `speed_annotate`, `joint_motion_speed`, and the runners `rebot_speed_pass` and `hybrid_diverse`. |
| `gripper_events/` | Depth gripper-event labels and their review videos. |
| `review/` | Review videos of training episodes with the labels the trainer consumes burned in; `audit_material` (one episode's labels, episode-local, and approach lines, for an audit checker). |
| `segments/` | Proprio segmentation (`semantic_segment`). |

Tests: `tests/annotation/`.

## A ReBot pass

To annotate a ReBot root, an agent needs only this page: it names what to read, the order of work and when to stop.

A pass is one results folder, `<work> = migration/<pass>_<date>/`. Everything specific to the pass is data in it:

| File in `<work>` | Content |
|---|---|
| `inventory.json` | One record per source episode: `idx`, `key`, `source`, `episode`, `frames`, `digest`, `kind`, `seams`. Written by `rebot.inventory`. |
| `labels/<IDX>.json` | Per source episode: keep ranges, split, task, segments, mistakes, cuts. |
| `pass.json` | `pool`, `dataset`, `staging`, `onset_thr`, `subtasks`, `class_rules`, `reuse`, `nearest`, `idle_exempt`. |
| `plan.json` | Only for a cut without labels: `[{source, episode, keep, cuts, task}]`. |

```bash
M=lerobot.annotation
# 1. look: numbers, inventory, traces, overview sheets (one tile per 2 s)
uv run python -m $M.rebot.screen <root> [<root> ...]
uv run python -m $M.rebot.inventory <work> <root> [--episodes E ...] --kind rollout|teleop --name <dataset>
uv run python -m $M.rebot.traces <work>
uv run python -m $M.rebot.sheets <root> <work>/review/<IDX> <episode> --step 60 --tag overview
# 2. cut only (no labels yet): <work>/plan.json -> root
uv run python -m $M.rebot.build_root <work> --out outputs/<root> --plan
# 3. labelled pass: check the label files, build the staging root, class the segments, render the class material
uv run python -m $M.rebot.validate_labels <work> 0 1 2 ...
uv run python -m $M.rebot.build_root <work> --out outputs/<staging> --split all
uv run python -m $M.rebot.classify <work>
uv run python -m $M.quality_v2.material --work <work>
uv run python -m $M.quality_v2.dense <class_slug> <uid> <from> <to>          # on demand, while grading
# 4. after grading: second read, compile, idle cuts
uv run python -m $M.quality_v2.second_read <pool> [--work <work>] --no-calibrate <every class of the pool>   # --work: re-annotation pools (class files from class_map.json)
uv run python -m $M.quality_v2.compile_labels --pool <pool> --deterministic-resolvers
uv run python -m $M.quality_v2.check_mistake_owner <pool>                     # also check_precision_chain, check_references, blind_agreement
uv run python -m $M.rebot.idle_scan <work>                                   # writes idle_cuts.json; build_root applies it
# 5. final root: build, tables, speed, gripper events, verify (each step refuses to overwrite)
uv run python -m $M.rebot.build_root <work> --out outputs/<root> --split train
uv run python -m $M.rebot.finalize <work> outputs/<root>
uv run python -m $M.speed.rebot_speed_pass outputs/<root>
uv run python -m $M.rebot.manual_speed <work> outputs/<root>
uv run python -m $M.gripper_events.depth_gripper_event_annotate --data-dir outputs/<root> --rule relative_travel
uv run python -m $M.rebot.verify_root outputs/<root>
# 6. review videos for the user (labels burned in)
uv run python -m $M.review.render_review_videos --rebot-root outputs/<root> --out <work>/review_videos
```

### How a pass is run

The agent that receives the request coordinates; the frame reading is done by agents it starts, in parallel.

| step | what |
|---|---|
| read | `rubrics/annotation_principles.md`, `rebot/AGENT_BRIEF.md`, `rubrics/quality_mistake_rubric_v2.md` section 8 (the user's verdicts are the calibration), and the source root's own `README.md` if it has one (its descriptions are a first guess; the frames decide) |
| set up | a new `<work>` folder, `inventory.json`, and `pass.json` with a pool and dataset name that do not exist yet under `paths.QUALITY_V2/classes`; `class_rules` and `reuse` name the existing class files each step is graded against |
| annotate | one agent per episode, all at once, each given "Annotating one episode" of `AGENT_BRIEF.md` plus its `idx`, root, episode, `seams` and operator runs (`review/events_<IDX>.txt`). Before going on, read every subtask text against principles section 2 and fix or send back the ones that fail |
| grade | staging root, `classify`, `material`, then one grader per episode, all at once, each given "Grading" of `AGENT_BRIEF.md` and its episode's units (each graded against its own class file, `final` in `class_map.json`). Then a second reader for every unsure row ("Second read" of `AGENT_BRIEF.md`; slices from `second_read <pool> --no-calibrate <every class of the pool>`), up to 12 readers at once. Then `compile_labels` |
| build | final root, tables, speed, gripper events; `verify_root` must pass |
| hand over | per episode a short table (frames, subtask, contact, precision, stretches, mistakes). Decide every call yourself (borderline grades, mistake types, short tails, range edges) and list them as decisions taken. Escalate only what would change the data substantially, each item with a review video and the frame range. Then stop for the user's feedback |
| feedback | written into the rubrics (the rule it changes, and a row in the case library) and into `AGENT_BRIEF.md`. Never into a new file |

- Nothing is overwritten: new work folder, new pool name, new root names. Leftovers of earlier passes are listed for
  the user, not deleted.
- A finished root is not added to a training config and not copied to the Spark until the user says so.
- Rendering runs on the Spark when the source root is there (quality rubric section 1.1). A root that exists only on
  this machine is rendered here.
- The annotator is independent (user 2026-10-06): the rubrics settle the calls, not questions to the user.

### Re-annotating a root

A root whose labels are bad is re-annotated as a new pass with the root as its source (`inventory --kind teleop|rollout`).
Everything is redone (keep ranges, segments, texts, channels, mistakes, grading); the old labels are a first guess. Go back
to the raw recording only when a cut looks wrong. When the request says to keep some fields, copy the old `labels/` and
redo only the rest (example: `migration/rollouts_annotation_v2_2026-10-06/`).

### Auditing annotated data by random sampling

Used when the data is too large to re-annotate. The population is the source roots of the training roots
(`<training root>/meta/provenance.json`); the request names any exclusions. A ledger `<work>/audit.md` holds one row per
round.

1. Draw 8 episodes at random; write the seed and the picks to the ledger.
2. One checker agent per episode ("Auditing one episode" of `rebot/AGENT_BRIEF.md`; material from
   `review.audit_material`) reads its labels against the frames, by the same rubrics as a pass: texts (principles
   2), boundaries and kept range, mistakes, contact and precision at the commit frames, quality on the approach line and
   dense strips. Each issue: what, frames, severity (`wrong`: teaches the model something false; `minor`: imprecise
   but harmless).
3. Fix every issue in the label source: quality, mistakes and precision as `F_` resolve rows in the class folder; texts,
   boundaries and keep ranges in new label files. An issue seen twice is a rule problem: fix the rule (rubric or
   `AGENT_BRIEF.md`, with a case-library row) and sweep that issue over the whole source.
4. Repeat until two rounds in a row find no `wrong` issue and at most 2 `minor`. Then rebuild the affected roots under
   new names (`verify_root` PASS) and hand over the ledger.


## Diverse dataset: sampled audit

Current contract, 2026-10-07. Review and repair existing annotations in small rounds; do not re-annotate the whole
corpus from scratch. Exclude every ReBot root, including external ReBot and rollouts: other sessions own them.
Do not use git. Do not change training configs or publish/adopt a replacement dataset as part of the audit.

### Establish the population and resume state

- Read the principles and channel rubrics listed above, including quality sections 4-6, 8 and 9. Read the source
  metadata before interpreting its frames or state. Routine annotation calls are the annotator's responsibility.
- Resolve the actual diverse roots from the current training configuration and store metadata. The existing class
  pass points to `outputs/diverse_robot_dataset_v3/{corpus,fmb}`; this is a discovery pointer, not proof that those
  are still the active roots. Include train, validation and test while preserving their existing split membership.
- Inventory unique source episodes, source family, split, native rate, retained intervals, camera availability,
  state/action schema and label provenance. Do not count multiple views, atoms or derived copies as new episodes.
  The families are DROID, DROID success, MolmoAct, RoboChallenge, UR7e, YAM and FMB; verify against the actual inventory.
- Create a new results folder `migration/diverse_annotation_audit_<date>/` (unique suffix if needed), or resume its
  recorded progress. `audit.md` is the ledger: roots and provenance, inventory, sampling policy, seed and picks per
  round, evidence paths, issues and severity, fixes/sweeps, validation and unresolved work. Record pre-fix counts;
  fixing a round does not make its original error count zero. Keep episode decisions in `checks/` there.

### Sample, review, repair, repeat

1. Draw **8 previously unchecked episodes per round**, without replacement. Randomize within source families;
   rotate allocation toward underrepresented families, with at least one from each non-exhausted family when the
   batch permits. Rotate task/robot/lab and split coverage across rounds. Record the seed, eligible pool, allocation
   and ordered picks so the draw is reproducible. Exhausted families need no duplicate picks.
2. Review all retained intervals and all current semantic labels of each selected episode against the images.
   First check task/subtask text, object identity, destinations, boundaries and retained ranges; then contact,
   precision level/window, mistakes and quality. Inspect the effective labels the trainer reads, including v2
   sidecars, rather than only the legacy atom `quality` or `mistake_events`. Use existing class references to judge
   quality, verifying their calibration; a missing or unsuitable reference needs a small comparison sheet, not a
   full-corpus regrade. Inspect frames before letting the old grade determine the verdict.
3. For each issue record episode, parent/atom identity, native half-open frame range, field, visible evidence,
   proposed correction and confidence. `wrong` teaches something false: wrong/ambiguous target, missed or invented
   mistake, visibly contradicted grade/contact/precision, a boundary more than 1 s off, or retained aimless motion.
   `minor` is harmless imprecision: a small boundary offset, a borderline adjacent grade, or an unnecessary modifier.
   Material effect takes precedence over boundary duration. Dense strips/closeups/video resolve uncertainties;
   unresolved rows require a second reader and remain open rather than silently counting as clean.
4. Fix every confirmed issue in versioned annotation sources. A recurring issue seen twice triggers a rule review:
   correct the relevant rubric if it is missing or wrong, add a case-library example, and sweep that issue across
   the affected source/classes. If the rule was already right, fix its application. Use metadata/code to find
   candidates; visually confirm semantic fixes. A targeted sweep is not a request to relabel unrelated fields.
   Keep sweep discoveries separate from the random-round counts.
5. Draw a fresh round after repairs. Stop after **two consecutive rounds each have zero `wrong` issues and at most
   two `minor` issues**, with all represented source families sampled at least once and no outstanding fixes,
   second reads or required sweeps. A new systematic issue resets the clean-round count. If the population is
   exhausted, report a complete review instead; a short final batch is not an eight-episode clean round.
   This is a budget-conscious heuristic, not a statistical guarantee of a low corpus-wide error rate. Report source
   and split coverage and any remaining blind spots alongside the stopping reason.

### Adapt the evidence to the source

- Think in seconds, store integer frames at the episode's native rate. Never assume 30 Hz. Preserve excluded gaps;
  neither labels nor action/history windows may bridge them. Cuts and mistakes may cross atom boundaries only
  within retained continuous footage. Atoms tile the revised retained parent intervals, not excluded time.
- Use wrist and external views where available; record missing views instead of inventing evidence. Contact and
  precision are read at the actual commit, not a fixed `to-15` offset across all frame rates. Noun-based priors
  may locate likely errors, but do not settle a reviewed episode's labels.
- The current FK approach-line implementation is **ReBot-only**. For diverse, use verified source-native TCP
  position/orientation when available; otherwise use source-appropriate traces plus dense strips/video of the
  whole action. Do not treat arbitrary state columns as xyz/Euler angles or apply ReBot kinematics to another arm.
  Record the schema, units and source-specific idle threshold. Missing gripper data is unknown, not a zero trace.
  RoboChallenge width and FMB binary commands need their own interpretation. Numbers locate a look; frames decide.
- Slow direct motion is not automatically poor. Searching at the object can be useful and stays with a critique.
  Trim confirmed idle and aimless tails under the principles; distinguish them from purposeful holds or contact work
  (insertion, pouring, cloth manipulation). ReBot hover timings are not diverse thresholds.
- Preserve speed and command-derived depth gripper-event semantics. If boundaries or retained ranges change,
  validate/remap affected labels and indexes using their own source-native rules; a failed grasp is not a reason
  to delete a commanded gripper event.

### Tools, storage and validation

- Reuse `quality_v2/{material,dense,classes,second_read,compile_labels}.py` and `atoms/{sheets,look,assemble}.py`
  where applicable. Inspect arguments and paths first: some still target the old global pass or v3 roots.
  `paths.QUALITY_V2` can be redirected with `QUALITY_V2_WORK`. Isolate this pass's files and pool/class names;
  never run a global inventory/compiler that also writes the concurrent ReBot sessions' outputs.
- Existing references and labels are discoverable under `migration/annotation_v2_2026-09-29/` through
  `paths.QUALITY_V2`. Reuse them as evidence; write audit-specific resolutions in the new pass. Existing `F_`
  resolve-row machinery can represent quality/mistake/precision fixes when compatible; text, boundary, contact
  and retention edits need the corresponding versioned sources too. Check the compiler's actual output coverage.
- Store sidecars are `subtask_atoms.jsonl`, `contact_atoms.jsonl`, `speed_atoms_hybrid_v1.jsonl`, and the v2
  `quality_spans.jsonl`, `mistakes_v2.jsonl`, `precision_windows.jsonl`. `precision_atoms.jsonl` is the older
  per-atom precision layer. Readers in `datasets/{diverse_corpus,fmb_corpus,diverse_actor_selection}.py` establish
  which file governs each channel; these paths are consumer references, not locations for new annotation code.
- The ReBot `review.audit_material`, rebuild and `verify_root` sequence is not a validated diverse workflow.
  Legacy atom validators also enforce old parent-grade inheritance and mistake ownership. Reuse their structural
  checks, but adapt obsolete semantic checks in this package before relying on a PASS; do not change correct new
  labels to satisfy a v1 checker. Missing source support should be implemented here, with explicit root/output
  arguments. Do not claim an unimplemented adapter exists.
- Run rendering/decoding and substantial compilation/validation on Spark as in quality section 1.1, with bounded
  workers and one writer per file. A source available only locally can be rendered locally; Spark unavailability
  alone does not authorize moving a remote workload here. Fetch evidence to inspect in this session.
- Build corrected stores under new names; preserve originals and unrelated labels. Validate schemas, native frame
  bounds, parent/atom coverage, excluded gaps, splits, channel joins after boundary edits, unique mistake events,
  containment of mistakes in quality 1/2 spans, precision windows, and effective trainer-facing labels. Apply
  critique/precision headroom exactly once and keep it within retained intervals. Recheck corrected examples and
  run the relevant existing structural/loader checks. If a retention edit invalidates an index/cache, rebuild the
  affected artifact; do not assume sidecar changes and physical cuts have the same consequences.
- Hand over the ledger, reviewed coverage, corrections and rule sweeps, validation results, replacement paths and
  stopping reason. Do not activate new roots or modify the other sessions' data. No fresh annotations are authorized
  merely by editing these instructions; start the audit when the user requests it.

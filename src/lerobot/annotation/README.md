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

For a **diverse-only sampled audit**, use [Diverse dataset: sampled audit](#diverse-dataset-sampled-audit) below;
since 2026-10-08 its current mode is the [Coverage screen](#coverage-screen-current-mode-2026-10-08).
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
   full-corpus regrade. Inspect frames before letting the old grade determine the verdict. After any retention or
   boundary repair, perform a fresh quality pass over the newly exposed motion: do not allow a parent/legacy
   quality 4 to hide a failed descent, miss or recovery. Check the raw span onset and the trainer-effective span
   with headroom, and make review-video overlays distinguish those effective grades from legacy fields.
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

### Coverage screen (current mode, 2026-10-08)

Replaces further random rounds. User 2026-10-08: after 19 rounds (17% of 909 episodes) errors were still frequent,
and the two-clean-rounds rule cannot trigger at that rate. The goal is now coverage and raising the flag on errors
that teach the model something false, not reviewing every episode. The rubrics are the same ones the ReBot passes
use, so diverse and ReBot labels stay on one framework.

1. **Select by stratum.** A phase is a seeded selection, stored as data in `<audit-work>/screen/selection_phase<N>.json`
   (seed, rule, quota per family/component, episodes, evidence path). Phase 1 (seed 20261008): every unreviewed DROID
   and DROID-success episode, 3 per RoboChallenge task, 33 MolmoAct household + 7 tabletop, 4 FMB multi + 6 single,
   plus the Rounds 20-23 campaign picks that had no primary review. Later phases go where phase flags are dense.
2. **Evidence.** The same packets and pages as a round:

   ```bash
   uv run --project lerobot python -m lerobot.annotation.diverse_audit screen-labels \
     --work <audit-work> --selection <audit-work>/screen/selection_phase<N>.json
   uv run --project lerobot python -m lerobot.annotation.diverse_audit_render_v2 \
     --root <review root> --work <audit-work> --evidence <audit-work>/screen/evidence
   ```

3. **Screen.** One screener per batch (one or two long episodes, up to four short ones). The screener reads
   `effective_labels.json`, every page and `trace_v2.png`, judges by the principles and channel rubrics, and flags
   only `wrong` issues. Images drop out of a long context, so the screener writes running notes to its own scratchpad subfolder
   (`<scratchpad>/<batch>/`) every 3-5 pages (images can drop out after a few reads). An image read can come back empty ("media removed"); the screener re-reads it until it displays, never
   describes a page it has not seen, and lists unread pages in `decision_note`. When the pages cannot settle a call, it reads the store's state arrays or video read-only
   (temporary frames in the scratchpad, not in the work dir). Flags:

   | channel | flag when |
   |---|---|
   | retention | purposeful motion excluded; aimless motion, or idle of 3 s or more (user 2026-10-04), kept; a pause over 1 s graded 4 or 5 (a pause inside an action is a critique of grade 3 or lower, user 2026-09-30); any excluded head, tail or mid-episode gap (whatever its stored reason) with more than 1 s of motion on a task action, or holding a gripper open/close that completes a step (any length; restore it into the adjacent action) |
   | task / subtask text | wrong action, object, destination or part; a missing destination (including one that does not separate start from goal, e.g. "to the table" for an object already on the table); a look-alike target the text does not single out |
   | boundaries | off by more than 1 s, or an atom (of any length) whose action does not happen in its frames |
   | quality | a miss, failed close, drop, collision or recovery inside a span graded 3 or higher; clean direct motion graded 1-2; any grade off by 2 or more |
   | mistakes | a visible mistake with no event, or an event with no visible mistake |
   | precision | level off by 2 or more, or a window on the wrong action |
   | contact | wrong element (NA on a grasp, top vs side pinch, grasp vs push) |

   Not flagged: offsets of 1 s or less, adjacent-grade borderlines, wording style, notes and provenance. Each flag uses
   the round check schema (native half-open frames, field, visible evidence, proposed correction, confidence
   `sure|unsure`) in `<audit-work>/screen/checks/<episode_id>.json`, `status: complete`. In place of `round` the
   check carries `screen_phase`, `batch` and `stratum`; each issue names its parent and atom (`parent_atom`, e.g.
   `p2a0`). A correction that another flag implies (a contact or span onset that moves with a text
   or mistake fix) goes inside that flag's `proposed_correction`, not in a separate flag. Actor-anchor fields on the pages are derived from retention and are not judged. Errors below the bar that a repair should clean up anyway (legacy fields, notes) go in `decision_note`.
   A clean episode gets a check with `issues: []`. Screeners are read-only on everything else. Issues already in
   `<audit-work>/screen/known_issues.md` are swept once and not flagged per episode, except
   those whose status there says "keep flagging per episode". Evidence paths in the batch
   file are relative to `<audit-work>`.
4. **Second read.** Every `unsure` flag gets an independent reader before it is repaired. The ledger counts unsure flags as pending until
   that read.
5. **Recurrence.** A flag seen in two or more episodes of a stratum is a rule question: fix the rubric if it is wrong,
   then sweep the unscreened episodes of that stratum by metadata predicate and visual confirmation.
6. **Repair once per phase.** All confirmed flags of a phase go into one new fix directory and one corrected sibling,
   with the gates under "Tools, storage and validation". The ledger `<audit-work>/screen/screen.md` reports flags per
   stratum (episodes screened, episodes flagged, episodes flagged per channel; flag counts depend on how a
   screener groups findings and are not compared), kept apart from the Rounds 1-19 counts.

There is no stopping rule. After each phase the coordinator reports the per-stratum flag rate and the user decides
whether to screen further, sweep, or adopt the corrected root.

### Concurrent multi-round campaigns

Use this mode only when the user explicitly authorizes several rounds at once. It changes scheduling, not the
meaning of a round: each round still has eight previously unchecked episodes, its own deterministic seed, checks,
pre-fix counts and ledger row. The launch prompt supplies only the current work directory, handoff/manifest,
authorized round range and requested agent count; all durable procedure belongs here.

1. **Freeze before review.** Record the review root, training root, inventory hash, first/last round, round size and
   seed rule in one campaign manifest. All campaign verdicts are against that frozen pre-campaign snapshot. Do not
   repair an early round before later campaign episodes have been judged against the snapshot.
2. **Plan sequential draws without false completion.** Later provisional draws treat earlier campaign picks as
   already selected for sampling coverage, but do not mark those rounds complete or append false ledger rows. Use:

   ```bash
   uv run --project lerobot python -m lerobot.annotation.diverse_audit plan-campaign \
     --work <audit-work> --start-round <N> --rounds <K> --seed-base <B>
   ```

   Round `r` uses seed `B+r`. The command writes
   `<audit-work>/campaigns/rounds_<N>_<M>/manifest.json`, incorporates an already-official first round when present,
   refuses a conflicting rerun, and does not create provisional official round/check/ledger records. Never hand-edit
   the ordered picks. Freeze effective-label packets for every official and provisional pick with:

   ```bash
   uv run --project lerobot python -m lerobot.annotation.diverse_audit campaign-labels \
     --work <audit-work> --manifest <campaign-manifest>
   ```

   That command refuses inventory/review-root drift and conflicting existing packets. Before official promotion, the
   normal `sample` command must reproduce the manifest's ordered picks exactly.
3. **Parallel reads, one writer.** Give every sampled episode one accountable primary reviewer. Reviewers may render
   separate episode directories concurrently but do not edit shared round, check, ledger, sidecar or corrected-root
   files. The coordinator is the only writer unless it designates one integrator. Use additional readers for
   retention/gaps, quality/mistakes and the remaining semantic channels; assign dense second reads independently.
   The launch prompt may request a minimum worker count. Reassign workers as episodes finish rather than encoding a
   fixed team layout in the prompt or checkpoint.
4. **Keep round accounting stable.** Record every finding under the sampled round that exposed it, using the labels
   in the frozen snapshot even if another campaign finding will repair the same rule. If a bounded sweep also finds a
   campaign-sampled episode, the sampled finding takes precedence and the sweep total excludes that same finding.
   Preserve separate `wrong`, `minor` and `open` counts for every round.
5. **Integrate after the read wave.** Adjudicate disagreements centrally, close second reads, run bounded recurrence
   sweeps, then compile confirmed repairs into versioned sources and corrected siblings. A campaign may consolidate
   repairs, but its manifest and per-round pre-fix decisions remain immutable. Refresh and recheck affected evidence
   after compilation, then run the normal containment, structural, loader, provenance and immutable-payload gates.
6. **Promote in order.** Create/finalize official rounds from first to last, verifying each official seed and ordered
   pick list against the manifest. Update the ledger and clean streak after each round. A fixed campaign explicitly
   authorized before review is completed through its recorded last round even if the ordinary stopping threshold is
   crossed in the middle; do not extend beyond that last round without new user authorization.

The campaign is reproducible only when the manifest, frozen review-root provenance and episode check records are
kept. Agent summaries or a long launch prompt are not substitutes for those artifacts.

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

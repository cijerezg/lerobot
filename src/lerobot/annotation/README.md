# Annotation

Everything that defines or produces a label lives in this folder.

## The rule

| What | Where |
|---|---|
| Every annotation script | here |
| Every rubric, label spec and vocabulary | here (`rubrics/`, `vocab.py`) |
| Results: inventories, label files, sheets, strips, review pages, reports, logs | outside the repo: `migration/<pass>_<date>/` or `outputs/_annotation/` |
| Labels of record | the `meta/` tables of each dataset root under `outputs/` |

- No annotation script or rubric is written in `migration/`. A script written there for a quick look moves
  here before the pass ends, or is deleted.
- No result is committed to this repo.
- Before writing a function, look for it in the shared modules below. If two passes need the same thing, it
  goes in a shared module and both call it. A pass script holds only what is specific to that pass (its source
  roots, its cut plan, its exemptions).
- Scripts take the results folder as an argument or an environment variable. They never write next to
  themselves. `paths.WORKSPACE` is the folder that holds `lerobot/`, `migration/` and `outputs/`.
- Commands run from the workspace root: `uv run python -m lerobot.annotation.<folder>.<script> ...`.

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
| `rebot/` | A ReBot pass, start to finish (order below). |
| `quality_v2/` | The rubric v2 class pass, shared by every pool: class rules, side-by-side material, dense strips, grading batches, second read, label compiler, checks. |
| `precision_contact/` | Frame grabbing, approach angle, row sheets and strips (used by `quality_v2`), and the text-rule priors for precision and contact. |
| `atoms/` | Subtask-atom pipeline over the diverse corpus: propose, sheets, verdict, agent_check, assemble, manual editor, validator; its reviewer briefs. |
| `speed/` | Speed label: `hybrid_motion_speed` (the adopted method), `speed_annotate`, `joint_motion_speed`, and the runners `rebot_speed_pass` and `hybrid_diverse`. |
| `gripper_events/` | Depth gripper-event labels and their review videos. |
| `review/` | Review videos of training episodes with the labels the trainer consumes burned in. |
| `segments/` | Proprio segmentation (`semantic_segment`). |

Tests: `tests/annotation/`.

## A ReBot pass

A pass is one results folder, `<work> = migration/<pass>_<date>/`. Everything specific to the pass is data in it:

| File in `<work>` | Content |
|---|---|
| `inventory.json` | One record per source episode: `idx`, `key`, `source`, `episode`, `frames`, `digest`, `kind`. |
| `labels/<IDX>.json` | Per source episode: keep ranges, split, task, segments, mistakes, cuts. |
| `pass.json` | `pool`, `dataset`, `staging`, `onset_thr`, `subtasks`, `class_rules`, `reuse`, `nearest`, `idle_exempt`. |
| `plan.json` | Only for a cut without labels: `[{source, episode, keep, cuts, task}]`. |

```bash
M=lerobot.annotation
# 1. look: numbers, sheets, traces
uv run python -m $M.rebot.screen <root> [<root> ...]
uv run python -m $M.rebot.sheets <root> <work>/sheets <episode> --step 90
uv run python -m $M.rebot.traces <work>
# 2. cut only (no labels yet): <work>/plan.json -> root
uv run python -m $M.rebot.build_root <work> --out outputs/<root> --plan
# 3. labelled pass: check the label files, build the staging root, class the segments, render the class material
uv run python -m $M.rebot.validate_labels <work> 0 1 2 ...
uv run python -m $M.rebot.build_root <work> --out outputs/<staging> --split all
uv run python -m $M.rebot.classify <work>
uv run python -m $M.quality_v2.material --work <work>
uv run python -m $M.quality_v2.dense <class_slug> <uid> <from> <to>          # on demand, while grading
# 4. after grading: second read, compile, idle cuts
uv run python -m $M.quality_v2.second_read <pool>
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
```

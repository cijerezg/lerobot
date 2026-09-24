# Archive

Original design/status docs, moved verbatim from the repo root on 2026-07-21 when
the wiki was consolidated. The wiki pages supersede them; kept for provenance
(decision dates, validation logs, full checklists). Code docstrings that reference
these filenames (e.g. `depth_pointmap_design.md`) now resolve here.

| File | Superseded by |
|---|---|
| `depth_pointmap_design.md` | [03 — Depth](../03_depth.md) |
| `depth_pointmap_build_plan.md` | [03 — Depth](../03_depth.md), [08 — Status](../08_status_roadmap.md) |
| `depth_pointmap_gate_gradient.md` | [03 — Depth §B.4](../03_depth.md) |
| `memory_build_plan.md` | [04 — Memory](../04_memory.md), [08 — Status](../08_status_roadmap.md) |
| `memory_probes_plan.md` | [04 — Memory §2.5](../04_memory.md) — moved from the repo root 2026-08-02; its P1/P2/P4/P6 probed the removed summary decode |
| `session_status_2026-07-18.md` | [07 — Data](../07_data_annotation.md), [08 — Status](../08_status_roadmap.md) |
| `state_audit_2026-07-19.md` | [05 — Training](../05_training.md), [08 — Status](../08_status_roadmap.md) |
| `ideas_to_revisit.md` | [08 — Status §Parked](../08_status_roadmap.md) |


## Cleanup record — 2026-09-24

Tier 1–2 cleanup removed disposable artifacts and completed recipes. Active model,
training, inference, annotation, and probe implementations and their configurations
were kept. Dataset/checkpoint payloads, calibration backups, published reports,
original media, saved annotations, and run evidence were kept.

- Root `DIVERSE_ROBOT_TRAINING_INTEGRATION_PLAN.md` and
  `SPEED_ANNOTATION_PROGRESS.md` were condensed into
  [07 — Data: retained integration decisions](../07_data_annotation.md#retained-integration-decisions-september-2026).
  Old snapshot configs may still name those retired documents in historical comments.
  The tracker-only `progress_snapshot.py` was removed; the review guide and editor
  documentation now point at the retained rules and actual collection/assembly tools.
- Completed August recipes were removed: `consolidate_stage_{a,b,c,c2}.py`,
  `verify_consolidation.py`, `annotate_v41.py`, `annotate_v42.py`, `review_v41.py`,
  `review_v42.py`, `split_v41.py`, `extend_val_v41.py`, `merge_v4_multitask.py`,
  `verify_v42.py`, and `verify_v4_multitask.py`. The five v41 and four v42 saved label
  files and `outputs/_staging/provenance.json` were present and retained.
  `verify_v41.py` remains because subsequent rollout/inference audits import it.
- The five September 7 memory-run shell launchers were retired. Their driver logs
  and config snapshot remain; current local/remote probe launchers are unchanged.
- The September 8 `memory_off.py` and September 20 `configure_suite.py` one-time
  config edit scripts were removed. Recent memory config snapshots remain as rollback
  records. The superseded embodiment source/launcher copies and bottle-build `.bak`
  were removed; deployment backups and checksum manifests were kept.
- `old_commands.txt` and 16 files of old scratch code, command/config dumps, and PI05
  inspection output were removed. The four original videos, `analyses.txt`, and the
  PI05 embedding-loader fix note in `old_files/` remain as original assets/findings.
- Generated synthetic/random-network representation HTML, arrays, summary JSON,
  and preview screenshots were removed. Generators and their template remain;
  real-checkpoint sweep results remain. Empty historical logs and development
  Python/test/lint caches were removed.

The action/ViT history dissections, source coordinate audits, September data
preparation pipelines, and reviewed scene manifests remain: they contain reusable
studies, useful evidence, or dependencies of current workflows. No capability was
retired from the active library as part of this pass.

Verification: this pass removed 783 files (59 obsolete/generated files and 724
cache files, including 24 Python scripts overall), totaling 55,247,759 bytes,
and 132 empty cache directories. All 769 inventoried active source/config files
were byte-identical afterward. The 420 retained custom/migration Python files
checked parsed successfully; no retained imports of deleted modules or new broken
links in the edited documentation were found. Prototype HTML/JSON regenerated
identically and all ten saved random-network arrays matched their regenerated values.

Separate filesystem changes occurred during the pass: 1,898 previously resolving
media symlinks lost targets under the public-dataset staging area and
`/home/user/.cache`. None of those targets was in this cleanup's deletion set;
the cause was not established. Dataset integrity therefore remains unverified.


## Custom library source cleanup — 2026-09-24

This follow-up simplified code inside `lerobot/`, with a net reduction of
1695 Python/shell lines and five deleted scripts. No new modules were introduced.

- `rl/inference_utils.py`: 1,070 to 107 lines. Kept RTC's observation conversion and
  action bounding functions unchanged; removed the unreachable non-RTC workers,
  shared state, terminal override console, duplicate action alignment, and episode
  logging. `inference_async.py` already rejects non-RTC execution.
- `probes/attention_budget.py`: removed 11 unused plotting helpers and their orphaned
  import/constants. Current budget, concentration, and history renderers remain.
- `rl/offline_dataset_utils.py`: removed the unused private path-list wrapper.
- Deleted `rl/eval_policy.py`, an unreferenced old evaluator that treated the current
  `(environment, teleoperator)` factory result as an environment.
- Deleted annotation scratch launchers `load_lerobot_high.py`, `annotate_libero.sh`,
  and `run_pgen.sh` (hardcoded paths from another machine and interactive debugging),
  plus `verify_relabelled.py` (a completed four-root historical acceptance recipe).
  The annotation engines, retained validators, original labels, and datasets remain.
- Updated affected inference documentation and the historical audit entry.

Validation: 44 focused action-bound, RTC routing, checkpoint-selection, subtask-state,
and offline-dataset tests passed before and after. Every retained function/class in
the edited implementation modules has the same AST as before. Saved attention data
rendered pixel-identical budget, concentration, and two-checkpoint trajectory PNGs.
No remaining source/test/experiment imports of the deleted modules were found.
These checks did not execute robot hardware, training, or a model forward.

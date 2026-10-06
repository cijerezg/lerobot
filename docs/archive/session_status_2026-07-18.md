# Session status — 2026-07-18 (memory pipeline on real rebot data)

To be unified into memory_build_plan.md / depth docs later. Companion notes: ideas_to_revisit.md (decode-order resolution, ledger-memory idea, pi0.7 takeaways).

## Where we are

**Dataset.** `outputs/rebot_socks_v1` = first real rebot B601 data (6 episodes, 29,192 frames @ 30fps, top + wrist cameras, wrist depth sidecar). Fully annotated copy: **`outputs/rebot-socks-annotated-v2` — training-ready.**

**Cadences (settled).** Subtask decode every 4 s (`subtask_regeneration_interval: 4.0` in root config_rl.yaml, line ~145); summary memory every 12 s (annotation-time grid, default in summary_annotate.py). 3:1 aligned. `chunk_size`/`n_action_steps` raised 30 → 50.

**Memory-first decode (settled + built).** HL answer format flipped to `Memory: <m_{t+1}> Subtask: <subtask>` in `build_generation_answer`/`parse_generation_answer` (processor_molmoact2.py); subtask conditions on fresh memory. Labels/seam/gRPC unchanged; tests updated and green. Summary is strictly retrospective (what happened; never plans). Prospective "ledger" memory (done/remaining) deliberately deferred — see ideas_to_revisit.md.

**Annotation pipeline** (scripts moved to `lerobot/src/lerobot/data_processing/annotate/`, no longer under policies/pi05_full):
1. `summary_annotate.py` — video-only recurrent summaries on a fixed 12 s grid, independent of subtasks; ffmpeg frame extraction (videos are AV1, cv2 can't decode). Output `meta/summaries.parquet` (+`summaries_info.json`), consumed unchanged by the training seam.
2. `subtask_annotate_grid.py` — memory-conditioned subtask label every 4 s window: input = last completed summary + 3 top frames + 2 wrist frames; output must be atomic and progress-free (no ordinals/counts — progress lives in the summary). Writes new dataset via add_features (per-frame `subtask_index`, `meta/subtasks.parquet`, `meta/subtask_windows.json` audit), copies summaries, hardlinks depth.
3. Old whole-episode `subtask_annotate_gemma_4.py` retired for this flow (depth-hardlink fix landed there too). Canonicalize is unnecessary for v2 (vocab already canonical: 13 labels = grasp/move/release × white/black/blue-heeled/blue-striped + return to home).

**Gemma 4 31B verdict.** Summaries: unusable (runaway counting — echo loop incrementing pairs every window; two prompt iterations failed; state-grounded prompt now in script but untested). Subtasks: ~62% window accuracy; systematic misses = releases almost never labeled, wrong colors in cluttered episodes, fumbles labeled "move". No better ≤62B open model exists (Gemma 4 is best-in-class per size; InternVL3.5-38B is a sidegrade; real upgrades are GLM-4.6V FP8 on the DGX or Molmo2-8B for temporal grounding — untried).

**Manual annotations (Claude watched all 6 episodes via contact sheets).** Summaries: 81 windows hand-written (12 s grid, `manual/claude-fable-5` in summaries_info.json). Subtasks: Gemma's 242 windows hand-corrected (91 changed, 38%); original Gemma label kept per-window as `gemma_subtask` in subtask_windows.json. Ground-truth event timelines per episode (drop times, pair orders, both mistakes) are recorded in Claude's memory (project_rebot_socks_v1.md): ep0 = 1 white pair; ep1 = white then black; ep2 = 2 white pairs then black; ep3 = white, black, blue-heeled; ep4 = white, black, blue-striped + MISTAKE (last sock lifted ~162 s, dropped back on table, recovered ~185 s); ep5 = interleaved order + MISTAKE (last black sock dropped beside basket ~176 s, recovered ~205 s). Socks always move ONE at a time; a pair is a completion unit only. ~1/3–1/2 of wall time is grasp fumbling (real, not annotation noise).

## Metadata (DESIGN REVISED + CODE BUILT 2026-07-18; annotation runs pending)

Final design (revised in-session; supersedes the earlier speed plan):
- **Quality**: per-episode 1–5, human, entered in the review tool (earlier proposed scores ep0/ep1/ep3 = 4, ep2/ep4/ep5 = 3 are a reference only; user scores each episode themselves).
- **Mistake**: boolean per 4 s subtask window (π0.7 term + granularity kept after considering "clean"/1–5 severity). Source = LLM pass tuned for recall over the hand-corrected windows, then human review of flagged windows only (unflagged = clean by definition).
- **Speed**: OMITTED. Single-operator data; pace variation is fumbling, which the mistake channel already carries. Backfillable later from step counts / confirmed transfer counts if rollout data ever varies in pace. Seam renders the clause partially (already did); `_extract_metadata` now tolerates the missing column.
- **Storage**: `meta/episode_metadata.parquet` (episode_index, quality, from/to_index) + `meta/mistakes.parquet` (one row per window, boolean) + `meta/metadata_info.json` audit. Window-range parquet instead of the earlier per-frame-column-in-data-parquet plan — mirrors summaries, no dataset rewrite, revisable in place.

Code (all built + tested, 13 tests green, every stage smoke-tested incl. video serving on the real dataset):
1. ONE script: `metadata_annotate.py` (annotate/) with two subcommands (page/finalize CLI removed after UX feedback — user wanted actual video playback + no manual export step).
   - `annotate` — high-recall LLM pass over `subtask_windows.json` windows (judges execution of the KNOWN label), writes `meta/mistake_candidates.json`; `--preview-episode` supported.
   - `review` — local video review UI (`review --data-dir DS`, opens browser itself, default port 8765): per-episode VIDEO playback (Range-streamed, camera switch top/wrist) + quality buttons 1–5; flagged windows play as in-place 4 s clips with LLM evidence, confirm/reject via buttons or j/k/y/n/r. Every click auto-saves server-side to `meta/metadata_review_state.json`; "Write to dataset" button validates (all episodes scored, all flagged windows decided, candidates present) and writes the three meta files directly. Works pre-annotate for quality-only scoring; restart after annotate to get the flagged clips.
2. `materialize_metadata` (buffer.py) reworked: takes (episode_rows, mistake_rows) from new `load_metadata_rows` (offline_dataset_utils.py, hard error if files missing); call sites in rl_offline.py + load_additional_offline_buffers updated; `metadata_default_*`/`metadata_speed_bucket_steps` removed from MemoryConfig (only `metadata_enabled` remains). Inference side already prompts {quality 5, mistake false} (rtc_actor_runtime.py).

## Next steps, in order

1. `metadata_annotate.py review` (score quality from video now) + `annotate` (GPU, Gemma 4 31B) in parallel; restart review, confirm flagged clips, click "Write to dataset".
2. Set `metadata_enabled: true` in config_rl.yaml memory block.
3. Training prep on rebot-socks-annotated-v2: point config_rl.yaml dataset at it, compute norm stats + `compute_delta_stats.py`, build memmap buffer cache (`--image-stride 5`, must match config).
4. First offline training run with summary seam + subtask CE + metadata; then offline eval (`probes/offline_inference.py` evaluates the HL query, shows GT+pred subtask and memory).
5. Later / parked: Phase 6 (frame-history consumption architecture — last unbuilt memory feature), ledger memory ablation, subtask-level exploration for online RL, CFG on metadata clause at inference, leader shadow takeover hardware, speed-channel backfill.

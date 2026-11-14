# VLM judge + Robometer for the sim and hardware evals (2026-10-30)

Two extra progress signals next to the hand-written scorers: a local VLM that names the current subtask from a fixed
vocabulary and the stage on the scorer's ladder, and Robometer-4B, a purpose-trained progress model. Both run on the
Spark behind HTTP; the repo code is a client plus CLIs. Nothing here touches the sim package or the twin folder.

What exists

| piece | where | state |
|---|---|---|
| judge client package | `lerobot/src/lerobot/rewards/vlm_judge/` (config, task_spec, prompt, parse, client, runner, frames, score_episode, compare_to_scorer, mock_server, tasks/*.json) | unit-tested, dry-run against a mock endpoint |
| Robometer client | `lerobot/src/lerobot/rewards/robometer/` (client, score_episode, server) | client + CLI dry-run; server not run |
| Spark scripts | `lerobot/scripts/spark_judge_setup.sh`, `spark_judge_serve.sh`, `spark_judge_smoke.sh`, `spark_judge_hf_server.py` | written, not run |
| tests | `lerobot/tests/rewards/test_vlm_judge.py` (12 tests) | pass |
| dry run outputs | `lerobot/scripts/sim_builders/digital_twin/judge/dry_run/` | mock answers, see below |

## 1. Model choice

| model | HF repo | params / bf16 weights | images per query | licence | role |
|---|---|---|---|---|---|
| Gemma 4 31B instruct | `google/gemma-4-31B-it` | 30.7 B / ~61 GB (vllm recipe: 59 GB) | multi-image and video (frame sequences); 280 visual tokens per image at the default budget (70-1120 selectable); 256K context | Apache 2.0 (licence acceptance on HF still required for download) | primary judge: subtask + stage + progress from an 8-frame strip |
| Qwen3-VL-32B-Instruct | `Qwen/Qwen3-VL-32B-Instruct` | 33 B / ~66 GB | multi-image and video; ~300 tokens per 640x480 image; 256K context | Apache 2.0 | alternative judge, same prompt; swap with `spark_judge_serve.sh qwen` |
| Robometer-4B | `lerobot/Robometer-4B` | Qwen3-VL-4B + 3 heads / ~9 GB | up to 8 frames of one camera, task string; per-frame progress (10-bin expectation) and success probability | Apache 2.0 | second progress signal, no vocabulary, no stage; upstream `reward_model.type=robometer` is inference-only |

Spark facts checked read-only on 10-30: aarch64, driver 580.126.09 (CUDA 13.0), GB10 compute capability 12.1, 121 GB
unified memory, 1.3 TB free, `uv` and `hf` in `~/.local/bin`, an HF token file exists, no Gemma/Qwen/Robometer in the
HF cache. The fork venv has torch 2.12.0+cu130; `.venv-flux3` holds UPSTREAM lerobot main d3793b7 (2026-10-23) at
`~/Documents/Research/RL/LeRobot/flux3_lerobot`, which already contains `lerobot/rewards/robometer/` but not
`qwen_vl_utils` or `fastapi`.

Serving: vllm 0.32.0 (2026-10-18) is on PyPI with `vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl`; the release notes
describe the default wheels as CUDA 13.0 for x86_64 and arm64 (the `+cu129` wheel is the alternate on GitHub releases).
Its supported-models table lists `Gemma4ForConditionalGeneration` and `Qwen3VLForConditionalGeneration` (Gemma 4 since
the 0.31 line; 0.32.0 pins transformers 5.16.1-5.19). GB10 / sm_121 is not a platform vllm lists; forum reports of
older cu130 builds on the Spark mention needing `--enforce-eager`. The setup script asserts `torch.version.cuda`
starts with 13 and the serve script takes `VLLM_EXTRA="--enforce-eager"`; if vllm does not come up, `--fallback`
builds the judge venv with transformers + the FastAPI server `spark_judge_hf_server.py` (same API on 8011, single
request at a time).

Memory plan: one judge + Robometer at a time. Gemma: `--gpu-memory-utilization 0.70` (~85 GB for 61 GB weights + KV)
leaves ~35 GB for Robometer (9 GB) and the system; Qwen: 0.75. Both judges together (127 GB) do not fit. A query is
<= 16 images (2 cameras x 8 frames) ~ 4.5k visual tokens + ~1.2k prompt; `--max-model-len 16384`.

## 2. Spark commands (user runs; nothing is daemonised)

```bash
# on the Spark, from ~/Documents/Research/RL/LeRobot (after syncing lerobot/scripts and lerobot/src/lerobot/rewards)
bash lerobot/scripts/spark_judge_setup.sh              # ~/judge-venv (vllm 0.32.0), ~/robometer-venv (upstream lerobot[robometer]), hf download x3 (~136 GB)
bash lerobot/scripts/spark_judge_setup.sh --fallback   # if vllm fails on GB10: judge venv with transformers + FastAPI instead
bash lerobot/scripts/spark_judge_setup.sh --hf-only    # downloads only

# shell 1                                   # shell 2
bash lerobot/scripts/spark_judge_serve.sh gemma        bash lerobot/scripts/spark_judge_serve.sh robometer
#   (or: ... qwen | ... hf gemma)            #   RBM_VENV=~/Documents/Research/RL/LeRobot/.venv-flux3 works too after
                                              #   `uv pip install --python .venv-flux3/bin/python qwen-vl-utils fastapi uvicorn`
# from the workstation
bash lerobot/scripts/spark_judge_smoke.sh dgx-ip        # /v1/models, one JSON answer about a red image, /health, /score on 2 frames
```

Scoring a recorded episode (workstation, LeRobot root):

```bash
S=outputs/rebot-mix40_20_40-resume1200-20261027/validation/step_00000600/sim/bottle_basket_v1
uv run python -m lerobot.rewards.vlm_judge.score_episode --sim-dir $S --episode 1 --task bottle_basket \
    --out lerobot/scripts/sim_builders/digital_twin/judge/runs/bb_v1_ep1.gemma.json          # default endpoint http://dgx-ip:8011/v1, model google/gemma-4-31B-it
uv run python -m lerobot.rewards.vlm_judge.compare_to_scorer --timeline lerobot/scripts/sim_builders/digital_twin/judge/runs/bb_v1_ep1.gemma.json --events $S/subtask_events.json
uv run python -m lerobot.rewards.robometer.score_episode --sim-dir $S --episode 1 --task bottle_basket \
    --out lerobot/scripts/sim_builders/digital_twin/judge/runs/bb_v1_ep1.robometer.json       # http://dgx-ip:8012
uv run python -m lerobot.rewards.vlm_judge.score_episode --root outputs/rebot_teleop_2026-10-29-cut-v1 --episode 0 --task "basket:pill bottle" --t-from 0 --t-to 60 --out ...
```

`--model Qwen/Qwen3-VL-32B-Instruct` when the qwen server is up; `--max-queries N` for a smoke run; `--period` (0.5 s)
sets both the sample spacing and the strip spacing; `--frames-per-camera` (4) the strip length.

Live use: `JudgeRunner(cfg, TaskSpec.load(name), JudgeClient(cfg))`, `runner.observe(step, t, {"top": rgb, "wrist": rgb})`
every control step; it keeps the last N frames per camera, fires one query every `every_k_steps` (15 = 0.5 s at 30 Hz)
on a worker thread with at most one in flight (a 31B answer takes seconds; the 30 Hz loop never waits), `runner.latest`
is the newest `Judgement` (subtask_id, stage_id, progress, confidence, evidence, cursor, stage_value) and
`runner.history` every answer. Not wired into `rtc_actor_runtime` yet (the seam is the `auto_subtasks.poll()` block,
line ~1403: the judge's `cursor` is the same quantity the scorer returns; the operator keys stay authoritative on hardware).

## 3. Task vocabularies (`vlm_judge/tasks/*.json`)

Each spec: `task` string, the policy's `script` (eval_subtasks), `subtasks` (id, text, observable criterion, contact
slug from annotation/vocab.py, precision 1-5, `cursor` = the script index it belongs to, -1 = off-task) and `stages`
(the scorer's STAGES with stage 0 = nothing reached, plus a criterion). `basket.json` is a template over
`{object}` in {pill bottle, cup, tape roll, bottle} with the grasp contact per object (`TaskSpec.load("basket:tape roll")`).

| spec | subtasks | stage ladder (value) | ladder source |
|---|---|---|---|
| bottle_basket | reach toward / grasp / lift / carry to the basket / place in the basket / retreat / idle-off-task | none 0, reach 0.1, held 0.3, lifted 0.5, over 0.8, inside 1.0 | `easy_tasks.BottleBasketScorer` (asserted equal in the tests) |
| basket:{object} | same seven, object substituted | same ladder | mirrors bottle_basket; the twin basket scenes should keep it |
| fold_kit | reach / pinch the bits-flap edge, swing it over the middle, release it, reach / pinch the card-flap edge, swing it over, set it down, press flat, retreat, idle | none 0, reach_bits_flap 0.1, bits_flap_pinched 0.2, bits_flap_over_middle 0.4, bits_flap_folded 0.5, card_flap_pinched 0.6, card_flap_over 0.8, kit_folded 0.9, pressed_flat 1.0 | PROPOSAL (T067 steps 1-5); align with the twin agent's fold scorer when it lands |
| close_lid | reach the rim, pinch the rim, swing closed, let go, retreat, idle | none, reach 0.1, quarter 0.4, three_quarters 0.7, closed 1.0 | `CloseLidScorer` |
| nail_clippers | reach / grasp / carry / insert / push for each clipper, retreat, idle | none, reach_large 0.1, held_large 0.2, near_large 0.3, seated_large 0.5, held_small 0.6, near_small 0.7, seated_both 1.0 | `NailTaskScorer` |

Prompt (`prompt.py`): one system message with the task, the numbered vocabulary (criterion, contact, precision) and the
numbered ladder, the rule "judge the latest frame, use earlier frames for motion", progress defined as a value between
the chosen stage's value and the next one, and the JSON contract `{subtask_id, stage_id, progress_0_1, confidence,
evidence}`; one user message interleaving `"<camera> camera, t=12.5 s:"` labels and JPEG data URIs, oldest first;
`response_format: json_object`. The parser strips fences, takes the first object, requires all five keys, integer ids
in range, numbers in [0, 1]; anything else raises `JudgeParseError` (the runner logs and keeps the last answer, the CLI
counts a failure).

## 4. Calibration procedure (against the hand-written scorers)

1. Pick episodes with scorer events: `outputs/rebot-mix40_20_40-resume1200-20261027/validation/step_00000600/sim/`
   has 8 episodes per layout with `subtask_events.json` and 30 fps top/wrist videos (bottle_basket_v1 ep 1, 7 and
   v2 ep 3-5, v5 ep 0, 2, 4-6 are successes; close_lid_v2/v4 ep 3 closed). Take ~10 basket, ~10 lid, ~10 clipper
   episodes spanning final stages.
2. `score_episode` each with Gemma, then with Qwen (same strip), then `robometer.score_episode`; collect the
   `compare_to_scorer` metrics: cursor accuracy per step (and within 1), stage accuracy (raw and running max), per-stage
   lag in seconds (judge first reaches the stage minus scorer's t; negative = early), Spearman(judge progress, scorer
   cumulative reward), confusion over cursors. Robometer timelines share the field names, so Spearman applies to them
   with the same tool (cursor/stage fields absent: run `compare` on progress only, or add the two columns as NaN).
3. Read the disagreements by `evidence`: systematic ones are prompt or criterion fixes (edit the JSON, rerun); random
   ones set the confidence threshold below which the judge's cursor is ignored.
4. Decide per signal: a stage lag p95 under ~1.5 s and cursor accuracy over ~0.8 on the sim set makes the judge a
   usable advance signal; Spearman over ~0.8 makes Robometer's progress a usable dense shaping term. Below that, keep
   them as logged side channels only. Then repeat on 3-5 hardware episodes (annotated roots carry the subtask ranges;
   the operator's n/b presses are the cursor truth there) before trusting either on hardware.
5. Alignment caveat: sim video frame k is at k/30 s on the env clock, events `t` are sim seconds since the reset; the
   two agree up to the inference_delay_steps lockstep (0.3 s at 9 steps). Lags under 0.5 s are within one period.

## 5. Dry run (mock endpoint, `vlm_judge/mock_server.py`; files in `judge/dry_run/`)

- `bottle_basket_v1_ep1.judge.json`: 91 queries at 0.5 s over the 45 s success episode, 0 failures, strips of 2 -> 8
  images, rows carry t, subtask, stage, progress, progress_max, confidence, evidence, latency.
- `bottle_basket_v1_ep1.compare.json`: the comparison tool ran end to end on the real `subtask_events.json` (5 events,
  success at 45.43 s). The numbers (cursor accuracy 0.077, Spearman 0.60, lags -17 to -36 s) are the mock's canned
  ramp and mean nothing.
- `bottle_basket_v1_ep1.robometer.json`: 6 queries, per-frame vectors of length 1..6 returned.
- `teleop_ep0.judge.json`: LeRobot v3 dataset path (AV1 video, episode timestamps from `meta/episodes`) on
  `rebot_teleop_2026-10-29-cut-v1` ep 0 with `basket:pill bottle`, 5 queries; no scorer there, so no comparison.

## 6. Untested

- Everything on the Spark: vllm 0.32.0 on GB10 (wheel CUDA, sm_121 kernels, CUDA graphs), Gemma 4 / Qwen3-VL image
  handling through vllm's OpenAI API with `response_format json_object`, the memory split, the HF fallback server,
  the Robometer server (its calls match upstream d3793b7: `RobometerEncoderProcessorStep.encode_samples`,
  `_compute_rbm_logits`, module-level `decode_progress_outputs(..., is_discrete_mode=config.use_discrete_progress)`,
  but it has not been executed).
- Judge quality: no real answers yet; the vocabulary criteria and the progress definition are untried on the models.
- Gemma download needs the HF licence click; the `hf` CLI will say so.
- `basket.json` and `fold_kit.json` ladders have no scorer to compare against until the twin agent's scenes exist.

## Revision 2026-10-30 afternoon (user decisions)

- No per-frame progress model: Robometer and Qwen3-VL-32B are optional (`spark_judge_setup.sh --all`), not part of the plan.
- Offline scorer = Gemma 4 31B-it on the recorded episode videos, after the fact: frames every 1 s, one query per 5 s window (`score_episode.py --period 1 --frames-per-camera 5`, stride = frames per camera). Latency is irrelevant offline.
- Live subtask assigner = a small model given the per-task vocabulary: Qwen/Qwen3-VL-8B-Instruct (`spark_judge_serve.sh assigner`, port 8013, 0.15 of the Spark memory so it runs beside Gemma). Gemma 4 12B is the fallback if the 8B misassigns.
- All model weights live on the Spark only (~/.cache/huggingface/hub there); nothing is downloaded to the workstation. The clients talk to the Spark over HTTP (dgx-ip while mDNS is down).

## Revision 2026-10-30 night: assigner + twin specs built, code on the Spark, models not yet downloaded

- Assigner model = `Qwen/Qwen3.5-9B` (2026-02, 9.65 B, natively multimodal, `Qwen3_5ForConditionalGeneration`): the newest small
  Qwen VLM on HF (the Qwen3.8 line, 2026-08, starts at 27B; Qwen3-VL-8B is 2025-10 and stays as the A/B `assigner_vl8b`, setup `--all`).
  Both architectures are in the vllm 0.32.0 registry; vllm 0.32.0 on aarch64 resolves to torch 2.13.0 + CUDA 13 runtime (checked by
  resolution, nothing installed). Gemma 4 31B-it is no longer gated on HF.
- Vocabulary for the assigner = the policy's own script (`eval_subtasks`: tasks.py per sim layout, config_rl.yaml on hardware),
  offered per object segment only (`vlm_judge/assigner.py`): a segment starts at every "grasp ..." step; the model sees the current
  segment + the next segment's first step + the step finished before it as context. Twin basket = 3-4 options, L4 = 4, the
  20-step hardware script <= 4. Answer `{"step", "confidence"}` (~15 tokens), thinking off (`chat_template_kwargs`), 3 frames per
  camera at 0.5 s. Per-step "working on it while" lines come from the verb (grasp/move/release/push/close/fold/press/return).
- Gemma specs for every sim layout are generated from tasks.py: `TaskSpec.load("sim:<layout>")` (`vlm_judge/sim_spec.py`):
  script, contacts, precisions and the scorer's STAGES (names, values, cursors) exactly, criteria from the twin_tasks thresholds;
  old layouts map to their JSON (asserted equal). The hand JSONs stay for hardware roots.
- Batch CLIs over a sim run (thread pool across episodes, resumable, summary.md per run): `score_episode --sim-run` (Gemma, compared
  to the scorer per episode) and `assign_episode --sim-run` (closed loop like live, or `--teacher` = window from the scorer's cursor).
  Dry run against the mock on the res600 twin run: 16 assigner + 4 judge episodes end to end.
- Memory split on the Spark: gemma 0.60 (~73 GB), assigner 0.25 (~30 GB); start gemma first.
- Not wired into rtc_actor_runtime yet: wait for the calibration numbers (seam unchanged, `AssignerRunner.cursor` in, `latest.cursor` out).
- 10-30 late (user): EVERYTHING runs on the Spark (sims, scoring, assigner, models). Client defaults = localhost (8011 / 8013 / 8012);
  lerobot/src, lerobot/scripts and the two twin eval runs (outputs/twin_eval_{res600,internal1200}_2026-10-30, 6.4 GB) rsynced to the
  Spark; the workstation only pushes code. The vllm processes stay (judge-venv torch 2.13 vs the fork venv torch 2.12), on localhost.

## Revision 2026-11-04: assigner dropped

User decision: no live subtask assigner. The sim scorer / hardware console keep the cursor. Gemma 4 31B-it remains as the offline
reward assigner over recorded videos; calibration against the scorer on the two twin runs is the open step (section 4 criteria,
reward/stage agreement; the cursor-accuracy criterion now only grades Gemma's stage reading, it drives nothing).

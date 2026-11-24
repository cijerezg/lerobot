# vlm_knowledge

Question: what do the fine-tuned models still know, measured on public benchmarks, and which part (language model,
vision encoder) loses it? Models: Molmo2-ER (`er`, the original VLM), MolmoAct2 (`base`), our checkpoints.

## Run (from the workspace root)

```bash
uv run python lerobot/adhocs/vlm_knowledge/prepare_benchmarks.py     # mmlu_redux, cv_bench, embspatial, vqa_v2 -> outputs/data/vlm_benchmarks
uv run python lerobot/adhocs/vlm_knowledge/prepare_sim_scenes.py     # sim_scenes (144 yes/no questions on 18 sim images)
uv run python lerobot/adhocs/vlm_knowledge/run_benchmarks.py --config_path=config_rl_validate.yaml \
  --models=er,base,internal=outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model,tube700=outputs/runs/rebot-v2-tube-b128-20261114/checkpoints/000700/pretrained_model \
  --benchmarks=mmlu_redux,cv_bench,embspatial,vqa_v2,sim_scenes --grid=true --mc_style=molmo \
  --out=outputs/analysis/vlm_knowledge_2026-11-17/answers
uv run python lerobot/adhocs/vlm_knowledge/summary.py                # summary.md: accuracy with intervals, paired differences
uv run python lerobot/adhocs/vlm_knowledge/mistakes.py               # mistakes.md: per-question overlap, categories, examples
uv run python lerobot/adhocs/vlm_knowledge/lost_questions.py         # lost_questions.md: what kind of questions are lost (needs question_types/ labels)
```

`--grid=true` runs every language model on every vision encoder for the image benchmarks. Finished answer files
are skipped, so the run resumes. The full grid takes under 2 h on one GPU (about 30 questions/s).

## Readings

Each question is answered once with one 378 px crop (`--crop_mode=resize`, as in our runs) and scored three ways:
`free` (greedy over the whole vocabulary), `masked` (action tokens forbidden), `letters` (most likely option
letter at the first position; multiple choice only). Use `letters` for multiple choice: base and tube 700 start a
sentence instead of a letter on about 10% of MMLU-Redux and `masked` counts those wrong.

## Findings (2026-11-17)

Report: `lerobot/docs/reports/vlm_knowledge_2026-11-17.md`.

| | er | base | tube 700 | internal 1200 |
|---|---|---|---|---|
| MMLU-Redux (`letters`) | 71.0 | 67.7 | 60.3 | 55.8 |
| CV-Bench | 85.4 | 86.9 | 72.4 | 72.0 |
| EmbSpatial | 77.7 | 78.4 | 51.5 | 42.6 |
| VQA v2 (2,000) | 84.1 | 83.0 | 56.0 | 51.1 |
| same with base vision: CV / Emb / VQA | | | 85.3 / 74.1 / 81.1 | 84.2 / 63.5 / 78.8 |

- Harness check: Molmo2-ER with its own crops gets CV-Bench 87.3 (published 87.8), EmbSpatial 79.5 (78.8).
- The mistakes are largely the same questions across models; ours add mistakes on top of base's, and the two runs
  add mostly the same ones.
- Most lost image questions come back with base's vision encoder. internal's language model also lost part
  (EmbSpatial) and prefers option A.
- Lost text questions are the ones that take several steps (42% lost vs 14% for one step), in any subject; robotics-related
  questions are not better kept. Lost image questions: counting more than one object, naming, viewer-relative position.

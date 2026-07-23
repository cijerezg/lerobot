# pi07 wiki

The single source of truth for **pi07** — our custom re-implementation of the π0.7
recipe (Physical Intelligence, arXiv 2604.15483) built on top of the **MolmoAct2**
VLA, extended with a **metric depth path** the original does not have. This wiki
consolidates the design docs that used to live scattered at the repo root (originals
preserved verbatim in [archive/](archive/)).

Target robot: **rebot B601** (7-DOF). Base checkpoint: `allenai/MolmoAct2`
(multi-embodiment foundation) at `outputs/MolmoAct2`. Live config: root
[`config_rl.yaml`](../../config_rl.yaml).

## Table of contents

| Page | Contents |
|---|---|
| [01 — Overview](01_overview.md) | What pi07 is, the four subsystems, system diagram, repo map, glossary |
| [02 — Base model: MolmoAct2](02_base_model.md) | VLM + action expert architecture, flow-matching math, token/prompt layout, anchor action encoding |
| [03 — Depth: point-map + co-evolving stream](03_depth.md) | Back-projection math, patch tokens, DepthStream, the gated additive read, bit-identity guarantee, critic depth read |
| [04 — Memory & prompts](04_memory.md) | Full prompt anatomy; short-term history; two-prompt subtask generation; MEM summary memory; metadata steering |
| [05 — Training](05_training.md) | `rl_offline.py` pipeline, all losses with formulas, freeze/optimizer rules, distributional critic (HL-Gauss), buffer + memmap cache |
| [06 — Inference runtime](06_inference.md) | RTC actor runtime, HL decode cadence, actor-side history deque, depth at inference, prompt defaults |
| [07 — Data & annotation](07_data_annotation.md) | rebot datasets, the annotation chain (summaries, subtasks, metadata), tooling |
| [08 — Status & roadmap](08_status_roadmap.md) | What is on/off in the current run, open items, parked ideas |

## Where the code lives

| Subsystem | Package |
|---|---|
| Policy wrapper + processor | `src/lerobot/policies/molmoact2/` |
| Depth point-map + stream | `src/lerobot/policies/depth_pointmap/` |
| RL trainer + critic | `src/lerobot/rl/molmoact2/` |
| Replay buffer, RTC runtime, shared config | `src/lerobot/rl/` |
| Annotation tools | `src/lerobot/data_processing/annotate/` |
| Probes (bit-identity, offline eval) | `src/lerobot/probes/` |

## Reading order

New to the project (or lost track): 01 → 02 → 03 → 04, then 05/06 as needed.
Resuming work: [08 — Status & roadmap](08_status_roadmap.md) first.

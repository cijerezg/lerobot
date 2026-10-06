This file provides guidance to AI agents when working with code in this repository.

> **Documentation index → [`docs/README.md`](./docs/README.md)**. Project state → [`docs/status.md`](./docs/status.md). Commands → [`CHEAT_SHEET.md`](./CHEAT_SHEET.md).

## Project Overview

A research fork of LeRobot (PyTorch robotics library) that trains and runs **MolmoAct2** on the
**rebot B601** (7-DOF: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll,
gripper) with a 102HD leader arm. The pipeline is offline-then-online RL (RECAP-style critic,
advantage-weighted regression), a metric depth path, prompt-rendered history and metadata
steering, and a probe suite. The design docs call the recipe **pi07**; that is a codename.
Base model: the multi-embodiment foundation checkpoint `allenai/MolmoAct2`.

MolmoAct2 is the only policy the offline trainer accepts. Upstream policies, sim envs and
robots are still present in `src/` but are not on the active path and are not maintained here.
The $\pi_{0.5}$ RL path (`policies/pi05_full/`, `rl/pi05/`) is legacy.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (transformers, datasets, Hub, accelerate) · draccus (config/CLI) · uv (package management) · Aim (metrics)

## Development Setup

```bash
uv sync --locked                                                  # Base dependencies
uv sync --locked --extra molmoact2 --extra async --extra training # The active pipeline
uv sync --locked --extra test --extra dev                         # Test + dev tools
```

The user's workspace keeps this clone as `lerobot/` next to `config_rl.yaml` (the live run
config), `outputs/` (datasets, checkpoints, probe runs) and `migration/` (scratch). Those are
not in git; `CHEAT_SHEET.md` commands are written from that workspace root.

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                                     # All tests
uv run python -m lerobot.scripts.rl_offline --config_path=config_rl.yaml  # Offline training
uv run python -m lerobot.rl.inference_async --config=config_rl.yaml       # RTC inference
uv run python -m lerobot.scripts.view_probes outputs/<run>                # Probe viewer
pre-commit run --all-files                                                # Lint + format
```

## Architecture (`src/lerobot/`)

Active path:

- **`policies/molmoact2/`** — policy wrapper, processor (prompt rendering, state discretization, advantage tokens), anchor encoding, action layout, future-visual auxiliary. `ARCHITECTURE.md` is the network reference.
- **`policies/depth_pointmap/`** — point-map back-projection, depth tokens, depth history.
- **`rl/`** — replay buffer, memmap cache, RTC actor runtime, shared config, AWR calibration, data sources (rebot + diverse corpus mixer). `rl/molmoact2/` holds the trainer, hybrid critic and val loss. `RL_NOTES.md` documents the `Trainer` seam.
- **`scripts/rl_offline.py`** — offline training entry point (single process or DDP via `accelerate launch`). Other fork scripts: `compute_delta_stats.py`, `lerobot_memmap_buffer_cache.py`, `view_probes.py`, `compare_probes.py`, `model_explorer.py`.
- **`probes/`** — validation probes run at every `val_freq`; each writes an `index.json` manifest the viewer reads. `README.md` and `MODEL_TENSORS.md` say where each probe reads the model.
- **`data_processing/annotate/`** — subtask, summary, metadata, speed, precision, contact and gripper-event annotation tools and review UIs.
- **`datasets/`** — `LeRobotDataset` plus the diverse-corpus loaders (`diverse_corpus.py`, `diverse_prompt.py`, `contact_vocab.py`).
- **`robots/rebot_b601_follower/`, `teleoperators/rebot_102_leader/`** — the hardware.

Upstream (inherited, not maintained): the other `policies/*`, `envs/`, the other robots and teleoperators, `examples/` except `examples/dataset/diverse_robot_dataset/`.

Conventions:

- **`configs/`** — Dataclass configs parsed by draccus. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")`.
- **`processor/`** — `ProcessorStep` pipeline between dataset, policy and robot.
- The generic layer (`rl/`, `data_processing/`) never imports a policy; policy-specific behavior lives in the policy seam (`policies/molmoact2/`, `rl/molmoact2/`). See `docs/design/overview.md`.

## Repository Structure (outside `src/`)

- **`docs/`** — all project documentation. `README.md` is the index and states the rules: `guide/` (how to use), `design/` (how it is built), `notes/` (dated notes with a status line), `runbooks/`, `reports/`, `archive/`, and `status.md` as the only snapshot page.
- **`CHEAT_SHEET.md`** — the one command sheet.
- **`tests/`** — pytest suite by module. Fork tests: `tests/rl/`, `tests/probes/`, `tests/policies/test_molmoact2_*.py`, `tests/datasets/test_diverse_*.py`, `tests/teleoperators/test_rebot_*.py`.
- **`scripts/`** — ops shell scripts: `remote_validate.sh`, `chase_validate.sh`, `run_probes.sh`, probe helpers.
- **`examples/dataset/diverse_robot_dataset/`** — the diverse-corpus build pipeline (production tooling, not an example).
- **`.github/workflows/`** — `quality.yml` (pre-commit), `fast_tests.yml`, `full_tests.yml`, `security.yml`.
- **Root files**: `pyproject.toml`, `Makefile`, `uv.lock`, `CONTRIBUTING.md`, `README.md`.

## Working rules for agents

- **Docs follow the lifecycle in `docs/README.md`.** A new idea gets a note in `docs/notes/` with a status line. When it lands, update the `design/` page and archive the note. When it dies, archive the note and delete the code in the same commit. Only `docs/status.md` carries "as of" statements.
- **Read the value, not the comment.** Comments in `config_rl.yaml` and prose in docs drift faster than code. Verify against the code before stating what is on.
- **Joint frames.** The pipeline operates entirely in the raw arm frame. The old SO-101 v3.0↔v2.1 conversion (`frame_so101.py`) was removed 2026-07-04; stats files produced with it are rejected at load. Norm stats come from the training dataset; anchor/delta stats from `compute_delta_stats.py` on the same dataset.
- **Optional dependencies**: new imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`.
- **Prioritize `uv run`** over raw `python` or `pip`.

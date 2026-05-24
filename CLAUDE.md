This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## MolmoAct2 — SO-101 Joint Frame Convention (CRITICAL)

MolmoAct2 was pretrained on SO-100/101 data in the **v2.1 joint convention**.
All datasets recorded with LeRobot v3.0 (this repo) are in the **v3.0 convention**.
Two joints differ between conventions:

| Joint | Transform: v3.0 → v2.1 |
|-------|------------------------|
| 1 · shoulder_lift | `v2.1 = −v3.0 + 90` |
| 2 · elbow_flex    | `v2.1 =  v3.0 + 90` |
| 0, 3, 4, 5        | unchanged              |

**Where this is handled:** `src/lerobot/policies/molmoact2/frame_so101.py`
- `SO101V3ToV21Step` — inserted before the normalizer in the input pipeline (converts state + action in training data and live observations)
- `SO101V21ToV3Step` — inserted after the unnormalizer in the output pipeline (converts model actions back to arm frame before sending to robot)

**Implications for training:**
- Training datasets recorded with LeRobot v3.0 are in v3.0 convention → the processor transform is correct and must stay in place.
- If you ever use a dataset recorded in v2.1 convention (e.g., raw HF hub datasets from before PR-777), remove or bypass `SO101V3ToV21Step` in the input pipeline — otherwise joint angles will be double-converted.
- To verify a dataset's convention: check `observation.state` mean for joint 1 (shoulder_lift). v3.0 ≈ −30° to +10°; v2.1 ≈ 90° to 130°.

**Implications for inference:**
- `norm_tag: so100_so101_molmoact2` is required when loading the zero-shot base checkpoint (`outputs/MolmoAct2-SO100_101`) because norm stats are not embedded in that checkpoint directory. Set `norm_tag: null` only when loading a fine-tuned checkpoint that already has norm stats in its `policy_preprocessor_*.safetensors`.
- `action_clamp_limits` in `config_rl.yaml` are defined in **v3.0 arm frame** (clamping runs after `SO101V21ToV3Step`).

**The TODO anchors** (`# TODO(anchor): remove when anchor deltas land`) in `frame_so101.py` and `processor_molmoact2.py` mark where to remove this transform once the model switches to anchor/delta action encoding, which is inherently frame-agnostic.

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).

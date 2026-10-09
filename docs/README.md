# Documentation

This fork of LeRobot trains and runs **MolmoAct2** on the **rebot B601** (7-DOF) with an
offline-then-online RL pipeline (RECAP-style critic, advantage-weighted regression), a metric
depth path, prompt-rendered memory and metadata steering, and a probe suite for looking
inside the model. The design pages still call the recipe **pi07** (our re-implementation of
the π0.7 recipe on MolmoAct2); that is the codename, not a separate model.

Everything that is not code lives here. One entry point, one cheat sheet, one status page.

| Where                                       | What                                                                  | Rule                                                |
| ------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------- |
| [`status.md`](status.md)                    | What is on/off in the current run, next steps, parked ideas, footguns | The **only** snapshot page. Dated.                  |
| [`guide/`](#guide--how-to-use-the-pipeline) | How to **use** the pipeline                                           | Timeless. No dates, no "as of".                     |
| [`design/`](#design--how-it-is-built)       | How the system **is built**, one page per subsystem                   | As-built reference. Kept current when code changes. |
| [`notes/`](#notes--the-ideas-lab)           | Dated design notes, investigations                                    | Every file opens with a **Status** line.            |
| [`runbooks/`](#runbooks)                    | What to do when a known failure happens                               | Symptom → root cause → fix.                         |
| [`reports/`](#reports)                      | Saved analyses with their numbers                                     | Frozen once written.                                |
| [`archive/`](archive/README.md)             | Superseded or abandoned documents                                     | Index says what replaced each file.                 |
| [`../CHEAT_SHEET.md`](../CHEAT_SHEET.md)    | The copy-paste command sheet for the rebot setup                      | The only cheat sheet.                               |

In-source markdown describes **that package's code only** and nothing about the project:
[`policies/molmoact2/ARCHITECTURE.md`](../src/lerobot/policies/molmoact2/ARCHITECTURE.md)
(network reference), [`probes/README.md`](../src/lerobot/probes/README.md) and
[`probes/MODEL_TENSORS.md`](../src/lerobot/probes/MODEL_TENSORS.md) (where probes read the
model), [`rl/RL_NOTES.md`](../src/lerobot/rl/RL_NOTES.md) (trainer seam and critic design).

## Start here

- **Run something**: [`../README.md`](../README.md) for install and the first training run,
  then [`guide/usage.md`](guide/usage.md).
- **Understand the build**: [`design/overview.md`](design/overview.md) →
  [`base_model.md`](design/base_model.md) → [`depth.md`](design/depth.md) →
  [`memory_prompts.md`](design/memory_prompts.md), then
  [`training.md`](design/training.md) / [`inference.md`](design/inference.md) as needed.
- **Resume work**: [`status.md`](status.md) first.

## `guide/` — how to use the pipeline

| Page                           | Covers                                                                                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`usage.md`](guide/usage.md)   | Action encodings, offline→online flow, full config reference, pretrained merge, the RL loop, v2.1 datasets, interventions, async inference, action post-processing, buffer caching, probes |
| [`recap.md`](guide/recap.md)   | RECAP: the critic, advantage conditioning, losses, training loop, freezing, rewards, hyperparameters, code map                                                                             |
| [`awr.md`](guide/awr.md)       | Frozen-critic advantage-weighted regression: weight calculation, calibration, launch, probe                                                                                                |
| [`probes.md`](guide/probes.md) | The validation probes: what each computes, standalone commands, config                                                                                                                     |

## `design/` — how it is built

| Page                                              | Subsystem                                                                                           |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| [`overview.md`](design/overview.md)               | The four subsystems, system diagram, the generic-layer / policy-seam rule, glossary                 |
| [`base_model.md`](design/base_model.md)           | MolmoAct2: VLM + action expert, flow matching, token and prompt layout, anchor encoding             |
| [`depth.md`](design/depth.md)                     | Point-map tokens, depth history, the depth read, critic depth read                                  |
| [`memory_prompts.md`](design/memory_prompts.md)   | Prompt anatomy, short-term history, subtask generation, metadata steering                           |
| [`training.md`](design/training.md)               | `rl_offline.py`: losses, freeze and optimizer rules, distributional critic, buffer and memmap cache |
| [`inference.md`](design/inference.md)             | RTC actor runtime, HL decode cadence, history deque, depth at inference                             |
| [`data_annotation.md`](design/data_annotation.md) | Datasets, the annotation chain and its tools, retained integration decisions                        |

## `notes/` — the ideas lab

Annotation rubrics and label specs are not here: they live with the annotation code in
[`src/lerobot/annotation/`](../src/lerobot/annotation/README.md) (`rubrics/`).

Each note opens with `> **Status:** ...`. Vocabulary: **idea** · **building** · **built, on** ·
**built, off** · **decided** · **fixed** / **resolved** · **in force** (rubrics) ·
**superseded by X** · **abandoned**. The date on the status line is the date of that status.

| Note                                                                             | Status                        | About                                                                               |
| -------------------------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------- |
| [`action_trajectory_losses.md`](notes/action_trajectory_losses.md)               | reference, 2026-09-02         | What the flow/FAST trajectory terms are, what is broken, option catalogue           |
| [`principled_action_losses.md`](notes/principled_action_losses.md)               | decided                       | Which objective change is defensible and what it assumes                            |
| [`fast_tokenizer_alphabet_bug.md`](notes/fast_tokenizer_alphabet_bug.md)         | fixed 2026-08-08              | Silent DCT coefficient deletion in the FAST tokenizer                               |
| [`depth_history_design.md`](notes/depth_history_design.md)                       | built 2026-07-25              | Temporal attention inside the depth patch encoder                                   |
| [`depth_redesign_options.md`](notes/depth_redesign_options.md)                   | decided + built 2026-07-26    | Decision record for the depth read                                                  |
| [`mem_temporal_attention_analysis.md`](notes/mem_temporal_attention_analysis.md) | built 2026-08-03; off         | RGB history temporal attention: spec, deviation, measurements                       |
| [`future_visual_prediction.md`](notes/future_visual_prediction.md)               | built, off                    | Predict a frame four seconds ahead to keep RGB memory informative                   |
| [`offline_accelerate_plan.md`](notes/offline_accelerate_plan.md)                 | built                         | DDP for `rl_offline.py` via `accelerate launch`                                     |
| [`leader_102hd_actuation.md`](notes/leader_102hd_actuation.md)                   | built 2026-09-03              | Driving the 102HD leader: HD driver, policy preview, shadowing                      |
| [`leader_bus_investigation.md`](notes/leader_bus_investigation.md)               | resolved 2026-09-06           | Leader byte loss and overvoltage: root causes and workarounds                       |
| [`open_questions.md`](notes/open_questions.md)                                   | living list                   | Unresolved repo-level questions                                                     |
| [`ee_mixture_loss/TODO.md`](ee_mixture_loss/TODO.md)                             | building, 2026-10-09          | The EE mixture loss (hand block, FK term, masked FAST): plan, decisions, phases; its folder holds the proposal and the loss notes |

## `runbooks/`

- [`system_overcommit.md`](runbooks/system_overcommit.md): `[Errno 12] Cannot allocate memory`
  from imageio/ffmpeg during probes; Linux overcommit accounting and the sysctl fix.

## `reports/`

Saved analyses. The HTML embeds its numbers and opens without a model or server; frame links
inside point at the original `outputs/probe_runs/...` directory on the training machine.

- [`critic_gradients_v2_2000/`](reports/critic_gradients_v2_2000/README.md): critic input
  sensitivity with continuous state and depth, checkpoint 2000, plus the comparison with v1.
- [`critic_gradients_2000/`](reports/critic_gradients_2000/README.md): the original critic,
  checkpoint 2000.

## Rules

1. **Status line on every note.** A note in `notes/` starts with `> **Status:** <word>, <date>`
   plus one sentence on what that means for the code or config. Change the line when the
   status changes; do not leave the reader to infer it from the prose.
2. **Design pages are the truth, notes are the history.** When an idea lands, fold the
   as-built description into the relevant `design/` page and move the note to `archive/`
   with a row in [`archive/README.md`](archive/README.md). When an idea dies, same move,
   and the code goes in the same commit.
3. **`guide/` and `design/` never carry dates.** Anything that would need "as of" belongs
   in `status.md` or in a note.
4. **The repo cites files outside itself by plain path, not by link.** The live run config
   (`config_rl.yaml`), the scratch directory (`migration/`) and run outputs (`outputs/`) sit
   at the workspace root next to this clone and are not in git. The in-package
   [`src/lerobot/rl/config_rl.yaml`](../src/lerobot/rl/config_rl.yaml) is the template.
5. **In-source markdown describes its package only.** Anything about the project, a decision
   or a status goes here.

### Adding a note

Create `notes/<topic>.md`:

```markdown
# <Title>

> **Status:** idea, 2026-MM-DD. <What this means for code/config right now.>

<Why, what, how to measure.>
```

Add a row to the table above. When the status changes, change the line and the row.

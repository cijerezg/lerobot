# Accelerate / multi-GPU plan for `rl_offline.py`

Status: implementation plan, not yet built. Written 2026-08-15 after the
pre-implementation cleanup.

## Outcome

Make the existing offline learner run unchanged on one GPU and use DDP through
Hugging Face Accelerate when launched on multiple GPUs.

The first supported production path is deliberately narrow:

- policy: `molmoact2_rl`;
- offline actor training: `skip_critic: true`;
- current action, depth, memory, metadata, and auxiliary losses;
- current replay-buffer iterator and multiple weighted dataset sources;
- all validation probes and `ValLoss` execute only on the main process/GPU;
- one Aim run and one set of checkpoints;
- one machine initially (two GPUs expected, four possible);
- DDP only: every GPU holds a complete model and optimizer replica.

The distributed plumbing should live at the generic Trainer/offline-loop seam so
future policies can opt in without rewriting process management. PI05, critic
updates, FSDP, DeepSpeed, multi-node execution, and distributed probes are not
part of this implementation.

## Current configuration contract

The active configuration in `src/lerobot/rl/config_rl.yaml` currently has:

- `policy.type: molmoact2_rl`;
- `skip_critic: true`;
- `batch_size: 32`;
- `policy.gradient_accumulation_steps: 4`;
- `policy.subtask_loss_weight: 0.0`;
- `policy.dtype: bfloat16`;
- CPU-backed memmap replay caches;
- `log_freq: 20`, `val_freq: 400`, and `offline_save_freq: 400`;
- `val_loss_frames: 128` and the full probe suite.

`batch_size` remains the **per-process** replay batch. The effective batch is:

```text
batch_size * gradient_accumulation_steps * number_of_processes
```

To preserve today's effective batch of 128:

| GPUs | batch/GPU | gradient accumulation | effective batch |
|---:|---:|---:|---:|
| 1 | 32 | 4 | 128 |
| 2 | 32 | 2 | 128 |
| 4 | 32 | 1 | 128 |
| 8 | 16 | 1 | 128 |

Do not silently divide accumulation in code. Log the effective batch prominently
and let the run config make this scientific choice explicit.

## Non-negotiable invariants

1. A normal single-process invocation still works; Accelerate barriers and
   reductions become no-ops.
2. Selecting physical GPU 1 works with `CUDA_VISIBLE_DEVICES=1`; inside the
   process that GPU is logical `cuda:0`. The config must not select `cuda:1`.
3. Every DDP rank samples different replay transitions. Identically seeded ranks
   would perform redundant training and are a correctness failure.
4. Actor forward/backward goes through the DDP-wrapped policy. Custom Molmo
   attributes and diagnostic stashes are read through the unwrapped policy.
5. The first `gradient_accumulation_steps - 1` microbatches do not all-reduce;
   only the last microbatch synchronizes gradients.
6. All ranks take the same number and order of DDP forward/backward operations.
7. Only the main process creates Aim, writes the shared log, saves checkpoints,
   runs `ValLoss`, and runs probes.
8. Non-main ranks wait on both sides of main-only checkpoint/validation work and
   never enter the next DDP update early.
9. Periodic pretrained merges happen on every rank from identical snapshots and
   clear the underlying raw optimizer state on every rank.
10. Multi-process execution fails early with an actionable error for a policy or
    training path that has not been made DDP-safe.

## Design

### 1. Small generic runtime seam

Add `src/lerobot/rl/training_runtime.py` with a small wrapper around an
`Accelerator`. It owns the mechanics that every future Trainer will need:

- `device`, `is_main_process`, `num_processes`, and `process_index`;
- `unwrap_model(policy)`;
- `backward(loss)`;
- `clip_grad_norm_(parameters, max_norm)`;
- a context manager that calls `accelerator.no_sync(policy)` on non-final
  accumulation microbatches and is a no-op in one process;
- `wait_for_everyone()`;
- a collective `any_process(bool)` for coordinated shutdown;
- reduction of scalar training metrics by mean;
- gathering equal-shaped tensor/NumPy histogram values across ranks.

The wrapper should not know anything about MolmoAct2, replay buffers, probes, or
Aim. An optional/single-process implementation keeps direct Trainer unit calls
working without constructing a distributed process group.

Pass this runtime through the existing generic `Trainer.update_actor(...,
**kwargs)` seam. Do not import Accelerate from the Molmo trainer.

### 2. Offline process lifecycle

Refactor `src/lerobot/scripts/rl_offline.py` using the established pattern in
`src/lerobot/scripts/lerobot_train.py`:

- require and construct `Accelerator` before logging/Aim initialization;
- use `DistributedDataParallelKwargs(find_unused_parameters=True)` because
  Molmo has conditional trainable paths;
- keep Accelerate mixed precision disabled initially (`mixed_precision="no"`):
  the current policy/config already owns BF16 casting, and numerical behavior
  should not change in the DDP patch;
- call `init_logging(..., accelerator=accelerator)` so only main logs to the
  console and shared file;
- instantiate Aim only on the main process;
- use `accelerator.device` instead of `learner_device` to construct the local
  policy and replay buffers on the rank's GPU;
- pass the real `is_main_process` value to dataset/stat/processor helpers instead
  of the current hard-coded `True` values;
- synchronize around any main-first filesystem/cache initialization;
- prepare the policy and every named optimizer, preserving optimizer dictionary
  keys and the raw optimizer objects needed by pretrained merging;
- reset RNG after model preparation to `seed + process_index` before constructing
  the replay iterators. This gives ranks distinct replay draws and dropout while
  DDP has already synchronized initial model weights;
- log rank count, per-rank batch, accumulation, and effective global batch.

Keep direct `python -m lerobot.scripts.rl_offline ...` valid. It constructs a
single-process Accelerator automatically; `accelerate launch` is only needed to
start multiple processes or explicitly select launch settings.

### 3. DDP-safe Molmo actor update

Modify only `MolmoAct2Trainer.update_actor` for the first implementation:

- accept the generic training runtime via `kwargs`;
- derive `raw_policy = runtime.unwrap_model(policy)`;
- call the wrapped policy as `policy(...)` for the training forward;
- use `raw_policy` for `config`, `_backbone()`, depth diagnostic stashes,
  `critic` inspection, named parameters, and all other custom attributes;
- wrap each non-final accumulation microbatch in the runtime's `no_sync`
  context;
- replace direct `.backward()` with `runtime.backward()`;
- replace `torch.nn.utils.clip_grad_norm_` with
  `runtime.clip_grad_norm_` for the actor parameter set;
- continue stepping every prepared named optimizer exactly once as today;
- preserve all existing scalar and histogram calculations.

Do not use `Accelerator.accumulate()` here. The Trainer owns an inner loop with
multiple forward/backward pairs per outer optimization step; explicit `no_sync`
preserves the existing step semantics and avoids an all-reduce on every
microbatch.

The optional subtask-generation loss directly forwards through internal model
modules and may select a different number of annotated samples on each rank.
Because the current configuration sets `subtask_loss_weight: 0.0`, add a
multi-process fail-fast guard for a positive weight rather than pretending that
path is safe. Future support must route that loss through a DDP-visible forward
whose collective sequence is identical on every rank.

### 4. Named optimizers and pretrained merges

`build_named_adamw_optimizers` may return both `policy` and `depth` optimizers.
Prepare them as an ordered list and reconstruct the dictionary with the same
keys.

Keep a second dictionary containing the original raw optimizers. Accelerate's
prepared optimizer wrappers are used for training steps; raw optimizers are used
to build/apply `PretrainedMerge` and clear actual AdamW state. The live parameter
objects are shared, so the merge changes the trained replicas correctly.

At a merge step:

1. barrier;
2. main process saves the pre-merge unwrapped model;
3. barrier;
4. every rank applies the same merge and clears its raw optimizer state;
5. barrier;
6. main process saves the post-merge unwrapped model;
7. barrier.

### 5. Metrics and logging

Every rank produces local `training_infos`. On log steps, all ranks must enter a
generic reducer before any main-only validation:

- numeric scalars and scalar tensors: mean across ranks;
- NumPy arrays/non-scalar tensors: gather along dimension zero;
- nonnumeric metadata: keep main-process value only;
- `Optimization step`: identical on all ranks; preserve it as an integer rather
  than averaging it to a float.

Only main passes the reduced result to `trainer.log_metrics` and Aim. Step-Hz
should measure synchronized wall-clock time (barrier around the timed training
step or maximum elapsed time across ranks), not rank 0's unsynchronized local
time.

### 6. Checkpoints

For ordinary and pretrained-merge checkpoints:

- all ranks enter a barrier before saving;
- main calls the existing save helper with `runtime.unwrap_model(policy)`;
- only main updates the `last` symlink and writes `training_state.pt`;
- all ranks enter a barrier after saving.

The current offline checkpoint intentionally saves no optimizer/scheduler state;
do not broaden this patch into resume support.

### 7. Main-GPU-only `ValLoss` and probes

Only main constructs `ValLoss`; otherwise its documented ~2.4 GB CPU sample
cache is duplicated per process. At every log step:

1. reduce normal training metrics on all ranks;
2. barrier;
3. main evaluates `ValLoss` on the unwrapped policy and merges its metrics;
4. barrier;
5. main logs.

For `val_on_start` and periodic probes:

1. barrier;
2. main calls `_run_validation_probes` with the unwrapped policy;
3. main clears any probe-created gradients and restores training mode;
4. barrier;
5. every rank resumes the same optimization step.

No probe code needs to know about Accelerate. Non-main GPUs intentionally idle
during probes in v1. Probe output paths and formats remain unchanged.

### 8. Coordinated shutdown

The current loop can break independently when its local signal event is set,
which would strand the other rank in a DDP collective. At the top of every outer
step, use the runtime's `any_process(shutdown_event.is_set())`; if any rank asks
to stop, all ranks leave the loop together after a final barrier.

## Fail-fast support boundary

When `num_processes > 1`, validate before loading the large model:

- require `cfg.policy.type == "molmoact2_rl"`;
- require `cfg.skip_critic is True`;
- require `cfg.policy.subtask_loss_weight == 0.0`;
- reject a CUDA storage device for the shared replay cache in the first version;
- report the computed effective batch and warn if it differs from the current
  single-GPU baseline of 128 (warning, not error).

Single-process behavior for other existing policies should not be intentionally
broken, but it is outside this plan's validation matrix.

## Implementation checklist

### Phase A — generic runtime and unit tests

- [ ] Add `src/lerobot/rl/training_runtime.py`.
- [ ] Implement unwrap/backward/clip/no-sync/barrier/process properties.
- [ ] Implement coordinated boolean reduction.
- [ ] Implement scalar reduction and histogram gathering.
- [ ] Unit-test single-process no-op behavior.
- [ ] Unit-test metric type/shape preservation with a fake runtime.

### Phase B — offline loop plumbing

- [ ] Construct Accelerator before logging and Aim.
- [ ] Add multi-process support guards.
- [ ] Replace fixed learner device with rank-local Accelerator device.
- [ ] Propagate `is_main_process` to setup helpers.
- [ ] Prepare the policy and all named optimizers.
- [ ] Preserve raw optimizer mapping for pretrained merges.
- [ ] Apply rank-specific RNG reset before replay iterator creation.
- [ ] Log world size and effective batch.
- [ ] Coordinate shutdown across ranks.

### Phase C — Molmo actor update

- [ ] Separate wrapped-forward policy from unwrapped attribute policy.
- [ ] Route training forward through DDP.
- [ ] Add explicit no-sync accumulation.
- [ ] Route backward and clipping through the runtime.
- [ ] Preserve depth/modality diagnostics from the raw policy.
- [ ] Add the positive-subtask-loss multi-GPU guard.

### Phase D — side effects and validation

- [ ] Reduce/gather training metrics on all ranks.
- [ ] Make Aim and shared logging main-only.
- [ ] Make ordinary checkpoints main-only with barriers and unwrapping.
- [ ] Make pretrained-merge checkpoints main-only; merge on every rank.
- [ ] Construct and execute `ValLoss` only on main.
- [ ] Execute every validation probe only on main with barriers.
- [ ] Clear probe gradients before resuming training.

### Phase E — verification

- [ ] Run focused unit tests for runtime and metric reduction.
- [ ] Run existing replay, Molmo loss-logging, and probe tests.
- [ ] Run formatting/lint/type checks on changed files.
- [ ] One-GPU smoke: two optimizer steps, no probes, Aim disabled.
- [ ] One-GPU probe smoke: `val_on_start`, one cheap enabled probe.
- [ ] Two-GPU smoke: two optimizer steps with accumulation 2.
- [ ] Verify ranks sample distinct replay indices or deterministic batch hashes.
- [ ] Verify a post-step trainable-parameter checksum agrees across ranks.
- [ ] Verify only one log, Aim run, checkpoint tree, and probe output tree exist.
- [ ] Verify a checkpoint reloads for ordinary single-GPU inference/probing.
- [ ] Interrupt a two-GPU smoke and verify both ranks exit without hanging.

## Test strategy

Use the existing subprocess/config-file pattern in
`tests/training/test_multi_gpu.py`, but keep heavy Molmo integration out of the
normal CPU suite.

Add focused tests under `tests/rl/` for:

- named optimizer preparation/reconstruction;
- no-sync selection for accumulation indices;
- training-info scalar reduction and histogram concatenation;
- main-only callable execution with pre/post barriers;
- raw-vs-wrapped policy selection in the Molmo actor update;
- fail-fast unsupported distributed configurations.

The real two-GPU smoke uses the current local caches/model and aggressive
overrides: two offline steps, Aim off, checkpoints off, `val_freq: 0`,
`val_on_start: false`, and `val_loss_frames: 0`. It is a manual/cloud validation,
not a lightweight CI test.

## Launch recipes

One physical GPU, selecting GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 \
  -m lerobot.scripts.rl_offline \
  --config_path=src/lerobot/rl/config_rl.yaml \
  --policy.gradient_accumulation_steps=4
```

Two GPUs, preserving effective batch 128:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  -m lerobot.scripts.rl_offline \
  --config_path=src/lerobot/rl/config_rl.yaml \
  --policy.gradient_accumulation_steps=2
```

Four GPUs, preserving effective batch 128:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  -m lerobot.scripts.rl_offline \
  --config_path=src/lerobot/rl/config_rl.yaml \
  --policy.gradient_accumulation_steps=1
```

With `CUDA_VISIBLE_DEVICES=1`, the selected physical GPU is exposed inside the
job as logical `cuda:0`. Keep `learner_device` at `cuda:0` or remove its authority
from offline placement; never set it to `cuda:1` in that launch.

## Expected performance and limitations

- DDP improves throughput, not model capacity. Each GPU must fit the full Molmo
  policy, optimizer, and its local batch.
- With two GPUs and accumulation reduced from four to two, ideal compute work per
  rank halves. Gradient all-reduce over the large trainable portion prevents a
  perfect 2x speedup.
- Explicit no-sync is important: synchronizing both accumulation microbatches
  would spend communication twice per optimizer step.
- Probes remain single-GPU and have historically dominated some validation
  intervals. That is accepted for v1; later work may shard independent probes or
  run them as checkpoint job arrays without changing training correctness.
- CPU memmaps benefit from the OS page cache, but non-image replay metadata and
  dataset objects are still replicated per process. Cloud host RAM must be sized
  accordingly.

## Definition of done

This phase is complete when the same codebase:

1. runs the current Molmo actor-only config on physical GPU 1 as a single process;
2. runs two synchronized training processes on GPUs 0 and 1;
3. preserves effective batch 128 with accumulation 2 on two GPUs;
4. demonstrates distinct replay data on the two ranks;
5. produces synchronized trainable weights after an optimizer step;
6. produces exactly one checkpoint/log/Aim/probe output set;
7. runs all probes on GPU 0 while GPU 1 waits and resumes without a DDP hang;
8. exits both processes cleanly on an interrupt;
9. leaves the single-GPU checkpoint format and downstream inference unchanged.

## Deferred follow-up

- DDP-safe Molmo critic updates and target-network handling;
- DDP-safe subtask-generation auxiliary loss;
- PI05 and future policy Trainer implementations;
- distributing `ValLoss` or independent probes across ranks;
- resumable optimizer/RNG state for offline training;
- FSDP/DeepSpeed if aggregate GPU memory, rather than throughput, is required;
- multi-node storage, rendezvous, and failure handling.

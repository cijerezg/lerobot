# Runbook: Linux Overcommit and Probe MP4 Failures

## Symptom

Attention-style probes can fail while writing MP4 artifacts with:

```text
[Errno 12] Cannot allocate memory
```

In the MolmoAct2 offline validation path, this showed up in:

- `lerobot.probes.attention`
- `lerobot.probes.action_drift_jacobian`

The model-side attention capture and rendering can already be complete. The failure may occur at the first or early calls to:

```python
writers[layer_idx][key].append_data(frame_np)
```

where `writers[layer_idx][key]` is an `imageio` ffmpeg writer.

## Root Cause

This can be Linux virtual-memory overcommit accounting, not a genuinely huge rendered frame.

PyTorch and CUDA may reserve very large virtual address ranges. These reservations inflate `Committed_AS` in `/proc/meminfo`, even when the pages are not physically resident. With:

```text
vm.overcommit_memory = 0
```

Linux uses heuristic overcommit checks. If the system is already far over `CommitLimit`, the kernel may reject allocations or ffmpeg writer startup with `ENOMEM`.

A representative bad state was:

```text
CommitLimit:    73,104,180 kB   (~73 GB)
Committed_AS:  212,584,596 kB   (~212 GB)
```

That is roughly `Committed_AS / CommitLimit = 2.9`.

In this state, small-looking video frames can still fail to write because imageio/ffmpeg needs subprocess, pipe, codec, and buffer allocation work inside a very large ML training process.

## Why PI05 Could Work While MolmoAct2 Failed

The PI05 probes used the same broad MP4 path: `imageio.get_writer(...).append_data(...)`.

The difference was the memory accounting situation. MolmoAct2 keeps a larger policy stack and associated PyTorch/CUDA reservations in process. The process can sit near the heuristic overcommit cliff even though the rendered frame itself is only a few MB.

So this is not primarily an image-size bug. It is an interaction between:

- large ML process virtual memory reservations,
- Linux heuristic overcommit mode,
- imageio/ffmpeg video writer initialization.

## Diagnosis

Check the overcommit mode:

```bash
cat /proc/sys/vm/overcommit_memory
```

Mode meanings:

```text
0 = heuristic overcommit
1 = always overcommit
2 = strict overcommit
```

Check commit accounting:

```bash
grep -E 'CommitLimit|Committed_AS' /proc/meminfo
```

If `vm.overcommit_memory` is `0` and `Committed_AS` is greater than `CommitLimit`, imageio/ffmpeg MP4 writers are at risk of failing with `Errno 12` from inside large training processes.

You can also inspect limits:

```bash
ulimit -a
```

If `virtual memory (-v)` is `unlimited`, this is not an `RLIMIT_AS` problem.

## Immediate Fix

Apply always-overcommit mode for the current boot:

```bash
sudo sysctl -w vm.overcommit_memory=1
```

Verify:

```bash
cat /proc/sys/vm/overcommit_memory
```

Expected output:

```text
1
```

Then restart the training/probe run.

## Persistent Fix

Make the setting survive reboot:

```bash
echo 'vm.overcommit_memory = 1' | sudo tee /etc/sysctl.d/99-overcommit.conf
sudo sysctl --system
```

Verify again:

```bash
cat /proc/sys/vm/overcommit_memory
```

## Tradeoff

With `vm.overcommit_memory=1`, the kernel stops rejecting virtual-memory commits up front. If processes actually touch more physical memory than the machine can satisfy, the OOM killer may terminate a process instead of an allocation failing cleanly.

For ML workstations running large PyTorch/CUDA jobs, mode `1` is often the practical setting because large virtual reservations are common and not necessarily physically backed.

## Repo-Side Guardrail

The attention probe has a warning for this condition. If it sees:

- `vm.overcommit_memory == 0`, and
- `Committed_AS / CommitLimit > 1`,

it warns that imageio/ffmpeg writers may fail with `Errno 12` and suggests:

```bash
sudo sysctl -w vm.overcommit_memory=1
```

## Related Files

- `lerobot/src/lerobot/probes/attention.py`
- `lerobot/src/lerobot/probes/action_drift_jacobian.py`
- `lerobot/src/lerobot/scripts/rl_offline.py`

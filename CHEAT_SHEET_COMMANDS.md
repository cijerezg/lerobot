# LeRobot uv Command Cheat Sheet

Run these from the workspace root:

```bash
cd /home/user/Documents/Research/RL/LeRobot
```

Use `uv run --project lerobot ...` because the Python project lives in
`lerobot/pyproject.toml`, while the active config and `outputs/` paths are rooted
at the workspace root.

For the RL/robot commands, use this prefix so `uv` includes the optional
dependencies these scripts usually need:

```bash
UV_RUN="uv run --project lerobot --extra training --extra hardware --extra feetech --extra molmoact2 --extra pi --extra async --extra viz"
```

If your environment is already synced with all needed extras, the shorter
`uv run --project lerobot ...` form also works.

## Shared Config

Most commands use the same config:

```bash
CONFIG=config_rl.yaml
```

The in-package copy is also useful when you specifically want the config tracked
next to the RL code:

```bash
CONFIG=lerobot/src/lerobot/rl/config_rl.yaml
```

The important knobs live in the config:

- `policy.type`: switch between `molmoact2_rl` and `pi05_rl`.
- `skip_critic`: `true` for actor-only / BC-style training, `false` for critic training.
- `offline_output_dir`, `job_name`: where checkpoints and logs go.
- `dataset.root`, `val_dataset_path`: training and validation datasets.
- `policy.pretrained_path` / `policy.checkpoint_path`: starting weights.
- `policy.offline_steps`, `batch_size`, `offline_save_freq`, `val_freq`: run size and cadence.

## Benchmark Inference

Proper command for `benchmark_inference.py`:

```bash
uv run --project lerobot python -m lerobot.scripts.benchmark_inference \
  --config_path="$CONFIG"
```

With the full extras prefix:

```bash
$UV_RUN python -m lerobot.scripts.benchmark_inference \
  --config_path="$CONFIG"
```

Useful benchmark-only options:

```bash
$UV_RUN python -m lerobot.scripts.benchmark_inference \
  --config_path="$CONFIG" \
  --benchmark-warmup=10 \
  --benchmark-measure=50 \
  --benchmark-inference-delay=0
```

Add camera/OpenCV load while benchmarking:

```bash
$UV_RUN python -m lerobot.scripts.benchmark_inference \
  --config_path="$CONFIG" \
  --benchmark-camera-load=true
```

The benchmark sweeps `policy.num_inference_steps` over `[1, 2, 3, 5]` and reports
latency against the RTC timing margin from `env.fps` and `policy.chunk_size`.

## Offline Training

Proper command for `rl_offline.py`:

```bash
uv run --project lerobot python -m lerobot.scripts.rl_offline \
  --config_path="$CONFIG"
```

With the full extras prefix:

```bash
$UV_RUN python -m lerobot.scripts.rl_offline \
  --config_path="$CONFIG"
```

Quick smoke test with fewer steps and no periodic validation:

```bash
$UV_RUN python -m lerobot.scripts.rl_offline \
  --config_path="$CONFIG" \
  --policy.offline_steps=20 \
  --val_freq=0 \
  --save_checkpoint=false \
  --wandb.enable=false
```

Actor-only / BC-style offline run:

```bash
$UV_RUN python -m lerobot.scripts.rl_offline \
  --config_path="$CONFIG" \
  --skip_critic=true
```

Full critic-training offline run:

```bash
$UV_RUN python -m lerobot.scripts.rl_offline \
  --config_path="$CONFIG" \
  --skip_critic=false
```

Override output identity without editing the config:

```bash
$UV_RUN python -m lerobot.scripts.rl_offline \
  --config_path="$CONFIG" \
  --offline_output_dir=outputs/my_offline_run \
  --job_name=my_offline_run
```

Logs land under:

```text
<offline_output_dir>/logs/rl_offline_<job_name>.log
```

Checkpoints land under:

```text
<offline_output_dir>/checkpoints/
```

## Standalone Robot Inference

Run the configured policy on the real robot without the online learner/actor
transport:

```bash
$UV_RUN python -m lerobot.rl.inference_async \
  --config_path="$CONFIG"
```

This uses the RTC inference runtime. Make sure the config points at the checkpoint
you want through `policy.pretrained_path` or `policy.checkpoint_path`, and that
`env.robot.port`, `env.teleop.port`, and camera indices match the machine.

## Online RL: Learner and Actor

Start the learner:

```bash
$UV_RUN python -m lerobot.rl.rl_learner \
  --config_path="$CONFIG"
```

Start the distributed actor on the robot machine:

```bash
$UV_RUN python -m lerobot.rl.rl_actor_async \
  --config_path="$CONFIG"
```

Before running actor/learner online, fill in `policy.actor_learner_config` in the
config so the actor can reach the learner host and port.

## Handy uv / Repo Commands

Run tests from the package project:

```bash
uv run --project lerobot pytest lerobot/tests -svv --maxfail=10
```

Run a single test file:

```bash
uv run --project lerobot pytest lerobot/tests/test_buffer_cache.py -v
```

Find cameras:

```bash
uv run --project lerobot lerobot-find-cameras
```

Find robot serial ports:

```bash
uv run --project lerobot lerobot-find-port
```

## Notes

- `rl_offline.py` and `benchmark_inference.py` are in `lerobot.scripts`, so the
  module form is `python -m lerobot.scripts.<name>`.
- `inference_async.py`, `rl_learner.py`, and `rl_actor_async.py` are in
  `lerobot.rl`, so their module form is `python -m lerobot.rl.<name>`.
- Keep the working directory at the workspace root unless you intentionally want
  config-relative paths to resolve somewhere else.

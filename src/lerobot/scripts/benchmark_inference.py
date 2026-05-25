#!/usr/bin/env python
"""
Benchmark MolmoAct2 raw policy inference latency.

This measures repeated calls to ``policy.predict_action_chunk`` on one
preprocessed dummy batch. It intentionally focuses on model-call latency rather
than full RTC runtime latency.

Usage:
  cd /home/user/Documents/Research/RL/LeRobot
  python lerobot/src/lerobot/scripts/benchmark_inference.py --config_path=config_rl.yaml

Useful benchmark-only options:
  --benchmark-warmup=10
  --benchmark-measure=50
  --benchmark-inference-delay=0
  --benchmark-camera-load=true
"""

from dataclasses import dataclass
import logging
import re
import sys
import threading
import time

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 - registers MolmoAct2RLConfig
import lerobot.rl.gym_manipulator  # noqa: F401 - registers robots, cameras, teleops

logger = logging.getLogger(__name__)

DEFAULT_N_WARMUP = 10
DEFAULT_N_MEASURE = 50
DEFAULT_INFERENCE_DELAY = 0

NUM_STEPS_LIST = [1, 2, 3, 5]


@dataclass(frozen=True)
class BenchmarkArgs:
    warmup: int = DEFAULT_N_WARMUP
    measure: int = DEFAULT_N_MEASURE
    inference_delay: int = DEFAULT_INFERENCE_DELAY
    camera_load: bool = False


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _pop_benchmark_args(argv: list[str]) -> BenchmarkArgs:
    """Parse benchmark-only CLI args and remove them before draccus sees argv."""
    values = {
        "warmup": DEFAULT_N_WARMUP,
        "measure": DEFAULT_N_MEASURE,
        "inference_delay": DEFAULT_INFERENCE_DELAY,
        "camera_load": False,
    }
    aliases = {
        "--benchmark-warmup": "warmup",
        "--benchmark.warmup": "warmup",
        "--benchmark-measure": "measure",
        "--benchmark.measure": "measure",
        "--benchmark-inference-delay": "inference_delay",
        "--benchmark.inference_delay": "inference_delay",
        "--benchmark-camera-load": "camera_load",
        "--benchmark.camera_load": "camera_load",
    }

    kept = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        key = None
        raw_value = None
        for prefix, name in aliases.items():
            if arg == prefix:
                key = name
                if i + 1 >= len(argv):
                    raise ValueError(f"Missing value for {prefix}.")
                raw_value = argv[i + 1]
                i += 2
                break
            if arg.startswith(f"{prefix}="):
                key = name
                raw_value = arg.split("=", 1)[1]
                i += 1
                break
        if key is None:
            kept.append(arg)
            i += 1
            continue
        if key == "camera_load":
            values[key] = _parse_bool(raw_value or "")
        else:
            values[key] = int(raw_value or "0")

    if values["warmup"] < 0:
        raise ValueError("--benchmark-warmup must be >= 0.")
    if values["measure"] <= 0:
        raise ValueError("--benchmark-measure must be > 0.")
    if values["inference_delay"] < 0:
        raise ValueError("--benchmark-inference-delay must be >= 0.")

    sys.argv = kept
    return BenchmarkArgs(**values)


BENCHMARK_ARGS = _pop_benchmark_args(sys.argv)


def _latency_label(p95_s: float, fps: int, chunk_size: int) -> str:
    delay_steps = int(np.ceil(p95_s * fps))
    steady_margin = chunk_size - 2 * delay_steps
    if steady_margin > 4:
        return f"OK      delay={delay_steps:2d} steady_margin=+{steady_margin:2d}"
    if steady_margin >= 0:
        return f"tight   delay={delay_steps:2d} steady_margin=+{steady_margin:2d}"
    return f"risk    delay={delay_steps:2d} steady_margin={steady_margin:+d}"


def _get_camera_indices(cfg) -> list[int]:
    """Extract integer camera indices from the env camera config."""
    cameras = getattr(cfg.env, "cameras", None) or {}
    indices = []
    for cam_cfg in cameras.values():
        val = getattr(cam_cfg, "index_or_path", None)
        if isinstance(val, int):
            indices.append(val)
    return sorted(indices) if indices else [0, 2]


def _camera_reader_worker(indices: list[int], stop_event: threading.Event, fps: int) -> None:
    """Read camera frames at env fps to add optional camera/OpenCV load."""
    import cv2

    caps = []
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            caps.append((idx, cap))
        else:
            print(f"  [camera thread] WARNING: could not open camera {idx}")

    if not caps:
        print("  [camera thread] No cameras opened - thread exits without load")
        return

    opened_ids = [i for i, _ in caps]
    print(f"  [camera thread] Reading cameras {opened_ids} at {fps} Hz")

    interval = 1.0 / fps
    while not stop_event.is_set():
        t0 = time.perf_counter()
        for _, cap in caps:
            cap.read()
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, interval - elapsed))

    for _, cap in caps:
        cap.release()
    print(f"  [camera thread] Released cameras {opened_ids}")


_TIMING_RE = re.compile(
    r"\[TIMING\] backbone=([\d.]+)ms\s+kv_setup=([\d.]+)ms\s+init=([\d.]+)ms"
    r"\s+steps=\[(.*?)\]\s+total=([\d.]+)ms"
)


class TimingAggregator(logging.Handler):
    """Captures [TIMING] INFO lines from modeling_molmoact2."""

    def __init__(self):
        super().__init__()
        self._rows: list[dict] = []

    def reset(self):
        self._rows.clear()

    def emit(self, record):
        m = _TIMING_RE.search(record.getMessage())
        if not m:
            return
        step_strs = [s.strip().strip("'\"") for s in m.group(4).split(",") if s.strip()]
        self._rows.append({
            "backbone": float(m.group(1)),
            "kv_setup": float(m.group(2)),
            "init": float(m.group(3)),
            "steps": [float(s) for s in step_strs],
            "total": float(m.group(5)),
        })

    def summary(self, skip: int) -> dict | None:
        rows = self._rows[skip:]
        if not rows:
            return None
        bb = np.mean([r["backbone"] for r in rows])
        kv = np.mean([r["kv_setup"] for r in rows])
        ini = np.mean([r["init"] for r in rows])
        total = np.mean([r["total"] for r in rows])
        n_steps = len(rows[0]["steps"])
        step_means = [
            np.mean([r["steps"][i] for r in rows if i < len(r["steps"])])
            for i in range(n_steps)
        ]
        return {"backbone": bb, "kv_setup": kv, "init": ini, "steps": step_means, "total": total}


def _run_sweep(
    policy,
    batch,
    prev_actions,
    device,
    fps,
    chunk_size,
    execution_horizon,
    benchmark_args: BenchmarkArgs,
    label,
    timing_agg: TimingAggregator | None = None,
):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    hdr = f"{'steps':>5}  {'delay':>5}  {'mean':>7}  {'p50':>7}  {'p95':>7}  {'max':>7}  status"
    print(hdr)
    print("-" * len(hdr))

    for num_steps in NUM_STEPS_LIST:
        policy.config.num_inference_steps = num_steps

        def _run():
            with torch.no_grad():
                policy.predict_action_chunk(
                    batch,
                    inference_delay=benchmark_args.inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=execution_horizon,
                )

        if timing_agg is not None:
            timing_agg.reset()

        for _ in range(benchmark_args.warmup):
            _run()
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(benchmark_args.measure):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _run()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        arr = np.array(latencies)
        p95 = float(np.percentile(arr, 95))
        print(
            f"{num_steps:>5}  {benchmark_args.inference_delay:>5}  "
            f"{arr.mean():.3f}s  {np.percentile(arr, 50):.3f}s  "
            f"{p95:.3f}s  {arr.max():.3f}s  "
            f"{_latency_label(p95, fps, chunk_size)}"
        )
        if timing_agg is not None:
            s = timing_agg.summary(skip=benchmark_args.warmup)
            if s:
                steps_str = "  ".join(f"{t:.0f}" for t in s["steps"])
                print(
                    f"{'':>5}  {'':>5}  "
                    f"  bb={s['backbone']:.0f}ms  kv={s['kv_setup']:.0f}ms"
                    f"  init={s['init']:.0f}ms  steps=[{steps_str}]ms"
                    f"  instrumented_total={s['total']:.0f}ms"
                )


def _feature_type_name(feature) -> str:
    value = getattr(feature, "type", None)
    return getattr(value, "name", str(value)).upper()


def _build_dummy_observation(cfg, action_dim: int) -> dict[str, torch.Tensor]:
    obs = {}
    for key, feature in cfg.policy.input_features.items():
        shape = tuple(getattr(feature, "shape", ()))
        feature_type = _feature_type_name(feature)
        key_upper = key.upper()
        if "VISUAL" in feature_type or "IMAGE" in key_upper:
            obs[key] = torch.randint(0, 256, shape, dtype=torch.uint8)
        elif "STATE" in feature_type or key == "observation.state":
            obs[key] = torch.zeros(shape or (action_dim,), dtype=torch.float32)
        else:
            obs[key] = torch.zeros(shape, dtype=torch.float32)
    return obs


def _action_dim(cfg) -> int:
    action_feature = getattr(cfg.policy, "output_features", {}).get("action")
    if action_feature is not None and getattr(action_feature, "shape", None):
        return int(action_feature.shape[0])
    return int(next(iter(cfg.policy.output_features.values())).shape[0])


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    for _noisy in ("transformers", "lerobot", "torch", "PIL"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    set_seed(cfg.seed)
    device = get_safe_torch_device(
        getattr(cfg.policy, "actor_device", None) or cfg.policy.device
    )
    cfg.policy.device = str(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Capture [TIMING] lines emitted by _generate_actions_from_inputs_with_rtc.
    # propagate=False keeps the raw INFO lines off the console; only timing_agg sees them.
    timing_agg = TimingAggregator()
    model_logger = logging.getLogger("lerobot.policies.molmoact2.modeling_molmoact2")
    model_logger.setLevel(logging.INFO)
    model_logger.propagate = False
    model_logger.addHandler(timing_agg)

    fps = cfg.env.fps
    chunk_size = cfg.policy.chunk_size
    action_dim = _action_dim(cfg)
    rtc_config = getattr(cfg.policy, "rtc_config", None)
    execution_horizon = int(getattr(rtc_config, "execution_horizon", chunk_size) or chunk_size)

    print(f"\n{'='*70}")
    print("  MolmoAct2 raw policy inference benchmark")
    print(f"  device={device}  fps={fps}  chunk_size={chunk_size}  action_dim={action_dim}")
    print(f"  warmup={BENCHMARK_ARGS.warmup}  measure={BENCHMARK_ARGS.measure}  inference_delay={BENCHMARK_ARGS.inference_delay}")
    print(f"  execution_horizon={execution_horizon}  camera_load={BENCHMARK_ARGS.camera_load}")
    print("  RTC status uses p95 with steady_margin = chunk_size - 2 * ceil(p95 * fps)")
    print(f"{'='*70}")

    print("Loading policy...")
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer

    trainer = MolmoAct2Trainer()
    policy = trainer.make_policy(cfg).to(device).eval()
    preprocessor, _ = trainer.make_processors(cfg)
    print("Done.\n")

    dummy_obs = _build_dummy_observation(cfg, action_dim)
    batch = trainer.build_inference_batch(
        dummy_obs, cfg.policy.task, cfg, preprocessor=preprocessor
    )
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    prev_actions = torch.randn(chunk_size // 2, action_dim, device=device, dtype=torch.float32)

    _run_sweep(
        policy,
        batch,
        prev_actions,
        device,
        fps,
        chunk_size,
        execution_horizon,
        BENCHMARK_ARGS,
        f"Baseline - execution_horizon={execution_horizon}",
        timing_agg,
    )

    if not BENCHMARK_ARGS.camera_load:
        return

    camera_indices = _get_camera_indices(cfg)
    print(f"Starting camera-reader thread  indices={camera_indices}  fps={fps} Hz")
    stop_event = threading.Event()
    cam_thread = threading.Thread(
        target=_camera_reader_worker,
        args=(camera_indices, stop_event, fps),
        daemon=True,
    )
    cam_thread.start()
    time.sleep(1.0)

    _run_sweep(
        policy,
        batch,
        prev_actions,
        device,
        fps,
        chunk_size,
        execution_horizon,
        BENCHMARK_ARGS,
        "With camera-reader thread (camera/OpenCV/scheduler load)",
        timing_agg,
    )

    stop_event.set()
    cam_thread.join(timeout=3.0)

    print(f"\n{'='*70}")
    print("  Camera-load deltas indicate contention/noise, not proof of GIL contention.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
RTC inference with live attention visualization (rerun).

Same inference loop as ``inference_async``, plus a throttled in-thread capture of
the action expert's cross-attention, streamed to rerun as per-layer heatmaps
(mean over heads). First iteration: 1-2 layers, raw heatmaps only.

Why in-thread (not a separate worker): MolmoAct2's attention capture flips a
process-global flag and patches the action-expert modules the live forward also
uses, so capture must not overlap inference. The capture therefore runs as a
``post_inference_hook`` inside the inference thread, self-throttled to
``ATTENTION_RATE_HZ`` so it only borrows the GPU a couple of times per second.

Usage (same as inference_async; requires use_rerun=true):
    python -m lerobot.rl.inference_w_attn \
        --config_path lerobot/src/lerobot/rl/config_rl.yaml
"""
import logging
import os
import signal
import time

import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.rl.rl_trainer import Trainer
from lerobot.rl.rtc_actor_runtime import act_with_policy_rtc_inference
import lerobot.rl.gym_manipulator  # noqa: F401 - registers robot/camera/teleop config choices
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 - registers MolmoAct2RLConfig
import lerobot.rl.pi05.rl_pi05  # noqa: F401 - registers PI05RLConfig

logger = logging.getLogger(__name__)

# ── Knobs (edit these) ───────────────────────────────────────────────────────
ATTENTION_LAYERS = [9]      # action-expert layers to capture (1-2 recommended)
ATTENTION_RATE_HZ = 2.5     # captures/sec; throttles GPU contention with inference


def _ensure_batch(obs: dict) -> dict:
    """Match the per-frame batch layout the probe adapters expect (batch dim 1)."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            if "image" in k and v.ndim == 3:
                v = v.unsqueeze(0)
            elif "state" in k and v.ndim == 1:
                v = v.unsqueeze(0)
        out[k] = v
    return out


def _make_attention_hook(policy, preprocessor, postprocessor, device, cfg):
    """Build the post-inference capture/render/log callable, or None if unsupported."""
    if not getattr(cfg, "use_rerun", False):
        logger.warning("[ATTN] use_rerun is false; attention viz disabled.")
        return None

    try:
        import rerun as rr
        from lerobot.probes.base import _adapter_for_type
        from lerobot.probes.spatial_memorization_attention import (
            _image_hw_for_segment,
            _render_heatmap,
            _segment_attention_vector,
            _upsample_patches,
        )

        adapter_cls = _adapter_for_type(getattr(cfg.policy, "type", None))
        adapter = adapter_cls(policy, preprocessor, postprocessor, device, cfg)
    except Exception as exc:
        logger.warning("[ATTN] Could not set up attention adapter (%s); viz disabled.", exc)
        return None

    task_str = cfg.policy.task
    layers = list(ATTENTION_LAYERS)
    min_interval = 1.0 / max(ATTENTION_RATE_HZ, 1e-6)
    state = {"last": 0.0}
    logger.info("[ATTN] Live attention viz on: layers=%s rate=%.1fHz", layers, ATTENTION_RATE_HZ)

    @torch.no_grad()
    def hook(latest_obs: dict) -> None:
        now = time.perf_counter()
        if now - state["last"] < min_interval or latest_obs is None:
            return
        state["last"] = now

        obs = _ensure_batch(latest_obs)
        result = adapter.capture_attention(obs, task_str, layers=layers)
        if not result.cross_attn_by_layer:
            return

        # Camera segments: prefer the adapter's overlay list (molmo pooled crops),
        # else fall back to contiguous "img*" segments (mirrors collect_aggregates).
        segment_lookup = {name: (s, e) for name, s, e in result.encoder_segments}
        overlay_segments = result.extras.get("image_overlay_segments")
        if overlay_segments:
            cam_segs = [(str(n), *segment_lookup.get(str(n), (None, None))) for n in overlay_segments]
        else:
            cam_segs = [(n, s, e) for n, s, e in result.encoder_segments if n.startswith("img")]

        for layer_idx, cross in result.cross_attn_by_layer.items():
            attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)  # [H, n_action, K]
            for cam_idx, (cam_name, cs, ce) in enumerate(cam_segs):
                try:
                    vec, n_p = _segment_attention_vector(attn, result, cam_name, cs, ce)
                except ValueError:
                    continue
                mean_vec = vec.mean(dim=0)  # mean over heads -> [K_cam]
                img_h, img_w = _image_hw_for_segment(result, cam_name, cam_idx)
                up = _upsample_patches(mean_vec, n_p, img_h, img_w)
                heat = _render_heatmap(up, f"L{layer_idx} {cam_name}", img_h, img_w)  # RGB uint8 HWC
                rr.log(f"attention/L{layer_idx:02d}/{cam_name}", rr.Image(heat))

    return hook


@parser.wrap()
def actor_vla_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()

    log_dir = os.path.join(cfg.output_dir or "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    init_logging(log_file=os.path.join(log_dir, f"actor_attn_{cfg.job_name}.log"), display_pid=False)
    logger.info("[INFERENCE] RTC inference with live attention viz starting.")

    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    def _sigint(sig, frame):
        logger.info("[INFERENCE] SIGINT - shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _sigint)

    try:
        trainer = Trainer.for_config(cfg)
        act_with_policy_rtc_inference(
            cfg=cfg,
            trainer=trainer,
            shutdown_event=shutdown_event,
            post_inference_hook_factory=_make_attention_hook,
        )
    finally:
        shutdown_event.set()
        logger.info("[INFERENCE] inference_w_attn finished.")


if __name__ == "__main__":
    actor_vla_cli()

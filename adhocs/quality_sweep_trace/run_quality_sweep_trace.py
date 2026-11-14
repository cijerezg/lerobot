"""Quality sweep in 3D: one frame, the same flow noise, the quality clause asked at 1..5.

    uv run python lerobot/adhocs/quality_sweep_trace/run_quality_sweep_trace.py --config_path=config_rl_validate.yaml \
        --policy.pretrained_path=outputs/rebot-corr10-nodepth-b128-20261111/checkpoints/002000/pretrained_model \
        --out_dir=outputs/quality_sweep_trace/corr10-ckpt2000_2026-11-13

Per anchor one ``predict_action_chunk_batch`` over five prompts that differ only in "The quality is
$q$ of 5." (no mistake, speed 5, the frame's own precision and contact), all rows on seed ``seed``'s
noise draw. The five chunks go through the action-trace probe's ``_analyse`` / ``_figure`` /
``_write_dashboard_html`` as its sample fan, named q1..q5, so the page is the Action Inspector with
quality where the flow draws usually are.

Anchors (val set, ``episodes`` x four stages, snapped to the image stride), from the segment table
and ``meta/depth_gripper_events.parquet``:
  approach      2 s before the FIRST close of the episode's first grasp segment
  near grasp    0.3 s before the LAST close of that segment (the one that lifts)
  move          middle of the move segment that follows
  near release  0.3 s before the first open of the segment after the move
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import matplotlib
import pandas as pd

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.probes import action_trace_probe
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import frame_metadata_lookup, probe_frame_inputs, probe_image_stride, register_config_choices
from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

QUALITY_LEVELS = (1, 2, 3, 4, 5)
APPROACH_S = 2.0
NEAR_S = 0.3


@dataclass
class QualitySweepTraceConfig(TrainRLServerPipelineConfig):
    out_dir: str = "outputs/quality_sweep_trace/corr10-ckpt2000_2026-11-13"
    episodes: str = "0,1,7,8"  # shirt, nail clipper, cup, bit
    seed: int = 0


def select_anchors(root: str, episodes: list[int], stride: int, fps: float) -> list[tuple[int, int, str]]:
    """(global_idx, frame_idx, stage) per episode: approach, near grasp, move, near release."""
    segs = pd.read_parquet(f"{root}/meta/episode_metadata.parquet")
    events = pd.read_parquet(f"{root}/meta/depth_gripper_events.parquet")
    starts = pd.read_parquet(f"{root}/meta/episodes/chunk-000/file-000.parquet")
    ep_start = dict(zip(starts.episode_index, starts.dataset_from_index))
    anchors = []
    for ep in episodes:
        base = int(ep_start[ep])
        grasp, move, release = (s for s in segs[segs.episode_index == ep].sort_values("segment_index").head(3).itertuples())
        ev = events[events.episode_index == ep]
        closes = sorted(int(f) for f in ev[ev.event_type == "close"].frame_index if grasp.from_index - base <= f < grasp.to_index - base)
        opens = sorted(int(f) for f in ev[ev.event_type == "open"].frame_index if release.from_index - base <= f < release.to_index - base)
        frames = {
            "approach": max(closes[0] - APPROACH_S * fps, 0),
            "near grasp": closes[-1] - NEAR_S * fps,
            "move": (move.from_index + move.to_index) / 2 - base,
            "near release": opens[0] - NEAR_S * fps,
        }
        for stage, frame in frames.items():
            frame = int(round(frame / stride)) * stride
            anchors.append((base + frame, frame, stage))
    return anchors


def run(cfg: QualitySweepTraceConfig) -> None:
    init_logging()
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.val_dataset_path)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    adapter._set_probe_cuda_graph_enabled(False)
    kin = RebotKinematics()
    p = cfg.probe_parameters
    fps = float(dataset.fps)
    labels = frame_metadata_lookup(dataset)
    anchors = select_anchors(cfg.val_dataset_path, [int(e) for e in cfg.episodes.split(",")], probe_image_stride(cfg), fps)
    names = [f"q{q}" for q in QUALITY_LEVELS]

    records = []
    for global_idx, expected_frame, stage in anchors:
        frame = probe_frame_inputs(dataset, cfg, global_idx, int(cfg.policy.chunk_size), metadata=None)
        assert frame["frame_idx"] == expected_frame
        own = labels[global_idx]
        metadatas = [{**own, "quality": q, "mistake": False, "speed": 5} for q in QUALITY_LEVELS]
        chunks, _ = adapter.predict_action_chunk_batch(
            frame["obs"], frame["task"], [frame["subtask"]] * len(metadatas),
            metadatas=metadatas, noise=adapter.flow_noise_like(len(metadatas), cfg.seed),
        )
        state = frame["obs"][OBS_STATE].reshape(-1).float().cpu()
        motor_speeds, joint_lower, joint_upper = action_trace_probe._resolve_actuator_config(cfg, int(frame["gt_actions"].shape[-1]))
        record = action_trace_probe._analyse(
            kin, state.numpy(), frame["gt_actions"].numpy(), [c.float().cpu().numpy() for c in chunks], p.trace_table_z,
            fps=fps, motor_speeds_deg_s=motor_speeds, joint_lower=joint_lower, joint_upper=joint_upper,
        )
        record.update(
            episode=int(frame["episode_idx"]), frame=int(frame["frame_idx"]), global_idx=int(global_idx),
            subtask=frame["subtask"], sample_names=names,
            trace_label=f"ep{frame['episode_idx']} fr{frame['frame_idx']} · {stage} · demo quality {own['quality']}",
            cameras=action_trace_probe._camera_context(frame["obs"]),
            camera_frames=action_trace_probe._camera_frames(frame["obs"]),
            measured_states=action_trace_probe._measured_states(dataset, int(global_idx), int(frame["episode_idx"]), int(frame["gt_actions"].shape[0])),
        )
        records.append(record)
        logging.info(f"[quality_sweep_trace] {record['trace_label']} · {frame['subtask']}")

    # q1 -> q5 dark to light, so the order of the levels can be read off the fan
    action_trace_probe.SAMPLE_COLORS = tuple(matplotlib.colors.to_hex(matplotlib.colormaps["viridis"](i / 4.4)) for i in range(5))
    html_path = os.path.join(cfg.out_dir, "quality_sweep_trace.html")
    action_trace_probe._write_dashboard_html(
        action_trace_probe._figure(records, p, fps=fps), records, html_path,
        page_title="Quality sweep",
        subtitle=f"One observation, flow noise fixed (seed {cfg.seed}); the five paths differ only in the quality the prompt asks for. "
                 f"Checkpoint {cfg.policy.pretrained_path}.",
        legend_note="q1 (dark) to q5 (light): quality asked, no mistake, speed 5, the frame's own precision and contact. Black = demonstration.",
    )
    logging.info(f"[quality_sweep_trace] wrote {html_path} ({len(records)} anchors)")


@parser.wrap()
def cli(cfg: QualitySweepTraceConfig):
    run(cfg)


def main() -> None:
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    register_config_choices()
    main()

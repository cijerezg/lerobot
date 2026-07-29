"""3D action-trace probe: where the policy would send the arm, from ground-truth states.

At anchor frames spaced through the validation episodes, draws ``n_samples`` flow samples
of the predicted action chunk and the ground-truth chunk in Cartesian space, via the reBot
URDF (``robots.rebot_b601_follower.kinematics``), and reports table clearance from the
full link geometry.

Three questions it answers, in decreasing order of what it can prove:

  1. **Clearance.** Would any link go through the table during the next chunk? Computed
     from per-link convex hulls, not link origins, so the gripper body and elbow count.
     This is a genuine pre-flight check: it runs before the arm ever moves.
  2. **Multimodality.** ``n_samples`` independent flow draws from one observation, drawn
     as a fan. A wide fan at a decision point means the policy is torn (which sock
     first); a tight bundle means it committed. No other probe exposes this.
  3. **Fit.** Cartesian error against GT, reported both as the mean over samples and the
     best sample, because they mean different things (see below).

OPEN-LOOP. Every anchor restarts the policy from a demo state, so this measures *intended*
motion, never closed-loop behaviour: compounding error and recovery from drift are
invisible by construction. Divergence from GT is also not automatically error — the task
is multimodal, and taking the other sock is a valid rollout that scores as a large
Cartesian distance. Read the fan, and prefer ``ee_err_best`` over ``ee_err_mean`` when
judging fit; use real rollouts for the closed-loop question.

Run:
    python -m lerobot.probes.action_trace_probe --config config_rl.yaml \
        --anchor_stride_s 2.0 --n_samples 8
"""

import csv
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import assemble_frame_history, build_episode_index, get_frame_data, makedirs
from lerobot.robots.rebot_b601_follower.kinematics import LINK_NAMES, RebotKinematics
from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ActionTraceProbeConfig(TrainRLServerPipelineConfig):
    episodes: str = ""  # comma-separated episode indices; empty = all
    anchor_stride_s: float = 2.0  # seconds between anchor frames
    max_anchors_per_episode: int = 30
    n_samples: int = 8  # independent flow draws per anchor — the fan
    table_z: float = 0.0  # table plane height (m). 0 = the arm's own mounting plane.
    clearance_warn_m: float = 0.01  # samples dipping below this are drawn red
    out_dir: str = "outputs/action_trace_probe"


def _anchor_frames(dataset, cfg) -> list[tuple[int, int]]:
    """``(episode_idx, global_idx)`` anchors, evenly spaced in time within each episode."""
    ep_to_indices = build_episode_index(dataset)
    wanted = {int(e) for e in cfg.episodes.split(",") if e.strip() != ""}
    stride = max(int(round(cfg.anchor_stride_s * dataset.fps)), 1)
    anchors = []
    for ep_idx in sorted(ep_to_indices):
        if wanted and ep_idx not in wanted:
            continue
        indices = ep_to_indices[ep_idx][::stride][: cfg.max_anchors_per_episode]
        anchors.extend((ep_idx, global_idx) for global_idx in indices)
    return anchors


def _observation(dataset, cfg, global_idx: int):
    """One frame's obs (with depth + short-term history attached), GT chunk, and context."""
    obs, gt_actions, _state, subtask, task_str, ep_idx, frame_idx = get_frame_data(
        dataset, global_idx, int(cfg.policy.chunk_size)
    )
    pointmap_config = getattr(cfg.policy, "pointmap_config", None)
    if pointmap_config is not None:
        depth_key = pointmap_config.depth_key
        depth = load_depth_png(dataset.root, f"{depth_key}.depth", ep_idx, frame_idx)
        obs[f"observation.depth.{depth_key}"] = torch.from_numpy(depth.astype(np.float32)).reshape(
            1, 1, *depth.shape
        )
    memory_cfg = getattr(cfg.policy, "memory", None)
    if memory_cfg is not None and memory_cfg.history_keys and memory_cfg.history_num_samples > 0:
        keys = [k for k in memory_cfg.history_keys if k == OBS_STATE or "images" in k]
        obs.update(assemble_frame_history(dataset, global_idx, memory_cfg, cfg.env.fps, keys))
    return obs, gt_actions, subtask, task_str, frame_idx


def _pairwise_spread(points: np.ndarray) -> np.ndarray:
    """Mean pairwise distance between samples at each timestep.

    Args:
        points: ``(K, T, 3)`` end-effector positions.

    Returns:
        ``(T,)``. Zero for K=1. This is the multimodality signal: it is large exactly
        where the samples disagree about where to go.
    """
    if len(points) < 2:
        return np.zeros(points.shape[1])
    deltas = points[:, None] - points[None, :]  # (K, K, T, 3)
    distances = np.linalg.norm(deltas, axis=-1)
    k = len(points)
    return distances.sum(axis=(0, 1)) / (k * (k - 1))


def _analyse(kin: RebotKinematics, gt_chunk: np.ndarray, samples: list[np.ndarray], table_z: float):
    """FK the GT and predicted chunks, and reduce them to traces + scalar metrics."""
    horizon = min(len(gt_chunk), min(len(s) for s in samples))
    gt_chunk = gt_chunk[:horizon]
    samples = [s[:horizon] for s in samples]

    gt_ee = kin.ee_path(gt_chunk)
    gt_z, _ = kin.min_height(gt_chunk)
    sample_ee = np.stack([kin.ee_path(s) for s in samples])  # (K, T, 3)
    sample_z, sample_owner = zip(*(kin.min_height(s) for s in samples))
    sample_z = np.stack(sample_z)  # (K, T)

    errors = np.linalg.norm(sample_ee - gt_ee[None], axis=-1).mean(axis=1)  # (K,)
    spread = _pairwise_spread(sample_ee)
    worst_sample = int(sample_z.min(axis=1).argmin())
    worst_step = int(sample_z[worst_sample].argmin())

    return {
        "gt_ee": gt_ee,
        "sample_ee": sample_ee,
        "sample_clearance": sample_z.min(axis=1) - table_z,  # (K,) per-sample worst
        "anchor_skeleton": kin.link_origins(gt_chunk[:1])[0],
        "worst_pose": samples[worst_sample][worst_step],
        "metrics": {
            "ee_err_mean": float(errors.mean()),
            "ee_err_best": float(errors.min()),
            "spread_mean": float(spread.mean()),
            "spread_terminal": float(spread[-1]),
            "clearance_gt": float(gt_z.min() - table_z),
            "clearance_pred": float(sample_z.min() - table_z),
            "clearance_link": LINK_NAMES[int(sample_owner[worst_sample][worst_step])],
        },
    }


def _figure(records: list[dict], cfg) -> "go.Figure":  # noqa: F821
    """One 3D scene with a slider over anchors — not one scene per anchor.

    Judging table clearance needs the viewer to orbit to a side view, which is only
    worth doing once; and a page of live 3D scenes is what made view_validation freeze.
    """
    import plotly.graph_objects as go

    points = np.concatenate([r["gt_ee"] for r in records] + [r["sample_ee"].reshape(-1, 3) for r in records])
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = 0.05
    table_x = [lo[0] - pad, hi[0] + pad]
    table_y = [lo[1] - pad, hi[1] + pad]

    def sample_traces(record):
        traces = []
        for ee, clearance in zip(record["sample_ee"], record["sample_clearance"]):
            unsafe = clearance < cfg.clearance_warn_m
            traces.append(
                go.Scatter3d(
                    x=ee[:, 0], y=ee[:, 1], z=ee[:, 2], mode="lines",
                    line=dict(color="#d62728" if unsafe else "#1f77b4", width=4),
                    opacity=0.95 if unsafe else 0.55,
                    name=f"sample (clr {clearance*1000:.0f} mm)",
                    showlegend=False,
                )
            )
        return traces

    def gt_trace(record):
        ee = record["gt_ee"]
        return go.Scatter3d(
            x=ee[:, 0], y=ee[:, 1], z=ee[:, 2], mode="lines+markers",
            line=dict(color="#111111", width=7), marker=dict(size=2.5, color="#111111"),
            name="ground truth",
        )

    def skeleton_trace(record):
        pts = record["anchor_skeleton"]
        return go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines+markers",
            line=dict(color="#888888", width=9), marker=dict(size=4, color="#555555"),
            name="arm at anchor",
        )

    first = records[0]
    table = go.Mesh3d(
        x=[table_x[0], table_x[1], table_x[1], table_x[0]],
        y=[table_y[0], table_y[0], table_y[1], table_y[1]],
        z=[cfg.table_z] * 4,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#c8b89a", opacity=0.35, name="table", showlegend=True, hoverinfo="skip",
    )
    fig = go.Figure(data=[table, gt_trace(first), skeleton_trace(first), *sample_traces(first)])

    updated = list(range(1, 3 + cfg.n_samples))
    fig.frames = [
        go.Frame(
            data=[gt_trace(r), skeleton_trace(r), *sample_traces(r)],
            traces=updated,
            name=str(i),
            layout=dict(title=dict(text=_title(r))),
        )
        for i, r in enumerate(records)
    ]
    fig.update_layout(
        title=dict(text=_title(first), font=dict(size=13)),
        scene=dict(
            xaxis=dict(title="x (m)"), yaxis=dict(title="y (m)"), zaxis=dict(title="z (m)"),
            aspectmode="data", bgcolor="white",
        ),
        paper_bgcolor="white",
        height=800,
        margin=dict(l=0, r=0, b=0, t=90),
        sliders=[dict(
            active=0, y=0, x=0.05, len=0.9, pad=dict(t=40),
            currentvalue=dict(prefix="anchor "),
            steps=[
                dict(method="animate", label=f"ep{r['episode']}:{r['frame']}",
                     args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                          transition=dict(duration=0))])
                for i, r in enumerate(records)
            ],
        )],
    )
    return fig


def _title(record: dict) -> str:
    m = record["metrics"]
    return (
        f"episode {record['episode']}  frame {record['frame']}   |   "
        f"clearance pred {m['clearance_pred']*1000:+.0f} mm ({m['clearance_link']}), "
        f"GT {m['clearance_gt']*1000:+.0f} mm   |   "
        f"fan spread {m['spread_terminal']*1000:.0f} mm   |   "
        f"EE err best {m['ee_err_best']*1000:.0f} / mean {m['ee_err_mean']*1000:.0f} mm"
    )


@parser.wrap()
def cli(cfg: ActionTraceProbeConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    makedirs(cfg.out_dir)

    from lerobot.datasets.factory import make_dataset

    dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None

    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    adapter._set_probe_cuda_graph_enabled(False)  # fresh flow noise per draw; keep eager
    kin = RebotKinematics()

    anchors = _anchor_frames(dataset, cfg)
    if not anchors:
        raise SystemExit(f"No anchors selected (episodes={cfg.episodes!r}).")
    logging.info(
        f"[action_trace] {len(anchors)} anchors x {cfg.n_samples} samples = "
        f"{len(anchors) * cfg.n_samples} forward passes"
    )

    records = []
    for n, (ep_idx, global_idx) in enumerate(anchors):
        obs, gt_actions, subtask, task_str, frame_idx = _observation(dataset, cfg, global_idx)
        samples = [
            # Deployment-condition metadata: ask for the best behaviour the steering
            # offers. Passing None omits the clause the policy was trained with.
            adapter.predict_action_chunk(
                obs, task_str, subtask=subtask, metadata={"quality": 5, "mistake": False}
            )[0].numpy()
            for _ in range(cfg.n_samples)
        ]
        record = _analyse(kin, gt_actions.numpy(), samples, cfg.table_z)
        record.update(episode=ep_idx, frame=frame_idx, global_idx=global_idx)
        records.append(record)
        logging.info(f"[{n + 1}/{len(anchors)}] {_title(record)}")

    fig = _figure(records, cfg)
    html_path = os.path.join(cfg.out_dir, "action_trace.html")
    fig.write_html(html_path, include_plotlyjs="cdn", auto_play=False)

    csv_path = os.path.join(cfg.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        columns = ["episode", "frame", "global_idx", *records[0]["metrics"]]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, r["metrics"].get(k)) for k in columns})

    logging.info("── worst predicted clearance ──")
    for r in sorted(records, key=lambda r: r["metrics"]["clearance_pred"])[:10]:
        m = r["metrics"]
        logging.info(
            f"  ep{r['episode']} frame {r['frame']:5d}  {m['clearance_pred']*1000:+7.1f} mm "
            f"({m['clearance_link']})   GT {m['clearance_gt']*1000:+7.1f} mm"
        )
    logging.info("── widest fan (most undecided) ──")
    for r in sorted(records, key=lambda r: -r["metrics"]["spread_terminal"])[:10]:
        m = r["metrics"]
        logging.info(
            f"  ep{r['episode']} frame {r['frame']:5d}  terminal spread "
            f"{m['spread_terminal']*1000:6.1f} mm   EE err best {m['ee_err_best']*1000:6.1f} mm"
        )
    breaches = [r for r in records if r["metrics"]["clearance_pred"] < cfg.clearance_warn_m]
    logging.info(
        f"{len(breaches)}/{len(records)} anchors predict a pass within "
        f"{cfg.clearance_warn_m*1000:.0f} mm of the table."
    )
    logging.info(f"wrote {html_path} and {csv_path}")


def main() -> None:
    # Same pre-parse machinery as rl_offline: register policy configs, strip inactive-model YAML fields.
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    import lerobot.rl.pi05.rl_pi05  # noqa: F401 — registers PI05RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401 — registers robot configs
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401 — registers teleop configs

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    main()

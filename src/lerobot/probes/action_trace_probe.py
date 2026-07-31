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

COMMANDED vs MEASURED. Every action — GT or predicted — is a commanded (leader) pose,
while the skeleton and the ``gripper now`` diamond are the measured follower pose. On
this arm the follower trails its command by a median 45 mm at the end-effector, so the
diamond does *not* sit on the traces, and the dotted ``follower lag`` segment is that
gap, not a step of motion.

OPEN-LOOP. Every anchor restarts the policy from a demo state, so this measures *intended*
motion, never closed-loop behaviour: compounding error and recovery from drift are
invisible by construction. Divergence from GT is also not automatically error — the task
is multimodal, and taking the other sock is a valid rollout that scores as a large
Cartesian distance. Read the fan, and prefer ``ee_err_best`` over ``ee_err_mean`` when
judging fit; use real rollouts for the closed-loop question.

Runs inside rl_offline's validation loop when ``probe_parameters.enable_action_trace`` is
set, or standalone:

    python -m lerobot.probes.action_trace_probe --config config_rl.yaml \
        --probe_parameters.trace_anchor_stride_s 2.0 --probe_parameters.trace_n_samples 5
"""

import csv
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import (
    build_episode_index,
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
)
from lerobot.robots.rebot_b601_follower.kinematics import LINK_NAMES, RebotKinematics
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ActionTraceProbeConfig(TrainRLServerPipelineConfig):
    """Tunables live under ``cfg.probe_parameters`` (the ``trace_*`` fields)."""


# One colour per flow draw, so a sample can be followed through the fan and matched to
# its clearance in the legend. Red is deliberately absent: it stays the table-breach
# signal, and reds/greens are kept apart for colour-blind readers.
SAMPLE_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#393b79", "#637939", "#8c6d31",
)


def _anchor_frames(dataset, p, depth_stride: int) -> list[tuple[int, int]]:
    """``(episode_idx, global_idx)`` anchors, evenly spaced in time within each episode.

    Anchors are snapped down onto the depth-sidecar grid: depth is written every
    ``depth_stride`` frames, and ``load_depth_png`` raises on any frame in between.
    """
    ep_to_indices = build_episode_index(dataset)
    wanted = {int(e) for e in (p.trace_episodes or "").split(",") if e.strip() != ""}
    stride = max(int(round(p.trace_anchor_stride_s * dataset.fps)), 1)
    stride -= stride % depth_stride
    stride = max(stride, depth_stride)
    anchors = []
    for ep_idx in sorted(ep_to_indices):
        if wanted and ep_idx not in wanted:
            continue
        indices = ep_to_indices[ep_idx][::stride][: p.trace_max_anchors_per_episode]
        anchors.extend((ep_idx, global_idx) for global_idx in indices)
    return anchors


def _observation(dataset, cfg, global_idx: int):
    """One frame's obs (with depth + short-term history attached), GT chunk, and context."""
    frame = probe_frame_inputs(dataset, cfg, global_idx, int(cfg.policy.chunk_size))
    return (
        frame["obs"],
        frame["gt_actions"],
        frame["subtask"],
        frame["task"],
        frame["metadata"],
        frame["frame_idx"],
    )


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


def _analyse(
    kin: RebotKinematics,
    state: np.ndarray,
    gt_chunk: np.ndarray,
    samples: list[np.ndarray],
    table_z: float,
):
    """FK the GT and predicted chunks, and reduce them to traces + scalar metrics.

    ``state`` is the *measured* follower pose at the anchor; every action in the dataset
    is a *commanded* leader pose. Those are not the same point in space: the follower
    trails its command by ~0.25 s of motion under load, which on this arm is a median
    45 mm at the end-effector — two thirds of the length of a whole 30-step chunk.

    So no trace is anchored on the measured pose. Chunks are drawn exactly as they are,
    starting at their own first action, and the state→first-command offset is reported
    as ``follower_lag`` and drawn as its own dashed segment. Joining them would fabricate
    a leading step longer than the motion the chunk actually contains.
    """
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

    start_ee = kin.ee_path(state[None])[0]  # (3,) — where the gripper actually is right now
    return {
        "start_ee": start_ee,
        "gt_ee": gt_ee,
        "sample_ee": sample_ee,
        "sample_clearance": sample_z.min(axis=1) - table_z,  # (K,) per-sample worst
        "sample_worst_link": [
            LINK_NAMES[int(owner[int(z.argmin())])] for owner, z in zip(sample_owner, sample_z, strict=True)
        ],
        "anchor_skeleton": kin.link_origins(state[None])[0],
        "worst_pose": samples[worst_sample][worst_step],
        "metrics": {
            "ee_err_mean": float(errors.mean()),
            "ee_err_best": float(errors.min()),
            "spread_mean": float(spread.mean()),
            "spread_terminal": float(spread[-1]),
            "clearance_gt": float(gt_z.min() - table_z),
            "clearance_pred": float(sample_z.min() - table_z),
            "clearance_link": LINK_NAMES[int(sample_owner[worst_sample][worst_step])],
            # Property of the teleop data, not of the checkpoint: how far the measured
            # pose sits behind the command it is chasing at this anchor.
            "follower_lag": float(np.linalg.norm(gt_ee[0] - start_ee)),
        },
    }


def _figure(records: list[dict], p) -> "go.Figure":  # noqa: F821
    """One 3D scene with a slider over anchors — not one scene per anchor.

    Judging table clearance needs the viewer to orbit to a side view, which is only
    worth doing once; and a page of live 3D scenes is what made view_validation freeze.
    """
    import plotly.graph_objects as go

    points = np.concatenate(
        [r["gt_ee"] for r in records]
        + [r["sample_ee"].reshape(-1, 3) for r in records]
        + [r["start_ee"][None] for r in records]
    )
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = 0.05
    table_x = [lo[0] - pad, hi[0] + pad]
    table_y = [lo[1] - pad, hi[1] + pad]

    def sample_traces(record):
        traces = []
        zipped = zip(
            record["sample_ee"], record["sample_clearance"], record["sample_worst_link"], strict=True
        )
        for k, (ee, clearance, link) in enumerate(zipped):
            unsafe = clearance < p.trace_clearance_warn_m
            # Identity comes from colour, so a breach cannot be recoloured red without
            # losing which sample it was: it is flagged by a dashed, heavier line and
            # by its legend entry instead.
            traces.append(
                go.Scatter3d(
                    x=ee[:, 0], y=ee[:, 1], z=ee[:, 2],
                    mode="lines+markers",
                    line=dict(
                        color=SAMPLE_COLORS[k % len(SAMPLE_COLORS)],
                        width=6 if unsafe else 3,
                        dash="dash" if unsafe else "solid",
                    ),
                    marker=dict(size=2, color=SAMPLE_COLORS[k % len(SAMPLE_COLORS)]),
                    opacity=1.0 if unsafe else 0.6,
                    name=(
                        f"sample {k} — {clearance*1000:.0f} mm ({link})"
                        + ("  ⚠ TABLE" if unsafe else "")
                    ),
                )
            )
        return traces

    def gt_trace(record):
        # Deliberately the heaviest line in the scene: it is the one thing every sample
        # is read against, and it has to stay findable inside the fan.
        ee = record["gt_ee"]
        return go.Scatter3d(
            x=ee[:, 0], y=ee[:, 1], z=ee[:, 2], mode="lines+markers",
            line=dict(color="#111111", width=11), marker=dict(size=4, color="#111111"),
            name="ground truth",
        )

    def gt_end_trace(record):
        x, y, z = record["gt_ee"][-1]
        return go.Scatter3d(
            x=[x], y=[y], z=[z], mode="markers",
            marker=dict(size=9, color="#111111", symbol="square",
                        line=dict(color="#ffffff", width=1)),
            name="ground truth — chunk end",
        )

    def lag_trace(record):
        """Measured pose → the command it is chasing. Not motion: teleop tracking error."""
        here, commanded = record["start_ee"], record["gt_ee"][0]
        return go.Scatter3d(
            x=[here[0], commanded[0]], y=[here[1], commanded[1]], z=[here[2], commanded[2]],
            mode="lines",
            line=dict(color="#999999", width=3, dash="dot"),
            name=f"follower lag — {record['metrics']['follower_lag']*1000:.0f} mm (not motion)",
        )

    def skeleton_trace(record):
        pts = record["anchor_skeleton"]
        return go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines+markers",
            line=dict(color="#888888", width=9), marker=dict(size=4, color="#555555"),
            name="arm now (measured state)",
        )

    def start_trace(record):
        x, y, z = record["start_ee"]
        return go.Scatter3d(
            x=[x], y=[y], z=[z], mode="markers",
            marker=dict(size=7, color="#111111", symbol="diamond",
                        line=dict(color="#ffffff", width=1)),
            name="gripper now (measured, behind the command)",
        )

    def anchor_traces(record):
        """Everything that is re-drawn when the slider moves, in a fixed order."""
        return [
            gt_trace(record), gt_end_trace(record), skeleton_trace(record),
            start_trace(record), lag_trace(record), *sample_traces(record),
        ]

    first = records[0]
    table = go.Mesh3d(
        x=[table_x[0], table_x[1], table_x[1], table_x[0]],
        y=[table_y[0], table_y[0], table_y[1], table_y[1]],
        z=[p.trace_table_z] * 4,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#c8b89a", opacity=0.35, name="table", showlegend=True, hoverinfo="skip",
    )
    fig = go.Figure(data=[table, *anchor_traces(first)])

    # Everything except the table (index 0) is re-drawn per anchor, and the slider frames
    # must name exactly those trace indices — so the count is taken from the data itself
    # rather than recounted by hand.
    updated = list(range(1, len(fig.data)))
    fig.frames = [
        go.Frame(
            data=anchor_traces(r),
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
        f"EE err best {m['ee_err_best']*1000:.0f} / mean {m['ee_err_mean']*1000:.0f} mm   |   "
        f"follower lag {m['follower_lag']*1000:.0f} mm"
    )


def run(adapter, dataset, cfg, output_dir):
    """Probe entry point shared by the CLI and rl_offline's validation loop."""
    p = cfg.probe_parameters
    makedirs(output_dir)
    kin = RebotKinematics()

    anchors = _anchor_frames(dataset, p, int(getattr(cfg.policy, "image_stride", 1)))
    if not anchors:
        raise ValueError(f"No anchors selected (trace_episodes={p.trace_episodes!r}).")
    logging.info(
        f"[action_trace] {len(anchors)} anchors x {p.trace_n_samples} samples = "
        f"{len(anchors) * p.trace_n_samples} forward passes"
    )

    # Fresh flow noise per draw: a replayed CUDA graph would hand every sample the
    # same noise and collapse the fan. Restored below so training is unaffected.
    adapter._set_probe_cuda_graph_enabled(False)
    records = []
    try:
        for n, (ep_idx, global_idx) in enumerate(anchors):
            obs, gt_actions, subtask, task_str, metadata, frame_idx = _observation(
                dataset, cfg, global_idx
            )
            samples = [
                # No generator: the spread across draws IS the measurement here, so
                # every sample must get fresh flow noise. Deployment-condition
                # metadata comes from probe_frame_inputs — ask for the best
                # behaviour the steering offers.
                adapter.predict_action_chunk(
                    obs, task_str, subtask=subtask, metadata=metadata
                )[0].numpy()
                for _ in range(p.trace_n_samples)
            ]
            state = obs[OBS_STATE].reshape(-1).numpy()
            record = _analyse(kin, state, gt_actions.numpy(), samples, p.trace_table_z)
            record.update(episode=ep_idx, frame=frame_idx, global_idx=global_idx)
            records.append(record)
            logging.info(f"[{n + 1}/{len(anchors)}] {_title(record)}")
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    fig = _figure(records, p)
    html_path = os.path.join(output_dir, "action_trace.html")
    fig.write_html(html_path, include_plotlyjs="cdn", auto_play=False)

    csv_path = os.path.join(output_dir, "metrics.csv")
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
    breaches = [r for r in records if r["metrics"]["clearance_pred"] < p.trace_clearance_warn_m]
    # WARNING so it survives the console filter rl_offline installs around probes.
    log = logging.warning if breaches else logging.info
    log(
        f"[action_trace] {len(breaches)}/{len(records)} anchors predict a pass within "
        f"{p.trace_clearance_warn_m*1000:.0f} mm of the table."
    )
    logging.info(f"wrote {html_path} and {csv_path}")


@parser.wrap()
def cli(cfg: ActionTraceProbeConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = load_probe_dataset(cfg)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    output_dir = os.path.join(cfg.probe_parameters.output_dir, "action_trace")
    run(adapter, dataset, cfg, output_dir)
    logging.info(f"Done. Output in {output_dir}/")


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

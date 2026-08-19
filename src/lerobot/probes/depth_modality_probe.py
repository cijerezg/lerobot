"""2×2 modality probe + sensitivity estimates for the joint depth read
(depth_redesign_options.md §5.1).

Runs predict_action_chunk on the same frame(s) under the four modality
conditions — {RGB+depth, RGB-only, depth-only, neither} — with IDENTICAL
fixed-seed flow noise, and reports:

  - four normalized-space trajectory criteria vs GT per condition — path MSE,
    temporal-shape MSE, final-position MSE, and pure final-direction alignment
    (is depth earning its keep under more than one definition of action quality?),
  - pairwise max|Δaction| between conditions (does depth influence output at all),
  - the RMS of the complete injected depth tokens over the pretrained embedding-table
    RMS (is the depth prefix becoming abnormally quiet or loud?),
  - finite-difference sensitivity: ‖Δactions‖ for a 1%-of-std perturbation of the
    raw depth input vs of the wrist RGB input (a directional-Jacobian estimate —
    separates "attended but ignored" from load-bearing).

Depth now enters the VLM prefix on DEPTH_TOKEN placeholders, so there is no gate and
no per-layer read bias to report — the α gate and the b_ℓ joint-softmax bias are both
gone. The tokens are in the sequence unconditionally. The copied visual path can still
learn features the trunk ignores or drift to an abnormal scale; the token-RMS ratio watches
scale, while the MSE conditions and finite-difference sensitivity measure whether depth
actually affects and improves the action.

Conditions are produced within one policy build: depth− removes the depth key
(learned null bank), while RGB− applies the exact training-time wrist-patch
attention mask, which also triggers the model-side bridge kill. Rebuild-to-rebuild
nondeterminism (~1e-1, from-scratch norm buffers) would swamp the comparisons.

Outputs under ``<output_dir>/``:
  depth_modality.json   every number below, plus per-frame rows
  depth_modality.png    condition MSE and FD sensitivity

Replaces probes/pointmap_bit_identity.py (retired 2026-07-26 with the α gate:
there is no gate-0 no-op to verify under the joint softmax).

Runs inside rl_offline's validation loop when ``probe_parameters.enable_depth_modality``
is set, or standalone:

    uv run python -m lerobot.probes.depth_modality_probe --config config_rl.yaml \
        --frame_indices 0,501,1200
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.depth_pointmap.modeling_stream import mask_camera_patch_span
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
    trajectory_error_components,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

CONDITIONS = ("rgb+depth", "rgb_only", "depth_only", "neither")
CONDITION_PAIRS = tuple(combinations(CONDITIONS, 2))


@dataclass
class DepthModalityProbeConfig(TrainRLServerPipelineConfig):
    frame_indices: str = ""  # comma-separated global dataset indices; empty = sample evenly
    fd_epsilon_rel: float = 0.01  # FD perturbation, fraction of the input's std


def _condition_obs(obs: dict, condition: str, *, depth_obs_key: str) -> dict:
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
    if condition in ("rgb_only", "neither"):
        # Encoder falls back to the learned null bank. The history window goes with
        # it: a depth-less condition that kept its past frames would not be depth-less.
        out.pop(depth_obs_key, None)
        for key in [k for k in out if str(k).startswith("history.depth.")]:
            out.pop(key)
    return out


def _write_manifest(output_dir: str, summary: dict) -> dict:
    """Describe the depth read to the manifest-driven viewer."""
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Depth Modality",
        group="Depth",
        claim="Does the point-map depth stream reach the actions, and does it improve them?",
        summary=summary,
        metrics=[
            Metric(
                "depth_benefit",
                "Depth benefit (MSE removed)",
                good="high",
                fmt=5,
                baseline=0.0,
                warn=0.0,
                primary=True,
                note="$\\mathrm{mse}(\\text{rgb only}) - \\mathrm{mse}(\\text{rgb+depth})$ at identical flow noise. Below zero the depth path is costing accuracy, which is the expected state early in a run while the from-scratch encoder is still noise.",
            ),
            Metric(
                "max_abs_delta.rgb+depth vs rgb_only",
                "Action shift when depth is removed",
                good="none",
                fmt=4,
                primary=True,
                note="Mean over frames of $\\max|\\Delta a|$ in normalized space. Near zero means the depth columns are not reaching the action at all — read before the benefit number, which is meaningless if nothing moved.",
            ),
            Metric(
                "fd_sensitivity.ratio",
                "Depth / wrist-RGB sensitivity",
                good="none",
                fmt=3,
                baseline=1.0,
                note="Finite-difference $\\|\\Delta a\\|$ for a 1%-of-std perturbation of raw depth over the same for wrist RGB. Separates attended-but-ignored from load-bearing; 1.0 means the two inputs move the action equally.",
            ),
            Metric(
                "depth_rgb_rms_ratio",
                "Depth / RGB token RMS",
                good="none",
                fmt=3,
                baseline=1.0,
                primary=True,
                note="RMS of final depth prefix tokens divided by final RGB prefix-token RMS at the exact text-embedding seam. This is the direct scale-matching measurement; sustained drift above 1 means depth is becoming louder than the tested RGB path.",
            ),
            Metric(
                "depth_pre_bound_token_rms",
                "Depth RMS before tanh bound",
                good="none",
                fmt=3,
                note="RMS of projected depth plus its marker immediately before the tanh. This may grow above 128 and reveals pressure against the bound.",
            ),
            Metric(
                "depth_injected_token_rms",
                "Depth RMS after tanh bound",
                good="none",
                fmt=3,
                note="RMS of the actual bounded depth tokens injected at the text-embedding seam. Every coordinate is limited to ±output_bound.",
            ),
            Metric(
                "depth_token_rms_ratio",
                "Depth projector / embedding RMS",
                good="none",
                fmt=3,
                baseline=1.0,
                note="RMS of the copied depth visual projector output over the pretrained text-token table RMS. This preserves the old chart's path-output/table interpretation; use depth_rgb_rms_ratio for direct modality matching.",
            ),
            Metric(
                "depth_late_early_rms_ratio",
                "H7 / H3 RMS",
                good="none",
                fmt=3,
                baseline=1.0,
                note="Residual-stream scale after depth block 7 divided by its scale after block 3. Persistent growth localizes scale drift inside the copied ViT blocks rather than the final projector.",
            ),
            Metric("n_frames", "Frames probed", good="none", fmt=0),
            Metric(
                "trajectory_depth_gain.path_mse",
                "Depth gain · path MSE",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Path MSE(rgb only) − path MSE(rgb+depth); positive means depth improves the full 30-target chunk.",
            ),
            Metric(
                "trajectory_depth_gain.shape_mse",
                "Depth gain · temporal shape",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Shape MSE(rgb only) − shape MSE(rgb+depth); positive means depth better matches adjacent-target changes.",
            ),
            Metric(
                "trajectory_depth_gain.terminal_mse",
                "Depth gain · final-position MSE",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Final-position MSE(rgb only) − final-position MSE(rgb+depth), at target 30.",
            ),
            Metric(
                "trajectory_depth_gain.terminal_direction_loss",
                "Depth gain · final direction",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Final-direction loss(rgb only) − loss(rgb+depth). Length is ignored; stationary GT endpoints are excluded.",
            ),
        ],
        panels=[
            Panel(
                "depth_modality.png",
                "Condition MSE and finite-difference sensitivity",
                how=(
                    "**Left** — normalized MSE against the demonstrated chunk under the four "
                    "modality conditions, all at identical flow noise. ``rgb+depth`` is the "
                    "deployment condition; ``rgb_only`` blanks the depth map, ``depth_only`` "
                    "blanks the wrist RGB, ``neither`` blanks both. The benefit is the gap "
                    "between the first two bars, and ``neither`` is the scale on which to "
                    "judge it.\n\n"
                    "**Right** — finite-difference sensitivity per frame: $\\|\\Delta a\\|$ for a "
                    "1%-of-std perturbation of the raw depth input against the same for wrist "
                    "RGB. This is the metric that separates 'depth is in the sequence but "
                    "inert' from 'depth is load-bearing'; the ratio is in the title bar.\n\n"
                    "There is no attention-mass panel any more. Depth tokens sit in the VLM "
                    "prefix behind an ordinary softmax over ~1.5k positions across 36 layers, "
                    "so measuring their mass means materializing full attention maps — and "
                    "mass was only ever a proxy for the influence the left and right panels "
                    "measure directly."
                ),
                primary=True,
            ),
            Panel(
                "depth_modality.json",
                "Every number above, plus the per-frame rows behind them",
                how="The summary dict and one row per probed frame, for checking whether a "
                    "mean is carried by a few frames.",
            ),
        ],
        see_also=["attention_budget", "action_trace"],
        extra={
            "viewer": {
                "metric_groups": [
                    {
                        "title": "Depth gain by trajectory criterion",
                        "keys": [
                            "trajectory_depth_gain.path_mse",
                            "trajectory_depth_gain.shape_mse",
                            "trajectory_depth_gain.terminal_mse",
                            "trajectory_depth_gain.terminal_direction_loss",
                        ],
                    }
                ]
            }
        },
    )


def _render(summary: dict, per_frame: list[dict], output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    conditions = summary["conditions"]
    mse = [summary["mse_norm"][c] for c in conditions]
    axes[0].bar(np.arange(len(conditions)), mse, color="#457B9D")
    axes[0].set_xticks(np.arange(len(conditions)), conditions, rotation=15, ha="right")
    axes[0].set_ylabel("normalized MSE vs GT")
    axes[0].set_title(
        f"Modality conditions — depth benefit {summary['depth_benefit']:+.5f}\n"
        "(mse(rgb_only) − mse(rgb+depth); positive ⇒ depth helps)",
        fontsize=10,
    )

    depth_s = [r["fd_sensitivity"]["depth"] for r in per_frame]
    rgb_s = [r["fd_sensitivity"]["rgb"] for r in per_frame]
    idx = np.arange(len(depth_s))
    axes[1].plot(idx, depth_s, marker="o", ms=3, color="#2A9D8F", label="depth")
    axes[1].plot(idx, rgb_s, marker="o", ms=3, color="#E76F51", label="wrist RGB")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("probed frame")
    axes[1].set_ylabel("‖Δactions‖ @ 1% input noise")
    axes[1].set_title("Finite-difference sensitivity")
    axes[1].legend(fontsize=8)

    fd = summary["fd_sensitivity"]
    fig.suptitle(
        f"Point-map depth read — n={summary['n_frames']} frames  |  "
        f"FD sensitivity depth={fd['depth']:.3e} rgb={fd['rgb']:.3e} "
        f"(ratio {fd['ratio']:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def run(adapter, dataset, cfg, output_dir: str) -> None:
    pointmap_config = getattr(cfg.policy, "pointmap_config", None)
    if pointmap_config is None:
        logging.info("[depth_modality] policy.pointmap_config is null — skipping.")
        return

    makedirs(output_dir)
    p = cfg.probe_parameters
    device = adapter.device
    policy = adapter.policy
    chunk_size = int(cfg.policy.chunk_size)
    depth_key = pointmap_config.depth_key
    depth_obs_key = f"observation.depth.{depth_key}"
    rgb_obs_key = f"observation.images.{depth_key}"
    image_patch_id, num_images, cam_index = policy._pointmap_wrist_meta()

    explicit = [int(s) for s in str(getattr(cfg, "frame_indices", "") or "").split(",") if s.strip()]
    if explicit:
        stride = probe_image_stride(cfg)
        frame_indices = [idx - idx % stride for idx in explicit]
        if frame_indices != explicit:
            logging.warning(
                f"[depth_modality] snapped frame indices onto the depth grid (stride {stride}): "
                f"{explicit} -> {frame_indices}"
            )
    else:
        frame_indices = [
            g for _, _, g in sample_episodes_evenly(
                dataset,
                int(getattr(p, "depth_modality_n_frames", None) or p.n_frames_per_episode),
                p.max_episodes,
                p.random_seed,
                probe_image_stride(cfg),
            )
        ]
    if not frame_indices:
        logging.warning("[depth_modality] no frames selected.")
        return

    adapter._set_probe_cuda_graph_enabled(False)

    def predict(obs, frame, *, rgb_on: bool = True) -> torch.Tensor:
        batch = adapter._make_batch(
            obs, frame["task"], subtask=frame["subtask"], metadata=frame["metadata"]
        )
        if not rgb_on:
            batch["attention_mask"] = mask_camera_patch_span(
                batch["attention_mask"],
                batch["input_ids"],
                image_patch_id=image_patch_id,
                num_images=num_images,
                cam_index=cam_index,
            )
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        return (
            policy.predict_action_chunk(
                batch, inference_action_mode="continuous", generator=generator
            )
            .float()
            .cpu()
            .squeeze(0)
        )

    mse_by_condition: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    # Keyed by whatever trajectory_error_components returns, so a new criterion
    # flows through to the summary instead of raising here.
    trajectory_by_condition: dict[str, dict[str, list[float]]] = {}
    pairwise_deltas: dict[tuple[str, str], list[float]] = {pair: [] for pair in CONDITION_PAIRS}
    sens_depth_list: list[float] = []
    sens_rgb_list: list[float] = []
    per_frame: list[dict] = []
    scale_samples: dict[str, list[float]] = {}
    n_context = 0

    def record_scale(name: str, value: float) -> None:
        scale_samples.setdefault(name, []).append(float(value))

    def capture_rgb_depth_scale() -> None:
        embed_rms = float(policy._depth_embed_rms)
        if policy._depth_token_rms is not None:
            projected = policy._depth_token_rms.item()
            record_scale("depth_projected_token_rms", projected)
            record_scale("depth_token_rms_ratio", projected / embed_rms)
        for stage, value in policy._depth_stage_rms.items():
            record_scale(f"depth_{stage}_token_rms", value.item())
        early_tap, late_tap = policy.config.pointmap_config.visual_feature_taps
        early = policy._depth_stage_rms.get(f"block{early_tap}")
        late = policy._depth_stage_rms.get(f"block{late_tap}")
        if early is not None and late is not None:
            record_scale(
                "depth_late_early_rms_ratio", late.item() / max(early.item(), 1e-12)
            )
        if late is not None and policy._depth_token_rms is not None:
            record_scale(
                "depth_projected_late_rms_ratio",
                policy._depth_token_rms.item() / max(late.item(), 1e-12),
            )
        backbone = policy._backbone()
        depth_rms = getattr(backbone, "_lerobot_depth_input_rms", None)
        rgb_rms = getattr(backbone, "_lerobot_rgb_input_rms", None)
        if depth_rms is not None:
            depth_value = depth_rms.item()
            record_scale("depth_injected_token_rms", depth_value)
            record_scale("depth_injected_rms_ratio", depth_value / embed_rms)
        if rgb_rms is not None:
            rgb_value = rgb_rms.item()
            record_scale("rgb_injected_token_rms", rgb_value)
            if depth_rms is not None:
                record_scale(
                    "depth_rgb_rms_ratio", depth_rms.item() / max(rgb_value, 1e-12)
                )

    try:
        for global_idx in frame_indices:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            obs = frame["obs"]
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"]).float()
            hold_raw = frame["state"][: frame["gt_actions"].shape[-1]].unsqueeze(0).repeat(
                frame["gt_actions"].shape[0], 1
            )
            hold_norm = adapter.normalize_gt_actions(hold_raw, frame["state"]).float()

            actions: dict[str, torch.Tensor] = {}
            for condition in CONDITIONS:
                cond_obs = _condition_obs(obs, condition, depth_obs_key=depth_obs_key)
                actions[condition] = predict(
                    cond_obs, frame, rgb_on=condition in ("rgb+depth", "rgb_only")
                )
                # Capture only the real RGB+depth condition. Reading after the final
                # "neither" condition would report the learned null bank instead.
                if condition == "rgb+depth":
                    capture_rgb_depth_scale()

            if not n_context:
                probe_batch = adapter._make_batch(
                    obs, frame["task"], subtask=frame["subtask"], metadata=frame["metadata"]
                )
                n_context = int(probe_batch["input_ids"].shape[-1])

            horizon = min(actions["rgb+depth"].shape[0], gt_norm.shape[0])
            row = {
                "global_idx": int(global_idx),
                "mse_norm": {},
                "trajectory": {},
                "max_abs_delta": {},
            }
            logging.info(f"frame {global_idx}:")
            for condition in CONDITIONS:
                mse = torch.nn.functional.mse_loss(
                    actions[condition][:horizon], gt_norm[:horizon]
                ).item()
                mse_by_condition[condition].append(mse)
                row["mse_norm"][condition] = mse
                components = trajectory_error_components(
                    actions[condition][:horizon],
                    gt_norm[:horizon],
                    hold_norm[:horizon],
                )
                row["trajectory"][condition] = {}
                for key, tensor in components.items():
                    value = float(tensor) if bool(torch.isfinite(tensor)) else None
                    row["trajectory"][condition][key] = value
                    if value is not None:
                        by_condition = trajectory_by_condition.setdefault(
                            key, {c: [] for c in CONDITIONS}
                        )
                        by_condition[condition].append(value)
                logging.info(f"  mse_norm[{condition:>9s}] = {mse:.5f}")
            for left, right in CONDITION_PAIRS:
                delta = (actions[left] - actions[right]).abs().max().item()
                pairwise_deltas[(left, right)].append(delta)
                row["max_abs_delta"][f"{left} vs {right}"] = delta
                logging.info(f"  max|Δ| {left} vs {right} = {delta:.4e}")

            # Finite-difference sensitivity: same seed, 1%-of-std input perturbation.
            eps = float(getattr(cfg, "fd_epsilon_rel", 0.01))
            depth_raw = obs[depth_obs_key]
            pert = torch.randn_like(depth_raw) * depth_raw.float().std() * eps
            sens_depth = (
                predict({**obs, depth_obs_key: depth_raw + pert}, frame) - actions["rgb+depth"]
            ).norm().item()
            rgb_raw = obs[rgb_obs_key].float()
            pert = torch.randn_like(rgb_raw) * rgb_raw.std() * eps
            sens_rgb = (
                predict(
                    {**obs, rgb_obs_key: (rgb_raw + pert).to(obs[rgb_obs_key].dtype)}, frame
                ) - actions["rgb+depth"]
            ).norm().item()
            sens_depth_list.append(sens_depth)
            sens_rgb_list.append(sens_rgb)
            row["fd_sensitivity"] = {"depth": sens_depth, "rgb": sens_rgb}
            per_frame.append(row)
            logging.info(
                f"  fd sensitivity (‖Δactions‖ @ {eps:.0%} input noise): depth={sens_depth:.4e} "
                f"rgb={sens_rgb:.4e} ratio={sens_depth / max(sens_rgb, 1e-12):.3f}"
            )
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    n = len(per_frame)
    mean_depth = sum(sens_depth_list) / n
    mean_rgb = sum(sens_rgb_list) / n
    trajectory = {
        key: {
            condition: (
                float(np.mean(values)) if values else None
            )
            for condition, values in by_condition.items()
        }
        for key, by_condition in trajectory_by_condition.items()
    }
    trajectory_depth_gain = {}
    for key, by_condition in trajectory.items():
        both_value = by_condition["rgb+depth"]
        rgb_value = by_condition["rgb_only"]
        trajectory_depth_gain[key] = (
            None
            if both_value is None or rgb_value is None
            else rgb_value - both_value
        )
    summary = {
        "n_frames": n,
        "frame_indices": [int(i) for i in frame_indices],
        "conditions": list(CONDITIONS),
        "mse_norm": {c: sum(v) / n for c, v in mse_by_condition.items()},
        "max_abs_delta": {f"{a} vs {b}": sum(v) / n for (a, b), v in pairwise_deltas.items()},
        "depth_benefit": trajectory_depth_gain["path_mse"],
        "trajectory": trajectory,
        "trajectory_depth_gain": trajectory_depth_gain,
        "fd_sensitivity": {
            "depth": mean_depth,
            "rgb": mean_rgb,
            "ratio": mean_depth / max(mean_rgb, 1e-12),
        },
    }
    summary["n_depth_tokens"] = int(policy.config.pointmap_config.num_pooled_tokens)
    summary["n_context_tokens"] = n_context
    summary.update(
        {name: sum(values) / len(values) for name, values in scale_samples.items() if values}
    )

    with open(os.path.join(output_dir, "depth_modality.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)

    # write_index drops panels whose file is not on disk yet, so render first or the
    # probe's only figure never reaches the viewer.
    _render(summary, per_frame, os.path.join(output_dir, "depth_modality.png"))
    _write_manifest(output_dir, summary)

    logging.info("── summary over frames ──")
    for condition in CONDITIONS:
        logging.info(f"mse_norm[{condition:>9s}] mean = {summary['mse_norm'][condition]:.5f}")
    for name, value in summary["max_abs_delta"].items():
        logging.info(f"max|Δ| {name} mean = {value:.4e}")
    if "depth_rgb_rms_ratio" in summary:
        logging.info(
            f"depth_rgb_rms_ratio = {summary['depth_rgb_rms_ratio']:.3f} "
            f"(final depth prefix RMS / final RGB prefix RMS)"
        )
    if "depth_pre_bound_token_rms" in summary and "depth_injected_token_rms" in summary:
        logging.info(
            f"depth token RMS: pre_bound={summary['depth_pre_bound_token_rms']:.3f} "
            f"bounded={summary['depth_injected_token_rms']:.3f} "
            f"(coordinate bound={policy.config.pointmap_config.output_bound:g})"
        )
    if "depth_token_rms_ratio" in summary:
        logging.info(
            f"depth_token_rms_ratio = {summary['depth_token_rms_ratio']:.3f} "
            f"(depth projector output / pretrained token-table RMS)"
        )
    logging.info(
        f"depth benefit  mse(rgb_only) − mse(rgb+depth) = {summary['depth_benefit']:+.5f} "
        f"(positive ⇒ depth helps)"
    )
    logging.info(
        f"mean fd sensitivity: depth={mean_depth:.4e} rgb={mean_rgb:.4e}"
    )
    logging.info(f"wrote {os.path.join(output_dir, 'depth_modality.json')} and .png")


@parser.wrap()
def cli(cfg: DepthModalityProbeConfig):
    init_logging()
    if getattr(cfg.policy, "pointmap_config", None) is None:
        raise SystemExit("policy.pointmap_config is null in this config — nothing to probe.")
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = load_probe_dataset(cfg)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    output_dir = os.path.join(cfg.probe_parameters.output_dir, "depth_modality")
    run(adapter, dataset, cfg, output_dir)


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

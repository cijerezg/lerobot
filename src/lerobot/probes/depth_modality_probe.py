"""2×2 modality probe + sensitivity estimates for the joint depth read
(depth_redesign_options.md §5.1).

Runs predict_action_chunk on the same frame(s) under the four modality
conditions — {RGB+depth, RGB-only, depth-only, neither} — with IDENTICAL
fixed-seed flow noise, and reports:

  - normalized-space MSE vs GT per condition (is depth earning its keep:
    mse(RGB) − mse(RGB+depth) is the headline number),
  - pairwise max|Δaction| between conditions (does depth influence output at all),
  - per-layer attention mass on the depth columns (via the mass-capture hook),
  - the learned per-layer read bias b_ℓ and the depth mass it implies on its own,
  - finite-difference sensitivity: ‖Δactions‖ for a 1%-of-std perturbation of the
    raw depth input vs of the wrist RGB input (a directional-Jacobian estimate —
    separates "attended but ignored" from load-bearing).

b_ℓ is the parameter that replaced the α gate. The α gate failed by never training
(a random walk at ~0.007 absmax) and that went unnoticed for weeks because nothing
reported it, so the raw values and their implied mass are written on every run
alongside the measured mass. Measured ≫ implied means the read is input-driven;
measured ≈ implied means the layer attends depth by prior alone.

Conditions are produced within one policy build: depth− removes the depth key
(learned null bank), while RGB− applies the exact training-time wrist-patch
attention mask, which also triggers the model-side bridge kill. Rebuild-to-rebuild
nondeterminism (~1e-1, from-scratch norm buffers) would swamp the comparisons.

Outputs under ``<output_dir>/``:
  depth_modality.json   every number below, plus per-frame rows
  depth_modality.png    condition MSE, per-layer mass vs b_ℓ, FD sensitivity

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
from lerobot.policies.molmoact2.modeling_molmoact2 import (
    drain_pointmap_mass_records,
    set_pointmap_mass_capture,
)
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
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


def _read_bias(policy, n_context: int, n_depth: int) -> dict:
    """The learned per-layer depth score bias and the mass it implies alone.

    Under the joint softmax a layer's depth columns carry an extra score b_ℓ. With
    all content scores equal, that alone puts

        n_d·e^{b} / (n_d·e^{b} + n_ctx)

    of the row's mass on depth. Comparing this prior against the measured
    ``depth_attn_mass`` says whether a layer's depth read is driven by the input or
    only by the bias.
    """
    stream = getattr(policy, "depth_stream", None)
    bias = getattr(stream, "depth_bias", None)
    if bias is None:
        return {}
    values = bias.detach().float().cpu().numpy()
    weights = n_depth * np.exp(values)
    return {
        "depth_bias": values.tolist(),
        "depth_bias_implied_mass": (weights / (weights + max(n_context, 1))).tolist(),
        "depth_bias_init": -2.0,
    }


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
                "depth_attn_mass_mean",
                "Mean depth attention mass",
                good="none",
                fmt=4,
                note="Averaged over stream layers and frames. Compare against ``depth_bias_implied_mass`` in the JSON: measured ≫ implied means the read is input-driven, measured ≈ implied means the layer attends depth by prior alone.",
            ),
            Metric("depth_attn_mass_max", "Peak layer depth mass", good="none", fmt=4),
            Metric("n_frames", "Frames probed", good="none", fmt=0),
        ],
        panels=[
            Panel(
                "depth_modality.png",
                "Condition MSE, depth mass per layer, and the learned read bias",
                how=(
                    "**Left** — normalized MSE against the demonstrated chunk under the four "
                    "modality conditions, all at identical flow noise. ``rgb+depth`` is the "
                    "deployment condition; ``rgb_only`` blanks the depth map, ``depth_only`` "
                    "blanks the wrist RGB, ``neither`` blanks both. The benefit is the gap "
                    "between the first two bars, and ``neither`` is the scale on which to "
                    "judge it.\n\n"
                    "**Middle** — softmax mass the action queries put on the depth columns, "
                    "per action-expert layer: measured (solid) against the mass the learned "
                    "bias $b_\\ell$ would produce on its own with no input (dashed). Solid "
                    "above dashed is an input-driven read; the two on top of each other means "
                    "the layer attends depth by prior alone.\n\n"
                    "**Right** — $b_\\ell$ itself against its initialisation. A line still flat "
                    "on the init value is an untrained read gate, which is the expected state "
                    "early in a run and makes the middle panel uninformative.\n\n"
                    "The finite-difference sensitivities are in the figure's title bar, not in "
                    "a panel: they are two numbers, not a series."
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
    )


def _render(summary: dict, output_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))

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

    mass = summary.get("depth_attn_mass_by_layer")
    implied = summary.get("depth_bias_implied_mass")
    if mass:
        layers = np.arange(len(mass))
        axes[1].plot(layers, mass, color="#2A9D8F", linewidth=1.6, label="measured mass")
        if implied:
            axes[1].plot(layers, implied, color="#E76F51", linestyle="--", linewidth=1.4,
                         label="implied by b$_\\ell$ alone")
        axes[1].set_xlabel("action-expert layer")
        axes[1].set_ylabel("softmax mass on depth columns")
        axes[1].set_title("Depth read per layer")
        axes[1].legend(fontsize=8)

    bias = summary.get("depth_bias")
    if bias:
        axes[2].plot(np.arange(len(bias)), bias, color="#9B5DE5", linewidth=1.6)
        axes[2].axhline(summary.get("depth_bias_init", -2.0), color="#888888", linestyle="--",
                        linewidth=1.0, label="init")
        axes[2].set_xlabel("action-expert layer")
        axes[2].set_ylabel("b$_\\ell$")
        axes[2].set_title("Learned depth read bias (flat at init ⇒ untrained)")
        axes[2].legend(fontsize=8)

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
                int(getattr(p, "depth_modality_n_frames", 6)),
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
    pairwise_deltas: dict[tuple[str, str], list[float]] = {pair: [] for pair in CONDITION_PAIRS}
    mass_by_layer: list[np.ndarray] = []
    sens_depth_list: list[float] = []
    sens_rgb_list: list[float] = []
    per_frame: list[dict] = []
    n_context = 0

    try:
        for global_idx in frame_indices:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            obs = frame["obs"]
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"]).float()

            mass: list[float] = []
            actions: dict[str, torch.Tensor] = {}
            for condition in CONDITIONS:
                cond_obs = _condition_obs(obs, condition, depth_obs_key=depth_obs_key)
                capture_mass = condition == "rgb+depth"
                if capture_mass:
                    set_pointmap_mass_capture(True)
                try:
                    actions[condition] = predict(
                        cond_obs, frame, rgb_on=condition in ("rgb+depth", "rgb_only")
                    )
                finally:
                    if capture_mass:
                        mass = drain_pointmap_mass_records()
                        set_pointmap_mass_capture(False)

            if not n_context:
                probe_batch = adapter._make_batch(
                    obs, frame["task"], subtask=frame["subtask"], metadata=frame["metadata"]
                )
                n_context = int(probe_batch["input_ids"].shape[-1])

            horizon = min(actions["rgb+depth"].shape[0], gt_norm.shape[0])
            row = {"global_idx": int(global_idx), "mse_norm": {}, "max_abs_delta": {}}
            logging.info(f"frame {global_idx}:")
            for condition in CONDITIONS:
                mse = torch.nn.functional.mse_loss(
                    actions[condition][:horizon], gt_norm[:horizon]
                ).item()
                mse_by_condition[condition].append(mse)
                row["mse_norm"][condition] = mse
                logging.info(f"  mse_norm[{condition:>9s}] = {mse:.5f}")
            for left, right in CONDITION_PAIRS:
                delta = (actions[left] - actions[right]).abs().max().item()
                pairwise_deltas[(left, right)].append(delta)
                row["max_abs_delta"][f"{left} vs {right}"] = delta
                logging.info(f"  max|Δ| {left} vs {right} = {delta:.4e}")
            if mass:
                # mass has one record per (denoise step × layer); fold to per-layer means.
                layers = len(policy.depth_stream.blocks)
                if len(mass) % layers:
                    raise RuntimeError(
                        f"Captured {len(mass)} depth-mass records for {layers} layers."
                    )
                per_layer = np.asarray(mass, dtype=np.float64).reshape(-1, layers).mean(axis=0)
                mass_by_layer.append(per_layer)
                row["depth_attn_mass_by_layer"] = per_layer.tolist()
                logging.info(
                    f"  depth_attn_mass: mean={per_layer.mean():.4f} max={per_layer.max():.4f} "
                    f"argmax=layer {int(per_layer.argmax())}"
                )

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
        set_pointmap_mass_capture(False)
        adapter._restore_probe_cuda_graph_enabled()

    n = len(per_frame)
    both = sum(mse_by_condition["rgb+depth"]) / n
    rgb = sum(mse_by_condition["rgb_only"]) / n
    mean_depth = sum(sens_depth_list) / n
    mean_rgb = sum(sens_rgb_list) / n
    summary = {
        "n_frames": n,
        "frame_indices": [int(i) for i in frame_indices],
        "conditions": list(CONDITIONS),
        "mse_norm": {c: sum(v) / n for c, v in mse_by_condition.items()},
        "max_abs_delta": {f"{a} vs {b}": sum(v) / n for (a, b), v in pairwise_deltas.items()},
        "depth_benefit": rgb - both,
        "fd_sensitivity": {
            "depth": mean_depth,
            "rgb": mean_rgb,
            "ratio": mean_depth / max(mean_rgb, 1e-12),
        },
    }
    if mass_by_layer:
        mean_mass = np.stack(mass_by_layer).mean(axis=0)
        summary["depth_attn_mass_by_layer"] = mean_mass.tolist()
        summary["depth_attn_mass_mean"] = float(mean_mass.mean())
        summary["depth_attn_mass_max"] = float(mean_mass.max())
        summary["n_depth_tokens"] = int(policy.pointmap_encoder.num_tokens)
        summary["n_context_tokens"] = n_context
        summary.update(_read_bias(policy, n_context, int(policy.pointmap_encoder.num_tokens)))

    with open(os.path.join(output_dir, "depth_modality.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)

    # write_index drops panels whose file is not on disk yet, so render first or the
    # probe's only figure never reaches the viewer.
    _render(summary, os.path.join(output_dir, "depth_modality.png"))
    _write_manifest(output_dir, summary)

    logging.info("── summary over frames ──")
    for condition in CONDITIONS:
        logging.info(f"mse_norm[{condition:>9s}] mean = {summary['mse_norm'][condition]:.5f}")
    for name, value in summary["max_abs_delta"].items():
        logging.info(f"max|Δ| {name} mean = {value:.4e}")
    if "depth_attn_mass_by_layer" in summary:
        logging.info(
            f"depth_attn_mass mean={summary['depth_attn_mass_mean']:.4f} "
            f"max={summary['depth_attn_mass_max']:.4f}"
        )
        bias = summary.get("depth_bias")
        if bias:
            logging.info(
                f"depth_bias b_l: min={min(bias):+.3f} max={max(bias):+.3f} "
                f"mean={sum(bias) / len(bias):+.3f} (init -2.0); implied mass mean="
                f"{sum(summary['depth_bias_implied_mass']) / len(bias):.4f}"
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

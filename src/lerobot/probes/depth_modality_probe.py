"""Matched-depth counterfactuals + sensor-loss stress tests for the joint depth read.

Runs predict_action_chunk on the same frame(s) with IDENTICAL fixed-seed flow noise.
The primary counterfactual replaces the complete current-plus-history depth window with
real depth from a different episode, matched by task/subtask when possible and then by
robot state and episode progress. A same-episode stale window is a secondary control.
The old all-null depth condition remains only as a missing-sensor stress test; because
training uses ``pointmap_config.dropout_prob: 0.0``, it is not evidence for the value of
depth content.

The probe reports:

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

Conditions are produced within one policy build. Foreign/stale treatments swap all
``observation.depth.*`` and ``history.depth.*`` tensors together. Null depth removes
those keys (learned null bank), while RGB− applies the wrist-patch attention mask and
triggers the model-side bridge kill. Rebuild-to-rebuild nondeterminism (~1e-1, from-
scratch norm buffers) would swamp the comparisons.

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
from collections import Counter
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
    assemble_frame_history,
    build_episode_index,
    get_subtask_idx,
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
    trajectory_error_components,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

REAL = "rgb+depth"
FOREIGN = "foreign_depth"
STALE = "stale_depth"
NULL = "rgb_only"
DEPTH_ONLY = "depth_only"
NEITHER = "neither"
CONDITIONS = (REAL, FOREIGN, STALE, NULL, DEPTH_ONLY, NEITHER)
CONDITION_PAIRS = tuple(combinations(CONDITIONS, 2))


@dataclass
class DepthModalityProbeConfig(TrainRLServerPipelineConfig):
    frame_indices: str = ""  # comma-separated global dataset indices; empty = sample evenly
    fd_epsilon_rel: float = 0.01  # FD perturbation, fraction of the input's std


def _drop_depth(obs: dict, *, depth_obs_key: str) -> dict:
    """Remove the complete depth window, retaining the legacy sensor-loss stress test."""
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
    out.pop(depth_obs_key, None)
    for key in [k for k in out if str(k).startswith("history.depth.")]:
        out.pop(key)
    return out


def _replace_depth_window(obs: dict, donor: dict, *, depth_obs_key: str) -> dict:
    """Replace current depth and every depth-history slot as one treatment.

    Missing donor keys are an error rather than a silent mixture of recipient and donor
    depth: that mixture would make the counterfactual impossible to interpret.
    """
    depth_keys = [depth_obs_key, *sorted(k for k in obs if str(k).startswith("history.depth."))]
    missing = [key for key in depth_keys if key not in donor]
    if missing:
        raise KeyError(f"donor depth window is missing keys: {missing}")
    out = dict(obs)
    for key in depth_keys:
        out[key] = donor[key].to(dtype=obs[key].dtype)
    return out


def _scalar(value, default: int = -1) -> int:
    if value is None:
        return default
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _depth_match_descriptors(dataset, stride: int) -> tuple[list[dict], dict[int, list[int]]]:
    """Low-dimensional, stride-valid frames used to match foreign depth donors."""
    by_episode = build_episode_index(dataset)
    descriptors: list[dict] = []
    for episode_idx, indices in by_episode.items():
        episode_length = max(len(indices) - 1, 1)
        for global_idx in indices:
            row = dataset.hf_dataset[global_idx]
            frame_idx = _scalar(row["frame_index"])
            if frame_idx % max(stride, 1):
                continue
            state = row.get("observation.state")
            if state is None:
                state_array = np.empty(0, dtype=np.float64)
            else:
                state_array = torch.as_tensor(state).detach().cpu().double().reshape(-1).numpy()
            descriptors.append(
                {
                    "global_idx": int(global_idx),
                    "episode_idx": int(episode_idx),
                    "frame_idx": int(frame_idx),
                    "progress": float(frame_idx / episode_length),
                    "task_idx": _scalar(row.get("task_index")),
                    "subtask_idx": int(get_subtask_idx(dataset, global_idx)),
                    "state": state_array,
                }
            )
    return descriptors, by_episode


def _match_foreign_depth_donors(dataset, anchors: list[int], stride: int) -> dict[int, dict]:
    """Choose a real depth donor from another episode for every anchor.

    Matching is deliberately progressive so small validation sets still produce a result:
    same task+subtask, same task, then any different episode. Within the strongest
    non-empty tier, standardized state distance plus episode-progress distance chooses the
    donor deterministically.
    """
    descriptors, _ = _depth_match_descriptors(dataset, stride)
    by_global = {row["global_idx"]: row for row in descriptors}
    if len({row["episode_idx"] for row in descriptors}) < 2:
        return {}

    state_rows = [row["state"] for row in descriptors if row["state"].size]
    state_scale = None
    if state_rows and len({row.shape for row in state_rows}) == 1:
        state_scale = np.std(np.stack(state_rows), axis=0)
        state_scale = np.maximum(state_scale, 1e-6)

    matches: dict[int, dict] = {}
    for anchor_idx in anchors:
        anchor = by_global.get(int(anchor_idx))
        if anchor is None:
            continue
        foreign = [row for row in descriptors if row["episode_idx"] != anchor["episode_idx"]]
        tiers = [
            (
                "same_task_subtask",
                [
                    row
                    for row in foreign
                    if anchor["task_idx"] >= 0
                    and row["task_idx"] == anchor["task_idx"]
                    and anchor["subtask_idx"] >= 0
                    and row["subtask_idx"] == anchor["subtask_idx"]
                ],
            ),
            (
                "same_task",
                [row for row in foreign if anchor["task_idx"] >= 0 and row["task_idx"] == anchor["task_idx"]],
            ),
            ("different_episode", foreign),
        ]
        tier_name, candidates = next((name, rows) for name, rows in tiers if rows)

        def distance(candidate: dict, anchor: dict = anchor) -> tuple[float, int]:
            progress_distance = abs(candidate["progress"] - anchor["progress"])
            state_distance = 0.0
            if (
                state_scale is not None
                and anchor["state"].shape == state_scale.shape
                and candidate["state"].shape == state_scale.shape
            ):
                state_distance = float(
                    np.sqrt(np.mean(((candidate["state"] - anchor["state"]) / state_scale) ** 2))
                )
            return state_distance + progress_distance, candidate["global_idx"]

        donor = min(candidates, key=distance)
        state_distance = None
        if (
            state_scale is not None
            and anchor["state"].shape == state_scale.shape
            and donor["state"].shape == state_scale.shape
        ):
            state_distance = float(np.sqrt(np.mean(((donor["state"] - anchor["state"]) / state_scale) ** 2)))
        matches[int(anchor_idx)] = {
            "global_idx": int(donor["global_idx"]),
            "episode_idx": int(donor["episode_idx"]),
            "frame_idx": int(donor["frame_idx"]),
            "tier": tier_name,
            "progress_distance": abs(donor["progress"] - anchor["progress"]),
            "state_distance": state_distance,
        }
    return matches


def _stale_depth_index(
    global_idx: int,
    frame_idx: int,
    *,
    stale_frames: int,
    stride: int,
) -> tuple[int, int] | None:
    """A stride-valid earlier frame in the same contiguous episode, or ``None`` at start."""
    target_frame = max(int(frame_idx) - max(int(stale_frames), 1), 0)
    target_frame -= target_frame % max(int(stride), 1)
    if target_frame >= int(frame_idx):
        return None
    episode_start = int(global_idx) - int(frame_idx)
    return episode_start + target_frame, int(frame_idx) - target_frame


def _load_depth_window(dataset, cfg, global_idx: int, obs: dict, *, depth_obs_key: str) -> dict:
    """Load only current and historical depth for a stride-valid donor anchor."""
    from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png

    pointmap_cfg = cfg.policy.pointmap_config
    row = dataset.hf_dataset[global_idx]
    episode_idx = _scalar(row["episode_index"])
    frame_idx = _scalar(row["frame_index"])
    depth = load_depth_png(
        dataset.root,
        f"{pointmap_cfg.depth_key}.depth",
        episode_idx,
        frame_idx,
    )
    current = torch.from_numpy(depth.astype(np.float32)).reshape_as(obs[depth_obs_key])
    out = {depth_obs_key: current}

    history_keys = [key.removeprefix("history.") for key in obs if str(key).startswith("history.depth.")]
    memory_cfg = getattr(cfg.policy, "memory", None)
    if history_keys and memory_cfg is not None:
        out.update(assemble_frame_history(dataset, global_idx, memory_cfg, cfg.env.fps, history_keys))
    return out


def _write_manifest(output_dir: str, summary: dict) -> dict:
    """Describe the matched depth counterfactual and legacy stress tests."""
    trajectory_keys = [
        "trajectory_counterfactual_penalty.foreign_depth.path_mse",
        "trajectory_counterfactual_penalty.foreign_depth.shape_mse",
        "trajectory_counterfactual_penalty.foreign_depth.terminal_mse",
        "trajectory_counterfactual_penalty.foreign_depth.terminal_direction_loss",
    ]
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Depth Counterfactual",
        group="Depth",
        claim="Does scene-aligned depth improve actions relative to real but mismatched depth?",
        summary=summary,
        metrics=[
            Metric(
                "foreign_depth_penalty",
                "Matched foreign-depth penalty",
                good="high",
                fmt=5,
                baseline=0.0,
                warn=0.0,
                primary=True,
                note="$\\mathrm{mse}(\\text{foreign depth}) - \\mathrm{mse}(\\text{real depth})$ at identical flow noise. The donor comes from another episode and swaps current plus historical depth together. Positive means scene-aligned depth improves the demonstrated path.",
            ),
            Metric(
                "max_abs_delta.rgb+depth vs foreign_depth",
                "Action shift under foreign depth",
                good="none",
                fmt=4,
                primary=True,
                note="Mean $\\max|\\Delta a|$ in normalized space. Near zero means replacing depth content does not reach the action, regardless of its GT-MSE effect.",
            ),
            Metric(
                "stale_depth_penalty",
                "Same-episode stale-depth penalty",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Path MSE(stale depth) − path MSE(real depth). The full depth window comes from an earlier point in the same episode; start frames without an earlier donor are excluded.",
            ),
            Metric(
                "missing_depth_penalty",
                "Missing-depth stress penalty",
                good="none",
                fmt=5,
                baseline=0.0,
                note="Legacy all-null depth contrast. Training used depth dropout 0, so this measures sensor-loss robustness, not the value of depth content.",
            ),
            Metric(
                "fd_sensitivity.ratio",
                "Depth / wrist-RGB sensitivity",
                good="none",
                fmt=3,
                baseline=1.0,
                note="Finite-difference action sensitivity for a 1%-of-std raw-input perturbation. This remains a secondary local diagnostic.",
            ),
            Metric(
                "depth_rgb_rms_ratio",
                "Depth / RGB token RMS",
                good="none",
                fmt=3,
                baseline=1.0,
                note="Final depth prefix-token RMS divided by final RGB prefix-token RMS at the text-embedding seam.",
            ),
            Metric("n_frames", "Frames probed", good="none", fmt=0),
            Metric(
                trajectory_keys[0],
                "Foreign depth · path MSE",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Path MSE(foreign depth) − path MSE(real depth).",
            ),
            Metric(
                trajectory_keys[1],
                "Foreign depth · temporal shape",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Shape MSE(foreign depth) − shape MSE(real depth).",
            ),
            Metric(
                trajectory_keys[2],
                "Foreign depth · final position",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Final-position MSE(foreign depth) − final-position MSE(real depth).",
            ),
            Metric(
                trajectory_keys[3],
                "Foreign depth · final direction",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Final-direction loss(foreign depth) − loss(real depth).",
            ),
        ],
        panels=[
            Panel(
                "depth_modality.png",
                "Matched depth counterfactuals and local sensitivity",
                how=(
                    "**Left** — normalized MSE against the demonstrated chunk, all at "
                    "identical flow noise. ``rgb+depth`` is deployment. ``foreign_depth`` "
                    "swaps current plus historical depth with a matched real window from a "
                    "different episode; this is the primary counterfactual. ``stale_depth`` "
                    "uses an earlier same-episode window. ``rgb_only`` and ``neither`` route "
                    "through the all-null bank and are sensor-loss stress tests only.\n\n"
                    "**Right** — the existing finite-difference sensitivity to 1%-of-std raw "
                    "depth and wrist-RGB perturbations. It is secondary to the matched real-"
                    "depth comparison."
                ),
                primary=True,
            ),
            Panel(
                "depth_modality.json",
                "Summary, donor provenance, and per-frame measurements",
                how="Each row records donor episode/frame, matching tier and distance, stale lag, condition errors, and paired action shifts.",
            ),
        ],
        see_also=["attention_budget", "action_trace"],
        extra={
            "viewer": {
                "metric_groups": [
                    {
                        "title": "Matched foreign-depth penalty by trajectory criterion",
                        "keys": trajectory_keys,
                    }
                ]
            }
        },
    )


def _render(summary: dict, per_frame: list[dict], output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    conditions = summary["conditions"]
    mse = [summary["mse_norm"][c] for c in conditions]
    colors = {
        REAL: "#264653",
        FOREIGN: "#E76F51",
        STALE: "#F4A261",
        NULL: "#9D4EDD",
        DEPTH_ONLY: "#2A9D8F",
        NEITHER: "#6C757D",
    }
    axes[0].bar(np.arange(len(conditions)), mse, color=[colors[c] for c in conditions])
    axes[0].set_xticks(np.arange(len(conditions)), conditions, rotation=20, ha="right")
    axes[0].set_ylabel("normalized MSE vs GT")
    axes[0].set_title(
        f"Depth counterfactuals — foreign penalty {summary['foreign_depth_penalty']:+.5f}\n"
        "(mse(foreign depth) − mse(real depth); positive ⇒ aligned depth helps)",
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

    stride = probe_image_stride(cfg)
    explicit = [int(s) for s in str(getattr(cfg, "frame_indices", "") or "").split(",") if s.strip()]
    if explicit:
        frame_indices = [idx - idx % stride for idx in explicit]
        if frame_indices != explicit:
            logging.warning(
                f"[depth_modality] snapped frame indices onto the depth grid (stride {stride}): "
                f"{explicit} -> {frame_indices}"
            )
    else:
        frame_indices = [
            g
            for _, _, g in sample_episodes_evenly(
                dataset,
                int(getattr(p, "depth_modality_n_frames", None) or p.n_frames_per_episode),
                p.max_episodes,
                p.random_seed,
                stride,
            )
        ]
    if not frame_indices:
        logging.warning("[depth_modality] no frames selected.")
        return

    donor_matches = _match_foreign_depth_donors(dataset, frame_indices, stride)
    if not donor_matches:
        logging.warning("[depth_modality] matched foreign depth requires at least two episodes; skipping.")
        return
    missing_matches = sorted(set(frame_indices) - set(donor_matches))
    if missing_matches:
        logging.warning(
            f"[depth_modality] no foreign donor for {len(missing_matches)} anchors; dropping them."
        )
        frame_indices = [idx for idx in frame_indices if idx in donor_matches]

    stale_seconds = float(getattr(p, "depth_stale_seconds", 2.0))
    stale_frames = max(int(round(stale_seconds * float(cfg.env.fps))), 1)
    adapter._set_probe_cuda_graph_enabled(False)

    def predict(obs, frame, *, rgb_on: bool = True) -> torch.Tensor:
        batch = adapter._make_batch(obs, frame["task"], subtask=frame["subtask"], metadata=frame["metadata"])
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
            policy.predict_action_chunk(batch, inference_action_mode="continuous", generator=generator)
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
            record_scale("depth_late_early_rms_ratio", late.item() / max(early.item(), 1e-12))
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
                record_scale("depth_rgb_rms_ratio", depth_rms.item() / max(rgb_value, 1e-12))

    try:
        for global_idx in frame_indices:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            obs = frame["obs"]
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"]).float()
            hold_raw = (
                frame["state"][: frame["gt_actions"].shape[-1]]
                .unsqueeze(0)
                .repeat(frame["gt_actions"].shape[0], 1)
            )
            hold_norm = adapter.normalize_gt_actions(hold_raw, frame["state"]).float()

            donor = donor_matches[int(global_idx)]
            foreign_window = _load_depth_window(
                dataset, cfg, donor["global_idx"], obs, depth_obs_key=depth_obs_key
            )
            condition_obs = {
                REAL: obs,
                FOREIGN: _replace_depth_window(obs, foreign_window, depth_obs_key=depth_obs_key),
                NULL: _drop_depth(obs, depth_obs_key=depth_obs_key),
                DEPTH_ONLY: obs,
                NEITHER: _drop_depth(obs, depth_obs_key=depth_obs_key),
            }
            stale = _stale_depth_index(
                global_idx,
                frame["frame_idx"],
                stale_frames=stale_frames,
                stride=stride,
            )
            if stale is not None:
                stale_idx, stale_lag_frames = stale
                stale_window = _load_depth_window(dataset, cfg, stale_idx, obs, depth_obs_key=depth_obs_key)
                condition_obs[STALE] = _replace_depth_window(obs, stale_window, depth_obs_key=depth_obs_key)

            actions: dict[str, torch.Tensor] = {}
            for condition in CONDITIONS:
                if condition not in condition_obs:
                    continue
                actions[condition] = predict(
                    condition_obs[condition],
                    frame,
                    rgb_on=condition not in (DEPTH_ONLY, NEITHER),
                )
                # Capture only the real RGB+depth condition. Reading after a donor or
                # null condition would report the treatment's token scale instead.
                if condition == REAL:
                    capture_rgb_depth_scale()

            if not n_context:
                probe_batch = adapter._make_batch(
                    obs, frame["task"], subtask=frame["subtask"], metadata=frame["metadata"]
                )
                n_context = int(probe_batch["input_ids"].shape[-1])

            horizon = min(actions[REAL].shape[0], gt_norm.shape[0])
            row = {
                "global_idx": int(global_idx),
                "episode_idx": int(frame["episode_idx"]),
                "frame_idx": int(frame["frame_idx"]),
                "foreign_donor": donor,
                "stale_donor_global_idx": None if stale is None else int(stale_idx),
                "stale_lag_frames": None if stale is None else int(stale_lag_frames),
                "mse_norm": {},
                "trajectory": {},
                "max_abs_delta": {},
            }
            logging.info(
                f"frame {global_idx}: foreign donor={donor['global_idx']} "
                f"ep={donor['episode_idx']} tier={donor['tier']}"
            )
            for condition in CONDITIONS:
                if condition not in actions:
                    continue
                mse = torch.nn.functional.mse_loss(actions[condition][:horizon], gt_norm[:horizon]).item()
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
                        by_condition = trajectory_by_condition.setdefault(key, {c: [] for c in CONDITIONS})
                        by_condition[condition].append(value)
                logging.info(f"  mse_norm[{condition:>9s}] = {mse:.5f}")
            for left, right in CONDITION_PAIRS:
                if left not in actions or right not in actions:
                    continue
                delta = (actions[left] - actions[right]).abs().max().item()
                pairwise_deltas[(left, right)].append(delta)
                row["max_abs_delta"][f"{left} vs {right}"] = delta
                logging.info(f"  max|Δ| {left} vs {right} = {delta:.4e}")

            # Finite-difference sensitivity: same seed, 1%-of-std input perturbation.
            eps = float(getattr(cfg, "fd_epsilon_rel", 0.01))
            depth_raw = obs[depth_obs_key]
            pert = torch.randn_like(depth_raw) * depth_raw.float().std() * eps
            sens_depth = (
                (predict({**obs, depth_obs_key: depth_raw + pert}, frame) - actions[REAL]).norm().item()
            )
            rgb_raw = obs[rgb_obs_key].float()
            pert = torch.randn_like(rgb_raw) * rgb_raw.std() * eps
            sens_rgb = (
                (
                    predict({**obs, rgb_obs_key: (rgb_raw + pert).to(obs[rgb_obs_key].dtype)}, frame)
                    - actions[REAL]
                )
                .norm()
                .item()
            )
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
            condition: (float(np.mean(values)) if values else None)
            for condition, values in by_condition.items()
        }
        for key, by_condition in trajectory_by_condition.items()
    }
    active_conditions = [condition for condition in CONDITIONS if mse_by_condition[condition]]
    trajectory_counterfactual_penalty = {
        condition: {
            key: (
                None
                if values[REAL] is None or values[condition] is None
                else values[condition] - values[REAL]
            )
            for key, values in trajectory.items()
        }
        for condition in (FOREIGN, STALE, NULL)
    }
    match_tier_counts = dict(sorted(Counter(row["foreign_donor"]["tier"] for row in per_frame).items()))
    state_distances = [
        row["foreign_donor"]["state_distance"]
        for row in per_frame
        if row["foreign_donor"]["state_distance"] is not None
    ]
    summary = {
        "n_frames": n,
        "frame_indices": [int(i) for i in frame_indices],
        "conditions": active_conditions,
        "mse_norm": {
            condition: float(np.mean(mse_by_condition[condition])) for condition in active_conditions
        },
        "max_abs_delta": {
            f"{left} vs {right}": float(np.mean(values))
            for (left, right), values in pairwise_deltas.items()
            if values
        },
        "foreign_depth_penalty": trajectory_counterfactual_penalty[FOREIGN]["path_mse"],
        "stale_depth_penalty": trajectory_counterfactual_penalty[STALE]["path_mse"],
        "missing_depth_penalty": trajectory_counterfactual_penalty[NULL]["path_mse"],
        "trajectory": trajectory,
        "trajectory_counterfactual_penalty": trajectory_counterfactual_penalty,
        "foreign_matching": {
            "tier_counts": match_tier_counts,
            "mean_progress_distance": float(
                np.mean([row["foreign_donor"]["progress_distance"] for row in per_frame])
            ),
            "mean_state_distance": (float(np.mean(state_distances)) if state_distances else None),
        },
        "stale_seconds_requested": stale_seconds,
        "n_stale_frames": len(mse_by_condition[STALE]),
        "fd_sensitivity": {
            "depth": mean_depth,
            "rgb": mean_rgb,
            "ratio": mean_depth / max(mean_rgb, 1e-12),
        },
    }
    summary["n_depth_tokens"] = int(policy.config.pointmap_config.num_pooled_tokens)
    summary["n_context_tokens"] = n_context
    summary.update({name: sum(values) / len(values) for name, values in scale_samples.items() if values})

    with open(os.path.join(output_dir, "depth_modality.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)

    # write_index drops panels whose file is not on disk yet, so render first or the
    # probe's only figure never reaches the viewer.
    _render(summary, per_frame, os.path.join(output_dir, "depth_modality.png"))
    _write_manifest(output_dir, summary)

    logging.info("── summary over frames ──")
    for condition in summary["conditions"]:
        logging.info(f"mse_norm[{condition:>13s}] mean = {summary['mse_norm'][condition]:.5f}")
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
        f"foreign-depth penalty = {summary['foreign_depth_penalty']:+.5f} "
        f"(matched real donor depth vs deployment depth)"
    )
    if summary["stale_depth_penalty"] is not None:
        logging.info(
            f"stale-depth penalty = {summary['stale_depth_penalty']:+.5f} "
            f"({summary['n_stale_frames']} frames)"
        )
    logging.info(
        f"missing-depth penalty = {summary['missing_depth_penalty']:+.5f} "
        "(unsupported sensor-loss stress test)"
    )
    logging.info(f"mean fd sensitivity: depth={mean_depth:.4e} rgb={mean_rgb:.4e}")
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

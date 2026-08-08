r"""MEM history-influence probe (04_memory.md §2.4).

Does short-term memory move the policy's action chunk, and does the move help?

Every sampled frame is predicted four times from the same lookback window with the same
fixed flow noise, so the only thing that varies is which history channels are packed:

  condition   image history   state history
  full        kept            kept
  none        dropped         dropped
  images      kept            dropped
  states      dropped         kept

*Image history* is the stack of past frames the MEM video encoder attends over in its
temporal layers. *State history* is the proprioceptive lookback — one embedded position
per past timestep. Both are named the same way at every stage; there is no third thing.
A channel is dropped by removing its ``history.*`` keys from the observation before
packing, so ``none`` is the real training-time no-history prompt and not a full prompt
holding empty placeholders. Depth history travels through the point-map encoder and no
condition here touches it — ``depth_modality_probe`` covers that path.

``none`` has no bar in any figure: it is the origin the other three are measured from.
Write $a^{(c)}$ for the normalized chunk under condition $c$, $a^{\star}$ for the
demonstrated chunk, and $T$, $D$ for chunk steps and action dimensions. Influence is

$$\mathrm{RMSE}(c)=\sqrt{\frac{1}{TD}\sum_{t,d}\left(a^{(c)}_{t,d}-a^{(\mathrm{none})}_{t,d}\right)^2}$$

reported next to its worst-case counterpart $\max_{t,d}|a^{(c)}_{t,d}-a^{(\mathrm{none})}_{t,d}|$,
which separates a channel that shifts the whole chunk slightly from one that moves a
single timestep hard. Usefulness is the mean squared error the channel removes against
the demonstration,

$$\Delta\mathrm{MSE}(c)=\frac{1}{TD}\sum_{t,d}\left(a^{(\mathrm{none})}_{t,d}-a^{\star}_{t,d}\right)^2-\frac{1}{TD}\sum_{t,d}\left(a^{(c)}_{t,d}-a^{\star}_{t,d}\right)^2 .$$

Influence says the channel reaches the output; usefulness says it reached it in the
right direction. Positive $\Delta\mathrm{MSE}$ is memory helping. Negative is worse than
an unused channel: the model leans on the channel and it points the wrong way. When
influence is near zero the usefulness number is measuring nothing rather than reporting
good news — there is nothing to be right or wrong about.

``images`` and ``states`` do not sum to ``full``. Read them for which channel carries
the effect, never as an additive decomposition.

Registered probe: enable with ``probe_parameters.enable_mem_history_influence``.
"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.probes.manifest import Panel, write_index
from lerobot.probes.utils import (
    as_image,
    assemble_frame_history,
    get_frame_data,
    joint_names_for_dim,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)
from lerobot.utils.constants import OBS_STATE


_CONDITION_STYLE = {
    "full": ("#2A9D8F", "-"),
    "none": ("#E63946", "--"),
    "images": ("#277DA1", "-."),
    "states": ("#9B5DE5", ":"),
}

_REBOT_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper",
]


def _variant_observation(obs: dict, images_on: bool, states_on: bool) -> dict:
    """Remove disabled history channels before packing the prompt/model inputs.

    Depth history rides the point-map encoder rather than the MEM video encoder, so
    it is left alone by every condition here: this probe isolates the MEM channels,
    and `depth_modality_probe` covers the depth path. Keeping it constant means the
    four conditions differ only in what the caller intends.
    """
    out = dict(obs)
    for key in list(out):
        if not key.startswith("history.") or key.startswith("history.depth."):
            continue
        remove_state = key == f"history.{OBS_STATE}" and not states_on
        remove_images = "images" in key and not images_on
        if remove_state or remove_images:
            out.pop(key)
    return out


_EXAMPLE_HOW = {
    "best": "A frame where the full history helped most: the ``full`` trace should sit closer to GT than ``none``.",
    "worst": "A frame where the full history hurt most. A channel that is used but pointed the wrong way looks like this everywhere, not just here.",
    "strong": "A frame where history moved the chunk most, regardless of whether the move was useful — the influence side of the measurement in isolation.",
}


def _write_manifest(output_dir: str, summary: dict, examples: list[tuple[str, str]]) -> dict:
    """Describe the history channels' effect on the action chunk to the viewer."""
    panels = [
        Panel(
            "influence.png",
            "Influence, usefulness, and the per-frame relationship between them",
            how="Left: how far each history condition moves the chunk away from the no-history prediction. Middle: how much GT MSE that movement removes — the bar that matters, and the only one that can come out negative. Right: one point per frame, so a positive mean built out of a few large wins is distinguishable from a consistent small gain.",
            primary=True,
        )
    ]
    panels += [
        Panel(f"examples/{filename}", f"{label} — history conditions against GT", how=_EXAMPLE_HOW[label])
        for label, filename in examples
    ]
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="MEM History Influence",
        group="History",
        claim="Does short-term history move the action chunk, and does the move help?",
        summary=summary,
        # Every number the probe computes is already a bar in influence.png, where it
        # carries its distribution instead of collapsing to one row. The values stay in
        # influence.json.
        metrics=[],
        panels=panels,
        see_also=["mem_temporal_attention", "attention_budget", "action_trace"],
    )


def _select_examples(rows: list[dict], per_category: int = 2) -> list[tuple[str, int]]:
    """Pick unique frames where full history helped, hurt, or changed actions most."""
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    rankings = (
        ("best", "full_gt_mse_improvement", True),
        ("worst", "full_gt_mse_improvement", False),
        ("strong", "full_rmse", True),
    )
    for label, metric, descending in rankings:
        ranked = sorted(range(len(rows)), key=lambda i: rows[i][metric], reverse=descending)
        count = 0
        for index in ranked:
            if index in used:
                continue
            selected.append((label, index))
            used.add(index)
            count += 1
            if count >= per_category:
                break
    return selected


def _render_example(dataset, memory_cfg, fps: float, diagnostic: dict, label: str, output_path: str) -> None:
    """Show what memory saw and how each condition changed every predicted joint."""
    global_idx = diagnostic["global_idx"]
    obs, _gt, _state, _subtask, _task, ep_idx, frame_idx = get_frame_data(
        dataset, global_idx, diagnostic["gt"].shape[0]
    )
    image_keys = sorted(k for k in obs if k.startswith("observation.images."))
    history = assemble_frame_history(dataset, global_idx, memory_cfg, fps, image_keys)

    n_cols = 4
    n_image_rows = max(1, len(image_keys))
    n_action_rows = (diagnostic["gt"].shape[-1] + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(18, 3.0 * n_image_rows + 2.7 * n_action_rows))
    grid = fig.add_gridspec(n_image_rows + n_action_rows, n_cols, hspace=0.45, wspace=0.28)

    for row_idx, key in enumerate(image_keys):
        history_tensor = history.get(f"history.{key}")
        if history_tensor is not None:
            history_tensor = history_tensor.squeeze(0)
            slots = [0, len(history_tensor) // 2, len(history_tensor) - 1]
            for col_idx, slot in enumerate(slots):
                ax = fig.add_subplot(grid[row_idx, col_idx])
                ax.imshow(as_image(history_tensor[slot]))
                ax.set_title(f"{key.split('.')[-1]} history: {('oldest', 'middle', 'newest')[col_idx]}")
                ax.axis("off")
        ax = fig.add_subplot(grid[row_idx, 3])
        ax.imshow(as_image(obs[key]))
        ax.set_title(f"{key.split('.')[-1]} current")
        ax.axis("off")

    steps = np.arange(diagnostic["gt"].shape[0])
    action_dim = diagnostic["gt"].shape[-1]
    joint_names = joint_names_for_dim(action_dim)
    for joint in range(action_dim):
        row = n_image_rows + joint // n_cols
        col = joint % n_cols
        ax = fig.add_subplot(grid[row, col])
        ax.plot(steps, diagnostic["gt"][:, joint], color="black", linewidth=2.0, label="GT")
        for condition, actions in diagnostic["acts"].items():
            color, linestyle = _CONDITION_STYLE[condition]
            ax.plot(
                steps,
                actions[:, joint],
                color=color,
                linestyle=linestyle,
                linewidth=1.25,
                label=condition,
            )
        ax.set_title(f"{joint_names[joint]} (normalized)")
        ax.grid(True, alpha=0.25, linestyle=":")
        if joint == 0:
            ax.legend(fontsize=8, ncol=3)
    for unused in range(action_dim, n_action_rows * n_cols):
        fig.add_subplot(grid[n_image_rows + unused // n_cols, unused % n_cols]).axis("off")

    row = diagnostic["row"]
    fig.suptitle(
        f"{label.upper()} memory example — episode {ep_idx}, frame {frame_idx}  |  "
        f"full influence RMSE={row['full_rmse']:.4f}  |  "
        f"GT-MSE improvement={row['full_gt_mse_improvement']:+.4f}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def run(adapter, dataset, cfg, output_dir: str) -> None:
    memory_cfg = getattr(cfg.policy, "memory", None)
    if memory_cfg is None or not memory_cfg.history_keys or memory_cfg.history_num_samples <= 0:
        logging.info("[mem_history_influence] no short-term memory configured — skipping.")
        return
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[mem_history_influence] needs continuous flow actions — skipping.")
        return

    makedirs(output_dir)
    p = cfg.probe_parameters
    device = adapter.device
    fps = cfg.env.fps

    adapter._set_probe_cuda_graph_enabled(False)  # varying mask per condition; keep eager
    samples = sample_episodes_evenly(
        dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )

    conditions = {"full": (True, True), "none": (False, False), "images": (True, False), "states": (False, True)}
    rows: list[dict] = []
    diagnostics: list[dict] = []
    for _ep, _fr, global_idx in samples:
        frame = probe_frame_inputs(dataset, cfg, global_idx, int(cfg.policy.chunk_size))
        obs, gt_actions, state = frame["obs"], frame["gt_actions"], frame["state"]
        subtask, task_str, metadata = frame["subtask"], frame["task"], frame["metadata"]
        full_batch = adapter._make_batch(obs, task_str, subtask=subtask, metadata=metadata)
        if "history_images_mask" not in full_batch and "history_state_values" not in full_batch:
            logging.warning("[mem_history_influence] batch carries no history tensors — skipping frame.")
            continue

        gt_norm = adapter.normalize_gt_actions(gt_actions, state)
        acts: dict[str, torch.Tensor] = {}
        for name, (img_on, st_on) in conditions.items():
            gen = torch.Generator(device=device)
            gen.manual_seed(0)  # same flow noise across conditions
            acts[name] = adapter.predict_action_chunk(
                _variant_observation(obs, img_on, st_on),
                task_str,
                state=state,
                subtask=subtask,
                metadata=metadata,
                generator=gen,
            )[1]
        base = acts["none"]
        gt_mse = {name: float((act - gt_norm).pow(2).mean()) for name, act in acts.items()}
        row = {
            "global_idx": int(global_idx),
            "episode_idx": int(_ep),
            "frame_idx": int(_fr),
            "full_maxabs": float((acts["full"] - base).abs().max()),
            "full_rmse": float((acts["full"] - base).pow(2).mean().sqrt()),
            "images_maxabs": float((acts["images"] - base).abs().max()),
            "images_rmse": float((acts["images"] - base).pow(2).mean().sqrt()),
            "states_maxabs": float((acts["states"] - base).abs().max()),
            "states_rmse": float((acts["states"] - base).pow(2).mean().sqrt()),
        }
        for name in conditions:
            row[f"{name}_gt_mse"] = gt_mse[name]
            if name != "none":
                row[f"{name}_gt_mse_improvement"] = gt_mse["none"] - gt_mse[name]
        rows.append(row)
        diagnostics.append(
            {
                "global_idx": int(global_idx),
                "gt": gt_norm,
                "acts": acts,
                "row": row,
            }
        )

    adapter._restore_probe_cuda_graph_enabled()
    if not rows:
        logging.warning("[mem_history_influence] no frames produced measurements.")
        return

    metrics = [
        "full_maxabs", "full_rmse", "images_maxabs", "images_rmse", "states_maxabs", "states_rmse",
        "full_gt_mse", "images_gt_mse", "states_gt_mse", "none_gt_mse",
        "full_gt_mse_improvement", "images_gt_mse_improvement", "states_gt_mse_improvement",
    ]
    summary = {metric: float(np.mean([r[metric] for r in rows])) for metric in metrics}
    summary["n_frames"] = len(rows)
    with open(os.path.join(output_dir, "influence.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    channels = ["full", "images", "states"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(len(channels))
    axes[0].bar(x - 0.2, [summary[f"{c}_rmse"] for c in channels], width=0.4, label="RMSE")
    axes[0].bar(x + 0.2, [summary[f"{c}_maxabs"] for c in channels], width=0.4, label="max|Δ|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels)
    axes[0].set_ylabel("normalized action Δ vs no-history")
    axes[0].set_title("Influence")
    axes[0].legend()
    axes[1].bar(x, [summary[f"{c}_gt_mse_improvement"] for c in channels])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels)
    axes[1].set_ylabel("GT MSE improvement vs no-history")
    axes[1].set_title("Usefulness (positive is better)")
    for condition in channels:
        axes[2].scatter(
            [row[f"{condition}_rmse"] for row in rows],
            [row[f"{condition}_gt_mse_improvement"] for row in rows],
            s=18,
            alpha=0.55,
            label=condition,
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("influence RMSE vs no-history")
    axes[2].set_ylabel("GT MSE improvement")
    axes[2].set_title("Per-frame effect")
    axes[2].legend()
    fig.suptitle(f"MEM history effect on the action chunk (n={len(rows)})")
    fig.savefig(os.path.join(output_dir, "influence.png"), bbox_inches="tight", dpi=100)
    plt.close(fig)

    examples_dir = os.path.join(output_dir, "examples")
    makedirs(examples_dir)
    examples: list[tuple[str, str]] = []
    for label, index in _select_examples(rows):
        diagnostic = diagnostics[index]
        row = rows[index]
        filename = f"{label}_ep{row['episode_idx']:04d}_fr{row['frame_idx']:06d}.png"
        _render_example(
            dataset,
            memory_cfg,
            fps,
            diagnostic,
            label,
            os.path.join(examples_dir, filename),
        )
        examples.append((label, filename))

    _write_manifest(output_dir, summary, examples)
    logging.info(
        f"[mem_history_influence] n={len(rows)}  full RMSE={summary['full_rmse']:.4f}  "
        f"full GT-MSE improvement={summary['full_gt_mse_improvement']:+.4f}  "
        f"images={summary['images_gt_mse_improvement']:+.4f}  "
        f"states={summary['states_gt_mse_improvement']:+.4f}"
    )

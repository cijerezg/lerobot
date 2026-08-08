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

Every number here is a mean over frames, and on a checkpoint where memory helps half the
frames and hurts the other half that mean is near zero whatever the effect size. This
probe answers *which channel*; `mem_history_regime` answers *which frames*, with the
per-frame distribution and the nulls that say whether either sign is real. Read this one
first only to find out where to look.

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
    history_offsets,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)
from lerobot.utils.constants import OBS_STATE


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


def _provenance(rows: list[dict], dataset, cfg, memory_cfg, conditions: dict) -> dict:
    """Which frames of which dataset the bars average, and what one forward was.

    Derived from the measured rows rather than from config, so it describes the frames
    that actually produced numbers after stride snapping and the episode budget dropped
    whatever they dropped.
    """
    p = cfg.probe_parameters
    fps = float(cfg.env.fps)
    chunk = int(cfg.policy.chunk_size)
    offsets = history_offsets(memory_cfg, fps)
    keys = [f"``history.{k}``" for k in memory_cfg.history_keys]
    channels = (
        "``images`` = " + ", ".join(k for k in keys if "images" in k)
        + "; ``states`` = " + ", ".join(k for k in keys if k.endswith(f"{OBS_STATE}``"))
    )
    depth_keys = ", ".join(k for k in keys if "``history.depth." in k)
    if depth_keys:
        channels += f"; kept in every condition = {depth_keys}"
    episodes = sorted({r["episode_idx"] for r in rows})
    return {
        "val": {
            "n_frames": len(rows),
            "n_episodes": len(episodes),
            "sources": [
                {
                    "name": str(getattr(dataset, "repo_id", "val")),
                    "root": str(getattr(dataset, "root", "")),
                    "episodes": episodes,
                    "n_episodes": len(episodes),
                    "n_frames": len(rows),
                }
            ],
        },
        "frames_per_episode": int(p.n_frames_per_episode),
        "episode_budget": p.max_episodes,
        "image_stride": probe_image_stride(cfg),
        "chunk_size": chunk,
        "batch_size": 1,
        "forwards": len(conditions) * len(rows),
        "details": [
            [
                "Per frame",
                f"{len(conditions)} forwards — " + ", ".join(f"``{name}``" for name in conditions),
            ],
            [
                "History window",
                f"{memory_cfg.history_num_samples} past frames over "
                f"{memory_cfg.history_window_seconds:g} s at {fps:g} fps — offsets "
                + ", ".join(str(o) for o in offsets)
                + " frames back, oldest first, repeat-padded before the episode start",
            ],
            ["Channels toggled", channels],
            [
                "Demonstration",
                r"$a^{\star}$ is the recorded action at the sampled frame and the "
                f"{chunk - 1} that follow it, normalized the same way as the prediction",
            ],
        ],
        "sampling": (
            "Frames evenly spaced across each episode, snapped onto the image/depth stride "
            "grid; episodes drawn by a seeded subset when the budget is smaller than the "
            f"split (seed {int(p.random_seed)}). Every number is a within-frame difference "
            "between two conditions run on the same observation — nothing is compared across "
            "frames, so uneven sampling cannot produce an effect. A chunk that would run past "
            "the end of its episode is repeat-padded with the last recorded action, which "
            f"makes the demonstration partly constant for frames within {chunk} of the end and "
            "shrinks what any channel can improve there."
        ),
        "regime": (
            "one frame per forward, batch size 1; "
            f"{int(getattr(cfg.policy, 'num_inference_steps', 0))} flow denoising steps with the "
            "same seeded noise in all conditions; the batch carries no action target, so "
            "training-time prompt dropout is not armed"
        ),
    }


# Panel captions carry the algebra because the bars are differences of differences: a
# reader who guesses at what is subtracted from what reads the middle panel backwards.
_INFLUENCE_HOW = r"""Every bar is a mean over the sampled held-out frames listed in
*Data behind these numbers*. Write
$a^{(c)}$ for the normalized chunk predicted under condition $c$, $a^{(\mathrm{none})}$ for
the same frame with both history channels dropped, and $a^{\star}$ for the demonstrated
chunk over the same $T$ steps and $D$ joints.

**Left — influence.** How far the channel moves the chunk away from the no-history
prediction, in normalized action units:

  $$\mathrm{RMSE}(c)=\sqrt{\frac{1}{TD}\sum_{t,d}\left(a^{(c)}_{t,d}-a^{(\mathrm{none})}_{t,d}\right)^{2}}
  \qquad \max_{t,d}\left|a^{(c)}_{t,d}-a^{(\mathrm{none})}_{t,d}\right|$$

The two bars separate a channel that shifts the whole chunk slightly (RMSE and max close)
from one that moves a single timestep hard (max far above RMSE). Nothing here involves the
demonstration, so neither bar can say whether the movement was an improvement.

**Middle — usefulness.** The squared error against the demonstration that the channel
removes, written as the difference of two MSEs with the no-history one first:

  $$\Delta\mathrm{MSE}(c)=\frac{1}{TD}\sum_{t,d}\left(a^{(\mathrm{none})}_{t,d}-a^{\star}_{t,d}\right)^{2}
  -\frac{1}{TD}\sum_{t,d}\left(a^{(c)}_{t,d}-a^{\star}_{t,d}\right)^{2}$$

Positive means the chunk moved toward the demonstration and the channel helped; negative
means the model leaned on the channel and it pointed away. The plotted bar is the mean of
$\Delta\mathrm{MSE}(c)$ over the frames, so it is in squared normalized action units and is
not comparable to the RMSE bars on the left.

Neither bar says whether the mean it plots describes any frame. A channel that helps half
the frames and hurts the other half lands on the same bar as a channel nothing happens
under — ``mem_history_regime`` is where that distribution, its nulls and its per-frame
examples live."""


def _write_manifest(output_dir: str, summary: dict) -> dict:
    """Describe the history channels' effect on the action chunk to the viewer."""
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
        panels=[
            Panel(
                "influence.png",
                "Influence and usefulness, per history channel",
                how=_INFLUENCE_HOW,
                primary=True,
                refs=["mem_history_regime"],
            )
        ],
        see_also=["mem_history_regime", "mem_temporal_attention", "attention_budget", "action_trace"],
        extra={"provenance": summary.get("data", {})},
    )


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

    adapter._set_probe_cuda_graph_enabled(False)  # varying mask per condition; keep eager
    samples = sample_episodes_evenly(
        dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )

    conditions = {"full": (True, True), "none": (False, False), "images": (True, False), "states": (False, True)}
    rows: list[dict] = []
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
    summary["data"] = _provenance(rows, dataset, cfg, memory_cfg, conditions)
    with open(os.path.join(output_dir, "influence.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    channels = ["full", "images", "states"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(channels))
    axes[0].bar(x - 0.2, [summary[f"{c}_rmse"] for c in channels], width=0.4, label="RMSE")
    axes[0].bar(x + 0.2, [summary[f"{c}_maxabs"] for c in channels], width=0.4, label="max|Δ|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels)
    axes[0].set_ylabel("normalized action Δ vs no-history")
    axes[0].set_title(r"Influence: mean $\mathrm{RMSE}(c)$ and mean $\max_{t,d}|\Delta|$")
    axes[0].legend()
    axes[1].bar(x, [summary[f"{c}_gt_mse_improvement"] for c in channels])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels)
    axes[1].set_ylabel(r"$\mathrm{MSE}(\mathrm{none}) - \mathrm{MSE}(c)$ vs GT (normalized$^2$)")
    axes[1].set_title(r"Usefulness: mean $\Delta\mathrm{MSE}(c)$ (positive is better)")
    provenance = summary["data"]
    fig.suptitle(
        f"MEM history effect on the action chunk — {len(rows)} frames from "
        f"{provenance['val']['n_episodes']} held-out episodes, "
        f"{int(cfg.policy.chunk_size)}-step chunks"
    )
    fig.savefig(os.path.join(output_dir, "influence.png"), bbox_inches="tight", dpi=100)
    plt.close(fig)

    _write_manifest(output_dir, summary)
    logging.info(
        f"[mem_history_influence] n={len(rows)}  full RMSE={summary['full_rmse']:.4f}  "
        f"full GT-MSE improvement={summary['full_gt_mse_improvement']:+.4f}  "
        f"images={summary['images_gt_mse_improvement']:+.4f}  "
        f"states={summary['states_gt_mse_improvement']:+.4f}"
    )

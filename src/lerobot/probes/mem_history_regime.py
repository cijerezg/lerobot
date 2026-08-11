r"""When does short-term memory help, and when does it hurt? (04_memory §2.4)

`mem_history_influence` averages the usefulness of each history channel over frames and
reports one bar per channel. That mean is a sum over two populations rather than a
description of either: on v6/1600 it read $+0.00067$, built out of 47% of frames that
improved and 53% that got worse. This probe is the per-frame view the mean hides — how
many frames memory helps, how many it hurts, and whether either sign is real.

Four forwards per sampled frame. The present observation is identical in all four; only
the lookback window and the flow seed move:

  condition   history in the prompt                    flow seed
  none        dropped                                  0
  full        the real window at this frame            0
  stale       that window as it stood $W$ s earlier    0
  none'       dropped                                  1

$W$ is ``memory.history_window_seconds`` — the full span of the lookback — so every slot
of the ``stale`` window is strictly older than the slot it replaces and none of them is
the frame it should be. Depth history rides the point-map encoder and is left untouched
by all four, exactly as in `mem_history_influence`; `depth_modality` covers that path.

Write $a^{(c)}$ for the normalized chunk under condition $c$, $a^{\star}$ for the
demonstrated chunk, and $T$, $D$ for chunk steps and action dimensions. Everything below
is built from

$$\mathrm{MSE}(c)=\frac{1}{TD}\sum_{t,d}\left(a^{(c)}_{t,d}-a^{\star}_{t,d}\right)^{2},
\qquad \Delta(c)=\mathrm{MSE}(\mathrm{none})-\mathrm{MSE}(c) .$$

**Is the sign real?** $\Delta(\mathrm{full})$ is positive on a frame where memory helped,
but so is half of any symmetric noise. ``none'`` gives the scale of a $\Delta$ that means
nothing: it changes the flow seed and nothing else, so

$$\tau=Q_{0.9}\left(\left|\Delta(\mathrm{none}')\right|\right)$$

is the width within which a frame's $\Delta$ is not distinguishable from redrawing the
sampler. Frames are labelled *helped* at $\Delta(\mathrm{full})>\tau$, *hurt* at
$\Delta(\mathrm{full})<-\tau$, *indistinguishable* between. By construction the null puts
about 5% of frames on each side of $\tau$, so those are the fractions to beat.

**Is it the content or just the perturbation?** A prompt that changes at all changes the
chunk, and a changed chunk lands closer to the demonstration half the time. ``stale`` is
the same-sized perturbation carrying the wrong content, so

$$G=\mathrm{MSE}(\mathrm{stale})-\mathrm{MSE}(\mathrm{full})=\Delta(\mathrm{full})-\Delta(\mathrm{stale})$$

is per-frame evidence that *this* window, and not merely *a* window, is what the model
used. $G$ is paired within a frame, so the mean over frames divided by its standard error
is the whole test: $z<2$ is no evidence the content matters, however large
$\Delta(\mathrm{full})$ looks. Read $z$ before reading anything else here.

*Which* frames memory helps on is not answered here. A Spearman ranking of
$\Delta(\mathrm{full})$ against dataset-side covariates (episode phase, lookback travel,
scene change, chunk travel, gripper swing, baseline error, repeat-padded slots) was removed
2026-08-10: at the sample sizes this probe runs at, every $\rho$ sat inside its own
$\pm1.96/\sqrt{n-3}$ band or barely outside it, the covariates are not independent of each
other, and the bin panels resolved differences an order of magnitude below $\tau$. Nothing
was ever concluded from it. The examples below are the per-frame view that remains.

Registered probe: enable with ``probe_parameters.enable_mem_history_regime``.
"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    as_image,
    assemble_frame_history,
    get_frame_data,
    joint_names_for_dim,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
    trajectory_error_components,
)

_CONDITION_STYLE = {
    "full": ("#2A9D8F", "-"),
    "none": ("#E63946", "--"),
    "stale": ("#F4A261", "-."),
}

_VERDICT_COLOR = {"helped": "#2A9D8F", "hurt": "#E63946", "indistinguishable": "#9AA0A6"}


def _variant_observation(obs: dict, history_on: bool) -> dict:
    """Drop the MEM history channels, keeping depth history and the present frame.

    Same rule as `mem_history_influence._variant_observation` with both channels
    switched together: this probe asks about memory as one thing, and the parent
    probe is where the channels are separated.
    """
    if history_on:
        return obs
    return {
        k: v
        for k, v in obs.items()
        if not k.startswith("history.") or k.startswith("history.depth.")
    }


def _stale_observation(obs: dict, dataset, memory_cfg, fps: float, global_idx: int, lag: int) -> tuple[dict, int]:
    """The same frame with its lookback window taken from ``lag`` frames earlier.

    Returns the observation and the shift actually applied. The shift is short of ``lag``
    only when the episode does not reach that far back, and a short shift is a weak
    control — the stale window then overlaps the real one — so the caller drops those
    frames from the ``stale`` statistics rather than averaging a degenerate null in.
    """
    frame_idx = int(dataset.hf_dataset[global_idx]["frame_index"].item())
    stale_idx = global_idx - min(lag, frame_idx)
    keys = [str(k) for k in memory_cfg.history_keys if not str(k).startswith("depth.")]
    out = dict(obs)
    out.update(assemble_frame_history(dataset, stale_idx, memory_cfg, fps, keys))
    return out, global_idx - stale_idx


def _predict(adapter, frame: dict, obs: dict, seed: int) -> torch.Tensor:
    """One chunk from one prompt under one flow seed, in normalized action space."""
    generator = torch.Generator(device=adapter.device)
    generator.manual_seed(seed)
    return adapter.predict_action_chunk(
        obs,
        frame["task"],
        state=frame["state"],
        subtask=frame["subtask"],
        metadata=frame["metadata"],
        generator=generator,
    )[1]


def _render_regime(rows: list[dict], summary: dict, output_dir: str) -> None:
    """Is the split real, where do the frames sit, and does the content matter."""
    delta = np.array([row["delta_full"] for row in rows])
    delta_seed = np.array([row["delta_seed"] for row in rows])
    stale_rows = [row for row in rows if row["stale_valid"]]
    delta_stale = np.array([row["delta_stale"] for row in stale_rows])
    tau = summary["sampler_floor"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    span = float(np.percentile(np.abs(np.concatenate([delta, delta_seed, delta_stale])), 99))
    bins = np.linspace(-span, span, 41)
    for values, label, color in (
        (delta, "full — real history", "#2A9D8F"),
        (delta_stale, "stale — wrong history", "#F4A261"),
        (delta_seed, "none' — flow seed only", "#9AA0A6"),
    ):
        axes[0].hist(values, bins=bins, histtype="step", linewidth=1.6, color=color, label=label)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].axvspan(-tau, tau, color="#9AA0A6", alpha=0.18, label=r"$\pm\tau$ (sampler floor)")
    axes[0].set_xlabel(r"$\Delta(c)=\mathrm{MSE}(\mathrm{none})-\mathrm{MSE}(c)$")
    axes[0].set_ylabel("frames")
    axes[0].set_title(
        "Is the helped/hurt split real?\n"
        rf"$G=\mathrm{{MSE}}_\mathrm{{stale}}-\mathrm{{MSE}}_\mathrm{{full}}$: "
        rf"{summary['content_gain']:+.5f}, $z$={summary['content_gain_z']:+.2f}"
    )
    axes[0].legend(fontsize=8)

    for verdict, color in _VERDICT_COLOR.items():
        subset = [row for row in rows if row["verdict"] == verdict]
        axes[1].scatter(
            [row["rmse_full"] for row in subset],
            [row["delta_full"] for row in subset],
            s=20, alpha=0.75, color=color, label=f"{verdict} ({len(subset)})",
        )
    axes[1].scatter(
        [row["rmse_stale"] for row in stale_rows], delta_stale,
        s=14, alpha=0.30, marker="x", color="#F4A261", label=f"stale null ({len(stale_rows)})",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhspan(-tau, tau, color="#9AA0A6", alpha=0.18)
    axes[1].set_xlabel(r"$\mathrm{RMSE}$ vs no-history (how far the chunk moved)")
    axes[1].set_ylabel(r"$\Delta$ vs GT (up is better)")
    axes[1].set_title("Per frame: how far it moved vs whether that helped")
    axes[1].legend(fontsize=8)

    verdicts = list(_VERDICT_COLOR)
    x = np.arange(len(verdicts))
    real_counts = [sum(row["verdict"] == v for row in rows) for v in verdicts]
    null_counts = [sum(row["stale_verdict"] == v for row in stale_rows) for v in verdicts]
    axes[2].bar(x - 0.2, np.array(real_counts) / len(rows), width=0.4, color="#2A9D8F", label="full")
    if stale_rows:
        axes[2].bar(x + 0.2, np.array(null_counts) / len(stale_rows), width=0.4, color="#F4A261", label="stale")
    axes[2].axhline(0.05, color="black", linestyle="--", linewidth=0.9, label=r"5% expected past $\tau$ by chance")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(verdicts)
    axes[2].set_ylabel("fraction of frames")
    axes[2].set_title("Verdicts — real history against the wrong-content null")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"When short-term memory helps — {len(rows)} frames from {summary['n_episodes']} held-out "
        f"episodes, {summary['chunk_size']}-step chunks  |  "
        f"helped {summary['helped_fraction']:.0%} · hurt {summary['hurt_fraction']:.0%}",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(os.path.join(output_dir, "regime.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def _select_examples(rows: list[dict], per_category: int = 2) -> list[tuple[str, int]]:
    """Frames memory helped most, hurt most, and moved most while meaning nothing."""
    moved_but_null = [i for i, row in enumerate(rows) if row["verdict"] == "indistinguishable"]
    rankings = (
        ("helped", range(len(rows)), lambda i: rows[i]["delta_full"], True),
        ("hurt", range(len(rows)), lambda i: rows[i]["delta_full"], False),
        ("moved_but_null", moved_but_null, lambda i: rows[i]["rmse_full"], True),
    )
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, candidates, key, descending in rankings:
        ranked = sorted(candidates, key=key, reverse=descending)
        taken = 0
        for index in ranked:
            if index in used:
                continue
            selected.append((label, index))
            used.add(index)
            taken += 1
            if taken >= per_category:
                break
    return selected


def _render_example(dataset, memory_cfg, fps: float, diagnostic: dict, label: str, output_path: str) -> None:
    """What memory saw at this frame, and what each condition then predicted per joint."""
    row = diagnostic["row"]
    gt = diagnostic["gt"]
    obs, _gt, _state, _subtask, _task, ep_idx, frame_idx = get_frame_data(
        dataset, row["global_idx"], gt.shape[0]
    )
    image_keys = sorted(k for k in obs if k.startswith("observation.images."))
    history = assemble_frame_history(dataset, row["global_idx"], memory_cfg, fps, image_keys)

    n_cols = 4
    n_image_rows = max(1, len(image_keys))
    n_action_rows = (gt.shape[-1] + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(18, 3.0 * n_image_rows + 2.7 * n_action_rows))
    grid = fig.add_gridspec(n_image_rows + n_action_rows, n_cols, hspace=0.45, wspace=0.28)

    for row_idx, key in enumerate(image_keys):
        window = history.get(f"history.{key}")
        if window is not None:
            window = window.squeeze(0)
            for col_idx, slot in enumerate([0, len(window) // 2, len(window) - 1]):
                ax = fig.add_subplot(grid[row_idx, col_idx])
                ax.imshow(as_image(window[slot]))
                ax.set_title(f"{key.split('.')[-1]} history: {('oldest', 'middle', 'newest')[col_idx]}")
                ax.axis("off")
        ax = fig.add_subplot(grid[row_idx, 3])
        ax.imshow(as_image(obs[key]))
        ax.set_title(f"{key.split('.')[-1]} current")
        ax.axis("off")

    steps = np.arange(gt.shape[0])
    joint_names = joint_names_for_dim(gt.shape[-1])
    for joint in range(gt.shape[-1]):
        ax = fig.add_subplot(grid[n_image_rows + joint // n_cols, joint % n_cols])
        ax.plot(steps, gt[:, joint], color="black", linewidth=2.0, label="GT")
        for condition, actions in diagnostic["acts"].items():
            color, linestyle = _CONDITION_STYLE[condition]
            ax.plot(steps, actions[:, joint], color=color, linestyle=linestyle, linewidth=1.25, label=condition)
        ax.set_title(f"{joint_names[joint]} (normalized)", fontsize=9)
        ax.grid(True, alpha=0.25, linestyle=":")
        if joint == 0:
            ax.legend(fontsize=8, ncol=2)
    for unused in range(gt.shape[-1], n_action_rows * n_cols):
        fig.add_subplot(grid[n_image_rows + unused // n_cols, unused % n_cols]).axis("off")

    fig.suptitle(
        f"{label.upper()} — episode {ep_idx}, frame {frame_idx}  |  "
        rf"$\Delta$(full)={row['delta_full']:+.4f} ({row['verdict']}), "
        rf"$G$={row['content_gain']:+.4f}, moved RMSE={row['rmse_full']:.4f}",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


_REGIME_HOW = r"""**Left — is the split real?** Three distributions of the same quantity,
$\Delta(c)=\mathrm{MSE}(\mathrm{none})-\mathrm{MSE}(c)$, over the sampled frames. Green is
the real history, orange is the window as it stood $W$ s earlier (a perturbation of the
same size carrying the wrong content), grey changes only the flow seed. If green is not
wider than orange, memory is being *reacted to* rather than *read*: the chunk moves, and
lands closer to the demonstration exactly as often as a wrong window would. The shaded
band is $\tau=Q_{0.9}(|\Delta(\mathrm{none}')|)$, the width a $\Delta$ has to clear to be
more than a redrawn sampler.

**Middle — per frame.** How far the chunk moved against whether that move helped, one
point per frame, coloured by verdict, with the stale null drawn underneath as crosses.
Points at small $x$ are frames memory never reached; their height is noise. A green cloud
sitting above an orange cloud at the same $x$ is the shape of memory working.

**Right — verdicts.** The same frames counted. The dashed line is the ~5% each side that
the definition of $\tau$ puts outside the band by construction, so a helped fraction at
the line is a helped fraction of zero."""

_EXAMPLE_HOW = {
    "helped": "A frame where real history moved the chunk toward the demonstration. Read the "
              "history strip first: if nothing in it is informative, the improvement is a "
              "coincidence and this frame is the tail of the distribution, not a mechanism.",
    "hurt": "A frame where history moved the chunk away. This is the population the mean in "
            "`mem_history_influence` cancels against, and it is the same size as the helped one.",
    "moved_but_null": "A frame where history moved the chunk hard and it changed nothing "
                      "measurable — the influence bars' magnitude with none of their promise.",
}


def run(adapter, dataset, cfg, output_dir: str) -> None:
    memory_cfg = getattr(cfg.policy, "memory", None)
    if memory_cfg is None or not memory_cfg.history_keys or memory_cfg.history_num_samples <= 0:
        logging.info("[mem_history_regime] no short-term memory configured — skipping.")
        return
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[mem_history_regime] needs continuous flow actions — skipping.")
        return

    makedirs(output_dir)
    p = cfg.probe_parameters
    fps = float(cfg.env.fps)
    chunk_size = int(cfg.policy.chunk_size)
    lag = int(round(memory_cfg.history_window_seconds * fps))

    samples = sample_episodes_evenly(
        dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )
    logging.info(f"[mem_history_regime] {len(samples)} frames x 4 = {len(samples) * 4} forward passes")

    adapter._set_probe_cuda_graph_enabled(False)  # history mask varies per condition; keep eager
    rows: list[dict] = []
    diagnostics: list[dict] = []
    try:
        for _ep, _fr, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            obs, state = frame["obs"], frame["state"]
            stale_obs, stale_shift = _stale_observation(obs, dataset, memory_cfg, fps, global_idx, lag)
            no_history = _variant_observation(obs, False)

            acts = {
                "none": _predict(adapter, frame, no_history, 0),
                "full": _predict(adapter, frame, obs, 0),
                "stale": _predict(adapter, frame, stale_obs, 0),
            }
            reseeded = _predict(adapter, frame, no_history, 1)

            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], state)
            hold_raw = state[: frame["gt_actions"].shape[-1]].unsqueeze(0).repeat(
                frame["gt_actions"].shape[0], 1
            )
            hold_norm = adapter.normalize_gt_actions(hold_raw, state)
            mse = {name: float((act - gt_norm).pow(2).mean()) for name, act in acts.items()}
            mse["none_reseed"] = float((reseeded - gt_norm).pow(2).mean())
            base = acts["none"]
            trajectory = {}
            for name, act in {**acts, "none_reseed": reseeded}.items():
                horizon = min(act.shape[0], gt_norm.shape[0])
                components = trajectory_error_components(
                    act[:horizon], gt_norm[:horizon], hold_norm[:horizon]
                )
                trajectory[name] = {
                    key: float(value) if bool(torch.isfinite(value)) else None
                    for key, value in components.items()
                }

            row = {
                "global_idx": int(global_idx),
                "episode_idx": int(_ep),
                "frame_idx": int(_fr),
                "mse": mse,
                "trajectory": trajectory,
                "delta_full": mse["none"] - mse["full"],
                "delta_stale": mse["none"] - mse["stale"],
                "delta_seed": mse["none"] - mse["none_reseed"],
                "content_gain": mse["stale"] - mse["full"],
                "rmse_full": float((acts["full"] - base).pow(2).mean().sqrt()),
                "rmse_stale": float((acts["stale"] - base).pow(2).mean().sqrt()),
                "rmse_seed": float((reseeded - base).pow(2).mean().sqrt()),
                "stale_shift": int(stale_shift),
                "stale_valid": stale_shift == lag,
            }
            rows.append(row)
            diagnostics.append({"gt": gt_norm, "acts": acts, "row": row})
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not rows:
        logging.warning("[mem_history_regime] no frames produced measurements.")
        return

    delta = np.array([row["delta_full"] for row in rows])
    gain = np.array([row["content_gain"] for row in rows if row["stale_valid"]])
    tau = float(np.quantile(np.abs([row["delta_seed"] for row in rows]), 0.9))
    for row in rows:
        row["verdict"] = (
            "helped" if row["delta_full"] > tau else "hurt" if row["delta_full"] < -tau else "indistinguishable"
        )
        row["stale_verdict"] = (
            "helped" if row["delta_stale"] > tau else "hurt" if row["delta_stale"] < -tau else "indistinguishable"
        )

    trajectory_content_gain = {}
    trajectory_content_gain_valid = {}
    for key in ("path_mse", "shape_mse", "terminal_mse", "terminal_direction_loss"):
        paired = [
            (row["trajectory"]["full"][key], row["trajectory"]["stale"][key])
            for row in rows
            if row["stale_valid"]
            and row["trajectory"]["full"][key] is not None
            and row["trajectory"]["stale"][key] is not None
        ]
        gains = [stale - full for full, stale in paired]
        trajectory_content_gain[key] = float(np.mean(gains)) if gains else None
        trajectory_content_gain_valid[key] = len(gains)

    summary = {
        "n_frames": len(rows),
        "n_episodes": len({row["episode_idx"] for row in rows}),
        "chunk_size": chunk_size,
        "stale_lag_frames": lag,
        "n_stale_valid": int(len(gain)),
        "sampler_floor": tau,
        "helped_fraction": float(np.mean([row["verdict"] == "helped" for row in rows])),
        "hurt_fraction": float(np.mean([row["verdict"] == "hurt" for row in rows])),
        "indistinguishable_fraction": float(np.mean([row["verdict"] == "indistinguishable" for row in rows])),
        "delta_full_mean": float(delta.mean()),
        "delta_full_median": float(np.median(delta)),
        "content_gain": trajectory_content_gain["path_mse"] or 0.0,
        "trajectory_content_gain": trajectory_content_gain,
        "trajectory_content_gain_valid": trajectory_content_gain_valid,
        "content_gain_z": (
            float(gain.mean() / (gain.std(ddof=1) / np.sqrt(len(gain)))) if len(gain) > 1 and gain.std() else 0.0
        ),
        "rmse_full_mean": float(np.mean([row["rmse_full"] for row in rows])),
        "rmse_stale_mean": float(np.mean([row["rmse_stale"] for row in rows])),
        "rmse_seed_mean": float(np.mean([row["rmse_seed"] for row in rows])),
    }
    with open(os.path.join(output_dir, "regime.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    _render_regime(rows, summary, output_dir)

    examples_dir = os.path.join(output_dir, "examples")
    makedirs(examples_dir)
    examples: list[tuple[str, str]] = []
    for label, index in _select_examples(rows):
        row = rows[index]
        filename = f"{label}_ep{row['episode_idx']:04d}_fr{row['frame_idx']:06d}.png"
        _render_example(dataset, memory_cfg, fps, diagnostics[index], label, os.path.join(examples_dir, filename))
        examples.append((label, filename))

    write_index(
        output_dir,
        sys.modules[__name__],
        title="MEM History Regime",
        group="History",
        claim="Which frames does short-term history help on, which does it hurt, and is either real?",
        summary=summary,
        metrics=[
            Metric(
                "content_gain_z", "Content evidence (z on MSE(stale) − MSE(full))", good="high",
                fmt=2, baseline=0.0, warn=2.0, bad=0.0, primary=True,
                note="Paired within frame over the frames whose stale window cleared the full lag. "
                     "Below 2 there is no evidence the model read this window rather than any window; "
                     "below 0 a wrong window fit the demonstration better.",
            ),
            Metric(
                "helped_fraction", "Frames helped past the sampler floor", good="high", fmt=2,
                baseline=0.05, primary=True,
                note=r"$\tau$ is the 90th percentile of $|\Delta|$ under a seed change alone, so ~5% "
                     "lands here by construction.",
            ),
            Metric("hurt_fraction", "Frames hurt past the sampler floor", good="low", fmt=2, baseline=0.05),
            Metric("delta_full_mean", "Mean ΔMSE (the parent probe's bar)", good="high", fmt=5, baseline=0.0,
                   refs=["mem_history_influence"]),
            Metric("sampler_floor", "τ — |ΔMSE| a reseed alone produces", good="none", fmt=5),
            Metric(
                "trajectory_content_gain.path_mse",
                "History content gain · path MSE",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Path MSE(stale) − path MSE(full); positive means the correct window beats the wrong-content window.",
            ),
            Metric(
                "trajectory_content_gain.shape_mse",
                "History content gain · temporal shape",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Shape MSE(stale) − shape MSE(full); positive means correct history better matches adjacent-target changes.",
            ),
            Metric(
                "trajectory_content_gain.terminal_mse",
                "History content gain · final-position MSE",
                good="high",
                fmt=5,
                baseline=0.0,
                note="Final-position MSE(stale) − final-position MSE(full), at target 30.",
            ),
            Metric(
                "trajectory_content_gain.terminal_direction_loss",
                "History content gain · final direction",
                good="high",
                fmt=5,
                baseline=0.0,
                note=f"Final-direction loss(stale) − loss(full), independent of length; stationary GT endpoints are excluded ({trajectory_content_gain_valid['terminal_direction_loss']} valid pairs).",
            ),
        ],
        panels=[
            Panel("regime.png", "Real history against a wrong-content null and a sampler null",
                  how=_REGIME_HOW, primary=True, refs=["mem_history_influence"]),
        ] + [
            Panel(f"examples/{filename}", f"{label.replace('_', ' ')} — history, then every joint against GT",
                  how=_EXAMPLE_HOW[label])
            for label, filename in examples
        ],
        see_also=["mem_history_influence", "mem_temporal_attention", "attention_budget", "action_trace"],
        extra={
            "viewer": {
                "metric_groups": [
                    {
                        "title": "Correct history vs stale history",
                        "keys": [
                            "trajectory_content_gain.path_mse",
                            "trajectory_content_gain.shape_mse",
                            "trajectory_content_gain.terminal_mse",
                            "trajectory_content_gain.terminal_direction_loss",
                        ],
                    }
                ]
            }
        },
    )

    logging.info(
        f"[mem_history_regime] n={len(rows)}  helped={summary['helped_fraction']:.0%}  "
        f"hurt={summary['hurt_fraction']:.0%}  tau={tau:.5f}  "
        f"content gain={summary['content_gain']:+.5f} (z={summary['content_gain_z']:+.2f})"
    )

r"""MEM history-influence probe (04_memory.md §2.4).

Does short-term memory move the policy's action chunk, and does the move help?

The question sounds like it needs one ablation — drop history, see what happens — and that
is what this probe used to do. It cannot work on this policy. Training ran with
``memory.history_dropout: 0.0``, so a prompt with the history channels missing is a shape
the model has never once seen, and the damage done by removing them mixes the thing being
measured (history carries information) with a thing nobody asked about (the model degrades
off-distribution). The old ``none`` origin is that prompt. The same mistake, one level up,
made the age profile in ``mem_temporal_attention`` look like a fact about history when a
control showed it was a fact about sequence position.

So the history channels are never removed. They are kept present, correctly shaped and
fully populated, and only their *content* is varied. Each channel independently takes one
of three treatments:

  ``real``      the true lookback window
  ``constant``  the current frame / current state copied into every slot — a well-formed
                history carrying no information the present does not already have
  ``foreign``   a complete lookback window from a different held-out episode — real
                trajectory statistics, real image statistics, wrong scene

Both channels crossed gives a $3\times3$ factorial, run on every sampled frame with the
same seeded flow noise, so any two cells differ only in the treatment. ``none`` is still
run and reported, as one column, to keep the numbers comparable with captures taken before
this design — it is a reference, no longer an origin.

*Image history* is the stack of past frames the MEM video encoder attends over in its
temporal layers. *State history* is the proprioceptive lookback — one embedded position per
past timestep. Depth history travels through the point-map encoder and no treatment here
touches it (``depth_modality_probe`` covers that path), so the cells differ only in the MEM
channels.

Write $a^{(c)}$ for the normalized chunk under cell $c$, $a^{\star}$ for the demonstrated
chunk, and $T$, $D$ for chunk steps and action dimensions. Every claim is a paired
difference against the ``real/real`` cell on the same frame:

$$\Delta\mathrm{MSE}(c)=\frac{1}{TD}\sum_{t,d}\left(a^{(c)}_{t,d}-a^{\star}_{t,d}\right)^{2}
-\frac{1}{TD}\sum_{t,d}\left(a^{(\mathrm{real/real})}_{t,d}-a^{\star}_{t,d}\right)^{2} .$$

Positive means the real window predicted the demonstration better than the treatment did,
which is memory content being worth something. The two readings that matter:

* $\Delta\mathrm{MSE}(\mathrm{constant/constant})$ — **what the content is worth.** Near
  zero and the model gains nothing from history it does not already have from the current
  frame, whatever the old ``none`` gap said.
* $\Delta\mathrm{MSE}(\mathrm{foreign/foreign})$ — **whether it is _this_ history.** A model
  helped by any well-formed window is not remembering; it is being conditioned.

Because the cells are paired per frame, each difference has its own noise floor: the
standard error of the per-frame differences, plotted on every bar. A bar whose interval
crosses zero is not an effect, and this probe reports no number without one.

The factorial's off-diagonal cells answer the compensation question — whether state history
carries the load when image history is neutered — which neither single-channel ablation can
see. ``real/real`` minus a one-channel treatment is that channel's contribution *in the
presence of* the other, and those do not add up: read them as contrasts, never as a
decomposition.

Every number is a mean over frames, and on a checkpoint where memory helps half the frames
and hurts the other half that mean is near zero whatever the effect size. This probe
answers *which treatment*; ``mem_history_regime`` splits the mean into its halves.

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
    assemble_frame_history,
    history_offsets,
    makedirs,
    panel_caption,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)
from lerobot.utils.constants import OBS_STATE

TREATMENTS = ("real", "constant", "foreign")
# The factorial, plus the legacy origin. Ordered so the figure reads real → emptied →
# wrong, and so CELLS[0] is the reference every paired difference is taken against.
CELLS: tuple[tuple[str, str], ...] = tuple(
    (images, states) for images in TREATMENTS for states in TREATMENTS
)
REFERENCE = ("real", "real")


def cell_name(images: str, states: str) -> str:
    return f"{images}/{states}"


def _mem_history_keys(obs: dict) -> list[str]:
    """The MEM history keys in an observation: image and state, never depth.

    Depth history rides the point-map encoder rather than the MEM video encoder, so it is
    left alone by every treatment here and ``depth_modality_probe`` covers that path.
    Keeping it fixed is what makes two cells differ only in what the caller intends.
    """
    return [
        key
        for key in obs
        if key.startswith("history.")
        and not key.startswith("history.depth.")
        and ("images" in key or key == f"history.{OBS_STATE}")
    ]


def _constant_history(obs: dict, key: str) -> torch.Tensor:
    """The current frame or state, copied into every slot of ``key``'s window.

    A history that is present, correctly shaped and carries nothing the current
    observation does not already carry. This is the origin the probe measures from: unlike
    a dropped channel it is a prompt shape the model saw throughout training, so a
    difference against it is about information and not about distribution shift.
    """
    window = obs[key]
    current = obs[key.removeprefix("history.")]
    n_slots = int(window.shape[1])
    return current.unsqueeze(1).repeat_interleave(n_slots, dim=1).to(window.dtype)


def _variant_observation(obs: dict, images: str, states: str, foreign: dict) -> dict:
    """One factorial cell: each MEM channel set to real, constant or foreign content.

    ``none`` is spelled by the caller as the legacy drop and handled separately; every cell
    built here keeps both channels populated.
    """
    out = dict(obs)
    for key in _mem_history_keys(obs):
        treatment = states if key == f"history.{OBS_STATE}" else images
        if treatment == "constant":
            out[key] = _constant_history(obs, key)
        elif treatment == "foreign" and key in foreign:
            out[key] = foreign[key].to(obs[key].dtype)
    return out


def _drop_history(obs: dict) -> dict:
    """The legacy ``none`` prompt: MEM history keys removed rather than emptied.

    Kept so captures taken before the factorial stay comparable, and kept clearly labelled
    off-distribution: training ran at ``history_dropout: 0.0``, so the model has never seen
    this prompt shape and its damage is not attributable to missing information.
    """
    return {key: value for key, value in obs.items() if key not in set(_mem_history_keys(obs))}


def _predict(adapter, device, observation: dict, frame: dict) -> torch.Tensor:
    """One chunk under one cell, with the flow noise held fixed across cells.

    The seed is constant rather than per frame on purpose: every comparison in this probe
    is between two cells of the *same* frame, so shared noise removes the flow draw from
    the difference entirely and the paired standard errors measure the treatment alone.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    return adapter.predict_action_chunk(
        observation,
        frame["task"],
        state=frame["state"],
        subtask=frame["subtask"],
        metadata=frame["metadata"],
        generator=generator,
    )[1]


def _foreign_history(dataset, memory_cfg, fps: float, obs: dict, global_idx: int) -> dict:
    """A complete lookback window assembled at ``global_idx``, for the foreign treatment.

    The donor index comes from the probe's own sample list, so it is already snapped to the
    ``policy.image_stride`` grid the depth sidecar and the history offsets require — an
    arbitrary index is not safe to assemble from.
    """
    keys = [key.removeprefix("history.") for key in _mem_history_keys(obs)]
    if not keys:
        return {}
    return assemble_frame_history(dataset, global_idx, memory_cfg, fps, keys)


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


def _grid_panel(fig, ax, summary) -> None:
    """The factorial: mean squared error against the demonstration, per cell."""
    grid = np.asarray([
        [summary[f"{cell_name(images, states)}_gt_mse"] for states in TREATMENTS]
        for images in TREATMENTS
    ])
    image = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(len(TREATMENTS)), TREATMENTS)
    ax.set_yticks(range(len(TREATMENTS)), TREATMENTS)
    ax.set_xlabel("state history")
    ax.set_ylabel("image history")
    for row in range(len(TREATMENTS)):
        for col in range(len(TREATMENTS)):
            ax.text(col, row, f"{grid[row, col]:.4f}", ha="center", va="center",
                    fontsize=8, color="white" if grid[row, col] < grid.mean() else "black")
    ax.set_title("MSE against the demonstration, per cell")
    fig.colorbar(image, ax=ax, fraction=0.046, label=r"MSE vs GT (normalized$^2$)")
    panel_caption(ax, [
        r"Both MEM channels crossed with the three treatments, one forward per cell per frame",
        r"with the same seeded flow noise. Top-left is the real window; every other cell keeps",
        r"the channel present and populated and changes only what is in it. Lower is better.",
        r"Read down a column for what image content is worth at a fixed state treatment, and",
        r"across a row for the reverse; the off-diagonal is where one channel compensating for",
        r"the other would show, which no single-channel ablation can see.",
        r"*none* is not a cell — it removes the channels rather than emptying them, a prompt",
        r"shape training never produced, and it sits in the panel beside this one as a reference.",
    ])


def _contrast_panel(ax, summary) -> None:
    """Each treatment's penalty against the real window, with its own noise floor."""
    contrasts = [
        (cell_name("constant", "constant"), "both emptied\n(what content is worth)"),
        (cell_name("constant", "real"), "image emptied"),
        (cell_name("real", "constant"), "state emptied"),
        (cell_name("foreign", "foreign"), "both foreign\n(is it *this* history?)"),
        (cell_name("foreign", "real"), "image foreign"),
        (cell_name("real", "foreign"), "state foreign"),
        ("none", "none — OFF-DISTRIBUTION\n(legacy origin)"),
    ]
    values = [summary[f"{name}_gt_mse_penalty"] for name, _ in contrasts]
    errors = [1.96 * summary[f"{name}_gt_mse_penalty_sem"] for name, _ in contrasts]
    colors = ["#8D99AE" if name == "none" else "#457B9D" for name, _ in contrasts]
    y = np.arange(len(contrasts))
    ax.barh(y, values, xerr=errors, color=colors, height=0.66,
            error_kw={"ecolor": "#222222", "capsize": 3, "lw": 1.1})
    ax.axvline(0.0, color="#111111", linewidth=1.0)
    ax.set_yticks(y, [label for _, label in contrasts], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"MSE(treatment) $-$ MSE(real/real), normalized$^2$")
    ax.set_title("What does the real window buy? (95% CI)")
    panel_caption(ax, [
        r"Paired per frame against the real window: same frame, same flow noise, one treatment",
        r"apart. Positive means the real window predicted the demonstration better, so a long",
        r"bar is memory content being worth something and a bar whose interval crosses the zero",
        r"line is not an effect at all. Intervals are $1.96\times$ the standard error of the",
        r"per-frame differences — this contrast's own noise floor, not a pooled one.",
        r"*both emptied* is the headline: the history is present and correctly shaped but says",
        r"nothing the current frame does not, so this is what the *content* is worth. *both",
        r"foreign* holds a real window from another episode — a model helped as much by any",
        r"window as by its own is being conditioned, not remembering.",
        r"*none* removes the channels instead. Training ran at history_dropout 0, so that",
        r"prompt shape never occurred and its bar mixes lost information with distribution",
        r"shift; it is drawn grey and kept only to line up with captures made before this",
        r"design. When it towers over *both emptied*, the old headline was the shift.",
    ])


def _influence_panel(ax, summary) -> None:
    """How far each treatment moves the chunk, before asking whether the move was good."""
    names = [cell_name(i, s) for i, s in CELLS if (i, s) != REFERENCE] + ["none"]
    y = np.arange(len(names))
    ax.barh(y - 0.19, [summary[f"{n}_rmse"] for n in names], height=0.36,
            color="#457B9D", label="RMSE")
    ax.barh(y + 0.19, [summary[f"{n}_maxabs"] for n in names], height=0.36,
            color="#E76F51", label=r"max$|\Delta|$")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("normalized action Δ vs the real window")
    ax.set_title("Influence: how far the chunk moves")
    ax.legend(fontsize=8, loc="lower right")
    panel_caption(ax, [
        r"Distance from the real window's chunk, with no reference to the demonstration, so",
        r"nothing here says whether a move was an improvement — that is the panel to the left.",
        r"RMSE and max close together is a treatment that shifts the whole chunk slightly; max",
        r"far above RMSE is one that moves a single timestep hard.",
        r"Read it as the gate on the panel to the left: where a treatment barely moves the",
        r"chunk, its penalty bar is measuring nothing rather than reporting good news.",
    ])


def _render(summary: dict, rows: list[dict], cfg, output_path: str) -> None:
    fig = plt.figure(figsize=(21, 8.4))
    panels = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.42,
                              left=0.05, right=0.98, top=0.86, bottom=0.30)
    axes = [fig.add_subplot(panels[0, i]) for i in range(3)]
    _grid_panel(fig, axes[0], summary)
    _contrast_panel(axes[1], summary)
    _influence_panel(axes[2], summary)
    provenance = summary["data"]
    fig.suptitle(
        f"MEM history: what is the content worth? — {len(rows)} frames from "
        f"{provenance['val']['n_episodes']} held-out episodes, "
        f"{int(cfg.policy.chunk_size)}-step chunks",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# Panel captions carry the algebra because the bars are differences of differences: a
# reader who guesses at what is subtracted from what reads the middle panel backwards.
_INFLUENCE_HOW = r"""Each MEM history channel — the image lookback the video encoder
attends over, and the proprioceptive state lookback — is independently set to **real**, to
**constant** (the current frame or state copied into every slot), or to **foreign** (a
complete window from a different held-out episode). Every cell keeps the channel present
and correctly shaped; only the content changes.

That is the point of the design. Training ran at ``memory.history_dropout: 0.0``, so a
prompt with the history channels *missing* is a shape the model never saw, and the old
``none`` origin measured lost information and distribution shift together. It is still run
and still plotted, in grey, so older captures line up — but every claim is now a paired
difference against the ``real/real`` cell, on the same frame, under the same flow noise.

**Read the middle panel first.** *both emptied* is what the memory content is worth: the
window is there, it just says nothing the present does not. If that bar is short while
*none* is tall, the model's apparent need for history was its need for the prompt shape.
*both foreign* asks whether it is *this* history — a model helped as much by another
episode's window is being conditioned, not remembering. Every bar carries a 95% interval
built from the per-frame paired differences, and a bar crossing zero is not an effect.

The left panel is the raw factorial the contrasts are drawn from; its off-diagonal is where
one channel compensating for the other would show. The right panel is the gate: where a
treatment barely moves the chunk at all, its penalty bar is measuring nothing rather than
reporting good news."""


def _write_manifest(output_dir: str, summary: dict) -> dict:
    """Describe the history channels' effect on the action chunk to the viewer."""
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="MEM History Influence",
        group="History",
        claim="Is the model helped by what is in its history, or only by history being there?",
        summary=summary,
        # Every number the probe computes is already a bar in influence.png, where it
        # carries its distribution instead of collapsing to one row. The values stay in
        # influence.json.
        metrics=[],
        panels=[
            Panel(
                "influence.png",
                "What the history content is worth: a 3x3 treatment factorial with paired intervals",
                how=_INFLUENCE_HOW,
                primary=True,
                refs=["mem_history_regime", "mem_temporal_attention"],
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

    # Donor frames for the foreign treatment: the sample list itself, so every donor is
    # already snapped to the image_stride grid the history offsets and the depth sidecar
    # require. Chosen from a different episode so "unrelated" means a different scene and
    # not a nearby moment of the same one.
    by_episode: dict[int, list[int]] = {}
    for ep, _fr, idx in samples:
        by_episode.setdefault(int(ep), []).append(int(idx))
    if len(by_episode) < 2:
        logging.warning(
            "[mem_history_influence] the foreign treatment needs at least two episodes; "
            f"got {len(by_episode)} — skipping the probe."
        )
        return
    donor_rng = np.random.default_rng(p.random_seed)
    fps = cfg.env.fps

    rows: list[dict] = []
    for _ep, _fr, global_idx in samples:
        frame = probe_frame_inputs(dataset, cfg, global_idx, int(cfg.policy.chunk_size))
        obs, gt_actions, state = frame["obs"], frame["gt_actions"], frame["state"]
        subtask, task_str, metadata = frame["subtask"], frame["task"], frame["metadata"]
        full_batch = adapter._make_batch(obs, task_str, subtask=subtask, metadata=metadata)
        if "history_images_mask" not in full_batch and "history_state_values" not in full_batch:
            logging.warning("[mem_history_influence] batch carries no history tensors — skipping frame.")
            continue

        other_episodes = [ep for ep in by_episode if ep != int(_ep)]
        donor_episode = other_episodes[donor_rng.integers(len(other_episodes))]
        donor_pool = by_episode[donor_episode]
        donor_idx = donor_pool[donor_rng.integers(len(donor_pool))]
        foreign = _foreign_history(dataset, memory_cfg, fps, obs, donor_idx)

        def predict(observation, frame=frame):
            return _predict(adapter, device, observation, frame)

        gt_norm = adapter.normalize_gt_actions(gt_actions, state)
        acts = {
            cell_name(images, states): predict(_variant_observation(obs, images, states, foreign))
            for images, states in CELLS
        }
        acts["none"] = predict(_drop_history(obs))

        reference = acts[cell_name(*REFERENCE)]
        gt_mse = {name: float((act - gt_norm).pow(2).mean()) for name, act in acts.items()}
        row = {
            "global_idx": int(global_idx),
            "episode_idx": int(_ep),
            "frame_idx": int(_fr),
            "donor_episode_idx": int(donor_episode),
            "donor_global_idx": int(donor_idx),
        }
        for name, act in acts.items():
            row[f"{name}_gt_mse"] = gt_mse[name]
            # Paired, and against the real window rather than against a prompt shape the
            # model never trained on. Positive = the real window beat this treatment.
            row[f"{name}_gt_mse_penalty"] = gt_mse[name] - gt_mse[cell_name(*REFERENCE)]
            row[f"{name}_rmse"] = float((act - reference).pow(2).mean().sqrt())
            row[f"{name}_maxabs"] = float((act - reference).abs().max())
        rows.append(row)

    adapter._restore_probe_cuda_graph_enabled()
    if not rows:
        logging.warning("[mem_history_influence] no frames produced measurements.")
        return

    names = [cell_name(images, states) for images, states in CELLS] + ["none"]
    reference_name = cell_name(*REFERENCE)

    def paired(name: str, field: str) -> tuple[float, float]:
        """Mean of the per-frame values and the standard error of that mean.

        The differences are paired — same frame, same flow noise, one treatment apart — so
        their spread across frames is this contrast's own noise floor and the interval is
        the only thing that says whether a bar is an effect.
        """
        values = np.asarray([row[f"{name}_{field}"] for row in rows], dtype=np.float64)
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))

    summary: dict = {"n_frames": len(rows), "reference_cell": reference_name}
    for name in names:
        for field in ("gt_mse", "gt_mse_penalty", "rmse", "maxabs"):
            mean, sem = paired(name, field)
            summary[f"{name}_{field}"] = mean
            summary[f"{name}_{field}_sem"] = sem
    summary["data"] = _provenance(rows, dataset, cfg, memory_cfg, dict.fromkeys(names))
    with open(os.path.join(output_dir, "influence.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    _render(summary, rows, cfg, os.path.join(output_dir, "influence.png"))

    _write_manifest(output_dir, summary)

    def headline(name: str) -> str:
        return f"{summary[f'{name}_gt_mse_penalty']:+.5f}±{1.96 * summary[f'{name}_gt_mse_penalty_sem']:.5f}"

    logging.info(
        f"[mem_history_influence] n={len(rows)}  penalty vs {reference_name} (95% CI; "
        f"positive = the real window helped):  both emptied={headline(cell_name('constant', 'constant'))}  "
        f"both foreign={headline(cell_name('foreign', 'foreign'))}  "
        f"none[OOD]={headline('none')}"
    )

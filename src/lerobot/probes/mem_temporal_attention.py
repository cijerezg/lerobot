r"""Distributional probe for the MEM video encoder's temporal attention.

When and where does the video encoder read its history frames?

Every ``temporal_layer_stride``-th ViT block applies space-time separable attention: a
time-only step first, then the ordinary spatial block. This probe measures the time-only
step. Its query is a single patch $i$ of a single camera at the current frame, and its
keys are that *same patch position* at each of the $K$ timesteps — the $T$ history ages
plus the query's own — causally masked. A patch sees its own location at earlier times,
never the whole past frame.

The quantity everything is built from is the past mass of one patch in one head,

$$m_{i,h}=\sum_{\tau=1}^{T} w_{i,h,\tau} ,$$

the share of that query's temporal attention landing on history rather than on its own
timestep. Averaging $m$ over patches, heads, cameras and frames gives
``mean_temporal_mass``.

That number cannot be read on its own, because a softmax over $K = T+1$ keys hands the
past some mass whether or not the model wants it. An indifferent allocation gives every
timestep $1/(T+1)$ and therefore

$$m_{\text{uniform}}=\frac{T}{T+1},$$

which is the line drawn on every panel. Only distance from it is evidence, so the
headline number is the enrichment ratio $m/m_{\text{uniform}}$: at 1.0 the past receives
exactly what indifference would give it and no claim about reading history survives.
Above 1.0 the encoder pulls attention onto the past; below 1.0 it pushes attention away —
and unlike an enrichment measured against a spatial-competition null, values well below
1.0 are ordinary here, because concentrating on its own timestep is how this step
expresses "the past is not useful".

Note the null moved on 2026-08-03. Before that the temporal and spatial keys shared one
softmax, making the null $T/(N+T)$ with $N=729$ patches — about 0.7% rather than 83%.
Enrichment numbers from runs before that date are not comparable to these.

A mean over everything hides the regimes that matter — one layer reading hard looks the
same as every layer reading a little, and a read spread evenly over all ages looks the
same as one aimed at a particular moment. So the probe keeps the distribution and splits
the single mean into the three questions it conflates:

* **How much**, per layer and per frame. The spread of $m$ across frames at fixed layer
  is what separates a read the scene drives from a fixed per-layer schedule.
* **Where**, from the age profile — past mass renormalised to sum to 1 over the $T$ ages,
  so it carries nothing about how much was read. Summarised by the log slope $b$ in
  $\log \mathrm{share}(\Delta t)=a+b\,\Delta t$, in e-folds per second: $b<0$ is recency,
  $b>0$ prefers the oldest frame in the window.
* **Whether "where" is real.** That profile is a mean over heads, and a mean over heads is
  ambiguous: a layer whose heads split evenly between preferring the newest and the oldest
  frame produces the same flat profile as a layer with no age preference at all, and those
  are opposite findings. Two per-head numbers separate them — the normalized entropy of
  one head's age mass, which is peakedness carrying no direction, and head agreement
  $|\overline{r}|/\overline{|r|}$ with $r_h$ the newest-minus-oldest share, which is
  direction carrying no magnitude. Neither substitutes for the other.

Each question is answered at the level it lives at, so the panels are not redundant views
of one number: level is per frame, direction is per layer, consensus is per head.

This probe establishes only that history is *read*. Whether reading it helps the action
is ``mem_history_influence``.

Writes:

* ``temporal_attention.png`` — frame/layer distributions, age preference, head and
  camera specialization, and the check that the age preference is not head cancellation;
* ``examples/frame_p*.png`` — low/median/high temporal-read examples with the most-read
  history frame, current image, and spatial temporal-read overlay per camera;
* ``temporal_attention_data.npz`` — the per-frame/layer/camera/head/age/patch arrays;
* ``temporal_attention.json`` — human-readable summary and per-frame layer values.
"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from torch.nn import functional

from lerobot.policies.molmoact2.modeling_molmoact2 import _MEM_TEMPORAL_CAPTURE
from lerobot.probes.manifest import Panel, write_index
from lerobot.probes.utils import (
    as_image,
    assemble_frame_history,
    dataset_display_name,
    get_frame_data,
    makedirs,
    panel_caption,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)


def _temporal_layer_indices(policy) -> list[int]:
    resblocks = policy._backbone().vision_backbone.image_vit.transformer.resblocks
    stride = max(int(policy.config.temporal_layer_stride), 1)
    return [i for i in range(len(resblocks)) if (i + 1) % stride == 0]


def _age_entropy(head_age: np.ndarray) -> np.ndarray:
    r"""How peaked one head's read is over history ages, averaged to one number per layer.

    With $p_h(\Delta t)$ the head's mass on age $\Delta t$ normalized over the $T$ ages,

    $$H_h = -\frac{1}{\log T}\sum_{\Delta t} p_h(\Delta t)\,\log p_h(\Delta t) \in [0, 1] ,$$

    so 1 spreads over every age and 0 puts everything on one. The entropy is taken per
    head and only then averaged over frames, cameras and heads: entropy of the head-mean
    profile would be a different and weaker quantity, since averaging heads first fills in
    the ages any single head ignores. This carries no direction — a head biased to the
    newest age and its mirror biased to the oldest score identically — which is what makes
    it independent of the agreement in ``_age_shape`` rather than a restatement of it.

    Heads with no mass at all are dropped instead of counted as peaked.
    """
    n_ages = head_age.shape[-1]
    if n_ages <= 1:
        return np.zeros(head_age.shape[1], dtype=np.float32)
    total = head_age.sum(axis=-1, keepdims=True)
    probs = np.divide(head_age, total, out=np.zeros_like(head_age), where=total > 1e-12)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1) / np.log(n_ages)
    entropy[total.squeeze(-1) <= 1e-12] = np.nan
    return np.nanmean(entropy, axis=(0, 2, 3))


def _age_shape(head_age: np.ndarray, history_seconds: np.ndarray) -> dict[str, np.ndarray]:
    r"""How each layer weights history by age, and whether its heads agree about it.

    Two numbers per layer, both about the *shape* of the age profile rather than its
    level. The slope fits the age share to an exponential in age,

    $$\log \mathrm{share}(\Delta t) = a + b\,\Delta t ,$$

    so $b<0$ is recency (the share falls as the frame gets older) and $b>0$ is the
    opposite — the layer prefers the oldest frame it can see. The units are e-folds per
    second, which makes $b$ comparable across layers and across history windows.

    Agreement exists because the layer mean is a mean over heads, and a mean over heads
    cannot distinguish "no age preference" from "half the heads prefer the newest frame
    and half prefer the oldest". Per head, $r_h$ = share on the newest age minus share on
    the oldest; then

    $$\mathrm{agreement}=\frac{\left|\overline{r}\right|}{\overline{|r|}} \in [0,1] ,$$

    which is 1 when every head pulls the same way and 0 under exact cancellation. A flat
    row in the age panel means indifference only when this is near 1.

    $r_h$ is averaged over frames and cameras *before* the newest-minus-oldest difference,
    so agreement asks whether the head split is a stable property of the layer. Heads that
    merely disagree frame to frame average toward 0 tilt each and do not register here;
    what registers is a head that reliably pulls one way while its neighbour reliably
    pulls the other. The ratio is scale-free in $r$, so it says nothing about how strong
    the tilts are — read it beside the entropy, never instead of it. When no head tilts at
    all the ratio is $0/0$ and is reported as 1: nothing is being cancelled.
    """
    n_layers, n_ages = head_age.shape[1], head_age.shape[-1]
    total = head_age.sum(axis=-1, keepdims=True)
    share = np.divide(head_age, total, out=np.zeros_like(head_age), where=total > 1e-12)
    # Per head first, so opposed heads can be seen cancelling instead of averaging away.
    head_recency = share[..., -1].mean(axis=(0, 2)) - share[..., 0].mean(axis=(0, 2))
    mean_abs = np.abs(head_recency).mean(axis=-1)
    agreement = np.divide(
        np.abs(head_recency.mean(axis=-1)), mean_abs, out=np.ones(n_layers), where=mean_abs > 1e-12
    )

    layer_share = head_age.mean(axis=(0, 2, 3))
    layer_share = layer_share / np.clip(layer_share.sum(axis=-1, keepdims=True), 1e-12, None)
    if n_ages > 1:
        slope = np.polyfit(history_seconds, np.log(np.clip(layer_share, 1e-12, None)).T, 1)[0]
    else:
        slope = np.zeros(n_layers, dtype=np.float32)
    return {
        "share": layer_share.astype(np.float32),
        "slope": slope.astype(np.float32),
        "agreement": agreement.astype(np.float32),
        "head_recency": head_recency.astype(np.float32),
    }


def _write_manifest(output_dir: str, summary: dict, examples: list[tuple[int, str]]) -> dict:
    """Describe the temporal read to the manifest-driven viewer."""
    panels = [
        Panel(
            "temporal_attention.png",
            f"Temporal-read distributions across frames, layers, heads, cameras, and history age — {summary['provenance']}",
            how="Each panel states its own computation underneath, and is titled with the question it answers. **Top row** is how much history is read and whether the scene changes it, then which age the read lands on. **Bottom row** resolves the same read over heads and cameras, and checks whether a flat age profile is really indifference. Read the top-left panel first: it carries the $T/(T+1)$ baseline that every claim here is measured against, and no claim about reading history survives a level that sits on it. The bottom-right panel is a guard rather than a result — the age panel averages over heads, and a layer whose heads split evenly between newest and oldest looks there exactly like a layer that has no age preference at all, so check the agreement bar before calling any flat curve indifference.",
            primary=True,
        ),
        Panel(
            "temporal_attention_data.npz",
            "Per-frame / layer / camera / head / age / patch arrays for follow-up analysis",
            how="Saved so a new question about this capture does not need another model run.",
        ),
    ]
    panels += [
        Panel(
            f"examples/{filename}",
            f"p{percentile} temporal-read frame — most-read history image and spatial overlay",
            how="The overlay is on the current image: it marks which current patches spend their attention on the past. Compare the p10 and p90 frames — if they look alike, the read is not scene-driven.",
            # The band is what survives across runs; the frame that lands in it does not,
            # so the filename keeps the frame and the comparison viewer keys on the band.
            align=f"examples/frame_p{percentile:02d}.png",
        )
        for percentile, filename in examples
    ]
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="MEM Temporal Attention",
        group="History",
        claim="When and where does the video encoder read its history frames?",
        summary=summary,
        # The absolute mass, the per-layer maximum, the entropy and the baseline are all
        # panels in temporal_attention.png, where they carry their distribution instead
        # of collapsing to one number. The values stay in temporal_attention.json.
        metrics=[],
        panels=panels,
        see_also=["mem_history_influence", "attention_budget"],
    )


def _percentile_examples(scores: np.ndarray) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for percentile in (10, 50, 90):
        target = float(np.percentile(scores, percentile))
        for index in np.argsort(np.abs(scores - target)):
            index = int(index)
            if index not in used:
                selected.append((percentile, index))
                used.add(index)
                break
    return selected


def _provenance(dataset, frame_meta, layers, camera_names, n_heads, n_patches, history_seconds) -> str:
    """What was measured, in one line: every figure and the manifest carry the same one."""
    episodes = sorted({meta["episode_idx"] for meta in frame_meta})
    listed = (
        ", ".join(f"e{episode}" for episode in episodes)
        if len(episodes) <= 8
        else f"e{episodes[0]}…e{episodes[-1]}"
    )
    return (
        f"{dataset_display_name(dataset)} — {len(frame_meta)} frames from {len(episodes)} episodes "
        f"({listed}) · {len(camera_names)} cameras ({', '.join(camera_names)}) × {n_heads} heads × "
        f"{n_patches} patches · layers {layers} · {len(history_seconds)} history frames back to "
        f"-{history_seconds[0]:g}s"
    )


def _layer_colors(n_layers: int) -> np.ndarray:
    """Depth as colour, so the same layer keeps its identity across panels."""
    return plt.cm.viridis(np.linspace(0.08, 0.92, n_layers))


def _spread_panel(ax, temporal_mass, layer_labels, uniform_mass) -> None:
    ax.boxplot([temporal_mass[:, i] for i in range(len(layer_labels))], tick_labels=layer_labels)
    ax.axhline(uniform_mass, color="#E76F51", linestyle="--", label="uniform over time")
    ax.set_xlabel("ViT temporal layer")
    ax.set_ylabel("past attention mass")
    ax.set_title("How much history is read, per layer")
    ax.legend(fontsize=8)
    panel_caption(ax, [
        r"$m=\sum_{\tau} w_\tau$, the share of one query's temporal softmax landing on history",
        r"rather than on its own timestep, averaged over patches, heads and cameras. One box",
        r"per layer over the sampled frames; box = quartiles, whiskers = 1.5 IQR.",
        r"Dashed line is $T/(T+1)$, what a layer with no preference between timesteps gives the",
        r"past. Only the distance from it is evidence, and below it is a real reading:",
        r"the layer is holding attention on its own frame.",
    ])


def _scene_panel(fig, ax, temporal_mass, layer_labels, frame_meta) -> None:
    # The panel asks a contrast question — layer against layer, frame against frame — so
    # the colour scale is spent on the range m actually occupies. Anchoring it at 0 and
    # the uniform baseline instead put over half the map on values m never takes, which
    # compressed a real 0.09-wide layer effect into a few neighbouring shades. Clip both
    # ends to the middle 96% and print the range on the bar: this scale is data-adaptive
    # and therefore not comparable across runs, which the caption says out loud.
    low, high = (float(value) for value in np.percentile(temporal_mass, [2, 98]))
    if high - low < 1e-6:  # a degenerate spread would blow the norm up, not flatten it
        low, high = low - 5e-7, high + 5e-7
    image = ax.imshow(temporal_mass, aspect="auto", cmap="viridis", vmin=low, vmax=high)
    ax.set_xticks(range(len(layer_labels)), layer_labels)
    stride = max(1, len(frame_meta) // 10)
    ticks = list(range(0, len(frame_meta), stride))
    ax.set_yticks(ticks, [f"e{frame_meta[i]['episode_idx']}:f{frame_meta[i]['frame_idx']}" for i in ticks])
    for index in range(1, len(frame_meta)):
        if frame_meta[index]["episode_idx"] != frame_meta[index - 1]["episode_idx"]:
            ax.axhline(index - 0.5, color="white", linewidth=0.7, alpha=0.8)
    ax.set_xlabel("ViT temporal layer")
    ax.set_title("Is the amount read scene-dependent?")
    fig.colorbar(image, ax=ax, fraction=0.046, label=f"past mass (clipped {low:.2f}–{high:.2f})")
    panel_caption(ax, [
        r"The same $m$, one row per sampled frame, ordered by episode; white rules are episode",
        r"boundaries. Vertical stripes mean the layer sets the level and the scene does not",
        r"change it — a fixed schedule, not retrieval. Horizontal structure is the opposite:",
        r"some frames pull more history than others. Read the columns, not the rows: the",
        r"frame axis is a concatenation of episodes, not one trajectory.",
        r"Colour is clipped to the middle 96% of $m$ and so shows contrast, not level: the",
        r"level is the boxplot to the left, and the baseline is in the figure title. The",
        r"range on the bar is set from this run's data — colours do not carry across runs.",
    ])


def _age_panel(ax, shape, layers, history_seconds, colors) -> None:
    uniform_share = 1.0 / len(history_seconds)
    x = -history_seconds
    for index, layer in enumerate(layers):
        ax.plot(
            x, shape["share"][index], marker="o", markersize=3.5, linewidth=1.6,
            color=colors[index], label=f"L{layer}  $b$={shape['slope'][index]:+.2f}/s",
        )
    ax.axhline(uniform_share, color="#E76F51", linestyle="--", linewidth=1.0)
    ax.annotate(
        f"uniform over ages = {uniform_share:.2f}", (x[0], uniform_share), xytext=(2, 4),
        textcoords="offset points", fontsize=7.5, color="#E76F51",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, [f"-{value:g}s" for value in history_seconds])
    # Headroom below every curve, so the six-entry legend has somewhere to sit that is
    # not on top of the data. "best" cannot find one: the curves cross in the middle.
    ax.set_ylim(float(shape["share"].min()) / 2.6, float(shape["share"].max()) * 1.3)
    ax.set_xlabel("history age (oldest → newest)")
    ax.set_ylabel("share of past mass (log)")
    ax.set_title("Which moment is read, given that history is read")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    panel_caption(ax, [
        r"Past mass renormalised to sum to 1 over the $T$ ages, so this is *where* the read",
        r"lands and carries nothing about how much was read — that is the panel to its left.",
        r"Log axis because the profile is close to exponential: $\log\,$share$\,=a+b\,\Delta t$, and $b$ is in",
        r"the legend in e-folds per second. $b<0$ is recency, $b>0$ prefers the oldest frame",
        r"in the window, $b\approx0$ sits on the dashed uniform line. A sign flip with depth means",
        r"the shallow and deep layers are reading history for different reasons.",
        r"These curves average over heads, so a flat one is genuine indifference only where the",
        r"agreement bar (bottom right) is high; where it is low the flatness is cancellation.",
    ])


def _head_panel(fig, ax, head_mean, layer_labels, n_heads) -> None:
    image = ax.imshow(head_mean, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(n_heads), [str(i) for i in range(n_heads)])
    ax.set_yticks(range(len(layer_labels)), layer_labels)
    ax.set_xlabel("ViT head")
    ax.set_ylabel("ViT temporal layer")
    ax.set_title("Do individual heads specialise?")
    fig.colorbar(image, ax=ax, fraction=0.046, label="past mass")
    panel_caption(ax, [
        r"$m$ per head, averaged over frames, patches and cameras. A uniform row means the",
        r"layer spends the same budget in every head; a row with dark cells means some heads",
        r"stay on the current frame while others do the reading. This is the level per head —",
        r"whether they also disagree about *which* age to read is the bottom-right panel.",
        r"Weigh the spread along a row here against the frame-to-frame spread in the panel",
        r"above it: whichever dominates is what actually sets the read, head or scene.",
    ])


def _camera_panel(fig, ax, camera_mean, layer_labels, camera_names) -> None:
    image = ax.imshow(camera_mean, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(camera_names)), camera_names, rotation=20, ha="right")
    ax.set_yticks(range(len(layer_labels)), layer_labels)
    ax.set_ylabel("ViT temporal layer")
    ax.set_title("Does the camera change the read?")
    fig.colorbar(image, ax=ax, fraction=0.046, label="past mass")
    panel_caption(ax, [
        r"$m$ per camera, averaged over frames, patches and heads. The temporal step is shared",
        r"weights applied per camera, so a difference here comes from the images alone:",
        r"a wrist view that moves with the arm has less in common with its own past than a",
        r"fixed overhead view does. Columns that match to two decimals mean the camera is",
        r"not a factor and this panel can be ignored for that layer.",
    ])


def _selectivity_panel(ax, entropy, agreement, layer_labels) -> None:
    x = np.arange(len(layer_labels))
    ax.bar(x - 0.2, entropy, width=0.4, color="#457B9D", label="age entropy (1 = flat over ages)")
    ax.bar(x + 0.2, agreement, width=0.4, color="#E9C46A", label="head agreement (1 = same direction)")
    ax.set_xticks(x, layer_labels)
    # Both bars can reach 1.0, so the legend goes in a band above them rather than on them.
    ax.set_ylim(0.0, 1.45)
    ax.axhline(1.0, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_xlabel("ViT temporal layer")
    ax.set_ylabel("normalized [0, 1]")
    ax.set_title("Is a flat age profile really indifference?")
    ax.legend(fontsize=7, loc="upper center", ncol=2)
    panel_caption(ax, [
        r"Both bars are computed per head and then averaged; the age panel above averages the",
        r"heads first, so these recover exactly what that average destroys. Entropy is one head's",
        r"mass over the $T$ ages, normalised — how peaked that head is, carrying no direction.",
        r"Agreement is $|\overline{r}|/\overline{|r|}$ over heads, $r_h$ = newest-age share minus oldest — direction",
        r"only, carrying no magnitude: 1 when the heads pull the same way, 0 under cancellation.",
        r"So a flat row above is indifference only beside a high bar here; beside a low bar it is",
        r"two opposed head populations. Low on both is heads each reading a different moment.",
    ])


def _render_distribution(
    temporal_mass: np.ndarray,
    head_age: np.ndarray,
    shape: dict[str, np.ndarray],
    layers: list[int],
    camera_names: list[str],
    history_seconds: np.ndarray,
    frame_meta: list[dict],
    uniform_mass: float,
    provenance: str,
    output_path: str,
) -> None:
    """Six views, in reading order: how much history is read, then where, then by whom.

    The top row is the level and its scene dependence, then the age profile. The bottom
    row resolves that same read over heads and cameras and closes with the check that
    makes the age profile readable at all. Every panel prints its own computation, so the
    figure survives being pulled out of the viewer and read on its own.
    """
    n_heads = head_age.shape[3]
    layer_labels = [str(layer) for layer in layers]
    head_total = head_age.sum(axis=-1)
    colors = _layer_colors(len(layers))

    fig = plt.figure(figsize=(20, 12.5))
    panels = fig.add_gridspec(2, 3, hspace=0.66, wspace=0.26,
                              left=0.05, right=0.985, top=0.925, bottom=0.07)
    axes = np.array([[fig.add_subplot(panels[r, c]) for c in range(3)] for r in range(2)])

    _spread_panel(axes[0, 0], temporal_mass, layer_labels, uniform_mass)
    _scene_panel(fig, axes[0, 1], temporal_mass, layer_labels, frame_meta)
    _age_panel(axes[0, 2], shape, layers, history_seconds, colors)
    _head_panel(fig, axes[1, 0], head_total.mean(axis=(0, 2)), layer_labels, n_heads)
    _camera_panel(fig, axes[1, 1], head_total.mean(axis=(0, 3)), layer_labels, camera_names)
    _selectivity_panel(axes[1, 2], _age_entropy(head_age), shape["agreement"], layer_labels)

    fig.suptitle(
        f"MEM temporal-attention diagnostics — uniform-over-time past mass={uniform_mass:.4f}\n{provenance}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def _render_spatial_example(
    dataset,
    memory_cfg,
    fps: float,
    diagnostic: dict,
    percentile: int,
    uniform_mass: float,
    patch_range: tuple[float, float],
    output_path: str,
) -> None:
    """Show the most-read history moment and where the current frame reads history."""
    obs, _gt, _state, _subtask, _task, ep_idx, frame_idx = get_frame_data(
        dataset, diagnostic["global_idx"], 1
    )
    camera_keys = diagnostic["camera_keys"]
    history = assemble_frame_history(dataset, diagnostic["global_idx"], memory_cfg, fps, camera_keys)
    head_age = diagnostic["head_age"]
    patch_mass = diagnostic["patch_mass"].mean(axis=0)
    age_by_camera = head_age.mean(axis=(0, 2))

    # Diverging around the uniform baseline, so "reads history more than indifference
    # would" is a colour flip and not a shade you have to judge against a number. The
    # two slopes are scaled independently, which keeps full contrast across the
    # sub-baseline range where in practice every patch sits. Bounds are shared by all
    # three percentile examples so they stay comparable.
    low, high = patch_range
    norm = TwoSlopeNorm(
        vcenter=uniform_mass,
        vmin=min(low, uniform_mass - 1e-3),
        vmax=max(high, uniform_mass + 1e-3),
    )

    fig, axes = plt.subplots(len(camera_keys), 3, figsize=(15, 4.2 * len(camera_keys)), squeeze=False)
    overlay_images = []
    for camera_idx, key in enumerate(camera_keys):
        history_tensor = history[f"history.{key}"].squeeze(0)
        age_index = int(np.argmax(age_by_camera[camera_idx]))
        axes[camera_idx, 0].imshow(as_image(history_tensor[age_index]))
        axes[camera_idx, 0].set_title(
            f"{key.split('.')[-1]} most-read history (-{diagnostic['history_seconds'][age_index]:g}s)"
        )
        axes[camera_idx, 0].axis("off")

        current = as_image(obs[key])
        axes[camera_idx, 1].imshow(current)
        axes[camera_idx, 1].set_title(f"{key.split('.')[-1]} current")
        axes[camera_idx, 1].axis("off")

        n_patches = patch_mass.shape[-1]
        side = round(n_patches**0.5)
        if side * side != n_patches:
            raise ValueError(f"Temporal patch overlay needs a square patch grid, got {n_patches} patches.")
        heat = torch.as_tensor(patch_mass[camera_idx]).view(1, 1, side, side)
        heat = functional.interpolate(
            heat, size=current.shape[:2], mode="bilinear", align_corners=False
        )[0, 0].numpy()
        axes[camera_idx, 2].imshow(current)
        overlay = axes[camera_idx, 2].imshow(heat, cmap="RdBu_r", alpha=0.62, norm=norm)
        overlay_images.append(overlay)
        axes[camera_idx, 2].set_title("current-frame patches reading history")
        axes[camera_idx, 2].axis("off")

    score = float(diagnostic["temporal_mass"].mean())
    bar = fig.colorbar(
        overlay_images[-1], ax=axes[:, 2].tolist(), fraction=0.025, pad=0.02, label="past mass"
    )
    bar.ax.axhline(uniform_mass, color="#111111", linewidth=1.3)
    bar.ax.text(
        0.5, uniform_mass, "uniform", transform=bar.ax.get_yaxis_transform(), ha="center",
        va="bottom", fontsize=7.5, color="#111111",
    )
    fig.suptitle(
        f"Temporal-read p{percentile} — episode {ep_idx}, frame {frame_idx}  |  "
        f"mean mass={score:.4f} ({score / max(uniform_mass, 1e-12):.2f}× uniform baseline)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def run(adapter, dataset, cfg, output_dir: str) -> None:
    memory_cfg = getattr(cfg.policy, "memory", None)
    image_history_keys = (
        [key for key in memory_cfg.history_keys if "images" in key] if memory_cfg is not None else []
    )
    if not image_history_keys or memory_cfg.history_num_samples <= 0:
        logging.info("[mem_temporal_attention] no image history configured — skipping.")
        return

    makedirs(output_dir)
    p = cfg.probe_parameters
    fps = cfg.env.fps
    layers = _temporal_layer_indices(adapter.policy)
    samples = sample_episodes_evenly(
        dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )

    temporal_mass_rows: list[np.ndarray] = []
    head_age_rows: list[np.ndarray] = []
    patch_mass_rows: list[np.ndarray] = []
    frame_meta: list[dict] = []
    camera_keys: list[str] | None = None
    history_seconds = np.asarray(
        [memory_cfg.history_window_seconds * (memory_cfg.history_num_samples - i) / memory_cfg.history_num_samples
         for i in range(memory_cfg.history_num_samples)],
        dtype=np.float32,
    )

    adapter._set_probe_cuda_graph_enabled(False)
    try:
        for ep_idx, frame_idx, global_idx in samples:
            # The temporal read lives in the ViT, so the prompt clauses do not reach
            # it; the frame builder is used for the image history and for a prompt
            # consistent with the other probes.
            frame = probe_frame_inputs(dataset, cfg, global_idx, int(cfg.policy.chunk_size))
            obs, task_str = frame["obs"], frame["task"]
            batch = adapter._make_batch(
                obs, task_str, subtask=frame["subtask"], metadata=frame["metadata"]
            )
            if "history_images" not in batch:
                continue

            model_inputs = adapter.policy._model_inputs(batch)
            _MEM_TEMPORAL_CAPTURE["records"].clear()
            _MEM_TEMPORAL_CAPTURE["enabled"] = True
            try:
                with torch.no_grad():
                    adapter.policy._run_prefix_backbone(model_inputs)
            finally:
                _MEM_TEMPORAL_CAPTURE["enabled"] = False
            records = list(_MEM_TEMPORAL_CAPTURE["records"])
            _MEM_TEMPORAL_CAPTURE["records"].clear()
            if len(records) != len(layers):
                logging.warning(
                    f"[mem_temporal_attention] expected {len(layers)} temporal records, "
                    f"got {len(records)} — skipping."
                )
                continue

            means = torch.stack([record["mean"] for record in records]).float().cpu().numpy()
            head_age = torch.stack([record["by_bc_head_age"] for record in records]).float().cpu()
            patch_mass = torch.stack([record["by_bc_patch"] for record in records]).float().cpu()
            n_cameras = int(batch["history_images"].shape[1])
            if int(head_age.shape[1]) != n_cameras:
                logging.warning(
                    "[mem_temporal_attention] expected probe batch size 1: "
                    f"capture BC={head_age.shape[1]}, cameras={n_cameras} — skipping."
                )
                continue
            # Use the exact processor packing order so camera labels and spatial
            # overlays cannot be silently swapped when memory.history_keys has a
            # different order from policy.image_keys.
            current_camera_keys = adapter._image_keys_for_obs(obs)
            if len(current_camera_keys) != n_cameras:
                logging.warning("[mem_temporal_attention] could not align capture cameras to observation keys.")
                continue
            if camera_keys is None:
                camera_keys = current_camera_keys
            elif camera_keys != current_camera_keys:
                logging.warning("[mem_temporal_attention] camera order changed across frames — skipping frame.")
                continue

            temporal_mass_rows.append(means)
            head_age_rows.append(head_age.numpy())
            patch_mass_rows.append(patch_mass.numpy())
            frame_meta.append(
                {
                    "episode_idx": int(ep_idx),
                    "frame_idx": int(frame_idx),
                    "global_idx": int(global_idx),
                }
            )
    finally:
        _MEM_TEMPORAL_CAPTURE["enabled"] = False
        _MEM_TEMPORAL_CAPTURE["records"].clear()
        adapter._restore_probe_cuda_graph_enabled()

    if not temporal_mass_rows or camera_keys is None:
        logging.warning("[mem_temporal_attention] no frames produced measurements.")
        return

    temporal_mass = np.stack(temporal_mass_rows)
    head_age = np.stack(head_age_rows)
    patch_mass = np.stack(patch_mass_rows)
    n_patches = int(patch_mass.shape[-1])
    n_ages = int(head_age.shape[-1])
    # The temporal softmax normalises over the K = n_ages + 1 timesteps (the past plus
    # the query's own), and nothing else: an indifferent read leaves n_ages/(n_ages + 1)
    # on history. Spatial keys live in their own softmax and do not enter this null.
    uniform_mass = n_ages / (n_ages + 1)
    camera_names = [key.split(".")[-1] for key in camera_keys]

    quantiles = np.percentile(temporal_mass, [10, 25, 50, 75, 90], axis=0)
    layer_summary = {
        str(layer): {
            "mean": float(temporal_mass[:, index].mean()),
            "std": float(temporal_mass[:, index].std()),
            "p10": float(quantiles[0, index]),
            "p25": float(quantiles[1, index]),
            "p50": float(quantiles[2, index]),
            "p75": float(quantiles[3, index]),
            "p90": float(quantiles[4, index]),
            "mean_enrichment_vs_uniform": float(temporal_mass[:, index].mean() / uniform_mass),
        }
        for index, layer in enumerate(layers)
    }
    age_mean = head_age.mean(axis=(0, 2, 3))
    age_entropy = _age_entropy(head_age)
    shape = _age_shape(head_age, history_seconds)
    episodes = sorted({meta["episode_idx"] for meta in frame_meta})
    provenance = _provenance(
        dataset, frame_meta, layers, camera_names, int(head_age.shape[3]), n_patches, history_seconds
    )
    summary = {
        "dataset": dataset_display_name(dataset),
        "episodes": episodes,
        "n_episodes": len(episodes),
        "provenance": provenance,
        "layers": layers,
        "cameras": camera_names,
        "history_seconds_oldest_to_newest": history_seconds.tolist(),
        "uniform_temporal_mass": float(uniform_mass),
        "mean_temporal_mass": float(temporal_mass.mean()),
        "mean_enrichment_vs_uniform": float(temporal_mass.mean() / uniform_mass),
        "max_layer_enrichment_vs_uniform": float(temporal_mass.mean(axis=0).max() / uniform_mass),
        "mean_normalized_age_entropy": float(age_entropy.mean()),
        "n_frames": len(frame_meta),
        "n_heads": int(head_age.shape[3]),
        "n_patches": n_patches,
        "per_layer_distribution": layer_summary,
        "mean_age_mass_by_layer": age_mean.tolist(),
        "mean_age_share_by_layer": (
            age_mean / np.clip(age_mean.sum(axis=-1, keepdims=True), 1e-12, None)
        ).tolist(),
        "mean_normalized_age_entropy_by_layer": age_entropy.tolist(),
        # e-folds per second of history age: negative is recency, positive prefers the
        # oldest frame in the window. Agreement guards the flat rows — see _age_shape.
        "age_log_slope_per_second_by_layer": shape["slope"].tolist(),
        "age_head_agreement_by_layer": shape["agreement"].tolist(),
        "age_recency_index_by_layer_head": shape["head_recency"].tolist(),
        "per_frame": [
            meta
            | {
                "temporal_mass_by_layer": temporal_mass[index].tolist(),
                "enrichment_vs_uniform_by_layer": (temporal_mass[index] / uniform_mass).tolist(),
            }
            for index, meta in enumerate(frame_meta)
        ],
    }
    with open(os.path.join(output_dir, "temporal_attention.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(output_dir, "temporal_attention_data.npz"),
        temporal_mass=temporal_mass,
        head_age_mass=head_age,
        patch_temporal_mass=patch_mass,
        age_log_slope=shape["slope"],
        age_head_agreement=shape["agreement"],
        age_recency_index_by_head=shape["head_recency"],
        layers=np.asarray(layers),
        camera_names=np.asarray(camera_names),
        history_seconds=history_seconds,
        episode_idx=np.asarray([meta["episode_idx"] for meta in frame_meta]),
        frame_idx=np.asarray([meta["frame_idx"] for meta in frame_meta]),
        global_idx=np.asarray([meta["global_idx"] for meta in frame_meta]),
    )

    plot_path = os.path.join(output_dir, "temporal_attention.png")
    _render_distribution(
        temporal_mass,
        head_age,
        shape,
        layers,
        camera_names,
        history_seconds,
        frame_meta,
        uniform_mass,
        provenance,
        plot_path,
    )

    examples_dir = os.path.join(output_dir, "examples")
    makedirs(examples_dir)
    scores = temporal_mass.mean(axis=1)
    # The overlay draws the layer mean, so its scale is set from that same reduction —
    # percentiles of the raw per-layer array describe a population no panel ever shows.
    drawn_patch_mass = patch_mass.mean(axis=1)
    patch_range = tuple(float(value) for value in np.percentile(drawn_patch_mass, [2, 98]))
    examples: list[tuple[int, str]] = []
    for percentile, index in _percentile_examples(scores):
        diagnostic = {
            "global_idx": frame_meta[index]["global_idx"],
            "camera_keys": camera_keys,
            "history_seconds": history_seconds,
            "temporal_mass": temporal_mass[index],
            "head_age": head_age[index],
            "patch_mass": patch_mass[index],
        }
        meta = frame_meta[index]
        filename = (
            f"frame_p{percentile:02d}_ep{meta['episode_idx']:04d}_fr{meta['frame_idx']:06d}.png"
        )
        _render_spatial_example(
            dataset,
            memory_cfg,
            fps,
            diagnostic,
            percentile,
            uniform_mass,
            patch_range,
            os.path.join(examples_dir, filename),
        )
        examples.append((percentile, filename))

    _write_manifest(output_dir, summary, examples)

    mean_enrichment = temporal_mass.mean(axis=0) / uniform_mass
    logging.info(
        f"[mem_temporal_attention] n={len(frame_meta)}  uniform baseline={uniform_mass:.4f}  "
        f"per-layer mean enrichment={[round(float(value), 2) for value in mean_enrichment]}"
    )

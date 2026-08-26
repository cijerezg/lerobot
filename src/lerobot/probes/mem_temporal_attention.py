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
the single mean into the four questions it conflates:

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
* **Whether "where" is about content at all.** Everything above is measured from attention
  mass, and mass on an age says the slot was read without saying it was read *for what is
  in it*. Mass piled on the two ends of a sequence is also the signature of an attention
  sink, which is a property of position and would survive any content whatsoever. Two
  controls settle it, both re-running the same frame with only the history tensor changed:

  ``constant``   the newest history frame copied into every slot, so content is identical
                 across ages by construction and any surviving age structure is positional;
  ``shuffled``   the same $T$ frames permuted across slots with the age embedding left in
                 place, so the question becomes whether the preference follows the frame or
                 the slot.

  Neither is redundant. ``constant`` is the clean positional null but is off-distribution —
  the model never saw $T$ identical frames in training, so a flat result there could be a
  reaction to degenerate input rather than to the absence of temporal content. ``shuffled``
  keeps the input in-distribution and cannot be dismissed that way, but it leaves content
  present and so cannot isolate position on its own. Read them together: a profile that
  collapses under both is content-driven; one that survives ``constant`` is positional.

  A weight is not a contribution, either. The block's output for the current frame is
  $\sum_k w_k v_k$, so a key with a small value vector adds nothing however hard it was
  attended, and parking probability on such a key is exactly what an attention sink does
  with the mass softmax forces it to spend. Every number above is a $w$. The probe also
  records $w_k\lVert v_k\rVert$ over the same keys — free, one norm per temporal layer —
  and reports history's share of that *delivered* magnitude beside the attended mass $m$.
  Where $m$ is substantial and the delivered share is not, the attention on history never
  reached the output and no age profile drawn from $w$ means anything. These are the
  magnitudes of the additive terms and not an attribution of $\lVert\text{out}\rVert$: the
  terms are vectors and can cancel, so the delivered share bounds what history can have
  contributed rather than measuring what it did.

  Reported as *survival*, per layer: the age profile's total deviation from the uniform
  share under the control, over the same deviation under the real history. 1.0 means the
  entire shape is reproduced with no temporal information present and the age profile is
  not evidence about history; 0.0 means the shape requires the real frames. Costs two extra
  forwards per frame and is switched off with
  ``probe_parameters.mem_temporal_positional_control``.

Each question is answered at the level it lives at, so the panels are not redundant views
of one number: level is per frame, direction is per layer, consensus is per head, and
whether any of it is about content is the control.

This probe establishes only that history is *read*. Whether reading it helps the action
is ``mem_history_influence``.

Writes:

* ``temporal_attention.png`` — frame/layer distributions, age preference, head and
  camera specialization, the check that the age preference is not head cancellation, how
  much of the age profile survives the two positional controls, and whether the attention
  on history delivers anything once value magnitudes are counted;
* ``examples/frame_p*.png`` — low/median/high temporal-read examples with the most-read
  history frame, current image, and spatial temporal-read overlay per camera;
* ``mistakes/*.png`` — complete per-age attention sequences for labelled mistakes and
  their immediate recovery frames; red titles mark ages inside the annotated event;
* ``temporal_attention_data.npz`` — the per-frame/layer/camera/head/age/patch arrays;
* ``temporal_attention.json`` — human-readable summary and per-frame layer values.
"""

import json
import logging
import os
import sys
from bisect import bisect_right

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
    build_episode_index,
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
        "interior_min": _interior_min_fraction(head_age),
    }


def _interior_min_fraction(head_age: np.ndarray) -> np.ndarray:
    r"""Fraction of heads whose age profile bottoms out inside the window, per layer.

    The dip that the slope $b$ cannot see. $b$ fits a straight line to $\log$ share, so a
    profile high at both ends and low in the middle fits with $b\approx 0$ and reads as
    indifference; head agreement is newest-minus-oldest and is blind to the middle by
    construction. This asks the one question neither can: is the smallest share at an end
    of the window, or inside it?

    Location, not curvature. A curvature statistic — the endpoints against the middle,
    with or without the fitted exponential removed — fires on a steeply convex monotone
    profile, because a convex curve's endpoints average above its interior whether or not
    anything dips. Measured on rebot ckpt400 it scored L3 (0.088 → 0.422, strictly
    increasing, no dip anywhere) at +0.12 raw and +0.26 after removing the fit, which is
    the same sign and a comparable size to layers that genuinely trough. The argmin
    location has no such failure mode: L3 scores 0.00 and the layers with a real dip score
    1.00.

    Taken per head and then averaged, for the reason every other per-head number here is:
    the head mean fills in the age a single head avoids.
    """
    if head_age.shape[-1] <= 2:  # with two ages every minimum is at an end
        return np.zeros(head_age.shape[1], dtype=np.float32)
    per_head = head_age.mean(axis=(0, 2))  # (layers, heads, ages)
    location = per_head.argmin(axis=-1)
    interior = (location > 0) & (location < head_age.shape[-1] - 1)
    return interior.mean(axis=-1).astype(np.float32)


def _profile_survival(real: np.ndarray, control: np.ndarray) -> np.ndarray:
    r"""How much of each layer's age profile the control reproduces without its content.

    With $s$ the age share renormalised over the $T$ ages and $u = 1/T$ the indifferent
    share, the shape's size is its total deviation $D=\sum_{\Delta t}|s(\Delta t)-u|$, and

    $$\mathrm{survival} = D_{\text{control}} / D_{\text{real}} .$$

    1.0 is the whole profile reproduced with no temporal information present, which makes
    the age profile a fact about position and not about history. 0.0 is a profile that
    needs the real frames. Deviation rather than correlation, because a control that
    flattens the profile is the outcome of interest and correlation is undefined on a flat
    profile; deviation degrades to 0 there instead. Values above 1 are possible — a
    control can be *more* peaked than the real read — and are not clipped, since a control
    that overshoots is a result and not an error.

    Layers whose real profile is already flat have no shape to survive and report NaN.
    The floor is on $D_{\text{real}}$ and is deliberately well above zero: a merely nonzero
    denominator is not enough, because the ratio is then dividing one small number by
    another and reports a difference that is not there. It is set at one quarter of the
    uniform share, i.e. the deviation of a profile whose ages sit an average of 12.5% of
    $u$ away from indifference — below that a layer is flat for reading purposes. Measured
    on rebot ckpt400 this is what separates L23 ($D=0.286$, a real shape) from L15
    ($D=0.021$, flat), where the unguarded ratio returned 1.79 and drew a bar taller than
    the panel for a layer with no age preference at all.
    """
    n_ages = real.shape[-1]
    uniform = 1.0 / n_ages
    real_share = real / np.clip(real.sum(axis=-1, keepdims=True), 1e-12, None)
    control_share = control / np.clip(control.sum(axis=-1, keepdims=True), 1e-12, None)
    real_dev = np.abs(real_share - uniform).sum(axis=-1)
    control_dev = np.abs(control_share - uniform).sum(axis=-1)
    floor = 0.25 * uniform
    return np.where(real_dev > floor, control_dev / np.clip(real_dev, 1e-12, None), np.nan)


def _delivered(contribution: np.ndarray) -> dict[str, np.ndarray]:
    r"""Split the attention's *delivered* magnitude into what history sent and where from.

    ``contribution`` is $w_k\|v_k\|$ per frame, layer, camera, head and key, the keys being
    the $T$ history ages oldest → newest with the current frame last. A weight is not a
    contribution: the block's output is $\sum_k w_k v_k$, so a key with a small value
    vector adds nothing however hard it was attended, and parking probability on such a
    key is what an attention sink is. Two numbers follow:

    ``past_share``   history's share of the delivered magnitude, per frame and layer — the
                     counterpart of the attended past mass $m$. Where $m$ sits near its
                     $T/(T+1)$ null and this sits far below it, the attention on history is
                     not reaching the output and the level is a sink, not a read.
    ``age_share``    the delivered profile over the $T$ ages, per layer, renormalised the
                     same way the attended age profile is, so the two are read on one axis.
                     A shape here that the attended profile does not have — or an attended
                     shape that vanishes here — is the age panel measuring parking.

    Magnitudes of the additive terms, not an attribution of $\|\text{out}\|$: the terms are
    vectors and may cancel, so this bounds what a key can have delivered rather than
    measuring what it did. The exact statement needs an ablation.
    """
    total = contribution.sum(axis=-1)
    past = contribution[..., :-1].sum(axis=-1)
    past_share = np.divide(past, total, out=np.zeros_like(past), where=total > 1e-12)
    age = contribution[..., :-1].mean(axis=(0, 2, 3))
    return {
        # contribution is (frames, layers, cameras, heads, keys); average the camera and
        # head axes away and keep (frames, layers), matching temporal_mass.
        "past_share": past_share.mean(axis=(2, 3)),
        "age_share": age / np.clip(age.sum(axis=-1, keepdims=True), 1e-12, None),
    }


def _delivery_panel(ax, attended: np.ndarray, delivered: np.ndarray, layer_labels, uniform_mass) -> None:
    """Attended past mass against delivered past share, per layer."""
    x = np.arange(len(layer_labels))
    ax.bar(x - 0.2, attended, width=0.4, color="#457B9D", label=r"attended  $m=\sum_k w_k$")
    ax.bar(x + 0.2, delivered, width=0.4, color="#B5651D", label=r"delivered  $\sum_k w_k\|v_k\|$")
    for position, (a, b) in enumerate(zip(attended, delivered, strict=True)):
        ax.text(position - 0.2, a + 0.012, f"{a:.2f}", ha="center", fontsize=6.5)
        ax.text(position + 0.2, b + 0.012, f"{b:.2f}", ha="center", fontsize=6.5)
    ax.axhline(uniform_mass, color="#E76F51", linestyle="--", linewidth=1.0)
    ax.annotate(
        f"uniform over time = {uniform_mass:.2f}",
        (len(layer_labels) - 0.5, uniform_mass),
        xytext=(-2, 4),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color="#E76F51",
    )
    ax.set_xticks(x, layer_labels)
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel("ViT temporal layer")
    ax.set_ylabel("share of the current query's read")
    ax.set_title("Does the attention on history deliver anything?")
    ax.legend(fontsize=7, loc="lower left", ncol=1)
    panel_caption(
        ax,
        [
            r"The block's output for the current frame is $\sum_k w_k v_k$ over the $T$ history keys and",
            r"the current one, so a key delivers its weight *times the size of its value vector*.",
            r"Blue is the weight alone — the same $m$ the first panel plots, and every age panel in",
            r"this figure is a slice of it. Brown replaces $w_k$ with $w_k\|v_k\|$ and asks what actually",
            r"left the block. Brown far below blue is an attention sink: probability parked on keys",
            r"that carry nothing, which softmax must spend somewhere and which no age profile can",
            r"distinguish from a read. Brown tracking blue means the mass is moving content and the",
            r"level is a real read whatever the age profile turned out to be about.",
            r"Term magnitudes, not an attribution: $\sum_k w_k v_k$ is a vector sum and can cancel, so",
            r"brown bounds what history delivered rather than measuring it.",
        ],
    )


def _write_manifest(
    output_dir: str,
    summary: dict,
    examples: list[tuple[int, str]],
    mistake_examples: list[tuple[dict, str]],
) -> dict:
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
    panels += [
        Panel(
            f"mistakes/{filename}",
            f"Labelled {', '.join(meta['mistake_types'])}: episode {meta['episode_idx']}, "
            f"frame {meta['frame_idx']} — full temporal read by history age",
            how=(
                "Each history frame carries its own temporal-attention overlay; red titles are "
                "the exact sampled ages inside meta/mistakes.parquet. The final column is the "
                "current image. Compare labelled and unlabelled ages within the same row before "
                "comparing across examples, because camera and layer scales are shared globally."
            ),
        )
        for meta, filename in mistake_examples
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


def _load_mistake_spans(dataset) -> list[dict]:
    """Validated mistake events, preserving the semantic type and note.

    The Parquet indices are global dataset indices and use half-open intervals. Missing
    annotations are ordinary for a generic probe dataset, so this is optional rather than
    a hard requirement like training with ``metadata_enabled``.
    """
    from lerobot.rl.offline_dataset_utils import load_metadata_rows

    try:
        _episode_rows, rows = load_metadata_rows(dataset.root)
    except (FileNotFoundError, OSError):
        return []
    return [
        {
            "episode_idx": int(row["episode_index"]),
            "from_index": int(row["from_index"]),
            "to_index": int(row["to_index"]),
            "mistake_type": str(row.get("mistake_type", "mistake")),
            "note": str(row.get("note", "")),
        }
        for row in rows
        if bool(row.get("mistake", False)) and int(row["to_index"]) > int(row["from_index"])
    ]


def _snap_global_to_episode_grid(dataset, indices: list[int], target: int, stride: int) -> tuple[int, int]:
    """Nearest preceding in-episode frame on the stored image/depth grid."""
    pos = max(bisect_right(indices, int(target)) - 1, 0)
    global_idx = indices[pos]
    frame_idx = int(dataset.hf_dataset[global_idx]["frame_index"].item())
    pos = max(pos - frame_idx % max(int(stride), 1), 0)
    global_idx = indices[pos]
    frame_idx = int(dataset.hf_dataset[global_idx]["frame_index"].item())
    return frame_idx, global_idx


def _mistake_focus_samples(dataset, spans: list[dict], fps: float, stride: int) -> list[tuple[int, int, int]]:
    """One late-event and one recovery view per mistake, independent of uniform luck.

    The late-event anchor makes the newest history slots overlap the failure when its
    duration permits. The recovery anchor is one second after the annotated end, so the
    failure lies in history while the current image shows what happened next.
    """
    ep_to_indices = build_episode_index(dataset)
    samples: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for span in spans:
        ep_idx = int(span["episode_idx"])
        indices = ep_to_indices.get(ep_idx)
        if not indices:
            continue
        targets = (int(span["to_index"]) - 1, int(span["to_index"]) + round(float(fps)))
        for target in targets:
            frame_idx, global_idx = _snap_global_to_episode_grid(
                dataset, indices, min(max(target, indices[0]), indices[-1]), stride
            )
            if global_idx not in seen:
                samples.append((ep_idx, frame_idx, global_idx))
                seen.add(global_idx)
    return samples


def _merge_samples(*groups: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Stable episode/time order with duplicate global frames removed."""
    by_global = {sample[2]: sample for group in groups for sample in group}
    return sorted(by_global.values(), key=lambda sample: (sample[0], sample[1], sample[2]))


def _mistake_context(
    episode_idx: int,
    global_idx: int,
    spans: list[dict],
    history_offsets: np.ndarray,
) -> dict:
    """Which labelled event is current, and which exact history slots contain one."""
    age_mask = np.zeros(len(history_offsets), dtype=bool)
    events: list[dict] = []
    for span in spans:
        if int(span["episode_idx"]) != int(episode_idx):
            continue
        start, stop = int(span["from_index"]), int(span["to_index"])
        current = start <= int(global_idx) < stop
        mask = np.asarray(
            [start <= int(global_idx) - int(offset) < stop for offset in history_offsets],
            dtype=bool,
        )
        if current or mask.any():
            age_mask |= mask
            events.append(
                {
                    "from_index": start,
                    "to_index": stop,
                    "mistake_type": span["mistake_type"],
                    "note": span["note"],
                    "current": bool(current),
                    "history_age_indices": np.flatnonzero(mask).astype(int).tolist(),
                }
            )
    return {
        "mistake_current": any(event["current"] for event in events),
        "mistake_in_history": bool(age_mask.any()),
        "mistake_history_age_mask": age_mask.tolist(),
        "mistake_types": sorted({event["mistake_type"] for event in events}),
        "mistake_events": events,
    }


def _mistake_age_enrichment(
    head_age: np.ndarray, mistake_age_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Attention share on labelled ages divided by their share of available slots.

    ``head_age`` is (frames, layers, cameras, heads, ages), while the boolean mask is
    (frames, ages). An indifferent age profile scores 1 regardless of whether one or
    four slots overlap the event. Frames with no labelled history age report NaN.
    """
    age_mass = head_age.mean(axis=(2, 3))
    age_share = age_mass / np.clip(age_mass.sum(axis=-1, keepdims=True), 1e-12, None)
    selected_share = (age_share * mistake_age_mask[:, None, :]).sum(axis=-1)
    slot_share = mistake_age_mask.mean(axis=-1, keepdims=True)
    enrichment = np.full_like(selected_share, np.nan)
    np.divide(selected_share, slot_share, out=enrichment, where=slot_share > 0)
    return selected_share, enrichment


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
    panel_caption(
        ax,
        [
            r"$m=\sum_{\tau} w_\tau$, the share of one query's temporal softmax landing on history",
            r"rather than on its own timestep, averaged over patches, heads and cameras. One box",
            r"per layer over the sampled frames; box = quartiles, whiskers = 1.5 IQR.",
            r"Dashed line is $T/(T+1)$, what a layer with no preference between timesteps gives the",
            r"past. Only the distance from it is evidence, and below it is a real reading:",
            r"the layer is holding attention on its own frame.",
        ],
    )


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
    panel_caption(
        ax,
        [
            r"The same $m$, one row per sampled frame, ordered by episode; white rules are episode",
            r"boundaries. Vertical stripes mean the layer sets the level and the scene does not",
            r"change it — a fixed schedule, not retrieval. Horizontal structure is the opposite:",
            r"some frames pull more history than others. Read the columns, not the rows: the",
            r"frame axis is a concatenation of episodes, not one trajectory.",
            r"Colour is clipped to the middle 96% of $m$ and so shows contrast, not level: the",
            r"level is the boxplot to the left, and the baseline is in the figure title. The",
            r"range on the bar is set from this run's data — colours do not carry across runs.",
        ],
    )


def _age_panel(ax, shape, layers, history_seconds, colors) -> None:
    uniform_share = 1.0 / len(history_seconds)
    x = -history_seconds
    for index, layer in enumerate(layers):
        ax.plot(
            x,
            shape["share"][index],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=colors[index],
            # b and the dip fraction together: b alone reads a trough as indifference.
            label=f"L{layer}  $b$={shape['slope'][index]:+.2f}/s  dip={shape['interior_min'][index]:.0%}",
        )
    ax.axhline(uniform_share, color="#E76F51", linestyle="--", linewidth=1.0)
    ax.annotate(
        f"uniform over ages = {uniform_share:.2f}",
        (x[0], uniform_share),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=7.5,
        color="#E76F51",
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
    panel_caption(
        ax,
        [
            r"Past mass renormalised to sum to 1 over the $T$ ages, so this is *where* the read",
            r"lands and carries nothing about how much was read — that is the panel to its left.",
            r"Log axis because the profile is close to exponential: $\log\,$share$\,=a+b\,\Delta t$, and $b$ is in",
            r"the legend in e-folds per second. $b<0$ is recency, $b>0$ prefers the oldest frame",
            r"in the window, $b\approx0$ sits on the dashed uniform line.",
            r"$b$ is a line fit and cannot see a trough — high at both ends and low in the middle fits",
            r"$b\approx0$ too. *dip* is the share of that layer's heads whose least-read age is inside",
            r"the window, and is what separates the two: read it before calling a flat $b$ indifference.",
            r"These curves average over heads, so a flat one is genuine indifference only where the",
            r"agreement bar (bottom right) is high; where it is low the flatness is cancellation.",
        ],
    )


def _head_panel(fig, ax, head_mean, layer_labels, n_heads) -> None:
    image = ax.imshow(head_mean, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(n_heads), [str(i) for i in range(n_heads)])
    ax.set_yticks(range(len(layer_labels)), layer_labels)
    ax.set_xlabel("ViT head")
    ax.set_ylabel("ViT temporal layer")
    ax.set_title("Do individual heads specialise?")
    fig.colorbar(image, ax=ax, fraction=0.046, label="past mass")
    panel_caption(
        ax,
        [
            r"$m$ per head, averaged over frames, patches and cameras. A uniform row means the",
            r"layer spends the same budget in every head; a row with dark cells means some heads",
            r"stay on the current frame while others do the reading. This is the level per head —",
            r"whether they also disagree about *which* age to read is the bottom-right panel.",
            r"Weigh the spread along a row here against the frame-to-frame spread in the panel",
            r"above it: whichever dominates is what actually sets the read, head or scene.",
        ],
    )


def _camera_panel(fig, ax, camera_mean, layer_labels, camera_names) -> None:
    image = ax.imshow(camera_mean, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(camera_names)), camera_names, rotation=20, ha="right")
    ax.set_yticks(range(len(layer_labels)), layer_labels)
    ax.set_ylabel("ViT temporal layer")
    ax.set_title("Does the camera change the read?")
    fig.colorbar(image, ax=ax, fraction=0.046, label="past mass")
    panel_caption(
        ax,
        [
            r"$m$ per camera, averaged over frames, patches and heads. The temporal step is shared",
            r"weights applied per camera, so a difference here comes from the images alone:",
            r"a wrist view that moves with the arm has less in common with its own past than a",
            r"fixed overhead view does. Columns that match to two decimals mean the camera is",
            r"not a factor and this panel can be ignored for that layer.",
        ],
    )


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
    panel_caption(
        ax,
        [
            r"Both bars are computed per head and then averaged; the age panel above averages the",
            r"heads first, so these recover exactly what that average destroys. Entropy is one head's",
            r"mass over the $T$ ages, normalised — how peaked that head is, carrying no direction.",
            r"Agreement is $|\overline{r}|/\overline{|r|}$ over heads, $r_h$ = newest-age share minus oldest — direction",
            r"only, carrying no magnitude: 1 when the heads pull the same way, 0 under cancellation.",
            r"So a flat row above is indifference only beside a high bar here; beside a low bar it is",
            r"two opposed head populations. Low on both is heads each reading a different moment.",
        ],
    )


CONTROL_COLORS = {"constant": "#E76F51", "shuffled": "#2A9D8F"}


def _control_panel(ax, survival: dict[str, np.ndarray], layer_labels) -> None:
    """How much of each age profile the positional controls reproduce."""
    x = np.arange(len(layer_labels))
    names = [name for name in ("constant", "shuffled") if name in survival]
    width = 0.8 / max(len(names), 1)
    for index, name in enumerate(names):
        offset = (index - (len(names) - 1) / 2) * width
        values = np.nan_to_num(survival[name], nan=0.0)
        ax.bar(x + offset, values, width=width * 0.92, color=CONTROL_COLORS[name], label=name)
        # A NaN is "this layer had no shape to survive", which is not a zero bar.
        for position, (value, raw) in enumerate(zip(values, survival[name], strict=True)):
            if np.isnan(raw):
                ax.text(position + offset, 0.02, "n/a", ha="center", fontsize=6.5, color="#777777")
            elif value >= 0.5:
                ax.text(position + offset, value + 0.03, f"{raw:.2f}", ha="center", fontsize=6.5)
    ax.axhline(1.0, color="#111111", linewidth=1.0, linestyle="--")
    ax.annotate(
        "real history = 1.0",
        (len(layer_labels) - 0.5, 1.0),
        xytext=(-2, 4),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
    )
    ax.set_xticks(x, layer_labels)
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel("ViT temporal layer")
    ax.set_ylabel("share of the real profile's deviation")
    ax.set_title("Does the age profile need the frames' content?")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    panel_caption(
        ax,
        [
            r"Same frames, same sampling, only the history tensor changed. *constant* copies the",
            r"newest history frame into every slot, so the content is the same at every age and a",
            r"bar near 1 says the profile above is positional — an attention sink at the ends of",
            r"the sequence — and not a fact about history. *shuffled* permutes the real frames",
            r"across slots with the age embedding untouched, asking whether the preference follows",
            r"the frame or the slot; it stays in distribution, which *constant* does not, so a",
            r"collapse under *constant* alone is weaker evidence than a collapse under both.",
            r"Height is $\sum_{\Delta t}|s-1/T|$ under the control over the same sum under the real",
            r"read, so it is about shape and not level. n/a is a layer with no shape to lose.",
        ],
    )


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
    survival: dict[str, np.ndarray],
    delivered: dict[str, np.ndarray] | None,
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

    # The control is a fourth column rather than a third row: it qualifies the age panel
    # directly above it, and reading order across the row is what carries that.
    columns = 4 if (survival or delivered) else 3
    fig = plt.figure(figsize=(6.7 * columns, 12.5))
    panels = fig.add_gridspec(
        2, columns, hspace=0.66, wspace=0.26, left=0.05, right=0.985, top=0.925, bottom=0.07
    )
    axes = np.array([[fig.add_subplot(panels[r, c]) for c in range(columns)] for r in range(2)])

    _spread_panel(axes[0, 0], temporal_mass, layer_labels, uniform_mass)
    _scene_panel(fig, axes[0, 1], temporal_mass, layer_labels, frame_meta)
    _age_panel(axes[0, 2], shape, layers, history_seconds, colors)
    _head_panel(fig, axes[1, 0], head_total.mean(axis=(0, 2)), layer_labels, n_heads)
    _camera_panel(fig, axes[1, 1], head_total.mean(axis=(0, 3)), layer_labels, camera_names)
    _selectivity_panel(axes[1, 2], _age_entropy(head_age), shape["agreement"], layer_labels)
    if columns == 4:
        if survival:
            _control_panel(axes[0, 3], survival, layer_labels)
        else:
            axes[0, 3].axis("off")
        if delivered is not None:
            _delivery_panel(
                axes[1, 3],
                temporal_mass.mean(axis=0),
                delivered["past_share"].mean(axis=0),
                layer_labels,
                uniform_mass,
            )
        else:
            axes[1, 3].axis("off")

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
        heat = functional.interpolate(heat, size=current.shape[:2], mode="bilinear", align_corners=False)[
            0, 0
        ].numpy()
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
        0.5,
        uniform_mass,
        "uniform",
        transform=bar.ax.get_yaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#111111",
    )
    fig.suptitle(
        f"Temporal-read p{percentile} — episode {ep_idx}, frame {frame_idx}  |  "
        f"mean mass={score:.4f} ({score / max(uniform_mass, 1e-12):.2f}× uniform baseline)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _render_mistake_sequence(
    dataset,
    memory_cfg,
    fps: float,
    diagnostic: dict,
    per_age_uniform: float,
    patch_age_range: tuple[float, float],
    output_path: str,
) -> None:
    """All history ages for a labelled event, with one spatial read map per age."""
    obs, _gt, _state, _subtask, _task, ep_idx, frame_idx = get_frame_data(
        dataset, diagnostic["global_idx"], 1
    )
    camera_keys = diagnostic["camera_keys"]
    history = assemble_frame_history(dataset, diagnostic["global_idx"], memory_cfg, fps, camera_keys)
    # Captured as (layers, cameras, query patches, ages); the picture is deliberately
    # the layer mean, matching the existing history-summed spatial examples.
    patch_age = diagnostic["patch_age"].mean(axis=0)
    head_age = diagnostic["head_age"]
    age_mass_by_camera = head_age.mean(axis=(0, 2))
    age_share_by_camera = age_mass_by_camera / np.clip(
        age_mass_by_camera.sum(axis=-1, keepdims=True), 1e-12, None
    )
    mistake_mask = np.asarray(diagnostic["mistake_history_age_mask"], dtype=bool)

    low, high = patch_age_range
    norm = TwoSlopeNorm(
        vcenter=per_age_uniform,
        vmin=min(low, per_age_uniform - 1e-4),
        vmax=max(high, per_age_uniform + 1e-4),
    )
    n_ages = len(diagnostic["history_seconds"])
    fig, axes = plt.subplots(
        len(camera_keys),
        n_ages + 1,
        figsize=(3.4 * (n_ages + 1), 3.6 * len(camera_keys)),
        squeeze=False,
    )
    overlays = []
    for camera_idx, key in enumerate(camera_keys):
        history_tensor = history[f"history.{key}"].squeeze(0)
        for age_idx, seconds in enumerate(diagnostic["history_seconds"]):
            image = as_image(history_tensor[age_idx])
            n_patches = int(patch_age.shape[1])
            side = round(n_patches**0.5)
            if side * side != n_patches:
                raise ValueError(
                    f"Temporal patch-age overlay needs a square patch grid, got {n_patches} patches."
                )
            heat = torch.as_tensor(patch_age[camera_idx, :, age_idx]).view(1, 1, side, side)
            heat = functional.interpolate(heat, size=image.shape[:2], mode="bilinear", align_corners=False)[
                0, 0
            ].numpy()
            axes[camera_idx, age_idx].imshow(image)
            overlay = axes[camera_idx, age_idx].imshow(heat, cmap="RdBu_r", alpha=0.58, norm=norm)
            overlays.append(overlay)
            label = "  MISTAKE" if mistake_mask[age_idx] else ""
            axes[camera_idx, age_idx].set_title(
                f"{key.split('.')[-1]}  -{float(seconds):g}s{label}\n"
                f"past-age share={age_share_by_camera[camera_idx, age_idx]:.3f}",
                color="#C1121F" if mistake_mask[age_idx] else "#222222",
                fontweight="bold" if mistake_mask[age_idx] else "normal",
                fontsize=9,
            )
            axes[camera_idx, age_idx].axis("off")

        current = as_image(obs[key])
        axes[camera_idx, -1].imshow(current)
        current_label = "  MISTAKE NOW" if diagnostic["mistake_current"] else ""
        axes[camera_idx, -1].set_title(
            f"{key.split('.')[-1]}  current{current_label}",
            color="#C1121F" if diagnostic["mistake_current"] else "#222222",
            fontweight="bold" if diagnostic["mistake_current"] else "normal",
            fontsize=9,
        )
        axes[camera_idx, -1].axis("off")

    if overlays:
        bar = fig.colorbar(
            overlays[-1],
            ax=axes[:, :-1].ravel().tolist(),
            fraction=0.018,
            pad=0.012,
            label="attention weight on this history age",
        )
        bar.ax.axhline(per_age_uniform, color="#111111", linewidth=1.2)
    relation = []
    if diagnostic["mistake_current"]:
        relation.append("current inside event")
    if mistake_mask.any():
        relation.append("event visible in history")
    fig.suptitle(
        f"MEM temporal read around {', '.join(diagnostic['mistake_types'])} — "
        f"episode {ep_idx}, frame {frame_idx}\n{'; '.join(relation)}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _capture_temporal(adapter, batch: dict, n_layers: int) -> list[dict] | None:
    """One prefix forward with the temporal-attention capture on.

    Returns the per-layer records, or None when the capture did not produce one record
    per temporal layer — a shape the caller has to drop the frame over rather than pad.
    ``_model_inputs`` re-stashes the history on every call, and the stash is consume-once,
    so each condition must go through it again rather than reuse the previous inputs.
    """
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
    if len(records) != n_layers:
        logging.warning(
            f"[mem_temporal_attention] expected {n_layers} temporal records, got {len(records)} — skipping."
        )
        return None
    return records


def _control_batches(batch: dict, rng: np.random.Generator) -> dict[str, dict]:
    """The same batch with only ``history_images`` changed, one entry per control.

    ``history_images`` is ``(B, cameras, T, patches, patch_dim)``, oldest → newest.
    ``history_image_times`` and ``history_images_mask`` are deliberately left alone:
    times carries the age embedding, so holding it fixed is what makes the slot's
    *position* the only thing the model still knows, and the mask is one flag per sample
    rather than per slot (invalid slots are already repeat-padded at assembly time), so
    there is nothing per-age in it to permute.

    One permutation per frame, drawn from a stream seeded once, so the run averages over
    many orders rather than resting on a single lucky one. The same order goes to every
    camera: permuting cameras independently would leave a slot holding the top view at one
    age and the wrist view at another, which breaks temporal alignment across views rather
    than testing it.

    Known weakness of ``shuffled``: only the identity is rejected, not fixed points, and a
    uniform permutation has one fixed point in expectation at every length — so about one
    slot in five keeps its own frame and that survival is partly the real condition. Use a
    derangement if this control ever has to carry a claim on its own. It does not here:
    ``constant`` puts one frame in every slot and has no such loophole, and on rebot
    ckpt400 the two agree (0.98 and 1.00 at L23).
    """
    frames = batch["history_images"]
    n_ages = int(frames.shape[2])
    if n_ages < 2:
        return {}
    # Newest slot into every slot: identical content at every age, so anything the age
    # profile still says afterwards is said by position alone.
    constant = frames[:, :, -1:].expand_as(frames).clone()
    # A permutation that moves at least one frame; the identity would be a second copy
    # of the real condition wearing the name of a control.
    order = np.arange(n_ages)
    while n_ages > 1 and np.array_equal(order, np.arange(n_ages)):
        order = rng.permutation(n_ages)
    shuffled = frames[:, :, torch.as_tensor(order, device=frames.device)].clone()
    return {
        "constant": {**batch, "history_images": constant},
        "shuffled": {**batch, "history_images": shuffled},
    }


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
    image_stride = probe_image_stride(cfg)
    uniform_samples = sample_episodes_evenly(
        dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed, image_stride
    )
    all_mistake_spans = _load_mistake_spans(dataset)
    sampled_episodes = {sample[0] for sample in uniform_samples}
    mistake_spans = [span for span in all_mistake_spans if int(span["episode_idx"]) in sampled_episodes]
    mistake_samples = _mistake_focus_samples(dataset, mistake_spans, fps, image_stride)
    samples = _merge_samples(uniform_samples, mistake_samples)
    if mistake_spans:
        logging.info(
            f"[mem_temporal_attention] added {len(samples) - len(uniform_samples)} unique "
            f"mistake-focused frames for {len(mistake_spans)} labelled events."
        )

    control_on = bool(getattr(p, "mem_temporal_positional_control", False))
    temporal_mass_rows: list[np.ndarray] = []
    head_age_rows: list[np.ndarray] = []
    patch_mass_rows: list[np.ndarray] = []
    patch_age_rows: list[np.ndarray] = []
    control_head_age_rows: dict[str, list[np.ndarray]] = {"constant": [], "shuffled": []}
    contribution_rows: list[np.ndarray] = []
    # Seeded once, not per frame: the shuffle is meant to vary across frames so that no
    # single unlucky permutation stands in for "a different order".
    control_rng = np.random.default_rng(p.random_seed)
    frame_meta: list[dict] = []
    camera_keys: list[str] | None = None
    history_seconds = np.asarray(
        [
            memory_cfg.history_window_seconds
            * (memory_cfg.history_num_samples - i)
            / memory_cfg.history_num_samples
            for i in range(memory_cfg.history_num_samples)
        ],
        dtype=np.float32,
    )
    history_offsets = np.rint(history_seconds * float(fps)).astype(np.int64)

    adapter._set_probe_cuda_graph_enabled(False)
    try:
        for ep_idx, frame_idx, global_idx in samples:
            # The temporal read lives in the ViT, so the prompt clauses do not reach
            # it; the frame builder is used for the image history and for a prompt
            # consistent with the other probes.
            frame = probe_frame_inputs(dataset, cfg, global_idx, int(cfg.policy.chunk_size))
            obs, task_str = frame["obs"], frame["task"]
            batch = adapter._make_batch(obs, task_str, subtask=frame["subtask"], metadata=frame["metadata"])
            if "history_images" not in batch:
                continue

            records = _capture_temporal(adapter, batch, len(layers))
            if records is None:
                continue

            means = torch.stack([record["mean"] for record in records]).float().cpu().numpy()
            head_age = torch.stack([record["by_bc_head_age"] for record in records]).float().cpu()
            patch_mass = torch.stack([record["by_bc_patch"] for record in records]).float().cpu()
            patch_age = (
                torch.stack([record["by_bc_patch_age"] for record in records]).float().cpu()
                if all("by_bc_patch_age" in record for record in records)
                else None
            )
            # Absent on a checkpoint captured before the value-norm record existed.
            contribution = (
                torch.stack([record["by_bc_head_key_value"] for record in records]).float().cpu()
                if all("by_bc_head_key_value" in record for record in records)
                else None
            )
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
                logging.warning(
                    "[mem_temporal_attention] could not align capture cameras to observation keys."
                )
                continue
            if camera_keys is None:
                camera_keys = current_camera_keys
            elif camera_keys != current_camera_keys:
                logging.warning(
                    "[mem_temporal_attention] camera order changed across frames — skipping frame."
                )
                continue

            # Only after the real read has passed every shape check, so a frame is never
            # half-recorded: the control rows stay index-aligned with head_age_rows.
            control_rows: dict[str, np.ndarray] = {}
            if control_on:
                for name, control_batch in _control_batches(batch, control_rng).items():
                    control_records = _capture_temporal(adapter, control_batch, len(layers))
                    if control_records is None:
                        break
                    control_rows[name] = (
                        torch.stack([r["by_bc_head_age"] for r in control_records]).float().cpu().numpy()
                    )
                if len(control_rows) != len(control_head_age_rows):
                    logging.warning(
                        "[mem_temporal_attention] positional control capture failed — skipping frame."
                    )
                    continue

            temporal_mass_rows.append(means)
            if contribution is not None:
                contribution_rows.append(contribution.numpy())
            head_age_rows.append(head_age.numpy())
            patch_mass_rows.append(patch_mass.numpy())
            if patch_age is not None:
                patch_age_rows.append(patch_age.numpy())
            for name, row in control_rows.items():
                control_head_age_rows[name].append(row)
            frame_meta.append(
                {
                    "episode_idx": int(ep_idx),
                    "frame_idx": int(frame_idx),
                    "global_idx": int(global_idx),
                }
                | _mistake_context(ep_idx, global_idx, mistake_spans, history_offsets)
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
    patch_age = np.stack(patch_age_rows) if len(patch_age_rows) == len(frame_meta) else None
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
    control_head_age = {name: np.stack(rows) for name, rows in control_head_age_rows.items() if rows}
    survival = {
        name: _profile_survival(age_mean, rows.mean(axis=(0, 2, 3)))
        for name, rows in control_head_age.items()
    }
    control_interior_min = {name: _interior_min_fraction(rows) for name, rows in control_head_age.items()}
    contribution = np.stack(contribution_rows) if contribution_rows else None
    delivered = _delivered(contribution) if contribution is not None else None
    episodes = sorted({meta["episode_idx"] for meta in frame_meta})
    provenance = _provenance(
        dataset, frame_meta, layers, camera_names, int(head_age.shape[3]), n_patches, history_seconds
    )
    mistake_current = np.asarray([meta["mistake_current"] for meta in frame_meta], dtype=bool)
    mistake_in_history = np.asarray([meta["mistake_in_history"] for meta in frame_meta], dtype=bool)
    mistake_history_age_mask = np.asarray(
        [meta["mistake_history_age_mask"] for meta in frame_meta], dtype=bool
    )
    mistake_age_share, mistake_age_enrichment = _mistake_age_enrichment(head_age, mistake_history_age_mask)
    mistake_relevant = mistake_current | mistake_in_history
    clean = ~mistake_relevant

    def _group_mass(mask: np.ndarray) -> list[float] | None:
        return temporal_mass[mask].mean(axis=0).tolist() if mask.any() else None

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
        "mistakes": {
            "n_labelled_events": len(mistake_spans),
            "n_current_frames": int(mistake_current.sum()),
            "n_history_overlap_frames": int(mistake_in_history.sum()),
            "n_relevant_frames": int(mistake_relevant.sum()),
            "mean_temporal_mass_by_layer": {
                "current_mistake": _group_mass(mistake_current),
                "mistake_in_history": _group_mass(mistake_in_history),
                "neither": _group_mass(clean),
            },
            "mean_mistake_age_enrichment_by_layer": (
                np.nanmean(mistake_age_enrichment[mistake_in_history], axis=0).tolist()
                if mistake_in_history.any()
                else None
            ),
        },
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
        # Fraction of heads whose least-read age is inside the window: the trough the
        # slope cannot see. See _interior_min_fraction for why it is a location and not
        # a curvature.
        "age_interior_min_fraction_by_layer": shape["interior_min"].tolist(),
        # Positional controls. survival near 1 means the age profile is reproduced with no
        # temporal content present and is therefore not evidence about history.
        "age_profile_survival_by_layer": {
            name: [None if np.isnan(v) else float(v) for v in values] for name, values in survival.items()
        },
        "age_interior_min_fraction_by_layer_control": {
            name: values.tolist() for name, values in control_interior_min.items()
        },
        # Delivered rather than attended: w_k ||v_k|| instead of w_k. Where the attended
        # past mass sits near its null and this sits far below it, the attention on
        # history is not reaching the output. See _delivered.
        "delivered_past_share_by_layer": (
            delivered["past_share"].mean(axis=0).tolist() if delivered else None
        ),
        "delivered_age_share_by_layer": (delivered["age_share"].tolist() if delivered else None),
        "per_frame": [
            meta
            | {
                "temporal_mass_by_layer": temporal_mass[index].tolist(),
                "enrichment_vs_uniform_by_layer": (temporal_mass[index] / uniform_mass).tolist(),
                "mistake_age_share_by_layer": (
                    mistake_age_share[index].tolist() if meta["mistake_in_history"] else None
                ),
                "mistake_age_enrichment_by_layer": (
                    mistake_age_enrichment[index].tolist() if meta["mistake_in_history"] else None
                ),
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
        **({"patch_temporal_mass_by_age": patch_age} if patch_age is not None else {}),
        age_log_slope=shape["slope"],
        age_head_agreement=shape["agreement"],
        age_recency_index_by_head=shape["head_recency"],
        age_interior_min_fraction=shape["interior_min"],
        **(
            {
                "head_key_value_contribution": contribution,
                "delivered_past_share": delivered["past_share"],
                "delivered_age_share": delivered["age_share"],
            }
            if delivered
            else {}
        ),
        # Sibling arrays rather than a condition axis on head_age_mass: that array keeps
        # its shape and meaning, so npz files written before the controls existed and any
        # analysis already reading them stay valid.
        **{f"head_age_mass_{name}": rows for name, rows in control_head_age.items()},
        **{f"age_profile_survival_{name}": values for name, values in survival.items()},
        layers=np.asarray(layers),
        camera_names=np.asarray(camera_names),
        history_seconds=history_seconds,
        episode_idx=np.asarray([meta["episode_idx"] for meta in frame_meta]),
        frame_idx=np.asarray([meta["frame_idx"] for meta in frame_meta]),
        global_idx=np.asarray([meta["global_idx"] for meta in frame_meta]),
        mistake_current=mistake_current,
        mistake_in_history=mistake_in_history,
        mistake_history_age_mask=mistake_history_age_mask,
        mistake_age_share=mistake_age_share,
        mistake_age_enrichment=mistake_age_enrichment,
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
        survival,
        delivered,
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
        filename = f"frame_p{percentile:02d}_ep{meta['episode_idx']:04d}_fr{meta['frame_idx']:06d}.png"
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

    mistake_examples: list[tuple[dict, str]] = []
    if patch_age is not None and mistake_relevant.any():
        mistakes_dir = os.path.join(output_dir, "mistakes")
        makedirs(mistakes_dir)
        drawn_patch_age = patch_age.mean(axis=1)
        patch_age_range = tuple(float(value) for value in np.percentile(drawn_patch_age, [2, 98]))
        for index in np.flatnonzero(mistake_relevant):
            meta = frame_meta[int(index)]
            relation = "current" if meta["mistake_current"] else "recovery"
            filename = (
                f"{relation}_ep{meta['episode_idx']:04d}_fr{meta['frame_idx']:06d}_"
                f"global{meta['global_idx']:06d}.png"
            )
            diagnostic = {
                "global_idx": meta["global_idx"],
                "camera_keys": camera_keys,
                "history_seconds": history_seconds,
                "head_age": head_age[int(index)],
                "patch_age": patch_age[int(index)],
                "mistake_current": meta["mistake_current"],
                "mistake_history_age_mask": meta["mistake_history_age_mask"],
                "mistake_types": meta["mistake_types"],
            }
            _render_mistake_sequence(
                dataset,
                memory_cfg,
                fps,
                diagnostic,
                1.0 / (n_ages + 1),
                patch_age_range,
                os.path.join(mistakes_dir, filename),
            )
            mistake_examples.append((meta, filename))

    _write_manifest(output_dir, summary, examples, mistake_examples)

    mean_enrichment = temporal_mass.mean(axis=0) / uniform_mass
    logging.info(
        f"[mem_temporal_attention] n={len(frame_meta)}  uniform baseline={uniform_mass:.4f}  "
        f"per-layer mean enrichment={[round(float(value), 2) for value in mean_enrichment]}"
    )
    if mistake_spans:
        logging.info(
            f"[mem_temporal_attention] mistakes={len(mistake_spans)} events, "
            f"current={int(mistake_current.sum())} frames, "
            f"history-overlap={int(mistake_in_history.sum())} frames; "
            f"rendered={len(mistake_examples)} sequences."
        )
    if delivered is not None:
        attended = temporal_mass.mean(axis=0)
        logging.info(
            "[mem_temporal_attention] attended past mass="
            f"{[round(float(v), 3) for v in attended]}  delivered past share="
            f"{[round(float(v), 3) for v in delivered['past_share'].mean(axis=0)]} "
            "(delivered far below attended = the attention on history is a sink)"
        )
    for name, values in survival.items():
        logging.info(
            f"[mem_temporal_attention] {name} control: per-layer age-profile survival="
            f"{[None if np.isnan(v) else round(float(v), 2) for v in values]} "
            f"(1.0 = the profile is positional)"
        )

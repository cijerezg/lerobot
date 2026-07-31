"""Where the action tokens spend their cross-attention, and how that shifts over frames.

Every action query runs one softmax over the whole encoder key axis: both camera
token blocks, the point-map depth columns, each prompt clause, and the chat
scaffolding. That row sums to 1, so partitioning the columns gives a genuine
attention *budget* — and the question this probe exists to answer is not what the
budget is on average (images dominate, they have hundreds of columns) but **how it
moves**: does depth take a bigger share when the gripper is close to something, does
the top camera take over during transport, does the subtask clause get read at all.

## Reading the numbers

The budget is compositional: the row sums to 1, so if image mass rises everything
else falls *mechanically*. Segment levels are therefore mutually confounded and a
story about one segment's level is not safe. Two transforms fix this and are what
the panels actually plot:

* **centered log-ratio**, the standard statistic for simplex data —
  ``clr(m)_S = log m_S - mean_S' log m_S'``. Shifts in clr are shifts in the
  *composition*, not in one segment's arbitrary level.
* **pairwise log-ratio** ``log(m_A / m_B)`` for a specific hypothesis. It is
  invariant to whatever every other segment does, which makes
  ``log(depth / img_wrist)`` the honest test of "depth matters more up close".

Two controls ship alongside, because both failure modes are real:

* **Row entropy** per layer per frame. The whole distribution can sharpen or flatten
  across frames or checkpoints, moving every mass without any change in preference.
  Flat entropy + moving composition ⇒ the composition move is real.
* **Mass per token** for ``subtask`` and ``metadata`` only. Those are the sole
  segments whose column count varies frame to frame ("grasp the grey shirt" vs
  "return to home"), so their raw series confounds "attends more" with "clause got
  longer". Every other segment has a fixed column count, where the count cancels in
  any across-frame comparison and per-token normalization would only add noise.

## Two things this probe cannot tell you

1. **Mass is not contribution.** A segment can be attended and its values ignored.
   ``--probe_parameters.budget_fd_sensitivity`` adds the causal complement: a
   finite-difference ``||Δactions||`` per frame for depth and wrist RGB, which is a
   contribution series rather than a mass series. Off by default — it costs two
   extra forwards per frame.
2. **Correlation with distance is mediated.** Approaching an object changes task
   phase, gripper state and image content at once, so a depth-mass/distance
   correlation is suggestive, not causal.

Also note the capture is at a single flow timestep (``probe_parameters.timestep``,
last denoise step), so this is one point on the flow trajectory.

The summary memory never enters the action prompt by design — it reaches behaviour
only through the decoded subtask — so its absence from this budget is the invariant,
not a finding. Judge memory at the subtask decode (``offline_inference``).

Registered probe: enable with ``probe_parameters.enable_attention_budget``.
"""

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.probes.attention import _group_prompt_indices, _text_blocks_for_action_matrix
from lerobot.probes.utils import (
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)

_EPS = 1e-12

# Stable colours so a segment keeps its identity across panels and across runs.
_SEGMENT_COLORS = {
    "img_top": "#4363d8",
    "img_wrist": "#3cb44b",
    "depth": "#e6194b",
    "depth_nullbank": "#f58231",
    "task": "#911eb4",
    "subtask": "#42d4f4",
    "state": "#f032e6",
    "state history": "#bfef45",
    "metadata": "#ff8c00",
    "question": "#469990",
    "other prompt": "#9a6324",
    "residual": "#a9a9a9",
}
# Column count varies per frame only for these: the clause text itself changes length.
_VARIABLE_LENGTH_SEGMENTS = ("subtask", "metadata")


def _color(name: str) -> str:
    return _SEGMENT_COLORS.get(name, "#808080")


def _segment_columns(result, encoder_len: int) -> dict[str, list[int]]:
    """Partition EVERY encoder column into named segments, residual included.

    A budget that drops unlabelled columns does not sum to 1, and the columns most
    likely to be dropped are exactly the chat scaffolding that attracts attention
    sinks — a large, near-constant share that would silently rescale every trend.
    Camera crops are merged into their parent camera: they are the same physical
    view at a different resolution, and splitting them fragments the series.
    """
    segments: dict[str, list[int]] = {}

    for name, indices in (result.extras.get("image_patch_indices_by_segment") or {}).items():
        parent = str(name).split("_crop")[0]
        segments.setdefault(parent, []).extend(int(i) for i in indices)

    depth_segment = result.extras.get("depth_segment")
    if depth_segment:
        name = "depth_nullbank" if depth_segment.get("is_null_bank") else "depth"
        segments[name] = [int(i) for i in depth_segment["indices"]]

    encoder_valid = None
    if torch.is_tensor(result.encoder_pad_masks) and result.encoder_pad_masks.ndim >= 2:
        encoder_valid = result.encoder_pad_masks[0].detach().cpu().to(torch.bool)
    text_blocks = _text_blocks_for_action_matrix(result)
    if text_blocks:
        for group, entries in _group_prompt_indices(result, text_blocks, encoder_len, encoder_valid):
            if entries:
                segments[group] = [int(idx) for idx, _label in entries]

    claimed: set[int] = set()
    clean: dict[str, list[int]] = {}
    for name, indices in segments.items():
        unique = sorted({i for i in indices if 0 <= i < encoder_len and i not in claimed})
        if unique:
            clean[name] = unique
            claimed.update(unique)

    residual = sorted(set(range(encoder_len)) - claimed)
    if residual:
        clean["residual"] = residual
    return clean


def _frame_budget(attn: torch.Tensor, columns: dict[str, list[int]]):
    """One layer's budget for one frame.

    ``attn`` is ``[H, Q, N]`` after softmax. Returns per-segment mass averaged over
    heads and action queries, the same masses resolved per action query, and the
    row entropy normalized to [0, 1].
    """
    mass: dict[str, float] = {}
    by_query: dict[str, np.ndarray] = {}
    for name, indices in columns.items():
        block = attn.index_select(2, torch.as_tensor(indices, dtype=torch.long)).sum(dim=2)  # [H, Q]
        mass[name] = float(block.mean())
        by_query[name] = block.mean(dim=0).numpy()  # [Q]

    probs = attn.clamp_min(0)
    entropy = -(probs * torch.log(probs.clamp_min(_EPS))).sum(dim=2)  # [H, Q]
    normalized = float(entropy.mean()) / float(np.log(max(attn.shape[2], 2)))
    return mass, by_query, normalized


def _clr(mass: np.ndarray) -> np.ndarray:
    """Centered log-ratio over the last axis. Composition in, unconstrained out."""
    logs = np.log(np.clip(mass, _EPS, None))
    return logs - logs.mean(axis=-1, keepdims=True)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _median_valid_depth_mm(obs: dict, pointmap_config) -> float:
    """Median depth over valid pixels, in mm — the scene covariate.

    Zero is the invalid marker in the raw sidecar, so it is excluded rather than
    dragging the median toward the camera.
    """
    depth = obs.get(f"observation.depth.{pointmap_config.depth_key}")
    if not torch.is_tensor(depth):
        return float("nan")
    values = depth.detach().float().reshape(-1)
    valid = values[values > 0]
    if valid.numel() == 0:
        return float("nan")
    return float(valid.median()) * float(pointmap_config.depth_units_mm)


def _render(
    names: list[str],
    mass: np.ndarray,          # [frames, layers, segments]
    by_query: np.ndarray,      # [frames, layers, segments, Q]
    entropy: np.ndarray,       # [frames, layers]
    layers: list[int],
    frame_meta: list[dict],
    depth_mm: np.ndarray,
    token_counts: dict[str, int],
    output_path: str,
) -> None:
    focus = len(layers) // 2  # middle probed layer: past the near-positional early stack
    fig, axes = plt.subplots(2, 3, figsize=(20, 10.5))
    steps = np.arange(len(frame_meta))
    colors = [_color(n) for n in names]

    axes[0, 0].stackplot(steps, mass[:, focus, :].T, labels=names, colors=colors, alpha=0.9)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xlabel("sampled frame")
    axes[0, 0].set_ylabel("share of attention mass")
    axes[0, 0].set_title(f"Budget composition, layer {layers[focus]}")
    axes[0, 0].legend(fontsize=6, ncol=2, loc="upper right")

    clr = _clr(mass[:, focus, :])
    for index, name in enumerate(names):
        axes[0, 1].plot(steps, clr[:, index], color=_color(name), linewidth=1.3, label=name)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 1].set_xlabel("sampled frame")
    axes[0, 1].set_ylabel("centered log-ratio")
    axes[0, 1].set_title(f"Compositional shift, layer {layers[focus]}\n(level-invariant; this is the real signal)")
    axes[0, 1].legend(fontsize=6, ncol=2)

    image = axes[0, 2].imshow(mass.mean(axis=0).T, aspect="auto", cmap="viridis", vmin=0.0)
    axes[0, 2].set_xticks(range(len(layers)), [str(layer) for layer in layers])
    axes[0, 2].set_yticks(range(len(names)), names, fontsize=7)
    axes[0, 2].set_xlabel("action-expert layer")
    axes[0, 2].set_title("Mean mass by layer")
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046)

    ax = axes[1, 0]
    if "depth" in names and any(n.startswith("img_") and "wrist" in n for n in names):
        wrist = next(n for n in names if n.startswith("img_") and "wrist" in n)
        ratio = np.log(
            np.clip(mass[:, focus, names.index("depth")], _EPS, None)
            / np.clip(mass[:, focus, names.index(wrist)], _EPS, None)
        )
        finite = np.isfinite(depth_mm)
        if finite.sum() >= 3:
            rho, p = _spearman(depth_mm[finite], ratio[finite])
            ax.scatter(depth_mm[finite], ratio[finite], s=26, alpha=0.75, color="#e6194b")
            ax.set_title(f"log(depth / {wrist}) vs scene distance\nSpearman rho={rho:+.2f} (p={p:.3g})")
        else:
            ax.set_title(f"log(depth / {wrist}) vs scene distance — no valid depth")
        ax.set_xlabel("median valid wrist depth (mm)")
        ax.set_ylabel("log ratio")
        ax.axhline(0.0, color="black", linewidth=0.7)
    else:
        ax.text(0.5, 0.5, "no depth segment", ha="center", va="center")
        ax.axis("off")

    for index, layer in enumerate(layers):
        axes[1, 1].plot(steps, entropy[:, index], linewidth=1.3, label=f"layer {layer}")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("sampled frame")
    axes[1, 1].set_ylabel("normalized row entropy")
    axes[1, 1].set_title("Sharpening control\n(flat here ⇒ composition moves are real)")
    axes[1, 1].legend(fontsize=7)

    query_mass = by_query[:, focus, :, :].mean(axis=0)  # [segments, Q]
    image = axes[1, 2].imshow(query_mass, aspect="auto", cmap="magma", vmin=0.0)
    axes[1, 2].set_yticks(range(len(names)), names, fontsize=7)
    axes[1, 2].set_xlabel("action-chunk position")
    axes[1, 2].set_title(f"Budget by chunk position, layer {layers[focus]}")
    fig.colorbar(image, ax=axes[1, 2], fraction=0.046)

    variable = ", ".join(
        f"{name} {token_counts.get(name, 0)} tok" for name in _VARIABLE_LENGTH_SEGMENTS if name in names
    )
    fig.suptitle(
        f"Action-token attention budget — {len(frame_meta)} frames, layers {layers}"
        + (f"  |  variable-length clauses: {variable}" if variable else ""),
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight", dpi=125)
    plt.close(fig)


def run(adapter, dataset, cfg, output_dir: str) -> None:
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[attention_budget] needs continuous flow actions — skipping.")
        return

    makedirs(output_dir)
    p = cfg.probe_parameters
    chunk_size = int(cfg.policy.chunk_size)
    timestep = float(getattr(p, "timestep", 0.5))
    layers = [int(x.strip()) for x in p.spatial_layers.split(",")]
    pointmap_config = getattr(cfg.policy, "pointmap_config", None)
    want_fd = bool(getattr(p, "budget_fd_sensitivity", False))

    samples = sample_episodes_evenly(
        dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )
    if not samples:
        logging.warning("[attention_budget] no frames selected.")
        return

    adapter._set_probe_cuda_graph_enabled(False)
    names: list[str] | None = None
    token_counts: dict[str, int] = {}
    mass_rows: list[np.ndarray] = []
    query_rows: list[np.ndarray] = []
    entropy_rows: list[np.ndarray] = []
    frame_meta: list[dict] = []
    depth_mm: list[float] = []
    fd_rows: list[dict] = []

    try:
        for ep_idx, fr_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            result = adapter.capture_attention(
                frame["obs"], frame["task"], state=frame["state"], timestep=timestep,
                layers=layers, subtask=frame["subtask"], metadata=frame["metadata"],
            )
            if not result.cross_attn_by_layer:
                logging.warning(f"[attention_budget] ep={ep_idx} fr={fr_idx}: no cross-attn; skipping.")
                continue

            encoder_len = int(next(iter(result.cross_attn_by_layer.values())).shape[-1])
            columns = _segment_columns(result, encoder_len)
            if names is None:
                names = list(columns)
                token_counts = {name: len(indices) for name, indices in columns.items()}
            elif list(columns) != names:
                # A clause the packer dropped would silently re-index every array.
                logging.warning(
                    f"[attention_budget] ep={ep_idx} fr={fr_idx}: segment set changed "
                    f"({set(columns) ^ set(names)}); skipping frame."
                )
                continue

            frame_mass, frame_query, frame_entropy = [], [], []
            for layer in layers:
                cross = result.cross_attn_by_layer.get(layer)
                if cross is None:
                    break
                attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)
                mass, by_query, entropy = _frame_budget(attn, columns)
                frame_mass.append([mass[name] for name in names])
                frame_query.append([by_query[name] for name in names])
                frame_entropy.append(entropy)
            if len(frame_mass) != len(layers):
                continue

            mass_rows.append(np.asarray(frame_mass, dtype=np.float64))
            query_rows.append(np.asarray(frame_query, dtype=np.float32))
            entropy_rows.append(np.asarray(frame_entropy, dtype=np.float64))
            frame_meta.append(
                {"episode_idx": int(ep_idx), "frame_idx": int(fr_idx),
                 "global_idx": int(global_idx), "task": frame["task"],
                 "subtask": frame["subtask"]}
            )
            depth_mm.append(
                _median_valid_depth_mm(frame["obs"], pointmap_config)
                if pointmap_config is not None else float("nan")
            )

            if want_fd:
                fd_rows.append(_fd_sensitivity(adapter, frame, pointmap_config))
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not mass_rows or names is None:
        logging.warning("[attention_budget] no frames produced a budget.")
        return

    mass = np.stack(mass_rows)            # [frames, layers, segments]
    by_query = np.stack(query_rows)       # [frames, layers, segments, Q]
    entropy = np.stack(entropy_rows)      # [frames, layers]
    depth_array = np.asarray(depth_mm, dtype=np.float64)

    total = mass.sum(axis=-1)
    if not np.allclose(total, 1.0, atol=2e-2):
        logging.warning(
            f"[attention_budget] budget does not sum to 1 (min={total.min():.4f} "
            f"max={total.max():.4f}) — the column partition is missing mass."
        )

    summary = _summarize(names, mass, entropy, depth_array, token_counts, layers, frame_meta, fd_rows)
    with open(os.path.join(output_dir, "budget.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(output_dir, "budget_data.npz"),
        segment_names=np.asarray(names), layers=np.asarray(layers),
        mass=mass, mass_by_query=by_query, entropy=entropy,
        median_depth_mm=depth_array,
        token_counts=np.asarray([token_counts[n] for n in names]),
        episode_idx=np.asarray([m["episode_idx"] for m in frame_meta]),
        frame_idx=np.asarray([m["frame_idx"] for m in frame_meta]),
    )
    _render(names, mass, by_query, entropy, layers, frame_meta, depth_array,
            token_counts, os.path.join(output_dir, "budget.png"))

    focus = len(layers) // 2
    ranked = sorted(names, key=lambda n: -mass[:, focus, names.index(n)].mean())
    logging.info(
        f"[attention_budget] n={len(frame_meta)} layer {layers[focus]} mean budget: "
        + "  ".join(f"{n}={mass[:, focus, names.index(n)].mean():.3f}" for n in ranked[:6])
    )
    volatile = sorted(names, key=lambda n: -np.std(_clr(mass[:, focus, :])[:, names.index(n)]))
    logging.info(
        "[attention_budget] most variable across frames (clr std): "
        + "  ".join(f"{n}={np.std(_clr(mass[:, focus, :])[:, names.index(n)]):.3f}" for n in volatile[:5])
    )
    if summary.get("depth_distance_spearman") is not None:
        rho = summary["depth_distance_spearman"]["rho"]
        logging.info(
            f"[attention_budget] log(depth/wrist) vs scene distance: rho={rho:+.3f} "
            f"(p={summary['depth_distance_spearman']['p']:.3g}); negative ⇒ depth share rises up close"
        )


def _fd_sensitivity(adapter, frame, pointmap_config) -> dict:
    """Per-frame causal complement: ||Δactions|| under a 1%-of-std input perturbation.

    Mass says a segment was read; this says perturbing it changes the output. Two
    extra forwards per frame, hence the flag.
    """
    if pointmap_config is None:
        return {}
    obs = frame["obs"]
    depth_key = f"observation.depth.{pointmap_config.depth_key}"
    rgb_key = f"observation.images.{pointmap_config.depth_key}"

    def predict(observation):
        generator = torch.Generator(device=adapter.device)
        generator.manual_seed(0)
        return adapter.predict_action_chunk(
            observation, frame["task"], state=frame["state"], subtask=frame["subtask"],
            metadata=frame["metadata"], generator=generator,
        )[1]

    base = predict(obs)
    out = {}
    for label, key in (("depth", depth_key), ("rgb_wrist", rgb_key)):
        raw = obs.get(key)
        if not torch.is_tensor(raw):
            continue
        noise = torch.randn_like(raw.float()) * raw.float().std() * 0.01
        out[label] = float((predict({**obs, key: (raw.float() + noise).to(raw.dtype)}) - base).norm())
    return out


def _summarize(names, mass, entropy, depth_mm, token_counts, layers, frame_meta, fd_rows) -> dict:
    focus = len(layers) // 2
    clr = _clr(mass[:, focus, :])
    summary = {
        "n_frames": len(frame_meta),
        "layers": layers,
        "focus_layer": layers[focus],
        "segments": names,
        "token_counts": token_counts,
        "budget_sums_to": {"min": float(mass.sum(axis=-1).min()), "max": float(mass.sum(axis=-1).max())},
        "mean_mass_by_layer": {
            str(layer): {name: float(mass[:, i, j].mean()) for j, name in enumerate(names)}
            for i, layer in enumerate(layers)
        },
        "clr_std_at_focus_layer": {name: float(clr[:, j].std()) for j, name in enumerate(names)},
        "entropy_mean_by_layer": {str(layer): float(entropy[:, i].mean()) for i, layer in enumerate(layers)},
        "entropy_std_by_layer": {str(layer): float(entropy[:, i].std()) for i, layer in enumerate(layers)},
        "tasks": sorted({m["task"] for m in frame_meta}),
    }

    # Per-token only where the column count varies frame to frame; elsewhere the
    # count is constant and cancels in any across-frame comparison.
    summary["mean_mass_per_token"] = {
        name: float(mass[:, focus, names.index(name)].mean() / max(token_counts.get(name, 1), 1))
        for name in _VARIABLE_LENGTH_SEGMENTS
        if name in names
    }

    if "depth" in names:
        wrist = next((n for n in names if n.startswith("img_") and "wrist" in n), None)
        finite = np.isfinite(depth_mm)
        if wrist is not None and finite.sum() >= 3:
            ratio = np.log(
                np.clip(mass[:, focus, names.index("depth")], _EPS, None)
                / np.clip(mass[:, focus, names.index(wrist)], _EPS, None)
            )
            rho, p = _spearman(depth_mm[finite], ratio[finite])
            summary["depth_distance_spearman"] = {
                "rho": rho, "p": p, "against": wrist,
                "note": "negative rho ⇒ depth's share of the budget rises as the scene gets closer",
            }

    if len(summary["tasks"]) > 1:
        summary["mean_mass_by_task"] = {
            task: {
                name: float(
                    mass[[i for i, m in enumerate(frame_meta) if m["task"] == task], focus, j].mean()
                )
                for j, name in enumerate(names)
            }
            for task in summary["tasks"]
        }

    if fd_rows:
        keys = {k for row in fd_rows for k in row}
        summary["fd_sensitivity_mean"] = {
            key: float(np.mean([row[key] for row in fd_rows if key in row])) for key in keys
        }

    summary["per_frame"] = [
        meta | {"median_depth_mm": float(depth_mm[i]),
                "mass_at_focus_layer": {name: float(mass[i, focus, j]) for j, name in enumerate(names)}}
        for i, meta in enumerate(frame_meta)
    ]
    return summary

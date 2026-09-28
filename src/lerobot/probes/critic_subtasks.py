"""Small, shared calculations for the critic's subtask diagnostics.

G is the return of the recorded trajectory under the sampler's clipped reward rule,
not a counterfactual oracle or an independent estimate of policy advantage.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def reference_returns(boundary, mistake_onset, chunk, discount, norm, penalty, v_min, v_max=0.0):
    """Return from every anchor, including terminal-chunk penalties and per-step clipping."""
    boundary = np.asarray(boundary, dtype=bool)
    n = len(boundary)
    if n and not boundary[-1]:
        raise ValueError("The final frame must be a terminal.")
    stops = np.minimum(np.arange(n) + chunk, n)
    def in_window(flags):
        prefix = np.r_[0, np.cumsum(flags)]
        return prefix[stops] > prefix[np.arange(n)]
    done, mistake = in_window(boundary), in_window(mistake_onset)
    rewards = (done.astype(float) - 1.0 - penalty * mistake) / norm
    returns = np.empty(n, dtype=float)
    for i in range(n - 1, -1, -1):
        future = 0.0 if done[i] else discount * returns[i + chunk]
        returns[i] = np.clip(rewards[i] + future, v_min, v_max)
    return returns


def load_text_counts(path):
    if not path:
        raise ValueError("critic_subtask_swap requires critic_text_counts_path; run migration/critic_text_counts.py.")
    counts = json.loads(Path(path).read_text())
    if not isinstance(counts, dict) or not counts or any(
        not isinstance(k, str) or not k or type(v) is not int or v < 0 for k, v in counts.items()
    ):
        raise ValueError(f"{path}: expected a nonempty {{text: nonnegative episode count}} table.")
    return counts


def progress_error(a, b, chunk, v_min):
    """Only compare a one-chunk pair with unchanged conditioning and no terminal crossing."""
    if (
        b["global_idx"] - a["global_idx"] != chunk
        or a["episode"] != b["episode"]
        or a["segment_end"] != b["segment_end"]
        or a["subtask"] != b["subtask"]
        or a["metadata"] != b["metadata"]
        or a["task"] != b["task"]
        or a["value"] is None or b["value"] is None
        or (a["reference"] <= v_min and b["reference"] <= v_min)
    ):
        return None
    return abs((b["value"] - a["value"]) - (b["reference"] - a["reference"]))


def _sample_episode_buckets(buckets, budget):
    """Allocate the bounded sample first, then span each episode's full timeline."""
    quotas = [0] * len(buckets)
    remaining = min(max(0, budget), sum(map(len, buckets)))
    while remaining:
        for i, bucket in enumerate(buckets):
            if quotas[i] < len(bucket):
                quotas[i] += 1
                remaining -= 1
                if remaining == 0:
                    break
    sampled = []
    for bucket, quota in zip(buckets, quotas, strict=True):
        # A singleton uses the midpoint instead of always taking the first frame.
        picks = ([len(bucket) // 2] if quota == 1
                 else np.linspace(0, len(bucket) - 1, quota, dtype=int))
        sampled.append([bucket[i] for i in picks])
    return [bucket[j] for j in range(max(quotas, default=0)) for bucket in sampled if j < len(bucket)]


def sample_text_groups(records, budget, per_text, seed, max_labels=16):
    """Deterministic groups of exact texts, spread through each episode's trajectory."""
    if budget <= 0:
        return []
    if per_text < 2:
        raise ValueError("critic_grad_frames_per_subtask must be at least 2.")
    groups = defaultdict(list)
    for row in records:
        if row["subtask"] and row["value"] is not None:
            groups[row["subtask"]].append(row)
    names = sorted(k for k, rows in groups.items() if len(rows) >= min(8, per_text))
    np.random.default_rng(seed).shuffle(names)
    selected = []
    for name in names[:max_labels]:
        rows = groups[name]
        by_episode = defaultdict(list)
        for row in rows:
            by_episode[row["episode"]].append(row)
        buckets = [sorted(by_episode[episode], key=lambda r: r["global_idx"])
                   for episode in sorted(by_episode)]
        count = min(per_text, len(rows), budget - len(selected))
        if count < 2:
            break
        selected.extend(_sample_episode_buckets(buckets, count))
    return selected


def fit_report(records, counts, output_dir):
    """Two count plots; raw rows and per-text support counts remain available in JSON."""
    groups = defaultdict(list)
    all_texts = defaultdict(list)
    for row in records:
        if row["subtask"] and row["value"] is not None:
            groups[(row["domain"], row["subtask"])].append(row)
            all_texts[row["subtask"]].append(row)

    def stats(rows):
        pairs = [r["progress_error"] for r in rows if r.get("progress_error") is not None]
        return {
            "n_points": len(rows), "n_pairs": len(pairs),
            "n_episodes": len({r["episode"] for r in rows}),
            "value_error": float(np.mean([abs(r["value"] - r["reference"]) for r in rows])),
            "progress_error": float(np.mean(pairs)) if pairs else None,
        }

    per_text = [
        {"domain": domain, "text": text, "train_episodes": counts.get(text, 0), **stats(rows)}
        for (domain, text), rows in sorted(groups.items())
    ]
    summary = {}
    for label, accept in (("rare", lambda n: n <= 3), ("common", lambda n: n >= 10)):
        bucket = [stats(rows) for text, rows in all_texts.items() if accept(counts.get(text, 0))]
        for metric in ("value_error", "progress_error"):
            values = [r[metric] for r in bucket if r[metric] is not None]
            summary[f"critic_{label}_{metric}"] = float(np.median(values)) if values else None
            summary[f"critic_{label}_{metric}_texts"] = len(values)
        summary[f"critic_{label}_points"] = sum(r["n_points"] for r in bucket)
        summary[f"critic_{label}_pairs"] = sum(r["n_pairs"] for r in bucket)
    for variant in ("next", "unrelated"):
        gaps = [
            abs(r["swaps"][variant]["value"] - r["value"])
            for r in records if r["value"] is not None and r.get("swaps", {}).get(variant)
            and r["swaps"][variant]["value"] is not None
        ]
        summary[f"critic_swap_{variant}_gap"] = float(np.mean(gaps)) if gaps else None
        summary[f"critic_swap_{variant}_points"] = len(gaps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(
        axes, ("value_error", "progress_error"),
        ("True-text value error |V − G|", "One-step progress error |ΔV − ΔG|"), strict=True,
    ):
        for domain, color, marker in (("rebot", "steelblue", "o"), ("diverse", "darkorange", "^")):
            dots = [r for r in per_text if r["domain"] == domain and r[metric] is not None]
            nkey = "n_pairs" if metric == "progress_error" else "n_points"
            ax.scatter(
                [r["train_episodes"] or 0.35 for r in dots], [r[metric] for r in dots],
                s=[20 + 12 * np.sqrt(r[nkey]) for r in dots], alpha=0.7,
                color=color, marker=marker, label=f"{domain} ({len(dots)} texts)",
            )
        ax.set_xscale("log")
        ticks = [0.35, 1, 3, 10, 30, 100, 300, 1000]
        largest = max((r["train_episodes"] for r in per_text), default=1)
        ticks = [t for t in ticks if t <= max(10, largest)]
        ax.set_xlim(0.25, max(15, largest * 1.3))
        ax.set_xticks(ticks, ["unseen" if t == 0.35 else str(t) for t in ticks])
        ax.set(xlabel="Training episodes with exact text", ylabel="Mean absolute error", title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("One dot per text and domain; size = contributing points / pairs")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "critic_fit_vs_count.png", dpi=150)
    plt.close(fig)
    summary.update(shirt_family_report(records, counts, output_dir))
    Path(output_dir, "critic_fit_vs_count.json").write_text(json.dumps(
        {"summary": summary, "per_text": per_text, "records": records}, indent=2, allow_nan=False,
    ))
    return summary


def swap_plot(rows, boundary_seconds, episode, output_path, v_min, unrelated_count):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    seconds = [r["seconds"] for r in rows]
    for ax, variant, color in zip(axes, ("next", "unrelated"), ("darkorange", "purple"), strict=True):
        ax.plot(seconds, [r["value"] for r in rows], color="0.65", label="V(current text)")
        values = [(r.get("swaps", {}).get(variant) or {}).get("value") for r in rows]
        label = "V(next-segment text)" if variant == "next" else "V(control text)"
        ax.plot(seconds, values, color=color, label=label)
        if variant == "next":
            refs = [(r.get("swaps", {}).get(variant) or {}).get("reference") for r in rows]
            ax.plot(seconds, refs, "--", color=color, alpha=0.65, label="time-to-next-end reference")
        else:
            ax.axhline(v_min, color=color, linestyle=":", alpha=0.65, label="support floor")
            text = next((r["swaps"][variant]["text"] for r in rows if r["swaps"].get(variant)), "(none)")
            ax.set_title(f"Unrelated: {text} ({unrelated_count} training episodes)", fontsize=10)
        for b in boundary_seconds:
            ax.axvline(b, color="0.8", linestyle=":", linewidth=0.6)
        ax.set_ylabel("Value")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Seconds into episode")
    fig.suptitle(f"Episode {episode}: text sensitivity (swapped references are hypotheses)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def diverse_reference(row, anchor_frame, chunk_seconds, discount, norm, penalty, v_min, v_max):
    """Same native-frame windows as DiverseActorBuffer.critic_view, including release fold."""
    chunk = int(round(float(row["native_rate_hz"]) * chunk_seconds))
    end = int(row["critic_end_timestep_exclusive"])
    rewards = []
    start = int(anchor_frame)
    while start < end:
        done = end <= start + chunk
        mistake = any(start <= t < start + chunk for t in row["mistake_onset_timesteps"])
        rewards.append((float(done) - 1.0 - penalty * mistake) / norm)
        start += chunk
    value = 0.0
    for reward in reversed(rewards):
        value = float(np.clip(reward + discount * value, v_min, v_max))
    return value


def diverse_fit_records(adapter, cfg):
    """Bounded true-text anchors from the existing holdout, with native training metadata."""
    from lerobot.datasets.diverse_actor_selection import holdout_actor_selection, open_federated_corpus
    from lerobot.datasets.diverse_prompt import episode_task
    from lerobot.rl.data_sources.diverse_actor_buffer import DiverseActorBuffer
    from lerobot.rl.data_sources.diverse_integration import sample_spec_from_config
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer, _forwarded_complementary_keys

    budget = int(getattr(cfg.probe_parameters, "critic_diverse_frames", 96))
    if budget <= 0 or not cfg.diverse.enabled:
        return []
    p = cfg.policy
    buffer = DiverseActorBuffer(
        holdout_actor_selection(open_federated_corpus(cfg.diverse.root)), sample_spec_from_config(cfg),
        render_automatic_quality=cfg.diverse.render_automatic_quality,
    )
    by_episode = defaultdict(list)
    for i, row in enumerate(buffer.rows):
        if not buffer._critic.skip[i]:
            by_episode[str(row["episode_id"])].append(i)
    # Allocate episode quotas before spacing anchors; truncating a larger grid
    # after round-robin selection would bias every episode toward its beginning.
    buckets = [sorted(by_episode[episode], key=lambda i: buffer.rows[i]["anchor_frame"])
               for episode in sorted(by_episode)]
    selected = _sample_episode_buckets(buckets, budget)

    def inputs(index):
        transition = buffer.collate([index])
        comp = transition["complementary_info"]
        obs = MolmoAct2Trainer._inject_depth_observations(transition["state"], comp, cfg)
        extra = {k: comp[k] for k in _forwarded_complementary_keys(comp, cfg)}
        for k in ("subtask_index", "task_index", "metadata_quality", "metadata_quality_is_valid",
                  "metadata_mistake", "metadata_speed", "metadata_precision", "metadata_contact"):
            extra.pop(k, None)
        obs = {**obs, **{f"probe_complementary.{k}": v for k, v in extra.items()}}
        metadata = {"mistake": bool(comp["metadata_mistake"].item()), "speed": int(comp["metadata_speed"].item())}
        if bool(comp["metadata_quality_is_valid"].item()):
            metadata["quality"] = int(comp["metadata_quality"].item())
        for k in ("precision", "contact"):
            value = int(comp[f"metadata_{k}"].item())
            if value >= 0:
                metadata[k] = value
        return obs, metadata

    records = []
    chunk_seconds = buffer.spec.action_horizon / buffer.spec.action_rate_hz
    for i in selected:
        row = buffer.rows[i]
        obs, metadata = inputs(i)
        task = episode_task(buffer.selection.episode_records[str(row["episode_id"])], row["source"])[0]
        reference = diverse_reference(
            row, row["anchor_frame"], chunk_seconds, p.discount, p.reward_normalization_constant,
            p.critic_mistake_penalty, p.value_support_min, p.value_support_max,
        )
        value = float(adapter.predict_value(obs, task, str(row["subtask"]), metadata=metadata))
        result = {
            "domain": "diverse", "source": str(row["source"]), "episode": str(row["episode_id"]),
            "global_idx": i, "frame": int(row["anchor_frame"]), "subtask": str(row["subtask"]),
            "task": task, "metadata": metadata, "value": value, "reference": reference, "progress_error": None,
        }
        j = int(buffer._critic.next_row[i])
        if not buffer._critic.done[i] and j >= 0:
            next_row = buffer.rows[j]
            if (next_row["subtask"] == row["subtask"]
                    and next_row["critic_end_timestep_exclusive"] == row["critic_end_timestep_exclusive"]):
                next_obs, next_metadata = inputs(j)
                next_ref = diverse_reference(
                    row, next_row["anchor_frame"], chunk_seconds, p.discount, p.reward_normalization_constant,
                    p.critic_mistake_penalty, p.value_support_min, p.value_support_max,
                )
                if metadata == next_metadata and not (reference <= p.value_support_min and next_ref <= p.value_support_min):
                    next_value = float(adapter.predict_value(next_obs, task, str(row["subtask"]), metadata=metadata))
                    if np.isfinite(next_value):
                        result["progress_error"] = abs((next_value - value) - (next_ref - reference))
        if not np.isfinite(value):
            raise ValueError(f"Non-finite diverse critic value at {row['episode_id']}:{row['anchor_frame']}")
        records.append(result)
    return records


_SHIRT_COLOURS = (
    "beige", "black", "blue", "brown", "green", "grey", "gray", "navy",
    "pink", "red", "white", "light blue", "dark blue",
)
_SHIRT_TIME_BINS = (0, 2, 5, 10, 20, float("inf"))


def shirt_family_report(records, counts, output_dir):
    """Compare exact shirt-grasp prompts without changing critic inputs or training.

    Cells match seconds to terminal and current mistake status. Average frames
    within episode/cell, then episodes equally. Comparable per-colour scores use
    the SAME cells, requiring at least two usable points per colour in each cell.
    This controls observed stage/mistake mix, not scene, quality or causal colour.
    """
    labels = {f"grasp the {colour} shirt": colour for colour in _SHIRT_COLOURS}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        text = row["subtask"]
        colour = labels.get(text.strip().lower().rstrip("."))
        seconds = row.get("seconds_to_end")
        if (row["domain"] != "rebot" or colour is None or seconds is None
                or row["value"] is None or row["reference"] is None):
            continue
        time_bin = int(np.searchsorted(_SHIRT_TIME_BINS, seconds, side="right") - 1)
        mistake = bool((row.get("metadata") or {}).get("mistake", False))
        grouped[text][(time_bin, mistake)].append(row)
    if not grouped:
        return {}

    texts = sorted(grouped, key=lambda text: (counts.get(text, 0), text))
    cells = sorted({cell for buckets in grouped.values() for cell in buckets}, key=lambda cell: (cell[1], cell[0]))

    def measure(rows, metric):
        by_episode = defaultdict(list)
        for row in rows:
            if metric == "progress_error":
                value = row.get("progress_error")
            else:
                value = row["value"] - row["reference"]
                if metric == "value_error":
                    value = abs(value)
            if value is not None:
                by_episode[row["episode"]].append(value)
        n = sum(map(len, by_episode.values()))
        value = float(np.mean([np.mean(v) for v in by_episode.values()])) if n else None
        return {"mean": value, "n_points": n, "n_episodes": len(by_episode)}

    metrics = ("residual", "value_error", "progress_error")
    estimates = {
        text: {cell: {metric: measure(grouped[text].get(cell, []), metric) for metric in metrics} for cell in cells}
        for text in texts
    }
    common = {
        metric: [cell for cell in cells if all(estimates[text][cell][metric]["n_points"] >= 2 for text in texts)]
        for metric in metrics
    }
    per_text = []
    for text in texts:
        rows = [r for bucket in grouped[text].values() for r in bucket]
        item = {
            "text": text, "colour": labels[text.strip().lower().rstrip(".")],
            "train_episodes": counts.get(text, 0), "val_episodes": len({r["episode"] for r in rows}),
            "n_points": len(rows),
        }
        for metric in metrics:
            shared = common[metric]
            item[f"matched_{metric}"] = (
                float(np.mean([estimates[text][cell][metric]["mean"] for cell in shared])) if shared else None
            )
        per_text.append(item)

    summary = {"critic_shirt_texts": len(texts)}
    for metric in ("value_error", "progress_error"):
        summary[f"critic_shirt_{metric}_matched_cells"] = len(common[metric])
        for band, predicate in (("rare", lambda n: n <= 3), ("common", lambda n: n >= 10)):
            values = [r[f"matched_{metric}"] for r in per_text
                      if predicate(r["train_episodes"]) and r[f"matched_{metric}"] is not None]
            summary[f"critic_shirt_{band}_{metric}"] = float(np.median(values)) if values else None
            summary[f"critic_shirt_{band}_{metric}_texts"] = len(values)
    biases = [r["matched_residual"] for r in per_text if r["matched_residual"] is not None]
    summary["critic_shirt_colour_bias_range"] = float(np.ptp(biases)) if len(biases) >= 2 else None

    def cell_label(cell):
        i, mistake = cell
        stop = _SHIRT_TIME_BINS[i + 1]
        duration = f"{_SHIRT_TIME_BINS[i]:g}–{stop:g}s" if np.isfinite(stop) else "20+s"
        return duration + ("\nmistake" if mistake else "\nclean")

    fig, axes = plt.subplots(1, 2, figsize=(max(13, 2.5 * len(cells)), max(5, 0.6 * len(texts) + 2)))
    for ax, metric, title in zip(
        axes, ("residual", "progress_error"), ("Signed value residual V − G", "One-step progress error |ΔV − ΔG|"),
        strict=True,
    ):
        data = np.array([[estimates[text][cell][metric]["mean"] for cell in cells] for text in texts], dtype=float)
        cmap = plt.get_cmap("RdBu_r" if metric == "residual" else "magma").copy()
        cmap.set_bad("0.92")
        if metric == "residual":
            limit = max(0.01, float(np.nanmax(np.abs(data)))) if np.isfinite(data).any() else 0.01
            image = ax.imshow(data, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        else:
            image = ax.imshow(data, cmap=cmap, vmin=0, aspect="auto")
        for i, text in enumerate(texts):
            for j, cell in enumerate(cells):
                estimate = estimates[text][cell][metric]
                value = estimate["mean"]
                label = "—" if value is None else f"{value:+.2f}" if metric == "residual" else f"{value:.3f}"
                if value is not None:
                    label += f"\nn={estimate['n_points']}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})
        ax.set_xticks(range(len(cells)), [cell_label(cell) for cell in cells], fontsize=8)
        ax.set_yticks(range(len(texts)), [
            f"{r['colour']}  (train {r['train_episodes']}; val {r['val_episodes']} ep)" for r in per_text
        ], fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Time to completion × current mistake status")
        fig.colorbar(image, ax=ax, shrink=0.7)
    fig.suptitle("Grasp-shirt family: exact colour prompts retained; episode-balanced cell means")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "critic_shirt_family.png", dpi=150)
    plt.close(fig)
    Path(output_dir, "critic_shirt_family.json").write_text(json.dumps({
        "family": "grasp shirt",
        "conditioning_unchanged": True,
        "matching": "same time bin and current mistake status; episode means weighted equally",
        "minimum_cell_points_per_text": 2,
        "common_cells": {metric: [cell_label(c) for c in shared] for metric, shared in common.items()},
        "summary": summary, "per_text": per_text,
        "cells": [
            {"text": text, "seconds_bin": cell_label(cell), "mistake": cell[1], **estimates[text][cell]}
            for text in texts for cell in cells
        ],
    }, indent=2, allow_nan=False))
    return summary


def gradient_group_summary(token_norms, groups, active_positions, hidden_size):
    """Partition the full input gradient; never silently omit or double-count tokens."""
    norms = np.asarray(token_norms, dtype=float)
    active = set(int(i) for i in active_positions)
    if norms.ndim != 1 or not np.isfinite(norms).all() or np.any(norms < 0):
        raise ValueError("Expected finite nonnegative per-token gradient norms.")
    if hidden_size <= 0 or any(i < 0 or i >= len(norms) for i in active):
        raise ValueError("Invalid gradient dimensions or token positions.")
    owner = {}
    positions = {}
    for name, indices in groups.items():
        selected = sorted(set(int(i) for i in indices) & active)
        for i in selected:
            if i in owner:
                raise ValueError(f"Token {i} belongs to both {owner[i]} and {name}.")
            owner[i] = name
        positions[name] = selected
    remaining = sorted(active - owner.keys())
    if remaining:
        positions["other_prompt"] = remaining
        owner.update({i: "other_prompt" for i in remaining})
    total_squared = float(np.square(norms[sorted(active)]).sum())
    result = {}
    for name, indices in positions.items():
        n = len(indices)
        squared = float(np.square(norms[indices]).sum())
        result[name] = {
            "norm": float(np.sqrt(squared)) if n else None,
            "n_tokens": n, "n_dimensions": n * hidden_size,
            "rms": float(np.sqrt(squared / (n * hidden_size))) if n else None,
            "squared_norm_share": squared / total_squared if total_squared and n else (0.0 if n else None),
        }
    return {"norm": float(np.sqrt(total_squared)), "groups": result, "token_groups": owner}

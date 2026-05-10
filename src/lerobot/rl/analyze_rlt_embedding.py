#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualize saved RLT embedding tokens from a compact RLT replay buffer.

This mirrors the RL Token reproduction sanity check: collect successful and
failed rollouts, project the saved RL tokens with PCA/t-SNE, and inspect whether
normal task phases form smooth trajectories while repeated failures cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import torch

from lerobot.rl.rlt_buffer import RLTReplayBuffer, RLTReplaySample
from lerobot.utils.utils import init_logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,I001
from matplotlib.lines import Line2D  # noqa: E402


_LOGGER = logging.getLogger(__name__)

_OUTCOME_MARKERS = {
    "success": "o",
    "failure": "X",
    "open": "s",
}


@dataclass
class TokenPoint:
    index: int
    episode_id: int | None
    episode_index: int
    episode_length: int
    episode_progress: float
    chunk_start_step: int | None
    outcome: str
    done: bool
    reward: float
    is_intervention: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-buffer-path",
        type=Path,
        default=Path("outputs/rlt_tinypi05v2_online/rlt_online_replay.pt"),
        help="Path to an RLTReplayBuffer .pt file, e.g. rlt_online_replay.pt or rlt_review_archive.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rlt_tinypi05v2_embedding_eval"),
        help="Directory where plots and metadata will be written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick previews. Keeps replay order from the start of the buffer.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity. It is lowered automatically for small buffers.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for t-SNE.")
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Only run PCA. Useful when scikit-learn is unavailable.",
    )
    parser.add_argument(
        "--no-review-sidecar",
        action="store_true",
        help="Do not apply a sibling .review.json sidecar before analysis.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Tinypi05v2 RLT Tokens",
        help="Plot title prefix.",
    )
    return parser.parse_args()


def _sample_outcome(sample: RLTReplaySample) -> str:
    if sample.success:
        return "success"
    if sample.failure:
        return "failure"
    if sample.done:
        return "success" if sample.reward > 0 else "failure"
    return "open"


def _episode_outcomes(samples: list[RLTReplaySample]) -> dict[int, str]:
    outcomes: dict[int, str] = {}
    for sample in samples:
        if sample.episode_id is None:
            continue

        episode_id = int(sample.episode_id)
        sample_outcome = _sample_outcome(sample)
        if sample_outcome == "open":
            outcomes.setdefault(episode_id, "open")
        elif sample_outcome == "success":
            outcomes[episode_id] = "success"
        elif outcomes.get(episode_id) != "success":
            outcomes[episode_id] = "failure"
    return outcomes


def _load_points(
    replay_path: Path,
    *,
    apply_review_sidecar: bool,
    max_samples: int | None,
) -> tuple[torch.Tensor, list[TokenPoint]]:
    replay = RLTReplayBuffer.load(replay_path, apply_review_sidecar=apply_review_sidecar)
    samples = replay.samples()
    if max_samples is not None:
        samples = samples[:max_samples]

    if not samples:
        raise ValueError(f"No RLT samples found in {replay_path}")

    tokens: list[torch.Tensor] = []
    points: list[TokenPoint] = []
    grouped_indices: dict[int | None, list[int]] = {}
    episode_outcomes = _episode_outcomes(samples)

    for index, sample in enumerate(samples):
        token = sample.rl_token.detach().cpu().float().flatten()
        if not torch.isfinite(token).all():
            _LOGGER.warning("Skipping non-finite RL token at replay index %d", index)
            continue
        outcome = (
            episode_outcomes.get(int(sample.episode_id), _sample_outcome(sample))
            if sample.episode_id is not None
            else _sample_outcome(sample)
        )
        tokens.append(token)
        grouped_indices.setdefault(sample.episode_id, []).append(len(points))
        points.append(
            TokenPoint(
                index=index,
                episode_id=sample.episode_id,
                episode_index=0,
                episode_length=1,
                episode_progress=0.0,
                chunk_start_step=sample.chunk_start_step,
                outcome=outcome,
                done=bool(sample.done),
                reward=float(sample.reward),
                is_intervention=bool(sample.is_intervention),
            )
        )

    if len(tokens) < 2:
        raise ValueError(f"Need at least 2 finite RL tokens for PCA, got {len(tokens)}")

    for point_indices in grouped_indices.values():
        sorted_point_indices = sorted(
            point_indices,
            key=lambda point_index: (
                math.inf
                if points[point_index].chunk_start_step is None
                else int(points[point_index].chunk_start_step),
                points[point_index].index,
            ),
        )
        episode_length = len(sorted_point_indices)
        denominator = max(episode_length - 1, 1)
        for episode_index, point_index in enumerate(sorted_point_indices):
            points[point_index].episode_index = episode_index
            points[point_index].episode_length = episode_length
            points[point_index].episode_progress = episode_index / denominator

    return torch.stack(tokens, dim=0), points


def _pca_projection(tokens: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:2].T
    variances = singular_values.pow(2) / max(tokens.shape[0] - 1, 1)
    total_variance = variances.sum().clamp_min(torch.finfo(variances.dtype).eps)
    explained = (variances[:2] / total_variance).tolist()
    return projection, [float(value) for value in explained]


def _tsne_projection(tokens: torch.Tensor, *, requested_perplexity: float, seed: int) -> tuple[torch.Tensor, float]:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for t-SNE. Re-run with "
            "`uv run --with scikit-learn python -m lerobot.rl.analyze_rlt_embedding ...` "
            "or pass --skip-tsne."
        ) from exc

    sample_count = int(tokens.shape[0])
    perplexity = min(float(requested_perplexity), max(1.0, (sample_count - 1) / 3.0))
    if perplexity >= sample_count:
        perplexity = max(1.0, sample_count - 1.0)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coords = tsne.fit_transform(tokens.numpy())
    return torch.from_numpy(coords).float(), perplexity


def _outcome_counts(points: list[TokenPoint]) -> dict[str, int]:
    counts = {"success": 0, "failure": 0, "open": 0}
    for point in points:
        counts[point.outcome] = counts.get(point.outcome, 0) + 1
    return counts


def _episode_counts(points: list[TokenPoint]) -> dict[str, int]:
    outcomes_by_episode: dict[int | None, str] = {}
    for point in points:
        if point.episode_id is None:
            continue
        outcomes_by_episode.setdefault(point.episode_id, point.outcome)
    counts = {"success": 0, "failure": 0, "open": 0}
    for outcome in outcomes_by_episode.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def _episode_ids(points: list[TokenPoint]) -> list[int | None]:
    return sorted(
        {point.episode_id for point in points},
        key=lambda episode_id: (-1 if episode_id is None else int(episode_id)),
    )


def _episode_indices(points: list[TokenPoint]) -> dict[int | None, list[int]]:
    groups: dict[int | None, list[int]] = {}
    for index, point in enumerate(points):
        groups.setdefault(point.episode_id, []).append(index)
    for indices in groups.values():
        indices.sort(
            key=lambda index: (
                points[index].episode_index,
                math.inf if points[index].chunk_start_step is None else int(points[index].chunk_start_step),
                points[index].index,
            )
        )
    return groups


def _episode_color_map(points: list[TokenPoint]) -> dict[int | None, tuple[float, float, float, float]]:
    episode_ids = _episode_ids(points)
    cmap = plt.get_cmap("tab20")
    return {
        episode_id: cmap(color_index % cmap.N)
        for color_index, episode_id in enumerate(episode_ids)
    }


def _episode_outcome(points: list[TokenPoint], indices: list[int]) -> str:
    outcomes = {points[index].outcome for index in indices}
    if "success" in outcomes:
        return "success"
    if "failure" in outcomes:
        return "failure"
    return "open"


def _episode_label(episode_id: int | None, outcome: str) -> str:
    episode_name = "unknown" if episode_id is None else str(episode_id)
    return f"ep {episode_name} ({outcome})"


def _plot_projection(
    coords: torch.Tensor,
    points: list[TokenPoint],
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    xy = coords.detach().cpu()
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    groups = _episode_indices(points)
    colors = _episode_color_map(points)

    legend_handles: list[Line2D] = []
    for episode_id in _episode_ids(points):
        indices = groups[episode_id]
        if not indices:
            continue

        outcome = _episode_outcome(points, indices)
        color = colors[episode_id]
        episode_xy = xy[indices]
        ax.plot(
            episode_xy[:, 0],
            episode_xy[:, 1],
            color=color,
            linewidth=1.2,
            alpha=0.72,
        )
        ax.scatter(
            episode_xy[:, 0],
            episode_xy[:, 1],
            s=26,
            color=color,
            alpha=0.82,
            linewidths=0,
        )

        start_index = indices[0]
        terminal_indices = [index for index in indices if points[index].done]
        end_index = terminal_indices[-1] if terminal_indices else indices[-1]
        ax.scatter(
            xy[start_index, 0],
            xy[start_index, 1],
            s=70,
            marker="^",
            color=color,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.scatter(
            xy[end_index, 0],
            xy[end_index, 1],
            s=100,
            marker=_OUTCOME_MARKERS.get(outcome, "s"),
            color=color,
            edgecolors="black",
            linewidths=1.0,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=_OUTCOME_MARKERS.get(outcome, "s"),
                color=color,
                label=_episode_label(episode_id, outcome),
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=7,
                linewidth=1.5,
            )
        )

    intervention_indices = [idx for idx, point in enumerate(points) if point.is_intervention]
    if intervention_indices:
        ax.scatter(
            xy[intervention_indices, 0],
            xy[intervention_indices, 1],
            s=42,
            marker="+",
            c="black",
            linewidths=1.0,
            label="intervention",
        )

    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="^",
                color="black",
                label="episode start",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=7,
                linewidth=0,
            ),
            Line2D(
                [0],
                [0],
                marker="+",
                color="black",
                label="intervention",
                markersize=8,
                linewidth=0,
            ),
        ]
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    _LOGGER.info("Saved %s", output_path)


def _write_points_csv(path: Path, points: list[TokenPoint], projections: dict[str, torch.Tensor]) -> None:
    fieldnames = list(asdict(points[0]).keys())
    for name in projections:
        fieldnames.extend([f"{name}_x", f"{name}_y"])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, point in enumerate(points):
            row: dict[str, Any] = asdict(point)
            for name, coords in projections.items():
                row[f"{name}_x"] = float(coords[row_index, 0])
                row[f"{name}_y"] = float(coords[row_index, 1])
            writer.writerow(row)
    _LOGGER.info("Saved %s", path)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    _LOGGER.info("Saved %s", path)


def main() -> None:
    init_logging()
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokens, points = _load_points(
        args.replay_buffer_path,
        apply_review_sidecar=not args.no_review_sidecar,
        max_samples=args.max_samples,
    )

    pca_coords, pca_explained = _pca_projection(tokens)
    projections = {"pca": pca_coords}
    plot_paths: dict[str, str] = {}
    pca_path = args.output_dir / "rlt_tokens_pca.png"
    _plot_projection(
        pca_coords,
        points,
        pca_path,
        title=f"{args.title} PCA",
        xlabel=f"PC1 ({pca_explained[0] * 100:.1f}% var)",
        ylabel=f"PC2 ({pca_explained[1] * 100:.1f}% var)",
    )
    plot_paths["pca"] = str(pca_path)

    tsne_perplexity = None
    tsne_error = None
    if not args.skip_tsne:
        try:
            tsne_coords, tsne_perplexity = _tsne_projection(
                tokens,
                requested_perplexity=args.perplexity,
                seed=args.seed,
            )
            projections["tsne"] = tsne_coords
            tsne_path = args.output_dir / "rlt_tokens_tsne.png"
            _plot_projection(
                tsne_coords,
                points,
                tsne_path,
                title=f"{args.title} t-SNE",
                xlabel="t-SNE 1",
                ylabel="t-SNE 2",
            )
            plot_paths["tsne"] = str(tsne_path)
        except RuntimeError as exc:
            tsne_error = str(exc)
            _LOGGER.warning("%s", tsne_error)

    coordinates_path = args.output_dir / "rlt_token_embedding_eval.pt"
    torch.save(
        {
            "tokens": tokens,
            "points": [asdict(point) for point in points],
            "projections": projections,
            "pca_explained_variance_ratio": pca_explained,
            "tsne_perplexity": tsne_perplexity,
        },
        coordinates_path,
    )
    _LOGGER.info("Saved %s", coordinates_path)

    points_csv_path = args.output_dir / "rlt_token_points.csv"
    _write_points_csv(points_csv_path, points, projections)

    episode_ids = sorted({point.episode_id for point in points if point.episode_id is not None})
    summary: dict[str, Any] = {
        "replay_buffer_path": str(args.replay_buffer_path),
        "sample_count": len(points),
        "token_dim": int(tokens.shape[-1]),
        "episode_count": len(episode_ids),
        "episode_ids": episode_ids,
        "sample_outcome_counts": _outcome_counts(points),
        "episode_outcome_counts": _episode_counts(points),
        "intervention_sample_count": sum(1 for point in points if point.is_intervention),
        "terminal_sample_count": sum(1 for point in points if point.done),
        "pca_explained_variance_ratio": pca_explained,
        "tsne_perplexity": tsne_perplexity,
        "tsne_error": tsne_error,
        "plots": plot_paths,
        "coordinates_path": str(coordinates_path),
        "points_csv_path": str(points_csv_path),
    }
    summary_path = args.output_dir / "summary.json"
    _write_summary(summary_path, summary)

    _LOGGER.info(
        "Analyzed %d samples (%d episodes): sample_outcomes=%s episode_outcomes=%s",
        len(points),
        len(episode_ids),
        summary["sample_outcome_counts"],
        summary["episode_outcome_counts"],
    )


if __name__ == "__main__":
    main()

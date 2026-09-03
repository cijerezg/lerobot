"""Dataset diagnostic for choosing temporal action bands and understanding normalization.

This probe measures demonstrated action chunks; it does not run the policy. Its purpose
is to expose the quantities a fixed band loss would act on before choosing band weights
``beta``:

1. coefficient power and cumulative energy at every DCT-II index;
2. candidate-band energy by action dimension and by annotated subtask phase;
3. the loss contribution implied by MSE-width, equal-band, and corpus-equalising beta;
4. how anchor/delta preprocessing and the configured action normalizer change the
   temporal spectrum.

Three spaces are deliberately kept separate:

``encoded_raw``
    Action after anchor/delta encoding but before dataset normalization. Frequencies
    retain their physical-coordinate interpretation, but different dimensions have
    different units and scales.

``model_target``
    The exact tensor the policy is trained to predict, produced by
    ``adapter.normalize_gt_actions``. This includes the normalizer's affine offset.

``model_motion``
    ``model_target - normalized(encoded zero)``. The subtraction removes the affine
    offset, which cancels in a prediction residual anyway, while retaining the exact
    per-step and per-dimension scale used by training. This is the primary space for
    deciding how a DCT of the normalized flow residual will behave.

The distinction matters in this repository: encoded-action statistics may have shape
``[chunk_step, action_dim]``. A time-varying scale does not commute with a DCT, so a
smooth raw-coordinate error can spread across model-space frequency bins. The
normalization panel reports that scale instead of silently calling either spectrum the
"right" one.

Energy does not determine task importance. Corpus-equalising beta is reported as the
exact *unfloored descriptive solution* to equal expected band contribution, not as a
recommendation. Phase-conditioned rows show whether a band mostly describes common
transport or the rarer positioning/contact phases the task actually fails on.

Outputs under ``<output_dir>/action_spectrum/``:

``action_spectrum.json``
    Summary, coefficient arrays, beta comparison, normalization audit, and phase counts.
``coefficient_power.csv``
    One row per space, action dimension, and DCT coefficient.
``band_energy.csv``
    Corpus and phase-conditioned band statistics with per-chunk quantiles.
``chunk_band_energy.csv``
    Long-form sampled-chunk table for alternative groupings without rerunning the probe.
``action_spectrum.png``
    Mean coefficient power and cumulative energy in model-motion and raw encoded spaces.
``band_beta_comparison.png``
    Candidate beta values and the average contribution each would induce.
``phase_band_energy.png``
    Target-energy share by annotated phase; a dataset description, not a task weight.
``normalization_audit.png``
    Raw units per normalized unit across chunk time, and normalized encoded zero.
``action_spectrum_report.md``
    Human-readable normalization, band, beta, and phase tables with interpretation.

Registered probe: ``probe_parameters.enable_action_spectrum``. It is checkpoint
invariant, so normally run it standalone or enable it for one deliberate validation
pass rather than at every checkpoint.

Standalone example:

    uv run python -m lerobot.probes.action_spectrum --config config_rl.yaml \
        --probe_parameters.action_spectrum_n_frames_per_episode 256
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    find_normalizer_step,
    get_action_chunk_lowdim,
    get_subtask_idx,
    get_subtask_str,
    identity_columns,
    joint_names_for_dim,
    load_extra_dataset,
    load_probe_dataset,
    makedirs,
    pad_to_action_width,
    register_config_choices,
    sample_episodes_evenly,
    subtask_group,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass(frozen=True)
class FrequencyBand:
    name: str
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


@dataclass
class ActionSpectrumProbeConfig(TrainRLServerPipelineConfig):
    """Tunables live under ``cfg.probe_parameters``."""


class _NormalizerOnlyAdapter:
    """The two adapter attributes this dataset-only probe needs, without a model."""

    def __init__(self, preprocessor, cfg, device: torch.device):
        self._preprocessor = preprocessor
        self._cfg = cfg
        self._device = device

    @property
    def chunk_size(self) -> int:
        return int(self._cfg.policy.chunk_size)

    @torch.no_grad()
    def normalize_gt_actions(
        self, gt_actions: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        # ReBot's 7 raw dims meet an 8-wide stats table; pad both sides exactly as
        # the preprocessor's unified-layout step does before the normalizer sees them.
        width = int(self._cfg.policy.output_features[ACTION].shape[0])
        encoded = _encode_actions(
            pad_to_action_width(gt_actions, width),
            None if state is None else pad_to_action_width(state, width),
            str(getattr(self._cfg.policy, "action_encoding", "absolute")),
        )
        normalizer = find_normalizer_step(self._preprocessor)
        batch = {
            TransitionKey.ACTION: encoded.unsqueeze(0).to(self._device),
            # Called as a bare step, so nothing else supplies the per-row stats index.
            TransitionKey.COMPLEMENTARY_DATA: identity_columns(self._cfg),
        }
        return normalizer(batch)[TransitionKey.ACTION].squeeze(0).float().cpu()


def _normalizer_only_adapter(cfg, dataset, device: torch.device) -> _NormalizerOnlyAdapter:
    """Build the same processors as training while deliberately not constructing a policy."""
    from lerobot.rl.rl_trainer import Trainer

    trainer = Trainer.for_config(cfg)
    preprocessor, _ = trainer.make_processors(cfg, dataset=dataset, is_main_process=True)
    return _NormalizerOnlyAdapter(preprocessor, cfg, device)


def _automatic_bands(horizon: int) -> list[FrequencyBand]:
    """A coarse complete partition when no explicit spec is supplied."""
    if horizon <= 1:
        return [FrequencyBand("all", 0, horizon)]
    edges = sorted({0, 1, min(3, horizon), min(9, horizon), horizon})
    names = ("dc", "transport", "maneuver", "fine")
    return [
        FrequencyBand(names[i], start, stop)
        for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:], strict=True))
        if stop > start
    ]


def parse_band_spec(spec: str, horizon: int) -> list[FrequencyBand]:
    """Parse ``name=0;name=1-2;name=3-`` and require an exact partition of ``0:T``."""
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if not spec.strip():
        return _automatic_bands(horizon)

    bands: list[FrequencyBand] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Band {item!r} must have the form name=start-stop.")
        name, interval = (part.strip() for part in item.split("=", 1))
        if not name:
            raise ValueError(f"Band {item!r} has an empty name.")
        if "-" in interval:
            start_text, stop_text = interval.split("-", 1)
            start = int(start_text)
            stop_inclusive = horizon - 1 if stop_text == "" else int(stop_text)
        else:
            start = stop_inclusive = int(interval)
        stop = stop_inclusive + 1
        if start < 0 or stop <= start or stop > horizon:
            raise ValueError(
                f"Band {name!r} resolves to [{start}, {stop}) outside horizon {horizon}."
            )
        bands.append(FrequencyBand(name, start, stop))

    if len({band.name for band in bands}) != len(bands):
        raise ValueError("Band names must be unique.")
    coverage = [index for band in bands for index in band.indices]
    if coverage != list(range(horizon)):
        raise ValueError(
            "Bands must be ordered, non-overlapping, and cover every DCT index exactly once; "
            f"got coverage {coverage}, expected {list(range(horizon))}."
        )
    return bands


def orthonormal_dct_matrix(horizon: int, *, dtype=torch.float64) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``C[k,n]``."""
    n = torch.arange(horizon, dtype=dtype).unsqueeze(0)
    k = torch.arange(horizon, dtype=dtype).unsqueeze(1)
    matrix = torch.cos(math.pi * k * (n + 0.5) / horizon)
    scales = torch.full((horizon,), math.sqrt(2.0 / horizon), dtype=dtype)
    scales[0] = math.sqrt(1.0 / horizon)
    return scales.unsqueeze(1) * matrix


def dct_coefficients(chunks: torch.Tensor) -> torch.Tensor:
    """Apply the orthonormal DCT-II along ``[N,T,D]`` chunk time."""
    if chunks.ndim != 3:
        raise ValueError(f"Expected chunks [N,T,D], got {tuple(chunks.shape)}.")
    matrix = orthonormal_dct_matrix(chunks.shape[1], dtype=chunks.dtype).to(chunks.device)
    return torch.einsum("kt,ntd->nkd", matrix, chunks)


def _encode_actions(actions: torch.Tensor, state: torch.Tensor | None, encoding: str) -> torch.Tensor:
    """The pre-normalization action encoding mirrored from the policy adapter."""
    actions = actions.float()
    if encoding == "absolute":
        return actions
    if state is None:
        raise ValueError(f"action_encoding={encoding!r} requires observation.state.")
    anchor = state[: actions.shape[-1]].float().unsqueeze(0)
    if encoding == "anchor":
        return actions - anchor
    if encoding == "delta":
        first = actions[:1] - anchor
        return torch.cat([first, torch.diff(actions, dim=0)], dim=0)
    raise ValueError(f"Unsupported action_encoding {encoding!r}.")


def _encoded_zero_actions(
    actions: torch.Tensor, state: torch.Tensor | None, encoding: str
) -> torch.Tensor:
    """Raw absolute chunk whose encoded value is zero."""
    if encoding in ("anchor", "delta"):
        if state is None:
            raise ValueError(f"action_encoding={encoding!r} requires observation.state.")
        return state[: actions.shape[-1]].float().unsqueeze(0).expand_as(actions).clone()
    return torch.zeros_like(actions)


def _chunk_context(dataset, global_idx: int, horizon: int) -> tuple[int, bool]:
    """Real steps before episode padding and whether their subtask label changes."""
    episode = int(dataset.hf_dataset[global_idx]["episode_index"].item())
    labels: list[int] = []
    for offset in range(horizon):
        index = global_idx + offset
        if index >= len(dataset):
            break
        row = dataset.hf_dataset[index]
        if int(row["episode_index"].item()) != episode:
            break
        labels.append(get_subtask_idx(dataset, index))
    known = [label for label in labels if label >= 0]
    return len(labels), len(set(known)) > 1


def _collect(adapter, dataset, cfg) -> dict:
    p = cfg.probe_parameters
    horizon = int(adapter.chunk_size)
    encoding = str(getattr(cfg.policy, "action_encoding", "absolute"))
    action_width = int(cfg.policy.output_features[ACTION].shape[0])
    n_per_episode = int(
        getattr(p, "action_spectrum_n_frames_per_episode", None) or p.n_frames_per_episode
    )
    max_episodes = getattr(p, "action_spectrum_max_episodes", None)
    samples = sample_episodes_evenly(
        dataset,
        n_per_episode=n_per_episode,
        max_episodes=max_episodes,
        seed=int(p.random_seed),
        stride=1,
    )
    if not samples:
        raise ValueError("No validation frames were selected for the action-spectrum probe.")

    encoded_raw: list[torch.Tensor] = []
    model_target: list[torch.Tensor] = []
    model_zero: list[torch.Tensor] = []
    rows: list[dict] = []
    for episode_idx, frame_idx, global_idx in samples:
        actions, state, actual_episode, actual_frame = get_action_chunk_lowdim(
            dataset, global_idx, horizon
        )
        if int(actual_episode) != int(episode_idx) or int(actual_frame) != int(frame_idx):
            raise RuntimeError("Spectrum sampler and low-dimensional chunk reader disagree.")
        # ReBot's 7 raw dims live in an 8-wide action layout. Pad once here, as
        # RoleAlignedBuffer does for the training batches, so the raw, target and zero
        # chunks are all in the canonical width the stats table is indexed by.
        actions = pad_to_action_width(actions, action_width)
        if state is not None:
            state = pad_to_action_width(state, action_width)
        raw = _encode_actions(actions, state, encoding)
        target = adapter.normalize_gt_actions(actions, state)
        zero_actions = _encoded_zero_actions(actions, state, encoding)
        zero = adapter.normalize_gt_actions(zero_actions, state)
        if target.shape != raw.shape or zero.shape != raw.shape:
            raise ValueError(
                "Action normalizer changed chunk shape: "
                f"raw={tuple(raw.shape)}, target={tuple(target.shape)}, zero={tuple(zero.shape)}."
            )

        subtask = get_subtask_str(dataset, get_subtask_idx(dataset, global_idx))
        valid_length, crosses_subtask = _chunk_context(dataset, global_idx, horizon)
        rows.append(
            {
                "episode_idx": int(episode_idx),
                "frame_idx": int(frame_idx),
                "global_idx": int(global_idx),
                "subtask": subtask,
                "phase": subtask_group(subtask),
                "valid_length": int(valid_length),
                "is_full_chunk": bool(valid_length == horizon),
                "crosses_subtask": bool(crosses_subtask),
            }
        )
        encoded_raw.append(raw.cpu())
        model_target.append(target.cpu())
        model_zero.append(zero.cpu())

    target_tensor = torch.stack(model_target).double()
    zero_tensor = torch.stack(model_zero).double()
    return {
        "rows": rows,
        "spaces": {
            "encoded_raw": torch.stack(encoded_raw).double(),
            "model_target": target_tensor,
            "model_motion": target_tensor - zero_tensor,
        },
        "model_zero": zero_tensor,
        "samples_requested_per_episode": n_per_episode,
        "max_episodes": max_episodes,
    }


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.double().flatten()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p10": float(torch.quantile(values, 0.10)),
        "median": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
    }


def _safe_shares(values: torch.Tensor, dim: int) -> torch.Tensor:
    denominator = values.sum(dim=dim, keepdim=True)
    return torch.where(denominator > 0, values / denominator, torch.zeros_like(values))


def _space_analysis(
    chunks: torch.Tensor,
    bands: list[FrequencyBand],
    names: list[str],
    fps: float,
    mask: torch.Tensor,
) -> dict:
    selected = chunks[mask]
    coefficients = dct_coefficients(selected)
    energy = coefficients.square()
    mean_power = energy.mean(dim=0)
    coefficient_share = _safe_shares(mean_power, dim=0)
    cumulative_share = coefficient_share.cumsum(dim=0)
    time_energy = selected.square().sum(dim=1)
    frequency_energy = energy.sum(dim=1)
    parseval_relative = (
        (time_energy - frequency_energy).abs()
        / time_energy.abs().clamp_min(torch.finfo(time_energy.dtype).eps)
    )

    coefficient_rows = []
    for d, name in enumerate(names):
        for k in range(selected.shape[1]):
            coefficient_rows.append(
                {
                    "dimension": name,
                    "dimension_index": d,
                    "k": k,
                    "frequency_hz": k * fps / (2.0 * selected.shape[1]),
                    "mean_power": float(mean_power[k, d]),
                    "energy_share": float(coefficient_share[k, d]),
                    "cumulative_energy_share": float(cumulative_share[k, d]),
                }
            )

    band_rows = []
    for d, name in enumerate(names):
        dimension_total = mean_power[:, d].sum().clamp_min(torch.finfo(mean_power.dtype).eps)
        for band in bands:
            sample_mean_power = energy[:, band.start : band.stop, d].mean(dim=1)
            total_power = mean_power[band.start : band.stop, d].sum()
            row = {
                "dimension": name,
                "dimension_index": d,
                "band": band.name,
                "start": band.start,
                "stop": band.stop,
                "width": band.width,
                "frequency_start_hz": band.start * fps / (2.0 * selected.shape[1]),
                "frequency_stop_hz": (band.stop - 1) * fps / (2.0 * selected.shape[1]),
                "mean_band_power": float(sample_mean_power.mean()),
                "total_coefficient_power": float(total_power),
                "energy_share_within_dimension": float(total_power / dimension_total),
                **{f"chunk_{key}": value for key, value in _quantiles(sample_mean_power).items()},
            }
            band_rows.append(row)

    # Existing dataset normalization makes averaging dimensions meaningful in the
    # model spaces. For encoded_raw this remains descriptive only; the JSON labels it.
    corpus_power = torch.stack(
        [energy[:, band.start : band.stop, :].mean() for band in bands]
    )
    mse_beta = torch.tensor(
        [band.width / selected.shape[1] for band in bands], dtype=corpus_power.dtype
    )
    equal_beta = torch.full(
        (len(bands),), 1.0 / len(bands), dtype=corpus_power.dtype
    )
    if bool(torch.all(corpus_power > torch.finfo(corpus_power.dtype).eps)):
        inverse = corpus_power.reciprocal()
        corpus_beta = inverse / inverse.sum()
    else:
        corpus_beta = torch.full_like(corpus_power, float("nan"))

    def contribution(beta: torch.Tensor) -> list[float | None]:
        values = beta * corpus_power
        if not bool(torch.all(torch.isfinite(values))) or float(values.sum()) <= 0:
            return [None] * len(values)
        return [float(value) for value in values / values.sum()]

    candidates = {
        "mse_width": {
            "beta": [float(value) for value in mse_beta],
            "expected_loss_share": contribution(mse_beta),
        },
        "equal_band": {
            "beta": [float(value) for value in equal_beta],
            "expected_loss_share": contribution(equal_beta),
        },
        "corpus_equalising_unfloored": {
            "beta": [float(value) if bool(torch.isfinite(value)) else None for value in corpus_beta],
            "expected_loss_share": contribution(corpus_beta),
        },
    }
    finite_corpus_beta = corpus_beta[torch.isfinite(corpus_beta)]
    beta_ratio = (
        float(finite_corpus_beta.max() / finite_corpus_beta.min())
        if finite_corpus_beta.numel() and float(finite_corpus_beta.min()) > 0
        else None
    )
    return {
        "n_chunks": int(selected.shape[0]),
        "mean_power": mean_power.tolist(),
        "coefficient_share": coefficient_share.tolist(),
        "cumulative_share": cumulative_share.tolist(),
        "coefficient_rows": coefficient_rows,
        "band_rows": band_rows,
        "band_summary": {
            band.name: {
                "mean_band_power": float(corpus_power[index]),
                "target_energy_share": float(
                    mean_power[band.start : band.stop].sum() / mean_power.sum().clamp_min(1e-30)
                ),
            }
            for index, band in enumerate(bands)
        },
        "beta_candidates": candidates,
        "corpus_equalising_beta_max_min_ratio": beta_ratio,
        "parseval_max_relative_error": float(parseval_relative.max()),
        "parseval_mean_relative_error": float(parseval_relative.mean()),
    }


def _phase_analysis(
    chunks: torch.Tensor,
    rows: list[dict],
    bands: list[FrequencyBand],
    base_mask: torch.Tensor,
) -> tuple[list[dict], dict]:
    coefficients = dct_coefficients(chunks).square()
    phases = sorted({row["phase"] for row in rows})
    phase_rows: list[dict] = []
    phase_summary: dict[str, dict] = {}
    for phase in phases:
        phase_mask = torch.tensor([row["phase"] == phase for row in rows]) & base_mask
        count = int(phase_mask.sum())
        if count == 0:
            continue
        selected = coefficients[phase_mask]
        total = selected.sum().clamp_min(torch.finfo(selected.dtype).eps)
        phase_summary[phase] = {"n_chunks": count, "bands": {}}
        for band in bands:
            sample_power = selected[:, band.start : band.stop, :].mean(dim=(1, 2))
            band_total = selected[:, band.start : band.stop, :].sum()
            stats = _quantiles(sample_power)
            row = {
                "phase": phase,
                "n_chunks": count,
                "band": band.name,
                "start": band.start,
                "stop": band.stop,
                "width": band.width,
                "target_energy_share": float(band_total / total),
                "total_coefficient_power": float(band_total),
                **stats,
            }
            phase_rows.append(row)
            phase_summary[phase]["bands"][band.name] = row
    for band in bands:
        band_total = sum(
            row["total_coefficient_power"] for row in phase_rows if row["band"] == band.name
        )
        for row in phase_rows:
            if row["band"] == band.name:
                row["share_of_band_corpus_energy"] = (
                    row["total_coefficient_power"] / band_total if band_total > 0 else 0.0
                )
    return phase_rows, phase_summary


def _normalizer_profile(adapter, horizon: int, action_dim: int) -> dict:
    """Raw units represented by one normalized unit, when the mode is affine."""
    step = find_normalizer_step(adapter._preprocessor)
    stats = step._tensor_stats.get(ACTION, {})
    mode = next(
        (
            getattr(value, "value", str(value))
            for key, value in step.norm_map.items()
            if str(getattr(key, "value", str(key))).lower() == "action"
        ),
        "unknown",
    )
    mode = str(mode).lower()
    if mode == "mean_std" and "std" in stats:
        scale = stats["std"]
    elif mode == "min_max" and "min" in stats and "max" in stats:
        scale = (stats["max"] - stats["min"]) / 2.0
    elif mode == "quantiles" and "q01" in stats and "q99" in stats:
        scale = (stats["q99"] - stats["q01"]) / 2.0
    elif mode == "quantile10" and "q10" in stats and "q90" in stats:
        scale = (stats["q90"] - stats["q10"]) / 2.0
    else:
        scale = None

    # MolmoAct2 can leave selected action dimensions in raw units. Its masked
    # normalizer applies the affine transform only where this mask is true.
    if scale is not None and "mask" in stats:
        mask = stats["mask"].detach().to(dtype=torch.bool, device=scale.device)
        if mask.ndim == 1 and scale.shape[-1] == mask.shape[0]:
            while mask.ndim < scale.ndim:
                mask = mask.unsqueeze(0)
            scale = torch.where(mask, scale, torch.ones_like(scale))

    result = {
        "mode": mode,
        "stats_shapes": {
            key: list(value.shape) for key, value in stats.items() if torch.is_tensor(value)
        },
        "scale_raw_units_per_normalized_unit": None,
        "scale_step_cv_by_dim": None,
        "scale_step_max_min_ratio_by_dim": None,
        "max_step_scale_ratio": None,
    }
    if scale is None:
        return result
    scale = scale.detach().double().cpu()
    if scale.ndim == 1:
        scale = scale.unsqueeze(0).expand(horizon, -1)
    if scale.shape[0] == 1:
        scale = scale.expand(horizon, -1)
    scale = scale[:horizon, :action_dim]
    if scale.shape != (horizon, action_dim):
        result["scale_shape_error"] = list(scale.shape)
        return result
    absolute_mean = scale.abs().mean(dim=0).clamp_min(torch.finfo(scale.dtype).eps)
    cv = scale.std(dim=0, unbiased=False) / absolute_mean
    minimum = scale.abs().min(dim=0).values.clamp_min(torch.finfo(scale.dtype).eps)
    ratio = scale.abs().max(dim=0).values / minimum
    result.update(
        scale_raw_units_per_normalized_unit=scale.tolist(),
        scale_step_cv_by_dim=[float(value) for value in cv],
        scale_step_max_min_ratio_by_dim=[float(value) for value in ratio],
        max_step_scale_ratio=float(ratio.max()),
    )
    return result


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    # Corpus and phase rows intentionally carry different context columns. Preserve
    # their union instead of letting DictWriter reject the later row type.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _band_spans(ax, bands: list[FrequencyBand]) -> None:
    colors = plt.cm.Set3(np.linspace(0, 1, len(bands)))
    for color, band in zip(colors, bands, strict=True):
        ax.axvspan(band.start - 0.5, band.stop - 0.5, color=color, alpha=0.13)
        ax.text(
            (band.start + band.stop - 1) / 2,
            0.98,
            band.name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
        )


def _plot_spectrum(
    analyses: dict[str, dict],
    names: list[str],
    bands: list[FrequencyBand],
    fps: float,
    path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for row_index, (space, title) in enumerate(
        (("model_motion", "Hold-relative model space"), ("encoded_raw", "Encoded raw space"))
    ):
        mean_power = np.asarray(analyses[space]["mean_power"])
        cumulative = np.asarray(analyses[space]["cumulative_share"])
        k90_lines = []
        for d, name in enumerate(names):
            axes[row_index, 0].plot(mean_power[:, d], marker="o", ms=2.5, label=name)
            axes[row_index, 1].plot(cumulative[:, d], marker="o", ms=2.5, label=name)
            crossing = np.flatnonzero(cumulative[:, d] >= 0.9)
            k90 = int(crossing[0]) if len(crossing) else len(cumulative) - 1
            k90_lines.append(f"{name}: {k90}")
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_ylabel("mean squared coefficient")
        axes[row_index, 0].set_title(f"{title}: coefficient power")
        axes[row_index, 1].set_ylim(0, 1.02)
        axes[row_index, 1].axhline(0.9, color="black", linestyle="--", linewidth=0.8)
        axes[row_index, 1].set_ylabel("cumulative energy share")
        axes[row_index, 1].set_title(f"{title}: cumulative energy")
        axes[row_index, 1].text(
            0.985,
            0.04,
            "$k_{90}$ (first index reaching 90%)\n" + "\n".join(k90_lines),
            transform=axes[row_index, 1].transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82},
        )
        for ax in axes[row_index]:
            _band_spans(ax, bands)
            ax.grid(True, alpha=0.25, linestyle=":")
    for ax in axes[-1]:
        ax.set_xlabel(f"DCT index k  (approximately k x {fps / (2 * len(analyses['model_motion']['mean_power'])):.3g} Hz)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(names), 7), frameon=False)
    fig.suptitle(
        "Demonstrated action spectrum: training metric versus encoded physical coordinates",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.12, top=0.91, hspace=0.30)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_beta(analysis: dict, bands: list[FrequencyBand], path: str) -> None:
    candidates = analysis["beta_candidates"]
    candidate_names = list(candidates)
    labels = [band.name for band in bands]
    x = np.arange(len(bands))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for index, name in enumerate(candidate_names):
        beta = [np.nan if value is None else value for value in candidates[name]["beta"]]
        contribution = [
            np.nan if value is None else value
            for value in candidates[name]["expected_loss_share"]
        ]
        offset = (index - (len(candidate_names) - 1) / 2) * width
        beta_bars = axes[0].bar(x + offset, beta, width=width, label=name)
        contribution_bars = axes[1].bar(x + offset, contribution, width=width, label=name)
        axes[0].bar_label(beta_bars, fmt="%.3f", fontsize=6.5, rotation=90, padding=2)
        axes[1].bar_label(
            contribution_bars,
            labels=["" if not np.isfinite(value) else f"{value:.1%}" for value in contribution],
            fontsize=6.5,
            rotation=90,
            padding=2,
        )
    axes[0].set_title(r"Declared total band weights $\beta_b$")
    axes[0].set_ylabel("beta")
    axes[1].set_title("Expected share of target-energy score under each beta")
    axes[1].set_ylabel("share")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Band width and corpus power answer different questions",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "MSE-width is exactly ordinary MSE; equal-band gives every declared band the same budget; "
        "corpus-equalising is the unfloored inverse-power solution and can be extreme. None encodes task importance.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(bottom=0.23, top=0.85, wspace=0.25)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_phase(phase_rows: list[dict], bands: list[FrequencyBand], path: str) -> None:
    if not phase_rows:
        return
    phases = sorted({row["phase"] for row in phase_rows})
    counts = {row["phase"]: row["n_chunks"] for row in phase_rows}
    lookup = {(row["phase"], row["band"]): row for row in phase_rows}
    within_phase = np.asarray(
        [[lookup[(phase, band.name)]["target_energy_share"] for band in bands] for phase in phases]
    )
    within_band = np.asarray(
        [
            [lookup[(phase, band.name)]["share_of_band_corpus_energy"] for band in bands]
            for phase in phases
        ]
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(14, 2.8 * len(bands)), max(5, 0.58 * len(phases))),
        sharey=True,
    )
    views = (
        (axes[0], within_phase, "Spectral shape within each phase", "each row sums to 100%"),
        (
            axes[1],
            within_band,
            "Which phases supply each band's corpus energy",
            "each column sums to 100%",
        ),
    )
    for ax, values, title, normalization in views:
        image = ax.imshow(
            values, cmap="magma", aspect="auto", vmin=0, vmax=max(values.max(), 1e-12)
        )
        ax.set_xticks(np.arange(len(bands)), [band.name for band in bands])
        ax.set_yticks(
            np.arange(len(phases)), [f"{phase}  (n={counts[phase]})" for phase in phases]
        )
        for row in range(len(phases)):
            for column in range(len(bands)):
                color = "white" if values[row, column] > 0.55 * values.max() else "black"
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.1%}",
                    ha="center",
                    va="center",
                    color=color,
                )
        ax.set_title(f"{title}\n({normalization})")
        ax.set_xlabel("candidate frequency band")
        fig.colorbar(image, ax=ax, label="energy share", fraction=0.046, pad=0.04)
    axes[0].set_ylabel("anchor-frame subtask phase")
    fig.suptitle(
        "Phase and frequency are two different views of demonstrated energy", fontweight="bold"
    )
    fig.text(
        0.5,
        0.01,
        "The left compares spectral shape; the right includes phase prevalence and motion magnitude. "
        "Neither is a task-importance weight. Crossing chunks remain labelled by their anchor frame.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(bottom=0.15, top=0.84, wspace=0.30)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_normalization(
    profile: dict, model_zero: torch.Tensor, names: list[str], path: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    scale = profile.get("scale_raw_units_per_normalized_unit")
    if scale is not None:
        scale_array = np.asarray(scale)
        for d, name in enumerate(names):
            axes[0].plot(scale_array[:, d], marker="o", ms=2.5, label=name)
        axes[0].set_title(f"Normalizer scale across chunk time ({profile['mode']})")
        axes[0].set_ylabel("raw encoded units per normalized unit")
        ratios = profile.get("scale_step_max_min_ratio_by_dim") or []
        ratio_lines = [
            f"{name}: {ratio:.2f}x" for name, ratio in zip(names, ratios, strict=False)
        ]
        axes[0].text(
            0.985,
            0.97,
            "stepwise max/min\n" + "\n".join(ratio_lines),
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82},
        )
    else:
        axes[0].text(0.5, 0.5, "Affine scale unavailable", ha="center", va="center")
        axes[0].set_title(f"Normalizer mode: {profile['mode']}")
    zero_mean = model_zero.mean(dim=0).numpy()
    for d, name in enumerate(names):
        axes[1].plot(zero_mean[:, d], marker="o", ms=2.5, label=name)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Where encoded zero lands in exact model space")
    axes[1].set_ylabel("normalized action")
    for ax in axes:
        ax.set_xlabel("chunk step")
        ax.grid(True, alpha=0.25, linestyle=":")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(names), 7), frameon=False)
    fig.suptitle(
        "Normalization audit: offsets cancel in residuals; time-varying scales do not",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.18, top=0.84, wspace=0.25)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _markdown_value(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}g}"


def _write_report(path: str, summary: dict, bands: list[FrequencyBand]) -> None:
    """Write the numbers needed for a loss decision without requiring JSON inspection."""
    analysis = summary["model_motion"]
    candidates = analysis["beta_candidates"]
    band_summary = analysis["band_summary"]
    normalization = summary["normalization"]
    lines = [
        "# Action spectrum diagnostic",
        "",
        "## What was measured",
        "",
        f"- Dataset chunks sampled: **{summary['n_chunks_sampled']}**; full chunks used in the primary aggregate: **{summary['n_chunks_primary']}**.",
        f"- Tail-padded sample fraction: **{summary['padded_fraction']:.1%}**.",
        f"- Chunks crossing a subtask boundary: **{summary['cross_subtask_fraction']:.1%}**.",
        f"- Action encoding: **{summary['action_encoding']}**; normalizer mode: **{normalization['mode']}**.",
        f"- DCT resolution: **{summary['frequency_resolution_hz']:.3g} Hz/index** at {summary['fps']:.3g} Hz control rate.",
        "",
        "The primary spectrum is `model_motion = normalized(target) - normalized(encoded zero)`. "
        "It removes the affine normalizer offset, which cancels in a prediction residual, but retains "
        "the exact per-step and per-dimension training scale.",
        "",
        "## Normalization audit",
        "",
        f"- Statistics shapes: `{normalization['stats_shapes']}`.",
        f"- Largest per-dimension scale max/min across chunk time: **{_markdown_value(normalization.get('max_step_scale_ratio'), 3)}x**.",
        "- A value of 1 means temporal DCT semantics are preserved up to one constant per dimension. "
        "A larger value means step-dependent normalization mixes raw-coordinate temporal modes.",
        "",
        "## Candidate bands and beta",
        "",
        "| band | k | approximate Hz | width | target energy share | beta: MSE width | beta: equal band | beta: corpus equalising |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, band in enumerate(bands):
        hz_start = band.start * summary["frequency_resolution_hz"]
        hz_stop = (band.stop - 1) * summary["frequency_resolution_hz"]
        corpus_beta = candidates["corpus_equalising_unfloored"]["beta"][index]
        lines.append(
            f"| {band.name} | {band.start}–{band.stop - 1} | {hz_start:.2f}–{hz_stop:.2f} | "
            f"{band.width} | {band_summary[band.name]['target_energy_share']:.2%} | "
            f"{candidates['mse_width']['beta'][index]:.4f} | "
            f"{candidates['equal_band']['beta'][index]:.4f} | "
            f"{_markdown_value(corpus_beta, 4)} |"
        )
    lines.extend(
        [
            "",
            "`beta: MSE width` reproduces ordinary MSE exactly. `beta: equal band` gives every "
            "declared band equal total budget. `beta: corpus equalising` is the unfloored inverse-power "
            "solution; it is diagnostic, not a recommendation.",
            "",
            "### Expected loss share under each beta",
            "",
            "| band | MSE width | equal band | corpus equalising |",
            "|---|---:|---:|---:|",
        ]
    )
    for index, band in enumerate(bands):
        shares = [
            candidates[name]["expected_loss_share"][index]
            for name in ("mse_width", "equal_band", "corpus_equalising_unfloored")
        ]
        lines.append(
            f"| {band.name} | " + " | ".join(
                "n/a" if value is None else f"{value:.2%}" for value in shares
            ) + " |"
        )

    lines.extend(
        [
            "",
            "## Phase-conditioned energy share",
            "",
            "Each row sums to 100%. These values describe where each phase moves; they do not say "
            "how much the task should value that phase.",
            "",
            "| phase | chunks | " + " | ".join(band.name for band in bands) + " |",
            "|---|---:|" + "---:|" * len(bands),
        ]
    )
    for phase, phase_summary in summary["phases"].items():
        shares = [phase_summary["bands"][band.name]["target_energy_share"] for band in bands]
        lines.append(
            f"| {phase} | {phase_summary['n_chunks']} | "
            + " | ".join(f"{value:.2%}" for value in shares)
            + " |"
        )
    lines.extend(
        [
            "",
            "### Phase contribution to each band's corpus energy",
            "",
            "Each band column sums to 100%. Unlike the table above, this includes phase prevalence "
            "and motion magnitude, but it still does not encode task importance.",
            "",
            "| phase | chunks | " + " | ".join(band.name for band in bands) + " |",
            "|---|---:|" + "---:|" * len(bands),
        ]
    )
    for phase, phase_summary in summary["phases"].items():
        shares = [
            phase_summary["bands"][band.name]["share_of_band_corpus_energy"] for band in bands
        ]
        lines.append(
            f"| {phase} | {phase_summary['n_chunks']} | "
            + " | ".join(f"{value:.2%}" for value in shares)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reading order",
            "",
            "1. Check the normalization scale plot. If it varies strongly over chunk time, decide whether "
            "the loss should use model-space or raw-coordinate temporal semantics.",
            "2. Use the coefficient spectrum and cumulative curves to revise band edges.",
            "3. Compare phase rows: a band dominated by common transport energy is not automatically the "
            "band that deserves more weight.",
            "4. Choose beta by the intended objective, then use `chunk_band_energy.csv` to inspect its "
            "per-example distribution before training.",
            "",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def _write_manifest(output_dir: str, summary: dict, bands: list[FrequencyBand]) -> None:
    metrics = [
        Metric("n_chunks_primary", "Full chunks analysed", fmt=0, primary=True),
        Metric("padded_fraction", "Sampled chunks requiring tail padding", fmt=3),
        Metric("cross_subtask_fraction", "Chunks crossing a subtask boundary", fmt=3),
        Metric(
            "normalization.max_step_scale_ratio",
            "Largest normalizer scale max/min across chunk time",
            fmt=2,
            baseline=1.0,
            primary=True,
            note="1 means normalization commutes with the temporal DCT up to one constant per dimension. Larger values mean the model-space spectrum mixes physical timescales through step-dependent scaling.",
        ),
        Metric(
            "model_motion.corpus_equalising_beta_max_min_ratio",
            "Unfloored inverse-power beta max/min",
            fmt=1,
            baseline=1.0,
            note="Descriptive conditioning warning, not a recommendation. A large ratio means corpus equalisation would amplify low-power bands aggressively and needs an explicit floor.",
        ),
        Metric(
            "model_motion.parseval_max_relative_error",
            "DCT Parseval max relative error",
            good="low",
            fmt=2,
            warn=1e-9,
            bad=1e-7,
            primary=True,
        ),
    ]
    metrics.extend(
        Metric(
            f"model_motion.band_summary.{band.name}.target_energy_share",
            f"Target energy share · {band.name}",
            fmt=3,
            note="Share of hold-relative normalized demonstration energy. This describes the sampled validation corpus; it is not beta and not task importance.",
        )
        for band in bands
    )
    write_index(
        output_dir,
        sys.modules[__name__],
        title="Action Spectrum",
        group="Actions",
        claim="Where does demonstrated action energy live after the exact training normalization, and what would candidate band weights do?",
        summary=summary,
        metrics=metrics,
        panels=[
            Panel(
                "action_spectrum.png",
                "Coefficient power before and after the training normalization",
                "Top is the primary loss-design space: exact normalized target minus normalized encoded zero, so affine offsets that cancel in residuals are absent. Bottom is anchor/delta-encoded raw action; read each dimension separately because units differ. Shaded regions are only the configured candidate partition.",
                primary=True,
            ),
            Panel(
                "band_beta_comparison.png",
                "What width, equal-band, and inverse-power beta actually imply",
                "Left shows declared total band budgets. Right multiplies each budget by measured corpus band power and normalizes the result. MSE-width reproduces MSE exactly; equal-band changes coefficient weights; inverse-power forces equal expected target-energy contribution and is shown without a floor so instability stays visible.",
                primary=True,
            ),
            Panel(
                "phase_band_energy.png",
                "Phase spectral shape versus phase contribution to each band",
                "The left heatmap is row-normalized and answers where each phase's motion lives. The right is column-normalized and answers which phases supply each band's corpus energy, including their prevalence. Compare transport/reach with position/grasp/contact before choosing beta: neither measured energy nor prevalence is task importance.",
                primary=True,
                refs=["subtask_sweep", "action_trace"],
            ),
            Panel(
                "normalization_audit.png",
                "Does action normalization preserve temporal frequency meaning?",
                "Left is the raw-unit scale behind one normalized unit at every chunk step. A flat line means the DCT keeps its physical temporal interpretation up to a dimension constant. Right is encoded zero after normalization; this offset is removed in model_motion because prediction residuals cancel it.",
            ),
            Panel(
                "band_energy.csv",
                "Corpus and phase-conditioned band statistics",
                "Long-form table with per-dimension corpus rows and phase rows, including per-chunk quantiles. Use it to test a different beta calculation without rerunning collection.",
            ),
            Panel(
                "chunk_band_energy.csv",
                "Per-chunk band powers with phase and padding context",
                "One row per sampled chunk, dimension, and band. This is the permanent escape hatch for regrouping reach/contact phases or testing a denominator without changing the probe.",
            ),
            Panel(
                "action_spectrum.json",
                "Complete machine-readable diagnostic",
                "Includes coefficient arrays, candidate beta, normalization shapes/scales, phase counts, band definitions, and sampling caveats.",
            ),
            Panel(
                "action_spectrum_report.md",
                "Readable normalization, band, beta, and phase tables",
                "Start here for the values behind the plots and a four-step reading order. The report explicitly separates measured target energy from the beta decision.",
            ),
        ],
        see_also=["actions", "action_trace", "objective", "subtask_sweep"],
    )


def run(adapter, dataset, cfg, output_dir: str) -> dict:
    """Collect, summarize, render, and register the action-spectrum diagnostic."""
    makedirs(output_dir)
    collected = _collect(adapter, dataset, cfg)
    rows = collected["rows"]
    spaces = collected["spaces"]
    _, horizon, action_dim = spaces["model_motion"].shape
    bands = parse_band_spec(str(cfg.probe_parameters.action_spectrum_bands), horizon)
    names = joint_names_for_dim(action_dim)
    fps = float(dataset.fps)

    full_mask = torch.tensor([row["is_full_chunk"] for row in rows])
    if int(full_mask.sum()) == 0:
        logging.warning("[action_spectrum] no full chunks sampled; primary aggregate includes padding.")
        primary_mask = torch.ones(len(rows), dtype=torch.bool)
        primary_population = "all_sampled_chunks"
    else:
        primary_mask = full_mask
        primary_population = "full_chunks_only"

    analyses = {
        name: _space_analysis(chunks, bands, names, fps, primary_mask)
        for name, chunks in spaces.items()
    }
    phase_rows, phase_summary = _phase_analysis(
        spaces["model_motion"], rows, bands, primary_mask
    )
    normalization = _normalizer_profile(adapter, horizon, action_dim)
    zero_stats = _quantiles(collected["model_zero"])
    normalization["encoded_zero_in_model_space"] = zero_stats

    coefficient_rows = []
    band_rows = []
    for space, analysis in analyses.items():
        coefficient_rows.extend({"space": space, **row} for row in analysis["coefficient_rows"])
        band_rows.extend({"scope": "corpus", "space": space, **row} for row in analysis["band_rows"])
    band_rows.extend(
        {
            "scope": "phase",
            "space": "model_motion",
            "dimension": "all_normalized_dimensions",
            "dimension_index": -1,
            **row,
        }
        for row in phase_rows
    )

    full_coefficients = dct_coefficients(spaces["model_motion"]).square()
    chunk_band_rows = []
    for sample_index, context in enumerate(rows):
        for d, dimension in enumerate(names):
            for band in bands:
                values = full_coefficients[sample_index, band.start : band.stop, d]
                chunk_band_rows.append(
                    {
                        **context,
                        "dimension": dimension,
                        "dimension_index": d,
                        "band": band.name,
                        "start": band.start,
                        "stop": band.stop,
                        "width": band.width,
                        "mean_band_power": float(values.mean()),
                        "total_coefficient_power": float(values.sum()),
                    }
                )

    summary = {
        "n_chunks_sampled": len(rows),
        "n_chunks_primary": int(primary_mask.sum()),
        "primary_population": primary_population,
        "padded_fraction": float(np.mean([not row["is_full_chunk"] for row in rows])),
        "cross_subtask_fraction": float(np.mean([row["crosses_subtask"] for row in rows])),
        "action_encoding": str(getattr(cfg.policy, "action_encoding", "absolute")),
        "action_stats_path": str(getattr(cfg.policy, "action_encoding_stats_path", "")),
        "horizon": horizon,
        "action_dim": action_dim,
        "joint_names": names,
        "fps": fps,
        "frequency_resolution_hz": fps / (2.0 * horizon),
        "bands": [asdict(band) | {"width": band.width} for band in bands],
        "sampling": {
            "requested_per_episode": collected["samples_requested_per_episode"],
            "max_episodes": collected["max_episodes"],
            "seed": int(cfg.probe_parameters.random_seed),
        },
        "normalization": normalization,
        "model_motion": {
            key: value
            for key, value in analyses["model_motion"].items()
            if key not in ("coefficient_rows", "band_rows")
        },
        "model_target": {
            key: value
            for key, value in analyses["model_target"].items()
            if key not in ("coefficient_rows", "band_rows")
        },
        "encoded_raw": {
            key: value
            for key, value in analyses["encoded_raw"].items()
            if key not in ("coefficient_rows", "band_rows")
        },
        "phases": phase_summary,
        "phase_counts_all_sampled": {
            phase: sum(row["phase"] == phase for row in rows)
            for phase in sorted({row["phase"] for row in rows})
        },
        "interpretation": {
            "primary_space": "model_motion",
            "beta_is_not_inferred": True,
            "corpus_equalising_beta_is_unfloored": True,
            "phase_label": "subtask_group at the anchor frame",
        },
    }

    _write_csv(os.path.join(output_dir, "coefficient_power.csv"), coefficient_rows)
    _write_csv(os.path.join(output_dir, "band_energy.csv"), band_rows)
    _write_csv(os.path.join(output_dir, "chunk_band_energy.csv"), chunk_band_rows)
    with open(os.path.join(output_dir, "action_spectrum.json"), "w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    _write_report(os.path.join(output_dir, "action_spectrum_report.md"), summary, bands)

    _plot_spectrum(analyses, names, bands, fps, os.path.join(output_dir, "action_spectrum.png"))
    _plot_beta(
        analyses["model_motion"], bands, os.path.join(output_dir, "band_beta_comparison.png")
    )
    _plot_phase(phase_rows, bands, os.path.join(output_dir, "phase_band_energy.png"))
    _plot_normalization(
        normalization,
        collected["model_zero"],
        names,
        os.path.join(output_dir, "normalization_audit.png"),
    )
    _write_manifest(output_dir, summary, bands)

    logging.info(
        f"[action_spectrum] {len(rows)} sampled / {int(primary_mask.sum())} primary chunks; "
        f"padded={summary['padded_fraction']:.1%}, cross-subtask={summary['cross_subtask_fraction']:.1%}, "
        f"normalizer max step-scale ratio={normalization.get('max_step_scale_ratio')}"
    )
    beta_candidates = analyses["model_motion"]["beta_candidates"]
    logging.info("[action_spectrum] band summary (model_motion):")
    for index, band in enumerate(bands):
        corpus_beta = beta_candidates["corpus_equalising_unfloored"]["beta"][index]
        logging.info(
            f"  {band.name:>12s} k={band.start:02d}-{band.stop - 1:02d} "
            f"energy={analyses['model_motion']['band_summary'][band.name]['target_energy_share']:7.2%} "
            f"beta[mse/equal/corpus]={beta_candidates['mse_width']['beta'][index]:.4f}/"
            f"{beta_candidates['equal_band']['beta'][index]:.4f}/"
            f"{_markdown_value(corpus_beta, 4)}"
        )
    logging.info(f"[action_spectrum] wrote {output_dir}/")
    return summary


@parser.wrap()
def probe_cli(cfg: ActionSpectrumProbeConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    root_dataset = load_probe_dataset(cfg)
    dataset = root_dataset
    if getattr(cfg, "val_dataset_path", None):
        logging.info(f"[action_spectrum] loading validation dataset {cfg.val_dataset_path}")
        dataset = load_extra_dataset(cfg.dataset.repo_id, cfg.val_dataset_path)
    adapter = _normalizer_only_adapter(cfg, root_dataset, device)
    output_dir = os.path.join(cfg.probe_parameters.output_dir, "action_spectrum")
    run(adapter, dataset, cfg, output_dir)


def main() -> None:
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401
    import lerobot.rl.pi05.rl_pi05  # noqa: F401
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    probe_cli()


if __name__ == "__main__":
    register_config_choices()
    main()

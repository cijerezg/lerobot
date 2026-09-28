"""Frozen training-set calibration for subtask-normalized AWR.

The shared scale and each group's exponential normalizer are fitted before actor
updates. Only the final mean weight is normalized over an optimizer's effective
batch. Group keys use embodiment and text, never dataset-local vocabulary IDs.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def group_key(embodiment: int, subtask: str) -> str:
    return json.dumps([int(embodiment), subtask], ensure_ascii=False)


def td_advantage(value, next_value, reward, done, discount, support_min, support_max):
    """Reward-aware TD error; terminal successors have exactly zero value.

    where, rather than multiplication by zero, also prevents a terminal's unused
    successor NaN from contaminating the target.
    """
    value, next_value, reward = (x.detach().float().reshape(-1) for x in (value, next_value, reward))
    successor = torch.where(done.reshape(-1).bool(), torch.zeros_like(next_value), next_value)
    target = (reward + discount * successor).clamp(support_min, support_max)
    return target - value


def validate_weight_parameters(beta: float, clip: float, lam: float = 1.0) -> None:
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("advantage_beta must be finite and positive.")
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("advantage_clip must be finite and positive.")
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("advantage_lambda must lie in [0, 1].")


class AWRCalibration:
    """Empirical group centers, one pooled residual scale, and group log E[exp].

    Raw scalar calibration samples are retained so a beta sweep can recompute
    its normalizers from training data, without fitting anything to validation.
    Missing groups are an error: a pooled fallback would silently undo balancing.
    """

    def __init__(self, samples: dict[str, list[float]], beta: float, clip: float, provenance=None):
        validate_weight_parameters(beta, clip)
        self.beta, self.clip = float(beta), float(clip)
        self.provenance = provenance or {}
        self.samples = {key: list(values) for key, values in samples.items()}
        self.centers: dict[str, float] = {}
        tensors = {}
        residual_sum = 0.0
        self.n = 0
        for key, values in self.samples.items():
            a = torch.tensor(values, dtype=torch.float64)
            if not a.numel() or not torch.isfinite(a).all():
                raise ValueError(f"Nonempty finite calibration advantages required for {key}.")
            self.centers[key] = a.mean().item()
            residual_sum += (a - self.centers[key]).square().sum().item()
            self.n += a.numel()
            tensors[key] = a
        if not self.n:
            raise ValueError("AWR calibration has no critic-valid training samples.")
        self.scale = max(math.sqrt(residual_sum / self.n), 1e-6)
        self.log_normalizers = {}
        for key, a in tensors.items():
            logits = ((a - self.centers[key]) / self.scale).clamp(-clip, clip) / beta
            self.log_normalizers[key] = (torch.logsumexp(logits, 0) - math.log(a.numel())).item()

    @classmethod
    def fit(cls, advantages, groups, beta, clip, provenance=None):
        a = torch.as_tensor(advantages).detach().double().cpu().reshape(-1)
        if len(groups) != a.numel():
            raise ValueError("One AWR group is required per advantage.")
        samples = defaultdict(list)
        for key, value in zip(groups, a.tolist(), strict=True):
            samples[key].append(value)
        return cls(samples, beta, clip, provenance)

    def standardized(self, advantages, groups):
        a = advantages.detach().double().reshape(-1)
        if a.numel() != len(groups):
            raise ValueError("One AWR group is required per advantage.")
        missing = set(groups) - self.centers.keys()
        if missing:
            raise ValueError(
                f"AWR calibration is missing {len(missing)} group(s): {sorted(missing)[:5]}. "
                "Increase advantage_calibration_batches and regenerate the calibration before training."
            )
        if not torch.isfinite(a).all():
            raise ValueError("Critic-valid AWR advantages must be finite.")
        centers = a.new_tensor([self.centers[key] for key in groups])
        return (a - centers) / self.scale

    def log_weights(self, advantages, groups):
        z = self.standardized(advantages, groups)
        log_z = z.new_tensor([self.log_normalizers[key] for key in groups])
        return z.clamp(-self.clip, self.clip) / self.beta - log_z, z

    def with_beta(self, beta):
        return type(self)(self.samples, beta, self.clip, self.provenance)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1, "score": "clipped_td", "beta": self.beta, "clip": self.clip,
            "scale": self.scale, "n": self.n, "provenance": self.provenance,
            "groups": {
                key: {"n": len(values), "center": self.centers[key],
                      "log_normalizer": self.log_normalizers[key], "advantages": values}
                for key, values in sorted(self.samples.items())
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path, beta=None, clip=None, provenance=None):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != 1 or payload.get("score") != "clipped_td":
            raise ValueError("Unsupported AWR calibration schema or advantage definition.")
        if provenance is not None and provenance != payload["provenance"]:
            raise ValueError("AWR calibration does not match this critic, reward rule, or training mixture.")
        return cls(
            {key: row["advantages"] for key, row in payload["groups"].items()},
            payload["beta"] if beta is None else beta,
            payload["clip"] if clip is None else clip,
            payload["provenance"],
        )


def normalize_log_weights(log_weights, lam=1.0, runtime=None):
    """Global mean-one weights across all accumulation batches and all ranks.

    Subtracting the common maximum is algebraically neutral and avoids overflow.
    Empty local ranks still take part in every collective. An all-skipped update
    returns an empty valid-row tensor; the caller restores skipped rows to one.
    """
    if not 0 <= lam <= 1:
        raise ValueError("advantage_lambda must lie in [0, 1].")
    log_weights = log_weights.detach().double().reshape(-1)
    if not torch.isfinite(log_weights).all():
        raise ValueError("AWR log weights must be finite.")
    def reduce(value, reduction):
        return value if runtime is None else runtime.reduce_scalar(value, reduction=reduction)
    count = reduce(float(log_weights.numel()), "sum")
    if count == 0:
        return log_weights.float()
    maximum = reduce(log_weights.max().item() if log_weights.numel() else -math.inf, "max")
    scaled = (log_weights - maximum).exp()
    total = reduce(scaled.sum().item(), "sum")
    return ((1 - lam) + lam * scaled * (count / total)).float()


def weight_telemetry(advantages, weights, standardized, groups, terminal, mistake_onset, total_rows, clip):
    """Small scalar dashboard plus per-group rows for the periodic JSONL audit."""
    a, w, z = (torch.as_tensor(x).double().reshape(-1) for x in (advantages, weights, standardized))
    metrics = {"awr/kept_fraction": a.numel() / max(total_rows, 1), "awr/groups_in_batch": len(set(groups))}
    if not a.numel():
        return metrics, []
    mass = w.sum()
    probability = w / mass
    metrics.update({
        "advantage_mean": a.mean().item(), "advantage_std": a.std(correction=0).item(),
        "adv_weight_ess_frac": (mass.square() / (w.numel() * w.square().sum())).item(),
        "adv_weight_kl": (probability * (probability * w.numel()).clamp_min(1e-30).log()).sum().item(),
        "adv_weight_min": w.min().item(), "adv_weight_max": w.max().item(),
        "awr/weight_mean": w.mean().item(), "awr/clip_fraction": (z.abs() >= clip).double().mean().item(),
        "awr/top5_weight_share": probability.topk(max(1, math.ceil(w.numel() * .05))).values.sum().item(),
    })
    for name, mask in (("terminal", terminal), ("mistake_onset", mistake_onset)):
        mask = torch.as_tensor(mask).bool()
        metrics[f"awr/{name}_fraction"] = mask.double().mean().item()
        metrics[f"awr/{name}_weight_share"] = probability[mask].sum().item()
    rows = []
    for key in sorted(set(groups)):
        mask = torch.tensor([g == key for g in groups])
        embodiment, subtask = json.loads(key)
        rows.append({"embodiment": embodiment, "subtask": subtask, "n": int(mask.sum()),
                     "advantage_mean": a[mask].mean().item(), "weight_mean": w[mask].mean().item(),
                     "weight_share": probability[mask].sum().item()})
    metrics["awr/group_weight_mean_min"] = min(row["weight_mean"] for row in rows)
    metrics["awr/group_weight_mean_max"] = max(row["weight_mean"] for row in rows)
    return metrics, rows

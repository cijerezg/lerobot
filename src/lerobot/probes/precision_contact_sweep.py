r"""Do the precision and contact clauses reach the chunk where they can matter?

``metadata_steering`` sweeps "The precision is $N$ of 5." and "The contact is <phrase>."
on evenly sampled frames. Most of those are transport, where the true precision is 1 and
the true contact is "not applicable", and a chunk of one second there should not change
under either sentence — the clause describes an event outside the horizon. The null that
probe reports is therefore the right answer on most of its frames, not evidence about the
channels. This probe is the ``subtask_scene_sweep`` treatment of the same question: the
frames are hand-reviewed pre-commit frames (1.0 / 0.5 / 0.2 s before the gripper close of
a grasp or the opening of a release; ``migration/precision_contact_sweep_frames.py``), and
each frame is swept only with contact codes plausible for its object — top vs side pinch
on a bottle, cloth vs top pinch on a shirt, set-down vs drop over a bin — plus two codes
implausible everywhere (strike, pour) as a control population on the same frame. Every
other clause is held at the deployment prompt (quality 5, no mistake, speed 5) with the
frame's TRUE precision and contact, so the reference chunk is the truthful prompt and a
swept row differs from it in one sentence.

Per frame, one batched forward over all rows under one flow draw (seed 0) and one over
$n_s$ reseeds of the truthful prompt. With $a_c$ the normalized chunk under condition $c$,
$a^{\star}$ the truthful chunk and $f$ the mean pairwise RMSE of the reseeds (the seed floor),

    $$D_c = \frac{\lVert a_c - a^{\star}\rVert}{f}$$

is the displacement a wrong sentence causes in seed-floor units: 1 = no more than
reseeding. Reported pooled over the plausible non-true contact codes ($D_{\mathrm{pl}}$), the
control codes ($D_{\mathrm{ctl}}$) and the four non-true precision levels ($D_{p}$).
$D_{\mathrm{pl}} \approx D_{\mathrm{ctl}}$ says the sentence is read as text, not as a
contact; $D_{\mathrm{pl}} \gg 1 > D_{\mathrm{ctl}}$ is the physical read. The precision ramp
gets ``metadata_steering``'s ordering test (projection on the $p_1 \to p_5$ axis, Kendall's
$\tau$) and its range $\lVert a_{p_5} - a_{p_1}\rVert / f$.

Three task-space readouts, each against the reseed null (the same quantity between a
reseed and the truthful chunk), never against the demonstration:

* **approach angle** (grasp frames offering top AND side pinch): FK angle between the
  gripper axis and straight down at the chunk's last step, the rubric's own definition
  (0 = top-down, 90 = horizontal, boundary 45). $\Delta_{\mathrm{ang}} = \theta(\text{side}) -
  \theta(\text{top})$, expected positive. The recorded state at the commit gives the
  demonstrated angle for reference only.
* **release height** (release frames offering set-down AND drop): end-effector $z$ at the
  last step, $\Delta_z = z(\text{set-down}) - z(\text{drop})$, expected negative.
* **approach path** (every frame): end-effector path length over the chunk,
  $\rho = L(p_5)/L(p_1)$, expected below 1 if a fine step slows the approach.

Frame list from ``probe_parameters.precision_contact_sweep_frames``; the probe raises when
the list resolves to other episode/frame indices in the loaded root and warns when the
root's own precision / contact rows disagree with the list. Registered probe: enable with
``probe_parameters.enable_precision_contact_sweep``. Added 2026-10-05.
"""

import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.annotation.vocab import CONTACT_VOCAB, phrase_for
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.metadata_steering import _level_projection
from lerobot.probes.subtask_scene_sweep import _box, _caption, _fmt, _median, _rmse
from lerobot.probes.subtask_sweep import _pairwise_rmse
from lerobot.probes.utils import (
    DEPLOYMENT_METADATA,
    frame_metadata_lookup,
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    register_config_choices,
)
from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

PRECISION_LEVELS = (1, 2, 3, 4, 5)
TOP, SIDE, SET_DOWN, DROP = 0, 1, 9, 10
SLUG = {e.code: e.slug for e in CONTACT_VOCAB}


class PrecisionContactSweepProbeConfig(TrainRLServerPipelineConfig):
    pass


def _approach_deg(kin: RebotKinematics, q: np.ndarray) -> float:
    """Angle between the gripper axis (end_link x) and straight down; 0 = top-down."""
    rotation = kin.ee_rotations(q[None, :])[0]
    return float(np.degrees(np.arccos(np.clip(-rotation[2, 0], -1.0, 1.0))))


def _path_length_m(kin: RebotKinematics, chunk: np.ndarray) -> float:
    path = kin.ee_path(chunk)
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _measure_frame(row: dict, frame: dict, adapter, kin: RebotKinematics, commit_state: np.ndarray, n_seeds: int) -> dict:
    true_p, true_c = int(row["true_precision"]), int(row["true_contact"])
    base = {**DEPLOYMENT_METADATA, "precision": true_p, "contact": true_c}
    plausible = [int(c) for c in row["plausible_contacts"]]
    control = [int(c) for c in row["control_contacts"]]
    if true_c not in plausible:
        raise ValueError(f"frame {row['global_idx']}: true contact {true_c} is not in its plausible set {plausible}")

    names: list[str] = ["base"]
    metadatas: list[dict] = [base]
    for k in PRECISION_LEVELS:
        if k != true_p:
            names.append(f"p{k}")
            metadatas.append({**base, "precision": k})
    for code in plausible + control:
        if code != true_c:
            names.append(f"c{code}")
            metadatas.append({**base, "contact": code})
    unnorm, norm = adapter.predict_action_chunk_batch(
        frame["obs"], frame["task"], [frame["subtask"]] * len(names),
        metadatas=metadatas,
        noise=adapter.flow_noise_like(len(names), 0),
        inference_action_mode="continuous",
    )
    acts = {name: norm[i] for i, name in enumerate(names)}
    raw = {name: unnorm[i].float().cpu().numpy() for i, name in enumerate(names)}
    acts[f"p{true_p}"] = acts["base"]
    acts[f"c{true_c}"] = acts["base"]
    raw[f"p{true_p}"] = raw["base"]
    raw[f"c{true_c}"] = raw["base"]

    seed_noise = torch.cat([adapter.flow_noise_like(1, seed) for seed in range(1, n_seeds + 1)], dim=0)
    seed_unnorm, seed_norm = adapter.predict_action_chunk_batch(
        frame["obs"], frame["task"], [frame["subtask"]] * n_seeds,
        metadatas=[base] * n_seeds,
        noise=seed_noise,
        inference_action_mode="continuous",
    )
    seed_draws = [seed_norm[i] for i in range(n_seeds)]
    seed_raw = [seed_unnorm[i].float().cpu().numpy() for i in range(n_seeds)]
    seed_floor, _ = _pairwise_rmse(seed_draws)
    floor = max(seed_floor, 1e-9)

    displacement = {name: _rmse(acts[name], acts["base"]) / floor for name in names if name != "base"}
    other_plausible = [c for c in plausible if c != true_c]
    projection, tau = _level_projection(acts, "p")
    precision = {
        "range_separation": _rmse(acts["p5"], acts["p1"]) / floor,
        "kendall_tau": tau,
        "projection": {str(k): v for k, v in projection.items()},
        "displacement": {str(k): displacement[f"p{k}"] for k in PRECISION_LEVELS if k != true_p},
        "displacement_mean": float(np.mean([displacement[f"p{k}"] for k in PRECISION_LEVELS if k != true_p])),
        "path_ratio": _path_length_m(kin, raw["p5"]) / max(_path_length_m(kin, raw["p1"]), 1e-9),
        "path_ratio_null": [_path_length_m(kin, s) / max(_path_length_m(kin, raw["base"]), 1e-9) for s in seed_raw],
    }
    contact = {
        "displacement": {str(c): displacement[f"c{c}"] for c in other_plausible + control},
        "plausible_mean": float(np.mean([displacement[f"c{c}"] for c in other_plausible])) if other_plausible else None,
        "control_mean": float(np.mean([displacement[f"c{c}"] for c in control])),
        "spread_plausible": _pairwise_rmse([acts[f"c{c}"] for c in plausible])[0] / floor if len(plausible) >= 2 else None,
    }

    approach = None
    if TOP in plausible and SIDE in plausible:
        angle = {name: _approach_deg(kin, raw[name][-1]) for name in (f"c{TOP}", f"c{SIDE}", "base")}
        approach = {
            "delta_deg": angle[f"c{SIDE}"] - angle[f"c{TOP}"],
            "null_deg": [_approach_deg(kin, s[-1]) - angle["base"] for s in seed_raw],
            "top_deg": angle[f"c{TOP}"], "side_deg": angle[f"c{SIDE}"], "base_deg": angle["base"],
            "commit_deg": _approach_deg(kin, commit_state),
        }
    height = None
    if SET_DOWN in plausible and DROP in plausible:
        z = {name: float(kin.ee_path(raw[name])[-1, 2]) for name in (f"c{SET_DOWN}", f"c{DROP}", "base")}
        height = {
            "delta_m": z[f"c{SET_DOWN}"] - z[f"c{DROP}"],
            "null_m": [float(kin.ee_path(s)[-1, 2]) - z["base"] for s in seed_raw],
            "set_down_m": z[f"c{SET_DOWN}"], "drop_m": z[f"c{DROP}"], "base_m": z["base"],
        }

    keep = ("episode_idx", "frame_idx", "global_idx", "segment_index", "commit_frame", "offset_s", "seconds_before_commit",
            "kind", "gt_subtask", "quality", "true_contact", "true_contact_slug", "plausible_contacts",
            "control_contacts", "true_precision")
    return {
        **{k: row[k] for k in keep},
        "seed_floor_mean": seed_floor,
        "precision": precision,
        "contact": contact,
        "approach": approach,
        "height": height,
        "_acts": acts,
    }


# ── Figures ───────────────────────────────────────────────────────────────────


def _render_summary(rows: list[dict], summary: dict, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
    _box(
        axes[0],
        [
            ("precision\nother level", [r["precision"]["displacement_mean"] for r in rows]),
            ("contact\nplausible other", [r["contact"]["plausible_mean"] for r in rows]),
            ("contact\ncontrol", [r["contact"]["control_mean"] for r in rows]),
        ],
        "RMSE to the truthful chunk / seed floor",
        "How far one changed sentence moves the chunk\n"
        f"$D_p$ = {_fmt(summary['precision']['displacement_median'])}x   "
        f"$D_{{pl}}$ = {_fmt(summary['contact']['plausible_median'])}x   "
        f"$D_{{ctl}}$ = {_fmt(summary['contact']['control_median'])}x",
    )
    axes[0].axhline(1.0, color="#E63946", linestyle="--", linewidth=1.2, label="seed floor")
    axes[0].legend(fontsize=8, frameon=False)
    _caption(axes[0], [
        r"Per frame: mean RMSE between the chunk under each wrong sentence and the chunk under the TRUE",
        r"precision + contact (same seed), over the reseed floor. 1 = no more than reseeding.",
        r"Physical read: plausible $\gg$ 1 and plausible $>$ control. Lexical read: plausible $\approx$ control.",
    ])

    for r in rows:
        axes[1].plot(PRECISION_LEVELS, [r["precision"]["projection"][str(k)] for k in PRECISION_LEVELS],
                     color="#457B9D", alpha=0.18, linewidth=0.9)
    means = [float(np.mean([r["precision"]["projection"][str(k)] for r in rows])) for k in PRECISION_LEVELS]
    sems = [float(np.std([r["precision"]["projection"][str(k)] for r in rows]) / np.sqrt(len(rows))) for k in PRECISION_LEVELS]
    axes[1].errorbar(PRECISION_LEVELS, means, yerr=sems, color="black", marker="o", linewidth=1.8, capsize=3, label="mean ± s.e.")
    axes[1].set_xticks(list(PRECISION_LEVELS))
    axes[1].set_xlabel("precision asked for")
    axes[1].set_ylabel("projection on the $p_1 \\to p_5$ axis")
    axes[1].set_title(
        "Is the precision ramp ordered?\n"
        rf"mean $\tau$ = {_fmt(summary['precision']['kendall_tau_mean'])}, range {_fmt(summary['precision']['range_separation_median'])}x the floor",
        fontsize=10,
    )
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(True, alpha=0.25, linestyle=":")
    _caption(axes[1], [
        r"$\pi(k) = \langle a_{p_k} - \bar a, u\rangle / \|u\|$, $u = a_{p_5} - a_{p_1}$; one faint line per frame. $\tau = +1$ is a",
        r"monotone staircase, 0 is five unrelated strings. Read the range against the floor first: a ramp",
        r"inside the sampler's own noise is not a ramp.",
    ])

    codes = sorted({int(c) for r in rows for c in r["contact"]["displacement"]})
    medians = [summary["contact"]["by_code_median"].get(str(c)) for c in codes]
    counts = [summary["contact"]["by_code_n"].get(str(c), 0) for c in codes]
    control = {int(c) for r in rows for c in r["control_contacts"]}
    colors = ["#999999" if c in control else "#457B9D" for c in codes]
    x = np.arange(len(codes))
    axes[2].bar(x, [m if m is not None else 0.0 for m in medians], color=colors, edgecolor="white")
    axes[2].axhline(1.0, color="#E63946", linestyle="--", linewidth=1.2)
    axes[2].set_xticks(x, [f"{SLUG[c]}\n(n={n})" for c, n in zip(codes, counts, strict=True)], fontsize=7)
    axes[2].set_ylabel("median $D_c$ (seed-floor units)")
    axes[2].set_title("Per contact code: distance from the truthful chunk\nblue = plausible for the frame, grey = control", fontsize=10)
    axes[2].grid(True, axis="y", alpha=0.25, linestyle=":")
    _caption(axes[2], [
        r"Each code's median over the frames that swept it as a non-true sentence. The same code is",
        r"plausible on some frames (a top pinch on a bottle) and never a control, so the colour is per code.",
    ])
    fig.suptitle(
        f"Precision / contact sweep on pre-commit frames (n={len(rows)} frames, {summary['n_windows']} windows)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    fig.savefig(os.path.join(output_dir, "precision_contact_sweep.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def _render_physical(rows: list[dict], summary: dict, output_dir: str) -> None:
    approach = [r["approach"] for r in rows if r["approach"] is not None]
    height = [r["height"] for r in rows if r["height"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
    _box(
        axes[0],
        [
            ("side pinch −\ntop pinch", [a["delta_deg"] for a in approach]),
            ("reseed null", [v for a in approach for v in a["null_deg"]]),
            ("demonstrated −\ntruthful chunk", [a["commit_deg"] - a["base_deg"] for a in approach]),
        ],
        "approach angle at the last step (deg)",
        "Does the contact word set the approach angle?\n"
        f"median $\\Delta$ {_fmt(summary['approach']['delta_median'], 1)} deg, null |.| {_fmt(summary['approach']['null_abs_median'], 1)} deg "
        f"(n={summary['approach']['n']})",
    )
    axes[0].axhline(0.0, color="#E63946", linestyle="--", linewidth=1.2)
    _caption(axes[0], [
        r"$\theta$ = FK angle between the gripper axis and straight down (0 top-down, 90 horizontal; rubric boundary 45),",
        r"at the last step of the chunk. Grasp frames whose plausible set has both pinches. Right box: the recorded",
        r"state at the commit minus the truthful chunk's end — how far the chunk is from the demonstrated pose anyway.",
    ])
    _box(
        axes[1],
        [
            ("set-down −\ndrop", [h["delta_m"] for h in height]),
            ("reseed null", [v for h in height for v in h["null_m"]]),
        ],
        "end-effector $z$ at the last step (m)",
        "Does 'a set-down' lower the hand?\n"
        f"median $\\Delta z$ {_fmt(summary['height']['delta_median'], 3)} m, null |.| {_fmt(summary['height']['null_abs_median'], 3)} m "
        f"(n={summary['height']['n']})",
    )
    axes[1].axhline(0.0, color="#E63946", linestyle="--", linewidth=1.2)
    _caption(axes[1], [
        r"Release frames offering set-down and drop. Expected negative: a set-down lowers the object onto the",
        r"surface before opening, a drop opens in the air.",
    ])
    _box(
        axes[2],
        [
            ("$L(p_5) / L(p_1)$", [r["precision"]["path_ratio"] for r in rows]),
            ("reseed null", [v for r in rows for v in r["precision"]["path_ratio_null"]]),
        ],
        "end-effector path length ratio over the chunk",
        "Does asking for precision slow the approach?\n"
        f"median $\\rho$ {_fmt(summary['precision']['path_ratio_median'])}, null {_fmt(summary['precision']['path_ratio_null_median'])}",
    )
    axes[2].axhline(1.0, color="#E63946", linestyle="--", linewidth=1.2)
    _caption(axes[2], [
        r"$L$ = summed end-effector step distance over the chunk. Below 1: the fine-precision chunk covers less",
        r"ground in the same second. Null: a reseed over the truthful chunk.",
    ])
    fig.suptitle("Precision / contact sweep — task-space readouts against the reseed null", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    fig.savefig(os.path.join(output_dir, "precision_contact_sweep_physical.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────


def _commit_states(dataset, frame_rows: list[dict]) -> dict[int, np.ndarray]:
    """global frame index of the row -> recorded state at its commit frame (same episode)."""
    states = np.stack(dataset.hf_dataset.with_format(None)["observation.state"])
    return {
        int(r["global_idx"]): states[int(r["global_idx"]) - int(r["frame_idx"]) + int(r["commit_frame"])]
        for r in frame_rows
    }


def run(adapter, dataset, cfg, output_dir: str) -> None:
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[precision_contact_sweep] needs continuous flow actions — skipping.")
        return
    p = cfg.probe_parameters
    frames_path = getattr(p, "precision_contact_sweep_frames", None)
    if not frames_path:
        raise ValueError("probe_parameters.precision_contact_sweep_frames must point at the reviewed frames.json")
    frame_rows = json.load(open(frames_path))
    n_seeds = max(int(p.precision_contact_sweep_n_seeds or p.n_seeds), 2)
    chunk_size = int(cfg.policy.chunk_size)

    makedirs(output_dir)
    kin = RebotKinematics()
    commit_states = _commit_states(dataset, frame_rows)
    true_labels = frame_metadata_lookup(dataset)
    n_forwards = sum(
        1 + (len(PRECISION_LEVELS) - 1) + len(r["plausible_contacts"]) - 1 + len(r["control_contacts"]) + n_seeds
        for r in frame_rows
    )
    logging.info(f"[precision_contact_sweep] {len(frame_rows)} frames from {frames_path}: {n_forwards} forward passes")

    adapter._set_probe_cuda_graph_enabled(False)
    rows: list[dict] = []
    try:
        for row in frame_rows:
            frame = probe_frame_inputs(dataset, cfg, int(row["global_idx"]), chunk_size, metadata=None)
            if (frame["episode_idx"], frame["frame_idx"]) != (row["episode_idx"], row["frame_idx"]):
                raise RuntimeError(
                    f"{frames_path} entry global {row['global_idx']} = ep{row['episode_idx']} fr{row['frame_idx']} "
                    f"but {dataset.root} resolves it to ep{frame['episode_idx']} fr{frame['frame_idx']}: frame list built for another root"
                )
            labels = true_labels.get(int(row["global_idx"]), {})
            for key in ("precision", "contact"):
                if key in labels and int(labels[key]) != int(row[f"true_{key}"]):
                    logging.warning(
                        f"[precision_contact_sweep] global {row['global_idx']}: dataset {key} {labels[key]} != "
                        f"frame list {row[f'true_{key}']}; sweeping around the frame list's"
                    )
            rows.append(_measure_frame(row, frame, adapter, kin, commit_states[int(row["global_idx"])], n_seeds))
    finally:
        adapter._restore_probe_cuda_graph_enabled()
    if not rows:
        logging.warning("[precision_contact_sweep] no frames measured.")
        return

    approach = [r["approach"] for r in rows if r["approach"] is not None]
    height = [r["height"] for r in rows if r["height"] is not None]
    codes = sorted({c for r in rows for c in r["contact"]["displacement"]}, key=int)
    kinds = ("grasp", "release")
    offsets = sorted({r["offset_s"] for r in rows}, reverse=True)
    summary = {
        "n_frames": len(rows),
        "n_windows": len({(r["episode_idx"], r["segment_index"]) for r in rows}),
        "n_seeds": n_seeds,
        "frames_path": frames_path,
        "seed_floor_median": _median([r["seed_floor_mean"] for r in rows]),
        "precision": {
            "range_separation_median": _median([r["precision"]["range_separation"] for r in rows]),
            "kendall_tau_mean": float(np.mean([r["precision"]["kendall_tau"] for r in rows])),
            "displacement_median": _median([r["precision"]["displacement_mean"] for r in rows]),
            "by_level_median": {
                str(k): _median([r["precision"]["displacement"].get(str(k)) for r in rows]) for k in PRECISION_LEVELS
            },
            "path_ratio_median": _median([r["precision"]["path_ratio"] for r in rows]),
            "path_ratio_null_median": _median([v for r in rows for v in r["precision"]["path_ratio_null"]]),
            "true_level_counts": {str(k): sum(r["true_precision"] == k for r in rows) for k in PRECISION_LEVELS},
        },
        "contact": {
            "plausible_median": _median([r["contact"]["plausible_mean"] for r in rows]),
            "control_median": _median([r["contact"]["control_mean"] for r in rows]),
            "spread_plausible_median": _median([r["contact"]["spread_plausible"] for r in rows]),
            "by_code_median": {c: _median([r["contact"]["displacement"].get(c) for r in rows]) for c in codes},
            "by_code_n": {c: sum(c in r["contact"]["displacement"] for r in rows) for c in codes},
            "by_code_slug": {c: SLUG[int(c)] for c in codes},
        },
        "approach": {
            "n": len(approach),
            "delta_median": _median([a["delta_deg"] for a in approach]),
            "delta_positive_fraction": float(np.mean([a["delta_deg"] > 0 for a in approach])) if approach else None,
            "null_abs_median": _median([abs(v) for a in approach for v in a["null_deg"]]),
            "commit_minus_base_median": _median([a["commit_deg"] - a["base_deg"] for a in approach]),
        },
        "height": {
            "n": len(height),
            "delta_median": _median([h["delta_m"] for h in height]),
            "delta_negative_fraction": float(np.mean([h["delta_m"] < 0 for h in height])) if height else None,
            "null_abs_median": _median([abs(v) for h in height for v in h["null_m"]]),
        },
        "by_kind": {
            k: {
                "n": sum(r["kind"] == k for r in rows),
                "precision_displacement_median": _median([r["precision"]["displacement_mean"] for r in rows if r["kind"] == k]),
                "contact_plausible_median": _median([r["contact"]["plausible_mean"] for r in rows if r["kind"] == k]),
                "contact_control_median": _median([r["contact"]["control_mean"] for r in rows if r["kind"] == k]),
            }
            for k in kinds
        },
        "by_offset": {
            str(o): {
                "n": sum(r["offset_s"] == o for r in rows),
                "precision_displacement_median": _median([r["precision"]["displacement_mean"] for r in rows if r["offset_s"] == o]),
                "contact_plausible_median": _median([r["contact"]["plausible_mean"] for r in rows if r["offset_s"] == o]),
            }
            for o in offsets
        },
        "verdict_note": (
            "D_plausible ~1 => the contact word does not reach the chunk even where it could; "
            "D_plausible ~ D_control => read as text, not as a contact; "
            "approach.delta above its null => side/top pinch sets the FK approach angle; "
            "height.delta below minus its null => set-down lowers the hand; "
            "precision.range_separation ~1 => the precision number is not read here either."
        ),
        "per_frame": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    with open(os.path.join(output_dir, "precision_contact_sweep.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _render_summary(rows, summary, output_dir)
    _render_physical(rows, summary, output_dir)

    write_index(
        output_dir,
        sys.modules[__name__],
        title="Precision / Contact Sweep",
        group="Steering",
        claim="Do the precision and contact clauses move the chunk in the second before the gripper commits, with only plausible alternatives offered?",
        summary=summary,
        see_also=["metadata_steering", "subtask_scene_sweep"],
        metrics=[
            Metric("contact.plausible_median", "D plausible contact", good="high", fmt=2, baseline=1.0, primary=True, trend=True,
                   note="RMSE to the truthful chunk / seed floor, over the plausible non-true codes."),
            Metric("contact.control_median", "D control contact (strike, pour)", good="none", fmt=2, baseline=1.0, trend=True),
            Metric("contact.spread_plausible_median", "plausible-set spread S_c", good="high", fmt=2, baseline=1.0, trend=True),
            Metric("precision.range_separation_median", "precision range p5 vs p1", good="high", fmt=2, baseline=1.0, primary=True, trend=True),
            Metric("precision.kendall_tau_mean", "precision ramp tau", good="high", fmt=2, baseline=0.0, trend=True),
            Metric("approach.delta_median", "approach angle side − top (deg)", good="high", fmt=1, baseline=0.0, primary=True, trend=True,
                   note="Read against approach.null_abs_median; grasp frames offering both pinches only."),
            Metric("approach.null_abs_median", "approach angle null |delta| (deg)", good="none", fmt=1),
            Metric("height.delta_median", "release height set-down − drop (m)", good="low", fmt=3, baseline=0.0, trend=True,
                   note="Read against height.null_abs_median."),
            Metric("height.null_abs_median", "release height null |delta| (m)", good="none", fmt=3),
            Metric("precision.path_ratio_median", "path length p5 / p1", good="low", fmt=2, baseline=1.0, trend=True,
                   note="Read against precision.path_ratio_null_median."),
        ],
        panels=[
            Panel("precision_contact_sweep.png", "Displacement by sentence, precision ramp order, per contact code",
                  "Left: how far one changed sentence moves the chunk off the truthful prompt, in seed-floor units — "
                  "the channel is read only if the plausible box clears 1, and read physically only if it clears the "
                  "control box. Middle: projection of the five precision levels on their own axis; a staircase is a "
                  "read scale. Right: the same displacement per contact code, blue plausible, grey control.",
                  primary=True, refs=["metadata_steering"]),
            Panel("precision_contact_sweep_physical.png", "Task-space readouts against the reseed null",
                  "Left: FK approach angle under 'a side pinch' minus 'a top pinch' at the chunk's end (positive = the "
                  "word sets the angle). Middle: end-effector height under 'a set-down' minus 'a drop' (negative = "
                  "lowers). Right: end-effector path length under precision 5 over precision 1 (below 1 = slower).",
                  primary=True, refs=["subtask_scene_sweep"]),
        ],
    )
    c, pr = summary["contact"], summary["precision"]
    logging.info(
        f"[precision_contact_sweep] n={len(rows)}  D plausible {_fmt(c['plausible_median'])}x control {_fmt(c['control_median'])}x "
        f"spread {_fmt(c['spread_plausible_median'])}x  precision range {_fmt(pr['range_separation_median'])}x tau {_fmt(pr['kendall_tau_mean'])}  "
        f"approach delta {_fmt(summary['approach']['delta_median'], 1)} deg vs null {_fmt(summary['approach']['null_abs_median'], 1)} "
        f"({summary['approach']['n']})  height delta {_fmt(summary['height']['delta_median'], 3)} m vs null "
        f"{_fmt(summary['height']['null_abs_median'], 3)} ({summary['height']['n']})  path ratio {_fmt(pr['path_ratio_median'])} "
        f"vs null {_fmt(pr['path_ratio_null_median'])}"
    )


@parser.wrap()
def cli(cfg: PrecisionContactSweepProbeConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    val_path = getattr(cfg, "val_dataset_path", None)
    if val_path:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=val_path)
        dataset.delta_timestamps = None
        dataset.delta_indices = None
    else:
        dataset = load_probe_dataset(cfg)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    step = 0
    for part in str(getattr(cfg.policy, "pretrained_path", "") or "").split(os.sep):
        if part.isdigit():
            step = int(part)
    run(adapter, dataset, cfg, os.path.join(cfg.probe_parameters.output_dir, "validation", f"step_{step:08d}", "precision_contact_sweep"))


def main() -> None:
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    import lerobot.rl.pi05.rl_pi05  # noqa: F401 — registers PI05RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401 — registers robot configs
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401 — registers teleop configs

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    register_config_choices()
    main()

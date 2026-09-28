"""Critic input sensitivity, connected to individual frames.

The primary gradient covers RGB patches, state values and consumed depth features entering
critic fusion, holding task/subtask, metadata and other prompt inputs fixed.
The full input gradient remains available as a diagnostic. It is
not a gradient of the training loss, nor a derivative in physical joint units.
MolmoAct2 discretizes state and categorical inputs; their embedding derivatives
are measured at this explicitly named continuous boundary. Depth/history
placeholder embeddings are distinct from raw depth/history, which this critic
forward does not consume. The learned value token and padding are excluded.

Open gradient_explorer.html, choose a source/episode when available, then an
input group, and click a dot to inspect its frames. Low/median/high buttons stay
within the selected exact subtask; the episode view also keeps the episode fixed.
The input breakdown partitions the squared total norm; RMS also accounts for
component size. A large norm is sensitivity, not a label for an execution error.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from scipy.stats import rankdata

from lerobot.probes.critic_subtasks import sample_text_groups
from lerobot.probes.critic_gradient_view import image_state_gradient_fields
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import canonical_camera_obs, probe_frame_inputs


def _save_frame_images(obs, cfg, row, root):
    from lerobot.probes.critic import _frame_to_uint8
    images = []
    folder = root / "frames"
    folder.mkdir(exist_ok=True)
    for key, value in sorted(canonical_camera_obs(obs, cfg).items()):
        if not key.startswith("observation.images."):
            continue
        camera = key.removeprefix("observation.images.")
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", camera)
        filename = f"frames/{row['global_idx']:08d}_{safe}.jpg"
        image = Image.fromarray(_frame_to_uint8(value))
        image.thumbnail((640, 480))
        image.save(root / filename, quality=85)
        images.append({"camera": camera, "path": filename})
    return images


def _validate_measurement(result):
    norm = float(result["norm"])
    if not np.isfinite(norm) or norm < 0:
        raise ValueError("Invalid full input gradient norm.")
    groups = result["groups"]
    if not groups:
        raise ValueError("A complete input gradient requires its input-group partition.")
    squares = []
    for group in groups.values():
        value = group["norm"]
        if value is not None:
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid input-group norm.")
            squares.append(float(value) ** 2)
    if not np.isclose(sum(squares), norm ** 2, rtol=2e-5, atol=1e-12):
        raise ValueError("Input-group squared norms do not sum to the full gradient squared norm.")
    if not result.get("boundary"):
        raise ValueError("The differentiation boundary must be explicit.")
    return norm


def render_gradient_report(measured, summary, output_dir):
    """Render from saved measurements; no model or image decoding required."""
    root = Path(output_dir)
    if not measured:
        return
    measured = [{**r, **image_state_gradient_fields(r["gradient"])} for r in measured]
    order = sorted({r["subtask"] for r in measured})
    labels = {text: f"{text} (n={sum(r['subtask'] == text for r in measured)})" for text in order}
    data = {"norm": [r["observation_grad_norm"] for r in measured], "text": [labels[r["subtask"]] for r in measured]}
    fig, ax = plt.subplots(figsize=(13, max(4, 0.5 * len(order) + 1.5)))
    sns.boxplot(data=data, x="norm", y="text", order=list(labels.values()), ax=ax, color="lightblue", fliersize=0)
    sns.stripplot(data=data, x="norm", y="text", order=list(labels.values()), ax=ax, color="0.3", size=3)
    ax.set_xscale("symlog", linthresh=1e-8)
    ax.set_xlim(max(0, min(data["norm"]) * 0.8), max(1e-8, max(data["norm"]) * 1.2))
    ax.set(xlabel="||∂E[V]/∂(image features, state-value embeddings, depth features)||₂", ylabel="",
           title="RGB + state + available depth gradient distribution within exact subtask")
    fig.tight_layout()
    fig.savefig(root / "gradient_magnitudes.png", dpi=150)
    plt.close(fig)
    payload = {"records": measured, "summary": summary}
    # Data is never parsed as executable script or interpolated into HTML labels.
    packed = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_suffix(".html").read_text()
    (root / "gradient_explorer.html").write_text(template.replace("__PROBE_DATA__", packed))


def run_critic_gradients(adapter, dataset, cfg, records, output_dir, *, selected_records=None):
    """Measure image/state sensitivity, retaining the full gradient breakdown."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    p = cfg.probe_parameters
    selected = (list(selected_records) if selected_records is not None else
                sample_text_groups(records, int(p.critic_grad_frames), int(p.critic_grad_frames_per_subtask),
                                   int(p.random_seed) + 1, int(p.max_labels)))
    measured, failures = [], []
    consecutive_failures = 0
    disabled = selected_records is None and int(p.critic_grad_frames) <= 0
    status = "disabled" if disabled else "no_eligible_groups" if not selected else "complete"
    for number, row in enumerate(selected):
        try:
            frame = probe_frame_inputs(dataset, cfg, row["global_idx"], adapter.chunk_size, metadata=row["metadata"])
            result = adapter.critic_input_gradients(frame["obs"], frame["task"], frame["subtask"], metadata=frame["metadata"])
            norm = _validate_measurement(result)
            value = float(result["value"])
            if not np.isfinite(value):
                raise ValueError("Non-finite value during gradient evaluation.")
            # The value and gradient come from the same forward. The recorded
            # return stays available as context; it is not an execution-error label.
            item = {**row, "value": value, "grad_norm": norm, "gradient": result}
            item.update(image_state_gradient_fields(result))
            item["images"] = _save_frame_images(frame["obs"], cfg, row, root)
            measured.append(item)
            consecutive_failures = 0
        except NotImplementedError as exc:
            status = "unsupported"
            logging.warning("[CRITIC SENSITIVITY] %s", exc)
            break
        except Exception as exc:
            failures.append({"global_idx": row["global_idx"], "error": str(exc)})
            logging.warning("[CRITIC SENSITIVITY] frame %s failed: %s", row["global_idx"], exc)
            consecutive_failures += 1
            status = "partial"
            if consecutive_failures >= 3:
                status = "stopped_after_failures"
                break
        if (number + 1) % 20 == 0:
            logging.info("[CRITIC SENSITIVITY] %s/%s frames", number + 1, len(selected))
    for text in sorted({r["subtask"] for r in measured}):
        group = [r for r in measured if r["subtask"] == text]
        ranks = (rankdata([r["observation_grad_norm"] for r in group]) - 0.5) / len(group)
        for row, rank in zip(group, ranks, strict=True):
            row["within_text_rank"] = float(rank)
    summary = {
        "grad_n_frames": len(measured), "grad_n_failed": len(failures), "grad_n_requested": len(selected),
        "grad_n_subtasks": len({r["subtask"] for r in measured}), "grad_status": status,
        "grad_norm_median": float(np.median([r["grad_norm"] for r in measured])) if measured else None,
        "image_state_grad_norm_median": float(np.median([r["image_state_grad_norm"] for r in measured])) if measured else None,
        "observation_grad_norm_median": float(np.median([r["observation_grad_norm"] for r in measured])) if measured else None,
        "gradient_scope": "observation", "within_text_rank_scope": "observation",
    }
    (root / "critic_gradients.json").write_text(json.dumps({
        "schema": 3, "summary": summary, "records": measured, "failures": failures,
        "interpretation": "Primary norm and within-text ranks use RGB patch, state-value and consumed depth embeddings. grad_norm retains the full input norm for comparison; no inference about robot failure probability.",
    }, indent=2, allow_nan=False))
    render_gradient_report(measured, summary, root)
    return {"grad_mags": torch.tensor([r["observation_grad_norm"] for r in measured])}, summary


def write_sensitivity_manifest(output_dir, summary):
    root = Path(output_dir)
    panels = [
        Panel("gradient_explorer.html", "Select a gradient point and inspect its frames", primary=True,
              how="RGB + state + available depth is the default. Click a dot to inspect its frames; image-only, state-only and full-input diagnostics remain available. Low/median/high buttons stay within the exact subtask and selected episode. Derivatives are in embedding coordinates, not physical units."),
        Panel("gradient_magnitudes.png", "RGB + state + available depth gradient distribution", primary=False,
              how="Joint L2 norm over RGB patch features, state-value embeddings and consumed depth features. Fixed state wording, delimiters, task/subtask text, metadata and depth/history placeholders are excluded. Token count and feature scale affect the norm."),
    ] if summary.get("grad_n_frames", 0) else []
    panels += [Panel(str(path.relative_to(root)), f"Episode {path.stem[2:]}: next and unrelated text",
                     how="Only the subtask text is changed. Dashed references describe a hypothesis, not supervised targets.")
               for path in sorted((root / "episode_swaps").glob("*.png"))
               if "critic_swap_next_gap" in summary]
    return write_index(str(root), sys.modules[__name__], title="Critic Input Sensitivity", group="Critic",
        claim="Which inputs change the critic's value, and what do high- and low-sensitivity frames look like?",
        summary=summary, status="warn" if summary.get("grad_status") in {"partial", "stopped_after_failures", "unsupported"} else None,
        metrics=[
            Metric("grad_n_frames", "Frames with observation gradients", good="none", fmt=0, primary=True),
            Metric("grad_n_subtasks", "Exact subtasks", good="none", fmt=0, primary=True),
            *([Metric("grad_n_episodes", "Episodes", good="none", fmt=0, primary=True)]
              if "grad_n_episodes" in summary else []),
            Metric("grad_n_failed", "Failed evaluations", good="low", fmt=0),
            *([Metric("observation_grad_norm_median", "Median observation gradient norm", good="none", fmt=4)]
              if "observation_grad_norm_median" in summary else []),
            *([Metric("critic_swap_next_gap", "Mean |V(next) − V(true)|", good="none", fmt=4, trend=True),
               Metric("critic_swap_unrelated_gap", "Mean |V(unrelated) − V(true)|", good="none", fmt=4, trend=True)]
              if "critic_swap_next_gap" in summary else []),
        ], panels=panels, see_also=["critic"], extra={"gradient_status": summary.get("grad_status")})


def run(adapter, dataset, cfg, output_dir, val_ep_indices=None, *, records=None):
    """Run alone or share the value probe's traces; produce a separate viewer entry."""
    from lerobot.probes.critic import run_episode_critic_traces
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    trace_root = root.parent / "critic" / "episode_traces"
    if records is None:
        trace = run_episode_critic_traces(adapter, dataset, val_ep_indices, cfg, str(trace_root))
        records = trace["records"] if trace else []
    raw, summary = run_critic_gradients(adapter, dataset, cfg, records, root)
    if bool(getattr(cfg.probe_parameters, "critic_subtask_swap", False)):
        for variant in ("next", "unrelated"):
            gaps = [abs(r["value"] - r["swaps"][variant]["value"]) for r in records
                    if r["value"] is not None and r.get("swaps", {}).get(variant)
                    and r["swaps"][variant]["value"] is not None]
            summary[f"critic_swap_{variant}_gap"] = float(np.mean(gaps)) if gaps else None
        for path in sorted(trace_root.glob("*/critic_subtask_swap.png")):
            target = root / "episode_swaps" / f"{path.parent.name}.png"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
    write_sensitivity_manifest(root, summary)
    return raw, summary

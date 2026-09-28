"""Text conditioning across a broad sample of recorded episodes.

V(current text), V(next-segment text), and V(control text) all use the same
current frame, task and metadata. Only the subtask clause changes. The next
text comes from the first frame after the current critic terminal, with release
folded into the preceding segment. Missing or identical swaps are skipped.

The next-end duration curve is a hypothetical sequential-completion reference,
not a supervised target for the swapped text. The control is a frequent training
text absent from the selected annotations; this does not guarantee that the
instruction is impossible in every scene. Larger gaps show text sensitivity,
not necessarily correct understanding.

Training episodes are sampled round-robin over source/task groups. Aggregate
numbers average within each episode and then equally across episodes, so long
episodes do not dominate. This exploratory sample does not estimate training
mixture prevalence or held-out generalization. Validation is shown separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lerobot.probes.manifest import Metric, Panel, write_index


def _mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def episode_metrics(rows):
    result = {"points": len(rows), "failed_values": sum(r["value"] is None for r in rows)}
    for variant in ("next", "unrelated"):
        attempted = [(r, r.get("swaps", {}).get(variant)) for r in rows]
        attempted = [(r, s) for r, s in attempted if s is not None]
        pairs = [(r["value"], s["value"]) for r, s in attempted if r["value"] is not None and s["value"] is not None]
        result.update({
            f"{variant}_pairs": len(pairs),
            f"{variant}_failed": sum(s["value"] is None for _, s in attempted),
            f"{variant}_gap": _mean([abs(b-a) for a, b in pairs]),
            f"{variant}_lower": _mean([float(b < a) for a, b in pairs]),
            f"{variant}_value": _mean([b for _, b in pairs]),
        })
    return result


def aggregate(episodes):
    """Equal weight per episode; missing pairs never become zero effects."""
    metrics = [ep["metrics"] for ep in episodes]
    result = {"episodes": len(episodes)}
    for key in ("points", "failed_values", "next_pairs", "unrelated_pairs", "next_failed", "unrelated_failed"):
        result[key] = sum(m[key] for m in metrics)
    for variant in ("next", "unrelated"):
        result[f"{variant}_episodes"] = sum(m[f"{variant}_pairs"] > 0 for m in metrics)
        for key in ("gap", "lower", "value"):
            result[f"{variant}_{key}"] = _mean([m[f"{variant}_{key}"] for m in metrics])
    return result


def load_episodes(root):
    episodes = []
    for path in sorted((root / "sources").glob("*/episode_traces/*/critic_values.json")):
        payload = json.loads(path.read_text())
        rows = payload["records"]
        if not rows:
            continue
        first = rows[0]
        source = path.parents[2].name
        if any(r["source"] != source or r["episode"] != first["episode"] or r["split"] != first["split"] for r in rows):
            raise ValueError(f"Mixed episode identities in {path}")
        episodes.append({
            "id": f"{source}/{first['episode']}", "source": source, "split": first["split"],
            "episode": first["episode"], "task": first["task"],
            "records": rows, "boundaries": payload["segment_end_seconds"],
            "metrics": episode_metrics(rows), "file": str(path.relative_to(root)),
            "plot": str((path.parent / "critic_subtask_swap.png").relative_to(root)),
        })
    return episodes


def render(run_root):
    run_root = Path(run_root)
    provenance = json.loads((run_root / "provenance.json").read_text())
    root = run_root / f"step_{provenance['step']:08d}" / "critic_text_swap"
    episodes = load_episodes(root)
    if not episodes:
        raise ValueError(f"No episode traces in {root}")
    groups = defaultdict(list)
    for ep in episodes:
        groups[ep["source"]].append(ep)
    by_source = {key: aggregate(value) for key, value in sorted(groups.items())}
    by_split = {split: aggregate([ep for ep in episodes if ep["split"] == split]) for split in ("training", "validation")}
    summary = {
        "checkpoint_step": provenance["step"], "episodes": len(episodes),
        "points": sum(len(ep["records"]) for ep in episodes),
        "exact_texts": len({r["subtask"] for ep in episodes for r in ep["records"]}),
        "snapshot_frames": sum(bool(r.get("images")) for ep in episodes for r in ep["records"]),
        "by_source": by_source, "by_split": by_split,
        "failed_predictions": sum(ep["metrics"][k] for ep in episodes for k in ("failed_values", "next_failed", "unrelated_failed")),
    }
    summary["complete"] = (
        {(ep["source"], ep["episode"]) for ep in episodes if ep["split"] == "training"}
        == {(ep["source"], ep["episode"]) for ep in provenance["selected_episodes"]}
        and by_split["validation"]["episodes"] == provenance.get("validation_episodes", -1)
        and summary["failed_predictions"] == 0
    )
    data = {"provenance": provenance, "summary": summary, "episodes": episodes}
    (root / "survey_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    packed = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = Path(__file__).with_suffix(".html").read_text().replace("__PROBE_DATA__", packed)
    (root / "episode_explorer.html").write_text(html)

    sources = list(by_source)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(sources)*.55)), sharey=True)
    rng = np.random.default_rng(42)
    jitter = {s: rng.uniform(-.13, .13, len(groups[s])) for s in sources}
    for ax, variant, color in zip(axes, ("next", "unrelated"), ("darkorange", "purple"), strict=True):
        for i, source in enumerate(sources):
            xs = [ep["metrics"][f"{variant}_gap"] for ep in groups[source]]
            ax.scatter(xs, i+jitter[source], s=15, color=color, alpha=.55)
            mean = by_source[source][f"{variant}_gap"]
            if mean is not None:
                ax.plot(mean, i, "D", color="black", markersize=6)
        ax.set_xlabel("Mean |V(swapped text) − V(current text)|")
        ax.set_title("Next-segment text" if variant == "next" else "Fixed off-label control")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(range(len(sources)), [f"{s} (n={len(groups[s])})" for s in sources])
    axes[0].invert_yaxis()
    fig.suptitle(f"Checkpoint {provenance['step']}: dots = episodes with valid pairs; diamonds = equal-episode means")
    fig.tight_layout()
    fig.savefig(root / "swap_by_source.png", dpi=160)
    plt.close(fig)
    write_index(str(root), sys.modules[__name__], title="Critic Text Swap Survey", group="Critic",
        claim="Does changing only the subtask change the value across many tasks and scenes?",
        summary=summary, status="info" if summary["complete"] else "warn",
        metrics=[
            Metric("episodes", "Episodes", fmt=0, primary=True),
            Metric("points", "Current-frame anchors", fmt=0, primary=True),
            Metric("exact_texts", "Distinct current texts", fmt=0),
            Metric("failed_predictions", "Failed predictions", fmt=0, good="low"),
            Metric("by_split.training.next_gap", "Training: next-text gap", fmt=3),
            Metric("by_split.training.unrelated_gap", "Training: control-text gap", fmt=3),
            Metric("by_split.validation.next_gap", "Validation: next-text gap", fmt=3),
            Metric("by_split.validation.unrelated_gap", "Validation: control-text gap", fmt=3),
        ], panels=[
            Panel("episode_explorer.html", "Browse episodes, text-swap curves and frames", primary=True,
                  how="Filter by source or task. Hover over a curve to read its exact texts; click a teal snapshot marker to see that frame. Every curve uses the same current image/state/metadata. Only the subtask changes."),
            Panel("swap_by_source.png", "Text sensitivity across sources", primary=False,
                  how="One dot per episode with valid pairs; source labels count all sampled episodes. Black diamond = equal-episode mean. Larger gaps indicate sensitivity, not correctness. Validation is a separate cohort. The sample balances source/task coverage, not training prevalence."),
        ], extra={"provenance": {"checkpoint": provenance["checkpoint"], "sampling": provenance["sampling"]}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render saved critic text-swap traces without loading the model")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run_root), indent=2))

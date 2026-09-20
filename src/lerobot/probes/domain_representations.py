#!/usr/bin/env python
"""Domain representations: how the policy represents ReBot frames against the diverse
corpus, layer by layer and token group by token group.

**Question.** Are the two halves of the mixture represented in separate regions of the
hidden state, or do they share one? Where they share, in which token group and at what
depth does the sharing appear?

**Method.** ReBot frames are sampled evenly per episode from the held-out ReBot set;
the diverse side draws the same total from the corpus cache, split evenly over its
sources with the anchors spread over an episode. Every frame goes through the deployed
action prompt (subtask clause from the annotation, deployment metadata clause on both
halves) and one forward at the first inference step: flow time $t = 0$, so the action
tokens are the SAME fixed noise for every frame and the expert's state is context and
nothing of the frame's own action. Every encoder block and every action-expert block is
hooked and its output mean-pooled per token group. Six groups are read out:

* ``img_external_0`` / ``img_wrist_0`` — that camera's image patches: the scene;
* ``state`` — the discrete proprio clause;
* ``question`` — the fixed "Given these, what action ..." sentence, byte-identical for
  every frame, so any domain signal in it arrived through attention: pure context leakage;
* ``action_output`` — the position the FAST head reads from;
* ``action`` — the action-expert tokens, what the flow head decodes from.

Every other clause is captured into the cache but not read out: task and subtask text
differ by domain by construction, the metadata clause is identical on both halves, the
template is markup, and embodiment / external_1 / depth exist on one side only (the
ReBot half carries no embodiment clause in training either).

Whole-sequence pooling is deliberately absent: the halves differ in cameras, prompt
wording, state width and depth presence, so one pooled vector per frame separates them
by construction and says nothing about what the policy learned.

**Readouts**, every one computed on the full-width hidden state after centering across
the frames of that (site, layer, group) and unit-normalising, so similarity is cosine:

* *domain purity@k*: the fraction of a frame's k most cosine-similar frames that are in
  its own domain, averaged over frames; a label shuffle gives its chance level;
* *linear probe*: 5-fold accuracy of a ridge (least-squares) linear classifier at reading
  the domain, with a shuffled-label control;
* *source purity@k*: the same neighbourhood readout against the source label (ReBot
  plus each diverse source), so a group can be checked for separating the diverse robots
  from one another and not only from ReBot.

Purity near 1 and probe accuracy near 1 at every depth means the two halves never meet
in that group. Purity falling toward chance with depth while the probe stays high means
the halves interleave locally but a direction still tells them apart. Both at chance
means the group is shared.

**Pictures.** One interactive page: 3-D UMAP (cosine, on the top-50 PCA directions) of
any of the six groups at the first, middle or last layer, coloured by source, chosen from a
dropdown, with that cell's numbers in the title. Illustration of the readouts, not the
measurement.

**Caveats.** The embodiment clause is rendered for diverse samples only (the ReBot half
carries no embodiment_index), so ``embodiment`` exists on one side and gets no domain
readout. ``img_external_1`` is never present on ReBot and is skipped for the same
reason. ``depth`` compares ReBot against FMB only. Cosine after centering removes the
shared mean direction that dominates transformer hidden states; without it every
readout saturates.

Output (under ``<output_dir>/domain_representations/``): ``by_layer.png`` (purity and
linear probe against depth), ``representations_3d.html`` (one page, dropdown over group x
layer), ``summary.json`` / ``metrics.csv`` (the numbers), ``activations_cache.pt`` (every
group and layer, for re-plots with ``--probe_parameters.mode=plot``).

Standalone (checkpoint, datasets and prompt path from the config):

    uv run python -m lerobot.probes.domain_representations --config config_rl.yaml \\
        --probe_parameters.enable_domain_representations=true
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    DEPLOYMENT_METADATA,
    EP_COLORS,
    load_probe_dataset,
    makedirs,
    plotly_3d_layout,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
    sample_episodes_evenly,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeDomainRepresentationsConfig(TrainRLServerPipelineConfig):
    """Tunables under ``cfg.probe_parameters`` (ProbeConfig)."""


REBOT = "rebot"
DIVERSE = "diverse"
SITES = ("encoder", "action_expert")
# The groups that carry an answer. Task and subtask text differ by domain by
# construction, the metadata clause is identical on both halves, the template is markup,
# and embodiment / external_1 / depth exist on one side only; all are captured into the
# cache but not read out.
GROUP_ORDER = ("img_external_0", "img_wrist_0", "state", "question", "action_output", "action")


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

def _rebot_samples(dataset, cfg) -> list[dict]:
    p = cfg.probe_parameters
    n_per_episode = p.domain_repr_n_frames_per_episode or p.n_frames_per_episode
    samples = sample_episodes_evenly(
        dataset, n_per_episode=n_per_episode, max_episodes=p.max_episodes,
        seed=p.random_seed, stride=probe_image_stride(cfg),
    )
    return [
        {"domain": REBOT, "source": REBOT, "embodiment": "Rebot B601",
         "episode": f"rebot/{ep}", "frame": int(fr), "index": int(gidx)}
        for ep, fr, gidx in samples
    ]


def _open_diverse(cfg):
    """The diverse buffer on this run's cache, exactly as training opens it."""
    from lerobot.datasets.diverse_actor_selection import open_federated_corpus, select_actor_anchors
    from lerobot.rl.data_sources.diverse_actor_buffer import DiverseActorBuffer
    from lerobot.rl.data_sources.diverse_actor_cache import resolve_cache
    from lerobot.rl.data_sources.diverse_integration import sample_spec_from_config

    diverse_cfg = cfg.diverse
    spec = sample_spec_from_config(cfg)
    selection = select_actor_anchors(open_federated_corpus(diverse_cfg.root))
    cache = resolve_cache(
        diverse_cfg.root,
        diverse_cfg.cache_dir or getattr(cfg, "buffer_cache_dir", None),
        spec, selection, cache_policy="require",
    )
    return DiverseActorBuffer(
        selection, spec, cache=cache,
        render_automatic_quality=diverse_cfg.render_automatic_quality,
    )


def _diverse_samples(buffer, n_total: int, max_episodes: int | None, seed: int) -> list[dict]:
    """``n_total`` anchors split evenly over the sources; within a source, random
    episodes and evenly spaced anchors inside each."""
    rng = np.random.RandomState(seed)
    built = set(int(i) for i in buffer.cache.built_row_indices) if buffer.cache is not None else None
    by_source: dict[str, dict[str, list[int]]] = {}
    for row_index, row in enumerate(buffer.rows):
        if built is not None and row_index not in built:
            continue
        by_source.setdefault(row["source"], {}).setdefault(row["episode_id"], []).append(row_index)

    sources = sorted(by_source)
    per_source = max(1, n_total // len(sources))
    samples = []
    for source in sources:
        episodes = sorted(by_source[source])
        n_eps = len(episodes) if max_episodes is None else min(max_episodes, len(episodes))
        chosen = sorted(rng.choice(episodes, size=n_eps, replace=False).tolist())
        per_episode = max(1, int(np.ceil(per_source / n_eps)))
        for episode_id in chosen:
            rows = sorted(by_source[source][episode_id], key=lambda i: float(buffer.rows[i]["anchor_s"]))
            picks = np.unique(np.linspace(0, len(rows) - 1, min(per_episode, len(rows)), dtype=int))
            for pos in picks:
                row_index = rows[int(pos)]
                row = buffer.rows[row_index]
                samples.append({
                    "domain": DIVERSE, "source": source, "embodiment": str(row["embodiment"]),
                    "episode": f"{source}/{episode_id}", "frame": float(row["anchor_s"]),
                    "index": int(row_index),
                })
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Frame inputs
# ──────────────────────────────────────────────────────────────────────────────

def _rebot_inputs(dataset, cfg, sample: dict, chunk_size: int) -> dict:
    frame = probe_frame_inputs(dataset, cfg, sample["index"], chunk_size)
    return {"obs": frame["obs"], "task": frame["task"], "subtask": frame["subtask"],
            "metadata": dict(DEPLOYMENT_METADATA), "extra": None}


def _diverse_inputs(buffer, cfg, sample: dict) -> dict:
    """One corpus anchor through the training collate, then the trainer's own lift of
    depth into the observation and its forwarded identity columns."""
    from lerobot.datasets.diverse_prompt import episode_task
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer, _forwarded_complementary_keys

    transition = buffer.collate([sample["index"]])
    memory_cfg = getattr(cfg.policy, "memory", None)
    keep_history = memory_cfg is not None and bool(memory_cfg.history_keys)
    obs = {k: v for k, v in transition["state"].items() if keep_history or not str(k).startswith("history.")}
    comp = transition["complementary_info"]
    obs = MolmoAct2Trainer._inject_depth_observations(obs, comp, cfg)
    extra = {key: comp[key] for key in _forwarded_complementary_keys(comp, cfg)}
    # The clauses come as strings below, the way the ReBot side passes them.
    for key in ("subtask_index", "metadata_quality", "metadata_mistake", "metadata_speed"):
        extra.pop(key, None)
    row = buffer.rows[sample["index"]]
    record = buffer.selection.episode_records[str(row["episode_id"])]
    return {"obs": obs, "task": episode_task(record, row["source"])[0], "subtask": str(row["subtask"]),
            "metadata": dict(DEPLOYMENT_METADATA), "extra": extra,
            "gt_actions": transition["action"], "native_width": int(row["native_action_dim"])}


# ──────────────────────────────────────────────────────────────────────────────
# Collection
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect(adapter, rebot_dataset, diverse_buffer, samples: list[dict], cfg) -> dict:
    chunk_size = adapter.chunk_size
    noise_seed = int(cfg.probe_parameters.random_seed)

    per_site: dict[str, dict[str, list[torch.Tensor | None]]] = {site: {} for site in SITES}
    meta: list[dict] = []
    for i, sample in enumerate(samples):
        if i % 50 == 0:
            logging.info(f"  [{i + 1}/{len(samples)}] {sample['episode']} @ {sample['frame']}")
        inputs = (
            _rebot_inputs(rebot_dataset, cfg, sample, chunk_size) if sample["domain"] == REBOT
            else _diverse_inputs(diverse_buffer, cfg, sample)
        )
        reps = adapter.capture_layer_representations(
            inputs["obs"], inputs["task"], subtask=inputs["subtask"], metadata=inputs["metadata"],
            extra_complementary=inputs["extra"], noise_seed=noise_seed,
        )
        for site in SITES:
            for group, vec in reps[site].items():
                per_site[site].setdefault(group, [None] * i).append(vec)
            for group, column in per_site[site].items():
                if len(column) < i + 1:
                    column.append(None)
        meta.append({**sample, "task": inputs["task"], "subtask": inputs["subtask"],
                     "n_tokens": dict(reps["n_tokens"])})

    sites: dict[str, dict[str, dict]] = {}
    for site, groups in per_site.items():
        sites[site] = {}
        for group, column in groups.items():
            present = torch.tensor([v is not None for v in column])
            if not bool(present.any()):
                continue  # a clause no frame carried (history with memory off)
            shape = next(v.shape for v in column if v is not None)
            x = torch.zeros((len(column), *shape), dtype=torch.float16)
            for n, v in enumerate(column):
                if v is not None:
                    x[n] = v
            sites[site][group] = {"x": x, "present": present}
    return {"meta": meta, "noise_seed": noise_seed, "sites": sites}


# ──────────────────────────────────────────────────────────────────────────────
# Readouts
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_ready(x: torch.Tensor) -> np.ndarray:
    z = x.float().numpy().astype(np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)


def purity_at_k(z: np.ndarray, labels: np.ndarray, k: int) -> float:
    sims = z @ z.T
    np.fill_diagonal(sims, -np.inf)
    k = min(k, len(z) - 1)
    neighbours = np.argpartition(-sims, k, axis=1)[:, :k]
    return float((labels[neighbours] == labels[:, None]).mean())


def shuffled_purity(z: np.ndarray, labels: np.ndarray, k: int, rng, n: int = 5) -> float:
    return float(np.mean([purity_at_k(z, rng.permutation(labels), k) for _ in range(n)]))


def linear_probe(z: np.ndarray, labels: np.ndarray, seed: int) -> tuple[float, float]:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from threadpoolctl import threadpool_limits

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)
    with threadpool_limits(limits=4, user_api="blas"):
        real = cross_val_score(RidgeClassifier(alpha=1.0), z, labels, cv=folds).mean()
        control = np.mean([
            cross_val_score(RidgeClassifier(alpha=1.0), z, rng.permutation(labels), cv=folds).mean()
            for _ in range(2)
        ])
    return float(real), float(control)


def compute_metrics(cache: dict, cfg) -> list[dict]:
    p = cfg.probe_parameters
    k = int(p.domain_repr_knn_k)
    seed = int(p.random_seed)
    rng = np.random.RandomState(seed)
    meta = cache["meta"]
    domain = np.array([m["domain"] for m in meta])
    source = np.array([m["source"] for m in meta])
    rows = []
    for site, groups in cache["sites"].items():
        for group, entry in groups.items():
            if group not in GROUP_ORDER:
                continue
            present = entry["present"].numpy()
            n_rebot = int((present & (domain == REBOT)).sum())
            n_diverse = int((present & (domain == DIVERSE)).sum())
            if n_rebot <= k or n_diverse <= k:
                logging.info(f"  skip {site}/{group}: rebot={n_rebot} diverse={n_diverse} frames")
                continue
            x = entry["x"][present]
            d, s = domain[present], source[present]
            for layer in range(x.shape[1]):
                z = _cosine_ready(x[:, layer])
                acc, acc_shuffled = linear_probe(z, d, seed)
                rows.append({
                    "site": site, "group": group, "layer": layer,
                    "n_rebot": n_rebot, "n_diverse": n_diverse,
                    "purity": purity_at_k(z, d, k), "purity_shuffled": shuffled_purity(z, d, k, rng),
                    "linear_acc": acc, "linear_acc_shuffled": acc_shuffled,
                    "source_purity": purity_at_k(z, s, k),
                    "source_purity_shuffled": shuffled_purity(z, s, k, rng),
                })
            logging.info(f"  {site}/{group}: last-layer purity {rows[-1]['purity']:.3f} "
                         f"(chance {rows[-1]['purity_shuffled']:.3f}), linear {rows[-1]['linear_acc']:.3f}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def _ordered_groups(names) -> list[str]:
    names = set(names)
    return [g for g in GROUP_ORDER if g in names]


def _group_color(group: str) -> str:
    return EP_COLORS[GROUP_ORDER.index(group) % len(EP_COLORS)]


def plot_by_layer(rows: list[dict], path: str, k: int) -> None:
    """One figure: purity@k (left) and the linear probe (right) against depth.
    Encoder groups are solid, the action-expert group dashed; the expert's layer i
    reads the encoder's layer i, so the two share the x axis."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in zip(axes, ("purity", "linear_acc"),
                              (f"domain purity@{k} (cosine, centred)", "linear probe: rebot vs diverse (ridge, 5-fold)")):
        for group in _ordered_groups(r["group"] for r in rows):
            series = sorted((r["layer"], r[key]) for r in rows if r["group"] == group)
            ax.plot([l for l, _ in series], [v for _, v in series], marker="o", markersize=3, linewidth=1.5,
                    linestyle="--" if group == "action" else "-", label=group, color=_group_color(group))
        chance = float(np.mean([r[f"{key}_shuffled"] for r in rows]))
        ax.axhline(chance, color="#888", linestyle=":", linewidth=1.2, label=f"shuffled labels ({chance:.2f})")
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("layer (encoder block; the expert's block i reads it)")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _umap3(z: np.ndarray, cfg) -> np.ndarray:
    import umap as umap_lib
    from sklearn.decomposition import PCA

    p = cfg.probe_parameters
    reduced = PCA(n_components=min(50, z.shape[0] - 1, z.shape[1]), random_state=0).fit_transform(z)
    reducer = umap_lib.UMAP(n_components=3, n_neighbors=p.umap_n_neighbors, min_dist=p.umap_min_dist,
                            metric="cosine", random_state=p.umap_seed)
    return reducer.fit_transform(reduced)


def _palette(labels: np.ndarray) -> dict[str, str]:
    names = [REBOT] + sorted(set(labels) - {REBOT})
    return {name: ("#111111" if name == REBOT else EP_COLORS[i % len(EP_COLORS)]) for i, name in enumerate(names)}


def write_3d(cache: dict, rows: list[dict], layers: list[int], cfg, path: str) -> None:
    """ONE interactive page: a dropdown over (group, layer) shows that embedding, coloured
    by source (rebot black), with that cell's purity and linear accuracy in the title."""
    import plotly.graph_objects as go

    meta = cache["meta"]
    source = np.array([m["source"] for m in meta])
    by_cell = {(r["site"], r["group"], r["layer"]): r for r in rows}
    cells = [(site, group, layer)
             for site in SITES
             for group in _ordered_groups(g for s, g, _ in by_cell if s == site)
             for layer in layers]
    traces, buttons = [], []
    for cell_idx, (site, group, layer) in enumerate(cells):
        entry = cache["sites"][site][group]
        present = entry["present"].numpy()
        kept = [m for m, keep in zip(meta, present) if keep]
        emb = _umap3(_cosine_ready(entry["x"][present][:, layer]), cfg)
        labels = source[present]
        palette = _palette(labels)
        hover = [f"{m['episode']} @ {m['frame']}<br>{m['embodiment']}<br>{m['subtask']}" for m in kept]
        first = len(traces)
        for name, colour in palette.items():
            idx = np.nonzero(labels == name)[0]
            traces.append(go.Scatter3d(
                x=emb[idx, 0], y=emb[idx, 1], z=emb[idx, 2], mode="markers", name=f"{name} (n={len(idx)})",
                visible=cell_idx == 0,
                marker=dict(size=3.5, color=colour, opacity=0.8, line=dict(width=0)),
                text=[hover[i] for i in idx], hovertemplate="%{text}<extra></extra>",
            ))
        r = by_cell.get((site, group, layer), {})
        title = (f"{site} · {group} · layer {layer}  —  purity@{cfg.probe_parameters.domain_repr_knn_k} "
                 f"{r.get('purity', float('nan')):.2f} (chance {r.get('purity_shuffled', float('nan')):.2f}), "
                 f"linear {r.get('linear_acc', float('nan')):.2f}")
        buttons.append(dict(
            label=f"{group} · L{layer}" + (" (expert)" if site == "action_expert" else ""),
            method="update",
            args=[{"visible": [first <= i < len(traces) for i in range(10_000)]}, {"title": title}],
        ))
    # Visibility masks must be as long as the final trace list.
    for button in buttons:
        button["args"][0]["visible"] = button["args"][0]["visible"][: len(traces)]
    fig = go.Figure(data=traces)
    layout = plotly_3d_layout(buttons[0]["args"][1]["title"])
    layout["updatemenus"] = [dict(buttons=buttons, direction="down", x=0.0, xanchor="left", y=1.12, yanchor="top",
                                  showactive=True)]
    fig.update_layout(**layout)
    fig.write_html(path, include_plotlyjs="cdn")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _picture_layers(n_layers: int) -> list[int]:
    return sorted({0, n_layers // 2, n_layers - 1})


def run_analysis(cache: dict, cfg, output_dir: str) -> dict:
    k = int(cfg.probe_parameters.domain_repr_knn_k)
    rows = compute_metrics(cache, cfg)
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_by_layer(rows, os.path.join(output_dir, "by_layer.png"), k)
    n_layers = max(r["layer"] for r in rows) + 1
    write_3d(cache, rows, _picture_layers(n_layers), cfg, os.path.join(output_dir, "representations_3d.html"))

    meta = cache["meta"]
    by_group: dict[str, dict] = {}
    for r in rows:
        g = by_group.setdefault(r["group"], {"site": r["site"], "n_rebot": r["n_rebot"], "n_diverse": r["n_diverse"]})
        if r["layer"] == 0:
            g["purity_first"] = r["purity"]
        if r["layer"] == n_layers - 1:
            g.update(purity_last=r["purity"], purity_chance=r["purity_shuffled"], linear_last=r["linear_acc"],
                     linear_chance=r["linear_acc_shuffled"], source_purity_last=r["source_purity"],
                     source_purity_chance=r["source_purity_shuffled"])
    summary = {
        "n_rebot": sum(m["domain"] == REBOT for m in meta),
        "n_diverse": sum(m["domain"] == DIVERSE for m in meta),
        "diverse_sources": sorted({m["source"] for m in meta if m["domain"] == DIVERSE}),
        "k": k,
        "n_layers": n_layers,
        "groups": {g: by_group[g] for g in _ordered_groups(by_group)},
    }
    for group in ("action", "question", "img_wrist_0"):
        g = by_group.get(group)
        if g is not None:
            summary[f"{group}_purity_last"] = g["purity_last"]
            summary[f"{group}_purity_chance"] = g["purity_chance"]
            summary[f"{group}_linear_last"] = g["linear_last"]
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_manifest(output_dir, summary)
    return summary


def _write_manifest(output_dir: str, summary: dict) -> dict:
    table = "  ".join(
        f"{g}: {v['purity_first']:.2f}→{v['purity_last']:.2f} (lin {v['linear_last']:.2f})"
        for g, v in summary["groups"].items()
    )
    panels = [
        Panel("by_layer.png", "Domain purity@k (left) and linear decodability (right) against depth, six token groups",
              how="1.0 = a frame's cosine neighbours are all its own domain; dotted = label-shuffled chance. "
                  "Purity at 1 everywhere = the halves never meet in that group. Purity falling to chance with the "
                  "linear probe still high = interleaved but a direction separates them. Both at chance = shared. "
                  "The question clause is identical text in every prompt, so its curve is context leakage only.",
              primary=True),
        Panel("representations_3d.html", "3-D UMAP (cosine) of any group at the first, middle or last layer, coloured by source",
              how="Pick the cell in the dropdown; its purity and linear accuracy are in the title. Rebot is black. "
                  "Structure inside the diverse cloud (one blob per source vs one shared blob) is what the 2-D "
                  "curves cannot show. Illustration, not the measurement."),
    ]
    metrics = [
        Metric("action_purity_last", "Action tokens: domain purity, last expert layer", good="none", fmt=2,
               note=f"chance {summary.get('action_purity_chance', float('nan')):.2f}; what the flow head decodes from", primary=True),
        Metric("action_linear_last", "Action tokens: linear probe, last expert layer", good="none", fmt=2),
        Metric("question_purity_last", "Question clause: domain purity, last encoder layer", good="none", fmt=2,
               note=f"chance {summary.get('question_purity_chance', float('nan')):.2f}; identical tokens in every prompt"),
        Metric("img_wrist_0_purity_last", "Wrist patches: domain purity, last encoder layer", good="none", fmt=2),
        Metric("n_rebot", "ReBot frames", good="none", fmt=0),
        Metric("n_diverse", "Diverse frames", good="none", fmt=0),
    ]
    return write_index(
        output_dir, sys.modules[__name__], title="Domain representations", group="Representation",
        claim="Do ReBot and the diverse corpus share a representation, and in which token group and at what depth?",
        summary=summary, metrics=metrics, panels=panels, status="info",
        extra={"first_to_last_purity": table},
        see_also=["representations", "input_swap", "attention_budget"],
    )


def run(adapter, dataset, cfg, output_dir: str) -> dict | None:
    """``dataset`` is the held-out ReBot set; the diverse side comes from ``cfg.diverse``."""
    p = cfg.probe_parameters
    makedirs(output_dir)
    cache_path = os.path.join(output_dir, "activations_cache.pt")
    cache = None
    if p.mode in ("collect", "all"):
        rebot = _rebot_samples(dataset, cfg)
        logging.info(f"  ReBot: {len(rebot)} frames; opening the diverse corpus …")
        diverse_buffer = _open_diverse(cfg)
        diverse = _diverse_samples(diverse_buffer, len(rebot), p.max_episodes, int(p.random_seed))
        logging.info(f"  Diverse: {len(diverse)} anchors over {len({s['source'] for s in diverse})} sources")
        cache = collect(adapter, dataset, diverse_buffer, rebot + diverse, cfg)
        torch.save(cache, cache_path)
    if p.mode in ("plot", "all"):
        if cache is None:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        return run_analysis(cache, cfg, output_dir)
    return None


@parser.wrap()
def cli(cfg: ProbeDomainRepresentationsConfig):
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
    adapter = None
    if cfg.probe_parameters.mode in ("collect", "all"):
        adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    step = 0
    for part in str(getattr(cfg.policy, "pretrained_path", "") or "").split(os.sep):
        if part.isdigit():
            step = int(part)
    run(adapter, dataset, cfg, os.path.join(
        cfg.probe_parameters.output_dir, "validation", f"step_{step:08d}", "domain_representations"))


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

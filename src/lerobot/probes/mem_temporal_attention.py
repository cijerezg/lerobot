"""Distributional probe for the MEM video encoder's temporal attention.

The union softmax gives every current-frame patch all current-frame spatial keys plus
one same-position key per history age. A single mean cannot reveal whether the model
selectively reads particular moments, heads, cameras, or image regions. This probe
retains compact summaries from the same prefix forward and writes:

* ``temporal_attention.png`` — frame/layer distributions, age preference, head and
  camera specialization, and age selectivity;
* ``examples/frame_p*.png`` — low/median/high temporal-read examples with the most-read
  history frame, current image, and spatial temporal-read overlay per camera;
* ``temporal_attention_data.npz`` — the per-frame/layer/camera/head/age/patch arrays;
* ``temporal_attention.json`` — human-readable summary and per-frame layer values.

The plots include the union-softmax uniform-key baseline T/(N+T), which is the relevant
reference for interpreting absolute past-attention mass.
"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional

from lerobot.policies.molmoact2.modeling_molmoact2 import _MEM_TEMPORAL_CAPTURE
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    assemble_frame_history,
    get_frame_data,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)


def _temporal_layer_indices(policy) -> list[int]:
    resblocks = policy._backbone().vision_backbone.image_vit.transformer.resblocks
    stride = max(int(policy.config.temporal_layer_stride), 1)
    return [i for i in range(len(resblocks)) if (i + 1) % stride == 0]


def _as_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().float().cpu().squeeze()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = image.numpy()
    if array.max() <= 1.0:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    return array


def _age_entropy(head_age: np.ndarray) -> np.ndarray:
    """Mean normalized entropy per layer; 0=one age, 1=uniform across ages."""
    n_ages = head_age.shape[-1]
    if n_ages <= 1:
        return np.zeros(head_age.shape[1], dtype=np.float32)
    total = head_age.sum(axis=-1, keepdims=True)
    probs = np.divide(head_age, total, out=np.zeros_like(head_age), where=total > 1e-12)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1) / np.log(n_ages)
    entropy[total.squeeze(-1) <= 1e-12] = np.nan
    return np.nanmean(entropy, axis=(0, 2, 3))


def _write_manifest(output_dir: str, summary: dict, examples: list[tuple[int, str]]) -> dict:
    """Describe the temporal read to the manifest-driven viewer."""
    panels = [
        Panel(
            "temporal_attention.png",
            "Temporal-read distributions across frames, layers, heads, cameras, and history age",
            how="Every panel carries the union-softmax uniform-key line $T/(N+T)$: mass at that line is what a model reading nothing in particular would produce, so only the distance from it is evidence. The age panels say which moment is read, the head/camera panels whether the read is specialised or spread.",
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
        )
        for percentile, filename in examples
    ]
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="MEM Temporal Attention",
        group="History",
        claim="When and where does the video encoder read its history frames?",
        summary=summary,
        metrics=[
            Metric(
                "mean_enrichment_vs_uniform",
                "Temporal mass over uniform baseline",
                good="none",
                fmt=2,
                baseline=1.0,
                primary=True,
                note="Mean past-attention mass divided by the union-softmax uniform-key mass $T/(N+T)$. At 1.0 the past keys receive exactly what an indifferent allocation gives them, and no claim about selective reading survives.",
            ),
            Metric(
                "mean_temporal_mass",
                "Mean past-attention mass",
                good="none",
                fmt=4,
                primary=True,
                note="Absolute share of each patch's attention landing on history keys, averaged over frames and temporal layers.",
            ),
            Metric(
                "max_layer_enrichment_vs_uniform",
                "Strongest layer's enrichment",
                good="none",
                fmt=2,
                baseline=1.0,
                note="One layer reading history hard is a different regime from every layer reading it a little; the mean hides that.",
            ),
            Metric(
                "mean_normalized_age_entropy",
                "Age-selectivity entropy",
                good="none",
                fmt=3,
                baseline=1.0,
                note="Entropy of the mass distribution over history ages, normalized so 1.0 is a flat read over every age. Lower means the model prefers particular moments.",
            ),
            Metric("uniform_key_temporal_mass", "Uniform-key baseline", good="none", fmt=4),
            Metric("n_frames", "Frames captured", good="none", fmt=0),
        ],
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


def _render_distribution(
    temporal_mass: np.ndarray,
    head_age: np.ndarray,
    layers: list[int],
    camera_names: list[str],
    history_seconds: np.ndarray,
    frame_meta: list[dict],
    uniform_mass: float,
    output_path: str,
) -> None:
    """Six complementary views of whether temporal reads are selective and contextual."""
    _, _, _, n_heads, _ = head_age.shape
    layer_labels = [str(layer) for layer in layers]
    head_total = head_age.sum(axis=-1)
    age_mean = head_age.mean(axis=(0, 2, 3))
    age_share = age_mean / np.clip(age_mean.sum(axis=-1, keepdims=True), 1e-12, None)
    head_mean = head_total.mean(axis=(0, 2))
    camera_mean = head_total.mean(axis=(0, 3))
    entropy = _age_entropy(head_age)

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    axes[0, 0].boxplot([temporal_mass[:, i] for i in range(len(layers))], tick_labels=layer_labels)
    axes[0, 0].axhline(uniform_mass, color="#E76F51", linestyle="--", label="uniform-key baseline")
    axes[0, 0].set_xlabel("ViT temporal layer")
    axes[0, 0].set_ylabel("past attention mass")
    axes[0, 0].set_title("Distribution across sampled frames")
    axes[0, 0].legend(fontsize=8)

    vmax = max(float(np.percentile(temporal_mass, 98)), uniform_mass)
    image = axes[0, 1].imshow(temporal_mass, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
    axes[0, 1].set_xticks(range(len(layers)), layer_labels)
    stride = max(1, len(frame_meta) // 10)
    ticks = list(range(0, len(frame_meta), stride))
    axes[0, 1].set_yticks(ticks, [f"e{frame_meta[i]['episode_idx']}:f{frame_meta[i]['frame_idx']}" for i in ticks])
    for index in range(1, len(frame_meta)):
        if frame_meta[index]["episode_idx"] != frame_meta[index - 1]["episode_idx"]:
            axes[0, 1].axhline(index - 0.5, color="white", linewidth=0.7, alpha=0.8)
    axes[0, 1].set_title("Scene dependence: frame × layer")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046)

    image = axes[0, 2].imshow(age_share, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 2].set_xticks(range(len(history_seconds)), [f"-{value:g}s" for value in history_seconds])
    axes[0, 2].set_yticks(range(len(layers)), layer_labels)
    axes[0, 2].set_xlabel("history age (oldest → newest)")
    axes[0, 2].set_ylabel("ViT temporal layer")
    axes[0, 2].set_title("Where past attention goes, conditional on reading history")
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046, label="share of past mass")

    image = axes[1, 0].imshow(head_mean, aspect="auto", cmap="viridis", vmin=0.0)
    axes[1, 0].set_xticks(range(n_heads), [str(i) for i in range(n_heads)])
    axes[1, 0].set_yticks(range(len(layers)), layer_labels)
    axes[1, 0].set_xlabel("ViT head")
    axes[1, 0].set_ylabel("ViT temporal layer")
    axes[1, 0].set_title("Head specialization")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, label="past mass")

    image = axes[1, 1].imshow(camera_mean, aspect="auto", cmap="viridis", vmin=0.0)
    axes[1, 1].set_xticks(range(len(camera_names)), camera_names, rotation=20, ha="right")
    axes[1, 1].set_yticks(range(len(layers)), layer_labels)
    axes[1, 1].set_xlabel("camera")
    axes[1, 1].set_ylabel("ViT temporal layer")
    axes[1, 1].set_title("Camera specialization")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, label="past mass")

    axes[1, 2].bar(layer_labels, entropy, color="#457B9D")
    axes[1, 2].set_ylim(0.0, 1.05)
    axes[1, 2].set_xlabel("ViT temporal layer")
    axes[1, 2].set_ylabel("normalized age entropy")
    axes[1, 2].set_title("Age selectivity: 0=one moment, 1=uniform")

    fig.suptitle(
        f"MEM temporal-attention diagnostics — uniform-key past mass={uniform_mass:.4f}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def _render_spatial_example(
    dataset,
    memory_cfg,
    fps: float,
    diagnostic: dict,
    percentile: int,
    uniform_mass: float,
    vmax: float,
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

    fig, axes = plt.subplots(len(camera_keys), 3, figsize=(15, 4.2 * len(camera_keys)), squeeze=False)
    overlay_images = []
    for camera_idx, key in enumerate(camera_keys):
        history_tensor = history[f"history.{key}"].squeeze(0)
        age_index = int(np.argmax(age_by_camera[camera_idx]))
        axes[camera_idx, 0].imshow(_as_image(history_tensor[age_index]))
        axes[camera_idx, 0].set_title(
            f"{key.split('.')[-1]} most-read history (-{diagnostic['history_seconds'][age_index]:g}s)"
        )
        axes[camera_idx, 0].axis("off")

        current = _as_image(obs[key])
        axes[camera_idx, 1].imshow(current)
        axes[camera_idx, 1].set_title(f"{key.split('.')[-1]} current")
        axes[camera_idx, 1].axis("off")

        n_patches = patch_mass.shape[-1]
        side = round(n_patches**0.5)
        if side * side != n_patches:
            raise ValueError(f"Temporal patch overlay needs a square patch grid, got {n_patches} patches.")
        heat = torch.as_tensor(patch_mass[camera_idx]).view(1, 1, side, side)
        heat = functional.interpolate(
            heat, size=current.shape[:2], mode="bilinear", align_corners=False
        )[0, 0].numpy()
        axes[camera_idx, 2].imshow(current)
        overlay = axes[camera_idx, 2].imshow(heat, cmap="magma", alpha=0.58, vmin=0.0, vmax=vmax)
        overlay_images.append(overlay)
        axes[camera_idx, 2].set_title("current-frame patches reading history")
        axes[camera_idx, 2].axis("off")

    score = float(diagnostic["temporal_mass"].mean())
    fig.colorbar(overlay_images[-1], ax=axes[:, 2].tolist(), fraction=0.025, pad=0.02, label="past mass")
    fig.suptitle(
        f"Temporal-read p{percentile} — episode {ep_idx}, frame {frame_idx}  |  "
        f"mean mass={score:.4f} ({score / max(uniform_mass, 1e-12):.2f}× uniform baseline)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


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
    samples = sample_episodes_evenly(
        dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )

    temporal_mass_rows: list[np.ndarray] = []
    head_age_rows: list[np.ndarray] = []
    patch_mass_rows: list[np.ndarray] = []
    frame_meta: list[dict] = []
    camera_keys: list[str] | None = None
    history_seconds = np.asarray(
        [memory_cfg.history_window_seconds * (memory_cfg.history_num_samples - i) / memory_cfg.history_num_samples
         for i in range(memory_cfg.history_num_samples)],
        dtype=np.float32,
    )

    adapter._set_probe_cuda_graph_enabled(False)
    try:
        for ep_idx, frame_idx, global_idx in samples:
            # The temporal read lives in the ViT, so the prompt clauses do not reach
            # it; the frame builder is used for the image history and for a prompt
            # consistent with the other probes.
            frame = probe_frame_inputs(dataset, cfg, global_idx, int(cfg.policy.chunk_size))
            obs, task_str = frame["obs"], frame["task"]
            batch = adapter._make_batch(
                obs, task_str, subtask=frame["subtask"], metadata=frame["metadata"]
            )
            if "history_images" not in batch:
                continue

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
            if len(records) != len(layers):
                logging.warning(
                    f"[mem_temporal_attention] expected {len(layers)} temporal records, "
                    f"got {len(records)} — skipping."
                )
                continue

            means = torch.stack([record["mean"] for record in records]).float().cpu().numpy()
            head_age = torch.stack([record["by_bc_head_age"] for record in records]).float().cpu()
            patch_mass = torch.stack([record["by_bc_patch"] for record in records]).float().cpu()
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
                logging.warning("[mem_temporal_attention] could not align capture cameras to observation keys.")
                continue
            if camera_keys is None:
                camera_keys = current_camera_keys
            elif camera_keys != current_camera_keys:
                logging.warning("[mem_temporal_attention] camera order changed across frames — skipping frame.")
                continue

            temporal_mass_rows.append(means)
            head_age_rows.append(head_age.numpy())
            patch_mass_rows.append(patch_mass.numpy())
            frame_meta.append(
                {
                    "episode_idx": int(ep_idx),
                    "frame_idx": int(frame_idx),
                    "global_idx": int(global_idx),
                }
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
    n_patches = int(patch_mass.shape[-1])
    n_ages = int(head_age.shape[-1])
    uniform_mass = n_ages / (n_patches + n_ages)
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
    summary = {
        "layers": layers,
        "cameras": camera_names,
        "history_seconds_oldest_to_newest": history_seconds.tolist(),
        "uniform_key_temporal_mass": float(uniform_mass),
        "mean_temporal_mass": float(temporal_mass.mean()),
        "mean_enrichment_vs_uniform": float(temporal_mass.mean() / uniform_mass),
        "max_layer_enrichment_vs_uniform": float(temporal_mass.mean(axis=0).max() / uniform_mass),
        "mean_normalized_age_entropy": float(age_entropy.mean()),
        "n_frames": len(frame_meta),
        "n_heads": int(head_age.shape[3]),
        "n_patches": n_patches,
        "per_layer_distribution": layer_summary,
        "mean_age_mass_by_layer": age_mean.tolist(),
        "mean_age_share_by_layer": (
            age_mean / np.clip(age_mean.sum(axis=-1, keepdims=True), 1e-12, None)
        ).tolist(),
        "mean_normalized_age_entropy_by_layer": age_entropy.tolist(),
        "per_frame": [
            meta
            | {
                "temporal_mass_by_layer": temporal_mass[index].tolist(),
                "enrichment_vs_uniform_by_layer": (temporal_mass[index] / uniform_mass).tolist(),
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
        layers=np.asarray(layers),
        camera_names=np.asarray(camera_names),
        history_seconds=history_seconds,
        episode_idx=np.asarray([meta["episode_idx"] for meta in frame_meta]),
        frame_idx=np.asarray([meta["frame_idx"] for meta in frame_meta]),
        global_idx=np.asarray([meta["global_idx"] for meta in frame_meta]),
    )

    plot_path = os.path.join(output_dir, "temporal_attention.png")
    _render_distribution(
        temporal_mass,
        head_age,
        layers,
        camera_names,
        history_seconds,
        frame_meta,
        uniform_mass,
        plot_path,
    )

    examples_dir = os.path.join(output_dir, "examples")
    makedirs(examples_dir)
    scores = temporal_mass.mean(axis=1)
    patch_vmax = max(float(np.percentile(patch_mass, 98)), 1e-8)
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
        filename = (
            f"frame_p{percentile:02d}_ep{meta['episode_idx']:04d}_fr{meta['frame_idx']:06d}.png"
        )
        _render_spatial_example(
            dataset,
            memory_cfg,
            fps,
            diagnostic,
            percentile,
            uniform_mass,
            patch_vmax,
            os.path.join(examples_dir, filename),
        )
        examples.append((percentile, filename))

    _write_manifest(output_dir, summary, examples)

    mean_enrichment = temporal_mass.mean(axis=0) / uniform_mass
    logging.info(
        f"[mem_temporal_attention] n={len(frame_meta)}  uniform baseline={uniform_mass:.4f}  "
        f"per-layer mean enrichment={[round(float(value), 2) for value in mean_enrichment]}"
    )

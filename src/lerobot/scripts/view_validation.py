"""Dependency-free validation artifact viewer.

This is a stdlib-only alternative to view_validation.py. It serves a small
browser app that resolves validation artifacts on demand and streams videos
with HTTP range support, so large MP4s do not have to be loaded upfront.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


SPATIAL_ATTENTION_DIRS = ("spatial_memorization_attention", "spatial_memorization")
SPATIAL_ACTION_JACOBIAN_DIRS = (
    "spatial_memorization_action_jacobian",
    "spatial_memorization_jacobian",
)

ACTION_VIEWS = {
    "Overview": "actions/2d/overview.png",
    "Trajectories": ("actions/2d/trajectories.png", "actions/2d/val/trajectories.png"),
    "By frame": ("actions/2d/by_frame.png", "actions/2d/val/by_frame.png"),
    "By subtask": ("actions/2d/by_subtask.png", "actions/2d/val/by_subtask.png"),
    "NN distances": ("actions/2d/nn_distances.png", "actions/2d/val/nn_distances.png"),
}

ACTION_3D_VIEWS = {
    "3D Overview": "actions/3d/overview.html",
    "3D by episode": "actions/3d/by_episode.html",
    "3D by frame": "actions/3d/by_frame.html",
    "3D by subtask": "actions/3d/by_subtask.html",
}

REPR_COLORINGS = ["by_episode", "by_frame", "by_subtask"]
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class MediaItem:
    step: str
    label: str
    path: Path


def discover_steps(val_dir: Path) -> list[str]:
    return sorted(d.name for d in val_dir.iterdir() if d.is_dir() and d.name.startswith("step_"))


def step_label(name: str) -> str:
    return f"Step {int(name.split('_')[-1])}"


def _first_existing_rel(base: Path, rels: str | tuple[str, ...]) -> Path:
    aliases = (rels,) if isinstance(rels, str) else rels
    for rel in aliases:
        path = base / rel
        if path.exists():
            return path
    return base / aliases[0]


def discover_episodes(val_dir: Path, steps: list[str]) -> list[str]:
    for step in steps:
        ep_dir = _first_existing_rel(
            val_dir / step,
            ("actions/2d/episodes", "actions/2d/val/episodes"),
        )
        if ep_dir.exists():
            return sorted(p.stem for p in ep_dir.glob("ep*.png"))
    return []


def discover_attention_episodes_layers(val_dir: Path, steps: list[str]) -> tuple[list[str], list[str]]:
    eps: set[str] = set()
    layers: set[str] = set()
    for step in steps:
        att_dir = val_dir / step / "attention"
        if not att_dir.exists():
            continue
        for path in att_dir.iterdir():
            if path.is_dir() and "_L" in path.name:
                ep_part, layer_part = path.name.rsplit("_L", 1)
                eps.add(ep_part)
                layers.add(f"L{layer_part}")
    return sorted(eps), sorted(layers)


def discover_attention_files(
    val_dir: Path, steps: list[str], episodes: list[str], layers: list[str]
) -> list[str]:
    found: set[str] = set()
    for step in steps:
        for episode in episodes:
            for layer in layers:
                layer_dir = val_dir / step / "attention" / f"{episode}_{layer}"
                if not layer_dir.exists():
                    continue
                for path in layer_dir.glob("*.mp4"):
                    if path.name.startswith("overlay_"):
                        found.add(path.name.removeprefix("overlay_"))
                    elif path.name.startswith("heatmap_"):
                        found.add(path.name.removeprefix("heatmap_"))
                    else:
                        found.add(path.name)
        if found:
            break
    return sorted(found)


def discover_offline_inference_frames(val_dir: Path, steps: list[str]) -> list[str]:
    for step in steps:
        frame_dir = _first_existing_rel(
            val_dir / step / "offline_inference",
            ("unnormalized_eval", "unnormalized"),
        )
        if frame_dir.exists():
            return sorted(p.stem for p in frame_dir.glob("ep*.png"))
    return []


def discover_action_drift_jacobian_groups(val_dir: Path, steps: list[str]) -> list[str]:
    for step in steps:
        path = val_dir / step / "action_drift_jacobian"
        if path.exists():
            return sorted(p.name for p in path.iterdir() if p.is_dir())
    return []


def discover_action_drift_jacobian_layers(val_dir: Path, steps: list[str], groups: list[str]) -> list[str]:
    for step in steps:
        for group in groups:
            path = val_dir / step / "action_drift_jacobian" / group
            if path.exists():
                return sorted(p.name for p in path.iterdir() if p.is_dir() and p.name.startswith("L"))
    return []


def discover_action_drift_jacobian_files(
    val_dir: Path, steps: list[str], groups: list[str], layers: list[str]
) -> list[str]:
    for step in steps:
        for group in groups:
            for layer in layers:
                path = val_dir / step / "action_drift_jacobian" / group / layer
                if not path.exists():
                    continue
                idx = int(layer.replace("L", ""))
                prefix = f"causal_L{idx}_"
                files = [
                    p.name.removeprefix(prefix) if p.name.startswith(prefix) else p.name
                    for p in path.glob("*.mp4")
                ]
                return sorted(files)
    return []


def discover_repr_spaces(val_dir: Path, steps: list[str]) -> list[str]:
    for step in steps:
        path = val_dir / step / "representations" / "2d"
        if path.exists():
            return sorted(p.name for p in path.iterdir() if p.is_dir())
    return []


def discover_scree_files(val_dir: Path, steps: list[str]) -> list[str]:
    for step in steps:
        path = val_dir / step / "representations" / "pca_variance"
        if path.exists():
            return sorted(p.stem.removesuffix("_pca_scree") for p in path.glob("*_pca_scree.png"))
    return []


def discover_repr_3d_spaces(val_dir: Path, steps: list[str]) -> list[str]:
    for step in steps:
        path = val_dir / step / "representations" / "3d"
        if path.exists() and path.is_dir():
            return sorted(p.name for p in path.iterdir() if p.is_dir())
    return []


def discover_repr_3d_files(val_dir: Path, steps: list[str], spaces: list[str]) -> list[str]:
    files: set[str] = set()
    for step in steps:
        for space in spaces:
            path = val_dir / step / "representations" / "3d" / space
            if path.exists() and path.is_dir():
                files.update(p.stem for p in path.glob("*.html"))
        if files:
            break
    return sorted(files)


def discover_spatial_memorization_layers(
    val_dir: Path, steps: list[str], probe_dirs: str | tuple[str, ...]
) -> list[str]:
    layers: set[str] = set()
    aliases = (probe_dirs,) if isinstance(probe_dirs, str) else probe_dirs
    for step in steps:
        for probe_dir in aliases:
            path = val_dir / step / probe_dir
            if path.exists():
                layers.update(p.name for p in path.iterdir() if p.is_dir() and p.name.startswith("L"))
    return sorted(layers)


def discover_spatial_memorization_files(
    val_dir: Path, steps: list[str], probe_dirs: str | tuple[str, ...], layers: list[str]
) -> list[str]:
    files: set[str] = set()
    aliases = (probe_dirs,) if isinstance(probe_dirs, str) else probe_dirs
    for step in steps:
        for probe_dir in aliases:
            for layer in layers:
                path = val_dir / step / probe_dir / layer
                if path.exists():
                    files.update(p.name for p in path.glob("*.png"))
    return sorted(files)


def discover_critic_values_files(val_dir: Path, steps: list[str]) -> list[str]:
    seen: set[str] = set()
    for step in steps:
        path = val_dir / step / "critic"
        if path.exists():
            seen.update(p.name for p in path.glob("*.png"))
    if not seen:
        return []

    fixed_order = {
        "predicted_distributions.png": 0,
        "advantage_dist.png": 1,
        "advantage_squashed_dist.png": 2,
        "gradient_magnitudes.png": 3,
    }

    def sort_key(name: str) -> tuple[int, str]:
        if name in fixed_order:
            return (fixed_order[name], name)
        if name.startswith("frame_p") and name.endswith(".png"):
            try:
                pct = int(name.removeprefix("frame_p").removesuffix(".png"))
                return (4, f"{pct:03d}")
            except ValueError:
                pass
        return (5, name)

    return sorted(seen, key=sort_key)


def discover_critic_trace_episodes(val_dir: Path, steps: list[str]) -> list[str]:
    episodes: set[str] = set()
    for step in steps:
        path = val_dir / step / "critic" / "episode_traces"
        if path.exists():
            episodes.update(p.name for p in path.iterdir() if p.is_dir() and p.name.startswith("ep"))
    return sorted(episodes)


def _probe_dir_for_step(val_dir: Path, step: str, probe_dirs: str | tuple[str, ...]) -> Path:
    aliases = (probe_dirs,) if isinstance(probe_dirs, str) else probe_dirs
    for probe_dir in aliases:
        path = val_dir / step / probe_dir
        if path.exists():
            return path
    return val_dir / step / aliases[0]


def _attention_real_fname(layer: str, fname: str) -> str:
    if fname.startswith(("matrix_", "cross_matrix", "self_matrix", "action_text_matrix")):
        return fname
    return f"{'overlay_' if layer == 'L00' else 'heatmap_'}{fname}"


def _attention_path(layer_dir: Path, layer: str, fname: str) -> Path:
    candidates = [fname, f"overlay_{fname}", f"heatmap_{fname}", _attention_real_fname(layer, fname)]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = layer_dir / candidate
        if path.exists():
            return path
    return layer_dir / candidates[0]


def _resolve_validation_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    for candidate in (root / "validation", root / "probes", root):
        if candidate.exists() and discover_steps(candidate):
            return candidate
    raise FileNotFoundError(f"No validation/probes step_* directories found under {root}")


def _default_steps(steps: list[str]) -> list[str]:
    indices = sorted({0, len(steps) // 2, len(steps) - 1})
    return [steps[i] for i in indices]


def _ordered_union(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                values.append(item)
    return values


class ValidationViewer:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.val_dir = _resolve_validation_dir(self.run_dir)
        self.steps = discover_steps(self.val_dir)
        if not self.steps:
            raise FileNotFoundError(f"No step_* directories in {self.val_dir}")

        self.episodes = discover_episodes(self.val_dir, self.steps)
        self.att_episodes, self.att_layers = discover_attention_episodes_layers(self.val_dir, self.steps)
        self.att_files = discover_attention_files(self.val_dir, self.steps, self.att_episodes, self.att_layers)
        self.offline_inference_frames = discover_offline_inference_frames(self.val_dir, self.steps)

        self.adj_groups = discover_action_drift_jacobian_groups(self.val_dir, self.steps)
        self.adj_layers = discover_action_drift_jacobian_layers(self.val_dir, self.steps, self.adj_groups)
        self.adj_files = discover_action_drift_jacobian_files(
            self.val_dir, self.steps, self.adj_groups, self.adj_layers
        )

        self.repr_spaces = discover_repr_spaces(self.val_dir, self.steps)
        self.repr_3d_spaces = discover_repr_3d_spaces(self.val_dir, self.steps)
        self.repr_3d_files = discover_repr_3d_files(self.val_dir, self.steps, self.repr_3d_spaces)
        self.scree_spaces = discover_scree_files(self.val_dir, self.steps)
        self.sm_layers = discover_spatial_memorization_layers(self.val_dir, self.steps, SPATIAL_ATTENTION_DIRS)
        self.sm_files = discover_spatial_memorization_files(
            self.val_dir, self.steps, SPATIAL_ATTENTION_DIRS, self.sm_layers
        )
        self.smj_layers = discover_spatial_memorization_layers(
            self.val_dir, self.steps, SPATIAL_ACTION_JACOBIAN_DIRS
        )
        self.smj_files = discover_spatial_memorization_files(
            self.val_dir, self.steps, SPATIAL_ACTION_JACOBIAN_DIRS, self.smj_layers
        )
        self.cv_files = discover_critic_values_files(self.val_dir, self.steps)
        self.cv_trace_eps = discover_critic_trace_episodes(self.val_dir, self.steps)

    def manifest(self) -> dict:
        sections = [
            {
                "id": "actions",
                "title": "Actions",
                "media": "image",
                "controls": [
                    {
                        "name": "view",
                        "label": "View",
                        "choices": list(ACTION_VIEWS)
                        + list(ACTION_3D_VIEWS)
                        + [f"Episode: {ep}" for ep in self.episodes],
                    }
                ],
            }
        ]
        if self.att_episodes and self.att_layers and self.att_files:
            sections.append(
                {
                    "id": "attention",
                    "title": "Attention",
                    "media": "video",
                    "controls": [
                        {"name": "episode", "label": "Episode", "choices": self.att_episodes},
                        {"name": "layer", "label": "Layer", "choices": self.att_layers},
                        {"name": "file", "label": "View", "choices": self.att_files},
                    ],
                }
            )
        if self.repr_spaces or self.repr_3d_spaces or self.scree_spaces:
            repr_modes = []
            if self.repr_spaces:
                repr_modes.append("2D embedding")
            if self.repr_3d_spaces and self.repr_3d_files:
                repr_modes.append("3D embedding")
            if self.scree_spaces:
                repr_modes.append("PCA scree")
            sections.append(
                {
                    "id": "representations",
                    "title": "Representations",
                    "media": "image",
                    "controls": [
                        {"name": "mode", "label": "Mode", "choices": repr_modes},
                        {
                            "name": "space",
                            "label": "Space",
                            "choices": _ordered_union(self.repr_spaces, self.repr_3d_spaces, self.scree_spaces),
                        },
                        {
                            "name": "coloring",
                            "label": "Coloring",
                            "choices": REPR_COLORINGS,
                            "showWhen": {"mode": "2D embedding"},
                        },
                        {
                            "name": "html_view",
                            "label": "View",
                            "choices": self.repr_3d_files,
                            "showWhen": {"mode": "3D embedding"},
                        },
                    ],
                }
            )
        if self.offline_inference_frames:
            sections.append(
                {
                    "id": "offline_inference",
                    "title": "Offline Inference",
                    "media": "image",
                    "controls": [
                        {"name": "space", "label": "Space", "choices": ["Unnormalized", "Normalized"]},
                        {"name": "frame", "label": "Frame", "choices": self.offline_inference_frames},
                    ],
                }
            )
        if self.adj_groups and self.adj_layers and self.adj_files:
            sections.append(
                {
                    "id": "action_drift_jacobian",
                    "title": "Action Drift Jacobian",
                    "media": "video",
                    "controls": [
                        {"name": "group", "label": "Episode / timestep", "choices": self.adj_groups},
                        {"name": "layer", "label": "Layer", "choices": self.adj_layers},
                        {"name": "file", "label": "View", "choices": self.adj_files},
                    ],
                }
            )
        if self.sm_layers and self.sm_files:
            sections.append(
                {
                    "id": "spatial_memorization_attention",
                    "title": "Spatial Memorization Attention",
                    "media": "image",
                    "controls": [
                        {"name": "layer", "label": "Layer", "choices": self.sm_layers},
                        {"name": "file", "label": "View", "choices": self.sm_files},
                    ],
                }
            )
        if self.smj_layers and self.smj_files:
            sections.append(
                {
                    "id": "spatial_memorization_action_jacobian",
                    "title": "Spatial Memorization Action-Jacobian",
                    "media": "image",
                    "controls": [
                        {"name": "layer", "label": "Layer", "choices": self.smj_layers},
                        {"name": "file", "label": "View", "choices": self.smj_files},
                    ],
                }
            )
        if self.cv_files or self.cv_trace_eps:
            critic_modes = []
            if self.cv_files:
                critic_modes.append("Distributions")
            if self.cv_trace_eps:
                critic_modes.append("Trace curves")
            sections.append(
                {
                    "id": "critic",
                    "title": "Critic",
                    "media": "image",
                    "controls": [
                        {"name": "mode", "label": "Mode", "choices": critic_modes},
                        {
                            "name": "file",
                            "label": "View",
                            "choices": self.cv_files,
                            "showWhen": {"mode": "Distributions"},
                        },
                        {
                            "name": "episode",
                            "label": "Episode",
                            "choices": self.cv_trace_eps,
                            "showWhen": {"mode": "Trace curves"},
                        },
                    ],
                }
            )

        return {
            "runName": self.run_dir.name,
            "validationDir": str(self.val_dir),
            "steps": [{"id": step, "label": step_label(step)} for step in self.steps],
            "defaultSteps": _default_steps(self.steps),
            "sections": sections,
        }

    def resolve(self, section: str, params: dict[str, str], selected_steps: list[str]) -> dict:
        selected = set(selected_steps)
        steps = [step for step in self.steps if step in selected] or _default_steps(self.steps)
        items: list[MediaItem]
        media = "image"
        layout = "grid"

        if section == "actions":
            view = params.get("view", "Overview")
            if view.startswith("Episode: "):
                ep = view.removeprefix("Episode: ")
                items = [
                    MediaItem(
                        step,
                        step_label(step),
                        _first_existing_rel(
                            self.val_dir / step,
                            ("actions/2d/episodes", "actions/2d/val/episodes"),
                        )
                        / f"{ep}.png",
                    )
                    for step in steps
                ]
            else:
                if view in ACTION_3D_VIEWS:
                    media = "html"
                    rel = ACTION_3D_VIEWS[view]
                    items = [
                        MediaItem(step, step_label(step), self.val_dir / step / rel)
                        for step in steps
                    ]
                else:
                    rel = ACTION_VIEWS.get(view, ACTION_VIEWS["Overview"])
                    items = [
                        MediaItem(step, step_label(step), _first_existing_rel(self.val_dir / step, rel))
                        for step in steps
                    ]
        elif section == "attention":
            media = "video"
            episode = params.get("episode", self.att_episodes[0] if self.att_episodes else "")
            layer = params.get("layer", self.att_layers[0] if self.att_layers else "")
            fname = params.get("file", self.att_files[0] if self.att_files else "")
            items = [
                MediaItem(
                    step,
                    step_label(step),
                    _attention_path(self.val_dir / step / "attention" / f"{episode}_{layer}", layer, fname),
                )
                for step in steps
            ]
        elif section == "representations":
            mode = params.get("mode", "2D embedding")
            if mode == "3D embedding" and self.repr_3d_spaces and self.repr_3d_files:
                media = "html"
                space = params.get("space", self.repr_3d_spaces[0])
                if space not in self.repr_3d_spaces:
                    space = self.repr_3d_spaces[0]
                view = params.get("html_view", self.repr_3d_files[0])
                if view not in self.repr_3d_files:
                    view = self.repr_3d_files[0]
                items = [
                    MediaItem(
                        step,
                        step_label(step),
                        self.val_dir / step / "representations" / "3d" / space / f"{view}.html",
                    )
                    for step in steps
                ]
            elif mode == "PCA scree" and self.scree_spaces:
                space = params.get("space", self.scree_spaces[0])
                if space not in self.scree_spaces:
                    space = self.scree_spaces[0]
                items = [
                    MediaItem(
                        step,
                        step_label(step),
                        self.val_dir / step / "representations" / "pca_variance" / f"{space}_pca_scree.png",
                    )
                    for step in steps
                ]
            else:
                space = params.get("space", self.repr_spaces[0] if self.repr_spaces else "")
                if self.repr_spaces and space not in self.repr_spaces:
                    space = self.repr_spaces[0]
                coloring = params.get("coloring", REPR_COLORINGS[0])
                items = [
                    MediaItem(
                        step,
                        step_label(step),
                        self.val_dir / step / "representations" / "2d" / space / f"{coloring}.png",
                    )
                    for step in steps
                ]
        elif section == "pca_scree":
            space = params.get("space", self.scree_spaces[0] if self.scree_spaces else "")
            items = [
                MediaItem(
                    step,
                    step_label(step),
                    self.val_dir / step / "representations" / "pca_variance" / f"{space}_pca_scree.png",
                )
                for step in steps
            ]
        elif section == "offline_inference":
            space = params.get("space", "Unnormalized")
            frame = params.get(
                "frame", self.offline_inference_frames[0] if self.offline_inference_frames else ""
            )
            subdirs = ("unnormalized_eval", "unnormalized") if space == "Unnormalized" else (
                "normalized_eval",
                "normalized",
            )
            items = [
                MediaItem(
                    step,
                    step_label(step),
                    _first_existing_rel(self.val_dir / step / "offline_inference", subdirs) / f"{frame}.png",
                )
                for step in steps
            ]
        elif section == "action_drift_jacobian":
            media = "video"
            group = params.get("group", self.adj_groups[0] if self.adj_groups else "")
            layer = params.get("layer", self.adj_layers[0] if self.adj_layers else "")
            fname = params.get("file", self.adj_files[0] if self.adj_files else "")
            idx = int(layer.replace("L", "")) if layer.startswith("L") else 0
            real_fname = fname if fname.startswith("causal_") else f"causal_L{idx}_{fname}"
            items = [
                MediaItem(
                    step,
                    step_label(step),
                    self.val_dir / step / "action_drift_jacobian" / group / layer / real_fname,
                )
                for step in steps
            ]
        elif section == "spatial_memorization_attention":
            layer = params.get("layer", self.sm_layers[0] if self.sm_layers else "")
            fname = params.get("file", self.sm_files[0] if self.sm_files else "")
            items = [
                MediaItem(
                    step,
                    step_label(step),
                    _probe_dir_for_step(self.val_dir, step, SPATIAL_ATTENTION_DIRS) / layer / fname,
                )
                for step in steps
            ]
        elif section == "spatial_memorization_action_jacobian":
            layer = params.get("layer", self.smj_layers[0] if self.smj_layers else "")
            fname = params.get("file", self.smj_files[0] if self.smj_files else "")
            items = [
                MediaItem(
                    step,
                    step_label(step),
                    _probe_dir_for_step(self.val_dir, step, SPATIAL_ACTION_JACOBIAN_DIRS) / layer / fname,
                )
                for step in steps
            ]
        elif section == "critic":
            mode = params.get("mode", "Distributions" if self.cv_files else "Trace curves")
            if mode == "Trace curves" and self.cv_trace_eps:
                ep = params.get("episode", self.cv_trace_eps[0])
                if ep not in self.cv_trace_eps:
                    ep = self.cv_trace_eps[0]
                items = [
                    MediaItem(
                        step,
                        step_label(step),
                        self.val_dir / step / "critic" / "episode_traces" / ep / "critic_plot.png",
                    )
                    for step in steps
                ]
            else:
                fname = params.get("file", self.cv_files[0] if self.cv_files else "")
                if self.cv_files and fname not in self.cv_files:
                    fname = self.cv_files[0]
                items = [
                    MediaItem(step, step_label(step), self.val_dir / step / "critic" / fname)
                    for step in steps
                ]
        else:
            raise ValueError(f"Unknown section: {section}")

        return {
            "media": media,
            "layout": layout,
            "items": [self._serialize_item(item) for item in items],
        }

    def _serialize_item(self, item: MediaItem) -> dict:
        try:
            rel = item.path.resolve().relative_to(self.val_dir).as_posix()
        except ValueError:
            rel = ""
        exists = bool(rel) and item.path.exists()
        return {
            "step": item.step,
            "label": item.label,
            "exists": exists,
            "url": f"/asset/{quote(rel, safe='/')}" if exists else "",
            "size": item.path.stat().st_size if exists else 0,
        }

    def asset_path(self, rel: str) -> Path:
        rel = unquote(rel)
        if rel.startswith("/") or "\x00" in rel:
            raise FileNotFoundError(rel)
        path = (self.val_dir / rel).resolve()
        if self.val_dir not in path.parents and path != self.val_dir:
            raise FileNotFoundError(rel)
        if not path.is_file():
            raise FileNotFoundError(rel)
        return path


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _parse_range(range_header: str | None, size: int) -> tuple[int, int, bool]:
    if not range_header:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        return 0, size - 1, False
    start_s, end_s = match.groups()
    if start_s:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    else:
        suffix = int(end_s) if end_s else 0
        start = max(size - suffix, 0)
        end = size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable range")
    return start, min(end, size - 1), True


def _send_file(handler: BaseHTTPRequestHandler, path: Path, head_only: bool = False) -> None:
    size = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    try:
        start, end, partial = _parse_range(handler.headers.get("Range"), size)
    except ValueError:
        handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        handler.send_header("Content-Range", f"bytes */{size}")
        handler.end_headers()
        return

    length = end - start + 1
    handler.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    handler.send_header("Cache-Control", "public, max-age=3600")
    if partial:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    if head_only:
        return

    with path.open("rb") as file:
        file.seek(start)
        remaining = length
        while remaining:
            chunk = file.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def make_handler(viewer: ValidationViewer) -> type[BaseHTTPRequestHandler]:
    class ValidationRequestHandler(BaseHTTPRequestHandler):
        server_version = "ValidationViewer/1.0"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_HEAD(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/asset/"):
                self._handle_asset(path.removeprefix("/asset/"), head_only=True)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._handle_index()
            elif parsed.path == "/api/manifest":
                _json_response(self, viewer.manifest())
            elif parsed.path == "/api/view":
                self._handle_view(parse_qs(parsed.query))
            elif parsed.path.startswith("/asset/"):
                self._handle_asset(parsed.path.removeprefix("/asset/"), head_only=False)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _handle_index(self) -> None:
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _handle_view(self, query: dict[str, list[str]]) -> None:
            section = query.get("section", [""])[0]
            steps = query.get("steps", [""])[0].split(",") if query.get("steps") else []
            params = {key: values[0] for key, values in query.items() if key not in {"section", "steps"}}
            try:
                _json_response(self, viewer.resolve(section, params, steps))
            except ValueError as exc:
                _json_response(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def _handle_asset(self, rel: str, head_only: bool) -> None:
            try:
                _send_file(self, viewer.asset_path(rel), head_only=head_only)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
            except BrokenPipeError:
                return

    return ValidationRequestHandler


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Validation Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0d0f12;
      --panel: #151922;
      --panel-2: #1d2430;
      --line: #2a3342;
      --text: #edf2f7;
      --muted: #9aa8b8;
      --accent: #5dd6b7;
      --accent-2: #7da7ff;
      --danger: #ff7b72;
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
    }
    .appbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: grid;
      gap: 12px;
      padding: 14px 18px 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(13, 15, 18, 0.94);
      backdrop-filter: blur(14px);
    }
    .identity {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      min-width: 0;
    }
    h1, h2 { margin: 0; letter-spacing: 0; }
    h1 {
      min-width: 0;
      font-size: 18px;
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .subtle {
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .nav, .steps {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 2px;
      scrollbar-width: thin;
    }
    .nav button, .step button, .toolbar button {
      min-height: 34px;
      border: 1px solid var(--line);
      background: transparent;
      color: var(--text);
      border-radius: 7px;
      cursor: pointer;
      font: inherit;
      white-space: nowrap;
    }
    .nav button {
      padding: 7px 11px;
      color: var(--muted);
    }
    .nav button.active {
      color: var(--text);
      border-color: rgba(93, 214, 183, 0.72);
      background: rgba(93, 214, 183, 0.1);
    }
    .step button {
      padding: 6px 9px;
      font-size: 12px;
      color: var(--muted);
    }
    .step button.active {
      color: #06100d;
      background: var(--accent);
      border-color: var(--accent);
    }
    main {
      min-width: 0;
      padding: 18px;
    }
    .workbar {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    .title h2 {
      font-size: 26px;
      line-height: 1.1;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 10px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    select {
      min-width: 170px;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }
    .toolbar {
      display: none;
      align-items: center;
      gap: 10px;
      margin: 0 0 16px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .toolbar.visible { display: flex; }
    .toolbar button {
      padding: 6px 12px;
      background: var(--panel-2);
    }
    .toolbar input[type="range"] {
      min-width: 180px;
      flex: 1;
    }
    .toolbar select { min-width: 88px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(var(--cols, 3), minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }
    .grid.vertical {
      grid-template-columns: minmax(0, 980px);
      justify-content: center;
    }
    .card {
      min-width: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 9px 11px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .card-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      white-space: nowrap;
    }
    .size {
      color: var(--accent-2);
    }
    .open-link {
      color: var(--muted);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 3px 7px;
    }
    .open-link:hover {
      color: var(--text);
      border-color: var(--accent);
    }
    .media-wrap {
      display: grid;
      place-items: center;
      min-height: 180px;
      background: #090b0f;
    }
    img, video, iframe {
      display: block;
      width: 100%;
      border: 0;
    }
    img, video {
      height: auto;
      max-height: 78vh;
      object-fit: contain;
    }
    iframe {
      height: 72vh;
      background: white;
    }
    .missing {
      padding: 44px 16px;
      color: var(--danger);
      text-align: center;
      font-size: 13px;
    }
    .empty {
      padding: 48px 16px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
    }
    @media (max-width: 900px) {
      .identity, .workbar { display: grid; }
      .controls { justify-content: stretch; }
      label, select { width: 100%; }
      main { padding: 14px; }
      .grid { grid-template-columns: 1fr; }
      .toolbar { flex-wrap: wrap; }
    }
  </style>
</head>
<body>
  <header class="appbar">
    <div class="identity">
      <h1 id="runName">Validation</h1>
      <div id="validationDir" class="subtle"></div>
    </div>
    <nav id="sections" class="nav"></nav>
    <div id="steps" class="steps"></div>
  </header>
  <main>
    <div class="workbar">
      <div class="title">
        <h2 id="sectionTitle"></h2>
      </div>
      <div id="controls" class="controls"></div>
    </div>
    <div id="videoToolbar" class="toolbar">
      <button id="playPause" type="button">Play</button>
      <button id="resetVideos" type="button">Reset</button>
      <input id="seek" type="range" min="0" max="1000" value="0" />
      <select id="speed" aria-label="Playback speed">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
      </select>
    </div>
    <div id="content" class="grid"></div>
  </main>
  <script>
    const state = {
      manifest: null,
      activeSection: null,
      selectedSteps: new Set(),
      controlValues: {},
      loading: null,
      observer: null,
    };

    const $ = (id) => document.getElementById(id);

    function fmtBytes(bytes) {
      if (!bytes) return "";
      const units = ["B", "KB", "MB", "GB"];
      let value = bytes;
      let unit = 0;
      while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit++;
      }
      return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
    }

    function sectionById(id) {
      return state.manifest.sections.find((section) => section.id === id);
    }

    function controlKey(section, controlName) {
      return `${section.id}.${controlName}`;
    }

    function controlVisible(section, control) {
      if (control.showWhen) {
        return Object.entries(control.showWhen).every(([name, expected]) => {
          return state.controlValues[controlKey(section, name)] === expected;
        });
      }
      if (control.showWhenAny) {
        return Object.entries(control.showWhenAny).every(([name, expected]) => {
          return expected.includes(state.controlValues[controlKey(section, name)]);
        });
      }
      return true;
    }

    function selectedStepList() {
      return state.manifest.steps
        .map((step) => step.id)
        .filter((step) => state.selectedSteps.has(step));
    }

    function renderSteps() {
      const root = $("steps");
      root.innerHTML = "";
      state.manifest.steps.forEach((step) => {
        const wrap = document.createElement("div");
        wrap.className = "step";
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = step.label.replace("Step ", "");
        button.classList.toggle("active", state.selectedSteps.has(step.id));
        button.onclick = () => {
          if (state.selectedSteps.has(step.id)) {
            state.selectedSteps.delete(step.id);
          } else {
            state.selectedSteps.add(step.id);
          }
          if (state.selectedSteps.size === 0) {
            state.selectedSteps.add(step.id);
          }
          renderSteps();
          loadView();
        };
        wrap.appendChild(button);
        root.appendChild(wrap);
      });
    }

    function renderSections() {
      const nav = $("sections");
      nav.innerHTML = "";
      state.manifest.sections.forEach((section) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = section.title;
        button.classList.toggle("active", section.id === state.activeSection);
        button.onclick = () => setSection(section.id);
        nav.appendChild(button);
      });
    }

    function setSection(sectionId) {
      state.activeSection = sectionId;
      renderSections();
      renderControls();
      loadView();
    }

    function renderControls() {
      const section = sectionById(state.activeSection);
      $("sectionTitle").textContent = section.title;
      const root = $("controls");
      root.innerHTML = "";
      section.controls.forEach((control) => {
        const key = controlKey(section, control.name);
        if (!state.controlValues[key]) {
          state.controlValues[key] = control.choices[0] || "";
        }
      });
      section.controls.forEach((control) => {
        if (!controlVisible(section, control)) return;
        const key = controlKey(section, control.name);
        const label = document.createElement("label");
        label.textContent = control.label;
        const select = document.createElement("select");
        control.choices.forEach((choice) => {
          const option = document.createElement("option");
          option.value = choice;
          option.textContent = choice;
          select.appendChild(option);
        });
        select.value = state.controlValues[key];
        select.onchange = () => {
          state.controlValues[key] = select.value;
          renderControls();
          loadView();
        };
        label.appendChild(select);
        root.appendChild(label);
      });
    }

    function queryForView() {
      const section = sectionById(state.activeSection);
      const query = new URLSearchParams();
      query.set("section", section.id);
      query.set("steps", selectedStepList().join(","));
      section.controls.forEach((control) => {
        if (controlVisible(section, control)) {
          query.set(control.name, state.controlValues[controlKey(section, control.name)] || "");
        }
      });
      return query;
    }

    async function loadView() {
      if (state.loading) {
        state.loading.abort();
      }
      state.loading = new AbortController();
      const content = $("content");
      content.innerHTML = `<div class="empty">Loading</div>`;
      try {
        const response = await fetch(`/api/view?${queryForView()}`, { signal: state.loading.signal });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Failed to load view");
        renderView(data);
      } catch (err) {
        if (err.name !== "AbortError") {
          content.innerHTML = `<div class="empty">${err.message}</div>`;
        }
      }
    }

    function renderView(data) {
      const content = $("content");
      content.innerHTML = "";
      content.className = `grid ${data.layout === "vertical" ? "vertical" : ""}`;
      content.style.setProperty("--cols", String(Math.max(1, Math.min(3, data.items.length))));
      $("videoToolbar").classList.toggle("visible", data.media === "video");

      if (state.observer) {
        state.observer.disconnect();
      }
      state.observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const video = entry.target;
          if (!video.src && video.dataset.src) {
            video.src = video.dataset.src;
            video.load();
          }
          state.observer.unobserve(video);
        });
      }, { rootMargin: "240px" });

      if (!data.items.length) {
        content.innerHTML = `<div class="empty">No artifacts</div>`;
        return;
      }

      data.items.forEach((item) => {
        const card = document.createElement("article");
        card.className = "card";
        const header = document.createElement("div");
        header.className = "card-header";
        const label = document.createElement("span");
        label.textContent = item.label;
        const actions = document.createElement("div");
        actions.className = "card-actions";
        const size = document.createElement("span");
        size.className = "size";
        size.textContent = fmtBytes(item.size);
        actions.appendChild(size);
        if (item.exists) {
          const link = document.createElement("a");
          link.className = "open-link";
          link.href = item.url;
          link.target = "_blank";
          link.rel = "noreferrer";
          link.textContent = "Open";
          actions.appendChild(link);
        }
        header.append(label, actions);
        const wrap = document.createElement("div");
        wrap.className = "media-wrap";

        if (!item.exists) {
          wrap.innerHTML = `<div class="missing">Not found</div>`;
        } else if (data.media === "video") {
          const video = document.createElement("video");
          video.controls = true;
          video.preload = "metadata";
          video.dataset.src = item.url;
          video.addEventListener("play", syncPlaybackRate);
          video.addEventListener("error", () => {
            const msg = document.createElement("div");
            msg.className = "missing";
            msg.textContent = "Browser cannot decode this MP4. Use Open to download/view it externally.";
            wrap.appendChild(msg);
          }, { once: true });
          wrap.appendChild(video);
          state.observer.observe(video);
        } else if (data.media === "html") {
          const frame = document.createElement("iframe");
          frame.loading = "lazy";
          frame.src = item.url;
          frame.title = item.label;
          wrap.appendChild(frame);
        } else {
          const img = document.createElement("img");
          img.loading = "lazy";
          img.decoding = "async";
          img.src = item.url;
          img.alt = item.label;
          wrap.appendChild(img);
        }
        card.append(header, wrap);
        content.appendChild(card);
      });
    }

    function videos() {
      return Array.from(document.querySelectorAll("video"));
    }

    function mountVideos() {
      videos().forEach((video) => {
        if (!video.src && video.dataset.src) {
          video.src = video.dataset.src;
          video.load();
        }
      });
    }

    function syncPlaybackRate() {
      const rate = parseFloat($("speed").value);
      videos().forEach((video) => { video.playbackRate = rate; });
    }

    $("playPause").onclick = async () => {
      mountVideos();
      const vids = videos();
      const playing = vids.some((video) => !video.paused);
      if (playing) {
        vids.forEach((video) => video.pause());
        $("playPause").textContent = "Play";
      } else {
        syncPlaybackRate();
        for (const video of vids) {
          try { await video.play(); } catch (_) {}
        }
        $("playPause").textContent = "Pause";
      }
    };

    $("resetVideos").onclick = () => {
      mountVideos();
      videos().forEach((video) => {
        video.pause();
        video.currentTime = 0;
      });
      $("seek").value = 0;
      $("playPause").textContent = "Play";
    };

    $("seek").oninput = () => {
      mountVideos();
      const frac = parseFloat($("seek").value) / 1000;
      videos().forEach((video) => {
        if (video.duration) video.currentTime = frac * video.duration;
      });
    };

    $("speed").onchange = syncPlaybackRate;

    function tick() {
      const first = videos().find((video) => video.duration);
      if (first && !$("seek").matches(":active")) {
        $("seek").value = String((first.currentTime / first.duration) * 1000);
      }
      requestAnimationFrame(tick);
    }

    async function init() {
      const response = await fetch("/api/manifest");
      state.manifest = await response.json();
      $("runName").textContent = state.manifest.runName;
      $("validationDir").textContent = state.manifest.validationDir;
      state.manifest.defaultSteps.forEach((step) => state.selectedSteps.add(step));
      state.activeSection = state.manifest.sections[0].id;
      renderSteps();
      renderSections();
      renderControls();
      await loadView();
      tick();
    }

    init().catch((err) => {
      $("content").innerHTML = `<div class="empty">${err.message}</div>`;
    });
  </script>
</body>
</html>
"""


def _make_server(viewer: ValidationViewer, host: str, port: int, retries: int = 20) -> ThreadingHTTPServer:
    handler = make_handler(viewer)
    last_error: OSError | None = None
    for candidate in range(port, port + retries + 1):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            last_error = exc
            if port == 0:
                break
    raise RuntimeError(f"Could not bind validation viewer server: {last_error}") from last_error


def view_validation(
    run_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = True,
) -> None:
    """Launch the local validation viewer and block until interrupted."""
    viewer = ValidationViewer(run_dir)
    server = _make_server(viewer, host, port)
    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}/"
    print(f"Validation viewer: {url}")
    print(f"Serving artifacts from: {viewer.val_dir}")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping validation viewer.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="View validation outputs in a dependency-free browser UI.")
    parser.add_argument("run_dir_pos", nargs="?", help="Training run directory, or validation/probes directory.")
    parser.add_argument("--run_dir", help="Training run directory, or validation/probes directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab automatically.")
    args = parser.parse_args()

    run_dir = args.run_dir or args.run_dir_pos
    if not run_dir:
        parser.error("provide run_dir as a positional argument or with --run_dir")
    view_validation(run_dir, host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()

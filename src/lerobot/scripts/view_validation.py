"""Side-by-side comparison of validation outputs across training steps."""

import argparse
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_steps(val_dir: Path) -> list[str]:
    return sorted(
        d.name for d in val_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    )


def step_label(name: str) -> str:
    return f"Step {int(name.split('_')[-1])}"


def discover_episodes(val_dir: Path, steps: list[str]) -> list[str]:
    ep_dir = val_dir / steps[0] / "actions" / "2d" / "val" / "episodes"
    if not ep_dir.exists():
        return []
    return sorted(p.stem for p in ep_dir.glob("ep*.png"))


def discover_attention_episodes_layers(val_dir: Path, steps: list[str]) -> tuple[list[str], list[str]]:
    """Scan attention/ep{XXXX}_L{YY} dirs across steps; return (episodes, layers)."""
    eps: set[str] = set()
    layers: set[str] = set()
    for s in steps:
        att_dir = val_dir / s / "attention"
        if not att_dir.exists():
            continue
        for d in att_dir.iterdir():
            if not d.is_dir():
                continue
            if "_L" not in d.name:
                continue
            ep_part, l_part = d.name.rsplit("_L", 1)
            eps.add(ep_part)
            layers.add(f"L{l_part}")
    return sorted(eps), sorted(layers)


def discover_attention_files(val_dir: Path, steps: list[str], episodes: list[str], layers: list[str]) -> list[str]:
    """Return normalized MP4 names (overlay_/heatmap_ prefix stripped)."""
    found: set[str] = set()
    for s in steps:
        for ep in episodes:
            for l in layers:
                d = val_dir / s / "attention" / f"{ep}_{l}"
                if not d.exists() or not d.is_dir():
                    continue
                for p in d.glob("*.mp4"):
                    n = p.name
                    if n.startswith("overlay_"):
                        found.add(n[len("overlay_"):])
                    elif n.startswith("heatmap_"):
                        found.add(n[len("heatmap_"):])
                    else:
                        found.add(n)
        if found:
            break
    return sorted(found)


def discover_offline_inference_frames(val_dir: Path, steps: list[str]) -> list[str]:
    """Discover per-frame PNGs produced by the offline_inference probe."""
    for s in steps:
        frame_dir = val_dir / s / "offline_inference" / "unnormalized"
        if frame_dir.exists():
            return sorted(p.stem for p in frame_dir.glob("ep*.png"))
    return []


def discover_action_drift_jacobian_groups(val_dir: Path, steps: list[str]) -> list[str]:
    for s in steps:
        d = val_dir / s / "action_drift_jacobian"
        if d.exists() and d.is_dir():
            return sorted(p.name for p in d.iterdir() if p.is_dir())
    return []


def discover_action_drift_jacobian_layers(val_dir: Path, steps: list[str], groups: list[str]) -> list[str]:
    for s in steps:
        for g in groups:
            d = val_dir / s / "action_drift_jacobian" / g
            if d.exists() and d.is_dir():
                return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("L"))
    return []


def discover_action_drift_jacobian_files(val_dir: Path, steps: list[str], groups: list[str], layers: list[str]) -> list[str]:
    for s in steps:
        for g in groups:
            for l in layers:
                d = val_dir / s / "action_drift_jacobian" / g / l
                if d.exists() and d.is_dir():
                    idx = int(l.replace("L", ""))
                    prefix = f"causal_L{idx}_"
                    files = []
                    for p in d.glob("*.mp4"):
                        if p.name.startswith(prefix):
                            files.append(p.name[len(prefix):])
                        else:
                            files.append(p.name)
                    return sorted(files)
    return []


def discover_repr_spaces(val_dir: Path, steps: list[str]) -> list[str]:
    for s in steps:
        d = val_dir / s / "representations" / "2d"
        if d.exists() and d.is_dir():
            return sorted(p.name for p in d.iterdir() if p.is_dir())
    return []


def discover_scree_files(val_dir: Path, steps: list[str]) -> list[str]:
    for s in steps:
        d = val_dir / s / "representations" / "pca_variance"
        if d.exists() and d.is_dir():
            return sorted(p.stem.removesuffix("_pca_scree") for p in d.glob("*_pca_scree.png"))
    return []


def discover_spatial_memorization_layers(val_dir: Path, steps: list[str], probe_dir: str) -> list[str]:
    for s in steps:
        d = val_dir / s / probe_dir
        if d.exists() and d.is_dir():
            return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("L"))
    return []


def discover_spatial_memorization_files(val_dir: Path, steps: list[str], probe_dir: str, layers: list[str]) -> list[str]:
    for s in steps:
        for l in layers:
            d = val_dir / s / probe_dir / l
            if d.exists() and d.is_dir():
                return sorted(p.name for p in d.glob("*.png"))
    return []


def discover_critic_values_files(val_dir: Path, steps: list[str]) -> list[str]:
    """Discover PNGs produced by the critic_values_distribution probe.

    The probe writes flat PNGs into step_NNNN/critic_values_distribution/.
    Returned names preserve a friendly ordering: distribution plots first,
    then percentile frames in numeric order.
    """
    seen: set[str] = set()
    for s in steps:
        d = val_dir / s / "critic_values_distribution"
        if d.exists() and d.is_dir():
            for p in d.glob("*.png"):
                seen.add(p.name)
    if not seen:
        return []

    def _sort_key(name: str) -> tuple[int, str]:
        # advantage_dist first, then gradient_magnitudes, then frame_pXX in numeric order
        if name == "advantage_dist.png":
            return (0, name)
        if name == "gradient_magnitudes.png":
            return (1, name)
        if name.startswith("frame_p") and name.endswith(".png"):
            try:
                pct = int(name[len("frame_p"):-len(".png")])
                return (2, f"{pct:03d}")
            except ValueError:
                return (3, name)
        return (4, name)

    return sorted(seen, key=_sort_key)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def file_url(path: Path) -> str:
    """Return a Gradio-served URL for `path`, or '' if the file is missing.

    The path must be inside a directory passed to `gr.set_static_paths`.
    """
    if not path.exists():
        return ""
    return f"/gradio_api/file={path}"


def render_image_grid(paths_and_labels: list[tuple[Path, str]]) -> str:
    """Render a responsive grid of images with step labels."""
    n = len(paths_and_labels)
    if n == 0:
        return "<p>Select at least one step.</p>"

    html = '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap:8px; width:100%;">'
    for path, label in paths_and_labels:
        uri = file_url(path)
        html += f'''<div style="text-align:center;">
            <div style="font-weight:bold; margin-bottom:4px; font-size:0.95em;">{label}</div>'''
        if uri:
            html += f'<img src="{uri}" loading="lazy" style="width:100%; height:auto; border-radius:4px;" />'
        else:
            html += '<p style="color:#888;">Not found</p>'
        html += '</div>'
    html += '</div>'
    return html


def _video_sync_controls(uid: str) -> str:
    return f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px; padding:8px;
                background:#2a2a3a; border-radius:8px;">
      <button onclick="
        const vids = document.querySelectorAll('#{uid} video');
        const playing = vids[0] && !vids[0].paused;
        vids.forEach(v => playing ? v.pause() : v.play());
        this.textContent = playing ? '▶ Play' : '⏸ Pause';
      " style="padding:6px 16px; cursor:pointer; border-radius:4px; border:1px solid #555;
              background:#3a3a4a; color:white;">▶ Play</button>
      <button onclick="
        const vids = document.querySelectorAll('#{uid} video');
        vids.forEach(v => {{ v.pause(); v.currentTime = 0; }});
        document.querySelector('#{uid}-seek').value = 0;
      " style="padding:6px 12px; cursor:pointer; border-radius:4px; border:1px solid #555;
              background:#3a3a4a; color:white;">⏮ Reset</button>
      <input id="{uid}-seek" type="range" min="0" max="1000" value="0" style="flex:1;" oninput="
        const vids = document.querySelectorAll('#{uid} video');
        const frac = this.value / 1000;
        vids.forEach(v => {{ if(v.duration) v.currentTime = frac * v.duration; }});
      " />
      <select onchange="
        const vids = document.querySelectorAll('#{uid} video');
        vids.forEach(v => v.playbackRate = parseFloat(this.value));
      " style="padding:4px; border-radius:4px; border:1px solid #555; background:#3a3a4a; color:white;">
        <option value='0.25'>0.25x</option>
        <option value='0.5'>0.5x</option>
        <option value='1' selected>1x</option>
        <option value='2'>2x</option>
      </select>
    </div>
    """


def _video_seek_script(uid: str) -> str:
    return f"""
    <script>
      (function() {{
        const update = () => {{
          const vids = document.querySelectorAll('#{uid} video');
          const seek = document.querySelector('#{uid}-seek');
          if (vids[0] && seek && vids[0].duration) {{
            seek.value = (vids[0].currentTime / vids[0].duration) * 1000;
          }}
          requestAnimationFrame(update);
        }};
        requestAnimationFrame(update);
      }})();
    </script>
    """


def _render_video_item(path: Path, label: str) -> str:
    uri = file_url(path)
    html = f'<div style="text-align:center;">'
    html += f'<div style="font-weight:bold; margin-bottom:4px; font-size:0.95em;">{label}</div>'
    if uri:
        html += f'<video src="{uri}" preload="metadata" style="width:100%; height:auto; border-radius:4px;"></video>'
    else:
        html += '<p style="color:#888;">Not found</p>'
    html += '</div>'
    return html


def render_video_grid(paths_and_labels: list[tuple[Path, str]], vertical: bool = False) -> str:
    """Render synced videos with shared controls. vertical=True stacks them."""
    if not paths_and_labels:
        return "<p>Select at least one step.</p>"

    uid = "v" + str(abs(hash(str(paths_and_labels[0][0]))))[-8:]
    html = _video_sync_controls(uid)

    if vertical:
        html += f'<div id="{uid}" style="display:flex; flex-direction:column; gap:12px; width:100%; max-width:900px; margin:0 auto;">'
    else:
        html += f'<div id="{uid}" style="display:grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap:8px; width:100%;">'

    for path, label in paths_and_labels:
        html += _render_video_item(path, label)
    html += '</div>'
    html += _video_seek_script(uid)
    return html


# ---------------------------------------------------------------------------
# Path resolvers
# ---------------------------------------------------------------------------

ACTION_VIEWS = {
    "Overview": "actions/2d/overview.png",
    "Trajectories": "actions/2d/val/trajectories.png",
    "By frame": "actions/2d/val/by_frame.png",
    "By subtask": "actions/2d/val/by_subtask.png",
    "NN distances": "actions/2d/val/nn_distances.png",
}

def _attention_real_fname(layer: str, fname: str) -> str:
    """Resolve the on-disk filename: matrix_* has no prefix; layer 0 uses
    'overlay_', deeper layers use 'heatmap_'."""
    if fname.startswith("matrix_"):
        return fname
    prefix = "overlay_" if layer == "L00" else "heatmap_"
    return f"{prefix}{fname}"

REPR_COLORINGS = ["by_episode", "by_frame", "by_subtask"]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app(run_dir: str):
    val_dir = Path(run_dir).resolve() / "validation"
    if not val_dir.exists():
        raise FileNotFoundError(f"No validation directory at {val_dir}")

    gr.set_static_paths([str(val_dir)])

    all_steps = discover_steps(val_dir)
    if not all_steps:
        raise FileNotFoundError(f"No step_* directories in {val_dir}")

    episodes = discover_episodes(val_dir, all_steps)
    att_episodes, att_layers = discover_attention_episodes_layers(val_dir, all_steps)
    att_files = discover_attention_files(val_dir, all_steps, att_episodes, att_layers)
    offline_inference_frames = discover_offline_inference_frames(val_dir, all_steps)

    adj_groups = discover_action_drift_jacobian_groups(val_dir, all_steps)
    adj_layers = discover_action_drift_jacobian_layers(val_dir, all_steps, adj_groups)
    adj_files = discover_action_drift_jacobian_files(val_dir, all_steps, adj_groups, adj_layers)

    repr_spaces = discover_repr_spaces(val_dir, all_steps)
    scree_spaces = discover_scree_files(val_dir, all_steps)

    sm_layers = discover_spatial_memorization_layers(val_dir, all_steps, "spatial_memorization")
    sm_files = discover_spatial_memorization_files(val_dir, all_steps, "spatial_memorization", sm_layers)

    smj_layers = discover_spatial_memorization_layers(val_dir, all_steps, "spatial_memorization_jacobian")
    smj_files = discover_spatial_memorization_files(val_dir, all_steps, "spatial_memorization_jacobian", smj_layers)

    cv_files = discover_critic_values_files(val_dir, all_steps)
    default_indices = sorted(list({0, len(all_steps) // 2, len(all_steps) - 1}))
    default_steps = [all_steps[i] for i in default_indices]

    with gr.Blocks(title="Validation Comparison") as app:
        gr.Markdown(f"## Validation — {Path(run_dir).name}")

        step_selector = gr.CheckboxGroup(
            choices=[(step_label(s), s) for s in all_steps],
            value=default_steps,
            label="Steps to compare",
        )

        # ---- Episode thumbnails (shown above tabs if available) ----
        # Thumbnails are step-independent, so just find the first one that exists
        thumb_path = None
        for s in all_steps:
            candidate = val_dir / s / "episode_thumbnails.png"
            if candidate.exists():
                thumb_path = candidate
                break
        if thumb_path:
            thumb_uri = file_url(thumb_path)
            if thumb_uri:
                gr.HTML(f'''<details style="margin-bottom:8px;">
                    <summary style="cursor:pointer; font-weight:bold; font-size:1em;">Episode thumbnails</summary>
                    <img src="{thumb_uri}" loading="lazy" style="width:100%; max-width:1200px; height:auto; margin-top:6px; border-radius:4px;" />
                </details>''')

        # ---- Actions tab ----
        with gr.Tab("Actions"):
            action_choices = list(ACTION_VIEWS.keys()) + [f"Episode: {e}" for e in episodes]
            action_dd = gr.Dropdown(choices=action_choices, value=action_choices[0], label="View")
            action_html = gr.HTML()

            def render_actions(selected_steps, view):
                if not selected_steps or not view:
                    return ""
                if view.startswith("Episode: "):
                    ep = view.split("Episode: ")[1]
                    items = [(val_dir / s / "actions" / "2d" / "val" / "episodes" / f"{ep}.png", step_label(s))
                             for s in selected_steps]
                else:
                    rel = ACTION_VIEWS[view]
                    items = [(val_dir / s / rel, step_label(s)) for s in selected_steps]
                return render_image_grid(items)

            action_dd.change(render_actions, [step_selector, action_dd], action_html)
            step_selector.change(render_actions, [step_selector, action_dd], action_html)
            app.load(render_actions, [step_selector, action_dd], action_html)

        # ---- Attention tab ----
        if att_episodes and att_layers and att_files:
            with gr.Tab("Attention"):
                with gr.Row():
                    att_ep_dd = gr.Dropdown(
                        choices=att_episodes, value=att_episodes[0], label="Episode"
                    )
                    att_layer_dd = gr.Dropdown(
                        choices=att_layers, value=att_layers[0], label="Layer"
                    )
                    att_file_dd = gr.Dropdown(
                        choices=att_files, value=att_files[0], label="View"
                    )
                att_html = gr.HTML()

                def render_attention(selected_steps, ep, layer, fname):
                    if not selected_steps or not ep or not layer or not fname:
                        return ""
                    real_fname = _attention_real_fname(layer, fname)
                    items = [
                        (val_dir / s / "attention" / f"{ep}_{layer}" / real_fname, step_label(s))
                        for s in selected_steps
                    ]
                    vertical = "mean" in fname or "summary" in fname
                    return render_video_grid(items, vertical=vertical)

                att_ep_dd.change(render_attention, [step_selector, att_ep_dd, att_layer_dd, att_file_dd], att_html)
                att_layer_dd.change(render_attention, [step_selector, att_ep_dd, att_layer_dd, att_file_dd], att_html)
                att_file_dd.change(render_attention, [step_selector, att_ep_dd, att_layer_dd, att_file_dd], att_html)
                step_selector.change(render_attention, [step_selector, att_ep_dd, att_layer_dd, att_file_dd], att_html)
                app.load(render_attention, [step_selector, att_ep_dd, att_layer_dd, att_file_dd], att_html)

        # ---- Representations tab ----
        if repr_spaces:
            with gr.Tab("Representations"):
                with gr.Row():
                    repr_space_dd = gr.Dropdown(choices=repr_spaces, value=repr_spaces[0], label="Space")
                    repr_color_dd = gr.Dropdown(choices=REPR_COLORINGS, value=REPR_COLORINGS[0], label="Coloring")
                repr_html = gr.HTML()

                def render_repr(selected_steps, space, coloring):
                    if not selected_steps or not space or not coloring:
                        return ""
                    items = [(val_dir / s / "representations" / "2d" / space / f"{coloring}.png", step_label(s))
                             for s in selected_steps]
                    return render_image_grid(items)

                repr_space_dd.change(render_repr, [step_selector, repr_space_dd, repr_color_dd], repr_html)
                repr_color_dd.change(render_repr, [step_selector, repr_space_dd, repr_color_dd], repr_html)
                step_selector.change(render_repr, [step_selector, repr_space_dd, repr_color_dd], repr_html)
                app.load(render_repr, [step_selector, repr_space_dd, repr_color_dd], repr_html)

                # PCA scree
                if scree_spaces:
                    gr.Markdown("### PCA scree")
                    scree_dd = gr.Dropdown(choices=scree_spaces, value=scree_spaces[0], label="Space")
                    scree_html = gr.HTML()

                    def render_scree(selected_steps, space):
                        if not selected_steps or not space:
                            return ""
                        fname = f"{space}_pca_scree.png"
                        items = [(val_dir / s / "representations" / "pca_variance" / fname, step_label(s))
                                 for s in selected_steps]
                        return render_image_grid(items)

                    scree_dd.change(render_scree, [step_selector, scree_dd], scree_html)
                    step_selector.change(render_scree, [step_selector, scree_dd], scree_html)
                    app.load(render_scree, [step_selector, scree_dd], scree_html)

        # ---- Offline Inference tab ----
        if offline_inference_frames:
            with gr.Tab("Offline Inference"):
                with gr.Row():
                    oe_space_dd = gr.Dropdown(
                        choices=["Unnormalized", "Normalized"],
                        value="Unnormalized",
                        label="Space",
                    )
                    oe_frame_dd = gr.Dropdown(
                        choices=offline_inference_frames,
                        value=offline_inference_frames[0] if offline_inference_frames else None,
                        label="Frame",
                    )
                oe_html = gr.HTML()

                def render_offline_inference(selected_steps, space, frame):
                    if not selected_steps or not space or not frame:
                        return ""
                    subdir = "unnormalized" if space == "Unnormalized" else "normalized"
                    items = [
                        (val_dir / s / "offline_inference" / subdir / f"{frame}.png", step_label(s))
                        for s in selected_steps
                    ]
                    return render_image_grid(items)

                oe_space_dd.change(render_offline_inference, [step_selector, oe_space_dd, oe_frame_dd], oe_html)
                oe_frame_dd.change(render_offline_inference, [step_selector, oe_space_dd, oe_frame_dd], oe_html)
                step_selector.change(render_offline_inference, [step_selector, oe_space_dd, oe_frame_dd], oe_html)
                app.load(render_offline_inference, [step_selector, oe_space_dd, oe_frame_dd], oe_html)

        # ---- Action Drift Jacobian tab ----
        if adj_groups and adj_layers and adj_files:
            with gr.Tab("Action Drift Jacobian"):
                with gr.Row():
                    adj_group_dd = gr.Dropdown(
                        choices=adj_groups, value=adj_groups[0], label="Episode / timestep"
                    )
                    adj_layer_dd = gr.Dropdown(
                        choices=adj_layers, value=adj_layers[0], label="Layer"
                    )
                    adj_file_dd = gr.Dropdown(
                        choices=adj_files, value=adj_files[0], label="View"
                    )
                adj_html = gr.HTML()

                def render_adj(selected_steps, group, layer, fname):
                    if not selected_steps or not group or not layer or not fname:
                        return ""
                    idx = int(layer.replace("L", ""))
                    prefix = f"causal_L{idx}_"
                    real_fname = fname if fname.startswith("causal_") else f"{prefix}{fname}"
                    items = [
                        (val_dir / s / "action_drift_jacobian" / group / layer / real_fname, step_label(s))
                        for s in selected_steps
                    ]
                    vertical = "summary" in fname
                    return render_video_grid(items, vertical=vertical)

                adj_group_dd.change(render_adj, [step_selector, adj_group_dd, adj_layer_dd, adj_file_dd], adj_html)
                adj_layer_dd.change(render_adj, [step_selector, adj_group_dd, adj_layer_dd, adj_file_dd], adj_html)
                adj_file_dd.change(render_adj, [step_selector, adj_group_dd, adj_layer_dd, adj_file_dd], adj_html)
                step_selector.change(render_adj, [step_selector, adj_group_dd, adj_layer_dd, adj_file_dd], adj_html)
                app.load(render_adj, [step_selector, adj_group_dd, adj_layer_dd, adj_file_dd], adj_html)

        # ---- Spatial Memorization tab ----
        if sm_layers and sm_files:
            with gr.Tab("Spatial Memorization"):
                with gr.Row():
                    sm_layer_dd = gr.Dropdown(
                        choices=sm_layers, value=sm_layers[0], label="Layer"
                    )
                    sm_file_dd = gr.Dropdown(
                        choices=sm_files, value=sm_files[0], label="View"
                    )
                sm_html = gr.HTML()

                def render_sm(selected_steps, layer, fname):
                    if not selected_steps or not layer or not fname:
                        return ""
                    items = [
                        (val_dir / s / "spatial_memorization" / layer / fname, step_label(s))
                        for s in selected_steps
                    ]
                    return render_image_grid(items)

                sm_layer_dd.change(render_sm, [step_selector, sm_layer_dd, sm_file_dd], sm_html)
                sm_file_dd.change(render_sm, [step_selector, sm_layer_dd, sm_file_dd], sm_html)
                step_selector.change(render_sm, [step_selector, sm_layer_dd, sm_file_dd], sm_html)
                app.load(render_sm, [step_selector, sm_layer_dd, sm_file_dd], sm_html)

        # ---- Spatial Memorization Jacobian tab ----
        if smj_layers and smj_files:
            with gr.Tab("Spatial Memorization Jacobian"):
                with gr.Row():
                    smj_layer_dd = gr.Dropdown(
                        choices=smj_layers, value=smj_layers[0], label="Layer"
                    )
                    smj_file_dd = gr.Dropdown(
                        choices=smj_files, value=smj_files[0], label="View"
                    )
                smj_html = gr.HTML()

                def render_smj(selected_steps, layer, fname):
                    if not selected_steps or not layer or not fname:
                        return ""
                    items = [
                        (val_dir / s / "spatial_memorization_jacobian" / layer / fname, step_label(s))
                        for s in selected_steps
                    ]
                    return render_image_grid(items)

                smj_layer_dd.change(render_smj, [step_selector, smj_layer_dd, smj_file_dd], smj_html)
                smj_file_dd.change(render_smj, [step_selector, smj_layer_dd, smj_file_dd], smj_html)
                step_selector.change(render_smj, [step_selector, smj_layer_dd, smj_file_dd], smj_html)
                app.load(render_smj, [step_selector, smj_layer_dd, smj_file_dd], smj_html)

        # ---- Critic Values tab ----
        if cv_files:
            with gr.Tab("Critic Values"):
                cv_file_dd = gr.Dropdown(
                    choices=cv_files, value=cv_files[0], label="View"
                )
                cv_html = gr.HTML()

                def render_cv(selected_steps, fname):
                    if not selected_steps or not fname:
                        return ""
                    items = [
                        (val_dir / s / "critic_values_distribution" / fname, step_label(s))
                        for s in selected_steps
                    ]
                    return render_image_grid(items)

                cv_file_dd.change(render_cv, [step_selector, cv_file_dd], cv_html)
                step_selector.change(render_cv, [step_selector, cv_file_dd], cv_html)
                app.load(render_cv, [step_selector, cv_file_dd], cv_html)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View validation outputs across training steps.")
    parser.add_argument("--run_dir", help="Path to the training run directory (contains validation/)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.run_dir)
    app.launch(server_port=args.port, share=args.share)

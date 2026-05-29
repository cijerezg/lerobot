"""Side-by-side comparison of validation outputs across training steps."""

import argparse
import html as html_lib
from pathlib import Path
from urllib.parse import quote

import gradio as gr

SPATIAL_ATTENTION_DIRS = ("spatial_memorization_attention", "spatial_memorization")
SPATIAL_ACTION_JACOBIAN_DIRS = (
    "spatial_memorization_action_jacobian",
    "spatial_memorization_jacobian",
)

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
        frame_dir = _first_existing_rel(
            val_dir / s / "offline_inference",
            ("unnormalized_eval", "unnormalized"),
        )
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


def _probe_dir_for_step(val_dir: Path, step: str, probe_dirs: str | tuple[str, ...]) -> Path:
    aliases = (probe_dirs,) if isinstance(probe_dirs, str) else probe_dirs
    for probe_dir in aliases:
        d = val_dir / step / probe_dir
        if d.exists() and d.is_dir():
            return d
    return val_dir / step / aliases[0]


def discover_spatial_memorization_layers(
    val_dir: Path, steps: list[str], probe_dirs: str | tuple[str, ...]
) -> list[str]:
    layers: set[str] = set()
    aliases = (probe_dirs,) if isinstance(probe_dirs, str) else probe_dirs
    for s in steps:
        for probe_dir in aliases:
            d = val_dir / s / probe_dir
            if d.exists() and d.is_dir():
                layers.update(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("L"))
    return sorted(layers)


def discover_spatial_memorization_files(
    val_dir: Path, steps: list[str], probe_dirs: str | tuple[str, ...], layers: list[str]
) -> list[str]:
    files: set[str] = set()
    aliases = (probe_dirs,) if isinstance(probe_dirs, str) else probe_dirs
    for s in steps:
        for probe_dir in aliases:
            for l in layers:
                d = val_dir / s / probe_dir / l
                if d.exists() and d.is_dir():
                    files.update(p.name for p in d.glob("*.png"))
    return sorted(files)


def discover_critic_values_files(val_dir: Path, steps: list[str]) -> list[str]:
    """Discover the flat distribution PNGs produced by the critic probe.

    The probe writes them into step_NNNN/critic/. Returned names preserve a
    friendly ordering: predicted distributions, then advantage plots, then
    gradient plot, then percentile frames in numeric order.
    """
    seen: set[str] = set()
    for s in steps:
        d = val_dir / s / "critic"
        if d.exists() and d.is_dir():
            for p in d.glob("*.png"):
                seen.add(p.name)
    if not seen:
        return []

    fixed_order = {
        "predicted_distributions.png": 0,
        "advantage_dist.png": 1,
        "advantage_squashed_dist.png": 2,
        "gradient_magnitudes.png": 3,
    }

    def _sort_key(name: str) -> tuple[int, str]:
        if name in fixed_order:
            return (fixed_order[name], name)
        if name.startswith("frame_p") and name.endswith(".png"):
            try:
                pct = int(name[len("frame_p"):-len(".png")])
                return (4, f"{pct:03d}")
            except ValueError:
                return (5, name)
        return (6, name)

    return sorted(seen, key=_sort_key)


def discover_critic_trace_episodes(val_dir: Path, steps: list[str]) -> list[str]:
    """Scan step_NNNN/critic/episode_traces/ep* across steps; return episode names."""
    eps: set[str] = set()
    for s in steps:
        d = val_dir / s / "critic" / "episode_traces"
        if d.exists() and d.is_dir():
            eps.update(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("ep"))
    return sorted(eps)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def file_url(path: Path) -> str:
    """Return a Gradio-served URL for `path`, or '' if the file is missing.

    The path must be inside a directory passed to `gr.set_static_paths`.
    """
    if not path.exists():
        return ""
    return f"/gradio_api/file={quote(str(path), safe='/')}"


def not_loaded_html(kind: str) -> str:
    return f'<p style="color:#888; margin:8px 0;">{html_lib.escape(kind)} view not loaded.</p>'


# Panels stack into a single vertical column when each one is wider than this
# ratio (W/H); otherwise they sit side by side in a row. So media wider than 2:1
# gets full width instead of thin strips. Measured client-side — no external tools.
ASPECT_VERTICAL_THRESHOLD = 2.0


def _media_tag(path: Path, kind: str) -> str:
    uri = file_url(path)
    if not uri:
        return '<p style="color:#888;">Not found</p>'
    if kind == "video":
        # No src until visible — the observer in _grid_script sets it on scroll-in,
        # so hidden tabs and off-screen panels never download. Prevents load hangs.
        return (
            f'<video class="vv-media" data-src="{uri}" muted loop playsinline '
            f'preload="none" style="width:100%; height:auto; border-radius:4px;"></video>'
        )
    return (
        f'<img class="vv-media" src="{uri}" loading="lazy" '
        f'style="width:100%; height:auto; border-radius:4px;" />'
    )


def _grid_script(uid: str, n: int) -> str:
    """Client-side: orient the grid by aspect ratio, and lazy-load videos on visibility.

    Videos carry only a data-src; the IntersectionObserver assigns src and plays them
    when they scroll into view and pauses when they leave, so off-screen and hidden-tab
    panels never download — that is what keeps the page from hanging on many videos.
    """
    template = """
    <script>
      (function() {
        var grid = document.getElementById("__UID__");
        if (!grid) return;
        var n = __N__, limit = __LIMIT__, laidOut = false;
        function apply(r) {
          var vertical = r > limit;
          grid.style.gridTemplateColumns = vertical ? "1fr" : "repeat(" + n + ", 1fr)";
          grid.style.maxWidth = vertical ? "1100px" : "none";
          grid.style.margin = vertical ? "0 auto" : "0";
        }
        // Grid starts as a single column (inline style); we only switch to a row
        // once measured AND the panels are narrow, so wide videos are never all
        // on-screen at once.
        function measure(el) {
          if (laidOut) return;
          var w = el.naturalWidth || el.videoWidth, h = el.naturalHeight || el.videoHeight;
          if (w && h) { laidOut = true; apply(w / h); }
        }
        var medias = grid.querySelectorAll(".vv-media");
        medias.forEach(function(el) {
          if (el.tagName === "IMG") {
            el.complete ? measure(el) : el.addEventListener("load", function() { measure(el); });
          }
        });
        var vids = [].filter.call(medias, function(el) { return el.tagName === "VIDEO"; });
        if (!vids.length) return;

        // Load at most one video at a time, and only those scrolled into view, so the
        // browser never downloads/decodes several large streams at once (the freeze).
        var visible = [], loading = false;
        function pump() {
          if (loading) return;
          for (var i = 0; i < vids.length; i++) {
            var v = vids[i];
            if (visible.indexOf(v) !== -1 && !v.src && v.dataset.src) {
              loading = true;
              v.addEventListener("loadedmetadata", (function(el) { return function() { measure(el); }; })(v), { once: true });
              var done = function() { loading = false; pump(); };
              v.addEventListener("canplay", done, { once: true });
              v.addEventListener("error", done, { once: true });
              v.src = v.dataset.src;
              v.play().catch(function() {});
              return;
            }
          }
        }
        function show(v) { if (visible.indexOf(v) === -1) visible.push(v); if (v.src) v.play().catch(function() {}); }
        function hide(v) { var i = visible.indexOf(v); if (i !== -1) visible.splice(i, 1); v.pause(); }
        if ("IntersectionObserver" in window) {
          var io = new IntersectionObserver(function(entries) {
            entries.forEach(function(e) { e.isIntersecting ? show(e.target) : hide(e.target); });
            pump();
          }, { rootMargin: "100px" });
          vids.forEach(function(v) { io.observe(v); });
        } else {
          vids.forEach(show); pump();
        }
      })();
    </script>
    """
    return (
        template.replace("__UID__", uid)
        .replace("__N__", str(n))
        .replace("__LIMIT__", str(ASPECT_VERTICAL_THRESHOLD))
    )


def render_media_grid(paths_and_labels: list[tuple[Path, str]], kind: str = "image") -> str:
    """Render images or autoplaying synced videos, auto-oriented by measured aspect ratio."""
    if not paths_and_labels:
        return "<p>Select at least one step.</p>"

    n = len(paths_and_labels)
    uid = "v" + str(abs(hash(str(paths_and_labels[0][0]))))[-8:]

    html = _video_sync_controls(uid) if kind == "video" else ""
    html += f'<div id="{uid}" style="display:grid; grid-template-columns: 1fr; gap:10px; width:100%;">'
    for path, label in paths_and_labels:
        html += (
            '<div style="text-align:center;">'
            f'<div style="font-weight:bold; margin-bottom:4px; font-size:0.95em;">{html_lib.escape(label)}</div>'
            f'{_media_tag(path, kind)}</div>'
        )
    html += '</div>'
    if kind == "video":
        html += _video_seek_script(uid)
    html += _grid_script(uid, n)
    return html


def render_image_grid(paths_and_labels: list[tuple[Path, str]]) -> str:
    return render_media_grid(paths_and_labels, "image")


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
              background:#3a3a4a; color:white;">⏸ Pause</button>
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
    # Event-driven, NOT a requestAnimationFrame loop: 'timeupdate' fires only while
    # the video plays and stops on its own, and the listener dies with the element.
    # The old rAF version never terminated and piled up on every re-render, saturating
    # the main thread and freezing the whole page (every tab, images included).
    return f"""
    <script>
      (function() {{
        const seek = document.getElementById("{uid}-seek");
        const first = document.querySelector('#{uid} video');
        if (!seek || !first) return;
        first.addEventListener("timeupdate", function() {{
          if (first.duration) seek.value = (first.currentTime / first.duration) * 1000;
        }});
      }})();
    </script>
    """


def render_video_grid(paths_and_labels: list[tuple[Path, str]]) -> str:
    return render_media_grid(paths_and_labels, "video")


# ---------------------------------------------------------------------------
# Path resolvers
# ---------------------------------------------------------------------------

ACTION_VIEWS = {
    "Overview": "actions/2d/overview.png",
    "Trajectories": ("actions/2d/trajectories.png", "actions/2d/val/trajectories.png"),
    "By frame": ("actions/2d/by_frame.png", "actions/2d/val/by_frame.png"),
    "By subtask": ("actions/2d/by_subtask.png", "actions/2d/val/by_subtask.png"),
    "NN distances": ("actions/2d/nn_distances.png", "actions/2d/val/nn_distances.png"),
}

def _attention_real_fname(layer: str, fname: str) -> str:
    """Best legacy guess for an attention filename."""
    unprefixed = (
        "matrix_",
        "cross_matrix",
        "self_matrix",
        "action_text_matrix",
    )
    if fname.startswith(unprefixed):
        return fname
    prefix = "overlay_" if layer == "L00" else "heatmap_"
    return f"{prefix}{fname}"


def _attention_path(layer_dir: Path, layer: str, fname: str) -> Path:
    """Resolve generic and older pi05 attention filename conventions."""
    candidates = [
        fname,
        f"overlay_{fname}",
        f"heatmap_{fname}",
        _attention_real_fname(layer, fname),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = layer_dir / candidate
        if path.exists():
            return path
    return layer_dir / candidates[0]

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

    sm_layers = discover_spatial_memorization_layers(val_dir, all_steps, SPATIAL_ATTENTION_DIRS)
    sm_files = discover_spatial_memorization_files(val_dir, all_steps, SPATIAL_ATTENTION_DIRS, sm_layers)

    smj_layers = discover_spatial_memorization_layers(val_dir, all_steps, SPATIAL_ACTION_JACOBIAN_DIRS)
    smj_files = discover_spatial_memorization_files(val_dir, all_steps, SPATIAL_ACTION_JACOBIAN_DIRS, smj_layers)

    cv_files = discover_critic_values_files(val_dir, all_steps)
    cv_trace_eps = discover_critic_trace_episodes(val_dir, all_steps)
    default_indices = sorted(list({0, len(all_steps) // 2, len(all_steps) - 1}))
    default_steps = [all_steps[i] for i in default_indices]

    with gr.Blocks(title="Validation Comparison") as app:
        gr.Markdown(f"## Validation — {Path(run_dir).name}")

        step_selector = gr.CheckboxGroup(
            choices=[(step_label(s), s) for s in all_steps],
            value=default_steps,
            label="Steps to compare",
        )

        def wire(fn, inputs, outputs):
            """Render on any input change and on first load — no buttons to click."""
            for comp in inputs:
                comp.change(fn, inputs, outputs)
            app.load(fn, inputs, outputs)

        # ---- Episode thumbnails (shown above tabs if available) ----
        # Thumbnails are step-independent, so just find the first one that exists
        thumb_path = None
        for s in all_steps:
            for candidate in (
                val_dir / s / "episode_thumbnails.png",
                val_dir / s / "representations" / "episode_thumbnails.png",
            ):
                if candidate.exists():
                    thumb_path = candidate
                    break
            if thumb_path:
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
                    items = [
                        (_first_existing_rel(
                            val_dir / s,
                            ("actions/2d/episodes", "actions/2d/val/episodes"),
                        ) / f"{ep}.png", step_label(s))
                        for s in selected_steps
                    ]
                else:
                    rel = ACTION_VIEWS[view]
                    items = [(_first_existing_rel(val_dir / s, rel), step_label(s)) for s in selected_steps]
                return render_image_grid(items)

            wire(render_actions, [step_selector, action_dd], action_html)

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
                att_html = gr.HTML(not_loaded_html("Attention"))

                def render_attention(selected_steps, ep, layer, fname):
                    if not selected_steps or not ep or not layer or not fname:
                        return ""
                    items = [
                        (_attention_path(val_dir / s / "attention" / f"{ep}_{layer}", layer, fname), step_label(s))
                        for s in selected_steps
                    ]
                    return render_video_grid(items)

                wire(render_attention, [step_selector, att_ep_dd, att_layer_dd, att_file_dd], att_html)

        # ---- Representations tab ----
        if repr_spaces:
            with gr.Tab("Representations"):
                with gr.Row():
                    repr_space_dd = gr.Dropdown(choices=repr_spaces, value=repr_spaces[0], label="Space")
                    repr_color_dd = gr.Dropdown(choices=REPR_COLORINGS, value=REPR_COLORINGS[0], label="Coloring")
                repr_html = gr.HTML(not_loaded_html("Representations"))

                def render_repr(selected_steps, space, coloring):
                    if not selected_steps or not space or not coloring:
                        return ""
                    items = [(val_dir / s / "representations" / "2d" / space / f"{coloring}.png", step_label(s))
                             for s in selected_steps]
                    return render_image_grid(items)

                wire(render_repr, [step_selector, repr_space_dd, repr_color_dd], repr_html)

                # PCA scree
                if scree_spaces:
                    gr.Markdown("### PCA scree")
                    scree_dd = gr.Dropdown(choices=scree_spaces, value=scree_spaces[0], label="Space")
                    scree_html = gr.HTML(not_loaded_html("PCA scree"))

                    def render_scree(selected_steps, space):
                        if not selected_steps or not space:
                            return ""
                        fname = f"{space}_pca_scree.png"
                        items = [(val_dir / s / "representations" / "pca_variance" / fname, step_label(s))
                                 for s in selected_steps]
                        return render_image_grid(items)

                    wire(render_scree, [step_selector, scree_dd], scree_html)

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
                oe_html = gr.HTML(not_loaded_html("Offline Inference"))

                def render_offline_inference(selected_steps, space, frame):
                    if not selected_steps or not space or not frame:
                        return ""
                    subdirs = (
                        ("unnormalized_eval", "unnormalized")
                        if space == "Unnormalized"
                        else ("normalized_eval", "normalized")
                    )
                    items = [
                        (_first_existing_rel(val_dir / s / "offline_inference", subdirs) / f"{frame}.png", step_label(s))
                        for s in selected_steps
                    ]
                    return render_image_grid(items)

                wire(render_offline_inference, [step_selector, oe_space_dd, oe_frame_dd], oe_html)

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
                adj_html = gr.HTML(not_loaded_html("Action Drift Jacobian"))

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
                    return render_video_grid(items)

                wire(render_adj, [step_selector, adj_group_dd, adj_layer_dd, adj_file_dd], adj_html)

        # ---- Spatial Memorization Attention tab ----
        if sm_layers and sm_files:
            with gr.Tab("Spatial Memorization Attention"):
                with gr.Row():
                    sm_layer_dd = gr.Dropdown(
                        choices=sm_layers, value=sm_layers[0], label="Layer"
                    )
                    sm_file_dd = gr.Dropdown(
                        choices=sm_files, value=sm_files[0], label="View"
                    )
                sm_html = gr.HTML(not_loaded_html("Spatial Memorization Attention"))

                def render_sm(selected_steps, layer, fname):
                    if not selected_steps or not layer or not fname:
                        return ""
                    items = [
                        (_probe_dir_for_step(val_dir, s, SPATIAL_ATTENTION_DIRS) / layer / fname, step_label(s))
                        for s in selected_steps
                    ]
                    return render_image_grid(items)

                wire(render_sm, [step_selector, sm_layer_dd, sm_file_dd], sm_html)

        # ---- Spatial Memorization Action-Jacobian tab ----
        if smj_layers and smj_files:
            with gr.Tab("Spatial Memorization Action-Jacobian"):
                with gr.Row():
                    smj_layer_dd = gr.Dropdown(
                        choices=smj_layers, value=smj_layers[0], label="Layer"
                    )
                    smj_file_dd = gr.Dropdown(
                        choices=smj_files, value=smj_files[0], label="View"
                    )
                smj_html = gr.HTML(not_loaded_html("Spatial Memorization Action-Jacobian"))

                def render_smj(selected_steps, layer, fname):
                    if not selected_steps or not layer or not fname:
                        return ""
                    items = [
                        (_probe_dir_for_step(val_dir, s, SPATIAL_ACTION_JACOBIAN_DIRS) / layer / fname, step_label(s))
                        for s in selected_steps
                    ]
                    return render_image_grid(items)

                wire(render_smj, [step_selector, smj_layer_dd, smj_file_dd], smj_html)

        # ---- Critic Values tab ----
        if cv_files or cv_trace_eps:
            with gr.Tab("Critic Values"):
                if cv_files:
                    gr.Markdown("### Distribution plots")
                    cv_file_dd = gr.Dropdown(
                        choices=cv_files, value=cv_files[0], label="View"
                    )
                    cv_html = gr.HTML(not_loaded_html("Critic Values"))

                    def render_cv(selected_steps, fname):
                        if not selected_steps or not fname:
                            return ""
                        items = [
                            (val_dir / s / "critic" / fname, step_label(s))
                            for s in selected_steps
                        ]
                        return render_image_grid(items)

                    wire(render_cv, [step_selector, cv_file_dd], cv_html)

                if cv_trace_eps:
                    gr.Markdown("### Episode traces")
                    cv_ep_dd = gr.Dropdown(
                        choices=cv_trace_eps, value=cv_trace_eps[0], label="Episode"
                    )
                    cv_curve_html = gr.HTML(not_loaded_html("V(s) curve"))
                    cv_video_html = gr.HTML(not_loaded_html("Overlay video"))

                    def render_cv_traces(selected_steps, ep):
                        if not selected_steps or not ep:
                            return "", ""
                        ep_dirs = [
                            (val_dir / s / "critic" / "episode_traces" / ep, step_label(s))
                            for s in selected_steps
                        ]
                        curve = render_image_grid(
                            [(d / "critic_plot.png", label) for d, label in ep_dirs]
                        )
                        video = render_video_grid(
                            [(d / "episode_video.mp4", label) for d, label in ep_dirs]
                        )
                        return curve, video

                    wire(render_cv_traces, [step_selector, cv_ep_dd], [cv_curve_html, cv_video_html])

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View validation outputs across training steps.")
    parser.add_argument("--run_dir", help="Path to the training run directory (contains validation/)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.run_dir)
    app.launch(server_port=args.port, share=args.share)

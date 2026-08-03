"""Side-by-side 3-D visualization of MolmoAct2 FAST and flow action chunks.

Both scenes share one cubic range and one camera, so a shape difference between the
two columns is a real difference in commanded geometry and not a difference in scale.

Two asymmetries make this a visual comparison rather than a score. FAST is greedy —
one reconstruction exists, and it carries the action tokenizer's quantization error,
which flow does not. And MolmoAct2's discrete path never runs the action expert, so
when point-map depth is configured the depth stream reaches the flow samples and
structurally cannot reach the FAST reconstruction.
"""

from __future__ import annotations

import numpy as np

FAST_COLOR = "#D97706"
FLOW_COLORS = ("#2563EB", "#0891B2", "#0F766E")
GT_COLOR = "#18181B"
NOW_COLOR = "#D81B60"

# Flow draws shown against the single greedy FAST reconstruction. The probe raises its
# fan to at least this many samples when the comparison is enabled.
FLOW_COMPARE_COUNT = len(FLOW_COLORS)


def _hover_data(chunk: np.ndarray, fps: float) -> np.ndarray:
    step = np.arange(len(chunk))
    time_ms = step / float(fps) * 1000.0
    roll = chunk[:, 5] if chunk.shape[1] > 5 else np.full(len(chunk), np.nan)
    grip = chunk[:, 6] if chunk.shape[1] > 6 else np.full(len(chunk), np.nan)
    return np.column_stack([step, time_ms, roll, grip])


def _path_trace(go, ee, chunk, *, label, color, fps, width, opacity, dash="solid"):
    if ee is None or chunk is None:
        return go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="lines",
            line=dict(color=color, width=width),
            name=label,
            showlegend=True,
            hoverinfo="skip",
        )
    custom = _hover_data(chunk, fps)
    return go.Scatter3d(
        x=ee[:, 0],
        y=ee[:, 1],
        z=ee[:, 2],
        mode="lines+markers",
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=3, color=color),
        opacity=opacity,
        customdata=custom,
        name=label,
        hovertemplate=(
            "<b>" + label + "</b><br>step %{customdata[0]:.0f} · %{customdata[1]:.0f} ms"
            "<br>x %{x:.3f} m · y %{y:.3f} m · z %{z:.3f} m"
            "<br>wrist roll %{customdata[2]:.1f}° · gripper %{customdata[3]:.1f}°"
            "<extra></extra>"
        ),
    )


def _gap_trace(go, start, ee, *, label, color, width, opacity):
    if ee is None:
        return go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="lines",
            line=dict(color=color, width=width, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    return go.Scatter3d(
        x=[start[0], ee[0, 0]],
        y=[start[1], ee[0, 1]],
        z=[start[2], ee[0, 2]],
        mode="lines",
        line=dict(color=color, width=width, dash="dot"),
        opacity=opacity,
        name=label,
        showlegend=False,
        hovertemplate=(
            "<b>" + label + "</b>"
            "<br>measured pose → first commanded target"
            "<br><i>This is the initial target gap, not a trace timestep.</i><extra></extra>"
        ),
    )


def _common_traces(go, record, *, showlegend, fps):
    gt = _path_trace(
        go,
        record["gt_ee"],
        record["gt_chunk"],
        label="ground truth",
        color=GT_COLOR,
        fps=fps,
        width=7,
        opacity=0.9,
    )
    gt.showlegend = showlegend
    skeleton = record["anchor_skeleton"]
    arm = go.Scatter3d(
        x=skeleton[:, 0],
        y=skeleton[:, 1],
        z=skeleton[:, 2],
        mode="lines+markers",
        line=dict(color="#A1A1AA", width=8),
        marker=dict(size=4, color="#71717A"),
        name="measured arm now",
        showlegend=showlegend,
        hovertemplate="measured follower arm<extra></extra>",
    )
    now = go.Scatter3d(
        x=[record["start_ee"][0]],
        y=[record["start_ee"][1]],
        z=[record["start_ee"][2]],
        mode="markers",
        marker=dict(size=9, color=NOW_COLOR, symbol="diamond"),
        name="measured gripper now",
        showlegend=showlegend,
        hovertemplate="measured gripper pose now<extra></extra>",
    )
    gt_gap = _gap_trace(
        go,
        record["start_ee"],
        record["gt_ee"],
        label="GT initial target gap",
        color=GT_COLOR,
        width=4,
        opacity=0.55,
    )
    return [gt, arm, now, gt_gap]


def _record_traces(record, fps):
    import plotly.graph_objects as go

    traces = [(trace, 1) for trace in _common_traces(go, record, showlegend=True, fps=fps)]

    fast_error = record.get("fast_error")
    fast_label = "FAST · greedy" if not fast_error else "FAST unavailable · " + str(fast_error)
    fast_ee = record.get("fast_ee")
    fast_chunk = record.get("fast_chunk")
    traces.extend(
        [
            (
                _path_trace(
                    go,
                    fast_ee,
                    fast_chunk,
                    label=fast_label,
                    color=FAST_COLOR,
                    fps=fps,
                    width=6,
                    opacity=0.95,
                ),
                1,
            ),
            (
                _gap_trace(
                    go,
                    record["start_ee"],
                    fast_ee,
                    label="FAST initial target gap",
                    color=FAST_COLOR,
                    width=4,
                    opacity=0.8,
                ),
                1,
            ),
        ]
    )

    traces.extend((trace, 2) for trace in _common_traces(go, record, showlegend=False, fps=fps))
    for sample_idx in range(FLOW_COMPARE_COUNT):
        ee = record["sample_ee"][sample_idx]
        chunk = record["sample_chunks"][sample_idx]
        color = FLOW_COLORS[sample_idx]
        traces.extend(
            [
                (
                    _path_trace(
                        go,
                        ee,
                        chunk,
                        label=f"flow · seed {sample_idx}",
                        color=color,
                        fps=fps,
                        width=5,
                        opacity=0.82,
                    ),
                    2,
                ),
                (
                    _gap_trace(
                        go,
                        record["start_ee"],
                        ee,
                        label=f"flow seed {sample_idx} initial target gap",
                        color=color,
                        width=3,
                        opacity=0.65,
                    ),
                    2,
                ),
            ]
        )
    return traces


def _title(record):
    subtask = record.get("subtask") or "no subtask"
    return (
        f"<b>FAST vs flow action trace</b> · episode {record['episode']} · "
        f"frame {record['frame']}<br><span style='font-size:12px;color:#52525b'>{subtask}</span>"
    )


def build_figure(records, table_z, fps=30.0):
    """Build two locked-scale 3-D scenes with no decoder-comparison score."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    point_sets = []
    for record in records:
        point_sets.extend(
            [
                record["gt_ee"],
                record["sample_ee"][:FLOW_COMPARE_COUNT].reshape(-1, 3),
                record["start_ee"][None],
            ]
        )
        if record.get("fast_ee") is not None:
            point_sets.append(record["fast_ee"])
    all_points = np.concatenate(point_sets)
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    # The table plane is drawn at table_z, which is usually well below every commanded
    # pose. Explicit axis ranges clip anything outside them, so the plane has to be in
    # the extent or it silently disappears.
    lo[2] = min(lo[2], table_z)
    hi[2] = max(hi[2], table_z)
    # aspectmode="cube" renders each axis at the same screen length, so the range has to
    # be a cube too — otherwise the short axis is stretched and the two columns are
    # compared in a distorted space.
    centre = (lo + hi) / 2.0
    half = max((hi - lo).max() / 2.0, 0.05) * 1.08
    lo = centre - half
    hi = centre + half

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "FAST · greedy reconstruction",
            f"Flow matching · {FLOW_COMPARE_COUNT} fixed noise seeds",
        ),
        horizontal_spacing=0.025,
    )

    for col in (1, 2):
        fig.add_trace(
            go.Mesh3d(
                x=[lo[0], hi[0], hi[0], lo[0]],
                y=[lo[1], lo[1], hi[1], hi[1]],
                z=[table_z] * 4,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="#C8B89A",
                opacity=0.30,
                name="table plane",
                showlegend=col == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

    for trace, col in _record_traces(records[0], fps):
        fig.add_trace(trace, row=1, col=col)

    dynamic_indices = list(range(2, len(fig.data)))
    frames = []
    for frame_idx, record in enumerate(records):
        dynamic = []
        for trace, col in _record_traces(record, fps):
            trace.scene = "scene" if col == 1 else "scene2"
            dynamic.append(trace)
        frames.append(
            go.Frame(
                data=dynamic,
                traces=dynamic_indices,
                name=str(frame_idx),
                layout=dict(title=dict(text=_title(record))),
            )
        )
    fig.frames = frames

    camera = dict(eye=dict(x=1.45, y=1.45, z=1.05), up=dict(x=0, y=0, z=1))
    scene = dict(
        xaxis=dict(title="x (m)", range=[float(lo[0]), float(hi[0])]),
        yaxis=dict(title="y (m)", range=[float(lo[1]), float(hi[1])]),
        zaxis=dict(title="z (m)", range=[float(lo[2]), float(hi[2])]),
        aspectmode="cube",
        bgcolor="#FAFAFB",
        camera=camera,
        uirevision="fast-flow-comparison-camera",
    )
    fig.update_layout(
        title=dict(text=_title(records[0]), x=0.01, xanchor="left"),
        scene=scene,
        scene2=dict(scene),
        paper_bgcolor="#FFFFFF",
        height=760,
        margin=dict(l=8, r=8, b=45, t=105),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#D4D4D8",
            borderwidth=1,
            font=dict(size=10),
            orientation="h",
            x=0.01,
            y=0.01,
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
        sliders=[
            dict(
                active=0,
                y=-0.02,
                x=0.04,
                len=0.92,
                pad=dict(t=42),
                currentvalue=dict(prefix="anchor ", font=dict(size=12)),
                steps=[
                    dict(
                        method="animate",
                        label=f"ep{record['episode']}:{record['frame']}",
                        value=str(index),
                        args=[
                            [str(index)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for index, record in enumerate(records)
                ],
            )
        ],
    )
    return fig


def write_html(fig, html_path):
    """Write the comparison and mirror orbit/zoom changes between both 3-D scenes."""
    sync_script = """
(function() {
  var plot = document.getElementById('decoder-comparison-plot');
  var syncing = false;
  plot.on('plotly_relayout', function(update) {
    if (syncing) return;
    var mirrored = {};
    Object.keys(update).forEach(function(key) {
      if (key.indexOf('scene.camera') === 0) {
        mirrored[key.replace('scene.camera', 'scene2.camera')] = update[key];
      } else if (key.indexOf('scene2.camera') === 0) {
        mirrored[key.replace('scene2.camera', 'scene.camera')] = update[key];
      }
    });
    if (Object.keys(mirrored).length) {
      syncing = true;
      Plotly.relayout(plot, mirrored).then(function() { syncing = false; });
    }
  });
})();
"""
    fig.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
        auto_play=False,
        div_id="decoder-comparison-plot",
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
        post_script=sync_script,
    )

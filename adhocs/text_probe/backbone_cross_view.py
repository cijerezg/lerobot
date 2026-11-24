"""Viewer of a backbone crossing (``backbone_cross.py``): numbers per cell and the chunks of one frame in 3-D.

    uv run python lerobot/adhocs/text_probe/backbone_cross_view.py outputs/analysis/backbone_cross_2026-11-16/chunks.pt

Open http://localhost:8771. No model is loaded; the file can be read while the run is still writing cells.

Chunks are drawn the way chat.py's Act tab and the action inspector (probes/action_trace_probe) draw them:
deployed low-pass on the predicted chunk, reBot kinematics, tip paths in cm next to the arm as it stands and the
recorded chunk (not filtered). Tip numbers are on the filtered chunk, joint numbers on the chunk as predicted.

A number of ``METRICS`` is one value per (cell, frame, seed); the page averages the seeds of a frame or takes
the best one, then the frames picked by the episode and step filters. A new quantity is one more entry there.
"""
from __future__ import annotations

import argparse
import io
import json

import numpy as np
import torch
from flask import Flask, jsonify, send_file

from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.action_smoothing import apply_butterworth_filter

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>backbone crossing</title>
<script src="plotly.js"></script>
<style>
  :root { color-scheme: dark; --bg: #15171a; --panel: #1d2024; --line: #30353c; --text: #e6e8eb; --dim: #8b929c; --accent: #4c9aff; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, sans-serif; }
  #layout { display: grid; grid-template-columns: 300px 1fr; min-height: 100vh; }
  #side { background: var(--panel); border-right: 1px solid var(--line); padding: 14px; position: sticky; top: 0; height: 100vh; overflow-y: auto; }
  #main { padding: 18px 22px 60px; min-width: 0; }
  h2 { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--dim); margin: 18px 0 8px; font-weight: 600; }
  h2:first-child { margin-top: 0; }
  h3 { font-size: 14px; margin: 0 0 2px; font-weight: 600; }
  p { margin: 0 0 8px; color: var(--dim); font-size: 12px; }
  select, button { background: var(--bg); color: var(--text); border: 1px solid var(--line); border-radius: 8px; font: inherit; padding: 6px 10px; }
  select { width: 100%; } button { cursor: pointer; }
  .seg { display: flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .seg div { flex: 1; text-align: center; padding: 6px 8px; cursor: pointer; color: var(--dim); border-right: 1px solid var(--line); user-select: none; }
  .seg div:last-child { border-right: 0; } .seg div.on { background: #4c9aff26; color: var(--text); }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; margin-bottom: 18px; padding: 14px; }
  table { border-collapse: collapse; width: 100%; }
  th { text-align: left; font-size: 12px; color: var(--dim); font-weight: 600; padding: 6px 12px; }
  td { border-top: 1px solid var(--line); padding: 8px 12px; font-variant-numeric: tabular-nums; }
  td b { font-size: 18px; font-weight: 600; } td span { color: var(--dim); font-size: 12px; margin-left: 8px; }
  .swatch { display: inline-block; width: 22px; height: 0; border-top: 3px solid; vertical-align: middle; margin-right: 6px; }
  .grid { display: grid; gap: 10px; } .scene { height: 330px; } .wide { height: 300px; }
  .cell { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .cell header { padding: 5px 10px; font-size: 12px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; gap: 8px; }
  .cell header span { color: var(--dim); font-variant-numeric: tabular-nums; }
  .shots { display: flex; gap: 10px; align-items: flex-start; margin: 8px 0 12px; }
  .shots img { height: 190px; border-radius: 6px; }
  .shots div { font-size: 13px; } .shots div b { font-weight: 600; }
  .nav { display: flex; gap: 8px; align-items: center; }
</style>
<div id="layout">
  <aside id="side">
    <h2>Quantity</h2>
    <select id="metric"></select>
    <h2>Seeds of a frame</h2>
    <div class="seg" id="reduce"><div class="on" data-v="mean">mean</div><div data-v="best">best seed</div></div>
    <h2>Episode</h2>
    <select id="episode"></select>
    <h2>Step</h2>
    <select id="subtask"></select>
    <h2>Frame</h2>
    <div class="nav"><button id="prev">&larr;</button><button id="next">&rarr;</button><span id="where"></span></div>
    <p style="margin-top: 8px">Click a point of the per-frame chart, or use the arrow keys.</p>
    <h2>File</h2>
    <p id="about"></p>
  </aside>
  <main id="main">
    <div class="card">
      <h3 id="title"></h3>
      <p id="note"></p>
      <table id="table"></table>
    </div>
    <div class="card"><h3>Per frame</h3><p>One point per frame and cell, frames in recording order; vertical lines separate episodes.</p><div class="wide" id="frames"></div></div>
    <div class="card"><h3>Along the chunk</h3><p>Tip distance at every step of the chunk (seeds averaged), mean over the frames picked on the left.</p>
      <div class="seg" id="along" style="max-width: 460px"></div><div class="wide" id="horizon"></div></div>
    <div class="card">
      <h3 id="frame_title"></h3>
      <div class="shots" id="shots"></div>
      <p>Each scene: grey = arm now, white dotted = recorded chunk, colour = the cell's chunks (seed 0 strong, other seeds faint). Turning one scene turns all.</p>
      <div class="grid" id="scenes"></div>
      <h3 style="margin-top: 16px">All cells in one scene (seed 0)</h3>
      <div style="height: 480px" id="overlay"></div>
      <h3 style="margin-top: 16px">Joints over the chunk</h3>
      <p>Chunk as predicted (not filtered), all seeds; white dotted = recorded.</p>
      <div style="height: 520px" id="joints"></div>
    </div>
  </main>
</div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const COLOURS = ["#4c9aff", "#f0a04b", "#3ecf8e", "#c084fc", "#f472b6", "#22d3ee"], DASHES = ["dash", "solid", "dot", "dashdot"];
const DARK = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#15171a", font: { color: "#e6e8eb", size: 11 }, margin: { l: 52, r: 12, t: 10, b: 40 } };
const ax = (title, extra) => ({ title: { text: title, font: { size: 11 } }, gridcolor: "#30353c", zerolinecolor: "#30353c", linecolor: "#30353c", ...extra });
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
const median = values => { const s = [...values].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
let D, metric, reduce = "mean", along, shown = [], current = 0;

// colour = language model, dash = vision path
const look = c => ({ colour: COLOURS[D.languages.indexOf(D.cells[c][1]) % COLOURS.length], dash: DASHES[D.visions.indexOf(D.cells[c][0]) % DASHES.length] });
const cellName = c => `vision ${D.cells[c][0]} · language model ${D.cells[c][1]}`;
const cellOf = (v, l) => D.cells.findIndex(c => c[0] === v && c[1] === l);
const value = (c, f) => { const seeds = D.metrics[metric][c][f]; return reduce === "mean" ? mean(seeds) : Math.min(...seeds); };

function summary() {
  shown = D.frames.map((_, f) => f).filter(f => ($("episode").value === "" || D.frames[f].episode === +$("episode").value) &&
    ($("subtask").value === "" || D.frames[f].subtask === $("subtask").value));
  $("title").textContent = metric;
  $("note").textContent = `${shown.length} frames x ${D.seeds} seeds per cell · per frame: ${reduce === "mean" ? "mean of the seeds" : "best seed (lowest value)"} · large = mean over the frames, small = median` +
    (D.reference < 0 ? "" : ` · reference cell = ${cellName(D.reference)}`);
  $("table").innerHTML = `<tr><th>vision path \\ language model</th>${D.languages.map(l => `<th>${esc(l)}</th>`).join("")}</tr>` + D.visions.map(v =>
    `<tr><th>${esc(v)}</th>${D.languages.map(l => { const c = cellOf(v, l); if (c < 0 || !shown.length) return "<td>not in the file</td>";
      const values = shown.map(f => value(c, f)), { colour, dash } = look(c);
      return `<td><i class="swatch" style="border-top-color: ${colour}; border-top-style: ${dash === "solid" ? "solid" : "dashed"}"></i><b>${mean(values).toFixed(2)}</b><span>${median(values).toFixed(2)}</span></td>`; }).join("")}</tr>`).join("");

  const breaks = shown.map((f, k) => k && D.frames[f].episode !== D.frames[shown[k - 1]].episode ? k - 0.5 : null).filter(x => x !== null);
  Plotly.react("frames", D.cells.map((_, c) => { const { colour, dash } = look(c);
    return { x: shown.map((_, k) => k), y: shown.map(f => value(c, f)), customdata: shown, name: cellName(c), mode: "lines+markers",
      line: { color: colour, width: 1.2, dash }, marker: { size: 5, color: colour },
      text: shown.map(f => `episode ${D.frames[f].episode} · frame ${D.frames[f].frame}<br>${esc(D.frames[f].subtask || D.frames[f].task)}`),
      hovertemplate: `%{y:.2f}<br>%{text}<extra>${esc(cellName(c))}</extra>` }; }),
    { ...DARK, xaxis: ax("frame (of those picked)"), yaxis: ax(metric), legend: { orientation: "h", y: -0.25 }, margin: { ...DARK.margin, b: 80 },
      shapes: [...breaks.map(x => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: "#30353c", width: 1 } })),
        ...(shown.includes(current) ? [{ type: "line", x0: shown.indexOf(current), x1: shown.indexOf(current), yref: "paper", y0: 0, y1: 1, line: { color: "#e6e8eb", width: 1, dash: "dot" } }] : [])] },
    { displaylogo: false, responsive: true });
  $("frames").removeAllListeners?.("plotly_click");
  $("frames").on("plotly_click", event => show(event.points[0].customdata));

  Plotly.react("horizon", shown.length ? D.cells.map((_, c) => { const { colour, dash } = look(c), rows = shown.map(f => D.horizon[along][c][f]);
    return { x: rows[0].map((_, t) => t), y: rows[0].map((_, t) => mean(rows.map(r => r[t]))), name: cellName(c), mode: "lines", line: { color: colour, width: 2, dash } }; }) : [],
    { ...DARK, xaxis: ax(`step of the chunk (${D.fps} per second)`), yaxis: ax(along), showlegend: false }, { displaylogo: false, responsive: true });
}

function path3d(points, name, colour, extra) {
  return { type: "scatter3d", x: points.map(p => p[0]), y: points.map(p => p[1]), z: points.map(p => p[2]), name,
    mode: "lines+markers", line: { color: colour, width: 6 }, marker: { size: 2.5, color: colour },
    hovertemplate: `<b>${esc(name)}</b><br>step %{pointNumber}<br>x %{x:.1f} · y %{y:.1f} · z %{z:.1f} cm<extra></extra>`, ...extra };
}
let turning = false, camera = { eye: { x: 1.4, y: 1.4, z: 1.0 } };
function turn(source) {
  source.on("plotly_relayout", event => {
    if (turning || !event["scene.camera"]) return;
    turning = true; camera = event["scene.camera"];
    Promise.all([...document.querySelectorAll(".scene, #overlay")].filter(p => p !== source).map(p => Plotly.relayout(p, { "scene.camera": camera }))).then(() => { turning = false; });
  });
}

async function show(f) {
  current = f;
  const frame = D.frames[f], out = await (await fetch(`frame/${f}`)).json();
  $("where").textContent = `${shown.indexOf(f) + 1} of ${shown.length}`;
  $("frame_title").textContent = `Episode ${frame.episode}, frame ${frame.frame}`;
  $("shots").innerHTML = D.cameras.map(c => `<img src="image/${f}/${c}" title="${c}">`).join("") +
    `<div><b>task</b> ${esc(frame.task)}<br><b>step</b> ${esc(frame.subtask || "(none)")}<br><b>recorded tip path</b> ${out.recorded_length.toFixed(1)} cm in ${out.recorded.length} steps</div>`;
  const all = [out.arm, out.recorded, ...out.tips.flat()].flat(), range = k => { const v = all.map(p => p[k]); return [Math.min(...v) - 2, Math.max(...v) + 2]; };
  const axis = (title, k) => ({ title, range: range(k), color: "#8b929c", gridcolor: "#30353c", backgroundcolor: "#1d2024", showbackground: true });
  const layout = legend => ({ margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#e6e8eb", size: 11 }, showlegend: legend,
    legend: { orientation: "h", y: 0, yanchor: "top", font: { size: 11 } },
    scene: { aspectmode: "data", xaxis: axis("x (cm)", 0), yaxis: axis("y (cm)", 1), zaxis: axis("z (cm)", 2), camera } });
  const fixed = () => [
    path3d(out.arm, "arm now", "#8b929c", { line: { color: "#8b929c", width: 9 }, marker: { size: 4, color: "#8b929c" }, hoverinfo: "name", hovertemplate: null }),
    path3d(out.recorded, "recorded", "#d4d7dc", { line: { color: "#d4d7dc", width: 3, dash: "dot" }, marker: { size: 1.5, color: "#d4d7dc" } }),
  ];
  $("scenes").style.gridTemplateColumns = `repeat(${D.languages.length}, 1fr)`;
  $("scenes").innerHTML = D.visions.map(v => D.languages.map(l => { const c = cellOf(v, l);
    return c < 0 ? `<div class="cell"><header>not in the file</header></div>` :
      `<div class="cell"><header style="color: ${look(c).colour}">${esc(cellName(c))}<span>${value(c, f).toFixed(2)}</span></header><div class="scene" data-c="${c}"></div></div>`; }).join("")).join("");
  for (const plot of document.querySelectorAll(".scene")) {
    const c = +plot.dataset.c, { colour } = look(c);
    await Plotly.newPlot(plot, [...fixed(), ...out.tips[c].map((points, seed) => path3d(points, `seed ${seed}`, colour, { opacity: seed ? 0.45 : 1 }))], layout(false), { displaylogo: false, responsive: true });
    turn(plot);
  }
  await Plotly.newPlot("overlay", [...fixed(), ...D.cells.map((_, c) => { const { colour, dash } = look(c);
    return path3d(out.tips[c][0], cellName(c), colour, { line: { color: colour, width: 5, dash } }); })], layout(true), { displaylogo: false, responsive: true });
  turn($("overlay"));

  const traces = [], columns = 4, rows = Math.ceil(D.joints.length / columns), steps = out.recorded_joints.map((_, t) => t);
  for (const [j, name] of D.joints.entries()) {
    const axes = { xaxis: `x${j + 1}`, yaxis: `y${j + 1}` };
    traces.push({ x: steps, y: out.recorded_joints.map(q => q[j]), name: "recorded", mode: "lines", line: { color: "#d4d7dc", width: 2, dash: "dot" }, showlegend: j === 0, legendgroup: "recorded", ...axes });
    for (const c of D.cells.keys()) for (const [seed, chunk] of out.joints[c].entries()) { const { colour, dash } = look(c);
      traces.push({ x: steps, y: chunk.map(q => q[j]), name: cellName(c), mode: "lines", line: { color: colour, width: 1.6, dash }, opacity: seed ? 0.35 : 1,
        showlegend: j === 0 && seed === 0, legendgroup: `c${c}`, ...axes }); }
  }
  const jointLayout = { ...DARK, grid: { rows, columns, pattern: "independent" }, legend: { orientation: "h", y: -0.08 }, margin: { l: 40, r: 10, t: 24, b: 60 },
    annotations: D.joints.map((name, j) => ({ text: `${name} (deg)`, showarrow: false, xref: `x${j + 1} domain`, yref: `y${j + 1} domain`, x: 0, y: 1.12, font: { size: 11, color: "#8b929c" } })) };
  for (const j of D.joints.keys()) { jointLayout[`xaxis${j + 1}`] = ax(""); jointLayout[`yaxis${j + 1}`] = ax(""); }
  Plotly.react("joints", traces, jointLayout, { displaylogo: false, responsive: true });
  summary();
}

function segment(id, choices, pick) {
  $(id).innerHTML = choices.map((c, k) => `<div class="${k ? "" : "on"}" data-v="${esc(c)}">${esc(c)}</div>`).join("");
  for (const div of $(id).children) div.onclick = () => { for (const other of $(id).children) other.classList.toggle("on", other === div); pick(div.dataset.v); };
}
const step = by => { const k = shown.indexOf(current); if (shown.length) show(shown[Math.max(0, Math.min(shown.length - 1, (k < 0 ? 0 : k + by)))]); };

(async () => {
  D = await (await fetch("data")).json();
  const names = Object.keys(D.metrics);
  metric = names[0]; along = Object.keys(D.horizon)[0];
  $("metric").innerHTML = names.map(n => `<option>${esc(n)}</option>`).join("");
  $("metric").onchange = () => { metric = $("metric").value; show(current); };
  for (const div of $("reduce").children) div.onclick = () => { for (const other of $("reduce").children) other.classList.toggle("on", other === div); reduce = div.dataset.v; show(current); };
  segment("along", Object.keys(D.horizon), v => { along = v; summary(); });
  $("episode").innerHTML = `<option value="">all episodes</option>` + [...new Set(D.frames.map(f => f.episode))].map(e => `<option value="${e}">episode ${e}</option>`).join("");
  $("subtask").innerHTML = `<option value="">all steps</option>` + [...new Set(D.frames.map(f => f.subtask))].sort().map(s => `<option value="${esc(s)}">${esc(s || "(none)")}</option>`).join("");
  $("episode").onchange = $("subtask").onchange = () => { summary(); if (shown.length && !shown.includes(current)) show(shown[0]); };
  $("prev").onclick = () => step(-1); $("next").onclick = () => step(1);
  document.addEventListener("keydown", e => { if (e.target.tagName === "SELECT") return; if (e.key === "ArrowLeft") step(-1); if (e.key === "ArrowRight") step(1); });
  $("about").innerHTML = `${esc(D.path)}<br>action expert: ${esc(D.expert)}<br>data: ${esc(D.root)}<br>depth tokens: ${D.depth ? "on" : "off"}<br>${D.cells.length} cells, ${D.frames.length} frames, ${D.seeds} seeds`;
  summary();
  show(shown[0]);
})();
</script>
"""


def load(path: str) -> dict:
    data = torch.load(path, weights_only=False)
    cells = list(data["chunks"])
    recorded = data["recorded"].double().numpy()  # [frames, steps, joints]
    joints, seeds = recorded.shape[-1], data["chunks"][cells[0]].shape[1]
    # [cells, frames, seeds, steps, joints]; a chunk wider than the recorded action keeps the recorded joints
    raw = np.stack([data["chunks"][cell].double().numpy() for cell in cells])[..., :joints]
    smooth = np.moveaxis(apply_butterworth_filter(np.moveaxis(raw, 3, 0)), 0, 3)
    kinematics = RebotKinematics()

    def tip(q):  # [..., joints] degrees -> [..., 3] cm
        return kinematics.ee_path(q.reshape(-1, joints)).reshape(*q.shape[:-1], 3) * 100

    def length(path):
        return np.linalg.norm(np.diff(path, axis=-2), axis=-1).sum(axis=-1)

    tips, recorded_tip = tip(smooth), tip(recorded)
    to_recorded = np.linalg.norm(tips - recorded_tip[None, :, None], axis=-1)  # [cells, frames, seeds, steps]
    between = np.linalg.norm(tips[:, :, :, None] - tips[:, :, None], axis=-1).mean(axis=-1)  # [cells, frames, seeds, seeds]
    norm = np.stack([data["chunks_norm"][cell].double().numpy() for cell in cells])[..., :joints]
    recorded_norm = data["recorded_norm"].double().numpy()[..., :joints]

    # one value per (cell, frame, seed)
    METRICS = {
        "tip distance to the recorded chunk, mean over the steps (cm)": to_recorded.mean(axis=-1),
        "tip distance to the recorded chunk, last step (cm)": to_recorded[..., -1],
        "joint RMSE to the recorded chunk (deg)": np.sqrt(((raw - recorded[None, :, None]) ** 2).mean(axis=(-1, -2))),
        "squared error to the recorded chunk, normalized action space": ((norm - recorded_norm[None, :, None]) ** 2).mean(axis=(-1, -2)),
        "gripper: mean absolute difference to the recorded chunk (deg)": np.abs(raw[..., -1] - recorded[None, :, None, :, -1]).mean(axis=-1),
        "tip path length minus the recorded one (cm)": length(tips) - length(recorded_tip)[None, :, None],
        "first step: tip distance to the arm now (cm)": np.linalg.norm(tips[..., 0, :] - tip(data["state"].double().numpy()[:, :joints])[None, :, None], axis=-1),
        "seed variation: mean tip distance to the other seeds (cm)": between.sum(axis=-1) / max(seeds - 1, 1),
    }
    horizon = {"tip distance to the recorded chunk (cm)": to_recorded.mean(axis=2)}  # [cells, frames, steps]
    reference = (data["expert"], data["expert"])
    if reference in cells:
        at = cells.index(reference)
        to_reference = np.linalg.norm(tips - tips[at][None], axis=-1)
        METRICS["tip distance to the reference cell, same seed (cm)"] = to_reference.mean(axis=-1)
        METRICS["joint RMSE to the reference cell, same seed (deg)"] = np.sqrt(((raw - raw[at][None]) ** 2).mean(axis=(-1, -2)))
        horizon["tip distance to the reference cell, same seed (cm)"] = to_reference.mean(axis=2)

    summary = {
        "path": path, "expert": data["expert"], "root": data["root"], "depth": data["depth"], "fps": data["fps"], "seeds": seeds,
        "cells": cells, "visions": data["visions"], "languages": data["languages"], "joints": data["joints"][:joints],
        "reference": cells.index(reference) if reference in cells else -1, "cameras": list(data["images"][0]),
        "frames": [{"episode": int(e), "frame": int(f), "task": t, "subtask": s} for e, f, t, s in zip(data["episode"], data["frame"], data["task"], data["subtask"])],
        "metrics": {name: np.round(value, 3).tolist() for name, value in METRICS.items()},
        "horizon": {name: np.round(value, 2).tolist() for name, value in horizon.items()},
    }
    arm = kinematics.link_origins(data["state"].double().numpy()[:, :joints]) * 100  # [frames, links, 3]
    return {"summary": summary, "images": data["images"], "arm": arm, "tips": tips, "recorded_tip": recorded_tip, "raw": raw, "recorded": recorded,
            "recorded_length": length(recorded_tip)}


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("path", nargs="?", default="outputs/analysis/backbone_cross_2026-11-16/chunks.pt")
    args.add_argument("--port", type=int, default=8771)
    args = args.parse_args()
    loaded = load(args.path)
    app = Flask(__name__)

    @app.get("/")
    def page():
        return PAGE

    @app.get("/plotly.js")
    def plotly():
        from plotly.offline import get_plotlyjs  # the copy shipped with the package: no CDN needed

        return app.response_class(get_plotlyjs(), mimetype="application/javascript")

    @app.get("/data")
    def data():
        return app.response_class(json.dumps(loaded["summary"]), mimetype="application/json")

    @app.get("/frame/<int:index>")
    def frame(index):
        return jsonify(
            arm=np.round(loaded["arm"][index], 2).tolist(), recorded=np.round(loaded["recorded_tip"][index], 2).tolist(),
            tips=np.round(loaded["tips"][:, index], 2).tolist(), joints=np.round(loaded["raw"][:, index], 2).tolist(),
            recorded_joints=np.round(loaded["recorded"][index], 2).tolist(), recorded_length=float(loaded["recorded_length"][index]),
        )

    @app.get("/image/<int:index>/<camera>")
    def image(index, camera):
        return send_file(io.BytesIO(loaded["images"][index][camera]), mimetype="image/jpeg")

    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

"""Viewer of a step swap (``subtask_swap.py``): numbers per model and kind of step text, and one frame in 3-D.

    uv run python lerobot/adhocs/text_probe/subtask_swap_view.py outputs/analysis/subtask_swap_2026-11-16/chunks.pt

Open http://localhost:8772. No model is loaded; the file can be read while the run is still writing models.

Chunks are drawn as in ``backbone_cross_view.py``: deployed low-pass on the predicted chunk, reBot kinematics,
tip paths in cm. Tip numbers are on the filtered chunk, the gripper on the chunk as predicted.

A number of ``METRICS`` is one value per (model, step text of a frame, seed). The page averages the seeds, then
the texts of one kind within a frame, then the frames picked on the left. A new quantity is one more entry there.
"""
from __future__ import annotations

import argparse
import io
import json

import numpy as np
import torch
from flask import Flask, send_file

from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.action_smoothing import apply_butterworth_filter

KINDS = {"own": "own step", "home": "return to home", "later": "later step of the episode", "earlier": "earlier step of the episode", "none": "no step"}

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>step swap</title>
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
  .grid { display: grid; gap: 10px; } .scene { height: 380px; } .small { height: 230px; } .wide { height: 300px; }
  .cell { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .cell header { padding: 5px 10px; font-size: 12px; border-bottom: 1px solid var(--line); }
  .shots { display: flex; gap: 10px; align-items: flex-start; margin: 8px 0 12px; }
  .shots img { height: 190px; border-radius: 6px; }
  .shots div { font-size: 13px; } .shots div b { font-weight: 600; }
  .nav { display: flex; gap: 8px; align-items: center; }
</style>
<div id="layout">
  <aside id="side">
    <h2>Quantity</h2>
    <select id="metric"></select>
    <h2>Episode</h2>
    <select id="episode"></select>
    <h2>Own step of the frame</h2>
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
    <div class="card"><h3>Kind of step text for the charts</h3><div class="seg" id="kind" style="max-width: 760px; margin-top: 8px"></div></div>
    <div class="card"><h3>Distance to the first model's chunk: under the own step against under the swapped text</h3>
      <p>Tip distance to the first model's chunk predicted under the same text and seed, mean over the steps (cm), seeds averaged. One point per frame and model.
        On the diagonal the model is as far from the first model under the swapped text as under the own step; above it, it follows the swapped text less than the first model does.
        Click a point to open the frame.</p>
      <div style="height: 460px" id="scatter"></div></div>
    <div class="card"><h3>Per frame</h3><p>One point per frame and model; frames in recording order, vertical lines separate episodes.</p>
      <div class="wide" id="frames"></div></div>
    <div class="card"><h3>Along the chunk</h3><p>Tip distance to the own-step chunk of the same seed at every step of the chunk, same kind of step text, mean over the frames picked on the left.
      Dotted = the model's seed variation under the own step (mean tip distance between two seeds).</p><div class="wide" id="horizon"></div></div>
    <div class="card">
      <h3 id="frame_title"></h3>
      <div class="shots" id="shots"></div>
      <p>One scene per model: grey = arm now, white dotted = recorded chunk, one colour per step text (seed 0 strong, other seeds faint),
        diamond = where that step ends in the recording. Turning one scene turns all.</p>
      <div class="grid" id="scenes"></div>
      <h3 style="margin-top: 16px">Gripper over the chunk (deg)</h3>
      <p>Chunk as predicted (not filtered), all seeds; white dotted = recorded.</p>
      <div class="grid" id="grippers"></div>
    </div>
  </main>
</div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const COLOURS = ["#4c9aff", "#f0a04b", "#3ecf8e", "#c084fc", "#f472b6", "#22d3ee", "#facc15"];
const KIND_COLOURS = { own: "#4c9aff", home: "#f0a04b", none: "#8b929c" }, OTHER = ["#3ecf8e", "#c084fc", "#f472b6", "#22d3ee", "#facc15"];
const DARK = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#15171a", font: { color: "#e6e8eb", size: 11 }, margin: { l: 52, r: 12, t: 10, b: 40 } };
const ax = (title, extra) => ({ title: { text: title, font: { size: 11 } }, gridcolor: "#30353c", zerolinecolor: "#30353c", linecolor: "#30353c", ...extra });
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
const median = values => { const s = [...values].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
let D, metric, kind, shown = [], current = 0;

// mean over the frame's step texts of one kind (seeds already averaged); null when the frame has none or the number does not exist
function value(m, f, k, name = metric) {
  const values = D.frames[f].rows.filter((_, i) => D.frames[f].kinds[i] === k).map(r => D.metrics[name][m][r]).filter(v => v !== null);
  return values.length ? mean(values) : null;
}
// one colour per step text of a frame: fixed for own / home / none, the others in order
function colours(frame) { let other = 0; return frame.kinds.map(k => KIND_COLOURS[k] || OTHER[other++ % OTHER.length]); }

function summary() {
  shown = D.frames.map((_, f) => f).filter(f => ($("episode").value === "" || D.frames[f].episode === +$("episode").value) &&
    ($("subtask").value === "" || D.frames[f].steps[0] === $("subtask").value));
  $("title").textContent = metric;
  $("note").textContent = `${shown.length} frames x ${D.seeds} seeds · seeds averaged, then the step texts of one kind within a frame · large = mean over the frames, small = median and number of frames`;
  $("table").innerHTML = `<tr><th>model \\ step text</th>${D.kinds.map(k => `<th>${esc(D.kind_names[k])}</th>`).join("")}</tr>` + D.models.map((name, m) =>
    `<tr><th><i class="swatch" style="border-top-color: ${COLOURS[m]}"></i>${esc(name)}</th>${D.kinds.map(k => { const values = shown.map(f => value(m, f, k)).filter(v => v !== null);
      return values.length ? `<td><b>${mean(values).toFixed(2)}</b><span>${median(values).toFixed(2)} · ${values.length}</span></td>` : "<td></td>"; }).join("")}</tr>`).join("");

  const pairs = D.models.map((_, m) => shown.filter(f => value(m, f, kind, D.error) !== null).map(f => [value(m, f, "own", D.error), value(m, f, kind, D.error), f]));
  const top = Math.max(1, ...pairs.flat().flatMap(p => [p[0], p[1]])) * 1.05;
  Plotly.react("scatter", D.models.map((name, m) => ({ x: pairs[m].map(p => p[0]), y: pairs[m].map(p => p[1]), customdata: pairs[m].map(p => p[2]), mode: "markers",
      name: pairs[m].length ? `${name} · mean ${mean(pairs[m].map(p => p[0])).toFixed(2)} → ${mean(pairs[m].map(p => p[1])).toFixed(2)} cm · above the diagonal ${pairs[m].filter(p => p[1] > p[0]).length} of ${pairs[m].length}` : name,
      marker: { size: 7, color: COLOURS[m], opacity: 0.75 },
      text: pairs[m].map(p => `episode ${D.frames[p[2]].episode} · frame ${D.frames[p[2]].frame}<br>own step: ${esc(D.frames[p[2]].steps[0])}`),
      hovertemplate: `own %{x:.2f} · swapped %{y:.2f} cm<br>%{text}<extra>${esc(name)}</extra>` })).slice(1),
    { ...DARK, xaxis: ax(`distance to ${D.models[0]}, both under the own step (cm)`, { range: [0, top] }),
      yaxis: ax(`distance to ${D.models[0]}, both under: ${D.kind_names[kind]} (cm)`, { range: [0, top], scaleanchor: "x" }),
      legend: { orientation: "h", y: -0.18 }, margin: { ...DARK.margin, b: 90 },
      shapes: [{ type: "line", x0: 0, y0: 0, x1: top, y1: top, line: { color: "#8b929c", width: 1, dash: "dot" } }] }, { displaylogo: false, responsive: true });
  $("scatter").removeAllListeners?.("plotly_click");
  $("scatter").on("plotly_click", event => show(event.points[0].customdata));

  const breaks = shown.map((f, i) => i && D.frames[f].episode !== D.frames[shown[i - 1]].episode ? i - 0.5 : null).filter(x => x !== null);
  Plotly.react("frames", D.models.map((name, m) => ({ x: shown.map((_, i) => i), y: shown.map(f => value(m, f, kind)), customdata: shown, name, mode: "lines+markers",
      line: { color: COLOURS[m], width: 1.2 }, marker: { size: 5, color: COLOURS[m] }, connectgaps: false,
      text: shown.map(f => `episode ${D.frames[f].episode} · frame ${D.frames[f].frame}<br>own step: ${esc(D.frames[f].steps[0])}`),
      hovertemplate: `%{y:.2f}<br>%{text}<extra>${esc(name)}</extra>` })),
    { ...DARK, xaxis: ax("frame (of those picked)"), yaxis: ax(metric), legend: { orientation: "h", y: -0.25 }, margin: { ...DARK.margin, b: 80 },
      shapes: [...breaks.map(x => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: "#30353c", width: 1 } })),
        ...(shown.includes(current) ? [{ type: "line", x0: shown.indexOf(current), x1: shown.indexOf(current), yref: "paper", y0: 0, y1: 1, line: { color: "#e6e8eb", width: 1, dash: "dot" } }] : [])] },
    { displaylogo: false, responsive: true });
  $("frames").removeAllListeners?.("plotly_click");
  $("frames").on("plotly_click", event => show(event.points[0].customdata));

  const along = (table, m, k) => { const rows = shown.flatMap(f => D.frames[f].rows.filter((_, i) => D.frames[f].kinds[i] === k)).map(r => table[m][r]);
    return rows.length ? rows[0].map((_, t) => mean(rows.map(r => r[t]))) : []; };
  Plotly.react("horizon", D.models.flatMap((name, m) => [
      { y: along(D.horizon, m, kind), name, mode: "lines", line: { color: COLOURS[m], width: 2 } },
      { y: along(D.spread, m, "own"), name: `${name}: seed variation`, mode: "lines", line: { color: COLOURS[m], width: 1.5, dash: "dot" }, showlegend: false }]),
    { ...DARK, xaxis: ax(`step of the chunk (${D.fps} per second)`), yaxis: ax("tip distance (cm)"), showlegend: false }, { displaylogo: false, responsive: true });
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
    Promise.all([...document.querySelectorAll(".scene")].filter(p => p !== source).map(p => Plotly.relayout(p, { "scene.camera": camera }))).then(() => { turning = false; });
  });
}

async function show(f) {
  current = f;
  const frame = D.frames[f], out = await (await fetch(`frame/${f}`)).json(), colour = colours(frame);
  const text = i => frame.steps[i] || "(no step)";
  $("where").textContent = `${shown.indexOf(f) + 1} of ${shown.length}`;
  $("frame_title").textContent = `Episode ${frame.episode}, frame ${frame.frame}`;
  $("shots").innerHTML = D.cameras.map(c => `<img src="image/${f}/${c}" title="${c}">`).join("") + `<div><b>task</b> ${esc(frame.task)}<br>` +
    frame.steps.map((_, i) => `<i class="swatch" style="border-top-color: ${colour[i]}"></i>${esc(text(i))} <span style="color: var(--dim)">${esc(D.kind_names[frame.kinds[i]])}</span>`).join("<br>") + "</div>";
  const ends = out.ends.filter(p => p !== null), all = [out.arm, out.recorded, ends, ...out.tips.flat(2)].flat();
  const range = k => { const v = all.map(p => p[k]); return [Math.min(...v) - 2, Math.max(...v) + 2]; };
  const axis = (title, k) => ({ title, range: range(k), color: "#8b929c", gridcolor: "#30353c", backgroundcolor: "#1d2024", showbackground: true });
  const layout = { margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#e6e8eb", size: 11 }, showlegend: false,
    scene: { aspectmode: "data", xaxis: axis("x (cm)", 0), yaxis: axis("y (cm)", 1), zaxis: axis("z (cm)", 2), camera } };
  const columns = `repeat(${D.models.length}, 1fr)`;
  $("scenes").style.gridTemplateColumns = $("grippers").style.gridTemplateColumns = columns;
  $("scenes").innerHTML = D.models.map((name, m) => `<div class="cell"><header style="color: ${COLOURS[m]}">${esc(name)}</header><div class="scene" data-m="${m}"></div></div>`).join("");
  $("grippers").innerHTML = D.models.map((_, m) => `<div class="cell"><div class="small" data-m="${m}"></div></div>`).join("");
  for (const plot of document.querySelectorAll(".scene")) {
    await Plotly.newPlot(plot, [
      path3d(out.arm, "arm now", "#8b929c", { line: { color: "#8b929c", width: 9 }, marker: { size: 4, color: "#8b929c" }, hoverinfo: "name", hovertemplate: null }),
      path3d(out.recorded, "recorded", "#d4d7dc", { line: { color: "#d4d7dc", width: 3, dash: "dot" }, marker: { size: 1.5, color: "#d4d7dc" } }),
      ...out.ends.flatMap((p, i) => p === null ? [] : [{ type: "scatter3d", x: [p[0]], y: [p[1]], z: [p[2]], mode: "markers", name: `end of: ${text(i)}`,
        marker: { size: 6, symbol: "diamond", color: colour[i] }, hovertemplate: `<b>${esc("end of: " + text(i))}</b><extra></extra>` }]),
      ...out.tips[+plot.dataset.m].flatMap((seeds, i) => seeds.map((points, seed) => path3d(points, `${text(i)} · seed ${seed}`, colour[i], { opacity: seed ? 0.4 : 1 }))),
    ], layout, { displaylogo: false, responsive: true });
    turn(plot);
  }
  for (const plot of document.querySelectorAll(".small")) {
    Plotly.newPlot(plot, [
      { y: out.recorded_gripper, mode: "lines", line: { color: "#d4d7dc", width: 2, dash: "dot" } },
      ...out.gripper[+plot.dataset.m].flatMap((seeds, i) => seeds.map((values, seed) => ({ y: values, name: text(i), mode: "lines", line: { color: colour[i], width: 1.6 }, opacity: seed ? 0.4 : 1 }))),
    ], { ...DARK, xaxis: ax("step of the chunk"), yaxis: ax("gripper (deg)"), showlegend: false }, { displaylogo: false, responsive: true });
  }
  summary();
}

const step = by => { const i = shown.indexOf(current); if (shown.length) show(shown[Math.max(0, Math.min(shown.length - 1, (i < 0 ? 0 : i + by)))]); };

(async () => {
  D = await (await fetch("data")).json();
  metric = Object.keys(D.metrics)[0]; kind = D.kinds.includes("home") ? "home" : D.kinds[0];
  $("metric").innerHTML = Object.keys(D.metrics).map(n => `<option>${esc(n)}</option>`).join("");
  $("metric").onchange = () => { metric = $("metric").value; summary(); };
  $("kind").innerHTML = D.kinds.map(k => `<div class="${k === kind ? "on" : ""}" data-v="${k}">${esc(D.kind_names[k])}</div>`).join("");
  for (const div of $("kind").children) div.onclick = () => { for (const other of $("kind").children) other.classList.toggle("on", other === div); kind = div.dataset.v; summary(); };
  $("episode").innerHTML = `<option value="">all episodes</option>` + [...new Set(D.frames.map(f => f.episode))].map(e => `<option value="${e}">episode ${e}</option>`).join("");
  $("subtask").innerHTML = `<option value="">all steps</option>` + [...new Set(D.frames.map(f => f.steps[0]))].sort().map(s => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
  $("episode").onchange = $("subtask").onchange = () => { summary(); if (shown.length && !shown.includes(current)) show(shown[0]); };
  $("prev").onclick = () => step(-1); $("next").onclick = () => step(1);
  document.addEventListener("keydown", e => { if (e.target.tagName === "SELECT") return; if (e.key === "ArrowLeft") step(-1); if (e.key === "ArrowRight") step(1); });
  $("about").innerHTML = `${esc(D.path)}<br>checkpoint: ${esc(D.checkpoint)}<br>data: ${esc(D.root)}<br>depth tokens: ${D.depth ? "on" : "off"}<br>${D.models.length} models, ${D.frames.length} frames, ${D.seeds} seeds`;
  summary();
  show(shown[0]);
})();
</script>
"""


def listed(values: np.ndarray, digits: int) -> list:
    """Nested lists for JSON, nan as null."""
    out = np.round(values, digits).astype(object)
    out[np.isnan(values)] = None
    return out.tolist()


def load(path: str) -> dict:
    data = torch.load(path, weights_only=False)
    models = list(data["chunks"])
    recorded = data["recorded"].double().numpy()  # [frames, steps, joints]
    joints = recorded.shape[-1]
    counts = [len(steps) for steps in data["steps"]]
    first = np.cumsum([0] + counts[:-1])  # a frame's rows start here; its first row is the own step
    of = np.repeat(np.arange(len(counts)), counts)  # row -> frame
    own = first[of]  # row -> the own-step row of its frame
    # [models, rows, seeds, steps, joints], one row per (frame, step text); a chunk wider than the recorded action keeps the recorded joints
    raw = np.stack([torch.cat(data["chunks"][model]).double().numpy() for model in models])[..., :joints]
    smooth = np.moveaxis(apply_butterworth_filter(np.moveaxis(raw, 3, 0)), 0, 3)
    seeds = raw.shape[2]
    kinematics = RebotKinematics()

    def tip(q):  # [..., joints] degrees -> [..., 3] cm
        return kinematics.ee_path(q.reshape(-1, joints)).reshape(*q.shape[:-1], 3) * 100

    def length(path):
        return np.linalg.norm(np.diff(path, axis=-2), axis=-1).sum(axis=-1)

    tips = tip(smooth)
    now = tip(data["state"].double().numpy()[:, :joints])[of]  # [rows, 3]
    ends = torch.cat(data["ends"]).double().numpy()[:, :joints]
    absent = np.isnan(ends).any(axis=-1)
    end = tip(np.nan_to_num(ends))
    end[absent] = np.nan
    change = np.linalg.norm(tips - tips[:, own], axis=-1)  # [models, rows, seeds, steps], same seed
    between = np.linalg.norm(tips[:, :, :, None] - tips[:, :, None], axis=-1)  # [models, rows, seeds, seeds, steps]
    spread = between.sum(axis=(2, 3)) / (seeds * max(seeds - 1, 1))  # [models, rows, steps]
    to_end = np.linalg.norm(tips[..., -1, :] - end[None, :, None], axis=-1)  # [models, rows, seeds]
    to_own_end = np.linalg.norm(tips[..., -1, :] - end[own][None, :, None], axis=-1)

    # one value per (model, row, seed)
    METRICS = {
        "tip distance to the own-step chunk, same seed, mean over the steps (cm)": change.mean(axis=-1),
        "the same, divided by the seed variation under the own step": change.mean(axis=-1) / spread.mean(axis=-1)[:, own, None],
        "tip distance to the own-step chunk, same seed, last step (cm)": change[..., -1],
        "approach to where the text's step ends: tip distance now minus at the last step (cm)": np.linalg.norm(now - end, axis=-1)[None, :, None] - to_end,
        "approach to where the own step ends: tip distance now minus at the last step (cm)": np.linalg.norm(now - end[own], axis=-1)[None, :, None] - to_own_end,
        f"tip distance to the chunk of {models[0]} under the same text, same seed, mean over the steps (cm)": np.linalg.norm(tips - tips[:1], axis=-1).mean(axis=-1),
        "tip distance to the recorded chunk, mean over the steps (cm)": np.linalg.norm(tips - tip(recorded)[None, of, None], axis=-1).mean(axis=-1),
        "tip path length (cm)": length(tips),
        "gripper minus the own-step chunk's, same seed, mean over the steps (deg)": (raw[..., -1] - raw[:, own][..., -1]).mean(axis=-1),
        "seed variation: mean tip distance between two seeds (cm)": np.repeat(spread.mean(axis=-1)[:, :, None], seeds, axis=2),
    }
    kinds = [kind for kind in KINDS if any(kind in frame for frame in data["kinds"])]
    summary = {
        "path": path, "checkpoint": data["checkpoint"], "root": data["root"], "depth": data["depth"], "fps": data["fps"], "seeds": seeds,
        "models": models, "kinds": kinds, "kind_names": KINDS, "cameras": list(data["images"][0]),
        "frames": [{"episode": int(e), "frame": int(f), "task": t, "kinds": k, "steps": s, "rows": list(range(int(a), int(a) + len(s)))}
                   for e, f, t, k, s, a in zip(data["episode"], data["frame"], data["task"], data["kinds"], data["steps"], first)],
        "metrics": {name: listed(value.mean(axis=-1), 3) for name, value in METRICS.items()},  # seeds averaged
        "error": f"tip distance to the chunk of {models[0]} under the same text, same seed, mean over the steps (cm)",  # the scatter's quantity
        "horizon": listed(change.mean(axis=2), 2), "spread": listed(spread, 2),  # [models, rows, steps]
    }
    arm = kinematics.link_origins(data["state"].double().numpy()[:, :joints]) * 100  # [frames, links, 3]
    return {"summary": summary, "images": data["images"], "arm": arm, "tips": tips, "end": end, "raw": raw, "recorded": recorded,
            "recorded_tip": tip(recorded), "first": first, "counts": counts}


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("path", nargs="?", default="outputs/analysis/subtask_swap_2026-11-16/chunks.pt")
    args.add_argument("--port", type=int, default=8772)
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
        rows = slice(loaded["first"][index], loaded["first"][index] + loaded["counts"][index])
        return app.response_class(json.dumps({
            "arm": listed(loaded["arm"][index], 2), "recorded": listed(loaded["recorded_tip"][index], 2),
            "tips": listed(loaded["tips"][:, rows], 2),  # [models, texts, seeds, steps, 3]
            "ends": [None if point[0] is None else point for point in listed(loaded["end"][rows], 2)],
            "gripper": listed(loaded["raw"][:, rows, :, :, -1], 2), "recorded_gripper": listed(loaded["recorded"][index, :, -1], 2),
        }), mimetype="application/json")

    @app.get("/image/<int:index>/<camera>")
    def image(index, camera):
        return send_file(io.BytesIO(loaded["images"][index][camera]), mimetype="image/jpeg")

    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

"""Position, velocity and torque of every motor along a recorded episode, next to the videos.

uv run python lerobot/adhocs/motor_viewer/view.py <dataset root> [--episode 0] [--joint gripper]
    [--out-dir outputs/analysis/motor_viewer] [--no-open]

One HTML per episode plus the clip it plays (top and wrist views stacked into one H.264 file: one clock, no drift),
then the first page opens in the browser. An existing clip is reused, so a second launch is immediate.
"Velocity input" and "Histogram" show the velocity as the model would read it (5-frame causal mean, 256 levels).
The page only draws: no detection, no model of the torque (that is adhocs/torque_viewer). Needs the dataset columns
``observation.velocity`` / ``observation.effort``, so any root recorded since 2026-11-13 (lerobot_record or the
online recorder).
"""

import argparse
import glob
import json
import subprocess
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.utils.motor_velocity import smooth_velocity

parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("--episode", type=int, default=None, help="default: every episode of the root")
parser.add_argument("--joint", default="gripper", help="joint shown when the page opens")
parser.add_argument("--out-dir", default="outputs/analysis/motor_viewer")
parser.add_argument("--no-open", action="store_true", help="build the pages without opening a browser tab")
args = parser.parse_args()
root, out_dir = Path(args.root), Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

info = json.load(open(root / "meta/info.json"))
assert "observation.effort" in info["features"], f"{root} has no observation.effort column (recorded before 2026-11-13)"
fps = info["fps"]
joints = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]]
data = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
episodes = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{root}/meta/episodes/*/*.parquet")).set_index("episode_index")
video_keys = sorted((k for k, ft in info["features"].items() if ft["dtype"] == "video"), key=lambda k: "top" not in k)
stem = root.parent.name if root.name == "inference_dataset" else root.name
shown = [args.episode] if args.episode is not None else sorted(data.episode_index.unique().tolist())

# Velocity as the model reads it (utils/motor_velocity.py): causal mean of the last 5 frames, then the 256 state
# tokens between q01 and q99 of the velocity column, as for the state. Here the range is this root's meta/stats.json;
# a training run pools it over its roots.
LEVELS = 256
smooth_of = {
    ep: smooth_velocity(np.stack(df.sort_values("frame_index")["observation.velocity"].to_numpy()).astype(np.float64))
    for ep, df in data.groupby("episode_index")
}
all_smooth = np.concatenate(list(smooth_of.values()))
velocity_stats = json.load(open(root / "meta/stats.json"))["observation.velocity"]
q01, q99 = np.array(velocity_stats["q01"]), np.array(velocity_stats["q99"])
level_of = {
    ep: np.rint(np.clip((smooth - q01) / (q99 - q01), 0.0, 1.0) * (LEVELS - 1)).astype(np.int64)
    for ep, smooth in smooth_of.items()
}
all_levels = np.concatenate(list(level_of.values()))
inputs = {
    joint: dict(q01=q01[j], q99=q99[j], hist=np.bincount(all_levels[:, j], minlength=LEVELS).tolist(),
                clipped=float(((all_smooth[:, j] < q01[j]) | (all_smooth[:, j] > q99[j])).mean()))
    for j, joint in enumerate(joints)
}
print(f"velocity input, {len(all_levels)} frames: joint, low, high (deg/s), deg/s per level, clipped, levels used, share in the fullest level")
for joint, row in inputs.items():
    hist = np.array(row["hist"])
    print(f"  {joint:14s} {row['q01']:7.2f} {row['q99']:7.2f} {(row['q99'] - row['q01']) / (LEVELS - 1):6.3f} {row['clipped']:6.1%} "
          f"{(hist > 0).sum():4d} {hist.max() / hist.sum():6.1%}")

PAGE = r"""<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --plane: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,.10); --wash: rgba(11,11,11,.05);
  --pos: #2a78d6; --vel: #1baf7a; --tq: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --plane: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10); --wash: rgba(255,255,255,.07);
    --pos: #3987e5; --vel: #199e70; --tq: #d95926;
  }
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; overflow: hidden; }
body { font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--plane);
       display: flex; gap: 12px; padding: 12px; }
button, select { font: inherit; color: inherit; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
#left { width: clamp(230px, calc((100vh - 190px) * 0.6667), 30vw); display: flex; flex-direction: column; padding: 12px; }
#left h1 { margin: 0; font-size: 14px; font-weight: 600; overflow-wrap: anywhere; }
#left .sub { color: var(--ink2); margin: 2px 0 10px; display: flex; gap: 8px; align-items: center; }
#left .sub a { color: var(--ink2); text-decoration: none; padding: 1px 7px; border: 1px solid var(--border); border-radius: 6px; }
#left .sub a:hover { background: var(--wash); color: var(--ink); }
#left video { width: 100%; background: #000; display: block; border-radius: 6px; }
#transport { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
#play { width: 64px; padding: 4px 0; border: 1px solid var(--border); border-radius: 6px; background: var(--wash); cursor: pointer; }
#rate { border: 1px solid var(--border); border-radius: 6px; background: var(--surface); padding: 3px 4px; }
#time { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--ink2); }
#keys { margin-top: auto; padding-top: 10px; color: var(--muted); font-size: 12px; }
#right { flex: 1; min-width: 0; display: flex; flex-direction: column; padding: 12px 14px 8px; }
#bar { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; margin-bottom: 10px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; padding: 2px; gap: 2px; }
.seg button { border: 0; background: none; padding: 3px 10px; border-radius: 6px; cursor: pointer; color: var(--ink2); white-space: nowrap; }
.seg button:hover { background: var(--wash); }
.seg button.on { background: var(--ink); color: var(--surface); }
#bar label { color: var(--ink2); white-space: nowrap; }
#bar select { border: 1px solid var(--border); border-radius: 6px; background: var(--surface); padding: 3px 4px; }
#reset { border: 1px solid var(--border); border-radius: 6px; background: none; padding: 3px 10px; cursor: pointer; color: var(--ink2); margin-left: auto; }
#reset:hover { background: var(--wash); }
#reset:disabled { opacity: .4; cursor: default; }
#overviewBox { height: 46px; position: relative; margin-bottom: 12px; }
#plotBox { flex: 1; min-height: 0; position: relative; }
canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
#plot { cursor: crosshair; }
#overview { cursor: pointer; }
#tip { position: fixed; display: none; pointer-events: none; z-index: 10; background: var(--surface); border: 1px solid var(--border);
       border-radius: 8px; padding: 7px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.14); font-variant-numeric: tabular-nums; }
#tip .when { color: var(--ink2); margin-bottom: 3px; }
#tip table { border-collapse: collapse; }
#tip td { padding: 1px 0; white-space: nowrap; }
#tip td.name { color: var(--ink2); padding-right: 12px; }
#tip td.value { text-align: right; font-weight: 600; }
#tip td.unit { color: var(--muted); padding-left: 4px; }
#tip i { display: inline-block; width: 10px; height: 2px; margin-right: 6px; vertical-align: middle; border-radius: 1px; }
</style>
<div id="left" class="card">
  <h1>__TITLE__</h1>
  <div class="sub"><span id="episode"></span><span id="nav"></span></div>
  <video id="video" src="__CLIP__" muted playsinline preload="auto"></video>
  <div id="transport">
    <button id="play">Play</button>
    <select id="rate" title="playback speed"><option>0.1</option><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
    <span id="time"></span>
  </div>
  <div id="keys">Click a chart to seek. Wheel zooms the time axis, drag pans, double click resets.
    Space plays, arrows step one frame (shift: one second).</div>
</div>
<div id="right" class="card">
  <div id="bar">
    <span class="seg" id="mode"><button data-v="joint">One joint</button><button data-v="all">All joints</button><button data-v="input">Velocity input</button><button data-v="hist">Histogram</button></span>
    <span class="seg" id="joint"></span>
    <span class="seg" id="measure"><button data-v="position">Position</button><button data-v="velocity">Velocity</button><button data-v="torque">Torque</button></span>
    <label id="logBox" title="bar height = log10(frames + 1)"><input type="checkbox" id="log" checked> log counts</label>
    <label id="smoothBox" title="centered mean, velocity and torque only; the raw signal stays visible underneath">Smoothing <select id="smooth"><option value="1">off</option><option value="3">3 frames</option><option value="7">7 frames</option><option value="15">15 frames</option></select></label>
    <button id="reset">Whole episode</button>
  </div>
  <div id="overviewBox"><canvas id="overview"></canvas></div>
  <div id="plotBox"><canvas id="plot"></canvas></div>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;
const dt = 1 / D.fps, last = (D.n - 1) * dt, LEFT = 58, RIGHT = 14, HEAD = 22, GAP = 12, AXIS = 24;
const MEASURES = {
  position: {label: "Position", unit: "deg", color: "--pos", floor: 2, digits: 2},
  velocity: {label: "Velocity", unit: "deg/s", color: "--vel", floor: 4, digits: 1},
  torque: {label: "Torque", unit: "N·m", color: "--tq", floor: 0.1, digits: 3},
};
const $ = id => document.getElementById(id);
const video = $("video"), plot = $("plot"), overview = $("overview"), tip = $("tip");
const state = {mode: "joint", joint: D.joint, measure: "torque"};
let view = [0, last], layout = [], hover = null, drag = null, drawnFrame = -1, dirty = true;

$("episode").textContent = `episode ${D.episode} · ${D.n} frames · ${last.toFixed(1)} s`;
$("nav").innerHTML = D.episodes.filter(e => e !== D.episode).map(e => `<a href="${D.stem}_ep${e}.html">ep ${e}</a>`).join(" ");
$("joint").innerHTML = D.joints.map(j => `<button data-v="${j}">${j.replace("_", " ")}</button>`).join("");

const signed = (v, digits) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(digits);
const frameAt = t => Math.min(D.n - 1, Math.max(0, Math.round(t / dt)));
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const cache = {};
function smoothed(joint, key, width) {
  const id = joint + key + width;
  if (!cache[id]) {
    const raw = D.series[joint][key], half = (width - 1) / 2, sums = [0];
    raw.forEach(v => sums.push(sums[sums.length - 1] + v));
    cache[id] = raw.map((_, i) => { const a = Math.max(0, i - half), b = Math.min(D.n - 1, i + half); return (sums[b + 1] - sums[a]) / (b + 1 - a); });
  }
  return cache[id];
}
// kind: main = the measured signal in the measure's color; command = dashed gray; raw = faint, under its smoothed copy.
function linesOf(joint, measure) {
  const series = D.series[joint], width = Number($("smooth").value);
  if (measure === "position") { return [{values: series.action, kind: "command", name: "command"}, {values: series.state, kind: "main", name: "measured"}]; }
  if (width === 1) { return [{values: series[measure], kind: "main", name: "measured"}]; }
  return [{values: series[measure], kind: "raw", name: "raw"}, {values: smoothed(joint, measure, width), kind: "main", name: `mean of ${width} frames`}];
}
function strips() {
  if (state.mode === "input") {
    return D.joints.map(j => ({title: j.replace("_", " "), measure: "velocity", joint: j, input: D.input[j], lines: [
      {values: D.series[j].velocity, kind: "raw", name: "raw"}, {values: D.series[j].input, kind: "main", name: "model input (last 5 frames 0.25 / 0.25 / 0.2 / 0.15 / 0.15, 256 levels, y axis = their range)"}]}));
  }
  if (state.mode === "joint") { return Object.keys(MEASURES).map(m => ({title: MEASURES[m].label, measure: m, joint: state.joint, lines: linesOf(state.joint, m)})); }
  return D.joints.map(j => ({title: j.replace("_", " "), measure: state.measure, joint: j, lines: linesOf(j, state.measure)}));
}

function fit(canvas) {
  const box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = box.width * ratio; canvas.height = box.height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  return [ctx, box.width, box.height];
}
const toX = (t, range, width) => LEFT + (t - range[0]) / (range[1] - range[0]) * (width - LEFT - RIGHT);
const toT = (x, range, width) => range[0] + (x - LEFT) / (width - LEFT - RIGHT) * (range[1] - range[0]);
function niceStep(span, count) {
  const raw = span / count, power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].find(m => m * power >= raw) * power;
}
function polyline(ctx, values, first, lastIndex, range, width, toY) {
  ctx.beginPath();
  for (let i = first; i <= lastIndex; i++) { ctx.lineTo(toX(i * dt, range, width), toY(values[i])); }
  ctx.stroke();
}

function drawPlot() {
  const [ctx, width, height] = fit(plot), list = strips(), frame = frameAt(video.currentTime);
  const stripHeight = (height - AXIS - GAP * (list.length - 1)) / list.length;
  const first = Math.max(0, Math.floor(view[0] / dt)), lastIndex = Math.min(D.n - 1, Math.ceil(view[1] / dt));
  const ink = css("--ink"), ink2 = css("--ink2"), muted = css("--muted"), grid = css("--grid"), axis = css("--axis"), surface = css("--surface");
  layout = [];
  list.forEach((strip, k) => {
    const measure = MEASURES[strip.measure], color = css(measure.color);
    const head = k * (stripHeight + GAP), top = head + HEAD, bottom = head + stripHeight;
    let low = Infinity, high = -Infinity;
    for (const line of strip.lines) { for (let i = first; i <= lastIndex; i++) { low = Math.min(low, line.values[i]); high = Math.max(high, line.values[i]); } }
    if (strip.input) { low = strip.input.q01; high = strip.input.q99; }
    else if (high - low < measure.floor) { const middle = (high + low) / 2; low = middle - measure.floor / 2; high = middle + measure.floor / 2; }
    const pad = (high - low) * 0.1; low -= pad; high += pad;
    const toY = v => bottom - (v - low) / (high - low) * (bottom - top);
    layout.push({strip, top: head, bottom, toY});

    // Header: title, unit, key of the lines, value under the playhead.
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.font = "600 13px system-ui, sans-serif"; ctx.fillStyle = ink; ctx.fillText(strip.title, LEFT, head + 9);
    let x = LEFT + ctx.measureText(strip.title).width + 8;
    ctx.font = "12px system-ui, sans-serif"; ctx.fillStyle = muted;
    const unit = state.mode === "joint" ? measure.unit : `${measure.label.toLowerCase()}, ${measure.unit}`;
    ctx.fillText(unit, x, head + 9); x += ctx.measureText(unit).width + 18;
    if (strip.lines.length > 1 && (state.mode === "joint" || k === 0)) {
      for (const line of strip.lines) {
        ctx.strokeStyle = line.kind === "command" ? muted : color; ctx.globalAlpha = line.kind === "raw" ? 0.35 : 1;
        ctx.lineWidth = 2; ctx.setLineDash(line.kind === "command" ? [4, 3] : []);
        ctx.beginPath(); ctx.moveTo(x, head + 9); ctx.lineTo(x + 18, head + 9); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
        ctx.fillStyle = ink2; ctx.fillText(line.name, x + 23, head + 9); x += 23 + ctx.measureText(line.name).width + 14;
      }
    }
    const main = strip.lines[strip.lines.length - 1].values;
    ctx.textAlign = "right"; ctx.font = "600 13px system-ui, sans-serif"; ctx.fillStyle = ink;
    ctx.fillText((strip.input ? `level ${D.series[strip.joint].level[frame]} · ` : "") + `${signed(main[frame], strip.input ? 2 : measure.digits)} ${measure.unit}`, width - RIGHT, head + 9);

    // Grid and y ticks.
    const step = niceStep(high - low, stripHeight > 120 ? 5 : 3), digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "right";
    for (let v = Math.ceil(low / step) * step; v <= high; v += step) {
      const y = Math.round(toY(v)) + 0.5, zero = Math.abs(v) < step / 2;
      ctx.strokeStyle = zero ? axis : grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(LEFT, y); ctx.lineTo(width - RIGHT, y); ctx.stroke();
      ctx.fillStyle = muted; ctx.fillText((zero ? 0 : v).toFixed(digits).replace("-", "−"), LEFT - 8, y);
    }

    ctx.save();
    ctx.beginPath(); ctx.rect(LEFT, top - 4, width - LEFT - RIGHT, bottom - top + 8); ctx.clip();
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    for (const line of strip.lines) {
      ctx.strokeStyle = line.kind === "command" ? muted : color; ctx.globalAlpha = line.kind === "raw" ? 0.3 : 1;
      ctx.lineWidth = line.kind === "main" ? 1.6 : 1.2; ctx.setLineDash(line.kind === "command" ? [4, 3] : []);
      polyline(ctx, line.values, first, lastIndex, view, width, toY);
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.restore();
  });

  // Time axis.
  const bottom = height - AXIS, step = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60].find(s => (view[1] - view[0]) / s <= 12) || 120;
  ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = muted;
  for (let t = Math.ceil(view[0] / step - 1e-9) * step; t <= view[1] + 1e-9; t += step) {
    const x = Math.round(toX(t, view, width)) + 0.5;
    ctx.strokeStyle = axis; ctx.beginPath(); ctx.moveTo(x, bottom + 2); ctx.lineTo(x, bottom + 6); ctx.stroke();
    ctx.fillText(t.toFixed(step < 1 ? 1 : 0) + " s", x, bottom + 9);
  }

  // Hover crosshair, then the playhead with a dot on every measured line.
  const mark = (t, stroke, lineWidth) => {
    if (t < view[0] || t > view[1]) { return null; }
    const x = toX(t, view, width);
    ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth;
    for (const {top, bottom} of layout) { ctx.beginPath(); ctx.moveTo(x, top + HEAD - 4); ctx.lineTo(x, bottom + 4); ctx.stroke(); }
    return x;
  };
  if (hover !== null) { mark(hover * dt, muted, 1); }
  const x = mark(frame * dt, ink, 1.5);
  if (x !== null) {
    for (const {strip, toY} of layout) {
      const y = toY(strip.lines[strip.lines.length - 1].values[frame]);
      ctx.fillStyle = css(MEASURES[strip.measure].color); ctx.strokeStyle = surface; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 7); ctx.fill(); ctx.stroke();
    }
  }
}

// Frames per level of the velocity input over the whole root, one row per joint; the line is the level under the playhead.
function drawHist() {
  const [ctx, width, height] = fit(plot), frame = frameAt(video.currentTime), log = $("log").checked;
  const rowHeight = (height - GAP * (D.joints.length - 1)) / D.joints.length, slot = (width - LEFT - RIGHT) / 256;
  const ink = css("--ink"), ink2 = css("--ink2"), muted = css("--muted"), axis = css("--axis"), color = css("--vel");
  layout = [];
  D.joints.forEach((joint, k) => {
    const input = D.input[joint], head = k * (rowHeight + GAP), top = head + HEAD, bottom = head + rowHeight - 16;
    const size = c => log ? Math.log10(c + 1) : c, peak = Math.max(...input.hist.map(size)), step = (input.q99 - input.q01) / 255;
    layout.push({joint, top: head, bottom: head + rowHeight});
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.font = "600 13px system-ui, sans-serif"; ctx.fillStyle = ink; ctx.fillText(joint.replace("_", " "), LEFT, head + 9);
    const x = LEFT + ctx.measureText(joint.replace("_", " ")).width + 8;
    ctx.font = "12px system-ui, sans-serif"; ctx.fillStyle = muted;
    ctx.fillText(`${step.toFixed(3)} deg/s per level · ${(input.clipped * 100).toFixed(1)}% of frames clipped · ${input.hist.filter(c => c > 0).length} levels used · fullest level ${(Math.max(...input.hist) / D.frames * 100).toFixed(1)}%`, x, head + 9);
    const level = D.series[joint].level[frame];
    ctx.textAlign = "right"; ctx.font = "600 13px system-ui, sans-serif"; ctx.fillStyle = ink;
    ctx.fillText(`level ${level} · ${signed(D.series[joint].input[frame], 2)} deg/s`, width - RIGHT, head + 9);
    ctx.fillStyle = color;
    input.hist.forEach((c, i) => { if (c) { const h = Math.max(1, size(c) / peak * (bottom - top)); ctx.fillRect(LEFT + i * slot, bottom - h, Math.max(1, slot - (slot > 3 ? 1 : 0)), h); } });
    ctx.strokeStyle = axis; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(LEFT, Math.round(bottom) + 0.5); ctx.lineTo(width - RIGHT, Math.round(bottom) + 0.5); ctx.stroke();
    ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "top"; ctx.fillStyle = muted;
    const stepV = niceStep(input.q99 - input.q01, 8);
    for (let v = Math.ceil(input.q01 / stepV) * stepV; v <= input.q99; v += stepV) {
      const tx = LEFT + ((v - input.q01) / step + 0.5) * slot;
      ctx.beginPath(); ctx.moveTo(tx, bottom + 1); ctx.lineTo(tx, bottom + 4); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText((Math.abs(v) < stepV / 2 ? 0 : v).toFixed(stepV < 1 ? 1 : 0).replace("-", "−"), tx, bottom + 5);
    }
    ctx.textAlign = "right"; ctx.fillText(k === 0 ? `deg/s · ${log ? "log" : "linear"} count of ${D.frames} frames, ${D.all_episodes} episodes` : "", width - RIGHT, top - 2);
    ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(LEFT + (level + 0.5) * slot, top); ctx.lineTo(LEFT + (level + 0.5) * slot, bottom); ctx.stroke();
  });
}
function histTip(e) {
  const row = layout.find(r => e.offsetY >= r.top && e.offsetY <= r.bottom), i = Math.floor((e.offsetX - LEFT) / ((plot.clientWidth - LEFT - RIGHT) / 256));
  if (!row || i < 0 || i > 255) { tip.style.display = "none"; return; }
  const input = D.input[row.joint], count = input.hist[i];
  tip.innerHTML = `<div class="when">${row.joint.replace("_", " ")} · level ${i}</div>${signed(input.q01 + i / 255 * (input.q99 - input.q01), 2)} deg/s · ${count} frames (${(count / D.frames * 100).toFixed(2)}%)`;
  tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 16, window.innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = Math.min(e.clientY + 16, window.innerHeight - tip.offsetHeight - 8) + "px";
}

// Whole episode: gripper position as the landmark (opening, grasp, release), the zoom window and the playhead.
function drawOverview() {
  const [ctx, width, height] = fit(overview), values = D.series[D.landmark].state, range = [0, last];
  const low = Math.min(...values), high = Math.max(...values), top = 16, bottom = height - 3;
  const toY = v => bottom - (v - low) / (high - low || 1) * (bottom - top);
  ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "top"; ctx.textAlign = "left"; ctx.fillStyle = css("--muted");
  ctx.fillText(`${D.landmark} position, whole episode`, LEFT, 0);
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(LEFT, bottom + 2.5); ctx.lineTo(width - RIGHT, bottom + 2.5); ctx.stroke();
  if (view[0] > 0 || view[1] < last) {
    const x0 = toX(view[0], range, width), x1 = toX(view[1], range, width);
    ctx.fillStyle = css("--wash"); ctx.fillRect(x0, top - 3, x1 - x0, bottom - top + 6);
    ctx.strokeStyle = css("--axis"); ctx.strokeRect(Math.round(x0) + 0.5, top - 2.5, Math.round(x1 - x0), bottom - top + 5);
  }
  ctx.strokeStyle = css("--ink2"); ctx.lineWidth = 1.2; ctx.lineJoin = "round";
  polyline(ctx, values, 0, D.n - 1, range, width, toY);
  const x = toX(Math.min(video.currentTime, last), range, width);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, top - 3); ctx.lineTo(x, bottom + 3); ctx.stroke();
}

function refresh() {
  for (const group of ["mode", "joint", "measure"]) {
    document.querySelectorAll(`#${group} button`).forEach(b => b.classList.toggle("on", b.dataset.v === state[group]));
  }
  $("joint").style.display = state.mode === "joint" ? "" : "none";
  $("measure").style.display = state.mode === "all" ? "" : "none";
  $("smoothBox").style.display = state.mode === "joint" || state.mode === "all" ? "" : "none";
  $("logBox").style.display = state.mode === "hist" ? "" : "none";
  $("reset").disabled = view[0] === 0 && view[1] === last;
  dirty = true;
}
function setView(start, span) {
  span = Math.min(Math.max(span, 1), last);
  start = Math.min(Math.max(start, 0), last - span);
  view = [start, start + span];
  refresh();
}
function seek(t) { video.currentTime = Math.min(Math.max(t, 0), last); }
function toggle() { if (video.paused) { video.play(); } else { video.pause(); } }

function showTip(event) {
  const frame = hover, rows = [];
  const row = (color, name, value, unit, dashed) => rows.push(`<tr><td class="name"><i style="background:${dashed ? "none" : color};${dashed ? `border-top:2px dashed ${color};height:0` : ""}"></i>${name}</td><td class="value">${value}</td><td class="unit">${unit}</td></tr>`);
  if (state.mode === "joint") {
    const series = D.series[state.joint];
    row("var(--muted)", "command", signed(series.action[frame], 2), "deg", true);
    row("var(--pos)", "position", signed(series.state[frame], 2), "deg");
    row("transparent", "command − position", signed(series.action[frame] - series.state[frame], 2), "deg");
    row("var(--vel)", "velocity", signed(series.velocity[frame], 1), "deg/s");
    row("var(--tq)", "torque", signed(series.torque[frame], 3), "N·m");
  } else if (state.mode === "input") {
    D.joints.forEach(j => row("var(--vel)", j.replace("_", " "), signed(D.series[j].input[frame], 2), `deg/s · level ${D.series[j].level[frame]} · raw ${signed(D.series[j].velocity[frame], 1)}`));
  } else {
    const measure = MEASURES[state.measure], key = state.measure === "position" ? "state" : state.measure;
    D.joints.forEach(j => row(`var(${measure.color})`, j.replace("_", " "), signed(D.series[j][key][frame], measure.digits), measure.unit));
  }
  const title = state.mode === "joint" ? state.joint.replace("_", " ") : state.mode === "input" ? "velocity input" : MEASURES[state.measure].label.toLowerCase();
  tip.innerHTML = `<div class="when">${(frame * dt).toFixed(2)} s · frame ${frame} · ${title}</div><table>${rows.join("")}</table>`;
  tip.style.display = "block";
  tip.style.left = Math.min(event.clientX + 16, window.innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = Math.min(event.clientY + 16, window.innerHeight - tip.offsetHeight - 8) + "px";
}

for (const group of ["mode", "joint", "measure"]) {
  $(group).onclick = e => { if (e.target.dataset.v) { state[group] = e.target.dataset.v; refresh(); } };
}
$("smooth").onchange = e => { e.target.blur(); refresh(); };
$("log").onchange = e => { e.target.blur(); refresh(); };
$("reset").onclick = () => setView(0, last);
$("play").onclick = toggle;
$("rate").onchange = e => { video.playbackRate = Number(e.target.value); e.target.blur(); };
plot.onmousedown = e => { if (state.mode === "hist") { return; } drag = {x: e.offsetX, start: view[0], moved: false}; };
plot.onmousemove = e => {
  const width = plot.clientWidth;
  if (state.mode === "hist") { histTip(e); return; }
  if (drag && Math.abs(e.offsetX - drag.x) > 3) { drag.moved = true; }
  if (drag && drag.moved) {
    setView(drag.start - (e.offsetX - drag.x) / (width - LEFT - RIGHT) * (view[1] - view[0]), view[1] - view[0]);
    hover = null; tip.style.display = "none";
    return;
  }
  if (e.offsetX < LEFT || e.offsetX > width - RIGHT) { hover = null; tip.style.display = "none"; dirty = true; return; }
  hover = frameAt(toT(e.offsetX, view, width));
  showTip(e);
  dirty = true;
};
plot.onmouseup = e => { if (state.mode === "hist") { return; } if (drag && !drag.moved) { seek(toT(e.offsetX, view, plot.clientWidth)); } drag = null; };
plot.onmouseleave = () => { drag = null; hover = null; tip.style.display = "none"; dirty = true; };
plot.ondblclick = () => setView(0, last);
plot.onwheel = e => {
  e.preventDefault();
  if (state.mode === "hist") { return; }
  const t = toT(e.offsetX, view, plot.clientWidth), span = Math.min(Math.max((view[1] - view[0]) * (e.deltaY < 0 ? 0.8 : 1.25), 1), last);
  setView(t - (t - view[0]) / (view[1] - view[0]) * span, span);
};
overview.onclick = e => {
  const t = toT(e.offsetX, [0, last], overview.clientWidth);
  seek(t);
  if (view[1] - view[0] < last) { setView(t - (view[1] - view[0]) / 2, view[1] - view[0]); }
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey) { return; }
  if (e.code === "Space") { toggle(); }
  else if (e.code === "ArrowRight") { video.pause(); seek(video.currentTime + (e.shiftKey ? 1 : dt)); }
  else if (e.code === "ArrowLeft") { video.pause(); seek(video.currentTime - (e.shiftKey ? 1 : dt)); }
  else { return; }
  e.preventDefault();
});
function tick() {
  const t = Math.min(video.currentTime, last), frame = frameAt(t);
  if (!video.paused && (t < view[0] || t > view[1])) { setView(t - (view[1] - view[0]) * 0.1, view[1] - view[0]); }
  if (dirty || frame !== drawnFrame) {
    dirty = false; drawnFrame = frame;
    if (state.mode === "hist") { drawHist(); } else { drawPlot(); }
    drawOverview();
    $("time").textContent = `${t.toFixed(2)} s · frame ${frame}`;
    $("play").textContent = video.paused ? "Play" : "Pause";
  }
  requestAnimationFrame(tick);
}
video.onplay = video.onpause = () => { dirty = true; };
new ResizeObserver(() => { dirty = true; }).observe($("plotBox"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { dirty = true; });
refresh();
requestAnimationFrame(tick);
</script>
"""

for ep in shown:
    df = data[data.episode_index == ep].sort_values("frame_index")
    episode = episodes.loc[ep]
    name = f"{stem}_ep{ep}"

    # Both views of the episode in one clip: the browser plays one stream, so the views stay frame-aligned.
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for key in video_keys:
        path = root / info["video_path"].format(
            video_key=key, chunk_index=episode[f"videos/{key}/chunk_index"], file_index=episode[f"videos/{key}/file_index"]
        )
        command += ["-ss", str(episode[f"videos/{key}/from_timestamp"]), "-t", str(len(df) / fps), "-i", str(path)]
    command += ["-filter_complex", f"vstack=inputs={len(video_keys)}", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-g", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_dir / f"{name}.mp4")]
    if not (out_dir / f"{name}.mp4").exists():
        subprocess.run(command, check=True)

    columns = {"action": "action", "state": "observation.state", "velocity": "observation.velocity", "torque": "observation.effort"}
    arrays = {key: np.stack(df[column].to_numpy()) for key, column in columns.items()}
    arrays["input"] = q01 + level_of[ep] / (LEVELS - 1) * (q99 - q01)
    arrays["level"] = level_of[ep]
    series = {joint: {key: np.round(array[:, j], 3).tolist() for key, array in arrays.items()} for j, joint in enumerate(joints)}
    page_data = dict(fps=fps, n=len(df), joints=joints, series=series, joint=args.joint, landmark="gripper" if "gripper" in joints else joints[-1],
                     episode=ep, episodes=shown, stem=stem, input=inputs, frames=len(data), all_episodes=data.episode_index.nunique())

    page = PAGE
    for key, value in {"__TITLE__": stem, "__CLIP__": f"{name}.mp4", "__DATA__": json.dumps(page_data)}.items():
        page = page.replace(key, value)
    (out_dir / f"{name}.html").write_text(page)
    print(out_dir / f"{name}.html")

if not args.no_open:
    webbrowser.open((out_dir / f"{stem}_ep{shown[0]}.html").resolve().as_uri())

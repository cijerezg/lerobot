"""Torque events of recorded inference episodes: detect peaks in motor torque, then check them against the videos.

uv run python lerobot/adhocs/torque_viewer/view.py outputs/train/<date>/<run>/inference_dataset [--episode 0]
    [--baseline 10] [--smooth 4] [--out-dir outputs/analysis/torque_viewer]

One HTML per episode plus the clip it plays (top and wrist views stacked into one H.264 file: one clock, no drift).

Detector, per joint (everything is computed here; the page only filters and groups, so the slider is instant):
1. torque is unwrapped (the wrist motors wrap at +-10 N*m) and gets a 3-frame median against single-frame glitches;
2. expected torque = slow baseline (rolling median over --baseline seconds: posture / gravity; the gripper carries no
   gravity load and uses the episode median) + motion model (robust fit on the joint's own velocity:
   a * tanh(v / v0) + b * v + c * dv/dt, i.e. friction and inertia);
3. score = |low-pass(torque - expected)| / robust sigma (1.4826 * median absolute deviation over the episode);
4. candidate peaks = scipy find_peaks on the score, each with prominence, signed height, foot-to-foot extent (90 % of
   the prominence down from the top) and width at half prominence.
The page keeps the candidates whose prominence passes the slider, merges candidates of one joint with overlapping
extents into one event (several crests of the same peak), and groups events of different joints whose peaks lie
within the coincidence window. "raw" detection skips the motion model (torque - baseline only), for comparison.

Torque and velocity come from the ``{joint}.torque`` / ``{joint}.vel`` columns of ``meta/online_labels.parquet``
(lerobot.rl.online_recorder); roots without velocity use the derivative of observation.state instead.
"""

import argparse
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks, peak_widths, savgol_filter

parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("--episode", type=int, default=None, help="default: every episode of the root")
parser.add_argument("--baseline", type=float, default=10.0, help="window of the rolling-median baseline, s")
parser.add_argument("--smooth", type=float, default=4.0, help="low-pass cutoff of the score, Hz")
parser.add_argument("--floor", type=float, default=2.0, help="smallest prominence kept as a candidate, sigma")
parser.add_argument("--out-dir", default="outputs/analysis/torque_viewer")
args = parser.parse_args()
root, out_dir = Path(args.root), Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

info = json.load(open(root / "meta/info.json"))
fps = info["fps"]
names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]]
joints = ["gripper"] + [n for n in names if n != "gripper"]
data = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
all_labels = pd.read_parquet(root / "meta/online_labels.parquet")
assert f"{names[0]}.torque" in all_labels, f"{root} has no torque columns (recorded before torque capture)"
episodes = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{root}/meta/episodes/*/*.parquet")).set_index("episode_index")
video_keys = sorted((k for k, ft in info["features"].items() if ft["dtype"] == "video"), key=lambda k: "top" not in k)
smooth_b, smooth_a = butter(2, args.smooth / (fps / 2))
baseline_frames = int(args.baseline * fps) // 2 * 2 + 1


def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def huber_fit(features, target):
    """Least squares with Huber weights (iteratively reweighted), so contact frames do not pull the motion model."""
    weight = np.ones(len(target))
    for _ in range(20):
        coefficients = np.linalg.lstsq(features * weight[:, None], target * weight, rcond=None)[0]
        error = np.abs(target - features @ coefficients)
        weight = np.sqrt(np.minimum(1, 1.345 * 1.4826 * np.median(error) / (error + 1e-9)))
    return coefficients


def slow_baseline(x, joint):
    if joint == "gripper":
        return np.full(len(x), np.median(x))
    return median_filter(x, baseline_frames, mode="nearest")


def expected_torque(torque, velocity, joint):
    """Baseline and motion part of the torque. Two passes: each one is estimated on the torque minus the other."""
    v0 = 0.05 * np.percentile(np.abs(velocity), 99)
    features = np.stack([np.tanh(velocity / v0), velocity, savgol_filter(velocity, 9, 2, deriv=1, delta=1 / fps)], axis=1)
    features = features / features.std(axis=0)
    baseline = slow_baseline(torque, joint)
    for _ in range(2):
        motion = features @ huber_fit(features, torque - baseline)
        baseline = slow_baseline(torque - motion, joint)
    return baseline, motion


def detect(torque, expected):
    """Score of one joint and its candidate peaks: [frame, prominence, signed score, foot start, foot end, half start, half end]."""
    sigma = robust_sigma(torque - expected)
    signed = filtfilt(smooth_b, smooth_a, torque - expected) / sigma
    score = np.abs(signed)
    peaks, found = find_peaks(score, prominence=args.floor)
    prominence = (found["prominences"], found["left_bases"], found["right_bases"])
    foot = peak_widths(score, peaks, rel_height=0.9, prominence_data=prominence)
    half = peak_widths(score, peaks, rel_height=0.5, prominence_data=prominence)
    rows = np.stack([peaks, found["prominences"], signed[peaks], foot[2], foot[3], half[2], half[3]], axis=1)
    return dict(expected=np.round(expected, 3).tolist(), score=np.round(score, 2).tolist(), sigma=round(float(sigma), 4),
                peaks=np.round(rows, 2).tolist())


PAGE = r"""<!doctype html><meta charset="utf-8"><title>torque __TITLE__</title>
<style>
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; overflow: hidden; }
body { font: 13px system-ui, sans-serif; color: #1c2430; background: #fff; display: flex; }
button, select, input { font: inherit; }
#left { width: clamp(240px, calc((100vh - 150px) * 0.6667), 30vw); padding: 8px; display: flex; flex-direction: column; border-right: 1px solid #d8dce2; }
#left video { width: 100%; background: #000; display: block; }
#left input[type=range] { width: 100%; margin: 6px 0 2px; }
#transport { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
#time { font-variant-numeric: tabular-nums; }
#held { color: #b3261e; font-weight: 600; }
#subtask { margin-top: 6px; color: #4a5568; min-height: 2.6em; }
h3 { margin: 0 0 6px; font-size: 13px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#right { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.bar { display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: center; padding: 6px 10px; border-bottom: 1px solid #d8dce2; }
.bar label { white-space: nowrap; }
#threshold { width: 200px; vertical-align: middle; }
#thresholdValue { display: inline-block; min-width: 44px; font-weight: 600; font-variant-numeric: tabular-nums; }
#joints { position: absolute; left: 0; top: 2px; width: 92px; z-index: 2; }
#joints label { display: flex; align-items: center; justify-content: flex-end; gap: 3px; height: 16px; font-size: 11px; cursor: pointer; }
#joints input { margin: 0; width: 11px; height: 11px; }
.key { color: #4a5568; white-space: nowrap; }
.key i { display: inline-block; width: 14px; height: 9px; margin-right: 3px; vertical-align: baseline; }
#body { flex: 1; min-height: 0; display: flex; }
#plots { flex: 1; min-width: 0; display: flex; flex-direction: column; position: relative; }
#rasterBox { height: 134px; position: relative; border-bottom: 1px solid #d8dce2; }
#panelBox { flex: 1; min-height: 0; position: relative; }
canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; cursor: crosshair; }
.cursor { position: absolute; top: 0; bottom: 0; width: 0; border-left: 1.5px solid #e11d48; pointer-events: none; }
#hair { position: absolute; top: 0; bottom: 0; width: 0; border-left: 1px solid #9aa3af; pointer-events: none; display: none; }
#tip { position: fixed; background: #fff; border: 1px solid #b6bcc6; border-radius: 4px; padding: 4px 7px; pointer-events: none; display: none;
       font-size: 12px; white-space: pre; z-index: 10; box-shadow: 0 1px 4px rgba(0,0,0,.15); font-variant-numeric: tabular-nums; }
#list { width: 310px; border-left: 1px solid #d8dce2; display: flex; flex-direction: column; }
#listHead { padding: 6px 8px; border-bottom: 1px solid #d8dce2; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
#count { font-weight: 600; margin-right: auto; }
#events { flex: 1; overflow-y: auto; }
.event { padding: 4px 8px; border-bottom: 1px solid #eceef1; cursor: pointer; border-left: 4px solid transparent; }
.event:hover { background: #f3f5f8; }
.event.selected { background: #fff3c4; border-left-color: #1c2430; }
.event .when { font-weight: 600; font-variant-numeric: tabular-nums; }
.event .tag { color: #4a5568; font-size: 11px; margin-left: 6px; }
.event .member { display: flex; gap: 6px; font-variant-numeric: tabular-nums; font-size: 12px; }
.event .member span:first-child { flex: 1; }
.event .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
</style>
<div id="left">
  <h3 title="__TITLE__">__TITLE__</h3>
  <video id="video" src="__CLIP__" muted playsinline preload="auto"></video>
  <input id="seek" type="range" min="0" max="__LAST__" step="__DT__" value="0">
  <div id="transport">
    <button id="play">play</button>
    <select id="rate"><option>0.1</option><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>x
    <span id="time"></span> <span id="held"></span>
  </div>
  <div id="subtask"></div>
</div>
<div id="right">
  <div class="bar">
    <label>prominence &ge; <input id="threshold" type="range" min="__FLOOR__" max="30" step="0.25" value="6"> <span id="thresholdValue"></span></label>
    <label>detect on <select id="mode"><option value="residual">torque - baseline - motion</option><option value="raw">torque - baseline</option></select></label>
    <label>joints coincide within <select id="gap"><option>0.1</option><option>0.2</option><option selected>0.3</option><option>0.5</option><option>1</option></select> s</label>
    <label>joints &ge; <select id="minJoints"><option>1</option><option>2</option><option>3</option><option>4</option></select></label>
    <label><input id="skipHeld" type="checkbox"> skip operator spans</label>
    <label>window <select id="span"><option value="0">whole episode</option><option>60</option><option>20</option><option>8</option></select></label>
    <label><input id="showPosition" type="checkbox"> position</label>
    <label><input id="showVelocity" type="checkbox"> velocity</label>
    <span class="key"><i style="background:#1c2430;height:2px"></i>torque <i style="background:#8b93a1;height:2px"></i>expected
      <i style="background:#d9620b"></i>one joint <i style="background:#1d5fd1"></i>several joints
      <i style="background:rgba(60,70,90,.16)"></i>operator <i style="border-left:1px dashed #6b7280;width:4px"></i>subtask change</span>
  </div>
  <div id="body">
    <div id="plots">
      <div id="rasterBox"><div id="joints"></div><canvas id="raster"></canvas><div class="cursor"></div></div>
      <div id="panelBox"><canvas id="panels"></canvas><div id="hair"></div><div class="cursor"></div></div>
    </div>
    <div id="list">
      <div id="listHead"><span id="count"></span>
        <button id="prev" title="previous event (p)">prev</button><button id="next" title="next event (n)">next</button>
        <label title="replay the selected event, 1 s before to 1 s after (l)"><input id="loop" type="checkbox"> loop</label>
        <select id="sort"><option value="time">by time</option><option value="strength">by strength</option></select>
      </div>
      <div id="events"></div>
    </div>
  </div>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;
const dt = 1 / D.fps, last = (D.n - 1) * dt, LEFT = 96, RIGHT = 10, LONE = "#d9620b", SHARED = "#1d5fd1";
const $ = id => document.getElementById(id);
const video = $("video"), seekBar = $("seek"), tip = $("tip"), hair = $("hair");
const raster = $("raster"), panels = $("panels"), cursors = document.querySelectorAll(".cursor");
let events = [], selected = null, view = [0, last], strips = [], rasterMarks = [], pending = false;

$("joints").innerHTML = D.joints.map(j => `<label>${j}<input type="checkbox" checked value="${j}"></label>`).join("");
const enabled = () => [...document.querySelectorAll("#joints input:checked")].map(el => el.value);
const isHeld = t => D.held.some(h => h[0] <= t && t < h[1]);
const mode = () => $("mode").value;

// Candidates above the slider -> per joint, overlapping extents merge into one event (crests of the same peak)
// -> across joints, events whose peaks lie within the coincidence window form one group.
function detect() {
  const threshold = Number($("threshold").value), gap = Number($("gap").value), single = [];
  for (const joint of enabled()) {
    const series = D.series[joint][mode()];
    let event = null;
    for (const [frame, prominence, signed, start, end, halfStart, halfEnd] of series.peaks) {
      if (prominence < threshold) { continue; }
      if (!event || start > event.end) {
        event = {joint, start, end, frame, signed, prominence, crests: 0, halfStart, halfEnd};
        single.push(event);
      }
      event.crests += 1;
      event.start = Math.min(event.start, start);
      event.end = Math.max(event.end, end);
      if (Math.abs(signed) > Math.abs(event.signed) || event.crests === 1) {
        Object.assign(event, {frame, signed, prominence, halfStart, halfEnd});
      }
    }
  }
  single.forEach(e => { e.t = e.frame * dt; e.newtonMeters = e.signed * D.series[e.joint][mode()].sigma; e.width = (e.halfEnd - e.halfStart) * dt; });
  single.sort((a, b) => a.t - b.t);
  const groups = [];
  for (const event of single) {
    let group = groups[groups.length - 1];
    if (!group || event.t - group.lastPeak > gap) { group = {members: []}; groups.push(group); }
    group.members.push(event);
    group.lastPeak = event.t;
  }
  const previous = selected && selected.t;
  events = groups.filter(g => g.members.length >= Number($("minJoints").value));
  for (const group of events) {
    group.members.sort((a, b) => Math.abs(b.signed) - Math.abs(a.signed));
    group.t = group.members[0].t;
    group.t0 = Math.min(...group.members.map(e => e.start)) * dt;
    group.t1 = Math.max(...group.members.map(e => e.end)) * dt;
    group.strength = Math.abs(group.members[0].signed);
    group.held = isHeld(group.t);
    group.color = group.members.length > 1 ? SHARED : LONE;
  }
  if ($("skipHeld").checked) { events = events.filter(g => !g.held); }
  selected = events.find(g => g.t === previous) || null;
  $("thresholdValue").textContent = threshold.toFixed(2).replace(/\.?0+$/, "") + " sigma";
  renderList();
  redraw();
}

function renderList() {
  const order = events.slice();
  if ($("sort").value === "strength") { order.sort((a, b) => b.strength - a.strength); }
  const shared = events.filter(g => g.members.length > 1).length;
  $("count").textContent = `${events.length} events (${shared} on several joints)`;
  $("events").innerHTML = order.map(group => {
    const widest = Math.max(...group.members.map(e => e.width));
    const tags = [widest >= 1 ? `sustained ${widest.toFixed(1)} s` : `bump ${widest.toFixed(2)} s`];
    if (group.held) { tags.push("operator"); }
    return `<div class="event" data-i="${events.indexOf(group)}"><span class="dot" style="background:${group.color}"></span>` +
      `<span class="when">${group.t.toFixed(2)} s</span><span class="tag">${tags.join(" / ")}</span>` +
      group.members.map(e => `<div class="member"><span>${e.joint}${e.crests > 1 ? ` <span class="tag">${e.crests} crests, one peak</span>` : ""}</span>` +
        `<span>${e.newtonMeters >= 0 ? "+" : "-"}${Math.abs(e.newtonMeters).toFixed(2)} N*m</span><span>${Math.abs(e.signed).toFixed(1)} sigma</span></div>`).join("") + "</div>";
  }).join("");
  markSelected();
}
function markSelected() {
  document.querySelectorAll(".event").forEach(el => el.classList.toggle("selected", events[el.dataset.i] === selected));
  const el = document.querySelector(".event.selected");
  if (el) { el.scrollIntoView({block: "nearest"}); }
}

function fit(canvas) {
  const box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = box.width * ratio; canvas.height = box.height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  ctx.font = "11px system-ui, sans-serif";
  return [ctx, box.width, box.height];
}
const toX = (t, range, width) => LEFT + (t - range[0]) / (range[1] - range[0]) * (width - LEFT - RIGHT);
const toT = (x, range, width) => range[0] + (x - LEFT) / (width - LEFT - RIGHT) * (range[1] - range[0]);

function backdrop(ctx, range, width, top, bottom) {
  ctx.save();
  ctx.beginPath(); ctx.rect(LEFT, top, width - LEFT - RIGHT, bottom - top); ctx.clip();
  ctx.fillStyle = "rgba(60,70,90,.16)";
  D.held.forEach(h => ctx.fillRect(toX(h[0], range, width), top, toX(h[1], range, width) - toX(h[0], range, width), bottom - top));
  ctx.strokeStyle = "#6b7280"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  D.segments.slice(1).forEach(s => { const x = Math.round(toX(s.t, range, width)) + 0.5; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke(); });
  ctx.restore();
}
function timeAxis(ctx, range, width, y) {
  const step = [1, 2, 5, 10, 20, 30, 60].find(s => (range[1] - range[0]) / s <= 12) || 60;
  ctx.fillStyle = "#4a5568"; ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = Math.ceil(range[0] / step) * step; t <= range[1]; t += step) { ctx.fillText(t + " s", toX(t, range, width), y); }
}

// Whole episode, one row per joint: a bar per event from foot to foot, a tick at its peak, a line joining coincident joints.
function drawRaster() {
  const [ctx, width, height] = fit(raster), range = [0, last], rows = D.joints, rowHeight = 16;
  backdrop(ctx, range, width, 2, height - 18);
  rasterMarks = [];
  ctx.textBaseline = "middle";
  rows.forEach((joint, r) => {
    ctx.fillStyle = "#eceef1"; ctx.fillRect(LEFT, 2 + (r + 1) * rowHeight - 1, width - LEFT - RIGHT, 1);
  });
  if (view[0] > 0 || view[1] < last) {
    ctx.fillStyle = "rgba(250,204,21,.22)"; ctx.fillRect(toX(view[0], range, width), 0, toX(view[1], range, width) - toX(view[0], range, width), height - 18);
  }
  for (const group of events) {
    const x = toX(group.t, range, width), ys = group.members.map(e => 2 + (rows.indexOf(e.joint) + 0.5) * rowHeight);
    if (group === selected) { ctx.fillStyle = "rgba(28,36,48,.14)"; ctx.fillRect(x - 6, 0, 12, height - 18); }
    ctx.strokeStyle = group.color; ctx.fillStyle = group.color; ctx.lineWidth = 1;
    if (ys.length > 1) { ctx.beginPath(); ctx.moveTo(x, Math.min(...ys)); ctx.lineTo(x, Math.max(...ys)); ctx.stroke(); }
    group.members.forEach((e, k) => {
      const x0 = toX(e.start * dt, range, width), x1 = toX(e.end * dt, range, width), tall = Math.min(rowHeight - 4, 4 + Math.abs(e.signed));
      ctx.globalAlpha = 0.35; ctx.fillRect(x0, ys[k] - tall / 2, Math.max(x1 - x0, 1), tall);
      ctx.globalAlpha = 1; ctx.fillRect(toX(e.t, range, width) - 1.5, ys[k] - tall / 2, 3, tall);
    });
    rasterMarks.push({x, group});
  }
  timeAxis(ctx, range, width, height - 15);
}

function curve(ctx, values, first, lastIndex, toY, width) {
  ctx.beginPath();
  for (let i = first; i <= lastIndex; i++) { ctx.lineTo(toX(i * dt, view, width), toY(values[i])); }
  ctx.stroke();
}
function extent(arrays, first, lastIndex, floor) {
  let low = Infinity, high = -Infinity;
  for (const values of arrays) { for (let i = first; i <= lastIndex; i++) { if (values[i] < low) { low = values[i]; } if (values[i] > high) { high = values[i]; } } }
  if (floor !== undefined) { low = 0; high = Math.max(high, floor); }
  const pad = (high - low) * 0.08 || 1;
  return [floor !== undefined ? 0 : low - pad, high + pad];
}

// Per enabled joint: torque with the expected torque and a marker at each event's peak; the score with the same events;
// optionally position (state, action) and velocity. Each strip has its own y axis.
function drawPanels() {
  const [ctx, width, height] = fit(panels), on = D.joints.filter(j => enabled().includes(j)), threshold = Number($("threshold").value);
  const first = Math.max(0, Math.floor(view[0] / dt)), lastIndex = Math.min(D.n - 1, Math.ceil(view[1] / dt));
  const kinds = [["torque", 3], ["score", 2]].concat($("showPosition").checked ? [["position", 2]] : [], $("showVelocity").checked ? [["velocity", 2]] : []);
  const weight = kinds.reduce((sum, k) => sum + k[1], 0), jointHeight = (height - 18) / on.length;
  strips = [];
  on.forEach((joint, row) => {
    const series = D.series[joint], detected = series[mode()], mine = [];
    events.forEach(group => group.members.forEach(e => { if (e.joint === joint) { mine.push([e, group]); } }));
    let top = row * jointHeight + 3;
    for (const [kind, share] of kinds) {
      const bottom = top + (jointHeight - 6) * share / weight - 3;
      const arrays = {torque: [series.torque, detected.expected], score: [detected.score], position: [series.state, series.action], velocity: [series.velocity]}[kind];
      const [low, high] = extent(arrays, first, lastIndex, kind === "score" ? threshold * 1.5 : undefined);
      const toY = v => bottom - (v - low) / (high - low) * (bottom - top);
      strips.push({joint, kind, top, bottom, low, high});
      ctx.fillStyle = "#fafbfc"; ctx.fillRect(LEFT, top, width - LEFT - RIGHT, bottom - top);
      backdrop(ctx, view, width, top, bottom);
      ctx.save();
      ctx.beginPath(); ctx.rect(LEFT, top, width - LEFT - RIGHT, bottom - top); ctx.clip();
      for (const [e, group] of mine) {
        const x0 = toX(e.start * dt, view, width), x1 = toX(e.end * dt, view, width);
        ctx.fillStyle = group.color; ctx.globalAlpha = group === selected ? 0.32 : 0.16;
        ctx.fillRect(x0, top, Math.max(x1 - x0, 1), bottom - top);
        ctx.globalAlpha = 1;
      }
      if (low < 0 && high > 0) { ctx.fillStyle = "#d8dce2"; ctx.fillRect(LEFT, Math.round(toY(0)), width, 1); }
      ctx.lineJoin = "round";
      if (kind === "torque") {
        ctx.strokeStyle = "#8b93a1"; ctx.lineWidth = 1.5; curve(ctx, detected.expected, first, lastIndex, toY, width);
        ctx.strokeStyle = "#1c2430"; ctx.lineWidth = 1.2; curve(ctx, series.torque, first, lastIndex, toY, width);
      } else if (kind === "score") {
        ctx.beginPath(); ctx.moveTo(toX(first * dt, view, width), bottom);
        for (let i = first; i <= lastIndex; i++) { ctx.lineTo(toX(i * dt, view, width), toY(detected.score[i])); }
        ctx.lineTo(toX(lastIndex * dt, view, width), bottom); ctx.fillStyle = "#c3c9d3"; ctx.fill();
        ctx.strokeStyle = "#4a5568"; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(LEFT, toY(threshold)); ctx.lineTo(width, toY(threshold)); ctx.stroke(); ctx.setLineDash([]);
      } else if (kind === "position") {
        ctx.strokeStyle = "#8b93a1"; ctx.lineWidth = 1; ctx.setLineDash([4, 3]); curve(ctx, series.action, first, lastIndex, toY, width); ctx.setLineDash([]);
        ctx.strokeStyle = "#1c2430"; ctx.lineWidth = 1.2; curve(ctx, series.state, first, lastIndex, toY, width);
      } else {
        ctx.strokeStyle = "#1c2430"; ctx.lineWidth = 1.2; curve(ctx, series.velocity, first, lastIndex, toY, width);
      }
      if (kind === "torque" || kind === "score") {
        for (const [e, group] of mine) {
          const x = toX(e.t, view, width), y = kind === "torque" ? toY(series.torque[e.frame]) : toY(detected.score[e.frame]);
          ctx.fillStyle = group.color; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, y, group === selected ? 5.5 : 4, 0, 7); ctx.stroke(); ctx.fill();
        }
      }
      ctx.restore();
      const digits = v => v.toFixed(Math.abs(v) < 10 ? 1 : 0), unit = {torque: "N*m", score: "sigma", position: "deg", velocity: "deg/s"}[kind];
      ctx.fillStyle = "#6b7280"; ctx.textAlign = "right";
      if (bottom - top >= 26) {
        ctx.textBaseline = "top"; ctx.fillText(digits(high), LEFT - 5, top);
        ctx.textBaseline = "bottom"; ctx.fillText(digits(low), LEFT - 5, bottom);
        ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(unit, 4, (top + bottom) / 2);
      } else {
        ctx.textBaseline = "middle"; ctx.fillText(`${digits(low)} to ${digits(high)} ${unit}`, LEFT - 5, (top + bottom) / 2);
      }
      if (kind === "torque") {
        ctx.font = "600 12px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "top";
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 4; ctx.strokeText(joint, LEFT + 5, top + 2);
        ctx.fillStyle = "#1c2430"; ctx.fillText(joint, LEFT + 5, top + 2); ctx.font = "11px system-ui, sans-serif";
      }
      top = bottom + 3;
    }
  });
  timeAxis(ctx, view, width, height - 15);
}
function redraw() {
  if (pending) { return; }
  pending = true;
  requestAnimationFrame(() => { pending = false; drawRaster(); drawPanels(); });
}

function seek(t) { video.currentTime = Math.min(Math.max(t, 0), last); }
function toggle() { if (video.paused) { video.play(); } else { video.pause(); } }
function setView(center) {
  const span = Number($("span").value);
  view = span ? [Math.max(0, Math.min(center - span / 2, last - span)), 0] : [0, last];
  if (span) { view[1] = view[0] + span; }
  redraw();
}
function select(group) {
  selected = group;
  if ($("loop").checked) { seek(group.t0 - 1); video.play(); } else { video.pause(); seek(group.t); }
  if (Number($("span").value)) { setView(group.t); }
  markSelected();
  redraw();
}
function jump(direction) {
  const t = selected && Math.abs(video.currentTime - selected.t) < 2 * dt ? selected.t : video.currentTime;
  const group = direction > 0 ? events.find(g => g.t > t + dt) : events.filter(g => g.t < t - dt).pop();
  if (group) { select(group); }
}
function tick() {
  let t = Math.min(video.currentTime, last);
  if ($("loop").checked && selected && !video.paused && (t > selected.t1 + 1 || t < selected.t0 - 1.2)) { seek(selected.t0 - 1); }
  if (Number($("span").value) && (t < view[0] || t > view[1])) { setView(t); }
  const rasterWidth = raster.clientWidth, panelWidth = panels.clientWidth;
  cursors[0].style.left = toX(t, [0, last], rasterWidth) + "px";
  cursors[1].style.left = toX(t, view, panelWidth) + "px";
  if (document.activeElement !== seekBar) { seekBar.value = t; }
  $("time").textContent = t.toFixed(2) + " s  frame " + Math.round(t / dt);
  $("play").textContent = video.paused ? "play" : "pause";
  $("subtask").textContent = (D.segments.filter(s => s.t <= t).pop() || {text: ""}).text;
  $("held").textContent = isHeld(t) ? "operator" : "";
  requestAnimationFrame(tick);
}

document.querySelectorAll(".bar select, .bar input[type=checkbox], #sort").forEach(el => el.addEventListener("change", () => { el.blur(); }));
["mode", "gap", "minJoints", "skipHeld"].forEach(id => $(id).addEventListener("change", detect));
$("joints").addEventListener("change", detect);
$("threshold").addEventListener("input", detect);
$("sort").addEventListener("change", renderList);
["showPosition", "showVelocity"].forEach(id => $(id).addEventListener("change", redraw));
$("span").addEventListener("change", () => setView(video.currentTime));
$("events").onclick = e => { const el = e.target.closest(".event"); if (el) { select(events[el.dataset.i]); } };
$("prev").onclick = () => jump(-1);
$("next").onclick = () => jump(1);
$("play").onclick = toggle;
seekBar.oninput = () => seek(Number(seekBar.value));
seekBar.onchange = () => seekBar.blur();
$("rate").onchange = e => { video.playbackRate = Number(e.target.value); e.target.blur(); };
raster.onclick = e => {
  const x = e.offsetX, near = rasterMarks.filter(m => Math.abs(m.x - x) <= 4).sort((a, b) => Math.abs(a.x - x) - Math.abs(b.x - x))[0];
  if (near) { select(near.group); } else { seek(toT(x, [0, last], raster.clientWidth)); }
};
panels.onclick = e => seek(toT(e.offsetX, view, panels.clientWidth));
panels.onwheel = e => {
  e.preventDefault();
  const options = [...$("span").options].map(o => Number(o.value)), spans = [0].concat(options.filter(Boolean).sort((a, b) => b - a));
  const index = Math.min(Math.max(spans.indexOf(Number($("span").value)) + (e.deltaY < 0 ? 1 : -1), 0), spans.length - 1);
  $("span").value = spans[index];
  setView(toT(e.offsetX, view, panels.clientWidth));
};
panels.onmousemove = e => {
  const t = toT(e.offsetX, view, panels.clientWidth), frame = Math.round(t / dt), strip = strips.find(s => s.top <= e.offsetY && e.offsetY <= s.bottom);
  if (!strip || frame < 0 || frame >= D.n || e.offsetX < LEFT) { tip.style.display = hair.style.display = "none"; return; }
  const series = D.series[strip.joint], detected = series[mode()];
  const lines = [`${(frame * dt).toFixed(2)} s  ${strip.joint}`, `torque    ${series.torque[frame].toFixed(2)} N*m`,
                 `expected  ${detected.expected[frame].toFixed(2)} N*m`, `score     ${detected.score[frame].toFixed(1)} sigma (sigma = ${detected.sigma.toFixed(3)} N*m)`];
  if (strip.kind === "position") { lines.push(`state ${series.state[frame].toFixed(1)}  action ${series.action[frame].toFixed(1)} deg`); }
  if (strip.kind === "velocity") { lines.push(`velocity ${series.velocity[frame].toFixed(1)} deg/s (${D.velocitySource})`); }
  events.forEach(group => group.members.forEach(m => {
    if (m.joint === strip.joint && m.start <= frame && frame <= m.end) {
      lines.push(`event: peak ${m.t.toFixed(2)} s, prominence ${m.prominence.toFixed(1)}, ${group.members.length > 1 ? "with " + group.members.filter(o => o !== m).map(o => o.joint).join(", ") : "this joint only"}`);
    }
  }));
  tip.textContent = lines.join("\n");
  tip.style.display = hair.style.display = "block";
  hair.style.left = e.offsetX + "px";
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8) + "px";
};
panels.onmouseleave = () => { tip.style.display = hair.style.display = "none"; };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey) { return; }
  if (e.code === "Space") { toggle(); }
  else if (e.code === "KeyN") { jump(1); }
  else if (e.code === "KeyP") { jump(-1); }
  else if (e.code === "KeyL") { $("loop").checked = !$("loop").checked; if (selected && $("loop").checked) { select(selected); } }
  else if (e.code === "ArrowRight") { video.pause(); seek(video.currentTime + (e.shiftKey ? 1 : dt)); }
  else if (e.code === "ArrowLeft") { video.pause(); seek(video.currentTime - (e.shiftKey ? 1 : dt)); }
  else { return; }
  e.preventDefault();
});
$("loop").addEventListener("change", e => { e.target.blur(); if (selected && e.target.checked) { select(selected); } });
new ResizeObserver(redraw).observe($("panelBox"));
detect();
requestAnimationFrame(tick);
</script>
"""

for ep in [args.episode] if args.episode is not None else sorted(all_labels.episode_index.unique()):
    df = data[data.episode_index == ep].sort_values("frame_index")
    labels = all_labels[all_labels.episode_index == ep].sort_values("frame_index")
    assert len(labels) == len(df), f"episode {ep}: {len(labels)} label rows for {len(df)} frames"
    episode = episodes.loc[ep]
    name = f"{root.parent.name}_ep{ep}"
    t = df.timestamp.to_numpy()
    state, action = np.stack(df["observation.state"].to_numpy()), np.stack(df.action.to_numpy())

    # Both views of the episode in one clip: the browser plays one stream, so the views stay frame-aligned.
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for key in video_keys:
        path = root / info["video_path"].format(
            video_key=key, chunk_index=episode[f"videos/{key}/chunk_index"], file_index=episode[f"videos/{key}/file_index"]
        )
        command += ["-ss", str(episode[f"videos/{key}/from_timestamp"]), "-t", str(len(df) / fps), "-i", str(path)]
    command += ["-filter_complex", f"vstack=inputs={len(video_keys)}", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-g", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_dir / f"{name}.mp4")]
    subprocess.run(command, check=True)

    has_vel = f"{names[0]}.vel" in labels  # roots recorded before velocity capture have torque only
    series = {}
    for joint in joints:
        j = names.index(joint)
        torque = median_filter(np.unwrap(labels[f"{joint}.torque"].to_numpy(), period=20.0), 3, mode="nearest")
        velocity = labels[f"{joint}.vel"].to_numpy() if has_vel else savgol_filter(state[:, j], 9, 2, deriv=1, delta=1 / fps)
        baseline, motion = expected_torque(torque, velocity, joint)
        series[joint] = dict(
            torque=np.round(torque, 3).tolist(), state=np.round(state[:, j], 2).tolist(), action=np.round(action[:, j], 2).tolist(),
            velocity=np.round(velocity, 2).tolist(),
            residual=detect(torque, baseline + motion), raw=detect(torque, slow_baseline(torque, joint)),
        )

    spans = np.flatnonzero(np.diff(np.concatenate([[0], labels.is_intervention.to_numpy().astype(int), [0]])))
    held = [[float(t[start]), float(t[end - 1] + 1 / fps)] for start, end in zip(spans[::2], spans[1::2])]
    subtask = labels.subtask.to_numpy()
    segments = [dict(t=float(t[i]), text=str(subtask[i])) for i in np.flatnonzero(np.concatenate([[True], subtask[1:] != subtask[:-1]]))]
    page_data = dict(fps=fps, n=len(df), joints=joints, series=series, held=held, segments=segments,
                     velocitySource="motor" if has_vel else "derivative of state")

    page = PAGE
    for key, value in {
        "__TITLE__": f"{root.parent.name} ep {ep}", "__CLIP__": f"{name}.mp4", "__LAST__": f"{t[-1]:.4f}", "__DT__": f"{1 / fps:.6f}",
        "__FLOOR__": f"{args.floor:g}", "__DATA__": json.dumps(page_data),
    }.items():
        page = page.replace(key, value)
    (out_dir / f"{name}.html").write_text(page)
    print(out_dir / f"{name}.html")

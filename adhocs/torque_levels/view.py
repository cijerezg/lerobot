"""Whether the gripper reads as holding something (level low / high), from its smoothed torque, over every episode of a root.

uv run python lerobot/adhocs/torque_levels/view.py <dataset root> [--threshold 0.18] [--closed 2] [--opened 2]
    [--margin 0.06] [--out-dir outputs/analysis/torque_levels] [--no-open]

One HTML for the root plus one clip per episode (top and wrist views stacked). The torque is the gripper's
``observation.effort`` smoothed with a causal weighted mean of the last 4 frames (0.4 on the newest, then 0.3, 0.2,
0.1). Two rules, switched in the page:
- latch (default): the level goes high when the torque reaches the threshold and the gripper closed by at least
  ``closed`` deg within the last second; it stays high until the gripper has opened by ``opened`` deg from its
  tightest position since.
- threshold alone, with hysteresis: high when the torque reaches threshold + margin, low when it falls below
  threshold - margin, otherwise unchanged.
An existing clip is reused.
"""

import argparse
import glob
import json
import subprocess
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1])  # newest frame first

parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("--threshold", type=float, default=0.18, help="torque threshold when the page opens, N*m")
parser.add_argument("--closed", type=float, default=2, help="latch: closing within the last second needed to go high, deg")
parser.add_argument("--opened", type=float, default=2, help="latch: opening from the tightest position that goes back to low, deg")
parser.add_argument("--margin", type=float, default=0.06, help="threshold alone: hysteresis margin, N*m")
parser.add_argument("--out-dir", default="outputs/analysis/torque_levels")
parser.add_argument("--no-open", action="store_true", help="build the page without opening a browser tab")
args = parser.parse_args()
root, out_dir = Path(args.root), Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

info = json.load(open(root / "meta/info.json"))
assert "observation.effort" in info["features"], f"{root} has no observation.effort column (recorded before 2026-11-13)"
fps = info["fps"]
gripper = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]].index("gripper")
data = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
episodes = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{root}/meta/episodes/*/*.parquet")).set_index("episode_index")
video_keys = sorted((k for k, ft in info["features"].items() if ft["dtype"] == "video"), key=lambda k: "top" not in k)
stem = root.parent.name if root.name == "inference_dataset" else root.name

PAGE = r"""<!doctype html><meta charset="utf-8"><title>__TITLE__ torque levels</title>
<style>
:root {
  color-scheme: light;
  --plane: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,.10); --wash: rgba(11,11,11,.05);
  --low: #2a78d6; --high: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --plane: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10); --wash: rgba(255,255,255,.07);
    --low: #3987e5; --high: #199e70;
  }
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; overflow: hidden; }
body { font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--plane);
       display: flex; gap: 12px; padding: 12px; }
button, select, input { font: inherit; color: inherit; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
#left { width: clamp(230px, calc((100vh - 190px) * 0.6667), 30vw); display: flex; flex-direction: column; padding: 12px; }
#left h1 { margin: 0; font-size: 14px; font-weight: 600; overflow-wrap: anywhere; }
#left .sub { color: var(--ink2); margin: 2px 0 10px; }
#left video { width: 100%; background: #000; display: block; border-radius: 6px; }
#transport { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
#play { width: 64px; padding: 4px 0; border: 1px solid var(--border); border-radius: 6px; background: var(--wash); cursor: pointer; }
#rate { border: 1px solid var(--border); border-radius: 6px; background: var(--surface); padding: 3px 4px; }
#time { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--ink2); }
#keys { margin-top: auto; padding-top: 10px; color: var(--muted); font-size: 12px; }
#right { flex: 1; min-width: 0; display: flex; flex-direction: column; padding: 12px 14px 8px; }
#bar { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin-bottom: 10px; }
#bar label { color: var(--ink2); white-space: nowrap; display: inline-flex; gap: 8px; align-items: center; }
#bar input[type=number] { width: 74px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); padding: 3px 6px; font-variant-numeric: tabular-nums; }
#stats { color: var(--ink2); font-variant-numeric: tabular-nums; }
#stats b { color: var(--ink); font-weight: 600; }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
#top { display: flex; gap: 14px; height: 44%; min-height: 0; margin-bottom: 10px; }
#scatterBox { width: 40%; position: relative; }
#detailBox { flex: 1; min-width: 0; position: relative; }
#rowsBox { flex: 1; min-height: 0; position: relative; }
canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; cursor: crosshair; }
#rows { cursor: pointer; }
</style>
<div id="left" class="card">
  <h1>__TITLE__</h1>
  <div class="sub" id="episode"></div>
  <video id="video" muted playsinline preload="auto"></video>
  <div id="transport">
    <button id="play">Play</button>
    <select id="rate" title="playback speed"><option>0.1</option><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
    <span id="time"></span>
  </div>
  <div id="keys">Drag the threshold line in either chart, or type its value (up / down arrows move it by 0.005 N·m).
    Latch: high once the torque reaches the threshold right after the gripper closed, until the gripper opens again.
    Without the latch the gray band is the margin: inside it the level keeps its previous value.
    Click a point, the episode chart or an episode row to see that frame. Space plays, left / right arrows step one
    frame (shift: one second).</div>
</div>
<div id="right" class="card">
  <div id="bar">
    <label>threshold <input id="bound" type="number" step="0.005"> N·m</label>
    <label><input id="latch" type="checkbox"> latch</label>
    <label>closed by <input id="closed" type="number" step="0.5" min="0"> deg</label>
    <label>until opened by <input id="opened" type="number" step="0.5" min="0"> deg</label>
    <label>margin ± <input id="margin" type="number" step="0.005" min="0"> N·m</label>
    <span id="stats"></span>
  </div>
  <div id="top">
    <div id="scatterBox"><canvas id="scatter"></canvas></div>
    <div id="detailBox"><canvas id="detail"></canvas></div>
  </div>
  <div id="rowsBox"><canvas id="rows"></canvas></div>
</div>
<script>
const D = __DATA__;
const dt = 1 / D.fps, LEFT = 52, RIGHT = 14, HEAD = 22, AXIS = 24, ROW_RIGHT = 78, LEVELS = ["low", "high"];
const longest = Math.max(...D.episodes.map(e => e.n - 1)) * dt, total = D.episodes.reduce((sum, e) => sum + e.n, 0);
const $ = id => document.getElementById(id);
const video = $("video"), scatter = $("scatter"), detail = $("detail"), rows = $("rows");
let bound = D.threshold, margin = D.margin, closed = D.closed, opened = D.opened, latch = true, current = -1, pending = null, drag = null, dirty = true, drawnFrame = -1;
let scatterMap = null, detailMap = null, rowsMap = null;

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const signed = (v, digits) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(digits);
const episode = () => D.episodes[current];
const frameNow = () => Math.min(episode().n - 1, Math.max(0, Math.round(video.currentTime / dt)));

function show(index, t) {
  t = Math.min(Math.max(t, 0), (D.episodes[index].n - 1) * dt);
  if (index === current) { video.currentTime = t; return; }
  current = index; pending = t;
  video.src = episode().clip;
  $("episode").textContent = `episode ${episode().episode} · ${episode().n} frames · ${((episode().n - 1) * dt).toFixed(1)} s`;
  dirty = true;
}
video.onloadedmetadata = () => { video.currentTime = pending; video.playbackRate = Number($("rate").value); };

// Level of every frame of one episode. Latch: high when the torque reaches the threshold and the gripper closed by
// `closed` deg within the last second, until the gripper has opened by `opened` deg from its tightest position since.
// Otherwise: high at threshold + margin, low below threshold - margin, unchanged in between.
function levelsOf(e, latch, margin) {
  const levels = new Uint8Array(e.n);
  let high = false, tightest = 0;
  for (let i = 0; i < e.n; i++) {
    if (!latch) { high = e.torque[i] >= bound + (high ? -margin : margin); }
    else if (high) { tightest = Math.max(tightest, e.state[i]); high = e.state[i] >= tightest - opened; }
    else if (e.torque[i] >= bound && e.state[i] - Math.min(...e.state.slice(Math.max(i - D.fps, 0), i + 1)) >= closed) { high = true; tightest = e.state[i]; }
    levels[i] = high;
  }
  return levels;
}
// Level changes and stretches (runs of one level) shorter than 0.5 s, over all episodes.
function flicker(levelsOfEpisode) {
  let changes = 0, short = 0;
  for (const e of D.episodes) {
    const levels = levelsOfEpisode(e);
    for (let i = 1, start = 0; i <= e.n; i++) {
      if (i === e.n || levels[i] !== levels[start]) { changes += i < e.n; short += (i - start) * dt < 0.5; start = i; }
    }
  }
  return [changes, short];
}
function setLevels() {
  bound = Math.round(bound / 0.005) * 0.005;
  $("bound").value = bound.toFixed(3); $("margin").value = margin.toFixed(3); $("closed").value = closed; $("opened").value = opened; $("latch").checked = latch;
  $("closed").disabled = $("opened").disabled = !latch; $("margin").disabled = latch;
  let high = 0;
  for (const e of D.episodes) {
    e.levels = levelsOf(e, latch, margin); e.share = e.levels.reduce((sum, level) => sum + level, 0) / e.n;
    high += e.share * e.n;
  }
  const [changes, short] = flicker(e => e.levels), [plainChanges, plainShort] = flicker(e => levelsOf(e, false, 0));
  $("stats").innerHTML = `<span class="key" style="background:var(--high)"></span>high <b>${(100 * high / total).toFixed(1)} %</b>`
    + ` &nbsp;·&nbsp; level changes <b>${changes}</b> (${plainChanges} with the threshold alone) · stretches shorter than 0.5 s <b>${short}</b> (${plainShort})`;
  dirty = true;
}
// Fills the stretches of the high level along a time axis.
function bands(ctx, e, toX, y, height, alpha) {
  ctx.globalAlpha = alpha;
  for (let i = 1, start = 0; i <= e.n; i++) {
    if (i < e.n && e.levels[i] === e.levels[start]) { continue; }
    if (e.levels[start]) {
      ctx.fillStyle = css("--" + LEVELS[e.levels[start]]);
      ctx.fillRect(toX((start - 0.5) * dt), y, Math.max(toX((i - 0.5) * dt) - toX((start - 0.5) * dt), 1), height);
    }
    start = i;
  }
  ctx.globalAlpha = 1;
}

function fit(canvas) {
  const box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = box.width * ratio; canvas.height = box.height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  return [ctx, box.width, box.height];
}
function niceStep(span, count) {
  const raw = span / count, power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].find(m => m * power >= raw) * power;
}
// Horizontal grid lines with their values at the left.
function yGrid(ctx, range, toY, x0, x1, count) {
  const step = niceStep(range[1] - range[0], count), digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
  for (let v = Math.ceil(range[0] / step) * step; v <= range[1]; v += step) {
    const y = Math.round(toY(v)) + 0.5, zero = Math.abs(v) < step / 2;
    ctx.strokeStyle = zero ? css("--axis") : css("--grid");
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    ctx.fillStyle = css("--muted"); ctx.fillText((zero ? 0 : v).toFixed(digits).replace("-", "−"), x0 - 8, y);
  }
}
// The threshold dashed with its value at the right; without the latch, the margin as a gray band.
function boundaryLines(ctx, toY, x0, x1) {
  const y = toY(bound), band = latch ? 0 : margin, text = `threshold  ${bound.toFixed(3)}`;
  ctx.fillStyle = css("--ink"); ctx.globalAlpha = 0.1; ctx.fillRect(x0, toY(bound + band), x1 - x0, toY(bound - band) - toY(bound + band)); ctx.globalAlpha = 1;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]);
  ctx.font = "600 11px system-ui, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
  const top = toY(bound + band) - 17;
  ctx.fillStyle = css("--surface"); ctx.fillRect(x1 - ctx.measureText(text).width - 8, top, ctx.measureText(text).width + 8, 15);
  ctx.fillStyle = css("--ink"); ctx.fillText(text, x1 - 4, top + 14);
}
function title(ctx, text, note, x, y) {
  ctx.textBaseline = "middle"; ctx.textAlign = "left";
  ctx.font = "600 13px system-ui, sans-serif"; ctx.fillStyle = css("--ink"); ctx.fillText(text, x, y);
  const width = ctx.measureText(text).width;
  ctx.font = "12px system-ui, sans-serif"; ctx.fillStyle = css("--muted"); ctx.fillText(note, x + width + 8, y);
}

// Every frame of the root: gripper position against smoothed torque.
function drawScatter() {
  const [ctx, width, height] = fit(scatter), top = HEAD + 4, bottom = height - AXIS - 14;
  const toX = v => LEFT + (v - D.position[0]) / (D.position[1] - D.position[0]) * (width - LEFT - RIGHT);
  const toY = v => bottom - (Math.min(Math.max(v, D.torque[0]), D.torque[1]) - D.torque[0]) / (D.torque[1] - D.torque[0]) * (bottom - top);
  scatterMap = {toX, toY, toTorque: y => D.torque[0] + (bottom - y) / (bottom - top) * (D.torque[1] - D.torque[0])};
  title(ctx, "All frames", "smoothed torque (N·m), clipped to the axis, colored by level", LEFT, 9);
  yGrid(ctx, D.torque, toY, LEFT, width - RIGHT, 6);
  const step = niceStep(D.position[1] - D.position[0], 7);
  ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--muted");
  for (let v = Math.ceil(D.position[0] / step) * step; v <= D.position[1]; v += step) { ctx.fillText(v.toFixed(0).replace("-", "−"), toX(v), bottom + 6); }
  ctx.fillText("gripper position, deg (0 = closed)", (LEFT + width - RIGHT) / 2, bottom + 21);
  ctx.globalAlpha = 0.3;
  LEVELS.forEach((name, level) => {
    ctx.fillStyle = css("--" + name);
    for (const e of D.episodes) {
      for (let i = 0; i < e.n; i++) { if (e.levels[i] === level) { ctx.fillRect(toX(e.state[i]) - 1, toY(e.torque[i]) - 1, 2, 2); } }
    }
  });
  ctx.globalAlpha = 1;
  boundaryLines(ctx, toY, LEFT, width - RIGHT);
  const frame = frameNow();
  ctx.fillStyle = css("--ink"); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(toX(episode().state[frame]), toY(episode().torque[frame]), 5, 0, 7); ctx.fill(); ctx.stroke();
}

// The episode being played: gripper position on top, torque with the threshold below.
function drawDetail() {
  const [ctx, width, height] = fit(detail), e = episode(), frame = frameNow(), last = (e.n - 1) * dt;
  const stripHeight = (height - AXIS - 12) / 2, toX = t => LEFT + t / last * (width - LEFT - RIGHT);
  const strip = (k, range) => {
    const top = k * (stripHeight + 12) + HEAD, bottom = k * (stripHeight + 12) + stripHeight;
    return {top, bottom, toY: v => bottom - (v - range[0]) / (range[1] - range[0]) * (bottom - top)};
  };
  const line = (values, toY, color, lineWidth, alpha, dash) => {
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.globalAlpha = alpha; ctx.setLineDash(dash); ctx.lineJoin = "round";
    ctx.beginPath(); values.forEach((v, i) => ctx.lineTo(toX(i * dt), toY(v))); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  };
  const value = (text, y) => { ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.font = "600 13px system-ui, sans-serif"; ctx.fillStyle = css("--ink"); ctx.fillText(text, width - RIGHT, y); };

  const position = strip(0, D.position), torque = strip(1, D.torque);
  detailMap = {top: torque.top, toTime: x => (x - LEFT) / (width - LEFT - RIGHT) * last,
               toTorque: y => D.torque[0] + (torque.bottom - y) / (torque.bottom - torque.top) * (D.torque[1] - D.torque[0])};
  title(ctx, `Episode ${e.episode}: gripper position`, "deg, command dashed", LEFT, 9);
  value(`${signed(e.state[frame], 2)} deg`, 9);
  yGrid(ctx, D.position, position.toY, LEFT, width - RIGHT, 3);
  line(e.action, position.toY, css("--muted"), 1.2, 1, [4, 3]);
  line(e.state, position.toY, css("--ink2"), 1.6, 1, []);

  title(ctx, "Gripper torque", "N·m, raw faint, smoothed solid", LEFT, torque.top - HEAD + 9);
  value(`${LEVELS[e.levels[frame]]} · ${signed(e.torque[frame], 3)} N·m`, torque.top - HEAD + 9);
  yGrid(ctx, D.torque, torque.toY, LEFT, width - RIGHT, 4);
  bands(ctx, e, toX, position.top, torque.bottom - position.top, 0.25);
  ctx.save(); ctx.beginPath(); ctx.rect(LEFT, torque.top - 4, width - LEFT - RIGHT, torque.bottom - torque.top + 8); ctx.clip();
  line(e.raw, torque.toY, css("--ink2"), 1, 0.35, []);
  line(e.torque, torque.toY, css("--ink"), 1.5, 1, []);
  ctx.restore();
  boundaryLines(ctx, torque.toY, LEFT, width - RIGHT);

  const step = [1, 2, 5, 10, 20, 30, 60].find(s => last / s <= 12) || 120;
  ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = css("--muted");
  for (let t = 0; t <= last; t += step) { ctx.fillText(t + " s", toX(t), height - AXIS + 9); }
  const x = toX(frame * dt);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x, position.top - 4); ctx.lineTo(x, torque.bottom + 4); ctx.stroke();
}

// One row per episode on a shared time axis: the gripper position, with the high stretches shaded.
function drawRows() {
  const [ctx, width, height] = fit(rows), top = HEAD, rowHeight = (height - top - AXIS) / D.episodes.length;
  const toX = t => LEFT + t / longest * (width - LEFT - ROW_RIGHT);
  rowsMap = {top, rowHeight, toTime: x => (x - LEFT) / (width - LEFT - ROW_RIGHT) * longest};
  title(ctx, "Every episode", "line = gripper position (up = closed), shaded = high, right = share of the episode high", LEFT, 9);
  D.episodes.forEach((e, k) => {
    const y0 = top + k * rowHeight, y1 = y0 + rowHeight - 3;
    if (k === current) { ctx.fillStyle = css("--wash"); ctx.fillRect(0, y0 - 1.5, width, rowHeight); }
    ctx.font = (k === current ? "600 " : "") + "11px system-ui, sans-serif"; ctx.textBaseline = "middle";
    ctx.textAlign = "right"; ctx.fillStyle = css(k === current ? "--ink" : "--muted"); ctx.fillText(`ep ${e.episode}`, LEFT - 8, (y0 + y1) / 2);
    ctx.textAlign = "left"; ctx.fillText(`${(100 * e.share).toFixed(0)} %`, width - ROW_RIGHT + 8, (y0 + y1) / 2);
    bands(ctx, e, toX, y0, y1 - y0, 0.6);
    ctx.strokeStyle = css("--ink2"); ctx.lineWidth = 1.1; ctx.lineJoin = "round";
    ctx.beginPath();
    e.state.forEach((v, i) => ctx.lineTo(toX(i * dt), y1 - 1 - (v - D.position[0]) / (D.position[1] - D.position[0]) * (y1 - y0 - 2)));
    ctx.stroke();
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(LEFT, Math.round(y1) + 1.5); ctx.lineTo(toX((e.n - 1) * dt), Math.round(y1) + 1.5); ctx.stroke();
    if (k === current) {
      const x = toX(frameNow() * dt);
      ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, y0 - 1); ctx.lineTo(x, y1 + 1); ctx.stroke();
    }
  });
  const step = [1, 2, 5, 10, 20, 30, 60].find(s => longest / s <= 14) || 120;
  ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = css("--muted");
  for (let t = 0; t <= longest; t += step) { ctx.fillText(t + " s", toX(t), height - AXIS + 9); }
}

function toggle() { if (video.paused) { video.play(); } else { video.pause(); } }
// Whether y is within 7 px of the threshold line.
const lineAt = (map, y) => Math.abs(map.toTorque(y) - bound) < map.toTorque(0) - map.toTorque(7);
const move = value => { bound = value; setLevels(); };

scatter.onmousedown = e => {
  if (lineAt(scatterMap, e.offsetY)) { drag = {canvas: scatter, map: scatterMap}; return; }
  let best = 144, hit = null;
  D.episodes.forEach((ep, k) => {
    for (let i = 0; i < ep.n; i++) {
      const dx = scatterMap.toX(ep.state[i]) - e.offsetX, dy = scatterMap.toY(ep.torque[i]) - e.offsetY;
      if (dx * dx + dy * dy < best) { best = dx * dx + dy * dy; hit = [k, i]; }
    }
  });
  if (hit) { video.pause(); show(hit[0], hit[1] * dt); }
};
detail.onmousedown = e => {
  if (e.offsetY > detailMap.top - 8 && lineAt(detailMap, e.offsetY)) { drag = {canvas: detail, map: detailMap}; return; }
  show(current, detailMap.toTime(e.offsetX));
};
scatter.onmousemove = e => { scatter.style.cursor = lineAt(scatterMap, e.offsetY) ? "ns-resize" : "crosshair"; };
detail.onmousemove = e => { detail.style.cursor = e.offsetY > detailMap.top - 8 && lineAt(detailMap, e.offsetY) ? "ns-resize" : "crosshair"; };
document.addEventListener("mousemove", e => {
  if (drag) { move(drag.map.toTorque(e.clientY - drag.canvas.getBoundingClientRect().top)); }
});
document.addEventListener("mouseup", () => { drag = null; });
rows.onclick = e => {
  const k = Math.floor((e.offsetY - rowsMap.top) / rowsMap.rowHeight);
  if (k >= 0 && k < D.episodes.length) { show(k, rowsMap.toTime(e.offsetX)); }
};
$("bound").onchange = e => { move(Number(e.target.value)); e.target.blur(); };
$("margin").onchange = e => { margin = Math.max(Number(e.target.value), 0); setLevels(); e.target.blur(); };
$("closed").onchange = e => { closed = Math.max(Number(e.target.value), 0); setLevels(); e.target.blur(); };
$("opened").onchange = e => { opened = Math.max(Number(e.target.value), 0); setLevels(); e.target.blur(); };
$("latch").onchange = e => { latch = e.target.checked; setLevels(); e.target.blur(); };
$("play").onclick = toggle;
$("rate").onchange = e => { video.playbackRate = Number(e.target.value); e.target.blur(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey) { return; }
  if (e.code === "Space") { toggle(); }
  else if (e.code === "ArrowRight") { video.pause(); show(current, video.currentTime + (e.shiftKey ? 1 : dt)); }
  else if (e.code === "ArrowLeft") { video.pause(); show(current, video.currentTime - (e.shiftKey ? 1 : dt)); }
  else if (e.code === "ArrowUp") { move(bound + 0.005); }
  else if (e.code === "ArrowDown") { move(bound - 0.005); }
  else { return; }
  e.preventDefault();
});
function tick() {
  const frame = frameNow();
  if (dirty || frame !== drawnFrame) {
    dirty = false; drawnFrame = frame;
    drawScatter(); drawDetail(); drawRows();
    $("time").textContent = `${(frame * dt).toFixed(2)} s · frame ${frame}`;
    $("play").textContent = video.paused ? "Play" : "Pause";
  }
  requestAnimationFrame(tick);
}
video.onplay = video.onpause = () => { dirty = true; };
new ResizeObserver(() => { dirty = true; }).observe($("right"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { dirty = true; });
setLevels();
show(0, 0);
requestAnimationFrame(tick);
</script>
"""

page_episodes = []
for ep in sorted(data.episode_index.unique().tolist()):
    df = data[data.episode_index == ep].sort_values("frame_index")
    episode = episodes.loc[ep]
    clip = f"{stem}_ep{ep}.mp4"

    # Both views of the episode in one clip: the browser plays one stream, so the views stay frame-aligned.
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for key in video_keys:
        path = root / info["video_path"].format(
            video_key=key, chunk_index=episode[f"videos/{key}/chunk_index"], file_index=episode[f"videos/{key}/file_index"]
        )
        command += ["-ss", str(episode[f"videos/{key}/from_timestamp"]), "-t", str(len(df) / fps), "-i", str(path)]
    command += ["-filter_complex", f"vstack=inputs={len(video_keys)}", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-g", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_dir / clip)]
    if not (out_dir / clip).exists():
        subprocess.run(command, check=True)

    raw = np.stack(df["observation.effort"].to_numpy())[:, gripper]
    # Causal: frame i uses frames i-3..i. The first frames repeat the episode's first sample.
    torque = np.convolve(np.concatenate([np.full(len(WEIGHTS) - 1, raw[0]), raw]), WEIGHTS, "valid")
    page_episodes.append(dict(
        episode=ep, n=len(df), clip=clip, raw=np.round(raw, 3).tolist(), torque=np.round(torque, 4).tolist(),
        state=np.round(np.stack(df["observation.state"].to_numpy())[:, gripper], 2).tolist(),
        action=np.round(np.stack(df["action"].to_numpy())[:, gripper], 2).tolist(),
    ))


def padded(low, high):
    return [low - 0.04 * (high - low), high + 0.04 * (high - low)]


# The torque axis covers the middle 99.6 % of the frames: the few hard squeezes would flatten everything else.
torques = np.concatenate([e["torque"] for e in page_episodes])
positions = np.concatenate([e["state"] + e["action"] for e in page_episodes])
page_data = dict(fps=fps, threshold=args.threshold, margin=args.margin, closed=args.closed, opened=args.opened, episodes=page_episodes, torque=padded(*np.percentile(torques, [0.2, 99.8]).tolist()),
                 position=padded(positions.min(), positions.max()))
page = PAGE.replace("__TITLE__", stem).replace("__DATA__", json.dumps(page_data))
(out_dir / f"{stem}.html").write_text(page)
print(out_dir / f"{stem}.html")

if not args.no_open:
    webbrowser.open((out_dir / f"{stem}.html").resolve().as_uri())

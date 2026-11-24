"""Text probe chat page: ask every checkpoint about an image, or make it act on a robot frame.

    uv run python lerobot/adhocs/text_probe/chat.py --config_path=config_rl_validate.yaml

then open http://localhost:8770. All checkpoints are loaded once and stay on the GPU. The image choices are
the top-camera validation frames listed in ``frames``, the frames ``clutter_frames`` of ``clutter_root`` (an
old cluttered recording) and every file of ``image_dir``.

Ask: the prompt and the decoding are those of ``run_text_probe.answer`` (plain chat turn, greedy, action
expert never called). The answer is shown raw, action tokens included.

Act: each line of the instruction box is put in the deployed action prompt as the subtask
(or as the task; "original MolmoAct2" drops the step, velocity, torque and metadata clauses) and decoded twice, action tokens through ``lm_head`` (greedy) and the action expert (one
chunk per flow seed). The chunks are drawn the way the action inspector (``probes/action_trace_probe``) draws
them: deployed low-pass, reBot kinematics, tip paths in one 3-D scene, next to the arm as it stands and the
recorded chunk. When the action tokens do not decode to a chunk, the cell says why; the raw tokens are shown
either way. The old root has no velocity or torque column: its prompt gets zero velocity, the level of zero
torque and no tip target clause. A photo is acted on as the top camera of ``body_frame`` (letterboxed to the
camera size); the arm state and the wrist camera stay those of that frame, and there is no recorded chunk.
Layers: two runs (checkpoint, image, chat or action prompt, text) are forwarded once each and the output of
every language-model block is compared position by position with a raw cosine (``layer_capture.py``): map over
layers and positions, mean and distribution per kind of token, the same for the action expert's blocks when
both runs are action prompts, and inside each run the mean cosine between positions, the participation ratio
and a position-by-position matrix. The last ``kept_runs`` runs stay in memory.

With no image picked, Act is text only: the state is that of ``body_frame`` and both cameras are marked absent
(``camera_is_present`` false), which removes their patches from the attention mask.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import probe_frame_inputs, register_config_choices
from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.action_smoothing import apply_butterworth_filter
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.gripper_torque import GRIPPER_TORQUE_LEVEL_KEY, gripper_torque_levels
from lerobot.utils.motor_velocity import VELOCITY_KEY
from lerobot.utils.tip_target import TIP_TARGET_KEY
from lerobot.utils.utils import init_logging
from layer_capture import between, capture, label_tokens, paired, within
from run_text_probe import IMAGE_KEY, TextProbeConfig, answer, label

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>text probe</title>
<script src="plotly.js"></script>
<style>
  :root { color-scheme: dark; --bg: #15171a; --panel: #1d2024; --line: #30353c; --text: #e6e8eb; --dim: #8b929c;
          --accent: #4c9aff; --act: #f0a04b; --bad: #ff6b6b; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, sans-serif; }
  #layout { display: grid; grid-template-columns: 330px 1fr; min-height: 100vh; }
  #side { background: var(--panel); border-right: 1px solid var(--line); padding: 14px; position: sticky; top: 0;
          height: 100vh; overflow-y: auto; }
  #main { padding: 18px 22px 60px; min-width: 0; }
  h2 { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--dim); margin: 18px 0 8px; font-weight: 600; }
  h2:first-child { margin-top: 0; }
  h2 button { float: right; }
  .chips { display: flex; flex-direction: column; gap: 6px; }
  .chip { border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; cursor: pointer; color: var(--dim);
          overflow-wrap: anywhere; user-select: none; }
  .chip.on { border-color: var(--accent); color: var(--text); background: #4c9aff1c; }
  .thumbs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
  .thumbs figure { margin: 0; cursor: pointer; border: 2px solid transparent; border-radius: 6px; overflow: hidden; opacity: .6; }
  .thumbs figure:hover { opacity: .9; }
  .thumbs figure.on { border-color: var(--accent); opacity: 1; }
  .thumbs img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; display: block; }
  .thumbs figcaption { font-size: 10px; text-align: center; color: var(--dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 1px 2px; }
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--line); margin-bottom: 14px; }
  .tab { padding: 8px 18px; cursor: pointer; color: var(--dim); border-bottom: 2px solid transparent; font-weight: 600; }
  .tab.on { color: var(--text); border-color: var(--accent); }
  .pane { display: none; } .pane.on { display: block; }
  .hint { color: var(--dim); font-size: 13px; margin: 0 0 8px; }
  textarea, input[type=text], input[type=number], select { background: var(--panel); color: var(--text); border: 1px solid var(--line);
          border-radius: 8px; font: inherit; padding: 7px 10px; }
  textarea { width: 100%; resize: vertical; }
  textarea:focus, input:focus { outline: 1px solid var(--accent); }
  .row { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin: 10px 0 0; }
  .row label { color: var(--dim); font-size: 13px; display: flex; gap: 6px; align-items: center; }
  .row input[type=number] { width: 5.5em; } .row input[type=text] { width: 15em; }
  button { font: inherit; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); color: var(--text); padding: 6px 12px; cursor: pointer; }
  button.go { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; padding: 7px 22px; }
  button.small { padding: 1px 8px; font-size: 11px; color: var(--dim); text-transform: none; letter-spacing: 0; }
  #status { color: var(--dim); font-size: 13px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; margin-top: 18px; overflow: hidden; }
  .card header { display: flex; gap: 10px; align-items: flex-start; padding: 10px 14px; border-bottom: 1px solid var(--line); }
  .card header .what { flex: 1; white-space: pre-wrap; font-weight: 600; }
  .card header .note { color: var(--dim); font-weight: 400; font-size: 12px; display: block; }
  .badge { font-size: 11px; font-weight: 700; border-radius: 5px; padding: 1px 7px; background: #4c9aff33; color: var(--accent); }
  .badge.act { background: #f0a04b33; color: var(--act); }
  table { border-collapse: collapse; width: 100%; table-layout: fixed; }
  th { text-align: left; font-size: 12px; color: var(--dim); font-weight: 600; padding: 8px 12px; overflow-wrap: anywhere; }
  td { border-top: 1px solid var(--line); padding: 10px 12px; vertical-align: top; }
  th:first-child, td:first-child { width: 170px; }
  td:first-child img { width: 100%; border-radius: 6px; display: block; }
  td:first-child span { font-size: 11px; color: var(--dim); }
  td.cell { white-space: pre-wrap; overflow-wrap: anywhere; }
  td.wait { color: var(--dim); } td.busy { color: var(--accent); }
  .tok { color: var(--act); }
  .error { color: var(--bad); }
  .sub { display: block; margin-top: 8px; font-size: 12px; color: var(--dim); white-space: normal; }
  .plot { height: 440px; }
  .line { margin-top: 8px; font-size: 13px; white-space: normal; }
  .line b { font-weight: 600; }
  [hidden] { display: none !important; }
  #zoom { position: fixed; z-index: 10; top: 50%; left: calc(330px + (100vw - 330px) / 2); transform: translate(-50%, -50%); max-width: calc(100vw - 380px);
          max-height: 92vh; border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 10px 50px #000c; pointer-events: none; }
  #zoom.pinned { pointer-events: auto; cursor: zoom-out; }
  td:first-child img { cursor: zoom-in; }
  .form { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
  .form textarea, .form input[type=text], .form input[type=number], .form select { background: var(--bg); }
  .field { display: block; font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--dim); font-weight: 600; margin: 12px 0 5px; }
  .field:first-child { margin-top: 0; }
  .seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .seg div { padding: 6px 14px; cursor: pointer; color: var(--dim); border-right: 1px solid var(--line); user-select: none; }
  .seg div:last-child { border-right: 0; }
  .seg div.on { background: #4c9aff26; color: var(--text); }
  #preview { font: 12px/1.6 ui-monospace, monospace; color: var(--dim); background: var(--bg); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  #preview b { color: var(--accent); font-weight: 600; } #preview i { color: var(--act); font-style: normal; }
  .res { margin-top: 10px; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: 13px; white-space: normal; }
  .res .name { font-weight: 600; }
  .res .state { font-size: 12px; margin: 2px 0 6px; }
  .ok { color: #3ecf8e; }
  .raw { font: 11px/1.45 ui-monospace, monospace; overflow-wrap: anywhere; max-height: 120px; overflow-y: auto; background: var(--bg); border-radius: 6px; padding: 6px 8px; }
  .duo { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .run { border: 1px solid var(--line); border-radius: 10px; padding: 12px; border-top: 3px solid var(--accent); }
  .run.b { border-top-color: var(--act); }
  .run .letter { font-weight: 700; font-size: 15px; color: var(--accent); } .run.b .letter { color: var(--act); }
  .run .pick { display: flex; gap: 10px; align-items: flex-start; margin-top: 8px; }
  .run .pick img { width: 96px; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 6px; cursor: zoom-in; }
  .run select { max-width: 100%; } .run textarea { margin-top: 8px; }
  .run .cols { display: flex; flex-direction: column; gap: 8px; flex: 1; min-width: 0; }
  .body { padding: 14px; }
  .body h3 { font-size: 13px; margin: 22px 0 2px; font-weight: 600; } .body h3:first-child { margin-top: 0; }
  .body p { margin: 0 0 6px; color: var(--dim); font-size: 12px; }
  .info { font: 12px/1.5 ui-monospace, monospace; color: var(--text); background: var(--bg); border-radius: 6px; padding: 5px 9px; min-height: 28px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .key { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; color: var(--dim); margin: 6px 0; }
  .key i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
  .trio { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .spin { display: inline-block; width: 11px; height: 11px; border: 2px solid var(--line); border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite; vertical-align: -1px; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
<img id="zoom" hidden>
<div id="layout">
  <aside id="side">
    <h2>Checkpoints</h2>
    <div class="chips" id="checkpoints"></div>
    <h2>Images <button class="small" id="clear">clear</button></h2>
    <p class="hint" id="count"></p>
    <div id="groups"></div>
  </aside>
  <main id="main">
    <div class="tabs"><div class="tab on" data-pane="ask">Ask</div><div class="tab" data-pane="act">Act</div><div class="tab" data-pane="layers">Layers</div></div>
    <div class="pane on form" id="ask">
      <span class="field">Prompt · plain chat turn on each picked image (none picked: text only)</span>
      <textarea id="prompt" rows="3" placeholder="Ctrl+Enter to run">Describe this image.</textarea>
      <div class="row">
        <button class="go" id="run">Ask</button>
        <label>max new tokens <input id="tokens" type="number" value="96"></label>
        <label><input id="mask" type="checkbox"> forbid action tokens</label>
        <label>assistant starts with <input id="prefix" type="text"></label>
      </div>
    </div>
    <div class="pane form" id="act">
      <span class="field">Instructions · one per line, each becomes one path per picked image (none picked: text only, no camera)</span>
      <textarea id="lines" rows="3" placeholder="Ctrl+Enter to run">grasp the purple bottle
grasp the cup</textarea>
      <span class="field">Where the instruction goes</span>
      <div class="row" style="margin: 0">
        <div class="seg" id="style">
          <div data-style="step" class="on">our prompt, as the step</div>
          <div data-style="task">our prompt, as the task</div>
          <div data-style="original">original MolmoAct2 prompt, as the task</div>
        </div>
        <label id="tasklabel">task <input id="task" type="text" placeholder="the frame's own task"></label>
      </div>
      <span class="field">Prompt the model gets (after the images)</span>
      <div id="preview"></div>
      <div class="row">
        <button class="go" id="go">Act</button>
        <label><input id="decode" type="checkbox" checked> action tokens too (a few seconds per instruction)</label>
        <label>action expert samples <input id="seeds" type="number" value="1" min="1"></label>
      </div>
    </div>
    <div class="pane form" id="layers">
      <span class="field">Two runs, compared position by position at every layer (raw cosine of the hidden vectors)</span>
      <div class="duo" id="duo"></div>
      <div class="row">
        <button class="go" id="compare">Compare</button>
        <button class="small" id="copy">copy A to B</button>
        <label>layers for the distributions <input id="picks" type="text" value="0,7,14,21,28,35"></label>
      </div>
    </div>
    <div class="row"><span id="status"></span><button class="small" id="stop" hidden>stop</button><button class="small" id="wipe">remove all results</button></div>
    <div id="cards"></div>
  </main>
</div>
<script>
const IMAGES = __IMAGES__, CHECKPOINTS = __CHECKPOINTS__;
const COLOURS = ["#4c9aff", "#ff6b6b", "#3ecf8e", "#f0a04b", "#c084fc", "#22d3ee", "#f472b6", "#a3e635"];
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const tokens = text => esc(text).replace(/(?:&lt;action_\w+&gt;)+/g, m => `<span class="tok">${m}</span>`);
const picked = new Set(), models = new Set(CHECKPOINTS);
let ticket = 0;

// sidebar
for (const name of CHECKPOINTS) {
  const chip = document.createElement("div");
  chip.className = "chip on";
  chip.textContent = name;
  chip.onclick = () => { models.has(name) ? models.delete(name) : models.add(name); chip.classList.toggle("on"); };
  $("checkpoints").append(chip);
}
const count = () => { $("count").textContent = picked.size ? `${picked.size} picked` : "none picked"; };
for (const group of [...new Set(IMAGES.map(i => i.group))]) {
  const title = document.createElement("h2"), box = document.createElement("div");
  title.textContent = group;
  box.className = "thumbs";
  for (const image of IMAGES.filter(i => i.group === group)) {
    const figure = document.createElement("figure");
    figure.innerHTML = `<img src="image/${encodeURIComponent(image.name)}" loading="lazy"><figcaption>${esc(image.name)}</figcaption>`;
    figure.onclick = () => { picked.has(image.name) ? picked.delete(image.name) : picked.add(image.name); figure.classList.toggle("on"); count(); };
    box.append(figure);
  }
  $("groups").append(title, box);
}
$("clear").onclick = () => { picked.clear(); document.querySelectorAll(".thumbs figure").forEach(f => f.classList.remove("on")); count(); };
count();
for (const tab of document.querySelectorAll(".tab")) tab.onclick = () => {
  document.querySelectorAll(".tab, .pane").forEach(e => e.classList.remove("on"));
  tab.classList.add("on");
  $(tab.dataset.pane).classList.add("on");
};

// hovering any image shows it large next to the sidebar; clicking one in a result keeps it up until the next click
let pinned = false;
const zoom = (src, pin) => { pinned = pin; $("zoom").src = src || ""; $("zoom").hidden = !src; $("zoom").classList.toggle("pinned", pin); };
document.addEventListener("mouseover", e => { if (!pinned && e.target.matches(".thumbs img, td img, .run img")) zoom(e.target.src, false); });
document.addEventListener("mouseout", e => { if (!pinned && e.target.matches(".thumbs img, td img, .run img")) zoom("", false); });
document.addEventListener("click", e => { if (pinned) zoom("", false); else if (e.target.matches("td img")) zoom(e.target.src, true); });

async function post(url, body) {
  try {
    const reply = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!reply.ok) return { error: `server answered ${reply.status}, see its terminal` };
    return await reply.json();
  } catch (error) { return { error: `no answer from the server (${error.message})` }; }
}

// one result card: a row per image, a column per checkpoint, cells filled one by one
function card(kind, what, note, images, checkpoints) {
  const element = document.createElement("div");
  element.className = "card";
  element.innerHTML = `<header><span class="badge ${kind}">${kind}</span><span class="what">${esc(what)}<span class="note">${esc(note)}</span></span>
    <button class="small">remove</button></header>
    <table><tr><th>image</th>${checkpoints.map(c => `<th>${esc(c)}</th>`).join("")}</tr>` +
    images.map(n => `<tr><td>${n ? `<img src="image/${encodeURIComponent(n)}"><span>${esc(n)}</span>` : "<span>no image</span>"}</td>` +
      checkpoints.map(() => `<td class="cell wait">waiting</td>`).join("") + "</tr>").join("") + "</table>";
  element.querySelector("button").onclick = () => element.remove();
  $("cards").prepend(element);
  return (i, j) => element.querySelectorAll("tr")[i + 1].children[j + 1];
}

async function each(images, checkpoints, cell, work) {
  const mine = ++ticket, total = images.length * checkpoints.length;
  let done = 0;
  $("stop").hidden = false;
  for (const [i, image] of images.entries()) for (const [j, checkpoint] of checkpoints.entries()) {
    if (mine !== ticket) return;
    $("status").innerHTML = `<span class="spin"></span>running ${done + 1} of ${total}`;
    const box = cell(i, j);
    box.className = "cell busy";
    box.innerHTML = `<span class="spin"></span>running`;
    try { await work(box, image, checkpoint); } catch (error) { fail(box, `page error: ${error.message}`); }
    box.classList.remove("busy");
    done++;
  }
  if (mine === ticket) { $("status").textContent = "done"; $("stop").hidden = true; }
}
$("stop").onclick = () => { ticket++; $("status").textContent = "stopped"; $("stop").hidden = true; };
$("wipe").onclick = () => { $("cards").innerHTML = ""; };
const fail = (box, error) => { box.innerHTML = `<span class="error">${esc(error)}</span>`; };

function ask() {
  const checkpoints = CHECKPOINTS.filter(c => models.has(c));
  if (!checkpoints.length) { $("status").textContent = "pick a checkpoint"; return; }
  const images = picked.size ? IMAGES.map(i => i.name).filter(n => picked.has(n)) : [null];
  const body = { prompt: $("prompt").value, prefix: $("prefix").value, mask: $("mask").checked, max_new_tokens: +$("tokens").value };
  const note = [body.mask ? "action tokens forbidden" : "", body.prefix ? `assistant starts with ${body.prefix}` : ""].filter(s => s).join(" · ");
  const cell = card("ask", body.prompt, note, images, checkpoints);
  each(images, checkpoints, cell, async (box, image, checkpoint) => {
    const out = await post("answer", { ...body, image, checkpoint });
    if (out.error) return fail(box, out.error);
    box.innerHTML = (out.text ? tokens(out.text) : `<span class="sub">(empty answer)</span>`) +
      `<span class="sub">most likely first tokens: ${out.top.map(([token, p]) => `${esc(JSON.stringify(token))} ${p.toFixed(2)}`).join(", ")}</span>`;
  });
}

function path3d(points, name, colour, extra) {
  return { type: "scatter3d", x: points.map(p => p[0]), y: points.map(p => p[1]), z: points.map(p => p[2]), name,
    mode: "lines+markers", line: { color: colour, width: 6 }, marker: { size: 2.5, color: colour },
    hovertemplate: `<b>${esc(name)}</b><br>step %{pointNumber}<br>x %{x:.1f} · y %{y:.1f} · z %{z:.1f} cm<extra></extra>`, ...extra };
}

function act() {
  const checkpoints = CHECKPOINTS.filter(c => models.has(c));
  const images = picked.size ? IMAGES.filter(i => picked.has(i.name)).map(i => i.name) : [null];
  const lines = $("lines").value.split("\n").map(l => l.trim()).filter(l => l);
  if (!checkpoints.length || !images.length || !lines.length) { $("status").textContent = !checkpoints.length ? "pick a checkpoint" : "write an instruction"; return; }
  const body = { lines, style, task: style === "step" ? $("task").value : "", seeds: Math.max(1, +$("seeds").value) };
  const decode = $("decode").checked;
  const cell = card("act", lines.join("\n"), document.querySelector("#style .on").textContent + (body.task ? ` · task: ${body.task}` : ""), images, checkpoints);
  each(images, checkpoints, cell, async (box, image, checkpoint) => {
    const out = await post("act", { ...body, image, checkpoint, decoder: "expert" });
    if (out.error) return fail(box, out.error);
    const traces = [
      path3d(out.arm, "arm now", "#8b929c", { line: { color: "#8b929c", width: 9 }, marker: { size: 4, color: "#8b929c" }, hoverinfo: "name", hovertemplate: null }),
    ];
    if (out.recorded) traces.push(path3d(out.recorded, "recorded", "#d4d7dc", { line: { color: "#d4d7dc", width: 3, dash: "dot" }, marker: { size: 1.5, color: "#d4d7dc" } }));
    for (const [k, line] of out.lines.entries())
      line.expert.forEach((points, seed) => traces.push(path3d(points, `${line.line} · action expert${seed ? " " + (seed + 1) : ""}`, COLOURS[k % COLOURS.length],
        { legendgroup: `e${k}`, showlegend: seed === 0, opacity: seed ? 0.5 : 1 })));
    box.innerHTML = `<div class="plot"></div>` + out.lines.map((line, k) =>
      `<div class="res"><div class="name" style="color: ${COLOURS[k % COLOURS.length]}">${esc(line.line)}</div><div class="state">` +
      (decode ? `<span class="spin"></span>action tokens: generating` : `<span class="sub" style="margin: 0">action tokens: not asked</span>`) + `</div></div>`).join("") +
      `<span class="sub">solid = action expert · dashed = action tokens · ` +
      (out.recorded ? `dotted = recorded chunk (${esc(out.subtask || out.task)}) · grey = arm now` : (image ? `grey = arm (photo as the top camera; arm state and wrist camera from __BODY__)` : `grey = arm (no image: both cameras hidden from the model; arm state from __BODY__)`)) + `</span>`;
    const axis = title => ({ title, color: "#8b929c", gridcolor: "#30353c", backgroundcolor: "#1d2024", showbackground: true });
    const plot = box.querySelector(".plot"), results = box.querySelectorAll(".res");
    await Plotly.newPlot(plot, traces, {
      margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#e6e8eb", size: 11 },
      legend: { orientation: "h", y: 0, yanchor: "top", font: { size: 11 } },
      scene: { aspectmode: "data", xaxis: axis("x (cm)"), yaxis: axis("y (cm)"), zaxis: axis("z (cm)"), camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } } },
    }, { displaylogo: false, responsive: true });
    if (!decode) return;
    // the action tokens are written one at a time (seconds per instruction), so each is added after the action expert is on screen
    for (const [k, line] of out.lines.entries()) {
      const state = results[k].querySelector(".state"), colour = COLOURS[k % COLOURS.length], end = line.expert[0].at(-1);
      const slow = await post("act", { ...body, lines: [line.line], image, checkpoint, decoder: "tokens" });
      if (slow.error) { state.innerHTML = `<span class="error">action tokens: ${esc(slow.error)}</span>`; continue; }
      const got = slow.lines[0];
      if (got.tokens) await Plotly.addTraces(plot, path3d(got.tokens, `${line.line} · action tokens`, colour,
        { line: { color: colour, width: 5, dash: "dash" }, marker: { size: 3.5, color: colour, symbol: "diamond" } }));
      state.innerHTML = got.tokens
        ? `<span class="ok">action tokens: valid chunk</span>, ends ${Math.hypot(...got.tokens.at(-1).map((v, i) => v - end[i])).toFixed(1)} cm from the action expert's end`
        : `<span class="error">action tokens: no valid chunk</span> · ${esc(got.tokens_error)}`;
      results[k].insertAdjacentHTML("beforeend", `<div class="raw">${got.raw ? tokens(got.raw) : "(nothing generated)"}</div>`);
    }
  });
}

// what each choice renders; {line} is the instruction, shown with the first line of the box
// ---- Layers: two runs, hidden states compared
const GROUP_COLOURS = { image: "#4c9aff", text: "#e6e8eb", special: "#8b929c", task: "#3ecf8e", subtask: "#a3e635", state: "#f0a04b", velocity: "#f472b6",
  gripper_torque: "#c084fc", question: "#22d3ee", embodiment: "#facc15", control_mode: "#fb923c", metadata: "#94a3b8", history: "#2dd4bf", masked: "#30353c", all: "#ffffff" };
const colourOf = group => GROUP_COLOURS[group] || "#d4d7dc";
const STYLES = { step: "action prompt, text = the step", task: "action prompt, text = the task", original: "original MolmoAct2 action prompt, text = the task" };
for (const [k, letter] of ["a", "b"].entries()) {
  const run = document.createElement("div");
  run.className = `run ${letter}`;
  run.id = `run_${letter}`;
  run.innerHTML = `<span class="letter">${letter.toUpperCase()}</span>
    <div class="pick"><img hidden><div class="cols">
      <select class="checkpoint">${CHECKPOINTS.map((c, i) => `<option${i === Math.min(k, CHECKPOINTS.length - 1) ? " selected" : ""}>${esc(c)}</option>`).join("")}</select>
      <select class="image"><option value="">no image</option>${IMAGES.map(i => `<option${i.name === IMAGES[0].name ? " selected" : ""}>${esc(i.name)}</option>`).join("")}</select>
      <select class="kind"><option value="chat">chat prompt (as in Ask)</option>${Object.entries(STYLES).map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select>
    </div></div>
    <textarea class="text" rows="2" placeholder="prompt, or instruction for an action prompt">Describe this image.</textarea>`;
  const show = () => { const img = run.querySelector("img"), name = run.querySelector(".image").value; img.hidden = !name; if (name) img.src = `image/${encodeURIComponent(name)}`; };
  run.querySelector(".image").onchange = show;
  $("duo").append(run);
  show();
}
const spec = letter => { const run = $(`run_${letter}`), kind = run.querySelector(".kind").value;
  return { checkpoint: run.querySelector(".checkpoint").value, image: run.querySelector(".image").value || null, text: run.querySelector(".text").value,
    kind: kind === "chat" ? "chat" : "action", style: kind === "chat" ? "" : kind, task: "" }; };
const tell = s => `${s.checkpoint} · ${s.image || "no image"} · ${s.kind === "chat" ? "chat" : STYLES[s.style]} · “${s.text}”`;
$("copy").onclick = () => { for (const field of ["checkpoint", "image", "kind", "text"]) $("run_b").querySelector("." + field).value = $("run_a").querySelector("." + field).value;
  $("run_b").querySelector(".image").onchange(); };
const DARK = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#15171a", font: { color: "#e6e8eb", size: 11 }, margin: { l: 48, r: 12, t: 10, b: 40 } };
const ax = (title, extra) => ({ title: { text: title, font: { size: 11 } }, gridcolor: "#30353c", zerolinecolor: "#30353c", linecolor: "#30353c", ...extra });
const draw = (element, traces, layout) => Plotly.newPlot(element, traces, { ...DARK, ...layout }, { displaylogo: false, responsive: true });
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;

async function compare() {
  const a = spec("a"), b = spec("b"), mine = ++ticket;
  const element = document.createElement("div");
  element.className = "card";
  element.innerHTML = `<header><span class="badge" style="background: #3ecf8e33; color: #3ecf8e">layers</span>
    <span class="what"><span style="color: var(--accent)">A</span>  ${esc(tell(a))}\n<span style="color: var(--act)">B</span>  ${esc(tell(b))}</span>
    <button class="small">remove</button></header><div class="body"><span class="spin"></span>running both forwards</div>`;
  element.querySelector("button").onclick = () => element.remove();
  $("cards").prepend(element);
  $("status").innerHTML = `<span class="spin"></span>running`;
  const out = await post("layers", { a, b }), body = element.querySelector(".body");
  if (mine === ticket) $("status").textContent = "done";
  if (out.error) return fail(body, out.error);
  const layers = out.cos.map((_, l) => l), shared = out.cos[0].length;
  const picks = $("picks").value.split(",").map(v => +v.trim()).filter(l => Number.isInteger(l) && l >= 0 && l < layers.length);
  const groups = out.first.map(i => out.a.groups[i]), names = [...new Set(groups)], seen = names;
  const left = (run, used) => { const count = {}; run.groups.forEach((g, i) => { if (g !== "masked" && !used.has(i)) count[g] = (count[g] || 0) + 1; });
    return Object.entries(count).map(([g, n]) => `${n} ${g}`).join(", "); };
  const leftA = left(out.a, new Set(out.first)), leftB = left(out.b, new Set(out.second));
  body.innerHTML = `
    <h3>A against B, every layer and position</h3>
    <p>${shared} input positions matched by kind (k-th image patch with k-th image patch, k-th state token with k-th state token, ...), in A's order.
       ${leftA || leftB ? `Not matched: ${leftA ? `A ${esc(leftA)}` : ""}${leftA && leftB ? " · " : ""}${leftB ? `B ${esc(leftB)}` : ""}.` : "Nothing left out."}
       The strip on top is the kind of token. Hover for the tokens.</p>
    <div class="key">${names.map(g => `<span><i style="background: ${colourOf(g)}"></i>${esc(g)} (${groups.filter(x => x === g).length})</span>`).join("")}</div>
    <div class="heat" style="height: 420px"></div><div class="info heatinfo">hover the map</div>
    <h3>Mean cosine per layer, by kind of token</h3>
    <div class="means" style="height: 300px"></div>
    <h3>Distribution over positions at layers ${picks.join(", ")}</h3>
    <div class="boxes" style="height: 340px"></div>
    <h3>Inside each run: are the positions collapsing onto each other?</h3>
    <p>Left: mean cosine between two different positions of the same run (1 = all the same vector); click a kind in the legend to show it. Middle: number of directions the positions spread over (participation ratio; 1 = one line). Right: mean vector length. Solid = A, dashed = B.</p>
    <div class="trio"><div class="pair" style="height: 280px"></div><div class="spread" style="height: 280px"></div><div class="norm" style="height: 280px"></div></div>
    <h3>Inside one run: position against position</h3>
    <div class="row" style="margin: 0 0 8px"><label>run <select class="mrun"><option value="a">A</option><option value="b">B</option></select></label>
      <label>layer <select class="mlayer">${layers.map(l => `<option${l === picks[picks.length - 1] ? " selected" : ""}>${l}</option>`).join("")}</select></label>
      <label><input class="mimages" type="checkbox"> with image patches</label><button class="small mgo">show</button></div>
    <div class="matrix"></div><div class="info matrixinfo" hidden></div>` +
    (out.expert ? `<h3>Action expert: A against B, every layer and action position (first flow step, same noise)</h3><div class="expert" style="height: 320px"></div>` : "");
  const at = name => body.querySelector("." + name);

  const strip = { type: "heatmap", z: [groups.map(g => names.indexOf(g))], x: groups.map((_, i) => i), y: [0], yaxis: "y2", showscale: false, hoverinfo: "skip",
    zmin: 0, zmax: Math.max(1, names.length - 1), colorscale: names.length > 1 ? names.map((g, i) => [i / (names.length - 1), colourOf(g)]) : [[0, colourOf(names[0])], [1, colourOf(names[0])]] };
  await draw(at("heat"), [{ type: "heatmap", z: out.cos, x: groups.map((_, i) => i), y: layers, colorscale: "RdBu", zmin: -1, zmax: 1, hoverinfo: "none", colorbar: { thickness: 10, title: { text: "cos", side: "right" } } }, strip],
    { xaxis: ax("matched position"), yaxis: ax("layer", { domain: [0, 0.93], autorange: "reversed" }), yaxis2: { domain: [0.95, 1], visible: false }, margin: { l: 48, r: 12, t: 6, b: 40 } });
  at("heat").on("plotly_hover", event => { const point = event.points.find(p => p.curveNumber === 0); if (!point) return;
    at("heatinfo").textContent = `layer ${point.y} · ${groups[point.x]} · cos ${point.z.toFixed(3)}   A position ${out.first[point.x]} ${JSON.stringify(out.a.tokens[out.first[point.x]])}   B position ${out.second[point.x]} ${JSON.stringify(out.b.tokens[out.second[point.x]])}`; });

  const lines = seen.map(g => { const index = groups.map((x, i) => x === g ? i : -1).filter(i => i >= 0);
    return { type: "scatter", mode: "lines+markers", name: g, x: layers, y: out.cos.map(row => mean(index.map(i => row[i]))), line: { color: colourOf(g), width: 2 }, marker: { size: 4 } }; });
  draw(at("means"), lines, { xaxis: ax("layer"), yaxis: ax("cosine A · B"), legend: { orientation: "h", y: 1.12 } });

  draw(at("boxes"), seen.map(g => { const index = groups.map((x, i) => x === g ? i : -1).filter(i => i >= 0);
    return { type: "box", name: g, x: picks.flatMap(l => index.map(() => `layer ${l}`)), y: picks.flatMap(l => index.map(i => out.cos[l][i])), marker: { color: colourOf(g), size: 2 }, line: { width: 1 }, boxpoints: "outliers" }; }),
    { boxmode: "group", xaxis: ax(""), yaxis: ax("cosine A · B"), legend: { orientation: "h", y: 1.12 } });

  const both = [["A", out.a, "solid"], ["B", out.b, "dash"]];
  draw(at("pair"), both.flatMap(([letter, run, dash]) => Object.entries(run.pair).map(([g, values]) =>
    ({ type: "scatter", mode: "lines", name: `${letter} ${g}`, x: layers, y: values, line: { color: colourOf(g), dash, width: g === "all" ? 3 : 1.5 }, visible: g === "all" ? true : "legendonly" }))),
    { xaxis: ax("layer"), yaxis: ax("mean cosine between positions"), legend: { font: { size: 10 } } });
  const twin = key => both.map(([letter, run, dash], k) => ({ type: "scatter", mode: "lines", name: letter, x: layers, y: run[key], line: { color: k ? "#f0a04b" : "#4c9aff", dash, width: 2.5 } }));
  draw(at("spread"), twin("spread"), { xaxis: ax("layer"), yaxis: ax("directions spread over") });
  draw(at("norm"), twin("norm"), { xaxis: ax("layer"), yaxis: ax("mean vector length", { type: "log" }) });

  at("mgo").onclick = async () => {
    at("matrix").style.height = "auto";
    at("matrix").innerHTML = `<span class="spin"></span>running`;
    const got = await post("matrix", { run: at("mrun").value === "a" ? a : b, layer: +at("mlayer").value, images: at("mimages").checked });
    if (got.error) return fail(at("matrix"), got.error);
    at("matrix").innerHTML = "";
    at("matrix").style.height = "560px";
    at("matrixinfo").hidden = false;
    at("matrixinfo").textContent = `${got.tokens.length} positions · hover the map`;
    const index = got.tokens.map((_, i) => i);
    await draw(at("matrix"), [{ type: "heatmap", z: got.z, x: index, y: index, colorscale: "RdBu", zmin: -1, zmax: 1, hoverinfo: "none", colorbar: { thickness: 10 } }],
      { xaxis: ax("position", { constrain: "domain" }), yaxis: ax("position", { autorange: "reversed", scaleanchor: "x" }) });
    at("matrix").on("plotly_hover", event => { const point = event.points[0];
      at("matrixinfo").textContent = `cos ${point.z.toFixed(2)}   ${got.positions[point.y]} ${got.groups[point.y]} ${JSON.stringify(got.tokens[point.y])}   ×   ${got.positions[point.x]} ${got.groups[point.x]} ${JSON.stringify(got.tokens[point.x])}`; });
  };

  if (out.expert) draw(at("expert"), [{ type: "heatmap", z: out.expert, y: layers, colorscale: "RdBu", zmin: -1, zmax: 1, colorbar: { thickness: 10 },
    hovertemplate: "layer %{y} · action position %{x} · cos %{z:.3f}<extra></extra>" }], { xaxis: ax("action position"), yaxis: ax("layer", { autorange: "reversed" }) });
}
$("compare").onclick = compare;

const TEMPLATES = {
  step: "The task is to {task}. The current step is {line}. The current state of the robot is {state}. The current velocity of the robot is {velocity}. The gripper torque is {torque} of 8. Given these, what action should the robot take to complete the task?",
  task: "The task is to {line}. The current state of the robot is {state}. The current velocity of the robot is {velocity}. The gripper torque is {torque} of 8. Given these, what action should the robot take to complete the task?",
  original: "The task is to {line}. The current state of the robot is {state}. Given these, what action should the robot take to complete the task?",
};
let style = "step";
function preview() {
  const line = $("lines").value.split("\n").map(l => l.trim()).filter(l => l)[0] || "instruction";
  const fill = { line: `<b>${esc(line)}</b>`, task: `<i>${esc($("task").value || "the frame's own task")}</i>`,
    state: "&lt;state tokens&gt;", velocity: "&lt;velocity tokens&gt;", torque: "N" };
  $("preview").innerHTML = TEMPLATES[style].replace(/\{(\w+)\}/g, (_, key) => fill[key]);
  $("tasklabel").hidden = style !== "step";
}
for (const choice of document.querySelectorAll("#style div")) choice.onclick = () => {
  document.querySelectorAll("#style div").forEach(e => e.classList.remove("on"));
  choice.classList.add("on");
  style = choice.dataset.style;
  preview();
};
$("lines").oninput = $("task").oninput = preview;
preview();
$("run").onclick = ask;
$("go").onclick = act;
$("prompt").onkeydown = e => { if (e.key === "Enter" && e.ctrlKey) ask(); };
$("lines").onkeydown = e => { if (e.key === "Enter" && e.ctrlKey) act(); };
</script>
"""


@dataclass
class ChatConfig(TextProbeConfig):
    checkpoints: str = (
        "base,"
        "outputs/runs/rebot-v2-tube-b128-20261114/checkpoints/001200/pretrained_model,"
        "outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model"
    )
    image_dir: str = "outputs/data/coco_val2017_sample"
    frames: str = "200,600,1200,2000"  # validation-set frames offered next to the files of image_dir
    clutter_root: str = "outputs/data/rebot_2026-10-14-val-v1"  # old cluttered recording (tidy up the table, bottles, ...)
    clutter_frames: str = "1180,2180,3215,4310,6040"
    body_frame: str = "clutter_3215"  # Act on a photo: the arm state and the wrist camera come from this frame
    kept_runs: int = 8  # Layers: captured runs kept in memory (a run is every layer at every position)
    matrix_positions: int = 500  # Layers: most positions drawn in a position-by-position matrix
    port: int = 8770


def run(cfg: ChatConfig) -> None:
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.val_dataset_path)
    clutter = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.clutter_root)
    robot = {f"robot_{frame}": (dataset, int(frame)) for frame in cfg.frames.split(",")}
    robot |= {f"clutter_{frame}": (clutter, int(frame)) for frame in cfg.clutter_frames.split(",")}
    images, groups = {}, {}
    for name, (source, index) in robot.items():
        images[name] = (source[index][IMAGE_KEY].permute(1, 2, 0) * 255).round().byte().numpy()
        groups[name] = "Validation frames (tube)" if source is dataset else "Cluttered scenes"
    for name in sorted(os.listdir(cfg.image_dir)):
        images[name] = np.asarray(Image.open(os.path.join(cfg.image_dir, name)).convert("RGB"))
        groups[name] = "Photos"

    models = {}
    for path in cfg.checkpoints.split(","):
        cfg.policy.pretrained_path = None if path == "base" else path
        adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
        adapter._set_probe_cuda_graph_enabled(False)
        processor = next(s for s in adapter.preprocessor.steps if isinstance(s, MolmoAct2PackInputsProcessorStep)).processor
        models[label(path)] = (adapter, processor)

    kinematics = RebotKinematics()
    torque = cfg.policy.gripper_torque
    frames = {}
    app = Flask(__name__)

    @app.errorhandler(Exception)
    def failed(error):
        # the page prints the error in the cell instead of waiting forever
        logging.exception("[text_probe] request failed")
        return jsonify(error=f"{type(error).__name__}: {error}")

    @app.get("/")
    def page():
        listing = [{"name": name, "group": groups[name], "robot": name in robot} for name in images]
        return PAGE.replace("__IMAGES__", json.dumps(listing)).replace("__CHECKPOINTS__", json.dumps(list(models))).replace("__BODY__", cfg.body_frame)

    @app.get("/plotly.js")
    def plotly():
        from plotly.offline import get_plotlyjs  # the copy shipped with the package: no CDN needed

        return app.response_class(get_plotlyjs(), mimetype="application/javascript")

    @app.get("/image/<name>")
    def image(name):
        buffer = io.BytesIO()
        Image.fromarray(images[name]).save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    @app.post("/answer")
    def reply():
        ask = request.get_json()
        adapter, processor = models[ask["checkpoint"]]
        picture = images[ask["image"]] if ask["image"] is not None else None
        text, _, first_top = answer(
            adapter.policy, processor, [ask["prompt"]], picture, ask["max_new_tokens"], ask["prefix"], ask["mask"]
        )
        return jsonify(text=text, top=first_top)

    def action_inputs(name, style):
        """The frame behind an image choice (None = text only) and the observation + metadata of a prompt style."""
        if name not in frames:
            source, index = robot.get(name, robot[cfg.body_frame])
            frame = probe_frame_inputs(source, cfg, index, cfg.policy.chunk_size, with_gripper_event_targets=False)
            # a root without the motor columns: zero velocity and the level of zero torque
            frame["obs"].setdefault(f"probe_complementary.{VELOCITY_KEY}", torch.zeros(1, 7))
            frame["obs"].setdefault(
                f"probe_complementary.{GRIPPER_TORQUE_LEVEL_KEY}",
                torch.as_tensor(gripper_torque_levels(np.zeros(1), torque.levels, torque.low, torque.high)),
            )
            if name is None:
                # text only: both cameras are marked absent, so their patches leave the attention mask
                for key in [key for key in frame["obs"] if key.startswith("observation.images.")]:
                    frame["obs"][key] = torch.zeros_like(frame["obs"][key])
                    frame["obs"][f"probe_complementary.camera_is_present.{key}"] = torch.tensor([False])
                frame["gt_actions"] = None
            elif name not in robot:
                # a photo: it replaces the top camera; arm state and wrist camera stay those of body_frame
                top = frame["obs"]["observation.images.external_0"]
                photo = Image.fromarray(images[name])
                photo.thumbnail((top.shape[-1], top.shape[-2]))
                canvas = Image.new("RGB", (top.shape[-1], top.shape[-2]))
                canvas.paste(photo, ((canvas.width - photo.width) // 2, (canvas.height - photo.height) // 2))
                frame["obs"]["observation.images.external_0"] = (
                    torch.from_numpy(np.asarray(canvas)).permute(2, 0, 1)[None].float() / 255
                )
                frame["gt_actions"] = None
            frames[name] = frame
        frame = frames[name]
        obs, metadata = frame["obs"], frame["metadata"]
        if style == "original":  # task + state only, the prompt of the released checkpoint
            extra = [f"probe_complementary.{key}" for key in (VELOCITY_KEY, GRIPPER_TORQUE_LEVEL_KEY, TIP_TARGET_KEY)]
            obs, metadata = {key: value for key, value in obs.items() if key not in extra}, None
        return frame, obs, metadata

    @app.post("/act")
    def act():
        ask = request.get_json()
        adapter, processor = models[ask["checkpoint"]]
        policy = adapter.policy
        frame, obs, metadata = action_inputs(ask["image"], ask["style"])

        def tip_path(mode: str, task: str, subtask: str | None, seed: int) -> np.ndarray:
            """One predicted chunk as [steps, 3] tip positions in cm, after the deployed low-pass."""
            chunk, _, _ = adapter.predict_action_chunk_in_mode(
                obs, task, inference_action_mode=mode, state=frame["state"], subtask=subtask,
                metadata=metadata, generator=torch.Generator(device=device).manual_seed(seed),
            )
            return kinematics.ee_path(apply_butterworth_filter(chunk.double().numpy())) * 100

        lines = []
        for line in ask["lines"]:
            task, subtask = ask["task"] or frame["task"], line
            if ask["style"] != "step":
                task, subtask = line, None
            if ask["decoder"] == "expert":
                expert = [tip_path("continuous", task, subtask, seed) for seed in range(ask["seeds"])]
                lines.append({"line": line, "expert": [np.round(path, 2).tolist() for path in expert]})
                continue
            # the generated ids are read on their way into the decoder, so they are shown even when it refuses them
            generated = {}
            decode = policy._decode_discrete_action_chunk
            policy._decode_discrete_action_chunk = lambda ids, **kwargs: generated.update(ids=ids) or decode(ids, **kwargs)
            tokens, tokens_error = None, ""
            try:
                tokens = tip_path("discrete", task, subtask, 0)
            except (RuntimeError, ValueError) as error:
                tokens_error = str(error)
            finally:
                del policy._decode_discrete_action_chunk
            raw = ""
            if "ids" in generated:
                ids = generated["ids"].reshape(-1).tolist()
                eos = int(policy.model.config.eos_token_id)
                raw = processor.tokenizer.decode(ids[: ids.index(eos)] if eos in ids else ids, skip_special_tokens=False)
            lines.append({
                "tokens": None if tokens is None else np.round(tokens, 2).tolist(),
                "tokens_error": tokens_error,
                "raw": raw,
            })
        return jsonify(
            lines=lines,
            arm=np.round(kinematics.link_origins(frame["state"].double().numpy())[0] * 100, 2).tolist(),
            recorded=None
            if frame["gt_actions"] is None
            else np.round(kinematics.ee_path(frame["gt_actions"].double().numpy()) * 100, 2).tolist(),
            task=frame["task"],
            subtask=frame["subtask"] or "",
        )

    runs = {}  # the last few captured runs, hidden states on the CPU

    def captured(spec):
        """Hidden states of one run: {checkpoint, kind: chat | action, image, text, style, task}."""
        key = json.dumps(spec, sort_keys=True)
        if key in runs:
            return runs[key]
        adapter, processor = models[spec["checkpoint"]]
        policy = adapter.policy

        def chat():
            picture = images[spec["image"]] if spec["image"] is not None else None
            text = "<|image|>" if picture is not None else ""
            text += f"<|im_start|>user\n{spec['text']}<|im_end|>\n<|im_start|>assistant\n"
            inputs = processor(text=text, images=picture, return_tensors="pt", padding=True)
            inputs = policy._model_inputs({key: value.to(device) for key, value in inputs.items()})
            policy.model(**policy._drop_trivial_attention_mask(inputs), use_cache=True)
            mask = inputs.get("attention_mask")
            return inputs["input_ids"][0], None if mask is None else mask[0]

        def action():
            frame, obs, metadata = action_inputs(spec["image"], spec["style"])
            task, subtask = spec["task"] or frame["task"], spec["text"]
            if spec["style"] != "step":
                task, subtask = spec["text"], None
            seen = {}
            pack = policy._model_inputs  # the packed prompt is read on its way into the model
            policy._model_inputs = lambda batch: seen.update(batch) or pack(batch)
            try:
                adapter.predict_action_chunk_in_mode(
                    obs, task, inference_action_mode="continuous", state=frame["state"], subtask=subtask,
                    metadata=metadata, generator=torch.Generator(device=device).manual_seed(0),
                )
            finally:
                del policy._model_inputs
            mask = seen.get("attention_mask")
            return seen["input_ids"][0], None if mask is None else mask[0]

        run = capture(adapter, chat if spec["kind"] == "chat" else action)
        run["tokens"], run["groups"] = label_tokens(adapter, processor.tokenizer, run)
        run["within"] = within(run["encoder"], run["groups"], device)
        if len(runs) >= cfg.kept_runs:
            del runs[next(iter(runs))]
        runs[key] = run
        return run

    @app.post("/layers")
    def layers():
        ask = request.get_json()
        a, b = captured(ask["a"]), captured(ask["b"])
        first, second = paired(a["groups"], b["groups"])
        cosine = between(a["encoder"], b["encoder"], device, first, second)
        expert = None
        if a["expert"] is not None and b["expert"] is not None:
            expert = np.round(between(a["expert"], b["expert"], device, *[list(range(min(a["expert"].shape[1], b["expert"].shape[1])))] * 2).numpy(), 3).tolist()
        return jsonify(
            cos=np.round(cosine.numpy(), 3).tolist(),
            first=first,
            second=second,
            expert=expert,
            a={"tokens": a["tokens"], "groups": a["groups"], **a["within"]},
            b={"tokens": b["tokens"], "groups": b["groups"], **b["within"]},
        )

    @app.post("/matrix")
    def matrix():
        """Cosine between every two positions of one run at one layer; image patches only on request."""
        ask = request.get_json()
        run = captured(ask["run"])
        keep = [i for i, g in enumerate(run["groups"]) if g != "masked" and (ask["images"] or g != "image")]
        keep = keep[:: -(-len(keep) // cfg.matrix_positions)]  # every n-th position when there are too many to draw
        unit = torch.nn.functional.normalize(run["encoder"][ask["layer"]][keep].to(device).float(), dim=-1)
        return jsonify(
            z=np.round((unit @ unit.T).cpu().numpy(), 2).tolist(),
            tokens=[run["tokens"][i] for i in keep],
            groups=[run["groups"][i] for i in keep],
            positions=keep,
        )

    logging.info(f"[text_probe] chat page on http://localhost:{cfg.port}")
    app.run(host="127.0.0.1", port=cfg.port, threaded=False)  # one request at a time: the models share the GPU


@parser.wrap()
def cli(cfg: ChatConfig):
    run(cfg)


def main() -> None:
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    register_config_choices()
    main()

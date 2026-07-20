#!/usr/bin/env python

"""
π0.7-style steering metadata, end to end: per-episode quality 1-5 (human) +
per-subtask-window mistake flag (LLM suspicion score 0-10, thresholded into
flags at review time, human-confirmed).
Speed is deliberately omitted (single-operator data; pace variation is carried
by the mistake channel; the prompt clause renders partially).

Subcommands:
- annotate: LLM mistake pass over the meta/subtask_windows.json windows
  (judges execution of the KNOWN label, evidence-first suspicion score 0-10),
  writes raw scores to meta/mistake_candidates.json — no threshold baked in.
- review: local review UI (opens the browser itself). --threshold picks which
  scores count as flagged (default 4); restart review with a different value to
  widen/narrow the candidate set without re-annotating. Streams the dataset
  videos: watch each episode and click its quality 1-5; play each flagged 4 s
  clip and confirm/reject the mistake. Every click auto-saves to
  meta/metadata_review_state.json; the "Write to dataset" button writes
  meta/episode_metadata.parquet + meta/mistakes.parquet + meta/metadata_info.json
  directly. Works before annotate has run (quality scoring only; mistakes and
  the write button unlock once candidates exist — just restart review).

Flow:
    python metadata_annotate.py annotate --data-dir DS --top-key observation.images.top \
        --wrist-key observation.images.wrist        # GPU
    python metadata_annotate.py review --data-dir DS   # can start before annotate finishes
"""

import argparse
import glob
import json
import re
import textwrap
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console

from lerobot.data_processing.annotate.canonicalize_subtasks import parse_json_response
from lerobot.data_processing.annotate.subtask_annotate_grid import frame_at, video_path_for

console = Console()

ANNOTATE_TOP_FRACS = (0.0, 0.5, 1.0)
WRIST_FRACS = (0.25, 0.75)
MISTAKE_SCORE_THRESHOLD = 4


def load_meta(root: Path):
    info = json.load(open(root / "meta" / "info.json"))
    windows = json.load(open(root / "meta" / "subtask_windows.json"))["episodes"]
    episodes_meta = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))]
    ).set_index("episode_index")
    return info, float(info["fps"]), windows, episodes_meta


# ── annotate: LLM mistake suspicion pass ─────────────────────────────────────

def create_mistake_prompt(coarse_goal: str, subtask: str, n_top: int, n_wrist: int) -> str:
    wrist_clause = (
        f" The first {n_top} frames are the overhead view in time order; the last {n_wrist} are the wrist camera."
        if n_wrist else ""
    )
    return textwrap.dedent(f"""\
        A robot is working on this task: "{coarse_goal}"
        During these frames it is meant to be performing: "{subtask}"{wrist_clause}

        Rate how suspicious these frames are of a mistake, on a 0-10 scale.
        A mistake is a concrete visible event, such as:
        - a failed grasp (gripper closes on nothing, or the object slips out),
        - an object dropped or knocked somewhere unintended,
        - the robot clearly doing something other than "{subtask}".

        First describe what you actually see happening, then score. High scores
        need specific visual evidence of a mistake in these frames; frames that
        are merely still, ambiguous, or hard to read are low, not high. Use the
        middle of the scale when genuinely unsure.
        0 = clearly clean execution, 10 = certain mistake.

        Reply with only JSON: {{"evidence": "<one short sentence: what you see>", "score": <integer 0-10>}}
    """)


def judge_window(model, processor, device, frames, coarse_goal, subtask, n_top, n_wrist) -> tuple[float, str]:
    prompt = create_mistake_prompt(coarse_goal, subtask, n_top, n_wrist)
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, videos=frames, num_frames=len(frames), return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    response = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()
    parsed = parse_json_response(response)
    if not isinstance(parsed.get("score"), (int, float)):
        raise ValueError(f"No numeric 'score' in response: {response[:200]}")
    return float(parsed["score"]), str(parsed.get("evidence", "")).strip()


def cmd_annotate(args):
    root = Path(args.data_dir)
    info, fps, windows, episodes_meta = load_meta(root)
    coarse_goal = str(pd.read_parquet(root / "meta" / "tasks.parquet").index[0])
    total = sum(len(w) for w in windows.values())
    console.print(f'goal: "{coarse_goal}" | {len(windows)} episodes -> {total} LLM calls')

    from transformers import AutoModelForCausalLM, AutoProcessor

    console.print(f"[cyan]Loading {args.model}...[/cyan]")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device)

    episode_keys = (
        [str(args.preview_episode)] if args.preview_episode is not None else sorted(windows, key=int)
    )
    results: dict[str, list[dict]] = {}
    for ep_key in episode_keys:
        ep = episodes_meta.loc[int(ep_key)]
        top_path, top_from = video_path_for(root, info, ep, args.top_key)
        if args.wrist_key:
            wrist_path, wrist_from = video_path_for(root, info, ep, args.wrist_key)
        ep_from_index = int(ep["dataset_from_index"])
        rows = []
        for win in windows[ep_key]:
            t0 = (win["from_index"] - ep_from_index) / fps
            t1 = (win["to_index"] - 1 - ep_from_index) / fps
            frames = [f for fr in ANNOTATE_TOP_FRACS if (f := frame_at(top_path, top_from + t0 + fr * (t1 - t0)))]
            n_top = len(frames)
            n_wrist = 0
            if args.wrist_key:
                wrist = [f for fr in WRIST_FRACS if (f := frame_at(wrist_path, wrist_from + t0 + fr * (t1 - t0)))]
                frames += wrist
                n_wrist = len(wrist)
            score, evidence = judge_window(
                model, processor, args.device, frames, coarse_goal, win["subtask"], n_top, n_wrist
            )
            color = "red" if score >= MISTAKE_SCORE_THRESHOLD else "green"
            console.print(f"[{color}]ep {ep_key} {t0:5.0f}-{t1:3.0f}s: score={score:g}[/{color}] {evidence}")
            rows.append({
                "from_index": win["from_index"], "to_index": win["to_index"],
                "subtask": win["subtask"], "score": score, "evidence": evidence,
            })
        results[ep_key] = rows

    if args.preview_episode is not None:
        console.print("[yellow]Preview only — nothing written.[/yellow]")
        return

    out_path = root / "meta" / "mistake_candidates.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "episodes": results}, f, indent=2)
    flagged = sum(r["score"] >= MISTAKE_SCORE_THRESHOLD for rows in results.values() for r in rows)
    console.print(
        f"[bold green]✓ {flagged}/{total} windows at score >= {MISTAKE_SCORE_THRESHOLD} -> {out_path}[/bold green]"
    )


# ── review: video UI, server-side state, writes the dataset ──────────────────

def write_metadata(root: Path, windows, episodes_meta, candidates, state) -> str:
    """Validate the review state and write the three meta files. Raises ValueError."""
    if candidates is None:
        raise ValueError("Mistake flags are pending — run the annotate stage, then restart review.")
    quality = {int(k): v for k, v in state.get("quality", {}).items()}
    missing = sorted(set(int(k) for k in windows) - set(quality))
    if missing:
        raise ValueError(f"No quality score yet for episodes {missing}")
    bad = {ep: q for ep, q in quality.items() if not isinstance(q, int) or not 1 <= q <= 5}
    if bad:
        raise ValueError(f"Quality must be int 1-5, got {bad}")

    decisions = {}
    for win_id, mistake in state.get("windows", {}).items():
        ep_key, from_index, to_index = win_id.split(":")
        decisions[(int(ep_key), int(from_index), int(to_index))] = bool(mistake)
    flagged = {
        (int(ep_key), w["from_index"], w["to_index"])
        for ep_key, wins in candidates.items()
        for w in wins
        if w["mistake"]
    }
    undecided = sorted(flagged - set(decisions))
    if undecided:
        raise ValueError(f"{len(undecided)} flagged windows not yet confirmed/rejected: {undecided[:5]}")

    episode_rows = [
        {
            "episode_index": int(ep_key),
            "quality": quality[int(ep_key)],
            "from_index": int(episodes_meta.loc[int(ep_key), "dataset_from_index"]),
            "to_index": int(episodes_meta.loc[int(ep_key), "dataset_to_index"]),
        }
        for ep_key in sorted(windows, key=int)
    ]
    mistake_rows = [
        {
            "episode_index": int(ep_key),
            "from_index": w["from_index"],
            "to_index": w["to_index"],
            "mistake": decisions.get((int(ep_key), w["from_index"], w["to_index"]), False),
        }
        for ep_key in sorted(windows, key=int)
        for w in windows[ep_key]
    ]
    pd.DataFrame(episode_rows).to_parquet(root / "meta" / "episode_metadata.parquet", engine="pyarrow")
    pd.DataFrame(mistake_rows).to_parquet(root / "meta" / "mistakes.parquet", engine="pyarrow")
    n_mistakes = sum(r["mistake"] for r in mistake_rows)
    with open(root / "meta" / "metadata_info.json", "w") as f:
        json.dump({
            "quality_by_episode": {str(r["episode_index"]): r["quality"] for r in episode_rows},
            "n_windows": len(mistake_rows),
            "n_flagged": len(flagged),
            "n_mistakes": n_mistakes,
            "speed": "omitted",
        }, f, indent=2)
    return (
        f"Wrote quality for {len(episode_rows)} episodes and "
        f"{n_mistakes}/{len(mistake_rows)} mistake windows to {root / 'meta'}"
    )


APP_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Metadata review</title>
<style>
  body { font-family: system-ui, sans-serif; background: #111; color: #ddd; margin: 0; padding: 1rem 2rem; }
  h1 { font-size: 1.2rem; margin: 0; }
  h2 { font-size: 1.05rem; border-top: 1px solid #333; padding-top: 1rem; }
  video { width: 100%; max-width: 900px; border-radius: 8px; background: #000; }
  .row { border: 2px solid #333; border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.5rem 0; }
  .row.current { border-color: #58a6ff; }
  .row.mistake { background: #2a1416; }
  .row.clean { background: #14231a; }
  .row .evidence { color: #e3b341; }
  .row b { color: #fff; }
  button { font-size: 0.9rem; padding: 0.3rem 0.9rem; border-radius: 6px; border: 1px solid #555;
           background: #222; color: #ddd; cursor: pointer; margin-right: 0.4rem; }
  button.active-mistake { background: #b62324; color: #fff; border-color: #b62324; }
  button.active-clean { background: #238636; color: #fff; border-color: #238636; }
  button.qbtn { min-width: 2.2rem; margin-right: 0.15rem; }
  button.active-quality { background: #58a6ff; color: #000; border-color: #58a6ff; }
  button.cam.active-cam { background: #444; color: #fff; }
  .pending { color: #e3b341; }
  #topbar { position: sticky; top: 0; background: #111; padding: 0.6rem 0; z-index: 2;
            display: flex; align-items: center; gap: 1rem; border-bottom: 1px solid #333; }
  #status { color: #7ee787; }
</style>
</head>
<body>
<div id="topbar">
  <h1>Metadata review</h1>
  <span id="progress"></span>
  <button onclick="writeDataset()">Write to dataset</button>
  <span id="status"></span>
</div>
<p>Watch each episode, click its quality (1 = poor … 5 = flawless). Click a flagged window to
play the clip, then confirm (<b>y</b> / mistake) or reject (<b>n</b> / clean); j/k move, r replays.
<span class="pending" id="pendingnote"></span></p>
<div id="episodes"></div>
<script>
const DATA = __PAYLOAD__;
let state = DATA.state;
let rows = [];
let current = -1;
const players = {};   // ep -> {video, camIdx, clip: [t0, t1] | null}

function post(path, body, cb) {
  fetch(path, { method: "POST", body: JSON.stringify(body) })
    .then((r) => r.text().then((t) => cb(r.ok, t)));
}
function saveState() {
  post("/state", state, (ok) => {
    document.getElementById("status").textContent = ok ? "saved" : "SAVE FAILED";
  });
  updateProgress();
}
function videoUrl(ep, camIdx) { return `/video/${camIdx}/${ep}`; }

function build() {
  const container = document.getElementById("episodes");
  if (!DATA.has_candidates) {
    document.getElementById("pendingnote").textContent =
      "Mistake flags pending — score quality now; restart review after the annotate stage.";
  }
  for (const ep of DATA.episodes) {
    const sec = document.createElement("div");
    sec.id = `sec-${ep.ep}`;
    const cams = ep.videos.map((v, i) =>
      `<button class="cam" data-cam="${i}" onclick="switchCam('${ep.ep}', ${i})">${v.name}</button>`).join("");
    const qbtns = [1, 2, 3, 4, 5].map((q) =>
      `<button class="qbtn" data-q="${q}" onclick="setQuality('${ep.ep}', ${q})">${q}</button>`).join("");
    const flaggedNote = DATA.has_candidates ? ` — ${ep.flagged.length} flagged windows` : "";
    sec.innerHTML = `<h2>Episode ${ep.ep} (${Math.round(ep.duration)}s${flaggedNote})
        &nbsp; quality: ${qbtns} &nbsp; ${cams}</h2>
      <video id="video-${ep.ep}" controls preload="metadata"></video>`;
    container.appendChild(sec);
    const video = sec.querySelector("video");
    players[ep.ep] = { video: video, camIdx: 0, clip: null, ep: ep };
    video.src = videoUrl(ep.ep, 0);
    video.addEventListener("loadedmetadata", () => {
      if (players[ep.ep].clip === null) video.currentTime = ep.videos[0].from_ts;
    });
    video.addEventListener("timeupdate", () => {
      const p = players[ep.ep];
      const offset = p.ep.videos[p.camIdx].from_ts;
      const end = p.clip ? offset + p.clip[1] : offset + p.ep.duration;
      if (video.currentTime >= end) video.pause();
    });
    for (const win of ep.flagged) {
      const row = document.createElement("div");
      row.className = "row";
      row.dataset.id = win.id;
      row.dataset.ep = ep.ep;
      row.innerHTML = `<b>${Math.round(win.t0)}–${Math.round(win.t1)}s</b> &nbsp; <b>${win.subtask}</b>
        &nbsp; <span class="evidence">${win.score}/10 · ${win.evidence}</span>
        &nbsp; <button class="btn-mistake" onclick="decide(event, this, true)">mistake</button>
        <button class="btn-clean" onclick="decide(event, this, false)">clean</button>`;
      row.addEventListener("click", () => { focusRow(rows.indexOf(row)); });
      container.appendChild(row);
      rows.push(row);
    }
  }
  for (const ep of DATA.episodes) { applyQuality(ep.ep); applyCam(ep.ep); }
  rows.forEach(applyRow);
  updateProgress();
}
function applyCam(ep) {
  document.querySelectorAll(`#sec-${CSS.escape(ep)} .cam`).forEach((b) => {
    b.classList.toggle("active-cam", Number(b.dataset.cam) === players[ep].camIdx);
  });
}
function switchCam(ep, camIdx) {
  const p = players[ep];
  p.camIdx = camIdx;
  p.video.src = videoUrl(ep, camIdx);
  const offset = p.ep.videos[camIdx].from_ts;
  const t = p.clip ? p.clip[0] : 0;
  p.video.addEventListener("loadedmetadata", () => { p.video.currentTime = offset + t; p.video.play(); },
                           { once: true });
  applyCam(ep);
}
function playClip(ep, t0, t1) {
  const p = players[ep];
  p.clip = [t0, t1];
  const offset = p.ep.videos[p.camIdx].from_ts;
  p.video.currentTime = offset + t0;
  p.video.play();
  p.video.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
function focusRow(i) {
  if (!rows.length) return;
  current = Math.max(0, Math.min(rows.length - 1, i));
  rows.forEach((r, j) => r.classList.toggle("current", j === current));
  const row = rows[current];
  const win = findWin(row.dataset.id);
  playClip(row.dataset.ep, win.t0, win.t1);
}
function findWin(id) {
  for (const ep of DATA.episodes) for (const w of ep.flagged) if (w.id === id) return w;
}
function applyRow(row) {
  const v = state.windows[row.dataset.id];
  row.classList.toggle("mistake", v === true);
  row.classList.toggle("clean", v === false);
  row.querySelector(".btn-mistake").className = "btn-mistake" + (v === true ? " active-mistake" : "");
  row.querySelector(".btn-clean").className = "btn-clean" + (v === false ? " active-clean" : "");
}
function decide(ev, btn, mistake) {
  if (ev) ev.stopPropagation();
  const row = btn ? btn.closest(".row") : rows[current];
  if (!row) return;
  state.windows[row.dataset.id] = mistake;
  saveState();
  applyRow(row);
}
function setQuality(ep, q) {
  state.quality[ep] = q;
  saveState();
  applyQuality(ep);
}
function applyQuality(ep) {
  const q = state.quality[ep];
  document.querySelectorAll(`#sec-${CSS.escape(ep)} .qbtn`).forEach((b) => {
    b.className = "qbtn" + (Number(b.dataset.q) === q ? " active-quality" : "");
  });
}
function updateProgress() {
  document.getElementById("progress").textContent =
    Object.keys(state.windows).length + " / " + rows.length + " windows · " +
    Object.keys(state.quality).length + " / " + DATA.episodes.length + " episodes scored";
}
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "VIDEO" || e.target.tagName === "INPUT") return;
  if (e.key === "j") focusRow(current + 1);
  else if (e.key === "k") focusRow(current - 1);
  else if (e.key === "y") { decide(null, null, true); focusRow(current + 1); }
  else if (e.key === "n") { decide(null, null, false); focusRow(current + 1); }
  else if (e.key === "r" && current >= 0) focusRow(current);
});
function writeDataset() {
  post("/finalize", {}, (ok, text) => {
    document.getElementById("status").textContent = "";
    alert(text);
  });
}
build();
</script>
</body>
</html>
"""


def cmd_review(args):
    root = Path(args.data_dir)
    info, fps, windows, episodes_meta = load_meta(root)
    candidates_path = root / "meta" / "mistake_candidates.json"
    candidates = json.load(open(candidates_path))["episodes"] if candidates_path.exists() else None
    if candidates:
        for wins in candidates.values():
            for w in wins:
                w["mistake"] = w["score"] >= args.threshold
    state_path = root / "meta" / "metadata_review_state.json"
    state = json.load(open(state_path)) if state_path.exists() else {"quality": {}, "windows": {}}

    video_keys = sorted(k for k, ft in info["features"].items() if ft.get("dtype") == "video")
    video_files: dict[tuple[int, str], Path] = {}
    episodes_payload = []
    for ep_key in sorted(windows, key=int):
        ep = episodes_meta.loc[int(ep_key)]
        ep_from_index = int(ep["dataset_from_index"])
        duration = (int(ep["dataset_to_index"]) - ep_from_index) / fps
        videos = []
        for cam_idx, key in enumerate(video_keys):
            path, from_ts = video_path_for(root, info, ep, key)
            video_files[(cam_idx, ep_key)] = path
            videos.append({"name": key.split(".")[-1], "from_ts": from_ts})
        flagged = [w for w in candidates.get(ep_key, []) if w["mistake"]] if candidates else []
        episodes_payload.append({
            "ep": ep_key,
            "duration": duration,
            "videos": videos,
            "flagged": [
                {
                    "id": f"{ep_key}:{w['from_index']}:{w['to_index']}",
                    "t0": (w["from_index"] - ep_from_index) / fps,
                    "t1": (w["to_index"] - 1 - ep_from_index) / fps,
                    "subtask": w["subtask"],
                    "score": w["score"],
                    "evidence": w.get("evidence", ""),
                }
                for w in flagged
            ],
        })
    payload = {"episodes": episodes_payload, "has_candidates": candidates is not None, "state": state}
    page = APP_HTML.replace("__PAYLOAD__", json.dumps(payload))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, content_type="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == "/":
                    self._send(200, page.encode(), "text/html")
                    return
                m = re.match(r"^/video/(\d+)/(\d+)$", self.path)
                if m:
                    self._serve_video(video_files.get((int(m.group(1)), m.group(2))))
                    return
                self._send(404, b"not found")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/state":
                new_state = json.loads(body)
                state.clear()
                state.update(new_state)
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2)
                self._send(200, b"ok")
            elif self.path == "/finalize":
                try:
                    message = write_metadata(root, windows, episodes_meta, candidates, state)
                    console.print(f"[bold green]✓ {message}[/bold green]")
                    self._send(200, message.encode())
                except ValueError as e:
                    self._send(400, str(e).encode())
            else:
                self._send(404, b"not found")

        def _serve_video(self, path: Path | None):
            if path is None or not path.exists():
                self._send(404, b"video not found")
                return
            size = path.stat().st_size
            range_header = self.headers.get("Range")
            if range_header:
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                start, end = 0, size - 1
                self.send_response(200)
            length = end - start + 1
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    n_flagged = sum(len(e["flagged"]) for e in episodes_payload)
    status = (
        f"{n_flagged} flagged windows (score >= {args.threshold:g})"
        if candidates else "mistake flags pending (quality scoring only)"
    )
    console.print(f"[bold green]Review UI at {url}[/bold green] — {len(episodes_payload)} episodes, {status}")
    console.print("Decisions auto-save; 'Write to dataset' writes the meta files. Ctrl-C to stop.")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\nstopped")


def main():
    parser = argparse.ArgumentParser(description="π0.7 metadata: annotate mistakes, review + write")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("annotate", help="LLM mistake suspicion pass (score 0-10 per window)")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--top-key", type=str, required=True)
    p.add_argument("--wrist-key", type=str, default=None)
    p.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--preview-episode", type=int, default=None,
                   help="judge a single episode and print flags; no writes")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("review", help="video review UI; saves state and writes the dataset")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--threshold", type=float, default=MISTAKE_SCORE_THRESHOLD,
                   help="windows with LLM suspicion score >= this are flagged for review")
    p.set_defaults(func=cmd_review)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

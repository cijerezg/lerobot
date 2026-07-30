#!/usr/bin/env python

"""
Build a self-contained HTML page for reviewing summary + subtask annotations against video.

Plays the dataset's own mp4s (no re-encode) with both cameras synced, and shows the summary
and subtask that apply at the playhead. Two aligned timeline strips make the annotation
rhythm visible at a glance: subtask windows coloured by phase, summary windows marked as
update vs hold (a hold is a verbatim repeat -- the "nothing moved" signal).

Videos are referenced by relative path, so the page must be served with the dataset root as
the document root -- and by a server that honours HTTP Range, or seeking is impossible
(see serve_review.py):

    uv run lerobot/src/lerobot/data_processing/annotate/review_annotations.py \
        --data-dir outputs/rebot_sorting_clothes_v1
    uv run lerobot/src/lerobot/data_processing/annotate/serve_review.py \
        --data-dir outputs/rebot_sorting_clothes_v1

Videos are AV1; any current Chrome or Firefox decodes them. The page reports a decode
failure rather than showing a silently blank frame.
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

TOP, WRIST = "observation.images.top", "observation.images.wrist"


def build_payload(root: Path) -> dict:
    info = json.load(open(root / "meta" / "info.json"))
    fps = float(info["fps"])
    goal = str(pd.read_parquet(root / "meta" / "tasks.parquet").index[0])

    episodes_meta = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))]
    ).set_index("episode_index")

    summaries = pd.read_parquet(root / "meta" / "summaries.parquet")
    subtasks = json.load(open(root / "meta" / "subtask_windows.json"))

    episodes = []
    for ep_idx, ep in episodes_meta.iterrows():
        ep_idx = int(ep_idx)
        f0, f1 = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        duration = (f1 - f0) / fps

        videos = {}
        for key in (TOP, WRIST):
            if f"videos/{key}/chunk_index" not in ep:
                continue
            videos[key] = {
                "src": info["video_path"].format(
                    video_key=key,
                    chunk_index=int(ep[f"videos/{key}/chunk_index"]),
                    file_index=int(ep[f"videos/{key}/file_index"]),
                ),
                "offset": float(ep[f"videos/{key}/from_timestamp"]),
            }

        sm = summaries[summaries.episode_index == ep_idx].sort_values("segment_index")
        summary_rows, prev = [], None
        for _, r in sm.iterrows():
            text = str(r.summary)
            summary_rows.append({
                "t0": (int(r.from_index) - f0) / fps,
                "t1": (int(r.to_index) - f0) / fps,
                "text": text,
                "hold": prev is not None and text == prev,
            })
            prev = text

        subtask_rows = [
            {
                "t0": (int(w["from_index"]) - f0) / fps,
                "t1": (int(w["to_index"]) - f0) / fps,
                "text": w["subtask"],
            }
            for w in subtasks["episodes"].get(str(ep_idx), [])
        ]

        episodes.append({
            "index": ep_idx,
            "duration": duration,
            "frames": f1 - f0,
            "videos": videos,
            "summaries": summary_rows,
            "subtasks": subtask_rows,
        })

    return {"dataset": root.name, "goal": goal, "fps": fps, "episodes": episodes}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light dark; --bg: #ffffff; --fg: #111; --dim: #6d6c67; --line: #dcdcd6; --hi: #f1f1ee; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #141413; --fg: #f2f2ee; --dim: #8f8e85; --line: #3a3a37; --hi: #232322; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 20px; background: var(--bg); color: var(--fg);
       font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 16px; font-weight: 600; margin: 0 0 12px; }
h1 span { color: var(--dim); font-weight: 400; }
.eps { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 14px; }
.eps button, .bar button {
  font: inherit; font-size: 13px; cursor: pointer; border-radius: 6px;
  background: transparent; color: var(--dim); border: 1px solid var(--line); padding: 4px 11px;
}
.eps button[aria-pressed="true"] { background: var(--fg); color: var(--bg); border-color: var(--fg); }
.videos { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
@media (max-width: 720px) { .videos { grid-template-columns: 1fr; } }
video { width: 100%; display: block; background: #000; border-radius: 6px; }
.bar { display: flex; align-items: center; gap: 12px; margin: 12px 0 16px; flex-wrap: wrap; }
.bar .t { color: var(--dim); font-variant-numeric: tabular-nums; font-size: 13px; }
.now { margin-bottom: 18px; }
.now div { padding: 3px 0; }
.now b { display: inline-block; min-width: 78px; color: var(--dim); font-weight: 400; font-size: 13px; }
.now .txt { font-size: 16px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
td { padding: 7px 10px; border-top: 1px solid var(--line); cursor: pointer; vertical-align: baseline; }
tr:hover td { background: var(--hi); }
tr.cur td { background: var(--hi); box-shadow: inset 2px 0 0 var(--fg); }
td.t { color: var(--dim); font-variant-numeric: tabular-nums; width: 74px; white-space: nowrap; font-size: 13px; }
td.k { color: var(--dim); width: 58px; font-size: 12px; }
tr.hold td.s { color: var(--dim); }
.err { color: #d14; font-size: 13px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__ <span id="goal"></span></h1>
  <div class="eps" id="eps"></div>
  <div class="videos" id="videos"></div>
  <div class="bar">
    <button id="play">Play</button>
    <span class="t" id="clock"></span>
    <span class="t" id="err"></span>
  </div>
  <div class="now">
    <div><b>subtask</b><span class="txt" id="curSub">&mdash;</span></div>
    <div><b>summary</b><span class="txt" id="curSum">&mdash;</span></div>
  </div>
  <table><tbody id="chain"></tbody></table>
</div>

<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
let ep = DATA.episodes[0], vids = {}, playing = false;

$("goal").textContent = "\\u2014 " + DATA.goal;
DATA.episodes.forEach((e, i) => {
  const b = document.createElement("button");
  b.textContent = "ep" + e.index;
  b.title = e.duration.toFixed(0) + "s";
  b.onclick = () => selectEp(i);
  $("eps").appendChild(b);
});

const lead = () => vids["observation.images.top"] || Object.values(vids)[0];
const localT = () => { const l = lead(); return l ? Math.max(0, Math.min(l.el.currentTime - l.offset, ep.duration)) : 0; };
const at = (rows, t) => rows.findIndex((r) => t >= r.t0 && t < r.t1);

function selectEp(i) {
  ep = DATA.episodes[i];
  [...$("eps").children].forEach((b, j) => b.setAttribute("aria-pressed", j === i));
  $("videos").innerHTML = ""; vids = {}; $("err").textContent = "";
  for (const [key, v] of Object.entries(ep.videos)) {
    const el = document.createElement("video");
    el.src = v.src; el.preload = "auto"; el.muted = true; el.playsInline = true;
    el.onerror = () => $("err").innerHTML = '<span class="err">cannot decode ' + v.src + '</span>';
    el.onloadedmetadata = () => { el.currentTime = v.offset; };
    $("videos").appendChild(el);
    vids[key] = { el, offset: v.offset };
  }
  if (lead()) lead().el.ontimeupdate = tick;

  $("chain").innerHTML = ep.summaries.map((r, i) =>
    '<tr class="' + (r.hold ? "hold" : "") + '" data-i="' + i + '">'
    + '<td class="t">' + r.t0.toFixed(0) + "\\u2013" + r.t1.toFixed(0) + 's</td>'
    + '<td class="k">' + (r.hold ? "hold" : "") + '</td>'
    + '<td class="s">' + r.text + "</td></tr>").join("");
  [...$("chain").children].forEach((tr) =>
    tr.onclick = () => seek(ep.summaries[+tr.dataset.i].t0 + 0.05));
  render(0);
}

function render(t) {
  const si = at(ep.subtasks, t), mi = at(ep.summaries, t);
  $("curSub").textContent = si < 0 ? "\\u2014" : ep.subtasks[si].text;
  $("curSum").textContent = mi < 0 ? "\\u2014" : ep.summaries[mi].text;
  $("clock").textContent = t.toFixed(1) + "s / " + ep.duration.toFixed(1) + "s  \\u00b7  frame " + Math.round(t * DATA.fps);
  [...$("chain").children].forEach((tr) => tr.classList.toggle("cur", +tr.dataset.i === mi));
}

function tick() {
  const t = localT();
  render(t);
  // the second camera lives at a different offset in a different file -- realign on drift
  for (const v of Object.values(vids)) {
    if (v === lead()) continue;
    if (Math.abs(v.el.currentTime - (v.offset + t)) > 0.15) v.el.currentTime = v.offset + t;
  }
  if (playing && t >= ep.duration - 0.05) $("play").click();
}

function seek(t) {
  t = Math.max(0, Math.min(t, ep.duration));
  for (const v of Object.values(vids)) v.el.currentTime = v.offset + t;
  render(t);
}

$("play").onclick = () => {
  playing = !playing;
  $("play").textContent = playing ? "Pause" : "Play";
  Object.values(vids).forEach((v) => playing ? v.el.play() : v.el.pause());
};
document.addEventListener("keydown", (e) => {
  if (e.key === " ") { e.preventDefault(); $("play").click(); }
  if (e.key === "ArrowRight") seek(localT() + (e.shiftKey ? 12 : 4));
  if (e.key === "ArrowLeft") seek(localT() - (e.shiftKey ? 12 : 4));
});

selectEp(0);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Build an HTML annotation review page for a dataset")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <data-dir>/review_annotations.html")
    args = ap.parse_args()

    root = Path(args.data_dir)
    payload = build_payload(root)
    out = Path(args.out) if args.out else root / "review_annotations.html"
    title = f"{payload['dataset']} annotations"
    out.write_text(
        PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )

    n_sum = sum(len(e["summaries"]) for e in payload["episodes"])
    n_sub = sum(len(e["subtasks"]) for e in payload["episodes"])
    print(f"wrote {out}  ({len(payload['episodes'])} episodes, {n_sum} summary + {n_sub} subtask windows)")
    print("\n  uv run lerobot/src/lerobot/data_processing/annotate/serve_review.py "
          f"--data-dir {root}")


if __name__ == "__main__":
    main()

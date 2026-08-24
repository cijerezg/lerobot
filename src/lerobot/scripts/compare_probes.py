# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare probe output from one to four runs or checkpoints in a browser.

Examples::

    python -m lerobot.scripts.compare_probes outputs/ckpt400 outputs/ckpt800 outputs/ckpt1200
    python -m lerobot.scripts.compare_probes outputs/run/validation/step_00000400 \
        outputs/run/validation/step_00000800 --label step-400 --label step-800

The existing :mod:`lerobot.scripts.view_probes` remains the detailed single-run
viewer. This companion keeps runs as columns and aligns figures by their exact
probe-relative filename. With a fixed validation set, paths such as
``ep0000_L14/overlay_img_top_summary.mp4`` identify the same sampled view.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from lerobot.scripts.view_probes import _json_response, _send_file, build_index, resolve_validation_dir

MAX_RUNS = 4


@dataclass(frozen=True)
class CompareSource:
    """A validation tree and the column metadata derived from one CLI input."""

    input_path: Path
    val_dir: Path
    label: str
    initial_step: int | None


def _step_number(path: Path) -> int | None:
    if not path.name.startswith("step_"):
        return None
    suffix = path.name.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def resolve_compare_sources(
    run_dirs: list[str | Path], labels: list[str] | None = None
) -> list[CompareSource]:
    """Resolve one to four inputs while retaining an explicitly selected step."""
    if not 1 <= len(run_dirs) <= MAX_RUNS:
        raise ValueError(f"Expected between 1 and {MAX_RUNS} run paths, got {len(run_dirs)}.")
    labels = labels or []
    if len(labels) > len(run_dirs):
        raise ValueError(f"Got {len(labels)} labels for {len(run_dirs)} run paths.")

    sources = []
    for index, raw in enumerate(run_dirs):
        input_path = Path(raw).expanduser().resolve()
        val_dir = resolve_validation_dir(input_path)
        requested_step = _step_number(input_path)
        default_label = (
            f"{val_dir.parent.name} · {input_path.name}"
            if requested_step is not None
            else val_dir.parent.name
        )
        label = labels[index].strip() if index < len(labels) and labels[index].strip() else default_label
        sources.append(
            CompareSource(
                input_path=input_path,
                val_dir=val_dir,
                label=label,
                initial_step=requested_step,
            )
        )
    return sources


def _interactive_samples(
    probe_dir: Path, probe_id: str, panel_file: str
) -> tuple[str, list[dict]] | None:
    """Return the internal frame order for inspectors the comparison UI can coordinate."""
    if probe_id == "action_trace" and panel_file == "action_trace.html":
        metrics_path = probe_dir / "metrics.csv"
        if not metrics_path.is_file():
            return None
        with metrics_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        samples = [
            {
                "key": f"{row['episode']}:{row['frame']}",
                "label": f"episode {row['episode']} · frame {row['frame']}",
                "episode": int(row["episode"]),
                "frame": int(row["frame"]),
                "global_idx": int(row["global_idx"]),
            }
            for row in rows
        ]
        return "frame", samples

    if probe_id == "objective" and panel_file == "action_exemplars.html":
        summary_path = probe_dir / "objective.json"
        if not summary_path.is_file():
            return None
        try:
            with summary_path.open() as file:
                summary = json.load(file)
        except (OSError, ValueError):
            return None
        samples = []
        for band in summary.get("exemplars", []):
            for row in band.get("frames", []):
                samples.append(
                    {
                        "key": f"{row['episode_idx']}:{row['frame_idx']}",
                        "label": (
                            f"p{band['percentile']} {band['label']} · "
                            f"episode {row['episode_idx']} · frame {row['frame_idx']}"
                        ),
                        "episode": int(row["episode_idx"]),
                        "frame": int(row["frame_idx"]),
                        "global_idx": int(row["global_idx"]),
                    }
                )
        return "rank", samples
    return None


def build_compare_index(sources: list[CompareSource]) -> dict:
    """Build each normal viewer index and namespace its asset URLs by column."""
    runs = []
    groups: list[str] = []
    for source_index, source in enumerate(sources):
        index = build_index(source.val_dir)
        for group in index["groups"]:
            if group not in groups:
                groups.append(group)
        for probe in index["probes"]:
            for panels in probe["panels"].values():
                for panel in panels:
                    relative = panel["url"].removeprefix("/asset/")
                    step_name = relative.split("/", 1)[0]
                    interactive = _interactive_samples(
                        source.val_dir / step_name / probe["id"], probe["id"], panel["file"]
                    )
                    if interactive is not None:
                        panel["sample_alignment"], panel["samples"] = interactive
                    panel["url"] = f"/asset/{source_index}/{relative}"
        available_steps = {step["step"] for step in index["steps"]}
        initial_step = source.initial_step if source.initial_step in available_steps else None
        if initial_step is None and index["steps"]:
            initial_step = index["steps"][-1]["step"]
        index["run"].update(
            label=source.label,
            input_path=str(source.input_path),
            source_index=source_index,
            initial_step=initial_step,
        )
        runs.append(index)
    return {"runs": runs, "groups": groups, "max_runs": MAX_RUNS}


def make_compare_handler(sources: list[CompareSource]) -> type[BaseHTTPRequestHandler]:
    def asset_path(source_index: int, relative: str) -> Path:
        if not 0 <= source_index < len(sources):
            raise FileNotFoundError(relative)
        root = sources[source_index].val_dir
        path = (root / unquote(relative)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(relative)
        return path

    class CompareRequestHandler(BaseHTTPRequestHandler):
        server_version = "ProbeCompare/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_HEAD(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/asset/"):
                self._handle_asset(path.removeprefix("/asset/"), head_only=True)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(COMPARE_HTML)
            elif path == "/api/index":
                _json_response(self, build_compare_index(sources))
            elif path.startswith("/asset/"):
                self._handle_asset(path.removeprefix("/asset/"), head_only=False)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _handle_asset(self, relative: str, head_only: bool) -> None:
            try:
                source_text, asset_relative = relative.split("/", 1)
                path = asset_path(int(source_text), asset_relative)
            except (FileNotFoundError, OSError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                _send_file(self, path, head_only=head_only)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return CompareRequestHandler


COMPARE_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Probe comparison</title><style>
:root{color-scheme:dark;--bg:#0b0d12;--surface:#11151d;--surface2:#171c26;--surface3:#202735;
 --line:#2a3343;--text:#edf1f7;--muted:#8d99aa;--blue:#79a8ff;--green:#4ed49b;--amber:#f0b65b;
 --red:#ff746c;--info:#738196;--col-min:340px;font-family:Inter,ui-sans-serif,-apple-system,
 BlinkMacSystemFont,"Segoe UI",sans-serif}
@media(prefers-color-scheme:light){:root{color-scheme:light;--bg:#eef1f5;--surface:#fff;--surface2:#f5f7fa;
 --surface3:#e7ecf3;--line:#d5dce6;--text:#17202b;--muted:#697687;--blue:#2868cf;--green:#168a63;
 --amber:#b6791d;--red:#c44740;--info:#8995a4}}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--text)}
body{font:13px/1.45 Inter,ui-sans-serif,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
button,select,input{font:inherit}button,select,input[type=search]{border:1px solid var(--line);border-radius:7px;
 background:var(--surface2);color:var(--text);padding:6px 9px}button{cursor:pointer}button:hover,select:hover{border-color:#52647d}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
button.primary{background:#2769d8;border-color:#3479ea;color:#fff}button.active{border-color:var(--blue);background:#273852}
a{color:var(--blue)}.spacer{flex:1}.muted{color:var(--muted)}.mono{font:11px ui-monospace,SFMono-Regular,Menlo,monospace}
.shell{height:100%;display:flex;flex-direction:column}.top{height:58px;flex:none;display:flex;align-items:center;gap:13px;
 padding:9px 14px;background:var(--surface);border-bottom:1px solid var(--line);z-index:20}.mark{width:28px;height:28px;
 border-radius:8px;background:linear-gradient(145deg,var(--blue),#805adf);flex:none}.brand{min-width:0}.brand strong{display:block}
.brand small{display:block;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:48ch}
.top button{white-space:nowrap}.syncstate{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:11px}
.layout{min-height:0;flex:1;display:grid;grid-template-columns:224px minmax(0,1fr);transition:grid-template-columns .14s}
.layout.norail{grid-template-columns:0 minmax(0,1fr)}.layout.norail .rail{display:none}.rail{min-height:0;overflow:auto;
 background:var(--surface);border-right:1px solid var(--line);padding:10px}.rail input{width:100%;margin-bottom:8px}
.group{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);padding:12px 8px 4px}
.probe{width:100%;border:0;background:transparent;display:flex;gap:8px;align-items:center;padding:7px 8px;text-align:left}
.probe:hover{background:var(--surface2)}.probe.on{background:#285b9e;color:#fff}.probe .name{min-width:0;flex:1;
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.probe .count{font-size:10px;color:var(--muted)}.probe.on .count{color:#dceaff}
.dot{width:7px;height:7px;border-radius:50%;background:var(--info);display:inline-block;flex:none}.dot.good{background:var(--green)}
.dot.warn{background:var(--amber)}.dot.bad{background:var(--red)}.main{min-width:0;overflow:auto;padding:16px 16px 56px}
.intro{display:flex;align-items:flex-end;gap:16px;margin-bottom:12px}.eyebrow{font-size:10px;text-transform:uppercase;letter-spacing:.11em;
 color:var(--blue);font-weight:700}.intro h1{font-size:22px;line-height:1.2;letter-spacing:-.02em;margin:3px 0}.intro p{margin:3px 0;color:var(--muted)}
.alignnote{text-align:right;color:var(--muted);font-size:11px;max-width:48ch}.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
 position:sticky;top:-16px;z-index:12;padding:10px 11px;background:color-mix(in srgb,var(--surface) 95%,transparent);
 border:1px solid var(--line);border-radius:10px;box-shadow:0 4px 14px rgba(0,0,0,.12)}.field{display:flex;align-items:center;
 gap:5px;color:var(--muted);white-space:nowrap}.field select{max-width:min(44vw,520px);color:var(--text)}.check{display:flex;gap:5px;
 align-items:center;color:var(--muted);white-space:nowrap}.check input{accent-color:var(--blue)}.pill{padding:3px 8px;border-radius:99px;
 background:var(--surface3);color:var(--muted);font-size:10px}.pill.warn{color:var(--amber)}
.compare-scroll{overflow-x:auto;padding:12px 1px 5px}.compare-grid{display:grid;grid-template-columns:repeat(var(--run-count),
 minmax(var(--col-min),1fr));gap:10px;min-width:calc(var(--run-count) * var(--col-min) + (var(--run-count) - 1) * 10px)}
.compare-grid.one{grid-template-columns:minmax(300px,1000px);min-width:0;justify-content:center}.run-card{min-width:0;background:var(--surface);
 border:1px solid var(--line);border-radius:11px;padding:9px}.runhead{display:flex;align-items:center;gap:7px;padding:1px 2px 9px;min-height:34px}
.runhead strong{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.runhead select{margin-left:auto;max-width:132px;padding:4px 7px}
.media{position:relative;background:#050608;border-radius:7px;overflow:auto;max-height:74vh;min-height:130px;display:block}
.media img{display:block;height:auto;max-width:none;margin:0 auto}.media video{display:block;width:100%;height:auto;max-height:72vh;margin:0 auto}
.media iframe{display:block;width:100%;height:900px;border:0;background:#fff}.media.data{display:flex;align-items:center;justify-content:center;padding:38px 16px;
 background:var(--surface2)}.media.data a{font-weight:650}.missing{min-height:240px;display:grid;place-items:center;text-align:center;
.inspectnav{display:flex;align-items:center;gap:6px;padding:7px 1px}.inspectnav button{padding:4px 8px}.inspectnav .sample{min-width:0;
 flex:1;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--muted);font-size:10px}.alignmentwarn{margin-top:10px;
 padding:10px 12px;border:1px solid color-mix(in srgb,var(--amber) 50%,var(--line));border-radius:9px;background:color-mix(in srgb,var(--amber) 8%,var(--surface));color:var(--muted)}
 color:var(--muted);background:repeating-linear-gradient(135deg,var(--surface2),var(--surface2) 12px,var(--surface) 12px,var(--surface) 24px);
 border-radius:7px}.cardfoot{display:flex;gap:7px;align-items:center;padding:8px 2px 1px;color:var(--muted);font-size:10px}
.cardfoot .file{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.native{margin-left:auto;white-space:nowrap}.openfull{white-space:nowrap}
.guide{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-top:6px}.guide h3{font-size:13px;margin:0 0 4px}
.guide p{color:var(--muted);margin:4px 0}.metrics{margin-top:12px;background:var(--surface);border:1px solid var(--line);border-radius:10px;
 overflow:hidden}.sectionhead{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--line)}.sectionhead h3{font-size:13px;margin:0}
.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;min-width:650px}th,td{padding:8px 12px;border-bottom:1px solid var(--line);
 text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--surface);z-index:1}
th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}.metricname{font-weight:600}.metricnote{font-size:10px;
 color:var(--muted);font-weight:400;max-width:52ch}.reading{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap}
.reading.best{color:var(--green)}.delta{display:block;font-size:9px;color:var(--muted)}details.support summary{padding:9px 12px;cursor:pointer;
 color:var(--muted);border-top:1px solid var(--line)}.empty{padding:48px 20px;text-align:center;color:var(--muted)}
.lightbox{display:none;position:fixed;inset:0;z-index:50;background:rgba(2,4,7,.92);padding:48px 28px 28px;overflow:auto}
.lightbox.on{display:block}.lightbox img{display:block;max-width:none;height:auto;margin:0 auto;box-shadow:0 8px 40px #000}.lightclose{position:fixed;
 top:14px;right:16px;z-index:51}.refreshmsg{min-width:8ch;color:var(--muted);font-size:10px}.railbtn{font-size:16px;padding:3px 8px}
@media(max-width:850px){.layout{grid-template-columns:0 minmax(0,1fr)}.rail{display:none}.intro{align-items:flex-start;flex-direction:column}
 .alignnote{text-align:left}.top .syncstate{display:none}.main{padding:12px 10px 48px}.toolbar{top:-12px}}
</style></head><body><div class="shell">
<header class="top"><button class="railbtn" id="railbtn" title="Toggle probe list">◀</button><i class="mark"></i>
 <div class="brand"><strong>Probe comparison</strong><small id="runsummary">Loading runs…</small></div><div class="spacer"></div>
 <button id="playbtn" class="primary" onclick="togglePlayback()">▶ Play all</button>
 <label class="syncstate"><input id="synccheck" type="checkbox" checked onchange="syncEnabled=this.checked"><span class="dot good"></span>sync</label>
 <button onclick="reload()">Refresh</button><span class="refreshmsg" id="refreshmsg"></span></header>
<div class="layout" id="layout"><aside class="rail"><input id="filter" type="search" placeholder="filter probes…" aria-label="Filter probes">
 <div id="probelist"></div></aside><main class="main" id="main"><div class="empty">Loading probe output…</div></main></div>
<div class="lightbox" id="lightbox" onclick="if(event.target===this)closeLightbox()"><button class="lightclose" onclick="closeLightbox()">Close</button>
 <img id="lightimg" alt="Full-size probe figure"></div>
<script>
let DATA=null, stepIndices=[], cur=null, selectedFile=null, selectedContext="", sharedOnly=true;
let zoom="fit", columnSize="auto", syncEnabled=true, synchronizing=false, refreshTimer=null;
let interactiveKey=null, runSampleIndices={};
const severity={bad:3,warn:2,good:1,info:0};
function esc(s){return String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function attr(s){return esc(s).replace(/"/g,"&quot;");}
function natural(a,b){return String(a).localeCompare(String(b),undefined,{numeric:true});}
function stepKey(ri){const run=DATA.runs[ri], i=stepIndices[ri];return run.steps[i]?String(run.steps[i].step):null;}
function probeAt(ri,id=cur){return DATA.runs[ri].probes.find(p=>p.id===id&&p.panels[stepKey(ri)]!==undefined)||null;}
function panelsAt(ri,id=cur){const p=probeAt(ri,id);return p?(p.panels[stepKey(ri)]||[]):[];}
function statusAt(ri,id=cur){const p=probeAt(ri,id);return p?(p.status[stepKey(ri)]||"info"):"info";}
function worstStatus(statuses){return statuses.reduce((a,b)=>(severity[b]||0)>(severity[a]||0)?b:a,"info");}
function probeCatalog(){
 const found=new Map(); DATA.runs.forEach((run,ri)=>run.probes.forEach(p=>{if(p.panels[stepKey(ri)]===undefined)return;
  const row=found.get(p.id)||{id:p.id,title:p.title,group:p.group,claim:p.claim,count:0,statuses:[]};
  row.count++;row.statuses.push(p.status[stepKey(ri)]||"info");found.set(p.id,row);}));
 return [...found.values()].sort((a,b)=>{const ga=DATA.groups.indexOf(a.group),gb=DATA.groups.indexOf(b.group);
  return (ga<0?999:ga)-(gb<0?999:gb)||natural(a.title,b.title);});
}
function panelCatalog(all=false){
 const found=new Map();DATA.runs.forEach((_,ri)=>panelsAt(ri).forEach(pn=>{const row=found.get(pn.file)||{file:pn.file,
  caption:pn.caption||pn.file,how:pn.how||"",kind:pn.kind||"data",primary:false,count:0};row.count++;row.primary=row.primary||pn.primary;
  if(!row.how&&pn.how)row.how=pn.how;found.set(pn.file,row);}));
 let rows=[...found.values()].sort((a,b)=>(b.primary?1:0)-(a.primary?1:0)||natural(a.file,b.file));
 return all||!sharedOnly?rows:rows.filter(x=>x.count===DATA.runs.length);
}
function dirOf(file){const i=file.lastIndexOf("/");return i<0?"":file.slice(0,i);}
function baseOf(file){const i=file.lastIndexOf("/");return i<0?file:file.slice(i+1);}
function normalizeSelection(){
 const rows=panelCatalog();if(rows.some(x=>x.file===selectedFile)){selectedContext=dirOf(selectedFile);return;}
 const contexts=[...new Set(rows.map(x=>dirOf(x.file)))].sort(natural);
 if(!contexts.includes(selectedContext))selectedContext=contexts[0]||"";
 const here=rows.filter(x=>dirOf(x.file)===selectedContext);selectedFile=(here.find(x=>x.primary)||here[0]||{}).file||null;
}
function resetInteractive(){interactiveKey=null;runSampleIndices={};}
function setProbe(id){cur=id;selectedFile=null;selectedContext="";resetInteractive();normalizeSelection();draw();document.getElementById("main").scrollTop=0;}
function setContext(value){selectedContext=value;const here=panelCatalog().filter(x=>dirOf(x.file)===value);
 selectedFile=(here.find(x=>x.primary)||here[0]||{}).file||null;resetInteractive();drawMain();}
function setFile(value){selectedFile=value;selectedContext=dirOf(value);resetInteractive();drawMain();}
function setStep(ri,value){const at=DATA.runs[ri].steps.findIndex(s=>String(s.step)===value);if(at>=0)stepIndices[ri]=at;
 const available=probeCatalog();if(!available.some(p=>p.id===cur))cur=available[0]?.id||null;resetInteractive();normalizeSelection();draw();}
function setShared(value){sharedOnly=value;normalizeSelection();drawMain();}
function currentPanel(){return panelCatalog(true).find(x=>x.file===selectedFile)||null;}
function interactiveInfo(){
 const panels=DATA.runs.map((_,ri)=>panelsAt(ri).find(pn=>pn.file===selectedFile)||null);
 const perRun=panels.map(pn=>pn?.samples||[]),hasSamples=perRun.some(rows=>rows.length);
 if(!hasSamples)return {panels,perRun,common:[],hasSamples:false,alignment:null};
 const common=perRun.every(rows=>rows.length)?perRun[0].filter(sample=>
  perRun.slice(1).every(rows=>rows.some(other=>other.key===sample.key))):[];
 return {panels,perRun,common,hasSamples:true,alignment:panels.find(pn=>pn?.sample_alignment)?.sample_alignment||"frame"};
}
function normalizeInteractive(info){
 if(info.common.length&&!info.common.some(sample=>sample.key===interactiveKey))interactiveKey=info.common[0].key;
 if(!info.common.length)interactiveKey=null;
 info.perRun.forEach((samples,ri)=>{const value=runSampleIndices[ri];
  if(value===undefined||value<0||value>=samples.length)runSampleIndices[ri]=0;});
}
function commandInspector(frame,index,attempt=0){
 if(!frame||index<0||!frame.isConnected)return;
 try{
  const win=frame.contentWindow,doc=frame.contentDocument,plot=doc.getElementById("action-inspector-plot");
  if(typeof win.probeInspectorGo==="function"){win.probeInspectorGo(index);frame.dataset.sampleIndex=index;return;}
  if(win.Plotly&&plot&&typeof plot.emit==="function"){
   win.Plotly.animate(plot,[String(index)],{mode:"immediate",frame:{duration:0,redraw:true},transition:{duration:0}});
   win.Plotly.relayout(plot,{"sliders[0].active":index});
   plot.emit("plotly_sliderchange",{step:{value:String(index),args:[[String(index)]]}});
   frame.dataset.sampleIndex=index;return;
  }
  const info=interactiveInfo(),count=info.perRun[+frame.dataset.run]?.length||0,current=+(frame.dataset.sampleIndex||0);
  const forward=count?(index-current+count)%count:0,backward=count?(current-index+count)%count:0;
  const button=doc.getElementById(forward<=backward?"next":"prev"),clicks=Math.min(forward,backward);
  if(button){for(let click=0;click<clicks;click++)button.click();frame.dataset.sampleIndex=index;return;}
 }catch(error){}
 if(attempt<12)setTimeout(()=>commandInspector(frame,index,attempt+1),180);
}
function onInspectorLoad(frame,ri){
 const info=interactiveInfo(),samples=info.perRun[ri]||[];
 const index=info.common.length?samples.findIndex(sample=>sample.key===interactiveKey):(runSampleIndices[ri]||0);
 if(index>=0)commandInspector(frame,index);
}
function updateInspectorLabels(info){
 const select=document.getElementById("interactive-select");if(select&&interactiveKey)select.value=interactiveKey;
 info.perRun.forEach((samples,ri)=>{const label=document.getElementById("sample-"+ri),index=runSampleIndices[ri]||0;
  if(label)label.textContent=samples[index]?.label||"no samples";});
}
function setInteractiveKey(key){
 interactiveKey=key;const info=interactiveInfo();
 document.querySelectorAll("iframe[data-run]").forEach(frame=>{const ri=+frame.dataset.run;
  const index=(info.perRun[ri]||[]).findIndex(sample=>sample.key===key);if(index>=0)commandInspector(frame,index);});
 updateInspectorLabels(info);
}
function stepInteractive(delta){
 const info=interactiveInfo();if(!info.common.length)return;
 const at=Math.max(0,info.common.findIndex(sample=>sample.key===interactiveKey));
 setInteractiveKey(info.common[(at+delta+info.common.length)%info.common.length].key);
}
function stepRunSample(ri,delta){
 const info=interactiveInfo(),samples=info.perRun[ri]||[];if(!samples.length)return;
 runSampleIndices[ri]=((runSampleIndices[ri]||0)+delta+samples.length)%samples.length;
 const frame=document.querySelector(`iframe[data-run="${ri}"]`);commandInspector(frame,runSampleIndices[ri]);updateInspectorLabels(info);
}
function interactiveControls(info){
 if(!info.hasSamples)return "";
 if(!info.common.length)return `<span class="pill warn">no common frames</span>`;
 const options=info.common.map(sample=>`<option value="${attr(sample.key)}" ${sample.key===interactiveKey?'selected':''}>${esc(sample.label)}</option>`).join("");
 return `<button onclick="stepInteractive(-1)">← Previous frame</button><label class="field">Frame <select id="interactive-select"
  onchange="setInteractiveKey(this.value)">${options}</select></label><button onclick="stepInteractive(1)">Next frame →</button>
  <span class="pill">${info.common.length} aligned frames</span>`;
}
function individualNav(ri,pn,info){
 if(!pn?.samples?.length||info.common.length)return "";
 const index=runSampleIndices[ri]||0,label=pn.samples[index]?.label||"";
 return `<div class="inspectnav"><button onclick="stepRunSample(${ri},-1)">← Prev</button><span class="sample" id="sample-${ri}"
  title="${attr(label)}">${esc(label)}</span><button onclick="stepRunSample(${ri},1)">Next →</button></div>`;
}
function alignmentWarning(info){
 if(!info.hasSamples||info.common.length)return "";
 const ranked=info.alignment==="rank";
 const shortcut=probeCatalog().some(probe=>probe.id==="action_trace")?
  ` <button onclick="setProbe('action_trace')">Open fixed-frame Action Inspector</button>`:"";
 return `<div class="alignmentwarn"><strong>${ranked?'Rank-aligned, not frame-aligned.':'No shared internal frames.'}</strong>
  ${ranked?'Each Objective inspector selects its own p5/p50/p95 loss exemplars, so the same rank refers to a different observation in each run.':
  'The selected inspector files do not contain one frame set shared by every run.'}
  Same-frame comparison is not possible from this artifact.${shortcut}</div>`;
}

function figureHtml(pn,ri){
 if(pn.kind==="image")return `<div class="media"><img src="${attr(pn.url)}" alt="${attr(pn.caption)}" onload="fitImage(this)"
  ondblclick="openLightbox(this.src)"></div>`;
 if(pn.kind==="video")return `<div class="media"><video controls preload="metadata" playsinline muted src="${attr(pn.url)}"></video></div>`;
 if(pn.kind==="html")return `<div class="media"><iframe data-run="${ri}" src="${attr(pn.url)}" loading="lazy" onload="onInspectorLoad(this,${ri})" title="${attr(pn.caption)}"></iframe></div>`;
 return `<div class="media data"><a href="${attr(pn.url)}" target="_blank">Open ${esc(baseOf(pn.file))} ↗</a></div>`;
}
function imageScale(img){const box=img.parentElement,nat=img.naturalWidth||box.clientWidth;
 if(zoom==="fit")return Math.min(1,box.clientWidth/nat);return parseFloat(zoom);}
function fitImage(img){const scale=imageScale(img),nat=img.naturalWidth||img.parentElement.clientWidth;img.style.width=Math.round(nat*scale)+"px";
 const label=img.closest(".run-card").querySelector(".native");if(label)label.textContent=`${nat}×${img.naturalHeight} · ${Math.round(scale*100)}%`;}
function setZoom(value){zoom=value;document.querySelectorAll(".media img").forEach(fitImage);}
function openLightbox(url){lightimg.src=url;lightbox.classList.add("on");}
function closeLightbox(){lightbox.classList.remove("on");lightimg.removeAttribute("src");}
function drawList(){
 const q=filter.value.trim().toLowerCase(),catalog=probeCatalog();
 probelist.innerHTML=DATA.groups.map(group=>{const rows=catalog.filter(p=>p.group===group&&(!q||(p.title+p.id+p.claim).toLowerCase().includes(q)));
  if(!rows.length)return "";return `<div class="group">${esc(group)}</div>`+rows.map(p=>`<button class="probe ${p.id===cur?'on':''}"
   onclick="setProbe('${attr(p.id)}')" title="${attr(p.claim)}"><i class="dot ${worstStatus(p.statuses)}"></i><span class="name">${esc(p.title)}</span>
   <span class="count">${p.count}/${DATA.runs.length}</span></button>`).join("");}).join("")||`<div class="empty">No matching probes</div>`;
}
function panelControls(){
 const rows=panelCatalog(),contexts=[...new Set(rows.map(x=>dirOf(x.file)))].sort(natural),here=rows.filter(x=>dirOf(x.file)===selectedContext);
 const sharedCount=currentPanel()?.count||0;
 return `<label class="field">Sample / facet <select onchange="setContext(this.value)">${contexts.map(x=>`<option value="${attr(x)}"
  ${x===selectedContext?'selected':''}>${esc(x||'top level')}</option>`).join('')}</select></label>
 <label class="field">Figure <select onchange="setFile(this.value)">${here.map(x=>`<option value="${attr(x.file)}" ${x.file===selectedFile?'selected':''}>
  ${x.primary?'★ ':''}${esc(x.caption!==x.file?x.caption:baseOf(x.file))}</option>`).join('')}</select></label>
 <span class="pill ${sharedCount<DATA.runs.length?'warn':''}">${sharedCount}/${DATA.runs.length} runs</span>`;
}
function stepSelect(run,ri){if(run.steps.length<=1)return `<span class="pill">step ${run.steps[0]?.step??'—'}</span>`;
 return `<select aria-label="Step for ${attr(run.run.label)}" onchange="setStep(${ri},this.value)">${run.steps.map((s,i)=>`<option value="${s.step}"
  ${i===stepIndices[ri]?'selected':''}>step ${s.step.toLocaleString()}</option>`).join('')}</select>`;}
function comparisonGrid(panel,info){
 return `<div class="compare-scroll"><section class="compare-grid ${DATA.runs.length===1?'one':''}" style="--run-count:${DATA.runs.length}">
 ${DATA.runs.map((run,ri)=>{const pn=panelsAt(ri).find(x=>x.file===selectedFile),status=statusAt(ri);
  return `<article class="run-card"><div class="runhead"><i class="dot ${status}"></i><strong title="${attr(run.run.dir)}">${esc(run.run.label)}</strong>
   ${stepSelect(run,ri)}</div>${pn?individualNav(ri,pn,info):''}${pn?figureHtml(pn,ri):`<div class="missing"><div><strong>Not written for this run</strong><br><span class="mono">${esc(selectedFile)}</span></div></div>`}
   <div class="cardfoot"><span class="file mono">${esc(selectedFile)}</span><span class="native"></span>${pn?`<a class="openfull" href="${attr(pn.url)}"
    target="_blank">open ↗</a>`:''}</div></article>`;}).join('')}</section></div>
 ${panel?`<section class="guide"><h3>${esc(panel.caption)}</h3>${panel.how?`<p>${esc(panel.how)}</p>`:''}
  <p class="mono">${esc(panel.file)}</p></section>`:''}`;
}
function metricCatalog(){
 const found=new Map();DATA.runs.forEach((_,ri)=>{const p=probeAt(ri);if(!p)return;p.metrics.forEach(m=>{const row=found.get(m.key)||
  {key:m.key,label:m.label,note:m.note||"",good:m.good,fmt:m.fmt,primary:false,cells:Array(DATA.runs.length).fill(null)};
  row.primary=row.primary||m.primary;row.cells[ri]={value:m.values[stepKey(ri)],status:m.statuses?.[stepKey(ri)]||"info",fmt:m.fmt};found.set(m.key,row);});});
 return [...found.values()].sort((a,b)=>(b.primary?1:0)-(a.primary?1:0)||natural(a.label,b.label));
}
function fmt(value,digits){if(value===null||value===undefined)return "—";if(digits===0)return String(Math.round(value));
 return value!==0&&Math.abs(value)<Math.pow(10,-digits)?value.toExponential(2):value.toFixed(digits);}
function metricTable(rows){if(!rows.length)return "";return `<div class="tablewrap"><table><thead><tr><th>Metric</th>${DATA.runs.map(r=>
 `<th title="${attr(r.run.dir)}">${esc(r.run.label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>{const values=row.cells.map(c=>c?.value)
  .filter(v=>v!==null&&v!==undefined),best=row.good==="none"||!values.length?null:(row.good==="high"?Math.max(...values):Math.min(...values));
  const baseline=row.cells[0]?.value;return `<tr><td><div class="metricname">${row.primary?'★ ':''}${esc(row.label)}</div>${row.note?`<div class="metricnote">${esc(row.note)}</div>`:''}</td>`+
  row.cells.map(cell=>{if(!cell||cell.value===null||cell.value===undefined)return `<td class="reading">—</td>`;const delta=baseline===null||baseline===undefined?null:cell.value-baseline;
   return `<td class="reading ${cell.value===best?'best':''}"><i class="dot ${cell.status}"></i> ${fmt(cell.value,cell.fmt)}${delta&&Math.abs(delta)>1e-12?
    `<span class="delta">${delta>0?'+':''}${fmt(delta,cell.fmt)} vs first</span>`:''}</td>`;}).join('')+`</tr>`;}).join('')}</tbody></table></div>`;}
function metricsHtml(){const rows=metricCatalog(),primary=rows.filter(x=>x.primary),support=rows.filter(x=>!x.primary);if(!rows.length)return "";
 return `<section class="metrics"><div class="sectionhead"><h3>Metric comparison</h3><span class="pill">selected steps</span></div>${metricTable(primary.length?primary:rows)}
 ${primary.length&&support.length?`<details class="support"><summary>${support.length} supporting metrics</summary>${metricTable(support)}</details>`:''}</section>`;}
function drawMain(){
 const main=document.getElementById("main"),catalog=probeCatalog(),meta=catalog.find(p=>p.id===cur);if(!meta){main.innerHTML=`<div class="empty">No probes at the selected steps.</div>`;return;}
 normalizeSelection();const all=panelCatalog(true),visible=panelCatalog(),panel=currentPanel(),info=interactiveInfo();
 normalizeInteractive(info);
 const kind=panel?.kind||"data",automatic=kind==="html"?"520":kind==="image"?"380":"340";
 document.documentElement.style.setProperty("--col-min",(columnSize==="auto"?automatic:columnSize)+"px");
 const noShared=sharedOnly&&!visible.length&&all.length;
 main.innerHTML=`<div class="intro"><div><div class="eyebrow">${esc(meta.group)}</div><h1>${esc(meta.title)}</h1><p>${esc(meta.claim||"")}</p></div>
  <div class="spacer"></div><div class="alignnote">Files align by relative name; interactive inspectors align by episode and frame when those keys are available.</div></div>
 <div class="toolbar">${visible.length?panelControls():''}<label class="check"><input type="checkbox" ${sharedOnly?'checked':''} onchange="setShared(this.checked)">shared files only</label>${interactiveControls(info)}
  <span class="spacer"></span>${kind==="image"?`<label class="field">Zoom <select onchange="setZoom(this.value)"><option value="fit" ${zoom==='fit'?'selected':''}>fit column</option>
   <option value="0.5" ${zoom==='0.5'?'selected':''}>50%</option><option value="1" ${zoom==='1'?'selected':''}>100%</option><option value="1.5" ${zoom==='1.5'?'selected':''}>150%</option></select></label>`:''}
  <label class="field">Columns <select onchange="columnSize=this.value;drawMain()"><option value="auto" ${columnSize==='auto'?'selected':''}>adaptive</option>
   <option value="300" ${columnSize==='300'?'selected':''}>compact</option><option value="440" ${columnSize==='440'?'selected':''}>wide</option>
   <option value="600" ${columnSize==='600'?'selected':''}>inspect</option></select></label></div>
 ${alignmentWarning(info)}${noShared?`<div class="empty">No exact-path figures are shared by all ${DATA.runs.length} runs.<br><button onclick="setShared(false)">Show the union with missing cells</button></div>`:
   (!visible.length?`<div class="empty">This probe wrote no viewable panels at the selected steps.</div>`:comparisonGrid(panel,info))}${metricsHtml()}`;
 wireMedia();document.querySelectorAll(".media img").forEach(img=>{if(img.complete)fitImage(img);});
}
function wireMedia(){
 const videos=[...document.querySelectorAll(".media video")];videos.forEach(v=>{v.addEventListener("play",()=>syncFrom(v,"play"));
  v.addEventListener("pause",()=>syncFrom(v,"pause"));v.addEventListener("seeked",()=>syncFrom(v,"seek"));v.addEventListener("ratechange",()=>syncFrom(v,"rate"));});
}
function syncFrom(source,action){if(!syncEnabled||synchronizing)return;synchronizing=true;const videos=[...document.querySelectorAll(".media video")];
 videos.forEach(v=>{if(v===source)return;if(Math.abs(v.currentTime-source.currentTime)>.08)v.currentTime=source.currentTime;v.playbackRate=source.playbackRate;
  if(action==="play")v.play().catch(()=>{});if(action==="pause")v.pause();});synchronizing=false;}
function togglePlayback(){const videos=[...document.querySelectorAll(".media video")];if(!videos.length)return;const play=videos.some(v=>v.paused);
 videos.forEach(v=>play?v.play().catch(()=>{}):v.pause());playbtn.textContent=play?"Ⅱ Pause all":"▶ Play all";}
function draw(){drawList();drawMain();runsummary.textContent=`${DATA.runs.length} run${DATA.runs.length===1?'':'s'} · up to 4 columns`;}
function initializeSteps(previousValues=[]){stepIndices=DATA.runs.map((run,ri)=>{const wanted=previousValues[ri]??run.run.initial_step;
 const at=run.steps.findIndex(s=>s.step===wanted);return at>=0?at:Math.max(0,run.steps.length-1);});}
function flash(message){refreshmsg.textContent=message;clearTimeout(refreshTimer);refreshTimer=setTimeout(()=>refreshmsg.textContent="",2200);}
async function reload(){const old=DATA,oldSteps=old?old.runs.map((_,ri)=>+stepKey(ri)):[];try{const response=await fetch("/api/index");
 if(!response.ok)throw new Error(response.status);DATA=await response.json();initializeSteps(oldSteps);const catalog=probeCatalog();
 if(!catalog.some(p=>p.id===cur))cur=(catalog.find(p=>p.count===DATA.runs.length)||catalog[0])?.id||null;normalizeSelection();draw();flash(old?"Up to date":"");}
 catch(error){console.error(error);main.innerHTML=`<div class="empty">Could not load probe outputs.</div>`;flash("Refresh failed");}}
filter.addEventListener("input",drawList);railbtn.addEventListener("click",()=>{layout.classList.toggle("norail");railbtn.textContent=layout.classList.contains("norail")?"▶":"◀";});
addEventListener("resize",()=>document.querySelectorAll(".media img").forEach(fitImage));addEventListener("keydown",event=>{
 if(event.key==="Escape"){closeLightbox();return}if(event.target.tagName==="INPUT"||event.target.tagName==="SELECT")return;
 const probes=probeCatalog(),at=probes.findIndex(p=>p.id===cur);if(event.key==="j"&&probes.length)setProbe(probes[Math.min(probes.length-1,at+1)].id);
 if(event.key==="k"&&probes.length)setProbe(probes[Math.max(0,at-1)].id);if(event.key==="n"||event.key==="p"){
  const rows=panelCatalog(),i=rows.findIndex(x=>x.file===selectedFile),next=Math.max(0,Math.min(rows.length-1,i+(event.key==="n"?1:-1)));
  if(rows[next])setFile(rows[next].file)}if(event.code==="Space"){event.preventDefault();togglePlayback()}});
if(innerWidth<=850){layout.classList.add("norail");railbtn.textContent="▶"}reload();
</script></body></html>"""


def _make_server(
    sources: list[CompareSource], host: str, port: int, retries: int = 20
) -> ThreadingHTTPServer:
    handler = make_compare_handler(sources)
    last_error: OSError | None = None
    for candidate in range(port, port + retries + 1):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            last_error = exc
            if port == 0:
                break
    raise RuntimeError(f"Could not bind probe comparison server: {last_error}") from last_error


def compare_probes(
    run_dirs: list[str | Path],
    labels: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 7871,
    open_browser: bool = True,
) -> None:
    """Launch the comparison viewer and block until interrupted."""
    sources = resolve_compare_sources(run_dirs, labels)
    server = _make_server(sources, host, port)
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    index = build_compare_index(sources)
    print(f"Probe comparison: {url}")
    for position, run in enumerate(index["runs"], start=1):
        print(f"  {position}. {run['run']['label']}: {run['run']['val_dir']}")
    print(f"{len(sources)} run(s), {len(index['groups'])} probe group(s)")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping probe comparison.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare probe output across one to four runs or checkpoints.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Run, validation, or step_* directories to compare (one to four paths).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Column label, repeated in the same order as the paths.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7871)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab.")
    args = parser.parse_args()
    try:
        compare_probes(
            args.run_dirs,
            labels=args.label,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

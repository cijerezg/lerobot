"""Review recorded inference episodes in the browser and cut them into a training root.

    uv run python -m lerobot.annotation.rebot.rollout_review outputs/train/<date>/<run> [<run> ...] [--out outputs/<root>] [--port 7871]

Renders one review video per episode (top | wrist, burned-in t = episode seconds, red INTERVENTION tag) into
migration/rollout_review_<date>/videos when missing, then opens http://127.0.0.1:<port>. In the page: pick an episode,
play it (space; 1x / 2x / 4x; arrows seek 1 s, shift+arrows 10 s), press [ at the start and ] at the end of every stretch
to REMOVE (or type the spans: "15-31, 36-43, 50-"), then Save. Save writes <work>/plan.json, builds the root with
build_root --plan (the kept spans of a source episode are spliced into one episode, boundaries snapped to multiples of 3
frames so the depth phase survives), runs verify_root and writes the README ledger. The output path is --out, the path
in the page, or outputs/rebot_rollouts_<date>-cut-v<N> with the first free N. Episodes without marks are kept whole;
untick an episode to leave it out of the root.
"""

import argparse
import io
import json
import subprocess
import threading
import traceback
import webbrowser
from contextlib import redirect_stdout, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot import build_root, verify_root
from lerobot.scripts.view_probes import _json_response, _send_file

FPS = 30


def dataset(run):
    run = Path(run).resolve()
    return run / "inference_dataset" if (run / "inference_dataset").is_dir() else run


def episodes(root):
    return pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))], ignore_index=True
    )


def intervention_spans(root, episode):
    """[(start s, end s)] of operator teleop inside an episode."""
    if not (root / "meta/online_labels.parquet").exists():
        return []
    o = pd.read_parquet(root / "meta/online_labels.parquet")
    iv = o[o.episode_index == episode].sort_values("frame_index").is_intervention.to_numpy().astype(int)
    pad = np.r_[0, iv, 0]
    return [
        (s / FPS, e / FPS)
        for s, e in zip(np.flatnonzero(np.diff(pad) == 1), np.flatnonzero(np.diff(pad) == -1), strict=True)
    ]


def video(root, e, cam):
    pre = f"videos/observation.images.{cam}"
    return (
        root
        / pre
        / f"chunk-{int(e[pre + '/chunk_index']):03d}"
        / f"file-{int(e[pre + '/file_index']):03d}.mp4"
    )


def inventory(runs, work):
    """One row per episode of the runs: id, run, episode, frames, seconds, task, interventions, review video path."""
    rows = []
    runs = [Path(run).resolve() for run in runs]
    short = len({run.name[:8] for run in runs}) == len(runs)  # rollout runs start HH-MM-SS; teleop roots share a prefix
    for run in runs:
        tag = run.name[:8] if short else run.name
        root = dataset(run)
        for _, e in episodes(root).iterrows():
            ep, n = int(e.episode_index), int(e.length)
            rows.append(
                {
                    "id": f"{tag} ep{ep}",
                    "run": str(run),
                    "episode": ep,
                    "frames": n,
                    "seconds": n / FPS,
                    "task": list(e.tasks)[0],
                    "interventions": intervention_spans(root, ep),
                    "video": work / "videos" / f"{tag}_ep{ep}.mp4",
                    "_e": e,
                }
            )
    return rows


def render(rows):
    """Review video per episode (missing ones only): top | wrist at 1x with the episode clock and the INTERVENTION tag."""
    jobs = []
    for r in rows:
        if r["video"].exists():
            continue
        r["video"].parent.mkdir(parents=True, exist_ok=True)
        root, e, n = dataset(r["run"]), r["_e"], r["frames"]
        enable = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in r["interventions"]) or "0"
        fc = (
            f"[0:v]setpts=PTS-STARTPTS,drawtext=text='{r['id']}   t=%{{pts\\:hms}}':x=10:y=10:fontsize=28:fontcolor=yellow"
            f":box=1:boxcolor=black@0.6,drawtext=text='INTERVENTION':x=10:y=50:fontsize=28:fontcolor=red:box=1"
            f":boxcolor=black@0.6:enable='{enable}'[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]hstack=inputs=2,fps={FPS}[v]"
        )
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
        for cam in ("top", "wrist"):
            pre = f"videos/observation.images.{cam}"
            cmd += [
                "-ss",
                f"{e[pre + '/from_timestamp']:.4f}",
                "-t",
                f"{n / FPS:.4f}",
                "-i",
                str(video(root, e, cam)),
            ]
        cmd += [
            "-filter_complex",
            fc,
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(r["video"]),
        ]
        print("rendering", r["video"].relative_to(WORKSPACE), f"({r['seconds']:.0f} s)", flush=True)
        jobs.append((subprocess.Popen(cmd), r["video"]))
    for p, target in jobs:
        assert p.wait() == 0, target


def seconds(text):
    """'83', '1:23', '1:23.5' -> seconds."""
    parts = text.strip().split(":")
    return float(parts[-1]) + 60 * (int(parts[-2]) if len(parts) > 1 else 0)


def snap(s, n):
    """Episode seconds -> frame, a multiple of 3 inside [0, n - n % 3]."""
    return int(min(max(3 * round(s * FPS / 3), 0), n - n % 3))


def spans(text, n):
    """'15-31, 36-43, 50-' -> [(a, b)) frames, sorted, overlaps merged; an open end runs to the episode end."""
    out = []
    for item in text.split(","):
        if not item.strip():
            continue
        a, b = item.split("-")
        out.append((snap(seconds(a), n), snap(seconds(b), n) if b.strip() else n - n % 3))
    merged = []  # overlapping or touching marks remove their union
    for a, b in sorted(out):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        elif a < b:
            merged.append((a, b))
    return merged


def plan_entry(row, remove, why):
    """Plan row for build_root: keep = [first kept, last kept), cuts = the removed stretches between them."""
    n = row["frames"]
    kept, cursor = [], 0
    for a, b in spans(remove, n) + [(n - n % 3, n - n % 3)]:
        if a > cursor:
            kept.append((cursor, a))
        cursor = max(cursor, b)
    assert kept, f"{row['id']}: nothing kept"
    run, src = Path(row["run"]), dataset(row["run"])
    return {
        "source": str(src.relative_to(WORKSPACE) if src.is_relative_to(WORKSPACE) else src),
        "episode": row["episode"],
        "keep": [kept[0][0], kept[-1][1]],
        "cuts": [[kept[i][1], kept[i + 1][0]] for i in range(len(kept) - 1)],
        "task": row["task"],
        "run": run.name,
        "recorded_date": run.parent.name,
        "remove": remove,
        "why": why,
    }


def default_out(date):
    n = 1
    while (WORKSPACE / f"outputs/rebot_rollouts_{date}-cut-v{n}").exists():
        n += 1
    return f"outputs/rebot_rollouts_{date}-cut-v{n}"


def build(work, out, plan):
    """plan.json -> root (build_root --plan) -> verify_root -> README ledger."""
    out = WORKSPACE / out
    assert not out.exists(), f"{out} exists; pick another path"
    (work / "plan.json").write_text(json.dumps(plan, indent=1))
    build_root.build(build_root.rows_from_plan(work), out)
    verify_root.check(out)
    rows = "\n".join(
        f"| {i} | {p['run']} | {p['episode']} | {p['keep']} | {p['cuts']} | {p['remove'] or 'whole'} | {p['why'] or ''} |"
        for i, p in enumerate(plan)
    )
    (out / "README.md").write_text(
        f"# {out.name}\n\nPolicy rollouts cut by the user's review (`lerobot.annotation.rebot.rollout_review`), NOT yet annotated."
        f" Sources untouched under outputs/train. Plan: `{(work / 'plan.json').relative_to(WORKSPACE)}`; verify_root PASS ->"
        f" meta/cleanup_verification.json. `meta/tasks.parquet` = operator-typed task strings, to revise during annotation."
        f" `meta/online_labels.parquet` keeps is_intervention + the typed `recorded_subtask`.\n\n"
        f"| ep | source run | src ep | kept (source frames) | spliced out | removed (episode s) | why |\n"
        f"|---|---|---|---|---|---|---|\n{rows}\n"
    )
    print("DONE", out.relative_to(WORKSPACE), flush=True)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Rollout review</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;display:grid;grid-template-columns:300px 1fr;height:100vh;background:#1e1e1e;color:#ddd}
#side{border-right:1px solid #444;padding:10px;overflow:auto}#main{padding:10px;overflow:auto}
video{width:100%;max-width:1280px;background:#000;display:block}
#timeline{width:100%;max-width:1280px;height:44px;background:#333;display:block;cursor:pointer}
.ep{padding:4px;cursor:pointer;border-radius:4px}.ep.sel{background:#345}
textarea,input{width:100%;box-sizing:border-box;background:#111;color:#ddd;border:1px solid #555;padding:4px;font:inherit}
button{background:#345;color:#ddd;border:1px solid #567;padding:4px 10px;margin:2px;cursor:pointer}
pre{background:#111;padding:8px;max-height:260px;overflow:auto;white-space:pre-wrap;font-size:12px}
.k{color:#999;font-size:12px}h3{margin:12px 0 4px}
</style></head><body>
<div id="side"><h3>Episodes</h3><div id="eps"></div>
<p class="k">Tick = goes into the root. Episodes without marks are kept whole.</p>
<h3>Output root</h3><input id="out"><p class="k">note (why)</p><input id="why">
<p><button id="save">Save: build + verify</button></p><div id="state" class="k"></div></div>
<div id="main"><div id="title" class="k"></div>
<video id="v" preload="auto"></video><canvas id="timeline" height="44"></canvas>
<div><button id="play">play</button> speed <select id="speed"><option>1</option><option selected>2</option><option>4</option></select>
<span id="t" style="margin:0 12px"></span>
<button id="mstart">[ remove from here</button><button id="mend">] to here</button><button id="mtoend">remove to the end</button><button id="clear">clear</button></div>
<p class="k">Remove spans in episode seconds, comma separated, "50-" = to the end. Keys: space play/pause, &larr;/&rarr; 1 s, shift 10 s, [ ] mark.</p>
<textarea id="spans" rows="2"></textarea><div id="summary" class="k"></div><pre id="log"></pre></div>
<script>
const $=id=>document.getElementById(id);
const v=$('v'),tl=$('timeline'),ta=$('spans');let eps=[],cur=null,marks={},include={},pending=null;
fetch('/api/episodes').then(r=>r.json()).then(d=>{eps=d.episodes;$('out').value=d.out;eps.forEach(e=>{marks[e.id]='';include[e.id]=true});select(eps[0].id)});
function list(){const el=$('eps');el.innerHTML='';eps.forEach(e=>{const d=document.createElement('div');d.className='ep'+(cur&&cur.id===e.id?' sel':'');
const cb=document.createElement('input');cb.type='checkbox';cb.checked=include[e.id];cb.style.width='auto';cb.onchange=()=>include[e.id]=cb.checked;cb.onclick=ev=>ev.stopPropagation();
d.appendChild(cb);d.appendChild(document.createTextNode(` ${e.id}  ${e.seconds.toFixed(0)} s`+(marks[e.id].trim()?'  *':'')));d.onclick=()=>select(e.id);el.appendChild(d)})}
function select(id){if(cur)marks[cur.id]=ta.value;cur=eps.find(e=>e.id===id);v.src=cur.video;v.playbackRate=+$('speed').value;ta.value=marks[id];pending=null;
$('title').textContent=`${cur.id}: ${cur.task}  (${cur.frames} frames, ${cur.seconds.toFixed(1)} s; interventions ${cur.interventions.map(s=>s.map(x=>x.toFixed(1)).join('-')).join(', ')||'none'})`;list();draw()}
function sec(t){const p=t.trim().split(':');return parseFloat(p[p.length-1])+(p.length>1?60*parseInt(p[p.length-2]):0)}
function parse(text,D){return text.split(',').map(s=>s.trim()).filter(Boolean).map(s=>{const i=s.lastIndexOf('-');const a=sec(s.slice(0,i)),b=s.slice(i+1).trim()?sec(s.slice(i+1)):D;return [a,Math.min(b,D)]}).filter(([a,b])=>b>a).sort((x,y)=>x[0]-y[0]).reduce((m,[a,b])=>{if(m.length&&a<=m[m.length-1][1])m[m.length-1][1]=Math.max(m[m.length-1][1],b);else m.push([a,b]);return m},[])}
function fmt(s){return `${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,'0')}`}
function draw(){if(!cur)return;const W=tl.width=tl.clientWidth,H=tl.height,c=tl.getContext('2d'),D=cur.seconds;c.fillStyle='#2a6';c.fillRect(0,0,W,H);
c.fillStyle='#c33';cur.interventions.forEach(([a,b])=>c.fillRect(a/D*W,0,(b-a)/D*W,H/2));let rm=[];try{rm=parse(ta.value,D)}catch(e){}
c.fillStyle='#777';rm.forEach(([a,b])=>c.fillRect(a/D*W,H/2,(b-a)/D*W,H/2));if(pending!==null){c.fillStyle='#fc0';c.fillRect(pending/D*W-1,0,3,H)}
c.fillStyle='#fff';c.fillRect(v.currentTime/D*W-1,0,3,H);const kept=D-rm.reduce((s,[a,b])=>s+(b-a),0);
$('summary').textContent=`${rm.length} remove span(s); kept ${kept.toFixed(1)} s of ${D.toFixed(1)} s`;$('t').textContent=fmt(v.currentTime)}
function add(a,b){ta.value=(ta.value.trim()?ta.value.trim().replace(/,\s*$/,'')+', ':'')+a.toFixed(1)+'-'+(b===null?'':b.toFixed(1));marks[cur.id]=ta.value;list();draw()}
$('play').onclick=()=>v.paused?v.play():v.pause();v.onplay=()=>$('play').textContent='pause';v.onpause=()=>$('play').textContent='play';
$('speed').onchange=()=>v.playbackRate=+$('speed').value;v.ontimeupdate=draw;ta.oninput=()=>{marks[cur.id]=ta.value;draw()};
$('mstart').onclick=()=>{pending=v.currentTime;draw()};$('mend').onclick=()=>{if(pending!==null&&v.currentTime>pending){add(pending,v.currentTime);pending=null}};
$('mtoend').onclick=()=>add(v.currentTime,null);$('clear').onclick=()=>{ta.value='';pending=null;marks[cur.id]='';list();draw()};
tl.onclick=e=>{v.currentTime=e.offsetX/tl.clientWidth*cur.seconds;draw()};
document.onkeydown=e=>{if(['INPUT','TEXTAREA'].includes(e.target.tagName))return;const s=e.shiftKey?10:1;
if(e.key===' '){e.preventDefault();$('play').click()}else if(e.key==='ArrowLeft')v.currentTime=Math.max(0,v.currentTime-s);else if(e.key==='ArrowRight')v.currentTime=Math.min(cur.seconds,v.currentTime+s);
else if(e.key==='[')$('mstart').click();else if(e.key===']')$('mend').click();draw()};
$('save').onclick=async()=>{marks[cur.id]=ta.value;const body={out:$('out').value,why:$('why').value,episodes:eps.filter(e=>include[e.id]).map(e=>({id:e.id,remove:marks[e.id]}))};
$('save').disabled=true;$('state').textContent='building...';const r=await (await fetch('/api/save',{method:'POST',body:JSON.stringify(body)})).json();
if(r.error){$('state').textContent='error: '+r.error;$('save').disabled=false;return}
const poll=async()=>{const s=await (await fetch('/api/status')).json();$('log').textContent=s.log;$('log').scrollTop=1e9;$('state').textContent=s.state;if(s.state==='building')setTimeout(poll,2000);else $('save').disabled=false};poll()};
</script></body></html>"""


def serve(rows, work, out, port):
    status = {"state": "idle", "log": ""}
    by_id = {r["id"]: r for r in rows}

    def run_build(plan, target):
        buf = io.StringIO()
        status.update(state="building", log="")
        try:
            with redirect_stdout(buf):
                build(work, target, plan)
            status["state"] = "done"
        except Exception:
            buf.write(traceback.format_exc())
            status["state"] = "error"
        finally:
            status["log"] = buf.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/episodes":
                eps = [
                    {k: v for k, v in r.items() if k != "_e"} | {"video": f"/video/{r['video'].name}"}
                    for r in rows
                ]
                _json_response(self, {"episodes": eps, "out": out})
            elif self.path == "/api/status":
                if status["state"] == "building":
                    status["log"] = status["log"] or "running build_root..."
                _json_response(self, status)
            elif self.path.startswith("/video/"):
                _send_file(self, work / "videos" / Path(self.path).name)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/api/save":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if status["state"] == "building":
                _json_response(self, {"error": "a build is running"})
                return
            try:
                plan = [plan_entry(by_id[e["id"]], e["remove"], req["why"]) for e in req["episodes"]]
                assert plan, "no episode ticked"
                assert not (WORKSPACE / req["out"]).exists(), f"{req['out']} exists; pick another path"
            except Exception as exc:
                _json_response(self, {"error": str(exc)})
                return
            threading.Thread(target=run_build, args=(plan, req["out"]), daemon=True).start()
            _json_response(self, {"ok": True})

    server = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError:
            continue
    assert server is not None, "no free port"
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"review at {url}  (Ctrl-C to stop)", flush=True)
    threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    with suppress(KeyboardInterrupt):
        server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="outputs/train/<date>/<run> directories")
    ap.add_argument("--out", help="root to build; default outputs/rebot_rollouts_<date>-cut-v<N>")
    ap.add_argument("--work", help="default migration/rollout_review_<date>")
    ap.add_argument("--port", type=int, default=7871)
    args = ap.parse_args()
    date = Path(args.runs[0]).resolve().parent.name
    work = WORKSPACE / (args.work or f"migration/rollout_review_{date}")
    rows = inventory(args.runs, work)
    for r in rows:
        print(
            f"{r['id']}: {r['frames']} frames, {r['seconds']:.1f} s, interventions "
            + (", ".join(f"{a:.1f}-{b:.1f}" for a, b in r["interventions"]) or "none"),
            flush=True,
        )
    render(rows)
    serve(rows, work, args.out or default_out(date), args.port)


if __name__ == "__main__":
    main()

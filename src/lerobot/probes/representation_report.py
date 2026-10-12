"""A portable, linked Plotly inspector for cached representation trajectories.

``trajectories.html`` holds only the sampled rows and a panel index. Each
group/layer/prompt/geometry panel (temporal metrics, ranked matches and the
per-episode cosine matrices) is written to ``panels/<key>.js`` and loaded on
demand through a script tag, which works both from the probe viewer and from a
plain ``file://`` open, as long as the directory moves as a whole. The packed
every-frame-against-every-frame cosine matrix behind the aligned cross-episode
grid is a second sidecar, ``panels/<key>.cross.js``, loaded only when that grid
is used. Camera previews are referenced from ``cache/thumbs`` rather than embedded.

Distance and norm-ratio views are derived in the browser from cosines and the
stored per-frame norms, so one stored matrix serves every measure. The aligned
grid is also computed in the browser: for the selected frame and the match it is
aligned with, it reads the cosine between the frames nearest to each pair of time
offsets out of the packed matrix, so changing the offset step or span costs nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

from plotly.offline import get_plotlyjs

ROW_KEYS = ("row", "sample", "text", "robot", "episode", "episode_label", "frame", "time_s",
            "task", "subtask", "holdout")
PANEL_INDEX_KEYS = ("key", "group", "layer", "text", "geometry", "centering")
CAMERAS = ("external_0", "wrist_0")


def _json(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def write_report(payload, output_dir):
    root = Path(output_dir)
    (root / "panels").mkdir(parents=True, exist_ok=True)
    for stale in (root / "panels").glob("*.js"):
        stale.unlink()
    index = []
    for panel in payload["panels"]:
        data = {k: v for k, v in panel.items() if k not in PANEL_INDEX_KEYS and k != "cross"}
        (root / "panels" / f"{panel['key']}.js").write_text(
            f"window.__trajectoryPanel({_json(panel['key'])},{_json(data)});")
        entry = {**{k: panel[k] for k in PANEL_INDEX_KEYS}, "file": f"panels/{panel['key']}.js",
                 "frames": len(panel["ids"]), "cross_file": None}
        if panel.get("cross") is not None:
            entry["cross_file"] = f"panels/{panel['key']}.cross.js"
            (root / entry["cross_file"]).write_text(
                f"window.__trajectoryCross({_json(panel['key'])},{_json(panel['cross'])});")
        index.append(entry)
    thumbs = {}
    for sample in sorted({r["sample"] for r in payload["rows"]}):
        cameras = [c for c in CAMERAS if (root / "cache" / "thumbs" / f"{sample:04d}.{c}.jpg").exists()]
        if cameras:
            thumbs[str(sample)] = cameras
    light = {"rows": [{k: r.get(k) for k in ROW_KEYS} for r in payload["rows"]], "panels": index,
             "window_s": payload["window_s"], "match_top_k": payload["match_top_k"],
             "stride_s": payload.get("stride_s"), "view": payload["view"],
             "checkpoint": payload["checkpoint"], "thumbs": thumbs, "thumb_dir": "cache/thumbs"}
    html = _HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", _json(light))
    (root / "trajectories.html").write_text(html)


_HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Representation trajectories</title>
<style>
*{box-sizing:border-box}body{margin:0;padding:20px;background:#f6f7f9;color:#202631;font:14px system-ui,sans-serif}
h1{font-size:23px;margin:0 0 8px}h2{font-size:17px;margin:14px 0 8px}p{line-height:1.5;margin:8px 0}.muted{color:#5e6878}.caption{font-size:12px;color:#5e6878;line-height:1.45;margin:6px 0 0}
.controls{display:flex;flex-wrap:wrap;gap:12px;background:white;padding:12px;border-radius:8px;margin:14px 0}
label{display:flex;flex-direction:column;gap:5px;font-size:12px;color:#505b6e}select,button{font:inherit;border:1px solid #b8c0ce;border-radius:4px;padding:7px;background:white;color:#202631;max-width:420px}
button{cursor:pointer}button:hover{background:#e9f1ff}button.small{padding:3px 7px;font-size:12px}.grid{display:grid;grid-template-columns:minmax(400px,1fr) minmax(400px,1fr);gap:14px;margin-top:14px}.box{background:white;border:1px solid #e0e4eb;border-radius:8px;padding:12px;min-width:0}
#matrix,#profile{height:520px}#timeline{height:420px}#cross{height:620px}#crossLines{height:330px}.cards{display:flex;flex-wrap:wrap;gap:10px}.card{border:1px solid #d8dfeb;border-radius:6px;padding:10px;flex:1;min-width:270px;max-width:580px}.cameras{display:flex;gap:4px;margin:7px 0}.cameras img{width:49%;object-fit:contain;max-height:190px;background:#edf0f5}.cameras span{display:block;font-size:11px}.details{font-size:12px;overflow-wrap:anywhere}.tag{font-size:11px;background:#edf0f5;border-radius:3px;padding:2px 5px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:6px 8px;border-bottom:1px solid #e1e6ee;text-align:left;vertical-align:top}tr.kind td{background:#f3f5f8;font-weight:600}tr.aligned td{background:#fff3e8}.scroll{overflow:auto}.note{padding:9px 12px;background:#edf2f8;border-radius:5px}#status,#episodeStats,#crossInfo{white-space:pre-wrap}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}.row input[type=range]{flex:1;min-width:160px}.row label.inline{flex-direction:row;align-items:center;gap:8px;flex:1}
.crossgrid{display:grid;grid-template-columns:minmax(400px,3fr) minmax(300px,2fr);gap:14px}
@media(max-width:950px){.grid,.crossgrid{grid-template-columns:1fr}body{padding:10px}#matrix,#profile{height:420px}}
</style>
<script>__PLOTLY__</script>
<h1>Representation trajectories</h1>
<p class="muted">Follow an episode frame by frame. Every pairwise view compares two frames' pooled vectors in the original dimensions of the chosen geometry, under the chosen measure: cosine (direction only), L2 distance (direction and magnitude), or log2 norm ratio (magnitude only). Next to each value sits the level it reaches against <em>other</em> episodes, so a high similarity can be read as trivial or not.</p>
<p class="note">Raw cosines between model states sit near 1 because all states share a large common direction. “Subtract robot mean” removes the per-robot part; “subtract episode mean” also removes the scene so only motion within the episode remains. The background line is the frame's mean value against other episodes of the same robot: a similarity is only informative where it clears that line. Dataset replays at the first flow step with fixed noise.</p>
<div class="controls">
<label>Token group<select id="group"></select></label><label>Layer<select id="layer"></select></label>
<label>Prompt<select id="text"><option value="real">Real task / subtask</option><option value="neutral">Constant task / subtask</option></select></label>
<label>Geometry<select id="geometry"><option value="raw">Raw vectors</option><option value="robot_centered" selected>Subtract robot mean</option><option value="episode_centered">Subtract episode mean</option></select></label>
<label>Measure<select id="measure"><option value="cosine">Cosine</option><option value="distance">L2 distance</option><option value="lognorm">log2 norm ratio (column / row)</option></select></label>
<label>Show robot<select id="robot"></select></label><label>Show task<select id="task"></select></label>
<label>Color range<select id="crange"><option value="auto">Fit to this episode</option><option value="panel">Same for all episodes</option></select></label>
</div>
<p id="status" class="muted"></p>
<section class="box"><h2>Episode</h2>
<div class="row"><label>Episode<select id="episode"></select></label><p id="episodeStats" class="muted" style="margin:0"></p></div>
<div class="row"><button id="play">▶ Play</button><button id="previous">←</button><input type="range" id="frame" min="0" max="0" value="0"><button id="next">→</button><span id="frameStats" class="muted"></span></div>
<div id="anchor" class="cards"></div></section>
<div class="grid"><section class="box"><h2>Within-episode matrix</h2><div id="matrix"></div>
<p class="caption" id="matrixCaption"></p></section>
<section class="box"><h2>Selected frame against its episode</h2>
<div class="row"><label class="inline"><span>Local radius: ±<span id="radiusLabel"></span> s</span><input type="range" id="radius" min="0.25" max="30" step="0.25"></label><span id="radiusStats" class="muted"></span></div><div id="profile"></div>
<p class="caption">Black: the selected frame against every other frame of its episode, by time; the shaded band is the local radius and the vertical line the selected frame. Dashed grey: this frame's mean value against all frames of other episodes of the same robot (background). Dotted: its single best cross-episode match (found by cosine), same task and different task, under the current measure. The episode's structure is informative only where the black line is clearly on the similar side of the dashed line.</p></section></div>
<section class="box"><h2>Nearest frames in other episodes</h2><p class="muted">Matches are found by cosine on the selected frame. Each category lists its top episodes (best first), one frame per episode, searched over every captured episode in this group, layer and prompt, including episodes hidden by display filters. “Same task” uses exact recorded task text; different wording may describe the same behavior. Target-window cosine compares the selected frame to frames within ±<span id="windowLabel"></span> s of the match, inside the match's episode and task; a support count of 1 is just the matched frame. <b>Align grid</b> centers the grid below on that match; the highlighted row is the one currently shown.</p>
<div class="scroll"><table><thead><tr><th>Candidate set</th><th>Rank</th><th>Cosine</th><th>Distance</th><th>Raw norm ratio<br>target / anchor</th><th>Target-window cosine</th><th>Support</th><th>Episode / time</th><th></th></tr></thead><tbody id="matches"></tbody></table></div>
<div id="matchCards" class="cards" style="margin-top:12px"></div></section>
<section class="box"><h2>Aligned cross-episode grid</h2>
<div class="row"><label>Align with<select id="crossWith"><option value="any">Best match: any category</option><option value="same_robot_same_task">Best match: same robot · same task</option><option value="same_robot_different_task">Best match: same robot · different task</option><option value="different_robot_same_task">Best match: other robot · same task</option><option value="different_robot_different_task">Best match: other robot · different task</option><option value="episode">Chosen episode (its best frame by current measure)</option></select></label>
<label>Rank<select id="rank"></select></label><label>Episode<select id="crossEpisode"></select></label>
<label>Offset step Δ<select id="delta"><option value="1">1 s</option><option value="2" selected>2 s</option><option value="4">4 s</option><option value="8">8 s</option></select></label>
<label>Span<select id="span"><option value="5" selected>±5 steps (11 × 11)</option><option value="7">±7 steps (15 × 15)</option></select></label></div>
<p id="crossInfo" class="muted"></p>
<div class="crossgrid"><div><div id="cross"></div></div><div><div id="crossLines"></div><div id="crossCards" class="cards"></div></div></div>
<p class="caption">The selected frame and the frame it is aligned with sit at the center cell (0, 0). Rows are the selected episode at offsets of −Δ·span … +Δ·span seconds from the selected frame; columns are the other episode at the same offsets from the aligned frame. Each cell is the current measure between the two sampled frames nearest to those times (hover shows the frames actually used; cells past an episode's ends are blank; with Δ below the sampling stride, neighbouring cells reuse the same frame). <b>Reading it:</b> a bright diagonal from top-left to bottom-right means the two episodes pass through the same sequence of states at the same pace; a bright diagonal with another slope means the same sequence at a different speed; a bright row or column means one episode holds still in a state the other passes through; a single bright center cell is a momentary coincidence. The side plot pulls three lines out of the grid against the selected frame's background level: the diagonal (both episodes move together), the center row (selected frame held, the other episode moves) and the center column (aligned frame held, the selected episode moves). Click a cell to select its row frame.</p></section>
<section class="box"><h2>Along the episode</h2><div id="timeline"></div>
<p class="caption"><b>What is plotted.</b> One point per sampled frame of the selected episode, at its time. Dashed grey, <em>background</em>: that frame's mean value against every frame of every other episode of the same robot, i.e. how similar the frame is to anything else this robot was recorded doing; dotted light grey, the same against other robots (absent when only one robot was captured). Colored lines, <em>best match</em> per category: the single most similar frame found anywhere in other episodes of that category, so each point may come from a different episode and a different moment. <b>How to read it.</b> The background is a mean and the match lines are maxima over hundreds of candidates, so the match lines always sit above the background: their height alone says nothing. What matters is the gap between a match line and the background, and how that gap changes along the episode: a stretch where a match line rises well clear of the background is where this episode has a specific counterpart elsewhere (select a frame there and read the aligned grid to see whether the counterpart persists in time); a stretch where every match line hugs the background is where the representation is generic. A match line for a category that is missing for some frames leaves a gap rather than a zero. Click a point to select that frame.</p></section>
<p class="muted">Local values exclude the selected frame and stay inside its episode; the radius slider recomputes them from the stored matrix. Gaps mean no valid temporal neighbors or zero-length vectors; no zeros are substituted. Raw norm is always measured before centering; distance and norm ratio use the norms of the chosen geometry. Keyboard: ← → step frames, space plays.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), uniq=a=>[...new Set(a)], esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(n,d=3)=>n==null?'—':Number(n).toFixed(d), mean=a=>{const v=a.filter(x=>x!=null);return v.length?v.reduce((s,x)=>s+x,0)/v.length:null;};
const median=a=>{const v=a.filter(x=>x!=null).sort((p,q)=>p-q);return v.length?v[Math.floor(v.length/2)]:null;};
const kinds=['same_robot_same_task','same_robot_different_task','different_robot_same_task','different_robot_different_task'];
const labels=['Same robot · same task','Same robot · different task','Other robot · same task','Other robot · different task'];
const colors=['#3366cc','#dd6633','#119988','#9933aa','#cc4477','#667733','#0099c6','#b47700'];
const MEASURES={cosine:{label:'Cosine',scale:'Viridis',reverse:false,bg:'background_same_robot',bgOther:'background_other_robot'},distance:{label:'L2 distance',scale:'Viridis',reverse:true,bg:'background_distance_same_robot',bgOther:'background_distance_other_robot'},lognorm:{label:'log2 norm ratio (column / row)',scale:'RdBu',reverse:false,bg:'background_lognorm_same_robot',bgOther:'background_lognorm_other_robot'}};
const cache={}, crossCache={}, pending={};
window.__trajectoryPanel=(key,data)=>{cache[key]=data;if(pending[key]){pending[key](data);delete pending[key];}};
window.__trajectoryCross=(key,text)=>{const bytes=Uint8Array.from(atob(text),c=>c.charCodeAt(0));crossCache[key]=new Uint16Array(bytes.buffer);if(pending['cross:'+key]){pending['cross:'+key](crossCache[key]);delete pending['cross:'+key];}};
function loadScript(id,file,store){return new Promise((resolve,reject)=>{if(store[id.replace(/^cross:/,'')])return resolve(store[id.replace(/^cross:/,'')]);pending[id]=resolve;const s=document.createElement('script');s.src=file;s.onerror=()=>{delete pending[id];reject(new Error(`Could not load ${file}; keep panels/ next to this page`));};document.head.appendChild(s);});}
let panel=null, localRows=[], selected=0, episode=null, epIds=[], epTimes=[], radius=D.window_s, playing=null, loadToken=0, cross=null, alignedTarget=null;
function options(id,values,all=false){const old=$(id).value;$(id).replaceChildren();if(all)$(id).add(new Option('All',''));values.forEach(v=>$(id).add(new Option(v,v)));if(values.includes(old)||all&&old==='')$(id).value=old;}
function visible(i){const r=localRows[i];return (!$('robot').value||r.robot===$('robot').value)&&(!$('task').value||r.task===$('task').value);}
// Ranked matches: panel.matches[kind][key][i] is the list over ranks (best first) for frame i.
function match(i,kind,rank=0){const m=panel.matches[kind];const t=m.target[i];if(!t||rank>=t.length)return null;return Object.fromEntries(Object.keys(m).map(k=>[k,m[k][i][rank]]));}
function bestAny(i){let best=null;kinds.forEach(kind=>{const m=match(i,kind,0);if(m&&(best==null||m.cosine>best.m.cosine))best={m,kind};});return best;}
function temporal(i){return Object.fromEntries(Object.keys(panel.temporal).map(k=>[k,panel.temporal[k][i]]));}
function measure(){return $('measure').value;}
function gn(i){return panel.temporal.geometry_norm[i];}
// One stored cosine serves every measure: distance and norm ratio follow from the two norms.
function val(c,i,j){if(c==null)return null;const m=measure();if(m==='cosine')return c;const a=gn(i),b=gn(j);if(!(a>0&&b>0))return null;if(m==='distance')return Math.sqrt(Math.max(0,a*a+b*b-2*a*b*c));return Math.log2(b/a);}
function matchValue(i,m){if(!m)return null;const k=measure();return k==='cosine'?m.cosine:k==='distance'?m.distance:gn(i)>0&&gn(m.target)>0?Math.log2(gn(m.target)/gn(i)):null;}
function episodeMatrix(ep){const e=panel.episodes[ep];return e.cosine.map((row,a)=>row.map((c,b)=>val(c,e.ids[a],e.ids[b])));}
function localValue(pos,r,matrix){const t=epTimes[pos],row=matrix[pos];let s=0,n=0;for(let k=0;k<epTimes.length;k++){if(k!==pos&&row[k]!=null&&Math.abs(epTimes[k]-t)<=r){s+=row[k];n++;}}return n?[s/n,n]:[null,0];}
function thumb(sample,camera){const cams=D.thumbs[String(sample)]||[];return cams.includes(camera)?`${D.thumb_dir}/${String(sample).padStart(4,'0')}.${camera}.jpg`:null;}
function colorRange(values){let lo=Infinity,hi=-Infinity;for(const v of values){if(v==null)continue;if(v<lo)lo=v;if(v>hi)hi=v;}if(!(lo<hi))return [lo-1e-3,hi+1e-3];if(measure()==='lognorm'){const m=Math.max(Math.abs(lo),Math.abs(hi));return [-m,m];}return [lo,hi];}
function zRange(matrix){if($('crange').value==='auto')return colorRange(matrix.flat());return colorRange(Object.keys(panel.episodes).flatMap(ep=>episodeMatrix(ep).flat()));}
async function updatePanel(){
 const sample=localRows[selected]?.sample, token=++loadToken;
 let candidates=D.panels.filter(p=>p.group===$('group').value);
 options('layer',uniq(candidates.map(p=>String(p.layer))));
 candidates=candidates.filter(p=>String(p.layer)===$('layer').value);
 const allowed=uniq(candidates.map(p=>p.text));for(const o of $('text').options)o.disabled=!allowed.includes(o.value);
 if(!allowed.includes($('text').value))$('text').value=allowed[0];
 const entry=candidates.find(p=>p.text===$('text').value&&p.geometry===$('geometry').value);if(!entry)return;
 $('status').textContent=`Loading ${entry.key}…`;
 let data;try{data=await loadScript(entry.key,entry.file,cache);}catch(e){$('status').textContent=e.message;return;}
 if(token!==loadToken)return;
 panel={...entry,...data};cross=crossCache[panel.key]||null;localRows=panel.ids.map(i=>D.rows[i]);selected=Math.max(0,localRows.findIndex(r=>r.sample===sample));
 options('robot',uniq(localRows.map(r=>r.robot)).sort(),true);options('task',uniq(localRows.map(r=>r.task)).sort(),true);
 const center={raw:'Cosine uses the uncentered captured vectors.',robot_centered:'Robot means: '+Object.entries(panel.centering).map(([r,p])=>`${r}: ${p.fit}, n=${p.rows}`).join('; '),episode_centered:'Each episode centered on its own sampled mean (all of its frames).'}[panel.geometry];
 const bg=mean(panel.temporal.background_same_robot), bgOther=mean(panel.temporal.background_other_robot), gap=median(panel.temporal.previous_dt_s);
 $('status').textContent=`Capture: ${D.view} · ${panel.group} · L${panel.layer} · ${localRows.length} frames in ${uniq(localRows.map(r=>r.episode)).length} episodes, ${D.stride_s!=null?`sampling stride ${D.stride_s} s, `:''}median gap ${fmt(gap,2)} s.\n${center}\nBackground cosine to other episodes: same robot ${fmt(bg)}, other robots ${fmt(bgOther)} (mean over frames). Checkpoint: ${D.checkpoint}`;
 updateFilter();
}
function updateFilter(){
 const eps=uniq(localRows.filter((r,i)=>visible(i)).map(r=>r.episode));
 const previous=localRows[selected]?.episode;
 $('episode').replaceChildren();eps.forEach(ep=>{const r=localRows.find(r=>r.episode===ep);$('episode').add(new Option(`${r.robot} · ${r.episode_label} · ${r.holdout?'held out':'train'}`,ep));});
 if(eps.includes(previous))$('episode').value=previous;
 if(!eps.length){for(const id of ['matrix','profile','timeline','cross','crossLines'])Plotly.purge(id);$('anchor').replaceChildren();$('matches').replaceChildren();$('matchCards').replaceChildren();$('crossCards').replaceChildren();$('frameStats').textContent='';$('crossInfo').textContent='';$('episodeStats').textContent='No frames match these filters';return;}
 if(!visible(selected))selected=localRows.findIndex((r,i)=>visible(i)&&r.episode===$('episode').value);
 setEpisode($('episode').value);
}
function setEpisode(ep){
 episode=ep;const e=panel.episodes[ep];epIds=e.ids;epTimes=e.times;
 if(localRows[selected]?.episode!==ep)selected=epIds[0];
 $('frame').max=epIds.length-1;
 const gaps=epIds.map(i=>panel.temporal.previous_dt_s[i]), r=localRows[epIds[0]];
 $('episodeStats').textContent=`${epIds.length} frames over ${fmt(epTimes[epTimes.length-1]-epTimes[0],1)} s, median gap ${fmt(median(gaps),2)} s · ${esc(r.task)}`;
 const others=Object.keys(panel.episodes).filter(x=>x!==ep).sort();const old=$('crossEpisode').value;$('crossEpisode').replaceChildren();others.forEach(x=>{const rr=localRows[panel.episodes[x].ids[0]];$('crossEpisode').add(new Option(`${rr.robot} · ${rr.episode_label} · ${rr.task}`,x));});if(others.includes(old))$('crossEpisode').value=old;
 renderMatrix();renderTimeline();renderFrame();
}
function heat(z,x,y,xtitle,ytitle,zrange,extraShapes,hover){
 const M=MEASURES[measure()];
 return [[{type:'heatmap',x,y,z,colorscale:M.scale,reversescale:M.reverse,zmin:zrange[0],zmax:zrange[1],hoverongaps:false,...(hover||{hovertemplate:'row %{y:.2f} s · col %{x:.2f} s<br>%{z:.4f}<extra></extra>'}),colorbar:{title:{text:M.label.split(' (')[0]},len:.9}}],
  {margin:{l:60,r:10,t:10,b:55},xaxis:{title:{text:xtitle}},yaxis:{title:{text:ytitle},scaleanchor:'x',autorange:'reversed'},shapes:extraShapes||[]},{responsive:true,displaylogo:false}];
}
function renderMatrix(){
 const matrix=episodeMatrix(episode);
 Plotly.react('matrix',...heat(matrix,epTimes,epTimes,'Episode time (s)','Episode time (s)',zRange(matrix)));
 $('matrixCaption').textContent=`Rows and columns are the ${epIds.length} sampled frames of this episode in time order (median gap ${fmt(median(epIds.map(i=>panel.temporal.previous_dt_s[i])),2)} s); each cell is the ${MEASURES[measure()].label.toLowerCase()} between the two frames' vectors. Blocks along the diagonal are stretches where the representation holds still; patches far from the diagonal are returns to an earlier state. The color range ${$('crange').value==='auto'?'is fitted to this episode, so structure shows even when all values are close':'is shared by all episodes of this panel'}.`;
 $('matrix').removeAllListeners?.('plotly_click');$('matrix').on('plotly_click',event=>{const p=event.points[0];const idx=p?.pointIndex??p?.pointNumber;if(Array.isArray(idx))pick(epIds[idx[0]]);});
}
function renderProfile(){
 const pos=epIds.indexOf(selected), t=epTimes[pos], matrix=episodeMatrix(episode), row=matrix[pos], m=temporal(selected), M=MEASURES[measure()];
 const span=[epTimes[0],epTimes[epTimes.length-1]], traces=[{type:'scatter',mode:'lines+markers',name:`${M.label} (selected, frame)`,x:epTimes,y:row,customdata:epIds,line:{color:'#202631'},marker:{size:4},connectgaps:false,hovertemplate:'%{x:.2f} s · %{y:.4f}<extra></extra>'}];
 const ref=(name,y,color,dash)=>{if(y!=null)traces.push({type:'scatter',mode:'lines',name,x:span,y:[y,y],line:{color,dash,width:1.5},hoverinfo:'name+y'});};
 ref('Background: other episodes, same robot',m[M.bg],'#949daa','dash');
 ref('Best match: same robot · same task',matchValue(selected,match(selected,kinds[0])),colors[0],'dot');
 ref('Best match: same robot · different task',matchValue(selected,match(selected,kinds[1])),colors[1],'dot');
 const [lc,n]=localValue(pos,radius,matrix), prev=pos?val(m.previous_cosine,epIds[pos-1],selected):null;
 $('radiusStats').textContent=`local ${M.label.toLowerCase()} ${fmt(lc)} over ${n} neighbors · previous frame ${fmt(prev)} (${fmt(m.previous_dt_s,2)} s)`;
 Plotly.react('profile',traces,{margin:{l:55,r:10,t:10,b:55},xaxis:{title:{text:'Episode time (s)'}},yaxis:{title:{text:M.label}},legend:{orientation:'h',font:{size:10},y:-.18},
  shapes:[{type:'rect',xref:'x',yref:'paper',x0:t-radius,x1:t+radius,y0:0,y1:1,fillcolor:'#3366cc',opacity:.08,line:{width:0}},{type:'line',xref:'x',yref:'paper',x0:t,x1:t,y0:0,y1:1,line:{color:'#777',width:1}}]},{responsive:true,displaylogo:false});
 $('profile').removeAllListeners?.('plotly_click');$('profile').on('plotly_click',event=>{const i=event.points[0]?.customdata;if(Number.isInteger(i))pick(i);});
}
function renderTimeline(){
 const ids=epIds, traces=[], t=epTimes, T=panel.temporal, M=MEASURES[measure()];
 const add=(name,values,color,dash,width)=>{if(values.every(v=>v==null))return;traces.push({type:'scatter',mode:'lines+markers',name,x:t,y:values,customdata:ids,line:{color,dash,width:width||2},marker:{size:4},connectgaps:false,hovertemplate:'%{x:.2f} s · %{y:.4f}<extra>%{fullData.name}</extra>'});};
 add('Background: other episodes, same robot (mean)',ids.map(i=>T[M.bg][i]),'#6b7482','dash');
 add('Background: other robots (mean)',ids.map(i=>T[M.bgOther][i]),'#b8c0ce','dot');
 kinds.forEach((kind,k)=>add(`Best match: ${labels[k].toLowerCase()} (max)`,ids.map(i=>matchValue(i,match(i,kind))),colors[k],k%2?'dash':'solid'));
 Plotly.react('timeline',traces,{margin:{l:58,r:15,t:12,b:85},uirevision:panel.key+episode,xaxis:{title:{text:'Episode time (seconds)'}},yaxis:{title:{text:M.label.split(' (')[0]}},legend:{orientation:'h',font:{size:10},y:-.2},shapes:[cursor()]},{responsive:true,displaylogo:false});
 $('timeline').removeAllListeners?.('plotly_click');$('timeline').on('plotly_click',event=>{const i=event.points[0]?.customdata;if(Number.isInteger(i))pick(i);});
}
function cursor(){return {type:'line',xref:'x',yref:'paper',x0:localRows[selected].time_s,x1:localRows[selected].time_s,y0:0,y1:1,line:{color:'#777',width:1}};}
function crossCos(i,j){const q=cross[i*localRows.length+j];return q?(q-1)/65534*2-1:null;}
// Position of the sampled frame nearest to time t, or null when t lies more than half a step past either end.
function nearestPos(times,t,delta){if(t<times[0]-delta/2||t>times[times.length-1]+delta/2)return null;let lo=0,hi=times.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(times[mid]<=t)lo=mid;else hi=mid;}return Math.abs(times[hi]-t)<Math.abs(times[lo]-t)?hi:lo;}
function clearCross(message){$('crossInfo').textContent=message;Plotly.purge('cross');Plotly.purge('crossLines');$('crossCards').replaceChildren();alignedTarget=null;renderMatchRows();}
async function renderCross(){
 const token=loadToken;
 if(!panel.cross_file)return clearCross('Cross-episode matrices were not written (repr_cross_matrices=false); re-run in plot mode with it enabled.');
 if(!cross){$('crossInfo').textContent='Loading the every-frame matrix for this panel…';try{cross=await loadScript('cross:'+panel.key,panel.cross_file,crossCache);}catch(e){$('crossInfo').textContent=e.message;return;}if(token!==loadToken)return;}
 const mode=$('crossWith').value, delta=Number($('delta').value), K=Number($('span').value), rank=Number($('rank').value)-1, pos=epIds.indexOf(selected), M=MEASURES[measure()];
 let target, how;
 if(mode==='episode'){const other=$('crossEpisode').value;if(!other)return clearCross('No other episode in this panel.');
  const e=panel.episodes[other];let best=null;e.ids.forEach(j=>{const v=val(crossCos(selected,j),selected,j);if(v==null)return;const better=measure()==='distance'?v<best?.v:measure()==='lognorm'?Math.abs(v)<Math.abs(best?.v):v>best?.v;if(best==null||better)best={v,j};});
  if(!best)return clearCross('No valid frames to align with in that episode.');target=best.j;how=`best frame of the chosen episode by ${M.label.toLowerCase()}`;
 }else if(mode==='any'){const b=bestAny(selected);if(!b)return clearCross('No eligible frames in any other episode.');target=b.m.target;how=`best cosine match over all categories (${labels[kinds.indexOf(b.kind)].toLowerCase()})`;
 }else{const m=match(selected,mode,rank);if(!m)return clearCross(`No match of rank ${rank+1} for “${labels[kinds.indexOf(mode)]}” (${match(selected,mode,0)?'fewer eligible episodes':'no eligible frames'}).`);target=m.target;how=`cosine match of rank ${rank+1}, ${labels[kinds.indexOf(mode)].toLowerCase()}`;}
 alignedTarget=target;
 const rA=localRows[selected], rB=localRows[target], A=panel.episodes[episode], B=panel.episodes[rB.episode], posB=B.ids.indexOf(target);
 const tA=A.times[pos], tB=B.times[posB], offs=[];for(let k=-K;k<=K;k++)offs.push(k*delta);
 const cellA=offs.map(o=>nearestPos(A.times,tA+o,delta)), cellB=offs.map(o=>nearestPos(B.times,tB+o,delta));
 const z=cellA.map(a=>cellB.map(b=>a==null||b==null?null:val(crossCos(A.ids[a],B.ids[b]),A.ids[a],B.ids[b])));
 const text=cellA.map((a,i)=>cellB.map((b,j)=>a==null||b==null?'':`selected ${offs[i]>=0?'+':''}${offs[i]} s → frame at ${fmt(A.times[a],2)} s · other ${offs[j]>=0?'+':''}${offs[j]} s → frame at ${fmt(B.times[b],2)} s`));
 const h=delta/2, shapes=[{type:'rect',xref:'x',yref:'y',x0:-h,x1:h,y0:-h,y1:h,line:{color:'white',width:2}},
  {type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:0,y1:0,line:{color:'white',width:.7,dash:'dot'}},{type:'line',xref:'x',yref:'paper',y0:0,y1:1,x0:0,x1:0,line:{color:'white',width:.7,dash:'dot'}}];
 const range=$('crange').value==='auto'?colorRange(z.flat()):zRange(episodeMatrix(episode));
 Plotly.react('cross',...heat(z,offs,offs,`${rB.robot} · ${rB.episode_label}: seconds from the aligned frame`,`${rA.robot} · ${rA.episode_label}: seconds from the selected frame`,range,shapes,{text,hovertemplate:'%{text}<br>%{z:.4f}<extra></extra>'}));
 $('cross').removeAllListeners?.('plotly_click');$('cross').on('plotly_click',event=>{const p=event.points[0];const idx=p?.pointIndex??p?.pointNumber;if(Array.isArray(idx)&&cellA[idx[0]]!=null)pick(A.ids[cellA[idx[0]]]);});
 const line=(name,y,color,dash)=>({type:'scatter',mode:'lines+markers',name,x:offs,y,line:{color,dash,width:2},marker:{size:5},connectgaps:false,hovertemplate:'%{x} s · %{y:.4f}<extra>%{fullData.name}</extra>'});
 const bg=panel.temporal[M.bg][selected], lines=[line('Diagonal: both move together',offs.map((_,k)=>z[k][k]),'#202631'),line('Center row: selected frame held, other moves',z[K],colors[1],'dash'),line('Center column: aligned frame held, selected moves',z.map(r=>r[K]),colors[0],'dot')];
 if(bg!=null)lines.push({type:'scatter',mode:'lines',name:'Background of the selected frame',x:[offs[0],offs[offs.length-1]],y:[bg,bg],line:{color:'#949daa',dash:'dash',width:1.5},hoverinfo:'name+y'});
 Plotly.react('crossLines',lines,{margin:{l:55,r:10,t:10,b:50},xaxis:{title:{text:'Offset (s)'},zeroline:true,zerolinecolor:'#c8ced8'},yaxis:{title:{text:M.label.split(' (')[0]}},legend:{orientation:'h',font:{size:10},y:-.3}},{responsive:true,displaylogo:false});
 const aligned=val(crossCos(selected,target),selected,target), gapA=median(A.ids.map(i=>panel.temporal.previous_dt_s[i])), gapB=median(B.ids.map(i=>panel.temporal.previous_dt_s[i]));
 const coarse=Math.max(gapA??0,gapB??0)>delta*1.05?` Δ = ${delta} s is below the sampling gap (${fmt(gapA,1)} s / ${fmt(gapB,1)} s), so neighbouring cells reuse frames.`:'';
 $('crossInfo').textContent=`Rows: ${rA.robot} · ${rA.episode_label}, selected frame at ${fmt(rA.time_s,2)} s. Columns: ${rB.robot} · ${rB.episode_label}, aligned frame at ${fmt(rB.time_s,2)} s (${how}). Aligned pair ${M.label.toLowerCase()}: ${fmt(aligned,4)}; selected frame's background: ${fmt(bg,4)}.${coarse}\nTasks: “${rA.task}” vs “${rB.task}”.`;
 $('crossCards').innerHTML=frameCard(target,'Aligned frame in the other episode');
 document.querySelectorAll('#crossCards [data-pick]').forEach(el=>el.onclick=()=>pick(Number(el.dataset.pick),true));
 renderMatchRows();
}
function frameCard(i,title){
 const r=localRows[i];
 const cameras=['external_0','wrist_0'].map(cam=>{const src=thumb(r.sample,cam);return src?`<img src="${src}" alt="${cam}" loading="lazy">`:`<span>${cam}: unavailable</span>`;}).join('');
 return `<div class="card"><strong>${esc(title)}</strong><div class="details">${esc(r.robot)} · ${esc(r.episode_label)} <span class="tag">${r.holdout?'held out':'train'}</span><br>Frame ${r.frame} · ${fmt(r.time_s,2)} s</div><div class="cameras">${cameras}</div><div class="details"><b>Task:</b> ${esc(r.task)}<br><b>Step:</b> ${esc(r.subtask)}</div><button data-pick="${i}" style="margin-top:8px">Follow this frame</button></div>`;
}
function renderMatchRows(){
 $('matches').innerHTML=kinds.map((kind,k)=>{
  const first=match(selected,kind,0);
  if(!first)return `<tr class="kind"><td>${labels[k]}</td><td colspan="8" style="font-weight:normal">No eligible frames</td></tr>`;
  const rows=[`<tr class="kind"><td>${labels[k]}</td><td colspan="8" style="font-weight:normal" class="muted">${first.candidate_count} candidate frames in ${first.candidate_episodes} episodes; best frame of each of the top ${Math.min(D.match_top_k,first.candidate_episodes)} episodes below</td></tr>`];
  for(let rank=0;rank<D.match_top_k;rank++){const mm=match(selected,kind,rank);if(!mm)break;const r=localRows[mm.target];
   rows.push(`<tr class="${mm.target===alignedTarget?'aligned':''}"><td></td><td>${rank+1}</td><td>${fmt(mm.cosine)}</td><td>${fmt(mm.distance)}</td><td>${fmt(mm.raw_norm_ratio)}</td><td>${fmt(mm.target_window_cosine)}</td><td>${mm.target_window_count}</td><td>${esc(r.episode_label)} · ${fmt(r.time_s,2)} s<br><span class="muted">${esc(r.task)}</span></td><td><button class="small" data-align="${kind}" data-rank="${rank+1}">Align grid</button> <button class="small" data-pick="${mm.target}">Follow</button></td></tr>`);}
  return rows.join('');
 }).join('');
 document.querySelectorAll('#matches [data-pick]').forEach(el=>el.onclick=()=>pick(Number(el.dataset.pick),true));
 document.querySelectorAll('#matches [data-align]').forEach(el=>el.onclick=()=>{$('crossWith').value=el.dataset.align;$('rank').value=el.dataset.rank;renderCross();});
}
function renderFrame(){
 const pos=epIds.indexOf(selected), m=temporal(selected), t=epTimes[pos];
 $('frame').value=pos;
 $('frameStats').textContent=`frame ${pos+1}/${epIds.length} · ${fmt(t,2)} s · raw norm ${fmt(m.raw_norm)} · ${panel.geometry==='raw'?'':`centered norm ${fmt(m.geometry_norm)} · `}step distance ${fmt(m.step_l2)}`;
 $('anchor').innerHTML=frameCard(selected,'Selected frame');
 Plotly.relayout('matrix',{shapes:[{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:t,y1:t,line:{color:'white',width:1}},{type:'line',xref:'x',yref:'paper',x0:t,x1:t,y0:0,y1:1,line:{color:'white',width:1}}]});
 Plotly.relayout('timeline',{shapes:[cursor()]});
 renderProfile();renderMatchRows();renderCross();
 $('matchCards').innerHTML=kinds.map((kind,k)=>{const mm=match(selected,kind,0);return mm?frameCard(mm.target,`Best: ${labels[k]}`):'';}).join('');
 document.querySelectorAll('#matchCards [data-pick], #anchor [data-pick]').forEach(el=>el.onclick=()=>pick(Number(el.dataset.pick),true));
}
function pick(i,reveal=false){
 selected=i;const r=localRows[i];
 if(reveal&&!visible(i)){$('robot').value='';$('task').value='';updateFilter();return;}
 if(r.episode!==episode){$('episode').value=r.episode;setEpisode(r.episode);return;}
 renderFrame();
}
function step(delta){const pos=epIds.indexOf(selected);pick(epIds[Math.max(0,Math.min(epIds.length-1,pos+delta))]);}
function stopPlaying(){if(playing){clearInterval(playing);playing=null;$('play').textContent='▶ Play';}}
function setRadius(r){radius=r;$('radius').value=r;$('radiusLabel').textContent=r;if(panel)renderProfile();}
options('group',uniq(D.panels.map(p=>p.group)));$('windowLabel').textContent=D.window_s;$('radius').value=radius;$('radiusLabel').textContent=radius;
for(let k=1;k<=D.match_top_k;k++)$('rank').add(new Option(String(k),String(k)));
['group','layer','text','geometry'].forEach(id=>$(id).onchange=updatePanel);
['robot','task'].forEach(id=>$(id).onchange=updateFilter);
['measure','crange'].forEach(id=>$(id).onchange=()=>{if(panel)setEpisode(episode);});
['crossWith','rank','crossEpisode','delta','span'].forEach(id=>$(id).onchange=()=>{if(panel)renderCross();});
$('episode').onchange=()=>setEpisode($('episode').value);
$('frame').oninput=()=>{if(panel)pick(epIds[Number($('frame').value)]);};
$('radius').oninput=()=>setRadius(Number($('radius').value));
$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('play').onclick=()=>{if(playing)return stopPlaying();if(!panel)return;$('play').textContent='⏸ Pause';playing=setInterval(()=>{const pos=epIds.indexOf(selected);if(pos>=epIds.length-1)stopPlaying();else step(1);},250);};
document.addEventListener('keydown',e=>{if(/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)&&e.target.id!=='frame')return;if(e.key==='ArrowLeft'){step(-1);e.preventDefault();}else if(e.key==='ArrowRight'){step(1);e.preventDefault();}else if(e.key===' '){$('play').click();e.preventDefault();}});
updatePanel();
</script></html>'''

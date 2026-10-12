"""A portable, linked Plotly inspector for cached representation trajectories.

``trajectories.html`` holds only the sampled rows and a panel index. Each
group/layer/prompt/geometry panel (coordinates, temporal metrics, matches and the
per-episode cosine matrices) is written to ``panels/<key>.js`` and loaded on
demand through a script tag, which works both from the probe viewer and from a
plain ``file://`` open, as long as the directory moves as a whole. The packed
every-frame-against-every-frame cosine matrix behind the anchored cross-episode
grid is a second sidecar, ``panels/<key>.cross.js``, loaded only when that grid
is used. Camera previews are referenced from ``cache/thumbs`` rather than embedded.

Distance and norm-ratio views are derived in the browser from cosines and the
stored per-frame norms, so one stored matrix serves every measure.
"""
from __future__ import annotations

import json
from pathlib import Path

from plotly.offline import get_plotlyjs

ROW_KEYS = ("row", "sample", "text", "robot", "episode", "episode_label", "frame", "time_s",
            "task", "subtask", "holdout")
PANEL_INDEX_KEYS = ("key", "group", "layer", "text", "geometry", "variance", "pca_fit", "centering")
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
             "window_s": payload["window_s"], "lag_edges_s": payload["lag_edges_s"], "view": payload["view"],
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
button{cursor:pointer}button:hover{background:#e9f1ff}.grid{display:grid;grid-template-columns:minmax(400px,1fr) minmax(400px,1fr);gap:14px;margin-top:14px}.box{background:white;border:1px solid #e0e4eb;border-radius:8px;padding:12px;min-width:0}
#space,#matrix,#profile{height:520px}#timeline{height:560px}#lag{height:420px}#cross{height:640px}#crossSummary{height:300px}.cards{display:flex;flex-wrap:wrap;gap:10px}.card{border:1px solid #d8dfeb;border-radius:6px;padding:10px;flex:1;min-width:270px;max-width:580px}.cameras{display:flex;gap:4px;margin:7px 0}.cameras img{width:49%;object-fit:contain;max-height:190px;background:#edf0f5}.cameras span{display:block;font-size:11px}.details{font-size:12px;overflow-wrap:anywhere}.tag{font-size:11px;background:#edf0f5;border-radius:3px;padding:2px 5px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:8px;border-bottom:1px solid #e1e6ee;text-align:left}tr.pick{cursor:pointer}tr.pick:hover{background:#edf5ff}.scroll{overflow:auto}.note{padding:9px 12px;background:#edf2f8;border-radius:5px}#status,#episodeStats,#crossInfo{white-space:pre-wrap}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}.row input[type=range]{flex:1;min-width:160px}.row label.inline{flex-direction:row;align-items:center;gap:8px;flex:1}
.crossgrid{display:grid;grid-template-columns:minmax(400px,2fr) minmax(300px,1fr);gap:14px}
@media(max-width:950px){.grid,.crossgrid{grid-template-columns:1fr}body{padding:10px}#space,#matrix,#profile{height:420px}}
</style>
<script>__PLOTLY__</script>
<h1>Representation trajectories</h1>
<p class="muted">Follow an episode frame by frame. Every pairwise view compares two frames' pooled vectors in the original dimensions of the chosen geometry, under the chosen measure: cosine (direction only), L2 distance (direction and magnitude), or log2 norm ratio (magnitude only). Next to each value sits the level it reaches against <em>other</em> episodes, so a high local similarity can be read as trivial or not.</p>
<p class="note">Raw cosines between model states sit near 1 because all states share a large common direction. “Subtract robot mean” removes the per-robot part; “subtract episode mean” also removes the scene so only motion within the episode remains. The background line is the frame's mean value against other episodes of the same robot: structure is only non-trivial where the profile separates from that line. Dataset replays at the first flow step with fixed noise.</p>
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
<section class="box"><h2>Aligned cross-episode grid</h2>
<div class="row"><label>Other episode<select id="crossWith"><option value="same_robot_same_task">Best match: same robot · same task</option><option value="same_robot_different_task">Best match: same robot · different task</option><option value="different_robot_same_task">Best match: other robot · same task</option><option value="different_robot_different_task">Best match: other robot · different task</option><option value="episode">Chosen episode (aligned by current measure)</option></select></label>
<label>Episode<select id="crossEpisode"></select></label><label>Tile size (frames)<select id="tile"><option>10</option><option>15</option><option selected>20</option><option>25</option><option>30</option></select></label></div>
<p id="crossInfo" class="muted"></p>
<div class="crossgrid"><div><div id="cross"></div></div><div><div id="crossSummary"></div><div id="crossCards" class="cards"></div></div></div>
<p class="caption">Rows: the selected episode around the selected frame; columns: the other episode around its aligned frame (the best match for that category, or the best frame of a chosen episode). The aligned pair sits at the center of the center tile; each tile is the chosen number of frames, so tile (i, j) compares the selected episode shifted by i tiles with the other episode shifted by j tiles. White lines are tile borders, the crosshair is the aligned pair; windows are clipped at episode ends. A bright band running through tiles (−1,−1), (0,0), (+1,+1) means the two episodes move through the same states at the same pace; a band bending away means one runs faster; bright patches away from the diagonal are states revisited at different times. The small grid is the mean of each tile. Click a cell to select its row frame.</p></section>
<div class="grid"><section class="box"><h2>How local is local</h2><div id="lag"></div>
<p class="caption">Each curve is one shown episode (selected bold): the mean value over all pairs of its frames whose time gap falls in the bin (edges in seconds, last bin open; for norm ratio the later frame is divided by the earlier). Reference lines are means over the selected episode's frames: dashed, background against other episodes of the same robot; dotted, best cross-episode match. Where a curve reaches the dashed line by gap G, frames more than G apart are no closer than frames of other episodes.</p></section>
<section class="box"><h2>Shared 3-D PCA</h2><div class="row"><label>3-D view<select id="view3d"><option value="off" selected>Hidden</option><option value="selected">Selected episode only</option><option value="all">All shown episodes</option></select></label><label>Color markers<select id="color"><option value="time">Episode progress</option><option value="episode">Episode</option></select></label></div><div id="space"></div>
<p class="caption">Three principal axes fitted once on all training frames of this group, layer, prompt and geometry (explained variance in the status line), not refitted when filtering. Hidden by default because WebGL redraws of all trajectories are slow on large captures. Diamonds: the selected frame (dark) and its best matches (orange). A crossing in 3-D is a candidate for inspection, not evidence.</p></section></div>
<section class="box"><h2>Along the episode</h2><div id="timeline"></div>
<p class="caption">Top: values under the current measure along the episode: local (mean over the radius), previous frame, background against other episodes of the same robot, and the best cross-episode match per category (each point may match a different episode). Middle: vector norms, raw and after centering. Bottom: how many frames fall inside the radius, so a flat local curve on one neighbor is visible as such.</p></section>
<section class="box"><h2>Nearest frames in other episodes</h2><p class="muted">Matches are found by cosine. Search covers every captured episode in this group, layer and prompt, including episodes hidden by display filters. “Same task” uses exact recorded task text; different wording may describe the same behavior. Target-window cosine compares the anchor to frames within ±<span id="windowLabel"></span> s of its best match, within that match's episode and task. A support count of 1 is just the matched frame.</p>
<div class="scroll"><table><thead><tr><th>Candidate set</th><th>Best cosine</th><th>Distance</th><th>Raw norm ratio<br>target / anchor</th><th>Target-window cosine</th><th>Support</th><th>Candidates</th><th>Episode / time</th></tr></thead><tbody id="matches"></tbody></table></div>
<div id="matchCards" class="cards" style="margin-top:12px"></div></section>
<p class="muted">Local values exclude the selected frame and stay inside its episode; the radius slider recomputes them from the stored matrix. Gaps mean no valid temporal neighbors or zero-length vectors; no zeros are substituted. Display filters preserve the fitted PCA axes. Axes are refitted when group, layer, prompt, geometry or checkpoint changes. Raw norm is always measured before centering; distance and norm ratio use the norms of the chosen geometry. Keyboard: ← → step frames, space plays.</p>
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
const MEASURES={cosine:{label:'Cosine',scale:'Viridis',reverse:false,bg:'background_same_robot'},distance:{label:'L2 distance',scale:'Viridis',reverse:true,bg:'background_distance_same_robot'},lognorm:{label:'log2 norm ratio (column / row)',scale:'RdBu',reverse:false,bg:'background_lognorm_same_robot'}};
const cache={}, crossCache={}, pending={};
window.__trajectoryPanel=(key,data)=>{cache[key]=data;if(pending[key]){pending[key](data);delete pending[key];}};
window.__trajectoryCross=(key,text)=>{const bytes=Uint8Array.from(atob(text),c=>c.charCodeAt(0));crossCache[key]=new Uint16Array(bytes.buffer);if(pending['cross:'+key]){pending['cross:'+key](crossCache[key]);delete pending['cross:'+key];}};
function loadScript(id,file,store){return new Promise((resolve,reject)=>{if(store[id.replace(/^cross:/,'')])return resolve(store[id.replace(/^cross:/,'')]);pending[id]=resolve;const s=document.createElement('script');s.src=file;s.onerror=()=>{delete pending[id];reject(new Error(`Could not load ${file}; keep panels/ next to this page`));};document.head.appendChild(s);});}
let panel=null, localRows=[], selected=0, episode=null, epIds=[], epTimes=[], epCos=[], radius=D.window_s, playing=null, loadToken=0, cross=null;
function options(id,values,all=false){const old=$(id).value;$(id).replaceChildren();if(all)$(id).add(new Option('All',''));values.forEach(v=>$(id).add(new Option(v,v)));if(values.includes(old)||all&&old==='')$(id).value=old;}
function visible(i){const r=localRows[i];return (!$('robot').value||r.robot===$('robot').value)&&(!$('task').value||r.task===$('task').value);}
function match(i,kind){const m=panel.matches[kind];return m.target[i]==null?null:Object.fromEntries(Object.keys(m).map(k=>[k,m[k][i]]));}
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
 const variance=panel.variance.map(v=>(100*v).toFixed(1)+'%').join(' + ');
 const center={raw:'PCA centers globally; cosine uses the uncentered captured vectors.',robot_centered:Object.entries(panel.centering).map(([r,p])=>`${r}: ${p.fit}, n=${p.rows}`).join('; '),episode_centered:'Each episode centered on its own sampled mean (all of its frames).'}[panel.geometry];
 const bg=mean(panel.temporal.background_same_robot), bgOther=mean(panel.temporal.background_other_robot), gap=median(panel.temporal.previous_dt_s);
 $('status').textContent=`Capture: ${D.view} · ${panel.group} · L${panel.layer} · ${localRows.length} frames in ${uniq(localRows.map(r=>r.episode)).length} episodes, median sampling gap ${fmt(gap,2)} s.\nPCA variance: ${variance} = ${(100*panel.variance.reduce((a,b)=>a+b,0)).toFixed(1)}%. Basis fit: ${panel.pca_fit}. ${center}\nBackground cosine to other episodes: same robot ${fmt(bg)}, other robots ${fmt(bgOther)} (mean over frames). Checkpoint: ${D.checkpoint}`;
 updateFilter();
}
function updateFilter(){
 const eps=uniq(localRows.filter((r,i)=>visible(i)).map(r=>r.episode));
 const previous=localRows[selected]?.episode;
 $('episode').replaceChildren();eps.forEach(ep=>{const r=localRows.find(r=>r.episode===ep);$('episode').add(new Option(`${r.robot} · ${r.episode_label} · ${r.holdout?'held out':'train'}`,ep));});
 if(eps.includes(previous))$('episode').value=previous;
 if(!eps.length){for(const id of ['space','matrix','profile','timeline','lag','cross','crossSummary'])Plotly.purge(id);$('anchor').replaceChildren();$('matches').replaceChildren();$('matchCards').replaceChildren();$('frameStats').textContent='';$('episodeStats').textContent='No frames match these filters';return;}
 if(!visible(selected))selected=localRows.findIndex((r,i)=>visible(i)&&r.episode===$('episode').value);
 renderSpace();setEpisode($('episode').value);
}
function setEpisode(ep){
 episode=ep;const e=panel.episodes[ep];epIds=e.ids;epTimes=e.times;epCos=e.cosine;
 if(localRows[selected]?.episode!==ep)selected=epIds[0];
 $('frame').max=epIds.length-1;
 const gaps=epIds.map(i=>panel.temporal.previous_dt_s[i]), r=localRows[epIds[0]];
 $('episodeStats').textContent=`${epIds.length} frames over ${fmt(epTimes[epTimes.length-1]-epTimes[0],1)} s, median gap ${fmt(median(gaps),2)} s · ${esc(r.task)}`;
 const others=Object.keys(panel.episodes).filter(x=>x!==ep).sort();const old=$('crossEpisode').value;$('crossEpisode').replaceChildren();others.forEach(x=>{const rr=localRows[panel.episodes[x].ids[0]];$('crossEpisode').add(new Option(`${rr.robot} · ${rr.episode_label} · ${rr.task}`,x));});if(others.includes(old))$('crossEpisode').value=old;
 renderMatrix();renderTimeline();renderLag();if($('view3d').value==='selected')renderSpace();else styleSpace();renderFrame();
}
function renderSpace(){
 if($('view3d').value==='off'){Plotly.purge('space');$('space').textContent='';return;}
 const episodes=uniq(localRows.filter((r,i)=>visible(i)).map(r=>r.episode)).filter(ep=>$('view3d').value==='all'||ep===episode), traces=[];
 episodes.forEach((ep,k)=>{
  const ids=panel.episodes[ep].ids, times=panel.episodes[ep].times, lo=times[0], hi=times[times.length-1], r=localRows[ids[0]], color=colors[k%colors.length];
  traces.push({type:'scatter3d',mode:'lines+markers',name:`${r.robot} · ${r.episode_label}`,x:ids.map(i=>panel.xyz[i][0]),y:ids.map(i=>panel.xyz[i][1]),z:ids.map(i=>panel.xyz[i][2]),customdata:ids,
   text:ids.map(i=>`${esc(localRows[i].episode_label)} · ${fmt(localRows[i].time_s,2)} s<br>${esc(localRows[i].subtask)}`),hovertemplate:'%{text}<extra></extra>',
   line:{color,width:2},marker:{size:3,color:$('color').value==='time'?times.map(t=>(t-lo)/Math.max(hi-lo,1e-8)):color,colorscale:'Viridis',cmin:0,cmax:1,showscale:k===0&&$('color').value==='time',colorbar:{title:{text:'Progress'},len:.45}},opacity:.8});
 });
 traces.push({type:'scatter3d',mode:'markers',name:'Selected / matched frames',x:[],y:[],z:[],text:[],hovertemplate:'%{text}<extra></extra>',marker:{size:7,color:[],symbol:'diamond'}});
 const ranges=[0,1,2].map(axis=>{let lo=Infinity,hi=-Infinity;for(const p of panel.xyz){if(p[axis]<lo)lo=p[axis];if(p[axis]>hi)hi=p[axis];}const pad=Math.max((hi-lo)*.05,1e-6);return [lo-pad,hi+pad];});
 Plotly.react('space',traces,{margin:{l:0,r:0,t:5,b:0},uirevision:panel.key,legend:{orientation:'h',font:{size:10},y:-.05},scene:{aspectmode:'data',xaxis:{title:{text:'PC1'},range:ranges[0]},yaxis:{title:{text:'PC2'},range:ranges[1]},zaxis:{title:{text:'PC3'},range:ranges[2]}}},{responsive:true,displaylogo:false});
 $('space').removeAllListeners?.('plotly_click');$('space').on('plotly_click',event=>{const i=event.points[0]?.customdata;if(Number.isInteger(i))pick(i,true);});
 styleSpace();
}
function styleSpace(){
 const g=$('space');if(!g.data||$('view3d').value==='off')return;const base=g.data.slice(0,-1);if(!base.length)return;
 const sel=base.map(t=>t.customdata&&localRows[t.customdata[0]].episode===episode);
 Plotly.restyle('space',{opacity:sel.map(s=>s?1:.3),'line.width':sel.map(s=>s?5:1.5)},base.map((_,k)=>k));
}
function drawHighlight(){
 const g=$('space');if(!g.data||$('view3d').value==='off')return;
 const points=[selected],names=['Selected frame'];kinds.forEach((kind,k)=>{const m=match(selected,kind);if(m){points.push(m.target);names.push(labels[k]);}});
 Plotly.restyle('space',{x:[points.map(i=>panel.xyz[i][0])],y:[points.map(i=>panel.xyz[i][1])],z:[points.map(i=>panel.xyz[i][2])],text:[names],'marker.color':[points.map((_,i)=>i?'#e37525':'#171e2b')]},[g.data.length-1]);
}
function heat(z,x,y,xtitle,ytitle,zrange,extraShapes){
 const M=MEASURES[measure()];
 return [[{type:'heatmap',x,y,z,colorscale:M.scale,reversescale:M.reverse,zmin:zrange[0],zmax:zrange[1],hoverongaps:false,hovertemplate:'row %{y:.2f} s · col %{x:.2f} s<br>%{z:.4f}<extra></extra>',colorbar:{title:{text:M.label.split(' (')[0]},len:.9}}],
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
 const ids=epIds, traces=[], t=epTimes, T=panel.temporal, M=MEASURES[measure()], matrix=episodeMatrix(episode);
 const add=(name,values,yaxis,color,dash)=>traces.push({type:'scatter',mode:'lines+markers',name,x:t,y:values,customdata:ids,yaxis,line:{color,dash},marker:{size:4},connectgaps:false,hovertemplate:'%{x:.2f} s · %{y:.4f}<extra>%{fullData.name}</extra>'});
 const local=ids.map((_,pos)=>localValue(pos,radius,matrix));
 add(`Local ${M.label.toLowerCase()} ±${radius} s`,local.map(v=>v[0]),'y','#202631');
 add('Previous frame',ids.map((i,pos)=>pos?val(T.previous_cosine[i],ids[pos-1],i):null),'y','#949daa','dot');
 add('Background: other episodes, same robot',ids.map(i=>T[M.bg][i]),'y','#949daa','dash');
 kinds.forEach((kind,k)=>add(labels[k],ids.map(i=>matchValue(i,match(i,kind))),'y',colors[k],k%2?'dash':'solid'));
 add('Raw norm',ids.map(i=>T.raw_norm[i]),'y2','#3366cc');
 if(panel.geometry!=='raw')add('Centered norm',ids.map(i=>T.geometry_norm[i]),'y2','#dd6633');
 add('Neighbors in radius',local.map(v=>v[1]),'y3','#119988');
 Plotly.react('timeline',traces,{margin:{l:58,r:15,t:12,b:85},uirevision:panel.key+episode,xaxis:{title:{text:'Episode time (seconds)'},anchor:'y3'},yaxis:{title:{text:M.label.split(' (')[0]},domain:[.47,1]},yaxis2:{title:{text:'Norm'},domain:[.22,.4],rangemode:'tozero'},yaxis3:{title:{text:'Neighbors'},domain:[0,.15],rangemode:'tozero'},legend:{orientation:'h',font:{size:10},y:-.2},shapes:[cursor()]},{responsive:true,displaylogo:false});
 $('timeline').removeAllListeners?.('plotly_click');$('timeline').on('plotly_click',event=>{const i=event.points[0]?.customdata;if(Number.isInteger(i))pick(i);});
}
function cursor(){return {type:'line',xref:'x',yref:'paper',x0:localRows[selected].time_s,x1:localRows[selected].time_s,y0:0,y1:1,line:{color:'#777',width:1}};}
function lagCurve(ep){
 const edges=D.lag_edges_s, e=panel.episodes[ep], matrix=episodeMatrix(ep), sums=edges.map(()=>0), counts=edges.map(()=>0);
 for(let a=0;a<e.times.length;a++)for(let b=a+1;b<e.times.length;b++){const v=matrix[a][b];if(v==null)continue;const gap=e.times[b]-e.times[a];let k=0;while(k+1<edges.length&&gap>edges[k+1])k++;sums[k]+=v;counts[k]++;}
 return {values:sums.map((s,k)=>counts[k]?s/counts[k]:null),counts};
}
function renderLag(){
 const edges=D.lag_edges_s, names=edges.map((lo,k)=>k+1<edges.length?`${lo}–${edges[k+1]}`:`${lo}+`), traces=[], M=MEASURES[measure()];
 const eps=uniq(localRows.filter((r,i)=>visible(i)).map(r=>r.episode));
 eps.forEach(ep=>{const e=panel.episodes[ep], r=localRows[e.ids[0]], sel=ep===episode, c=lagCurve(ep);
  traces.push({type:'scatter',mode:'lines+markers',name:`${r.robot} · ${r.episode_label}`,x:names,y:c.values,text:c.counts.map(n=>`${n} pairs`),connectgaps:false,hovertemplate:'%{x} s · %{y:.4f} · %{text}<extra>%{fullData.name}</extra>',line:{color:sel?'#202631':'#b8c0ce',width:sel?3:1},marker:{size:sel?6:3},opacity:sel?1:.6,showlegend:sel});});
 const ids=epIds, ref=(name,y,color,dash)=>{if(y!=null)traces.push({type:'scatter',mode:'lines',name,x:[names[0],names[names.length-1]],y:[y,y],line:{color,dash,width:1.5},hoverinfo:'name+y'});};
 ref('Background: other episodes, same robot (episode mean)',mean(ids.map(i=>panel.temporal[M.bg][i])),'#949daa','dash');
 ref('Best match: same robot · same task (episode mean)',mean(ids.map(i=>matchValue(i,match(i,kinds[0])))),colors[0],'dot');
 ref('Best match: same robot · different task (episode mean)',mean(ids.map(i=>matchValue(i,match(i,kinds[1])))),colors[1],'dot');
 Plotly.react('lag',traces,{margin:{l:55,r:10,t:10,b:60},xaxis:{title:{text:'Time gap between frames (s)'},type:'category'},yaxis:{title:{text:`Mean ${M.label.split(' (')[0].toLowerCase()}`}},legend:{orientation:'h',font:{size:10},y:-.25}},{responsive:true,displaylogo:false});
}
function crossCos(i,j){const q=cross[i*localRows.length+j];return q?(q-1)/65534*2-1:null;}
async function renderCross(){
 const token=loadToken;
 if(!panel.cross_file){$('crossInfo').textContent='Cross-episode matrices were not written (repr_cross_matrices=false); re-run in plot mode with it enabled.';Plotly.purge('cross');Plotly.purge('crossSummary');$('crossCards').replaceChildren();return;}
 if(!cross){$('crossInfo').textContent='Loading the every-frame matrix for this panel…';try{cross=await loadScript('cross:'+panel.key,panel.cross_file,crossCache);}catch(e){$('crossInfo').textContent=e.message;return;}if(token!==loadToken)return;}
 const mode=$('crossWith').value, tile=Number($('tile').value), pos=epIds.indexOf(selected);
 let other, posB, how;
 if(mode==='episode'){other=$('crossEpisode').value;if(!other){$('crossInfo').textContent='No other episode in this panel.';return;}
  const e=panel.episodes[other];let best=null;e.ids.forEach((j,k)=>{const v=val(crossCos(selected,j),selected,j);if(v==null)return;const better=measure()==='distance'?v<best?.v:measure()==='lognorm'?Math.abs(v)<Math.abs(best?.v):v>best?.v;if(best==null||better)best={v,k};});
  if(!best){$('crossInfo').textContent='No valid frames to align with in that episode.';return;}posB=best.k;how=`best frame of the chosen episode by ${MEASURES[measure()].label.toLowerCase()}`;
 }else{const m=match(selected,mode);if(!m){$('crossInfo').textContent=`No eligible frames for “${labels[kinds.indexOf(mode)]}”.`;Plotly.purge('cross');Plotly.purge('crossSummary');$('crossCards').replaceChildren();return;}
  other=localRows[m.target].episode;posB=panel.episodes[other].ids.indexOf(m.target);how=`best cosine match, ${labels[kinds.indexOf(mode)].toLowerCase()}`;}
 const A=panel.episodes[episode], B=panel.episodes[other], half=Math.floor(1.5*tile);
 const ra=[Math.max(0,pos-half),Math.min(A.ids.length,pos+half+1)], rb=[Math.max(0,posB-half),Math.min(B.ids.length,posB+half+1)];
 const ia=A.ids.slice(ra[0],ra[1]), ib=B.ids.slice(rb[0],rb[1]), ta=A.times.slice(ra[0],ra[1]), tb=B.times.slice(rb[0],rb[1]);
 const z=ia.map(i=>ib.map(j=>val(crossCos(i,j),i,j)));
 const edge=(times,k)=>k<=0?times[0]-(times[1]-times[0])/2:k>=times.length?times[times.length-1]+(times[times.length-1]-times[times.length-2])/2:(times[k-1]+times[k])/2;
 const shapes=[];
 for(const s of [-1,1]){const ka=pos-ra[0]+s*tile/2, kb=posB-rb[0]+s*tile/2;
  if(ka>0&&ka<ia.length)shapes.push({type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:edge(ta,Math.round(ka)),y1:edge(ta,Math.round(ka)),line:{color:'white',width:1.5}});
  if(kb>0&&kb<ib.length)shapes.push({type:'line',xref:'x',yref:'paper',y0:0,y1:1,x0:edge(tb,Math.round(kb)),x1:edge(tb,Math.round(kb)),line:{color:'white',width:1.5}});}
 shapes.push({type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:ta[pos-ra[0]],y1:ta[pos-ra[0]],line:{color:'#e37525',width:1,dash:'dot'}},{type:'line',xref:'x',yref:'paper',y0:0,y1:1,x0:tb[posB-rb[0]],x1:tb[posB-rb[0]],line:{color:'#e37525',width:1,dash:'dot'}});
 const rA=localRows[selected], rB=localRows[B.ids[posB]];
 const range=$('crange').value==='auto'?colorRange(z.flat()):zRange(episodeMatrix(episode));
 Plotly.react('cross',...heat(z,tb,ta,`${rB.robot} · ${rB.episode_label} time (s)`,`${rA.robot} · ${rA.episode_label} time (s)`,range,shapes));
 $('cross').removeAllListeners?.('plotly_click');$('cross').on('plotly_click',event=>{const p=event.points[0];const idx=p?.pointIndex??p?.pointNumber;if(Array.isArray(idx))pick(ia[idx[0]]);});
 const tiles=[-1,0,1], summary=tiles.map(i=>tiles.map(j=>{const vals=[];for(let a=0;a<ia.length;a++){if(Math.floor((a-(pos-ra[0])+tile/2)/tile)!==i)continue;for(let b=0;b<ib.length;b++){if(Math.floor((b-(posB-rb[0])+tile/2)/tile)!==j)continue;if(z[a][b]!=null)vals.push(z[a][b]);}}return vals.length?mean(vals):null;}));
 const M=MEASURES[measure()];
 Plotly.react('crossSummary',[{type:'heatmap',x:['−1','0','+1'],y:['−1','0','+1'],z:summary,colorscale:M.scale,reversescale:M.reverse,zmin:range[0],zmax:range[1],showscale:false,text:summary.map(r=>r.map(v=>fmt(v))),texttemplate:'%{text}',hovertemplate:'tiles (%{y}, %{x}) · mean %{z:.4f}<extra></extra>'}],
  {margin:{l:70,r:10,t:30,b:50},title:{text:'Mean per tile',font:{size:13}},xaxis:{title:{text:`${rB.episode_label} tile shift`},type:'category'},yaxis:{title:{text:`${rA.episode_label} tile shift`},type:'category',autorange:'reversed'}},{responsive:true,displaylogo:false});
 const aligned=val(crossCos(selected,B.ids[posB]),selected,B.ids[posB]);
 $('crossInfo').textContent=`Rows: ${rA.robot} · ${rA.episode_label} frames ${ra[0]+1}–${ra[1]} of ${A.ids.length} around ${fmt(rA.time_s,2)} s. Columns: ${rB.robot} · ${rB.episode_label} frames ${rb[0]+1}–${rb[1]} of ${B.ids.length} around ${fmt(rB.time_s,2)} s (${how}). Aligned pair ${M.label.toLowerCase()}: ${fmt(aligned,4)}. Tasks: “${rA.task}” vs “${rB.task}”.`;
 $('crossCards').innerHTML=frameCard(B.ids[posB],'Aligned frame in the other episode');
 document.querySelectorAll('#crossCards [data-pick]').forEach(el=>el.onclick=()=>pick(Number(el.dataset.pick),true));
}
function frameCard(i,title){
 const r=localRows[i];
 const cameras=['external_0','wrist_0'].map(cam=>{const src=thumb(r.sample,cam);return src?`<img src="${src}" alt="${cam}" loading="lazy">`:`<span>${cam}: unavailable</span>`;}).join('');
 return `<div class="card"><strong>${esc(title)}</strong><div class="details">${esc(r.robot)} · ${esc(r.episode_label)} <span class="tag">${r.holdout?'held out':'train'}</span><br>Frame ${r.frame} · ${fmt(r.time_s,2)} s</div><div class="cameras">${cameras}</div><div class="details"><b>Task:</b> ${esc(r.task)}<br><b>Step:</b> ${esc(r.subtask)}</div><button data-pick="${i}" style="margin-top:8px">Follow this frame</button></div>`;
}
function renderFrame(){
 const pos=epIds.indexOf(selected), m=temporal(selected), t=epTimes[pos];
 $('frame').value=pos;
 $('frameStats').textContent=`frame ${pos+1}/${epIds.length} · ${fmt(t,2)} s · raw norm ${fmt(m.raw_norm)} · ${panel.geometry==='raw'?'':`centered norm ${fmt(m.geometry_norm)} · `}step distance ${fmt(m.step_l2)}`;
 $('anchor').innerHTML=frameCard(selected,'Selected frame');
 Plotly.relayout('matrix',{shapes:[{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:t,y1:t,line:{color:'white',width:1}},{type:'line',xref:'x',yref:'paper',x0:t,x1:t,y0:0,y1:1,line:{color:'white',width:1}}]});
 Plotly.relayout('timeline',{shapes:[cursor()]});
 renderProfile();drawHighlight();renderCross();
 $('matches').innerHTML=kinds.map((kind,k)=>{
  const mm=match(selected,kind);
  if(!mm)return `<tr><td>${labels[k]}</td><td colspan="7">No eligible frames</td></tr>`;
  const r=localRows[mm.target];
  return `<tr class="pick" data-pick="${mm.target}"><td>${labels[k]}</td><td>${fmt(mm.cosine)}</td><td>${fmt(mm.distance)}</td><td>${fmt(mm.raw_norm_ratio)}</td><td>${fmt(mm.target_window_cosine)}</td><td>${mm.target_window_count}</td><td>${mm.candidate_count} frames / ${mm.candidate_episodes} episodes</td><td>${esc(r.episode_label)} · ${fmt(r.time_s,2)} s</td></tr>`;
 }).join('');
 $('matchCards').innerHTML=kinds.map((kind,k)=>{const mm=match(selected,kind);return mm?frameCard(mm.target,labels[k]):'';}).join('');
 document.querySelectorAll('#matches [data-pick], #matchCards [data-pick], #anchor [data-pick]').forEach(el=>el.onclick=()=>pick(Number(el.dataset.pick),true));
}
function pick(i,reveal=false){
 selected=i;const r=localRows[i];
 if(reveal&&!visible(i)){$('robot').value='';$('task').value='';updateFilter();return;}
 if(r.episode!==episode){$('episode').value=r.episode;setEpisode(r.episode);return;}
 renderFrame();
}
function step(delta){const pos=epIds.indexOf(selected);pick(epIds[Math.max(0,Math.min(epIds.length-1,pos+delta))]);}
function stopPlaying(){if(playing){clearInterval(playing);playing=null;$('play').textContent='▶ Play';}}
function setRadius(r){radius=r;$('radius').value=r;$('radiusLabel').textContent=r;if(panel){renderTimeline();renderProfile();}}
options('group',uniq(D.panels.map(p=>p.group)));$('windowLabel').textContent=D.window_s;$('radius').value=radius;$('radiusLabel').textContent=radius;
['group','layer','text','geometry'].forEach(id=>$(id).onchange=updatePanel);
['robot','task'].forEach(id=>$(id).onchange=updateFilter);$('color').onchange=renderSpace;$('view3d').onchange=()=>{renderSpace();drawHighlight();};
['measure','crange'].forEach(id=>$(id).onchange=()=>{if(panel)setEpisode(episode);});
['crossWith','crossEpisode','tile'].forEach(id=>$(id).onchange=()=>{if(panel)renderCross();});
$('episode').onchange=()=>setEpisode($('episode').value);
$('frame').oninput=()=>{if(panel)pick(epIds[Number($('frame').value)]);};
$('radius').oninput=()=>setRadius(Number($('radius').value));
$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('play').onclick=()=>{if(playing)return stopPlaying();if(!panel)return;$('play').textContent='⏸ Pause';playing=setInterval(()=>{const pos=epIds.indexOf(selected);if(pos>=epIds.length-1)stopPlaying();else step(1);},250);};
document.addEventListener('keydown',e=>{if(/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)&&e.target.id!=='frame')return;if(e.key==='ArrowLeft'){step(-1);e.preventDefault();}else if(e.key==='ArrowRight'){step(1);e.preventDefault();}else if(e.key===' '){$('play').click();e.preventDefault();}});
updatePanel();
</script></html>'''

"""A portable, linked Plotly inspector for cached representation trajectories.

``trajectories.html`` holds only the sampled rows and a panel index. Each
group/layer/prompt/geometry panel (coordinates, temporal metrics, matches and the
per-episode cosine matrices) is written to ``panels/<key>.js`` and loaded on
demand through a script tag, which works both from the probe viewer and from a
plain ``file://`` open, as long as the directory moves as a whole. Camera previews
are referenced from ``cache/thumbs`` rather than embedded.
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
        data = {k: v for k, v in panel.items() if k not in PANEL_INDEX_KEYS}
        (root / "panels" / f"{panel['key']}.js").write_text(
            f"window.__trajectoryPanel({_json(panel['key'])},{_json(data)});")
        index.append({**{k: panel[k] for k in PANEL_INDEX_KEYS}, "file": f"panels/{panel['key']}.js",
                      "frames": len(panel["ids"])})
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
h1{font-size:23px;margin:0 0 8px}h2{font-size:17px;margin:14px 0 8px}p{line-height:1.5;margin:8px 0}.muted{color:#5e6878}
.controls{display:flex;flex-wrap:wrap;gap:12px;background:white;padding:12px;border-radius:8px;margin:14px 0}
label{display:flex;flex-direction:column;gap:5px;font-size:12px;color:#505b6e}select,button{font:inherit;border:1px solid #b8c0ce;border-radius:4px;padding:7px;background:white;color:#202631;max-width:420px}
button{cursor:pointer}button:hover{background:#e9f1ff}.grid{display:grid;grid-template-columns:minmax(400px,1fr) minmax(400px,1fr);gap:14px;margin-top:14px}.box{background:white;border:1px solid #e0e4eb;border-radius:8px;padding:12px;min-width:0}
#space,#matrix{height:520px}#profile{height:520px}#timeline{height:560px}#lag{height:420px}.cards{display:flex;flex-wrap:wrap;gap:10px}.card{border:1px solid #d8dfeb;border-radius:6px;padding:10px;flex:1;min-width:270px;max-width:580px}.cameras{display:flex;gap:4px;margin:7px 0}.cameras img{width:49%;object-fit:contain;max-height:190px;background:#edf0f5}.cameras span{display:block;font-size:11px}.details{font-size:12px;overflow-wrap:anywhere}.tag{font-size:11px;background:#edf0f5;border-radius:3px;padding:2px 5px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:8px;border-bottom:1px solid #e1e6ee;text-align:left}tr.pick{cursor:pointer}tr.pick:hover{background:#edf5ff}.scroll{overflow:auto}.note{padding:9px 12px;background:#edf2f8;border-radius:5px}#status,#episodeStats{white-space:pre-wrap}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}.row input[type=range]{flex:1;min-width:160px}.row label.inline{flex-direction:row;align-items:center;gap:8px;flex:1}
@media(max-width:950px){.grid{grid-template-columns:1fr}body{padding:10px}#space,#matrix,#profile{height:420px}}
</style>
<script>__PLOTLY__</script>
<h1>Representation trajectories</h1>
<p class="muted">Follow an episode frame by frame. The cosine matrix shows every frame of the episode against every other, in the original dimensions; the profile shows the selected frame against its episode next to the level it reaches against <em>other</em> episodes, so a high local cosine can be read as trivial or not.</p>
<p class="note">Raw cosines between model states sit near 1 because all states share a large common direction. “Subtract robot mean” removes the per-robot part; “subtract episode mean” also removes the scene so only motion within the episode remains. The background line is the frame's mean cosine to other episodes of the same robot: structure is only non-trivial where the profile separates from that line. 3-D PCA is a projection; a visual crossing is a candidate for inspection, not evidence. Dataset replays at the first flow step with fixed noise.</p>
<div class="controls">
<label>Token group<select id="group"></select></label><label>Layer<select id="layer"></select></label>
<label>Prompt<select id="text"><option value="real">Real task / subtask</option><option value="neutral">Constant task / subtask</option></select></label>
<label>Geometry<select id="geometry"><option value="raw">Raw vectors</option><option value="robot_centered" selected>Subtract robot mean</option><option value="episode_centered">Subtract episode mean</option></select></label>
<label>Show robot<select id="robot"></select></label><label>Show task<select id="task"></select></label>
<label>3-D view<select id="view3d"><option value="all">All shown episodes</option><option value="selected">Selected episode only</option><option value="off">Hidden</option></select></label>
<label>Color markers<select id="color"><option value="time">Episode progress</option><option value="episode">Episode</option></select></label>
</div>
<p id="status" class="muted"></p>
<section class="box"><h2>Episode</h2>
<div class="row"><label>Episode<select id="episode"></select></label><p id="episodeStats" class="muted" style="margin:0"></p></div>
<div class="row"><button id="play">▶ Play</button><button id="previous">←</button><input type="range" id="frame" min="0" max="0" value="0"><button id="next">→</button><span id="frameStats" class="muted"></span></div>
<div id="anchor" class="cards"></div></section>
<div class="grid"><section class="box"><h2>Within-episode cosine matrix</h2>
<div class="row"><label>Color range<select id="crange"><option value="auto">Fit to this episode</option><option value="positive">0 to 1</option><option value="unit">−1 to 1</option></select></label><span class="muted">Click a cell to select its row frame. Diagonal is 1 by definition.</span></div><div id="matrix"></div></section>
<section class="box"><h2>Selected frame against its episode</h2>
<div class="row"><label class="inline"><span>Local radius: ±<span id="radiusLabel"></span> s</span><input type="range" id="radius" min="0.25" max="30" step="0.25"></label><span id="radiusStats" class="muted"></span></div><div id="profile"></div></section></div>
<div class="grid"><section class="box"><h2>Shared 3-D PCA</h2><div id="space"></div></section>
<section class="box"><h2>How local is local</h2><p class="muted">Mean cosine over frame pairs by time gap, one curve per shown episode (selected episode bold). Where a curve meets the background or best-match level, frames that far apart are no longer closer than other episodes are.</p><div id="lag"></div></section></div>
<section class="box"><h2>Along the episode</h2><div id="timeline"></div></section>
<section class="box"><h2>Nearest frames in other episodes</h2><p class="muted">Search covers every captured episode in this group, layer and prompt, including episodes hidden by display filters. “Same task” uses exact recorded task text; different wording may describe the same behavior. Target-window cosine compares the anchor to frames within ±<span id="windowLabel"></span> s of its best match, within that match's episode and task. A support count of 1 is just the matched frame.</p>
<div class="scroll"><table><thead><tr><th>Candidate set</th><th>Best cosine</th><th>Raw norm ratio<br>target / anchor</th><th>Distance</th><th>Target-window cosine</th><th>Support</th><th>Candidates</th><th>Episode / time</th></tr></thead><tbody id="matches"></tbody></table></div>
<div id="matchCards" class="cards" style="margin-top:12px"></div></section>
<p class="muted">Local cosine excludes the selected frame and stays inside its episode; the radius slider recomputes it from the matrix. Gaps mean no valid temporal neighbors or zero-length vectors; no zeros are substituted. Nearest-match curves may switch episodes at each time. Display filters preserve the fitted PCA axes. Axes are refitted when group, layer, prompt, geometry or checkpoint changes. Raw norm is always measured before centering or unit normalization. Keyboard: ← → step frames, space plays.</p>
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
const cache={}, pending={};
window.__trajectoryPanel=(key,data)=>{cache[key]=data;if(pending[key]){pending[key](data);delete pending[key];}};
function loadPanel(entry){return new Promise((resolve,reject)=>{if(cache[entry.key])return resolve(cache[entry.key]);pending[entry.key]=resolve;const s=document.createElement('script');s.src=entry.file;s.onerror=()=>{delete pending[entry.key];reject(new Error(`Could not load ${entry.file}; keep panels/ next to this page`));};document.head.appendChild(s);});}
let panel=null, localRows=[], selected=0, episode=null, epIds=[], epTimes=[], epCos=[], radius=D.window_s, playing=null, loadToken=0;
function options(id,values,all=false){const old=$(id).value;$(id).replaceChildren();if(all)$(id).add(new Option('All',''));values.forEach(v=>$(id).add(new Option(v,v)));if(values.includes(old)||all&&old==='')$(id).value=old;}
function visible(i){const r=localRows[i];return (!$('robot').value||r.robot===$('robot').value)&&(!$('task').value||r.task===$('task').value);}
function match(i,kind){const m=panel.matches[kind];return m.target[i]==null?null:Object.fromEntries(Object.keys(m).map(k=>[k,m[k][i]]));}
function temporal(i){return Object.fromEntries(Object.keys(panel.temporal).map(k=>[k,panel.temporal[k][i]]));}
function localCosine(pos,r){const t=epTimes[pos],row=epCos[pos];let s=0,n=0;for(let k=0;k<epTimes.length;k++){if(k!==pos&&row[k]!=null&&Math.abs(epTimes[k]-t)<=r){s+=row[k];n++;}}return n?[s/n,n]:[null,0];}
function thumb(sample,camera){const cams=D.thumbs[String(sample)]||[];return cams.includes(camera)?`${D.thumb_dir}/${String(sample).padStart(4,'0')}.${camera}.jpg`:null;}
async function updatePanel(){
 const sample=localRows[selected]?.sample, token=++loadToken;
 let candidates=D.panels.filter(p=>p.group===$('group').value);
 options('layer',uniq(candidates.map(p=>String(p.layer))));
 candidates=candidates.filter(p=>String(p.layer)===$('layer').value);
 const allowed=uniq(candidates.map(p=>p.text));for(const o of $('text').options)o.disabled=!allowed.includes(o.value);
 if(!allowed.includes($('text').value))$('text').value=allowed[0];
 const entry=candidates.find(p=>p.text===$('text').value&&p.geometry===$('geometry').value);if(!entry)return;
 $('status').textContent=`Loading ${entry.key}…`;
 let data;try{data=await loadPanel(entry);}catch(e){$('status').textContent=e.message;return;}
 if(token!==loadToken)return;
 panel={...entry,...data};localRows=panel.ids.map(i=>D.rows[i]);selected=Math.max(0,localRows.findIndex(r=>r.sample===sample));
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
 if(!eps.length){for(const id of ['space','matrix','profile','timeline','lag'])Plotly.purge(id);$('anchor').replaceChildren();$('matches').replaceChildren();$('matchCards').replaceChildren();$('frameStats').textContent='';$('episodeStats').textContent='No frames match these filters';return;}
 if(!visible(selected))selected=localRows.findIndex((r,i)=>visible(i)&&r.episode===$('episode').value);
 renderSpace();setEpisode($('episode').value);
}
function setEpisode(ep){
 episode=ep;const e=panel.episodes[ep];epIds=e.ids;epTimes=e.times;epCos=e.cosine;
 if(localRows[selected]?.episode!==ep)selected=epIds[0];
 $('frame').max=epIds.length-1;
 const gaps=epIds.map(i=>panel.temporal.previous_dt_s[i]), r=localRows[epIds[0]];
 $('episodeStats').textContent=`${epIds.length} frames over ${fmt(epTimes[epTimes.length-1]-epTimes[0],1)} s, median gap ${fmt(median(gaps),2)} s · ${esc(r.task)}`;
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
function zRange(){
 if($('crange').value==='unit')return [-1,1];if($('crange').value==='positive')return [0,1];
 let lo=Infinity,hi=-Infinity;for(const row of epCos)for(const v of row){if(v==null)continue;if(v<lo)lo=v;if(v>hi)hi=v;}
 return lo<hi?[lo,hi]:[lo-1e-3,hi+1e-3];
}
function renderMatrix(){
 const [zmin,zmax]=zRange();
 Plotly.react('matrix',[{type:'heatmap',x:epTimes,y:epTimes,z:epCos,colorscale:'Viridis',zmin,zmax,hoverongaps:false,hovertemplate:'row %{y:.2f} s · col %{x:.2f} s<br>cos %{z:.4f}<extra></extra>',colorbar:{title:{text:'cos'},len:.9}}],
  {margin:{l:60,r:10,t:10,b:55},xaxis:{title:{text:'Episode time (s)'}},yaxis:{title:{text:'Episode time (s)'},scaleanchor:'x',autorange:'reversed'},shapes:[]},{responsive:true,displaylogo:false});
 $('matrix').removeAllListeners?.('plotly_click');$('matrix').on('plotly_click',event=>{const p=event.points[0];if(p&&Array.isArray(p.pointIndex))pick(epIds[p.pointIndex[0]]);});
}
function renderProfile(){
 const pos=epIds.indexOf(selected), t=epTimes[pos], row=epCos[pos], r=localRows[selected], m=temporal(selected);
 const span=[epTimes[0],epTimes[epTimes.length-1]], traces=[{type:'scatter',mode:'lines+markers',name:'cos(selected, frame)',x:epTimes,y:row,customdata:epIds,line:{color:'#202631'},marker:{size:4},connectgaps:false,hovertemplate:'%{x:.2f} s · %{y:.4f}<extra></extra>'}];
 const ref=(name,y,color,dash)=>{if(y!=null)traces.push({type:'scatter',mode:'lines',name,x:span,y:[y,y],line:{color,dash,width:1.5},hoverinfo:'name+y'});};
 ref('Background: other episodes, same robot',m.background_same_robot,'#949daa','dash');
 ref('Best match: same robot · same task',match(selected,kinds[0])?.cosine,colors[0],'dot');
 ref('Best match: same robot · different task',match(selected,kinds[1])?.cosine,colors[1],'dot');
 const [lc,n]=localCosine(pos,radius);$('radiusStats').textContent=`local cosine ${fmt(lc)} over ${n} neighbors · previous frame ${fmt(m.previous_cosine)} (${fmt(m.previous_dt_s,2)} s)`;
 Plotly.react('profile',traces,{margin:{l:55,r:10,t:10,b:55},xaxis:{title:{text:'Episode time (s)'}},yaxis:{title:{text:'Cosine'}},legend:{orientation:'h',font:{size:10},y:-.18},
  shapes:[{type:'rect',xref:'x',yref:'paper',x0:t-radius,x1:t+radius,y0:0,y1:1,fillcolor:'#3366cc',opacity:.08,line:{width:0}},{type:'line',xref:'x',yref:'paper',x0:t,x1:t,y0:0,y1:1,line:{color:'#777',width:1}}]},{responsive:true,displaylogo:false});
 $('profile').removeAllListeners?.('plotly_click');$('profile').on('plotly_click',event=>{const i=event.points[0]?.customdata;if(Number.isInteger(i))pick(i);});
}
function renderTimeline(){
 const ids=epIds, traces=[], t=epTimes, T=panel.temporal;
 const add=(name,values,yaxis,color,dash)=>traces.push({type:'scatter',mode:'lines+markers',name,x:t,y:values,customdata:ids,yaxis,line:{color,dash},marker:{size:4},connectgaps:false,hovertemplate:'%{x:.2f} s · %{y:.4f}<extra>%{fullData.name}</extra>'});
 const local=ids.map((_,pos)=>localCosine(pos,radius));
 add(`Local cosine ±${radius} s`,local.map(v=>v[0]),'y','#202631');
 add('Previous-frame cosine',ids.map(i=>T.previous_cosine[i]),'y','#949daa','dot');
 add('Background: other episodes, same robot',ids.map(i=>T.background_same_robot[i]),'y','#949daa','dash');
 kinds.forEach((kind,k)=>add(labels[k],ids.map(i=>panel.matches[kind].cosine[i]),'y',colors[k],k%2?'dash':'solid'));
 add('Raw norm',ids.map(i=>T.raw_norm[i]),'y2','#3366cc');
 if(panel.geometry!=='raw')add('Centered norm',ids.map(i=>T.geometry_norm[i]),'y2','#dd6633');
 add('Neighbors in radius',local.map(v=>v[1]),'y3','#119988');
 Plotly.react('timeline',traces,{margin:{l:58,r:15,t:12,b:85},uirevision:panel.key+episode,xaxis:{title:{text:'Episode time (seconds)'},anchor:'y3'},yaxis:{title:{text:'Cosine'},domain:[.47,1]},yaxis2:{title:{text:'Norm'},domain:[.22,.4],rangemode:'tozero'},yaxis3:{title:{text:'Neighbors'},domain:[0,.15],rangemode:'tozero'},legend:{orientation:'h',font:{size:10},y:-.2},shapes:[cursor()]},{responsive:true,displaylogo:false});
 $('timeline').removeAllListeners?.('plotly_click');$('timeline').on('plotly_click',event=>{const i=event.points[0]?.customdata;if(Number.isInteger(i))pick(i);});
}
function cursor(){return {type:'line',xref:'x',yref:'paper',x0:localRows[selected].time_s,x1:localRows[selected].time_s,y0:0,y1:1,line:{color:'#777',width:1}};}
function renderLag(){
 const edges=D.lag_edges_s, names=edges.map((lo,k)=>k+1<edges.length?`${lo}–${edges[k+1]}`:`${lo}+`), traces=[];
 const eps=uniq(localRows.filter((r,i)=>visible(i)).map(r=>r.episode));
 eps.forEach(ep=>{const e=panel.episodes[ep], r=localRows[e.ids[0]], sel=ep===episode;
  traces.push({type:'scatter',mode:'lines+markers',name:`${r.robot} · ${r.episode_label}`,x:names,y:e.lags.map(l=>l.mean_cosine),text:e.lags.map(l=>`${l.pairs} pairs`),connectgaps:false,hovertemplate:'%{x} s · %{y:.4f} · %{text}<extra>%{fullData.name}</extra>',line:{color:sel?'#202631':'#b8c0ce',width:sel?3:1},marker:{size:sel?6:3},opacity:sel?1:.6,showlegend:sel});});
 const ids=epIds, ref=(name,y,color,dash)=>{if(y!=null)traces.push({type:'scatter',mode:'lines',name,x:[names[0],names[names.length-1]],y:[y,y],line:{color,dash,width:1.5},hoverinfo:'name+y'});};
 ref('Background: other episodes, same robot (episode mean)',mean(ids.map(i=>panel.temporal.background_same_robot[i])),'#949daa','dash');
 ref('Best match: same robot · same task (episode mean)',mean(ids.map(i=>panel.matches[kinds[0]].cosine[i])),colors[0],'dot');
 ref('Best match: same robot · different task (episode mean)',mean(ids.map(i=>panel.matches[kinds[1]].cosine[i])),colors[1],'dot');
 Plotly.react('lag',traces,{margin:{l:55,r:10,t:10,b:60},xaxis:{title:{text:'Time gap between frames (s)'},type:'category'},yaxis:{title:{text:'Mean cosine'}},legend:{orientation:'h',font:{size:10},y:-.25}},{responsive:true,displaylogo:false});
}
function frameCard(i,title){
 const r=localRows[i];
 const cameras=['external_0','wrist_0'].map(cam=>{const src=thumb(r.sample,cam);return src?`<img src="${src}" alt="${cam}" loading="lazy">`:`<span>${cam}: unavailable</span>`;}).join('');
 return `<div class="card"><strong>${esc(title)}</strong><div class="details">${esc(r.robot)} · ${esc(r.episode_label)} <span class="tag">${r.holdout?'held out':'train'}</span><br>Frame ${r.frame} · ${fmt(r.time_s,2)} s</div><div class="cameras">${cameras}</div><div class="details"><b>Task:</b> ${esc(r.task)}<br><b>Step:</b> ${esc(r.subtask)}</div><button data-pick="${i}" style="margin-top:8px">Follow this frame</button></div>`;
}
function renderFrame(){
 const pos=epIds.indexOf(selected), m=temporal(selected), t=epTimes[pos];
 $('frame').value=pos;
 $('frameStats').textContent=`frame ${pos+1}/${epIds.length} · ${fmt(t,2)} s · raw norm ${fmt(m.raw_norm)} · step distance ${fmt(m.step_l2)}`;
 $('anchor').innerHTML=frameCard(selected,'Selected frame');
 Plotly.relayout('matrix',{shapes:[{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:t,y1:t,line:{color:'white',width:1}},{type:'line',xref:'x',yref:'paper',x0:t,x1:t,y0:0,y1:1,line:{color:'white',width:1}}]});
 Plotly.relayout('timeline',{shapes:[cursor()]});
 renderProfile();drawHighlight();
 $('matches').innerHTML=kinds.map((kind,k)=>{
  const mm=match(selected,kind);
  if(!mm)return `<tr><td>${labels[k]}</td><td colspan="7">No eligible frames</td></tr>`;
  const r=localRows[mm.target];
  return `<tr class="pick" data-pick="${mm.target}"><td>${labels[k]}</td><td>${fmt(mm.cosine)}</td><td>${fmt(mm.raw_norm_ratio)}</td><td>${fmt(mm.distance)}</td><td>${fmt(mm.target_window_cosine)}</td><td>${mm.target_window_count}</td><td>${mm.candidate_count} frames / ${mm.candidate_episodes} episodes</td><td>${esc(r.episode_label)} · ${fmt(r.time_s,2)} s</td></tr>`;
 }).join('');
 $('matchCards').innerHTML=kinds.map((kind,k)=>{const mm=match(selected,kind);return mm?frameCard(mm.target,labels[k]):'';}).join('');
 document.querySelectorAll('[data-pick]').forEach(el=>el.onclick=()=>pick(Number(el.dataset.pick),true));
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
$('crange').onchange=()=>{renderMatrix();renderFrame();};
$('episode').onchange=()=>setEpisode($('episode').value);
$('frame').oninput=()=>{if(panel)pick(epIds[Number($('frame').value)]);};
$('radius').oninput=()=>setRadius(Number($('radius').value));
$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('play').onclick=()=>{if(playing)return stopPlaying();if(!panel)return;$('play').textContent='⏸ Pause';playing=setInterval(()=>{const pos=epIds.indexOf(selected);if(pos>=epIds.length-1)stopPlaying();else step(1);},250);};
document.addEventListener('keydown',e=>{if(/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)&&e.target.id!=='frame')return;if(e.key==='ArrowLeft'){step(-1);e.preventDefault();}else if(e.key==='ArrowRight'){step(1);e.preventDefault();}else if(e.key===' '){$('play').click();e.preventDefault();}});
updatePanel();
</script></html>'''

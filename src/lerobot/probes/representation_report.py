"""A portable, linked Plotly inspector for cached representation trajectories."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from plotly.offline import get_plotlyjs


def write_report(payload, output_dir):
    root = Path(output_dir)
    images = {}
    for sample in sorted({r["sample"] for r in payload["rows"]}):
        images[str(sample)] = {}
        for camera in ("external_0", "wrist_0"):
            path = root / "cache" / "thumbs" / f"{sample:04d}.{camera}.jpg"
            if path.exists():
                images[str(sample)][camera] = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    data = json.dumps({**payload, "images": images}, allow_nan=False).replace("<", "\\u003c")
    html = _HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", data)
    (root / "trajectories.html").write_text(html)


_HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Representation trajectories</title>
<style>
*{box-sizing:border-box}body{margin:0;padding:20px;background:#f6f7f9;color:#202631;font:14px system-ui,sans-serif}
h1{font-size:23px;margin:0 0 8px}h2{font-size:17px;margin:14px 0 8px}p{line-height:1.5;margin:8px 0}.muted{color:#5e6878}
.controls{display:flex;flex-wrap:wrap;gap:12px;background:white;padding:12px;border-radius:8px;margin:14px 0}
label{display:flex;flex-direction:column;gap:5px;font-size:12px;color:#505b6e}select,button{font:inherit;border:1px solid #b8c0ce;border-radius:4px;padding:7px;background:white;color:#202631;max-width:420px}
button{cursor:pointer}button:hover{background:#e9f1ff}.grid{display:grid;grid-template-columns:minmax(400px,1fr) minmax(400px,1fr);gap:14px}.box{background:white;border:1px solid #e0e4eb;border-radius:8px;padding:12px;min-width:0}
#space{height:590px}#timeline{height:590px}.cards{display:flex;flex-wrap:wrap;gap:10px}.card{border:1px solid #d8dfeb;border-radius:6px;padding:10px;flex:1;min-width:270px;max-width:580px}.cameras{display:flex;gap:4px;margin:7px 0}.cameras img{width:49%;object-fit:contain;max-height:170px;background:#edf0f5}.cameras span{display:block;font-size:11px}.details{font-size:12px;overflow-wrap:anywhere}.tag{font-size:11px;background:#edf0f5;border-radius:3px;padding:2px 5px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:8px;border-bottom:1px solid #e1e6ee;text-align:left}tr.pick{cursor:pointer}tr.pick:hover{background:#edf5ff}.scroll{overflow:auto}.note{padding:9px 12px;background:#edf2f8;border-radius:5px}#status{white-space:pre-wrap}
@media(max-width:950px){.grid{grid-template-columns:1fr}body{padding:10px}#space,#timeline{height:480px}}
</style>
<script>__PLOTLY__</script>
<h1>Representation trajectories</h1>
<p class="muted">Follow an episode, then inspect where its full-dimensional representation resembles another task. Click a point in either plot to see frames and matches.</p>
<p class="note">3-D PCA is a projection. A visual crossing is a candidate for inspection, not evidence of confusion. Similarities and distances below use the original vectors. These are dataset replays at the first flow step, with fixed noise.</p>
<div class="controls">
<label>Token group<select id="group"></select></label><label>Layer<select id="layer"></select></label>
<label>Prompt<select id="text"><option value="real">Real task / subtask</option><option value="neutral">Constant task / subtask</option></select></label>
<label>Geometry<select id="geometry"><option value="raw">Raw vectors</option><option value="robot_centered">Subtract robot mean</option></select></label>
<label>Show robot<select id="robot"></select></label><label>Show task<select id="task"></select></label>
<label>Color markers<select id="color"><option value="time">Episode progress</option><option value="episode">Episode</option></select></label>
</div>
<p id="status" class="muted"></p>
<div class="grid"><section class="box"><h2>Shared 3-D PCA</h2><div id="space"></div></section>
<section class="box"><h2>Episode timeline</h2><label>Episode<select id="episode"></select></label><div id="timeline"></div></section></div>
<section class="box" style="margin-top:14px"><h2>Selected frame</h2><div style="display:flex;gap:8px;margin:8px 0"><button id="previous">← Previous frame</button><button id="next">Next frame →</button><span id="frameStats" class="muted"></span></div><div id="anchor" class="cards"></div>
<h2>Nearest frames in other episodes</h2><p class="muted">Search covers every captured episode in this group, layer and prompt, including episodes hidden by display filters. “Same task” uses exact recorded task text; different wording may describe the same behavior. Target-window cosine compares the anchor to frames within ±<span id="radius"></span> s of its best match, within that match's episode and task. A support count of 1 is just the matched frame.</p>
<div class="scroll"><table><thead><tr><th>Candidate set</th><th>Best cosine</th><th>Raw norm ratio<br>target / anchor</th><th>Distance</th><th>Target-window cosine</th><th>Support</th><th>Candidates</th><th>Episode / time</th></tr></thead><tbody id="matches"></tbody></table></div>
<div id="matchCards" class="cards" style="margin-top:12px"></div></section>
<p class="muted">Local cosine excludes the selected frame and stays inside its episode. Gaps mean no valid temporal neighbors or zero-length vectors; no zeros are substituted. Nearest-match curves may switch episodes at each time. Display filters preserve the fitted PCA axes. Axes are refitted when group, layer, text, geometry or checkpoint changes. Raw norm is always measured before external centering or unit normalization.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), uniq=a=>[...new Set(a)], esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(n,d=3)=>n==null?'—':Number(n).toFixed(d);
const kinds=['same_robot_same_task','same_robot_different_task','different_robot_same_task','different_robot_different_task'];
const labels=['Same robot · same task','Same robot · different task','Other robot · same task','Other robot · different task'];
const colors=['#3366cc','#dd6633','#119988','#9933aa','#cc4477','#667733','#0099c6','#b47700'];
let panel, selected=0, localRows=[], baseTraces=[];
function options(id,values,all=false){const old=$(id).value;$(id).replaceChildren();if(all)$(id).add(new Option('All',''));values.forEach(v=>$(id).add(new Option(v,v)));if(values.includes(old)||all&&old==='')$(id).value=old;}
function selectedPanel(){return D.panels.find(p=>p.group===$('group').value&&String(p.layer)===$('layer').value&&p.text===$('text').value&&p.geometry===$('geometry').value);}
function visible(i){const r=localRows[i];return (!$('robot').value||r.robot===$('robot').value)&&(!$('task').value||r.task===$('task').value);}
function episodeIndices(ep){return localRows.map((r,i)=>i).filter(i=>localRows[i].episode===ep).sort((a,b)=>localRows[a].time_s-localRows[b].time_s);}
function updatePanel(){
 const sample=localRows[selected]?.sample;
 let candidates=D.panels.filter(p=>p.group===$('group').value);
 options('layer',uniq(candidates.map(p=>String(p.layer))));
 candidates=candidates.filter(p=>String(p.layer)===$('layer').value);
 const allowed=uniq(candidates.map(p=>p.text));for(const o of $('text').options)o.disabled=!allowed.includes(o.value);
 if(!allowed.includes($('text').value))$('text').value=allowed[0];
 panel=selectedPanel();if(!panel)return;
 localRows=panel.ids.map(i=>D.rows[i]);selected=Math.max(0,localRows.findIndex(r=>r.sample===sample));
 options('robot',uniq(localRows.map(r=>r.robot)).sort(),true);options('task',uniq(localRows.map(r=>r.task)).sort(),true);
 const variance=panel.variance.map(v=>(100*v).toFixed(1)+'%').join(' + ');
 const center=panel.geometry==='raw'?'PCA centers globally; cosine uses the uncentered captured vectors.':Object.entries(panel.centering).map(([r,p])=>`${r}: ${p.fit}, n=${p.rows}`).join('; ');
 $('status').textContent=`Capture: ${D.view} · ${panel.group} · L${panel.layer}. PCA variance: ${variance} = ${(100*panel.variance.reduce((a,b)=>a+b,0)).toFixed(1)}%. Basis fit: ${panel.pca_fit}.\n${center}\nCheckpoint: ${D.checkpoint}`;
 updateFilter();
}
function updateFilter(){
 const eps=uniq(localRows.filter((r,i)=>visible(i)).map(r=>r.episode));
 const previous=localRows[selected]?.episode;
 $('episode').replaceChildren();eps.forEach(ep=>{const r=localRows.find(r=>r.episode===ep);$('episode').add(new Option(`${r.robot} · ${r.episode_label} · ${r.holdout?'held out':'train'}`,ep));});
 if(eps.includes(previous))$('episode').value=previous;
 if(!eps.length){Plotly.react('space',[],{title:'No frames match these filters'});Plotly.react('timeline',[],{title:'No episode selected'});$('anchor').replaceChildren();$('matches').replaceChildren();$('matchCards').replaceChildren();$('frameStats').textContent='';return;}
 if(!visible(selected))selected=localRows.findIndex((r,i)=>visible(i)&&r.episode===$('episode').value);
 renderSpace();renderTimeline();renderSelection();
}
function renderSpace(){
 const episodes=uniq(localRows.filter((r,i)=>visible(i)).map(r=>r.episode));baseTraces=[];
 episodes.forEach((ep,k)=>{
  const ids=episodeIndices(ep).filter(visible), times=ids.map(i=>localRows[i].time_s), lo=Math.min(...times),hi=Math.max(...times);
  const r=localRows[ids[0]], color=colors[k%colors.length];
  baseTraces.push({type:'scatter3d',mode:'lines+markers',name:`${r.robot} · ${r.episode_label}`,x:ids.map(i=>panel.xyz[i][0]),y:ids.map(i=>panel.xyz[i][1]),z:ids.map(i=>panel.xyz[i][2]),customdata:ids,
   text:ids.map(i=>`${esc(localRows[i].episode_label)} · ${fmt(localRows[i].time_s,2)} s<br>${esc(localRows[i].subtask)}`),hovertemplate:'%{text}<extra></extra>',
   line:{color,width:2},marker:{size:3,color:$('color').value==='time'?times.map(t=>(t-lo)/Math.max(hi-lo,1e-8)):color,colorscale:'Viridis',cmin:0,cmax:1,showscale:k===0&&$('color').value==='time',colorbar:{title:{text:'Progress'},len:.45}},opacity:.8});
 });
 drawSpace();
}
function drawSpace(){
 const points=[selected], names=['Selected frame'];
 if(panel.matches[selected])kinds.forEach((kind,k)=>{const m=panel.matches[selected][kind];if(m){points.push(m.target);names.push(labels[k]);}});
 const selection={type:'scatter3d',mode:'markers',name:'Selected / matched frames',x:points.map(i=>panel.xyz[i][0]),y:points.map(i=>panel.xyz[i][1]),z:points.map(i=>panel.xyz[i][2]),customdata:points,text:names,hovertemplate:'%{text}<extra></extra>',marker:{size:7,color:points.map((_,i)=>i?'#e37525':'#171e2b'),symbol:'diamond'}};
 const ranges=[0,1,2].map(axis=>{const a=panel.xyz.map(p=>p[axis]),lo=Math.min(...a),hi=Math.max(...a),pad=Math.max((hi-lo)*.05,1e-6);return [lo-pad,hi+pad];});
 Plotly.react('space',[...baseTraces,selection],{margin:{l:0,r:0,t:5,b:0},uirevision:panel.key,legend:{orientation:'h',font:{size:10},y:-.05},scene:{aspectmode:'data',xaxis:{title:{text:'PC1'},range:ranges[0]},yaxis:{title:{text:'PC2'},range:ranges[1]},zaxis:{title:{text:'PC3'},range:ranges[2]}}},{responsive:true,displaylogo:false});
}
function renderTimeline(){
 const ids=episodeIndices($('episode').value), traces=[], t=ids.map(i=>localRows[i].time_s);
 const add=(name,values,yaxis,color,dash)=>traces.push({type:'scatter',mode:'lines+markers',name,x:t,y:values,customdata:ids,yaxis,line:{color,dash},marker:{size:4},connectgaps:false,hovertemplate:'%{x:.2f} s · %{y:.4f}<extra>%{fullData.name}</extra>'});
 add('Local cosine',ids.map(i=>panel.temporal[i].local_cosine),'y','#202631');
 add('Previous-frame cosine',ids.map(i=>panel.temporal[i].previous_cosine),'y','#949daa','dot');
 kinds.forEach((kind,k)=>add(labels[k],ids.map(i=>panel.matches[i][kind]?.cosine??null),'y',colors[k],k%2?'dash':'solid'));
 add('Raw norm',ids.map(i=>panel.temporal[i].raw_norm),'y2','#3366cc');
 if(panel.geometry==='robot_centered')add('Centered norm',ids.map(i=>panel.temporal[i].geometry_norm),'y2','#dd6633');
 add('Temporal neighbors',ids.map(i=>panel.temporal[i].local_count),'y3','#119988');
 const at=localRows[selected].time_s;
 Plotly.react('timeline',traces,{margin:{l:58,r:15,t:12,b:85},uirevision:panel.key+$('episode').value,xaxis:{title:{text:'Episode time (seconds)'},anchor:'y3'},yaxis:{title:{text:'Cosine'},domain:[.47,1],range:[-1.05,1.05]},yaxis2:{title:{text:'Norm'},domain:[.22,.4],rangemode:'tozero'},yaxis3:{title:{text:'Neighbors'},domain:[0,.15],rangemode:'tozero'},legend:{orientation:'h',font:{size:10},y:-.2},shapes:[{type:'line',xref:'x',yref:'paper',x0:at,x1:at,y0:0,y1:1,line:{color:'#777',width:1}}]},{responsive:true,displaylogo:false});
}
function frameCard(i,title){
 const r=localRows[i], imgs=D.images[String(r.sample)]||{};
 const cameras=['external_0','wrist_0'].map(cam=>imgs[cam]?`<img src="${imgs[cam]}" alt="${cam}">`:`<span>${cam}: unavailable</span>`).join('');
 return `<div class="card"><strong>${esc(title)}</strong><div class="details">${esc(r.robot)} · ${esc(r.episode_label)} <span class="tag">${r.holdout?'held out':'train'}</span><br>Frame ${r.frame} · ${fmt(r.time_s,2)} s</div><div class="cameras">${cameras}</div><div class="details"><b>Task:</b> ${esc(r.task)}<br><b>Step:</b> ${esc(r.subtask)}</div><button data-pick="${i}" style="margin-top:8px">Follow this frame</button></div>`;
}
function renderSelection(){
 const m=panel.temporal[selected];
 $('frameStats').textContent=`Local cosine ${fmt(m.local_cosine)} (n=${m.local_count}) · raw norm ${fmt(m.raw_norm)} · previous gap ${fmt(m.previous_dt_s,2)} s · step distance ${fmt(m.step_l2)}`;
 $('anchor').innerHTML=frameCard(selected,'Anchor');
 $('matches').innerHTML=kinds.map((kind,k)=>{
  const match=panel.matches[selected][kind];
  if(!match)return `<tr><td>${labels[k]}</td><td colspan="7">No eligible frames</td></tr>`;
  const r=localRows[match.target];
  return `<tr class="pick" data-pick="${match.target}"><td>${labels[k]}</td><td>${fmt(match.cosine)}</td><td>${fmt(match.raw_norm_ratio)}</td><td>${fmt(match.distance)}</td><td>${fmt(match.target_window_cosine)}</td><td>${match.target_window_count}</td><td>${match.candidate_count} frames / ${match.candidate_episodes} episodes</td><td>${esc(r.episode_label)} · ${fmt(r.time_s,2)} s</td></tr>`;
 }).join('');
 $('matchCards').innerHTML=kinds.map((kind,k)=>{const m=panel.matches[selected][kind];return m?frameCard(m.target,labels[k]):'';}).join('');
 document.querySelectorAll('[data-pick]').forEach(el=>el.onclick=()=>pick(Number(el.dataset.pick),true));
}
function pick(i,reveal=false){
 selected=i;const r=localRows[i];
 if(reveal&&!visible(i)){$('robot').value='';$('task').value='';updateFilter();}
 $('episode').value=r.episode;drawSpace();renderTimeline();renderSelection();
}
options('group',uniq(D.panels.map(p=>p.group)));$('radius').textContent=D.window_s;
['group','layer','text','geometry'].forEach(id=>$(id).onchange=updatePanel);
['robot','task'].forEach(id=>$(id).onchange=updateFilter);$('color').onchange=renderSpace;
$('episode').onchange=()=>pick(episodeIndices($('episode').value)[0]);
['previous','next'].forEach((id,k)=>$(id).onclick=()=>{if(!panel||!$('episode').value)return;const ids=episodeIndices(localRows[selected].episode),pos=ids.indexOf(selected);pick(ids[Math.max(0,Math.min(ids.length-1,pos+(k?1:-1)))]);});
updatePanel();
$('space').on('plotly_click',event=>{const i=event.points[0]?.customdata;if(Number.isInteger(i))pick(i,true);});
$('timeline').on('plotly_click',event=>{const i=event.points[0]?.customdata;if(Number.isInteger(i))pick(i);});
</script></html>'''

/* Probe comparison shell: one chip row per axis of the probe's outputs, parsed from panel keys.
   Probes the suite no longer reads (HIDDEN) stay out of the rail; their output is still served. */
(function(){
'use strict';
const L=window.ProbeCompare,h=L.h;
const natural=(a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true});
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const HIDDEN=/spatial_memorization|jacobian|mem_temporal_attention|critic_value|value_distribution|future_visual_cases/i;
const CAMERA={img_external_0:'external_0',img_external_1:'external_1',img_wrist_0:'wrist_0',depth:'depth'};

/* One rule per probe id. parse(key) -> {facets:{Axis:value}} for a faceted output,
   {facets, extra:file} for a per-cell side file, or null for a flat card.
   order: fixed chip order per axis (natural sort otherwise). axes: row order. wall: axis fanned out on the compare stage. */
const RULES={
 attention:{
  parse(key){const m=key.match(/^ep(\d+)_L(\d+)\/(.+)$/);if(!m)return null;const f={Episode:String(+m[1]),Layer:String(+m[2])};
   const v=m[3].match(/^overlay_(img_external_0|img_external_1|img_wrist_0|depth)_(summary|heads)\.mp4$/);
   if(v){f.Camera=CAMERA[v[1]];f.View=v[2]==='summary'?'mean over heads':'per head';return {facets:f};}
   return {facets:f,extra:m[3]};},
  order:{Camera:['external_0','external_1','wrist_0','depth'],View:['mean over heads','per head']},
  extraLabel:f=>`Also for episode ${f.Episode}, layer ${f.Layer}`,
  wall:{axis:'Camera',columns:2}
 },
 sim:{
  parse(key){const m=key.match(/^([^/]+)\/videos\/ep(\d+)_(top|wrist|wrist_depth)\.mp4$/);if(!m)return null;return {facets:{Scene:m[1],'Sim episode':m[2],Camera:m[3]}};},
  order:{Camera:['top','wrist','wrist_depth']},
  wall:{axis:'Camera',columns:3}
 },
 sim_nail:{
  parse(key){const m=key.match(/^([^/]+)\/videos\/ep(\d+)_(top|wrist|wrist_depth)\.mp4$/);if(!m)return null;return {facets:{Scene:m[1],'Sim episode':m[2],Camera:m[3]}};},
  order:{Camera:['top','wrist','wrist_depth']},
  wall:{axis:'Camera',columns:3}
 },
 input_swap:{
  family:'Examples',
  parse(key){const m=key.match(/^examples\/(same_episode|matched|random)_(p\d+)\.png$/);if(!m)return null;return {facets:{Kind:m[1],Band:m[2]}};},
  order:{Kind:['same_episode','matched','random']}
 },
 conditions_matrix:{
  parse(key){let m;
   if((m=key.match(/^matrices(_wrist_state)?(?:_L(\d+))?\.png$/)))return {facets:{Group:m[1]?'wrist+state':'class',Layer:m[2]?'L'+m[2]:'headline'}};
   if(/^rebot_instances\.png$/.test(key))return {facets:{Group:'instances'}};
   if(/^decoding\.png$/.test(key))return {facets:{Group:'decoding'}};
   if((m=key.match(/^frames_(\w+)\.png$/)))return {facets:{Group:'frames',Robot:m[1]}};
   return null;},
  axes:['Group','Layer','Robot'],
  order:{Group:['class','wrist+state','instances','decoding','frames'],Layer:['headline','L14','L32']}
 }
};
// Generic fallback: anything in a sub-directory becomes Folder x File; top-level files stay flat cards.
const GENERIC={parse(key){const i=key.lastIndexOf('/');if(i<0)return null;return {facets:{Folder:key.slice(0,i),File:key.slice(i+1)}};}};

const facetMemory={};   // probe id -> last facets seen, so Camera/View survive a detour to action_to_prompt.png
const hashParam=k=>new URLSearchParams(location.hash.slice(1)).get(k);
const setHashParam=(k,v)=>{const q=new URLSearchParams(location.hash.slice(1));q.set(k,v);history.replaceState(null,'','#'+q);};
let wall=false;   // read from the hash on every render; the hash is the single source of truth
let railFilter='';

function layout(){
 const rule=RULES[L.probeId]||GENERIC,panels=L.panels();
 const items=panels.map(p=>Object.assign({},p,rule.parse(p.key)||{facets:null}));
 const cards=items.filter(p=>!p.facets),faceted=items.filter(p=>p.facets&&!p.extra),extras=items.filter(p=>p.extra);
 const axes=[];faceted.forEach(p=>Object.keys(p.facets).forEach(a=>{if(!axes.includes(a))axes.push(a);}));
 if(rule.axes)axes.sort((a,b)=>{const i=rule.axes.indexOf(a),j=rule.axes.indexOf(b);return (i<0?99:i)-(j<0?99:j);});
 const values={};
 axes.forEach(a=>{const set=new Set();faceted.forEach(p=>{if(p.facets[a]!==undefined)set.add(p.facets[a]);});const order=rule.order?.[a];const idx=v=>{const i=order?order.indexOf(v):-1;return i<0?999:i;};values[a]=[...set].sort((x,y)=>idx(x)-idx(y)||natural(x,y));});
 const current=items.find(p=>p.key===L.panelKey)||null;
 if(current?.facets)facetMemory[L.probeId]=Object.assign({},facetMemory[L.probeId],current.facets);
 const S=Object.assign({},facetMemory[L.probeId],current?.facets);
 return {rule,items,cards,faceted,extras,axes,values,current,S};
}
const agrees=(lay,p,S,skip)=>lay.axes.every(a=>a===skip||p.facets[a]===undefined||S[a]===undefined||p.facets[a]===S[a]);
function exists(lay,axis,value){return lay.faceted.some(p=>p.facets[axis]===value&&agrees(lay,p,lay.S,axis));}
function retarget(lay,axis,value){
 const cand=lay.faceted.filter(p=>p.facets[axis]===value);if(!cand.length)return;
 const score=p=>lay.axes.reduce((s,a)=>a===axis||p.facets[a]===undefined||lay.S[a]===undefined?s:s+(p.facets[a]===lay.S[a]?1:-1),0)+(p.primary?.01:0);
 cand.sort((a,b)=>score(b)-score(a)||natural(a.key,b.key));
 L.setPanel(cand[0].key);
}
function wallKeys(lay){
 const w=lay.rule.wall;if(!w||!lay.values[w.axis])return [];
 return lay.values[w.axis].map(v=>({value:v,key:lay.faceted.find(p=>p.facets[w.axis]===v&&agrees(lay,p,lay.S,w.axis))?.key||null}));
}

/* ---------- rail ---------- */
function visibleProbes(){return L.probes().filter(p=>!HIDDEN.test(p.id));}
function runsWithOutput(id){return L.data.runs.filter((r,i)=>(L.probe(i,id)?.panels[L.step(i)]||[]).length).length;}
function renderRail(){
 const probes=visibleProbes(),q=railFilter.trim().toLowerCase();
 document.getElementById('probe-count').textContent=probes.length;
 document.getElementById('probe-list').innerHTML=probes.map((p,i)=>{
  const hide=q&&!(p.title+' '+p.group+' '+p.id).toLowerCase().includes(q);
  return `<button class="probe-item${p.id===L.probeId?' active':''}" data-probe="${h(p.id)}"${hide?' hidden':''}><span>${h(p.group)}<small>${runsWithOutput(p.id)}/${L.data.runs.length} runs</small></span><strong><i class="probe-number">${String(i+1).padStart(2,'0')}</i>${h(p.title)}</strong></button>`;
 }).join('');
 document.querySelectorAll('#probe-list [data-probe]').forEach(b=>b.onclick=()=>L.setProbe(b.dataset.probe));
}

/* ---------- title ---------- */
function renderTitle(lay){
 const probes=visibleProbes(),row=probes.find(p=>p.id===L.probeId),n=probes.findIndex(p=>p.id===L.probeId)+1;
 const total=L.panels(L.probeId,true).length,steps=new Set(L.data.runs.flatMap(r=>r.steps.map(s=>s.step))).size;
 document.getElementById('eyebrow').innerHTML=`<span>PROBE ${String(n).padStart(2,'0')} · ${h((row?.group||'').toUpperCase())}</span><i class="eyebrow-rule"></i><span>${L.data.runs.length} RUNS / ${steps} CHECKPOINTS</span><i class="eyebrow-rule"></i><span>${total} OUTPUT${total===1?'':'S'} AT THESE CHECKPOINTS</span>`;
 document.getElementById('title').textContent=row?.title||L.probeId||'No probe';
 document.getElementById('claim').textContent=row?.claim||'';
 const shape=lay.axes.length?(lay.cards.length?'overview + examples':'faceted media'):'flat';
 document.getElementById('title-side').innerHTML=`<div class="shape-tag"><small>SHAPE</small><strong>${shape}</strong><span>${lay.axes.length?lay.axes.map(a=>`${h(a)} ${lay.values[a].length}`).join(' × '):`${lay.cards.length} figure${lay.cards.length===1?'':'s'}`}</span></div>`;
}

/* ---------- axis rails ---------- */
const KIND_ICON={image:'IMG',video:'VID',html:'HTML',data:'DATA'};
function cardHTML(p,compact){
 return `<button class="card${p.key===L.panelKey?' active':''}" data-panel="${h(p.key)}" title="${h(p.caption)}"><small><b class="kind kind-${h(p.kind)}">${KIND_ICON[p.kind]||'FILE'}</b> ${p.count}/${L.data.runs.length} runs</small><span>${h(p.caption)}</span><code>${h(p.key)}</code></button>`;
}
function chipRow(label,chips,note){
 return `<div class="axis-row"><div class="axis-label">${h(label)}${note?`<em>${h(note)}</em>`:''}</div><div class="chips">${chips}</div></div>`;
}
function renderRails(lay){
 const host=document.getElementById('rails');
 if(!lay.items.length){host.innerHTML=`<div class="rails-head"><div class="strip-label">OUTPUTS<span>0</span></div><p class="rails-empty">${L.shared?'No output is present in every run at these checkpoints. Switch to All outputs.':'Nothing recorded at these checkpoints.'}</p></div>`;return;}
 const sections=[];
 if(lay.cards.length)sections.push(chipRow('Figures',`<div class="cards">${lay.cards.map(p=>cardHTML(p)).join('')}</div>`));
 if(lay.axes.length){
  const rows=lay.axes.map(a=>chipRow(a,lay.values[a].map(v=>{const on=lay.S[a]===v&&lay.current?.facets?.[a]===v,ok=exists(lay,a,v);return `<button class="chip${on?' active':''}${ok?'':' dim'}" data-axis="${h(a)}" data-value="${h(v)}" aria-pressed="${on}" title="${ok?h(a)+' '+h(v):'No output with '+h(a)+' '+h(v)+' at the current selection; picks the nearest one'}">${h(v)}</button>`;}).join(''),lay.values[a].length>1?String(lay.values[a].length):''));
  sections.push(lay.rule.family?`<div class="family"><div class="family-title">${h(lay.rule.family)} <small>${lay.faceted.length} panels</small></div>${rows.join('')}</div>`:rows.join(''));
 }
 if(lay.extras.length){
  const mine=lay.extras.filter(p=>agrees(lay,p,lay.S,null));
  if(mine.length)sections.push(chipRow(lay.rule.extraLabel?lay.rule.extraLabel(lay.S):'Also',mine.map(p=>`<button class="chip extra${p.key===L.panelKey?' active':''}" data-panel="${h(p.key)}" title="${h(p.caption)}"><b class="kind kind-${h(p.kind)}">${KIND_ICON[p.kind]||'FILE'}</b>${h(p.extra)}</button>`).join('')));
 }
 const cur=lay.current;
 host.innerHTML=`<div class="rails-head"><div class="strip-label">OUTPUTS<span>${lay.items.length}</span></div><div class="current"><small>SHOWING · ${h((cur?.kind||'').toUpperCase())} · ${cur?.count||0}/${L.data.runs.length} RUNS</small><strong>${h(cur?.caption||'')}</strong><code>${h(cur?.key||'')}</code></div></div><div class="rails-body">${sections.join('')}</div>`;
 host.querySelectorAll('[data-panel]').forEach(b=>b.onclick=()=>L.setPanel(b.dataset.panel));
 host.querySelectorAll('[data-axis]').forEach(b=>b.onclick=()=>retarget(lay,b.dataset.axis,b.dataset.value));
 const active=host.querySelector('.card.active');if(active)active.scrollIntoView({block:'nearest',inline:'nearest'});
}

/* ---------- runs ---------- */
function sparkline(i,metric){
 const probe=L.data.runs[i].probes.find(p=>p.id===L.probeId),m=probe?.metrics.find(x=>x.key===metric?.key);
 const steps=L.data.runs[i].steps.map(s=>s.step),pts=steps.map(s=>[s,m?.values?.[String(s)]]).filter(([,v])=>finite(v));
 if(pts.length<2)return '<svg class="run-trace" viewBox="0 0 90 36"><path class="trace-floor" d="M2 30H88"/></svg>';
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const X=s=>4+(s-x0)/(x1-x0||1)*82,Y=v=>30-(v-y0)/(y1-y0||1)*24;
 const d=pts.map(([s,v],k)=>`${k?'L':'M'}${X(s).toFixed(1)} ${Y(v).toFixed(1)}`).join(' ');
 const sel=pts.find(([s])=>String(s)===L.step(i));
 return `<svg class="run-trace" viewBox="0 0 90 36" aria-label="${h(metric?.label||'')} through training"><path class="trace-floor" d="M2 33H88"/><path d="${d}"/>${sel?`<circle cx="${X(sel[0]).toFixed(1)}" cy="${Y(sel[1]).toFixed(1)}" r="2.6"/>`:''}</svg>`;
}
function renderRuns(){
 const host=document.getElementById('runs'),metric=L.metrics().find(m=>m.primary)||L.metrics()[0];
 const common=L.data.runs.map(r=>r.steps.map(s=>s.step)).reduce((a,b)=>a.filter(s=>b.includes(s)));
 const probeAt=(i,s)=>L.data.runs[i].probes.some(p=>p.id===L.probeId&&p.panels[String(s)]!==undefined);
 host.innerHTML=`<div class="common-lane"><span>ALL RUNS AT</span><div>${common.map(s=>`<button data-all-step="${s}" class="${L.steps.every(x=>x===s)?'active':''}">${s.toLocaleString()}</button>`).join('')}</div><em>${metric?`trace: ${h(metric.label)}`:'no scalar metric for this probe'}</em></div>`+
 `<div class="run-cards">${L.data.runs.map((r,i)=>`<div class="run-card" data-run="${i}"><div class="run-top"><span class="run-marker">${i+1}</span><div class="run-name"><small>RUN ${String(i+1).padStart(2,'0')}</small><strong>${h(r.run.label)}</strong></div>${sparkline(i,metric)}<div class="selected-step"><small>CHECKPOINT</small>${Number(L.step(i)).toLocaleString()}</div></div><div class="step-buttons">${r.steps.map(s=>`<button data-run-step="${i}:${s.step}" class="${String(s.step)===L.step(i)?'active':''}${probeAt(i,s.step)?'':' absent'}" title="${probeAt(i,s.step)?'':'Probe not recorded at this step'}">${s.step.toLocaleString()}</button>`).join('')}<button class="baseline${i===L.baseline?' active':''}" data-baseline="${i}">${i===L.baseline?'● Baseline':'Set baseline'}</button></div></div>`).join('')}</div>`;
 host.querySelectorAll('[data-all-step]').forEach(b=>b.onclick=()=>L.setSteps(L.data.runs.map(()=>Number(b.dataset.allStep))));
 host.querySelectorAll('[data-run-step]').forEach(b=>b.onclick=()=>{const [i,s]=b.dataset.runStep.split(':');L.setStep(Number(i),Number(s));});
 host.querySelectorAll('[data-baseline]').forEach(b=>b.onclick=()=>L.setBaseline(Number(b.dataset.baseline)));
}

/* ---------- stage ---------- */
function renderTools(lay){
 const host=document.getElementById('stage-tools'),canWall=!!lay.rule.wall&&lay.axes.length>0;
 if(!canWall)wall=false;
 host.innerHTML=`<div class="stage-title"><strong>Compare</strong><span>${L.data.runs.length} runs · ${wall?'wall: every '+h(lay.rule.wall.axis.toLowerCase())+' of the selected cell':'one output per run'}</span></div><div class="segmented"><button data-shared="0" class="${L.shared?'':'active'}">All outputs</button><button data-shared="1" class="${L.shared?'active':''}">Shared outputs</button></div>${canWall?`<div class="segmented"><button data-wall="0" class="${wall?'':'active'}">Single</button><button data-wall="1" class="${wall?'active':''}">Wall · ${h(lay.rule.wall.axis)}</button></div>`:''}<span class="lock-help">${L.shared?'Only outputs present in every run.':'Outputs missing from a run show as not recorded.'}</span>`;
 host.querySelectorAll('[data-shared]').forEach(b=>b.onclick=()=>L.setShared(b.dataset.shared==='1'));
 host.querySelectorAll('[data-wall]').forEach(b=>b.onclick=()=>{setHashParam('wall',b.dataset.wall);L.emit();});
}
function fmt(t){if(!finite(t))return '—';const n=Math.round(Math.max(0,t)*10);return `${Math.floor(n/600)}:${((n%600)/10).toFixed(1).padStart(4,'0')}`;}
function wallSync(section){
 const videos=[...section.querySelectorAll('video')],bar=section.querySelector('.wall-transport');
 if(!videos.length){bar.hidden=true;return;}
 const seeking=new WeakSet(),playing=new WeakSet(),pausing=new WeakSet();
 const move=(v,t)=>{if(finite(v.duration)){const x=Math.max(0,Math.min(v.duration,t));if(Math.abs(v.currentTime-x)>.06){seeking.add(v);v.currentTime=x;}}};
 const play=v=>{if(v.paused){playing.add(v);v.play().catch(()=>playing.delete(v));}};
 const pause=v=>{if(!v.paused){pausing.add(v);v.pause();}};
 const others=v=>videos.filter(x=>x!==v);
 const lead=()=>videos.find(v=>!v.paused)||videos[0];
 const update=()=>{const v=lead(),d=Math.max(...videos.map(x=>finite(x.duration)?x.duration:0)),range=bar.querySelector('input');range.max=d||1;if(document.activeElement!==range)range.value=v.currentTime;bar.querySelector('output').textContent=`${fmt(v.currentTime)} / ${fmt(d||NaN)}`;bar.querySelector('[data-w-play]').textContent=videos.some(x=>!x.paused&&!x.ended)?'Ⅱ Pause':'▶ Play';};
 videos.forEach(v=>{
  v.addEventListener('play',()=>{if(!playing.delete(v))others(v).forEach(x=>{move(x,v.currentTime);play(x);});update();});
  v.addEventListener('pause',()=>{if(!pausing.delete(v)&&!v.ended)others(v).forEach(pause);update();});
  v.addEventListener('seeked',()=>{if(!seeking.delete(v))others(v).forEach(x=>move(x,v.currentTime));update();});
  v.addEventListener('ratechange',()=>others(v).forEach(x=>{if(x.playbackRate!==v.playbackRate)x.playbackRate=v.playbackRate;}));
  v.addEventListener('timeupdate',()=>{if(v===lead())update();});
  v.addEventListener('loadedmetadata',update);
 });
 bar.querySelector('[data-w-play]').onclick=()=>{if(videos.some(v=>!v.paused&&!v.ended))videos.forEach(pause);else{const t=lead().ended?0:lead().currentTime;videos.forEach(v=>{move(v,t);play(v);});}update();};
 bar.querySelector('input').oninput=e=>{videos.forEach(v=>move(v,Number(e.target.value)));update();};
 bar.querySelector('[data-w-back]').onclick=()=>{const t=lead().currentTime-1;videos.forEach(v=>move(v,t));update();};
 bar.querySelector('[data-w-next]').onclick=()=>{const t=lead().currentTime+1;videos.forEach(v=>move(v,t));update();};
 bar.querySelectorAll('[data-w-rate]').forEach(b=>b.onclick=()=>{videos.forEach(v=>v.playbackRate=Number(b.dataset.wRate));bar.querySelectorAll('[data-w-rate]').forEach(x=>x.setAttribute('aria-pressed',x===b));});
 section.wallPlayback={play:()=>videos.forEach(play),pause:()=>videos.forEach(pause),seek:t=>videos.forEach(v=>move(v,t))};
 update();
}
function renderStage(lay){
 const host=document.getElementById('stage');host.className='stage';delete host.dataset.probe;delete host.dataset.panel;
 const anyPanels=L.data.runs.some((r,i)=>(L.probe(i)?.panels[L.step(i)]||[]).length);
 if(!anyPanels&&/sim/i.test(L.probeId||'')){
  host.innerHTML=`<div class="rerun-note"><strong>Rerun recordings only at these checkpoints</strong><p>This probe writes .rrd logs, not browser media. Open them in rerun from the run directories below; exported MP4s appear here once a run writes them.</p><ul>${L.data.runs.map((r,i)=>`<li><span class="run-marker" data-run="${i}">${i+1}</span><code>${h(r.run.val_dir)}/step_${String(L.step(i)).padStart(8,'0')}/${h(L.probeId)}/</code></li>`).join('')}</ul></div>`;return;
 }
 if(!wall||!lay.rule.wall){L.evidence(host,{});return;}
 const keys=wallKeys(lay).filter(k=>k.key);
 host.classList.add('wall');
 host.innerHTML=`<div class="wall-all"><button data-wall-all="play">▶ Play every run</button><button data-wall-all="pause">Ⅱ Pause every run</button><small>Videos are synced within a run; runs advance independently.</small></div>`+L.data.runs.map((r,i)=>{
  const present=keys.filter(k=>L.panel(i,k.key));
  return `<section class="wall-run" data-run="${i}"><header class="wall-head"><span class="run-marker">${i+1}</span><strong>${h(r.run.label)}</strong><span class="wall-step">step ${L.step(i)}${i===L.baseline?' · baseline':''}</span><div class="wall-transport"${present.length?'':' hidden'}><button data-w-play>▶ Play</button><button data-w-back>−1s</button><button data-w-next>+1s</button><input type="range" min="0" max="1" step="0.05" value="0" aria-label="Seek this run"><output>—</output><span class="evidence-speeds">${[.5,1,2].map(x=>`<button data-w-rate="${x}" aria-pressed="${x===1}">${x}×</button>`).join('')}</span></div></header>${present.length?`<div class="wall-tiles" style="--cols:${lay.rule.wall.columns}">${keys.map(k=>`<div class="wall-tile"><div class="wall-tile-label">${h(lay.rule.wall.axis)} · ${h(k.value)}</div><div class="wall-ev" data-key="${h(k.key)}"></div></div>`).join('')}</div>`:`<div class="evidence-empty">Not recorded for this run at step ${L.step(i)}.</div>`}</section>`;
 }).join('');
 host.querySelectorAll('.wall-run').forEach(section=>{const i=Number(section.dataset.run);section.querySelectorAll('.wall-ev').forEach(ev=>L.evidence(ev,{probe:L.probeId,key:ev.dataset.key,runs:[i]}));wallSync(section);});
 host.querySelectorAll('[data-wall-all]').forEach(b=>b.onclick=()=>host.querySelectorAll('.wall-run').forEach(s=>s.wallPlayback?.[b.dataset.wallAll]()));
}

/* ---------- radar (Objective) ----------
   One polygon per run at its selected checkpoint. Each axis is min-max scaled over the compared
   runs and flipped so outward is always good; a run missing a value leaves a gap, not a zero.
   Values come from the objective index.json metrics and, when a sim index exists at the same
   step, the sim reward metrics. Axes nobody has are omitted and listed under the chart. */
const RADAR_AXES=[
 {probe:'objective',key:'loss_flow_joint.val',label:'flow loss (val)',good:'low'},
 {probe:'objective',key:'loss_discrete_ce.val',label:'FAST CE (val)',good:'low'},
 {probe:'objective',key:'loss_discrete_ce.train',label:'FAST CE (train)',good:'low'},
 {probe:'objective',key:'offset.val',label:'offset MSE (val)',good:'low'},
 {probe:'objective',key:'shape.val',label:'shape MSE (val)',good:'low'},
 {probe:'objective',key:'collapse.val',label:'spread (collapse)',good:'high'},
 {probe:'objective',key:'depth_event_reference.close.val.captured',label:'close captured',good:'high'},
 {probe:'objective',key:'depth_event_reference.open.val.captured',label:'open captured',good:'high'},
 {probe:'sim',key:'reward.easy',label:'sim reward (easy)',good:'high'},
 {probe:'sim',key:'reward.hard',label:'sim reward (hard)',good:'high'}
];
const RADAR_WINDOWS=[1.25,1.5,2,4];
const RADAR_MAX=8;
const radarWindow=()=>RADAR_WINDOWS.includes(Number(hashParam('radar_r')))?Number(hashParam('radar_r')):2;
const RADAR_DIFF_AXES=/captured$|^reward\./;
// Within one run the polygons of different steps share the run's colour family. The dash is the
// primary step cue (rank 0 solid, 1 dashed, 2 dotted, 3 dash-dot, ...), the marker shape the
// secondary one, and the lightness steps toward white with the distance from the selected step,
// which is drawn at full colour. Rank = the step's position among that run's toggled steps.
const MARKERS=[
 (x,y,r)=>`<circle cx="${x}" cy="${y}" r="${r}"/>`,
 (x,y,r)=>`<rect x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}"/>`,
 (x,y,r)=>`<path d="M${x} ${y-r*1.3}L${x+r*1.3} ${y}L${x} ${y+r*1.3}L${x-r*1.3} ${y}Z"/>`,
 (x,y,r)=>`<path d="M${x} ${y-r*1.3}L${x+r*1.25} ${y+r}L${x-r*1.25} ${y}Z"/>`
];
const DASHES=['','8 4','2 4','8 4 2 4','12 4 2 4 2 4','4 4'];
const polyColor=p=>p.selected?'var(--run-accent)':`color-mix(in srgb,var(--run-accent) ${Math.max(40,100-22*p.distance)}%,#fff)`;
const swatch=(p,cls='')=>`<svg class="radar-swatch ${cls}" viewBox="0 0 30 14" aria-hidden="true" style="color:${polyColor(p)}"><path class="radar-line" d="M1 7H29"${p.dash?` stroke-dasharray="${p.dash}"`:''}/><g class="radar-mark">${p.marker(15,7,4)}</g></svg>`;
const polyKey=p=>`${p.run}:${p.step}`;
const hashList=k=>(hashParam(k)||'').split(',').filter(Boolean);
// A step counts once its objective index is written (panels present), not while the probe dir is still empty.
const objectiveAt=(i,s)=>L.data.runs[i]?.probes.some(p=>p.id==='objective'&&p.panels[String(s)]?.length>0);
const objectiveValue=(i,s,probe,key)=>{const m=L.data.runs[i].probes.find(p=>p.id===probe)?.metrics.find(x=>x.key===key);const v=m?.values?.[String(s)];return {value:finite(v)?v:null,fmt:m?.fmt??4};};
/* The toggled polygons: each run's selected step unless hidden (radar_hide), plus the
   extras the user turned on (radar_show); ordered by run then step, at most RADAR_MAX. */
function radarPolygons(){
 const hide=new Set(hashList('radar_hide')),set=new Map();
 L.data.runs.forEach((_,i)=>{const s=Number(L.step(i));if(objectiveAt(i,s)&&!hide.has(`${i}:${s}`))set.set(`${i}:${s}`,{run:i,step:s});});
 hashList('radar_show').forEach(k=>{const [i,s]=k.split(':').map(Number);if(objectiveAt(i,s))set.set(k,{run:i,step:s});});
 const polys=[...set.values()].sort((a,b)=>a.run-b.run||a.step-b.step).slice(0,RADAR_MAX);
 polys.forEach(p=>{
  const own=polys.filter(q=>q.run===p.run),selected=Number(L.step(p.run));
  p.rank=own.indexOf(p);p.key=polyKey(p);p.selected=p.step===selected;
  p.distance=Math.abs(p.rank-(own.findIndex(q=>q.step===selected)<0?own.length-1:own.findIndex(q=>q.step===selected)));
  p.dash=DASHES[p.rank%DASHES.length];p.marker=MARKERS[p.rank%MARKERS.length];
 });
 return polys;
}
function toggleRadarPolygon(i,s){
 const key=`${i}:${s}`,isDefault=Number(L.step(i))===s,show=new Set(hashList('radar_show')),hide=new Set(hashList('radar_hide'));
 const on=radarPolygons().some(p=>p.key===key);
 if(on){show.delete(key);if(isDefault)hide.add(key);}else{hide.delete(key);if(!isDefault)show.add(key);}
 setHashParam('radar_show',[...show].join(','));setHashParam('radar_hide',[...hide].join(','));renderRadar();
}
/* Baseline-relative geometry. Ratio axes (losses, offset, shape, collapse):
   t = s * log2(value / baseline) / log2(R), s = -1 lower-is-better, +1 for collapse.
   Difference axes (captured fractions, sim rewards, all in [-1, 1]):
   t = (value - baseline) / D with D = 0.25 * log2(R), so R = 2 gives D = 0.25.
   r = 0.5 + 0.5 * clamp(t, -1, 1): the baseline (its run at its selected step) sits on the
   0.5 ring, the rim is an R-fold (or +D) improvement, the centre an R-fold (or -D)
   degradation; beyond that the point is clamped. */
function radarRows(polys,baseline){
 const R=radarWindow(),D=0.25*Math.log2(R);
 return RADAR_AXES.map(a=>{
  const diff=RADAR_DIFF_AXES.test(a.key),s=a.good==='low'?-1:1;
  const cells=polys.map(p=>objectiveValue(p.run,p.step,a.probe,a.key)),values=cells.map(c=>c.value);
  const b=objectiveValue(baseline.run,baseline.step,a.probe,a.key).value,fmt=cells.find(c=>c.value!==null)?.fmt??4;
  const rel=values.map(v=>v===null||b===null?null:diff?v-b:(v>0&&b>0?v/b:null));
  const t=rel.map(x=>x===null?null:diff?x/D:s*Math.log2(x)/Math.log2(R));
  const scaled=t.map(x=>x===null?null:0.5+0.5*Math.max(-1,Math.min(1,x)));
  const clamped=t.map(x=>x!==null&&Math.abs(x)>1);
  const rim=b===null?null:diff?b+D:s<0?b/R:b*R,centre=b===null?null:diff?b-D:s<0?b*R:b/R;
  return Object.assign({},a,{values,rel,scaled,clamped,diff,fmt,present:values.filter(v=>v!==null).length,baseline:b,rim,centre,R,D});
 });
}
const relText=(a,x)=>x===null?'—':a.diff?(x>0?'+':'')+x.toFixed(3)+' vs baseline':x.toFixed(2)+'x baseline';
function renderRadar(){
 const host=document.getElementById('radar');
 if(L.probeId!=='objective'){host.hidden=true;host.innerHTML='';return;}
 host.hidden=false;
 const runs=L.data.runs,base=L.baseline,baseline={run:base,step:Number(L.step(base))},polys=radarPolygons(),full=polys.length>=RADAR_MAX;
 const chips=runs.map((r,i)=>{
  const steps=r.steps.map(s=>s.step).filter(s=>objectiveAt(i,s));
  return `<div class="radar-chip-run" data-run="${i}"><span class="run-marker">${i+1}</span><strong>${h(r.run.label)}</strong><div class="chips">${steps.map(s=>{const p=polys.find(q=>q.run===i&&q.step===s),isBase=i===base&&s===baseline.step;return `<button class="chip${p?' active':''}" data-radar-toggle="${i}:${s}" aria-pressed="${!!p}"${!p&&full?' disabled':''} title="${isBase?'baseline polygon (the 0.5 ring)':p?'remove this polygon':full?'at most '+RADAR_MAX+' polygons':'add this polygon'}">${p?swatch(p):''}${s.toLocaleString()}${isBase?' ·':''}${s===Number(L.step(i))?'<small>sel</small>':''}</button>`;}).join('')}</div></div>`;
 }).join('');
 const chipRow=`<div class="radar-chips"><span class="axis-label">Polygons<em>${polys.length} of ${RADAR_MAX}</em></span>${chips}${full?`<small class="radar-cap">Cap reached: turn a polygon off to add another.</small>`:''}</div>`;
 // Every axis keeps its slot and angle whatever the selection: an axis without values (or
 // without a baseline value) is drawn greyed, with no points, and listed under the chart.
 const axes=radarRows(polys,baseline);axes.forEach(a=>{a.active=a.present>0&&a.baseline!==null;});
 const live=axes.filter(a=>a.active),omitted=axes.filter(a=>!a.present),unanchored=axes.filter(a=>a.present&&a.baseline===null);
 if(!polys.length||!live.length){host.innerHTML=`<div class="radar-head"><strong>Objective radar</strong><span>${polys.length?'No objective metrics with a baseline value at these checkpoints.':'No polygon selected.'}</span></div>${chipRow}`;host.querySelectorAll('[data-radar-toggle]').forEach(b=>b.onclick=()=>toggleRadarPolygon(...b.dataset.radarToggle.split(':').map(Number)));return;}
 const W=560,H=470,cx=W/2,cy=H/2+6,RAD=150,n=axes.length,R=radarWindow(),D=0.25*Math.log2(R);
 const angle=j=>-Math.PI/2+j*2*Math.PI/n;
 const pos=(j,s)=>[cx+Math.cos(angle(j))*RAD*s,cy+Math.sin(angle(j))*RAD*s];
 const pt=p=>`${p[0].toFixed(1)} ${p[1].toFixed(1)}`;
 const ring=(s,cls)=>`<polygon class="${cls}" points="${axes.map((_,j)=>pt(pos(j,s))).join(' ')}"/>`;
 const rings=ring(.25,'radar-ring')+ring(.75,'radar-ring')+ring(1,'radar-limit')+ring(.5,'radar-baseline')+`<circle class="radar-limit" cx="${cx}" cy="${cy}" r="3"/>`;
 const spokes=axes.map((_,j)=>{const [x,y]=pos(j,1);return `<line class="radar-spoke" x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"/>`;}).join('');
 const ringText=a=>`baseline ${L.format(a.baseline,a.fmt)} on the 0.5 ring · rim ${L.format(a.rim,a.fmt)} · centre ${L.format(a.centre,a.fmt)}`;
 const labels=axes.map((a,j)=>{const [x,y]=pos(j,1.16),c=Math.cos(angle(j));const anchor=Math.abs(c)<.2?'middle':c>0?'start':'end';return `<text class="radar-label${a.active?'':' radar-label-off'}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" dominant-baseline="middle">${h(a.label)}<title>${h(a.label)} (${a.good==='low'?'lower is better':'higher is better'})\n${a.active?ringText(a):'no value for this selection'}</title></text>`;}).join('');
 const polyLabel=p=>`${h(runs[p.run].run.label)} · step ${p.step}`;
 const isBaseline=p=>p.run===baseline.run&&p.step===baseline.step;
 const polygons=polys.map((p,k)=>{
  const pts=axes.map((a,j)=>a.scaled[k]===null?null:pos(j,a.scaled[k]));
  let d='',open=false;pts.forEach(q=>{if(!q){open=false;return;}d+=(open?'L':'M')+pt(q);open=true;});
  const complete=pts.every(Boolean);if(complete)d+='Z';else if(pts[0]&&pts.at(-1))d+='L'+pt(pts[0]);
  const marks=axes.map((a,j)=>pts[j]?`<g class="radar-mark${a.clamped[k]?' radar-clamped':''}">${p.marker(pts[j][0],pts[j][1],4.5)}<title>${polyLabel(p)}\n${h(a.label)}: ${L.format(a.values[k],a.fmt)}${isBaseline(p)?' (baseline)':', '+relText(a,a.rel[k])}${a.clamped[k]?' · beyond the window, clamped':''}\n${ringText(a)}</title></g>`:'').join('');
  return `<g class="radar-run" data-run="${p.run}" style="color:${polyColor(p)}">${complete&&!isBaseline(p)?`<path class="radar-fill" d="${d}"/>`:''}<path class="radar-line" d="${d}"${p.dash?` stroke-dasharray="${p.dash}"`:''}/>${marks}</g>`;
 }).join('');
 const legend=polys.map(p=>`<span class="radar-key" data-run="${p.run}">${swatch(p)}${polyLabel(p)}${isBaseline(p)?' <small>baseline</small>':''}${p.selected?'':' <small>(not selected)</small>'}</span>`).join('')+`<span class="radar-key radar-key-note"><svg class="radar-swatch" viewBox="0 0 30 14" aria-hidden="true"><g class="radar-mark radar-clamped"><circle cx="15" cy="7" r="4"/></g></svg>hollow = clamped at the window</span>`;
 const picker=`<div class="radar-picker"><span>window R</span><div class="segmented">${RADAR_WINDOWS.map(w=>`<button data-radar-r="${w}" class="${w===R?'active':''}">${w}x</button>`).join('')}</div><small>rim = ${R}x better, centre = ${R}x worse · captured and reward: ±${D.toFixed(3)} absolute</small></div>`;
 const table=`<table><thead><tr><th>Axis</th>${polys.map(p=>`<th><span class="radar-th" data-run="${p.run}">${swatch(p)}<span class="run-marker">${p.run+1}</span> step ${p.step}</span>${isBaseline(p)?' <small>baseline</small>':''}</th>`).join('')}</tr></thead><tbody>${axes.map(a=>{
  const best=a.good==='low'?Math.min(...a.values.filter(finite)):Math.max(...a.values.filter(finite));
  return `<tr><th>${h(a.label)}<small>${a.good==='low'?'↓ lower is better · outward = low':'↑ higher is better · outward = high'}</small></th>${a.values.map((v,k)=>`<td class="${finite(v)&&v===best?'evidence-best':''}">${v===null?'—':L.format(v,a.fmt)}${!isBaseline(polys[k])&&a.rel[k]!==null?`<small>${relText(a,a.rel[k])}</small>`:''}</td>`).join('')}</tr>`;
 }).join('')}</tbody></table>`;
 const notes=[];
 if(!polys.some(isBaseline))notes.push(`The baseline polygon (run ${base+1} at step ${baseline.step}) is turned off; it is still the 0.5 ring every other polygon is measured against.`);
 if(omitted.length)notes.push(`No value (axis kept in place, greyed): ${omitted.map(a=>`${h(a.label)} <code>${h(a.probe)}/${h(a.key)}</code>`).join(', ')}.`);
 if(unanchored.length)notes.push(`No baseline value (axis kept in place, greyed): ${unanchored.map(a=>h(a.label)).join(', ')}.`);
 host.innerHTML=`<div class="radar-head"><strong>Objective radar</strong><span>${n} axes · ${polys.length} polygon${polys.length===1?'':'s'}, every one relative to the baseline run at its selected step, which sits on the 0.5 ring · outward is better on every axis · losses, offset, shape and spread by log2 ratio over a ${R}x window, captured and reward by absolute difference over ±${D.toFixed(3)} · the rim and centre rings are the clamp limits · hover a marker or an axis label for the raw values</span></div>${chipRow}${picker}<div class="radar-body${polys.length>=4?' radar-wide':''}"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Objective radar">${rings}${spokes}${labels}${polygons}</svg><div class="radar-side"><div class="radar-legend">${legend}</div><div class="evidence-metrics radar-table">${table}</div>${notes.map(t=>`<p class="radar-note">${t}</p>`).join('')}</div></div>`;
 host.querySelectorAll('[data-radar-r]').forEach(b=>b.onclick=()=>{setHashParam('radar_r',b.dataset.radarR);renderRadar();});
 host.querySelectorAll('[data-radar-toggle]').forEach(b=>b.onclick=()=>toggleRadarPolygon(...b.dataset.radarToggle.split(':').map(Number)));
}

/* ---------- paired per-frame scatter (Objective) ----------
   Each run's objective/objective.json carries one row per probed frame. Runs probing the same
   seeded frames are joined on (source, split, global_idx): x = the baseline run, y = each other
   run, three panels per pair (global error with a picker, then offset and shape). The JSON is
   fetched lazily, only when the Objective page shows two or more runs. */
const PAIR_GLOBAL=[
 {key:'loss_flow_joint',label:'joint flow loss',log:true},
 {key:'loss_discrete_ce',label:'FAST CE',log:true},
 {key:'loss_depth_event',label:'depth event loss',log:true},
 {key:'collapse',label:'collapse (std over samples)',log:false}
];
const PAIR_FIXED=[{key:'offset',label:'offset (DC-band MSE, normalized units)',log:true},{key:'shape',label:'shape (above-DC MSE, normalized units)',log:true}];
const SPLIT_COLOR={val:'#E63946',train:'#457B9D'};
const SOURCE_COLORS=['#80baff','#eda361','#8ed3a2','#e58ad0','#f2d56b','#8fd6e8','#c9a6ff','#f29b8a','#a7c957','#6fb3b8','#d4a373','#b8b8ff','#ff9f1c'];
const pairCache=new Map();
function pairFrames(i){
 const panel=L.probe(i,'objective')?.panels[L.step(i)]?.[0];
 if(!panel)return {error:'no objective output at this step'};
 const url=panel.url.replace(/[^/]+$/,'objective.json'),entry=pairCache.get(url);
 if(entry)return entry;
 const pending={loading:true};pairCache.set(url,pending);
 fetch(url,{cache:'no-store'}).then(r=>r.ok?r.json():Promise.reject(Error('HTTP '+r.status))).then(j=>pairCache.set(url,{rows:j.per_frame||[]})).catch(e=>pairCache.set(url,{error:e.message})).finally(()=>{if(L.probeId==='objective')renderPairs();});
 return pending;
}
const frameKey=r=>`${r.source}|${r.split}|${r.global_idx}`;
function ranks(v){const idx=v.map((_,i)=>i).sort((a,b)=>v[a]-v[b]),r=Array(v.length);for(let i=0;i<idx.length;){let j=i;while(j+1<idx.length&&v[idx[j+1]]===v[idx[i]])j++;const avg=(i+j)/2+1;for(let k=i;k<=j;k++)r[idx[k]]=avg;i=j+1;}return r;}
function spearman(x,y){if(x.length<3)return null;const rx=ranks(x),ry=ranks(y),m=(x.length+1)/2;let sxy=0,sxx=0,syy=0;for(let i=0;i<x.length;i++){const a=rx[i]-m,b=ry[i]-m;sxy+=a*b;sxx+=a*a;syy+=b*b;}return sxx&&syy?sxy/Math.sqrt(sxx*syy):null;}
function worstDecile(pts,get){const k=Math.max(1,Math.round(pts.length/10));return new Set([...pts].sort((a,b)=>get(b)-get(a)).slice(0,k).map(p=>p.id));}
function pairStats(pts){
 return ['val','train'].map(split=>{
  const q=pts.filter(p=>p.split===split);if(!q.length)return {split,n:0};
  const A=worstDecile(q,p=>p.x),B=worstDecile(q,p=>p.y);let disc=0;A.forEach(id=>{if(!B.has(id))disc++;});
  return {split,n:q.length,rho:spearman(q.map(p=>p.x),q.map(p=>p.y)),below:q.filter(p=>p.y<p.x).length/q.length,disc,decile:A.size};
 });
}
const tickText=v=>{const a=Math.abs(v);return a===0?'0':a>=100?v.toFixed(0):String(Number(v.toPrecision(a>=1?3:2)));};
function logTicks(lo,hi){const out=[];for(let e=Math.floor(lo);e<=Math.floor(hi);e++)for(const m of (hi-lo>=2.5?[1]:[1,2,5])){const v=m*10**e,l=Math.log10(v);if(l>=lo&&l<=hi)out.push(v);}return out;}
function linTicks(lo,hi){const raw=(hi-lo)/4,p=10**Math.floor(Math.log10(raw)),step=[1,2,5,10].map(m=>m*p).find(s=>s>=raw);const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-12;v+=step)out.push(Number(v.toPrecision(10)));return out;}
function scatterSVG(pts,m,colorOf,xLabel,yLabel){
 const W=380,H=380,l=52,r=12,t=12,b=44,pw=W-l-r,ph=H-t-b,tr=m.log?Math.log10:v=>v;
 const vals=pts.flatMap(p=>[tr(p.x),tr(p.y)]);let lo=Math.min(...vals),hi=Math.max(...vals);if(lo===hi){lo-=1;hi+=1;}const pad=(hi-lo)*.05;lo-=pad;hi+=pad;
 const X=v=>l+(tr(v)-lo)/(hi-lo)*pw,Y=v=>t+ph-(tr(v)-lo)/(hi-lo)*ph;
 const ticks=m.log?logTicks(lo,hi):linTicks(lo,hi);
 const grid=ticks.map(v=>`<line class="pair-gridline" x1="${X(v).toFixed(1)}" y1="${t}" x2="${X(v).toFixed(1)}" y2="${t+ph}"/><line class="pair-gridline" x1="${l}" y1="${Y(v).toFixed(1)}" x2="${l+pw}" y2="${Y(v).toFixed(1)}"/><text class="pair-tick" x="${X(v).toFixed(1)}" y="${t+ph+14}" text-anchor="middle">${tickText(v)}</text><text class="pair-tick" x="${l-6}" y="${Y(v).toFixed(1)}" text-anchor="end" dominant-baseline="middle">${tickText(v)}</text>`).join('');
 const order=[...pts].sort((a,b)=>(a.split==='val')-(b.split==='val'));
 const dots=order.map(p=>`<circle class="pair-dot${p.id===pairFrame?' pair-dot-pinned':''}" data-frame="${h(p.id)}" cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="3" fill="${colorOf(p)}"><title>${h(p.row.source)} · episode ${p.row.episode_idx} · frame ${p.row.frame_idx} · ${p.split}\n${h(p.row.subtask||'(no subtask)')}\nx ${h(xLabel)}: ${L.format(p.x,4)}\ny ${h(yLabel)}: ${L.format(p.y,4)}</title></circle>`).join('');
 return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${h(m.label)}: ${h(yLabel)} against ${h(xLabel)}">${grid}<rect class="pair-frame" x="${l}" y="${t}" width="${pw}" height="${ph}"/><line class="pair-identity" x1="${l}" y1="${t+ph}" x2="${l+pw}" y2="${t}"/>${dots}<text class="pair-axis" x="${l+pw/2}" y="${H-6}" text-anchor="middle">x · ${h(xLabel)}${m.log?' (log)':''}</text><text class="pair-axis" transform="translate(12 ${t+ph/2}) rotate(-90)" text-anchor="middle">y · ${h(yLabel)}${m.log?' (log)':''}</text></svg>`;
}
function pairPanel(m,base,other,joined,colorOf,xLabel,yLabel){
 const pts=joined.map(([k,a,c])=>({id:k,row:a,split:a.split,x:a[m.key],y:c[m.key]})).filter(p=>finite(p.x)&&finite(p.y)&&(!m.log||(p.x>0&&p.y>0)));
 const skipped=joined.length-pts.length;
 if(!pts.length)return `<article class="pair-panel"><h4>${h(m.label)}</h4><p class="evidence-empty">No usable rows (${skipped} skipped: missing${m.log?' or non-positive':''}).</p></article>`;
 const stats=pairStats(pts).map(s=>s.n?`<div class="pair-stat" style="--split:${SPLIT_COLOR[s.split]}"><b>${s.split}</b><span>n <strong>${s.n}</strong></span><span>Spearman ρ <strong>${s.rho===null?'—':s.rho.toFixed(3)}</strong></span><span>y &lt; x <strong>${(100*s.below).toFixed(0)}%</strong></span><span title="Frames in the worst decile (highest value) of one run but not the other">discordant <strong>${s.disc}</strong> of ${s.decile}</span></div>`:`<div class="pair-stat" style="--split:${SPLIT_COLOR[s.split]}"><b>${s.split}</b><span>no rows</span></div>`).join('');
 const gap=p=>m.log?Math.abs(Math.log(p.y)-Math.log(p.x)):Math.abs(p.y-p.x);
 const top=[...pts].sort((a,b)=>gap(b)-gap(a)).slice(0,10);
 const table=`<table><colgroup><col style="width:14%"><col style="width:24%"><col style="width:26%"><col style="width:14%"><col style="width:14%"><col style="width:8%"></colgroup><thead><tr><th>ep:frame</th><th>source</th><th>subtask</th><th>x</th><th>y</th><th title="Run with the higher value">↑</th></tr></thead><tbody>${top.map(p=>`<tr data-frame="${h(p.id)}" class="${p.id===pairFrame?'pair-row-pinned':''}" title="${h(p.row.source)} · episode ${p.row.episode_idx} · frame ${p.row.frame_idx} · ${p.split}\n${h(p.row.subtask||'')}\nclick to show the frame"><td>${p.row.episode_idx}:${p.row.frame_idx}</td><td>${h(p.row.source)}</td><td>${h(p.row.subtask||'—')}</td><td>${L.format(p.x,3)}</td><td>${L.format(p.y,3)}</td><td><span class="run-marker" data-run="${p.y>p.x?other:base}">${(p.y>p.x?other:base)+1}</span></td></tr>`).join('')}</tbody></table>`;
 return `<article class="pair-panel"><h4>${h(m.label)}${skipped?` <small>${skipped} skipped</small>`:''}</h4>${scatterSVG(pts,m,colorOf,xLabel,yLabel)}<div class="pair-stats">${stats}</div><div class="pair-table"><small>10 frames with the largest ${m.log?'|log y − log x|':'|y − x|'} · click a row to show the frame</small>${table}</div></article>`;
}
/* Frame panel: the frame behind a clicked dot or table row, pinned in the hash as pair_frame
   (the joined row key) until closed. Images are fetched only on that click, through fetch so a
   404 can show its reason; the object URLs are kept per request URL. */
let pairFrame=null;
const frameImages=new Map();
function frameImage(url){
 if(!frameImages.has(url))frameImages.set(url,fetch(url,{cache:'force-cache'}).then(async r=>{if(r.ok)return {src:URL.createObjectURL(await r.blob())};const body=await r.json().catch(()=>({}));return {error:body.error||('HTTP '+r.status)};}).catch(e=>({error:e.message})));
 return frameImages.get(url);
}
function renderFramePanel(host,rowsByRun){
 const el=host.querySelector('#pair-frame');
 if(!pairFrame){el.hidden=true;el.innerHTML='';return;}
 const rows=rowsByRun.map(m=>m.get(pairFrame)||null),row=rows.find(Boolean);
 if(!row){el.hidden=true;return;}
 el.hidden=false;
 const url=cam=>`/api/frame?root=${encodeURIComponent(row.source_root)}&idx=${row.global_idx}&cam=${cam}`;
 const metrics=[['loss_flow_joint','flow loss'],['offset','offset MSE'],['shape','shape MSE']];
 el.innerHTML=`<header class="pair-frame-head"><strong>${h(row.source)} · episode ${row.episode_idx} · frame ${row.frame_idx} <small>${h(row.split)} · global ${row.global_idx}</small></strong><button data-frame-close aria-label="Close frame panel">Close ×</button></header><div class="pair-frame-body"><div class="pair-frame-cams">${['top','wrist'].map(cam=>`<figure><figcaption>${cam}</figcaption><div class="pair-frame-img" data-cam="${cam}"><span>Loading…</span></div></figure>`).join('')}</div><div class="pair-frame-info"><p><b>subtask</b> ${h(row.subtask||'—')}</p><p><b>progress</b> ${finite(row.progress)?(100*row.progress).toFixed(0)+'% of the episode':'—'}</p><div class="evidence-metrics"><table><thead><tr><th></th>${L.data.runs.map((r,i)=>`<th><span class="run-marker" data-run="${i}">${i+1}</span>${i===L.baseline?'<small>baseline</small>':''}</th>`).join('')}</tr></thead><tbody>${metrics.map(([k,label])=>`<tr><th>${label}</th>${rows.map(x=>`<td>${x&&finite(x[k])?L.format(x[k],4):'—'}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div></div>`;
 el.querySelector('[data-frame-close]').onclick=()=>{pairFrame=null;setHashParam('pair_frame','');renderPairs();};
 el.querySelectorAll('.pair-frame-img').forEach(box=>frameImage(url(box.dataset.cam)).then(result=>{if(!box.isConnected)return;box.innerHTML=result.src?`<img src="${result.src}" alt="${box.dataset.cam} camera, episode ${row.episode_idx} frame ${row.frame_idx}">`:`<span class="pair-frame-error">Frame not available: ${h(result.error)}</span>`;}));
}
function renderPairs(){
 const host=document.getElementById('pairs'),runs=L.data.runs;
 if(L.probeId!=='objective'||runs.length<2){host.hidden=true;host.innerHTML='';return;}
 host.hidden=false;
 pairFrame=hashParam('pair_frame')||null;
 const metric=PAIR_GLOBAL.find(m=>m.key===hashParam('pair_metric'))||PAIR_GLOBAL[0],bySource=hashParam('pair_color')==='source';
 const base=L.baseline,baseLabel=runs[base].run.label,frames=runs.map((_,i)=>pairFrames(i));
 const rowsByRun=frames.map(f=>new Map((f.rows||[]).map(r=>[frameKey(r),r])));
 const sources=[...new Set(frames.flatMap(f=>f.rows||[]).map(r=>r.source))].sort(natural);
 const colorOf=bySource?p=>SOURCE_COLORS[sources.indexOf(p.row.source)%SOURCE_COLORS.length]:p=>SPLIT_COLOR[p.split];
 const legend=bySource?sources.map((s,i)=>`<span class="pair-key"><i style="background:${SOURCE_COLORS[i%SOURCE_COLORS.length]}"></i>${h(s)}</span>`).join(''):Object.entries(SPLIT_COLOR).map(([s,c])=>`<span class="pair-key"><i style="background:${c}"></i>${s}</span>`).join('');
 const picker=`<div class="segmented">${PAIR_GLOBAL.map(m=>`<button data-pair-metric="${m.key}" class="${m.key===metric.key?'active':''}">${h(m.label)}</button>`).join('')}</div><div class="segmented"><button data-pair-color="split" class="${bySource?'':'active'}">by split</button><button data-pair-color="source" class="${bySource?'active':''}">by source</button></div>`;
 const sections=runs.map((r,j)=>{
  if(j===base)return '';
  const a=frames[base],c=frames[j];
  let body;
  if(a.loading||c.loading)body='<p class="evidence-empty">Loading per-frame rows…</p>';
  else if(a.error||c.error)body=`<p class="evidence-empty">${h(a.error||c.error)}</p>`;
  else{
   const byKey=new Map(c.rows.map(x=>[frameKey(x),x]));
   const joined=a.rows.map(x=>[frameKey(x),x,byKey.get(frameKey(x))]).filter(x=>x[2]);
   const partial=joined.length<a.rows.length||joined.length<c.rows.length;
   body=`<p class="pair-join${partial?' pair-partial':''}">${joined.length} frames matched on (source, split, global_idx) · baseline ${a.rows.length} rows, this run ${c.rows.length}${partial?' · partial match: the runs did not probe the same frames (config or data differ), read with care':''}.</p>`+
    (joined.length?`<div class="pair-grid">${[metric,...PAIR_FIXED].map(m=>pairPanel(m,base,j,joined,colorOf,baseLabel,r.run.label)).join('')}</div>`:'');
  }
  return `<section class="pair-run" data-run="${j}"><header class="pair-head"><span class="run-marker">${j+1}</span><strong>y · ${h(r.run.label)} <small>step ${L.step(j)}</small></strong><span class="pair-vs">against x · <span class="run-marker" data-run="${base}">${base+1}</span> ${h(baseLabel)} <small>step ${L.step(base)}</small></span></header>${body}</section>`;
 }).join('');
 host.innerHTML=`<div class="radar-head"><strong>Paired per-frame scatter</strong><span>Same frames, two checkpoints: a point above the diagonal is a frame this run scores higher (worse for losses, offset and shape) than the baseline. Global error (picker), then its offset and shape components: DC-band and above-DC MSE of the sample mean against the GT chunk in normalized action units, which add up to the chunk MSE. Hover a point for the frame, click it to see the cameras.</span></div><div class="pair-tools">${picker}<div class="pair-legend">${legend}</div></div><div id="pair-frame" class="pair-frame" hidden></div>${sections}`;
 renderFramePanel(host,rowsByRun);
 host.querySelectorAll('[data-pair-metric]').forEach(b=>b.onclick=()=>{setHashParam('pair_metric',b.dataset.pairMetric);renderPairs();});
 host.querySelectorAll('[data-pair-color]').forEach(b=>b.onclick=()=>{setHashParam('pair_color',b.dataset.pairColor);renderPairs();});
 host.querySelectorAll('[data-frame]').forEach(el=>el.onclick=()=>{setHashParam('pair_frame',el.dataset.frame);renderPairs();});
}

/* ---------- metrics ---------- */
function renderMetrics(){
 const host=document.getElementById('metrics'),all=L.metrics(),primary=all.filter(m=>m.primary),rows=L.allMetrics||!primary.length?all:primary;
 host.innerHTML=`<div class="metrics-heading"><h2>Metrics at the selected checkpoints <small>${rows.length} of ${all.length}</small></h2>${primary.length&&primary.length<all.length?`<button id="all-metrics" aria-pressed="${!!L.allMetrics}">${L.allMetrics?'Primary only':'All metrics'}</button>`:''}</div><div id="metric-table"></div>`;
 L.metricTable('#metric-table',rows);
 const b=document.getElementById('all-metrics');if(b)b.onclick=()=>{L.allMetrics=!L.allMetrics;L.emit();};
}

function render(){
 if(!L.data)return;
 wall=hashParam('wall')==='1';
 const lay=layout();
 renderRail();renderTitle(lay);renderRails(lay);renderRuns();renderRadar();renderPairs();renderTools(lay);renderStage(lay);renderMetrics();
}

document.getElementById('probe-filter').oninput=e=>{railFilter=e.target.value;renderRail();};
document.getElementById('copy-link').onclick=()=>L.copyLink();
document.getElementById('refresh').onclick=()=>L.refresh();
addEventListener('keydown',e=>{
 if(!L.data||e.target.closest('input,textarea,select,[contenteditable]')||e.metaKey||e.ctrlKey||e.altKey)return;
 const k=e.key.toLowerCase();
 if(k==='j'||k==='k'){const probes=visibleProbes(),i=probes.findIndex(p=>p.id===L.probeId),n=probes[(i+(k==='j'?1:-1)+probes.length)%probes.length];if(n)L.setProbe(n.id);}
 else if(k==='n'||k==='p'){const panels=L.panels(),i=panels.findIndex(p=>p.key===L.panelKey),n=panels[(i+(k==='n'?1:-1)+panels.length)%panels.length];if(n)L.setPanel(n.key);}
 else if(k===' '){const b=document.querySelector('#stage [data-ev-play],#stage [data-wall-all]');if(b){e.preventDefault();b.click();}}
});
L.start(render);
})();

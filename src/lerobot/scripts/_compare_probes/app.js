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
 sim_nail:{
  parse(key){const m=key.match(/^ep(\d+)\/videos\/ep(\d+)_(top|wrist|wrist_depth)\.mp4$/);if(!m)return null;return {facets:{Scene:'ep'+m[1],'Sim episode':m[2],Camera:m[3]}};},
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
 renderRail();renderTitle(lay);renderRails(lay);renderRuns();renderTools(lay);renderStage(lay);renderMetrics();
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

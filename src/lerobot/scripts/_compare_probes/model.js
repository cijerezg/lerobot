/* Probe comparison data model and media components. */
(function(){
'use strict';
const listeners=new Set();
const h=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const natural=(a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true});
const finite=x=>typeof x==='number'&&Number.isFinite(x);
function selectionFromURL(){
 const q=new URLSearchParams(location.hash.slice(1));
 return {
  steps:(q.get('steps')||'').split(',').map(s=>s===''?undefined:Number(s)),
  probeId:q.get('probe'),panelKey:q.get('panel'),
  baseline:Math.max(0,Math.trunc(Number(q.get('baseline')))||0),
  shared:q.get('shared')==='1',archived:q.get('archived')==='1',
  view:['map','aligned'].includes(q.get('view'))?q.get('view'):null,
  locked:q.get('locked')==='1',metric:q.get('metric'),allMetrics:q.get('all_metrics')==='1'
 };
}
const L=window.ProbeCompare={data:null,h,...selectionFromURL()};
L.step=i=>L.steps[i]===undefined?null:String(L.steps[i]);
L.probe=(i,id=L.probeId)=>L.data?.runs[i]?.probes.find(p=>p.id===id&&p.panels[L.step(i)]!==undefined)||null;
L.panel=(i,key=L.panelKey,id=L.probeId)=>L.probe(i,id)?.panels[L.step(i)]?.find(p=>(p.align||p.file)===key)||null;
L.probes=function(){
 const map=new Map();if(!L.data)return [];
 L.data.runs.forEach((r,i)=>r.probes.forEach(p=>{
  if(p.panels[L.step(i)]===undefined||(!L.archived&&/jacobian/i.test(p.id)))return;
  const row=map.get(p.id)||{id:p.id,title:p.title,group:p.group,claim:p.claim,count:0,statuses:[]};
  row.count++;row.statuses.push(p.status?.[L.step(i)]||'info');map.set(p.id,row);
 }));
 return [...map.values()].sort((a,b)=>{const ai=L.data.groups.indexOf(a.group),bi=L.data.groups.indexOf(b.group);return (ai<0?999:ai)-(bi<0?999:bi)||natural(a.title,b.title);});
};
L.panels=function(id=L.probeId,all=false){
 const map=new Map();if(!L.data)return [];
 L.data.runs.forEach((r,i)=>(L.probe(i,id)?.panels[L.step(i)]||[]).forEach(p=>{
  const key=p.align||p.file,row=map.get(key)||{key,caption:p.caption||p.file,kind:p.kind||'data',primary:false,count:0,files:[],how:p.how||''};
  row.count++;row.primary||=!!p.primary;if(!row.files.includes(p.file))row.files.push(p.file);map.set(key,row);
 }));
 return [...map.values()].filter(p=>all||!L.shared||p.count===L.data.runs.length).sort((a,b)=>Number(b.primary)-Number(a.primary)||natural(a.key,b.key));
};
L.metrics=function(id=L.probeId){
 const map=new Map();if(!L.data)return [];
 L.data.runs.forEach((r,i)=>(L.probe(i,id)?.metrics||[]).forEach(m=>{
  const row=map.get(m.key)||{key:m.key,label:m.label,note:m.note||'',good:m.good||'none',fmt:m.fmt??4,primary:false,refs:m.refs||[],cells:Array(L.data.runs.length).fill(null)};
  row.primary||=!!m.primary;row.cells[i]={value:m.values[L.step(i)],status:m.statuses?.[L.step(i)]||'info',fmt:m.fmt??4};map.set(m.key,row);
 }));
 return [...map.values()].sort((a,b)=>Number(b.primary)-Number(a.primary)||natural(a.label,b.label));
};
L.format=function(v,n=4){if(!finite(v))return '—';n=Number.isInteger(n)?Math.max(0,Math.min(10,n)):4;return n===0?String(Math.round(v)):v!==0&&Math.abs(v)<10**(-n)?v.toExponential(2):v.toFixed(n);};
L.thumb=function(row,id=L.probeId){const key=typeof row==='string'?row:row?.key;return L.data?.runs.map((_,i)=>L.panel(i,key,id)).find(p=>p?.kind==='image')||null;};
function normalize(){
 if(!L.data)return;
 if(!L.view)L.view=L.data.runs.every(r=>r.run.explicit_step)?'aligned':'map';
 L.steps=L.data.runs.map((r,i)=>{const want=L.steps[i]??r.run.initial_step;return r.steps.some(s=>s.step===want)?want:r.steps.at(-1)?.step;});
 if(L.locked&&!L.steps.every(s=>s!==undefined&&s===L.steps[0]))L.locked=false;
 L.baseline=Math.max(0,Math.min(L.data.runs.length-1,Math.trunc(L.baseline)||0));
 const probes=L.probes();if(!probes.some(p=>p.id===L.probeId)){L.probeId=probes[0]?.id||null;L.panelKey=null;}
 const panels=L.panels();if(!panels.some(p=>p.key===L.panelKey))L.panelKey=(panels.find(p=>p.key==='objective.png')||panels.find(p=>p.primary&&p.kind==='image')||panels.find(p=>p.primary)||panels[0])?.key||null;
}
function save(){
 if(!L.data)return;
 const q=new URLSearchParams(location.hash.slice(1));
 for(const [k,v] of Object.entries({probe:L.probeId||'',panel:L.panelKey||'',steps:L.steps.join(','),baseline:L.baseline,shared:L.shared?'1':'0',archived:L.archived?'1':'0',view:L.view,locked:L.locked?'1':'0',metric:L.metric||'',all_metrics:L.allMetrics?'1':'0'}))q.set(k,String(v));
 history.replaceState(null,'','#'+q);
}
// Pasted comparison links and browser hash navigation must restore the same state
// as opening the link in a new tab. replaceState in save() does not fire this event.
addEventListener('hashchange',()=>{Object.assign(L,selectionFromURL());if(L.data)L.emit();});
L.persist=save;
L.emit=function(){normalize();save();listeners.forEach(fn=>fn());};
L.subscribe=fn=>{listeners.add(fn);return ()=>listeners.delete(fn);};
L.start=async function(render){L.subscribe(render);await L.refresh(true);};
L.refresh=async function(first=false){
 try{const r=await fetch('/api/index',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const data=await r.json();
  if(first)Object.assign(L,selectionFromURL());
  L.data=data;L.emit();if(!first)L.flash('Probe outputs refreshed');
 }catch(err){console.error(err);L.flash('Could not read probe output: '+err.message);if(!L.data){const app=document.getElementById('app');if(app){app.innerHTML='<div class="evidence-empty">Could not load probe output. <button id="compare-retry">Retry</button></div>';document.getElementById('compare-retry').onclick=()=>L.refresh(true);}}}
};
L.setProbe=id=>{if(L.probes().some(p=>p.id===id)){L.probeId=id;L.panelKey=null;L.emit();}};
L.setPanel=key=>{if(L.panels().some(p=>p.key===key)){L.panelKey=key;L.emit();}};
L.setStep=(i,step)=>{step=Number(step);if(L.data.runs[i]?.steps.some(s=>s.step===step)){L.steps[i]=step;L.emit();}};
// Validate the whole selection before emitting, so shared-panel normalization only
// runs once after all columns have moved to their common checkpoint.
L.setSteps=steps=>{const values=steps.map(Number);if(values.length!==L.data.runs.length||!values.every((step,i)=>L.data.runs[i].steps.some(s=>s.step===step)))return false;L.steps=values;L.emit();return true;};
L.setBaseline=i=>{L.baseline=Number(i);L.emit();};
L.setShared=value=>{L.shared=!!value;L.emit();};L.setArchived=value=>{L.archived=!!value;L.emit();};
L.copyLink=async function(){save();try{await navigator.clipboard.writeText(location.href);L.flash('Comparison link copied');}catch(_){L.flash('Copy the current browser address to share this comparison.');}};
L.flash=function(message){let el=document.getElementById('compare-toast');if(!el){el=document.createElement('div');el.id='compare-toast';el.setAttribute('role','status');document.body.append(el);}el.textContent=message;el.hidden=false;clearTimeout(L._toast);L._toast=setTimeout(()=>el.hidden=true,3500);};
L.jumpSim=function(){const rows=L.probes().filter(p=>/sim/i.test(p.id));const p=rows.find(p=>L.panels(p.id,true).some(x=>x.kind==='video'))||rows.find(p=>p.id==='sim_nail')||rows[0];if(!p){L.flash('No simulation output at these checkpoints');return;}L.shared=false;L.view='aligned';L.probeId=p.id;L.panelKey=L.panels(p.id,true).find(x=>x.kind==='video')?.key||null;L.emit();};
L.metricTable=function(host,rows=L.metrics()){
 if(typeof host==='string')host=document.querySelector(host);if(!host)return;
 if(!rows.length){host.innerHTML='<p class="evidence-empty">No scalar metrics recorded at these checkpoints.</p>';return;}
 host.innerHTML=`<div class="evidence-metrics"><table><thead><tr><th>Measurement</th>${L.data.runs.map((r,i)=>`<th>${h(r.run.label)}${i===L.baseline?' <small>baseline</small>':''}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>{
  const vals=row.cells.map(c=>c?.value).filter(finite),best=vals.length?(row.good==='low'?Math.min(...vals):row.good==='high'?Math.max(...vals):null):null,b=row.cells[L.baseline]?.value;
  return `<tr><th><strong>${h(row.label)}</strong><small>${row.good==='low'?'↓ lower preferred':row.good==='high'?'↑ higher preferred':'descriptive'}</small>${row.note?`<details><summary>Definition</summary>${h(row.note)}</details>`:''}</th>${row.cells.map((c,i)=>`<td class="${finite(c?.value)&&c.value===best?'evidence-best':''}">${L.format(c?.value,c?.fmt??row.fmt)}${i!==L.baseline&&finite(b)&&finite(c?.value)?`<small>${c.value-b>0?'+':''}${L.format(c.value-b,c.fmt)} vs baseline</small>`:''}</td>`).join('')}</tr>`;
 }).join('')}</tbody></table></div>`;
};
let lightbox;
function openImage(url,label){
 if(!lightbox){lightbox=document.createElement('dialog');lightbox.className='evidence-lightbox';lightbox.innerHTML='<button class="evidence-lightbox-close" aria-label="Close image">Close ×</button><img alt="">';lightbox.querySelector('button').onclick=()=>lightbox.close();lightbox.onclick=e=>{if(e.target===lightbox)lightbox.close();};document.body.append(lightbox);}
 lightbox.querySelector('img').src=url;lightbox.querySelector('img').alt=label||'Probe figure';lightbox.showModal();
}
function commandInspector(frame,index,count,attempt=0){
 if(index<0||!frame.isConnected)return;
 try{
  const w=frame.contentWindow,d=frame.contentDocument,plot=d?.getElementById('action-inspector-plot');
  if(typeof w?.probeInspectorGo==='function'){w.probeInspectorGo(index);frame.dataset.sampleIndex=index;return;}
  if(w?.Plotly&&plot&&typeof plot.emit==='function'){w.Plotly.animate(plot,[String(index)],{mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}});w.Plotly.relayout(plot,{'sliders[0].active':index});plot.emit('plotly_sliderchange',{step:{value:String(index),args:[[String(index)]]}});frame.dataset.sampleIndex=index;return;}
  const current=Number(frame.dataset.sampleIndex||0),forward=count?(index-current+count)%count:0,backward=count?(current-index+count)%count:0,button=d?.getElementById(forward<=backward?'next':'prev');
  if(button){for(let i=0;i<Math.min(forward,backward);i++)button.click();frame.dataset.sampleIndex=index;return;}
 }catch(_){}
 if(attempt<24)setTimeout(()=>commandInspector(frame,index,count,attempt+1),200);
}
function time(t){if(!finite(t))return '—';const n=Math.round(Math.max(0,t)*10);return `${Math.floor(n/600)}:${((n%600)/10).toFixed(1).padStart(4,'0')}`;}
function videoControls(host){
 const videos=[...host.querySelectorAll('video')];if(!videos.length)return;
 let leader=null,sync=true;
 const seeking=new WeakSet(),playing=new WeakSet(),pausing=new WeakSet(),rates=new WeakSet();
 const lead=()=>videos.includes(leader)?leader:videos.reduce((a,b)=>(b.duration||0)>(a?.duration||0)?b:a,null);
 const move=(v,t)=>{if(finite(v.duration)){const x=Math.max(0,Math.min(v.duration,t));if(Math.abs(v.currentTime-x)>.06){seeking.add(v);v.currentTime=x;}}};
 const play=v=>{if(v.paused&&(!v.ended||v.currentTime<v.duration)){playing.add(v);v.play().catch(()=>playing.delete(v));}};
 const pause=v=>{if(!v.paused){pausing.add(v);v.pause();}};
 const rate=(v,r)=>{if(v.playbackRate!==r){rates.add(v);v.playbackRate=r;}};
 const propagate=(source,what)=>{if(!sync)return;videos.forEach(v=>{if(v===source)return;if(what!=='rate')move(v,source.currentTime);rate(v,source.playbackRate);if(what==='play')play(v);if(what==='pause')pause(v);});};
 const bar=document.createElement('div');bar.className='evidence-transport';bar.innerHTML='<button data-ev-play>▶ Play all</button><button data-ev-back aria-label="Back one second">−1s</button><button data-ev-next aria-label="Forward one second">+1s</button><input type="range" aria-label="Seek these recordings" min="0" max="1" step="0.05" value="0"><output>Loading…</output><span class="evidence-speeds" aria-label="Playback speed">'+[.25,.5,1,2].map(r=>`<button data-ev-rate="${r}" aria-pressed="${r===1}">${r}×</button>`).join('')+'</span><button data-ev-sync aria-pressed="true">Sync on</button><small>Elapsed time aligned; outcomes can diverge.</small>';
 host.prepend(bar);
 const update=()=>{const v=lead();if(!v)return;const duration=Math.max(...videos.map(x=>finite(x.duration)?x.duration:0));const range=bar.querySelector('input');range.max=duration||1;if(document.activeElement!==range)range.value=v.currentTime;bar.querySelector('output').textContent=`${time(v.currentTime)} / ${time(duration||NaN)}`;bar.querySelector('[data-ev-play]').textContent=videos.some(x=>!x.paused&&!x.ended)?'Ⅱ Pause all':'▶ Play all';};
 const seek=t=>{leader=lead();videos.forEach(v=>move(v,t));update();};
 bar.querySelector('input').oninput=e=>seek(Number(e.target.value));
 bar.querySelector('[data-ev-back]').onclick=()=>seek((lead()?.currentTime||0)-1);
 bar.querySelector('[data-ev-next]').onclick=()=>seek((lead()?.currentTime||0)+1);
 bar.querySelector('[data-ev-play]').onclick=()=>{leader=lead();if(videos.some(v=>!v.paused&&!v.ended))videos.forEach(pause);else{const t=leader.ended?0:leader.currentTime;videos.forEach(v=>{move(v,t);play(v);});}update();};
 bar.querySelector('[data-ev-sync]').onclick=e=>{sync=!sync;e.target.textContent=sync?'Sync on':'Sync off';e.target.setAttribute('aria-pressed',sync);};
 bar.querySelectorAll('[data-ev-rate]').forEach(b=>b.onclick=()=>{videos.forEach(v=>rate(v,Number(b.dataset.evRate)));bar.querySelectorAll('[data-ev-rate]').forEach(x=>x.setAttribute('aria-pressed',x===b));});
 videos.forEach(v=>{
  v.addEventListener('play',()=>{if(!playing.delete(v)){leader=v;propagate(v,'play');}update();});
  v.addEventListener('pause',()=>{if(!pausing.delete(v)&&!v.ended){leader=v;propagate(v,'pause');}update();});
  v.addEventListener('seeked',()=>{if(!seeking.delete(v)){leader=v;propagate(v,'seek');}update();});
  v.addEventListener('ratechange',()=>{if(!rates.delete(v)){leader=v;propagate(v,'rate');}});
  v.addEventListener('timeupdate',()=>{if(v===lead()&&!v.paused)propagate(v,'time');update();});
  v.addEventListener('ended',()=>{if(v===leader)leader=videos.find(x=>x!==v&&!x.paused&&!x.ended)||v;update();});
  v.addEventListener('loadedmetadata',update);
 });
 update();
 // A scoped controller is useful for design-specific bookmarks/loops.
 host.evidencePlayback={videos,seek,getTime:()=>lead()?.currentTime||0,pause:()=>videos.forEach(pause),play:()=>videos.forEach(play)};
}
const sampleMemory=new Map();
L.evidence=function(host,options={}){
 if(typeof host==='string')host=document.querySelector(host);if(!host||!L.data)return;
 const id=options.probe??L.probeId,key=options.key??L.panelKey,runs=options.runs??L.data.runs.map((_,i)=>i);
 host.classList.add('evidence-component');host.dataset.probe=id||'';host.dataset.panel=key||'';
 const rows=runs.map(i=>L.panel(i,key,id)),meta=L.panels(id,true).find(p=>p.key===key);
 if(!key||!rows.some(Boolean)){
  host.innerHTML=`<div class="evidence-empty"><strong>${/sim/i.test(id||'')?'No browser recording at these checkpoints':'No matching panel at these checkpoints'}</strong><p>${/sim/i.test(id||'')?'Rerun logs are separate artifacts. Exported top, wrist and depth videos appear after their MP4s and manifest are written.':'Choose another panel or checkpoint. A missing output is not a failed probe.'}</p></div>`;return;
 }
 const samples=rows.map(p=>p?.samples||[]),common=samples.every(s=>s.length)?samples[0].filter(s=>samples.slice(1).every(other=>other.some(x=>x.key===s.key))):[];
 const memoryKey=[id,key,...runs.map(i=>i+':'+L.step(i))].join('|');
 const memory=sampleMemory.get(memoryKey)||{shared:0,indices:runs.map(()=>0)};sampleMemory.set(memoryKey,memory);if(sampleMemory.size>100)sampleMemory.delete(sampleMemory.keys().next().value);
 const warnings=[];
 if(meta?.files.length>1)warnings.push('Band aligned: the same percentile can refer to different observations. Filenames below identify the actual samples.');
 if(samples.some(s=>s.length)&&!common.length)warnings.push(rows.some(p=>p?.sample_alignment==='rank')?'Rank aligned, not frame aligned. Each run selected its own exemplars.':'No common episode/frame keys. Navigate each inspector independently.');
 host.innerHTML=warnings.map(w=>`<div class="evidence-warning">${h(w)}</div>`).join('')+
 (common.length?`<div class="evidence-samples"><strong>${common.length} shared frames</strong><div>${common.map((s,i)=>`<button data-ev-sample="${i}" aria-pressed="${i===memory.shared}" title="${h(s.label)}">${h(s.label)}</button>`).join('')}</div></div>`:'')+
 `<div class="evidence-grid" style="--evidence-runs:${runs.length}">${runs.map((ri,position)=>{
  const p=rows[position],r=L.data.runs[ri];let media='';
  if(!p)media='<div class="evidence-empty">Not recorded for this run / checkpoint.</div>';
  else if(p.kind==='image')media=`<div class="evidence-media"><img src="${h(p.url)}" alt="${h(p.caption||p.file)}" loading="lazy"></div>`;
  else if(p.kind==='video')media=`<div class="evidence-media"><video controls playsinline muted preload="metadata" src="${h(p.url)}"></video></div>`;
  else if(p.kind==='html')media=`<div class="evidence-media"><iframe src="${h(p.url)}" title="${h(p.caption||p.file)}" loading="lazy" data-ev-frame="${position}"></iframe></div>`;
  else media=`<div class="evidence-empty"><a href="${h(p.url)}" target="_blank" rel="noopener">Open ${h(p.file)} ↗</a></div>`;
  const nav=p?.samples?.length&&!common.length?`<div class="evidence-individual"><button data-ev-prev="${position}">←</button><span data-ev-label="${position}">${h(p.samples[memory.indices[position]||0]?.label)}</span><button data-ev-next-sample="${position}">→</button></div>`:'';
  return `<article class="evidence-card" data-ev-run="${ri}"><header class="evidence-head"><strong>${h(r.run.label)}</strong><span>step ${L.step(ri)}${ri===L.baseline?' · baseline':''}</span></header>${nav}${media}<footer class="evidence-foot"><span title="${h(p?.file||key)}">${h(p?.file||key)}</span>${p?.kind==='image'?'<button data-ev-zoom aria-pressed="false">1:1</button><button data-ev-expand>Expand ↗</button>':''}${p?`<a href="${h(p.url)}" target="_blank" rel="noopener">Raw ↗</a>`:''}</footer></article>`;
 }).join('')}</div>`;
 host.querySelectorAll('[data-ev-zoom]').forEach(b=>b.onclick=()=>{const img=b.closest('article').querySelector('img'),native=b.getAttribute('aria-pressed')!=='true';b.setAttribute('aria-pressed',native);b.textContent=native?'Fit':'1:1';img.style.width=native?'auto':'100%';img.style.maxWidth=native?'none':'100%';});
 host.querySelectorAll('[data-ev-expand]').forEach(b=>b.onclick=()=>{const img=b.closest('article').querySelector('img');openImage(img.src,img.alt);});
 host.querySelectorAll('.evidence-media img').forEach(img=>img.ondblclick=()=>openImage(img.src,img.alt));
 function applySamples(){
  host.querySelectorAll('[data-ev-frame]').forEach(frame=>{const i=Number(frame.dataset.evFrame),list=samples[i],index=common.length?list.findIndex(s=>s.key===common[memory.shared]?.key):(memory.indices[i]||0);commandInspector(frame,index,list.length);});
  host.querySelectorAll('[data-ev-label]').forEach(label=>{const i=Number(label.dataset.evLabel);label.textContent=samples[i][memory.indices[i]||0]?.label||'';});
  host.querySelectorAll('[data-ev-sample]').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.evSample)===memory.shared));
 }
 host.querySelectorAll('[data-ev-frame]').forEach(f=>f.onload=applySamples);
 host.querySelectorAll('[data-ev-sample]').forEach(b=>b.onclick=()=>{memory.shared=Number(b.dataset.evSample);applySamples();});
 for(const [attr,delta] of [['evPrev',-1],['evNextSample',1]])host.querySelectorAll(attr==='evPrev'?'[data-ev-prev]':'[data-ev-next-sample]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset[attr]),count=samples[i].length;memory.indices[i]=((memory.indices[i]||0)+delta+count)%count;applySamples();});
 host.querySelectorAll('img,video,iframe').forEach(el=>el.addEventListener('error',()=>{let warning=el.parentElement.querySelector('.evidence-load-error');if(!warning){warning=document.createElement('p');warning.className='evidence-load-error';warning.textContent='This artifact could not be displayed. Open its raw file to inspect it.';el.parentElement.append(warning);}}));
 videoControls(host);
};
})();

'use strict';
const $ = id => document.getElementById(id);
const data = JSON.parse($('probe-data').textContent), rows = data.records;
const names = {observation:'RGB + state + depth',depth_only:'Depth features only',depth:'Depth features',depth_null:'Missing-depth null tokens',images_state:'Images + state values',images_only:'Image features only',state_values:'State-value embeddings',total:'All encoded inputs',img_external_0:'Camera · external 0',img_external_1:'Camera · external 1',img_wrist_0:'Camera · wrist',state:'State clause (including wording)',task:'Task',subtask:'Subtask',metadata:'Metadata',question:'Question',template:'Template',control_mode:'Control mode',embodiment:'Embodiment',action_output:'Action-output marker',depth_placeholders:'Depth placeholders',history_placeholders:'History placeholders',other_prompt:'Other prompt',other_image_patches:'Other image patches'};
const labels = {phase:'Action (grasp, move, …)',family:'Colour-free subtask',subtask:'Exact subtask',source:'Dataset source',cameras:'Camera configuration',seconds_to_end:'Seconds to segment end',segment_progress:'Elapsed segment fraction',value:'Critic value V',delta_value:'Signed ΔV (next − current)',advantage:'Signed raw advantage',abs_advantage:'|Raw advantage|',terminal:'Terminal transition'};
const categorical = new Set(['phase','family','subtask','source','cameras','terminal']);
const label = k => names[k] || k;
const fmt = v => v == null ? '—' : v === 0 ? '0' : Math.abs(v) < .001 ? v.toExponential(2) : Number(v).toPrecision(4);
const ep = r => data.episodes[r.episode_ref];
const norm = r => {const k=$('component').value;return k==='observation'?r.observation_grad_norm:k==='depth_only'?(r.raw_depth_consumed?r.depth_grad_norm:null):k==='images_state'?r.image_state_grad_norm:k==='images_only'?r.image_grad_norm:k==='state_values'?r.state_value_grad_norm:k==='total'?r.grad_norm:r.norms[k]??null;};
function field(r, key) {
  if (key === 'source') return ep(r).source;
  if (key === 'cameras') return r.images.map(i=>i.camera).join(' + ');
  if (key === 'abs_advantage') return r.advantage == null ? null : Math.abs(r.advantage);
  if (key === 'terminal') return r.terminal == null ? null : r.terminal ? 'Terminal' : 'Within segment';
  return r[key] ?? null;
}
function option(id, value, text) {
  const node = document.createElement('option'); node.value = value; node.textContent = text; $(id).append(node);
}
for(const key of [...(rows.some(r=>r.raw_depth_consumed)?['observation','depth_only']:[]),'images_state','images_only','state_values','total'])option('component',key,label(key));
[...new Set(rows.flatMap(r=>Object.keys(r.norms)))].filter(k=>rows.some(r=>r.norms[k]!=null)).forEach(k=>option('component',k,label(k)));
Object.entries(labels).forEach(([k,v])=>option('colour',k,v));
['value','seconds_to_end','segment_progress','delta_value','advantage','abs_advantage'].forEach(k=>option('axis',k,labels[k]));
[...new Set(rows.map(r=>r.phase))].sort().forEach(v=>option('phase',v,v));
[...new Set(data.episodes.map(e=>e.source))].sort().forEach(v=>option('source',v,v));
let sorted = [], plotted = [], rankList = [], positions = [], bins = [], selected = -1;
let activeBin = null, zoomDomain = null, fullDomain = null, detailVersion = 0;
let curveSummary = null;
let rankMap = new Map(), selectedPosition = null, plotSpec = null;
const detailCache = new Map(), palette = ['#0072b2','#d55e00','#009e73','#cc79a7','#8a6800','#56b4e9','#6b4c9a','#5b6876'];
const W = 1200, L = 84, R = 22, TOP = 27, BOTTOM = 329, NX = 48, NY = 28;
const svg = (tag, attrs, text, parent=$('chart')) => {
  const n = document.createElementNS('http://www.w3.org/2000/svg',tag);
  for (const [key,value] of Object.entries(attrs)) n.setAttribute(key,value);
  if (text != null) n.textContent = text;
  parent.append(n); return n;
};
const valid = v => v != null && Number.isFinite(v);
const extent = values => {
  let lo=Infinity, hi=-Infinity;
  for (const value of values) { lo=Math.min(lo,value); hi=Math.max(hi,value); }
  if (!Number.isFinite(lo)) return [0,1];
  const pad = hi===lo ? Math.max(.01,Math.abs(lo)*.05) : (hi-lo)*.025;
  return [lo-pad,hi+pad];
};
const logNorm = v => Math.log10(Math.max(v,10**fullDomain[0]));
function textAt(i) {
  const r=rows[i], e=ep(r);
  return `${e.source} · episode ${e.episode} · frame ${r.frame_idx} · ${r.subtask} · gradient ${fmt(norm(r))} · V ${fmt(r.value)} · A ${fmt(r.advantage)}`;
}
function passes(r) {
  return (!$('phase').value || r.phase === $('phase').value)
    && (!$('source').value || ep(r).source === $('source').value)
    && r.subtask.toLowerCase().includes($('text-filter').value.trim().toLowerCase())
    && ($('transition').value==='all' || ($('transition').value==='terminal' ? r.terminal===true : r.terminal===false));
}
function rebuild(resetZoom=true) {
  if (resetZoom) zoomDomain=null;
  activeBin=null; $('bin-note').hidden=true;
  const density=$('view').value==='density', curves=$('view').value==='curves';
  $('color-control').hidden=density||curves; $('axis-control').hidden=!density&&!curves; $('normalization-control').hidden=!density;
  $('bins-control').hidden=!curves; $('numeric-card').hidden=!curves; $('numeric-summary').hidden=!curves; $('zoom').hidden=curves;
  sorted=rows.map((r,i)=>({i,v:norm(r)})).filter(p=>passes(rows[p.i])&&valid(p.v)).sort((a,b)=>a.v-b.v||a.i-b.i);
  rankMap=new Map(sorted.map((p,k)=>[p.i,k]));
  const positives=sorted.filter(p=>p.v>0), floor=positives.length?positives[0].v*.8:1e-8;
  fullDomain=[Math.log10(floor),Math.max(Math.log10(floor)+.2,Math.log10(Math.max(floor,sorted.at(-1)?.v||floor)*1.2))];
  const [lo,hi]=zoomDomain||fullDomain;
  plotted=sorted.filter(p=>(!(density||curves)||valid(field(rows[p.i],$('axis').value)))&&logNorm(p.v)>=lo&&logNorm(p.v)<=hi);
  rankList=plotted.map(p=>p.i);
  $('no-data').hidden=!!plotted.length;
  $('frame-card').hidden=!plotted.length;
  for (const id of ['lower','higher','rank','rank-slider','zoom','reset']) $(id).disabled=!plotted.length;
  document.querySelectorAll('[data-q]').forEach(b=>b.disabled=!plotted.length);
  renderChart();
  if (plotted.length) select(rankList.includes(selected)?selected:rankList[Math.floor(rankList.length/2)]);
  else {selected=-1;detailVersion++;$('rank').value='';$('rank-total').textContent='of 0';$('hover').textContent='';}
}
function numericColour(t, signed=false) {
  t=Math.max(0,Math.min(1,t));
  const stops=signed?[[33,102,172],[245,245,245],[178,24,43]]:[[68,1,84],[33,145,140],[253,231,37]];
  const a=t<.5?0:1, u=t<.5?t*2:(t-.5)*2;
  return `rgb(${stops[a].map((v,i)=>Math.round(v+(stops[a+1][i]-v)*u)).join(',')})`;
}
function legendItem(colour,text) {
  const span=document.createElement('span'), swatch=document.createElement('i');swatch.style.background=colour;
  span.append(swatch,document.createTextNode(text));$('legend').append(span);
}
function colourFunction() {
  const key=$('colour').value, values=plotted.map(p=>field(rows[p.i],key));
  $('legend').replaceChildren();
  if (categorical.has(key)) {
    const categories=[...new Set(rows.map(r=>field(r,key)).filter(v=>v!=null))].sort();
    const colours=new Map(categories.map((v,i)=>[v,categories.length<=palette.length?palette[i]:`hsl(${(i*137.508)%360} 58% 43%)`]));
    const counts=new Map();values.forEach(v=>counts.set(v,(counts.get(v)||0)+1));
    [...counts].sort((a,b)=>b[1]-a[1]).forEach(([v,n])=>legendItem(colours.get(v)||'#aab3bf',`${v??'Unavailable'} (${n})`));
    return i=>colours.get(field(rows[i],key))||'#aab3bf';
  }
  const finite=values.filter(valid), signed=key==='advantage'||key==='delta_value';
  let [lo,hi]=extent(finite);
  if (key==='segment_progress') [lo,hi]=[0,1];
  if (signed) {hi=Math.max(Math.abs(lo),Math.abs(hi));lo=-hi;}
  const ramp=document.createElement('span');ramp.textContent=`${labels[key]}: ${fmt(lo)} `;
  const swatch=document.createElement('i');swatch.className='ramp';
  swatch.style.background=`linear-gradient(to right,${[0,.25,.5,.75,1].map(t=>numericColour(t,signed)).join(',')})`;
  ramp.append(swatch,document.createTextNode(` ${fmt(hi)}`));$('legend').append(ramp);
  if (finite.length<values.length) legendItem('#aab3bf',`Unavailable (${values.length-finite.length})`);
  return i=>{const v=field(rows[i],key);return valid(v)?numericColour((v-lo)/(hi-lo),signed):'#aab3bf';};
}
function axes(xDomain,yDomain,xLabel,yLabel,xLog,yLog) {
  const x=v=>L+(v-xDomain[0])/(xDomain[1]-xDomain[0])*(W-L-R);
  const y=v=>BOTTOM-(v-yDomain[0])/(yDomain[1]-yDomain[0])*(BOTTOM-TOP);
  for(let k=0;k<=5;k++) {
    const xv=xDomain[0]+(xDomain[1]-xDomain[0])*k/5, yv=yDomain[0]+(yDomain[1]-yDomain[0])*k/5;
    svg('line',{x1:x(xv),x2:x(xv),y1:TOP,y2:BOTTOM,stroke:'#e7edf2'});
    svg('text',{x:x(xv),y:BOTTOM+21,'text-anchor':'middle','font-size':12,fill:'#54667f'},xLog?fmt(10**xv):fmt(xv));
    svg('line',{x1:L,x2:W-R,y1:y(yv),y2:y(yv),stroke:'#e7edf2'});
    svg('text',{x:L-10,y:y(yv)+4,'text-anchor':'end','font-size':12,fill:'#54667f'},yLog?fmt(10**yv):fmt(yv));
  }
  svg('text',{x:(L+W-R)/2,y:382,'text-anchor':'middle','font-size':13,fill:'#34495e'},xLabel);
  svg('text',{transform:`translate(18 ${(TOP+BOTTOM)/2}) rotate(-90)`,'text-anchor':'middle','font-size':13,fill:'#34495e'},yLabel);
  return {x,y};
}
function renderChart() {
  $('chart').replaceChildren();positions=[];bins=[];selectedPosition=null;
  const density=$('view').value==='density', curves=$('view').value==='curves', missing=sorted.length-sorted.filter(p=>!(density||curves)||valid(field(rows[p.i],$('axis').value))).length;
  $('status').textContent=`${plotted.length.toLocaleString()} / ${rows.length.toLocaleString()} frames plotted${missing?` · ${missing} unavailable for this axis`:''}`;
  if (!plotted.length) {curveSummary=null;$('bin-table').querySelector('tbody').replaceChildren();$('numeric-summary').textContent='No numerical summary for this selection.';$('legend').replaceChildren();$('plot-note').textContent='Change the filters or comparison axis to see available points.';return;}
  const gDomain=zoomDomain||fullDomain;
  if (curves) {renderCurves();markSelected();return;}
  if (!density) {
    const colour=colourFunction(), first=rankMap.get(plotted[0].i),last=rankMap.get(plotted.at(-1).i);
    const percentile=k=>100*(k+.5)/sorted.length;
    const xDomain=zoomDomain?extent([percentile(first),percentile(last)]):[0,100];
    const {x,y}=axes(xDomain,gDomain,'Gradient percentile among filtered frames',`${label($('component').value)} · gradient norm (log)`,false,true);
    plotSpec={density:false,x,y};
    const marks=svg('g',{'aria-label':'Measured frame points'});
    for(const p of plotted) {
      const at={i:p.i,x:x(percentile(rankMap.get(p.i))),y:y(logNorm(p.v))};positions.push(at);
      const dot=svg('circle',{cx:at.x,cy:at.y,r:2.4,fill:colour(p.i),opacity:.82,'data-index':p.i,style:'cursor:pointer'},null,marks);
      dot.addEventListener('click',event=>{event.stopPropagation();clearBin(false);select(p.i)});
      dot.addEventListener('mouseenter',()=>$('hover').textContent=textAt(p.i));
    }
    $('plot-note').textContent='One dot per frame, sorted from smaller to larger gradient. Colour reveals which tasks or phases occupy each range. Click a dot; rank controls reach overlapping points. Low/middle/high use the plotted frames.';
  } else {
    $('legend').replaceChildren();
    const key=$('axis').value, yDomain=extent(plotted.map(p=>field(rows[p.i],key)));
    const {x,y}=axes(gDomain,yDomain,`${label($('component').value)} · gradient norm (log)`,labels[key],true,false);
    plotSpec={density:true,x,y,gDomain,yDomain};
    const map=new Map(), columns=new Array(NX).fill(0);
    for(const p of plotted) {
      const gx=logNorm(p.v), vy=field(rows[p.i],key);
      const bx=Math.min(NX-1,Math.max(0,Math.floor((gx-gDomain[0])/(gDomain[1]-gDomain[0])*NX)));
      const by=Math.min(NY-1,Math.max(0,Math.floor((vy-yDomain[0])/(yDomain[1]-yDomain[0])*NY)));
      const id=`${bx}:${by}`;
      if(!map.has(id))map.set(id,{id,bx,by,indices:[]});map.get(id).indices.push(p.i);columns[bx]++;
      positions.push({i:p.i,x:x(gx),y:y(vy)});
    }
    bins=[...map.values()];const conditional=$('normalization').value==='conditional';
    const maxCount=Math.max(...bins.map(b=>b.indices.length));
    for(const b of bins) {
      b.columnCount=columns[b.bx]; b.fraction=b.indices.length/b.columnCount;
      b.x0=gDomain[0]+b.bx/NX*(gDomain[1]-gDomain[0]);b.x1=gDomain[0]+(b.bx+1)/NX*(gDomain[1]-gDomain[0]);
      b.y0=yDomain[0]+b.by/NY*(yDomain[1]-yDomain[0]);b.y1=yDomain[0]+(b.by+1)/NY*(yDomain[1]-yDomain[0]);
      const intensity=conditional?b.fraction:Math.log1p(b.indices.length)/Math.log1p(maxCount);
      const rect=svg('rect',{x:x(b.x0),y:y(b.y1),width:x(b.x1)-x(b.x0),height:y(b.y0)-y(b.y1),fill:`hsl(205 65% ${97-66*intensity}%)`,'data-bin':b.id,class:'density-bin'});
      const description=`${b.indices.length} frames · gradient ${fmt(10**b.x0)}–${fmt(10**b.x1)} · ${labels[key]} ${fmt(b.y0)}–${fmt(b.y1)} · ${(b.fraction*100).toFixed(1)}% of ${b.columnCount} frames in this gradient column`;
      rect.addEventListener('mouseenter',()=>$('hover').textContent=description);
      rect.addEventListener('click',event=>{event.stopPropagation();activeBin=b;rankList=[...b.indices];$('bin-note').hidden=false;$('bin-note').querySelector('span').textContent=description;select(rankList[Math.floor(rankList.length/2)]);});
    }
    legendItem('hsl(205 65% 90%)',conditional?'Low fraction':'Low count');
    legendItem('hsl(205 65% 31%)',conditional?'100% of a gradient column':`Up to ${maxCount} frames (log colour scale)`);
    if(yDomain[0]<0&&yDomain[1]>0)svg('line',{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:'#a85c22','stroke-dasharray':'4 4','pointer-events':'none'});
    const terminals=plotted.filter(p=>rows[p.i].terminal).length;
    $('plot-note').textContent=`Click a bin to browse its frames with the rank controls. ${terminals.toLocaleString()} plotted frames are terminal transitions; use Transitions to separate them. ${conditional?'Each gradient column sums to 100%; hover shows sample counts.':'Darker cells contain more frames; colour uses log counts.'}`;
  }
  markSelected();
}
function markSelected() {
  document.querySelectorAll('#chart [data-selected]').forEach(n=>n.remove());
  document.querySelectorAll('#chart [data-bin]').forEach(n=>{n.removeAttribute('stroke');n.removeAttribute('stroke-width');});
  if(activeBin) {const rect=$('chart').querySelector(`[data-bin="${activeBin.id}"]`);if(rect){rect.setAttribute('stroke','#e07416');rect.setAttribute('stroke-width','2.5');}}
  const point=positions.find(p=>p.i===selected);selectedPosition=point||null;
  if(point)svg('circle',{cx:point.x,cy:point.y,r:6,fill:'#e07416',stroke:'white','stroke-width':1.6,'pointer-events':'none','data-selected':selected});
}
function clearBin(update=true) {
  activeBin=null;rankList=plotted.map(p=>p.i);$('bin-note').hidden=true;
  if(update&&rankList.length)select(rankList.includes(selected)?selected:rankList[0]);
}
function selectRank(rank) {
  if(!rankList.length||!Number.isFinite(rank))return;
  rank=Math.max(0,Math.min(rankList.length-1,Math.round(rank)));select(rankList[rank]);
}
function choose(q) {clearBin(false);selectRank(Math.round(q*(rankList.length-1)));}
function select(i) {
  selected=i;const r=rows[i],e=ep(r),v=norm(r),rank=rankList.indexOf(i);
  $('frame-card').hidden=false;$('rank').value=rank+1;$('rank').max=rankList.length;
  $('rank-slider').value=rank;$('rank-slider').max=rankList.length-1;
  // Setting max can clamp the previous value, so set the chosen value last.
  $('rank-slider').value=rank;
  $('rank-total').textContent=`of ${rankList.length.toLocaleString()}${activeBin?' in selected bin':' plotted'}`;
  $('rank-label').textContent=activeBin?'Bin rank':'Rank';
  $('lower').disabled=rank<=0;$('higher').disabled=rank>=rankList.length-1;
  $('frame-title').textContent=r.subtask;
  $('frame-info').textContent=`${label($('component').value)}: ${fmt(v)} · gradient percentile ${(100*(rankMap.get(i)+.5)/sorted.length).toFixed(2)} among filtered frames · V: ${fmt(r.value)}`;
  $('identity').textContent=`${e.source} · ${e.split} · episode ${e.episode} · frame ${r.frame_idx} · ${Number(r.seconds).toFixed(1)} s into episode · ${fmt(r.seconds_to_end)} s to segment end · ${r.segment_progress==null?'—':(100*r.segment_progress).toFixed(1)+'%'} segment progress`;
  $('task').textContent=`Task: ${r.task}`;
  const reasons={paired:'matched next frame',terminal:'terminal; no next value needed',conditioning_changes:'next frame has different conditioning',camera_inputs_change:'next frame has different cameras',missing_next_frame:'matching next frame was not saved',annotations_unavailable:'transition annotations unavailable'};
  $('transition-info').textContent=`ΔV: ${fmt(r.delta_value)} · raw advantage: ${fmt(r.advantage)} · reward: ${fmt(r.reward)} · ${reasons[r.transition_status]||r.transition_status}${r.target_clipped?' · TD target clipped to value support':''}`;
  $('images').replaceChildren(...r.images.map(im=>{const figure=document.createElement('figure'),a=document.createElement('a'),img=document.createElement('img'),caption=document.createElement('figcaption');a.href=im.path;a.target='_blank';a.rel='noopener';img.src=im.path;img.alt=`${im.camera}, ${e.source}, episode ${e.episode}, frame ${r.frame_idx}`;caption.textContent=im.camera;a.append(img);figure.append(a,caption);return figure}));
  $('metadata').textContent=Object.entries(r.metadata||{}).map(([k,v])=>`${k}: ${v}`).join(' · ');
  $('record-file').href=e.file;$('record-index').textContent=`Record ${r.record_index} (zero-based), global frame ${r.global_idx}`;
  $('groups').replaceChildren();$('tokens').replaceChildren();$('detail-status').textContent='';detailVersion++;
  if($('breakdown').open)loadDetails();
  markSelected();$('hover').textContent=textAt(i);
}
async function loadDetails() {
  const i=selected;if(i<0)return;
  const r=rows[i],file=ep(r).file,version=++detailVersion;
  $('detail-status').textContent='Loading saved input breakdown…';
  try {
    if(!detailCache.has(file)) {
      const request=fetch(file).then(response=>{if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json();}).catch(error=>{detailCache.delete(file);throw error;});
      detailCache.set(file,request);if(detailCache.size>3)detailCache.delete(detailCache.keys().next().value);
    }
    const payload=await detailCache.get(file);if(version!==detailVersion||i!==selected)return;
    const full=payload.records[r.record_index];
    if(full.global_idx!==r.global_idx||full.episode!==ep(r).episode||full.subtask!==r.subtask)throw new Error('Saved frame identity mismatch');
    $('groups').replaceChildren();
    for(const [k,g]of Object.entries(full.gradient.groups)) {
      if(g.norm==null)continue;const tr=document.createElement('tr');
      for(const value of [label(k),fmt(g.norm),`${(g.squared_norm_share*100).toFixed(2)}%`,g.n_tokens,fmt(g.rms)]) {const td=document.createElement('td');td.textContent=value;tr.append(td);}
      $('groups').append(tr);
    }
    const tokens=full.gradient.tokens||[],max=Math.max(1e-12,...tokens.map(t=>t.norm));
    $('tokens').replaceChildren(...tokens.map(t=>{const span=document.createElement('span');span.className='token';span.textContent=t.text;span.title=`${label(t.group)} · norm ${fmt(t.norm)}`;span.style.background=`rgba(47,128,173,${.05+.55*Math.sqrt(t.norm/max)})`;return span;}));
    $('detail-status').textContent='Groups partition the squared total norm. RMS accounts for token count and embedding width.';
  } catch(error) {if(version===detailVersion)$('detail-status').textContent=`Could not load saved details: ${error.message}`;}
}
$('chart').addEventListener('click',event=>{
  if(!positions.length||$('view').value!=='sorted')return;
  const pt=$('chart').createSVGPoint();pt.x=event.clientX;pt.y=event.clientY;
  const at=pt.matrixTransform($('chart').getScreenCTM().inverse());let nearest=positions[0],best=Infinity;
  for(const p of positions){const d=(p.x-at.x)**2+(p.y-at.y)**2;if(d<best){best=d;nearest=p;}}
  clearBin(false);select(nearest.i);
});
$('chart').addEventListener('mouseleave',()=>{if(selected>=0)$('hover').textContent=textAt(selected);});
$('breakdown').addEventListener('toggle',()=>{if($('breakdown').open)loadDetails();else detailVersion++;});
for(const id of ['component','view','axis','phase','source','transition','n-bins'])$(id).addEventListener('change',()=>rebuild());
$('colour').addEventListener('change',renderChart);
$('normalization').addEventListener('change',()=>{clearBin(false);renderChart();if(selected>=0)select(selected);});
let filterTimer;
$('text-filter').addEventListener('input',()=>{clearTimeout(filterTimer);filterTimer=setTimeout(()=>rebuild(),180);});
$('clear-filters').addEventListener('click',()=>{$('phase').value='';$('source').value='';$('text-filter').value='';$('transition').value='all';rebuild();});
$('clear-bin').addEventListener('click',()=>clearBin());
document.querySelectorAll('button[data-q]').forEach(b=>b.addEventListener('click',()=>choose(Number(b.dataset.q))));
$('rank').addEventListener('change',()=>selectRank(Number($('rank').value)-1));
$('rank-slider').addEventListener('input',()=>selectRank(Number($('rank-slider').value)));
$('lower').addEventListener('click',()=>selectRank(Number($('rank').value)-2));
$('higher').addEventListener('click',()=>selectRank(Number($('rank').value)));
$('zoom').addEventListener('click',()=>{if(selected<0)return;const at=logNorm(norm(rows[selected])),d=zoomDomain||fullDomain,half=Math.max(.0001,(d[1]-d[0])/4);zoomDomain=[Math.max(fullDomain[0],at-half),Math.min(fullDomain[1],at+half)];rebuild(false);});
$('reset').addEventListener('click',()=>rebuild());
$('coverage').textContent=`Checkpoint ${data.summary.checkpoint_step} · ${rows.length.toLocaleString()} frames · ${data.episodes.length} episodes · ${data.summary.grad_n_subtasks} subtasks. Select a point or bin to inspect its frames.`;
$('transition-coverage').textContent=`Saved-data coverage: ${data.summary.delta_value_n??0} ΔV pairs; ${data.summary.advantage_n??0} raw advantages. ${Object.entries(data.summary.transition_coverage||{}).map(([k,v])=>`${k.replaceAll('_',' ')}: ${v}`).join(' · ')}.`;
$('axis').value='segment_progress';
const query=new URLSearchParams(location.search);
for(const [key,id]of [['view','view'],['component','component'],['axis','axis'],['phase','phase'],['transition','transition'],['bins','n-bins']]){
 const value=query.get(key);if(value!=null&&[...$(id).options].some(o=>o.value===value))$(id).value=value;
}
rebuild();
if(query.has('lo')&&query.has('hi')){
 const lo=Number(query.get('lo')),hi=Number(query.get('hi')),inclusive=query.get('inclusive')==='true';
 if(Number.isFinite(lo)&&Number.isFinite(hi)&&lo<=hi){
  const members=plotted.filter(p=>{const v=field(rows[p.i],$('axis').value);return valid(v)&&v>=lo&&(v<hi||(inclusive&&v<=hi));}).map(p=>p.i);
  if(members.length){
   activeBin={id:'report-bin',indices:members};rankList=members;$('bin-note').hidden=false;
   $('bin-note').querySelector('span').textContent=`Selected report bin: ${labels[$('axis').value]} [${fmt(lo)}, ${fmt(hi)}${inclusive?']':')'} · ${members.length} frames. The curve/table uses this page's filtered bin edges; browsing retains the report's exact selection.`;
   select(members[Math.floor(members.length/2)]);
  }
 }
}


function chooseCurveBin(b) {
  if(!b.indices.length)return;
  activeBin=b;rankList=[...b.indices];$('bin-note').hidden=false;
  $('bin-note').querySelector('span').textContent=`${labels[$('axis').value]} ${fmt(b.lo)}–${fmt(b.hi)} · ${b.n_frames} frames from ${b.n_episodes} episodes · median gradient ${fmt(b.median)}`;
  select(rankList[Math.floor(rankList.length/2)]);
}
function renderCurves() {
  const key=$('axis').value;
  curveSummary=GradientStats.summarize(plotted.map(p=>({x:field(rows[p.i],key),y:p.v,episode:rows[p.i].episode_ref,index:p.i})),key,null,Number($('n-bins').value));
  bins=curveSummary.bins.map((b,i)=>({...b,id:`curve-${i}`}));
  const present=bins.filter(b=>b.n_frames), xDomain=key==='segment_progress'?[0,1]:extent(curveSummary.edges);
  const yDomain=extent(present.flatMap(b=>[b.q25,b.q75,b.episode_median].map(logNorm)));
  const {x,y}=axes(xDomain,yDomain,labels[key],`${label($('component').value)} · gradient norm (log)`,false,true);
  const paths=[];let run=[];
  for(const b of bins){if(b.n_frames)run.push(b);else if(run.length){paths.push(run);run=[];}}
  if(run.length)paths.push(run);
  for(const group of paths) {
    const band=[...group.map(b=>`${x(b.x_median)},${y(logNorm(b.q25))}`),...group.slice().reverse().map(b=>`${x(b.x_median)},${y(logNorm(b.q75))}`)].join(' ');
    svg('polygon',{points:band,fill:'#0072b2',opacity:.13,'pointer-events':'none'});
    for(const [stat,colour,dash]of [['median','#0072b2',''],['episode_median','#d55e00','5 4']])
      svg('polyline',{points:group.map(b=>`${x(b.x_median)},${y(logNorm(b[stat]))}`).join(' '),fill:'none',stroke:colour,'stroke-width':2,'stroke-dasharray':dash,'pointer-events':'none'});
  }
  $('bin-table').querySelector('tbody').replaceChildren();
  for(const b of bins) {
    if(b.n_frames) {
      const xx=x(b.x_median),yy=y(logNorm(b.median));
      svg('line',{x1:xx,x2:xx,y1:y(logNorm(b.q25)),y2:y(logNorm(b.q75)),stroke:'#0072b2','stroke-width':2,'pointer-events':'none'});
      const dot=svg('circle',{cx:xx,cy:yy,r:6,fill:'#0072b2','data-bin':b.id,'data-curve-bin':b.id,style:'cursor:pointer'});
      dot.addEventListener('click',event=>{event.stopPropagation();chooseCurveBin(b);});
      dot.addEventListener('mouseenter',()=>$('hover').textContent=`${b.n_frames} frames · ${b.n_episodes} episodes · median ${fmt(b.median)} · middle 50% ${fmt(b.q25)}–${fmt(b.q75)} · episode median ${fmt(b.episode_median)}`);
      for(const i of b.indices)positions.push({i,x:xx,y:yy});
    }
    const tr=document.createElement('tr');tr.dataset.curveRow=b.id;
    for(const val of [`[${fmt(b.lo)}, ${fmt(b.hi)}${b.upper_inclusive?']':')'}`,b.n_frames,b.n_episodes,fmt(b.median),fmt(b.q25),fmt(b.q75),fmt(b.mean),fmt(b.episode_median)]) {
      const td=document.createElement('td');td.textContent=val;tr.append(td);
    }
    if(b.n_frames){tr.style.cursor='pointer';tr.addEventListener('click',()=>chooseCurveBin(b));}
    $('bin-table').querySelector('tbody').append(tr);
  }
  $('legend').replaceChildren();legendItem('#0072b2','Median gradient; band = frame 25th–75th percentiles');legendItem('#d55e00','Dashed: median of episode medians');
  $('numeric-summary').textContent=`${curveSummary.n_frames.toLocaleString()} frames · ${curveSummary.n_episodes} episodes · Spearman ρ (${labels[key]}, gradient) = ${fmt(curveSummary.spearman)} · ${curveSummary.method}.`;
  $('plot-note').textContent='Each point summarizes an X bin, positioned at its median X. Click a point or table row to inspect the underlying frames. Statistics use the current filters. The pooled correlation describes association; the per-action filter can reveal different trends.';
}
$('export-bins').addEventListener('click',()=>{
  if(!curveSummary)return;
  const keys=['lo','hi','upper_inclusive','n_frames','n_episodes','x_median','median','q25','q75','mean','episode_median'];
  const context={axis:$('axis').value,component:$('component').value,action:$('phase').value||'all',subtask_filter:$('text-filter').value,source:$('source').value||'all',transitions:$('transition').value,binning:curveSummary.method};
  const escape=v=>`"${String(v??'').replaceAll('"','""')}"`;
  const csv=[[...Object.keys(context),...keys].map(escape).join(','),...curveSummary.bins.map(b=>[...Object.values(context),...keys.map(k=>b[k])].map(escape).join(','))].join('\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const link=document.createElement('a');link.href=url;link.download=`gradient_bins_${$('axis').value}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});

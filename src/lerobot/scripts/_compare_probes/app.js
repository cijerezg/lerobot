(() => {
  const local = { search: '' };
  const h = value => ProbeCompare.h(value);
  const btn = (action, value, label, active = false, extra = '') => `<button data-action="${h(action)}" data-value="${h(value)}" class="${active ? 'active' : ''}" aria-pressed="${active}" ${extra}>${label}</button>`;
  const rawProbe = (ri) => ProbeCompare.data.runs[ri].probes.find(p => p.id === ProbeCompare.probeId);
  const steps = () => [...new Set(ProbeCompare.data.runs.flatMap(r => r.steps.map(s => +s.step)))].sort((a,b) => a-b);
  const commonSteps = () => steps().filter(s => ProbeCompare.data.runs.every(r => r.steps.some(t => +t.step === s)));
  function historicalMetrics() {
    const found = new Map();
    ProbeCompare.data.runs.forEach((r, ri) => (rawProbe(ri)?.metrics || []).forEach(m => { if(!found.has(m.key)) found.set(m.key,m); }));
    return [...found.values()].sort((a,b) => Number(b.primary)-Number(a.primary));
  }
  const icons = {
    map: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/></svg>',
    image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="8" cy="9" r="1.5"/><path d="m4 18 6-6 4 4 3-3 4 5"/></svg>',
    video: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="5"/><path d="m10 8 6 4-6 4Z"/></svg>',
    html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="m9 6-6 6 6 6m6-12 6 6-6 6"/></svg>'
  };
  function trace(ri) {
    const metric = rawProbe(ri)?.metrics?.find(m => m.key === ProbeCompare.metric);
    const samples = ProbeCompare.data.runs[ri].steps.map(s => ({step:+s.step,value:metric?.values?.[String(s.step)]})).sort((a,b)=>a.step-b.step);
    const valid = samples.filter(s=>Number.isFinite(s.value));
    if(valid.length < 2) return '';
    const min=Math.min(...valid.map(s=>s.value)), max=Math.max(...valid.map(s=>s.value));
    const first=samples[0].step, last=samples.at(-1).step;
    const x=s=>4+(s.step-first)/(last-first||1)*100, y=s=>29-(s.value-min)/(max-min||1)*23;
    let pen=false;
    const path=samples.map(s=>{if(!Number.isFinite(s.value)){pen=false;return '';}const cmd=pen?'L':'M';pen=true;return `${cmd}${x(s).toFixed(1)},${y(s).toFixed(1)}`;}).join(' ');
    const selected=valid.find(s=>s.step===+ProbeCompare.steps[ri]);
    return `<svg class="run-trace" viewBox="0 0 108 36" role="img" aria-label="${h(metric.label)} across recorded checkpoints"><title>${h(metric.label)} · each run uses its own vertical scale</title><path class="trace-floor" d="M4 32H104"/><path d="${path}"/>${selected?`<circle cx="${x(selected)}" cy="${y(selected)}" r="3"/>`:''}</svg>`;
  }
  function tile(ri, step) {
    const run = ProbeCompare.data.runs[ri], exists = run.steps.some(s => +s.step === step), p = rawProbe(ri);
    if (!exists) return `<div class="step-tile absent" data-run="${ri}"><span class="tile-step">${h(step.toLocaleString())}</span><span class="no-data">Checkpoint absent</span></div>`;
    const panels = p?.panels?.[String(step)], pn = panels?.find(p => (p.align || p.file) === ProbeCompare.panelKey);
    const imagePanel = pn?.kind === 'image' ? pn : panels?.find(p => p.kind === 'image');
    const m = p?.metrics?.find(m => m.key === ProbeCompare.metric), value = m?.values?.[String(step)];
    const selected = +ProbeCompare.steps[ri] === step;
    return `<button data-action="step" data-run="${ri}" data-value="${step}" class="step-tile ${selected ? 'selected' : ''}" aria-label="${h(run.run.label)} checkpoint ${step}" aria-pressed="${selected}" ${ProbeCompare.locked && !commonSteps().includes(step) ? 'disabled title="Unlock checkpoints to select a step absent from another run"' : ''}>
      <span class="tile-top"><span class="tile-step">${h(step.toLocaleString())}</span><span class="tile-state">${selected ? '● SELECTED' : '↗'}</span></span>
      ${imagePanel ? `<img loading="lazy" src="${h(imagePanel.url)}" alt="${h(imagePanel.caption || imagePanel.file)} at step ${step}">` : `<span class="no-data">${!panels ? 'Probe not recorded' : pn?.kind === 'video' ? 'Video · open aligned view to play' : pn ? `${h(pn.kind)} · open aligned view` : 'No image for this probe'}</span>`}
      <span class="tile-reading">${h(ProbeCompare.format(value, m?.fmt ?? 4))}<small>${h(m?.label || 'No metric available')}</small></span>
      <span class="tile-caption">${imagePanel ? `${pn === imagePanel ? '' : 'Preview: '}${h(imagePanel.caption || imagePanel.file)}` : h(pn?.caption || 'No preview asset at this checkpoint')}</span>
    </button>`;
  }
  function render() {
    const probes = ProbeCompare.probes(), current = probes.find(p => p.id === ProbeCompare.probeId);
    const metrics = historicalMetrics();
    if(!metrics.some(m => m.key === ProbeCompare.metric)) ProbeCompare.metric = metrics[0]?.key || null;
    const allSteps = steps(), common = commonSteps(), panels = ProbeCompare.panels();
    document.getElementById('app').innerHTML = `
      <header class="mast"><span class="identity"><span class="mark">${icons.map}</span><span>Checkpoint<span class="identity-light"> / map</span></span></span><span class="mast-note">A field guide to your training runs</span><div class="utilities">${btn('sim','','Sim outputs')}${btn('copy','','Copy link')}${btn('refresh','','Refresh')}</div></header>
      <div class="workspace"><aside class="probe-rail"><div class="rail-title">PROBES <span>${probes.length}</span></div><label class="search-label">Find a probe<input id="probe-search" type="search" value="${h(local.search)}" placeholder="Filter by name or group"></label><nav id="probe-list">${probes.map((p,pi) => `<button data-action="probe" data-value="${h(p.id)}" data-search="${h((p.title+' '+p.group).toLowerCase())}" class="probe-item ${p.id === ProbeCompare.probeId ? 'active' : ''}" aria-pressed="${p.id===ProbeCompare.probeId}" ${local.search && !(p.title+' '+p.group).toLowerCase().includes(local.search.toLowerCase())?'hidden':''}><span>${h(p.group || 'Probe')} <small>${p.count}/${ProbeCompare.data.runs.length} runs</small></span><strong><i class="probe-number">${String(pi+1).padStart(2,'0')}</i>${h(p.title)}</strong></button>`).join('')}</nav><div class="archive">${btn('archive','',ProbeCompare.archived ? 'Hide Jacobian archive' : 'Show Jacobian archive',ProbeCompare.archived)}</div></aside>
      <section class="work"><div class="titlebar"><div><span class="eyebrow">EXPERIMENT EXPLORER <span class="eyebrow-rule"></span> ${String(ProbeCompare.data.runs.length).padStart(2,'0')} RUNS / ${allSteps.length} CHECKPOINTS</span><h1>${h(current?.title || 'No probes available')}</h1>${current?.claim ? `<p>${h(current.claim)}</p>` : ''}</div><div class="mode-buttons">${btn('mode','map',`${icons.map} Map`,ProbeCompare.view==='map')}${btn('mode','aligned','<span class=aligned-icon>◫</span> Aligned view',ProbeCompare.view==='aligned')}</div></div>
      <section class="panel-strip" aria-label="Visible output choices"><div class="strip-label">OUTPUTS <span>${panels.length}</span></div><div class="output-buttons">${panels.map(p=>btn('panel',p.key,`<span class="output-icon">${icons[p.kind]||icons.html}</span><span class="output-copy"><small>${h(p.kind)} <b>· ${p.count}/${ProbeCompare.data.runs.length} runs</b></small><span class="output-caption">${h(p.caption)}</span><code>${h(p.key)}</code></span>`,p.key===ProbeCompare.panelKey)).join('') || '<p>No outputs at the selected steps.</p>'}</div></section>
      <div class="alignment-tools"><div class="segmented">${btn('shared','union','All outputs',!ProbeCompare.shared)}${btn('shared','shared','Shared outputs',ProbeCompare.shared)}</div>${btn('lock','',ProbeCompare.locked?'● Checkpoints locked':'○ Independent checkpoints',ProbeCompare.locked)}<span class="lock-help">${ProbeCompare.locked?'A tile moves every run to that common step.':'Click any tile to move only that run.'}</span></div>
      <section class="common-lane"><span>ALL RUNS AT</span><div>${common.length ? common.map(step=>btn('common',step,h(step.toLocaleString()),ProbeCompare.steps.every(s=>+s===step))).join('') : '<em>No common checkpoints. Use each run independently.</em>'}</div></section>
      <div class="run-settings">${ProbeCompare.data.runs.map((r,ri)=>`<div class="run-setting" data-run="${ri}"><span class="run-marker">${ri+1}</span><div class="run-name"><small>RUN ${String(ri+1).padStart(2,'0')}</small><strong>${h(r.run.label)}</strong></div>${trace(ri)}<span class="selected-step"><small>CHECKPOINT</small>${h(Number(ProbeCompare.steps[ri]).toLocaleString())}</span>${btn('baseline',ri,ProbeCompare.baseline===ri?'● Baseline':'Set baseline',ProbeCompare.baseline===ri)}</div>`).join('')}</div>
      ${ProbeCompare.view === 'map' ? `<section class="storyboard"><div class="map-tools"><div><strong>Through training<span class="section-dot">.</span></strong><span>${allSteps.length} recorded steps · scroll horizontally to inspect history →</span></div><span class="selection-key"><i></i> Selected checkpoint</span></div><div class="metric-choices"><span>TILE METRIC</span>${metrics.map(m=>btn('metric',m.key,h(m.label),m.key===ProbeCompare.metric)).join('') || '<em>No recorded numeric metrics</em>'}</div><div class="matrix-scroll" id="matrix"><div class="matrix" style="--steps:${allSteps.length}"><div class="corner">RUN / STEP →</div>${allSteps.map(step=>`<div class="column-heading"><i></i><span>${h(step.toLocaleString())}</span></div>`).join('')}${ProbeCompare.data.runs.map((r,ri)=>`<div class="run-heading" data-run="${ri}"><span class="run-marker">${ri+1}</span><h2>${h(r.run.label)}</h2><span>${r.steps.length} checkpoints</span><p>${h(rawProbe(ri)?.title || 'Probe unavailable')}</p></div>${allSteps.map(step=>tile(ri,step)).join('')}`).join('')}</div></div><p class="matrix-note">Each tile uses evidence from its own checkpoint. When the selected output is not an image, a labeled image from the same probe supplies the preview. Missing checkpoints and probe outputs are shown explicitly.</p></section><section class="compare-dock"><div><strong>Compare the selected checkpoints</strong><p>${ProbeCompare.data.runs.map((r,ri)=>`${h(r.run.label)} · ${h(ProbeCompare.steps[ri])}`).join(' &nbsp;↔&nbsp; ')}</p></div>${btn('mode','aligned','Open aligned evidence ↗')}</section>` : `<section class="aligned-steps">${ProbeCompare.data.runs.map((r,ri)=>`<div data-run="${ri}"><strong><span class="run-marker">${ri+1}</span> ${h(r.run.label)}</strong><div class="step-buttons">${r.steps.map(s=>btn('step',s.step,h(Number(s.step).toLocaleString()),+ProbeCompare.steps[ri]===+s.step,`data-run="${ri}" ${ProbeCompare.locked&&!common.includes(+s.step)?'disabled':''}`)).join('')}</div></div>`).join('')}</section><section id="evidence" class="aligned-evidence"></section>`}
      <section class="metrics-section"><div class="metrics-heading"><h2>Measurements at selected checkpoints</h2>${btn('metrics','',ProbeCompare.allMetrics?'Primary metrics':'All metrics',ProbeCompare.allMetrics)}</div><div id="metric-table"></div></section>
      </section></div>`;
    if(ProbeCompare.view === 'aligned') ProbeCompare.evidence(document.getElementById('evidence'));
    const rows = ProbeCompare.metrics();
    ProbeCompare.metricTable(document.getElementById('metric-table'),ProbeCompare.allMetrics?rows:rows.filter(m=>m.primary).length?rows.filter(m=>m.primary):rows.slice(0,4));
    const matrix = document.getElementById('matrix');
    if(matrix) matrix.scrollLeft = local.scrollLeft ?? Math.max(0,(allSteps.indexOf(+ProbeCompare.steps[0])-3)*250);
    ProbeCompare.persist();
  }
  document.addEventListener('input',e=>{if(e.target.id !== 'probe-search')return;local.search=e.target.value;document.querySelectorAll('.probe-item').forEach(el=>el.hidden=!el.dataset.search.includes(local.search.toLowerCase()));});
  document.addEventListener('click',e=>{
    const b=e.target.closest('button[data-action]'); if(!b)return;
    const action=b.dataset.action,value=b.dataset.value;
    const matrix=document.getElementById('matrix');if(matrix)local.scrollLeft=matrix.scrollLeft;
    if(action==='probe')ProbeCompare.setProbe(value);
    else if(action==='panel')ProbeCompare.setPanel(value);
    else if(action==='baseline')ProbeCompare.setBaseline(+value);
    else if(action==='shared')ProbeCompare.setShared(value==='shared');
    else if(action==='archive')ProbeCompare.setArchived(!ProbeCompare.archived);
    else if(action==='copy')ProbeCompare.copyLink();
    else if(action==='refresh')ProbeCompare.refresh();
    else if(action==='sim')ProbeCompare.jumpSim();
    else if(action==='step'){if(ProbeCompare.locked){if(!commonSteps().includes(+value))return;ProbeCompare.setSteps(ProbeCompare.data.runs.map(()=>+value));}else ProbeCompare.setStep(+b.dataset.run,+value);}
    else if(action==='common')ProbeCompare.setSteps(ProbeCompare.data.runs.map(()=>+value));
    else if(action==='mode'){ProbeCompare.view=value;ProbeCompare.emit();}
    else if(action==='metric'){ProbeCompare.metric=value;ProbeCompare.emit();}
    else if(action==='metrics'){ProbeCompare.allMetrics=!ProbeCompare.allMetrics;ProbeCompare.emit();}
    else if(action==='lock'){ProbeCompare.locked=!ProbeCompare.locked;if(ProbeCompare.locked){const common=commonSteps();if(!common.length){ProbeCompare.locked=false;ProbeCompare.flash?.('No common checkpoints are available.');}else{const target=common.includes(+ProbeCompare.steps[0])?+ProbeCompare.steps[0]:common[0];ProbeCompare.setSteps(ProbeCompare.data.runs.map(()=>target));}}ProbeCompare.emit();}
  });
  // Preserve the original viewer's probe/output shortcuts; native controls keep
  // their normal keyboard behavior, including Space on focused buttons.
  document.addEventListener('keydown',event=>{
    if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||!ProbeCompare.data||document.querySelector('dialog[open]'))return;
    if(event.target.closest('input,textarea,select,[contenteditable="true"]'))return;
    const key=event.key.toLowerCase();
    if(key==='j'||key==='k'){
      const rows=ProbeCompare.probes(),at=rows.findIndex(p=>p.id===ProbeCompare.probeId);
      const next=rows[Math.max(0,Math.min(rows.length-1,at+(key==='j'?1:-1)))];
      if(next){event.preventDefault();ProbeCompare.setProbe(next.id);}
    }else if(key==='n'||key==='p'){
      const rows=ProbeCompare.panels(),at=rows.findIndex(p=>p.key===ProbeCompare.panelKey);
      const next=rows[Math.max(0,Math.min(rows.length-1,at+(key==='n'?1:-1)))];
      if(next){event.preventDefault();ProbeCompare.setPanel(next.key);}
    }else if(event.code==='Space'&&!event.target.closest('button,a,video')){
      const play=document.querySelector('#evidence [data-ev-play]');
      if(play){event.preventDefault();play.click();}
    }
  });
  ProbeCompare.start(render);
})();

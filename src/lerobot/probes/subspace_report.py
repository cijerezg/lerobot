"""Read mean-centred representation spans without confusing geometry with task performance.

Each frame contributes one arithmetic mean over a token group. The capture view is
recorded in the cache and displayed above every chart. Native attention-input captures
read actual VLM attention-normalization outputs (including learned scaling) and actual
expert cross-attention inputs after normalization and time modulation. Expert-key/value
views read prepared cross-attention arguments; heads are flattened without changing values.
There is no external normalization, whitening, or unit-length conversion in this analysis.
Native views are pooled and stored in float32. Explicit block_output selects legacy raw
post-block fp16 representations. The action group, when present, is averaged over future
positions at flow time zero with fixed seeded noise; it is not a predicted action.

Layer indices identify the named capture site. Native attention-input L17 is inside
block 17, reading block 16's residual output. Expert K/V at L16 were computed before
VLM block 16's MLP output; its new output first affects expert context L17.

Subspace promotes cached vectors to float64 and centers them on each robot's own
training mean; holdout rows subtract the same mean. Mean norms and between-robot mean
cosines are reported separately. Rescaling rows can change singular values, numerical
rank, and energy residuals. The relative cutoff is a measurement choice: a larger first
singular value can lower reported rank without shrinking the remaining variation.

Training rows here are the reference frames defining the span; they are not a new
training run. The conditions cache samples interior frames per episode/class/phase using the
configured conditions_* budget (defaults: one frame per episode and cell, 250 training
frames per robot filled round-robin over its cells, 1000 total frames). summary.json
"available" gives the episodes each robot could have supplied, the ceiling of its count. Only the selected text condition (real by default) and
present groups enter this report; robots need at least eight reference frames. ReBot
holdout comes from validation windows; corpus holdout follows its episode ledger.
Correlated windows and unequal frame counts limit comparisons. UR7e has no holdout.

Numerical rank k(tau) counts singular values strictly above tau times the largest.
Training residual is discarded squared-singular-value energy divided by total energy;
holdout residual is squared energy outside the training basis divided by holdout energy.
These are energy-weighted fractions of the centred vectors, not average frame errors or task accuracy.

Principal-angle cosines are singular values of Q_A transpose Q_B, sorted largest first:
index 1 is the smallest angle, hence the best-aligned direction. There are min(p,q) values,
so curve length changes with rank, tolerance and available frames. Overlap is their mean
square; a value of one means the smaller span lies inside the larger, not necessarily equality.
The isotropic random-space expectation is max(p,q)/D. This is a geometric reference,
not a task-aware null or a statistical significance test. At least max(0,p+q-D) shared
directions are forced by dimension alone. Cosine >0.9 is an explicitly chosen descriptive
threshold; it does not establish an exactly shared direction.

The angle reference is the pointwise 95th percentile of 100 matched-dimension random
space pairs, not a simultaneous confidence band. The cache-precision floor estimates independent
rounding error: sigma*(sqrt(n)+sqrt(D)), sigma squared = mean(ulp(x)^2)/12. It is a heuristic,
not proof that a direction below it is noise, and does not include upstream bfloat16 error.
Pivoted QR chooses the frame with the largest residual after previous pivots. Its index
is a selection order, not time; pivot frames are geometric examples, not causal evidence.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path
import os
import sys
import numpy as np
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.representation_views import VIEW_METADATA, resolve_view_cache
from lerobot.probes.report_html import CHART_JS, clean, data_json, page, table

GROUPS = {
 'img_external_0': 'External-camera image patch tokens, averaged within each frame; encoder, D=2560.',
 'img_wrist_0': 'Wrist-camera image patch tokens, averaged within each frame; encoder, D=2560. Missing cameras are excluded.',
 'subtask': 'Tokens assigned to the subtask clause, averaged within each frame; encoder, D=2560. Real text can carry object and phase labels.',
 'state': 'Packed state positions, averaged within each frame; encoder, D=2560. BC uses continuous state embeddings; Diverse-v3 uses discrete state tokens.',
 'action_output': 'The single <action_output> readout position in the encoder, D=2560; not the expert trajectory or output joints.',
 'action': 'Action-expert block states, averaged over 30 future-step positions at t=0, D=768. The same Gaussian input noise is used for every frame.'}


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            try:
                row[k] = float(v) if any(c in v.lower() for c in '.en') else int(v)
            except ValueError:
                pass
    return rows


def detail_data(cache_dir, summary, spans, pivots, n_null=100, seed=42):
    """Only the configured detailed layers need new CPU linear algebra. No model imports/run."""
    from lerobot.probes.subspace_spans import centred_matrices, robot_rows, fp16_floor, null_cosines
    meta = json.loads((cache_dir/'meta.json').read_text())
    details, nulls = [], {}
    rng = np.random.RandomState(seed)
    for group in meta['groups']:
        logging.info('Rendering cached detail layers: %s',group)
        arr = np.load(cache_dir/f'{group}.npy', mmap_mode='r')
        present = np.load(cache_dir/f'{group}.present.npy')
        sel = robot_rows(meta['rows'],present,summary['text'])
        for layer in summary['layers']:
            bases = {}
            for robot, ids in sel.items():
                x16, x, _, _ = centred_matrices(arr, ids, layer)
                _, s, vt = np.linalg.svd(x, full_matrices=False)
                k = int((s > summary['headline_tau']*s[0]).sum())
                saved = next(r for r in spans if r['group']==group and r['layer']==layer and r['robot']==robot and r['tau']==summary['headline_tau'])
                if k != saved['k'] or not np.isclose(s[0],saved['s1'],rtol=1e-8):
                    raise ValueError(f'Cache does not reproduce saved span: {group}/{layer}/{robot}')
                bases[robot] = vt[:k].T
                pr = [r for r in pivots if r['group']==group and r['layer']==layer and r['robot']==robot]
                details.append(dict(kind='spectrum',group=group,layer=layer,robot=robot,s=s/s[0],qr=[r['residual_rel'] for r in pr],floor=fp16_floor(x16)/s[0]))
            robots = list(bases)
            for i,a in enumerate(robots):
                for b in robots[i+1:]:
                    qa,qb = bases[a],bases[b]
                    cos = np.linalg.svd(qa.T@qb,compute_uv=False)
                    key=(qa.shape[1],qb.shape[1],arr.shape[2])
                    if key not in nulls:
                        nulls[key]=null_cosines(*key,n_null,rng)
                    details.append(dict(kind='angle',group=group,layer=layer,a=a,b=b,cos=cos,null=nulls[key]))
    return clean(details),meta


def render(output_dir, cache_dir=None, *, n_null=100, seed=42, n_pivots=12):
    out=Path(output_dir)
    summary=json.loads((out/'summary.json').read_text())
    view=summary.get('representation_view','block_output')
    cache=Path(resolve_view_cache(cache_dir or out.parent/'conditions_matrix/cache',view))
    spans=read_csv(out/'spans.csv'); pairs=read_csv(out/'pairs.csv'); pivots=read_csv(out/'pivots.csv')
    detail_path = out/'report_details.json'
    sources = [out/name for name in ('summary.json', 'spans.csv', 'pairs.csv', 'pivots.csv')]
    sources += [cache/'meta.json', *cache.glob('*.npy')]
    cached = json.loads(detail_path.read_text()) if detail_path.exists() else {}
    fresh = detail_path.exists() and detail_path.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in sources)
    if fresh and cached.get('n_null') == n_null and cached.get('seed') == seed:
        details = cached['details']
        meta = json.loads((cache/'meta.json').read_text())
    else:
        details,meta=detail_data(cache,summary,spans,pivots,n_null,seed)
        detail_path.write_text(json.dumps(dict(n_null=n_null,seed=seed,details=details),allow_nan=False))
    actual_view=meta.get('representation_view','block_output')
    if actual_view!=view:
        raise ValueError(f'Report view {view} does not match cache view {actual_view}; reanalyze the selected cache.')
    capture=VIEW_METADATA[view]
    descriptions={}
    for group,shape in meta['groups'].items():
        detail=capture['action'] if group=='action' else capture['encoder']
        descriptions[group]=f'{group}: {detail} Width D={shape[-1]}; arithmetic mean over the selected group positions.'
    precision=meta.get('cache_dtype',str(np.load(cache/f"{next(iter(meta['groups']))}.npy",mmap_mode='r').dtype))
    tensor_context=f"{capture['layers']} {precision} cache; per-robot training-mean centering and float64 analysis. {summary['text']} text."
    robots=list(summary['robots'][summary['headline_group']])
    counts=[]
    for robot in robots:
        rr=[r for r in meta['rows'] if r['robot']==robot and r['text']==summary['text']]
        counts.append([robot, sum(not r['holdout'] for r in rr), len({str(r['episode']) for r in rr if not r['holdout']}), sum(r['holdout'] for r in rr), len({str(r['episode']) for r in rr if r['holdout']})])
    data=dict(summary=summary,spans=spans,pairs=pairs,pivots=pivots,details=details,groups=descriptions,robots=robots,counts=counts,
              tensor_context=tensor_context,capture=capture,cache_dtype=precision,
              cache=os.path.relpath(cache,out),thumbs=os.path.relpath(cache/meta.get('thumbs_path','thumbs'),out),n_null=n_null,seed=seed,n_pivots=n_pivots)
    body='''<div class="eyebrow">Representation geometry · saved-cache report</div><h1>Which directions are shared across robots?</h1>
<p class="callout" id="capture-site"></p>
<p class="note">One view at a time. Select a token group, then explore depth, tolerance, or a robot pair. Hover a curve point for its exact value.</p>
<div class="controls"><label>Question<select id="view"><option value="rank">How large is each span?</option><option value="overlap">How much do two spans overlap?</option><option value="residual">Does the span cover unseen frames?</option><option value="spectra">Which singular directions survive?</option><option value="angles">How are the directions aligned?</option><option value="pivots">Which frames define the span?</option></select></label>
<label>Token group<select id="group"></select></label><label id="tauLabel">Tolerance τ<select id="tau"></select></label><label id="layerLabel">Detailed layer<select id="layer"></select></label><label id="pairLabel">Robot pair<select id="pair"></select></label><label id="robotLabel">Robot<select id="robot"></select></label></div>
<p id="context" class="note"></p><div id="finding" class="callout"></div><div id="plot" class="chart"></div><div id="gallery" class="pivots"></div><p id="reading"></p><div id="numbers"></div>
<details><summary>Sampling, token groups and mathematical definitions</summary>DOC</details>
<p><a href="spans.csv">All span measurements (CSV)</a> · <a href="pairs.csv">All pairs, layers and tolerances (CSV)</a> · <a href="pivots.csv">Every pivot frame (CSV)</a> · <a href="report_details.json">Detailed curves and reference provenance (JSON)</a></p>'''
    import html
    body=body.replace('DOC', '<p>Cache counts before token-presence filtering (the selected view gives actual group counts). One episode can contribute several class/phase frames.</p>' + table(['Robot', 'Reference frames', 'Reference episodes', 'Holdout frames', 'Holdout episodes'], counts) + ''.join('<p>'+html.escape(p)+'</p>' for p in __doc__.split('\n\n')))
    page(out/'explorer.html','Subspace spans',body,'const D='+data_json(data)+';\n'+CHART_JS+SCRIPT)
    g=summary['headline_group']; layer=summary['headline_layer']; tau=summary['headline_tau']
    metrics=[Metric('rank',f'ReBot rank · {g}, L{layer}, τ={tau}',value=summary['dimension'][g]['rebot']['k'],fmt=0,primary=True,
        note='Frame-count limited; lower is not inherently better.')]
    for pair, r in summary['overlap'][g].items():
        if pair.startswith('rebot_'):
            metrics.append(Metric(pair, pair.replace('_',' vs ')+' overlap',value=r['overlap'],fmt=4,note=f"Random expectation {r['null_overlap']:.4f}; {r['shared']} / {min(r['p'],r['q'])} cosines >0.9. L{layer}, τ={tau}, {g}."))
    panels=[Panel('explorer.html','Explore the centred spans, one readable chart at a time',primary=True,
        how='Selectors preserve access to every group, all captured layers and configured tolerances. Angle spectra use the saved middle tolerance at the configured detailed layers. Tables show exact counts and energy residuals; downloads preserve all measurements.')]
    if (out/'comparison.html').exists():
        panels.append(Panel('comparison.html','BC 1400 versus Diverse-v3 1200: measured results and limitations',how='Matched report settings; checkpoints differ in state encoding and other saved settings, so this is not a controlled training ablation.'))
    return write_index(str(out),sys.modules[__name__],title='Subspace spans',group='Representation',
        claim='Shared directions are present; neither overlap nor low residual alone establishes a shared task representation.',summary=summary,metrics=metrics,panels=panels,status='info',
        extra={'viewer': {'show_headlines': False}, 'provenance':{'sampling':f"Conditions-matrix sampling budget; {summary['text']} text only. Present masks exclude missing groups. Exact counts appear in the explorer.",
            'details':[['Cache',str(cache)],['Reference draws',f'{n_null} isotropic random pairs per dimension tuple; report seed {seed}. Recomputed on CPU; pointwise references can differ from the original Monte Carlo draw.'],['Capture site',capture['label']],['Layer convention',capture['layers'] + ' Detail ' + ', '.join(f'L{layer}' for layer in summary['layers']) + '.']] }},see_also=['conditions_matrix'])


SCRIPT = r'''
const $=id=>document.getElementById(id);
$('capture-site').textContent=D.capture.label+' · '+D.capture.encoder+' '+D.capture.layers;
function options(id,vals,labels){$(id).innerHTML=vals.map((v,i)=>`<option value="${esc(v)}">${esc(labels?labels[i]:v)}</option>`).join('')}
const groups=Object.keys(D.summary.robots), pairs=[...new Set(D.pairs.map(r=>r.a+'|'+r.b))];
options('group',groups);$('group').value=D.summary.headline_group;options('tau',D.summary.tolerances);$('tau').value=D.summary.headline_tau;options('layer',D.summary.layers);$('layer').value=D.summary.headline_layer;options('pair',pairs,pairs.map(p=>p.replace('|',' ↔ ')));options('robot',D.robots);
const color=r=>palette[D.robots.indexOf(r)%palette.length];
function draw(){
 const v=$('view').value,g=$('group').value,t=Number($('tau').value),l=Number($('layer').value),[a,b]=$('pair').value.split('|'),robot=$('robot').value;
 $('tauLabel').hidden=['spectra','angles','pivots'].includes(v);$('layerLabel').hidden=['rank','overlap','residual'].includes(v);$('pairLabel').hidden=!['overlap','angles'].includes(v);$('robotLabel').hidden=!['spectra','residual','pivots'].includes(v);
 const ss=D.spans.filter(r=>r.group===g&&r.tau===t), ps=D.pairs.filter(r=>r.group===g&&r.tau===t&&r.a===a&&r.b===b), end=ss.filter(r=>r.layer===D.summary.headline_layer);
 const rankMax=Math.max(...D.spans.map(r=>Math.min(r.n_frames,r.d))), lastLayer=Math.max(...D.spans.map(r=>r.layer));
 $('context').textContent=D.groups[g]+' '+D.tensor_context;
 $('gallery').innerHTML='';$('plot').innerHTML='';let lines=[],spec={},reading='',finding='',headers=[],rows=[];
 if(v==='rank'){
  for(const r of D.robots){let z=ss.filter(x=>x.robot===r);if(!z.length)continue;lines.push({name:r,x:z.map(x=>x.layer),y:z.map(x=>x.k),color:color(r)});lines.push({name:r+' frame ceiling ('+z[0].n_frames+')',x:[0,lastLayer],y:[z[0].n_frames,z[0].n_frames],color:color(r),dash:'2 6',width:1})}
  spec={title:`Numerical rank · ${g} · τ=${t}`,xlabel:'Block index (zero-based)',ylabel:'Retained directions k(τ)',xmax:lastLayer,ymax:rankMax};
  reading='Solid: k = number of singular values strictly above τ × s₁. Dotted: each robot’s training-frame ceiling min(n,D). The common vertical scale is fixed across groups and tolerances. A ceiling-level rank is sample-limited, not evidence of a high-dimensional full population.';
  finding='At L'+D.summary.headline_layer+', ReBot retains '+end.find(r=>r.robot==='rebot').k+' of '+end.find(r=>r.robot==='rebot').n_frames+' possible frame directions. Change τ to see how strongly that count depends on the cutoff.';
  headers=['Robot · L'+D.summary.headline_layer,'Reference frames','Holdout frames','Rank','Train outside (%)','Holdout outside (%)'];rows=end.map(r=>[r.robot,r.n_frames,r.n_holdout,r.k,fmt(100*r.train_residual),r.holdout_residual===null?'— no holdout':fmt(100*r.holdout_residual)]);
 } else if(v==='overlap'){
  lines=[{name:a+' ↔ '+b,x:ps.map(r=>r.layer),y:ps.map(r=>r.overlap),color:palette[0]},{name:'Random expectation max(p,q)/D',x:ps.map(r=>r.layer),y:ps.map(r=>r.null_overlap),color:palette[1],dash:'8 5'}];
  spec={title:`Span overlap · ${a} ↔ ${b} · τ=${t}`,xlabel:'Block index (zero-based)',ylabel:'Mean squared principal cosine',xmax:lastLayer};
  let r=ps.find(r=>r.layer===D.summary.headline_layer);finding=`At L${r.layer}: overlap ${fmt(r.overlap)} versus random expectation ${fmt(r.null_overlap)}; ${r.shared} of ${Math.min(r.p,r.q)} cosines exceed 0.9. This is partial alignment, not equal spans.`;
  reading='Solid: mean cos² over min(p,q) directions. Dashed: expected overlap for independent isotropic subspaces with these ranks and ambient width. Both use a fixed 0–1 scale. Above-reference overlap is descriptive, not a p-value. The shared raw mean direction is removed before the spans are formed; the cosine between the two robots’ raw means is in the table. A↔B geometry is symmetric, but energy coverage is directional.';
  headers=['Layer','Rank A','Rank B','Overlap','Random','cos >0.9','Raw means cos','B outside A (%)','A outside B (%)'];rows=ps.map(r=>[r.layer,r.p,r.q,fmt(r.overlap),fmt(r.null_overlap),r.shared,fmt(r.mean_cos),fmt(100*r.residual_b_outside_a),fmt(100*r.residual_a_outside_b)]);
 } else if(v==='residual'){
  let z=ss.filter(r=>r.robot===robot);lines=[{name:'Reference / training',x:z.map(r=>r.layer),y:z.map(r=>r.train_residual),color:palette[0]},{name:'Holdout',x:z.map(r=>r.layer),y:z.map(r=>r.holdout_residual),color:palette[1],dash:'8 5'}];
  spec={title:`Energy outside the span · ${robot} · τ=${t}`,xlabel:'Block index (zero-based)',ylabel:'Fraction of centred energy',xmax:lastLayer};
  let r=z.find(r=>r.layer===D.summary.headline_layer);finding=r.n_holdout?`At L${r.layer}: ${fmt(100*r.train_residual)}% reference energy and ${fmt(100*r.holdout_residual)}% holdout energy remain outside the ${r.k}-direction basis.`:'No holdout frames for this robot. A missing curve is not a zero residual.';
  reading='Training = discarded Σsᵢ² / total Σsᵢ². Holdout = 1 − ‖Xhold Qk‖²F / ‖Xhold‖²F, using the training basis unchanged. Lower means better energy coverage, not task generalisation. Curves share a 0–1 scale; exact small values remain in the table. Training residuals near zero are expected when rank reaches the frame count.';
  headers=['Layer','Rank','Train outside (%)','Holdout outside (%)','Reference / holdout n'];rows=z.map(r=>[r.layer,r.k,fmt(100*r.train_residual),r.holdout_residual===null?'—':fmt(100*r.holdout_residual),r.n_frames+' / '+r.n_holdout]);
 } else if(v==='spectra'){
  let r=D.details.find(r=>r.kind==='spectrum'&&r.group===g&&r.layer===l&&r.robot===robot);const x=r.s.map((_,i)=>i+1);
  lines=[{name:'Singular values sᵢ / s₁',x,y:r.s,color:palette[0]},{name:'Pivot residual |Rⱼⱼ| / |R₁₁|',x,y:r.qr,color:palette[1],dash:'2 5'},...D.summary.tolerances.map(t=>({name:'τ='+t,x:[1,rankMax],y:[t,t],color:'#a9b9c9',dash:'8 5',width:1})),{name:'Estimated cache rounding floor',x:[1,rankMax],y:[r.floor,r.floor],color:palette[3],dash:'10 4 2 4'}];
  spec={title:`Centred spectrum · ${robot} · ${g} · L${l}`,xlabel:'Singular-value rank / QR pivot order (one-based)',ylabel:'Relative amplitude (log scale)',xmax:rankMax,ymin:1e-7,ymax:1,log:true};
  finding=`${r.s.length} singular values; estimated cache rounding floor / s₁ = ${r.floor.toExponential(2)}. Different robots have different curve lengths because they contribute different numbers of frames.`;
  reading='Solid blue: descending singular values of the mean-centred frame matrix. Dotted orange: greedy frame residual sequence, separately normalised by its first pivot. Grey dashed lines mark all cutoffs; purple dash-dot is this robot’s estimated rounding floor, not a median across robots. The two sequence indices are orders, not matching directions. Curves share fixed axes across groups, robots and models.';
  headers=['Index','sᵢ / s₁','QR residual ratio'];rows=r.s.map((s,i)=>[i+1,fmt(s,7),fmt(r.qr[i],7)]);
 } else if(v==='angles'){
  let r=D.details.find(r=>r.kind==='angle'&&r.group===g&&r.layer===l&&r.a===a&&r.b===b);let p=D.pairs.find(r=>r.group===g&&r.layer===l&&r.tau===D.summary.headline_tau&&r.a===a&&r.b===b),x=r.cos.map((_,i)=>i+1);
  lines=[{name:'Measured cos(θᵢ)',x,y:r.cos,color:palette[0]},{name:D.n_null+' random draws: pointwise 95th percentile',x,y:r.null,color:palette[1],dash:'8 5'},{name:'Descriptive cos >0.9 threshold',x:[1,rankMax],y:[.9,.9],color:'#a9b9c9',dash:'2 5'}];
  spec={title:`Principal angles · ${a} ↔ ${b} · L${l} · τ=${D.summary.headline_tau}`,xlabel:'Principal-angle index (smallest angle first)',ylabel:'Cosine of principal angle',xmax:rankMax};
  finding=`Ranks p=${p.p}, q=${p.q}; ${r.cos.length} angles. ${p.shared} cosines exceed 0.9. Dimension alone forces at least ${Math.max(0,p.p+p.q-p.d)} intersecting directions.`;
  reading='Cosines descend from best to worst alignment (angles ascend). 1 means aligned; 0 means orthogonal. There are min(p,q) values, so shorter curves do not mean missing measurements. Same axes in every selection. Dashed random curve is a pointwise 95th percentile, not a simultaneous confidence band or a corrected hypothesis test. Detailed angle curves use only the saved middle tolerance; overlap tables retain all tolerances.';
  headers=['Angle index','Measured cosine','Random pointwise p95'];rows=r.cos.map((c,i)=>[i+1,fmt(c,6),fmt(r.null[i],6)]);
 } else {
  let r=D.pivots.filter(r=>r.group===g&&r.layer===l&&r.robot===robot);finding='The first '+D.n_pivots+' greedy frame pivots for '+robot+'; all '+r.length+' pivots remain in the table and CSV.';
  $('gallery').innerHTML=r.slice(0,D.n_pivots).map(p=>`<figure><img loading="lazy" src="${esc(D.thumbs)}/${String(Math.floor(p.row/2)).padStart(4,'0')}.external_0.jpg" alt="External camera, pivot ${p.pivot}" onerror="this.replaceWith(document.createTextNode('Thumbnail unavailable'))"><figcaption><b>#${p.pivot} · residual ${fmt(p.residual_rel,3)}</b><br>${esc(p.object_class)} / ${esc(p.phase)}<br>${esc(p.episode)} · frame ${esc(p.frame)}</figcaption></figure>`).join('');
  reading='Column-pivoted QR of Xᵀ: choose the frame with maximum remaining residual after earlier pivots. Residual ratios divide by the first pivot norm; they are not singular values. External-camera thumbnails identify the frame even when another token group defines the geometry. The order is geometric novelty, not chronology or task importance.';
  headers=['Pivot','Cache row','Episode','Frame / corpus time (s)','Class','Phase','Residual ratio'];rows=r.map(p=>[p.pivot,p.row,p.episode,p.frame,p.object_class,p.phase,fmt(p.residual_rel,7)]);
 }
 if(lines.length)$('plot').innerHTML=chart(lines,spec);$('finding').textContent=finding;$('reading').textContent=reading;$('numbers').innerHTML=table(headers,rows);
}
for(const id of ['view','group','tau','layer','pair','robot'])$(id).addEventListener('change',draw);
let resizeFrame;window.addEventListener('resize',()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(draw)});draw();
'''

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Rebuild the subspace report from saved CSVs and caches; no model inference.')
    p.add_argument('--n_pivots',type=int,default=12);p.add_argument('--output_dir',required=True);p.add_argument('--cache_dir');p.add_argument('--n_null',type=int,default=100);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();logging.basicConfig(level=logging.INFO);render(a.output_dir,a.cache_dir,n_null=a.n_null,seed=a.seed,n_pivots=a.n_pivots)

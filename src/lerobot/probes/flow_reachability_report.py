"""Reachability of arbitrary action targets, independent of the input prior."""
from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lerobot.probes.flow_roundtrip_report import COLORS, EXPLORER_JS, label
from lerobot.probes.flow_synthetic import DISTRIBUTIONS
from lerobot.probes.report_html import data_json, page


def render(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lerobot.probes.utils import joint_names_for_dim

    out = Path(out_dir)
    data = json.loads((out/'roundtrip.json').read_text())
    header, all_rows = data['header'], data['records']
    rows = [dict(r) for r in all_rows if r['start_step']==0 and r['target_kind'] in header['synthetic_distributions']]
    settings = sorted({(r['num_steps'],r['refine']) for r in rows})
    kinds = list(header['synthetic_distributions'])
    titles = {k: DISTRIBUTIONS[k][0] for k in header['synthetic_distributions']}
    titles.update(demo='Recorded actions', generated='Model-generated actions')
    ids = list(dict.fromkeys(r['global_idx'] for r in rows))
    by_frame = {f:[r for r in rows if r['global_idx']==f] for f in ids}
    errors, targets, examples = defaultdict(list), defaultdict(list), []
    for i,f in enumerate(ids):
        with np.load(out/f'trajectories_{i:04d}.npz') as a:
            valid = a['valid'].astype(bool)
            for kind in kinds:
                paths = {}
                for n,r in settings:
                    key=f'{kind}_n{n}_r{r}'
                    target=a[key+'_inverse'][-1,0][...,valid].astype(float)
                    recon=a[key+'_deployed'][0][...,valid].astype(float)
                    delta=recon-target
                    errors[kind,n,r].append(delta)
                    row=next(v for v in by_frame[f] if (v['target_kind'],v['num_steps'],v['refine'])==(kind,n,r))
                    row.update(mse=float(np.mean(delta**2)),max_abs=float(np.abs(delta).max()),
                               fp32_mse=float(np.mean((a[key+'_forward_fp32'][-1,0][...,valid].astype(float)-target)**2)))
                    paths[f'{n}_{r}']=a[key+'_forward_k0'][:,0][...,valid].astype(float).round(7).tolist()
                targets[kind].append(target)
                if kind!='generated':
                    row=by_frame[f][0]
                    examples.append(dict(global_idx=f,episode_idx=row['episode_idx'],frame_idx=row['frame_idx'],
                                         kind=kind,kind_label=titles[kind],target=target.round(7).tolist(),paths=paths))
    groups=defaultdict(list)
    for row in rows: groups[row['target_kind'],row['num_steps'],row['refine']].append(row)
    measurements=[]
    for (kind,n,r), samples in groups.items():
        measurements.append(dict(target_kind=kind,num_steps=n,refine=r,n=len(samples),
                                 mean_mse=float(np.mean([s['mse'] for s in samples])),
                                 worst_frame_mse=max(s['mse'] for s in samples),
                                 worst_coordinate_error=max(s['max_abs'] for s in samples),
                                 mean_fp32_mse=float(np.mean([s['fp32_mse'] for s in samples])),
                                 mean_cast_only_mse=float(np.mean([s['cast_only_mse'] for s in samples])),
                                 fraction_coordinates_outside_training_range=float(np.mean([s['target_fraction_outside_training_range'] for s in samples]))))
    def measure(kind,s): return next(v for v in measurements if (v['target_kind'],v['num_steps'],v['refine'])==(kind,*s))
    focus=(20,4) if (20,4) in settings else settings[-1]
    uniform=measure('uniform',focus) if 'uniform' in kinds else measure(kinds[0],focus)
    summary=dict(header=header,measurements=measurements,focus=dict(num_steps=focus[0],refine=focus[1]),
                 sampler_parity_max_abs=max(r['sampler_parity_max_abs'] for r in rows),
                 interpretation='Finite-sample approximate reachability; no input-prior constraint and no universal surjectivity claim')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (out/'roundtrip.csv').open('w',newline='') as file:
        fields=sorted({k for row in rows for k in row if not k.endswith('_by_step')})
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9','savefig.facecolor':'#fafbf9'})
    def save(fig,name):fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig)
    def grid(ax,xlabel):
        ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.set_xscale('symlog',linthresh=1e-9)
    step_counts=sorted({n for n,r in settings})
    fig,axes=plt.subplots(1,len(step_counts),figsize=(7*len(step_counts),6.5),squeeze=False,layout='constrained')
    for ax,n in zip(axes.flat,step_counts):
        refinements=[r for ns,r in settings if ns==n]
        for j,r in enumerate(refinements):
            color=COLORS[j];ys=np.arange(len(kinds))+(j-(len(refinements)-1)/2)*.25
            means=[measure(k,(n,r))['mean_mse'] for k in kinds]
            ax.plot(means,ys,'o',ms=9,color=color,label=f'{r} corrections per reverse step')
            for y,k,m in zip(ys,kinds,means):
                ax.scatter([v['mse'] for v in groups[k,n,r]],np.full(len(groups[k,n,r]),y),s=14,alpha=.3,color=color)
                if r==max(refinements):ax.annotate(f'{m:.3g}',(m,y),xytext=(8,3),textcoords='offset points',fontsize=9)
        ax.set_yticks(range(len(kinds)),[titles[k] for k in kinds]);ax.invert_yaxis()
        ax.set_title(f'{n} reverse flow steps → {n} forward steps',loc='left');grid(ax,'Round-trip MSE · lower is better');ax.legend(fontsize=9,loc='lower right');ax.margins(x=.3)
    save(fig,'reachability_mse.png')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    x=np.arange(len(kinds));short=[titles[k].replace('Constant-in-time','Constant').replace('Model-generated actions','Generated').replace('Recorded actions','Recorded') for k in kinds]
    for j,s in enumerate(settings):
        axes[0].plot(x,[measure(k,s)['worst_frame_mse'] for k in kinds],'o-',color=COLORS[j],label=label(*s))
        axes[1].plot(x,[measure(k,s)['worst_coordinate_error'] for k in kinds],'o-',color=COLORS[j],label=label(*s))
    for ax in axes:
        ax.set_xticks(x,short,rotation=35,ha='right',fontsize=9);ax.set_yscale('symlog',linthresh=1e-8);ax.grid(axis='y',alpha=.2)
    axes[0].set(title='Worst target chunk in each family',ylabel='Largest per-chunk MSE')
    axes[1].set(title='Largest individual coordinate error',ylabel='Max |reconstruction − target| · normalized units')
    axes[0].legend(fontsize=8)
    save(fig,'reachability_worst_errors.png')
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,kind in zip(axes.flat,header['synthetic_distributions']):
        values=np.array(targets[kind]).ravel();ax.hist(values,bins=40,color='#1878bc',alpha=.85,density=True)
        ax.axvline(-1,color='#de752d',ls='--');ax.axvline(1,color='#de752d',ls='--')
        ax.set(title=titles[kind],xlabel='Normalized target coordinate',ylabel='Density')
        ax.text(.97,.93,f'{np.mean(np.abs(values)>1):.1%} outside [−1,1]',transform=ax.transAxes,ha='right',va='top',fontsize=10)
    for ax in list(axes.flat)[len(header['synthetic_distributions']):]:ax.set_visible(False)
    save(fig,'target_distributions.png')
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    ax.plot(x,[measure(k,focus)['mean_mse'] for k in kinds],'o-',label='Normal forward generation',color='#1878bc')
    ax.plot(x,[measure(k,focus)['mean_fp32_mse'] for k in kinds],'s--',label='Float32 diagnostic',color='#e07a24')
    ax.plot(x,[measure(k,focus)['mean_cast_only_mse'] for k in kinds],'d:',label='Target rounded to sampler precision only',color='#27957a')
    ax.set_xticks(x,short,rotation=30,ha='right');ax.set_yscale('symlog',linthresh=1e-9);ax.grid(axis='y',alpha=.2)
    ax.set(ylabel='Mean squared error',title=label(*focus));ax.legend(fontsize=10)
    save(fig,'reachability_precision.png')
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ax,kind in zip(axes.flat,header['synthetic_distributions']):
        e=np.array(errors[kind,*focus]);idx=int(np.argmax(np.mean(e**2,axis=(1,2))));joint=int(np.argmax(np.mean(e[idx]**2,axis=0)))
        t=targets[kind][idx];recon=t+e[idx]
        ax.plot(t[:,joint],color='#172b38',lw=2,label='Target action (black solid)')
        ax.plot(recon[:,joint],'--',color='#1878bc',lw=1.8,label='Reconstruction (blue dashed)')
        ax.set(title=f'{titles[kind]} · {joint_names_for_dim(t.shape[-1])[joint]}',xlabel='Future robot step within chunk',ylabel='Normalized action')
    for ax in list(axes.flat)[len(header['synthetic_distributions']):]:ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    save(fig,'synthetic_action_examples.png')

    def figure(file,title,text):return f'<section><h2>{title}</h2><p>{text}</p><figure><a href="{file}"><img src="{file}" alt="{html.escape(title)}" loading="lazy"></a></figure></section>'
    body='''<style>body{max-width:1400px;margin:auto;padding:30px 28px 80px;background:#f4f5f0;color:#172b38;font-size:17px;--muted:#536570;--card:#fff;--bg:#f4f5f0;--fg:#172b38;--line:#cdd7d8;--accent:#166ba5}h1{font-size:38px}h2{font-size:26px}p{max-width:1100px}section{margin-top:38px}.hero{background:#fff;border-top:5px solid #1878bc;border-radius:10px;padding:24px}.big{font-size:48px;font-weight:750}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}.card{background:#fff;padding:18px;border-radius:8px}.card b{display:block;font-size:24px}figure{margin:16px 0}figure img{width:100%;border-radius:8px}.controls{background:#e5eef0;align-items:center}.controls label{min-width:0;max-width:100%}.controls select{background:#fff;color:#172b38;max-width:100%}.jointplots{display:grid;grid-template-columns:1fr 1fr;gap:12px}.jointplots svg{width:100%;background:#fff}.live{padding:18px 0;font-size:19px;font-weight:650}.legendbox{position:sticky;top:0;background:#f4f5f0ee;padding:12px 0;z-index:1}.swatch{display:inline-block;width:35px;border-top:3px solid #172b38;vertical-align:middle;margin-right:8px}.blue{border-color:#1878bc;border-top-style:dashed}input[type=range]{width:min(400px,80vw)}code{overflow-wrap:anywhere}a{color:#166ba5}@media(max-width:700px){body{padding:18px 12px}.jointplots{grid-template-columns:1fr}h1{font-size:29px}.big{font-size:38px}}</style>'''
    body+='<div class="eyebrow">ACTION EXPERT · ARBITRARY TARGETS</div><h1>Can the expert produce arbitrary actions?</h1>'
    body+=f'<div class="hero">Uniform [−1,1] actions · mean round-trip MSE<div class="big">{uniform["mean_mse"]:.9f}</div><p>{label(*focus)}. Worst target-chunk MSE: <b>{uniform["worst_frame_mse"]:.7g}</b>; largest coordinate error: <b>{uniform["worst_coordinate_error"]:.5g}</b> normalized units.</p></div>'
    if not header.get('complete', False):
        body+=f'<p><b>Partial results:</b> {len(ids)} completed observation contexts. The run was stopped to make these results available now; all plots below use only fully saved targets.</p>'
    body+='<p><b>The question is existence, not noise likelihood.</b> Pick a target action, search for an input by reverse flow, then feed that input through the normal expert. A small final error supplies a concrete approximate preimage for that target. There is no constraint on how typical the recovered input is under a Gaussian prior.</p>'
    body+=f'<p>This run uses {len(ids)} observation contexts, with one independently seeded action chunk per context and synthetic distribution. Each target is held fixed across the solver settings. It tests {len(ids)*len(header["synthetic_distributions"])} synthetic target/context pairs. Recorded-action and generated-action controls are excluded.</p>'
    body+='<p><b>What the numbers mean:</b> MSE averages squared errors over all future positions and the 7 real joints, then over target chunks. Zero means exact reconstruction; lower is better. The output is compared before action clipping or physical-unit postprocessing. A finite list of successful targets cannot prove that every action is reachable; a failed inverse search cannot prove that a target is unreachable.</p>'
    body+=figure('reachability_mse.png','1. Round-trip MSE for each action distribution','Large dots are mean MSE; faint dots are individual target chunks. Colors compare 0 versus 4 inverse corrections. Each panel uses a different number of flow steps. The labels next to the corrected means give their numeric values. The x axis is logarithmic above 10⁻⁹. Lower means the expert reproduced the arbitrary target more closely.')
    body+='<div class="cards">'+''.join(f'<div class="card">Uniform [−1,1] · {label(*s)}<b>{measure("uniform",s)["mean_mse"]:.9f}</b>mean MSE</div>' for s in settings if 'uniform' in kinds)+'</div>'
    body+='<section><h2>2. What actions did we ask for?</h2><p>These are distributions over the <b>target action chunk</b>, not over the starting noise. Each coordinate is already in the action expert’s normalized space.</p><div class="cards">'+''.join(f'<div class="card"><h3>{titles[k]}</h3><p>{DISTRIBUTIONS[k][1]}</p></div>' for k in header['synthetic_distributions'])+'</div></section>'
    body+=figure('target_distributions.png','Check the actual target values','Histograms pool all sampled target coordinates. Orange dashed lines mark the training clamp range [−1,1]. Gaussian, uniform and heavy-tailed targets have the same population variance (1/3); their tails differ. Random ±1 has variance 1, and wide uniform has variance 3, so those stress tests deliberately change scale as well as shape. No out-of-range coordinate was clipped.')
    body+='<section><h2>3. Inspect the target and its reconstruction</h2><p>Choose a distribution, frame, and solver setting. The default is the worst uniform target for the selected solver. At the last iteration, blue is the final expert output; moving the slider left shows the forward return from the recovered input.</p>'
    body+='<div class="controls"><label>Target family<select id="distribution"></select></label><label>Target / observation frame<select id="frame"></select></label><label>Solver setting<select id="setting"></select></label><button id="worst">Show worst target in this family</button><label>Forward iteration<input id="iteration" type="range"></label></div><div class="legendbox"><span class="swatch"></span><b>Black solid = target action</b> &nbsp; <span class="swatch blue"></span><b>Blue dashed = reconstruction / current forward state</b></div><div id="live" class="live"></div><div id="jointplots" class="jointplots"></div></section>'
    body+=figure('synthetic_action_examples.png','Examples of the strangest requested actions',f'At {label(*focus)}, each panel selects that family’s highest-MSE target chunk and its highest-MSE joint. Black solid is the target; blue dashed is the final reconstruction. If the curves overlap, the expert produced a close match. The x axis is future robot position within the chunk, not flow time.')
    body+=figure('reachability_worst_errors.png','4. Does the average hide a target we could not reproduce?','Left: the largest per-chunk MSE among sampled targets in each family. Right: the largest absolute error in any valid joint/position, across the same targets. This second metric is in normalized action units, not squared units. These are observed worst cases in this sample, not bounds over all possible actions.')
    # Precision diagnostics remain downloadable; show only target reachability here.
    unused_precision_caption=figure('reachability_precision.png','5. How close are we to the sampler’s numerical precision?',f'Same recovered inputs, at {label(*focus)}. Blue: normal forward generation. Orange: float32 accumulation and time-conditioning diagnostic. Green: error from simply rounding the target to the sampler dtype, with no flow integration. Random ±1 is exactly representable, so that rounding error is zero. This separates a finite-precision limitation from imperfect numerical inversion; it does not certify unreachability.')
    body+='<section><h2>Normalization and what “any action” refers to</h2><p>The checkpoint encodes actions relative to the current state (anchor encoding), normalizes them with its saved ReBot quantiles, then clamps training targets to [−1,1]. For an encoded coordinate a, the transform is <code>2 × (a − q01) / (q99 − q01) − 1</code>.</p><p>We draw synthetic targets <b>directly after those preprocessing steps</b>. Thus [−1,1] is the normalized training box, not a range of raw joint angles. We verify the frame’s action-layout index against the saved ReBot normalization row, fill only the 7 valid coordinates, and keep padding zero. Targets are not normalized twice. Out-of-range targets and reconstructed outputs remain unclipped.</p><p>The question here is the range of the <b>action expert and flow integrator</b>. Downstream clamping and robot limits are separate constraints. These synthetic chunks are mathematical targets, not commands to a robot.</p></section>'
    body+='<section><h2>Flow steps and corrections</h2><p>20 steps means 20 reverse-flow updates from action to input, followed by 20 forward-flow updates back to action. Every model call is a forward neural-network evaluation; reverse integration subtracts the velocity instead of adding it. There is no backpropagation or weight training.</p><p>Zero corrections uses one explicit reverse estimate per step. Four corrections repeatedly evaluate the velocity at the estimated previous state to better undo that one forward update: <code>previous ← destination − Δt × velocity(previous, previous_time)</code>. Corrections are a numerical search method, not a restriction on the allowed input.</p></section>'
    body+=f'<p>Production-sampler parity: maximum coordinate difference <b>{summary["sampler_parity_max_abs"]:.6g}</b>. Captured {header["completed_frames"]}/{header["n_frames"]} requested contexts. Frames within an episode are correlated.</p>'
    body+='<details><summary>Checkpoint and saved evidence</summary><p><code>'+html.escape(header['checkpoint'])+'</code></p><p><a href="summary.json">MSE summary JSON</a> · <a href="roundtrip.csv">Per-target metrics CSV</a> · <a href="roundtrip.json">Capture provenance JSON</a>. Each trajectories_*.npz stores the exact target, recovered input, production output and full paths. Those arrays are the witnesses for the reported reconstruction errors.</p></details>'
    payload=dict(frames=examples,joints=joint_names_for_dim(targets[kinds[0]][0].shape[-1]),settings=[dict(key=f'{n}_{r}',n=n,label=label(n,r)) for n,r in settings])
    script='const DATA='+data_json(payload)+';\n'+EXPLORER_JS
    script=script.replace("`Episode ${f.episode_idx}, frame ${f.frame_idx}`", "`${f.kind_label} · episode ${f.episode_idx}, frame ${f.frame_idx}`")
    script=script.replace("DATA.frames.forEach((f,i)=>{let p=f.paths[key]", "DATA.frames.forEach((f,i)=>{if(f.kind!==document.getElementById('distribution').value)return;let p=f.paths[key]")
    # Populate the family filter before the shared explorer's initial worst-frame selection.
    setup="const dist=document.getElementById('distribution'); [...new Set(DATA.frames.map(f=>f.kind))].forEach(k=>dist.add(new Option(DATA.frames.find(f=>f.kind===k).kind_label,k)));dist.value=DATA.frames.some(f=>f.kind==='uniform')?'uniform':DATA.frames[0].kind;\n"
    script=script.replace('const frameSel=',setup+'const frameSel=')
    script+="\nfunction filterFrames(){[...frameSel.options].forEach(o=>{o.hidden=DATA.frames[Number(o.value)].kind!==dist.value;});worst();}dist.onchange=filterFrames;filterFrames();\n"
    page(out/'roundtrip.html','Action expert reachability · arbitrary targets',body,script)
    print(f'Reachability report: {out / "roundtrip.html"}',flush=True)

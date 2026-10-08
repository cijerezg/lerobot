"""Show fixed camera contexts, action-space blends, and the recovered noise paths."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from lerobot.probes.flow_interpolation import ALPHAS,COLORS,NAMES,path_geometry
from lerobot.probes.flow_noise_pca import shared_pca
from lerobot.probes.report_html import data_json,page


def render(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    from lerobot.probes.utils import joint_names_for_dim

    out=Path(out_dir);capture=json.loads((out/'interpolation.json').read_text())
    header,frames=capture['header'],capture['frames']
    generated=header.get('generated_actions',False)
    path_names=header.get('names',NAMES)
    arrays=[]
    for i in range(len(frames)):
        with np.load(out/f'paths_{i:02d}.npz') as a:
            valid=a['valid'].astype(bool)
            arrays.append(dict(target=a['target'][...,valid].astype(float),noise=a['noise_input'][...,valid].astype(float),
                               recon=a['reconstruction'][...,valid].astype(float),indices=a['path_indices'],
                               sampled_noise=a['sampled_noise'][...,valid].astype(float) if generated else None))
    lengths=[len(a['noise']) for a in arrays]
    flat=np.concatenate([a['noise'].reshape(len(a['noise']),-1) for a in arrays])
    fit=shared_pca(flat);ratio=fit['explained_variance_ratio'];scores=np.split(fit['scores'],np.cumsum(lengths)[:-1])
    np.savez_compressed(out/'path_pca.npz',mean=fit['mean'],components=fit['components'],
                        explained_variance_ratio=ratio,scores=fit['scores'])
    (out/'plotly.min.js').write_text(get_plotlyjs())
    reference_scores=[(a['sampled_noise'].reshape(3,-1)-fit['mean'])@fit['components'].T for a in arrays] if generated else []
    all_scores=np.concatenate([fit['scores'],*reference_scores])
    low=all_scores.min(0);high=all_scores.max(0);center=(low+high)/2;half=max((high-low).max()*.55,1e-5)
    scene={axis:dict(title=f'PC{i+1} · {ratio[i]:.1%} of pooled variance',range=[center[i]-half,center[i]+half],
                     backgroundcolor='#f1f4f5',showbackground=True,gridcolor='#d6dfe2')
           for i,axis in enumerate(['xaxis','yaxis','zaxis'])}
    scene.update(aspectmode='cube',camera=dict(eye=dict(x=1.4,y=1.4,z=1)))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9','savefig.facecolor':'#fafbf9'})
    body='''<style>body{max-width:1400px;margin:auto;padding:28px 24px 80px;background:#f4f5f0;color:#172b38;font-size:17px;--muted:#536570;--card:#fff;--bg:#f4f5f0;--fg:#172b38;--line:#cdd7d8;--accent:#166ba5}h1{font-size:38px}h2{font-size:27px}p{max-width:1120px}.hero{background:#fff;border-top:5px solid #1878bc;padding:22px;border-radius:9px}.photos{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.photos figure{margin:0}.photos img{width:100%;border-radius:8px}.photos figcaption{font-size:14px;padding:5px}.context{border-top:2px solid #ccd7db;padding-top:25px;margin-top:45px}.plotwrap{background:#fafbf9;border-radius:8px}.diagnostic{width:100%;margin:15px 0}.controls{background:#e7eef0;align-items:center}.controls label{max-width:100%;min-width:0}.controls select{background:#fff;color:#172b38;max-width:100%}.live{font-size:18px;font-weight:650;padding:12px 0}.jointplot svg{width:100%;max-width:900px;background:#fff;border-radius:8px}.legend{background:#fff;padding:10px;border-radius:8px;font-size:15px}.swatch{display:inline-block;width:28px;border-top:3px solid #172b38;vertical-align:middle;margin:0 7px 0 12px}.blue{border-color:#1878bc;border-top-style:dashed}.muted{color:#536570}a{color:#166ba5}code{overflow-wrap:anywhere}input[type=range]{width:min(340px,75vw)}@media(max-width:650px){body{padding:18px 10px}h1{font-size:29px}}</style>'''
    body+='<div class="eyebrow">FIXED OBSERVATIONS · ACTION INTERPOLATION</div><h1>How does an action path map into noise space?</h1>'
    body+=f'<div class="hero"><b>{len(frames)} own-ReBot frames · {len(path_names)} paths per frame · α = 0, 0.1, …, 1</b><p>Each frame keeps one observation and prompt fixed. It has one original action and one recovered original-noise point. Each synthetic endpoint is drawn once, then held fixed while we interpolate.</p><p><code>action(α) = (1 − α) × start action + α × end action</code></p><p><b>α is an action-blend fraction, not flow time.</b> We independently invert every blended action using {header["num_steps"]} flow steps and {header["refine"]} inverse corrections, then check it through the production sampler.</p></div>'
    if not header['complete']:
        body+=f'<p><b>Partial results: {len(frames)} of 3 frames complete.</b> Refresh this page as more frames finish.</p>'
    if generated:
        body+='<p><b>Endpoints are model-generated actions:</b> for each fixed frame, draw three independent standard Gaussian noise inputs (mean 0, standard deviation 1) and run the normal 20-step production sampler. These are model samples, not a guarantee of successful robot behavior. α=0 is the recorded action; α=1 is one of the three generated actions. Intermediate actions are linear blends of the two.</p>'
    else:
        body+='<p>The four spokes run from the recorded action to uniform [−1,1], Gaussian (σ=1/√3), random ±1, and heavy-tailed (t₃/3) actions. The dashed bridge runs from that same uniform endpoint to that same ±1 endpoint. Uniform and ±1 are independent draws; the ±1 target is not obtained by taking the sign of the uniform target.</p>'
    body+=f'<p>All three 3D panels share <b>one PCA basis and identical axis ranges</b>, fitted to the {len(flat)} unique recovered inputs. It retains <b>{ratio[:3].sum():.1%}</b> of their pooled variance. Shared endpoints are counted once. No group rescaling or noise normalization is applied. The distance and bending plots use all 210 coordinates, so their conclusions do not depend on PCA.</p>'
    if generated:
        all_mse=np.concatenate([np.mean((a['recon']-a['target'])**2,axis=(1,2)) for a in arrays])
        body+=f'<div class="hero"><b>Round-trip MSE across {len(all_mse)} unique action targets: mean {all_mse.mean():.3g} · worst {all_mse.max():.3g}</b><p>Each MSE averages squared reconstruction error over the 30 future steps × 7 valid joints in normalized action space. These targets include the recorded actions, generated endpoints, and intermediate blends; shared recorded targets are counted once.</p></div>'
    body+='<script src="plotly.min.js"></script>' 
    exported=[];summary_frames=[];interactive=[]
    for i,(frame,a,coords) in enumerate(zip(frames,arrays,scores)):
        geometries=path_geometry(a['target'],a['noise'],a['indices'])
        mse=np.mean((a['recon']-a['target'])**2,axis=(1,2))
        fig=go.Figure()
        for j,(name,color,index) in enumerate(zip(path_names,COLORS,a['indices'])):
            custom=np.column_stack([ALPHAS,mse[index],geometries[j]['straight_line_deviation'],index])
            fig.add_trace(go.Scatter3d(x=coords[index,0],y=coords[index,1],z=coords[index,2],name=name,mode='lines+markers',
                                       line=dict(color=color,width=5,dash='dash' if j==4 else 'solid'),
                                       marker=dict(size=4,color=color),customdata=custom,
                                       hovertemplate=name+'<br>Action blend α=%{customdata[0]:.1f}<br>Round-trip MSE=%{customdata[1]:.6g}'
                                       '<br>Full-space bend RMS=%{customdata[2]:.4f}<extra></extra>'))
            # Endpoint labels make the direction legible without having to hover.
            if j<4:
                k=index[-1]
                fig.add_trace(go.Scatter3d(x=[coords[k,0]],y=[coords[k,1]],z=[coords[k,2]],mode='markers+text',
                                           marker=dict(size=6,color=color,symbol='square'),text=[name.split(' → ')[-1]],
                                           textposition='top center',showlegend=False,hoverinfo='skip'))
        endpoint_noise_rms=None
        if generated:
            endpoint_index=a['indices'][:,-1]
            endpoint_noise_rms=np.sqrt(np.mean((a['noise'][endpoint_index]-a['sampled_noise'])**2,axis=(1,2)))
            for j,(color,q,k) in enumerate(zip(COLORS,reference_scores[i],endpoint_index)):
                # A dotted link exposes recovery error; it is not another interpolation path.
                fig.add_trace(go.Scatter3d(x=[coords[k,0],q[0]],y=[coords[k,1],q[1]],z=[coords[k,2],q[2]],
                    mode='lines+markers',line=dict(color=color,width=2,dash='dot'),
                    marker=dict(size=[0,7],color=color,symbol='cross'),name=f'Actual sampled noise {j+1}',showlegend=False,
                    hovertemplate=f'Actual sampled noise {j+1}<br>Recovered-versus-sampled noise RMS {endpoint_noise_rms[j]:.6g}<extra></extra>'))
        fig.add_trace(go.Scatter3d(x=[coords[0,0]],y=[coords[0,1]],z=[coords[0,2]],mode='markers+text',
                                   marker=dict(size=7,color='#172b38',symbol='diamond'),text=['Original action noise'],
                                   textposition='bottom center',name='Original action noise',showlegend=False,
                                   hovertemplate=f'Original action noise<br>Round-trip MSE {mse[0]:.7g}<extra></extra>'))
        fig.update_layout(height=630,scene=scene,margin=dict(l=0,r=0,t=35,b=10),paper_bgcolor='#fafbf9',
                          font=dict(family='system-ui',size=12,color='#173044'),legend=dict(orientation='h',x=0,y=1.06))
        plot=fig.to_html(full_html=False,include_plotlyjs=False,div_id=f'path-3d-{i}',config=dict(displaylogo=False,responsive=True))
        diagnostics,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
        stretch_fig,stretch=plt.subplots(figsize=(10,3.6),layout='constrained')
        path_summaries=[]
        for j,(name,color,index,g) in enumerate(zip(path_names,COLORS,a['indices'],geometries)):
            linestyle='--' if j==4 else '-'
            axes[0].plot(ALPHAS,g['distance_from_original'],'o',ls=linestyle,color=color,ms=3,label=name)
            axes[1].plot(ALPHAS,g['straight_line_deviation'],'o',ls=linestyle,color=color,ms=3)
            axes[2].plot(ALPHAS,mse[index],'o',ls=linestyle,color=color,ms=3)
            stretch.plot((ALPHAS[:-1]+ALPHAS[1:])/2,g['step_stretch'],'o',ls=linestyle,color=color,ms=4,label=name)
            for k,alpha in enumerate(ALPHAS):
                exported.append(dict(frame=i,episode_idx=frame['episode_idx'],frame_idx=frame['frame_idx'],path=name,
                                     alpha=float(alpha),target_index=int(index[k]),mse=float(mse[index[k]]),
                                     distance_from_original=float(g['distance_from_original'][k]),
                                     straight_line_deviation=float(g['straight_line_deviation'][k])))
            path_summaries.append(dict(name=name,max_mse=float(mse[index].max()),mean_mse=float(mse[index].mean()),
                                       max_bend_rms=float(g['straight_line_deviation'].max()),
                                       end_distance_from_original=float(g['distance_from_original'][-1])))
        for ax,title,y in zip(axes,['Movement away from original noise','Bending away from a straight noise line','Does every blend reconstruct?'],
                              ['RMS distance in noise space','RMS distance from endpoint chord','Round-trip MSE']):
            ax.set(title=title,xlabel='Action blend α',ylabel=y);ax.grid(alpha=.2)
        axes[2].set_yscale('symlog',linthresh=1e-8)
        handles,names=axes[0].get_legend_handles_labels();diagnostics.legend(handles,names,loc='outside upper center',ncol=3,fontsize=9)
        diagnostics.savefig(out/f'geometry_{i}.png',dpi=170,bbox_inches='tight');plt.close(diagnostics)
        stretch.set(xlabel='Midpoint of each 0.1 action-blend interval',ylabel='Noise step RMS / action step RMS',title='Where does the inverse expand or contract a small action change?')
        stretch.grid(alpha=.2);stretch.legend(fontsize=9,ncol=2)
        stretch_fig.savefig(out/f'stretch_{i}.png',dpi=170,bbox_inches='tight');plt.close(stretch_fig)
        summary_frames.append(dict(episode_idx=frame['episode_idx'],frame_idx=frame['frame_idx'],task=frame['task'],
                                   original_mse=float(mse[0]),paths=path_summaries,
                                   endpoint_noise_recovery_rms=None if endpoint_noise_rms is None else endpoint_noise_rms.tolist()))
        interactive.append(dict(target=a['target'].round(7).tolist(),recon=a['recon'].round(7).tolist(),indices=a['indices'].tolist(),
                                joint_names=joint_names_for_dim(a['target'].shape[-1]),mse=mse.tolist()))
        body+=f'<section class="context"><h2>Frame {i+1}: {html.escape(frame["task"])}</h2><p>Validation episode {frame["episode_idx"]}, frame {frame["frame_idx"]}. Current subtask: <b>{html.escape(str(frame["subtask"]))}</b>. Original-action round-trip MSE: <b>{mse[0]:.7g}</b>.</p>'
        body+='<div class="photos">'+''.join(f'<figure><img src="{p["file"]}" alt="{html.escape(p["camera"])} camera at selected frame"><figcaption>{html.escape(p["camera"])}</figcaption></figure>' for p in frame['photographs'])+'</div>'
        body+='<p>These are the actual RGB observations held fixed throughout this frame’s experiment. Depth, state, and prompt also stay fixed. The robot is not moving through these synthetic targets.</p>'
        if generated:
            body+='<h3>Recovered-noise paths</h3><p><b>Black diamond:</b> input recovered for the recorded action. <b>Colored lines and dots:</b> independently recovered inputs as α increases from 0 to 1. <b>Squares:</b> recovered inputs at generated-action endpoints. <b>Crosses:</b> actual Gaussian inputs that generated those endpoints. Dotted connectors show any difference between a square and its cross; they are not interpolation paths. Rotate or hover to inspect.</p><div class="plotwrap">'+plot+'</div>'
        else:
            body+='<h3>Recovered-noise paths</h3><p>The black diamond is the single input recovered for the recorded action. Colored dots follow α in increments of 0.1; squares mark the synthetic endpoints. The dashed bridge connects the uniform and ±1 endpoints. Rotate the plot or hover over a point to read α and its reconstruction MSE.</p><div class="plotwrap">'+plot+'</div>'
        body+=f'<h3>Full-dimensional path geometry</h3><img class="diagnostic" src="geometry_{i}.png" alt="Full-space movement, bending, and reconstruction error for frame {i+1}">'
        body+='<p><b>Left:</b> distance from the original-action noise, using all 210 coordinates. The bridge starts away from the original because it starts at uniform. <b>Middle:</b> distance from the straight line between each path’s own noise endpoints. If this were zero throughout, linear interpolation in action space would map to linear interpolation in noise space. <b>Right:</b> reconstruction MSE for each blended action, using the actual production sampler; its y axis is logarithmic above 10⁻⁸.</p>'
        if generated:
            endpoint_fig,endpoint_axes=plt.subplots(1,2,figsize=(11,3.4),layout='constrained')
            labels=['Sample 1','Sample 2','Sample 3']
            for ax,values,title,ylabel in zip(endpoint_axes,[mse[a['indices'][:,-1]],endpoint_noise_rms],
                ['Do generated actions reconstruct?','Do we recover their actual sampled inputs?'],
                ['Generated-endpoint round-trip MSE','RMS difference in noise (210 coordinates)']):
                ax.bar(labels,values,color=COLORS[:3]);ax.set(title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.2)
                ax.set_ylim(0,max(float(np.max(values))*1.25,1e-10))
                for k,value in enumerate(values):ax.text(k,value,f'{value:.3g}',ha='center',va='bottom',fontsize=10)
            endpoint_fig.savefig(out/f'endpoint_check_{i}.png',dpi=150,bbox_inches='tight');plt.close(endpoint_fig)
            body+=f'<h3>Check the known generated endpoints</h3><img class="diagnostic" src="endpoint_check_{i}.png" alt="Generated action reconstruction and known-input recovery errors"><p>The left plot measures action reconstruction error. The right compares the recovered noise with the <b>known input</b> used to generate that action. These are different questions: low action MSE does not by itself establish that the recovered noise equals the original sample. The finite-precision sampler and approximate inverse can produce discrepancies.</p>'
        body+=f'<details><summary>Local expansion along the paths</summary><img class="diagnostic" src="stretch_{i}.png" alt="Local inverse expansion"><p>Each point divides the noise-space movement between consecutive samples by the corresponding action-space movement. Both use RMS over 210 coordinates. A rising curve means the same-size action increment requires a larger change in the recovered input. The horizontal position is the midpoint of a 0.1 α interval.</p></details>'
        body+=f'<h3>Inspect the blended action itself</h3><div class="controls"><label>Path<select class="path-select" data-frame="{i}">'+''.join(f'<option value="{j}">{name}</option>' for j,name in enumerate(path_names))+f'</select></label><label>Joint<select class="joint-select" data-frame="{i}">'+''.join(f'<option value="{j}">{name}</option>' for j,name in enumerate(interactive[-1]['joint_names']))+f'</select></label><label>Action blend α<input class="alpha" data-frame="{i}" type="range" min="0" max="10" value="0" step="1"></label></div><p class="legend"><span class="swatch"></span>Black solid = blended target action <span class="swatch blue"></span>Blue dashed = final reconstructed action</p><div class="live" id="live-{i}"></div><div class="jointplot" id="joint-{i}"></div></section>'
    summary=dict(header=header,unique_inputs=len(flat),pca_variance_3d=float(ratio[:3].sum()),frames=summary_frames)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (out/'path_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    body+='<section class="context"><h2>Sampling and normalization</h2><p>These are your own ReBot episodes, identified from dataset provenance; external demonstration episodes are excluded. We select a frame about 40% through each of three different tasks and snap it to the image/depth stride. This choice is independent of the inversion results.</p><p>Each synthetic family has <b>one sampled endpoint per frame</b>. These paths are examples, not estimates of an entire family’s typical geometry. The uniform-to-±1 bridge shares its endpoints with the two spokes; the original and all endpoints are inverted only once per frame.</p><p>Interpolation happens in the expert’s normalized action coordinates, after anchor encoding and quantile normalization. All paths use the same 7 valid joints and 30-step horizon. Padding stays zero. Gaussian and heavy-tailed endpoints and their interpolations remain unclipped. These synthetic action targets are never sent to a robot.</p>'
    body+=f'<p>Compute: {len(flat)} unique inversions; {header["num_steps"]} flow steps and {header["refine"]} inverse corrections, batch size {header["inverse_batch_size"]}. Every target is reconstructed individually by the actual production sampler. Capture took <b>{header["elapsed_seconds"]:.1f} seconds</b> after loading the model.</p><p>Data: <code>{html.escape(header["val_root"])}</code><br>Checkpoint: <code>{html.escape(header["checkpoint"])}</code></p>'
    body+='<p><a href="path_metrics.csv">Per-path metrics CSV</a> · <a href="summary.json">Summary</a> · <a href="interpolation.json">Frame identities, task prompts, provenance and timing</a> · <a href="path_pca.npz">Shared PCA basis</a>. Files paths_*.npz contain exact targets, recovered inputs, and production outputs.</p></section>'
    if generated:
        body=body.replace('Each synthetic endpoint is drawn once, then held fixed while we interpolate.',
            'Three noise draws are each passed through the production sampler to make three action endpoints, which stay fixed while we interpolate.')
        body=body.replace('The bridge starts away from the original because it starts at uniform. ', '')
        body=body.replace('Each synthetic family has <b>one sampled endpoint per frame</b>. These paths are examples, not estimates of an entire family’s typical geometry. The uniform-to-±1 bridge shares its endpoints with the two spokes; the original and all endpoints are inverted only once per frame.',
            'Each frame has <b>three independently generated action endpoints</b>, and each path has 11 α values. The original is shared, giving 31 unique targets per frame and 93 across all three frames. We draw N(0,1) noise directly in the production dtype, mask padding, and save the exact seeds and inputs. These are examples, not a distribution-level estimate.')
        body=body.replace('Gaussian and heavy-tailed endpoints and their interpolations remain unclipped. These synthetic action targets are never sent to a robot.',
            'Generated actions are the normalized outputs of the production flow sampler before postprocessing. We blend those outputs directly with the normalized recorded action, without clipping or applying normalization a second time. Nothing is sent to a robot.')
        body=body.replace('<h1>How does an action path map into noise space?</h1>',
            '<h1>Recorded → model-generated actions: what happens in noise space?</h1>')
        body=body.replace('<script src="plotly.min.js"></script>',
            '<p><b>How inversion works:</b> walk through flow time from action back to input, subtracting the predicted velocity. At each step, four correction evaluations update the candidate previous point. Every model evaluation is an ordinary forward pass; no gradients or backpropagation are used. This numerical inverse is approximate. Read its path geometry together with the round-trip MSE: a large MSE means the recovered point does not closely reproduce that target.</p><script src="plotly.min.js"></script>')
    page(out/'interpolation.html','Action interpolation → recovered noise paths',body,'const DATA='+data_json(interactive)+';\n'+SCRIPT)
    print(f'Interpolation report: {out / "interpolation.html"}',flush=True)


SCRIPT=r'''
function draw(i){const d=DATA[i],p=Number(document.querySelector(`.path-select[data-frame="${i}"]`).value),j=Number(document.querySelector(`.joint-select[data-frame="${i}"]`).value),step=Number(document.querySelector(`.alpha[data-frame="${i}"]`).value),index=d.indices[p][step];
 document.getElementById(`live-${i}`).textContent=`Action blend α = ${(step/10).toFixed(1)} · final round-trip MSE = ${d.mse[index].toPrecision(6)}`;
 const a=d.target[index].map(x=>x[j]),b=d.recon[index].map(x=>x[j]),v=[...a,...b];let lo=Math.min(...v),hi=Math.max(...v),pad=Math.max((hi-lo)*.1,.005);lo-=pad;hi+=pad;
 const W=900,H=270,L=75,R=25,T=25,B=55,X=k=>L+k/(a.length-1)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
 let s=`<svg role="img" aria-label="Blended target and reconstructed action" viewBox="0 0 ${W} ${H}">`;
 for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4,y=Y(v);s+=`<path d="M${L} ${y}H${W-R}" stroke="#dce4e7"/><text x="${L-10}" y="${y+4}" text-anchor="end" font-size="12" fill="#536570">${v.toFixed(3)}</text>`;}
 for(const [values,color,dash] of [[a,'#172b38',''],[b,'#1878bc','6 3']])s+=`<path d="${values.map((v,k)=>(k?'L':'M')+X(k)+','+Y(v)).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5" stroke-dasharray="${dash}"/>`;
 for(const k of [0,14,29])s+=`<text x="${X(k)}" y="${H-B+22}" text-anchor="middle" font-size="13" fill="#536570">${k}</text>`;
 s+=`<text x="${W/2}" y="${H-10}" text-anchor="middle" font-size="13" fill="#536570">Future robot step within the action chunk</text><text transform="translate(18 ${H/2}) rotate(-90)" text-anchor="middle" font-size="13" fill="#536570">Normalized action</text></svg>`;
 document.getElementById(`joint-${i}`).innerHTML=s;}
 document.querySelectorAll('.path-select,.joint-select,.alpha').forEach(e=>e.addEventListener('input',()=>draw(Number(e.dataset.frame))));DATA.forEach((_,i)=>draw(i));
'''

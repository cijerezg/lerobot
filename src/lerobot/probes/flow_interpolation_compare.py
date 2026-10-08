"""Paired step-200 versus step-1200 action interpolation, on one common PCA."""
from __future__ import annotations

import json
import html
from pathlib import Path

import numpy as np

from lerobot.probes.flow_interpolation import ALPHAS,COLORS,NAMES,path_geometry
from lerobot.probes.flow_noise_pca import shared_pca
from lerobot.probes.report_html import page


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    root=Path(root);steps=['000200','001200'];captures=[];data=[]
    for step in steps:
        capture=json.loads((root/f'step_{step}/interpolation.json').read_text());captures.append(capture)
    count=min(len(c['frames']) for c in captures)
    if not count:raise ValueError('No matched frames have finished yet')
    for step in steps:
        entries=[]
        for i in range(count):
            with np.load(root/f'step_{step}/paths_{i:02d}.npz') as a:
                mask=a['valid'].astype(bool)
                entries.append(dict(target=a['target'][...,mask].astype(float),noise=a['noise_input'][...,mask].astype(float),
                                    reconstruction=a['reconstruction'][...,mask].astype(float),indices=a['path_indices']))
        data.append(entries)
    for i in range(count):
        assert captures[0]['frames'][i]['global_idx']==captures[1]['frames'][i]['global_idx']
        np.testing.assert_array_equal(data[0][i]['target'],data[1][i]['target'])
        np.testing.assert_array_equal(data[0][i]['indices'],data[1][i]['indices'])
    for key in ['num_steps','refine','seed','val_root']:
        assert captures[0]['header'][key]==captures[1]['header'][key],key
    flat=np.concatenate([a['noise'].reshape(len(a['noise']),-1) for model in data for a in model])
    pca=shared_pca(flat);ratio=pca['explained_variance_ratio'];coords=pca['scores'].reshape(2,count,50,3)
    np.savez_compressed(root/'shared_path_pca.npz',mean=pca['mean'],components=pca['components'],scores=coords,
                        explained_variance_ratio=ratio,checkpoint_steps=np.array([200,1200]))
    (root/'plotly.min.js').write_text(get_plotlyjs())
    low=pca['scores'].min(0);high=pca['scores'].max(0);center=(low+high)/2;half=max(float((high-low).max())*.55,1e-6)
    scene={axis:dict(title=f'PC{i+1} · {ratio[i]:.1%}',range=[center[i]-half,center[i]+half],
                     backgroundcolor='#f0f4f5',showbackground=True,gridcolor='#d6dfe2')
           for i,axis in enumerate(['xaxis','yaxis','zaxis'])}
    scene.update(aspectmode='cube',camera=dict(eye=dict(x=1.4,y=1.4,z=1)))
    body='''<style>body{max-width:1600px;margin:auto;padding:28px 22px 70px;background:#f4f5f0;color:#172b38;font-size:17px;--muted:#536570;--card:#fff;--bg:#f4f5f0;--fg:#172b38;--line:#cdd7d8;--accent:#166ba5}h1{font-size:36px}h2{font-size:26px}.hero{background:#fff;border-top:5px solid #1878bc;border-radius:8px;padding:22px}.photos{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.photos figure{margin:0}.photos img{width:100%;border-radius:8px}.photos figcaption{font-size:14px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.model{background:#fafbf9;min-width:0;border:1px solid #d4dfe1;border-radius:8px;padding:8px}.model h3{margin:8px 14px}.model p{margin:8px 14px;font-size:15px}section{border-top:2px solid #ccd7db;margin-top:40px;padding-top:22px}.diagnostic{width:100%;margin:12px 0}p{max-width:1200px}a{color:#166ba5}code{overflow-wrap:anywhere}.badge{font-weight:700;color:#166ba5}@media(max-width:800px){body{padding:18px 10px}.pair{grid-template-columns:1fr}h1{font-size:28px}}</style>'''
    body+='<div class="eyebrow">SAME ACTION PATHS · TWO CHECKPOINTS</div><h1>Step 200 versus step 1200: how does the inverse change?</h1>'
    body+=f'<div class="hero"><b>{count} matched ReBot frames · identical action targets · α = 0, 0.1, …, 1</b><p>Both checkpoints see exactly the same observation frame, prompt, sampled endpoints, and interpolated action targets. The saved action normalization is identical, and the complete target tensors are verified bit-for-bit equal.</p><p><b>20 flow steps, 4 inverse corrections per step</b> for both checkpoints. The comparison uses one common PCA across both checkpoints and all matched frames; every 3D panel uses the same axes and ranges. These axes retain <b>{ratio[:3].sum():.1%}</b> of pooled variance.</p></div>'
    complete=count==3 and all(c['header']['complete'] for c in captures)
    if not complete:body+=f'<p class="badge">Partial side-by-side results: {count}/3 matched frames. Reload as the next frame finishes.</p>'
    body+='<p><b>α blends actions, not flow time:</b> action(α) = (1−α) × start + α × end. We independently invert each target. Each frame has four paths from its original action to uniform, Gaussian, random ±1, and heavy-tailed targets; a dashed fifth path connects the same uniform and ±1 endpoints.</p>'
    body+='<p>The black diamond in each panel is the single noise input recovered for the original action under that checkpoint. Colored dots follow α in 0.1 increments; squares mark synthetic endpoints. Rotate a panel or hover for α and the actual reconstruction MSE. Both panels rotate together so their viewpoints stay comparable.</p><script src="plotly.min.js"></script>'
    summary=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9','savefig.facecolor':'#fafbf9'})
    for i in range(count):
        frame=captures[1]['frames'][i]
        body+=f'<section><h2>{i+1}. {html.escape(frame["task"])}</h2><p>Validation episode {frame["episode_idx"]}, frame {frame["frame_idx"]}. Subtask: <b>{html.escape(str(frame["subtask"]))}</b>. Same fixed context for every point below.</p>'
        body+='<div class="photos">'+''.join(f'<figure><img src="step_001200/{p["file"]}" alt="{html.escape(p["camera"])} camera"><figcaption>{html.escape(p["camera"])}</figcaption></figure>' for p in frame['photographs'])+'</div><div class="pair">'
        diag,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey='col',layout='constrained')
        frame_summary=dict(task=frame['task'],episode_idx=frame['episode_idx'],frame_idx=frame['frame_idx'],checkpoints={})
        for m,step in enumerate(steps):
            a=data[m][i];xyz=coords[m,i];mse=np.mean((a['reconstruction']-a['target'])**2,axis=(1,2));g=path_geometry(a['target'],a['noise'],a['indices'])
            fig=go.Figure()
            paths=[]
            for j,(name,color,index) in enumerate(zip(NAMES,COLORS,a['indices'])):
                fig.add_trace(go.Scatter3d(x=xyz[index,0],y=xyz[index,1],z=xyz[index,2],name=name,mode='lines+markers',
                                           line=dict(color=color,width=5,dash='dash' if j==4 else 'solid'),marker=dict(size=4,color=color),
                                           customdata=np.column_stack([ALPHAS,mse[index]]),
                                           hovertemplate=f'Step {int(step)} · '+name+'<br>α=%{customdata[0]:.1f}<br>MSE=%{customdata[1]:.7g}<extra></extra>'))
                if j<4:
                    k=index[-1];fig.add_trace(go.Scatter3d(x=[xyz[k,0]],y=[xyz[k,1]],z=[xyz[k,2]],mode='markers',marker=dict(size=6,color=color,symbol='square'),showlegend=False,hoverinfo='skip'))
                for ax,values in zip(axes[m],[g[j]['distance_from_original'],g[j]['straight_line_deviation'],mse[index]]):
                    ax.plot(ALPHAS,values,'o',ls='--' if j==4 else '-',ms=3,color=color,label=name);ax.grid(alpha=.2)
                paths.append(dict(name=name,max_bend_rms=float(g[j]['straight_line_deviation'].max()),
                                  mean_mse=float(mse[index].mean()),max_mse=float(mse[index].max())))
            fig.add_trace(go.Scatter3d(x=[xyz[0,0]],y=[xyz[0,1]],z=[xyz[0,2]],mode='markers+text',marker=dict(size=7,color='#172b38',symbol='diamond'),text=['Original'],textposition='bottom center',showlegend=False,hovertemplate=f'Original action · MSE {mse[0]:.7g}<extra></extra>'))
            fig.update_layout(height=570,scene=scene,margin=dict(l=0,r=0,t=70,b=5),paper_bgcolor='#fafbf9',
                              font=dict(family='system-ui',size=11,color='#173044'),legend=dict(orientation='h',x=0,y=1.16,font=dict(size=10)))
            body+=f'<div class="model"><h3>Checkpoint {int(step)}</h3><p>Original-action MSE: <b>{mse[0]:.7g}</b></p>'+fig.to_html(full_html=False,include_plotlyjs=False,div_id=f'compare-{i}-{m}',config=dict(displaylogo=False,responsive=True))+f'<p><a href="step_{step}/interpolation.html">Inspect action blends and individual-checkpoint details</a></p></div>'
            axes[m,0].set_ylabel(f'Step {int(step)}\nRMS distance from original noise')
            axes[m,1].set_ylabel('RMS deviation from straight noise line')
            axes[m,2].set_ylabel('Round-trip MSE');axes[m,2].set_yscale('symlog',linthresh=1e-8)
            frame_summary['checkpoints'][str(int(step))]=dict(original_mse=float(mse[0]),paths=paths)
        body+='</div>'
        for ax,title in zip(axes[0],['Movement away from original noise','Bending of the recovered noise path','Reconstruction along the whole path']):ax.set_title(title)
        for ax in axes[1]:ax.set_xlabel('Action blend α')
        handles,names=axes[0,0].get_legend_handles_labels();diag.legend(handles,names,loc='outside upper center',ncol=3,fontsize=10)
        diag.savefig(root/f'comparison_geometry_{i}.png',dpi=170,bbox_inches='tight');plt.close(diag)
        body+=f'<h3>Compare the full 210-dimensional paths</h3><img class="diagnostic" src="comparison_geometry_{i}.png" alt="Matched noise-path geometry and errors for the two checkpoints"><p>Top row: checkpoint 200. Bottom row: checkpoint 1200. <b>Each column has the same y scale in both rows.</b> Left measures movement away from each checkpoint’s own original-action noise. Middle measures departure from each path’s straight endpoint-to-endpoint line. Right checks whether the recovered input reconstructs the blended action. The bridge starts at uniform, so it starts away from the original-noise point.</p></section>'
        summary.append(frame_summary)
    elapsed=[c['header']['elapsed_seconds'] for c in captures]
    body+=f'<section><h2>Scope and compute</h2><p>One synthetic endpoint per distribution and frame; these are path examples, not distribution averages. Shared original and endpoint targets are inverted only once per checkpoint and frame. There are 50 unique targets per frame, hence 150 per checkpoint and 300 in the completed comparison.</p><p>Targets are blended directly in the checkpoint’s normalized action coordinates. No clipping is applied to synthetic endpoints, blended targets, or reconstructions. Padding is excluded from geometry and error metrics. Both checkpoints use the actual production forward sampler for every reconstruction; the noise plotted is the input after its sampler-dtype cast.</p><p>Capture time after model loading: checkpoint 200 <b>{elapsed[0]:.1f}s</b>; checkpoint 1200 <b>{elapsed[1]:.1f}s</b>. Data: <code>{html.escape(captures[0]["header"]["val_root"])}</code>. Source images and original-source episode provenance are included in each checkpoint’s report.</p><p><a href="comparison_summary.json">Comparison measurements</a> · <a href="shared_path_pca.npz">Common PCA coordinates and basis</a>.</p></section>'
    _summary=dict(complete=complete,matched_frames=count,pca_variance_3d=float(ratio[:3].sum()),identical_targets_verified=True,frames=summary)
    (root/'comparison_summary.json').write_text(json.dumps(_summary,indent=2,allow_nan=False))
    # Synchronize camera movements between paired panels without changing either PCA.
    script=f'''for(let i=0;i<{count};i++){{let lock=false;const a=document.getElementById(`compare-${{i}}-0`),b=document.getElementById(`compare-${{i}}-1`);for(const [src,dst] of [[a,b],[b,a]])src.on('plotly_relayout',e=>{{if(lock||!e['scene.camera'])return;lock=true;Plotly.relayout(dst,{{'scene.camera':e['scene.camera']}}).then(()=>lock=false);}});}}'''
    page(root/'comparison.html','Action interpolation · checkpoint 200 versus 1200',body,script)
    print(f'Paired interpolation report: {root / "comparison.html"} ({count} matched frames)',flush=True)


if __name__=='__main__':
    import sys
    render(sys.argv[1])

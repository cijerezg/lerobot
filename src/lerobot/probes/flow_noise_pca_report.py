"""Interactive shared 3D PCA of recovered inputs, with honest projection context."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from lerobot.probes.flow_noise_pca import COLORS, KINDS, LABELS, shared_pca
from lerobot.probes.report_html import page


def render(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    out=Path(out_dir)
    capture=json.loads((out/'noise_capture.json').read_text())
    header=capture['header']; count=header['completed_contexts']
    if not count:return
    chunks=[]
    metrics={(r['context_index'],r['kind']):dict(r) for r in capture['records']}
    for i in range(count):
        with np.load(out/f'noise_{i:04d}.npz') as arrays:
            valid=arrays['valid'].astype(bool)
            z=arrays['noise_input'][...,valid].astype(float)
            chunks.append(z.reshape(len(KINDS),-1))
            for j,kind in enumerate(KINDS):
                error=(arrays['reconstruction'][j].astype(float)-arrays['target'][j].astype(float))[...,valid]
                metrics[i,kind]['mse']=float(np.mean(error**2))
                metrics[i,kind]['noise_rms']=float(np.sqrt(np.mean(z[j]**2)))
    vectors=np.stack(chunks)  # [contexts, families, 210]
    flat=vectors.reshape(-1,vectors.shape[-1])
    fit=shared_pca(flat)
    scores=fit['scores'].reshape(count,len(KINDS),3)
    ratios=fit['explained_variance_ratio']
    np.savez_compressed(out/'noise_pca.npz',noise=vectors,scores=scores,pca_mean=fit['mean'],
                        components=fit['components'],explained_variance_ratio=ratios,kinds=np.array(KINDS),
                        global_indices=np.array([header['samples'][i]['global_idx'] for i in range(count)]))
    records=[]
    for i in range(count):
        for j,kind in enumerate(KINDS):
            records.append(dict(metrics[i,kind],pc1=float(scores[i,j,0]),pc2=float(scores[i,j,1]),pc3=float(scores[i,j,2])))
    with (out/'points.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    stats={}
    for j,kind in enumerate(KINDS):
        group=vectors[:,j];error=np.array([metrics[i,kind]['mse'] for i in range(count)])
        centroid=group.mean(axis=0)
        stats[kind]=dict(n=count,mean_roundtrip_mse=float(error.mean()),max_roundtrip_mse=float(error.max()),
                         mean_noise_sq_norm_over_d=float(np.mean(group**2)),
                         full_space_centroid_norm=float(np.linalg.norm(centroid)),
                         mean_squared_distance_to_centroid=float(np.mean(np.sum((group-centroid)**2,axis=1))),
                         projection_energy_fraction=float(np.sum(scores[:,j]**2)/np.sum((group-fit['mean'])**2)))
    summary=dict(header=header,dimensions=flat.shape[1],total_points=len(flat),points_per_group=count,
                 pca=dict(center='one pooled mean across all groups',standardize=False,whiten=False,
                          explained_variance_ratio=ratios[:3].tolist(),explained_variance_3d=float(ratios[:3].sum())),groups=stats)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    axes_titles=[f'PC{i+1} · {ratios[i]:.1%} of pooled variance' for i in range(3)]
    fig=go.Figure()
    for j,(kind,name,color) in enumerate(zip(KINDS,LABELS,COLORS)):
        custom=[[metrics[i,kind]['episode_idx'],metrics[i,kind]['frame_idx'],metrics[i,kind]['mse'],
                 metrics[i,kind]['noise_rms'],metrics[i,kind]['global_idx']] for i in range(count)]
        fig.add_trace(go.Scatter3d(x=scores[:,j,0],y=scores[:,j,1],z=scores[:,j,2],name=name,mode='markers',
                                   marker=dict(size=4,color=color,opacity=.8),customdata=custom,
                                   hovertemplate=name+'<br>Episode %{customdata[0]} · frame %{customdata[1]}'
                                   '<br>Round-trip MSE %{customdata[2]:.7g}<br>Noise RMS %{customdata[3]:.4f}'
                                   '<br>PC1 %{x:.3f} · PC2 %{y:.3f} · PC3 %{z:.3f}<extra></extra>'))
    # An isotropic geometric display: equal distance has the same meaning on each axis.
    low=scores.reshape(-1,3).min(axis=0); high=scores.reshape(-1,3).max(axis=0)
    span=np.maximum(high-low,1e-6);center=(high+low)/2;half=span.max()*.56
    scene={axis:dict(title=title,range=[center[i]-half,center[i]+half],backgroundcolor='#f1f4f5',gridcolor='#d6dfe2',showbackground=True)
           for i,(axis,title) in enumerate(zip(['xaxis','yaxis','zaxis'],axes_titles))}
    scene.update(aspectmode='cube',camera=dict(eye=dict(x=1.5,y=1.5,z=1.1)))
    fig.update_layout(height=760,margin=dict(l=0,r=0,t=20,b=10),paper_bgcolor='#fafbf9',font=dict(family='system-ui',size=13,color='#173044'),
                      scene=scene,legend=dict(orientation='h',y=1.05,x=0),uirevision='noise-pca')
    plot=fig.to_html(full_html=False,include_plotlyjs=False,div_id='noise-pca-3d',config=dict(displaylogo=False,responsive=True,
                        toImageButtonOptions=dict(format='png',filename='recovered_noise_pca_3d',scale=2)))
    (out/'plotly.min.js').write_text(get_plotlyjs())

    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9','savefig.facecolor':'#fafbf9'})
    def save(f,name):f.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(f)
    projection,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for ax,(a,b) in zip(axs,[(0,1),(0,2),(1,2)]):
        for j,name in enumerate(LABELS):
            ax.scatter(scores[:,j,a],scores[:,j,b],color=COLORS[j],s=15,alpha=.65,label=name)
        ax.set(xlabel=axes_titles[a],ylabel=axes_titles[b]);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    handles,names=axs[0].get_legend_handles_labels();projection.legend(handles,names,loc='outside upper center',ncol=3,fontsize=10)
    save(projection,'pca_projections.png')
    static=plt.figure(figsize=(10,8));ax=static.add_subplot(projection='3d')
    for j,name in enumerate(LABELS):ax.scatter(scores[:,j,0],scores[:,j,1],scores[:,j,2],s=16,alpha=.65,color=COLORS[j],label=name)
    ax.set(xlabel=axes_titles[0],ylabel=axes_titles[1],zlabel=axes_titles[2]);ax.set_box_aspect((1,1,1))
    ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half);ax.set_zlim(center[2]-half,center[2]+half)
    ax.legend(fontsize=9,loc='upper left');save(static,'pca_3d.png')
    diagnostics,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for j,name in enumerate(LABELS):
        norm=np.sqrt(np.mean(vectors[:,j]**2,axis=1))
        sorted_norm=np.sort(norm)
        axs[0].plot(sorted_norm,np.arange(1,count+1)/count,color=COLORS[j],label=name,lw=2)
        errors=np.array([metrics[i,KINDS[j]]['mse'] for i in range(count)])
        axs[1].scatter(np.full(count,j)+np.linspace(-.14,.14,count),errors,s=12,alpha=.5,color=COLORS[j])
        axs[1].plot([j-.25,j+.25],[errors.mean()]*2,color='#172b38',lw=3)
        axs[1].annotate(f'{errors.mean():.3g}',(j,max(errors)),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    axs[0].set(xlabel='Recovered input RMS over all 210 coordinates',ylabel='Fraction of chunks at or below x')
    axs[1].set_xticks(range(5),['Recorded','Uniform','Gaussian','Random ±1','Heavy tails'],rotation=25)
    axs[1].set_ylabel('Actual production round-trip MSE');axs[1].set_yscale('symlog',linthresh=1e-8);axs[1].margins(y=.25)
    cumulative=np.cumsum(ratios)
    axs[2].plot(np.arange(1,len(ratios)+1),cumulative,color='#173044',lw=2)
    axs[2].axvline(3,color='#e57b25',ls='--');axs[2].scatter([3],[cumulative[2]],color='#e57b25')
    axs[2].set(xlabel='Number of pooled principal components',ylabel='Fraction of variance retained',ylim=(0,1.03))
    for ax in axs:ax.grid(alpha=.2)
    diagnostics.legend(handles,names,loc='outside upper center',ncol=3,fontsize=10)
    save(diagnostics,'pca_diagnostics.png')

    body='''<style>body{max-width:1450px;margin:auto;padding:28px 24px 70px;background:#f4f5f0;color:#172b38;font-size:17px;--muted:#536570;--card:#fff;--bg:#f4f5f0;--fg:#172b38;--line:#cdd7d8;--accent:#166ba5}h1{font-size:37px}h2{font-size:25px}p{max-width:1150px}.hero{background:#fff;border-top:5px solid #1878bc;padding:22px;border-radius:9px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.card{background:#fff;border-radius:8px;padding:17px}.card b{font-size:23px;display:block}section{margin-top:34px}figure{margin:15px 0}figure img{max-width:100%;width:100%;border-radius:8px}.plotwrap{background:#fafbf9;border-radius:8px;margin:18px 0}.badge{font-weight:700;color:#166ba5}a{color:#166ba5}code{overflow-wrap:anywhere}@media(max-width:650px){body{padding:18px 10px}h1{font-size:29px}}</style>'''
    body+='<div class="eyebrow">RECOVERED INPUTS · COMMON 3D PCA</div><h1>Where do different actions land in noise space?</h1>'
    body+=f'<div class="hero"><b>{count} points per group · {len(flat)} total · {flat.shape[1]} coordinates per input</b><p>The same {count} observation contexts are used in all five groups. Each dot is one complete recovered input that was fed back through the production sampler.</p><p><b>These three axes retain {ratios[:3].sum():.1%} of the pooled variance.</b> They are one shared PCA, fitted to all groups together. No per-group centering, scaling, or whitening is applied.</p></div>'
    if not header['complete']:
        body+=f'<p class="badge">Live partial result: {count}/{header["points_per_group"]} contexts complete. Reload after an update. PCA axes may rotate as more points arrive.</p>'
    body+='<p><b>Color names refer to the target actions, not to the recovered noise distribution.</b> For example, “Uniform actions” means a uniform random action was inverted; the resulting noise need not be uniform. “Original recorded actions” uses the demonstrated action at the same observation frame.</p>'
    body+='<section><h2>Rotate the shared noise cloud</h2><p>Drag to rotate, scroll to zoom, and click a legend entry to hide or show a group. Double-click a legend entry to isolate it. Hover for the observation frame, reconstruction MSE, and input magnitude. All three spatial axes use equal geometric scale.</p><script src="plotly.min.js"></script><div class="plotwrap">'+plot+'</div></section>'
    body+='<p>Points are not individual joints or timesteps. We flatten each 30-step × 7-joint recovered input into one 210-dimensional vector, subtract the <b>same pooled mean</b>, and project onto the three directions with the most pooled variance. PC1 explains the most variation; PC2 and PC3 explain the next largest variations along perpendicular directions. These are combinations of noise coordinates, not individual joints. Nearby dots have similar projections; separation outside these three axes is not visible. Broader groups have more influence on the shared PCA directions; the full-dimensional magnitude plot below checks spread without this projection.</p>'
    body+='<section><h2>The same projection, without 3D occlusion</h2><p>Left: PC1 versus PC2. Middle: PC1 versus PC3. Right: PC2 versus PC3. These use exactly the same fitted axes as the interactive plot. No PCA is re-fitted for a group or panel.</p><figure><img src="pca_projections.png" alt="Three pairwise projections of one shared noise PCA"></figure></section>'
    body+='<section><h2>Check what PCA leaves out, and whether the inputs reconstruct</h2><p><b>Left:</b> recovered-input RMS in the full 210-dimensional space; a curve farther right means larger inputs. <b>Middle:</b> round-trip MSE for each recovered input, evaluated using the actual production sampler. Dark marks are arithmetic means; dots are individual contexts. <b>Right:</b> cumulative variance retained as more PCA directions are included; the orange marker is the 3D view.</p><figure><img src="pca_diagnostics.png" alt="Full-dimensional noise magnitude, reconstruction errors, and PCA variance coverage"></figure></section>'
    body+='<div class="cards">'+''.join(f'<div class="card" style="border-top:4px solid {COLORS[j]}">{name}<b>{stats[KINDS[j]]["mean_roundtrip_mse"]:.7g}</b>mean round-trip MSE</div>' for j,name in enumerate(LABELS))+'</div>'
    body+=f'<section><h2>Exactly what was run</h2><p><b>{header["num_steps"]} flow steps and {header["refine"]} inverse corrections per step</b> for every group. Each network evaluation is a normal forward pass. Inversion subtracts the learned velocity; refinement revises the previous-state estimate. There is no training or gradient-based optimization.</p>'
    body+='<p>The five targets are: recorded actions; independent uniform [−1,1]; independent Gaussian with σ=1/√3; independent random −1 or +1; and independent Student t₃/3. Uniform, Gaussian and heavy-tailed targets have matching population variance 1/3; random ±1 has variance 1.</p>'
    body+='<p>Targets are sampled directly in the expert’s normalized action space after anchor encoding and quantile normalization. Gaussian and heavy-tailed targets stay unclipped. Padding remains zero and is excluded from PCA and MSE. We plot the recovered input <b>after the sampler dtype cast</b>, because this is the input actually used for the reconstruction. The float32 inverse endpoint is also saved.</p>'
    body+='<p>The five inversions share each observation context and are evaluated as independent batch rows for speed. Each recovered candidate is then checked individually through the production generation method, restoring the same depth/state/history conditioning. Reported MSE therefore measures the actual sampler, not only a batched approximation.</p>'
    episode_counts={}
    for row in header['samples'][:count]:episode_counts[row['episode_idx']]=episode_counts.get(row['episode_idx'],0)+1
    body+='<p><b>Data source:</b> the local validation split <code>'+html.escape(header['val_root'])+'</code>. These are validation observations and their recorded action chunks; synthetic targets reuse the same observations.</p>'
    body+=f'<p>Sampling seed: <b>{header["seed"]}</b>. Completed anchors per episode: '+', '.join(f'{ep}: {n}' for ep,n in sorted(episode_counts.items()))+'. Anchors are approximately evenly spaced within episodes and snapped to the 3-frame image/depth grid. Selection is independent of reconstruction error.</p>'
    body+=f'<p>Contexts are spread across {len(episode_counts)} episodes with nearly equal episode quotas, then evenly spaced within each episode and snapped to the image stride. Frames in the same episode can be dependent; repeated episode-end targets are retained. MSE and coordinates are saved for every point; no high-error or large-radius point is removed. A 3D cloud alone cannot establish Gaussianity or full-dimensional overlap.</p></section>'
    body+='<details><summary>Downloads and checkpoint</summary><p><code>'+html.escape(header['checkpoint'])+'</code></p><p><a href="pca_3d.png">Static 3D figure</a> · <a href="noise_pca.npz">Full vectors, shared PCA basis and coordinates</a> · <a href="points.csv">Point identities and reconstruction MSE</a> · <a href="summary.json">Summary</a> · <a href="noise_capture.json">Capture provenance</a>. Per-context noise_*.npz files contain targets, recovered inputs and actual reconstructions. No external scripts or network access are needed to view this page.</p></details>'
    page(out/'noise_pca.html','Recovered noise · shared 3D PCA',body)
    print(f'Noise PCA report: {out / "noise_pca.html"} ({count} points/group, {ratios[:3].sum():.1%} variance)',flush=True)

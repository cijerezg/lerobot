"""Readable, explicitly GT-rooted comparisons of recorded and fake endpoints."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from lerobot.probes.flow_interpolation import ALPHAS, path_geometry
from lerobot.probes.flow_noise_pca import shared_pca
from lerobot.probes.report_html import page

COLORS = ['#ce671a', '#168166', '#9258af', '#cf475a']
RECORDED_COLOR = '#17658c'
FAKE_COLOR = '#c15322'
FAKE_NAMES = ['Uniform U[−1, 1]', 'Gaussian σ=1/√3', 'Random ±1', 'Student-t(3)/3']
ANCHOR_NAMES = ['Cup / tape / bottle sorting', 'Put shirts in bin', 'Fold the bit kit closed']


def path_label(pair, family):
    number = pair['pair']
    return f'GT → Fake F{number}' if family else f'GT → Recorded R{number}'


def render(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    out = Path(out_dir)
    capture = json.loads((out / 'matched.json').read_text())
    header, frames = capture['header'], capture['frames']
    if header.get('comparison') != 'recorded_vs_direct_fake':
        raise ValueError('This report requires direct fake-distribution captures')
    arrays = []
    for i in range(len(frames)):
        with np.load(out / f'matched_{i:02d}.npz') as a:
            valid = a['valid'].astype(bool)
            arrays.append(dict(target=a['target'][..., valid].astype(float),
                               noise=a['noise_input'][..., valid].astype(float),
                               recon=a['reconstruction'][..., valid].astype(float),
                               indices=a['path_indices']))
    flat = np.concatenate([a['noise'].reshape(len(a['noise']), -1) for a in arrays])
    fit = shared_pca(flat)
    scores = np.split(fit['scores'], np.cumsum([len(a['noise']) for a in arrays])[:-1])
    ratios = fit['explained_variance_ratio']
    np.savez_compressed(out / 'matched_pca.npz', **fit)
    (out / 'plotly.min.js').write_text(get_plotlyjs())

    def asset(name):
        # Content-based URLs prevent an old PNG legend surviving a page refresh.
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()[:12]
        return f'{name}?v={digest}'

    def photograph(photo, caption):
        return f'<figure><img src="{asset(photo["file"])}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

    low, high = fit['scores'].min(0), fit['scores'].max(0)
    center = (low + high) / 2
    half = max(float((high - low).max()) * .55, 1e-5)
    scene = {axis: dict(title=f'PC{j+1} · {ratios[j]:.1%}', range=[center[j]-half, center[j]+half],
                        backgroundcolor='#f4f7f9', showbackground=True, gridcolor='#dce4e8')
             for j, axis in enumerate(['xaxis', 'yaxis', 'zaxis'])}
    scene.update(aspectmode='cube', camera=dict(eye=dict(x=1.8, y=1.8, z=1.35)))
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    metrics = [np.mean((a['recon']-a['target'])**2, axis=(1, 2)) for a in arrays]
    geometries = [path_geometry(a['target'], a['noise'], a['indices']) for a in arrays]
    bend_limit = max(float(g['straight_line_deviation'].max()) for gs in geometries for g in gs) * 1.12
    mse_low = min(float(m[m > 0].min()) for m in metrics) * .65
    mse_high = max(float(m.max()) for m in metrics) * 1.5
    distance_limit = max(max(p['recorded_endpoint_mse'], p['fake_endpoint_mse']) for f in frames for p in f['pairs']) * 1.28

    body = '''<style>
:root{color-scheme:light;--bg:#f4f6f8;--fg:#1b2b39;--card:#fff;--muted:#536371;--line:#d6e0e7;--accent:#17658c}
body{max-width:1500px;margin:auto;padding:28px 28px 70px;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
h1{font-size:32px;line-height:1.2;margin:8px 0 14px}h2{font-size:23px;margin:0 0 8px}h3{font-size:18px;margin:0 0 8px}p{margin:6px 0 12px}.muted{color:var(--muted);font-size:14px}.intro{max-width:1000px}.routes,.two,.pair-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.route{padding:12px 16px;border-left:4px solid;background:white}.recorded{border-color:#17658c}.fake{border-color:#c15322}.route strong{display:block;font-size:19px}.route span{font-size:14px}
nav{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;background:#f4f6f8f5;padding:14px 0;z-index:5;margin-top:10px;border-bottom:1px solid var(--line)}nav button{border-radius:7px;padding:10px 15px;background:white;font-size:14px}nav button[aria-selected=true]{background:#193f58;color:white;border-color:#193f58}
.panel[hidden]{display:none}.block{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;margin-top:20px}.anchor{display:grid;grid-template-columns:1.3fr 1fr;gap:24px;align-items:center}.photos{display:flex;gap:10px}.photos figure{flex:1;min-width:0;margin:0}.photos img{width:100%;max-height:175px;object-fit:contain;border-radius:6px;background:#e9eef1}figcaption{font-size:12px;color:var(--muted)}.plot-image{display:block;width:100%;height:auto}.warning{padding:10px 14px;background:#fff1da;border-left:4px solid #c48424;font-size:14px;margin:12px 0}.model{min-width:0}.model h3{margin-top:8px}.pair-card{border:1px solid var(--line);border-radius:8px;padding:14px;min-width:0}.pair-card p{font-size:14px}.endpoint-id{font-weight:750;font-size:20px}.bad{color:#995b13;font-weight:650}.pair-map{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.pair-map span{padding:6px 11px;background:#edf2f5;border-radius:5px;font-size:14px}.endpoint-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.endpoint-grid .photos img{max-height:210px}.formula{font-family:ui-monospace,monospace;font-size:14px}a{color:#17658c}code{overflow-wrap:anywhere}.anchor-number{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#536371}.status{font-size:13px;color:#536371}.definition{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}
@media(max-width:850px){body{padding:15px 12px}h1{font-size:26px}.routes,.two,.pair-grid,.anchor,.endpoint-grid{grid-template-columns:1fr}.block{padding:14px}nav{position:static}.photos img{max-height:150px}}
</style><script src="plotly.min.js"></script>'''
    body += f'<div class="status">CHECKPOINT 1200 · {len(frames)}/3 ANCHORS COMPLETE · 30 STEPS × 7 JOINTS</div>'
    body += '<h1>Two action paths from the same ground truth</h1><p class="intro">For each observation, start at its recorded ground-truth action (<b>GT</b>). Blend toward either another recorded chunk or a directly sampled fake action. Independently invert every blend, then measure reconstruction with the production sampler.</p>'
    body += '<div class="routes"><div class="route recorded"><strong>GT → Recorded R1–R4</strong><span>Endpoints from four other dataset frames.</span></div><div class="route fake"><strong>GT → Fake F1–F4</strong><span>Endpoints sampled from four named distributions.</span></div></div>'
    body += '<p class="muted">R1 is distance-matched with F1, R2 with F2, and so on. Pairing compares distances; both interpolation paths start at GT. The same eight endpoints are fixed across all three anchors.</p>'
    body += '<nav aria-label="Report sections">'
    for i in range(len(frames)):
        body += f'<button role="tab" data-panel="anchor-{i}" aria-selected="{str(i == 0).lower()}">{i+1}. {ANCHOR_NAMES[i]}</button>'
    body += '<button role="tab" data-panel="endpoints" aria-selected="false">Endpoint reference</button><button role="tab" data-panel="methods" aria-selected="false">Method & downloads</button></nav>'
    summary = []
    for i, (frame, a, xyz, mse, geometry) in enumerate(zip(frames, arrays, scores, metrics, geometries)):
        body += f'<div class="panel" id="anchor-{i}" {"hidden" if i else ""}><div class="block anchor"><div><div class="anchor-number">Anchor {i+1} · fixed observation</div><h2>{html.escape(ANCHOR_NAMES[i])}</h2><p>Episode {frame["episode_idx"]} · frame {frame["frame_idx"]} · global index {frame["global_idx"]}</p><p class="muted">Task: {html.escape(frame["task"])}<br>Subtask: {html.escape(str(frame["subtask"]))}</p><p class="muted">GT reconstruction MSE: {mse[0]:.3g}. Images, depth, state, task and subtask stay fixed for every path below.</p></div><div class="photos">'
        body += ''.join(photograph(p, 'Anchor · '+p['camera']) for p in frame['photographs'])+'</div></div>'
        body += '<section class="block"><h2>1. Compare endpoint distances from GT</h2><p>Each pair compares one recorded endpoint (R) with one fake endpoint (F). Bar heights and numbers are <b>endpoint-distance MSE</b>; percentage gaps describe matching quality.</p>'
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.1), layout='constrained', sharey=True)
        for j, (pair, ax) in enumerate(zip(frame['pairs'], axes)):
            n = j+1
            bars = ax.bar([0, 1], [pair['recorded_endpoint_mse'], pair['fake_endpoint_mse']], color=[RECORDED_COLOR, FAKE_COLOR], width=.58)
            ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=11)
            ax.set_xticks([0, 1], [f'Recorded R{n}', f'Fake F{n}\n{FAKE_NAMES[j]}'], fontsize=10)
            ax.set_title(f'Pair {n} · gap {pair["relative_mismatch"]:.1%}', fontsize=12)
            ax.set_ylim(0, distance_limit);ax.grid(axis='y', alpha=.18);ax.set_axisbelow(True)
        axes[0].set_ylabel('Endpoint-distance MSE')
        filename = f'endpoint_distances_{i}.png'
        fig.savefig(out / filename, dpi=160);plt.close(fig)
        body += f'<img class="plot-image" src="{asset(filename)}" alt="Four pairs: distance from GT to Recorded R1–R4 and Fake F1–F4">'
        gap = frame['pairs'][2]['relative_mismatch']
        body += f'<div class="warning"><b>Pair 3 is poorly matched:</b> the random ±1 sample F3 is farther from GT than recorded R3; the gap is {gap:.1%}. Differences in this pair remain confounded by action distance.</div><p class="muted">Gap = |recorded distance − fake distance| / fake distance. Endpoint distance averages squared differences over the 210 normalized action coordinates. It is separate from reconstruction error.</p></section>'

        body += '<section class="block"><h2>2. Inspect recovered-noise paths</h2><p>Every curve starts at the black <b>GT, α=0</b> marker. A square marks α=1 at its named endpoint. These axes show recovered noise, after inversion of the action blends.</p>'
        body += f'<p class="muted">One PCA basis and identical axis ranges across all anchors and both panels; 3D retains {ratios[:3].sum():.1%} of pooled noise variance. Drag to rotate; the paired views rotate together.</p><div class="two">'
        for family in range(2):
            fig = go.Figure()
            for j, (pair, color) in enumerate(zip(frame['pairs'], COLORS)):
                idx = a['indices'][2*j+family]
                dist = pair['fake_endpoint_mse' if family else 'recorded_endpoint_mse']
                name = f'F{j+1} · {FAKE_NAMES[j]}' if family else f'R{j+1} · episode {pair["recorded_episode_idx"]}, frame {pair["recorded_frame_idx"]}'
                fig.add_trace(go.Scatter3d(x=xyz[idx, 0].tolist(), y=xyz[idx, 1].tolist(), z=xyz[idx, 2].tolist(), mode='lines+markers', name=name,
                    line=dict(color=color, width=5, dash='dash' if family else 'solid'), marker=dict(color=color, size=4), customdata=np.column_stack([ALPHAS, mse[idx]]).tolist(),
                    hovertemplate=f'{path_label(pair, family)}<br>{html.escape(name)}<br>Endpoint-distance MSE: {dist:.6g}<br>α=%{{customdata[0]:.1f}}<br>Reconstruction MSE: %{{customdata[1]:.7g}}<extra></extra>'))
                k = idx[-1]
                fig.add_trace(go.Scatter3d(x=[xyz[k,0]], y=[xyz[k,1]], z=[xyz[k,2]], mode='markers', marker=dict(color=color, size=6, symbol='square'), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter3d(x=[xyz[0,0]], y=[xyz[0,1]], z=[xyz[0,2]], mode='markers+text', marker=dict(color='#1b2b39', size=7, symbol='diamond'), text=['GT, α=0'], textposition='bottom center', showlegend=False, hovertemplate=f'Anchor GT<br>Reconstruction MSE {mse[0]:.7g}<extra></extra>'))
            fig.update_layout(height=540, scene=scene, margin=dict(l=0,r=0,t=100,b=0), paper_bgcolor='white', font=dict(family='system-ui',size=12), legend=dict(x=0,y=1.2,orientation='h',font=dict(size=11)))
            title = 'GT → Fake F1–F4' if family else 'GT → Recorded R1–R4'
            formula = '(1 − α) × GT + α × F' if family else '(1 − α) × GT + α × R'
            body += f'<div class="model"><h3>{title}</h3><div class="formula">{formula}</div>'+fig.to_html(full_html=False,include_plotlyjs=False,div_id=f'matched-{i}-{family}',config=dict(displaylogo=False,responsive=True))+'</div>'
        body += '</div></section>'

        body += '<section class="block"><h2>3. Compare bending and reconstruction, one pair at a time</h2><p>Each card contains only two paths. <b>Blue solid: GT → recorded R.</b> <b>Orange dashed: GT → fake F.</b> Every bending plot uses the same scale; every reconstruction plot uses the same log scale.</p><div class="pair-grid">'
        paths = []
        for j, pair in enumerate(frame['pairs']):
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), layout='constrained')
            for family, color, style in [(0, RECORDED_COLOR, '-'), (1, FAKE_COLOR, '--')]:
                k=2*j+family;idx=a['indices'][k];label=path_label(pair,family)
                axes[0].plot(ALPHAS, geometry[k]['straight_line_deviation'], color=color, ls=style, marker='o', ms=3, label=label)
                axes[1].plot(ALPHAS, mse[idx], color=color, ls=style, marker='o', ms=3)
                paths.append(dict(pair=j+1, family='fake' if family else 'recorded', mean_reconstruction_mse=float(mse[idx].mean()),max_reconstruction_mse=float(mse[idx].max()),endpoint_reconstruction_mse=float(mse[idx[-1]]),max_bend_rms=float(geometry[k]['straight_line_deviation'].max())))
            axes[0].set(title='Noise-path bending', ylabel='RMS deviation from straight line', ylim=(0,bend_limit))
            axes[1].set(title='Round-trip reconstruction', ylabel='Reconstruction MSE', yscale='log', ylim=(mse_low,mse_high))
            for ax in axes:
                ax.set_xlabel('Action blend α');ax.set_xticks([0,.5,1]);ax.grid(alpha=.2)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside upper center', ncol=2, fontsize=11)
            filename=f'pair_paths_{i}_{j}.png';fig.savefig(out/filename,dpi=150);plt.close(fig)
            body += f'<div class="pair-card"><h3>Pair {j+1}: Recorded R{j+1} and Fake F{j+1}</h3><p>R{j+1}: episode {pair["recorded_episode_idx"]}, frame {pair["recorded_frame_idx"]}<br>F{j+1}: {FAKE_NAMES[j]}</p>'
            if j==2:body += f'<p class="bad">Poor distance match · {pair["relative_mismatch"]:.1%} gap</p>'
            body += f'<img class="plot-image" src="{asset(filename)}" alt="Two paths: GT to Recorded R{j+1} and GT to Fake F{j+1}; bending and reconstruction MSE"></div>'
        body += '</div><p class="definition"><b>Bending</b> is RMS deviation from the straight line between that path’s recovered-noise endpoints, using all 210 coordinates. <b>Reconstruction MSE</b> compares each blended target action with the production sampler’s output from its recovered input. Larger reconstruction error makes the apparent inverse-path geometry less reliable.</p></section></div>'
        summary.append(dict(global_idx=frame['global_idx'],task=frame['task'],pairs=frame['pairs'],paths=paths))

    body += '<div class="panel" id="endpoints" hidden><section class="block"><h2>The eight fixed endpoints</h2><p><b>R means recorded; F means fake.</b> Each R is a chunk from the dataset. Each F is a direct random draw in normalized action space. Their IDs and tensors stay fixed across the three anchors.</p><div class="pair-map">'
    body += ''.join(f'<span>Pair {j+1}: R{j+1} and F{j+1}</span>' for j in range(4))+'</div><div class="endpoint-grid">'
    for j, pair in enumerate(capture['shared_pairs']):
        body += f'<article class="pair-card"><h3>Pair {j+1} · paired by distance</h3><div class="endpoint-id">Recorded R{j+1}</div><p>Episode {pair["recorded_episode_idx"]} · frame {pair["recorded_frame_idx"]} · global index {pair["recorded_global_idx"]}<br>{html.escape(pair["task"])}</p><div class="photos">'
        body += ''.join(photograph(p, f'R{j+1} · '+p['camera']) for p in pair['photographs'])+'</div>'
        body += f'<p class="muted">Subtask: {html.escape(str(pair["subtask"]))}. Training clamp affected {pair["recorded_clipped_fraction"]:.1%} of coordinates.</p><div class="definition"><div class="endpoint-id">Fake F{j+1} · {FAKE_NAMES[j]}</div><p>Independent values at each of 30 steps × 7 joints. Seed {pair["seed"]}. No rescaling or clipping.</p><p class="muted">|value| &gt; 1: {pair["fake_fraction_outside_unit"]:.1%}; maximum |value|: {pair["fake_max_abs"]:.3g}.</p></div></article>'
    body += '</div></section></div>'
    body += f'''<div class="panel" id="methods" hidden><section class="block"><h2>Coordinate system and fixed conditioning</h2><p>Recorded chunks use their own starting states for anchor encoding, followed by the checkpoint’s saved per-timestep quantile normalization and training clamp to [−1,1]. R1–R4 are fixed normalized displacement patterns, not absolute joint positions transferred into a new anchor. The valid-joint mask selects the same 30 × 7 coordinates throughout.</p><p>Fake endpoints are sampled directly in normalized coordinates: U[−1,1], Gaussian σ=1/√3, independent ±1, and Student-t(3)/3. Gaussian and Student-t samples remain unbounded and unclipped. No fake sample is normalized twice.</p><p>For each anchor, images, depth, history, state, task and subtask stay fixed. Every production forward rebuilds _model_inputs(batch), preserving consume-once conditioning stashes.</p></section><section class="block"><h2>Distance selection and limits</h2><p>The recorded pool contains {header['pool_size']} complete chunks from validation episodes {header['pool_episodes']}, sampled every three frames. Chunks within 60 frames of an anchor are excluded, and eligible recorded endpoints have MSE ≥ {header['minimum_recorded_endpoint_mse']} from every anchor. Selected recorded chunks do not overlap.</p><p>The fake pool contains 64 draws per distribution. Starting with the hardest distribution to match, selection minimizes the worst relative endpoint-distance discrepancy across all three anchors. No endpoint is rescaled. Selection uses no recovered-noise or reconstruction outcomes.</p><p>Distance matching conditions the chosen samples; four pairs do not characterize entire distributions. Matching does not control temporal smoothness, direction or task compatibility. The ±1 pair remains poorly matched and is marked throughout.</p></section><section class="block"><h2>Solver and downloadable evidence</h2><p>α = 0, 0.1, …, 1. Each target is independently inverted with {header['num_steps']} flow steps and {header['refine']} fixed-point inverse corrections. No warm start from neighboring α values. GT at α=0 is shared, giving 81 unique targets per anchor and 243 total. Each recovered input is checked individually using the production sampler.</p><p>Capture time: {header['elapsed_seconds']:.1f} seconds after model loading. The shared PCA basis explains {ratios[:3].sum():.1%} of pooled variance in three dimensions; all quantitative geometry and MSE use 210 dimensions. During a partial run the PCA basis is recomputed as anchors finish; within each report version all panels share it.</p><p><a href="matched.json">Experiment metadata</a> · <a href="summary.json">Measurements</a> · <a href="shared_endpoints.npz">Exact endpoints and candidate pools</a> · <a href="matched_pca.npz">Shared PCA basis</a></p><p>Per-anchor captures: <a href="matched_00.npz">Anchor 1</a> · <a href="matched_01.npz">Anchor 2</a> · <a href="matched_02.npz">Anchor 3</a>. Captures contain targets, recovered inputs, reconstructions, path indices and masks.</p></section></div>'''
    (out/'summary.json').write_text(json.dumps(dict(complete=header['complete'],pca_variance_3d=float(ratios[:3].sum()),max_relative_mismatch=max(p['relative_mismatch'] for f in frames for p in f['pairs']),frames=summary),indent=2,allow_nan=False))
    script = '''
const tabs=[...document.querySelectorAll('nav button')];
function showPanel(id){
 if(!document.getElementById(id))return;
 for(const p of document.querySelectorAll('.panel'))p.hidden=p.id!==id;
 for(const t of tabs)t.setAttribute('aria-selected',String(t.dataset.panel===id));
 history.replaceState(null,'','#'+id);
 requestAnimationFrame(()=>document.querySelectorAll('#'+id+' .js-plotly-plot').forEach(p=>Plotly.Plots.resize(p)));
}
for(const t of tabs)t.addEventListener('click',()=>showPanel(t.dataset.panel));
if(location.hash)showPanel(location.hash.slice(1));
'''
    script += f'''for(let i=0;i<{len(frames)};i++){{let lock=false;const a=document.getElementById(`matched-${{i}}-0`),b=document.getElementById(`matched-${{i}}-1`);for(const [src,dst] of [[a,b],[b,a]])src.on('plotly_relayout',e=>{{if(lock||!e['scene.camera'])return;lock=true;Plotly.relayout(dst,{{'scene.camera':e['scene.camera']}}).then(()=>lock=false);}});}}'''
    page(out/'matched.tmp.html', 'GT → recorded vs GT → fake · checkpoint 1200', body, script)
    (out/'matched.tmp.html').replace(out/'matched.html')
    print(f'Matched report ready: {out / "matched.html"} ({len(frames)}/3 anchors)',flush=True)

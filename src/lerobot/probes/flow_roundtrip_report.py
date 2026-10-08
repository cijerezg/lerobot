"""Explain action reconstruction with MSE, action overlays, and flow-time plots."""
from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lerobot.probes.report_html import data_json, page


COLORS = ['#1878bc', '#e07a24', '#27957a', '#9561b5', '#ca5570', '#707b86']


def mse_summary(rows):
    """Average squared errors, never square an average/median of per-frame RMS."""
    values = np.array([r.get('mse', r['rms'] ** 2) for r in rows], dtype=float)
    return dict(mean_mse=float(values.mean()), median_mse=float(np.median(values)),
                p95_mse=float(np.quantile(values, .95)), max_mse=float(values.max()))


def label(n, refine, multiline=False):
    sep = '\n' if multiline else ' · '
    return f'{n} flow steps{sep}' + ('no backward corrections' if refine == 0 else f'{refine} corrections per backward step')


def render(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    from lerobot.probes.utils import joint_names_for_dim

    out = Path(out_dir)
    data = json.loads((out / 'roundtrip.json').read_text())
    records, header = data['records'], data['header']
    for row in records:
        row.setdefault('mse', row['rms'] ** 2)
        row.setdefault('fp32_mse', row['fp32_rms'] ** 2)
        row.setdefault('random_baseline_mse', row['random_baseline_rms'] ** 2)
    frame_ids = list(dict.fromkeys(r['global_idx'] for r in records))
    frame_rows = {f: next(r for r in records if r['global_idx'] == f) for f in frame_ids}
    groups = defaultdict(list)
    for row in records:
        groups[row['target_kind'], row['num_steps'], row['refine'], row['start_step']].append(row)
    settings = sorted({(r['num_steps'], r['refine']) for r in records})
    base = (20, 0) if (20, 0) in settings else settings[0]
    settings = [base] + [s for s in settings if s != base]
    names = [label(*s, multiline=True) for s in settings]
    colors = {s: COLORS[i % len(COLORS)] for i, s in enumerate(settings)}

    # Existing captures contain full tensors. Recompute endpoint MSE directly in
    # float64; old JSON only has rounded float32 RMS. Keep its fallback for portability.
    errors, explorer = defaultdict(list), []
    joints = None
    for i, f in enumerate(frame_ids):
        path = out / f'trajectories_{i:04d}.npz'
        if not path.exists():
            continue
        with np.load(path) as arrays:
            valid = arrays['valid'].astype(bool)
            target = arrays['demo'][0][..., valid].astype(float)
            joints = joint_names_for_dim(target.shape[-1])
            entry = dict(global_idx=f, episode_idx=frame_rows[f]['episode_idx'], frame_idx=frame_rows[f]['frame_idx'],
                         target=target.round(7).tolist(), paths={})
            for n, refine in settings:
                key = f'demo_n{n}_r{refine}'
                forward = arrays[key + '_forward_k0'][:, 0][..., valid].astype(float)
                entry['paths'][f'{n}_{refine}'] = forward.round(7).tolist()
                error = forward[-1] - target
                errors[n, refine].append(error)
                for kind in ('demo', 'generated'):
                    prefix = f'{kind}_n{n}_r{refine}'
                    if prefix + '_inverse' not in arrays:
                        continue
                    original = arrays[prefix + '_inverse'][-1, 0][..., valid].astype(float)
                    for row in records:
                        if (row['global_idx'], row['target_kind'], row['num_steps'], row['refine']) != (f, kind, n, refine):
                            continue
                        restored = arrays[f"{prefix}_forward_k{row['start_step']}"][-1, 0][..., valid].astype(float)
                        row['mse'] = float(np.mean((restored - original) ** 2))
                        fp32 = arrays[prefix + '_forward_fp32'][-1, 0][..., valid].astype(float)
                        row['fp32_mse'] = float(np.mean((fp32 - original) ** 2))
            explorer.append(entry)
    summaries = []
    for (kind, n, refine, k), rows in sorted(groups.items()):
        summaries.append(dict(target_kind=kind, num_steps=n, refine=refine, start_time=k/n, n=len(rows),
                              **mse_summary(rows), mean_fp32_mse=float(np.mean([r['fp32_mse'] for r in rows])),
                              median_rms=float(np.median([r['rms'] for r in rows])),
                              p95_rms=float(np.quantile([r['rms'] for r in rows], .95))))
    summary = dict(header=header, metric='mean of per-frame mean squared error over chunk positions and valid joints',
                   measurements=summaries,
                   sampler_parity_max_abs=max(r.get('sampler_parity_max_abs', 0) for r in records))
    summary['headline'] = dict(num_steps=base[0], refine=base[1], **mse_summary(groups['demo', *base, 0]))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    scalar_keys = sorted({k for r in records for k in r if not k.endswith('_by_step')})
    with (out / 'roundtrip.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#fafbf9', 'axes.facecolor': '#fafbf9', 'savefig.facecolor': '#fafbf9'})
    def finish(fig, filename):
        fig.savefig(out / filename, dpi=170, bbox_inches='tight')
        plt.close(fig)

    def style(ax, xlabel='', ylabel='Mean squared error', log=False):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(axis='y', alpha=.22)
        ax.set_axisbelow(True)
        if log:
            # Keep exact zeros visible in analytic tests without replacing data.
            ax.set_yscale('symlog', linthresh=1e-8)

    means = [mse_summary(groups['demo', *s, 0])['mean_mse'] for s in settings]
    fig, ax = plt.subplots(figsize=(11, 5.6), layout='constrained')
    for i, s in enumerate(settings):
        values = [r['mse'] for r in groups['demo', *s, 0]]
        ax.scatter(i + np.linspace(-.11, .11, len(values)), values, color=colors[s], alpha=.7, s=38)
        ax.plot([i-.23, i+.23], [means[i]]*2, color='#182f40', lw=3.5)
        ax.annotate(f'Mean {means[i]:.7f}', (i, max(values + [means[i]])), xytext=(0, 15), textcoords='offset points', ha='center', weight='bold')
    ax.set_xticks(range(len(settings)), names)
    style(ax, ylabel='Round-trip MSE · lower is better', log=True)
    ax.margins(y=.25)
    ax.set_title('Does the recovered noise reproduce the original action?', loc='left', pad=18, weight='bold')
    finish(fig, 'roundtrip.png')

    fig, ax = plt.subplots(figsize=(12, 4.6), layout='constrained')
    for s in settings:
        lookup = {r['global_idx']: r['mse'] for r in groups['demo', *s, 0]}
        ax.plot(range(len(frame_ids)), [lookup[f] for f in frame_ids], 'o-', color=colors[s], label=label(*s))
    ax.set_xticks(range(len(frame_ids)), [f"Episode {frame_rows[f]['episode_idx']}\nframe {frame_rows[f]['frame_idx']}" for f in frame_ids], fontsize=9)
    style(ax, ylabel='Round-trip MSE for this frame', log=True)
    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(.5, 1.25), ncol=2)
    finish(fig, 'mse_by_frame.png')

    if errors:
        cols = min(2, len(settings)); rows_n = (len(settings)+cols-1)//cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(12, 3.5*rows_n), squeeze=False, layout='constrained')
        maps = {s: np.mean(np.square(errors[s]), axis=0).T for s in settings}
        norm = Normalize(0, max(float(v.max()) for v in maps.values()) or 1e-12)
        for ax, s in zip(axes.flat, settings):
            m = maps[s]
            im = ax.imshow(m, aspect='auto', origin='lower', cmap='magma', norm=norm,
                           extent=[-.5, m.shape[1]-.5, -.5, m.shape[0]-.5])
            ax.set_title(label(*s), fontsize=11)
            ax.set(xlabel='Future robot step within the action chunk', yticks=range(len(joints)), yticklabels=joints)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for ax in list(axes.flat)[len(settings):]: ax.set_visible(False)
        fig.colorbar(im, ax=list(axes.flat), label='Mean squared reconstruction error · same scale in every panel', shrink=.85)
        finish(fig, 'mse_by_joint_and_step.png')

    fig, ax = plt.subplots(figsize=(10, 4.6), layout='constrained')
    for s in settings:
        selected = sorted([r for r in summaries if r['target_kind']=='demo' and (r['num_steps'],r['refine'])==s], key=lambda r:r['start_time'])
        ax.plot([r['start_time'] for r in selected], [r['mean_mse'] for r in selected], 'o-', color=colors[s], label=label(*s))
    style(ax, xlabel='How far backward did we go? Restart flow time s', ylabel='Final MSE after returning to t = 1')
    ax.set_xticks(header['restart_times'], [f'{t:g}' + ('\nFull round trip' if t==0 else '\nNo flow steps' if t==1 else '') for t in header['restart_times']])
    ax.legend(fontsize=9)
    finish(fig, 'mse_by_restart_time.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for s in settings:
        rows = groups['demo', *s, 0]
        times = np.linspace(0, 1, s[0]+1)
        for ax, key in zip(axes, ['target_error_by_step', 'retrace_error_by_step']):
            values = np.mean(np.square([r[key] for r in rows]), axis=0)
            ax.plot(times, values, color=colors[s], label=label(*s))
            ax.scatter([1], [values[-1]], color=colors[s], s=22)
    for ax in axes:
        style(ax, xlabel='Forward flow time t · noise (0) → action (1)', log=True)
    axes[0].set_title('Distance to the original action')
    axes[1].set_title('Distance to the inverse path at the same time')
    axes[0].legend(fontsize=8)
    finish(fig, 'mse_along_forward_flow.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    x = np.arange(len(settings))
    fp32 = [np.mean([r['fp32_mse'] for r in groups['demo', *s, 0]]) for s in settings]
    axes[0].plot(x, means, 'o-', color='#1878bc', label='Normal forward generation')
    axes[0].plot(x, fp32, 's--', color='#e07a24', label='Float32 forward diagnostic')
    axes[0].set_title('How much does forward precision matter?')
    axes[1].plot(x, means, 'o-', color='#1878bc', label='Recorded actions')
    axes[1].plot(x, [mse_summary(groups['generated', *s, 0])['mean_mse'] for s in settings], 's--', color='#27957a', label='Model-generated actions')
    axes[1].plot(x, [np.mean([r['random_baseline_mse'] for r in groups['demo', *s, 0]]) for s in settings], 'd:', color='#9561b5', label='Fresh noise → action (no inversion)')
    axes[1].set_title('Is inversion useful compared with fresh noise?')
    for ax in axes:
        ax.set_xticks(x, [f'{n} steps\n{r} corrections' for n,r in settings])
        style(ax, ylabel='Full round-trip MSE / baseline action MSE', log=True)
        ax.legend(fontsize=9)
    finish(fig, 'mse_controls.png')

    if explorer:
        worst = int(np.argmax([np.mean((np.array(e['paths'][f'{base[0]}_{base[1]}'][-1])-np.array(e['target']))**2) for e in explorer]))
        example = explorer[worst]; target = np.array(example['target'])
        cols = 2; fig, axes = plt.subplots((len(joints)+1)//2, cols, figsize=(12, 2.5*((len(joints)+1)//2)), squeeze=False, layout='constrained')
        for j, ax in enumerate(axes.flat):
            if j >= len(joints): ax.set_visible(False); continue
            ax.plot(target[:,j], color='#1c2d38', lw=2.5, label='Original recorded action')
            for s in settings:
                ax.plot(np.array(example['paths'][f'{s[0]}_{s[1]}'][-1])[:,j], '--', color=colors[s], label=label(*s))
            ax.set_title(joints[j], loc='left'); style(ax, xlabel='Future robot step within the chunk', ylabel='Normalized action')
        handles, labels_ = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='outside upper center', ncol=2, fontsize=9)
        fig.suptitle(f"Largest baseline error: episode {example['episode_idx']}, frame {example['frame_idx']}", fontsize=13)
        finish(fig, 'action_reconstruction_example.png')

    def figure(filename, title, description):
        return f'<section><h2>{title}</h2><p>{description}</p><figure><a href="{filename}"><img src="{filename}" alt="{html.escape(title)}" loading="lazy"></a></figure></section>'
    baseline = means[0]
    alternatives = [(s, m) for s, m in zip(settings, means) if s[0]==base[0] and s[1]>0]
    comparison = ''
    if alternatives:
        refined, refined_mse = min(alternatives, key=lambda v:v[1])
        ratio = baseline/refined_mse if refined_mse else float('inf')
        comparison = f'<p class="lead">With {refined[1]} corrections per backward step: <b>{refined_mse:.9f}</b> MSE — {ratio:.1f}× lower on these same frames.</p>'
    nframes = len(frame_ids); neps = len({r['episode_idx'] for r in records})
    body = '''<style>
body{max-width:1320px;margin:auto;padding:32px 28px 80px;background:#f4f5f0;color:#172b38;font-size:17px;--muted:#536570;--card:#fff;--bg:#f4f5f0;--fg:#172b38;--line:#cdd7d8;--accent:#166ba5}
h1{font-size:40px;line-height:1.14;max-width:950px}h2{font-size:26px;margin-top:0}h3{font-size:19px}p{max-width:1000px}.lead{font-size:20px}.eyebrow{font-weight:700}.hero{padding:28px;background:#fff;border-radius:12px;border-top:5px solid #1878bc}.big{font-size:55px;font-weight:750;letter-spacing:-2px;margin:6px 0}.metriclabel{font-size:17px;color:#536570}section{margin-top:42px}figure{margin:16px 0;background:#fafbf9;border:1px solid #d9e0df;border-radius:10px;overflow:hidden}figure img{display:block;width:100%}.flow{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:22px 0}.flow span{background:#e9f0f3;padding:14px;border-radius:8px}.definitions{display:grid;grid-template-columns:1fr 1fr;gap:18px}.definitions>div{background:#fff;padding:22px;border-radius:10px}.definitions p{margin:6px 0}.formula{padding:16px;background:#e7eef0;border-radius:8px;font-family:ui-monospace,monospace;font-size:15px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:22px 0}.card{border:1px solid #c8d5d9;padding:18px;background:#fff;border-radius:8px}.card b{font-size:24px;display:block;margin:6px 0}.note{color:#536570}.controls{background:#e7eef0;align-items:center}.controls label{min-width:0;max-width:100%}.controls select{background:#fff;color:#172b38;max-width:100%}.formula,code{overflow-wrap:anywhere}.jointplots{display:grid;grid-template-columns:1fr 1fr;gap:12px}.jointplots svg{width:100%;background:#fff;border-radius:8px}.live{padding:16px 0;font-size:19px;font-weight:650}input[type=range]{width:min(400px,80vw)}summary{color:#166ba5}a{color:#166ba5}
@media(max-width:700px){body{padding:20px 14px}h1{font-size:30px}.big{font-size:39px}.definitions,.jointplots{grid-template-columns:1fr}}
</style>'''
    body += '<div class="eyebrow">FLOW INVERSION · RECONSTRUCTION TEST</div><h1>Do we get the original action back?</h1>'
    body += f'<div class="hero"><div class="metriclabel">Mean full round-trip MSE · {label(*base)}</div><div class="big">{baseline:.9f}</div>{comparison}'
    body += f'<p>{nframes} recorded action chunks from {neps} validation episodes. Lower is better; <b>zero means exact reconstruction</b>. These values use the normal forward sampler.</p></div>'
    body += '<div class="flow"><span>Original recorded action<br><b>flow time 1</b></span><b>→ reverse flow →</b><span>Recovered noise<br><b>flow time 0</b></span><b>→ normal forward flow →</b><span>Reconstructed action<br><b>flow time 1</b></span></div>'
    body += '<p>For each camera observation and prompt, take its recorded action chunk, find an estimated noise input by running the flow backward, then generate forward from that input under the <b>same observation and prompt</b>. Compare the final action with the original.</p>'
    body += '<p><b>Every network evaluation is a normal forward pass.</b> “Reverse” describes the direction we move through flow time, not backpropagation. To generate, add Δt × model velocity; to estimate the previous state, subtract it. Refinement repeats normal network evaluations at revised estimates of that previous state. No gradients or weight updates are used.</p>'
    body += '<div class="formula">MSE = mean((reconstructed action − original action)²)</div>'
    body += '<p>We square each coordinate error, average across future robot steps and valid joints within a chunk, then average across sampled frames. The units are <b>squared normalized action units</b>, not joint degrees. Every frame has equal weight; padding joints are excluded. This is mean squared error, not RMS and not median error.</p>'
    body += '<section><h2>What are “steps” and “corrections”?</h2><div class="definitions"><div><h3>Flow steps (N)</h3><p>How many updates we take in each direction. <b>20 steps</b> means 20 updates from the action to noise and 20 updates back to the action. Each update moves flow time by 1/20. At 40 steps, each update is half as large.</p></div><div><h3>Inverse corrections (refinement)</h3><p>Extra attempts to undo <b>each individual forward update</b>. With 0, we take one backward estimate. With 4, we revise that estimate four times to try to make one forward update land back at the state we just left. The forward sampler is unchanged.</p></div></div>'
    body += '<p>More corrections cost more model evaluations. With R corrections, inversion uses N × (1 + R) velocity evaluations, followed by N for the full forward return. The corrections can improve the inverse but are not guaranteed to converge.</p>'
    body += '<div class="callout"><b>Why might a backward step need correcting?</b><p>Suppose a forward update is x → x + 0.2x: 1 becomes 1.2. Subtracting 0.2 × 1.2 from the destination gives 0.96, not the original 1. The update was evaluated at the wrong end of the step. Repeating y ← 1.2 − 0.2y gives 0.96 → 1.008 → 0.9984 → 1.00032, approaching 1.</p><p>That is what refinement tries to do with the model’s velocity at each backward step. It is optional numerical correction, not training or extra forward generation. The no-correction setting tests the original reverse-flow method.</p></div>'
    body += '<details><summary>The backward correction equation</summary><p>A forward step is x_next = y + Δt · v(y, t). Given x_next, start with a backward estimate y, then repeat y ← x_next − Δt · v(y, t). Four refinements means four such replacements per backward step.</p></details></section>'
    body += figure('roundtrip.png', '1. The full round-trip MSE, for every setting', 'Every dot is one recorded frame’s final MSE after action → noise → action. The dark horizontal mark and its label give the <b>arithmetic mean</b> across frames. All trips here reach noise at t=0 before returning. The vertical axis is logarithmic above 10⁻⁸ so small and large errors stay visible; lower is better.')
    body += '<div class="cards">' + ''.join(f'<div class="card">{label(*s)}<b>{m:.9f}</b><span>mean round-trip MSE</span></div>' for s,m in zip(settings,means)) + '</div>'
    body += figure('mse_by_frame.png', '2. Does the average hide a bad reconstruction?', 'The same endpoint MSE, now separated by episode and frame. Compare colors <b>at the same x position</b> to see which setting reconstructs that exact action best. Lines connect sampled frames only to help follow each setting; they do not show flow iterations. Frames within an episode are dependent.')
    if explorer:
        body += '<section><h2>3. Look at the actual reconstructed actions</h2><p>Select a frame and solver setting. Each small plot is one joint. <b>Black is the original recorded action; blue is the forward state.</b> The slider starts at the final reconstructed action. Move it left to watch the forward return from recovered noise.</p>'
        body += '<p>The x axis inside these plots is <b>future robot step within the action chunk</b>. The slider is <b>flow time</b>, progress from noise (0) to action (1). These are different axes.</p>'
        body += '<div class="controls"><label>Recorded frame<select id="frame"></select></label><label>Solver setting<select id="setting"></select></label><button id="worst">Show worst frame for this setting</button><label>Forward iteration<input id="iteration" type="range" min="0" value="20" max="20"></label></div><div id="live" class="live"></div><div id="jointplots" class="jointplots"></div></section>'
        body += figure('action_reconstruction_example.png', 'Action overlay to save or share', f"This fixed example is the largest-error frame under {label(*base)}: episode {example['episode_idx']}, frame {example['frame_idx']}. All colored curves are <b>final reconstructed actions</b>; black is the target. Each panel shows one joint across future robot steps. The interactive plots above let you inspect every other frame.")
        body += figure('mse_by_joint_and_step.png', '4. Where in the action chunk is the error?', 'Each cell is the squared error for one joint and one future robot step, averaged over the sampled frames after the full round trip. Brighter cells mean more error. <b>All panels use the same color scale</b>, so a darker panel really has smaller errors. This horizontal axis is robot-step position, not flow time.')
    body += '<section><h2>What does restarting at a flow time mean?</h2><p>The main result above always runs all the way back to noise: <b>1 → 0 → 1</b>. The next plot asks a separate question: what if we only reverse partway, then turn around?</p><div class="definitions"><div><h3>Restart at s = 0</h3><p>Action → recovered noise → action. This is the full round trip, and its endpoint error is the headline MSE.</p></div><div><h3>Restart at s = 0.75</h3><p>Action at 1 → intermediate state at 0.75 → action at 1. With 20 flow steps, that is 5 backward steps and 5 forward steps. The intermediate state is not pure noise.</p></div></div><p>Restart at s=1 takes no flow steps: it only casts the original action into the sampler’s numeric precision. A small error there is a precision floor, not successful noise recovery.</p></section>'
    body += figure('mse_by_restart_time.png', '5. How much error accumulates when we reverse farther?', 'X: the turnaround time s. Y: <b>mean final action MSE</b> after continuing from s back to 1. Each point compares a completed reconstruction to the same original action. Read left to right as full round trip → shorter round trips → no flow steps. These are separate experiments, not points along one forward trajectory.')
    body += figure('mse_along_forward_flow.png', '6. What happens during the forward return?', '<b>Both panels start at the recovered noise at t=0</b> and follow all forward iterations to t=1. Left: mean squared distance from the current forward state to the final target action; being far away near noise is expected. Right: mean squared distance from the current forward state to the saved inverse state at the same flow time; this shows whether the return retraces the backward path. At t=1, both compare against the original action. The endpoint dots therefore equal the full round-trip MSE. Axes are logarithmic above 10⁻⁸.')
    body += figure('mse_controls.png', '7. Precision and generated-action checks', '<b>Left:</b> take the same recovered noise and compare normal forward generation with a float32 diagnostic. This diagnostic changes both accumulation precision and time-conditioning precision, so the gap is their combined effect. More steps need not reduce error in the normal sampler. <b>Right:</b> blue repeats the recorded-action round trip; green starts with an action produced by the model from known noise, then inverts and reconstructs it; purple skips inversion and compares an action generated from fresh seeded noise against the recorded target. Purple is an action-error baseline, not a round trip. Every curve reports mean MSE; axes are logarithmic above 10⁻⁸.')
    body += f'<section><h2>What this run establishes</h2><p>These measurements tell us how closely the numerical inverse reconstructs these {nframes} sampled action chunks. They do not measure robot success or whether recovered noise is Gaussian. Samples from the same episode are correlated; these plots do not claim population confidence intervals.</p>'
    body += f'<p>The full forward continuation was checked against the production sampler: maximum coordinate discrepancy <b>{summary["sampler_parity_max_abs"]:.6g}</b>. {header.get("completed_frames", nframes)} of {header["n_frames"]} requested frames were captured.</p>'
    body += '<details><summary>Checkpoint, numerical precision, and downloads</summary><p>Checkpoint: <code>' + html.escape(header['checkpoint']) + '</code></p>'
    body += f'<p>Inverse accumulation: float32. Normal forward precision: {html.escape(header["forward_dtype"])}. Joint padding is excluded; repeated episode-end targets are retained. Step counts: {header["step_counts"]}; backward corrections: {header["refinements"]}.</p>'
    body += '<p><a href="roundtrip.csv">Frame metrics including MSE (CSV)</a> · <a href="summary.json">Mean MSE summaries (JSON)</a> · <a href="roundtrip.json">Original capture (JSON)</a>. Saved trajectories_*.npz contain complete tensors. Click any figure to open its full-resolution image.</p></details></section>'
    payload = dict(frames=explorer, joints=joints or [], settings=[dict(key=f'{n}_{r}', n=n, label=label(n,r)) for n,r in settings])
    script = 'const DATA=' + data_json(payload) + ';\n' + EXPLORER_JS if explorer else ''
    page(out / 'roundtrip.html', 'Round-trip MSE · action reconstruction', body, script)
    print(f'Round-trip MSE report: {out / "roundtrip.html"}', flush=True)


EXPLORER_JS = r'''
const frameSel=document.getElementById('frame'), settingSel=document.getElementById('setting'), slider=document.getElementById('iteration');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
DATA.frames.forEach((f,i)=>frameSel.add(new Option(`Episode ${f.episode_idx}, frame ${f.frame_idx}`,i)));
DATA.settings.forEach(s=>settingSel.add(new Option(s.label,s.key)));
function mse(a,b){let sum=0,n=0;for(let i=0;i<a.length;i++)for(let j=0;j<a[i].length;j++){sum+=(a[i][j]-b[i][j])**2;n++;}return sum/n;}
function worst(){let key=settingSel.value, best=-1;DATA.frames.forEach((f,i)=>{let p=f.paths[key],v=mse(p[p.length-1],f.target);if(v>best){best=v;frameSel.value=i;}});slider.value=slider.max;draw();}
function draw(){const f=DATA.frames[Number(frameSel.value)], path=f.paths[settingSel.value], n=path.length-1,k=Number(slider.value),current=path[k];
 document.getElementById('live').textContent=`${k===n?'Final reconstruction':'Intermediate forward state'} · iteration ${k}/${n} · flow time ${(k/n).toFixed(2)} · ${k===n?'Round-trip MSE':'MSE to final target'}: ${mse(current,f.target).toFixed(9)}`;
 document.getElementById('jointplots').innerHTML=DATA.joints.map((joint,j)=>{let a=f.target.map(v=>v[j]),b=current.map(v=>v[j]),vals=[...a,...b],lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.12,.005);lo-=pad;hi+=pad;
 const W=560,H=250,L=66,R=20,T=34,B=50,X=i=>L+i/Math.max(a.length-1,1)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
 let svg=`<svg role="img" aria-label="${esc(joint)} original action and forward state" viewBox="0 0 ${W} ${H}"><text x="${L}" y="22" fill="#172b38" font-size="16" font-weight="600">${esc(joint)}</text>`;
 for(let z=0;z<=4;z++){let v=lo+(hi-lo)*z/4,y=Y(v);svg+=`<path d="M${L} ${y}H${W-R}" stroke="#e0e7e9"/><text x="${L-8}" y="${y+4}" text-anchor="end" font-size="11" fill="#536570">${v.toFixed(3)}</text>`;}
 for(let i of [...new Set([0,Math.floor((a.length-1)/2),a.length-1])])svg+=`<text x="${X(i)}" y="${H-B+19}" text-anchor="middle" font-size="12" fill="#536570">${i}</text>`;
 for(const [values,color,dash] of [[a,'#1c2d38',''],[b,'#1878bc','6 3']])svg+=`<path d="${values.map((v,i)=>(i?'L':'M')+X(i)+','+Y(v)).join(' ')}" fill="none" stroke="${color}" stroke-width="2.4" stroke-dasharray="${dash}"/>`;
 svg+=`<text x="${W/2}" y="${H-9}" text-anchor="middle" font-size="12" fill="#536570">Future robot step within the chunk</text><text transform="translate(14 ${H/2}) rotate(-90)" text-anchor="middle" font-size="11" fill="#536570">Normalized action</text></svg>`;return svg;}).join('');}
settingSel.onchange=()=>{slider.max=DATA.frames[Number(frameSel.value)].paths[settingSel.value].length-1;slider.value=slider.max;draw();};
frameSel.onchange=draw;slider.oninput=draw;document.getElementById('worst').onclick=worst;
slider.max=DATA.frames[0].paths[settingSel.value].length-1;slider.value=slider.max;worst();
'''

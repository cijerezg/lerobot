"""Flow inversion follows a demonstrated action backward through the policy.

**Start with ordinary generation.** Given the camera images, state and prompt, the policy
starts with random numbers z drawn from a standard normal distribution. The action expert
repeatedly predicts a velocity that transforms that initial array into a 30-step action
chunk. The flow time t from 0 to 1 is progress through this transformation, not robot time.
The 30 positions inside a chunk are a separate axis: future robot timesteps.

**Now reverse the question.** Keep the same observation and prompt, start at the operator's
recorded action chunk, and follow the learned velocity field backward. The endpoint z is
an estimated noise array that would generate that demonstration under this context.
It is not sensor noise, action error, or a noise sample that was stored during training.
Each frame is inverted under its own context. We then compare the collection of recovered
arrays with the standard-normal distribution from which generation normally starts.

**Why expect normal noise?** If the learned conditional action distribution matched the
sampled demonstration distribution, and integration were exact, the backward-transformed
demonstrations would have the model's starting distribution. A mismatch in this experiment
can reflect the model, the sampled demonstrations, or numerical inversion. The probe alone
does not separate them or tell us how successful the robot would be.

**A forward check qualifies the result.** Feed each recovered z back through the same
context and measure how close it returns to the original demonstration. This is the
round-trip error. A perfect round trip would validate reconstruction for those samples,
but would not establish Gaussian noise or good policy behaviour. A nonzero error means
we must also account for imperfect numerical inversion before attributing noise mismatch.

Starting at the processed ground-truth action x(1), the probe takes 20 naive backward
Euler steps x(t-dt)=x(t)-dt*v(x(t),t), with dt=1/20 and no fixed-point refinement.
The recovered z=x(0) is integrated forward with 20 Euler steps and compared with the
same target. Backward Euler stepping here is explicit integration in reverse time,
not an exact inversion of the deployed discrete forward map. Round-trip error includes
integration and finite-precision effects; it is not action-prediction error.

Saved arrays contain 30 future timesteps by 7 valid joints (210 dimensions), excluding
padded joints. The checkpoint's saved processors and deployment preprocessing define
normalised action coordinates; values are not degrees or physical joint errors.
Validation uses 64 evenly spaced, stride-3-snapped frames per episode (512 / 8 episodes);
training uses 6 per episode (540 / 90 episodes), seed 42. Snapped duplicates are removed.
Frames in the same episode or overlapping chunks are dependent; neither frame counts
nor 256 projection directions are independent statistical replications.

Gaussian diagnostics compare uncentred, unstandardised recovered vectors with N(0,I).
Mean squared norm divided by d has expectation 1. The reference norm density is chi-square(d)/d;
the fractions below its first percentile or above its 99th percentile each have expectation .01.
Projection directions are random unit vectors; QQ pools their quantiles, while each KS
value is the maximum empirical-CDF distance from N(0,1). The Gaussian reference has the
same n and d and the same projection directions within a split (analysis RNG seed 0).
KS is a distance, not a p-value. Projection tests include scale and mean mismatch:
a narrow Gaussian also fails this standard-normal comparison.

Second moments are eigenvalues of Z transpose Z / n, with no mean subtraction. Their
population standard-normal reference is 1; finite Gaussian samples do not have all
sample eigenvalues exactly 1. Top-two directions are fitted separately to each split,
so the two scatter clouds do not share a fitted coordinate basis. Random 2D bases are
also drawn separately by split. Radius-1/2/3 circles are geometric radii, not 68/95/99.7%
coverage regions in 2D. Off-diagonal correlations are centred Pearson correlations.

Temporal correlation pools frames and valid within-chunk timestep pairs per joint,
then averages correlations over joints. Lag is in future chunk steps, not episode time.
The coloured band is the minimum-to-maximum range across joints, not uncertainty.
The grey ±2/sqrt(n*T) band is only a rough iid scale; overlapping pairs and frames break
its assumptions. Joint standard deviations pool frames and timesteps; each heatmap cell
instead computes variation across frames at one step and joint. All standard deviations
use ddof=0. Heatmap scales are shared across splits and models.

Observed contraction and correlation describe the saved approximate inverse. They do
not establish the distribution of an exact inverse, a calibrated likelihood, overfitting,
or closed-loop performance. Nonzero round-trip error cannot be translated into a bound
on recovered-noise bias without additional stability or convergence evidence. No finer
integration run was made for this presentation. BC and Diverse-v3 differ in state encoding,
checkpoint step and potentially other settings; comparisons are not a training-only ablation.
"""
import json
from pathlib import Path
import sys
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.report_html import page, table

# One definition drives both the readable split table and the viewer's metrics.
METRICS=[
 ('n','Sampled frames',0,'Number of saved chunks; frames within episodes are dependent.',None),
 ('d','Valid dimensions',0,'30 chunk steps × 7 non-padding joints.',None),
 ('sq_norm_over_d_mean','Mean ‖z‖² / d',4,'Uncentred squared norm per dimension; standard-normal expectation 1.',1),
 ('sq_norm_over_d_std','SD of ‖z‖² / d',4,'Across-frame spread of normalised squared norms; not a standard error.',None),
 ('ks_median_model','Median projected KS',4,'Median CDF distance to N(0,1), across 256 random unit directions; includes scale mismatch.',None),
 ('ks_median_ref','Matched Gaussian KS',4,'Same sample size and dimension; finite-sample distance is not zero.',None),
 ('recon_rms_median','Median round-trip RMS',4,'RMS over 210 coordinates, then median over frames; normalised action units.',0),
 ('recon_rms_p95','95th percentile round-trip RMS',4,'Upper-tail numerical reconstruction error; normalised action units.',0),
 ('autocorr_lag1_mean','Lag-1 temporal correlation',4,'Pearson correlation pooled over frames / within-chunk pairs, averaged over joints.',0),
 ('mean_abs_offdiag_corr_model','Mean |off-diagonal correlation|',4,'Centred coordinate correlations, averaged over distinct pairs.',None),
 ('mean_abs_offdiag_corr_ref','Gaussian |off-diagonal correlation|',4,'Finite Gaussian sample reference for coordinate correlation.',None),
 ('frac_below_chi2_01','Fraction below Gaussian norm p01',4,'Fraction of frames with ‖z‖² below chi-square(d) first percentile.',.01),
 ('frac_above_chi2_99','Fraction above Gaussian norm p99',4,'Fraction of frames with ‖z‖² above chi-square(d) 99th percentile.',.01),
 ('kurt_median_model','Median projected excess kurtosis',4,'Per-projection centred shape diagnostic; Gaussian population value 0.',0),
 ('kurt_median_ref','Gaussian projected excess kurtosis',4,'Matched finite-sample shape reference.',0),
 ('skew_abs_median_model','Median |projected skewness|',4,'Absolute per-projection centred skewness; Gaussian population value 0.',0),
 ('skew_abs_median_ref','Gaussian |projected skewness|',4,'Matched finite-sample reference; absolute skew has positive sample bias.',0),
]
GAUSSIANITY='''Read left to right, top then bottom. Blue = validation, orange = training; grey = Gaussian references.
Norm histogram: density of ‖z‖²/d, dashed chi-square(210)/210; expectation 1. QQ: standard-normal quantile on x, pooled projected quantile on y; dashed y=x, solid grey matched validation Gaussian sample.
KS histogram: x is distance to N(0,1), y is number of the 256 projections; grey dashed is the matched validation sample.
Random 2D scatter: one point per frame, coordinates z·u₁ and z·u₂; bases drawn separately by split. Top-two scatter: separate uncentred second-moment bases per split. Circles in both plots have radii 1, 2, 3, not one-dimensional sigma coverage.
Spectrum: descending eigenvalue index versus eigenvalue of ZᵀZ/n (log y); grey dashed is a finite validation-sized Gaussian sample. Legend correlations use centred Pearson correlations, unlike this uncentred spectrum. Below-unit norm and flattened QQ slope show contraction; they alone do not distinguish a narrow Gaussian from a non-Gaussian shape.'''
STRUCTURE='''Read left to right, top then bottom. Blue = validation, orange = training.
Temporal plot: x is lag in future chunk steps; y is within-joint Pearson correlation. Solid = mean across joints, coloured fill = joint min–max, dashed grey = matched validation Gaussian, horizontal line = zero. Grey band ±2/√(nT) is a rough iid scale, not a calibrated confidence interval.
Joint-scale dots: joint on x and std pooled across frames and timesteps on y; dashed = unit Gaussian scale.
Round-trip histogram: x is per-frame RMS of forward(z) minus processed ground truth in normalised action coordinates; y is frame count. This is a numerical reconstruction diagnostic, not policy action error.
Bottom: validation and training std heatmaps (frames vary, step/joint fixed), then validation mean. x = chunk step, y = joint; blue encodes std on a shared 0–1.5 scale, red/blue mean on −0.6–0.6. Colourbar extensions mark values beyond the shown mean range. Nonzero correlations and joint-dependent scales describe the approximate inverse, with integration bias unresolved.'''


def write_primer(out, summary):
    """Teach the experiment before asking the reader to interpret its diagnostics."""
    from html import escape
    import math

    val = summary['val']
    train = summary.get('train')
    steps = summary['num_steps']
    norm = val['sq_norm_over_d_mean']
    rms_scale = math.sqrt(norm)
    body = '''<div class="eyebrow">Flow inversion · start here</div>
<h1>What noise would produce this demonstrated action?</h1>
<p>The policy normally turns random noise into an action. This probe starts with a recorded demonstration and runs that transformation backward. It asks whether the recovered starting points resemble the noise the policy normally receives.</p>
<style>
.flowpath{display:grid;grid-template-columns:1fr auto 1.3fr auto 1fr;gap:10px;align-items:center;margin:16px 0}
.flowbox{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;font-size:14px}.flowbox b{display:block;color:var(--accent);margin-bottom:5px}.arrow{font-size:24px;color:var(--accent)}
@media(max-width:650px){.flowpath{grid-template-columns:1fr}.arrow{text-align:center;transform:rotate(90deg)}}
</style>
<div class="flowpath" aria-label="Normal generation: Gaussian noise through the conditioned action expert produces an action chunk">
<div class="flowbox"><b>1. Starting noise z</b>30 future positions × 7 valid joints. Initially random numbers.</div><div class="arrow">→</div>
<div class="flowbox"><b>2. Learned transformation</b>The action expert updates those numbers, using the images, state and prompt.</div><div class="arrow">→</div>
<div class="flowbox"><b>3. Action chunk</b>A proposed sequence of robot actions in processed model coordinates.</div></div>
<p><b>The probe goes right to left.</b> At the right-hand end, substitute the operator's recorded action for a generated action. Keep that frame's observation and prompt fixed. Integrate backward to estimate z; then integrate forward again to check reconstruction.</p>
<p><b>Two different clocks:</b> flow time t=0…1 measures progress from noise to action. The 30 positions in the chunk are future robot timesteps. “Backward integration” reverses flow time; it does not rewind the video or execute robot actions backward.</p>
<h2>Why compare the recovered numbers with a Gaussian?</h2>
<p>Normal generation starts with independent standard-normal coordinates: mean 0, standard deviation 1, and no correlation between future positions or joints. If the model described the sampled demonstrations correctly and the inverse were exact, those demonstrations would map back to that starting distribution.</p>
<p>We invert many demonstrations, each with its own context, and inspect the resulting collection of z arrays. These are <b>estimated starting points</b>, not measured sensor noise, recorded training noise, or prediction errors. The Gaussian sample in the plots is a synthetic reference with the same number of frames and dimensions.</p>
<h2>Read this run in three passes</h2>'''
    body += f'''<div class="callout"><b>1. How large are the recovered starting points?</b><br>
Validation mean ‖z‖²/d is <b>{norm:.4f}</b>, versus the standard-normal reference 1.
That corresponds to a root-mean-square coordinate magnitude of <b>{rms_scale:.3f}</b> versus 1.
The recovered arrays therefore sit closer to zero overall. “30% of expected squared norm” means about 55% of reference RMS magnitude, not 30% of the standard deviation and not 30% action accuracy.
</div>'''
    body += f'''<p>Here, d={val['d']}: flatten the 30 × 7 chunk, square its coordinates, add them, and divide by 210. Then average over {val['n']} validation frames. No mean subtraction or rescaling is performed. “Contracted” describes this magnitude; it does not mean every joint or direction shrinks equally.</p>
<div class="callout"><b>2. Do their distributions and relationships look like independent unit Gaussians?</b><br>
Median projected KS is <b>{val['ks_median_model']:.4f}</b>, while the matched Gaussian reference is <b>{val['ks_median_ref']:.4f}</b>.
Lag-1 temporal correlation is <b>{val['autocorr_lag1_mean']:.4f}</b>, versus population reference 0.
</div>
<p>Imagine taking a randomly weighted sum of all 210 coordinates for every frame. Each such “projection” gives a one-dimensional histogram we can compare with N(0,1); the weights have unit length, so ideal Gaussian noise still has standard deviation 1 after projection.</p>
<p><b>QQ</b> lines match percentiles: a shallower-than-diagonal line indicates a narrower spread. <b>KS</b> is the biggest gap between cumulative fractions. A distance of 0.20 means a 20-percentage-point discrepancy at some threshold for that projection—not 20% incorrect actions or a 20% probability the model is wrong. The report takes the median distance over 256 directions.</p>
<p>A Gaussian with standard deviation 0.55 would already fail this comparison with a <i>unit</i> Gaussian. Thus the norm and KS findings are not independent proof of an unusual distributional shape. Skewness and excess kurtosis describe shape; the second-moment spectrum shows whether energy is concentrated in a few directions. Temporal correlation checks whether recovered values at neighbouring future positions move together, as independent starting noise would not.</p>
<div class="callout"><b>3. How faithfully did the numerical inverse reconstruct the demonstration?</b><br>
Validation median round-trip RMS is <b>{val['recon_rms_median']:.4f}</b>; its 95th percentile is <b>{val['recon_rms_p95']:.4f}</b>.
These are errors in processed action coordinates, not joint degrees or success rates.
</div>
<p>This run uses <b>{steps} explicit Euler steps in each direction, refinement={summary['refine']}</b>. At each step, the solver follows the local velocity in a straight line. On a curved trajectory, stepping backward and forward does not exactly retrace the same path. Model computation also has finite precision. The round trip measures their combined reconstruction discrepancy.</p>
<p>It does not tell us how far the recovered z is from an exact inverse: the transformation may amplify or attenuate errors. We have not measured convergence with smaller integration steps, so the amount of contraction caused by numerical inversion remains unresolved. A low round-trip error alone would not prove that the distribution of recovered z is correct.</p>'''
    if train:
        body += f'''<h2>How to read training versus validation</h2>
<p>For this checkpoint, mean ‖z‖²/d is {train['sq_norm_over_d_mean']:.4f} on training frames and {norm:.4f} on validation frames. Median round-trip RMS is {train['recon_rms_median']:.4f} and {val['recon_rms_median']:.4f}, respectively. These describe two sampled sets of demonstrations. Smaller noise norm is not automatically better fit, and a smaller round-trip error is not automatically better action prediction. The splits have different episode and frame sampling; this is not, on its own, an overfitting score.</p>'''
    body += '''<h2>Which figure should I open?</h2>
<p><b>Gaussianity:</b> start with squared norms and QQ for scale; use KS as a numerical standard-normal mismatch; then use the second-moment spectrum and projection scatters to inspect directionality.</p>
<p><b>Structure:</b> inspect correlation across future steps and scales across joints, then check the round-trip histogram before attributing differences to the learned model. The heatmaps locate which step/joint coordinates have a different mean or spread.</p>
<p><b>What this establishes:</b> the saved approximate inverse produces contracted, structured starting arrays. It does not establish why, give a likelihood for an action, or measure closed-loop robot performance. BC and Diverse-v3 also differ in state representation and checkpoint settings, so differences are not a controlled training-only effect.</p>
<p><a href="metrics.html">Exact metrics, sampling and mathematical definitions</a> · <a href="gaussianity.png">Gaussianity figure</a> · <a href="structure.png">Structure figure</a></p>'''
    body += '<p class="note">Checkpoint: '+escape(summary['checkpoint'])+'</p>'
    page(out/'overview.html', 'Understanding flow inversion', body)


def render(output_dir):
    out=Path(output_dir);s=json.loads((out/'summary.json').read_text());write_primer(out,s);splits=[n for n in ('val','train') if n in s]
    rows=[[label,*[f'{s[n][key]:.{digits}f}' for n in splits],note] for key,label,digits,note,_ in METRICS]
    body='<div class="eyebrow">Flow inversion · measured values</div><h1>Recovered noise is contracted</h1>'
    body+=f'<div class="callout">Validation mean ‖z‖²/d = {s["val"]["sq_norm_over_d_mean"]:.4f}, versus Gaussian expectation 1. Median round-trip RMS = {s["val"]["recon_rms_median"]:.4f}. These are measurements of a 20-step approximate inverse; its numerical bias has not been separated from model mismatch.</div>'
    body+='<div class="definitions">'+table(['Metric',*['Validation' if n=='val' else 'Training' for n in splits],'Definition / units'],rows)+'</div>'
    from lerobot.probes.utils import joint_names_for_dim
    joint_names = joint_names_for_dim(len(s[splits[0]]['per_joint_std']))
    body+='<h2>Per-joint recovered-noise scale</h2>'+table(['Valid joint',*splits],[[name,*[f'{s[n]["per_joint_std"][j]:.4f}' for n in splits]] for j,name in enumerate(joint_names)])
    import html
    body+='<h2>Data and interpretation</h2>'+''.join('<p>'+html.escape(p)+'</p>' for p in __doc__.split('\n\n'))
    body+='<p><a href="summary.json">Saved summary</a> · <a href="frames.csv">Frame-level metrics</a> · <a href="header.json">Capture provenance</a></p>'
    page(out/'metrics.html','Flow inversion measurements',body)
    metrics=[]
    for key,label,digits,note,baseline in METRICS:
        for n in splits:
            metrics.append(Metric(f'{n}.{key}',f'{n}: {label}',fmt=digits,note=note,baseline=baseline,primary=(n=='val' and key in ('sq_norm_over_d_mean','ks_median_model','recon_rms_median'))))
    provenance={}
    for n in splits:
        meta=json.loads((out/f'meta_{n}.json').read_text());eps=sorted({r['episode_idx'] for r in meta})
        provenance[n]={'n_frames':len(meta),'n_episodes':len(eps),'sources':[{'name':n,'root':s[f'{n}_root'],'n_frames':len(meta),'n_episodes':len(eps),'episodes':eps}]}
    provenance.update(details=[['Checkpoint',s['checkpoint']],['Integrator',f"{s['num_steps']} Euler steps; refinement={s['refine']}"],['Sampling','Validation 64 frames/episode; training 6; evenly spaced, stride 3, seed 42.'],['Units','30 × 7 valid coordinates; checkpoint-specific processed action space.']])
    panels=[Panel('overview.html','Start here: what flow inversion measures',primary=True,how='Follow normal generation, reverse a demonstration, then read scale, distribution and round-trip error in that order. This page explains the actual values for this checkpoint.'),Panel('metrics.html','Read the numerical results and interpretation',how='Validation and training are side by side. Values are verified against the saved summary; metric definitions and limitations accompany every row.'),Panel('gaussianity.png','Does recovered noise resemble N(0,I)?',how=GAUSSIANITY),Panel('structure.png','Where does the recovered noise have structure?',how=STRUCTURE)]
    if (out/'comparison.html').exists():panels.append(Panel('comparison.html','Compare BC and Diverse-v3',how='Both models and both splits, with numerical-integration and checkpoint-setting limitations.'))
    return write_index(str(out),sys.modules[__name__],title='Flow inversion',group='Actions',claim='Ground-truth chunks invert to contracted noise; integration error limits attribution.',summary=s,metrics=metrics,panels=panels,extra={'provenance':provenance},status='info')

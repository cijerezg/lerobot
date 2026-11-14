#!/usr/bin/env node
// Summarize saved measurements and refresh the browser without model evaluation.
const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
const root=process.argv[2];if(!root)throw new Error('Usage: node lerobot/scripts/probes/summarize_critic_gradients.cjs <critic_sensitivity_dir>');
const code=path.resolve(__dirname,'../../src/lerobot/probes');
const statsSource=fs.readFileSync(path.join(code,'critic_gradient_stats.js'),'utf8');
const context={};vm.createContext(context);vm.runInContext(statsSource,context);
const stats=context.GradientStats;
const data=JSON.parse(fs.readFileSync(path.join(root,'gradient_points.json'),'utf8'));
if(data.records.some(r=>!Number.isFinite(r.image_state_grad_norm)))throw new Error('Run critic_gradient_view --components-only to derive saved image/state gradients first');
const hasDepth=data.records.some(r=>r.raw_depth_consumed);
const scope=hasDepth?'observation':'images_state';
const gradientLabel=hasDepth?'RGB + state + depth':'RGB + state';
const gradient=r=>hasDepth?r.observation_grad_norm:r.image_state_grad_norm;
if(data.records.some(r=>!Number.isFinite(gradient(r))))throw new Error('Missing primary gradient norm');
const axes=['segment_progress','seconds_to_end','value','delta_value','advantage','abs_advantage'];
const actions=['all',...new Set(data.records.map(r=>r.phase))];
const val=(r,key)=>key==='abs_advantage'?(r.advantage==null?null:Math.abs(r.advantage)):r[key];
const reports=[];
for(const transition of ['all','interior']) {
 const available=data.records.filter(r=>transition==='all'||r.terminal===false);
 for(const axis of axes)for(const nBins of [10,20,40,80]) {
  // Shared pooled edges within each axis/regime/resolution keep action curves comparable.
  const points=available.map((r,index)=>({x:val(r,axis),y:gradient(r),episode:r.episode_ref,index}));
  const pooled=stats.summarize(points,axis,null,nBins);
  for(const action of actions) {
   const subset=action==='all'?points:points.filter(p=>available[p.index].phase===action);
   const result=action==='all'?pooled:stats.summarize(subset,axis,pooled.edges,nBins);
   reports.push({action,transition,axis,requested_bins:nBins,...result,bins:result.bins.map(({indices,...b})=>b)});
  }
 }
}
const out={gradient_scope:scope,gradient_label:gradientLabel,depth_frames:data.records.filter(r=>r.raw_depth_consumed).length,checkpoint:data.summary.checkpoint_step,frames:data.records.length,episodes:data.episodes.length,
 definition:'Gradient norm restricted to RGB image features, actual state-value embeddings (discrete or projected continuous), and real depth features when consumed, entering critic fusion. Fixed state wording/delimiters, task/subtask text, metadata and depth/history placeholders are excluded. The critic stays conditioned on the original prompt. Restricted norms are computed from saved disjoint token gradients. Norm scale depends on the representation and number of tokens. Each frame has equal weight in quantiles, means and Spearman. Episode-balanced statistic is median of within-bin episode medians. IQR is frame spread, not confidence. No significance or difficulty inference. No smoothing. All and within-segment transitions both reported. Shared pooled X edges across actions; identical X values never split across bins.',reports};
if(fs.existsSync(path.join(root,'architecture_comparison.html')))out.comparison_url='architecture_comparison.html';
out.durations={};
for(const action of actions){
 const segments=new Map();
 for(const r of data.records){
  if(action!=='all'&&r.phase!==action)continue;
  if(r.segment_progress==null||r.segment_progress>=1||!(r.seconds_to_end>0))continue;
  const duration=r.seconds_to_end/(1-r.segment_progress);
  segments.set(`${r.episode_ref}/${r.segment_end}`,duration);
 }
 const values=[...segments.values()].sort((a,b)=>a-b);
 out.durations[action]={n_segments:values.length,p10:stats.quantile(values,.1),median:stats.quantile(values,.5),p90:stats.quantile(values,.9)};
}
fs.writeFileSync(path.join(root,'gradient_binned_summary.json'),JSON.stringify(out));
const columns=['action','transition','axis','requested_bins','lo','hi','upper_inclusive','n_frames','n_episodes','x_median','median','q25','q75','mean','episode_median'];
const escape=v=>'"'+String(v??'').replaceAll('"','""')+'"';
const lines=[columns.map(escape).join(',')];
for(const report of reports)for(const bin of report.bins)lines.push(columns.map(k=>escape(bin[k]??report[k])).join(','));
fs.writeFileSync(path.join(root,'gradient_binned_summary.csv'),lines.join('\n'));
const fmt=n=>n==null?'—':n.toFixed(3);
const reportFor=(action,axis,transition='all',requested_bins=40)=>reports.find(r=>r.action===action&&r.axis===axis&&r.transition===transition&&r.requested_bins===requested_bins);
let md=`# ${gradientLabel} gradient numerical summaries — checkpoint ${data.summary.checkpoint_step}\n\n${data.records.length} measured frames, ${data.episodes.length} episodes.\n\n${out.definition}\n\n`;
md+='## Rank associations\n\nSpearman correlation between gradient norm and each quantity. These describe the recorded frames, not independent trials or causality. Raw numbers are independent of bin resolution.\n\n';
md+='| Action | Frames (progress) | Episodes | Progress | Seconds left | V | ΔV | Advantage | Absolute advantage |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n';
for(const a of actions){const p=reportFor(a,'segment_progress');md+=`| ${a} | ${p.n_frames} | ${p.n_episodes} | ${axes.map(k=>fmt(reportFor(a,k).spearman)).join(' | ')} |\n`;}
md+='\n## Advantage with terminal transitions excluded\n\n| Action | Paired frames | Episodes | Signed advantage ρ | Absolute advantage ρ |\n|---|---:|---:|---:|---:|\n';
for(const a of actions){const r=reportFor(a,'advantage','interior');md+=`| ${a} | ${r.n_frames} | ${r.n_episodes} | ${fmt(r.spearman)} | ${fmt(reportFor(a,'abs_advantage','interior').spearman)} |\n`;}
md+='\n## Progress-bin resolution check\n\nEach row compares the first and last progress bins at that resolution; their widths change, so ratios are not interchangeable effect estimates. Both tails and their support are retained at every resolution. Empty bins remain missing.\n\n| Action | Bins | First-bin frames / episodes | First median | Last-bin frames / episodes | Last median | Last / first | Episode-median ratio |\n|---|---:|---:|---:|---:|---:|---:|---:|\n';
for(const a of actions)for(const n of [10,20,40,80]){const r=reportFor(a,'segment_progress','all',n),f=r.bins[0],l=r.bins.at(-1);if(!f||!l)continue;md+=`| ${a} | ${n} | ${f.n_frames} / ${f.n_episodes} | ${fmt(f.median)} | ${l.n_frames} / ${l.n_episodes} | ${fmt(l.median)} | ${fmt(f.median&&l.median!=null?l.median/f.median:null)} | ${fmt(f.episode_median&&l.episode_median!=null?l.episode_median/f.episode_median:null)} |\n`;}
md+='\nThe browser defaults to 40 bins and allows 10/20/40/80. Progress bins have fixed width; other axes use pooled quantiles with tied edges merged. UI filters recalculate edges for the selected population; the saved report uses shared pooled edges across actions at each resolution. Frame count weighting can favour longer episodes; the episode-median curve gives each contributing episode equal weight within each bin, but neither controls for task/source composition. Advantage and ΔV exclude unmatched conditioning; use the plotted sample counts.\n\nFull numerical bins: [CSV](gradient_binned_summary.csv) · [JSON](gradient_binned_summary.json).\n';
fs.writeFileSync(path.join(root,'gradient_binned_summary.md'),md);
const template=fs.readFileSync(path.join(code,'critic_gradient_all.html'),'utf8');
const script=statsSource+'\n'+fs.readFileSync(path.join(code,'critic_gradient_all.js'),'utf8');
fs.writeFileSync(path.join(root,'gradient_explorer.html'),template.replace('__PROBE_SCRIPT__',()=>script).replace('__PROBE_DATA__',()=>JSON.stringify(data).replaceAll('<','\\u003c')));
const reportTemplate=fs.readFileSync(path.join(code,'critic_gradient_report.html'),'utf8');
fs.writeFileSync(path.join(root,'gradient_binned_summary.html'),reportTemplate.replace('__REPORT_DATA__',()=>JSON.stringify(out).replaceAll('<','\\u003c')));
const manifestFile=path.join(root,'index.json');
if(fs.existsSync(manifestFile)){
 const manifest=JSON.parse(fs.readFileSync(manifestFile,'utf8'));
 manifest.panels=manifest.panels.filter(p=>p.file!=='gradient_binned_summary.html');
 manifest.panels.unshift({file:'gradient_binned_summary.html',caption:gradientLabel+' gradients: numerical report',how:'Choose an action, bin resolution and transition regime. Inspect six curves together, expand exact bin tables, or click a point to open its saved frames.',kind:'html',primary:true,refs:[],align:''});
 fs.writeFileSync(manifestFile,JSON.stringify(manifest,null,2));
}
console.log(JSON.stringify({curves:reports.length,bin_rows:lines.length-1,frames:data.records.length}));
console.log(md.split('## Progress-bin resolution check')[0]);

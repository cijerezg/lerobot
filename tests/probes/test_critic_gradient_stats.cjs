const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const context={};vm.createContext(context);
vm.runInContext(fs.readFileSync(path.resolve(__dirname,'../../src/lerobot/probes/critic_gradient_stats.js'),'utf8'),context);
const S=context.GradientStats;
const p=(x,y,episode=0,index=0)=>({x,y,episode,index});
let result=S.summarize([p(0,1),p(.5,2),p(1,4)],'segment_progress',null,40);
assert.equal(result.bins.length,40);assert.equal(result.bins.reduce((n,b)=>n+b.n_frames,0),3);
assert.equal(result.bins[0].median,1);assert.equal(result.bins[20].median,2);assert.equal(result.bins[39].median,4);
assert.equal(result.bins[1].median,null);assert.equal(result.bins[1].n_episodes,0);
result=S.summarize([p(1,1),p(1,1),p(1,1),p(1,9,1)],'value');
assert.equal(result.bins.length,1);assert.equal(result.bins[0].median,1);assert.equal(result.bins[0].episode_median,5);
assert.equal(result.bins[0].mean,3);assert.equal(result.bins[0].n_episodes,2);assert.equal(result.spearman,null);
assert.equal(S.spearman([p(1,3),p(1,3),p(2,2),p(3,1)]),-1);
result=S.summarize([p(null,2),p(0,0),p(1,2),p(2,null)],'value');assert.equal(result.n_frames,2);
for(const n of [10,20,40,80]) {
 result=S.summarize(Array.from({length:201},(_,i)=>p(i%9,i,Math.floor(i/10),i)),'value',null,n);
 const ids=result.bins.flatMap(b=>b.indices);assert.equal(ids.length,201);assert.equal(new Set(ids).size,201);
 for(const b of result.bins)if(b.n_frames)assert.ok(b.q25<=b.median&&b.median<=b.q75);
}
console.log('Gradient stats: boundaries, empty bins, tied quantiles/ranks, episode weighting, missing values, and all resolutions passed.');

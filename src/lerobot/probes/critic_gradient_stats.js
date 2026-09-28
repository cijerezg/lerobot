/* Shared numerical summaries for the browser and saved-data report. */
(function(root) {
  'use strict';
  const finite = v => v != null && Number.isFinite(v);
  function quantile(sorted,q) {
    if (!sorted.length) return null;
    const p=(sorted.length-1)*q, i=Math.floor(p), f=p-i;
    return sorted[i]*(1-f)+sorted[Math.min(i+1,sorted.length-1)]*f;
  }
  const median = values => quantile([...values].sort((a,b)=>a-b),.5);
  function ranks(values) {
    const order=values.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v), result=new Array(values.length);
    for(let i=0;i<order.length;) {
      let j=i+1;while(j<order.length&&order[j].v===order[i].v)j++;
      for(let k=i;k<j;k++)result[order[k].i]=(i+j-1)/2;
      i=j;
    }
    return result;
  }
  function spearman(points) {
    if(points.length<3)return null;
    const x=ranks(points.map(p=>p.x)), y=ranks(points.map(p=>p.y)), mean=(points.length-1)/2;
    let xx=0,yy=0,xy=0;
    for(let i=0;i<x.length;i++){const a=x[i]-mean,b=y[i]-mean;xx+=a*a;yy+=b*b;xy+=a*b;}
    return xx&&yy?xy/Math.sqrt(xx*yy):null;
  }
  function edgesFor(points,key,nBins) {
    if(key==='segment_progress')return Array.from({length:nBins+1},(_,i)=>i/nBins);
    const xs=points.map(p=>p.x).sort((a,b)=>a-b);
    return [...new Set(Array.from({length:nBins+1},(_,i)=>quantile(xs,i/nBins)))];
  }
  function summarize(input,key,suppliedEdges=null,nBins=40) {
    const points=input.filter(p=>finite(p.x)&&finite(p.y));
    if(!points.length)return {n_frames:0,n_episodes:0,spearman:null,bins:[],edges:[],method:'no data'};
    let edges=suppliedEdges||edgesFor(points,key,nBins);
    if(edges.length===1)edges=[edges[0],edges[0]];
    const bins=edges.slice(0,-1).map((lo,i)=>({lo,hi:edges[i+1],upper_inclusive:i===edges.length-2,points:[]}));
    for(const p of points) {
      const bin=bins.find(b=>p.x>=b.lo&&(p.x<b.hi||(b.upper_inclusive&&p.x<=b.hi)));
      if(!bin)throw new Error('A point falls outside the summary bin edges');
      bin.points.push(p);
    }
    return {n_frames:points.length,n_episodes:new Set(points.map(p=>p.episode)).size,
      spearman:spearman(points),edges,method:key==='segment_progress'?`Fixed ${(100/nBins).toPrecision(3)}% progress bins`:'X-quantile bins; tied edges merged',
      bins:bins.map(b=>{
        const ys=b.points.map(p=>p.y).sort((a,b)=>a-b), episodes=new Map();
        for(const p of b.points){if(!episodes.has(p.episode))episodes.set(p.episode,[]);episodes.get(p.episode).push(p.y);}
        return {lo:b.lo,hi:b.hi,upper_inclusive:b.upper_inclusive,
          n_frames:ys.length,n_episodes:episodes.size,x_median:median(b.points.map(p=>p.x)),
          median:quantile(ys,.5),q25:quantile(ys,.25),q75:quantile(ys,.75),
          mean:ys.length?ys.reduce((a,b)=>a+b,0)/ys.length:null,
          episode_median:median([...episodes.values()].map(median)),
          indices:b.points.map(p=>p.index)};
      })};
  }
  root.GradientStats={quantile,median,spearman,summarize};
})(globalThis);

"""Interactive spread-vs-error scatter with the anchor's frame on hover (top + wrist camera), self-contained HTML.
Controls: y metric (mean / best / typical draw error), relative to motion or absolute, color by set / verb /
object family, set filter. Click pins a point. Thumbnails are decoded once and cached in thumbs.npz.

    uv run python lerobot/adhocs/draw_spread/followup_hover_page.py outputs/draw_spread/internal1200_2026-11-07
"""
import base64, io, json, os, re, sys
import numpy as np, pandas as pd
from PIL import Image

OUT = sys.argv[1]
run = json.load(open(f"{OUT}/run.json"))
roots = {os.path.basename(r): r for r, _ in run["train_episodes"]}
roots[os.path.basename(run["val_dataset_path"])] = run["val_dataset_path"]

a = pd.read_csv(f"{OUT}/anchors.csv")
a = a[a.setting == "clean"].copy()
a["root_base"] = a.root.map(os.path.basename)
e = pd.read_csv(f"{OUT}/spread_vs_error.csv")
e["root_base"] = e.episode.str.replace(r"^(train|val)_", "", regex=True).str.replace(r"_ep\d+$", "", regex=True)
e["ep"] = e.episode.str.extract(r"_ep(\d+)$").astype(int)
df = e.merge(a[["set", "root_base", "episode", "frame", "global_idx", "time_s", "bucket_level", "spread_ratio_total", "style_proj_offset", "style_proj_shape"]],
             left_on=["set", "root_base", "ep", "frame"], right_on=["set", "root_base", "episode", "frame"], suffixes=("", "_a"))
assert len(df) == len(e), (len(df), len(e))
floor = float(np.quantile(df.demo, 0.10))
den = np.maximum(df.demo, floor)
for k in ("spread", "bias", "best", "mean_draw_err"):
    df[f"{k}_rel"] = df[k] / den
df["verb"] = df.subtask.fillna("").str.split().str[0].where(lambda s: s.isin(["grasp", "move", "release"]), "other")


def family(s):
    s = s or ""
    if re.search(r"sock|shirt", s): return "clothes"
    if re.search(r"bottle|cup|pills", s): return "bottles/cup"
    if re.search(r"bit|clipper|flap|kit|headphone", s): return "bits/clipper/kit"
    return "other"


df["family"] = df.subtask.fillna("").map(family)

# thumbnails: top + wrist side by side, cached
cache = f"{OUT}/thumbs.json"
thumbs = json.load(open(cache)) if os.path.exists(cache) else {}
missing = [(r.set, r.root_base, int(r.global_idx)) for r in df.itertuples() if f"{r.set}|{r.root_base}|{int(r.global_idx)}" not in thumbs]
if missing:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    by_root = {}
    for s, rb, g in missing:
        by_root.setdefault(rb, []).append((s, g))
    for rb, items in by_root.items():
        ds = LeRobotDataset(repo_id="cijerezg/rebot-all-v1", root=roots[rb]); ds.delta_timestamps = None; ds.delta_indices = None
        for n, (s, g) in enumerate(items):
            fr = ds[g]
            ims = [v for k, v in fr.items() if k.startswith("observation.images.")]
            tiles = [Image.fromarray((v.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((200, 150)) for v in ims[:2]]
            canvas = Image.new("RGB", (200 * len(tiles), 150))
            for i, t in enumerate(tiles):
                canvas.paste(t, (200 * i, 0))
            buf = io.BytesIO(); canvas.save(buf, format="JPEG", quality=70)
            thumbs[f"{s}|{rb}|{g}"] = base64.b64encode(buf.getvalue()).decode()
            if n % 100 == 0:
                print(rb, n, "/", len(items), flush=True)
    json.dump(thumbs, open(cache, "w"))

points = []
for r in df.itertuples():
    points.append({"set": r.set, "ep": f"{r.root_base} ep{r.ep}", "frame": int(r.frame), "t": round(float(r.time_s), 1), "sub": r.subtask if isinstance(r.subtask, str) else "",
                   "verb": r.verb, "fam": r.family, "spread": r.spread, "bias": r.bias, "best": r.best, "typ": r.mean_draw_err, "demo": r.demo,
                   "spread_rel": r.spread_rel, "bias_rel": r.bias_rel, "best_rel": r.best_rel, "typ_rel": r.mean_draw_err_rel,
                   "ratio": r.spread_ratio_total ** 0.5, "style_off": r.style_proj_offset, "style_shape": r.style_proj_shape, "lvl": r.bucket_level,
                   "img": f"{r.set}|{r.root_base}|{int(r.global_idx)}"})

html = """<!doctype html><html><head><meta charset=utf-8><title>Draw spread hover</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#f9f9f7;color:#0b0b0b;font-size:13px}
#controls{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;align-items:center}
select{font-size:13px}#wrap{display:flex;gap:16px;align-items:flex-start}
svg{background:#fcfcfb;border:1px solid #e1e0d9}
#panel{width:420px;min-height:420px;background:#fcfcfb;border:1px solid #e1e0d9;padding:10px;font-variant-numeric:tabular-nums}
#panel img{width:400px;display:block;margin-bottom:8px;background:#ddd}
#panel table td{padding:1px 8px 1px 0;color:#52514e}#panel table td:first-child{color:#898781}
.legend span{display:inline-block;margin-right:12px}.dot{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:4px;vertical-align:middle}
.note{color:#898781}
</style></head><body>
<h3 style="margin:0 0 6px">Draw spread vs error, one point per anchor. Hover for the frame, click to pin.</h3>
<div id=controls>
 <label>y: <select id=ymet><option value=bias>error of the draw mean</option><option value=best>error of the best draw</option><option value=typ>error of a typical draw</option></select></label>
 <label>x: <select id=xmet><option value=spread>rms spread of the 64 draws</option><option value=demo>motion size (rms demo)</option><option value=ratio>spread / data std (reference)</option></select></label>
 <label><input type=checkbox id=rel checked> relative to motion (floored at p10 = FLOOR)</label>
 <label>color: <select id=col><option value=set>train / val</option><option value=verb>verb</option><option value=fam>object family</option></select></label>
 <label>show: <select id=show><option value=all>train + val</option><option value=val>val only</option><option value=train>train only</option></select></label>
 <span class=legend id=legend></span>
</div>
<div id=wrap><svg id=plot width=820 height=560></svg><div id=panel><span class=note>hover a point</span></div></div>
<p class=note>Dashed: y = x (only when both axes share units). Dotted: error = motion (holding still), relative mode only. Log axes.</p>
<script>
const P = POINTS;
const COL = {set:{train:"#2a78d6",val:"#eb6834"}, verb:{grasp:"#2a78d6",move:"#eb6834",release:"#1baf7a",other:"#898781"},
             fam:{"clothes":"#2a78d6","bottles/cup":"#eb6834","bits/clipper/kit":"#1baf7a","other":"#898781"}};
const svg=document.getElementById('plot'), panel=document.getElementById('panel');
const W=820,H=560,L=64,R=16,T=16,B=48; let pinned=null;
function val(p,key,rel){ if(key==='ratio') return p.ratio; if(key==='demo') return p.demo; return rel? p[key+'_rel'] : p[key]; }
function label(key,rel){ const n={bias:'error of the draw mean',best:'error of the best draw',typ:'error of a typical draw',spread:'rms spread of the 64 draws',demo:'motion size (rms demo)',ratio:'spread / data std'}[key]; return (rel&&key!=='ratio'&&key!=='demo')? n+' / motion' : n; }
function draw(){
  const ym=ymet.value, xm=xmet.value, rel=document.getElementById('rel').checked, c=col.value, sh=show.value;
  const pts=P.filter(p=>sh==='all'||p.set===sh);
  const xs=pts.map(p=>val(p,xm,rel)), ys=pts.map(p=>val(p,ym,rel));
  const lx=v=>Math.log10(v), xmin=lx(Math.min(...xs))-0.05, xmax=lx(Math.max(...xs))+0.05, ymin=lx(Math.min(...ys))-0.05, ymax=lx(Math.max(...ys))+0.05;
  const sx=v=>L+(lx(v)-xmin)/(xmax-xmin)*(W-L-R), sy=v=>H-B-(lx(v)-ymin)/(ymax-ymin)*(H-T-B);
  let s='';
  const ticks=(a,b)=>{const t=[];for(let e=Math.floor(a);e<=Math.ceil(b);e++)for(const m of [1,2,5]){const v=lx(m*Math.pow(10,e));if(v>=a&&v<=b)t.push(m*Math.pow(10,e));}return t;};
  for(const t of ticks(xmin,xmax)){s+=`<line x1=${sx(t)} x2=${sx(t)} y1=${T} y2=${H-B} stroke="#e1e0d9"/><text x=${sx(t)} y=${H-B+16} font-size=11 fill="#898781" text-anchor=middle>${t}</text>`;}
  for(const t of ticks(ymin,ymax)){s+=`<line y1=${sy(t)} y2=${sy(t)} x1=${L} x2=${W-R} stroke="#e1e0d9"/><text x=${L-6} y=${sy(t)+4} font-size=11 fill="#898781" text-anchor=end>${t}</text>`;}
  const sameUnits = (xm==='spread'||xm==='demo') && !(xm==='demo'&&rel);
  if(sameUnits){const lo=Math.pow(10,Math.max(xmin,ymin)),hi=Math.pow(10,Math.min(xmax,ymax)); s+=`<line x1=${sx(lo)} y1=${sy(lo)} x2=${sx(hi)} y2=${sy(hi)} stroke="#898781" stroke-dasharray="5 4"/>`;}
  if(rel && ymin<0 && ymax>0){s+=`<line x1=${L} x2=${W-R} y1=${sy(1)} y2=${sy(1)} stroke="#898781" stroke-dasharray="2 3"/>`;}
  s+=`<text x=${(L+W-R)/2} y=${H-10} font-size=12 text-anchor=middle>${label(xm,rel)}</text>`;
  s+=`<text transform="translate(14 ${(T+H-B)/2}) rotate(-90)" font-size=12 text-anchor=middle>${label(ym,rel)}</text>`;
  pts.forEach((p,i)=>{s+=`<circle data-i=${P.indexOf(p)} cx=${sx(val(p,xm,rel)).toFixed(1)} cy=${sy(val(p,ym,rel)).toFixed(1)} r=3.2 fill="${COL[c][p[c]]}" fill-opacity=0.6 stroke="${pinned===p?'#0b0b0b':'none'}" stroke-width="1.5"/>`;});
  svg.innerHTML=s;
  legend.innerHTML=Object.entries(COL[c]).map(([k,v])=>`<span><i class=dot style="background:${v}"></i>${k}</span>`).join('');
  svg.querySelectorAll('circle').forEach(el=>{el.onmouseenter=()=>{if(!pinned)showP(P[el.dataset.i]);}; el.onclick=()=>{pinned=(pinned===P[el.dataset.i])?null:P[el.dataset.i]; showP(P[el.dataset.i]); draw();};});
}
function f(v){return (v===null||v===undefined||Number.isNaN(v))?'-':(Math.abs(v)>=10?v.toFixed(0):v.toFixed(3));}
function showP(p){
  panel.innerHTML=`<img src="data:image/jpeg;base64,${THUMBS[p.img]}"><b>${p.set}</b> · ${p.ep} · frame ${p.frame} · t = ${p.t} s<br><b>${p.sub||'(no subtask)'}</b>
  <table><tr><td>rms spread</td><td>${f(p.spread)}</td><td>/ motion</td><td>${f(p.spread_rel)}</td></tr>
  <tr><td>error of the mean</td><td>${f(p.bias)}</td><td>/ motion</td><td>${f(p.bias_rel)}</td></tr>
  <tr><td>best draw</td><td>${f(p.best)}</td><td>/ motion</td><td>${f(p.best_rel)}</td></tr>
  <tr><td>typical draw</td><td>${f(p.typ)}</td><td>/ motion</td><td>${f(p.typ_rel)}</td></tr>
  <tr><td>motion (rms demo)</td><td>${f(p.demo)}</td><td>spread / data std</td><td>${f(p.ratio)}</td></tr>
  <tr><td>style proj offset</td><td>${f(p.style_off)}</td><td>shape</td><td>${f(p.style_shape)}</td></tr>
  <tr><td>reference level</td><td colspan=3>${p.lvl}</td></tr></table>${pinned===p?'<span class=note>pinned, click again to release</span>':''}`;
}
for(const id of ['ymet','xmet','rel','col','show']) document.getElementById(id).onchange=draw;
draw();
</script></body></html>"""
html = html.replace("POINTS", json.dumps(points)).replace("THUMBS", json.dumps(thumbs)).replace("FLOOR", f"{floor:.3f}")
open(f"{OUT}/hover.html", "w").write(html)
print(f"wrote {OUT}/hover.html  ({len(points)} points, {os.path.getsize(f'{OUT}/hover.html') / 1e6:.1f} MB)")

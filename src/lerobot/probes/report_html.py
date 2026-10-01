"""Small, offline HTML report helpers shared by saved-data probe reports."""
import html
import json
import math
from pathlib import Path

STYLE = '''
:root{color-scheme:dark;--bg:#111820;--card:#19232e;--fg:#e8eef5;--muted:#a9b9c9;--line:#344555;--accent:#8acbff}
*{box-sizing:border-box}body{margin:0;padding:24px;font:16px/1.55 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:25px;margin:0 0 8px}h2{font-size:20px;margin:24px 0 8px}h3{font-size:17px}p{max-width:110ch;margin:8px 0 16px}
a{color:var(--accent)}.eyebrow{color:var(--accent);font-size:12px;letter-spacing:.12em;text-transform:uppercase}
.note{color:var(--muted)}.callout{border-left:3px solid var(--accent);padding:12px 18px;background:var(--card);margin:16px 0}
.controls{display:flex;flex-wrap:wrap;gap:12px;padding:14px;background:var(--card);border-radius:8px}
label{font-size:13px;color:var(--muted)}select{display:block;font:15px system-ui;padding:8px;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:5px;max-width:100%}
.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line)}th{background:var(--card);color:var(--muted)}th:first-child,td:first-child{text-align:left}.definitions th:last-child,.definitions td:last-child{text-align:left;max-width:70ch}td small{color:var(--muted)}
.chart{overflow:auto;background:var(--card);border-radius:8px;margin:12px 0}.chart svg{width:100%;min-width:760px;display:block}.legend{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:14px;padding:6px 16px 14px}.swatch{display:inline-block;width:24px;border-top:3px solid;vertical-align:middle;margin-right:7px}
.pivots{display:grid;grid-template-columns:repeat(6,minmax(140px,1fr));gap:12px;overflow:auto}.pivots img{width:100%;border-radius:6px}.pivots figure{margin:0;font-size:13px}.pivots figcaption{padding:6px 0}details{margin:18px 0}summary{cursor:pointer;font-weight:600}code{font-size:.9em}button{padding:8px;background:var(--card);color:var(--fg);border:1px solid var(--line);cursor:pointer}
'''

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def data_json(value):
    return json.dumps(clean(value), separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')


def page(path, title, body, script=''):
    Path(path).write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>{STYLE}</style><body>{body}'
        f'<script>{script}</script></body></html>')


def table(headers, rows):
    def cell(v):
        return html.escape(str(v))
    return '<div class="tablewrap"><table><thead><tr>'+''.join(f'<th>{cell(v)}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{cell(v)}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

# One full-width chart, with fixed readable typography, explicit axes and exact hover values.
CHART_JS = r'''
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const palette=['#79bcff','#ffa95e','#64d2b1','#dc9cff','#ff8297'];
const fmt = (v,n=4) => v===null || v===undefined ? '—' : Number(v).toLocaleString('en-US',{maximumFractionDigits:n});
function table(headers,rows){return '<div class="tablewrap"><table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(c=>'<td>'+esc(c)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}
function chart(series,{title,xlabel,ylabel,xmax,ymax=1,ymin=0,log=false}){
 const W=Math.max(760,document.getElementById('plot').clientWidth),H=360,L=92,R=30,T=48,B=65, iw=W-L-R,ih=H-T-B;
 const X=x=>L+x/xmax*iw, Y=y=>T+ih*(1-(log?(Math.log10(Math.max(y,1e-12))-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin)):(y-ymin)/(ymax-ymin)));
 let s=`<svg role="img" aria-label="${esc(title)}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg"><title>${esc(title)}</title><defs><clipPath id="plotclip"><rect x="${L}" y="${T}" width="${iw}" height="${ih}"/></clipPath></defs><g font-family="system-ui" font-size="15" fill="#b7c8d8"><text x="${L}" y="26" fill="#e8eef5" font-size="19">${esc(title)}</text>`;
 const ticks=log?Array.from({length:Math.floor(Math.log10(ymax))-Math.ceil(Math.log10(ymin))+1},(_,i)=>10**(Math.ceil(Math.log10(ymin))+i)):Array.from({length:6},(_,i)=>ymin+(ymax-ymin)*i/5);
 for(const v of ticks){let y=Y(v);s+=`<path d="M${L} ${y}H${W-R}" stroke="#344555"/><text x="${L-12}" y="${y+5}" text-anchor="end">${log?v.toExponential(0):fmt(v,3)}</text>`;}
 for(let i=0;i<=5;i++){let xv=xmax*i/5,x=X(xv);s+=`<text x="${x}" y="${H-B+26}" text-anchor="middle">${fmt(xv,0)}</text>`}
 s+=`<text x="${L+iw/2}" y="${H-14}" text-anchor="middle">${esc(xlabel)}</text><text transform="translate(23 ${T+ih/2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text></g><g clip-path="url(#plotclip)">`;
 for(const t of series){let pts=t.y.map((y,i)=>y===null?null:[X(t.x?t.x[i]:i),Y(y),t.x?t.x[i]:i,y]).filter(Boolean);s+=`<path fill="none" stroke="${t.color}" stroke-width="${t.width||2.4}" ${t.dash?'stroke-dasharray="'+t.dash+'"':''} d="${pts.map((p,i)=>(i?'L':'M')+p[0]+','+p[1]).join(' ')}"/>`;s+=pts.map(p=>`<circle cx="${p[0]}" cy="${p[1]}" r="5" fill="transparent"><title>${esc(t.name)} · x=${p[2]} · y=${fmt(p[3],7)}</title></circle>`).join('')}
 s+='</g></svg><div class="legend">'+series.map(t=>`<span><i class="swatch" style="border-color:${t.color};border-top-style:${t.dash?'dashed':'solid'}"></i>${esc(t.name)}</span>`).join('')+'</div>';return s;
}
'''

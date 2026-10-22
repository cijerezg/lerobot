"""Two probe pages side by side, each pane with a dropdown to pick which one it shows (e.g. two checkpoints of the
same matrix). Pages keep loading their own panels and thumbs; paths are relative to the output file.

    uv run python lerobot/adhocs/intervention_episode_reps/compare_page.py <out.html> <label>=<trajectories.html> [<label>=<path> ...]
"""

import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
pages = [(a.split("=", 1)[0], os.path.relpath(a.split("=", 1)[1], out.parent)) for a in sys.argv[2:]]
out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{out.stem}</title>
<style>body{{margin:0;font-family:sans-serif;background:#111;color:#eee}}
.row{{display:flex;height:100vh}}.pane{{flex:1;display:flex;flex-direction:column;min-width:0;border-right:2px solid #444}}
nav{{padding:6px 10px;background:#222}}select{{font-size:14px;padding:4px 8px;background:#333;color:#eee;border:1px solid #555}}
iframe{{flex:1;width:100%;border:0;background:#fff}}</style></head><body><div class="row">
<div class="pane"><nav>left <select id="s0"></select></nav><iframe id="f0"></iframe></div>
<div class="pane"><nav>right <select id="s1"></select></nav><iframe id="f1"></iframe></div></div>
<script>
const P={json.dumps(pages)};
const pick=(location.hash.slice(1).split(",").map(Number));
[0,1].forEach(k=>{{const s=document.getElementById('s'+k),f=document.getElementById('f'+k);
P.forEach(([l,p],i)=>{{const o=document.createElement('option');o.value=i;o.textContent=l;s.append(o);}});
s.value=isNaN(pick[k])?(k?Math.min(1,P.length-1):0):pick[k];
const load=()=>{{f.src=P[s.value][1];location.hash=[document.getElementById('s0').value,document.getElementById('s1').value].join(',');}};
s.onchange=load;load();}});
</script></body></html>""")
print(out, [l for l, _ in pages])

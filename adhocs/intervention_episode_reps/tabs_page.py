"""One html page with a tab per probe page (e.g. one per checkpoint), each shown in an iframe. The pages keep loading
their own panels and thumbs from their own folders; paths are written relative to the output file.

    uv run python lerobot/adhocs/intervention_episode_reps/tabs_page.py <out.html> <label>=<trajectories.html> [<label>=<path> ...]
"""

import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
tabs = [a.split("=", 1) for a in sys.argv[2:]]
buttons = "".join(f'<button onclick="show({i})" id="b{i}">{label}</button>' for i, (label, _) in enumerate(tabs))
frames = "".join(f'<iframe id="f{i}" data-src="{os.path.relpath(path, out.parent)}"></iframe>' for i, (_, path) in enumerate(tabs))
out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{out.stem}</title>
<style>body{{margin:0;font-family:sans-serif;background:#111;color:#eee}}nav{{padding:6px 10px;background:#222}}
button{{font-size:14px;padding:6px 14px;margin-right:6px;background:#333;color:#eee;border:1px solid #555;border-radius:4px;cursor:pointer}}
button.on{{background:#4a7;color:#000;border-color:#4a7}}iframe{{display:none;width:100vw;height:calc(100vh - 44px);border:0;background:#fff}}
iframe.on{{display:block}}</style></head><body><nav>{buttons}</nav>{frames}
<script>
function show(k){{document.querySelectorAll('iframe').forEach((f,i)=>{{f.classList.toggle('on',i===k);if(i===k&&!f.src)f.src=f.dataset.src;}});
document.querySelectorAll('button').forEach((b,i)=>b.classList.toggle('on',i===k));location.hash=k;}}
show(parseInt(location.hash.slice(1))||0);
</script></body></html>""")
print(out, [l for l, _ in tabs])

"""Set the xy of free bodies in a layout (z and orientation kept): scene_nail.xml + scene_nail_photo.xml + objects.json.
usage: uv run python scripts/set_body_pos.py LAYOUT_DIR body x y [body x y ...]"""
import json, pathlib, re, sys
L = pathlib.Path(sys.argv[1]); args = sys.argv[2:]; moves = {args[i]: (float(args[i + 1]), float(args[i + 2])) for i in range(0, len(args), 3)}
for xml in ("scene_nail.xml", "scene_nail_photo.xml"):
    p = L / xml; s = p.read_text()
    for n, (x, y) in moves.items():
        pat = re.compile(rf'(<body name="{n}" pos=")([^"]*)(")'); m = pat.search(s); assert m, (xml, n); z = m.group(2).split()[2]
        s = pat.sub(lambda m: f'{m.group(1)}{x:.4f} {y:.4f} {z}{m.group(3)}', s, count=1)
    p.write_text(s)
oj = L / "objects.json"; o = json.loads(oj.read_text())
for n, (x, y) in moves.items():
    if n in o.get("objects", {}) and "xy" in o["objects"][n]: o["objects"][n]["xy"] = [x, y]
oj.write_text(json.dumps(o, indent=1)); print(f"{L.name}: {moves}")

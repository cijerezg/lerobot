"""Remove free bodies from a layout: the <body> blocks in scene_nail.xml and scene_nail_photo.xml (backups *.pre_remove.xml,
once) and the objects.json entries. usage: uv run python scripts/remove_bodies.py LAYOUT_DIR body [body ...]"""
import json, pathlib, re, sys
L = pathlib.Path(sys.argv[1]); names = sys.argv[2:]
for xml in ("scene_nail.xml", "scene_nail_photo.xml"):
    p = L / xml; s = p.read_text(); b = L / xml.replace(".xml", ".pre_remove.xml")
    if not b.exists(): b.write_text(s)
    for n in names:
        s, k = re.subn(rf'[ \t]*<body name="{n}"[^>]*>.*?</body>\n', "", s, count=1, flags=re.S); assert k == 1, (xml, n)
    p.write_text(s)
oj = L / "objects.json"; o = json.loads(oj.read_text())
for n in names: o.get("objects", {}).pop(n, None)
oj.write_text(json.dumps(o, indent=1)); print(f"{L.name}: removed {names}")

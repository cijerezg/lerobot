"""Objects of the object-choice scenes that the twin desk does not have: the YCB banana and the downloaded mug.

    uv run python -I lerobot/scripts/sim_builders/object_choice/build_assets.py

Writes scenes/assets/choice/<name>/ in the package (visual obj, texture, CoACD hulls, <name>.json with the scene lines
build_choice_envs.py adds to the twin text). Conventions as scripts/sim_builders/digital_twin/scripts/dl_build_assets.py: up = +z,
resting on z = 0, origin = footprint centre, visual geom group 2 without contacts, hulls group 3 with masses by hull volume.
banana: assets/sim_sources/choice_objects/downloads/ycb_011_banana (YCB object 011, google_16k scan, metres, z up), read as data.
mug: assets/sim_sources/digital_twin/assets_dl/mug (already converted, plain white), copied.
"""
import json
import re
import shutil
from pathlib import Path

import coacd
import numpy as np
import trimesh
from PIL import Image

WS = Path(__file__).resolve().parents[4]
SOURCES = WS / "assets/sim_sources"
OUT = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/assets/choice"
PALETTE = ["0.85 0.35 0.30", "0.30 0.65 0.85", "0.40 0.75 0.40", "0.90 0.70 0.25", "0.70 0.45 0.80", "0.35 0.75 0.75", "0.85 0.55 0.55", "0.55 0.55 0.85"]


def scene_lines(name, visual, hull_masses, friction):
    """The asset lines, default classes and body block of one object, in the twin scene's format (assetdir = scenes/)."""
    folder = f"assets/choice/{name}"
    assets = [f'    <mesh name="{name}_visual" file="{folder}/{name}_visual.obj" />']
    assets += [f'    <mesh name="{name}_hull{i:02d}" file="{folder}/{name}_hull{i:02d}.obj" />' for i in range(len(hull_masses))]
    if visual.startswith("material"):
        assets = [f'    <texture name="{name}_tex" type="2d" file="{folder}/{name}_tex.png" />',
                  f'    <material name="{name}_mat" texture="{name}_tex" specular="0.2" shininess="0.2" />'] + assets
    default = (f'    <default class="{name}_col">\n      <geom type="mesh" group="3" condim="4" friction="{friction} 0.005 0.0001" />\n    </default>\n'
               f'    <default class="{name}_vis">\n      <geom type="mesh" group="2" contype="0" conaffinity="0" mass="0" />\n    </default>\n')
    body = f'    <body name="{name}" pos="0 0 0.0020" quat="1 0 0 0">\n      <freejoint />\n      <geom class="{name}_vis" mesh="{name}_visual" {visual} />\n'
    body += "".join(f'      <geom class="{name}_col" mesh="{name}_hull{i:02d}" mass="{mass:.5f}" rgba="{PALETTE[i % len(PALETTE)]} 1" />\n' for i, mass in enumerate(hull_masses))
    return dict(asset_lines=assets, default=default, body=body + "    </body>\n")


def banana():
    src = SOURCES / "choice_objects/downloads/ycb_011_banana/011_banana/google_16k"
    out = OUT / "banana"; out.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(str(src / "textured.obj"), process=False, force="mesh")
    V, F, UV = np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.visual.uv)
    V = V - np.r_[(V[:, :2].min(0) + V[:, :2].max(0)) / 2, V[:, 2].min()]
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V] + [f"vt {u:.6f} {v:.6f}" for u, v in UV]
    lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in F]
    (out / "banana_visual.obj").write_text("\n".join(lines) + "\n")
    Image.open(src / "texture_map.png").convert("RGB").resize((1024, 1024), Image.LANCZOS).save(out / "banana_tex.png")
    welded = trimesh.Trimesh(V, F, process=True)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    parts = coacd.run_coacd(coacd.Mesh(welded.vertices, welded.faces), threshold=0.04, max_convex_hull=12, preprocess_mode="auto", seed=0)
    for old in out.glob("banana_hull*.obj"):
        old.unlink()
    volumes = []
    for i, (hv, hf) in enumerate(parts):
        hull = trimesh.Trimesh(hv, hf, process=False).convex_hull
        hull.export(out / f"banana_hull{i:02d}.obj"); volumes.append(float(hull.volume))
    mass = 0.07  # a real banana of this size is 0.1-0.15 kg; the YCB plastic one 0.066 kg
    meta = dict(name="banana", source="YCB object 011_banana, google_16k scan", url="http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/011_banana_google_16k.tgz",
                licence="CC-BY 4.0 (YCB Object and Model Set)", size_m=np.ptp(V, axis=0).round(4).tolist(), mass_kg=mass, hulls=len(volumes),
                **scene_lines("banana", 'material="banana_mat"', [mass * v / sum(volumes) for v in volumes], friction=0.8))
    (out / "banana.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"banana: size mm {(np.ptp(V, axis=0) * 1e3).round(1)}, {len(F)} triangles, {len(volumes)} hulls")


def mug():
    src = SOURCES / "digital_twin/assets_dl/mug"
    out = OUT / "mug"; out.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("mug_*.obj")):
        shutil.copyfile(f, out / f.name)
    xml = (src / "mug.xml").read_text()
    masses = [float(m) for m in re.findall(r'class="mug_col" mesh="mug_hull\d+" mass="([\d.]+)"', xml)]
    assert len(masses) == len(list(out.glob("mug_hull*.obj")))
    source = json.loads((src / "mug_meta.json").read_text())
    meta = dict(name="mug", source=source["source"], url=source["url"], licence=source["licence"], author=source["author"], size_m=source["size_m"],
                mass_kg=source["mass_kg"], hulls=len(masses), **scene_lines("mug", 'rgba="0.92 0.92 0.90 1"', masses, friction=0.8))
    (out / "mug.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"mug: size mm {(np.array(source['size_m']) * 1e3).round(1)}, {len(masses)} hulls")


if __name__ == "__main__":
    banana()
    mug()

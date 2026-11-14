"""Downloaded glTF/GLB models -> MuJoCo rigid assets at real size (assets_dl/<name>/).
Per object: <name>_visual.obj (+ <name>_tex.png + .mtl when the source has a texture), CoACD hulls, <name>.xml, <name>_meta.json.
Conventions follow lerobot/scripts/sim_builders/real2sim/scripts/build_rigid_asset.py: up = +z, resting on z = 0, origin = footprint centre,
visual geom group 2 contype 0 conaffinity 0 mass 0, hulls group 3 with masses proportional to hull volume summing to the nominal mass.
Reads the downloads only as data (trimesh). usage: python -I dl_build_assets.py [names...]"""
import json, pathlib, sys, numpy as np, trimesh, coacd, open3d as o3d
from PIL import Image

ROOT = pathlib.Path("/home/user/Documents/Research/RL/LeRobot/assets/sim_sources/digital_twin")
DL = ROOT / "downloads"; OUT = ROOT / "assets_dl"
R_YUP_TO_ZUP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])  # glTF Y-up -> MuJoCo Z-up
PALETTE = ["0.85 0.35 0.30", "0.30 0.65 0.85", "0.40 0.75 0.40", "0.90 0.70 0.25", "0.70 0.45 0.80", "0.35 0.75 0.75", "0.85 0.55 0.55", "0.55 0.55 0.85"]

# name: source file, geometry node filter (None = all), target size in source axes (x, y_up, z) or a scale rule, mass, hull params, colour
SPECS = {
    "pill_bottle": dict(file="01_pill_bottle/objaverse/104965efa64847dab5014c78eeee9d7b/104965efa64847dab5014c78eeee9d7b.glb",
        keep=lambda n: n.startswith(("Bottle", "Cap")), drop_note="7 loose pill meshes (Pill*) deleted", scale=("height", 0.100),
        mass=0.12, hulls=16, threshold=0.04, rgba=None, source="Sketchfab via Objaverse, uid 104965efa64847dab5014c78eeee9d7b",
        url="https://sketchfab.com/3d-models/104965efa64847dab5014c78eeee9d7b", licence="CC-BY 4.0", author="darthobsidian", thumb="104965efa64847dab5014c78eeee9d7b"),
    "pump_bottle": dict(file="02_pump_bottle/objaverse/36f768d11a6342b7b36a121e014f521e/36f768d11a6342b7b36a121e014f521e.glb",
        keep=lambda n: n in ("Object_11", "Object_9", "Object_6"), drop_note="ground plane (Object_4) and label decal (Object_7) deleted; recoloured black (no texture kept)",
        scale=("height", 0.180), mass=0.15, hulls=16, threshold=0.04, rgba="0.06 0.06 0.06 1", source="Sketchfab via Objaverse, uid 36f768d11a6342b7b36a121e014f521e",
        url="https://sketchfab.com/3d-models/36f768d11a6342b7b36a121e014f521e", licence="CC-BY 4.0", author="chohsunlu", thumb="36f768d11a6342b7b36a121e014f521e"),
    "mug": dict(file="05_mug/objaverse/19c8fe5702b544d0a1409d3dac1cf90e/19c8fe5702b544d0a1409d3dac1cf90e.glb", keep=None, drop_note="",
        scale=("geomean", (0.085, 0.095)), mass=0.30, hulls=32, threshold=0.03, rgba="0.92 0.92 0.90 1", source="Sketchfab via Objaverse, uid 19c8fe5702b544d0a1409d3dac1cf90e",
        url="https://sketchfab.com/3d-models/19c8fe5702b544d0a1409d3dac1cf90e", licence="CC-BY 4.0", author="edwardlewis450", thumb="19c8fe5702b544d0a1409d3dac1cf90e"),
    "tape_roll": dict(file="06_tape_roll/polyhaven_medical_tape/medical_tape_1k.gltf", keep=None, drop_note="", scale=("factor", 1.0),
        mass=0.03, hulls=32, threshold=0.02, rgba=None, source="Poly Haven medical_tape (1k glTF)", url="https://polyhaven.com/a/medical_tape",
        licence="CC0", author="Poly Haven", thumb=None),
    "power_strip": dict(file="07_power_strip/objaverse/78519c71cc1b4a1db9f64a79f4c9cdc2/78519c71cc1b4a1db9f64a79f4c9cdc2.glb",
        keep=lambda n: n.startswith("C_powerStripBody"), drop_note="cable (C_powerStripCable_low) and plug (C_powerStripPlug_low) meshes dropped; the 12-face blinn1 base box is kept as a second untextured visual geom",
        scale=("length_x", 0.300), mass=0.35, hulls=24, threshold=0.04, rgba=None, source="Sketchfab via Objaverse, uid 78519c71cc1b4a1db9f64a79f4c9cdc2",
        url="https://sketchfab.com/3d-models/78519c71cc1b4a1db9f64a79f4c9cdc2", licence="CC-BY 4.0", author="faceTheEdge", thumb="78519c71cc1b4a1db9f64a79f4c9cdc2"),
    "glasses_case": dict(file="03_headphone_case/objaverse/106573d6efd94e65860699eca5f9a6c7/106573d6efd94e65860699eca5f9a6c7.glb", keep=None, drop_note="source is an OPEN clamshell (lid open 135 deg); the lid half (z < -35.35 mm in source coords) was rotated about the hinge line onto the base to close it",
        scale=("xyz", (0.250, 0.100, 0.190)), fix="close_lid", mass=0.25, hulls=16, threshold=0.04, rgba="0.08 0.08 0.08 1", source="Poly by Google via Sketchfab/Objaverse, uid 106573d6efd94e65860699eca5f9a6c7",
        url="https://sketchfab.com/3d-models/106573d6efd94e65860699eca5f9a6c7", licence="CC-BY 4.0", author="ie-niels", thumb="106573d6efd94e65860699eca5f9a6c7"),
}


def load_parts(spec):
    """-> list of (name, Trimesh in world coords, texture image or None, baseColorFactor or None)."""
    s = trimesh.load(str(DL / spec["file"]), force="scene"); parts = []
    for node in s.graph.nodes_geometry:
        T, gname = s.graph[node]; g = s.geometry[gname]
        if spec["keep"] is not None and not spec["keep"](node): continue
        m = g.copy(); m.apply_transform(T); mat = getattr(g.visual, "material", None); img = None; fac = None
        if mat is not None:
            img = getattr(mat, "baseColorTexture", None) if hasattr(mat, "baseColorTexture") else getattr(mat, "image", None)
            fac = getattr(mat, "baseColorFactor", None) if hasattr(mat, "baseColorFactor") else None
        parts.append((node, m, img, fac))
    return parts


def merge(parts):
    V, F, UV, off = [], [], [], 0
    for _, m, _, _ in parts:
        V.append(m.vertices); F.append(m.faces + off); off += len(m.vertices)
        uv = m.visual.uv if hasattr(m.visual, "uv") and m.visual.uv is not None else np.zeros((len(m.vertices), 2))
        UV.append(uv)
    return np.vstack(V), np.vstack(F), np.vstack(UV)


def close_lid(V):
    """Glasses case GLB (Y-up, mm): base rim at y = -0.5, lid behind it (z < -35.35) open 135 deg. Rotate the lid about the hinge line
    (parallel to x, through y = -0.5, z = -35.5) so its rim lies in the base rim plane and the dome points up."""
    V = V * 1000.0; lid = V[:, 2] < -35.35
    u = np.array([49.1 - (-0.9), -85.2 - (-35.5)]); u /= np.linalg.norm(u)  # rim strip direction, hinge end (-0.9, -35.5) -> far end (49.1, -85.2)
    a = np.pi / 2 - np.arctan2(u[1], u[0]); c, s_ = np.cos(a), np.sin(a); piv = np.array([-0.5, -35.5])
    yz = V[lid][:, 1:] - piv; V = V.copy(); V[lid, 1] = c * yz[:, 0] - s_ * yz[:, 1] + piv[0]; V[lid, 2] = s_ * yz[:, 0] + c * yz[:, 1] + piv[1]
    print("close_lid: rotated %.1f deg; lid now y %.1f..%.1f z %.1f..%.1f (mm)" % (np.degrees(a), V[lid][:, 1].min(), V[lid][:, 1].max(), V[lid][:, 2].min(), V[lid][:, 2].max()))
    return V / 1000.0


def scale_factor(rule, ext_src):  # ext_src in source axes (x, y_up, z)
    kind, arg = rule
    if kind == "height": return np.full(3, arg / ext_src[1])
    if kind == "length_x": return np.full(3, arg / ext_src[0])
    if kind == "factor": return np.full(3, arg)
    if kind == "geomean": d, h = arg; return np.full(3, np.sqrt((d / ext_src[0]) * (h / ext_src[1])))
    if kind == "xyz": return np.array(arg) / ext_src
    raise ValueError(kind)


def write_obj(path, V, F, UV=None, mtl=None, mtlname=None):
    L = []
    if mtl is not None: L += [f"mtllib {mtl}", f"usemtl {mtlname}"]
    L += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    if UV is not None:
        L += [f"vt {u:.6f} {v:.6f}" for u, v in UV]
        L += [f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}" for a, b, c in F]
    else:
        L += [f"f {a+1} {b+1} {c+1}" for a, b, c in F]
    path.write_text("\n".join(L) + "\n")


def build(name):
    spec = SPECS[name]; A = OUT / name; A.mkdir(parents=True, exist_ok=True)
    parts = load_parts(spec); tri_before = sum(len(m.faces) for _, m, _, _ in parts)
    V, F, UV = merge(parts)
    if spec.get("fix") == "close_lid": V = close_lid(V); parts = [(parts[0][0], trimesh.Trimesh(V, F, process=False), parts[0][2], parts[0][3])]
    ext_src = np.ptp(V, axis=0); sf = scale_factor(spec["scale"], ext_src)
    R = np.eye(4) if spec.get("zup") else R_YUP_TO_ZUP  # the glasses case GLB already stacks its halves along z
    S = np.diag(np.r_[sf, 1.0]); V = trimesh.transform_points(V, R @ S)
    origin = np.r_[(V[:, :2].min(0) + V[:, :2].max(0)) / 2, V[:, 2].min()]; V = V - origin
    # decimate when over 50k triangles (none of the picked sources is; kept for completeness, drops the UVs)
    decimated = False
    if len(F) > 50000:
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)).simplify_quadric_decimation(20000)
        V, F, UV = np.asarray(m.vertices), np.asarray(m.triangles), None; decimated = True
    # texture: the first textured part's image (all kept parts of one object share it, checked per object below)
    imgs = [img for _, _, img, _ in parts if img is not None]; textured = len(imgs) > 0 and spec["rgba"] is None and UV is not None
    if textured:
        assert all(i is imgs[0] or np.array_equal(np.asarray(i), np.asarray(imgs[0])) for i in imgs), "parts use different textures"
        imgs[0].convert("RGB").save(A / f"{name}_tex.png")
        (A / f"{name}.mtl").write_text(f"newmtl {name}_mat\nKd 1 1 1\nmap_Kd {name}_tex.png\n")
    # visual geoms: one textured/plain obj per material group when the object mixes textured + plain parts (power strip base box)
    vis = []
    if textured and any(img is None for _, _, img, _ in parts):
        tex_parts = [p for p in parts if p[2] is not None]; plain_parts = [p for p in parts if p[2] is None]
        for tag, pp in (("", tex_parts), ("_base", plain_parts)):
            Vp, Fp, UVp = merge(pp); Vp = trimesh.transform_points(Vp, R @ S) - origin; fn = f"{name}_visual{tag}.obj"
            if tag == "": write_obj(A / fn, Vp, Fp, UVp, f"{name}.mtl", f"{name}_mat"); vis.append((fn, None))
            else:
                fac = pp[0][3]; rgba = " ".join(f"{c/255:.3f}" for c in fac[:3]) + " 1" if fac is not None else "0.1 0.1 0.1 1"
                write_obj(A / fn, Vp, Fp); vis.append((fn, rgba))
    elif textured:
        write_obj(A / f"{name}_visual.obj", V, F, UV, f"{name}.mtl", f"{name}_mat"); vis.append((f"{name}_visual.obj", None))
    else:
        write_obj(A / f"{name}_visual.obj", V, F); vis.append((f"{name}_visual.obj", spec["rgba"] or "0.6 0.6 0.6 1"))
    # collision: CoACD on the merged, vertex-welded mesh
    tm = trimesh.Trimesh(V, F, process=True)
    res = coacd.run_coacd(coacd.Mesh(tm.vertices, tm.faces), threshold=spec["threshold"], max_convex_hull=spec["hulls"], preprocess_mode="auto", seed=0)
    for old in A.glob(f"{name}_hull*.obj"): old.unlink()
    hulls = []
    for i, (hv, hf) in enumerate(res):
        h = trimesh.Trimesh(hv, hf, process=False).convex_hull; fn = f"{name}_hull{i:02d}.obj"; h.export(A / fn); hulls.append((fn, float(h.volume)))
    vol = sum(v for _, v in hulls); size = np.ptp(V, axis=0)
    # xml
    L = ["<mujoco>", "  <asset>"]
    if textured: L += [f'    <texture name="{name}_tex" type="2d" file="{name}_tex.png"/>', f'    <material name="{name}_mat" texture="{name}_tex" specular="0.2" shininess="0.2"/>']
    L += [f'    <mesh name="{fn[:-4]}" file="{fn}"/>' for fn, _ in vis] + [f'    <mesh name="{fn[:-4]}" file="{fn}"/>' for fn, _ in hulls]
    L += ["  </asset>", "  <default>", f'    <default class="{name}_col"><geom type="mesh" group="3" condim="4" friction="0.8 0.005 0.0001"/></default>',
          f'    <default class="{name}_vis"><geom type="mesh" group="2" contype="0" conaffinity="0" mass="0"/></default>', "  </default>", "  <worldbody>",
          f'    <body name="{name}" pos="0 0 0">', "      <freejoint/>"]
    for fn, rgba in vis:
        L.append(f'      <geom class="{name}_vis" mesh="{fn[:-4]}" ' + (f'material="{name}_mat"/>' if rgba is None else f'rgba="{rgba}"/>'))
    L += [f'      <geom class="{name}_col" mesh="{fn[:-4]}" mass="{spec["mass"] * v / vol:.5f}" rgba="{PALETTE[i % len(PALETTE)]} 1"/>' for i, (fn, v) in enumerate(hulls)]
    L += ["    </body>", "  </worldbody>", "</mujoco>"]; (A / f"{name}.xml").write_text("\n".join(L) + "\n")
    meta = {"name": name, "source": spec["source"], "url": spec["url"], "licence": spec["licence"], "author": spec["author"], "source_file": spec["file"],
            "kept_nodes": [p[0] for p in parts], "edits": spec["drop_note"], "source_extents_units": ext_src.tolist(), "scale_factor_xyz_source_axes": sf.tolist(),
            "uniform_scale": bool(np.allclose(sf, sf[0])), "rotation": "none (source already Z-up)" if spec.get("zup") else "glTF Y-up -> Z-up (+90 deg about x)", "size_m": size.tolist(), "triangles_source": int(tri_before),
            "triangles_visual": int(len(F)), "decimated": decimated, "textured": textured, "mass_kg": spec["mass"], "hulls": len(hulls), "hull_volume_m3": vol,
            "coacd": {"threshold": spec["threshold"], "max_convex_hull": spec["hulls"], "seed": 0}, "origin": "footprint centre, resting on z = 0"}
    json.dump(meta, open(A / f"{name}_meta.json", "w"), indent=1)
    print(f"{name}: tris {tri_before}->{len(F)} scale {sf.round(4)} size mm {(size*1000).round(1)} hulls {len(hulls)} vol {vol:.3e} textured {textured}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(SPECS)): build(n)

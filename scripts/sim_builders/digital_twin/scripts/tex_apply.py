"""Apply the cylindrical unwraps (textures/<obj>/unwrap.png, from tex_unwrap.py) to the downloaded bottle meshes.
Reads assets_dl/<obj>/ (visual obj + hulls + meta), scales the mesh to the real size (non-uniform xy allowed; hulls scaled the same
way, masses re-spread by volume), gives every face its own UVs: side faces u = atan2(y, x) (CCW from the cap end, like the unwrap),
v = height mapped onto the unwrap rows (h = h_per_z * z), top faces (nz > 0.8) and bottom faces (nz < -0.8) a flat colour tile.
The texture atlas = unwrap rolled so the least-covered column (the unseen back) is the seam, plus a 64-row strip of flat tiles.
Unseen texels take the row median of the seen ones; side faces above flat_above_z (cap, neck, pump head) use the flat tile.
Writes assets_dl/<obj>_real/<obj>_real_{visual.obj, tex.png, .mtl, hullNN.obj, .xml, meta.json}.
usage (LeRobot root): uv run python -I lerobot/scripts/sim_builders/digital_twin/scripts/tex_apply.py [pill_bottle pump_bottle]"""
import json, pathlib, sys, numpy as np, cv2, trimesh
ROOT = pathlib.Path(__file__).resolve().parents[5] / "assets/sim_sources/digital_twin"; DL = ROOT / "assets_dl"; TEX = ROOT / "textures"
PALETTE = ["0.85 0.35 0.30", "0.30 0.65 0.85", "0.40 0.75 0.40", "0.90 0.70 0.25", "0.70 0.45 0.80", "0.35 0.75 0.75", "0.85 0.55 0.55", "0.55 0.55 0.85"]
CFG = {  # scale: source asset -> real size; h_per_z: unwrap height per mesh height (label span of the scan / label span of the mesh)
    "pill_bottle": dict(scale=(0.050 / 0.05386, 0.050 / 0.05386, 1.0), h_per_z=1.0, label_px=330, cap_rows_h=(0.082, 0.098), mass=0.12, flat_above_z=0.078,
                        note="xy scaled 53.9 -> 50 mm (real diameter); z kept 100 mm; scan body top 72 mm = mesh body top 73 mm so h = z"),
    "pump_bottle": dict(scale=(0.045 / 0.061, 0.045 / 0.061, 0.230 / 0.180), h_per_z=0.165 / (0.125 * 0.230 / 0.180), label_px=1000, cap_rows_h=None, mass=0.15, flat_above_z=0.13 * 0.230 / 0.180,
                        max_gain=2.0, neutral=True,
                        note="2026-10-30: sized from the depth scan, 45 mm wide x 230 mm tall (the briefed 180 x 40 was a guess); scan body top 165 mm -> mesh body top 160 mm; gain capped at 2, body colour neutralised (the dim D405 scan had a green cast)"),
}
LABEL_TO_DEG = 0.0  # world azimuth (atan2) the label centre faces after mapping (+x: visible from the oblique az 135 and close az 200 cameras)
STRIP = 64


def main(name):
    C = CFG[name]; src = DL / name; out = DL / f"{name}_real"; out.mkdir(exist_ok=True); nn = f"{name}_real"
    meta0 = json.load(open(src / f"{name}_meta.json")); fr = json.load(open(TEX / name / "frames.json")); H_tex, TH_body = fr["H_m"], None
    m = trimesh.load(str(src / f"{name}_visual.obj"), force="mesh", process=False); S = np.array(C["scale"]); V = m.vertices * S; F = m.faces
    V[:, 2] -= V[:, 2].min(); V[:, :2] -= (V[:, :2].min(0) + V[:, :2].max(0)) / 2; size = np.ptp(V, axis=0)
    N = trimesh.Trimesh(V, F, process=False).face_normals
    # texture atlas: roll the unwrap so that the least covered body column is the seam
    unwrap = cv2.imread(str(TEX / name / "unwrap_raw.png")); wgt = cv2.imread(str(TEX / name / "weight.png"), 0).astype(float); TH, TW = wgt.shape
    body_rows = slice(int(TH * (1 - fr["body_m"] / H_tex)), TH); colcov = (wgt[body_rows] > 2).mean(0)
    k = 61; sm = np.convolve(np.concatenate([colcov[-k:], colcov, colcov[:k]]), np.ones(k) / k, "same")[k:-k]; gap = int(np.argmin(sm))
    rolled = np.roll(unwrap, -gap, axis=1); off = C["label_px"] - LABEL_TO_DEG / 360 * TW  # x_orig(angle) = angle/2pi*TW + off
    cov = wgt > 2
    for rr in range(TH):  # unseen texels take the row median of the seen ones (keeps the white / label / white bands); rows with nothing seen are filled below
        if cov[rr].sum() >= 50: unwrap[rr][~cov[rr]] = np.median(unwrap[rr][cov[rr]], 0)
    rolled = np.roll(unwrap, -gap, axis=1)
    if C["cap_rows_h"]:
        r0, r1 = [int(TH * (1 - h / H_tex)) for h in C["cap_rows_h"][::-1]]; sel = cov[r0:r1]; capcol = np.percentile(unwrap[r0:r1][sel], 80, axis=0) if sel.sum() > 100 else np.array([235, 235, 235.0])  # bright side of the cap rows (shading darkens the median)
    else: capcol = np.median(unwrap[body_rows][cov[body_rows]], 0)
    # exposure: the scans are dim (white cap ~ 90/255); the texture is meant as albedo, so gain the brightest 1 % of covered body pixels to 235
    gain = min(235.0 / max(float(np.percentile(unwrap[body_rows][cov[body_rows]].max(1), 99)), 1.0), C.get("max_gain", 1e9))
    if C.get("neutral"):  # grey-world on the body: equalise the channel medians of the seen body texels
        med = np.median(unwrap[body_rows][cov[body_rows]], 0).astype(float); rolled = np.clip(rolled.astype(float) * (med.mean() / med), 0, 255); capcol = capcol * (med.mean() / med)
    rolled = np.clip(rolled.astype(float) * gain, 0, 255).astype(np.uint8); capcol = np.clip(capcol * gain, 0, 255)
    atlas = np.zeros((TH + STRIP, TW, 3), np.uint8); atlas[:TH] = rolled; atlas[TH:] = capcol.astype(np.uint8); AH = TH + STRIP
    atlas[:TH][(cov.sum(1) < 50)] = capcol.astype(np.uint8)
    cv2.imwrite(str(out / f"{nn}_tex.png"), atlas)
    # per-corner UVs
    uv = np.zeros((len(F), 3, 2)); kind = np.where(N[:, 2] > 0.8, 1, np.where(N[:, 2] < -0.8, 2, 0)); kind[(V[F][:, :, 2] > C["flat_above_z"]).all(1)] = 1  # cap / neck / pump head: flat colour
    for i, (f, kd) in enumerate(zip(F, kind)):
        P = V[f]
        if kd == 0:
            ang = np.arctan2(P[:, 1], P[:, 0]) % (2 * np.pi); x = ((ang / (2 * np.pi)) * TW + off - gap) % TW; u = x / TW
            if u.max() - u.min() > 0.5: u = np.where(u < 0.5, u + 1, u)
            h = np.clip(C["h_per_z"] * P[:, 2], 0, H_tex); row = TH * (1 - h / H_tex); v = 1 - row / AH
        else:
            u = 0.25 + 0.5 * np.random.RandomState(i).rand(3) * 0; v = 1 - (TH + STRIP / 2) / AH + 0 * u
        uv[i, :, 0] = u; uv[i, :, 1] = v
    L = [f"mtllib {nn}.mtl", f"usemtl {nn}_mat"] + [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    L += [f"vt {u:.6f} {v:.6f}" for u, v in uv.reshape(-1, 2)]
    L += [f"f {a+1}/{3*i+1} {b+1}/{3*i+2} {c+1}/{3*i+3}" for i, (a, b, c) in enumerate(F)]
    (out / f"{nn}_visual.obj").write_text("\n".join(L) + "\n"); (out / f"{nn}.mtl").write_text(f"newmtl {nn}_mat\nKd 1 1 1\nmap_Kd {nn}_tex.png\n")
    # hulls: same scale (affine keeps convexity), origin shift identical to the visual
    hulls = []
    for i, hp in enumerate(sorted(src.glob(f"{name}_hull*.obj"))):
        h = trimesh.load(str(hp), force="mesh"); hv = h.vertices * S; hv[:, 2] -= (m.vertices[:, 2] * S[2]).min(); hv[:, :2] -= ((m.vertices[:, :2] * S[:2]).min(0) + (m.vertices[:, :2] * S[:2]).max(0)) / 2
        hh = trimesh.Trimesh(hv, h.faces, process=False).convex_hull; fn = f"{nn}_hull{i:02d}.obj"; hh.export(out / fn); hulls.append((fn, float(hh.volume)))
    vol = sum(v for _, v in hulls)
    X = ["<mujoco>", "  <asset>", f'    <texture name="{nn}_tex" type="2d" file="{nn}_tex.png"/>', f'    <material name="{nn}_mat" texture="{nn}_tex" specular="0.2" shininess="0.2"/>',
         f'    <mesh name="{nn}_visual" file="{nn}_visual.obj"/>'] + [f'    <mesh name="{fn[:-4]}" file="{fn}"/>' for fn, _ in hulls]
    X += ["  </asset>", "  <default>", f'    <default class="{nn}_col"><geom type="mesh" group="3" condim="4" friction="0.8 0.005 0.0001"/></default>',
          f'    <default class="{nn}_vis"><geom type="mesh" group="2" contype="0" conaffinity="0" mass="0"/></default>', "  </default>", "  <worldbody>",
          f'    <body name="{nn}" pos="0 0 0">', "      <freejoint/>", f'      <geom class="{nn}_vis" mesh="{nn}_visual" material="{nn}_mat"/>']
    X += [f'      <geom class="{nn}_col" mesh="{fn[:-4]}" mass="{C["mass"] * v / vol:.5f}" rgba="{PALETTE[i % len(PALETTE)]} 1"/>' for i, (fn, v) in enumerate(hulls)]
    X += ["    </body>", "  </worldbody>", "</mujoco>"]; (out / f"{nn}.xml").write_text("\n".join(X) + "\n")
    meta = dict(meta0, name=nn, base_asset=f"assets_dl/{name}", texture_source=f"textures/{name}/unwrap.png (tex_unwrap.py, D405 scan)", extra_scale_xyz=S.tolist(),
                uniform_scale=bool(np.allclose(S, S[0])) and meta0["uniform_scale"], size_m=size.tolist(), textured=True, mass_kg=C["mass"], hulls=len(hulls), hull_volume_m3=vol,
                uv=dict(side="u = atan2(y,x) CCW from +z, label centre facing +x; v: unwrap row = TH*(1 - h/H_tex), h = h_per_z * z; side faces above flat_above_z use the flat tile", flat_above_z=C["flat_above_z"], h_per_z=C["h_per_z"], H_tex_m=H_tex,
                        seam_col_px=gap, label_col_px=C["label_px"], top_bottom="flat colour tile (bgr %s)" % capcol.astype(int).tolist(), exposure_gain=round(gain, 3), faces=dict(side=int((kind == 0).sum()), top=int((kind == 1).sum()), bottom=int((kind == 2).sum()))),
                edits=meta0["edits"] + "; " + C["note"])
    json.dump(meta, open(out / f"{nn}_meta.json", "w"), indent=1)
    print(f"{nn}: size mm {(size*1000).round(1)} faces side/top/bottom {(kind==0).sum()}/{(kind==1).sum()}/{(kind==2).sum()} seam col {gap} cap colour bgr {capcol.astype(int).tolist()} gain {gain:.2f} hulls {len(hulls)} vol {vol:.3e}")


for n in (sys.argv[1:] or list(CFG)): main(n)

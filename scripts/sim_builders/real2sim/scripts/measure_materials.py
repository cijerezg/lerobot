"""Material colours of the nail scene measured from real pixels WELL INSIDE each region (hand-picked windows on 10-16 ep 5,
work/scene_nail/frames/r1016_ep5), in both cameras, plus the per-channel affine map wrist = a * top + b between the two
cameras fitted on the materials seen in both (wood, foam, black end, slot floor, clipper body, clipper head). Clutter colours =
mean of the real top frame 0 inside the eroded sim silhouette of each static primitive (ext camera). Writes OUT/measured.json
(BGR, 0-255) and OUT/samples.jpg (the windows drawn on the frames, to check they sit inside their region).
usage: MUJOCO_GL=egl uv run python scripts/measure_materials.py SCENE_XML FRAMES_DIR OUT_DIR"""
import argparse, json, pathlib, cv2, mujoco, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("frames"); ap.add_argument("out"); a = ap.parse_args()
fr = pathlib.Path(a.frames); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
# windows (x0, y0, x1, y1) in 640x480 pixels; several per material are pooled. Top frame 0: robot parked, case lit (2 mm/px).
TOP = {
    "foam": [(323, 200, 329, 227), (339, 200, 344, 227), (352, 200, 358, 227)],
    "black_end": [(323, 181, 329, 193), (339, 181, 344, 193), (352, 181, 358, 193)],
    "slot": [(332, 197, 336, 220), (346, 197, 350, 215)],
    "clipper_body": [(269, 211, 291, 215), (403, 204, 408, 214)],
    "clipper_head": [(291, 212, 297, 217), (403, 216, 408, 222)],
    "clipper_file": [(261, 209, 268, 214), (403, 196, 408, 203)],
    "wood": [(365, 195, 395, 230), (300, 190, 318, 230)],
}
# wrist frames: f1002 case close and lit on its left/middle (arm shadow right of it), f668 large clipper close, f2005 small clipper, f334 second case view
WRIST = {
    "foam": [(1002, 183, 320, 198, 400), (1002, 250, 280, 310, 350), (1002, 388, 300, 445, 400), (334, 150, 100, 190, 140)],
    "black_end": [(1002, 225, 440, 290, 470), (334, 30, 280, 100, 340)],
    "slot": [(1002, 212, 330, 228, 440), (1002, 322, 322, 348, 358)],
    "rim": [(1002, 175, 300, 183, 400), (1002, 250, 212, 350, 220)],
    "clipper_body": [(668, 320, 350, 380, 410), (2005, 340, 350, 356, 372)],
    "clipper_head": [(668, 322, 300, 378, 345), (2005, 360, 300, 378, 318)],
    "wood": [(1002, 20, 20, 200, 180), (668, 20, 200, 280, 320)],
    "wood_shadow": [(1002, 420, 60, 560, 150)],
}
imgs = {}
def img(cam, f):
    if (cam, f) not in imgs: imgs[(cam, f)] = cv2.imread(str(fr / f"{cam}_{f:05d}.jpg"))
    return imgs[(cam, f)]
res = {"top": {}, "wrist": {}}
for name, wins in TOP.items():
    px = np.concatenate([img("top", 0)[y0:y1, x0:x1].reshape(-1, 3) for x0, y0, x1, y1 in wins]).astype(float)
    res["top"][name] = dict(mean=px.mean(0).round(1).tolist(), median=np.median(px, 0).round(1).tolist(), std=px.std(0).round(1).tolist(), n=len(px))
for name, wins in WRIST.items():
    px = np.concatenate([img("wrist", f)[y0:y1, x0:x1].reshape(-1, 3) for f, x0, y0, x1, y1 in wins]).astype(float)
    per = [img("wrist", f)[y0:y1, x0:x1].reshape(-1, 3).mean(0).round(0).tolist() for f, x0, y0, x1, y1 in wins]
    res["wrist"][name] = dict(mean=px.mean(0).round(1).tolist(), median=np.median(px, 0).round(1).tolist(), std=px.std(0).round(1).tolist(), n=len(px), per_window=per)
# clutter: eroded sim silhouettes of the static primitives in the ext camera over the real top frame 0
model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, 480, 640); r.enable_segmentation_rendering(); r.update_scene(data, "ext"); seg = r.render()[:, :, 0]
res["clutter"] = {}
for g in range(model.ngeom):
    name = model.geom(g).name
    if model.geom_bodyid[g] != 0 or name in ("", "table", "floor", "table_top"): continue
    m = cv2.erode((seg == g).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if m.sum() < 20: m = seg == g
    px = img("top", 0)[m].astype(float); res["clutter"][name] = dict(mean=px.mean(0).round(1).tolist(), median=np.median(px, 0).round(1).tolist(), n=int(m.sum()))
# affine fit per channel on the materials measured in both cameras (medians)
pairs = ["wood", "foam", "black_end", "slot", "clipper_body", "clipper_head"]
T = np.array([res["top"][p]["median"] for p in pairs]); Wd = np.array([res["wrist"][p]["median"] for p in pairs])
fit = {}
for c, ch in enumerate("bgr"):
    A = np.c_[T[:, c], np.ones(len(pairs))]; (ga, gb), *_ = np.linalg.lstsq(A, Wd[:, c], rcond=None); g0 = (T[:, c] @ Wd[:, c]) / (T[:, c] @ T[:, c])
    fit[ch] = dict(gain=float(ga), offset=float(gb), resid=(Wd[:, c] - A @ [ga, gb]).round(1).tolist(), gain_only=float(g0), resid_gain_only=(Wd[:, c] - g0 * T[:, c]).round(1).tolist())
res["affine_wrist_from_top"] = dict(pairs=pairs, top_median=T.round(1).tolist(), wrist_median=Wd.round(1).tolist(), per_channel_bgr=fit)
json.dump(res, open(out / "measured.json", "w"), indent=1)
for name in TOP: print(f"top   {name:13s} median {res['top'][name]['median']} std {res['top'][name]['std']} n {res['top'][name]['n']}")
for name in WRIST: print(f"wrist {name:13s} median {res['wrist'][name]['median']} std {res['wrist'][name]['std']} per window {res['wrist'][name]['per_window']}")
for name, v in res["clutter"].items(): print(f"clutter {name:28s} median {v['median']} n {v['n']}")
for ch, v in fit.items(): print(f"affine {ch}: gain {v['gain']:.3f} offset {v['offset']:.1f} resid {v['resid']} | gain only {v['gain_only']:.3f} resid {v['resid_gain_only']}")
# review sheet
tiles = []
t = img("top", 0).copy()
for name, wins in TOP.items():
    for x0, y0, x1, y1 in wins: cv2.rectangle(t, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
tiles.append(cv2.resize(t[150:260, 230:440], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
for f in (1002, 668, 2005, 334):
    w = img("wrist", f).copy()
    for name, wins in WRIST.items():
        for ff, x0, y0, x1, y1 in wins:
            if ff == f: cv2.rectangle(w, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1); cv2.putText(w, name, (x0, max(y0 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    tiles.append(w)
h = max(t.shape[0] for t in tiles); cv2.imwrite(str(out / "samples.jpg"), np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("wrote", out)

"""Render ext + wrist of a scene at real recorded poses, side by side with the real frames.
usage: render_room.py SCENE_XML ROOT REAL_DIR OUT_DIR FRAMES... [--ref SCENE_XML_REF]"""
import sys, pathlib, cv2, numpy as np, mujoco, pandas as pd
args = sys.argv[1:]; ref = None
if "--ref" in args: i = args.index("--ref"); ref = args[i + 1]; args = args[:i] + args[i + 2:]
scene, root, real, out = args[0], args[1], pathlib.Path(args[2]), pathlib.Path(args[3]); frames = [int(f) for f in args[4:]]; out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "lerobot/src"); from lerobot.probes.action_trace_sim import to_qpos
df = pd.read_parquet(f"{root}/data/chunk-000/file-000.parquet", columns=["observation.state"]); states = np.stack(df["observation.state"].to_numpy())
def label(im, t):
    cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1); return im
def load(xml):
    m = mujoco.MjModel.from_xml_path(xml); return m, mujoco.MjData(m), mujoco.Renderer(m, 480, 640)
def render(m, d, r, fr, cam):
    d.qpos[:8] = to_qpos(states[fr]); mujoco.mj_forward(m, d); r.update_scene(d, cam); return np.ascontiguousarray(r.render()[:, :, ::-1])
m, d, r = load(scene); rows = []
for fr in frames:
    top, wr = render(m, d, r, fr, "ext"), render(m, d, r, fr, "wrist"); rt, rw = cv2.imread(str(real / f"top_f{fr}.png")), cv2.imread(str(real / f"wrist_f{fr}.png"))
    cv2.imwrite(str(out / f"sim_top_f{fr}.jpg"), top, [cv2.IMWRITE_JPEG_QUALITY, 92]); cv2.imwrite(str(out / f"sim_wrist_f{fr}.jpg"), wr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    row = np.hstack([label(rt.copy(), f"real top f{fr}"), label(top.copy(), "sim top (room)"), label(rw.copy(), f"real wrist f{fr}"), label(wr.copy(), "sim wrist (room)")])
    cv2.imwrite(str(out / f"side_by_side_f{fr}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 90]); rows.append(row)
    # colour patches: wall (top centre-left of the wrist view) and floor (if the real frame shows it) as mean BGR
    for name, (y0, y1, x0, x1) in dict(wall=(30, 120, 60, 200), low=(300, 420, 60, 200)).items():
        print(f"f{fr} {name} patch mean BGR real {rw[y0:y1, x0:x1].reshape(-1, 3).mean(0).round(0)} sim {wr[y0:y1, x0:x1].reshape(-1, 3).mean(0).round(0)}")
cv2.imwrite(str(out / "side_by_side_all.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
if ref:
    m2, d2, r2 = load(ref); a, b = render(m, d, r, 0, "ext"), render(m2, d2, r2, 0, "ext"); aw, bw = render(m, d, r, 0, "wrist"), render(m2, d2, r2, 0, "wrist")
    diff = np.abs(a.astype(int) - b.astype(int)).sum(-1); print("ext park: pixels differing > 30:", int((diff > 30).sum()), "of", diff.size, "; table-interior region rows 100-350 cols 150-500 differing:", int((diff[100:350, 150:500] > 30).sum()))
    cv2.imwrite(str(out / "ref_vs_room_ext.jpg"), np.hstack([label(b, "ext clippers_only"), label(a, "ext room"), label(cv2.applyColorMap(np.clip(diff, 0, 255).astype(np.uint8), cv2.COLORMAP_JET), "|diff|")]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(out / "ref_vs_room_wrist.jpg"), np.hstack([label(bw, "wrist clippers_only"), label(aw, "wrist room")]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("wrote", out)

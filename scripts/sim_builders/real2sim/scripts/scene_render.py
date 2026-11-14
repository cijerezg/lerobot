"""Render the full observation from scene_nail.xml for real frames: top RGB, wrist RGB, wrist depth (sidecar units),
next to the real ones. Robot posed from observation.state; objects stay at the scene's start pose.
usage: scene_render.py ROOT EP SCENE_XML OUT_DIR --frames f1 f2 ... [--tag name]
Depth: MuJoCo depth render (metres along the optical axis) -> uint16 at 0.1 mm per level, zeroed below the D405 minimum
range (0.07 m) and above DEPTH_MAX_M. Error metric: median |sim - real| on table pixels (real RANSAC plane inliers)."""
import argparse, json, pathlib, sys, cv2, numpy as np, mujoco
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scene_sim as S, scene_wrist as Wr
from scene_frames import episode_states, decode_frames, TOP, WRIST

DEPTH_MAX_M = 2.5   # the real sidecar holds values up to 2.37 m in 10-16 ep 5 (D405 spec range ends at 0.5 m)


def sim_depth_sidecar(depth_m):
    d = np.where((depth_m >= Wr.D405_MIN) & (depth_m <= DEPTH_MAX_M), depth_m, 0.0); return np.round(d / Wr.UNITS_M).astype(np.uint16)


def label(im, t):
    cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1); return im


def render_frame(model, data, V, states, f):
    S.set_state(model, data, states[f])
    return dict(top=np.ascontiguousarray(V.rgb(data, "ext")[:, :, ::-1]), wrist=np.ascontiguousarray(V.rgb(data, "wrist")[:, :, ::-1]), depth_m=V.depth(data, "wrist"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("ep", type=int); ap.add_argument("scene"); ap.add_argument("out"); ap.add_argument("--frames", type=int, nargs="+", required=True); ap.add_argument("--tag", default=""); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); V = S.Views(model)
    states, _ = episode_states(a.root, a.ep); real_top = decode_frames(a.root, a.ep, TOP, a.frames); real_wrist = decode_frames(a.root, a.ep, WRIST, a.frames)
    rows_w, rows_d, rows_t, table = [], [], [], []
    for f in a.frames:
        r = render_frame(model, data, V, states, f); sim16 = sim_depth_sidecar(r["depth_m"]); real16, fd = Wr.load_depth(a.root, a.ep, f)
        cv2.imwrite(str(out / f"sim_wrist_depth{a.tag}_{f:05d}.png"), sim16); cv2.imwrite(str(out / f"sim_wrist{a.tag}_{f:05d}.jpg"), r["wrist"]); cv2.imwrite(str(out / f"sim_top{a.tag}_{f:05d}.jpg"), r["top"])
        real_m = real16 * Wr.UNITS_M; sim_m = sim16 * Wr.UNITS_M; valid = (real16 > 0) & (sim16 > 0)
        # table pixels = inliers of the real plane (fit on valid real depth < 1 m)
        vr = (real16 > 0) & (real_m < 1.0); pts = Wr.backproject(np.where(vr, real_m, 0)); P = pts[vr]; sub = P[np.random.default_rng(0).choice(len(P), min(6000, len(P)), replace=False)]; n, d, _ = Wr.ransac_plane(sub)
        tab = np.zeros(vr.shape, bool); tab[vr] = np.abs(P @ n + d) < 0.004; m = tab & valid; err = sim_m[m] - real_m[m]
        table.append(dict(frame=f, depth_frame=fd, table_px=int(m.sum()), median_abs_mm=float(1e3 * np.median(np.abs(err))), median_signed_mm=float(1e3 * np.median(err)), p90_abs_mm=float(1e3 * np.percentile(np.abs(err), 90)), real_median_m=float(np.median(real_m[m])), sim_median_m=float(np.median(sim_m[m]))))
        print(f"f{f} (depth f{fd}) table px {m.sum()}: median |sim-real| {table[-1]['median_abs_mm']:.1f} mm, median signed {table[-1]['median_signed_mm']:+.1f} mm, p90 {table[-1]['p90_abs_mm']:.1f} mm, real median {table[-1]['real_median_m']:.3f} sim median {table[-1]['sim_median_m']:.3f}")
        rw = real_wrist[f]; rows_w.append(np.hstack([label(rw.copy(), f"real wrist f{f}"), label(r["wrist"].copy(), "sim wrist"), label(cv2.addWeighted(rw, 0.5, r["wrist"], 0.5, 0), "50/50 blend")]))
        diff = np.abs(sim_m - real_m); diff_im = cv2.applyColorMap((255 * np.clip(diff / 0.05, 0, 1)).astype(np.uint8), cv2.COLORMAP_VIRIDIS); diff_im[~valid] = 0
        rows_d.append(np.hstack([label(Wr.colourmap(real_m), f"real depth f{fd} (0.05-0.50 m)"), label(Wr.colourmap(sim_m), "sim depth"), label(diff_im, f"|diff| 0-50 mm; table median {table[-1]['median_abs_mm']:.0f} mm")]))
        rt = real_top[f]; rows_t.append(np.hstack([label(rt.copy(), f"real top f{f}"), label(r["top"].copy(), "sim top"), label(cv2.addWeighted(rt, 0.5, r["top"], 0.5, 0), "blend")]))
        cv2.imwrite(str(out / f"wrist_composite{a.tag}_{f:05d}.jpg"), rows_w[-1], [cv2.IMWRITE_JPEG_QUALITY, 90]); cv2.imwrite(str(out / f"depth_composite{a.tag}_{f:05d}.jpg"), rows_d[-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out / f"wrist_composites{a.tag}.jpg"), np.vstack(rows_w), [cv2.IMWRITE_JPEG_QUALITY, 85]); cv2.imwrite(str(out / f"depth_composites{a.tag}.jpg"), np.vstack(rows_d), [cv2.IMWRITE_JPEG_QUALITY, 85]); cv2.imwrite(str(out / f"top_composites{a.tag}.jpg"), np.vstack(rows_t), [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(table, open(out / f"depth_errors{a.tag}.json", "w"), indent=1); print("wrote", out)

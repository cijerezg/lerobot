"""Metric scale for VGGT: median ratio D405 depth / VGGT depth over all frames (valid D405 0.07-0.45 m, outside the gripper mask,
VGGT conf above --conf_pct). usage: d405_scale.py scan_dir work_dir [--conf_pct 50] -> work_dir/scale.json"""
import argparse, json, pathlib, cv2, numpy as np
p = argparse.ArgumentParser(); p.add_argument("scan"); p.add_argument("work"); p.add_argument("--conf_pct", type=float, default=50); a = p.parse_args()
S = pathlib.Path(a.scan); W = pathlib.Path(a.work); d = np.load(W / "vggt.npz"); K = json.load(open(S / "intrinsic.json"))
H, Wd = d["depth"].shape[1:3]; gm = cv2.resize(cv2.imread(str(S.parent / "gripper_mask.png"), 0), (Wd, H), interpolation=cv2.INTER_NEAREST) > 0
thr = np.percentile(d["depth_conf"], a.conf_pct); per = []; allr = []
for i, f in enumerate(d["frames"]):
    z = cv2.imread(str(S / "depth" / (pathlib.Path(str(f)).stem + ".png")), -1).astype(np.float32) * K["depth_scale"]
    z = cv2.resize(z, (Wd, H), interpolation=cv2.INTER_NEAREST); v = d["depth"][i, ..., 0]
    ok = (z > 0.07) & (z < 0.45) & ~gm & (d["depth_conf"][i] > thr) & (v > 1e-4)
    if ok.sum() < 500: per.append(np.nan); continue
    r = z[ok] / v[ok]; per.append(float(np.median(r))); allr.append(r[:: max(1, len(r) // 3000)])
per = np.array(per); allr = np.concatenate(allr); s = float(np.median(allr)); good = ~np.isnan(per)
res = dict(scale_m_per_unit=s, n_frames_used=int(good.sum()), per_frame_median_iqr=[float(np.nanpercentile(per, 25)), float(np.nanpercentile(per, 75))],
           per_frame_median_p5_p95=[float(np.nanpercentile(per, 5)), float(np.nanpercentile(per, 95))], pixel_ratio_iqr=[float(np.percentile(allr, 25)), float(np.percentile(allr, 75))],
           vggt_fx_median=float(np.median(d["intrinsics"][:, 0, 0])), d405_fx_at_vggt_res=K["fx"] * Wd / K["width"], conf_pct=[float(np.percentile(d["depth_conf"], q)) for q in (10, 50, 90)])
cams = -np.einsum("nij,nj->ni", d["extrinsics"][:, :3, :3].transpose(0, 2, 1), d["extrinsics"][:, :3, 3]) * s
res["camera_track_extent_m"] = np.ptp(cams, axis=0).tolist(); res["camera_step_median_m"] = float(np.median(np.linalg.norm(np.diff(cams, axis=0), axis=1)))
json.dump(res, open(W / "scale.json", "w"), indent=1); print(json.dumps(res))

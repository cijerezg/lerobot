"""Cylindrical unwrap texture of a bottle lying on the table, from D405 colour+depth frames (camera looks down on it).
Per frame: table plane (RANSAC), object = above-plane connected component (colour prior for the pill bottle), axis = in-plane
principal direction of the component, base end = the wider end (the cap / pump head tapers), axis position = fixed-radius
circle fit in the cross-section (robust LS, points well above the table), texels (angle, height) back-projected into the image
and sampled, weight = (normal . view)^2, per-frame registration = argmax of circular NCC against the accumulated map
(FFT over all angle shifts x height shifts of +-8 rows; mild continuity prior on the angle).
Texture: u = angle, CCW seen from the cap end (text reads unmirrored); v: row 0 = top (cap end), last row = base.
usage: tex_unwrap.py pill_bottle|pump_bottle [--stride 3] [--refine 2]
writes textures/<obj>/{unwrap.png, unwrap_raw.png, weight.png, cap.png, check.jpg, frames.json}"""
import argparse, json, pathlib, cv2, numpy as np, open3d as o3d
ROOT = pathlib.Path(__file__).resolve().parents[5] / "assets/sim_sources/digital_twin"; S = ROOT / "scans"; OUT = ROOT / "textures"
CFG = {  # dimensions measured from the depth profile along the axis (body = cylindrical part from the base, H = total length)
    "pill_bottle": dict(scan="obj_all", ranges=[(990, 1280), (1395, 1575), (1845, 2205)], r=0.025, H=0.115, body=0.072, roll=True,
                        prior=lambda hsv: ((hsv[..., 0] < 22) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 70), prior_min=0.15),
    "pump_bottle": dict(scan="obj_orbit", ranges=[(1170, 1680)], r=0.0225, H=0.255, body=0.190, prior=None, prior_min=0, roll=False),  # roll=False: bottle never rolled, angle search limited to +-40 deg of the previous frame
}
TW, TH, DV, WIN = 1024, 512, 16, 40
a = argparse.ArgumentParser(); a.add_argument("obj"); a.add_argument("--stride", type=int, default=3); a.add_argument("--zmax", type=float, default=0.45); a.add_argument("--refine", type=int, default=0, help="leave-one-out re-registration passes against the map built from the other frames"); args = a.parse_args()
C = CFG[args.obj]; K = json.load(open(S / C["scan"] / "intrinsic.json")); fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
gm = cv2.imread(str(S / "gripper_mask.png"), 0) > 0; r, H, BODY = C["r"], C["H"], C["body"]
uu, vv = np.meshgrid(np.arange(848), np.arange(480)); rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1)
frames = [i for lo, hi in C["ranges"] for i in range(lo, hi + 1, args.stride)]
theta = (np.arange(TW) + 0.5) / TW * 2 * np.pi; hgt = H * (1 - (np.arange(TH) + 0.5) / TH); TT, HH = np.meshgrid(theta, hgt)
BODY_ROWS = slice(int(TH * (1 - BODY / H)), TH)

def plane(pts):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)); m, _ = pc.segment_plane(0.003, 3, 400); n = np.array(m[:3]); d = m[3]
    if d < 0: n, d = -n, -d  # camera (origin) side: n.p + d > 0
    return n / np.linalg.norm(n), d / np.linalg.norm(np.array(m[:3]))

def fit_center(q):  # q: Nx2 cross-section points, coordinate 0 = up (towards the camera); fixed radius r; robust Gauss-Newton
    c = q.mean(0) + np.array([-0.7 * r, 0.0])
    for _ in range(15):
        d = q - c; L = np.linalg.norm(d, axis=1) + 1e-9; res = L - r; J = -d / L[:, None]; w = (np.abs(res) < 0.006).astype(float) + 0.05
        c = c - np.linalg.solve((J * w[:, None]).T @ J + 1e-9 * np.eye(2), (J * w[:, None]).T @ res)
    L = np.linalg.norm(q - c, axis=1); kasa = np.linalg.lstsq(np.c_[2 * q, np.ones(len(q))], (q ** 2).sum(1), rcond=None)[0]
    return c, L, float(np.sqrt(max(kasa[2] + kasa[0] ** 2 + kasa[1] ** 2, 0)))

def project(X): return fx * X[..., 0] / X[..., 2] + cx, fy * X[..., 1] / X[..., 2] + cy

def geometry(pts, n, hp, why):
    """returns ax (base->top), e2, axis0 (axis point at the base), L (radial distances of the fit points), rfree, length"""
    hi_pts = pts[hp > 0.010]
    if len(hi_pts) < 500: why["few"] += 1; return None
    q = hi_pts - (hi_pts @ n)[:, None] * n; qm = q.mean(0); _, _, vt = np.linalg.svd((q - qm)[::4], full_matrices=False); ax = vt[0]; e2 = np.cross(ax, n)
    t = (q - qm) @ ax; w = (q - qm) @ e2; B = 0.005; lo, hi = np.percentile(t, 0.5), np.percentile(t, 99.5); length = hi - lo
    nb = int((hi - lo) / B) + 1; bi = np.clip(((t - lo) / B).astype(int), 0, nb - 1); cnt = np.bincount(bi, minlength=nb); wid = np.zeros(nb)
    for b in np.where(cnt > 15)[0]: wb = w[bi == b]; wid[b] = np.percentile(wb, 97) - np.percentile(wb, 3)
    if wid.max() > 4.2 * r: why["wide"] += 1; return None  # merged with something (hand, case)
    def end_info(sign):  # sign +1: the hi end, -1: the lo end
        bins = range(nb - 4, nb) if sign > 0 else range(0, 4); ws = [wid[b] for b in bins if cnt[b] > 15]
        X = (hi_pts @ n).mean() * n + qm + (hi if sign > 0 else lo) * ax; u, v = project(X); border = not (8 < u < 840 and 8 < v < 472)
        return (np.mean(ws) if ws else 0.0), border
    w_hi, b_hi = end_info(+1); w_lo, b_lo = end_info(-1)
    if not b_hi and not b_lo:
        if length < 0.8 * H or length > 1.3 * H: why["len"] += 1; return None
        if max(w_hi, w_lo) < 1.2 * min(w_hi, w_lo) + 1e-9: why["ends"] += 1; return None
        base_hi = w_hi > w_lo
    elif b_hi != b_lo:
        base_hi = b_lo  # the end inside the image is the base
        if (w_hi if base_hi else w_lo) < 1.8 * r or length < 0.55 * H: why["cut"] += 1; return None
    else: why["border"] += 1; return None
    if base_hi: ax, e2, t, lo, hi = -ax, -e2, -t, -hi, -lo
    t_base = lo + 0.004  # depth fattening extends the silhouette by a few mm
    mid = (t > t_base + 0.15 * BODY) & (t < t_base + 0.85 * BODY); q3 = hi_pts[mid]
    if len(q3) < 100: why["few"] += 1; return None
    c2, L, rfree = fit_center(np.stack([q3 @ n, q3 @ e2], 1)); cen3 = c2[0] * n + c2[1] * e2
    axis0 = cen3 - (cen3 @ ax) * ax + (qm @ ax + t_base) * ax
    return ax, e2, axis0, L, rfree, length

def process(idx, why):
    z = cv2.imread(str(S / C["scan"] / "depth" / f"{idx:06d}.png"), -1).astype(np.float32) * K["depth_scale"]; z[(z < 0.07) | (z > args.zmax) | gm] = 0
    img = cv2.imread(str(S / C["scan"] / "color" / f"{idx:06d}.jpg")); P = rays * z[..., None]; ok = z > 0
    n, d = plane(P[ok][::8]); h = P @ n + d; h[~ok] = -1
    above = ok & (h > 0.006) & (h < 1.5 * H); num, lab, st, _ = cv2.connectedComponentsWithStats(above.astype(np.uint8), connectivity=8)
    pri = C["prior"](cv2.cvtColor(img, cv2.COLOR_BGR2HSV)) if C["prior"] else None; best = None
    for k in range(1, num):
        if st[k, 4] < 3000 or (pri is not None and pri[lab == k].mean() < C["prior_min"]): continue
        G = geometry(P[lab == k], n, h[lab == k], why)
        if G is not None and (best is None or st[k, 4] > best[1]): best = (k, st[k, 4], G)
    if best is None: why["none"] += 1; return None
    k, _, (ax, e2, axis0, L, rfree, length) = best; m = lab == k; pm = P[m]; e1 = n
    inl = float((np.abs(L - r) < 0.006).mean())
    if inl < 0.5: why["inl"] += 1; return None
    X = axis0[None, None] + HH[..., None] * ax + r * (np.cos(TT)[..., None] * e1 + np.sin(TT)[..., None] * e2)
    nrm = np.cos(TT)[..., None] * e1 + np.sin(TT)[..., None] * e2; view = -X / np.linalg.norm(X, axis=-1, keepdims=True); cosv = (nrm * view).sum(-1)
    px, py = project(X); px = px.astype(np.float32); py = py.astype(np.float32)
    inside = (px >= 1) & (px < 847) & (py >= 1) & (py < 479) & (cosv > 0.25)
    mdil = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8))
    col = cv2.remap(img, px, py, cv2.INTER_LINEAR).astype(np.float32); zs = cv2.remap(z, px, py, cv2.INTER_NEAREST); ms = cv2.remap(mdil, px, py, cv2.INTER_NEAREST) > 0
    okz = (zs == 0) | (np.abs(zs - X[..., 2]) < 0.015); okz[:BODY_ROWS.start] &= zs[:BODY_ROWS.start] > 0  # cap / head rows: only depth-confirmed pixels (radius is smaller there, table shows behind)
    W = np.where(inside & ms & okz, np.clip(cosv, 0, 1) ** 2, 0).astype(np.float32)
    pm3 = pm - axis0; hh = pm3 @ ax; c1 = pm3 @ e1; c2_ = pm3 @ e2; rad = np.hypot(c1, c2_); capm = (hh > BODY - 0.005) & (rad < 1.05 * r)
    return dict(col=col, W=W, inl=inl, rfree=rfree, length_mm=round(1e3 * length), cap=(c1[capm], c2_[capm], img[m][capm]), prof=(hh, rad), idx=idx, npx=int(m.sum()), img=img, m=m)

def ncc_shift(F, W, M, A):  # score(s) = NCC over the overlap between F shifted right by s and M, for every s (circular along u)
    g = lambda x: np.fft.rfft(x, axis=1); ig = lambda x: np.fft.irfft(x, n=TW, axis=1)
    def corr(x, y): return ig(np.conj(g(x)) * g(y)).sum(0)
    Fm = (F * W).sum() / max(W.sum(), 1); Mm = (M * A).sum() / max(A.sum(), 1); F0 = (F - Fm) * W; M0 = (M - Mm) * A
    ov = corr(W, A); num = corr(F0, M0); den = np.sqrt(np.maximum(corr(F0 ** 2 / np.maximum(W, 1e-9), A), 1e-9) * np.maximum(corr(W, M0 ** 2 / np.maximum(A, 1e-9)), 1e-9))
    s = num / den; s[ov < 0.1 * max(W.sum(), 1)] = -1; return s

def gray(x): return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

def register(R, acc, accw, prev):  # returns (u shift, v shift, score)
    if accw.sum() == 0: return 0, 0, 1.0
    Mg = gray(acc / np.maximum(accw, 1e-6)[..., None])[BODY_ROWS]; A = np.minimum(accw, 1.0)[BODY_ROWS]; Fg = gray(R["col"]); W = R["W"]
    sh = (np.arange(TW) - prev + TW // 2) % TW - TW // 2; prior = 0.08 * np.exp(-0.5 * (sh / (TW / 9)) ** 2); best = (0, 0, -1, -1)
    if not C["roll"]: prior = np.where(np.abs(sh) * 360 / TW <= WIN, prior, -9)
    for dv in range(-DV, DV + 1):
        Fs = np.roll(Fg, dv, axis=0)[BODY_ROWS]; Ws = np.roll(W, dv, axis=0)[BODY_ROWS]; sc = ncc_shift(Fs, Ws, Mg, A); s = int(np.argmax(sc + prior))
        if sc[s] + prior[s] > best[3]: best = (s, dv, float(sc[s]), sc[s] + prior[s])
    return best[0], best[1], best[2]

def render_preview(tex, cap):  # three orthographic side views of the textured cylinder + the cap patch
    views = []
    for az in (0, 120, 240):
        h, w = 256, int(256 * (2 * r) / H) + 2; img = np.zeros((h, w, 3), np.uint8) + 40
        xs = (np.arange(w) - w / 2 + 0.5) / (w / 2 - 1); vis = np.abs(xs) < 1; ang = (np.arcsin(np.clip(xs[vis], -1, 1)) + np.deg2rad(az)) % (2 * np.pi)
        u = (ang / (2 * np.pi) * TW).astype(int) % TW; v = np.arange(h) * TH // h; shade = (0.55 + 0.45 * np.sqrt(1 - xs[vis] ** 2))[None, :, None]
        img[:, vis] = np.clip(tex[v][:, u] * shade, 0, 255).astype(np.uint8); views += [img, np.zeros((h, 8, 3), np.uint8)]
    return np.hstack(views + [cv2.resize(cap, (256, 256))])

def main():
    acc = np.zeros((TH, TW, 3), np.float32); accw = np.zeros((TH, TW), np.float32); used = []; caps = []; profs = []; prev = 0; keepf = []; shifts = {}; store = []
    why = dict(few=0, wide=0, len=0, ends=0, cut=0, border=0, none=0, inl=0, ncc=0)
    for idx in frames:
        R = process(idx, why)
        if R is None: continue
        s, dv, score = register(R, acc, accw, prev); jump = used and abs((s - prev + TW // 2) % TW - TW // 2) * 360 / TW > 20
        if score < 0.45 or (jump and score < 0.8): why["ncc"] += 1; continue
        prev = s; store.append(dict(idx=idx, col=R["col"].astype(np.uint8), W=R["W"].astype(np.float16), s=s, dv=dv)); WS = np.roll(np.roll(R["W"], s, axis=1), dv, axis=0); acc += np.roll(np.roll(R["col"], s, axis=1), dv, axis=0) * WS[..., None]; accw += WS; shifts[idx] = s
        used.append(dict(frame=idx, shift_deg=round(360 * s / TW, 1), dv_rows=dv, ncc=round(score, 3), inl=round(R["inl"], 3), rfree_mm=round(1e3 * R["rfree"], 1), length_mm=R["length_mm"], npx=R["npx"], wsum=round(float(R["W"].sum()))))
        ang = 2 * np.pi * s / TW; c1, c2_, cc = R["cap"]; caps.append((np.cos(ang) * c1 - np.sin(ang) * c2_, np.sin(ang) * c1 + np.cos(ang) * c2_, cc)); profs.append(R["prof"])
        keepf.append(dict(idx=idx, img=R["img"], m=R["m"], col=R["col"], W=R["W"], w=float(R["W"].sum())))
        if len(keepf) > 3: keepf.pop(int(np.argmin([k["w"] for k in keepf])))
    print(f"{args.obj}: {len(used)}/{len(frames)} frames used, skipped {why}")
    if not used: return
    def placed(st_):
        WS = np.roll(np.roll(st_["W"].astype(np.float32), st_["s"], axis=1), st_["dv"], axis=0); return np.roll(np.roll(st_["col"].astype(np.float32), st_["s"], axis=1), st_["dv"], axis=0) * WS[..., None], WS
    refine_log = []
    for it in range(args.refine):  # leave-one-out: take the frame out of the map, re-register it against the rest, put it back
        moves = []
        for i, st_ in enumerate(store):
            CW, WS = placed(st_); acc -= CW; accw -= WS; accw = np.maximum(accw, 0)
            s, dv, sc = register(dict(col=st_["col"].astype(np.float32), W=st_["W"].astype(np.float32)), acc, accw, st_["s"])
            if sc >= 0.45: moves.append(abs((s - st_["s"] + TW // 2) % TW - TW // 2) * 360 / TW); st_["s"], st_["dv"] = s, dv; used[i]["shift_deg"] = round(360 * s / TW, 1); used[i]["dv_rows"] = dv; used[i]["ncc"] = round(sc, 3)
            CW, WS = placed(st_); acc += CW; accw += WS
        refine_log.append(dict(median_move_deg=round(float(np.median(moves)), 2), p90_move_deg=round(float(np.percentile(moves, 90)), 2), max_move_deg=round(float(np.max(moves)), 2), n=len(moves))); print("refine", it, refine_log[-1])
    tex = acc / np.maximum(accw, 1e-6)[..., None]; cov = accw > 0.05; coverage = float(cov[BODY_ROWS].mean())
    resid = []  # consistency: re-register every used frame against the final map (full search for the rolling bottle)
    for st_ in store:
        s, dv, sc = register(dict(col=st_["col"].astype(np.float32), W=st_["W"].astype(np.float32)), acc, accw, st_["s"]); resid.append(round(360 * ((s - st_["s"] + TW // 2) % TW - TW // 2) / TW, 1))
    resid = np.array(resid); half = [np.zeros((TH, TW, 3), np.float32) for _ in range(2)]; halfw = [np.zeros((TH, TW), np.float32) for _ in range(2)]
    for i, st_ in enumerate(store):  # odd / even half maps, registered against each other
        WS = np.roll(np.roll(st_["W"].astype(np.float32), st_["s"], axis=1), st_["dv"], axis=0); half[i % 2] += np.roll(np.roll(st_["col"].astype(np.float32), st_["s"], axis=1), st_["dv"], axis=0) * WS[..., None]; halfw[i % 2] += WS
    hs, hdv, hsc = register(dict(col=half[1] / np.maximum(halfw[1], 1e-6)[..., None], W=np.minimum(halfw[1], 1.0)), half[0], halfw[0], 0) if halfw[1].sum() > 0 else (0, 0, 0)
    halves = dict(offset_deg=round(360 * ((hs + TW // 2) % TW - TW // 2) / TW, 1), dv_rows=int(hdv), ncc=round(float(hsc), 3))
    tex8 = np.clip(tex, 0, 255).astype(np.uint8); miss = (~cov).astype(np.uint8)
    filled = cv2.inpaint(np.concatenate([tex8] * 3, 1), np.concatenate([miss] * 3, 1), 7, cv2.INPAINT_TELEA)[:, TW:2 * TW]
    ab = slice(0, BODY_ROWS.start); topc = cov[ab] if cov[ab].sum() > 2000 else cov[BODY_ROWS][:6]; src = tex8[ab] if cov[ab].sum() > 2000 else tex8[BODY_ROWS][:6]
    topcol = np.median(src[topc], 0).astype(np.uint8) if topc.any() else np.array([200, 200, 200], np.uint8)
    if BODY_ROWS.start > 0: filled[ab] = np.where(cov[ab][..., None], filled[ab], topcol)
    out = OUT / args.obj; out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "unwrap.png"), filled); cv2.imwrite(str(out / "unwrap_raw.png"), tex8); cv2.imwrite(str(out / "weight.png"), np.clip(accw * 40, 0, 255).astype(np.uint8))
    CP = 256; capimg = np.zeros((CP, CP, 3), np.float32); capw = np.zeros((CP, CP), np.float32)
    for xs, ys, cc in caps:
        ix = np.clip(((xs / r + 1) / 2 * CP).astype(int), 0, CP - 1); iy = np.clip(((1 - (ys / r + 1) / 2) * CP).astype(int), 0, CP - 1)
        np.add.at(capimg, (iy, ix), cc.astype(np.float32)); np.add.at(capw, (iy, ix), 1)
    capv = capw > 0; cap8 = np.clip(capimg / np.maximum(capw, 1)[..., None], 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[:CP, :CP]; disc = np.hypot(xx - CP / 2 + 0.5, yy - CP / 2 + 0.5) < CP / 2
    cap8 = cv2.inpaint(cap8, ((~capv) & disc).astype(np.uint8), 9, cv2.INPAINT_TELEA) if capv.any() else cap8; cap8[~disc] = 0; cv2.imwrite(str(out / "cap.png"), cap8)
    hh = np.concatenate([p[0] for p in profs]); rr = np.concatenate([p[1] for p in profs]); bins = np.arange(0, H + 0.005, 0.005); prof = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        sel = (hh >= b0) & (hh < b1); prof.append([round(1e3 * b0), int(sel.sum()), round(1e3 * float(np.percentile(rr[sel], 90)), 1) if sel.sum() > 30 else None])
    rep = dict(frames=used, n_candidates=len(frames), skipped=why, coverage_body=round(coverage, 3), coverage_all=round(float(cov.mean()), 3), cap_cover=round(float(capv[disc].mean()), 3),
               cap_median_bgr=np.median(cap8[disc & capv], 0).tolist() if (disc & capv).any() else None, rereg_resid_deg=dict(median_abs=round(float(np.median(np.abs(resid))), 1), p90_abs=round(float(np.percentile(np.abs(resid), 90)), 1), max_abs=round(float(np.abs(resid).max()), 1), n=len(resid)), halves=halves, refine=refine_log, profile_mm_h_count_r90=prof, body_m=BODY, r_m=r, H_m=H,
               rfree_mm_median=round(float(np.median([u["rfree_mm"] for u in used])), 1), inl_median=round(float(np.median([u["inl"] for u in used])), 3), length_mm_median=float(np.median([u["length_mm"] for u in used])))
    json.dump(rep, open(out / "frames.json", "w"), indent=1)
    crops = []
    for kf in sorted(keepf, key=lambda k: k["idx"]):
        ys, xs = np.where(kf["m"]); y0, y1, x0, x1 = max(ys.min() - 20, 0), min(ys.max() + 20, 480), max(xs.min() - 20, 0), min(xs.max() + 20, 848)
        cr = kf["img"][y0:y1, x0:x1]; cr = cv2.resize(cr, (max(int(cr.shape[1] * 256 / cr.shape[0]), 1), 256)); cv2.putText(cr, str(kf["idx"]), (5, 25), 0, 0.8, (0, 255, 0), 2); crops.append(cr)
    row = np.hstack(crops + [np.zeros((256, 8, 3), np.uint8), render_preview(filled, cap8)]); wmax = max(row.shape[1], TW)
    cv2.imwrite(str(out / "check.jpg"), np.vstack([np.pad(filled, ((0, 0), (0, wmax - TW), (0, 0))), np.pad(row, ((0, 0), (0, wmax - row.shape[1]), (0, 0)))]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(json.dumps({k: rep[k] for k in ("coverage_body", "coverage_all", "cap_cover", "rereg_resid_deg", "halves", "rfree_mm_median", "inl_median", "length_mm_median")}))

main()

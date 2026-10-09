"""Review videos of sampled TRAINING episodes with the labels the trainer consumes burned in.

Diverse v2, new blocks only (molmoact household / tabletop, DROID CLVR + RAIL, FMB multi-object), one
random episode per block: all cameras tiled, real time. Per frame: the reviewed parent segment
(subtask / quality / retention / mistake events), the reviewed one-action ATOM the prompt renders
(subtask, atom quality + provenance, hybrid speed), any atom-level pause / interruption / recovery
event, and the nearest retained 5 Hz anchor row exactly as select_actor_anchors hands it to the
buffer (subtask / quality / speed / mistake). Timeline strip: retained anchors green, rejected grey,
mistake spans red.

ReBot merged root (rebot_all-annotated-v1): top | wrist at 2x (every other frame at 30 fps), the
subtask window (meta/subtask_windows.json), segment quality (episode_metadata.parquet), hybrid speed
(speed_hybrid_v1.parquet = REBOT_SPEED_TABLE), mistakes.parquet spans, online_labels is_intervention.

    uv run python -m lerobot.annotation.review.render_review_videos [--seed 20260919]
    -> migration/annotation_review_2026-09-19/videos/*.mp4 + picks.json

2026-10-06: --rebot-root <annotated ReBot root> --out <dir> [--episodes E ...] renders those episodes of any ReBot root.

2026-09-25: --diverse-root <corpus root> --robots <group ...> [--n N] --out <dir> renders N random training
episodes, each from a different robot group (droid, droid_success, molmoact, rc_arx5, rc_ur5, ur7e, yam, fmb),
and adds the precision / contact atom (precision_atoms.jsonl, contact_atoms.jsonl) and the anchor's
precision / contact to the burned-in lines.
"""
import argparse, glob, json, os, random, subprocess, sys, textwrap
from pathlib import Path
import av, cv2, numpy as np, pandas as pd

from lerobot.datasets.diverse_actor_selection import open_federated_corpus, select_actor_anchors, _atom_at
from lerobot.annotation.vocab import phrase_for
from lerobot.rl.offline_dataset_utils import load_metadata_rows

PC = {"precision": {}, "contact": {}}  # episode_id -> precision / contact atoms (empty when the sidecars are absent)
QUALITY_SPANS = {}  # episode_id -> trainer-effective v2 quality spans


def pc_lines(eid, k):
    p, c = _atom_at(PC["precision"].get(eid, []), k), _atom_at(PC["contact"].get(eid, []), k)
    if p is None and c is None: return []
    return [(f"PRECISION {p['precision'] if p else '-'}   CONTACT {phrase_for(int(c['contact'])) if c else '-'}"
             f" ({c['contact_slug'] if c else '-'})   [{(c or p).get('provenance', '')}]", BLUE)]


def pc_anchor(r):
    if r.get("precision", -1) is None or r.get("precision", -1) < 0: return ""
    return f"   precision {r['precision']}   contact {r['contact']} ({phrase_for(int(r['contact']))})"


def effective_quality(eid, frame):
    """Return the trainer-facing frame grade: lowest overlapping critique, else exemplary 5, else 4."""
    rows = [r for r in QUALITY_SPANS.get(eid, []) if int(r["from_index"]) <= frame < int(r["to_index"])]
    return min((int(r["quality"]) for r in rows), default=4), rows

HERE = Path("migration/annotation_review_2026-09-19"); OUT = HERE / "videos"; OUT.mkdir(parents=True, exist_ok=True)
DIVERSE = Path("outputs/diverse_robot_dataset_v2"); REBOT = Path("outputs/rebot_all-annotated-v1")
WHITE, GREY, YELLOW, GREEN, RED, CYAN, ORANGE, BLUE = (255, 255, 255), (170, 170, 170), (80, 220, 255), (80, 230, 80), (60, 60, 255), (255, 220, 80), (0, 160, 255), (255, 120, 120)
LINE = 22; TILE_H = 420; MAX_LINES = 12; SCALE = 0.55


def put(img, text, y, color=WHITE, scale=SCALE):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def resize_h(img, h):
    return cv2.resize(img, (int(round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


class Writer:
    def __init__(self, path, fps):
        self.path, self.fps, self.p = str(path), fps, None

    def write(self, rgb):
        if self.p is None:
            self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{rgb.shape[1]}x{rgb.shape[0]}",
                                       "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", self.path], stdin=subprocess.PIPE)
        self.p.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()


def compose(tiles, lines, timeline, tile_h=TILE_H):
    """Text band (one row per line) + timeline strip + tiles side by side; tiles scaled to TILE_H, width padded to even."""
    body = np.concatenate([resize_h(t, tile_h) for t in tiles], axis=1)
    w = body.shape[1] + body.shape[1] % 2
    band = np.zeros((LINE * MAX_LINES + 10 + 14, w, 3), np.uint8)
    wrapped = [(part, color) for text, color in lines for part in textwrap.wrap(text, max(w // 10, 40), subsequent_indent="    ")]
    for j, (text, color) in enumerate(wrapped[:MAX_LINES]): put(band, text, 17 + LINE * j, color)
    # timeline: list of (x0_frac, x1_frac, color) + cursor at frac
    y0 = LINE * MAX_LINES + 4; segs, cursor = timeline
    band[y0:y0 + 12, :] = (40, 40, 40)
    for a, b, c in segs: band[y0:y0 + 12, int(a * (w - 1)):max(int(b * (w - 1)), int(a * (w - 1)) + 1)] = c
    x = int(cursor * (w - 1)); band[y0 - 2:y0 + 14, max(x - 1, 0):x + 2] = (255, 255, 255)
    padded = np.zeros((body.shape[0], w, 3), np.uint8); padded[:, :body.shape[1]] = body
    return np.concatenate([band, padded], axis=0)


def video_frames(path):
    with av.open(str(path)) as c:
        for f in c.decode(video=0): yield f.to_ndarray(format="rgb24")


def events_at(events, t, key0="start_s", key1="end_s"):
    return [e for e in events or [] if float(e[key0]) <= t < float(e[key1])]


def event_label(e):
    what = e.get("kind") or e.get("reason") or e.get("basis") or ""
    return f"{what}: {e.get('note', '')}" if e.get("note") else what


# ── diverse (common corpus: molmoact / droid) ──────────────────────────────────
def render_common(eid, corpus, sel_rows, all_anchors, atoms, speed_atoms):
    ep_dir = DIVERSE / "corpus" / "episodes" / eid; rec = json.loads((ep_dir / "episode.json").read_text()); ann = rec["annotations"]
    ts = np.load(ep_dir / "timestamp_s.npy"); n = len(ts); dur = float(ts[-1]) + 1.0 / rec["native_rate_hz"]; fps = float(rec["cameras"][0]["fps"])
    rows = sorted(sel_rows.get(eid, []), key=lambda r: r["anchor_frame"]); anchors = sorted(all_anchors.get(eid, []), key=lambda r: r["anchor_frame"])
    af = np.array([r["anchor_frame"] for r in rows], dtype=int)
    tl = [((a["anchor_frame"]) / n, (a["anchor_frame"] + 1) / n, (60, 160, 60) if a["retained"] else (110, 110, 110)) for a in anchors]
    for s in ann["segments"]:
        for e in s.get("mistake_events") or []: tl.append((float(e["start_s"]) / dur, float(e["end_s"]) / dur, (40, 40, 220)))
    name = f"diverse_{eid}"; w = Writer(OUT / f"{name}.mp4", fps)
    gens = [video_frames(ep_dir / c["path"]) for c in rec["cameras"]]
    for k, frames in enumerate(zip(*gens)):
        if k >= n: break
        t = float(ts[k]); seg = next((s for s in ann["segments"] if float(s["start_s"]) <= t < float(s["end_s"])), None)
        atom = _atom_at(atoms.get(eid, []), k); sp = _atom_at(speed_atoms.get(eid, []), k)
        frame_quality, quality_rows = effective_quality(eid, k)
        lines = [(f"{eid}  {rec['source']}/{rec['component']}  {rec['embodiment']}  outcome={ann.get('outcome')}  t={t:5.1f}s f{k}/{n}   TASK: {ann.get('task')}", WHITE)]
        if seg is None: lines.append(("SEGMENT: none (outside every reviewed interval)", GREY))
        elif seg["retention"] != "keep": lines.append((f"SEGMENT [{seg['start_s']:.1f}-{seg['end_s']:.1f}s]: REJECTED ({seg['retention_reason']})", GREY))
        elif "subtask" not in seg:
            lines.append((f"SEGMENT [{seg['start_s']:.1f}-{seg['end_s']:.1f}s]: retained bridge   ({seg['retention_reason']}); atom sidecar is authoritative", YELLOW))
        else: lines.append((f"SEGMENT [{seg['start_s']:.1f}-{seg['end_s']:.1f}s]: \"{seg['subtask']}\"   q{seg['quality']}   ({seg['retention_reason']})", YELLOW))
        if atom is None: lines.append(("ATOM: none", GREY))
        else:
            spd = f"speed {sp['speed']} ({sp.get('speed_source')}{', ' + ','.join(sp.get('speed_flags') or []) if sp.get('speed_flags') else ''})" if sp else "speed: none"
            lines.append((f"ATOM [{atom['start_s']:.1f}-{atom['end_s_exclusive']:.1f}s] #{atom['parent_interval_index']}.{atom['atom_index']}: \"{atom['subtask']}\"   legacy q{atom['quality']} ({atom['quality_provenance']})   {spd}   {atom['confidence']}", CYAN))
        detail = "; ".join(f"q{int(r['quality'])} {r.get('cause', '')} raw[{r.get('raw_from_index')},{r.get('raw_to_index')})" for r in quality_rows)
        lines.append((f"TRAINER-EFFECTIVE QUALITY q{frame_quality}" + (f"   {detail}" if detail else "   default (no v2 span)"), ORANGE if frame_quality < 4 else GREEN))
        lines += pc_lines(eid, k)
        mist = events_at(seg.get("mistake_events") if seg else [], t) + [e for e in events_at(atom.get("mistake_events") if atom else [], t) if e.get("provenance") != "parent"]
        for e in mist: lines.append((f"MISTAKE {event_label(e)}"[:240], RED))
        for key, col in (("pause_events", ORANGE), ("interruption_events", ORANGE), ("recovery_events", GREEN)):
            for e in events_at(atom.get(key) if atom else [], t): lines.append((f"{key[:-7].upper()} {event_label(e)}"[:240], col))
        j = int(np.searchsorted(af, k, side="right")) - 1
        if j < 0: lines.append(("TRAIN: no retained anchor yet", GREY))
        else:
            r = rows[j]; lines.append((f"TRAIN anchor f{r['anchor_frame']} ({r['anchor_s']:.1f}s): \"{r['subtask']}\"   q{r['quality']}   speed {r['speed']}{pc_anchor(r)}   mistake={r['mistake']}   [{r['action_layout']}]", GREEN if r["anchor_frame"] == k else WHITE))
        a = next((x for x in anchors if x["anchor_frame"] == k), None)
        if a is not None and not a["retained"]: lines.append((f"anchor f{k} REJECTED: {','.join(a.get('rejection_reasons') or [])}", GREY))
        w.write(compose(list(frames), lines, (tl, k / n)))
    w.close(); print("wrote", w.path, n, "frames", f"{len(rows)} retained anchors")


# ── diverse (FMB store) ────────────────────────────────────────────────────────
def render_fmb(eid, corpus, sel_rows, all_anchors, atoms, speed_atoms):
    rec = corpus.fmb._records[eid]; ep_dir = DIVERSE / "fmb" / "episodes" / eid; n = int(rec["frame_count"]); fps = float(rec["nominal_fps"])
    imgs = [np.load(ep_dir / f"{c}.npy", mmap_mode="r") for c in ("side_1_rgb", "side_2_rgb", "wrist_1_rgb")]
    depth = np.load(ep_dir / "wrist_1_depth_z16.npy", mmap_mode="r")
    rows = sorted(sel_rows.get(eid, []), key=lambda r: r["anchor_timestep"]); anchors = sorted(all_anchors.get(eid, []), key=lambda r: r["anchor_timestep"])
    af = np.array([r["anchor_timestep"] for r in rows], dtype=int)
    tl = [((a["anchor_timestep"]) / n, (a["anchor_timestep"] + 1) / n, (60, 160, 60) if a["retained"] else (110, 110, 110)) for a in anchors]
    for p in rec["primitive_intervals"]:
        for e in p.get("mistake_events") or []: tl.append((int(e["start_timestep"]) / n, int(e["end_timestep_exclusive"]) / n, (40, 40, 220)))
    name = f"diverse_fmb_{eid}"; w = Writer(OUT / f"{name}.mp4", fps); obj = rec["source"]["object"]
    for k in range(n):
        t = k / fps; p = next((p for p in rec["primitive_intervals"] if int(p["start_timestep"]) <= k < int(p["end_timestep_exclusive"])), None)
        atom = _atom_at(atoms.get(eid, []), k); sp = _atom_at(speed_atoms.get(eid, []), k)
        lines = [(f"{eid}  fmb/{rec.get('component')}  {obj}  t={t:5.1f}s f{k}/{n}", WHITE)]
        if p is None: lines.append(("PRIMITIVE: none", GREY))
        else: lines.append((f"PRIMITIVE [{p['start_timestep']}-{p['end_timestep_exclusive']}) {p['primitive']}: \"{p['normalized_description']}\"   q{p['quality']}   {p['classification']}   critic_eligible={p['critic_eligible']}", YELLOW))
        if atom is None: lines.append(("ATOM: none", GREY))
        else:
            spd = f"speed {sp['speed']} ({sp.get('speed_source')})" if sp else "speed: none"
            lines.append((f"ATOM [{atom['start_timestep']}-{atom['end_timestep_exclusive']}) #{atom['parent_interval_index']}.{atom['atom_index']}: \"{atom['subtask']}\"   q{atom['quality']} ({atom['quality_provenance']})   {spd}   {atom['confidence']}", CYAN))
        lines += pc_lines(eid, k)
        for e in (p.get("mistake_events") if p else []) or []:
            if int(e["start_timestep"]) <= k < int(e["end_timestep_exclusive"]): lines.append((f"MISTAKE {event_label(e)}"[:240], RED))
        for key, col in (("pause_events", ORANGE), ("interruption_events", ORANGE), ("recovery_events", GREEN), ("retry_events", ORANGE)):
            for e in (p.get(key) if p else []) or []:
                if int(e["start_timestep"]) <= k < int(e["end_timestep_exclusive"]): lines.append((f"{key[:-7].upper()} {event_label(e)}"[:240], col))
        j = int(np.searchsorted(af, k, side="right")) - 1
        if j < 0: lines.append(("TRAIN: no retained anchor yet", GREY))
        else:
            r = rows[j]; lines.append((f"TRAIN anchor ts{r['anchor_timestep']} ({r['anchor_s']:.1f}s): \"{r['subtask']}\"   q{r['quality']}   speed {r['speed']}{pc_anchor(r)}   mistake={r['mistake']}   [{r['action_layout']}]", GREEN if r["anchor_timestep"] == k else WHITE))
        a = next((x for x in anchors if x["anchor_timestep"] == k), None)
        if a is not None and not a["retained"]: lines.append((f"anchor ts{k} REJECTED: {','.join(a.get('rejection_reasons') or [])}", GREY))
        d = np.asarray(depth[k]).astype(np.float32); valid = (d > 0) & (d < 65535); dv = np.zeros_like(d)
        if valid.any(): lo, hi = np.percentile(d[valid], [2, 98]); dv[valid] = np.clip((d[valid] - lo) / max(hi - lo, 1), 0, 1)
        dimg = cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]; dimg[~valid] = 0
        tiles = [cv2.resize(np.asarray(im[k]), (512, 512), interpolation=cv2.INTER_NEAREST) for im in imgs] + [cv2.resize(dimg, (512, 512), interpolation=cv2.INTER_NEAREST)]
        w.write(compose(tiles, lines, (tl, k / n)))
    w.close(); print("wrote", w.path, n, "frames", f"{len(rows)} retained anchors")


# ── ReBot merged root ──────────────────────────────────────────────────────────
def render_rebot(ep, step=2):
    R = str(REBOT); eps = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{R}/meta/episodes/**/*.parquet", recursive=True)]).sort_values("episode_index")
    row = eps[eps.episode_index == ep].iloc[0]; off = int(row.dataset_from_index); n = int(row.length)
    segs = pd.read_parquet(f"{R}/meta/episode_metadata.parquet"); segs = segs[segs.episode_index == ep]
    mist = pd.read_parquet(f"{R}/meta/mistakes.parquet"); mist = mist[(mist.episode_index == ep) & mist.mistake]
    speed = pd.read_parquet(f"{R}/meta/speed_hybrid_v1.parquet"); speed = speed[speed.episode_index == ep]
    ol = pd.read_parquet(f"{R}/meta/online_labels.parquet") if os.path.exists(f"{R}/meta/online_labels.parquet") else pd.DataFrame(columns=["episode_index", "frame_index", "is_intervention"])  # external roots carry no online labels
    ol = ol[ol.episode_index == ep].set_index("frame_index").is_intervention
    windows = json.load(open(f"{R}/meta/subtask_windows.json"))["episodes"].get(str(ep), [])
    # 2026-10-04: rubric v2 roots (rebot_additions) also carry precision / contact per segment, quality spans and precision windows
    # q printed on the segment line = the grade the trainer reads (load_metadata_rows: split by quality_spans when present)
    pieces = pd.DataFrame(load_metadata_rows(R)[0]); pieces = pieces[pieces.episode_index == ep]
    extra = {k: (lambda t: t[t.episode_index == ep])(pd.read_parquet(f"{R}/meta/{k}.parquet")) for k in ("precision", "contact", "quality_spans", "precision_windows") if os.path.exists(f"{R}/meta/{k}.parquet")}
    prov = next(p for p in json.load(open(f"{R}/meta/provenance.json")) if p["episode_index"] == ep)
    block = prov.get("merged_from") or f"{prov.get('kind', '')} {prov.get('source_dataset', '')}".strip()
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/data/**/*.parquet", recursive=True))]).sort_values("index").set_index("index")
    st = np.stack(data.loc[off:off + n - 1, "observation.state"].values); task = pd.read_parquet(f"{R}/meta/tasks.parquet").index[int(data.loc[off, "task_index"])]

    def frames_of(cam):
        cam = {"top": "external_0", "wrist": "wrist_0"}[cam] if f"videos/observation.images.{cam}/chunk_index" not in row else cam  # training-mix roots name the cameras external_0 / wrist_0
        ci, fi = int(row[f"videos/observation.images.{cam}/chunk_index"]), int(row[f"videos/observation.images.{cam}/file_index"]); t0 = float(row[f"videos/observation.images.{cam}/from_timestamp"])
        with av.open(f"{R}/videos/observation.images.{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4") as c:
            s = c.streams.video[0]; c.seek(int(max(t0 - 0.5, 0) / float(s.time_base)), stream=s, backward=True); k = 0
            for f in c.decode(s):
                if float(f.pts * s.time_base) < t0 - 1 / 60: continue
                yield f.to_ndarray(format="rgb24"); k += 1
                if k == n: break

    tl = []
    for i, (_, s) in enumerate(segs.sort_values("segment_index").iterrows()): tl.append(((s.from_index - off) / n, (s.to_index - off) / n, (60, 140, 60) if i % 2 else (40, 100, 40)))
    for _, m in mist.iterrows(): tl.append(((m.from_index - off) / n, (m.to_index - off) / n, (40, 40, 220)))
    for _, q in extra.get("quality_spans", pd.DataFrame()).iterrows():
        if q.quality != 4: tl.append(((q.raw_from_index - off) / n, (q.raw_to_index - off) / n, (200, 60, 200) if q.quality <= 3 else (220, 220, 60)))
    iv = ol.reindex(range(n)).fillna(False).values.astype(bool)
    if iv.any():
        edges = np.flatnonzero(np.diff(np.concatenate([[0], iv.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]): tl.append((a / n, b / n, (220, 140, 40)))
    name = f"rebot_ep{ep:02d}_{block.replace('-annotated', '').replace('rebot_', '')}"; w = Writer(OUT / f"{name}.mp4", 30)
    for i, (a, b) in enumerate(zip(frames_of("top"), frames_of("wrist"))):
        if i % step: continue
        g = off + i; r = segs[(segs.from_index <= g) & (segs.to_index > g)]; sp = speed[(speed.from_index <= g) & (speed.to_index > g)]; mm = mist[(mist.from_index <= g) & (mist.to_index > g)]
        win = next((x for x in windows if x["from_index"] <= g < x["to_index"]), None)
        lines = [(f"rebot_all ep{ep} <- {block} ep{prov.get('merged_from_episode', prov.get('source_episode'))}   t={i/30:5.1f}s f{i}/{n}   grip {st[i,6]:.0f} pan {st[i,0]:.0f}   TASK: {task}", WHITE)]
        lines.append((f"SUBTASK window: \"{win['subtask']}\"" if win else "SUBTASK window: none", YELLOW))
        pq = pieces[(pieces.from_index <= g) & (pieces.to_index > g)]
        if len(r): lines.append((f"seg{int(r.iloc[0].segment_index)} [{int(r.iloc[0].from_index - off)}-{int(r.iloc[0].to_index - off)}): \"{r.iloc[0].subtask}\"   q{int(pq.iloc[0].quality)}   note: {r.iloc[0].note}"[:220], CYAN))
        else: lines.append(("segment: none", GREY))
        if len(sp): lines.append((f"SPEED {int(sp.iloc[0].speed)} (motion {int(sp.iloc[0].motion_speed)} @ {sp.iloc[0].motion_median_deg_s:.1f} deg/s, duration {int(sp.iloc[0].duration_speed)} @ {sp.iloc[0].duration_s:.1f}s)   {'; '.join(sp.iloc[0].speed_flags) if len(sp.iloc[0].speed_flags) else ''}"[:220], WHITE))
        else: lines.append(("SPEED: none", GREY))
        if "precision" in extra:
            pr = extra["precision"][(extra["precision"].from_index <= g) & (extra["precision"].to_index > g)]; co = extra["contact"][(extra["contact"].from_index <= g) & (extra["contact"].to_index > g)]
            pw = extra["precision_windows"][(extra["precision_windows"].from_index <= g) & (extra["precision_windows"].to_index > g)]
            win_txt = f"   WINDOW level {int(pw.iloc[0].precision)} commit f{int(pw.iloc[0].commit_index - off)}{' (COMMIT)' if abs(g - pw.iloc[0].commit_index) < 4 else ''}" if len(pw) else ""
            lines.append((f"PRECISION {int(pr.iloc[0].precision) if len(pr) else '-'}   CONTACT {co.iloc[0].contact_slug if len(co) else '-'}{win_txt}", WHITE))
            qs = extra["quality_spans"][(extra["quality_spans"].raw_from_index <= g) & (extra["quality_spans"].raw_to_index > g)]
            for _, q in qs.iterrows(): lines.append((f"SPAN q{int(q.quality)} {q.cause} ({q.confidence}): {q.note}"[:220], YELLOW if q.quality == 5 else ORANGE))
        for _, m in mm.iterrows(): lines.append((f"MISTAKE {m.mistake_type}: {m.note}"[:220], RED))
        if iv[i]: lines.append(("TELEOP INTERVENTION (is_intervention)", ORANGE))
        w.write(compose([a, b], lines, (tl, i / n), tile_h=480))
    w.close(); print("wrote", w.path, n, "frames")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20260919); ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--diverse-root", type=Path); ap.add_argument("--robots", nargs="*"); ap.add_argument("--n", type=int, default=5); ap.add_argument("--out", type=Path)
    ap.add_argument("--episode-ids", nargs="*", help="exact diverse episode ids to render")
    ap.add_argument("--rebot-root", type=Path); ap.add_argument("--episodes", type=int, nargs="*")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.rebot_root is not None: return render_rebot_root(args)
    if args.diverse_root is not None: return render_diverse_groups(args, rng)
    common = [json.loads(l) for l in open(DIVERSE / "corpus/episodes.jsonl")]; fmb = [json.loads(l) for l in open(DIVERSE / "fmb/episodes.jsonl")]
    holdout = set(json.load(open(DIVERSE / "holdout_episodes.json"))["episode_ids"])
    pools = {
        "molmoact_household": [r["episode_id"] for r in common if r["source"] == "molmoact" and r["component"] == "household" and r["episode_id"] not in holdout],
        "molmoact_tabletop": [r["episode_id"] for r in common if r["source"] == "molmoact" and r["component"] == "tabletop" and r["episode_id"] not in holdout],
        "droid_clvr_rail": [r["episode_id"] for r in common if r["source"] == "droid_success" and r["component"] in ("CLVR", "RAIL") and r["episode_id"] not in holdout],
        "fmb_multi_object": [r["episode_id"] for r in fmb if r.get("component") == "multi_object_manipulation"],
    }
    picks = {k: rng.choice(sorted(v)) for k, v in pools.items()}
    # rollouts: one autonomous (eps 0-16, rebot_inference_2026-09-17-v1) + one rollouts-v2 (eps 41-55); bottle: two of eps 70-81
    picks["rebot_rollout_autonomous"] = rng.choice(range(0, 17)); picks["rebot_rollout_v2"] = rng.choice(range(41, 56)); picks["rebot_bottle"] = rng.sample(range(70, 82), 2)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in pools.items()}), json.dumps(picks)); json.dump({"seed": args.seed, "pools": {k: len(v) for k, v in pools.items()}, "picks": picks}, open(HERE / "picks.json", "w"), indent=1)
    only = set(args.only) if args.only else None

    corpus = open_federated_corpus(DIVERSE); sel = select_actor_anchors(corpus, verify_counts=False); sel_rows = sel.rows_by_episode()
    all_anchors = {}
    for r in corpus.actor_anchors(split=None, retained_only=False): all_anchors.setdefault(r["episode_id"], []).append(r)
    atoms, speed_atoms = {}, {}
    for view, d in (("subtask_atoms", atoms), ("speed_atoms", speed_atoms)):
        for a in list(getattr(corpus.common, view)()) + list(getattr(corpus.fmb, view)()): d.setdefault(a["episode_id"], []).append(a)
    for k in ("molmoact_household", "molmoact_tabletop", "droid_clvr_rail"):
        if only is None or k in only: render_common(picks[k], corpus, sel_rows, all_anchors, atoms, speed_atoms)
    if only is None or "fmb_multi_object" in only: render_fmb(picks["fmb_multi_object"], corpus, sel_rows, all_anchors, atoms, speed_atoms)
    for k in ("rebot_rollout_autonomous", "rebot_rollout_v2"):
        if only is None or k in only: render_rebot(picks[k])
    if only is None or "rebot_bottle" in only:
        for ep in picks["rebot_bottle"]: render_rebot(ep)


def render_rebot_root(args):
    """The given episodes (default: all) of any annotated ReBot root, into --out."""
    global REBOT, OUT
    REBOT, OUT = args.rebot_root, args.out; OUT.mkdir(parents=True, exist_ok=True)
    eps = pd.read_parquet(REBOT / "meta/episodes/chunk-000/file-000.parquet").episode_index
    for ep in args.episodes or sorted(eps): render_rebot(int(ep))


def render_diverse_groups(args, rng):
    """One random non-holdout episode from each of N randomly chosen robot groups of --diverse-root."""
    global DIVERSE, OUT, MAX_LINES
    DIVERSE, MAX_LINES = args.diverse_root, 11
    OUT = args.out or HERE / "videos"; OUT.mkdir(parents=True, exist_ok=True)
    holdout = set(json.load(open(DIVERSE / "holdout_episodes.json"))["episode_ids"])
    group = lambda r: "rc_" + r["embodiment"].lower() if r["source"] == "robochallenge" else r["source"]  # noqa: E731
    pools, episodes = {}, {}
    for r in map(json.loads, open(DIVERSE / "corpus/episodes.jsonl")):
        episodes[r["episode_id"]] = ("common", r["episode_id"])
        if r["episode_id"] not in holdout: pools.setdefault(group(r), []).append(("common", r["episode_id"]))
    for r in map(json.loads, open(DIVERSE / "fmb/episodes.jsonl")):
        episodes[r["episode_id"]] = ("fmb", r["episode_id"])
        if r["episode_id"] not in holdout: pools.setdefault("fmb", []).append(("fmb", r["episode_id"]))
    if args.episode_ids:
        missing = sorted(set(args.episode_ids) - episodes.keys())
        if missing: raise ValueError(f"unknown diverse episode ids: {missing}")
        picks = {eid: episodes[eid] for eid in args.episode_ids}
        groups = []
    else:
        groups = args.robots or rng.sample(sorted(pools), args.n)
        picks = {g: rng.choice(sorted(pools[g])) for g in groups}
    print(json.dumps({g: len(pools[g]) for g in groups}), json.dumps(picks))
    json.dump({"seed": args.seed, "root": str(DIVERSE), "pools": {g: len(v) for g, v in pools.items()}, "picks": picks}, open(OUT / "picks.json", "w"), indent=1)
    corpus = open_federated_corpus(DIVERSE); sel_rows = select_actor_anchors(corpus, verify_counts=False).rows_by_episode()
    all_anchors = {}
    for r in corpus.actor_anchors(split=None, retained_only=False): all_anchors.setdefault(r["episode_id"], []).append(r)
    atoms, speed_atoms = {}, {}
    for view, d in (("subtask_atoms", atoms), ("speed_atoms", speed_atoms), ("precision_atoms", PC["precision"]), ("contact_atoms", PC["contact"])):
        for a in list(getattr(corpus.common, view)()) + list(getattr(corpus.fmb, view)()): d.setdefault(a["episode_id"], []).append(a)
    for store in (DIVERSE / "corpus", DIVERSE / "fmb"):
        path = store / "quality_spans.jsonl"
        if path.is_file():
            for row in map(json.loads, open(path)):
                QUALITY_SPANS.setdefault(row["episode_id"], []).append(row)
    for g, (kind, eid) in picks.items():
        (render_fmb if kind == "fmb" else render_common)(eid, corpus, sel_rows, all_anchors, atoms, speed_atoms)


if __name__ == "__main__":
    main()

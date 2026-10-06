"""Contact-sheet rows for the precision + contact pass (2026-09-25). Nothing is labelled.

One row per meta/episode_metadata.parquet row of each root, in root order (bits, pushbook,
rebot_all, additions, external, validation). move / return rows are derived (no sheet row). Every other row gets 4 tiles
(256x192) from the top and wrist cameras; 6 rows per sheet, sheets grouped by root and verb.

Frames (from = from_index, to = to_index, exclusive; every frame is clamped into [from, to-1]
and a clamped tile is flagged):
  grasp                      top to-15, wrist to-15 (commit), wrist to-45, wrist to-1
  release                    top onset, wrist onset (commit), wrist onset+8,
                             wrist open-event (depth_gripper_events.parquet) else wrist onset-10
  push press strike pry pour top from+15, wrist from+15 (commit), top to-15, wrist to-15
  drag fold wipe
  everything else            top to-1, wrist to-1, top to-30, wrist to-15 (commit)
Opening onset: first frame in the segment where the state gripper (0 shut, opens negative) is
more than 20 deg below its held value for 5 consecutive frames; held value = the more closed of
(median of the 15 frames before the segment, same episode) and (median of the segment's first 5
frames). The pre-segment median alone reproduces the proposal's 360/366 but misses two releases
that follow a grasp directly (the 15 frames before are the closing ramp): ep50 seg7, ep69 seg10.
No onset: from+15 and flag no_opening_in_segment.

Cameras: wrist = the key matching wrist|hand|gripper|arm_camera (own roots `wrist`, cache_ready roots
`wrist_0`); top = the first other key whose episode video is present (own `top`; cache_ready
`external_0`, falling back to `external_1`). A video resolving to placeholders/absent.mp4 (external_rebot_train:
43 episodes without wrist, 10 wrist only) counts as absent: black tile, flag wrist_absent / top_absent.

Contact prior: contact_prior.classify() (text rules).
Precision prior: no text-rule script exists; left null for the next step.

Usage (repo root): .venv/bin/python -m lerobot.annotation.precision_contact.render_rows [short ...]
Output: this dir/rows.jsonl, summary.txt, sheets/<root>/<verb>_<k>.jpg
"""
import glob, json, os, re, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from lerobot.annotation.paths import WORKSPACE  # noqa: E402

REPO = WORKSPACE
os.chdir(REPO)

import cv2, numpy as np, pandas as pd  # noqa: E402
from lerobot.annotation.precision_contact import sample_grasps as sg  # noqa: E402  approach_angle (FK)
from lerobot.annotation.precision_contact import sample_contacts as sc  # noqa: E402  grab_many

HERE = Path("migration/precision_contact_annotation_2026-09-25"); SHEETS = HERE / "sheets"
ROOTS = [("bits", "outputs/rebot_bits-annotated-v2"), ("pushbook", "outputs/rebot_push-book-annotated-v1"),
         ("rebot_all", "outputs/rebot_all-annotated-v1"),
         ("additions", "outputs/rebot_cache_ready_2026-09-21/main_additions_train"),
         ("external", "outputs/rebot_cache_ready_2026-09-21/external_rebot_train"),
         ("validation", "outputs/rebot_cache_ready_2026-09-21/validation_all")]
TILE_W, TILE_H = 256, 192; ROWS_PER_SHEET = 6
HEAD_H, STRIP_H = 22, 34
CONTACT_VERBS = {"push", "press", "strike", "pry", "pour", "drag", "fold", "wipe"}
DERIVED = {"move", "return"}
ONSET_DROP, ONSET_RUN, HOLD_WIN = 20.0, 5, 15

from lerobot.annotation.precision_contact.contact_prior import classify  # noqa: E402


def verb_of(sub):
    s = sub.lower().strip()
    if s.startswith("turn on") or s.startswith("turn off"): return "turn"
    return s.split()[0]


def layout_of(verb):
    if verb == "grasp": return "grasp"
    if verb == "release": return "release"
    if verb in CONTACT_VERBS: return "contact"
    return "end"


def onset(g, f0, f1, ep_start):
    """First frame in [f0, f1) where gripper < held - 20 for 5 consecutive frames; None if none."""
    pre = g[max(f0 - HOLD_WIN, ep_start):f0]; first = float(np.median(g[f0:f0 + 5]))
    held = max(float(np.median(pre)), first) if len(pre) else first  # the more closed (0 shut, open negative)
    below = g[f0:f1] < held - ONSET_DROP
    for t in range(len(below) - ONSET_RUN + 1):
        if below[t:t + ONSET_RUN].all(): return f0 + t, held
    return None, held


def load_root(short, root):
    root = Path(root); info = json.load(open(root / "meta/info.json")); fps = info["fps"]
    cams = [k for k in info["features"] if k.startswith("observation.images") and "depth" not in k]
    wrist = next((k for k in cams if re.search(r"wrist|hand|gripper|arm_camera", k)), None)
    tops = [k for k in cams if k != wrist]
    eps = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True)]).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f, columns=["index", "observation.state"]) for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))]).sort_values("index")
    assert (data["index"].values == np.arange(len(data))).all(), "non-contiguous global index"
    Q = np.stack(data["observation.state"].values).astype(float)
    ev_path = root / "meta/depth_gripper_events.parquet"
    ev = pd.read_parquet(ev_path) if ev_path.exists() else None
    meta = pd.read_parquet(root / "meta/episode_metadata.parquet")

    def video(ep_idx, cam):
        if cam is None: return None, 0.0
        e = eps.loc[ep_idx]
        vp = root / info["video_path"].format(video_key=cam, chunk_index=int(e[f"videos/{cam}/chunk_index"]), file_index=int(e[f"videos/{cam}/file_index"]))
        if not vp.exists() or "/placeholders/" in os.path.realpath(vp): return None, 0.0  # placeholders/absent.mp4
        return str(vp), float(e[f"videos/{cam}/from_timestamp"])

    def top_of(ep_idx):  # first external view present in this episode
        return next((k for k in tops if video(ep_idx, k)[0] is not None), tops[0] if tops else None)

    rows = []; n_onset = n_rel = 0; ang_cache = {}

    def ang(i):
        if i not in ang_cache: ang_cache[i] = round(sg.approach_angle(Q[i]), 1)
        return ang_cache[i]

    meta = meta.reset_index(drop=True)
    for k, s in meta.iterrows():
        ep_idx, seg = int(s.episode_index), int(s.segment_index); f0, f1 = int(s.from_index), int(s.to_index)
        e = eps.loc[ep_idx]; off, ep_end = int(e.dataset_from_index), int(e.dataset_to_index)
        assert off <= f0 < f1 <= ep_end, (short, ep_idx, seg, f0, f1)
        sub = str(s.subtask); verb = verb_of(sub); lay = None if verb in DERIVED else layout_of(verb)
        el, img_needed, rule = classify(sub)
        r = dict(row_id=f"{short}:ep{ep_idx}:seg{seg}", root=short, root_path=str(root), episode=ep_idx, segment_index=seg,
                 subtask=sub, verb=verb, from_index=f0, to_index=f1, to_exclusive=True, n_frames=f1 - f0, len_s=round((f1 - f0) / fps, 2),
                 derived=verb in DERIVED, layout=lay, sheet=None, sheet_row=None, tiles=[], wrist=None, approach_deg=None,
                 gripper_state=[], flags=[], contact_prior=el, contact_prior_rule=rule, contact_prior_image_needed=img_needed,
                 precision_prior=None, precision_prior_why="no text-rule script; filled in the next step")
        if verb in DERIVED:
            r.update(contact="na", precision_rule="next step minus 1, floor 1" if verb == "move" else "1 (return to home)")
            if verb == "move":
                nxt = meta[(meta.episode_index == ep_idx) & (meta.segment_index == seg + 1)]
                r["precision_from_row"] = f"{short}:ep{ep_idx}:seg{seg + 1}" if len(nxt) else None
                if not len(nxt): r["flags"].append("no_next_step")
            rows.append(r); continue

        last = f1 - 1; flags = []
        if f1 - f0 < 30: flags.append("short_segment")
        if lay == "grasp":
            spec = [("top", "to-15", f1 - 15), ("wrist", "to-15", f1 - 15), ("wrist", "to-45", f1 - 45), ("wrist", "to-1", f1 - 1)]; commit = 1
        elif lay == "release":
            n_rel += 1
            on, held = onset(Q[:, 6], f0, f1, off); r["held_gripper"] = round(held, 1)
            if on is None: flags.append("no_opening_in_segment"); on = f0 + 15; oname = "from+15"
            else: n_onset += 1; oname = "onset"
            r["onset_index"] = on if oname == "onset" else None
            evf = None
            if ev is not None:
                o = ev[(ev.event_type == "open") & (ev["index"] >= f0) & (ev["index"] < f1)]
                if len(o): evf = int(o["index"].min())
            r["open_event_index"] = evf
            spec = [("top", oname, on), ("wrist", oname, on), ("wrist", f"{oname}+8", on + 8),
                    ("wrist", "open-event", evf) if evf is not None else ("wrist", f"{oname}-10", on - 10)]; commit = 1
        elif lay == "contact":
            spec = [("top", "from+15", f0 + 15), ("wrist", "from+15", f0 + 15), ("top", "to-15", f1 - 15), ("wrist", "to-15", f1 - 15)]; commit = 1
        else:
            spec = [("top", "to-1", f1 - 1), ("wrist", "to-1", f1 - 1), ("top", "to-30", f1 - 30), ("wrist", "to-15", f1 - 15)]; commit = 3
        tiles, clamped = [], []
        for i, (cam, role, fr) in enumerate(spec):
            c = min(max(fr, f0), last)
            if c != fr: clamped.append(f"T{i + 1}:{role}")
            tiles.append(dict(tile=i + 1, role=role, camera=cam, frame=c, frame_requested=fr, clamped=c != fr, commit=i == commit,
                              local_frame=c - off, gripper=round(float(Q[c, 6]), 1), approach_deg=ang(c)))
        if clamped: flags.append("clamped_tile")
        top = top_of(ep_idx); wv, _ = video(ep_idx, wrist); tv, _ = video(ep_idx, top)
        if wv is None: flags.append("wrist_absent")
        if tv is None: flags.append("top_absent")
        r.update(tiles=tiles, clamped_tiles=clamped, wrist="present" if wv else "absent", approach_deg=tiles[commit]["approach_deg"],
                 gripper_state=[t["gripper"] for t in tiles], flags=flags,
                 _video={"top": video(ep_idx, top), "wrist": video(ep_idx, wrist)}, _fps=fps, _off=off)
        rows.append(r)
    return rows, dict(releases=n_rel, onset_found=n_onset)


# ---------------------------------------------------------------- rendering (workers)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, xy, scale=0.42, color=(255, 255, 255)):
    cv2.putText(img, text, xy, FONT, scale, color, 1, cv2.LINE_AA)


def fit(text, width, scale):
    while text and cv2.getTextSize(text, FONT, scale, 1)[0][0] > width - 6: text = text[:-1]
    return text


def render_row(r):
    by_cam = defaultdict(list)
    for t in r["tiles"]: by_cam[t["camera"]].append(t)
    imgs = {}
    for cam, ts in by_cam.items():
        vp, t0 = r["_video"][cam]
        if vp is None:
            for t in ts: imgs[t["tile"]] = None
            continue
        times = [t0 + (t["frame"] - r["_off"]) / r["_fps"] for t in ts]
        frames, sorted_t = sc.grab_many(vp, times, r["_fps"])
        lut = dict(zip(sorted_t, frames))
        for t, tm in zip(ts, times): imgs[t["tile"]] = lut[tm]
    W = 4 * TILE_W
    head = np.full((HEAD_H, W, 3), (40, 40, 90), np.uint8)
    flags = (" [" + ", ".join(r["flags"]) + "]") if r["flags"] else ""
    put(head, fit(f"{r['root']} ep{r['episode']} seg{r['segment_index']} | {r['subtask']} | f{r['from_index']}-{r['to_index']} ({r['len_s']}s){flags}", W, 0.5), (4, 16), 0.5)
    cols = []
    for t in r["tiles"]:
        img = imgs[t["tile"]]
        img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA) if img is not None else np.zeros((TILE_H, TILE_W, 3), np.uint8)
        if img is None or imgs[t["tile"]] is None: put(img, f"{t['camera']} absent", (70, 100), 0.6, (255, 80, 80))
        strip = np.zeros((STRIP_H, TILE_W, 3), np.uint8)
        if t["commit"]: strip[:] = (0, 70, 0)
        col = (255, 120, 120) if t["clamped"] else (255, 255, 255)
        put(strip, fit(f"T{t['tile']} {t['camera']} {t['role']}{' COMMIT' if t['commit'] else ''}", TILE_W, 0.42), (3, 13), 0.42, col)
        put(strip, fit(f"f{t['frame']}{' CLAMPED' if t['clamped'] else ''}  grip {int(round(t['gripper']))}  appr {t['approach_deg']:.0f}deg", TILE_W, 0.42), (3, 29), 0.42, col)
        cols.append(np.vstack([strip, img]))
    body = np.hstack([np.pad(c, ((0, 0), (0, 2), (0, 0))) if i < 3 else c for i, c in enumerate(cols)])
    body = cv2.resize(body, (W, body.shape[0])) if body.shape[1] != W else body
    return np.vstack([head, body])


def render_sheet(job):
    path, rows = job
    out = []
    for r in rows:
        out += [render_row(r), np.full((6, 4 * TILE_W, 3), 90, np.uint8)]
    img = np.vstack(out[:-1])
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return str(path), len(rows)


# ---------------------------------------------------------------- main
def main(only):
    rows_path = HERE / "rows.jsonl"
    all_rows, onset_stats = [], {}
    for short, root in ROOTS:
        rows, st = load_root(short, root); onset_stats[short] = st
        all_rows += rows; print(f"{short}: {len(rows)} rows, onset {st['onset_found']}/{st['releases']}", flush=True)
    jobs = []
    for short, _ in ROOTS:
        groups = defaultdict(list)
        for r in all_rows:
            if r["root"] == short and not r["derived"]: groups[r["verb"]].append(r)
        (SHEETS / short).mkdir(parents=True, exist_ok=True)
        for verb, rs in groups.items():
            for k in range(0, len(rs), ROWS_PER_SHEET):
                chunk = rs[k:k + ROWS_PER_SHEET]; path = SHEETS / short / f"{verb}_{k // ROWS_PER_SHEET + 1:03d}.jpg"
                for j, r in enumerate(chunk): r["sheet"], r["sheet_row"] = str(path.relative_to(HERE)), j + 1
                if short in only: jobs.append((path, chunk))
    with Pool(min(24, os.cpu_count())) as pool:
        for i, (p, n) in enumerate(pool.imap_unordered(render_sheet, jobs, chunksize=1)):
            if i % 20 == 0 or i == len(jobs) - 1: print(f"sheet {i + 1}/{len(jobs)} {p}", flush=True)

    with open(rows_path, "w") as f:
        for r in all_rows: f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")

    # ---------------- checks + summary
    bad = [(r["row_id"], t["tile"]) for r in all_rows for t in r["tiles"]
           if not (r["from_index"] <= t["frame"] < r["to_index"]) or (t["frame"] != t["frame_requested"] and not t["clamped"])]
    unflagged = [r["row_id"] for r in all_rows if any(t["clamped"] for t in r["tiles"]) and "clamped_tile" not in r["flags"]]
    L = ["# precision + contact rows, 2026-09-25 (nothing labelled)", ""]
    L.append(f"tile frames outside their segment: {len(bad)}; clamped tiles without the flag: {len(unflagged)}")
    L += ["", "| root | rows | derived | sheet rows | sheets | release onset found (proposal) |", "|---|---|---|---|---|---|"]
    prop = {"bits": "52/52", "pushbook": "no releases", "rebot_all": "360/366; +ep50 seg7, ep69 seg10 with the segment-start held value",
            "additions": "not given", "external": "259/276", "validation": "not given"}
    for short, _ in ROOTS:
        rs = [r for r in all_rows if r["root"] == short]; st = onset_stats[short]
        L.append(f"| {short} | {len(rs)} | {sum(r['derived'] for r in rs)} | {sum(not r['derived'] for r in rs)} | "
                 f"{len({r['sheet'] for r in rs if r['sheet']})} | {st['onset_found']}/{st['releases']} ({prop[short]}) |")
    L += ["", "| root | verb | rows | derived | sheets |", "|---|---|---|---|---|"]
    for (short, verb), n in sorted(Counter((r["root"], r["verb"]) for r in all_rows).items(), key=lambda x: ([s for s, _ in ROOTS].index(x[0][0]), x[0][1])):
        rs = [r for r in all_rows if r["root"] == short and r["verb"] == verb]
        L.append(f"| {short} | {verb} | {n} | {sum(r['derived'] for r in rs)} | {len({r['sheet'] for r in rs if r['sheet']})} |")
    L += ["", "| root | flag | rows |", "|---|---|---|"]
    for (short, fl), n in sorted(Counter((r["root"], fl) for r in all_rows for fl in r["flags"]).items()):
        L.append(f"| {short} | {fl} | {n} |")
    L += ["", "Clamped tiles by role: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(c.split(':', 1)[1] for r in all_rows for c in r.get('clamped_tiles', [])).items())),
          "No-onset rows: " + ", ".join(r["row_id"] for r in all_rows if "no_opening_in_segment" in r["flags"]),
          "Release rows with an open event in the segment: " + ", ".join(f"{s} {sum(1 for r in all_rows if r['root'] == s and r['verb'] == 'release' and r.get('open_event_index') is not None)}" for s, _ in ROOTS),
          "", "Contact prior: class_map.py classify(). Precision prior: none (no text-rule script); left null."]
    (HERE / "summary.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
    if bad or unflagged: sys.exit(f"CHECK FAILED: {bad[:5]} {unflagged[:5]}")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {s for s, _ in ROOTS})

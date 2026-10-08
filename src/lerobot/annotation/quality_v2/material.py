"""Pilot material for one side-by-side class: traces, coarse strips, commit-aligned strips, side-by-side sheets.

uv run python -m lerobot.annotation.quality_v2.material rebot_main/carry/sock ...   (sheet classes of units_classed.csv)
uv run python -m lerobot.annotation.quality_v2.material --work <work> [sheet_class ...]   (a labelled ReBot pass; default: all of
<work>/class_map.json; units from <work>/units_classed.csv, staging root and onset threshold from <work>/pass.json)
-> classes/<slug>/{units.jsonl, traces/<uid>.txt, coarse/<uid>.jpg, aligned/<uid>.jpg, sheets/aligned_NN.jpg, sheets/index.txt}
Strips are render_strips.strip (8 tiles, top over wrist). CPU only, 12 workers.
Commit per action (rubric 6): grasp = last close onset in the unit; release = first opening onset from 0.5 s before the unit;
steps with no single commit (WHOLE) = the whole unit on the aligned strip; everything else (carry, return, insert, press, ...)
= the unit end (end pose / seat). The sheet order is seeded per class slug. (Step 2 pilot sheets were built into pilot/.)
"""
import glob, json, os, sys, zlib
from multiprocessing import Pool
from pathlib import Path

from lerobot.annotation.quality_v2.commit_overrides import apply_commit_override, load_commit_overrides

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
REPO = WORKSPACE
os.chdir(REPO)
HERE = QUALITY_V2
_a = sys.argv; sys.argv = sys.argv[:1]
from lerobot.annotation.precision_contact import render_rows as rr  # noqa: E402
from lerobot.annotation.precision_contact import render_strips as rs  # noqa: E402
from lerobot.annotation.precision_contact import diverse_rows as dr  # noqa: E402
sys.argv = _a
import cv2, numpy as np, pandas as pd  # noqa: E402

REBOT = {"rebot_all": "outputs/rebot_all-annotated-v2", "additions": "outputs/rebot_cache_ready_2026-09-25/main_additions_train",
         "validation": "outputs/rebot_cache_ready_2026-09-25/validation_all", "bits_book": "outputs/rebot_bits-book-annotated-v1",
         "external": "outputs/rebot_cache_ready_2026-09-25/external_rebot_train"}
# bits_book: the gripper opens/closes on a small bit by only ~30 units, slowly; 25 over 0.67 s misses it
ONSET_THR = {"bits_book": 10.0,
             # rc_ur5 bare phone: the Robotiq closes on it to only ~18 %, under 25
             "rc_ur5/grasp/flat": 10.0, "rc_ur5/release/flat->container": 10.0,
             # rc_arx5 cup on the peg: the gripper opens by only 9-21 points
             "rc_arx5/release/cup_mug->fixture": 8.0}
SHORT = {"rebot_all": "rebot_all", "additions": "additions", "validation": "validation", "bits_book": "bits_book", "external": "external"}
STRIP_S, PRE_S, POST_S, PAD0, PAD1 = 6.0, 6.0, 1.5, 1.0, 2.0
OUT = HERE / "classes"
UNITS = HERE / "units_classed.csv"
WHOLE = {"cloth_work", "wipe_scrub", "pour", "stir", "drag_soft", "in_hand_rotate", "handshake"}
COMMIT_OVERRIDES = load_commit_overrides(HERE / "pilot_commit_overrides.json")


def runs(mask, min_len):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0])); return [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= min_len]


def onsets(g, n, thr=25.0):
    """frames where g rises (close) / falls (open) by thr over n frames; first frame of each run."""
    d = np.r_[g[n:] - g[:-n], np.zeros(n)]
    up, dn = d >= thr, d <= -thr
    step = np.r_[np.diff(g), 0]; tol = 0.02 * (np.nanmax(g) - np.nanmin(g) + 1e-9)

    def first_move(starts, sign):  # walk forward to the first frame the gripper actually moves
        return np.array([next((f for f in range(t, min(t + n, len(g) - 1)) if sign * step[f] > tol), t) for t in starts], int)
    return (first_move(np.flatnonzero(up & ~np.r_[False, up[:-1]]), 1), first_move(np.flatnonzero(dn & ~np.r_[False, dn[:-1]]), -1))


def commit_of(action, cl, op, f0, f1, fps):
    """(commit frame, kind, found)"""
    if action == "grasp":
        c = cl[(cl >= f0) & (cl < f1)]; return (int(c[-1]), "close", True) if len(c) else (f1 - int(0.5 * fps), "close", False)
    if action == "release":
        c = op[(op >= f0 - int(0.5 * fps)) & (op < f1)]; return (int(c[0]), "open", True) if len(c) else (f0 + int(0.5 * fps), "open", False)
    if action in WHOLE: return f1, "whole", True
    return f1, "end", True


RISE_CM, RISE_DEG, FLAT_CM, FLAT_DEG, FLAT_GRIP = 1.0, 5.0, 0.5, 2.0, 5.0  # rubric 5.13 starting values


def approach_line(Q, w0, w1, f0, commit, fps):
    """Rubric 5.13 on the window [w0, w1): d = distance (cm) of the gripper end link to its pose at the commit,
    turn = rotation (deg) still needed to reach the commit orientation, from the arm joints by forward kinematics.
    Flags from the farthest point (largest d between the unit start f0 and the commit, so leaving home does not count)
    to the commit: rises (d more than RISE_CM or turn more than RISE_DEG above its running minimum, for 0.5 s or more) and flats (over 1 s,
    d moves less than FLAT_CM, turn less than FLAT_DEG, gripper less than FLAT_GRIP)."""
    global FK
    if "FK" not in globals():
        from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
        FK = RebotKinematics()
    fr = FK.frames(Q[w0:w1, :6]); p, R = fr[:, -1, :3, 3], fr[:, -1, :3, :3]
    c = min(max(commit, w0), w1 - 1) - w0
    d = np.linalg.norm(p - p[c], axis=1) * 100
    turn = np.degrees(np.arccos(np.clip((np.einsum("tij,ij->t", R, R[c]) - 1) / 2, -1, 1)))
    k = max(1, int(fps / 3)); smooth = lambda x: np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[: len(x)]
    sm, st = smooth(d), smooth(turn)
    a, s = max(f0, w0) - w0, int(fps)
    a += int(np.argmax(sm[a:c + 1])) if c >= a else 0
    above = lambda x: x[a:c + 1] - np.minimum.accumulate(x[a:c + 1])
    rise = (above(sm) > RISE_CM) | (above(st) > RISE_DEG)
    g = Q[w0:w1, 6]; lag = lambda x: np.abs(x[a + s:c + 1] - x[a:c + 1 - s]) if c + 1 - s > a else np.zeros(0)
    flat = (lag(sm) < FLAT_CM) & (lag(st) < FLAT_DEG) & (lag(g) < FLAT_GRIP)
    rises = [[int(x) + a + w0, int(y) + a + w0] for x, y in runs(rise, max(1, int(fps / 2)))]
    flat_frames = np.zeros(c + 1 - a if c >= a else 0, bool)
    for x, y in runs(flat, 1): flat_frames[x:y + s] = True  # a flat 1 s window covers its whole second
    flats = [[int(x) + a + w0, int(y) + a + w0] for x, y in runs(flat_frames, 1)]
    return d, turn, rises, flats


def plot_approach(path, it, d, turn, rises, flats):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    w0, w1 = it["window"]; x = np.arange(w0, w1)
    fig, ax = plt.subplots(figsize=(14, 3.2)); ax2 = ax.twinx()
    for a, b in rises: ax.axvspan(a, b, color="tab:red", alpha=0.2)
    for a, b in flats: ax.axvspan(a, b, color="tab:orange", alpha=0.2)
    ax.plot(x, d, "k", lw=1.6, label="distance to commit pose (cm)"); ax2.plot(x, turn, "tab:blue", lw=1, label="turn to commit (deg)")
    for f, st in ((it["from_index"], "--"), (it["to_index"], "--"), (it["commit"], "-")): ax.axvline(f, color="grey", ls=st, lw=0.8)
    ax.set_xlabel("frame (red = rises again, orange = flat)"); ax.set_ylabel("cm"); ax2.set_ylabel("deg", color="tab:blue")
    ax.set_title(f"{it['uid']} | {it['subtask'][:70]} | commit f{it['commit']}", fontsize=9)
    ax.legend(loc="upper right", fontsize=8); ax2.legend(loc="upper center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def job(j):
    kind, out, r, g = j
    img = rs.strip(r, g) if kind == "one" else np.vstack([rs.strip(x, g) for x in r])
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def rebot_units(sel):
    items = []
    for ds, grp in sel.groupby("dataset"):
        root = REBOT[ds]; fps = json.load(open(f"{root}/meta/info.json"))["fps"]
        rows, _ = rr.load_root(SHORT[ds], root)
        h = {(r["episode"], r["segment_index"]): r for r in rows if "_video" in r}
        h_ep = {r["episode"]: r for r in rows if "_video" in r}  # move / return rows have no video; the paths are per episode
        data = pd.concat([pd.read_parquet(f, columns=["index", "episode_index", "observation.state"])
                          for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))]).sort_values("index")
        Q = np.stack(data["observation.state"].values).astype(float); ep = data.episode_index.values; g = Q[:, 6]
        v = np.r_[0, np.linalg.norm(np.diff(Q[:, :6], axis=0), axis=1) * fps]; v[np.r_[True, ep[1:] != ep[:-1]]] = 0
        dg = np.r_[0, np.abs(np.diff(g)) * fps]; still = (v < 1.0) & (dg < 1.0)
        vs = np.convolve(v, np.ones(15) / 15, mode="same")
        em = pd.read_parquet(f"{root}/meta/episode_metadata.parquet"); mk = pd.read_parquet(f"{root}/meta/mistakes.parquet")
        eps = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True)]).set_index("episode_index")
        cl, op = onsets(g, 20, ONSET_THR.get(ds, 25.0))
        for s in grp.itertuples():
            e, seg = int(s.episode), int(s.unit[3:]); m = em[(em.episode_index == e) & (em.segment_index == seg)].iloc[0]
            uid = f"{ds}_ep{e}_seg{seg}"
            f0, f1 = int(m.from_index), int(m.to_index); e0, e1 = int(eps.loc[e].dataset_from_index), int(eps.loc[e].dataset_to_index)
            w0, w1 = max(e0, f0 - int(PAD0 * fps)), min(e1, f1 + int(PAD1 * fps))
            commit, ckind, found = commit_of(s.action, cl, op, f0, f1, fps)
            commit, ckind, found = apply_commit_override(uid, f0, f1, (commit, ckind, found), COMMIT_OVERRIDES)
            segs = em[(em.episode_index == e) & (em.to_index > w0) & (em.from_index < w1)]
            mks = mk[(mk.episode_index == e) & (mk.to_index > w0) & (mk.from_index < w1)]
            items.append(dict(uid=uid, dataset=ds, episode=str(e), unit=s.unit, subtask=s.subtask, fps=fps,
                              action=s.action, sheet_class=s.sheet_class,
                              from_index=f0, to_index=f1, window=[w0, w1], episode_range=[e0, e1], commit=commit, commit_kind=ckind, commit_found=found,
                              v1_quality=int(m.quality), v1_note=str(m.note), contact=s.contact, precision_v1=s.precision,
                              close_onsets=[int(x) for x in cl[(cl >= w0) & (cl < w1)]], open_onsets=[int(x) for x in op[(op >= w0) & (op < w1)]],
                              still_runs=[[int(a) + w0, int(b) + w0] for a, b in runs(still[w0:w1], fps)],
                              v1_mistakes=[dict(from_index=int(x.from_index), to_index=int(x.to_index), type=x.mistake_type, note=x.note) for x in mks.itertuples()],
                              segments=[f"seg{int(t.segment_index)} f{int(t.from_index)}-{int(t.to_index)} q{int(t.quality)} {t.subtask}" for t in segs.itertuples()],
                              _h=h.get((e, seg), h_ep[e]), _g=g, _v=vs, _off=0, _Q=Q))
    return items


def diverse_episode_render_metadata(part, episode, episode_dir):
    """Build render metadata when an episode contains only derived atoms.

    diverse_rows.load_part intentionally omits camera and gripper payloads
    from move/return rows. Most derived rows can borrow that payload from a
    non-derived atom in the same episode, but derived-only episodes have no
    such row. Reconstruct the payload from the source episode using the same
    camera and gripper conventions as diverse_rows.
    """
    rate = float(episode["native_rate_hz"])
    if part == "fmb":
        raw_g = np.load(episode_dir / "actions.npy")[:, -1].astype(float)
        top = episode_dir / "side_1_rgb.npy"
        wrist = episode_dir / "wrist_1_rgb.npy"
        closes_up = True
    else:
        raw_g = np.load(episode_dir / "state.npy")[:, -1].astype(float)
        names = [camera if isinstance(camera, str) else camera["name"] for camera in episode["cameras"]]
        wrist_name = next((name for name in names if "wrist" in name), None)
        top_name = next((name for name in names if name != wrist_name), None)
        top = episode_dir / f"videos/{top_name}.mp4" if top_name else None
        wrist = episode_dir / f"videos/{wrist_name}.mp4" if wrist_name else None
        closes_up = episode["source"] != "robochallenge"
    return {
        "_video": {"top": (str(top), 0.0) if top else (None, 0.0),
                   "wrist": (str(wrist), 0.0) if wrist else (None, 0.0)},
        "_fps": rate,
        "_off": 0,
        "_g": dr.pct_closed(raw_g, closes_up),
    }


def diverse_units(sel):
    part = "fmb" if (sel.dataset == "fmb").all() else "corpus"; root = Path("outputs/diverse_robot_dataset_v3") / part
    rows = {r["row_id"]: r for r in dr.load_part(part)}
    # Derived diverse actions such as return have no dedicated render row.
    # Reuse the episode camera/gripper metadata from any non-derived atom.
    render_by_ep = {}
    for row in rows.values():
        if "_video" in row:
            render_by_ep.setdefault(row["episode"], row)
    atoms = {(a["episode_id"], f"p{a['parent_interval_index']}a{a['atom_index']}"): a
             for a in map(json.loads, open(root / "subtask_atoms.jsonl"))}
    episodes = {e["episode_id"]: e for e in map(json.loads, open(root / "episodes.jsonl"))}
    epdir = {eid: root / e.get("directory", f"episodes/{eid}") for eid, e in episodes.items()}
    fallback_render_by_ep = {}
    by_ep = {}
    for (eid, u), a in atoms.items(): by_ep.setdefault(eid, []).append(a)
    items = []
    for s in sel.itertuples():
        a = atoms[(s.episode, s.unit)]; r = rows[f"{dr.short_of(a)}:{s.episode}:{s.unit}"]; rate = float(a["native_rate_hz"])
        if part == "fmb":  # no state.npy: tcp_pose = xyz (m) + quaternion xyzw
            tp = np.load(epdir[s.episode] / "tcp_pose.npy").astype(float); xyz = tp[:, :3]
            rot = np.r_[0, 2 * np.arccos(np.clip(np.abs((tp[1:, 3:7] * tp[:-1, 3:7]).sum(1)), 0, 1)) * rate]  # rad/s
        else:
            st = np.load(epdir[s.episode] / "state.npy").astype(float); xyz = st[:, :3]
            rot = np.r_[0, np.abs(np.diff(np.unwrap(st[:, 3:6], axis=0), axis=0)).max(1) * rate]  # rad/s
        n = len(xyz)
        # Some diverse-corpus rows (for example, a DROID pour episode) have no
        # gripper channel at all rather than an explicit null value.
        h = r if "_video" in r else render_by_ep.get(s.episode)
        if h is None:
            if s.episode not in fallback_render_by_ep:
                fallback_render_by_ep[s.episode] = diverse_episode_render_metadata(
                    part, episodes[s.episode], epdir[s.episode])
            h = fallback_render_by_ep[s.episode]
        row_g = r.get("_g", h.get("_g"))
        g = row_g if row_g is not None else np.zeros(n)
        pos = np.r_[0, np.linalg.norm(np.diff(xyz, axis=0), axis=1) * rate]  # m/s
        still = (pos < 0.0087) & (rot < np.deg2rad(1.0)) & (np.r_[0, np.abs(np.diff(g)) * rate] < 1.0)
        k = max(1, int(rate / 2)); vs = np.convolve(pos * 1000, np.ones(k) / k, mode="same")  # mm/s
        f0, f1 = a["start_timestep"], a["end_timestep_exclusive"]; w0, w1 = max(0, f0 - int(PAD0 * rate)), min(n, f1 + int(PAD1 * rate))
        cl, op = onsets(g, max(2, int(round(0.67 * rate))), ONSET_THR.get(s.sheet_class, 25.0))
        uid = f"{s.episode.split('__', 1)[-1]}_{s.unit}"
        commit, ckind, found = commit_of(s.action, cl, op, f0, f1, rate)
        commit, ckind, found = apply_commit_override(uid, f0, f1, (commit, ckind, found), COMMIT_OVERRIDES)
        near = [b for b in sorted(by_ep[s.episode], key=lambda b: b["start_timestep"]) if b["end_timestep_exclusive"] > w0 and b["start_timestep"] < w1]
        items.append(dict(uid=uid, dataset=s.dataset, episode=s.episode, unit=s.unit, subtask=s.subtask,
                          fps=rate, action=s.action, sheet_class=s.sheet_class,
                          from_index=f0, to_index=f1, window=[w0, w1], episode_range=[0, n], commit=commit, commit_kind=ckind, commit_found=found,
                          v1_quality=a["quality"], v1_note=a["note"] + " || " + a["parent_note"], contact=s.contact, precision_v1=s.precision,
                          close_onsets=[int(x) for x in cl[(cl >= w0) & (cl < w1)]], open_onsets=[int(x) for x in op[(op >= w0) & (op < w1)]],
                          still_runs=[[int(x) + w0, int(y) + w0] for x, y in runs(still[w0:w1], int(rate))],
                          v1_mistakes=[dict(from_s=e["start_s"], to_s=e["end_s"], type=e.get("kind"), note=e.get("note", "")) for b in near for e in b["mistake_events"]],
                          v1_pauses=[dict(from_s=e["start_s"], to_s=e["end_s"]) for b in near for e in b["pause_events"]],
                          gaps=[dict(from_s=e["start_s"], to_s=e["end_s"]) for b in near for e in b["interruption_events"]],
                          segments=[f"p{b['parent_interval_index']}a{b['atom_index']} f{b['start_timestep']}-{b['end_timestep_exclusive']} q{b['quality']} {b['subtask']}" for b in near],
                          task=a.get("parent_subtask", ""), _h=h, _g=g, _v=vs, _off=0))
    return items


def build(cls):
    slug = cls.replace("/", "__").replace("->", "_to_"); out = OUT / slug; rng = np.random.default_rng(zlib.crc32(slug.encode()))
    for d in ("traces", "coarse", "aligned", "sheets"): (out / d).mkdir(parents=True, exist_ok=True)
    u = pd.read_csv(UNITS, dtype={"episode": str}); sel = u[u["sheet_class"] == cls]
    items = rebot_units(sel) if sel.half.iloc[0] == "rebot" else diverse_units(sel)
    jobs = []
    for it in items:
        h, fps, g, uid = it["_h"], it["fps"], it["_g"], it["uid"]
        base = {k: h[k] for k in ("_video", "_fps", "_off")}
        w0, w1 = it["window"]; unit_speed = "deg/s" if it["dataset"] in REBOT else "mm/s"
        # trace at 4 Hz
        tags = {}
        for f in it["close_onsets"]: tags.setdefault(f, []).append("CLOSE_START")
        for f in it["open_onsets"]: tags.setdefault(f, []).append("OPEN_START")
        tags.setdefault(it["from_index"], []).append("UNIT_START"); tags.setdefault(it["to_index"], []).append("UNIT_END")
        tags.setdefault(it["commit"], []).append(f"COMMIT_{it['commit_kind'].upper()}")
        L = [f"{uid} | {it['subtask']} | unit f{it['from_index']}-{it['to_index']} ({(it['to_index'] - it['from_index']) / fps:.1f} s) "
             f"| window f{w0}-{w1} | {fps} Hz | v1 q{it['v1_quality']} | contact {it['contact']} | precision v1 {it['precision_v1']} "
             f"| commit f{it['commit']} ({it['commit_kind']}{'' if it['commit_found'] else ', NOT FOUND: fallback'})",
             f"v1 note: {it['v1_note']}", "segments in window:"] + ["  " + x for x in it["segments"]]
        L += [f"v1 mistakes: {it['v1_mistakes']}", f"still runs (no motion >= 1 s): {it['still_runs']}"]
        if "v1_pauses" in it: L += [f"v1 pauses: {it['v1_pauses']}", f"excluded gaps: {it['gaps']}"]
        line = approach_line(it["_Q"], w0, w1, it["from_index"], it["commit"], fps) if "_Q" in it else None
        if line is not None:  # rubric 5.13
            (out / "approach").mkdir(exist_ok=True); plot_approach(out / "approach" / f"{uid}.png", it, *line)
            L.append(f"approach line (rubric 5.13, unit start to commit): rises again {line[2]} | flat {line[3]} | plot approach/{uid}.png")
        L.append(f"frame   t_from_commit_s  arm_speed_{unit_speed}  gripper" + ("  dist_cm  turn_deg" if line is not None else "") + "  tags")
        step = max(1, int(round(fps / 4)))
        for f in range(w0, w1, step):
            tg = [t for a, ts in tags.items() if f <= a < f + step for t in ts]
            tg += ["still"] if any(a <= f < b for a, b in it["still_runs"]) else []
            tg += ["RISE"] if line is not None and any(a <= f < b for a, b in line[2]) else []
            tg += ["FLAT"] if line is not None and any(a <= f < b for a, b in line[3]) else []
            dt = f" {line[0][f - w0]:8.1f} {line[1][f - w0]:9.1f}" if line is not None else ""
            L.append(f"{f:8d} {(f - it['commit']) / fps:+8.2f} {it['_v'][f]:12.1f} {g[f]:8.1f}{dt}  {' '.join(tg)}")
        (out / "traces" / f"{uid}.txt").write_text("\n".join(L) + "\n")
        # coarse strips over the window, 6 s each
        n = int(np.ceil((w1 - w0) / (STRIP_S * fps))); parts = []
        for k in range(n):
            a = w0 + int(k * STRIP_S * fps); b = min(w1, a + int(STRIP_S * fps))
            if b - a >= 8: parts.append(dict(base, row_id=f"{uid} [{k + 1}/{n}]", subtask=it["subtask"][:60], from_index=a, to_index=b, len_s=round((b - a) / fps, 1)))
        jobs.append(("many", str(out / "coarse" / f"{uid}.jpg"), parts, g))
        # commit-aligned strip: 6 s before the commit, 1.5 s after (the whole unit for steps with no single commit)
        c = it["commit"]; a, b = max(it["episode_range"][0], c - int(PRE_S * fps)), min(it["episode_range"][1], c + int(POST_S * fps))
        if it["commit_kind"] == "whole": a, b = it["from_index"], it["to_index"]
        jobs.append(("one", str(out / "aligned" / f"{uid}.jpg"),
                     dict(base, row_id=uid + ("" if it["commit_found"] else " NO-COMMIT-FOUND"), subtask=it["subtask"][:50], from_index=a, to_index=b,
                          len_s=round((b - a) / fps, 1)), g))
    if os.environ.get("META_ONLY"): jobs = []
    with Pool(12) as p:
        for k, _ in enumerate(p.imap_unordered(job, jobs, chunksize=2)):
            if k % 100 == 0: print(cls, k, "/", len(jobs), flush=True)
    order = [items[i]["uid"] for i in rng.permutation(len(items))]
    idx = []
    for k in range(0, len(order), 4):
        idx.append(f"aligned_{k // 4 + 1:02d}.jpg: " + " ".join(order[k:k + 4]))
        if os.environ.get("META_ONLY"): continue
        ims = [cv2.imread(str(out / "aligned" / f"{x}.jpg")) for x in order[k:k + 4]]
        wd = max(i.shape[1] for i in ims); ims = [cv2.copyMakeBorder(i, 0, 6, 0, wd - i.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in ims]
        p = out / "sheets" / f"aligned_{k // 4 + 1:02d}.jpg"; cv2.imwrite(str(p), np.vstack(ims), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "sheets" / "index.txt").write_text("\n".join(idx) + "\n")
    with open(out / "units.jsonl", "w") as fh:
        for it in items:
            e0, e1 = it["episode_range"]
            rec = {k: v for k, v in it.items() if not k.startswith("_")}
            rec["render"] = {k: it["_h"][k] for k in ("_video", "_fps", "_off")}
            rec["gripper_episode"] = [round(float(x), 1) for x in it["_g"][e0:e1]]
            fh.write(json.dumps(rec, default=str) + "\n")
    print(cls, len(items), "units ->", out)


if __name__ == "__main__":
    classes = sys.argv[1:]
    if classes[:1] == ["--work"]:  # a labelled ReBot pass: units, class list and pool settings from its results folder
        work = Path(classes[1]); config = json.loads((work / "pass.json").read_text()); name = config["dataset"]
        UNITS = work / "units_classed.csv"
        REBOT[name] = config["staging"]; SHORT[name] = name; ONSET_THR[name] = config["onset_thr"]
        classes = classes[2:] or list(json.loads((work / "class_map.json").read_text()))
    for c in classes: build(c)

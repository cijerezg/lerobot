"""Contact-sheet rows for the diverse half of the precision + contact pass (2026-09-25). Nothing is labelled.

One row per atom of outputs/diverse_robot_dataset_v3/{corpus,fmb}/subtask_atoms.jsonl, in file order.
move / return are derived (no sheet row). Sheets are drawn by the ReBot renderer
(precision_contact_annotation_2026-09-25/render_rows.py render_sheet); this file only builds the rows.

Differences from the ReBot rows:
  root        one short name per robot: droid, droid_success, molmoact, rc_arx5, rc_ur5, ur7e, yam, fmb
  offsets     the ReBot 30 Hz offsets kept in seconds: k frames -> round(k * rate / 30) native frames
              (to-15 = 0.5 s = to-8 at 15 Hz); tile roles print the native offset
  cameras     wrist = the camera named *wrist*; top = the first other camera in episodes.jsonl
              (droid left_external, molmoact primary, robochallenge global, ur7e realsense_topview,
              yam outside / top); FMB side_1_rgb / wrist_1_rgb arrays
  gripper     percent closed over the episode's range (100 = most closed in the episode); closing raises
              the value on droid / molmoact / ur7e / yam / fmb and lowers it on robochallenge (width)
  onset       release: first frame more than 20 % of the range below the held value for 0.17 s
              (5 frames at 30 Hz), held = the more closed of the 0.5 s before and the first 5 frames;
              none found: the steepest 0.17 s opening inside the atom (if > 5 points), else to-0.5 s; flagged
  angle       approach angle by FK (grasp_strategy_2026-09-24/diverse_angles.py angle_for); ARX5 / YAM nan
  layout      contact tiles also for scrub, stir, spread, water, unfold, flatten, straighten, tilt;
              contact / end-layout atoms over 4 s flagged long_segment (the fixed tiles miss the contact)
  FMB grasp   contact_fixed from the FK angle at the commit tile (< 45 top-pinch, >= 45 side-pinch);
              12 atoms spread over the angle range marked fk_check
  move        precision_from_row = the next atom of the episode (parent interval, atom order)

Usage (repo root): uv run python -m migration.precision_contact_diverse_2026-09-25.diverse_rows  (or the file path)
Output: this dir/rows.jsonl, summary.txt, sheets/<root>/<verb>_<k>.jpg. Arguments: root short names to render
(default all); --sample N renders only the first N sheets per (root, verb).
"""
import json, math, os, re, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from lerobot.annotation.paths import WORKSPACE  # noqa: E402

REPO = WORKSPACE
HERE = REPO / "migration/precision_contact_diverse_2026-09-25"  # results folder of the 09-25 pass
_argv = sys.argv; sys.argv = sys.argv[:1]
from lerobot.annotation.precision_contact import render_rows as rr  # noqa: E402  (chdir to REPO, render_sheet, classify, onset constants)
sys.argv = _argv
import numpy as np  # noqa: E402
from lerobot.annotation.precision_contact import sample_contacts as sc  # noqa: E402
from lerobot.annotation.precision_contact import diverse_angles as da  # noqa: E402
from lerobot.annotation.precision_contact.precision_prior import precision_prior  # noqa: E402

CORPUS = Path("outputs/diverse_robot_dataset_v3")
SHEETS = HERE / "sheets"
EXTRA_CONTACT = {"scrub", "stir", "spread", "water", "unfold", "flatten", "straighten", "tilt"}
DERIVED = {"move", "return"}
N_FK_CHECK = 12
LONG_S = 4.0  # contact / end-layout atoms longer than this get the 8-frame strip as well


# ---------------------------------------------------------------- FMB frames live in .npy arrays
_grab_video = sc.grab_many


def grab_many(path, ts, fps):
    if not str(path).endswith(".npy"): return _grab_video(path, ts, fps)
    arr = np.load(path, mmap_mode="r"); ts = sorted(set(ts))
    return [np.array(arr[min(int(round(t * fps)), len(arr) - 1)]) for t in ts], ts


sc.grab_many = grab_many


def short_of(a):
    if a["source"] == "robochallenge": return "rc_" + a["embodiment"].lower()
    return a["source"]


def layout_of(verb):
    return "contact" if verb in EXTRA_CONTACT else rr.layout_of(verb)


def native(k, rate):
    return int(round(k * rate / 30.0))


def pct_closed(g, closes_up):
    lo, hi = float(np.min(g)), float(np.max(g))
    if hi - lo < 1e-6: return None
    x = (g - lo) / (hi - lo)
    return 100.0 * (x if closes_up else 1.0 - x)


def onset(gp, f0, f1, rate):
    """First frame in [f0, f1) more than 20 points below the held value for 0.17 s; None if none."""
    run, pre_n = max(2, native(rr.ONSET_RUN, rate)), native(rr.HOLD_WIN, rate)
    pre = gp[max(f0 - pre_n, 0):f0]; first = float(np.median(gp[f0:f0 + 5]))
    held = max(float(np.median(pre)), first) if len(pre) else first
    below = gp[f0:f1] < held - 20.0
    for t in range(len(below) - run + 1):
        if below[t:t + run].all(): return f0 + t, held
    return None, held


def load_part(part):
    root = CORPUS / part
    atoms = [json.loads(l) for l in open(root / "subtask_atoms.jsonl")]
    eps = {e["episode_id"]: e for e in map(json.loads, open(root / "episodes.jsonl"))}
    by_ep = defaultdict(list)
    for a in atoms: by_ep[a["episode_id"]].append(a)
    for v in by_ep.values(): v.sort(key=lambda a: (a["parent_interval_index"], a["atom_index"]))
    grip, rows = {}, []

    def ep_dir(eid): return root / eps[eid].get("directory", f"episodes/{eid}")

    def gripper(eid, source):
        if eid not in grip:
            d = ep_dir(eid)
            if part == "fmb":  # FMB gripper width from the depth-gripper arrays is not in state; use actions[:, -1]
                g = np.load(d / "actions.npy")[:, -1].astype(float); up = True
            else:
                g = np.load(d / "state.npy")[:, -1].astype(float); up = source != "robochallenge"
            grip[eid] = pct_closed(g, up)
        return grip[eid]

    def cams(eid):
        if part == "fmb":
            d = ep_dir(eid); return str(d / "side_1_rgb.npy"), str(d / "wrist_1_rgb.npy")
        names = [c if isinstance(c, str) else c["name"] for c in eps[eid]["cameras"]]
        wrist = next((c for c in names if "wrist" in c), None); top = next((c for c in names if c != wrist), None)
        d = ep_dir(eid)
        return (str(d / f"videos/{top}.mp4") if top else None), (str(d / f"videos/{wrist}.mp4") if wrist else None)

    angle_cache = {}

    def angle(a, eid, frame):
        key = (eid, frame)
        if key not in angle_cache:
            rec = dict(root=str(root), _ep=eps[eid], episode=eid, commit=frame, source="fmb" if part == "fmb" else a["source"],
                       sub_source=a["source"], embodiment=a["embodiment"])
            try: angle_cache[key] = round(da.angle_for(rec), 1)
            except Exception: angle_cache[key] = float("nan")
        return angle_cache[key]

    for eid, seq in by_ep.items():
        for i, a in enumerate(seq):
            short = short_of(a); rate = float(a["native_rate_hz"]); f0, f1 = int(a["start_timestep"]), int(a["end_timestep_exclusive"])
            sub = a["subtask"]; verb = a["verb"]; seg = f"p{a['parent_interval_index']}a{a['atom_index']}"
            el, img_needed, rule = rr.classify(sub.lower())
            lay = None if verb in DERIVED else layout_of(verb)
            r = dict(row_id=f"{short}:{eid}:{seg}", part=part, root=short, root_path=str(root), episode=eid, segment_index=seg,
                     parent_interval_index=a["parent_interval_index"], atom_index=a["atom_index"], embodiment=a["embodiment"],
                     subtask=sub, verb=verb, object=a.get("object"), container=a.get("container"), rate_hz=rate,
                     from_index=f0, to_index=f1, to_exclusive=True, n_frames=f1 - f0, len_s=round((f1 - f0) / rate, 2),
                     derived=verb in DERIVED, layout=lay, sheet=None, sheet_row=None, tiles=[], wrist=None, approach_deg=None,
                     gripper_state=[], flags=[], contact_prior=el, contact_prior_rule=rule, contact_prior_image_needed=img_needed)
            r["precision_prior"], r["precision_prior_why"] = precision_prior(sub, verb, "fmb" if part == "fmb" else "diverse")
            if verb in DERIVED:
                r.update(contact="na", precision_rule="next step minus 1, floor 1" if verb == "move" else "1 (return to home)")
                if verb == "move":
                    nxt = seq[i + 1] if i + 1 < len(seq) else None
                    r["precision_from_row"] = f"{short}:{eid}:p{nxt['parent_interval_index']}a{nxt['atom_index']}" if nxt else None
                    if nxt is None: r["flags"].append("no_next_step")
                    elif nxt["verb"] in DERIVED: r["flags"].append("next_step_derived")
                rows.append(r); continue

            gp = gripper(eid, a["source"]); last = f1 - 1; flags = []
            if f1 - f0 < native(30, rate): flags.append("short_segment")
            if lay in ("contact", "end") and f1 - f0 > LONG_S * rate: flags.append("long_segment")  # 8-frame strip pass
            n = lambda k: native(k, rate)  # noqa: E731
            if lay == "grasp":
                spec = [("top", f"to-{n(15)}", f1 - n(15)), ("wrist", f"to-{n(15)}", f1 - n(15)), ("wrist", f"to-{n(45)}", f1 - n(45)), ("wrist", "to-1", f1 - 1)]; commit = 1
            elif lay == "release":
                if gp is None: on = None; held = None; flags.append("gripper_flat")
                else: on, held = onset(gp, f0, f1, rate)
                r["held_gripper"] = None if held is None else round(held, 1)
                if on is None:  # diverse release atoms end at the opening; from+15 would show the carry
                    flags.append("no_opening_in_segment"); w = max(2, n(rr.ONSET_RUN))
                    drop = gp[f0:f1 - w] - gp[f0 + w:f1] if gp is not None and f1 - f0 > w else np.zeros(0)
                    if len(drop) and drop.max() > 5.0: on = f0 + int(np.argmax(drop)); oname = "steepest"
                    else: on = f1 - n(15); oname = f"to-{n(15)}"
                else: oname = "onset"
                r["onset_index"] = on if oname == "onset" else None; r["open_event_index"] = None
                spec = [("top", oname, on), ("wrist", oname, on), ("wrist", f"{oname}+{n(8)}", on + n(8)), ("wrist", f"{oname}-{n(10)}", on - n(10))]; commit = 1
            elif lay == "contact":
                spec = [("top", f"from+{n(15)}", f0 + n(15)), ("wrist", f"from+{n(15)}", f0 + n(15)), ("top", f"to-{n(15)}", f1 - n(15)), ("wrist", f"to-{n(15)}", f1 - n(15))]; commit = 1
            else:
                spec = [("top", "to-1", f1 - 1), ("wrist", "to-1", f1 - 1), ("top", f"to-{n(30)}", f1 - n(30)), ("wrist", f"to-{n(15)}", f1 - n(15))]; commit = 3
            tiles, clamped = [], []
            for k, (cam, role, fr) in enumerate(spec):
                c = min(max(fr, f0), last)
                if c != fr: clamped.append(f"T{k + 1}:{role}")
                tiles.append(dict(tile=k + 1, role=role, camera=cam, frame=c, frame_requested=fr, clamped=c != fr, commit=k == commit,
                                  local_frame=c, gripper=None if gp is None else round(float(gp[c]), 1), approach_deg=angle(a, eid, c)))
            if clamped: flags.append("clamped_tile")
            top, wrist = cams(eid)
            if wrist is None: flags.append("wrist_absent")
            if top is None: flags.append("top_absent")
            r.update(tiles=tiles, clamped_tiles=clamped, wrist="present" if wrist else "absent", approach_deg=tiles[commit]["approach_deg"],
                     gripper_state=[t["gripper"] for t in tiles], flags=flags,
                     _video={"top": (top, 0.0), "wrist": (wrist, 0.0)}, _fps=rate, _off=0, _g=gp)
            if part == "fmb" and verb == "grasp":
                ang = r["approach_deg"]
                r["contact_fixed"] = "top-pinch" if ang < 45 else "side-pinch"
                r["contact_fixed_rule"] = f"FMB FK approach angle {ang:.1f} deg at the commit tile (< 45 top-pinch, >= 45 side-pinch)"
            rows.append(r)
    return rows


def mark_fk_check(rows):
    """12 FMB grasps nearest to 12 angles evenly spaced over the observed range."""
    g = [r for r in rows if r["root"] == "fmb" and r["verb"] == "grasp"]
    lo, hi = min(r["approach_deg"] for r in g), max(r["approach_deg"] for r in g)
    for target in np.linspace(lo, hi, N_FK_CHECK):
        min((r for r in g if not r.get("fk_check")), key=lambda r: abs(r["approach_deg"] - target))["fk_check"] = True


def clean(v):
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list): return [clean(x) for x in v]
    return v


def gripper_label(r):
    """render_row prints int(round(gripper)); give it 0 where the episode gripper is flat."""
    for t in r["tiles"]:
        if t["gripper"] is None: t["gripper"] = 0.0
    return r


STRIPS = HERE / "review/second_pass"


def strip_one(r):
    import cv2
    sys.argv, _a = sys.argv[:1], sys.argv
    from lerobot.annotation.precision_contact import render_strips as rs
    sys.argv = _a
    g = r["_g"] if r["_g"] is not None else np.zeros(r["to_index"] + 1)
    p = STRIPS / f"{r['row_id'].replace(':', '_')}.jpg"
    cv2.imwrite(str(p), cv2.cvtColor(rs.strip(r, g), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(p)


def strips(ids_file):
    """8-frame strips (render_strips.strip) for the row ids listed one per line in ids_file."""
    want = {l.strip() for l in open(ids_file) if l.strip()}
    STRIPS.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_part("corpus") + load_part("fmb") if r["row_id"] in want]
    missing = want - {r["row_id"] for r in rows}
    if missing: sys.exit(f"unknown or derived row ids: {sorted(missing)[:5]}")
    with Pool(min(24, os.cpu_count())) as pool:
        n = sum(1 for _ in pool.imap_unordered(strip_one, rows, chunksize=1))
    print(f"{n} strips in {STRIPS}")


def main():
    args = sys.argv[1:]; sample = None
    if "--strips" in args: return strips(args[args.index("--strips") + 1])
    if "--sample" in args:
        k = args.index("--sample"); sample = int(args[k + 1]); del args[k:k + 2]
    rows = load_part("corpus") + load_part("fmb"); mark_fk_check(rows)
    roots = list(dict.fromkeys(r["root"] for r in rows)); only = set(args) or set(roots)
    jobs = []
    for short in roots:
        groups = defaultdict(list)
        for r in rows:
            if r["root"] == short and not r["derived"]: groups[r["verb"]].append(r)
        (SHEETS / short).mkdir(parents=True, exist_ok=True)
        for verb, rs in groups.items():
            vname = verb.replace(" ", "-")
            for k in range(0, len(rs), rr.ROWS_PER_SHEET):
                chunk = rs[k:k + rr.ROWS_PER_SHEET]; path = SHEETS / short / f"{vname}_{k // rr.ROWS_PER_SHEET + 1:03d}.jpg"
                for j, r in enumerate(chunk): r["sheet"], r["sheet_row"] = str(path.relative_to(HERE)), j + 1
                if short in only and (sample is None or k // rr.ROWS_PER_SHEET < sample):
                    jobs.append((path, [gripper_label(dict(x, tiles=[dict(t) for t in x["tiles"]])) for x in chunk]))
    with Pool(min(24, os.cpu_count())) as pool:
        for i, (p, n) in enumerate(pool.imap_unordered(rr.render_sheet, jobs, chunksize=1)):
            if i % 50 == 0 or i == len(jobs) - 1: print(f"sheet {i + 1}/{len(jobs)} {p}", flush=True)
    with open(HERE / "rows.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(clean({k: v for k, v in r.items() if not k.startswith("_")})) + "\n")

    bad = [(r["row_id"], t["tile"]) for r in rows for t in r["tiles"] if not (r["from_index"] <= t["frame"] < r["to_index"])]
    L = ["# diverse precision + contact rows, 2026-09-25 (nothing labelled)", "",
         f"tile frames outside their atom: {len(bad)}", "",
         "| root | atoms | derived | sheet rows | sheets | release onset found |", "|---|---|---|---|---|---|"]
    for short in roots:
        rs = [r for r in rows if r["root"] == short]; rel = [r for r in rs if r["verb"] == "release"]
        L.append(f"| {short} | {len(rs)} | {sum(r['derived'] for r in rs)} | {sum(not r['derived'] for r in rs)} | "
                 f"{len({r['sheet'] for r in rs if r['sheet']})} | {sum(r.get('onset_index') is not None for r in rel)}/{len(rel)} |")
    tot = [sum(1 for r in rows if not r["derived"]), len({r["sheet"] for r in rows if r["sheet"]})]
    L += [f"| total | {len(rows)} | {sum(r['derived'] for r in rows)} | {tot[0]} | {tot[1]} | |", "",
          "| root | verb | sheet rows |", "|---|---|---|"]
    for (short, verb), n in sorted(Counter((r["root"], r["verb"]) for r in rows if not r["derived"]).items(), key=lambda x: (roots.index(x[0][0]), -x[1])):
        L.append(f"| {short} | {verb} | {n} |")
    L += ["", "| root | flag | rows |", "|---|---|---|"]
    for (short, fl), n in sorted(Counter((r["root"], fl) for r in rows for fl in r["flags"]).items()):
        L.append(f"| {short} | {fl} | {n} |")
    fmb = [r for r in rows if r.get("contact_fixed")]
    L += ["", f"FMB grasp contact from FK: {dict(Counter(r['contact_fixed'] for r in fmb))}; fk_check atoms: "
          + ", ".join(f"{r['row_id']} ({r['approach_deg']})" for r in rows if r.get("fk_check"))]
    (HERE / "summary.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
    if bad: sys.exit(f"CHECK FAILED: {bad[:5]}")


if __name__ == "__main__":
    main()

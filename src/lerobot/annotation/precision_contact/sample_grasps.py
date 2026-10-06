"""Design evidence for the grasp-strategy channel: sample grasp segments per object family
from every source, render top|wrist contact sheets at the commit frame (to-15 on ReBot,
0.5 s before the atom end elsewhere) and compute the ReBot approach angle by FK.

Nothing is annotated. Output: sheets/<source>__<family>.jpg + grasps.jsonl + angles.txt.
"""
import glob, json, os, random, re, sys
from pathlib import Path

import av, cv2, numpy as np, pandas as pd

from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics  # noqa: E402

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
os.chdir(WORKSPACE)  # every path below is workspace-relative
OUT = Path("migration/grasp_strategy_2026-09-24"); SHEETS = OUT / "sheets"; SHEETS.mkdir(parents=True, exist_ok=True)
PER_FAMILY = 12; TILE = (320, 240); COLS = 3; SEED = 0
FK = RebotKinematics()

FAMILIES = [  # (family, regex over the object noun)
    ("sock", r"sock"), ("shirt", r"shirt"), ("cloth", r"cloth|towel|rag|shorts|tissue|wrapper|sheet|plush"),
    ("cup", r"\bcup\b|mug"), ("bottle", r"bottle|can\b|tin\b"), ("tape_roll", r"tape"),
    ("block", r"block|cube|die\b|disk|eraser|chess"), ("fruit", r"apple|banana|kiwi|mango|lemon|pear|star fruit|avocado|mangosteen|fig"),
    ("pen", r"\bpen\b|marker|pencil|crayon|bolt|screw"), ("flower", r"flower"), ("utensil", r"spoon|fork|knife"),
    ("lid_bowl_plate", r"\blid\b|bowl|plate"), ("bag_packet", r"bag|packet|package|sweet|paper ball|tea bag|bar\b"),
    ("handle", r"handle|watering"), ("electronics", r"phone|mouse|charg|microcontroller|motor|speaker|cable"),
    ("fmb_object", r"^the object$"), ("other", r"."),
]


def family_of(obj):
    o = obj.lower()
    for fam, rx in FAMILIES:
        if re.search(rx, o): return fam
    return "other"


def grab(video, t, fps):
    """Decode the frame at time t (s) from a video file."""
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t - 0.6, 0) / float(s.time_base)), stream=s, backward=True)
        last = None
        for f in c.decode(s):
            ft = float(f.pts * s.time_base)
            last = f
            if ft >= t - 0.5 / fps: break
        return last.to_ndarray(format="rgb24") if last is not None else np.zeros((240, 320, 3), np.uint8)


def approach_angle(q_deg):
    """Angle (deg) between the gripper pointing axis (end_link x) and straight down. 0 = top-down."""
    R = FK.frames(np.asarray(q_deg, float))[0, -1, :3, :3]
    return float(np.degrees(np.arccos(np.clip(-R[2, 0], -1, 1))))


# ---------------------------------------------------------------- ReBot-style roots
def rebot_root(root, source, segs):
    """segs: DataFrame with episode_index, from_index, to_index, subtask (global indices)."""
    root = Path(root)
    info = json.load(open(root / "meta/info.json")); fps = info["fps"]
    cams = [k for k in info["features"] if k.startswith("observation.images") and "depth" not in k]
    wrist = next((k for k in cams if re.search(r"wrist|hand|gripper|arm_camera", k)), None)
    ext = next((k for k in cams if k != wrist), None)
    eps = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True)]).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f, columns=["index", "observation.state"]) for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))]).set_index("index")
    out = []
    for _, s in segs.iterrows():
        ep = eps.loc[int(s.episode_index)]; off = int(ep.dataset_from_index)
        commit = int(s.to_index) - 15
        if commit < int(s.from_index): commit = int(s.from_index) + (int(s.to_index) - int(s.from_index)) // 2
        local = commit - off
        q = np.asarray(data.loc[commit, "observation.state"], float)
        rec = dict(source=source, root=str(root), episode=int(s.episode_index), from_index=int(s.from_index), to_index=int(s.to_index),
                   subtask=s.subtask, object=re.sub(r"^grasp (the |a |an )?", "", s.subtask), commit=commit,
                   approach_deg=approach_angle(q), wrist_roll=float(q[5]), gripper=float(q[6]), seg_len_s=(int(s.to_index) - int(s.from_index)) / fps)
        rec["family"] = family_of(rec["object"])
        frames = []
        for cam in (ext, wrist):
            if cam is None: frames.append(None); continue
            ci, fi = int(ep[f"videos/{cam}/chunk_index"]), int(ep[f"videos/{cam}/file_index"]); t0 = float(ep[f"videos/{cam}/from_timestamp"])
            vp = root / info["video_path"].format(video_key=cam, chunk_index=ci, file_index=fi)
            frames.append((str(vp), t0 + local / fps, fps) if vp.exists() else None)
            if not vp.exists(): rec["missing_video"] = True
        rec["_frames"] = frames; out.append(rec)
    return out


def rebot_tiles(rec):
    return [grab(*f) if f else np.zeros((TILE[1], TILE[0], 3), np.uint8) for f in rec["_frames"]]


def segs_from_metadata(root):
    m = pd.read_parquet(f"{root}/meta/episode_metadata.parquet")
    return m[m.subtask.str.lower().str.startswith("grasp")].copy()


def segs_from_windows(root):
    w = json.load(open(f"{root}/meta/subtask_windows.json"))["episodes"]
    rows = [dict(episode_index=int(e), from_index=s["from_index"], to_index=s["to_index"], subtask=s["subtask"]) for e, ss in w.items() for s in ss if s["subtask"].startswith("grasp")]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- diverse corpus
def diverse(root, source):
    root = Path(root); eps = {e["episode_id"]: e for e in map(json.loads, open(root / "episodes.jsonl"))}
    atoms = [a for a in map(json.loads, open(root / "subtask_atoms.jsonl")) if a.get("verb") == "grasp"]
    out = []
    for a in atoms:
        e = eps[a["episode_id"]]; rate = float(a["native_rate_hz"]); d = root / e.get("directory", f"episodes/{a['episode_id']}")
        commit = max(int(a["end_timestep_exclusive"]) - 1 - int(round(0.5 * rate)), int(a["start_timestep"]))
        rec = dict(source=source, root=str(root), episode=a["episode_id"], atom=a["atom_index"], from_index=a["start_timestep"], to_index=a["end_timestep_exclusive"],
                   subtask=a["subtask"], object=a.get("object") or a["subtask"], commit=commit, embodiment=a["embodiment"], sub_source=a["source"],
                   seg_len_s=(a["end_timestep_exclusive"] - a["start_timestep"]) / rate)
        rec["family"] = family_of(rec["object"]); rec["_ep"] = e; rec["_rate"] = rate; out.append(rec)
    return out


def diverse_tiles(rec):
    e = rec["_ep"]; d = Path(rec["root"]) / e.get("directory", f"episodes/{rec['episode']}"); rate = rec["_rate"]; c = rec["commit"]
    if rec["source"] == "fmb":
        return [np.load(d / "side_1_rgb.npy", mmap_mode="r")[c].copy(), np.load(d / "wrist_1_rgb.npy", mmap_mode="r")[c].copy()]
    cams = [x["name"] if isinstance(x, dict) else x for x in e["cameras"]]
    wrist = next((k for k in cams if re.search(r"wrist", k)), None); ext = next((k for k in cams if k != wrist), None)
    return [grab(d / f"videos/{k}.mp4", c / rate, rate) if k else np.zeros((TILE[1], TILE[0], 3), np.uint8) for k in (ext, wrist)]


# ---------------------------------------------------------------- sheets
def label(img, text):
    img = cv2.resize(img, TILE); strip = np.zeros((22, TILE[0], 3), np.uint8)
    cv2.putText(strip, text[:52], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def sheet(recs, path):
    cells = []
    for r in recs:
        a, b = r["tiles"]
        txt = f"{r['source']} ep{r['episode']} {r['object']}" if isinstance(r["episode"], int) else f"{r.get('embodiment','')} {r['episode'][:22]} {r['object']}"
        txt2 = f"approach {r['approach_deg']:.0f}deg roll {r['wrist_roll']:.0f} grip {r['gripper']:.0f}" if "approach_deg" in r else f"{r.get('sub_source','')} atom{r.get('atom','')} {r['seg_len_s']:.1f}s"
        cells.append(np.hstack([label(a, txt), label(b, txt2)]))
    w = cells[0].shape[1]; h = cells[0].shape[0]
    rows = [np.hstack(cells[i:i + COLS] + [np.zeros((h, w, 3), np.uint8)] * (COLS - len(cells[i:i + COLS]))) for i in range(0, len(cells), COLS)]
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def pick(recs, n, rng):
    """Up to n records, spreading across episodes first."""
    recs = [r for r in recs if not r.get("missing_video")]; rng.shuffle(recs); seen = set(); first, rest = [], []
    for r in recs: (rest if r["episode"] in seen else first).append(r); seen.add(r["episode"])
    return (first + rest)[:n]


def main():
    rng = random.Random(SEED); all_recs = []
    # 1. own teleop
    all_recs += rebot_root("outputs/rebot_all-annotated-v1", "own", segs_from_metadata("outputs/rebot_all-annotated-v1"))
    # 2. model rollouts
    all_recs += rebot_root("outputs/rebot_rollouts-annotated-v2", "model", segs_from_metadata("outputs/rebot_rollouts-annotated-v2"))
    for r in ["outputs/rebot_inference_2026-09-20-v1", "outputs/rebot_inference_2026-09-17-v1"]:
        all_recs += rebot_root(r, "model", segs_from_windows(r))
    # 3. external ReBot operators
    for p in sorted(glob.glob("outputs/rebot_public_annotated_v1/*/*")):
        segs = segs_from_metadata(p)
        if len(segs): all_recs += rebot_root(p, "ext", segs)
    print("rebot-style records:", len(all_recs), "missing video:", sum(1 for r in all_recs if r.get("missing_video")), sorted({r["root"] for r in all_recs if r.get("missing_video")}), flush=True)
    # 4. diverse (tiles rendered lazily, only for the sampled ones)
    div = diverse("outputs/diverse_robot_dataset_v3/corpus", "diverse") + diverse("outputs/diverse_robot_dataset_v3/fmb", "fmb")
    print("diverse grasp atoms:", len(div), flush=True)

    groups = {}
    for r in all_recs + div: groups.setdefault((r["source"], r["family"]), []).append(r)
    with open(OUT / "grasps.jsonl", "w") as f:
        for r in all_recs + div: f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_") and k != "tiles"}) + "\n")
    for (src, fam), recs in sorted(groups.items()):
        chosen = pick(recs, PER_FAMILY, rng)
        if not chosen: print(f"{src:8s} {fam:15s} no videos ({len(recs)} grasps)"); continue
        for r in chosen:
            if "tiles" not in r: r["tiles"] = rebot_tiles(r) if "_frames" in r else diverse_tiles(r)
        sheet(chosen, SHEETS / f"{src}__{fam}.jpg"); print(f"{src:8s} {fam:15s} {len(chosen):3d}/{len(recs)}", flush=True)

    # FK angle summary for every ReBot grasp
    with open(OUT / "angles.txt", "w") as f:
        for (src, fam), recs in sorted(groups.items()):
            a = np.array([r["approach_deg"] for r in recs if "approach_deg" in r])
            if not len(a): continue
            h, _ = np.histogram(a, bins=[0, 15, 30, 45, 60, 75, 90, 180])
            f.write(f"{src:8s} {fam:15s} n={len(a):3d} median={np.median(a):5.1f} bins[0-15,15-30,30-45,45-60,60-75,75-90,90+]={h.tolist()}\n")
            for obj in sorted({r["object"] for r in recs}):
                b = np.array([r["approach_deg"] for r in recs if r["object"] == obj and "approach_deg" in r])
                h, _ = np.histogram(b, bins=[0, 15, 30, 45, 60, 75, 90, 180])
                f.write(f"    {obj:32s} n={len(b):3d} median={np.median(b):5.1f} {h.tolist()}\n")
    print(open(OUT / "angles.txt").read())


if __name__ == "__main__":
    main()

"""Design evidence for the contact-strategy vocabulary (2026-09-24). Nothing is annotated.

One row per sampled segment/atom: three external-camera frames and three wrist frames spread
over the step (fractions chosen per verb), labelled with source, episode, subtask and, on
ReBot roots, the FK approach angle at the last shown frame (0 = top-down, 90 = horizontal).
Rows are grouped into sheets by a candidate bucket so the unusual classes (push, pour, press,
fold, insert, release modes) can be read side by side.

Usage: uv run python -m lerobot.annotation.precision_contact.sample_contacts [spec_name]
Output: migration/contact_strategy_2026-09-24/sheets/<group>.jpg + sampled.jsonl
"""
import glob, json, random, re, sys
from pathlib import Path

import av, cv2, numpy as np, pandas as pd

from lerobot.annotation.precision_contact import sample_grasps as sg  # noqa: E402  (grab, approach_angle, FK)

HERE = Path("migration/contact_strategy_2026-09-24"); SHEETS = HERE / "sheets"; SHEETS.mkdir(parents=True, exist_ok=True)
TILE = (256, 192); SEED = 0
ANGLES = {}
for line in open("migration/grasp_strategy_2026-09-24/grasps.jsonl"):
    r = json.loads(line)
    if "approach_deg" in r: ANGLES[(r["root"], r["episode"], r["from_index"])] = r["approach_deg"]
for line in open("migration/grasp_strategy_2026-09-24/diverse_angles.jsonl"):
    r = json.loads(line)
    if "approach_deg" in r and r["approach_deg"] is not None: ANGLES[(r["root"], r["episode"], r["atom"])] = r["approach_deg"]

FRACS = {"grasp_rebot": [0.7, 0.9, 0.98], "grasp_div": [0.2, 0.6, 0.95], "release_rebot": [0.03, 0.35, 0.8],
         "release_div": [0.1, 0.5, 0.9], "other": [0.1, 0.5, 0.9]}


def grab_many(video, ts, fps):
    """Decode the frames nearest to each time in ts (s) with one seek."""
    out = {}
    ts = sorted(set(ts))
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        c.seek(int(max(ts[0] - 0.6, 0) / float(s.time_base)), stream=s, backward=True)
        i = 0; last = None
        for f in c.decode(s):
            ft = float(f.pts * s.time_base); last = f
            while i < len(ts) and ft >= ts[i] - 0.5 / fps:
                out[ts[i]] = f.to_ndarray(format="rgb24"); i += 1
            if i >= len(ts): break
        for t in ts[i:]: out[t] = last.to_ndarray(format="rgb24") if last is not None else np.zeros((TILE[1], TILE[0], 3), np.uint8)
    return [out[t] for t in ts], ts


# ---------------------------------------------------------------- ReBot-style roots
class RebotCtx:
    def __init__(self, root):
        self.root = Path(root); info = json.load(open(self.root / "meta/info.json")); self.fps = info["fps"]; self.info = info
        cams = [k for k in info["features"] if k.startswith("observation.images") and "depth" not in k]
        self.wrist = next((k for k in cams if re.search(r"wrist|hand|gripper|arm_camera", k)), None)
        self.ext = next((k for k in cams if k != self.wrist), None)
        if self.wrist is None and len(cams) > 1: self.wrist = next(k for k in cams if k != self.ext)  # two external views, no wrist
        self.eps = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True)]).set_index("episode_index")
        self.data = None
        mp = self.root / "meta/episode_metadata.parquet"
        if mp.exists():
            self.segs = pd.read_parquet(mp)
        else:
            w = json.load(open(self.root / "meta/subtask_windows.json"))["episodes"]
            self.segs = pd.DataFrame([dict(episode_index=int(e), from_index=s["from_index"], to_index=s["to_index"], subtask=s["subtask"]) for e, ss in w.items() for s in ss])
        self.segs["subtask"] = self.segs.subtask.str.lower()

    def state(self, gidx):
        if self.data is None:
            self.data = pd.concat([pd.read_parquet(f, columns=["index", "observation.state"]) for f in sorted(glob.glob(f"{self.root}/data/**/*.parquet", recursive=True))]).set_index("index")
        return np.asarray(self.data.loc[gidx, "observation.state"], float)

    def video(self, ep_idx, cam):
        ep = self.eps.loc[ep_idx]
        ci, fi = int(ep[f"videos/{cam}/chunk_index"]), int(ep[f"videos/{cam}/file_index"]); t0 = float(ep[f"videos/{cam}/from_timestamp"])
        vp = self.root / self.info["video_path"].format(video_key=cam, chunk_index=ci, file_index=fi)
        return (vp if vp.exists() else None), t0

    def has_video(self, ep_idx):
        return any(self.video(ep_idx, c)[0] is not None for c in (self.ext, self.wrist) if c)

    def row(self, seg, fracs):
        ep_idx = int(seg.episode_index); off = int(self.eps.loc[ep_idx].dataset_from_index)
        f0, f1 = int(seg.from_index), int(seg.to_index)
        gidx = [min(f1 - 1, int(f0 + fr * (f1 - f0))) for fr in fracs]
        tiles = []
        for cam in (self.ext, self.wrist):
            vp, t0 = self.video(ep_idx, cam) if cam else (None, 0.0)
            if vp is None: tiles += [np.zeros((TILE[1], TILE[0], 3), np.uint8)] * len(gidx); continue
            frames, _ = grab_many(vp, [t0 + (g - off) / self.fps for g in gidx], self.fps); tiles += frames
        q = self.state(gidx[-1]); ang = sg.approach_angle(q) if len(q) >= 6 else float("nan")
        return tiles, dict(approach_deg=ang, wrist_roll=float(q[5]) if len(q) >= 6 else None, gripper=float(q[6]) if len(q) >= 7 else None, frames=gidx, len_s=(f1 - f0) / self.fps)


# ---------------------------------------------------------------- diverse corpus
class DivCtx:
    def __init__(self, root, source):
        self.root = Path(root); self.source = source
        self.eps = {e["episode_id"]: e for e in map(json.loads, open(self.root / "episodes.jsonl"))}
        self.atoms = [json.loads(l) for l in open(self.root / "subtask_atoms.jsonl")]
        for a in self.atoms: a["subtask"] = a["subtask"].lower()

    def row(self, a, fracs):
        e = self.eps[a["episode_id"]]; rate = float(a["native_rate_hz"]); d = self.root / e.get("directory", f"episodes/{a['episode_id']}")
        s0, s1 = int(a["start_timestep"]), int(a["end_timestep_exclusive"])
        idx = [min(s1 - 1, int(s0 + fr * (s1 - s0))) for fr in fracs]
        if self.source == "fmb":
            ext = np.load(d / "side_1_rgb.npy", mmap_mode="r"); wr = np.load(d / "wrist_1_rgb.npy", mmap_mode="r")
            tiles = [ext[i].copy() for i in idx] + [wr[i].copy() for i in idx]
        else:
            cams = [x["name"] if isinstance(x, dict) else x for x in e["cameras"]]
            wrist = next((k for k in cams if re.search(r"wrist", k)), None); ext = next((k for k in cams if k != wrist), None)
            tiles = []
            for k in (ext, wrist):
                if k is None: tiles += [np.zeros((TILE[1], TILE[0], 3), np.uint8)] * len(idx); continue
                frames, _ = grab_many(d / f"videos/{k}.mp4", [i / rate for i in idx], rate); tiles += frames
        return tiles, dict(frames=idx, len_s=(s1 - s0) / rate, approach_deg=ANGLES.get((str(self.root), a["episode_id"], a["atom_index"]), float("nan")))


# ---------------------------------------------------------------- rendering
def label(img, text):
    img = cv2.resize(img, TILE); strip = np.zeros((20, TILE[0], 3), np.uint8)
    cv2.putText(strip, text[:44], (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def render_row(tiles, texts):
    return np.hstack([label(t, x) for t, x in zip(tiles, texts)])


def sheet(rows, path):
    h = max(r.shape[0] for r in rows); w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, h - r.shape[0]), (0, w - r.shape[1]), (0, 0))) for r in rows]
    sep = np.full((6, w, 3), 90, np.uint8)
    img = np.vstack(sum(([r, sep] for r in rows), []))
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- specs
ROOTS = {
    "own": "outputs/rebot_all-annotated-v1", "bottle": "outputs/bottle_grasping-train-annotated-v2",
    "bits": "outputs/rebot_bits-annotated-v2", "pushbook": "outputs/rebot_push-book-annotated-v1",
    "model": "outputs/rebot_rollouts-annotated-v2", "model20": "outputs/rebot_inference_2026-09-20-v1",
}
EXT = {p.split("/")[-1].split("-annotated")[0]: p for p in sorted(glob.glob("outputs/rebot_public_annotated_v1/*/*"))}
DIV = "outputs/diverse_robot_dataset_v3/corpus"; FMB = "outputs/diverse_robot_dataset_v3/fmb"

# spec row: (group, source_key, subtask regex, n, filter) ; filter: None | ("angle>=",45) | ("angle<",45) | ("sub_source", "molmoact")
SPECS = [
    # own teleop
    ("01_own_grasp_cloth", "own", r"^grasp the black sock$", 1, None), ("01_own_grasp_cloth", "own", r"^grasp the white sock$", 1, None),
    ("01_own_grasp_cloth", "own", r"^grasp the white shirt$", 1, None), ("01_own_grasp_cloth", "own", r"^grasp the navy shirt$", 1, None),
    ("01_own_grasp_cloth", "own", r"^grasp the .* shirt$", 2, ("angle>=", 45)),
    ("02_own_grasp_rigid", "own", r"^grasp the cup$", 1, ("angle<", 45)), ("02_own_grasp_rigid", "own", r"^grasp the cup$", 2, ("angle>=", 45)),
    ("02_own_grasp_rigid", "own", r"^grasp the pill bottle$", 1, None), ("02_own_grasp_rigid", "own", r"^grasp the black spray bottle$", 1, None),
    ("02_own_grasp_rigid", "own", r"^grasp the tape roll$", 1, None),
    ("03_own_grasp_small_and_model", "bits", r"^grasp the bit$", 3, None),
    ("03_own_grasp_small_and_model", "model", r"^grasp the (cup|.*bottle)$", 1, None), ("03_own_grasp_small_and_model", "model20", r"^grasp the (cup|.*bottle)$", 2, None),
    ("04_own_release", "own", r"^release the black sock in the basket$", 1, None), ("04_own_release", "own", r"^release the white shirt in the bin$", 1, None),
    ("04_own_release", "own", r"^release the cup in the basket$", 1, None), ("04_own_release", "bottle", r"^release the pill bottle in the bin$", 1, None),
    ("04_own_release", "own", r"^release the tape roll in the bin$", 1, None), ("04_own_release", "bits", r"^release the bit in the empty slot$", 1, None),
    ("05_own_release_bits_move", "bits", r"^release the bit in the empty slot$", 2, None), ("05_own_release_bits_move", "bits", r"^move the bit to the box$", 1, None),
    ("05_own_release_bits_move", "own", r"^move the black sock to the basket$", 1, None), ("05_own_release_bits_move", "own", r"^return to home$", 1, None),
    ("05_own_release_bits_move", "own", r"^release the black spray bottle in the (basket|bin)$", 1, None),
    ("06_own_pushbook", "pushbook", r"^push the book off the box$", 1, None), ("06_own_pushbook", "pushbook", r"^lift the box lid$", 1, None),
    ("06_own_pushbook", "pushbook", r"^push the box lid open$", 1, None),
    # external ReBot operators
    ("07_ext_nonprehensile", "ext:b601_paperball_pickplace_new", r"^knock the paper ball off the box$", 1, None),
    ("07_ext_nonprehensile", "ext:b601-dm-icehockey-v1", r"^hit the red puck back", 2, None), ("07_ext_nonprehensile", "ext:b601-dm-icehockey-v1", r"^grasp the red striker$", 1, None),
    ("07_ext_nonprehensile", "ext:b601-dm-icehockey-v1", r"^release the red striker on the table$", 1, None), ("07_ext_nonprehensile", "ext:rebot_shakehands", r"^shake the hand$", 1, None),
    ("08_ext_cloth", "ext:b601_towel_fold", r"^grasp the towel$", 1, None), ("08_ext_cloth", "ext:b601_towel_fold", r"^fold the towel over$", 2, None),
    ("08_ext_cloth", "ext:b601_towel_fold", r"^lift the towel and lay it flat$", 1, None), ("08_ext_cloth", "ext:b601_towel_fold", r"^release the towel on the (table|mat)$", 2, None),
    ("09_ext_pour", "ext:Seeed_Hackathon_egg_merged", r"^pour the egg into the pan$", 1, None), ("09_ext_pour", "ext:Seeed_rebot_hackathon_oil_1", r"^pour the oil into the pan$", 1, None),
    ("09_ext_pour", "ext:Seeed_rebot_miandanducaiji_real_1", r"^pour the noodles into the pan$", 1, None), ("09_ext_pour", "ext:Seeed_rebot_hackathon_oil_1", r"^grasp the white oil cup$", 1, None),
    ("09_ext_pour", "ext:Seeed_rebot_lianxucaiji_18_lian3_merged", r"^release the cooking ingredient in the pan$", 1, None), ("09_ext_pour", "ext:Seeed_rebot_lianxucaiji_18_lian3_merged", r"^grasp the next cooking ingredient$", 1, None),
    ("10_ext_rigid_grasp", "ext:portal-rebot-pick-microcontroller", r"^grasp the microcontroller$", 1, None), ("10_ext_rigid_grasp", "ext:pick_place_b601", r"^grasp the black disk$", 1, None),
    ("10_ext_rigid_grasp", "ext:rebot_pickplace_bolt_dataset_20260909_130237", r"^grasp the silver bolt$", 1, None), ("10_ext_rigid_grasp", "ext:rebot-pick-motor", r"^grasp the motor$", 1, None),
    ("10_ext_rigid_grasp", "ext:rs_b601_1", r"^grasp the red crayfish$", 1, None), ("10_ext_rigid_grasp", "ext:pick-up-red-die-place-in-basket-rebot-act-v02", r"^grasp the red die$", 1, None),
    ("11_ext_rigid_release", "ext:portal-rebot-pick-microcontroller", r"^release the microcontroller in the box$", 1, None), ("11_ext_rigid_release", "ext:pick_place_b601", r"^release the black disk in the box$", 1, None),
    ("11_ext_rigid_release", "ext:rebot_pickplace_bolt_dataset_20260909_130237", r"^release the silver bolt on the green pad$", 1, None), ("11_ext_rigid_release", "ext:b601_pusht_pick_and_place", r"^release the orange t block on the target outline$", 1, None),
    ("11_ext_rigid_release", "ext:rebot-pick-motor", r"^release the motor in the box$", 1, None), ("11_ext_rigid_release", "ext:rs_b601_1", r"^release the red crayfish in the black box$", 1, None),
    ("12_ext_misc", "ext:b601_pusht_pick_and_place", r"^grasp the orange t block$", 1, None), ("12_ext_misc", "ext:b601_paperball_pickplace_new", r"^grasp the paper ball$", 1, None),
    ("12_ext_misc", "ext:rebot-cansort-recycle-merged", r"^grasp the .*can$", 1, None), ("12_ext_misc", "ext:sort_b601", r"^release the white disk on the blue plate$", 1, None),
    ("12_ext_misc", "ext:sort_b601", r"^grasp the white disk$", 1, None), ("12_ext_misc", "ext:pick-up-red-die-place-in-basket-rebot-act-v02", r"^release the red die in the basket$", 1, None),
    # diverse corpus
    ("13_div_press_turn", "div", r"^press the yellow button$", 1, None), ("13_div_press_turn", "div", r"^press the pump$", 1, None),
    ("13_div_press_turn", "div", r"^press the toaster lever$", 1, None), ("13_div_press_turn", "div", r"^press the flush handle$", 1, None),
    ("13_div_press_turn", "div", r"^turn on the lamp$", 1, None), ("13_div_press_turn", "div", r"^turn on the light switch$", 1, None),
    ("14_div_rotate_knob", "div", r"^rotate the left faucet handle$", 1, None), ("14_div_rotate_knob", "div", r"^rotate the .*(knob)$", 1, None),
    ("14_div_rotate_knob", "div", r"^turn off the stove knob$", 1, None), ("14_div_rotate_knob", "div", r"^rotate the sanitizer bottle$", 1, None),
    ("14_div_rotate_knob", "div", r"^rotate the mug$", 1, None), ("14_div_rotate_knob", "fmb", r"^rotate the object$", 1, None),
    ("15_div_push_close", "div", r"^push the sanitizer bottle$", 1, None), ("15_div_push_close", "div", r"^push the box lid$", 1, None),
    ("15_div_push_close", "div", r"^push the portafilter handle$", 1, None), ("15_div_push_close", "div", r"^push the foosball handle$", 1, None),
    ("15_div_push_close", "div", r"^close the door$", 1, None), ("15_div_push_close", "div", r"^close the laptop lid$", 1, None),
    ("16_div_close_open", "div", r"^close the drawer$", 1, None), ("16_div_close_open", "div", r"^close the oven door$", 1, None),
    ("16_div_close_open", "div", r"^open the fridge$", 1, None), ("16_div_close_open", "div", r"^open the oven door$", 1, None),
    ("16_div_close_open", "div", r"^open the drawer$", 1, None), ("16_div_close_open", "div", r"^open the container lid$", 1, None),
    ("17_div_pull_lift", "div", r"^pull the paper towel$", 1, None), ("17_div_pull_lift", "div", r"^pull the rope$", 1, None),
    ("17_div_pull_lift", "div", r"^pull the lid$", 1, None), ("17_div_pull_lift", "div", r"^pull the cloth off the black stand$", 1, None),
    ("17_div_pull_lift", "div", r"^lift the lid$", 1, None), ("17_div_pull_lift", "div", r"^(open|lift) the tongues$", 1, None),
    ("18_div_cloth", "div", r"^fold the striped towel$", 1, None), ("18_div_cloth", "div", r"^fold the shorts$", 1, None),
    ("18_div_cloth", "div", r"^unfold the shorts$", 1, None), ("18_div_cloth", "div", r"^straighten the cloth$", 1, None),
    ("18_div_cloth", "div", r"^spread the tissue$", 1, None), ("18_div_cloth", "div", r"^flatten the white towel$", 1, None),
    ("19_div_wipe_tool", "div", r"^wipe the desk with the rag$", 1, None), ("19_div_wipe_tool", "div", r"^wipe the sink with the towel$", 1, None),
    ("19_div_wipe_tool", "div", r"^scrub the toilet bowl", 1, None), ("19_div_wipe_tool", "div", r"^stir the black beans", 1, None),
    ("19_div_wipe_tool", "div", r"^wipe the blue plate", 1, None), ("19_div_wipe_tool", "div", r"^scrub the sink", 1, None),
    ("20_div_pour", "div", r"^water the plant$", 1, None), ("20_div_pour", "div", r"^pour the pitcher into the glass$", 1, None),
    ("20_div_pour", "div", r"^pour the bowl into the box$", 1, None), ("20_div_pour", "div", r"^pour the cup into the red bowl$", 1, None),
    ("20_div_pour", "div", r"^tilt the copper mug$", 1, None), ("20_div_pour", "div", r"^pour the orange cup onto the large plate$", 1, None),
    ("21_div_insert", "div", r"^insert the portafilter into the group head$", 1, None), ("21_div_insert", "div", r"^insert the white tube into the black channel$", 1, None),
    ("21_div_insert", "fmb", r"^insert the object into the board$", 1, None), ("21_div_insert", "fmb", r"^place the object on the fixture$", 1, None),
    ("21_div_insert", "div", r"^release the paper in the shredder$", 1, None), ("21_div_insert", "div", r"^release the white flower in the vase$", 1, None),
    ("22_div_grasp_modes", "div", r"^grasp the watering can$", 1, None), ("22_div_grasp_modes", "div", r"^grasp the blue mug$", 1, ("angle>=", 45)),
    ("22_div_grasp_modes", "div", r"^grasp the blue mug$", 1, ("angle<", 45)), ("22_div_grasp_modes", "div", r"^grasp the cup$", 1, ("sub_source", "robochallenge")),
    ("22_div_grasp_modes", "div", r"^grasp the fridge handle$", 1, None), ("22_div_grasp_modes", "div", r"^grasp the .*bottle$", 1, ("angle>=", 45)),
    ("23_div_grasp_more", "div", r"^grasp the green block$", 1, None), ("23_div_grasp_more", "div", r"^grasp the rag$", 1, None),
    ("23_div_grasp_more", "div", r"^grasp the white flower$", 1, None), ("23_div_grasp_more", "div", r"^grasp the phone$", 1, None),
    ("23_div_grasp_more", "div", r"^grasp the banana$", 1, None), ("23_div_grasp_more", "div", r"^grasp the book$", 1, None),
    ("24_fmb_and_grasp", "fmb", r"^grasp the object$", 1, ("angle>=", 45)), ("24_fmb_and_grasp", "fmb", r"^grasp the object$", 1, ("angle<", 45)),
    ("24_fmb_and_grasp", "fmb", r"^lift the object$", 1, None), ("24_fmb_and_grasp", "fmb", r"^move the object to the board$", 1, None),
    ("24_fmb_and_grasp", "div", r"^grasp the lid$", 1, None), ("24_fmb_and_grasp", "div", r"^grasp the fork$", 1, None),
    ("25_div_release", "div", r"^release the green block in the basket$", 1, None), ("25_div_release", "div", r"^release the cup on the rack$", 1, None),
    ("25_div_release", "div", r"^release the watering can on the table$", 1, None), ("25_div_release", "div", r"^release the book on the shelf$", 1, None),
    ("25_div_release", "div", r"^release the rag on the tray$", 1, None), ("25_div_release", "div", r"^release the can in the trash bin$", 1, None),
    ("26_div_misc", "div", r"^press the white tube into the black channel$", 1, None), ("26_div_misc", "div", r"^hold the stove", 1, None),
    ("26_div_misc", "div", r"^place the plate on the table$", 1, None), ("26_div_misc", "div", r"^push the top drawer$", 1, None),
    ("26_div_misc", "div", r"^push the cloth$", 1, None), ("26_div_misc", "div", r"^straighten the shorts$", 1, None),
]


def verb_fracs(subtask, rebot):
    v = subtask.split()[0]
    if v == "grasp": return FRACS["grasp_rebot" if rebot else "grasp_div"]
    if v == "release": return FRACS["release_rebot" if rebot else "release_div"]
    return FRACS["other"]


def main(only=None):
    rng = random.Random(SEED); ctxs = {}; groups = {}; log = open(HERE / "sampled.jsonl", "a" if only else "w")
    for group, src, rx, n, flt in SPECS:
        if only and not any(group.startswith(o) for o in only.split(",")): continue
        try:
            if src.startswith("ext:"):
                key = src[4:]; root = EXT[key]; ctx = ctxs.setdefault(root, RebotCtx(root)); rebot = True
            elif src in ("div", "fmb"):
                root = DIV if src == "div" else FMB; ctx = ctxs.setdefault(root, DivCtx(root, src)); rebot = False
            else:
                root = ROOTS[src]; ctx = ctxs.setdefault(root, RebotCtx(root)); rebot = True
        except Exception as ex:
            print(f"SKIP {group} {src}: {ex}"); continue
        if rebot:
            cand = ctx.segs[ctx.segs.subtask.str.match(rx)]
            cand = [s for _, s in cand.iterrows() if ctx.has_video(int(s.episode_index))]
            if flt and flt[0].startswith("angle"):
                cand = [s for s in cand if (a := ANGLES.get((root, int(s.episode_index), int(s.from_index)))) is not None and ((a >= flt[1]) if flt[0] == "angle>=" else (a < flt[1]))]
        else:
            cand = [a for a in ctx.atoms if re.match(rx, a["subtask"])]
            if flt and flt[0] == "sub_source": cand = [a for a in cand if a["source"] == flt[1]]
            if flt and flt[0].startswith("angle"):
                cand = [a for a in cand if (x := ANGLES.get((root, a["episode_id"], a["atom_index"]))) is not None and ((x >= flt[1]) if flt[0] == "angle>=" else (x < flt[1]))]
        if not cand: print(f"NONE {group} {src} {rx} {flt}"); continue
        rng.shuffle(cand); seen = set(); first, rest = [], []
        for c in cand:
            e = int(c.episode_index) if rebot else c["episode_id"]; (rest if e in seen else first).append(c); seen.add(e)
        chosen = (first + rest)[:n]
        for c in chosen:
            fr = verb_fracs(c.subtask if rebot else c["subtask"], rebot)
            try:
                tiles, meta = ctx.row(c, fr)
            except Exception as ex:
                print(f"FAIL {group} {src} {rx}: {ex}"); continue
            if rebot:
                who = src if not src.startswith("ext:") else "ext " + src[4:][:14]; ep = f"ep{int(c.episode_index)}"; sub = c.subtask
            else:
                who = f"{c['source']} {c['embodiment']}"; ep = c["episode_id"].split("__")[-1]; sub = c["subtask"]
            head = f"{who} {ep} | {sub}"; ang = f"approach {meta['approach_deg']:.0f}deg" if meta["approach_deg"] == meta["approach_deg"] else ""
            texts = [head[:44], sub[:44] if len(head) > 44 else f"{meta['len_s']:.1f}s f{meta['frames'][0]}-{meta['frames'][-1]}", f"top {fr[0]:.2f}/{fr[1]:.2f}/{fr[2]:.2f}",
                     f"wrist {ang}", f"grip {meta['gripper']:.0f} roll {meta['wrist_roll']:.0f}" if meta.get("gripper") is not None else "wrist", f"{meta['len_s']:.1f}s"]
            groups.setdefault(group, []).append(render_row(tiles, texts))
            log.write(json.dumps(dict(group=group, source=src, root=root, subtask=sub, episode=int(c.episode_index) if rebot else c["episode_id"],
                                      from_index=int(c.from_index) if rebot else c["start_timestep"], to_index=int(c.to_index) if rebot else c["end_timestep_exclusive"],
                                      fracs=fr, **{k: v for k, v in meta.items()})) + "\n"); log.flush()
        print(f"{group:28s} {src:44s} {rx:52s} {len(chosen)}/{len(cand)}", flush=True)
    for g, rows in groups.items():
        sheet(rows, SHEETS / f"{g}.jpg"); print("wrote", g, len(rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

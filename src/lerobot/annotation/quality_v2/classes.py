"""Side-by-side class of every unit: (pool, verb group, object family[, target]). Draft rules, reviewed on the sheets.

uv run python -m lerobot.annotation.quality_v2.classes   (reads units.csv, overrides.csv if present)
-> units_classed.csv, classes.csv, and the count tables on stdout
"""
import re
from pathlib import Path

import pandas as pd

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
HERE = QUALITY_V2

# one pool = one robot in one kind of scene; sheets never mix pools
POOL = {"rebot_all": "rebot_main", "additions": "rebot_main", "validation": "rebot_main", "bits_book": "rebot_bits",
        "external": "rebot_external", "droid": "droid", "droid_success": "droid", "molmoact": "molmoact",
        "ur7e": "ur7e", "yam": "yam", "fmb": "fmb"}

VERB_GROUP = {
    "grasp": "grasp", "move": "carry", "lift": "carry", "release": "release", "place": "release", "stack": "release",
    "return": "return", "insert": "insert", "plug": "insert",
    "press": "press", "turn on": "press", "turn off": "press",
    "open": "articulate", "close": "articulate", "pull": "articulate", "push": "articulate", "rotate": "articulate", "hold": "articulate",
    "fold": "cloth_work", "unfold": "cloth_work", "spread": "cloth_work", "straighten": "cloth_work", "flatten": "cloth_work",
    "wipe": "surface_work", "scrub": "surface_work",
    "pour": "pour", "water": "pour", "tilt": "pour", "stir": "pour", "shake": "other", "hit": "strike", "knock": "strike",
}

# first match wins; order matters (spray bottle before bottle, paper towel before paper, ...)
FAMILY = [
    ("sock", r"\bsock"),
    ("garment", r"shirt|shorts|garment"),
    ("cloth", r"towel|rag\b|cloth|napkin|tissue|plastic sheet"),
    ("bottle", r"bottle|dispenser"),
    ("cup_mug", r"\bcup|mug|tumbler|glass\b|can\b|jar\b|pitcher|jug|canister"),
    ("bowl_plate", r"bowl|plate|dish\b|pan\b|tray"),
    ("lid", r"\blid|\bcap\b"),
    ("ring_disk", r"ring\b|tape|disk|pieces|puck"),
    ("bit_small_part", r"\bbit\b|bolt|sugar cube|lego|microcontroller|sweet\b|potato chip|die\b"),
    ("block", r"block|cube|eraser|chess|foam|arch"),
    ("fruit", r"apple|banana|kiwi|mango|lemon|avocado|pear\b|plum|orange$|tangerine|grape|star fruit|radish|egg|fig bar|protein bar"),
    ("long_thin", r"pen\b|pencil\b|marker|sharpie|crayon|spoon|fork|knife|tong|screwdriver|straw|cable|cord|rope|tube|strip|bar\b|scissors|brush|duster|crayfish"),
    ("flat", r"paper$|book|phone|card|packet|wrapper|tea bag|pencil case|paper bag|package|bag$|laptop"),
    ("soft_toy", r"plush|stuffed|teddy|doll|rabbit|toy|paper ball|crumpled|bread|noodle|sponge|tennis|spiky|pizza"),
    ("flower", r"flower"),
    ("handle_fixture", r"drawer|door|handle|knob|faucet|lever|switch|button|lamp|pump|stove|fridge|tap\b|portafilter|charger"),
    ("device", r"striker|glasses|litter|cereal container|divider|fruit|tin\b|pillow|desk|sink|counter|spill|screen|mouse|speaker|headphone|gamepad|compass|motor|box$|object$|item|piece$|ingredient|beans|oil|noodles|watering can|hand$|plant$"),
]
TARGET = [("container", r"basket|bin\b|trash|dishwasher|organizer|glass$|box|bowl|drawer|cup\b|mug|vase|jar|pot\b|pan\b|sink|bag|holder|container|compartment|slot|tray|shredder"),
          ("fixture", r"peg|rack|stack|base\b|tape roll|hook|group head|socket|channel|board|fixture|dish rack|outline"),
          ("surface", r"pad\b|zone|stand|dish\b|block|backrest|stove|refrigerator|lid$|dryer|microwave|barrier|side$|toaster|table|plate|counter|desk|shelf|floor|cutting board|mat\b|towel|cloth|paper$")]


def family(obj, pool):
    if pool == "fmb": return "fmb_object"
    o = re.sub(r"^(the|a|an) ", "", str(obj or "").lower())
    if not o or o == "to home" or o == "nan": return "none"
    for f, rx in FAMILY:
        if re.search(rx, o): return f
    return "unmapped"


def target(t):
    t = str(t or "").lower()
    for f, rx in TARGET:
        if re.search(rx, t): return f
    return "none" if t in ("", "nan") else "unmapped"


d = pd.read_csv(HERE / "units.csv", dtype={"episode": str})
d["pool"] = d.dataset.map(POOL)
rc = d.dataset == "robochallenge"  # split by arm: ARX5 vs UR5
if rc.any():
    import json
    e = {a["episode_id"]: a["embodiment"] for a in map(json.loads, open("outputs/diverse_robot_dataset_v3/corpus/subtask_atoms.jsonl"
                                                                          if Path("outputs").exists() else HERE.parents[1] / "outputs/diverse_robot_dataset_v3/corpus/subtask_atoms.jsonl"))}
    d.loc[rc, "pool"] = "rc_" + d.loc[rc, "episode"].map(e).str.lower()
d["verb_group"] = d.verb.map(VERB_GROUP).fillna("other")
d["family"] = [family(o, p) for o, p in zip(d.object, d.pool)]
d["target_type"] = d.target.map(target)
# ---- reviewed corrections (merge_reviews.py): string scope, then target types, then row scope
ov = HERE / "overrides.csv"
if ov.exists():
    o = pd.read_csv(ov); m = dict(zip(zip(o.pool, o.object), o.family)); un = set(zip(o[o.unsure].pool, o[o.unsure].object))
    d["family"] = [m.get((p, ob), f) for p, ob, f in zip(d.pool, d.object, d.family)]
    d["family_unsure"] = [(p, ob) in un for p, ob in zip(d.pool, d.object)]
to = HERE / "target_overrides.csv"
if to.exists():
    t = pd.read_csv(to); m = dict(zip(zip(t.pool, t.target), t.target_type))
    d["target_type"] = [m.get((p, x), f) for p, x, f in zip(d.pool, d.target, d.target_type)]
d.loc[(d.pool == "rc_arx5") & d.target.str.contains("tray", na=False), "target_type"] = "surface"  # S15e: shallow tray, rag laid flat
d["family_unsure"] = d.get("family_unsure", False); d["family_unsure"] = d.family_unsure.fillna(False).astype(bool)
ro = HERE / "row_overrides.csv"
if ro.exists():
    r = pd.read_csv(ro, dtype={"episode": str}); m = dict(zip(zip(r.dataset, r.episode, r.unit), zip(r.family, r.unsure)))
    for k, (ds, ep, un) in enumerate(zip(d.dataset, d.episode, d.unit)):
        if (ds, ep, un) in m:
            f, uns = m[(ds, ep, un)]
            if f in ("container", "fixture", "surface", "none") and d.at[k, "verb_group"] == "release": d.at[k, "target_type"] = f
            elif f not in ("container", "fixture", "surface"): d.at[k, "family"] = f
            d.at[k, "family_unsure"] = bool(d.at[k, "family_unsure"]) or bool(uns)
d["family"] = d.family.replace({"plate_upright": "bowl_plate", "lid_knob": "lid", "pouch": "flat", "spray_bottle": "bottle"})
d.loc[d.object.str.contains("watering can", na=False), "family"] = "handle_vessel"  # S15, S15e, S15f


# ---- task verbs: one class per kind of action (S08, S09, S11, S14, S15f found the verb groups mixed)
def action(r):
    v, o, t = r.verb, str(r.object).lower(), str(r.target).lower()
    if r.pool == "fmb": return {"insert": "insert", "place": "release", "rotate": "rotate_part", "grasp": "grasp", "move": "carry"}[v]
    if v in ("stack", "place") and r.dataset == "external": return "sequence"  # whole grasp-to-release units (S08)
    if v == "lift" and "lay it flat" in o: return "cloth_work"
    if v == "lift" and "lid" in o: return "panel_open"
    if v in ("grasp", "move", "lift", "release", "place", "stack", "return"): return r.verb_group
    if v in ("insert", "plug") or (v == "press" and "channel" in t): return "insert"
    if v in ("fold", "unfold", "spread", "straighten"): return "cloth_work"
    if v in ("wipe", "scrub", "flatten"): return "wipe_scrub"
    if v == "stir": return "stir"
    if v in ("pour", "water", "tilt"): return "pour"
    if v in ("hit",): return "strike"
    if v == "knock" or (v == "push" and re.search(r"t block|book|bottle|cloth", o)): return "push_object"
    if v == "shake": return "handshake"
    if v == "open" and "tong" in o: return "tool_use"
    if v == "rotate" and re.search(r"faucet|knob|handle", o): return "knob_turn"
    if v == "rotate": return "in_hand_rotate"
    if v in ("turn on", "turn off") and re.search(r"stove|knob|tap", o + " " + t): return "knob_turn"
    if v == "hold": return "other"
    if v in ("turn on", "turn off"): return "switch"
    if v == "press" and re.search(r"button", o): return "button"
    if v == "press" or (v == "push" and "sanitizer" in o and r.dataset == "molmoact" and False): return "press_lever"
    if v == "pull" and re.search(r"rope|cloth|paper towel", o): return "drag_soft"
    if v in ("close",) or (v == "push" and re.search(r"lid|drawer|door|handle", o)): return "panel_close"
    if v in ("open", "pull"): return "panel_open"
    return "other"


d["action"] = d.apply(action, axis=1)
COARSE = {"sock": "deformable", "garment": "deformable", "cloth": "deformable", "soft_toy": "deformable", "flat": "flat"}


def cls(r):
    a = r.action
    if a == "return": return f"{r.pool}/return"
    if a == "release":
        tgt = r.target_type
        if r.pool == "rebot_main" and tgt == "container": tgt = "bin" if "bin" in str(r.target) else "basket" if "basket" in str(r.target) else "container"
        return f"{r.pool}/release/{r.family}->{tgt}" if r.pool != "fmb" else "fmb/release/cradle"
    if a in ("grasp", "carry") or (r.pool == "fmb" and a == "insert"): return f"{r.pool}/{a}/{r.family}"
    return f"{r.pool}/{a}"


def coarse(r):  # the nearest class on the same robot, for classes under 8 (rubric 5.5 step 7)
    a = r.action
    if a in ("grasp", "carry", "release"):
        g = COARSE.get(r.family, "fmb" if r.pool == "fmb" else "rigid")
        return f"{r.pool}/{a}/{g}" + (f"->{r.target_type}" if a == "release" else "")
    return f"{r.pool}/tasks"


d["class"] = d.apply(cls, axis=1)
n = d["class"].map(d["class"].value_counts())
d["sheet_class"] = [c if k >= 8 else coarse(r) for c, k, r in zip(d["class"], n, d.itertuples())]
d.to_csv(HERE / "units_classed.csv", index=False)
agg = dict(pool=("pool", "first"), action=("action", "first"), units=("unit", "size"), episodes=("episode", "nunique"),
           minutes=("seconds", lambda x: round(x.sum() / 60, 1)), datasets=("dataset", lambda x: ",".join(sorted(set(x)))),
           objects=("object", lambda x: "; ".join(f"{k}({v})" for k, v in x.value_counts().head(6).items())))
c = d.groupby("class").agg(**agg, sheet_class=("sheet_class", "first"))
c.sort_values(["pool", "action", "units"], ascending=[True, True, False]).to_csv(HERE / "classes.csv")
sc = d.groupby("sheet_class").agg(**agg, n_classes=("class", "nunique"))
sc.sort_values(["pool", "action", "units"], ascending=[True, True, False]).to_csv(HERE / "sheet_classes.csv")
if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80); pd.set_option("display.max_rows", 500)
    print("unmapped:", d[d.family == "unmapped"].object.value_counts().to_dict(), "| actions other:", d[d.action == "other"].subtask.value_counts().to_dict())
    s = sc.assign(small=sc.units < 8).groupby("pool").agg(sheet_classes=("units", "size"), still_under_8=("small", "sum"), units=("units", "sum"))
    s["fine_classes"] = c.groupby("pool").size()
    print(s.to_string()); print("fine classes", len(c), "sheet classes", len(sc), "units", int(sc.units.sum()))

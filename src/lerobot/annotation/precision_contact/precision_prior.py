"""Precision prior by text rules, straight from annotation/rubrics/precision_rubric.md
("Reference values (priors; the frames override, with a note)" table, plus the level-5
anchor "plug into socket"). NO LABEL IS WRITTEN: the prior is a hypothesis shown to the
reviewer next to the images; the images decide.

    precision_prior(subtask, verb, root) -> (level 1-5 or None, one-line why)

A range in the table returns its lower value with the range in the why. No matching row
returns None, "no rubric row". move / return are derived later and return None.

Run as a script (repo root) for the coverage check -> precision_prior_coverage.md.
Import is side-effect free.
"""
import re

CLOTH = r"\b(sock|shirt|t-shirt|cloth|towel|rag)s?\b"
RIGID = r"\b(cup|bottle|pill bottle|spray bottle|tape roll|apple|block|mug)s?\b"
CONTAINER = r"\b(basket|bin)\b"
SURFACE = r"\b(table|desk|tray)\b"
PREP = r"\b(in|into|on|onto|at|beside|against|to|over)\b"


def _split(s, verb):
    """(object, target) of '<verb> the <object> <prep> <target>'."""
    rest = re.sub(rf"^{verb}( on| off)?\s+", "", s)
    m = re.search(PREP, rest)
    return (rest[:m.start()].strip(), rest[m.start():].strip()) if m else (rest, "")


def precision_prior(subtask: str, verb: str, root: str) -> tuple[int | None, str]:
    s = subtask.lower().strip(); v = verb.lower().strip(); r = (root or "").lower()
    obj, tgt = _split(s, v)
    fmb = "fmb" in r

    if v in ("move", "return"):
        return None, "derived (move = next step minus 1; return = 1), computed later"

    if v == "grasp":
        if re.search(r"\bbits?\b", obj): return 5, "row 'grasp the bit (bits-box)': 5 (6 mm bit lying flat 4, +1 pads across)"
        if re.search(CLOTH, obj): return 1, "row 'grasp the sock / shirt / cloth / towel / rag': 1 (any fold captures)"
        if re.search(r"\bwatering can\b", obj): return 3, "row 'grasp the watering can': 3 (handle)"
        if re.search(r"\bflower\b", obj): return 3, "row 'grasp the flower': 3"
        if re.search(r"\bhandle\b", obj): return 3, "row 'grasp the drawer handle': 3 (handle)"
        if fmb and re.search(r"\bobject\b", obj): return 3, "row 'grasp the object (FMB)': 3"
        if re.search(RIGID, obj): return 3, "row 'grasp the cup / bottle / ... / block / mug': 3 (straddle a rigid object, free yaw)"
        return None, "no rubric row"

    if v in ("release", "lean"):
        if re.search(r"\bbits?\b", obj) and re.search(r"\bslot\b", tgt): return 5, "row 'release the bit in the slot': 5 (compartment barely wider 4, +1 aligned)"
        if re.search(r"\bshredder\b", tgt): return 5, "row 'release in the shredder': 5 (1 cm slot, edge-on)"
        if re.search(r"\bvase\b", tgt): return 4, "row 'release in the vase': 4 (stem into a 3 cm mouth stem-first)"
        if re.search(r"\brack\b", tgt): return 3, "row 'release on the rack': 3 (rack slot a few cm wider)"
        if re.search(r"\bpillow\b", obj) and re.search(r"\bwall\b", tgt): return 2, "row 'lean the pillow against the wall': 2 (3-10 cm band)"
        if re.search(SURFACE, tgt): return 1, "row 'release ... any object on the table, desk or tray': 1 (open surface)"
        if re.search(CONTAINER, tgt):
            if re.search(CLOTH, obj): return 1, "row 'release the sock / shirt in the basket / bin': 1 (30 cm+ rim, deformable)"
            if re.search(RIGID, obj): return 1, "row 'release the cup / bottle in the basket or bin': 1-2 per rubric; 2 if the container is crowded"
        return None, "no rubric row"

    if v == "insert" and fmb: return 5, "row 'insert (FMB)': 5 (mating fit)"
    if v == "place" and re.search(r"\bfixture\b", s): return 4, "row 'place the object on the fixture (FMB)': 4 (fixture fit)"
    if v == "lift" and fmb and re.search(r"\bobject\b", s): return 4, "row 'lift (FMB)': 4 (out of a fitted fixture)"
    if v == "plug" and re.search(r"\bsocket\b", s): return 5, "level-5 anchor 'plug into socket' (mating fit); not a table row"
    if v == "press" and re.search(r"\bbutton\b", s): return 3, "row 'press the button (corpus)': 3 (2 cm face)"
    if v == "turn" and re.search(r"\b(lamp|switch|light)\b", s): return 3, "row 'turn on the lamp / switch': 3-4 per rubric (switch size)"
    if v == "water": return 2, "row 'water the plant': 2 (spout over a 10 cm pot)"
    if v == "dump" and re.search(r"\bboard\b", s): return 2, "row 'dump the strawberries on the cutting board': 2 (board-sized target)"
    if v == "wipe" and re.search(r"\b(desk|table)\b", s): return 1, "row 'wipe the desk': 1 (area)"
    if v in ("fold", "unfold"): return 1, "row 'fold / unfold the cloth': 1 (area)"
    if v == "pull" and re.search(r"\bdrawer\b", s): return 1, "row 'pull the drawer': 1 (free motion)"
    return None, "no rubric row"


def verb_of(sub):
    s = sub.lower().strip()
    if s.startswith("turn on") or s.startswith("turn off"): return "turn"
    return s.split()[0]


if __name__ == "__main__":
    import json, os
    from collections import Counter
    from pathlib import Path
    import pandas as pd

    from lerobot.annotation.paths import WORKSPACE

    os.chdir(WORKSPACE)
    HERE = Path("migration/precision_contact_annotation_2026-09-25")
    counts = Counter()
    for l in open(HERE / "rows.jsonl"):
        d = json.loads(l)
        if d["verb"] not in ("move", "return"): counts[(d["root"], d["subtask"].lower(), d["verb"])] += 1
    for name in ("main_additions_train", "external_rebot_train", "validation_all"):
        m = pd.read_parquet(f"outputs/rebot_cache_ready_2026-09-21/{name}/meta/episode_metadata.parquet", columns=["subtask"])
        for sub, n in m.subtask.str.lower().value_counts().items():
            if verb_of(sub) not in ("move", "return"): counts[(name, sub, verb_of(sub))] += int(n)

    rows = []
    for (root, sub, verb), n in counts.items():
        p, why = precision_prior(sub, verb, root)
        rows.append(dict(root=root, subtask=sub, n=n, prior=p, why=why))
    df = pd.DataFrame(rows).sort_values(["root", "n"], ascending=[True, False])

    L = ["# Precision prior coverage (2026-09-25)", "",
         "Text-rule priors from `precision_prior.py` (rubric reference table). Hypotheses only, no label written.",
         "Segments = rows of `rows.jsonl` (bits, pushbook, rebot_all) or `meta/episode_metadata.parquet`",
         "(rebot_cache_ready_2026-09-21 roots); move / return excluded (derived).", "",
         "| root | strings | segments | with prior | None (segments) | None (strings) |", "|---|---|---|---|---|---|"]
    for root, g in df.groupby("root", sort=True):
        nn = g[g.prior.isna()]
        L.append(f"| {root} | {len(g)} | {g.n.sum()} | {g[g.prior.notna()].n.sum()} | {nn.n.sum()} | {len(nn)} |")
    L += ["", "| root | subtask | n | prior | why |", "|---|---|---|---|---|"]
    for x in df.itertuples():
        L.append(f"| {x.root} | {x.subtask} | {x.n} | {'' if pd.isna(x.prior) else int(x.prior)} | {x.why} |")
    (HERE / "precision_prior_coverage.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[5:14]))
    none = df[df.prior.isna()].groupby("subtask").n.sum().sort_values(ascending=False)
    print(none.head(15).to_string())

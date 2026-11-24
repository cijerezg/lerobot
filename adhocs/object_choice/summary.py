"""Tables of the object-choice experiment: which object the arm went to, and what the cells answered about the sides.

    uv run python lerobot/adhocs/object_choice/summary.py [ROOT]

Reads ROOT/sim/<cell>/validation/step_00000000/sim/choice_*/subtask_events.json (run_sim_eval.sh) and ROOT/ask_sides/answers.json
(ask_sides.py), whichever exist; prints the tables and writes ROOT/summary.md. ROOT defaults to outputs/analysis/object_choice_2026-11-17.

Went to = the object the fingers reached first (within 5 cm, ``first_approach`` of choice_tasks.py); touched / lifted likewise.
Couples: p1+p2, p3+p4, p5+p6 are the same spots with the two objects exchanged. For one instruction, an arm that ignores the
instruction's noun and goes by position goes to the named object in one placement of a couple and to the other object in the
other; "follows the swap" counts the couples (per episode index) where it went to the named object in BOTH.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/analysis/object_choice_2026-11-17")
COUPLE = {1: "one each side", 2: "one each side", 3: "both right", 4: "both right", 5: "both left", 6: "both left"}
lines = []


def table(title, header, rows):
    lines.extend([f"\n### {title}\n", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def frac(n, d):
    return f"{n}/{d}" if d else "-"


episodes = []  # one per sim episode
for events in sorted(ROOT.glob("sim/*/validation/step_*/sim/choice_*/subtask_events.json")):
    cell, layout = events.parents[4].name, events.parent.name
    pair, asked, k = re.fullmatch(r"choice_(\w+)_ask_(\w+)_p(\d)", layout).groups()
    for ep in json.loads(events.read_text()):
        c = ep["choice"]
        went = {event: "named" if c[f"first_{event}"] == c["asked"] else "other" if c[f"first_{event}"] else "neither" for event in ("approach", "contact", "lifted")}
        episodes.append(dict(cell=cell, pair=pair, asked=asked, k=int(k), episode=ep["episode"], reward=ep["reward"], success=ep["success"], **went))

if episodes:
    cells = list(dict.fromkeys(e["cell"] for e in episodes))
    groups = defaultdict(list)
    for e in episodes:
        groups[(e["cell"], e["pair"], e["asked"])].append(e)
    rows = []
    for (cell, pair, asked), eps in groups.items():
        n = len(eps)
        by = {(e["k"], e["episode"]): e for e in eps}
        couples = [(by[(k, i)], by[(k + 1, i)]) for k in (1, 3, 5) for i in sorted({e["episode"] for e in eps}) if (k, i) in by and (k + 1, i) in by]
        rows.append([cell, pair, asked, n] + [frac(sum(e["approach"] == w for e in eps), n) for w in ("named", "other", "neither")]
                    + [frac(sum(e["contact"] == "named" for e in eps), n), frac(sum(e["contact"] == "other" for e in eps), n),
                       frac(sum(e["lifted"] == "named" for e in eps), n), frac(sum(e["lifted"] == "other" for e in eps), n),
                       frac(sum(a["approach"] == b["approach"] == "named" for a, b in couples), len(couples)),
                       frac(sum({a["approach"], b["approach"]} == {"named", "other"} for a, b in couples), len(couples)),
                       frac(sum(e["success"] for e in eps), n), f"{sum(e['reward'] for e in eps) / n:.2f}"])
    table("Which object the arm went to (per cell, pair, instruction; all placements)",
          ["cell", "pair", "asked", "episodes", "went to named", "went to other", "neither", "touched named", "touched other", "lifted named", "lifted other",
           "couples: named in both", "couples: one of each", "success", "mean reward"], rows)
    rows = []
    for cell in cells:
        for couple in dict.fromkeys(COUPLE.values()):
            for pair in dict.fromkeys(e["pair"] for e in episodes):
                eps = [e for e in episodes if e["cell"] == cell and e["pair"] == pair and COUPLE[e["k"]] == couple]
                if eps:
                    rows.append([cell, pair, couple, len(eps)] + [frac(sum(e["approach"] == w for e in eps), len(eps)) for w in ("named", "other", "neither")])
    table("By arrangement (both instructions together)", ["cell", "pair", "arrangement", "episodes", "went to named", "went to other", "neither"], rows)
    table("Every layout (went to: first approached, per episode)", ["cell", "layout", "went to", "reward"],
          [[cell, f"choice_{pair}_ask_{asked}_p{k}", " ".join(e["approach"] for e in eps), " ".join(f"{e['reward']:.2f}" for e in eps)]
           for (cell, pair, asked, k), eps in sorted({(e["cell"], e["pair"], e["asked"], e["k"]): [x for x in episodes if (x["cell"], x["pair"], x["asked"], x["k"]) == (e["cell"], e["pair"], e["asked"], e["k"])] for e in episodes}.items())])

answers = ROOT / "ask_sides/answers.json"
if answers.exists():
    rows_in = json.loads(answers.read_text())
    scored = [r for r in rows_in if r["truth"]]
    groups = defaultdict(list)
    for r in scored:
        groups[(r["cell"], r["kind"], r["object"])].append(r)
    table("Side questions (chance = half)", ["cell", "question", "object", "correct", "wrong", "no side named", "said left", "said right"],
          [[cell, kind, obj, frac(sum(r["said"] == r["truth"] for r in rs), len(rs)), sum(r["said"] not in (None, r["truth"]) for r in rs),
            sum(r["said"] is None for r in rs), sum(r["said"] == "left" for r in rs), sum(r["said"] == "right" for r in rs)] for (cell, kind, obj), rs in groups.items()])
    table("Open question: What objects are on the table?", ["cell", "pair", "placement", "answer"],
          [[r["cell"], r["pair"], r["placement"], r["answer"].replace("|", "\\|").replace("\n", " ")] for r in rows_in if r["kind"] == "open"])

if not lines:
    sys.exit(f"nothing under {ROOT}/sim or {ROOT}/ask_sides")
(ROOT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

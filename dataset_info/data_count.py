"""Live record of the training data: episodes, frames and hours per root and per sampling group, read from the
training config and the roots themselves. Writes lerobot/dataset_info/DATA.md (overwritten every run).

    uv run python lerobot/dataset_info/data_count.py [--config config_rl.yaml]

Sources: `dataset.sources` (ReBot roots and their episode_groups), `diverse.rebot_group_weights` (shares),
`diverse.rebot_exclusions_manifest` and `diverse.diverse_exclusions_manifest` (usefulness cuts: frames whose action targets are never drawn), `diverse.root`
(corpus/ and fmb/ episodes.jsonl minus holdout_episodes.json and excluded_episodes.json), `val_dataset_path`.
Roots built but not yet in the config are listed in lerobot/dataset_info/pending.yaml (`- root: ..., note: ...`).
Also writes lerobot/dataset_info/SUBTASKS.md: every subtask string in the sampled ReBot episodes (meta/episode_metadata.parquet).
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def episode_lengths(root):
    eps = pd.concat([pd.read_parquet(p, columns=["episode_index", "length"]) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    return dict(zip(eps.episode_index, eps.length, strict=True))


def hours(frames, fps=30):
    return f"{frames / fps / 3600:.2f}"


# Object families, matched in the object phrase (verb and destination stripped); the earliest match wins (ties: the longest), so
# "move the bit to the compartment with the gray bits" -> bit and "push the bit kit next to ..." -> bit kit.
FAMILIES = {
    "bit kit": r"\bflap with the (?:bits|white card|shipping label)|\bflap\b|\bbit kit\b|\bthe kit\b",
    "headphone case": r"headphone case|case with the headphones",
    "nail clipper": r"nail clipper",
    "bit": r"\bbits?\b",
    "wall plug": r"wall plug",
    "pill bottle": r"pill bottle|\bpills\b",
    "sock": r"\bsock\b",
    "shirt": r"\bshirt\b",
    "towel": r"\btowel\b|\bcorner\b",
    "tape roll": r"tape roll",
    "eraser": r"\beraser\b",
    "disk": r"\bdisk\b",
    "drawer": r"\bdrawer\b",
    "bowl": r"\bbowl\b",
    "book": r"\bbooks?\b",
    "microcontroller": r"microcontroller",
    "floss": r"\bfloss\b",
    "can": r"\bcan\b",
    "deck of cards": r"deck of cards",
}
STOP = r"\b(?:to|in|into|on|onto|off|next|over|down|upright|out|from|back|toward|away|near\w*|farther|farthest|left|right|between|with|lying|apart|under|hanging|of|at|and|flat|open)\b"

DEST = r"\b(?:to|in|into|on|onto|off|next|over|down|upright|out|from|back|toward|away|nearest|farther|farthest|between|lying|apart|under|hanging|at|and|flat|open)\b"


def subtask_object(subtask):
    """(family, variant) of a subtask string, or None for "return to home". Variant = the noun phrase that names the object."""
    rest = subtask.split(" ", 1)[1] if " " in subtask else ""
    if subtask == "return to home" or not rest:
        return None
    rest = re.sub(r"^(?:out|up)\s", "", rest)  # "shake out the towel", "line up the ends"
    head_part = re.split(DEST, rest)[0]  # the object phrase, before any destination
    hits = [(m.start(), -len(m.group()), fam, m) for fam, rx in FAMILIES.items() for m in [re.search(rx, head_part)] if m]
    if hits:
        _, _, fam, m = min(hits, key=lambda h: h[:2])
        head = re.split(r"\b(?:the|a)\s", rest[: m.start()])[-1]
        head = head if not re.search(STOP, head) else ""
        return fam, (head + m.group()).strip()
    phrase = re.sub(r"^(?:up\s|(?:one of )?the\s|a\s)", "", rest)
    phrase = re.split(STOP, phrase)[0].strip()
    return phrase, phrase


def subtask_tables(cfg, shares):
    """Segments and object grasp cycles in the sampled ReBot episodes. A cycle = a run of consecutive segments of one
    episode on the same object family, split at each fresh grasp (grasp -> move -> release = 1, back-to-back regrasps = 1,
    nine bits in a row = 9, each bit kit flap = 1)."""
    seg = defaultdict(lambda: [0, set(), 0, 0, 0])  # subtask -> segments, episodes, frames, internal, external
    obj = defaultdict(lambda: [0, set(), 0, 0, 0, defaultdict(int)])  # family -> cycles, episodes, frames, int, ext, variants
    for s in cfg["dataset"]["sources"]:
        meta = pd.read_parquet(WORKSPACE / s["root"] / "meta/episode_metadata.parquet")
        for g, eps in s["episode_groups"].items():
            if not shares.get(g, 0):
                continue
            col = 3 if g != "external_rebot" else 4
            for e, rows in meta[meta.episode_index.isin(eps)].sort_values(["episode_index", "segment_index"]).groupby("episode_index"):
                prev, prev_grasp = None, False
                for r in rows.itertuples():
                    n = r.to_index - r.from_index
                    x = seg[r.subtask]
                    x[0] += 1
                    x[1].add((s["root"], e))
                    x[2] += n
                    x[col] += 1
                    o = subtask_object(r.subtask)
                    if o is None:
                        prev = None
                        continue
                    y = obj[o[0]]
                    grasp = r.subtask.startswith("grasp ")
                    if o[0] != prev or (grasp and not prev_grasp):
                        y[0] += 1
                        y[col] += 1
                        y[5][o[1]] += 1
                    y[1].add((s["root"], e))
                    y[2] += n
                    prev, prev_grasp = o[0], grasp
    return seg, obj


def diverse_rows(root, cut_s):
    skip = set()
    for name in ("holdout_episodes.json", "excluded_episodes.json"):
        if (root / name).is_file():
            skip |= set(json.loads((root / name).read_text())["episode_ids"])
    by_source = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for sub in ("corpus", "fmb"):
        for line in (root / sub / "episodes.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["episode_id"] in skip:
                continue
            src = r["source"] if isinstance(r["source"], str) else sub
            s = by_source[src]
            s[0] += 1
            s[1] += r.get("frames", r.get("frame_count", 0))
            s[2] += r.get("duration_s", r.get("duration_s_nominal", 0.0))
            s[3] += cut_s.get(r["episode_id"], 0.0)
    return dict(sorted(by_source.items())), len(skip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_rl.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((WORKSPACE / args.config).read_text())
    shares = cfg["diverse"]["rebot_group_weights"]
    manifest = cfg["diverse"].get("rebot_exclusions_manifest")
    cut = defaultdict(int)  # (root, episode) -> excluded frames
    if manifest:
        for x in json.loads((WORKSPACE / manifest).read_text())["exclusions"]:
            cut[(x["source_root"], x["episode_index"])] += x["to_frame"] - x["from_frame"]

    group = defaultdict(lambda: [0, 0, 0])  # episodes, frames, usable frames
    src_rows = []
    for s in cfg["dataset"]["sources"]:
        lengths = episode_lengths(WORKSPACE / s["root"])
        per = {}
        for g, eps in s["episode_groups"].items():
            frames = sum(lengths[e] for e in eps)
            usable = frames - sum(cut[(s["root"], e)] for e in eps)
            per[g] = (len(eps), frames)
            group[g][0] += len(eps)
            group[g][1] += frames
            group[g][2] += usable if shares.get(g, 0) else 0
        sampled = sum(n for g, (n, _) in per.items() if shares.get(g, 0))
        src_rows.append((s["name"], s["root"], len(lengths), sum(lengths.values()), sampled, per))

    rebot_share = sum(v for g, v in shares.items() if g in group)
    diverse_share = cfg["diverse"]["weight"] if cfg["diverse"].get("enabled") else 0
    div_manifest = cfg["diverse"].get("diverse_exclusions_manifest")
    cut_s = defaultdict(float)  # episode_id -> excluded seconds
    if div_manifest:
        for x in json.loads((WORKSPACE / div_manifest).read_text())["exclusions"]:
            cut_s[x["episode_id"]] += x["to_s"] - x["from_s"]
    div, n_skip = diverse_rows(WORKSPACE / cfg["diverse"]["root"], cut_s)

    out = [f"# Data record\n\nGenerated {datetime.now():%Y-%m-%d %H:%M} by `uv run python lerobot/dataset_info/data_count.py` from "
           f"`{args.config}`. Do not edit by hand; rerun after every change to the data or the config.\n"]
    out.append("## Training mix by sampling group\n\nShare = fraction of all training samples. Usable = frames minus the usefulness cuts "
               f"(`{manifest}`; rule: `lerobot/src/lerobot/annotation/rubrics/usefulness_review_rubric.md`). ReBot hours at 30 fps.\n")
    out.append("| group | share | episodes | frames | usable frames | hours |\n|---|---|---|---|---|---|")
    for g, (n, f, u) in sorted(group.items(), key=lambda kv: -shares.get(kv[0], 0)):
        out.append(f"| {g} | {shares.get(g, 0):.4g} | {n} | {f:,} | {u:,} | {hours(f)} |")
    tot = [sum(v[i] for g, v in group.items() if shares.get(g, 0)) for i in range(3)]
    out.append(f"| **ReBot sampled** | **{rebot_share:.4g}** | **{tot[0]}** | **{tot[1]:,}** | **{tot[2]:,}** | **{hours(tot[1])}** |")
    dn, df, ds = (sum(v[i] for v in div.values()) for i in range(3))
    dc = sum(v[3] for v in div.values())
    out.append(f"| diverse corpus | {diverse_share:.4g} | {dn} | {df:,} | | {ds / 3600:.2f} (usable {(ds - dc) / 3600:.2f}) |\n")

    out.append("## ReBot roots in the config\n\n| source | root | episodes | frames | hours | sampled eps | groups (episodes / frames) |\n|---|---|---|---|---|---|---|")
    for name, root, n, f, sampled, per in src_rows:
        groups = ", ".join(f"{g} {k} / {fr:,}" for g, (k, fr) in per.items())
        out.append(f"| {name} | `{root}` | {n} | {f:,} | {hours(f)} | {sampled} | {groups} |")
    n_all, f_all = sum(r[2] for r in src_rows), sum(r[3] for r in src_rows)
    out.append(f"| **total** | | **{n_all}** | **{f_all:,}** | **{hours(f_all)}** | | |\n")

    out.append(f"## Diverse corpus\n\n`{cfg['diverse']['root']}`: corpus/ + fmb/ episodes, minus {n_skip} holdout or excluded episodes "
               f"(anchor-level retention not counted). Hours from each episode's duration; usable = minus `{div_manifest}`.\n")
    out.append("| source | episodes | frames | hours | usable hours |\n|---|---|---|---|---|")
    out += [f"| {k} | {n} | {f:,} | {s / 3600:.2f} | {(s - c) / 3600:.2f} |" for k, (n, f, s, c) in div.items()]

    val = cfg.get("val_dataset_path")
    if val:
        lengths = episode_lengths(WORKSPACE / val)
        out.append(f"\n## Validation\n\n`{val}`: {len(lengths)} episodes, {sum(lengths.values()):,} frames, {hours(sum(lengths.values()))} h.")

    pending = yaml.safe_load((HERE / "pending.yaml").read_text()) if (HERE / "pending.yaml").is_file() else None
    out.append("\n## Built, not in the config\n\nFrom `lerobot/dataset_info/pending.yaml`.\n")
    if pending:
        out.append("| root | episodes | frames | hours | note |\n|---|---|---|---|---|")
        for p in pending:
            lengths = episode_lengths(WORKSPACE / p["root"])
            out.append(f"| `{p['root']}` | {len(lengths)} | {sum(lengths.values()):,} | {hours(sum(lengths.values()))} | {p.get('note', '')} |")
    else:
        out.append("None.")
    seg, obj = subtask_tables(cfg, shares)
    out.append("\n## Objects (sampled ReBot episodes)\n\nFrom each root's `meta/episode_metadata.parquet`, groups with share > 0. "
               "Cycle = consecutive segments of one episode on the same object, split at each fresh grasp (grasp, move, release = 1; "
               "back-to-back regrasps = 1; each bit kit flap = 1; a retry after a failed close or insert = 1 more). "
               "Object = family matched in the subtask string (`FAMILIES` in data_count.py), else the noun phrase. Frames = segment frames, "
               "return to home not counted. Variants = the noun phrase of each cycle's first segment. Full subtask list: `SUBTASKS.md`.\n")
    out.append("| object | grasp cycles | internal | external | episodes | frames | hours | variants (cycles) |\n|---|---|---|---|---|---|---|---|")
    for k, (n, eps, f, i, x, var) in sorted(obj.items(), key=lambda kv: (-kv[1][0], kv[0])):
        v = ", ".join(f"{a} {b}" for a, b in sorted(var.items(), key=lambda ab: -ab[1])) if list(var) != [k] else ""
        out.append(f"| {k} | {n} | {i} | {x} | {len(eps)} | {f:,} | {hours(f)} | {v} |")
    verbs = defaultdict(lambda: [0, 0])
    for k, (n, _, f, _, _) in seg.items():
        v = "return to home" if k == "return to home" else k.split(" ")[0]
        verbs[v][0] += n
        verbs[v][1] += f
    out.append("\n## Subtasks by verb (sampled ReBot episodes)\n\n| verb | segments | frames | hours |\n|---|---|---|---|")
    out += [f"| {k} | {n} | {f:,} | {hours(f)} |" for k, (n, f) in sorted(verbs.items(), key=lambda kv: -kv[1][0])]
    general = defaultdict(lambda: [0, set(), 0, 0, 0, 0])  # (object, "verb the object") -> segments, episodes, frames, int, ext, strings
    for k, (n, eps, f, i, x) in seg.items():
        o = subtask_object(k)
        if o is None:
            continue
        verb = " ".join(k.split()[:2]) if re.match(r"(?:line up|shake out) ", k) else k.split()[0]
        g = general[(o[0], f"{verb} the {o[0]}")]
        g[0] += n
        g[1] |= eps
        g[2] += f
        g[3] += i
        g[4] += x
        g[5] += 1
    totals = defaultdict(lambda: [0, set(), 0, 0, 0])
    for (o, _), (n, eps, f, i, x, _) in general.items():
        t = totals[o]
        t[0] += n
        t[1] |= eps
        t[2] += f
        t[3] += i
        t[4] += x
    sub = [f"# Subtasks\n\nGenerated {datetime.now():%Y-%m-%d %H:%M} by `data_count.py` from `{args.config}`, sampled ReBot episodes. "
           "Object and the per-object grasp cycles: DATA.md. Segments split into internal / external ReBot; return to home not listed.\n",
           "## By object\n\nGeneral subtask = verb + object, all wordings merged (\"grasp the bit nearest the box\" and \"grasp the leftmost bit\" "
           "-> grasp the bit; every destination of \"move the bit to ...\" -> move the bit). Strings = distinct wordings merged.\n",
           "| object | general subtask | segments | internal | external | episodes | frames | hours | strings |\n|---|---|---|---|---|---|---|---|---|"]
    for o, (n, eps, f, i, x) in sorted(totals.items(), key=lambda kv: (-kv[1][0], kv[0])):
        sub.append(f"| **{o}** | **all** | **{n}** | **{i}** | **{x}** | **{len(eps)}** | **{f:,}** | **{hours(f)}** | |")
        rows = sorted(((k, v) for k, v in general.items() if k[0] == o), key=lambda kv: -kv[1][0])
        sub += [f"| | {g} | {n} | {i} | {x} | {len(eps)} | {f:,} | {hours(f)} | {m} |" for (_, g), (n, eps, f, i, x, m) in rows]
    sub.append("\n## Every subtask string\n\n| object | subtask | segments | internal | external | episodes | frames |\n|---|---|---|---|---|---|---|")
    keyed = sorted(seg.items(), key=lambda kv: ((subtask_object(kv[0]) or ("~",))[0], -kv[1][0], kv[0]))
    sub += [f"| {(subtask_object(k) or ('',))[0]} | {k} | {n} | {i} | {x} | {len(eps)} | {f:,} |" for k, (n, eps, f, i, x) in keyed]
    (HERE / "SUBTASKS.md").write_text("\n".join(sub) + "\n")
    (HERE / "DATA.md").write_text("\n".join(out) + "\n")
    print((HERE / "DATA.md").read_text())


if __name__ == "__main__":
    main()

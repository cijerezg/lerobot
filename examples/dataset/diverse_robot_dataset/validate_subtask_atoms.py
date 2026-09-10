#!/usr/bin/env python
"""Validate the atomic subtask layer of the diverse corpus.

    uv run python lerobot/examples/dataset/diverse_robot_dataset/validate_subtask_atoms.py \
        [--root outputs/diverse_robot_dataset] [--strict]

Checks, for corpus/subtask_atoms.jsonl and fmb/subtask_atoms.jsonl against the untouched
critic_intervals.jsonl of each store:

* every parent interval has children, and the children tile it exactly: integer
  boundaries, first start = parent start, each start = previous end, last end = parent end;
* every subtask string is one verb + one object in the closed grammar, with no ordinal,
  count or progress word, and equals the rendering of its fields;
* every reviewed mistake event of a parent lands in exactly one child (and inside it);
* quality is an integer 1-5, inherited values equal the parent's, reviewed values never
  exceed the parent except for siblings of an event-driven 1-2 parent;
* per-source counts and the verb x source table are printed.

This file is also the single source of the grammar vocabulary used by the annotation
scripts under migration/subtask_atoms_2026-09-08/.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

PICK_PLACE_VERBS = ("grasp", "move", "release", "return")
CONTACT_VERBS = (
    "wipe", "press", "pour", "water", "turn on", "turn off", "fold", "unfold", "stir", "scrub",
    "insert", "open", "close", "push", "pull", "spread", "straighten", "rotate", "lift", "place",
    "hang", "tilt", "shake", "flatten", "flip", "slide", "drag", "hold",
)
VERBS = PICK_PLACE_VERBS + CONTACT_VERBS
PREPOSITIONS = ("in", "on", "into", "onto", "to", "under", "over", "beside", "through", "from", "off", "at")
PROGRESS_WORDS = frozenset(
    {
        "first", "second", "third", "fourth", "fifth", "next", "remaining", "another", "last",
        "again", "other", "more", "all", "both", "1st", "2nd", "3rd", "4th", "final", "previous",
        "rest", "twice", "once", "then", "further", "additional", "each", "every", "one", "two",
        "three", "four", "five", "six",
    }
)
ARTICLES = ("the ", "a ", "an ")
BOUNDARY_PROVENANCE = ("gripper_event", "arm_settle", "vision", "parent")
QUALITY_PROVENANCE = ("inherited", "reviewed")


def render_subtask(atom: dict) -> str:
    """The only way a subtask string is produced."""
    verb, obj, cont = atom["verb"], atom.get("object"), atom.get("container")
    prep = atom.get("preposition")
    inst = atom.get("instrument")
    if verb == "return":
        return "return to home"
    if verb == "grasp":
        text = f"grasp {obj}"
    elif verb == "move":
        text = f"move {obj} {prep or 'to'} {cont}" if cont else f"lift {obj}"
    elif verb == "release":
        text = f"release {obj} {prep or 'in'} {cont}"
    else:
        text = f"{verb} {obj}"
        if cont:
            text += f" {prep or 'on'} {cont}"
    if inst:
        text += f" with {inst}"
    return text


def phrase_errors(name: str, phrase, where: str) -> list[str]:
    errors = []
    if not isinstance(phrase, str) or not phrase:
        return [f"{where}: {name} missing"]
    if not phrase.startswith(ARTICLES):
        errors.append(f"{where}: {name} '{phrase}' must start with an article")
    words = re.findall(r"[a-z0-9'-]+", phrase.lower())
    bad = [w for w in words if w in PROGRESS_WORDS]
    if bad:
        errors.append(f"{where}: {name} '{phrase}' carries progress/ordinal words {bad}")
    if " and " in f" {phrase} ":
        errors.append(f"{where}: {name} '{phrase}' names two things")
    if re.search(r"\d", phrase):
        errors.append(f"{where}: {name} '{phrase}' contains a digit")
    return errors


def subtask_errors(atom: dict, where: str) -> list[str]:
    errors = []
    verb = atom.get("verb")
    if verb not in VERBS:
        return [f"{where}: verb '{verb}' not in the grammar"]
    if verb == "return":
        if atom.get("object") or atom.get("container"):
            errors.append(f"{where}: return takes no object")
    else:
        errors += phrase_errors("object", atom.get("object"), where)
        if atom.get("container") is not None:
            errors += phrase_errors("container", atom.get("container"), where)
        if atom.get("instrument") is not None:
            errors += phrase_errors("instrument", atom.get("instrument"), where)
        if verb == "release" and not atom.get("container"):
            errors.append(f"{where}: release needs a container")
        if verb == "grasp" and atom.get("container"):
            errors.append(f"{where}: grasp takes no container")
        if atom.get("preposition") not in (None, *PREPOSITIONS):
            errors.append(f"{where}: preposition '{atom.get('preposition')}' not allowed")
    rendered = render_subtask(atom)
    if atom.get("subtask") != rendered:
        errors.append(f"{where}: subtask '{atom.get('subtask')}' != rendered '{rendered}'")
    # one verb: no second grammar verb inside the object/container/instrument phrases
    tail = rendered.split(" ", 1)[1] if " " in rendered else ""
    for v in VERBS:
        if v in ("open", "close", "lift", "hold", "place", "press", "turn on", "turn off"):
            continue  # common nouns/adjectives collide ("the open box", "the lid")
        pattern = rf"\b{re.escape(v)}\b"
        if v == "water":
            # "water bottle" / "bowl of water" use the noun. An embedded
            # action still takes an object phrase, e.g. "water the plant".
            pattern += r"(?=\s+(?:the|a|an)\b)"
        if re.search(pattern, tail):
            errors.append(f"{where}: second verb '{v}' inside '{rendered}'")
    words = re.findall(r"[a-z0-9'-]+", rendered.lower())
    bad = [w for w in words if w in PROGRESS_WORDS]
    if bad:
        errors.append(f"{where}: subtask '{rendered}' carries progress words {bad}")
    return errors


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def event_key(e: dict) -> tuple:
    return (e.get("kind"), round(float(e.get("source_start_s", e.get("start_s"))), 3), round(float(e.get("source_end_s", e.get("end_s"))), 3))


def validate_store(root: Path, name: str, strict: bool) -> tuple[list[str], dict]:
    errors: list[str] = []
    atoms_path = root / "subtask_atoms.jsonl"
    if not atoms_path.exists():
        return [f"{name}: {atoms_path} missing"], {}
    atoms = read_jsonl(atoms_path)
    parents = read_jsonl(root / "critic_intervals.jsonl")
    by_parent: dict[tuple, list[dict]] = collections.defaultdict(list)
    for a in atoms:
        by_parent[(a["episode_id"], int(a["parent_interval_index"]))].append(a)
    stats = {
        "atoms": len(atoms),
        "parents": len(parents),
        "per_source": collections.Counter(),
        "verb_by_source": collections.defaultdict(collections.Counter),
        "boundary_provenance": collections.Counter(),
        "quality_before": collections.Counter(),
        "quality_after": collections.Counter(),
        "quality_frames_before": collections.Counter(),
        "quality_frames_after": collections.Counter(),
        "parents_split": 0,
        "parents_passthrough": 0,
        "unsure_parents": [],
        "mistake_events_in_children": 0,
        "new_mistake_events": 0,
    }
    for prow in parents:
        key = (prow["episode_id"], int(prow["interval_index"]))
        where = f"{name} {key[0]} P{key[1]}"
        children = sorted(by_parent.get(key, []), key=lambda a: int(a["atom_index"]))
        source = prow.get("source", name)
        pstart, pend = int(prow["start_timestep"]), int(prow["end_timestep_exclusive"])
        stats["quality_before"][int(prow["quality"])] += 1
        stats["quality_frames_before"][int(prow["quality"])] += pend - pstart
        if not children:
            errors.append(f"{where}: no atoms")
            continue
        if len(children) > 1:
            stats["parents_split"] += 1
        else:
            stats["parents_passthrough"] += 1
        cursor = pstart
        for i, a in enumerate(children):
            w = f"{where} atom {i}"
            if int(a["atom_index"]) != i:
                errors.append(f"{w}: atom_index {a['atom_index']} out of order")
            s, e = a["start_timestep"], a["end_timestep_exclusive"]
            if not (isinstance(s, int) and isinstance(e, int)):
                errors.append(f"{w}: boundaries must be integers ({s!r}, {e!r})")
                s, e = int(s), int(e)
            if s != cursor:
                errors.append(f"{w}: starts at {s}, expected {cursor}")
            if e <= s:
                errors.append(f"{w}: empty span [{s},{e})")
            cursor = e
            errors += subtask_errors(a, w)
            if a.get("boundary_provenance") not in BOUNDARY_PROVENANCE:
                errors.append(f"{w}: boundary_provenance '{a.get('boundary_provenance')}'")
            if a.get("quality_provenance") not in QUALITY_PROVENANCE:
                errors.append(f"{w}: quality_provenance '{a.get('quality_provenance')}'")
            q = a.get("quality")
            if not isinstance(q, int) or not 1 <= q <= 5:
                errors.append(f"{w}: quality {q!r}")
            else:
                pq = int(prow["quality"])
                if a.get("quality_provenance") == "inherited" and q != pq:
                    errors.append(f"{w}: inherited quality {q} != parent {pq}")
                event_driven = pq <= 2 and len(prow["mistake_events"]) > 0
                if q > pq and not event_driven:
                    errors.append(f"{w}: quality {q} above parent {pq}")
                has_event = bool(a.get("mistake_events"))
                if has_event and q > 2:
                    errors.append(f"{w}: contains a mistake event but quality {q} > 2")
                stats["quality_after"][q] += 1
                stats["quality_frames_after"][q] += e - s
            for me in a.get("mistake_events", []):
                ms, mend = float(me["start_s"]), float(me["end_s"])
                if not (s / float(prow.get("native_rate_hz", 10.0)) - 1e-6 <= ms <= mend <= e / float(prow.get("native_rate_hz", 10.0)) + 1e-6):
                    if not me.get("outside_parent"):
                        errors.append(f"{w}: mistake span {ms:.2f}-{mend:.2f}s outside the atom")
                if me.get("provenance") == "atom_review":
                    stats["new_mistake_events"] += 1
            stats["per_source"][source] += 1
            stats["verb_by_source"][source][a["verb"]] += 1
            stats["boundary_provenance"][a.get("boundary_provenance")] += 1
            if a.get("confidence") == "unsure" and i == 0:
                stats["unsure_parents"].append(f"{key[0]} P{key[1]}: {a.get('note', '')}")
        if cursor != pend:
            errors.append(f"{where}: atoms end at {cursor}, parent ends at {pend}")
        # parent mistake events land in exactly one child
        for pe in prow["mistake_events"]:
            hits = [
                (i, ce)
                for i, a in enumerate(children)
                for ce in a.get("mistake_events", [])
                if ce.get("provenance") == "parent" and event_key(ce) == event_key(pe)
            ]
            if len(hits) != 1:
                errors.append(f"{where}: parent mistake {pe.get('kind')} {pe['start_s']}-{pe['end_s']}s lands in {len(hits)} children")
            else:
                stats["mistake_events_in_children"] += 1
    for key in by_parent:
        if key not in {(p["episode_id"], int(p["interval_index"])) for p in parents}:
            errors.append(f"{name}: atoms reference unknown parent {key}")
    return errors, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/diverse_robot_dataset")
    ap.add_argument("--strict", action="store_true", help="fail on unsure parents too")
    args = ap.parse_args()
    root = Path(args.root)
    total_errors = 0
    for store in ("corpus", "fmb"):
        errors, stats = validate_store(root / store, store, args.strict)
        print(f"== {store}: {stats.get('atoms', 0)} atoms over {stats.get('parents', 0)} parents, "
              f"{stats.get('parents_split', 0)} split / {stats.get('parents_passthrough', 0)} passed through")
        if stats:
            print("   per source:", dict(stats["per_source"]))
            sources = sorted(stats["verb_by_source"])
            verbs = sorted({v for c in stats["verb_by_source"].values() for v in c})
            print("   verb x source:")
            print("      " + f"{'verb':12s}" + "".join(f"{s[:14]:>15s}" for s in sources))
            for v in verbs:
                print("      " + f"{v:12s}" + "".join(f"{stats['verb_by_source'][s][v]:15d}" for s in sources))
            print("   boundary provenance:", dict(stats["boundary_provenance"]))
            print("   quality (parents -> atoms):", dict(sorted(stats["quality_before"].items())), "->", dict(sorted(stats["quality_after"].items())))
            fb, fa = stats["quality_frames_before"], stats["quality_frames_after"]
            tb, ta = sum(fb.values()) or 1, sum(fa.values()) or 1
            print("   quality by frames %:", {q: round(100 * fb[q] / tb, 1) for q in sorted(fb)}, "->", {q: round(100 * fa[q] / ta, 1) for q in sorted(fa)})
            print(f"   parent mistake events mapped: {stats['mistake_events_in_children']}, new from the atom review: {stats['new_mistake_events']}")
            if stats["unsure_parents"]:
                print(f"   unsure parents ({len(stats['unsure_parents'])}):")
                for u in stats["unsure_parents"]:
                    print("      -", u)
                if args.strict:
                    errors = errors + [f"{store}: {len(stats['unsure_parents'])} unsure parents"]
        for e in errors:
            print("ERROR", e)
        total_errors += len(errors)
    print("OK" if total_errors == 0 else f"{total_errors} errors")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

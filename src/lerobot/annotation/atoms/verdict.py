"""Verdict files: what the reviewer decided per parent after looking at the sheets.

    uv run python -m lerobot.annotation.atoms.verdict skeleton --episode E [--force]
    uv run python -m lerobot.annotation.atoms.verdict check --episode E | --source S | --all
    uv run python -m lerobot.annotation.atoms.verdict show --episode E

A verdict is outputs/_annotation/subtask_atoms_review/verdicts/<episode_id>.json:

{
  "episode_id": "...", "reviewer": "...",
  "parents": [
    {"parent_interval_index": 0, "confidence": "confident" | "unsure", "note": "what was seen",
     "atoms": [
        {"start_timestep": 180, "end_timestep_exclusive": 266,
         "verb": "grasp", "object": "the red flower", "container": null, "preposition": null,
         "instrument": null,
         "quality": null, "quality_note": "",
         "new_mistake_events": [{"start_s": 12.3, "end_s": 14.1, "kind": "failed_close", "note": "..."}],
         "note": ""}
     ]}
  ]
}

The subtask string is rendered from verb/object/container/preposition/instrument by
`render_subtask`, never typed by hand, so the vocabulary stays closed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from lerobot.annotation.atoms import validate_subtask_atoms as validator
from lerobot.annotation.atoms.atoms_common import REVIEW, WORK, load_corpus, read_jsonl

VERDICTS = REVIEW / "verdicts"

VERBS = set(validator.VERBS)
PREPOSITIONS = set(validator.PREPOSITIONS)
PROGRESS_WORDS = set(validator.PROGRESS_WORDS)
render_subtask = validator.render_subtask
MISTAKE_KINDS = {"failed_close", "slip", "drop", "knock", "wrong_target"}


def check_phrase(name: str, phrase: str | None, errors: list[str], where: str, required: bool = True) -> None:
    if phrase is None or phrase == "":
        if required:
            errors.append(f"{where}: {name} missing")
        return
    if not isinstance(phrase, str):
        errors.append(f"{where}: {name} not a string")
        return
    if not (phrase.startswith("the ") or phrase.startswith("a ") or phrase.startswith("an ")):
        errors.append(f"{where}: {name} '{phrase}' must start with 'the ' (or 'a '/'an ')")
    words = re.findall(r"[a-z0-9'-]+", phrase.lower())
    bad = [w for w in words if w in PROGRESS_WORDS]
    if bad:
        errors.append(f"{where}: {name} '{phrase}' carries progress/ordinal words {bad}")
    if " and " in f" {phrase} ":
        errors.append(f"{where}: {name} '{phrase}' names two things ('and'); one object per atom")
    if re.search(r"\d", phrase):
        errors.append(f"{where}: {name} '{phrase}' contains a digit")


def check_verdict(verdict: dict, proposals_by_parent: dict, episode: dict, parents_rows: dict) -> list[str]:
    errors: list[str] = []
    eid = verdict.get("episode_id")
    if eid != episode["episode_id"]:
        errors.append(f"episode_id mismatch: {eid}")
    rate = float(episode["native_rate_hz"])
    seen = set()
    for pv in verdict.get("parents", []):
        pidx = pv.get("parent_interval_index")
        where = f"{eid} P{pidx}"
        if pidx not in proposals_by_parent:
            errors.append(f"{where}: unknown parent index")
            continue
        seen.add(pidx)
        prop = proposals_by_parent[pidx]
        prow = parents_rows[pidx]
        pstart, pend = prop["parent_start"], prop["parent_end"]
        if pv.get("confidence") not in ("confident", "unsure"):
            errors.append(f"{where}: confidence must be 'confident' or 'unsure'")
        atoms = pv.get("atoms", [])
        if not atoms:
            errors.append(f"{where}: no atoms")
            continue
        cursor = pstart
        for i, a in enumerate(atoms):
            w = f"{where} atom {i}"
            try:
                s, e = int(a["start_timestep"]), int(a["end_timestep_exclusive"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{w}: start/end must be integers")
                continue
            if s != cursor:
                errors.append(f"{w}: starts at {s}, expected {cursor} (atoms must tile the parent, no gaps or overlap)")
            if e <= s:
                errors.append(f"{w}: empty or negative span [{s},{e})")
            if e > pend:
                errors.append(f"{w}: ends at {e} beyond the parent end {pend}")
            cursor = e
            verb = a.get("verb")
            if verb not in VERBS:
                errors.append(f"{w}: verb '{verb}' not in the closed vocabulary {sorted(VERBS)}")
                continue
            if verb == "return":
                if a.get("object") or a.get("container"):
                    errors.append(f"{w}: 'return' takes no object or container")
            else:
                check_phrase("object", a.get("object"), errors, w)
                if verb == "release" and not a.get("container"):
                    errors.append(f"{w}: release needs a container")
                check_phrase("container", a.get("container"), errors, w, required=False)
                check_phrase("instrument", a.get("instrument"), errors, w, required=False)
                if a.get("preposition") not in (None, *PREPOSITIONS):
                    errors.append(f"{w}: preposition '{a.get('preposition')}' not allowed")
                if verb in ("grasp",) and a.get("container"):
                    errors.append(f"{w}: grasp takes no container")
            q = a.get("quality")
            if q is not None:
                if not isinstance(q, int) or not 1 <= q <= 5:
                    errors.append(f"{w}: quality must be null or an integer 1-5")
                elif not a.get("quality_note"):
                    errors.append(f"{w}: a reviewed quality needs a quality_note")
                else:
                    parent_q = int(prow["quality"])
                    event_driven = parent_q <= 2 and len(prow["mistake_events"]) > 0
                    if q > parent_q and not event_driven:
                        errors.append(f"{w}: quality {q} above the parent's {parent_q} (only siblings of an event-driven 1-2 parent may be graded on their own)")
            for me in a.get("new_mistake_events", []) or []:
                if me.get("kind") not in MISTAKE_KINDS:
                    errors.append(f"{w}: new mistake kind '{me.get('kind')}' not in {sorted(MISTAKE_KINDS)}")
                try:
                    ms, mend = float(me["start_s"]), float(me["end_s"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{w}: new mistake needs start_s/end_s")
                    continue
                if not (s / rate - 1e-6 <= ms < mend <= e / rate + 1e-6):
                    errors.append(f"{w}: new mistake {ms:.2f}-{mend:.2f}s must lie inside the atom [{s / rate:.2f},{e / rate:.2f})s")
                if not me.get("note"):
                    errors.append(f"{w}: new mistake needs a note")
                if mend - ms < 0.3:
                    errors.append(f"{w}: new mistake span {mend - ms:.1f}s shorter than 0.3 s; justify or fix")
        if cursor != pend:
            errors.append(f"{where}: atoms end at {cursor}, parent ends at {pend}")
        # minimum atom length, unless the parent itself is short
        if (pend - pstart) / rate >= 1.0:
            for i, a in enumerate(atoms):
                try:
                    if (int(a["end_timestep_exclusive"]) - int(a["start_timestep"])) / rate < 0.5:
                        errors.append(f"{where} atom {i}: shorter than 0.5 s; merge it into a neighbour")
                except (KeyError, TypeError, ValueError):
                    pass
        # event-driven 1-2 parents: siblings without a reviewed event need their own grade;
        # atoms carrying a new event need quality 1-2
        parent_q = int(prow["quality"])
        event_driven = parent_q <= 2 and len(prow["mistake_events"]) > 0
        for i, a in enumerate(atoms):
            try:
                s, e = int(a["start_timestep"]) / rate, int(a["end_timestep_exclusive"]) / rate
            except (KeyError, TypeError, ValueError):
                continue
            def _overlap(ev):
                return max(0.0, min(e, float(ev["end_s"])) - max(s, float(ev["start_s"])))
            holds_event = any(_overlap(ev) > 0 for ev in prow["mistake_events"])
            # the assembler gives an event to the child with the largest overlap; approximate here
            best = {}
            for ev in prow["mistake_events"]:
                ovs = [max(0.0, min(int(b["end_timestep_exclusive"]) / rate, float(ev["end_s"])) - max(int(b["start_timestep"]) / rate, float(ev["start_s"]))) for b in atoms]
                best[id(ev)] = max(range(len(atoms)), key=lambda k: ovs[k]) if max(ovs) > 0 else None
            holds_event = any(best[id(ev)] == i for ev in prow["mistake_events"])
            if event_driven and not holds_event and a.get("quality") is None:
                errors.append(f"{where} atom {i}: sibling of an event-driven quality-{parent_q} parent must be graded on its own (set quality 3/4/5 with a quality_note)")
            if a.get("new_mistake_events") and (a.get("quality") is None or int(a.get("quality")) > 2):
                errors.append(f"{where} atom {i}: carries a new mistake event, so quality must be 1 or 2")
    missing = set(proposals_by_parent) - seen
    if missing:
        errors.append(f"{eid}: verdict lacks parents {sorted(missing)}")
    return errors


def skeleton(episode_id: str) -> dict:
    proposals = [p for p in read_jsonl(WORK / "proposals.jsonl") if p["episode_id"] == episode_id]
    out = {"episode_id": episode_id, "reviewer": "", "parents": []}
    for p in proposals:
        atoms = []
        for a in p["atoms"]:
            atoms.append(
                {
                    "start_timestep": a["start_timestep"],
                    "end_timestep_exclusive": a["end_timestep_exclusive"],
                    "verb": a["verb"],
                    "object": None if a["verb"] != "return" else None,
                    "container": None,
                    "preposition": None,
                    "instrument": None,
                    "quality": None,
                    "quality_note": "",
                    "new_mistake_events": [],
                    "note": "",
                }
            )
        out["parents"].append(
            {
                "parent_interval_index": p["parent_interval_index"],
                "parent_subtask": p["parent_subtask"],
                "parent_quality": p["parent_quality"],
                "parent_span": [p["parent_start"], p["parent_end"]],
                "confidence": "confident",
                "note": "",
                "atoms": atoms,
            }
        )
    return out


def load_verdict(episode_id: str) -> dict | None:
    path = VERDICTS / f"{episode_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_check(episode_ids: list[str]) -> int:
    episodes, parents = load_corpus()
    proposals = read_jsonl(WORK / "proposals.jsonl")
    by_ep = {}
    for p in proposals:
        by_ep.setdefault(p["episode_id"], {})[p["parent_interval_index"]] = p
    n_err = 0
    n_missing = 0
    for eid in episode_ids:
        v = load_verdict(eid)
        if v is None:
            n_missing += 1
            print(f"MISSING {eid}")
            continue
        errors = check_verdict(v, by_ep[eid], episodes[eid], {r["interval_index"]: r for r in parents[eid]})
        if errors:
            n_err += len(errors)
            for e in errors:
                print("ERROR", e)
        else:
            unsure = [pv["parent_interval_index"] for pv in v["parents"] if pv.get("confidence") == "unsure"]
            print(f"OK {eid}" + (f"  (unsure parents {unsure})" if unsure else ""))
    print(f"{len(episode_ids)} episodes, {n_missing} missing verdicts, {n_err} errors")
    return 1 if (n_err or n_missing) else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("skeleton")
    s.add_argument("--episode", required=True)
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--episode", default=None)
    c.add_argument("--source", default=None)
    c.add_argument("--all", action="store_true")
    sh = sub.add_parser("show")
    sh.add_argument("--episode", required=True)
    args = ap.parse_args()
    episodes, parents = load_corpus()
    if args.cmd == "skeleton":
        path = VERDICTS / f"{args.episode}.json"
        if path.exists() and not args.force:
            print(f"exists: {path} (use --force to overwrite)")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(skeleton(args.episode), indent=1), encoding="utf-8")
        print(path)
        print(path.read_text())
    elif args.cmd == "check":
        if args.episode:
            ids = [args.episode]
        elif args.source:
            ids = [e for e, ep in episodes.items() if ep["source"] == args.source]
        else:
            ids = list(episodes)
        sys.exit(run_check(ids))
    elif args.cmd == "show":
        v = load_verdict(args.episode)
        ep = episodes[args.episode]
        rate = float(ep["native_rate_hz"])
        print(args.episode, "|", ep["task"])
        for pv in (v or {}).get("parents", []):
            print(f"P{pv['parent_interval_index']} {pv.get('confidence')} | {pv.get('parent_subtask')} | note: {pv.get('note')}")
            for a in pv["atoms"]:
                s, e = a["start_timestep"], a["end_timestep_exclusive"]
                print(f"   [{s},{e}) {s / rate:6.1f}-{e / rate:6.1f}s  {render_subtask(a):45s} q={a.get('quality')} {a.get('note', '')}")


if __name__ == "__main__":
    main()

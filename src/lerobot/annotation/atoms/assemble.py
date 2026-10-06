"""Assemble the atom layer from the proposals and the reviewed verdicts.

    uv run python -m lerobot.annotation.atoms.assemble [--allow-missing]

Writes (never in place, new files only):
  outputs/diverse_robot_dataset/corpus/subtask_atoms.jsonl + subtask_atoms_info.json
  outputs/diverse_robot_dataset/fmb/subtask_atoms.jsonl    + subtask_atoms_info.json
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import hashlib
import os
import tempfile
import sys
from pathlib import Path

from lerobot.annotation.atoms.atoms_common import CORPUS, FMB, GRAMMAR_VERSION, WORK, load_corpus, read_jsonl, write_jsonl
from lerobot.annotation.atoms.verdict import VERDICTS, check_verdict, load_verdict, render_subtask, validator

FMB_PRIMITIVE_MAP = {
    "grasp": {"verb": "grasp", "object": "the object", "container": None, "preposition": None},
    "regrasp": {"verb": "grasp", "object": "the object", "container": None, "preposition": None},
    "move_up": {"verb": "move", "object": "the object", "container": None, "preposition": None},  # renders 'lift the object'
    "go_to_board": {"verb": "move", "object": "the object", "container": "the board", "preposition": "to"},
    "place_on_fixture": {"verb": "place", "object": "the object", "container": "the fixture", "preposition": "on"},
    "insert": {"verb": "insert", "object": "the object", "container": "the board", "preposition": "into"},
    "rotate": {"verb": "rotate", "object": "the object", "container": None, "preposition": None},
}

OBJECT_NAMING_RULES = [
    "object = noun phrase with an article, the thing handled; colour or a distinguishing attribute only when several similar things are present; the task string's own names win over what the frames suggest",
    "container = destination of move/release (and target of insert/pour), named as in the task string; preposition 'to' for move, 'in' for release into a container, 'on' for surfaces, racks, plates and stacking targets, 'into' for insert/pour",
    "instrument only when a tool acts on the object: 'wipe the desk with the rag', 'grasp the lemon with the tongs'",
    "no ordinals, counts or progress words (first, next, remaining, another, again, ...), no digits, never two objects joined by 'and'; identical objects keep the same name in every cycle",
    "FMB single-object manipulation keeps 'the object' (one object per episode, the source names it only by a shape/size code)",
    "the rendered string is produced by validate_subtask_atoms.render_subtask from the fields, never typed by hand",
]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def place_events(events: list[dict], atoms: list[dict], rate: float, clip: bool) -> list[list[dict]]:
    """Map parent events into the atom with the largest overlap (nearest atom if none)."""
    per_atom = [[] for _ in atoms]
    for e in events:
        es, ee = float(e["start_s"]), float(e["end_s"])
        best, best_ov = None, 0.0
        for i, a in enumerate(atoms):
            ov = overlap(es, ee, a["start_timestep"] / rate, a["end_timestep_exclusive"] / rate)
            if ov > best_ov:
                best, best_ov = i, ov
        row = dict(e)
        row["source_start_s"] = es
        row["source_end_s"] = ee
        row["provenance"] = "parent"
        if best is None:
            # entirely outside the parent (e.g. an interruption after its end): nearest atom, unclipped
            dist = [min(abs(es - a["end_timestep_exclusive"] / rate), abs(ee - a["start_timestep"] / rate)) for a in atoms]
            best = int(min(range(len(atoms)), key=lambda i: dist[i]))
            row["outside_parent"] = True
        elif clip:
            a = atoms[best]
            cs, ce = max(es, a["start_timestep"] / rate), min(ee, a["end_timestep_exclusive"] / rate)
            if (cs, ce) != (es, ee):
                row["clipped_to_atom"] = True
            row["start_s"], row["end_s"] = round(cs, 6), round(ce, 6)
        per_atom[best].append(row)
    return per_atom


def provenance_table(proposal: dict) -> dict[int, str]:
    table = {}
    for e in proposal["gripper_events"]:
        table[int(e["frame"])] = "gripper_event"
        table[int(e["ramp_end"])] = "gripper_event"
        table[int(e["ramp_begin"])] = "gripper_event"
    for c in proposal["carries_in_parent"]:
        table[int(c["close"])] = "gripper_event"
        table[int(c["open_end"])] = "gripper_event"
        table[int(c["open_begin"])] = "gripper_event"
        table[int(c["arrival"])] = "arm_settle"
    for a in proposal["atoms"]:
        table.setdefault(int(a["start_timestep"]), a["start_provenance"])
    return table


def assemble_corpus(allow_missing: bool) -> tuple[list[dict], dict]:
    episodes, parents = load_corpus()
    proposals = {(p["episode_id"], p["parent_interval_index"]): p for p in read_jsonl(WORK / "proposals.jsonl")}
    rows = []
    info = {
        "missing_verdicts": [],
        "verdict_errors": [],
        "unsure_parents": [],
        "reviewers": collections.Counter(),
        "new_mistake_events_dropped_as_duplicates": [],
        "quality_overrides": 0,
    }
    for eid, ep in episodes.items():
        rate = float(ep["native_rate_hz"])
        verdict = load_verdict(eid)
        if verdict is None:
            info["missing_verdicts"].append(eid)
            if not allow_missing:
                continue
            verdict = {"episode_id": eid, "reviewer": "", "parents": []}
        else:
            errors = check_verdict(verdict, {k[1]: v for k, v in proposals.items() if k[0] == eid}, ep, {r["interval_index"]: r for r in parents[eid]})
            if errors:
                info["verdict_errors"] += errors
                if not allow_missing:
                    continue
        info["reviewers"][verdict.get("reviewer", "")] += 1
        vparents = {pv["parent_interval_index"]: pv for pv in verdict.get("parents", [])}
        for prow in parents[eid]:
            pidx = int(prow["interval_index"])
            proposal = proposals[(eid, pidx)]
            pv = vparents.get(pidx)
            if pv is None:
                if not allow_missing:
                    continue
                atoms_in = [
                    {**a, "object": "the object", "container": None, "quality": None, "quality_note": "", "new_mistake_events": [], "note": "UNREVIEWED proposal"}
                    for a in proposal["atoms"]
                ]
                confidence, pnote = "unsure", "no verdict"
            else:
                atoms_in = pv["atoms"]
                confidence, pnote = pv.get("confidence", "unsure"), pv.get("note", "")
            if confidence == "unsure":
                info["unsure_parents"].append({"episode_id": eid, "parent_interval_index": pidx, "parent_subtask": prow["normalized_description"], "note": pnote})
            table = provenance_table(proposal)
            pstart, pend = int(prow["start_timestep"]), int(prow["end_timestep_exclusive"])
            parent_q = int(prow["quality"])
            event_driven = parent_q <= 2 and len(prow["mistake_events"]) > 0
            placed = {
                k: place_events(prow[k], atoms_in, rate, clip=(k == "mistake_events"))
                for k in ("mistake_events", "pause_events", "interruption_events", "recovery_events")
            }
            for i, a in enumerate(atoms_in):
                s, e = int(a["start_timestep"]), int(a["end_timestep_exclusive"])
                fields = {
                    "verb": a["verb"],
                    "object": a.get("object"),
                    "container": a.get("container"),
                    "preposition": a.get("preposition"),
                    "instrument": a.get("instrument"),
                }
                subtask = render_subtask(fields)
                start_prov = "parent" if s == pstart else table.get(s, "vision")
                if i + 1 < len(atoms_in):
                    nxt = int(atoms_in[i + 1]["start_timestep"])
                    end_prov = table.get(nxt, "vision")
                else:
                    end_prov = "parent"
                mistakes = placed["mistake_events"][i]
                new_events = []
                for me in a.get("new_mistake_events", []) or []:
                    dup = any(overlap(float(me["start_s"]), float(me["end_s"]), float(x["start_s"]), float(x["end_s"])) > 0 for x in mistakes)
                    if dup:
                        info["new_mistake_events_dropped_as_duplicates"].append({"episode_id": eid, "parent": pidx, "atom": i, "event": me})
                        continue
                    new_events.append({"type": "mistake", "kind": me["kind"], "start_s": float(me["start_s"]), "end_s": float(me["end_s"]), "note": me.get("note", ""), "provenance": "atom_review"})
                mistakes = mistakes + new_events
                q = a.get("quality")
                if q is None:
                    quality, qprov = parent_q, "inherited"
                    if event_driven and not mistakes:
                        qprov = "inherited"  # sibling left ungraded: checker demands a grade, so this is rare
                else:
                    quality, qprov = int(q), "reviewed"
                    info["quality_overrides"] += 1
                note = a.get("note", "") or ""
                if a.get("quality_note"):
                    note = (note + " | " if note else "") + "quality: " + a["quality_note"]
                rows.append(
                    {
                        "episode_id": eid,
                        "source": ep["source"],
                        "component": ep["component"],
                        "embodiment": ep["embodiment"],
                        "split": ep["split"],
                        "native_rate_hz": rate,
                        "parent_interval_index": pidx,
                        "parent_subtask": prow["normalized_description"],
                        "parent_quality": parent_q,
                        "parent_critic_eligible": bool(prow["critic_eligible"]),
                        "atom_index": i,
                        "start_timestep": s,
                        "end_timestep_exclusive": e,
                        "start_s": round(s / rate, 6),
                        "end_s_exclusive": round(e / rate, 6),
                        "duration_s": round((e - s) / rate, 6),
                        **fields,
                        "subtask": subtask,
                        "primitive": None,
                        "quality": quality,
                        "quality_provenance": qprov,
                        "mistake_events": mistakes,
                        "pause_events": placed["pause_events"][i],
                        "interruption_events": placed["interruption_events"][i],
                        "recovery_events": placed["recovery_events"][i],
                        "boundary_provenance": start_prov,
                        "end_boundary_provenance": end_prov,
                        "confidence": confidence,
                        "note": note,
                        "parent_note": pnote,
                    }
                )
    return rows, info


def fmb_event(x: dict, rate: float) -> dict:
    """FMB events are timestep-native ({start_timestep, end_timestep_exclusive, mistake_type}); carry them in the corpus shape with the native keys kept."""
    s, e = round(x["start_timestep"] / rate, 6), round(x["end_timestep_exclusive"] / rate, 6)
    return {
        "type": "mistake",
        "kind": x["mistake_type"],
        "start_s": s,
        "end_s": e,
        "note": x.get("note", ""),
        "start_timestep": int(x["start_timestep"]),
        "end_timestep_exclusive": int(x["end_timestep_exclusive"]),
        "source_start_s": s,
        "source_end_s": e,
        "provenance": "parent",
    }


def assemble_fmb() -> list[dict]:
    rows = []
    episodes = {e["episode_id"]: e for e in read_jsonl(FMB / "episodes.jsonl")}
    for prow in read_jsonl(FMB / "critic_intervals.jsonl"):
        ep = episodes[prow["episode_id"]]
        m = FMB_PRIMITIVE_MAP[prow["primitive"]]
        s, e = int(prow["start_timestep"]), int(prow["end_timestep_exclusive"])
        rate = float(ep.get("nominal_fps", 10.0))
        rows.append(
            {
                "episode_id": prow["episode_id"],
                "source": "fmb",
                "component": "fmb",
                "embodiment": "Franka",
                "split": prow["split"],
                "native_rate_hz": rate,
                "parent_interval_index": int(prow["interval_index"]),
                "parent_subtask": prow["normalized_description"],
                "parent_quality": int(prow["quality"]),
                "parent_critic_eligible": bool(prow["critic_eligible"]),
                "atom_index": 0,
                "start_timestep": s,
                "end_timestep_exclusive": e,
                "start_s": round(s / rate, 6),
                "end_s_exclusive": round(e / rate, 6),
                "duration_s": round((e - s) / rate, 6),
                **m,
                "instrument": None,
                "subtask": render_subtask(m),
                "primitive": prow["primitive"],
                "quality": int(prow["quality"]),
                "quality_provenance": "inherited",
                "mistake_events": [fmb_event(x, rate) for x in prow["mistake_events"]],
                "pause_events": [dict(x, provenance="parent") for x in prow.get("pause_events", [])],
                "interruption_events": [dict(x, provenance="parent") for x in prow["interruption_events"]],
                "recovery_events": [dict(x, provenance="parent") for x in prow["recovery_events"]],
                "boundary_provenance": "parent",
                "end_boundary_provenance": "parent",
                "confidence": "confident",
                "note": f"FMB primitive '{prow['primitive']}' passed through unchanged; quality {prow['quality']} from the FMB production review",
                "parent_note": "",
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    per_source = collections.Counter(r["source"] for r in rows)
    verb_by_source = collections.defaultdict(collections.Counter)
    prov = collections.Counter(r["boundary_provenance"] for r in rows)
    interior = collections.Counter(r["boundary_provenance"] for r in rows if r["atom_index"] > 0)
    for r in rows:
        verb_by_source[r["source"]][r["verb"]] += 1
    parents = {(r["episode_id"], r["parent_interval_index"]) for r in rows}
    n_children = collections.Counter((r["episode_id"], r["parent_interval_index"]) for r in rows)
    qb = collections.Counter()
    qa = collections.Counter(r["quality"] for r in rows)
    seen = set()
    for r in rows:
        key = (r["episode_id"], r["parent_interval_index"])
        if key not in seen:
            seen.add(key)
            qb[r["parent_quality"]] += 1
    return {
        "atoms": len(rows),
        "parents": len(parents),
        "parents_split": sum(1 for v in n_children.values() if v > 1),
        "parents_passthrough": sum(1 for v in n_children.values() if v == 1),
        "atoms_per_source": dict(per_source),
        "verb_by_source": {s: dict(c) for s, c in verb_by_source.items()},
        "boundary_provenance_all_atoms": dict(prov),
        "boundary_provenance_interior_cuts": dict(interior),
        "quality_parents": dict(sorted(qb.items())),
        "quality_atoms": dict(sorted(qa.items())),
        "quality_provenance": dict(collections.Counter(r["quality_provenance"] for r in rows)),
        "mistake_events": {
            "from_parents": sum(1 for r in rows for e in r["mistake_events"] if e.get("provenance") == "parent"),
            "new_from_atom_review": sum(1 for r in rows for e in r["mistake_events"] if e.get("provenance") == "atom_review"),
            "clipped_to_atom": sum(1 for r in rows for e in r["mistake_events"] if e.get("clipped_to_atom")),
            "outside_parent": sum(1 for r in rows for e in r["mistake_events"] if e.get("outside_parent")),
        },
        "confidence": dict(collections.Counter(r["confidence"] for r in rows if r["atom_index"] == 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true", help="write even if verdicts are missing or fail their check")
    args = ap.parse_args()
    corpus_rows, info = assemble_corpus(args.allow_missing)
    fmb_rows = assemble_fmb()
    if (info["missing_verdicts"] or info["verdict_errors"]) and not args.allow_missing:
        print(f"{len(info['missing_verdicts'])} missing verdicts, {len(info['verdict_errors'])} verdict errors; nothing written (use --allow-missing to write anyway)")
        for m in info["missing_verdicts"][:20]:
            print("  missing", m)
        for e in info["verdict_errors"][:40]:
            print("  error", e)
        sys.exit(1)
    today = dt.date.today().isoformat()
    common = {
        "grammar_version": GRAMMAR_VERSION,
        "created": today,
        "annotator": "Proprio proposals reviewed frame by frame by Claude (Fable 5.1) reviewing agents from the contact sheets, dense strips and closeups, with Codex visual corrections taking precedence where they exist; reviewer per parent is recorded in the verdict review_provenance. Machine visual annotation, not human approval.",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (CORPUS / "critic_intervals.jsonl", FMB / "critic_intervals.jsonl", WORK / "proposals.jsonl")},
        "verdict_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(VERDICTS.glob("*.json"))},
        "grammar": {
            "pick_and_place": {
                "grasp": "[end of previous release or parent start, gripper closes)  approach + closing; absorbs transit and failed closes",
                "move": "[gripper closed, arrival over the target)  rendered 'move the X to the C', or 'lift the X' with no destination",
                "release": "[arrival over the target, opening settles)  rendered 'release the X in|on the C'",
                "return": "[last release, parent end) final park only",
            },
            "contact_atoms": "non pick-and-place work keeps its verb, one atom per contact episode (press, turn on, wipe, water, pour, fold, unfold, stir, scrub, insert, open, close, push, pull, spread, straighten, rotate, place, hang, ...)",
            "verbs": list(validator.VERBS),
            "prepositions": list(validator.PREPOSITIONS),
            "forbidden_progress_words": sorted(validator.PROGRESS_WORDS),
            "rendering": "validate_subtask_atoms.render_subtask(fields)",
        },
        "object_naming_rules": OBJECT_NAMING_RULES,
        "boundary_provenance_values": {
            "parent": "the parent interval's own edge",
            "gripper_event": "a measured gripper close/open event (step in the measured gripper channel)",
            "arm_settle": "arrival over the target from proprio: the earlier of the base-yaw settling into 0.05 rad of its release value (when the carry swings the base > 0.3 rad) and the last frame above 40% of the carry's peak joint speed",
            "vision": "a frame the reviewer chose from the contact sheets or a dense strip",
        },
        "proprio_rules": {
            "gripper_closedness": "Franka/UR7e: measured position 0 open..1 closed; ARX5: 1 - width/0.0872 m; UR5: 1 - width/0.085 m",
            "event_detector": "step in closedness >= 0.20 (Franka, UR7e) / 0.15 (ARX5, UR5) between 0.4 s medians; event frame at the midpoint crossing",
            "failed_close": "closed run with arm-joint L2 displacement < 0.20 rad never lifted anything: stays inside the enclosing grasp; confirmed or rejected by vision",
            "park": "a closed run reaching the episode end within 3 s is the parked state (ARX5 rests shut)",
            "short_runs": "phase runs shorter than 0.5 s merge into their same-cycle neighbour",
        },
        "quality_rules": {
            "default": "inherit the parent's reviewed quality (quality_provenance = inherited)",
            "event_children": "an atom containing a parent mistake event keeps the parent's 1-2",
            "event_siblings": "siblings of an event-driven 1-2 parent are graded on their own by the reviewer (3 laboured, 4-5 direct), the only case a child exceeds its parent",
            "new_events": "an atom with a new visible failure gets 2 (one event) or 1",
            "never_above_parent_otherwise": True,
        },
        "event_mapping": "parent mistake/pause/interruption/recovery events go to the child with the largest time overlap; mistake spans are clipped to that child (clipped_to_atom=true); an event with no overlap (an interruption after the parent's end) is attached to the nearest child unclipped with outside_parent=true; source_start_s/source_end_s keep the parent's values",
    }
    corpus_info = {
        **common,
        "store": "corpus",
        "sources": ["droid", "droid_success", "robochallenge", "ur7e"],
        "what_proprio_did": "gripper events, carries vs failed closes, arrival, candidate atoms per parent (proposals.jsonl); overview and atom contact sheets from ffmpeg frames",
        "what_vision_did": "every parent's atoms, cuts, verbs, object and container names, confidence, quality overrides and new mistake events were decided by a reviewer looking at the sheets (verdict files); boundary provenance records which cuts the reviewer moved",
        "summary": summarize(corpus_rows),
        "unsure_parents": info["unsure_parents"],
        "missing_verdicts": info["missing_verdicts"],
        "verdict_errors": info["verdict_errors"],
        "reviewers": dict(info["reviewers"]),
        "new_mistake_events_dropped_as_duplicates": info["new_mistake_events_dropped_as_duplicates"],
    }
    fmb_info = {
        **common,
        "store": "fmb",
        "sources": ["fmb"],
        "fmb_primitive_mapping": {k: {**v, "subtask": render_subtask(v)} for k, v in FMB_PRIMITIVE_MAP.items()},
        "what_proprio_did": "nothing: FMB primitive intervals are source-native and already atomic; no re-cut",
        "what_vision_did": f"nothing new: the {len(fmb_rows)} intervals were visually reviewed in the FMB production passes (quality and mistake events come from those reviews, carried through unchanged); this layer only maps primitives onto the grammar",
        "summary": summarize(fmb_rows),
    }
    # Validate complete files against untouched parents before publishing either store.
    with tempfile.TemporaryDirectory(prefix="atoms-assemble-", dir=WORK) as temporary:
        staged = Path(temporary)
        for store, rows, metadata in ((CORPUS, corpus_rows, corpus_info), (FMB, fmb_rows, fmb_info)):
            folder = staged / store.name
            folder.mkdir()
            (folder / "critic_intervals.jsonl").symlink_to(store / "critic_intervals.jsonl")
            write_jsonl(folder / "subtask_atoms.jsonl", rows)
            (folder / "subtask_atoms_info.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
            errors, _ = validator.validate_store(folder, store.name, strict=False)
            if errors:
                raise ValueError("Assembly validation failed: " + "\n".join(errors))
        for store in (CORPUS, FMB):
            for name in ("subtask_atoms.jsonl", "subtask_atoms_info.json"):
                target = store / name
                if target.exists():
                    raise FileExistsError(f"{target} already exists; preserve the previous layer before rebuilding")
        for store in (CORPUS, FMB):
            for name in ("subtask_atoms.jsonl", "subtask_atoms_info.json"):
                os.replace(staged / store.name / name, store / name)
    print(f"corpus: {len(corpus_rows)} atoms; fmb: {len(fmb_rows)} atoms")
    print(json.dumps(corpus_info["summary"], indent=1))
    if info["missing_verdicts"]:
        print("MISSING verdicts:", len(info["missing_verdicts"]))
    if info["verdict_errors"]:
        print("VERDICT ERRORS:", len(info["verdict_errors"]))


if __name__ == "__main__":
    main()

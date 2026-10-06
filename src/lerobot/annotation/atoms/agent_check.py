"""Agent review files: skeleton, check, show.

    uv run python -m lerobot.annotation.atoms.agent_check skeleton --episode E [--force]
    uv run python -m lerobot.annotation.atoms.agent_check check --episode E
    uv run python -m lerobot.annotation.atoms.agent_check status

A review is outputs/_annotation/subtask_atoms_review/agent_reviews/<episode_id>.json, the
same shape as a manual override:

{"episode_id": E, "reviewer": "Claude visual review", "images": [sheet paths read],
 "parents": [{"parent_interval_index": 0, "confidence": "confident" | "unsure", "note": "...",
              "atoms": [{... same atom fields as a verdict ...}]}]}

Parents that already have a manual override (Codex visual corrections) are authoritative
and must not be reviewed here; the skeleton leaves them out and check ignores them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lerobot.annotation.atoms.atoms_common import REVIEW, WORK, load_corpus, read_jsonl
from lerobot.annotation.atoms.verdict import check_verdict, render_subtask, skeleton, validator

AGENT = REVIEW / "agent_reviews"
OVERRIDES = REVIEW / "manual_overrides"
REVIEWER = "Claude visual review"


def override_parents(episode_id: str) -> set[int]:
    path = OVERRIDES / f"{episode_id}.json"
    if not path.exists():
        return set()
    return {int(p["parent_interval_index"]) for p in json.loads(path.read_text())["parents"]}


def proposals_for(episode_id: str) -> dict[int, dict]:
    return {p["parent_interval_index"]: p for p in read_jsonl(WORK / "proposals.jsonl") if p["episode_id"] == episode_id}


def check_episode(episode_id: str, episodes, parents, verbose: bool = True) -> list[str]:
    path = AGENT / f"{episode_id}.json"
    if not path.exists():
        return [f"{episode_id}: no agent review at {path}"]
    try:
        review = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{episode_id}: invalid JSON: {exc}"]
    skip = override_parents(episode_id)
    props = {k: v for k, v in proposals_for(episode_id).items() if k not in skip}
    errors = []
    if review.get("reviewer") != REVIEWER:
        errors.append(f"{episode_id}: reviewer must be '{REVIEWER}'")
    if not review.get("images"):
        errors.append(f"{episode_id}: list the sheet images you read under 'images'")
    extra = [p["parent_interval_index"] for p in review.get("parents", []) if p["parent_interval_index"] in skip]
    if extra:
        errors.append(f"{episode_id}: parents {extra} have a manual override; remove them from the agent review")
    kept = {**review, "parents": [p for p in review.get("parents", []) if p["parent_interval_index"] not in skip]}
    errors += check_verdict(kept, props, episodes[episode_id], {r["interval_index"]: r for r in parents[episode_id]})
    for pv in kept["parents"]:
        if not (pv.get("note") or "").strip():
            errors.append(f"{episode_id} P{pv['parent_interval_index']}: note must say what you saw")
        for a in pv.get("atoms", []):
            try:
                errors += validator.subtask_errors(dict(a, subtask=render_subtask(a)), episode_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{episode_id} P{pv['parent_interval_index']}: cannot render atom {a}: {exc!r}")
    if verbose:
        ep = episodes[episode_id]
        rate = float(ep["native_rate_hz"])
        print(episode_id, "|", ep["task"])
        for pv in kept["parents"]:
            prop = props.get(pv["parent_interval_index"])
            head = f"P{pv['parent_interval_index']} {pv.get('confidence')}"
            if prop:
                head += f" | parent [{prop['parent_start']},{prop['parent_end']}) q{prop['parent_quality']} \"{prop['parent_subtask']}\""
            print(head)
            print("   note:", pv.get("note", ""))
            for a in pv.get("atoms", []):
                try:
                    s, e = int(a["start_timestep"]), int(a["end_timestep_exclusive"])
                    text = render_subtask(a)
                except Exception:  # noqa: BLE001
                    print("   (unrenderable atom)", a)
                    continue
                flags = []
                if a.get("quality") is not None:
                    flags.append(f"q={a['quality']} ({a.get('quality_note', '')})")
                for me in a.get("new_mistake_events", []) or []:
                    flags.append(f"NEW {me.get('kind')} {me.get('start_s')}-{me.get('end_s')}s")
                print(f"   [{s},{e}) {s / rate:6.1f}-{e / rate:6.1f}s  {text:48s} {' '.join(flags)}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("skeleton")
    s.add_argument("--episode", required=True)
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--episode", required=True)
    sub.add_parser("status")
    args = ap.parse_args()
    episodes, parents = load_corpus()
    AGENT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "skeleton":
        path = AGENT / f"{args.episode}.json"
        if path.exists() and not args.force:
            print(f"exists: {path} (use --force to overwrite)")
            return
        sk = skeleton(args.episode)
        skip = override_parents(args.episode)
        out = {
            "episode_id": args.episode,
            "reviewer": REVIEWER,
            "images": [],
            "parents": [p for p in sk["parents"] if p["parent_interval_index"] not in skip],
        }
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(path)
        if skip:
            print(f"parents {sorted(skip)} have a manual override and are left out")
        print(path.read_text())
    elif args.cmd == "check":
        errors = check_episode(args.episode, episodes, parents)
        for e in errors:
            print("ERROR", e)
        print("OK" if not errors else f"{len(errors)} errors")
        sys.exit(1 if errors else 0)
    elif args.cmd == "status":
        done, missing, bad = [], [], {}
        for eid in episodes:
            props = set(proposals_for(eid)) - override_parents(eid)
            if not props:
                done.append(eid)
                continue
            errors = check_episode(eid, episodes, parents, verbose=False)
            if errors and errors[0].endswith("no agent review at " + str(AGENT / f"{eid}.json")):
                missing.append(eid)
            elif errors:
                bad[eid] = errors
            else:
                done.append(eid)
        print(f"complete {len(done)}  missing {len(missing)}  with errors {len(bad)}")
        for eid, errs in bad.items():
            print("ERRORS", eid, errs[:3])


if __name__ == "__main__":
    main()

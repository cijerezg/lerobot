"""Grading batches for one pool (step 3): classes/grade_batches_<pool>.json.

uv run python -m lerobot.annotation.quality_v2.grade_batches rebot_main
Batches of at most 12 units of one sheet class, in sheet order. precision_hint per unit: precision_v1; for a carry,
the precision_v1 of the first later release before the next grasp (rubric 6: the end of a carry takes the release's level).
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
HERE = QUALITY_V2
CLASSES = HERE / "classes"
EXTRA = {"rebot_main__release__deformable_to_surface": "rebot_main__release__sock_to_basket"}  # graded with that FINAL.md
SEG_RE = re.compile(r"seg(\d+)$")
ATOM_RE = re.compile(r"p(\d+)a(\d+)$")


def short(slug, pool):
    s = slug[len(pool) + 2:].replace("__", "_").replace("_to_", "_")
    return s.replace("release", "rel")


def unit_order(row):
    """Chronological key for built units and source rows used by dry-run checks."""
    if row.get("from_index") is not None:
        return (0, int(row["from_index"]), int(row.get("to_index", row["from_index"])))
    match = SEG_RE.fullmatch(row["unit"])
    if match:
        return (1, int(match.group(1)), 0)
    match = ATOM_RE.fullmatch(row["unit"])
    if match:
        return (2, int(match.group(1)), int(match.group(2)))
    raise ValueError(f"unexpected unit id without frame indices: {row['unit']}")


def precision_hints(rows):
    """Return each unit's hint, pairing carries to the first eligible release.

    A later grasp starts a new manipulation chain, so a release after it must
    not supply the earlier carry's precision. This mirrors the validated DROID
    pairing rule while accepting ReBot ``segN`` and diverse ``pXaY`` IDs.
    """
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[(row["dataset"], row["episode"])].append(row)

    hints = {}
    for episode_rows in by_episode.values():
        episode_rows.sort(key=unit_order)
        for index, row in enumerate(episode_rows):
            level = row.get("precision_v1")
            if row["action"] == "carry":
                for later in episode_rows[index + 1:]:
                    if later["action"] == "release":
                        level = later.get("precision_v1")
                        break
                    if later["action"] == "grasp":
                        break
            hints[row["uid"]] = level
    return hints


def load_units(pool, classes=CLASSES):
    units = {}
    files = sorted(classes.glob(f"{pool}__*/units.jsonl"))
    for path in files:
        with path.open() as handle:
            for line in handle:
                unit = json.loads(line)
                unit["slug"] = path.parent.name
                units[unit["uid"]] = unit
    return units, files


def build_batches(pool, classes=CLASSES):
    """Build a pool manifest in memory without writing it."""
    units, files = load_units(pool, classes)
    hints = precision_hints(units.values())
    batches = {}
    for path in files:
        slug = path.parent.name
        if slug in EXTRA:
            continue
        order = []
        index_path = path.parent / "sheets" / "index.txt"
        if index_path.exists():
            with index_path.open() as handle:
                for line in handle:
                    order += line.split(":", 1)[1].split()
        with path.open() as handle:
            uids = [json.loads(line)["uid"] for line in handle]
        uids = [uid for uid in order if uid in uids] + [uid for uid in uids if uid not in order]
        extra = [uid for uid in units if units[uid]["slug"] in EXTRA and EXTRA[units[uid]["slug"]] == slug]
        count = max(1, math.ceil(len(uids) / 12))
        chunks = [uids[i * len(uids) // count:(i + 1) * len(uids) // count] for i in range(count)]
        chunks[-1] += extra
        for number, chunk in enumerate(chunks, 1):
            batches[f"G_{short(slug, pool)}_{number:02d}"] = {
                "class": slug,
                "final": f"classes/{EXTRA.get(slug, slug)}/strategy/FINAL.md",
                "units": [
                    {"uid": uid, "class": units[uid]["slug"], "precision_hint": hints[uid]}
                    for uid in chunk
                ],
            }
    return batches


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: grade_batches.py <pool>")
    pool = argv[0]
    batches = build_batches(pool)
    out = CLASSES / f"grade_batches_{pool}.json"
    with out.open("w") as handle:
        json.dump(batches, handle, indent=1)
    print(out, len(batches), "batches,", sum(len(batch["units"]) for batch in batches.values()), "units")


if __name__ == "__main__":
    main()

"""Consolidate the independent round-1 sampled-audit verdicts.

This is a result writer, not annotation policy. It exists because the workspace
sandbox failed every in-place apply_patch during the parallel review.
"""

from __future__ import annotations

import json
from pathlib import Path

WORK = Path("migration/diverse_annotation_audit_2026-10-07")


def issue(parent, atom, frames, field, evidence, correction, severity="wrong", looked="dense"):
    return {
        "parent_interval_index": parent,
        "atom_index": atom,
        "frames": list(frames),
        "field": field,
        "visible_evidence": evidence,
        "proposed_correction": correction,
        "confidence": "sure",
        "severity": severity,
        "looked_at": looked,
    }


RESULTS = {
    "droid_success__CLVR__ep009563": {
        "decision_note": (
            "Clean after source metadata, all eight pages, trace, trainer-facing sidecars, and class "
            "references. Retained [90,299) is continuous; text/boundaries/contact/precision are supported. "
            "The visible knock [173,195) and its q2 attempt plus near-cap q3 strategy are correct."
        ),
        "issues": [],
    },
    "droid__WEIRD__ep013616": {
        "decision_note": (
            "All pages, trace, metadata, effective sidecars, class references and dense tail reviewed. "
            "Grasp execution is q4 top-pinch p3 and no mistake is visible; target text, retained release "
            "tail, contact and precision commit are wrong."
        ),
        "issues": [
            issue(0, 0, (90, 183), "subtask_text.object_identity",
                  "Several purple foam pieces are visible; the selected upright target sits in the packed cluster between green pieces.",
                  "Use 'grasp the upright purple foam piece between the green pieces' for both split grasp atoms; later move/release may drop the modifier."),
            issue(2, 0, (332, 350), "retained_intervals",
                  "The current cut ends while the foam is held; commanded open begins f334, measured opening is f336-342, and the foam lands in the bag by about f350.",
                  "Extend the retained parent/atom end to f350; keep [350,364) excluded and remap dependent rows."),
            issue(2, 0, (240, 350), "contact",
                  "The final step opens while the foam is unsupported over the bag; na contradicts the visible release.",
                  "Set contact to code 10 drop over the repaired release atom."),
            issue(2, 0, (225, 350), "precision_window",
                  "The existing commit/end f332 precede the actual measured open; the bag target supports level 2.",
                  "Keep precision 2 and from_index 225; set commit_index 336 and to_index 350."),
        ],
    },
    "molmoact__household__ep002456": {
        "decision_note": (
            "All four pages, trace, metadata, effective sidecars and dense commits reviewed. Text, target, "
            "retention, release contact/precision, no mistakes and the exemplary grasp are supported."
        ),
        "issues": [
            issue(0, "0/1", (39, 46), "subtask_boundary/contact",
                  "At f39 the banana remains on the table and the gripper is still closing (0.3806 at f39, 0.6302 at f44, closed plateau near f46).",
                  "Move grasp-to-move boundary from f39 to f46 and remap every atom-keyed sidecar."),
            issue(0, 2, (92, 93), "precision_window_containment",
                  "Precision window [54,93) exceeds the retained/native interval [0,92); f92 is excluded.",
                  "Clip precision-window to_index from 93 to 92.", "minor", "effective_labels"),
        ],
    },
    "robochallenge__pick_out_the_green_blocks__ep000285": {
        "decision_note": (
            "Metadata, 17 pages, trace, effective sidecars, actor histories and closeups reviewed. ARX5 "
            "gripper is measured width (larger=open). Both grasps/releases succeed; no mistake is visible."
        ),
        "issues": [
            issue(0, 0, (180, 233), "subtask",
                  "Two green blocks are active at f180; the first selected block is the one nearest the blue basket.",
                  "Use 'grasp the green block nearest the basket'; the later grasp may stay generic."),
            issue(0, 0, (180, 360), "actor_history",
                  "Source metadata rejects [0,180), but anchor f180 history is [0,30,60,90,120,150,180] and anchors through f354 still include rejected frames.",
                  "Rebuild actor anchors so history stays in one retained interval; delay anchors until full history is available."),
            issue(0, 0, (150, 180), "quality_containment",
                  "Padded quality rows start in the rejected prefix although their raw spans start at/after f180.",
                  "Clamp materialized quality from_index to f180 while preserving raw spans.", "minor", "effective_labels"),
            issue(0, 0, (166, 180), "precision_window_containment",
                  "The precision-window lead starts in rejected pre-roll.",
                  "Clamp precision-window from_index to f180.", "minor", "effective_labels"),
            issue(0, "0/1", (233, 240), "boundary",
                  "ARX5 width is still closing at f233 and reaches the held-block plateau around f240.",
                  "Move grasp-to-move boundary to about f240 and remap dependent channels.", "minor"),
            issue(0, "2/3", (484, 490), "boundary",
                  "The first release continues opening after f484 and settles around f489-490.",
                  "Move release-to-next-grasp boundary to about f490 and remap.", "minor"),
            issue(0, "3/4", (790, 800), "boundary",
                  "The second close continues from f790 to the held-block plateau near f800.",
                  "Move grasp-to-move boundary to about f800 and remap.", "minor"),
            issue(0, "5/6", (987, 994), "boundary",
                  "The final opening continues after f987 and settles near f993-994.",
                  "Move release-to-return boundary to about f994 and remap.", "minor"),
        ],
    },
    "ur7e__stack_block__ep000041": {
        "decision_note": (
            "All metadata, effective labels, trace and 41 pages reviewed at native 30 Hz. Text/order/"
            "destinations, contact otherwise, precision, outcomes and no-mistake status are supported."
        ),
        "issues": [
            issue(1, 1, (300, 360), "keep/excluded_gap",
                  "Source metadata rejects the static 10-12 s interval, but the move atom [270,461) and labels bridge it.",
                  "Preserve [300,360) as an excluded gap; split/remap atoms, labels, indexes and history windows."),
            issue(2, 0, (733, 796), "keep",
                  "Dense pages and trace show an aimless stationary waypoint hold.",
                  "Exclude [733,796) and split/remap dependent artifacts."),
            issue(3, "0/1", (975, 1047), "keep",
                  "The held green block remains still with no task-required contact work.",
                  "Exclude [975,1047) and split/remap dependent artifacts."),
            issue(3, "3/4", (1396, 1456), "keep",
                  "The arm is stationary/parked between useful actions.",
                  "Exclude [1396,1456) and split/remap dependent artifacts."),
            issue(5, 0, (1630, 1702), "keep",
                  "The held blue block remains still away from contact work.",
                  "Exclude [1630,1702) and split/remap dependent artifacts."),
            issue(4, "0/5.0", (1620, 1628), "boundary/contact",
                  "The blue grasp changes to move/na at close onset; capture occurs around f1624 and close settles near f1628.",
                  "Extend grasp/top-pinch through f1628, then start move and remap."),
            issue(None, None, (0, 59), "quality_containment",
                  "Two quality spans lie wholly before the retained start f180.",
                  "Remove these out-of-retention rows.", "minor", "effective_labels"),
            *[
                issue(parent, atom, frames, "quality", evidence,
                      "Remove the raw q3 waypoint-settle span; leave retained frames at default quality 4.",
                      "minor")
                for parent, atom, frames, evidence in [
                    (0, 0, (196, 220), "A subsecond waypoint settle remains purposeful."),
                    (1, 0, (274, 304), "The retained part before the excluded gap is a short settle."),
                    (1, 1, (388, 402), "A subsecond waypoint settle remains purposeful."),
                    (1, 2, (479, 493), "A subsecond waypoint settle remains purposeful."),
                    (1, 2, (546, 564), "A short post-release/transition settle remains purposeful."),
                    (1, 3, (599, 613), "A subsecond waypoint settle remains purposeful."),
                    (2, 0, (898, 920), "A subsecond waypoint settle remains purposeful."),
                    (3, 2, (1218, 1235), "A subsecond waypoint settle remains purposeful."),
                    (3, 3, (1269, 1283), "A subsecond waypoint settle remains purposeful."),
                    (4, 0, (1555, 1576), "A subsecond waypoint settle remains purposeful."),
                    (5, 2, (1934, 1950), "A subsecond waypoint settle remains purposeful."),
                ]
            ],
        ],
    },
    "yam__espresso__ep000086": {
        "decision_note": (
            "Source metadata, effective sidecars, trace and all 19 outside+wrist pages reviewed at 30 Hz. "
            "Higher gripper values are closed here. Grasp/contact/precision, q5 insertion, handle push, and "
            "no-mistake status are otherwise supported."
        ),
        "issues": [
            issue(0, 0, (20, 82), "keep",
                  "Pages 1-3 and trace show complete stillness; metadata already rejects f30-60 but atom 0 and labels bridge it.",
                  "Retain [0,20) and [82,843); preserve [20,82) as a gap and rebuild dependent atoms/labels/index/history."),
            issue(0, 0, (0, 97), "quality",
                  "A useful reset precedes excluded idle; padded stall/idle rows grade across the gap.",
                  "Remove raw stall [8,38) and idle [38,50), [50,82); default-4 retained f0-20 and preserve only supported post-gap critique."),
            issue(0, 1, (390, 439), "quality",
                  "Purposeful small adjustments occur under the group head; nonzero joint motion is visible, but a cleanup fragment forces q1.",
                  "Delete idle raw [420,424); use one q3 hover/hold_still raw [390,424) with padding once."),
            issue(0, 3, (697, 747), "quality",
                  "Purposeful handle alignment precedes the successful push; the trace is not fully still, yet cleanup forces q1.",
                  "Delete idle raw [727,732) and retain/extend q3 hover raw [697,732), padding once."),
            issue(0, "3/4", (810, 874), "boundary",
                  "The push is complete by f810; withdrawal/return ends around f843, but push continues to f874.",
                  "End push/start return at f810, end return at f843, and remap contact/speed/channel rows."),
            issue(0, 3, (825, 889), "precision_window",
                  "The push window uses obsolete atom end f874 and remains p3 during return/home.",
                  "Set push commit f810 and window end about f825; return is precision 1."),
            issue(0, 4, (843, 924), "keep",
                  "The arm is parked from about f843; the current return atom contains no return motion.",
                  "End retained footage at f843 and remove/remap all terminal rows."),
        ],
    },
    "droid__AUTOLab__ep002365": {
        "decision_note": (
            "Metadata, trainer sidecars, trace, 17 pages and boundary/commit closeups reviewed. The first "
            "approach is wrongly excluded; repeated bar texts are ambiguous/incomplete, precision is too "
            "low, and two purposeful pre-open holds are mislabeled idle. Contact and no-mistake remain valid."
        ),
        "issues": [
            issue(0, "new/0", (0, 93), "retention_and_atom_boundary",
                  "Frames [0,90) visibly contain the continuous first approach/grasp; move begins only around f93.",
                  "Extend retained start to f0, add grasp [0,93), and start move at f93."),
            issue(0, 0, (90, 119), "subtask_destination",
                  "The bar is being aligned beside the other bars, not moved to an arbitrary table location.",
                  "Use 'move the black aluminium bar next to the other bars'."),
            issue(0, 1, (119, 133), "subtask_destination",
                  "The release completes the alignment beside the other bars.",
                  "Use 'release the black aluminium bar next to the other bars'."),
            issue(0, 1, (104, 141), "precision_level_and_window",
                  "The narrow aligned set-down pins orientation and neighbour clearance; opening command is near f126.",
                  "Add level-4 release window raw start f119, commit f126, with correct padding/end."),
            issue(0, 2, (133, 212), "subtask_object_identity",
                  "Several identical adjacent bars remain; the selected target is the right-hand bar in the pair under the gripper.",
                  "Add that visible discriminator to the grasp text."),
            issue(0, 2, (167, 221), "precision",
                  "The narrow bar and adjacent neighbour constrain pads and orientation.",
                  "Raise grasp precision from 3 to 4 and preserve the reviewed window."),
            issue(0, 3, (212, 285), "subtask_destination",
                  "The motion aligns the bar beside the other bars.",
                  "Use the destination 'next to the other bars'."),
            issue(0, 4, (285, 300), "subtask_destination",
                  "The release aligns beside the other bars and continues into the next parent.",
                  "Use the destination 'next to the other bars' and repair the continuous release boundary."),
            issue("0/1", "4/0", (270, 416), "precision_window",
                  "The aligned release begins around f285 and opens f401-408.",
                  "Add a level-4 window over the final alignment/open, with headroom applied once."),
            issue(1, 0, (300, 408), "subtask_destination",
                  "This is continuation of the same aligned release beside the bars.",
                  "Use the destination 'next to the other bars'."),
            issue(1, 0, (384, 411), "quality",
                  "The raw [399,403) interval is a purposeful pre-open hold, not idle.",
                  "Delete the idle/q1 span; use default 4."),
            issue(1, 1, (408, 495), "subtask_object_identity",
                  "Several identical adjacent bars remain; the selected target needs the reviewed visible discriminator.",
                  "Add the visible bar discriminator to the grasp text."),
            issue(1, 1, (451, 501), "precision",
                  "The narrow bar and adjacent neighbour constrain pads and orientation.",
                  "Raise grasp precision from 3 to 4."),
            issue(2, 0, (495, 539), "subtask_destination",
                  "The motion aligns the bar beside the other bars.",
                  "Use the destination 'next to the other bars'."),
            issue(2, 1, (539, 564), "subtask_destination",
                  "The release aligns the bar beside the other bars.",
                  "Use the destination 'next to the other bars'."),
            issue(2, 1, (524, 564), "precision_window",
                  "The aligned release starts around f539 and command-open begins f559 before the source gap.",
                  "Add level-4 window with raw start f539, commit f559, capped at f564."),
            issue(2, 1, (540, 564), "quality",
                  "Raw [555,559) is a purposeful pre-open hold, not idle; padded span incorrectly crosses the retained end.",
                  "Delete the idle/q1 span; use default 4 and retain the f564 gap."),
        ],
    },
}


def main() -> None:
    checks = WORK / "checks"
    for episode_id, result in RESULTS.items():
        path = checks / f"round_01__{episode_id}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row.update(status="complete", second_read_required=False, **result)
        path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fmb_path = checks / "round_01__episode_000061_6_S_L_4_vertical_n_5.json"
    fmb = json.loads(fmb_path.read_text(encoding="utf-8"))
    fmb["status"] = "complete"
    fmb_path.write_text(json.dumps(fmb, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(checks.glob("round_01__*.json"))]
    wrong = sum(item["severity"] == "wrong" for row in rows for item in row["issues"])
    minor = sum(item["severity"] == "minor" for row in rows for item in row["issues"])
    open_count = sum(row["status"] != "complete" or row["second_read_required"] for row in rows)
    round_path = WORK / "rounds" / "round_01.json"
    round_row = json.loads(round_path.read_text(encoding="utf-8"))
    round_row.update(status="reviewed_needs_repair", pre_fix_counts={"wrong": wrong, "minor": minor, "open": open_count})
    round_path.write_text(json.dumps(round_row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = (WORK / "audit.md").read_text(encoding="utf-8")
    lines = audit.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("| 1 | 2026100701 |"):
            parts = line.split("|")
            parts[-5:-1] = [f" {wrong} ", f" {minor} ", f" {open_count} ", " 0 "]
            lines[index] = "|".join(parts)
    lines += [
        "",
        "### Round 1 disposition",
        "",
        f"- Pre-fix random-round counts: **{wrong} wrong, {minor} minor, {open_count} open**. Fixing these does not change the counts.",
        "- Clean episodes: droid_success__CLVR__ep009563 only.",
        "- Recurring/systematic triggers: excluded-gap and history bridging; purposeful motion mislabeled idle/hover; ambiguous look-alike text; commit boundaries at close/open onset rather than settle; v2 containment outside retained intervals.",
        "- Separate sweep discoveries are recorded under sweeps/ and are not added to the random-round counts.",
    ]
    (WORK / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(rows), "wrong": wrong, "minor": minor, "open": open_count}))


if __name__ == "__main__":
    main()

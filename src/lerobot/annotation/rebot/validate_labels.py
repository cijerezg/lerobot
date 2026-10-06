"""Check the per-episode label files of a pass: <work>/labels/<IDX>.json against <work>/inventory.json.

A subtask string must be in <work>/pass.json "subtasks" or carry a 'NEW_SUBTASK: <string>' flag.

    uv run python -m lerobot.annotation.rebot.validate_labels <work> IDX [IDX ...]
"""

import json
import sys
from pathlib import Path

from lerobot.annotation.rebot.episode import records
from lerobot.annotation.vocab import CONTACT_SLUGS, MISTAKE_TYPES


def errors(d, r, subtasks):
    err = []
    flags = d.get("flags", []) + [x for e in d.get("episodes", []) for x in e.get("flags", [])]
    new = {f.split(":", 1)[1].strip() for f in flags if f.startswith("NEW_SUBTASK:")}
    if d.get("idx") != r["idx"] or d.get("key") != r["key"] or d.get("frames") != r["frames"]:
        err.append("idx/key/frames mismatch")
    if d.get("decision") not in ("keep", "out"):
        err.append("decision must be keep/out")
    if not d.get("reason"):
        err.append("reason missing")
    if d.get("decision") == "out" and d.get("episodes"):
        err.append("out with episodes")
    if d.get("decision") == "keep" and not d.get("episodes"):
        err.append("keep without episodes")
    last = -1
    for j, e in enumerate(d.get("episodes", [])):
        a, b = e["keep"]
        if not (0 <= a < b <= r["frames"]) or a % 3 or (b - a) % 3 or a < last:
            err.append(f"ep{j} keep {a},{b}: need 0<=a<b<=frames, a%3==0, (b-a)%3==0, ordered")
        last = b
        if e.get("split") not in ("train", "val"):
            err.append(f"ep{j} split")
        if not e.get("task") or not e["task"][0].isupper():
            err.append(f"ep{j} task missing / not sentence case")
        cur = a
        for i, s in enumerate(e["segments"]):
            if s["from_index"] != cur or s["to_index"] <= s["from_index"]:
                err.append(f"ep{j} seg{i} not gapless at {cur}")
            cur = s["to_index"]
            if s["subtask"] not in subtasks and s["subtask"] not in new:
                err.append(f"ep{j} seg{i} subtask {s['subtask']!r} not in vocab and no NEW_SUBTASK flag")
            if s.get("speed") not in (1, 2, 3, 4, 5) or s.get("precision") not in (1, 2, 3, 4, 5):
                err.append(f"ep{j} seg{i} speed/precision")
            if s.get("contact") not in CONTACT_SLUGS:
                err.append(f"ep{j} seg{i} contact {s.get('contact')!r}")
            if not isinstance(s.get("orientation_pin"), bool) or not s.get("what_happens"):
                err.append(f"ep{j} seg{i} orientation_pin/what_happens")
            if s["subtask"].startswith(("move the", "return to")) and s["contact"] != "na":
                err.append(f"ep{j} seg{i} move/return must be na")
        if cur != b:
            err.append(f"ep{j} segments end at {cur}, keep ends at {b}")
        for i, m in enumerate(e["mistakes"]):
            if m.get("type") not in MISTAKE_TYPES or not (a <= m["from_index"] < m["to_index"] <= b):
                err.append(f"ep{j} mistake{i} type/range")
            if m.get("confidence") not in ("sure", "unsure") or m.get("looked_at") not in ("coarse", "dense", "video"):
                err.append(f"ep{j} mistake{i} confidence/looked_at")
    return err


if __name__ == "__main__":
    work = Path(sys.argv[1])
    subtasks = json.loads((work / "pass.json").read_text())["subtasks"]
    ok_all = True
    for k in map(int, sys.argv[2:]):
        try:
            d = json.loads((work / f"labels/{k:02d}.json").read_text())
        except Exception as e:
            print(k, "FAIL cannot read:", e)
            ok_all = False
            continue
        err = errors(d, records(work)[k], subtasks)
        print(k, "PASS" if not err else "FAIL", *err, sep="\n  " if err else " ")
        ok_all &= not err
    sys.exit(0 if ok_all else 1)

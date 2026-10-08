"""Write the annotation tables into a freshly built root (build_root output) from the per-episode label files
and the compiled class-pass labels (QUALITY_V2/classes/labels_<pool>.jsonl, staging-root frames). Pool, dataset
name and staging root come from <work>/pass.json. Refuses tables already present.

    uv run python -m lerobot.annotation.rebot.finalize <work> outputs/<root>

Tables (rubric v2 section 10 + the loader tables):
  quality_spans.parquet      raw stretch + writer headroom (critiques 1 s before / 0.5 s after, clipped to the episode)
  mistakes.parquet           final mistake events (replaces the table build_root wrote)
  precision_windows.parquet  w_a / commit / w_b per unit with a window
  episode_metadata.parquet   `quality` per segment: lowest critique stretch overlapping the segment (raw frames),
                             else 5 when an exemplary stretch overlaps it, else 4
  precision.parquet, contact.parquet (+ _info.json)  per-segment values of the label files
  references/<class>.json    strategy file + reference uids of each class
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE
from lerobot.annotation.rebot.episode import kept, records, remap
from lerobot.annotation.vocab import CONTACT_VOCAB, CONTACT_VOCAB_VERSION, code_for

PRE, POST = 30, 15
RUBRICS = "lerobot/src/lerobot/annotation/rubrics"


def episode_map(staging, root, work):
    """staging episode -> (final episode, frame map, final [from, to)). The map takes a staging-root frame to the
    final root through the source frame; idle cuts (spliced or trimmed) collapse onto the splice point. A pass that
    re-annotates an annotated root may name that root as its staging root: its episodes are then the inventory's
    source episodes, uncut."""

    def prov(r):
        eps = pd.read_parquet(r / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
        return {
            (p["source_key"], p["part"]): (
                p["episode_index"],
                int(eps.loc[p["episode_index"]].dataset_from_index),
                int(eps.loc[p["episode_index"]].dataset_to_index),
                p,
            )
            for p in json.loads((r / "meta/provenance.json").read_text())
        }

    f = prov(root)
    if all("source_key" in p for p in json.loads((staging / "meta/provenance.json").read_text())):
        s = prov(staging)
    else:
        eps = pd.concat([pd.read_parquet(x) for x in (staging / "meta/episodes").rglob("*.parquet")]).set_index("episode_index")
        s = {}
        for r in records(work):
            start = int(eps.loc[r["episode"]].dataset_from_index)
            s.update({(r["key"], j): (r["episode"], start, None, dict(keep=[0, r["frames"]])) for j in range(8)})
    out = {}
    for key, (ef, f0, f1, p) in f.items():
        es, s0, _, sp = s[key]
        # staging frame -> source frame through the staging root's own kept frames (a staging root built with the
        # cuts already applied), then source frame -> final frame; the frame after the last maps past the end
        ks = np.append(kept(sp["keep"], sp.get("cuts") or []), sp["keep"][1])
        k = kept(p["keep"], p["cuts"])
        out[es] = (ef, lambda g, s0=s0, ks=ks, k=k, f0=f0: f0 + remap(k, ks[min(g - s0, len(ks) - 1)]), f0, f1)
    return out


def main(work, root):
    config = json.loads((work / "pass.json").read_text())
    pool, name = config["pool"], config["dataset"]
    meta = root / "meta"
    for t in ("quality_spans.parquet", "precision_windows.parquet", "precision.parquet", "contact.parquet"):
        assert not (meta / t).exists(), f"{t} already in {root}"
    emap = episode_map(WORKSPACE / config["staging"], root, work)
    labels = [json.loads(line) for line in open(QUALITY_V2 / f"classes/labels_{pool}.jsonl")]
    em = pd.read_parquet(meta / "episode_metadata.parquet")
    spans, mistakes, windows = [], [], []
    for r in labels:
        m = re.fullmatch(rf"{name}_ep(\d+)_seg(\d+)", r["uid"])
        if m is None:
            continue  # another dataset of the pool
        es, seg = map(int, m.groups())
        if es not in emap:
            continue
        ef, fm, e0, e1 = emap[es]
        for s in r.get("spans", []):
            a, b = fm(s["raw_from"]), fm(s["raw_to"])
            if a == b:
                continue  # wholly inside an idle cut
            crit = s["grade"] <= 3
            spans.append(
                dict(
                    episode_index=ef,
                    raw_from_index=a,
                    raw_to_index=b,
                    from_index=max(e0, a - PRE) if crit else a,
                    to_index=min(e1, b + POST) if crit else b,
                    quality=int(s["grade"]),
                    cause=s["cause"],
                    confidence=s["confidence"],
                    looked_at=s["looked_at"],
                    note=s.get("note", "") or s.get("what_happens", ""),
                    uid=r["uid"],
                )
            )
        for m in r.get("mistakes", []):
            if fm(m["from_index"]) == fm(m["to_index"]):
                continue
            mistakes.append(
                dict(
                    episode_index=ef,
                    from_index=fm(m["from_index"]),
                    to_index=fm(m["to_index"]),
                    mistake=True,
                    mistake_type=m["type"],
                    confidence=m["confidence"],
                    note=m.get("what_happens", ""),
                    uid=r["uid"],
                )
            )
        for k in sorted(x for x in r if x.startswith("precision")):  # precision, precision_2, ... (a unit with several commits)
            p = r[k]
            if not (p and fm(p["w_a"]) < fm(p["w_b"])):
                continue
            c = min(fm(p["commit_index"]), e1 - 1)  # a clip may end on its commit
            host = em[(em.episode_index == ef) & (em.from_index <= c) & (em.to_index > c)]
            assert len(host) == 1, r["uid"]  # final segment index (staging indices shift after reconcile merges)
            windows.append(
                dict(
                    episode_index=ef,
                    segment_index=int(host.segment_index.iloc[0]),
                    commit_index=fm(p["commit_index"]),
                    raw_from_index=fm(p["w_a"]),
                    from_index=max(e0, fm(p["w_a"]) - PRE),
                    to_index=fm(p["w_b"]),
                    precision=int(p["level"]),
                    confidence=p["confidence"],
                    uid=r["uid"],
                )
            )
    # columns named so a root without any span, mistake or window still has readable tables for the loader
    scols = ["episode_index", "raw_from_index", "raw_to_index", "from_index", "to_index", "quality", "cause", "confidence", "looked_at", "note", "uid"]
    mcols = ["episode_index", "from_index", "to_index", "mistake", "mistake_type", "confidence", "note", "uid"]
    wcols = ["episode_index", "segment_index", "commit_index", "raw_from_index", "from_index", "to_index", "precision", "confidence", "uid"]
    S, M, W = pd.DataFrame(spans, columns=scols), pd.DataFrame(mistakes, columns=mcols), pd.DataFrame(windows, columns=wcols)  # noqa: N806
    for df in (S, M):
        assert (df.from_index < df.to_index).all() if len(df) else True
    S.to_parquet(meta / "quality_spans.parquet", index=False)
    M.astype({"mistake": bool}).to_parquet(meta / "mistakes.parquet", index=False)
    W.to_parquet(meta / "precision_windows.parquet", index=False)

    # segment quality for the loader and the per-segment precision / contact tables (label-file values)
    prov = {p["episode_index"]: p for p in json.loads((meta / "provenance.json").read_text())}
    q, prec, cont = [], [], []
    for s in em.itertuples():
        overlap = (S.episode_index == s.episode_index) & (S.raw_from_index < s.to_index) & (S.raw_to_index > s.from_index)
        o = S[overlap] if len(S) else S
        crit = o[o.quality <= 3].quality if len(o) else []
        q.append(int(min(crit)) if len(crit) else 5 if len(o) and (o.quality == 5).any() else 4)
        p = prov[s.episode_index]
        file = json.loads((work / f"labels/{p['inventory_idx']:02d}.json").read_text())
        segs = file["episodes"][p["part"]]["segments"]
        seg = segs[s.segment_index]
        assert seg["subtask"] == s.subtask
        precision = int(seg["precision"])
        if s.subtask.startswith("move the"):  # a move takes the next non-move step's level minus 1, floor 1
            later = (t for t in segs[s.segment_index + 1 :] if not t["subtask"].startswith("move the"))
            nxt = next(later, None)
            precision = max(1, nxt["precision"] - 1) if nxt is not None else 1
        derived = s.subtask.startswith(("move the", "return to"))
        row = dict(
            episode_index=s.episode_index,
            segment_index=s.segment_index,
            from_index=s.from_index,
            to_index=s.to_index,
            source="derived" if derived else "read",
            row_id=f"{name}:ep{s.episode_index}:seg{s.segment_index}",
            flags="",
        )
        if s.subtask.startswith("move the"):
            note = "derived: next step minus 1, floor 1"
        else:
            note = "return to home: 1" if derived else seg.get("note", "")
        prec.append(dict(row, precision=precision, note=note + (" orientation_pin" if seg["orientation_pin"] else "")))
        note = "derived: no contact" if derived else seg.get("note", "")
        cont.append(dict(row, contact=code_for(seg["contact"]), contact_slug=seg["contact"], note=note))
    em["quality"] = np.array(q, dtype=em.quality.dtype)
    em.to_parquet(meta / "episode_metadata.parquet", index=False)
    cols = ["episode_index", "segment_index", "from_index", "to_index"]
    pv, cv = pd.DataFrame(prec), pd.DataFrame(cont)
    pv[cols + ["precision", "source", "note", "row_id", "flags"]].to_parquet(meta / "precision.parquet", index=False)
    cv[cols + ["contact", "contact_slug", "source", "note", "row_id", "flags"]].to_parquet(meta / "contact.parquet", index=False)
    common = dict(
        date=time.strftime("%Y-%m-%d"),
        annotator="claude reviewer agents (one agent per source episode)",
        new_root=str(root.relative_to(WORKSPACE)),
        labels_dir=f"{work}/labels",
    )

    def counts(column):
        return {str(k): int(v) for k, v in column.value_counts().sort_index().items()}

    precision_info = dict(
        common,
        channel="precision",
        table="meta/precision.parquet",
        rubric=f"{RUBRICS}/precision_rubric.md",
        counts=dict(rows=len(pv), per_value_all=counts(pv.precision)),
        windows="meta/precision_windows.parquet (rubric v2 section 6)",
    )
    (meta / "precision_info.json").write_text(json.dumps(precision_info, indent=2))
    contact_info = dict(
        common,
        channel="contact",
        table="meta/contact.parquet",
        rubric=f"{RUBRICS}/contact_strategy_rubric.md",
        vocab_version=CONTACT_VOCAB_VERSION,
        vocab=[e._asdict() for e in CONTACT_VOCAB],
        counts=dict(rows=len(cv), per_value_all=counts(cv.contact)),
    )
    (meta / "contact_info.json").write_text(json.dumps(contact_info, indent=2))
    metadata_info = dict(
        model="assistant-vision-review",
        created_date=time.strftime("%Y-%m-%d"),
        rubric=f"{RUBRICS}/quality_mistake_rubric_v2.md (draft 5)",
        quality_scope="episode_metadata.quality = lowest critique stretch overlapping the segment, else 5 if an exemplary "
        "stretch overlaps, else 4; frame grades come from meta/quality_spans.parquet (rubric 5.1)",
        method="per source episode: segments, mistakes, speed, precision, contact; per class, side by side: strategy files, "
        "critique / exemplary stretches, mistakes confirmed, precision windows; then a second reader",
        annotator=f"{work} + {QUALITY_V2.relative_to(WORKSPACE)}/classes/{pool}__*",
        segments=len(em),
        mistakes=len(M),
        quality_spans=len(S),
        segment_quality=counts(pd.Series(q)),
    )
    (meta / "metadata_info.json").write_text(json.dumps(metadata_info, indent=2))
    ref = root / "references"
    ref.mkdir(exist_ok=True)
    for slug, c in json.loads((work / "class_map.json").read_text()).items():
        fin = WORKSPACE / c["final"]
        line = next((x.strip() for x in open(fin) if x.startswith("References:")), "") if fin.exists() else ""
        reference = dict(sheet_class=slug, strategy_file=c["final"], reuse=c["reuse"], references=line)
        reference["sheets"] = f"{QUALITY_V2.relative_to(WORKSPACE)}/classes/{c['slug']}/sheets"
        (ref / f"{c['slug']}.json").write_text(json.dumps(reference, indent=2))
    print(root.name, "spans", len(S), "mistakes", len(M), "windows", len(W), "segment quality", counts(pd.Series(q)))


if __name__ == "__main__":
    main(Path(sys.argv[1]), WORKSPACE / sys.argv[2])

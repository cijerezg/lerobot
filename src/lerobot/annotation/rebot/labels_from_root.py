"""Label files of a pass that re-annotates an annotated root which has none (README "Re-annotating a root").

    uv run python -m lerobot.annotation.rebot.labels_from_root <work> [IDX ...]

For every inventory record (its source is the annotated root) writes <work>/labels/<IDX>.json from the root's own
tables: keep [0, frames - frames % 3), task, segments (subtask, frames, speed, precision, contact, note) and the mistake rows.
Frames become episode-local. Refuses to overwrite a label file. The pass then edits these files; the class-pass labels
stay in the root's pool (pass.json "staging" = the root, so finalize maps them through the inventory).
"""

import json
import sys
from pathlib import Path

import pandas as pd

from lerobot.annotation.rebot.episode import records

if __name__ == "__main__":
    work = Path(sys.argv[1])
    recs = records(work)
    for r in [recs[int(i)] for i in sys.argv[2:]] or recs:
        out = work / f"labels/{r['idx']:02d}.json"
        assert not out.exists(), out
        meta, e = Path(r["source"]) / "meta", r["episode"]

        def table(name, meta=meta, e=e):
            t = pd.read_parquet(meta / f"{name}.parquet")
            return t[t.episode_index == e]

        eps = pd.concat([pd.read_parquet(p) for p in (meta / "episodes").rglob("*.parquet")]).set_index("episode_index")
        o = int(eps.loc[e].dataset_from_index)
        em = table("episode_metadata").sort_values("segment_index")
        speed, prec, cont = (table(t).set_index("segment_index") for t in ("speed", "precision", "contact"))
        segments = [
            dict(
                from_index=int(s.from_index) - o,
                to_index=int(s.to_index) - o,
                subtask=s.subtask,
                what_happens=str(s.note),
                speed=int(speed.loc[s.segment_index].speed),
                precision=int(prec.loc[s.segment_index].precision),
                orientation_pin="orientation_pin" in str(prec.loc[s.segment_index].note),
                contact=str(cont.loc[s.segment_index].contact_slug),
                note=str(cont.loc[s.segment_index].note),
            )
            for s in em.itertuples()
        ]
        mistakes = [
            dict(
                type=m.mistake_type,
                from_index=int(m.from_index) - o,
                to_index=int(m.to_index) - o,
                what_happens=str(m.note),
                confidence=getattr(m, "confidence", "sure") or "sure",
                looked_at="dense",
                note="",
            )
            for m in table("mistakes").itertuples()
        ]
        end = r["frames"] - r["frames"] % 3  # build_root keeps (b - a) % 3 == 0 (depth phase)
        for s in segments:
            s["to_index"] = min(s["to_index"], end)
        segments = [s for s in segments if s["from_index"] < s["to_index"]]
        mistakes = [dict(m, to_index=min(m["to_index"], end)) for m in mistakes if m["from_index"] < end]
        episode = dict(keep=[0, end], split="train", task=str(eps.loc[e].tasks[0]), segments=segments,
                       mistakes=mistakes, flags=[], cuts=[])
        d = dict(idx=r["idx"], key=r["key"], frames=r["frames"], decision="keep",
                 reason=f"labels of {Path(r['source']).name} ep {e}", episodes=[episode], flags=[])
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(d, indent=1))
        print(out, len(segments), "segments", len(mistakes), "mistakes")

"""One clip per window with the labels burned in (top | wrist, real time): how the user reviews an agent pass.

    uv run python -m lerobot.annotation.rebot.window_render <work>

Reads table.json, frames.csv and mistakes.json of `window_assemble`; writes <work>/videos/<window>.mp4. Timeline:
segments in alternating grey, frames labelled high green, frames labelled mistake red (both look 1 s ahead).
"""

import json
import sys
from pathlib import Path

import av
import pandas as pd

from lerobot.annotation.review.render_review_videos import CYAN, GREEN, GREY, RED, WHITE, YELLOW, Writer, compose

WORK = Path(sys.argv[1])
table = pd.DataFrame(json.loads((WORK / "table.json").read_text()))
per_frame = pd.read_csv(WORK / "frames.csv", keep_default_na=False)
mistakes = json.loads((WORK / "mistakes.json").read_text())
(WORK / "videos").mkdir(exist_ok=True)


def frames(root, row, cam, start, count):
    pre = f"videos/observation.images.{cam}"
    path = root / pre / f"chunk-{int(row[pre + '/chunk_index']):03d}" / f"file-{int(row[pre + '/file_index']):03d}.mp4"
    t0 = float(row[pre + "/from_timestamp"]) + start / 30
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 0.5, 0) / float(s.time_base)), stream=s, backward=True)
        k = 0
        for f in c.decode(s):
            if float(f.pts * s.time_base) < t0 - 1 / 60:
                continue
            yield f.to_ndarray(format="rgb24")
            k += 1
            if k == count:
                break


for name, part in table.groupby("window"):
    root = Path(part.root.iloc[0])
    episodes = pd.concat(pd.read_parquet(p) for p in (root / "meta/episodes").rglob("*.parquet")).set_index("episode_index")
    signals = pd.read_csv(WORK / name / "signals.csv").set_index("frame")
    episode, start, end = int(part.episode.iloc[0]), int(part.start.min()), int(part.end.max())
    n = end - start
    label = per_frame[per_frame.window == name].set_index("frame")
    timeline = [((s.start - start) / n, (s.end - start) / n, (110, 110, 110) if i % 2 else (70, 70, 70)) for i, s in enumerate(part.itertuples())]
    timeline += [((f - start) / n, (f + 1 - start) / n, (60, 200, 60)) for f in label.index[label.high]]
    timeline += [((f - start) / n, (f + 1 - start) / n, (230, 50, 50)) for f in label.index[label.mistake != ""]]
    writer = Writer(WORK / "videos" / f"{name}.mp4", 30)
    row = episodes.loc[episode]
    for i, (top, wrist) in enumerate(zip(frames(root, row, "top", start, n), frames(root, row, "wrist", start, n))):
        f = start + i
        s = part[(part.start <= f) & (part.end > f)].iloc[0]
        c = s.checks
        lines = [(f"{name}   episode {episode}   frame {f}   gripper {signals.grip_angle[f]:.0f} deg   torque {signals.grip_torque[f]:.2f} N*m   tip {signals.tip_speed_smooth[f]:.1f} cm/s", WHITE),
                 (f"SUBTASK  {s.subtask}   [{s.start}-{s.end})", YELLOW),
                 (f"SPEED  {s.level}  ({s.speed:.1f} cm/s mean)", CYAN),
                 (f"QUALITY  {'HIGH' if label.high[f] else 'not high'}   (segment: {'high' if s.quality == 'high' else 'not high'}   direct {s.directness:.2f}   backs off {s.back_cm:.1f} cm   longest pause {s.pause_s:.1f} s   turns {s.reversals}   agent: centred {c['centred_in_wrist_view']}, decisive {c['decisive']})",
                  GREEN if label.high[f] else GREY)]
        lines += [(f"MISTAKE  {m['type']}  motion {m['start']}-{m['event_frame']} (label from 1 s before): {m['note']}"[:230], RED)
                  for m in mistakes if m["window"] == name and m["start"] - 30 <= f <= m["event_frame"]]
        lines.append((f"why: {s.evidence}"[:330], GREY))
        if s.note:
            lines.append((f"unsure: {s.note}"[:230], GREY))
        # the colour constants are BGR
        writer.write(compose([top, wrist], [(text, color[::-1]) for text, color in lines], (timeline, i / n), tile_h=480))
    writer.close()
    print("wrote", writer.path, n, "frames")

"""Anchor frames for trace_episode.py: 2 Hz over the reviewed spans, snapped to the image stride (3). Real seconds."""

import numpy as np

SPANS = {"clipper_ok": (5.0, 17.5), "clipper": (52.5, 132.0), "lid": (208.0, 248.5)}  # user spans, starts snapped to the 0.5 s grid
FPS, STRIDE, RATE_HZ = 30, 3, 2.0

frames = []
for a, b in SPANS.values():
    for t in np.arange(a, b + 1e-9, 1.0 / RATE_HZ):
        f = int(round(round(t * FPS) / STRIDE) * STRIDE)
        if f not in frames:
            frames.append(f)
open(__file__.replace("anchors.py", "anchors.txt"), "w").write(",".join(f"0:{f}" for f in frames))
print(len(frames), "anchors")

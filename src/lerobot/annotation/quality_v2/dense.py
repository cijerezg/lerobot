"""Dense strip (render_strips.strip: 8 tiles, top over wrist) of one pilot unit on a frame window. At most 12 renders at once.

uv run python -m lerobot.annotation.quality_v2.dense <class_slug> <uid> <from> <to>   (classes/<slug> or pilot/<slug>)
Frames: ReBot = global dataset index, diverse = native timestep in the episode (as in the traces). Prints the jpg path.
"""
import fcntl, json, os, sys, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
REPO = WORKSPACE; os.chdir(REPO)
HERE = QUALITY_V2
cls, uid, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); sys.argv = sys.argv[:1]
import cv2, numpy as np  # noqa: E402
from lerobot.annotation.precision_contact import render_strips as rs  # noqa: E402
from lerobot.annotation.precision_contact import diverse_rows  # noqa: E402,F401  (patches the frame grabber for FMB .npy arrays)

cdir = HERE / "classes" / cls if (HERE / "classes" / cls).exists() else HERE / "pilot" / cls
u = next(json.loads(l) for l in open(cdir / "units.jsonl") if f'"uid": "{uid}"' in l)
e0, e1 = u["episode_range"]; a, b = max(a, e0), min(b, e1)
g = np.zeros(e1 + 1); g[e0:e1] = u["gripper_episode"]
r = dict(u["render"], row_id=f"{uid} dense", subtask=u["subtask"][:50], from_index=a, to_index=b, len_s=round((b - a) / u["fps"], 2))
r["_video"] = {k: tuple(v) for k, v in r["_video"].items()}
out = cdir / "dense"; out.mkdir(exist_ok=True); p = out / f"{uid}_{a}_{b}.jpg"
slots = HERE / "pilot" / ".slots"; slots.mkdir(exist_ok=True)
while True:  # take one of 12 render slots
    for k in range(12):
        fh = open(slots / f"{k}", "w")
        try: fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB); break
        except BlockingIOError: fh.close()
    else: time.sleep(0.5); continue
    break
cv2.imwrite(str(p), cv2.cvtColor(rs.strip(r, g), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(p)

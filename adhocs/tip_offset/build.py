"""Tip-offset targets of a subtask-labelled ReBot root: per frame, the vector from the gripper tip now to the tip at
the next gripper event (the grasp, or the release at the peg), in the current tool frame. See README.md."""
import argparse, datetime, glob, json, os
import av, cv2, numpy as np, pandas as pd
from lerobot.annotation.corrections.arm import Arm, frame, project

FX, FY, CX, CY = 394.9832, 394.9832, 322.5604, 238.6966  # D405 at 640 x 480
SLACK_DEG = 2.0  # the gripper counts as holding while it is within this of the held angle

ap = argparse.ArgumentParser()
ap.add_argument("root")
args = ap.parse_args()
root = args.root.rstrip("/")
store = f"lerobot/annotation_data/tip_offset/{os.path.basename(root)}"  # the targets, read by later code
out = f"outputs/analysis/tip_offset/{os.path.basename(root)}"  # the check sheet and the view.py recordings
os.makedirs(store, exist_ok=True)
os.makedirs(out, exist_ok=True)

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(root + "/data/*/*.parquet"))]).sort_values("index").reset_index(drop=True)
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(root + "/meta/episodes/*/*.parquet"))]).set_index("episode_index")
segs = pd.read_parquet(root + "/meta/episode_metadata.parquet").sort_values("from_index").reset_index(drop=True)
state = np.stack(df["observation.state"])
grip = state[:, 6]  # 0 = closed, more negative = more open

arm = Arm()
n = len(df)
tip, rot, cam_p, cam_r = np.zeros((n, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros((n, 3, 3))
for i, q in enumerate(state):
    tip[i], a, j = arm.gripper.fk(q)
    rot[i] = frame(a, j)
    cam_p[i], cam_r[i] = arm.gripper.wrist(q)

# events: (frames that aim at it, event frame, kind, subtask of the event's segment, placed)
events = []
for k, s in segs.iterrows():
    word = s.subtask.split()[0]
    same_ep = lambda j: 0 <= j < len(segs) and segs.episode_index[j] == s.episode_index
    after = segs.subtask[k + 1].split()[0] if same_ep(k + 1) else ""
    if word == "grasp" and after in ("move", "insert"):
        # the hold starts after the last frame the gripper was wider than the angle it leaves the segment with
        wide = np.flatnonzero(grip[s.from_index : s.to_index] < grip[s.to_index - 1] - SLACK_DEG)
        event = s.from_index + (wide[-1] + 1 if len(wide) else 0)
        start = k
        while same_ep(start - 1) and segs.subtask[start - 1] == "approach" + s.subtask[len("grasp") :]:
            start -= 1
        events.append((segs.from_index[start], event, "grasp", s.subtask, True))
    if word == "insert":
        start = k
        while same_ep(start - 1) and segs.subtask[start - 1].split()[0] == "move":
            start -= 1
        a = segs.from_index[start]
        # the release starts after the last frame the gripper was still at the tightest angle of the carry
        event = a + np.flatnonzero(grip[a : s.to_index] >= grip[a : s.to_index].max() - SLACK_DEG)[-1]
        events.append((a, event, "release", s.subtask, after != "grasp"))

rows = pd.DataFrame({"index": df["index"], "episode_index": df.episode_index, "frame_index": df.frame_index, "subtask": ""})
for _, s in segs.iterrows():
    rows.loc[s.from_index : s.to_index - 1, "subtask"] = s.subtask
rows["kind"] = "none"
rows["placed"] = False
for c in ["event_index", "seconds_to_event", "x", "y", "z", "distance", "u", "v"]:
    rows[c] = np.nan
rows["in_view"] = False
for a, event, kind, _, placed in events:
    i = np.arange(a, event)
    world = tip[event] - tip[i]
    local = np.einsum("nji,nj->ni", rot[i], world)  # x jaw (finger to finger), y approach x jaw, z approach (along the fingers)
    cam = np.einsum("nji,nj->ni", cam_r[i], tip[event] - cam_p[i])
    u, v = FX * cam[:, 0] / cam[:, 2] + CX, FY * cam[:, 1] / cam[:, 2] + CY
    rows.loc[i, "kind"], rows.loc[i, "placed"], rows.loc[i, "event_index"] = kind, placed, event
    rows.loc[i, "seconds_to_event"] = (event - i) / 30
    rows.loc[i, ["x", "y", "z"]] = local
    rows.loc[i, "distance"] = np.linalg.norm(world, axis=1)
    rows.loc[i, ["u", "v"]] = np.stack([u, v], 1)
    rows.loc[i, "in_view"] = (cam[:, 2] > 0) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
rows.to_parquet(store + "/targets.parquet")
json.dump(
    dict(root=os.path.abspath(root), built=datetime.datetime.now().isoformat(timespec="seconds"), script="lerobot/adhocs/tip_offset/build.py",
         slack_deg=SLACK_DEG, intrinsics=[FX, FY, CX, CY], frames=n, segments=len(segs),
         events=[dict(first_index=int(a), event_index=int(e), kind=k, subtask=s, placed=bool(p)) for a, e, k, s, p in events]),
    open(store + "/provenance.json", "w"), indent=1)

print(f"{root}: {n} frames, {len(segs)} segments, {len(events)} events")
print("kind      events  frames  in view  distance cm p10 / p50 / p90  |x| |y| |z| cm p50")
for kind in ["grasp", "release"]:
    r = rows[rows.kind == kind]
    d = 100 * r.distance
    print(f"{kind:8s}  {sum(e[2] == kind for e in events):6d}  {len(r):6d}  {r.in_view.mean():7.2f}  {d.quantile(0.1):6.1f} / {d.quantile(0.5):.1f} / {d.quantile(0.9):.1f}"
          f"        {' '.join(f'{100 * r[c].abs().median():.1f}' for c in 'xyz')}")
print(f"none      {'':6s}  {(rows.kind == 'none').sum():6d}")
print(f"releases followed by a new grasp of the tube (placed = False): {sum(e[2] == 'release' and not e[4] for e in events)}")


def video_frame(key, i):
    r = eps.loc[df.episode_index[i]]
    f = f"{root}/videos/{key}/chunk-{r[f'videos/{key}/chunk_index']:03d}/file-{r[f'videos/{key}/file_index']:03d}.mp4"
    t = r[f"videos/{key}/from_timestamp"] + df.frame_index[i] / 30
    c = av.open(f)
    st = c.streams.video[0]
    c.seek(int(max(t - 1, 0) / st.time_base), stream=st)
    for fr in c.decode(st):
        if float(fr.pts * st.time_base) >= t - 1e-3:
            return fr.to_ndarray(format="bgr24")


# check sheet: per event, the top view at the first frame that aims at it and the wrist view 2 s before it
tiles = []
for a, event, kind, subtask, placed in events:
    color = (0, 0, 255) if kind == "grasp" else (0, 255, 255)
    top = video_frame("observation.images.top", a)
    u, v = project(arm.top, tip[event])
    cv2.circle(top, (int(u), int(v)), 12, color, 2)
    i = max(event - 60, a)
    wrist = video_frame("observation.images.wrist", i)
    cv2.circle(wrist, (int(np.clip(rows.u[i], -50, 690)), int(np.clip(rows.v[i], -50, 530))), 14, color, 2)
    cv2.putText(top, f"ep {df.episode_index[a]} fr {df.frame_index[a]} {subtask}"[:52], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(wrist, f"-{(event - i) / 30:.1f} s, {100 * rows.distance[i]:.1f} cm" + ("" if placed else ", regrasped after"), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(np.hstack([cv2.resize(top, (640, 480)), wrist]))
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
sheet = np.vstack([np.hstack(tiles[k : k + 3]) for k in range(0, len(tiles), 3)])
cv2.imwrite(out + "/check.jpg", cv2.resize(sheet, None, fx=0.4, fy=0.4))
print("wrote", store + "/targets.parquet", store + "/provenance.json", out + "/check.jpg")

"""Presentation video of a recorded inference episode: top RGB | wrist RGB | wrist depth, README style.

usage: python lerobot/scripts/episode_3cam_video.py <run_dir> [out.mp4] [--speed 2] [--height 448] [--label autonomous]
                                                    [--panels top,wrist,depth] [--fill-stalls] [--cut <seam frame> ...]
                                                    [--subtasks windows.json] [--depth-inset]
--depth-inset: no depth panel; the wrist depth is drawn as an inset in the top-right corner of the wrist panel (1/3 size).
--style card: dark canvas, rounded panels with camera chips, depth inset rounded, caption bar above the panels with the
step text (typewriter) on the left and "<label> · <speed>x · mm:ss" on the right. The blog standard since 2026-10-26.
Label box top-left: "<label>, <speed>x, mm:ss" (mm:ss = real episode time), URW Gothic like the README demo video.
--subtasks: JSON list of [start frame, text]; each text is drawn in a second box right of the label from its start frame
(recorded index) until the next entry. Written by hand to match what the robot actually did, not the operator's presses.
--labels: same form for the label ([start frame, "autonomous" | "operator" ...]); overrides --label per frame, so operator
spans of a raw run are never labelled autonomous. --shown-speed: the speed the caption states when it differs from --speed.
--fill-stalls: the recorder stamps synthetic 1/30 s timestamps, so a control-loop stall (wall-clock gap > 60 ms between
leader_trace.csv rows) is a 1-frame jump in the recording; insert round(dt*fps)-1 motion-interpolated frames (ffmpeg
minterpolate) at each seam so the video and the mm:ss counter run on wall-clock time. The seam frame is whichever of the
two candidate steps has the larger joint jump. --cut lists seam frames to leave as hard cuts (interpolation smears a
large arm swing into a ghost). Only the mp4 gets synthetic frames; the dataset is untouched.
"""
import argparse, glob, json, subprocess
import av, cv2, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("out", nargs="?")
p.add_argument("--speed", type=float, default=2); p.add_argument("--height", type=int, default=448)
p.add_argument("--label", default="autonomous"); p.add_argument("--crf", type=int, default=23)
p.add_argument("--font-size", type=int, default=26)
p.add_argument("--panels", default="top,wrist,depth")
p.add_argument("--fill-stalls", action="store_true"); p.add_argument("--cut", type=int, nargs="*", default=[])
p.add_argument("--subtasks"); p.add_argument("--depth-inset", action="store_true")
p.add_argument("--labels"); p.add_argument("--shown-speed", type=int)
p.add_argument("--style", default="plain", choices=["plain", "card"])
p.add_argument("--start", type=int, default=0); p.add_argument("--end", type=int)  # recorded frame range [start, end)
a = p.parse_args()
run = a.run.rstrip("/"); ds = f"{run}/inference_dataset"; panels = a.panels.split(",")
out = a.out or f"{run}/episode_{len(panels)}cam_{a.speed:g}x.mp4"
shown_speed = f"{a.shown_speed if a.shown_speed is not None else a.speed:g}"  # text in the caption; --speed may be a float (3.4)
fps = json.load(open(f"{ds}/meta/info.json"))["fps"]
H = a.height; W = 2 * round(H * 640 / 480 / 2)  # 4:3 panels, even width
font = ImageFont.truetype(subprocess.check_output(["fc-match", "-f", "%{file}", "URW Gothic:style=Book"], text=True), a.font_size)
cams = [c for c in panels if c != "depth"]

# ---- card style -------------------------------------------------------------------------------------------------
BG, FG, GREY, ACCENT = (16, 19, 24), (238, 238, 236), (150, 156, 166), (247, 190, 78)  # RGB
if a.style == "card":
    _fp = lambda style: subprocess.check_output(["fc-match", "-f", "%{file}", f"URW Gothic:style={style}"], text=True)
    DEMI, BOOK = _fp("Demi"), _fp("Book")
    F = lambda path, size: ImageFont.truetype(path, size)
    M, G, CB = 24, 24, 84                      # margin, gap, caption bar
    R = max(12, H // 32)                       # panel corner radius
    PY = CB + 12                               # panels sit below the caption bar so the player controls never cover it
    CWd, CHt = 2 * M + len(cams) * W + (len(cams) - 1) * G, PY + H + M
    CWd += CWd % 2; CHt += CHt % 2
    _m = Image.new("L", (W, H), 0); ImageDraw.Draw(_m).rounded_rectangle((0, 0, W - 1, H - 1), R, fill=255); pmask = np.array(_m)[..., None] / 255.0
    IW, IH, IM = W // 3, H // 3, 12
    _mi = Image.new("L", (IW, IH), 0); ImageDraw.Draw(_mi).rounded_rectangle((0, 0, IW - 1, IH - 1), R // 2, fill=255); imask = np.array(_mi)[..., None] / 255.0
    _mb = Image.new("L", (IW + 6, IH + 6), 0); ImageDraw.Draw(_mb).rounded_rectangle((0, 0, IW + 5, IH + 5), R // 2 + 3, fill=255); bmask = np.array(_mb)[..., None] / 255.0
    PX = {c: M + k * (W + G) for k, c in enumerate(cams)}

    def rgba(w, h): return Image.new("RGBA", (w, h), (0, 0, 0, 0))

    def paste(canvas, img_rgba, x, y):
        img = np.array(img_rgba); h, w = img.shape[:2]
        al = img[:, :, 3:4] / 255.0
        canvas[y:y+h, x:x+w] = (al * img[:, :, 2::-1] + (1 - al) * canvas[y:y+h, x:x+w]).astype(np.uint8)

    def text_img(txt, fnt, fill):
        l, t, r, b = fnt.getbbox(txt)
        img = rgba(r - l + 4, b - t + 4); ImageDraw.Draw(img).text((2 - l, 2 - t), txt, font=fnt, fill=fill + (255,))
        return img

    def chip(txt):
        tag = text_img(txt, F(DEMI, max(16, H // 26)), GREY)
        img = rgba(tag.width + 24, tag.height + 12); ImageDraw.Draw(img).rounded_rectangle((0, 0, img.width - 1, img.height - 1), 8, fill=BG + (190,))
        img.alpha_composite(tag, (12, 6)); return img

    CHIPS = {c: chip(f"{c} camera") for c in cams}
    CAP_F, CAP_L, CAP_R = F(BOOK, max(22, H // 17)), F(DEMI, max(14, H // 28)), F(BOOK, max(16, H // 22))
    class Typewriter:
        """Caption text reveals ~3 chars per output frame when it changes."""
        def __init__(self): self.txt, self.k, self.img = None, 0, None
        def get(self, txt):
            if txt != self.txt: self.txt, self.k, self.img = txt, 0, None
            if self.img is None or self.k < len(self.txt):
                self.k = min(self.k + 3, len(self.txt))
                img = rgba(CWd - 2 * M, CB); d = ImageDraw.Draw(img)
                d.text((0, CB // 2 - 10), "SUBTASK", font=CAP_L, fill=ACCENT + (255,))
                d.text((CAP_L.getlength("SUBTASK") + 22, CB // 2 - CAP_F.size // 2 - 4), self.txt[:self.k], font=CAP_F, fill=FG + (255,))
                self.img = img
            return self.img
    typewriter = Typewriter()

    def compose_card(fr, depth, sub_txt, sec, lab_txt):
        canvas = np.full((CHt, CWd, 3), BG[::-1], np.uint8)
        for c in cams:
            x = PX[c]; img = cv2.resize(fr[c], (W, H), interpolation=cv2.INTER_AREA)
            canvas[PY:PY+H, x:x+W] = (pmask * img + (1 - pmask) * canvas[PY:PY+H, x:x+W]).astype(np.uint8)
        if depth is not None and "wrist" in PX:
            x, y = PX["wrist"] + W - IW - IM, PY + IM
            bg = np.array(BG[::-1], np.uint8)
            canvas[y-3:y+IH+3, x-3:x+IW+3] = (bmask * bg + (1 - bmask) * canvas[y-3:y+IH+3, x-3:x+IW+3]).astype(np.uint8)
            img = cv2.resize(depth, (IW, IH), interpolation=cv2.INTER_AREA)
            canvas[y:y+IH, x:x+IW] = (imask * img + (1 - imask) * canvas[y:y+IH, x:x+IW]).astype(np.uint8)
        for c in cams:
            paste(canvas, CHIPS[c], PX[c] + 14, PY + 14)
        if sub_txt:
            paste(canvas, typewriter.get(sub_txt), M, 8)
        tag = text_img(f"{lab_txt}  ·  {shown_speed}x  ·  {sec // 60:02d}:{sec % 60:02d}", CAP_R, GREY)
        paste(canvas, tag, CWd - M - tag.width, 8 + CB // 2 - tag.height // 2)
        return canvas

def decode(cam):
    with av.open(f"{ds}/videos/observation.images.{cam}/chunk-000/file-000.mp4") as c:
        for f in c.decode(video=0):
            yield f.to_ndarray(format="bgr24")

def minterpolate(x, y, n):
    """n motion-compensated frames strictly between x and y: 3-frame clip (x, y, y) at 1 fps -> n+1 fps (minterpolate needs >= 3 inputs)."""
    h, w = x.shape[:2]
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "1", "-i", "-",
                        "-vf", f"minterpolate=fps={n + 1}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1", "-frames:v", str(n + 1),
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], input=np.stack([x, y, y]).tobytes(), capture_output=True, check=True)
    return list(np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w, 3)[1:])

fill = {}  # seam frame k -> frames to insert between k and k+1
if a.fill_stalls:
    t = np.loadtxt(f"{run}/leader_trace.csv", delimiter=",", skiprows=1, usecols=0)
    state = np.stack(pd.read_parquet(f"{ds}/data/chunk-000/file-000.parquet")["observation.state"].values)
    jump = np.abs(np.diff(state, axis=0)).max(axis=1)
    for s in np.where(np.diff(t)[1:] > 0.06)[0] + 1:  # row 0 -> row 1 is the pre-episode wait; the stall's frame is s-1 or s
        k = s - 1 + int(jump[s] > jump[s - 1])
        fill[k] = round((t[s + 1] - t[s]) * fps) - 1
        print(f"stall {1000 * (t[s + 1] - t[s]):.0f} ms at frame {k} ({k / fps:.1f} s): {'cut' if k in a.cut else f'{fill[k]} frames interpolated'}")

def frames():
    """(recorded index, {cam: bgr}) on the wall-clock grid: recorded frames plus the interpolated ones after each seam."""
    its = {c: decode(c) for c in cams}
    cur = {c: next(its[c]) for c in cams}
    i = 0
    while True:
        yield i, cur
        try:
            nxt = {c: next(its[c]) for c in cams}
        except StopIteration:
            return
        if i in fill and i not in a.cut:
            for f in zip(*[minterpolate(cur[c], nxt[c], fill[i]) for c in cams]):
                yield i, dict(zip(cams, f))
        cur, i = nxt, i + 1

depth_files = {int(q.split("frame-")[1][:6]): q for q in glob.glob(f"{ds}/depth/wrist.depth/episode-000000/*.png")}
depth_idx = sorted(depth_files)
lo, hi = 600.0, 8000.0  # raw z16 units, log scale; near = warm, far = cool, beyond hi saturates

def colorize(d):
    n = np.clip((np.log(np.maximum(d, 1).astype(np.float32)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    img = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d == 0] = 0
    return img

def box(txt):
    l, t, r, b = font.getbbox(txt)
    img = Image.new("RGBA", (r - l + 24, b - t + 12), (30, 30, 30, 175))
    ImageDraw.Draw(img).text((12 - l, 6 - t), txt, font=font, fill=(240, 240, 240, 255))
    return np.array(img)

def label(sec, txt):
    return box(f"{txt}, {shown_speed}x, {sec // 60:02d}:{sec % 60:02d}")

def blend(canvas, img, y0, x0):
    bh, bw = img.shape[:2]
    alpha = img[:, :, 3:4].astype(np.float32) / 255
    canvas[y0:y0+bh, x0:x0+bw] = (alpha * img[:, :, 2::-1] + (1 - alpha) * canvas[y0:y0+bh, x0:x0+bw]).astype(np.uint8)

windows = [(int(k), v) for k, v in json.load(open(a.subtasks))] if a.subtasks else []
starts = [k for k, _ in windows]
labels = [(int(k), v) for k, v in json.load(open(a.labels))] if a.labels else [(0, a.label)]
label_starts = [k for k, _ in labels]
def label_at(i): return labels[max(np.searchsorted(label_starts, i, side="right") - 1, 0)][1]

enc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", (f"{CWd}x{CHt}" if a.style == "card" else f"{len(panels)*W}x{H}"), "-r", str(fps), "-i", "-",
     "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
    stdin=subprocess.PIPE)
cur_depth, cur_idx, cur_sec, lab, n_out = None, None, None, None, 0
cur_sub, sub = None, None
next_out = 0.0  # wall-clock frame index of the next output frame; fractional --speed accumulates
for j, (i, fr) in enumerate(frames()):
    if i < a.start or j < next_out:
        continue
    next_out += a.speed
    if a.end is not None and i >= a.end:
        break
    if "depth" in panels or a.depth_inset:
        k = depth_idx[np.searchsorted(depth_idx, i, side="right") - 1]
        if k != cur_idx:
            cur_depth, cur_idx = cv2.resize(colorize(np.array(Image.open(depth_files[k]))), (W, H), interpolation=cv2.INTER_AREA), k
    sec = j // fps; lab_txt = label_at(i)
    if (sec, lab_txt) != cur_sec:
        lab, cur_sec = label(sec, lab_txt), (sec, lab_txt)
    if a.style == "card":
        sub_txt = windows[np.searchsorted(starts, i, side="right") - 1][1] if windows and i >= starts[0] else ""
        canvas = compose_card(fr, cur_depth if (a.depth_inset or "depth" in panels) else None, sub_txt, sec, lab_txt)
        enc.stdin.write(canvas.tobytes()); n_out += 1
        continue
    canvas = np.concatenate([cur_depth if c == "depth" else cv2.resize(fr[c], (W, H), interpolation=cv2.INTER_AREA) for c in panels], axis=1)
    if a.depth_inset and cur_depth is not None:
        iw, ih, m = W // 3, H // 3, 10
        x0 = panels.index("wrist") * W + W - iw - m
        canvas[m - 2:m + ih + 2, x0 - 2:x0 + iw + 2] = 20
        canvas[m:m + ih, x0:x0 + iw] = cv2.resize(cur_depth, (iw, ih), interpolation=cv2.INTER_AREA)
    blend(canvas, lab, 14, 14)
    if windows and i >= starts[0]:
        w = windows[np.searchsorted(starts, i, side="right") - 1][1]
        if w != cur_sub:
            sub, cur_sub = box(w), w
        blend(canvas, sub, 14, 14 + lab.shape[1] + 6)
    enc.stdin.write(canvas.tobytes()); n_out += 1
enc.stdin.close(); enc.wait()
print(f"wrote {out}: {n_out} frames, {a.speed:g}x, {n_out / fps:.1f} s")

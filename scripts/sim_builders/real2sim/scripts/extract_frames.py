"""Video -> evenly spaced sharp frames. usage: extract_frames.py video out_dir [--every N] [--width W]"""
import argparse, pathlib, cv2, numpy as np
p = argparse.ArgumentParser(); p.add_argument("video"); p.add_argument("out"); p.add_argument("--every", type=int, default=15); p.add_argument("--width", type=int, default=1036)
a = p.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(a.video); n = 0; cand = []
while True:
    ok, im = cap.read()
    if not ok: break
    if n % a.every == 0:
        h, w = im.shape[:2]; im = cv2.resize(im, (a.width, int(h * a.width / w)), interpolation=cv2.INTER_AREA)
        sharp = cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        cand.append((n, sharp, im))
    n += 1
med = np.median([c[1] for c in cand]); kept = 0
for fi, s, im in cand:
    if s < 0.5 * med: continue
    cv2.imwrite(str(out / f"{fi:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95]); kept += 1
print(f"{len(cand)} candidates, kept {kept} (sharpness median {med:.0f})")

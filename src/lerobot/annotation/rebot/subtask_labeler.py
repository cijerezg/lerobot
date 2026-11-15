"""Label the subtasks of a ReBot root by hand in the browser.

    uv run python -m lerobot.annotation.rebot.subtask_labeler outputs/data/<root> --subtasks "approach tube" "grasp tube" ...
        [--work outputs/annotation/passes/subtasks_<root>] [--port 7872] [--no-open]

Renders one clip per episode (top | wrist | wrist depth in one H.264 file: one clock, no drift; missing clips only) into
<work>/clips, then opens http://127.0.0.1:<port>. The page plays the clip (up to 4x) over the gripper / joint position,
velocity and torque traces. Key 1..9 ends the matching subtask at the playhead (it covers the stretch since the
previous boundary) and opens its text for a free edit ("approach tube" -> "approach left tube"). Every change is saved at once to
<work>/labels/episode_<E>.json: {root, episode, fps, frames, done, segments: [{start, end, class, subtask}]}, end
exclusive, frames without a segment unlabelled. The labels found at launch are copied to <work>/labels_history first.
The root is only read.
"""

import argparse
import json
import shutil
import subprocess
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot.rollout_review import episodes, video
from lerobot.scripts.view_probes import _json_response, _send_file

PAGE = Path(__file__).with_suffix(".html")
DEPTH_NEAR, DEPTH_FAR = 600, 6000  # depth units of 0.1 mm: 6 cm (the D405 floor) to 60 cm


def colorize(depth):
    """uint16 depth -> RGB: TURBO over a fixed log range (no flicker between frames), near = red, invalid black."""
    d = depth.astype(np.float32)
    scaled = 1 - np.clip(np.log(np.maximum(d, 1) / DEPTH_NEAR) / np.log(DEPTH_FAR / DEPTH_NEAR), 0, 1)
    rgb = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
    rgb[(depth == 0) | (depth == 65535)] = 0
    return rgb


def render(root, fps, e, depth, path):
    """Clip of one episode: top | wrist | colorized depth (each depth frame held until the next one)."""
    n = int(e.length)
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for cam in ("top", "wrist"):
        pre = f"videos/observation.images.{cam}"
        cmd += ["-ss", f"{e[pre + '/from_timestamp']:.4f}", "-t", f"{n / fps:.4f}", "-i", str(video(root, e, cam))]
    panes = "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]"
    frames = {}
    if depth is not None:
        frames = {int(p.stem[6:]): p for p in (root / "depth" / depth / f"episode-{int(e.episode_index):06d}").glob("*.png")}
        height, width = cv2.imread(str(next(iter(frames.values()))), cv2.IMREAD_UNCHANGED).shape
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        panes += "[2:v]"
    cmd += ["-filter_complex", f"{panes}hstack=inputs={3 if frames else 2},fps={fps}[v]", "-map", "[v]", "-frames:v", str(n)]
    cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-g", "15", "-pix_fmt", "yuv420p"]
    part = path.with_suffix(".part.mp4")
    print("rendering", path, f"({n / fps:.0f} s)", flush=True)
    ffmpeg = subprocess.Popen(cmd + ["-movflags", "+faststart", str(part)], stdin=subprocess.PIPE)
    if frames:
        image = np.zeros((height, width, 3), np.uint8)
        with suppress(BrokenPipeError):
            for i in range(n):
                if i in frames:
                    image = colorize(cv2.imread(str(frames[i]), cv2.IMREAD_UNCHANGED))
                ffmpeg.stdin.write(image.tobytes())
    ffmpeg.stdin.close()
    assert ffmpeg.wait() == 0, f"ffmpeg failed on {path}"
    part.rename(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--subtasks", nargs="+", required=True, help="one per number key, in key order")
    parser.add_argument("--work", default=None)
    parser.add_argument("--port", type=int, default=7872)
    parser.add_argument("--no-open", action="store_true", help="serve without opening a browser tab")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    work = Path(args.work).resolve() if args.work else WORKSPACE / "outputs/annotation/passes" / f"subtasks_{root.name}"
    clips, labels = work / "clips", work / "labels"
    clips.mkdir(parents=True, exist_ok=True)
    if labels.exists():
        shutil.copytree(labels, work / "labels_history" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    labels.mkdir(exist_ok=True)

    info = json.load(open(root / "meta/info.json"))
    fps = info["fps"]
    joints = [name.removesuffix(".pos") for name in info["features"]["action"]["names"]]
    table = episodes(root).set_index("episode_index", drop=False)
    data = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet")))
    depth = next((p.name for p in sorted((root / "depth").glob("*"))), None) if (root / "depth").is_dir() else None

    with ThreadPoolExecutor(3) as pool:
        jobs = [
            pool.submit(render, root, fps, e, depth, clips / f"episode_{ep:03d}.mp4")
            for ep, e in table.iterrows()
            if not (clips / f"episode_{ep:03d}.mp4").exists()
        ]
    for job in jobs:
        job.result()

    def stored(ep):
        path = labels / f"episode_{ep:03d}.json"
        return json.load(open(path)) if path.exists() else {"segments": [], "done": False}

    def meta():
        docs = {int(ep): stored(ep) for ep in table.index}
        return {
            "name": root.name,
            "fps": fps,
            "subtasks": args.subtasks,
            "joints": joints,
            "panes": ["Top", "Wrist"] + (["Wrist depth · 6–60 cm, near = red"] if depth else []),
            "episodes": [
                {
                    "episode": ep,
                    "frames": int(table.loc[ep].length),
                    "labelled": sum(s["end"] - s["start"] for s in doc["segments"]),
                    "done": doc["done"],
                }
                for ep, doc in docs.items()
            ],
            "texts": [[s["class"], s["subtask"]] for doc in docs.values() for s in doc["segments"]],
        }

    def episode(ep):
        df = data[data.episode_index == ep].sort_values("frame_index")
        columns = {"action": "action", "state": "observation.state", "velocity": "observation.velocity", "torque": "observation.effort"}
        arrays = {key: np.stack(df[column].to_numpy()) for key, column in columns.items()}
        series = {joint: {key: np.round(a[:, j].astype(float), 3).tolist() for key, a in arrays.items()} for j, joint in enumerate(joints)}
        return {"episode": ep, "frames": len(df), "series": series, **stored(ep)}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                _send_file(self, PAGE)
            elif path == "/api/meta":
                _json_response(self, meta())
            elif path.startswith("/api/episode/"):
                _json_response(self, episode(int(path.rsplit("/", 1)[1])))
            elif path.startswith("/clip/"):
                _send_file(self, clips / f"episode_{int(path.rsplit('/', 1)[1]):03d}.mp4")
            else:
                self.send_error(404)

        def do_POST(self):
            ep = int(self.path.rsplit("/", 1)[1])
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            doc = {
                "root": str(root),
                "episode": ep,
                "fps": fps,
                "frames": int(table.loc[ep].length),
                "done": body["done"],
                "segments": body["segments"],
            }
            part = labels / f"episode_{ep:03d}.json.part"
            part.write_text(json.dumps(doc, indent=1))
            part.rename(labels / f"episode_{ep:03d}.json")
            _json_response(self, {"ok": True})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"subtask labeler: {url}   labels: {labels}   (Ctrl+C stops)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    with suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()

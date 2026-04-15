#!/usr/bin/env python3
"""
Compress an MP4 video for sharing on GitHub.

Where to use each format:
  --format webp  → GitHub README (embedded as image: ![](file.webp))
  --format mp4   → GitHub issues/PR comments (drag-and-drop, plays inline)
  --format webm  → Same as mp4 but smaller; less universal outside browser

Usage:
    python compress_video.py input.mp4 --format webp          # for README
    python compress_video.py input.mp4 --format webp --scale 720 --fps 15
    python compress_video.py input.mp4 --target-mb 15         # MP4 for issues
    python compress_video.py input.mp4 --format webm
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def get_video_info(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def to_webp(input_path: str, output_path: str, scale: int | None, fps: int | None, quality: int):
    """Convert to animated WebP. No audio. Quality 0-100 (default 75)."""
    filters = []
    if scale:
        filters.append(f"scale=-2:{scale}")
    if fps:
        filters.append(f"fps={fps}")
    vf = ",".join(filters) if filters else None

    cmd = ["ffmpeg", "-y", "-i", input_path]
    if vf:
        cmd += ["-vf", vf]
    cmd += [
        "-c:v", "libwebp_anim",
        "-quality", str(quality),
        "-loop", "0",   # loop forever
        "-an",          # no audio (WebP can't carry audio)
        output_path,
    ]
    print(f"Encoding animated WebP at quality={quality}...")
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done: {output_path}  ({size_mb:.2f} MB)")
    return size_mb


def to_webp_target_size(
    input_path: str, output_path: str, target_mb: float,
    scale: int | None, fps: int | None,
):
    """Binary-search over WebP quality to hit the target file size."""
    lo, hi = 10, 90
    best_quality = lo

    for _ in range(7):  # ~7 iterations is enough to converge
        mid = (lo + hi) // 2
        size_mb = to_webp(input_path, output_path, scale, fps, mid)
        print(f"  quality={mid} → {size_mb:.2f} MB (target {target_mb} MB)")
        if size_mb <= target_mb:
            best_quality = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if lo != best_quality + 1:
        to_webp(input_path, output_path, scale, fps, best_quality)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nFinal: {output_path}  ({size_mb:.2f} MB, quality={best_quality})")


def compress_video(
    input_path: str,
    output_path: str,
    target_mb: float,
    fmt: str,
    scale: int | None,
    fps: int | None,
):
    info = get_video_info(input_path)
    duration = float(info["format"]["duration"])

    audio_kbps = 64
    target_bits = target_mb * 8 * 1024 * 1024
    video_kbps = int((target_bits / duration - audio_kbps * 1000) / 1000)

    if video_kbps <= 0:
        print(f"ERROR: Video is too long ({duration:.0f}s) to fit in {target_mb} MB. "
              "Try --scale or --fps to reduce size.")
        sys.exit(1)

    print(f"Duration: {duration:.1f}s | Target: {target_mb} MB | Video bitrate: {video_kbps} kbps")

    filters = []
    if scale:
        filters.append(f"scale=-2:{scale}")
    if fps:
        filters.append(f"fps={fps}")
    vf = ",".join(filters) if filters else None

    def base_args():
        args = ["ffmpeg", "-y", "-i", input_path]
        if vf:
            args += ["-vf", vf]
        return args

    if fmt == "webm":
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "ffmpeg2pass")
            pass1 = base_args() + [
                "-c:v", "libvpx-vp9", "-b:v", f"{video_kbps}k",
                "-pass", "1", "-passlogfile", log,
                "-an", "-f", "null", "/dev/null",
            ]
            print("Pass 1/2...")
            subprocess.run(pass1, check=True, stderr=subprocess.DEVNULL)
            pass2 = base_args() + [
                "-c:v", "libvpx-vp9", "-b:v", f"{video_kbps}k",
                "-pass", "2", "-passlogfile", log,
                "-c:a", "libopus", "-b:a", f"{audio_kbps}k",
                output_path,
            ]
            print("Pass 2/2...")
            subprocess.run(pass2, check=True, stderr=subprocess.DEVNULL)
    else:
        # H.264 MP4
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "ffmpeg2pass")
            pass1 = base_args() + [
                "-c:v", "libx264", "-b:v", f"{video_kbps}k",
                "-pass", "1", "-passlogfile", log,
                "-an", "-f", "null", "/dev/null",
            ]
            print("Pass 1/2...")
            subprocess.run(pass1, check=True, stderr=subprocess.DEVNULL)
            pass2 = base_args() + [
                "-c:v", "libx264", "-b:v", f"{video_kbps}k",
                "-pass", "2", "-passlogfile", log,
                "-c:a", "aac", "-b:a", f"{audio_kbps}k",
                "-movflags", "+faststart",
                output_path,
            ]
            print("Pass 2/2...")
            subprocess.run(pass2, check=True, stderr=subprocess.DEVNULL)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done: {output_path}  ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Compress video for GitHub sharing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Format guide:\n"
            "  webp  → embed in README with ![](file.webp)  (no audio)\n"
            "  mp4   → drag into GitHub issues/PR comments  (has audio)\n"
            "  webm  → smaller than mp4; plays in browser, less universal\n"
        ),
    )
    parser.add_argument("input", help="Input MP4 file")
    parser.add_argument("-o", "--output", help="Output path (default: <input>_compressed.<ext>)")
    parser.add_argument("--target-mb", type=float, default=15.0,
                        help="Target file size in MB (default: 15)")
    parser.add_argument("--format", choices=["mp4", "webm", "webp"], default="mp4",
                        help="Output format (default: mp4)")
    parser.add_argument("--scale", type=int, metavar="HEIGHT",
                        help="Downscale to this height in pixels, e.g. 720 or 480")
    parser.add_argument("--fps", type=int,
                        help="Reduce framerate, e.g. 15 or 10")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    ext = args.format
    if args.output:
        output_path = args.output
    else:
        stem = Path(args.input).stem
        output_path = str(Path(args.input).parent / f"{stem}_compressed.{ext}")

    if args.format == "webp":
        to_webp_target_size(args.input, output_path, args.target_mb, args.scale, args.fps)
    else:
        compress_video(args.input, output_path, args.target_mb, args.format, args.scale, args.fps)


if __name__ == "__main__":
    main()

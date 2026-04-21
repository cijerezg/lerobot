"""ffmpeg helper for encoding numpy frames to MP4.

Mirrors `examples/encode_video.py` from
https://github.com/villekuosmanen/physical-AI-interpretability so the runner
script doesn't depend on that external repo.
"""

from __future__ import annotations

import subprocess
from typing import Sequence

import numpy as np


def encode_video_ffmpeg(
    frames: Sequence[np.ndarray],
    output_filename: str,
    fps: int,
    pix_fmt_in: str = "bgr24",
) -> None:
    """Encode a list of HxWxC numpy frames to MP4 using ffmpeg via stdin.

    Args:
        frames: Iterable of `(H, W, 3)` uint8 frames in `pix_fmt_in` order.
        output_filename: Path to write the encoded MP4.
        fps: Output framerate.
        pix_fmt_in: Pixel format of the input frames (default `bgr24` to match
            OpenCV).
    """
    if not frames:
        print(f"No frames to encode for {output_filename}.")
        return

    height, width, channels = frames[0].shape
    if channels != 3:
        print(f"Error: Frames must be 3-channel. Got {channels} channels.")
        return

    command = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", pix_fmt_in,
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output_filename,
    ]

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for frame in frames:
            process.stdin.write(frame.tobytes())

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error encoding video {output_filename}:")
            print(f"FFmpeg stderr:\n{stderr.decode(errors='ignore')}")
        else:
            print(f"Successfully encoded video: {output_filename}")
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
    except Exception as exc:  # noqa: BLE001
        print(f"An unexpected error occurred during video encoding for {output_filename}: {exc}")

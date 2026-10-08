"""Render audit-specific image pages for sampled diverse episodes.

Pages show every available RGB view at source-native frame indices and the exact
trainer-facing atom/contact plus v2 quality/mistake/precision labels active there.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import av
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = {
    "white": (245, 245, 245),
    "cyan": (100, 225, 255),
    "yellow": (255, 225, 80),
    "red": (255, 90, 90),
    "grey": (165, 165, 165),
}


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def rows_at(rows: list[dict[str, Any]], frame: int) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row.get("from_index", row.get("start_timestep", -1)))
        <= frame
        < int(row.get("to_index", row.get("end_timestep_exclusive", -1)))
    ]


def atom_at(rows: list[dict[str, Any]], frame: int) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if int(row["start_timestep"]) <= frame < int(row["end_timestep_exclusive"])
        ),
        None,
    )


def keyed(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["parent_interval_index"]), int(row["atom_index"])): row
        for row in rows
    }


def selected_frames(packet: dict[str, Any], step_s: float) -> list[int]:
    inv, channels = packet["inventory"], packet["channels"]
    rate, total = float(inv["native_rate_hz"]), int(inv["frames"])
    step, frames = max(1, int(round(step_s * rate))), set()
    for start, stop in inv["retained_intervals"]:
        frames.update(range(int(start), int(stop), step))
        frames.update((int(start), int(stop) - 1))
    for name in (
        "subtask_atoms.jsonl",
        "quality_spans.jsonl",
        "mistakes_v2.jsonl",
        "precision_windows.jsonl",
    ):
        for row in channels[name]:
            start = int(row.get("from_index", row.get("start_timestep", 0)))
            stop = int(row.get("to_index", row.get("end_timestep_exclusive", total)))
            frames.update((start, max(start, stop - 1)))
            for key in ("raw_from_index", "commit_index"):
                if key in row:
                    frames.add(int(row[key]))
            if "raw_to_index" in row:
                frames.add(max(0, int(row["raw_to_index"]) - 1))
    return sorted(frame for frame in frames if 0 <= frame < total)


def decode_common(record, episode_dir: Path, wanted: list[int]):
    want, names, result = set(wanted), [], {}
    for camera in record["cameras"]:
        name = camera if isinstance(camera, str) else camera["name"]
        rel = f"videos/{name}.mp4" if isinstance(camera, str) else camera["path"]
        names.append(name)
        result[name] = {}
        with av.open(str(episode_dir / rel)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in want:
                    result[name][index] = Image.fromarray(frame.to_ndarray(format="rgb24"))
                if index > wanted[-1]:
                    break
    return names, result


def decode_fmb(episode_dir: Path, wanted: list[int]):
    names = [
        name
        for name in ("side_1", "side_2", "wrist_1")
        if (episode_dir / f"{name}_rgb.npy").is_file()
    ]
    result = {}
    for name in names:
        array = np.load(episode_dir / f"{name}_rgb.npy", mmap_mode="r")
        result[name] = {
            frame: Image.fromarray(np.asarray(array[frame]).astype(np.uint8))
            for frame in wanted
        }
    return names, result


def fit_tile(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), (25, 25, 25))
    tile.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return tile


def wrap(draw, text: str, width: int, fnt) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = (current + " " + word).strip()
        if current and draw.textlength(trial, font=fnt) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def frame_text(packet: dict[str, Any], frame: int) -> list[tuple[str, str]]:
    inv, ch = packet["inventory"], packet["channels"]
    atom = atom_at(ch["subtask_atoms.jsonl"], frame)
    contacts, old_precision = keyed(ch["contact_atoms.jsonl"]), keyed(ch["precision_atoms.jsonl"])
    out = [
        (
            f"f{frame} t={frame / float(inv['native_rate_hz']):.2f}s "
            f"{inv['split']} {inv['family']}/{inv['component']}",
            "white",
        )
    ]
    if atom:
        key = (int(atom["parent_interval_index"]), int(atom["atom_index"]))
        contact, precision = contacts.get(key), old_precision.get(key)
        out.append(
            (
                f"ATOM p{key[0]}a{key[1]} [{atom['start_timestep']},"
                f"{atom['end_timestep_exclusive']}): {atom['subtask']}",
                "cyan",
            )
        )
        out.append(
            (
                f"contact={contact.get('contact_slug') if contact else 'MISSING'}; "
                f"atom precision={precision.get('precision') if precision else 'MISSING'}",
                "cyan",
            )
        )
    else:
        out.append(("ATOM: none / excluded footage", "grey"))
    qrows = rows_at(ch["quality_spans.jsonl"], frame)
    critiques = [int(row["quality"]) for row in qrows if int(row["quality"]) <= 3]
    quality = min(critiques) if critiques else (
        5 if any(int(row["quality"]) == 5 for row in qrows) else 4
    )
    qwhy = ", ".join(f"{row['cause']}:{row['confidence']}" for row in qrows) or "default"
    windows = rows_at(ch["precision_windows.jsonl"], frame)
    plevel = max((int(row["precision"]) for row in windows), default=1)
    out.append(
        (
            f"TRAINER: quality={quality} ({qwhy}); precision={plevel} "
            f"({'window' if windows else 'outside window'})",
            "yellow",
        )
    )
    for row in rows_at(ch["mistakes_v2.jsonl"], frame):
        out.append(
            (
                f"MISTAKE {row['mistake_type']} [{row['from_index']},{row['to_index']}): "
                f"{row.get('note', '')}",
                "red",
            )
        )
    return out


def render_pages(packet: dict[str, Any], out: Path, step_s: float, per_page: int) -> int:
    inv = packet["inventory"]
    episode_dir = Path(inv["label_store"]) / "episodes" / inv["episode_id"]
    record = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    wanted = selected_frames(packet, step_s)
    names, decoded = (
        decode_fmb(episode_dir, wanted)
        if inv["store"] == "fmb"
        else decode_common(record, episode_dir, wanted)
    )
    out.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h, text_w, header_h = 360, 250, 780, 90
    width = tile_w * len(names) + text_w
    pages = math.ceil(len(wanted) / per_page)
    for page_index in range(pages):
        frames = wanted[page_index * per_page : (page_index + 1) * per_page]
        canvas = Image.new(
            "RGB", (width, header_h + tile_h * len(frames)), (18, 18, 18)
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 8),
            f"{inv['episode_id']} | task: {inv['task']}",
            font=font(22, True),
            fill="white",
        )
        draw.text(
            (12, 40),
            f"native {inv['native_rate_hz']:g} Hz | retained {inv['retained_intervals']} | "
            f"views: {', '.join(names)} | page {page_index + 1}/{pages}",
            font=font(16),
            fill="white",
        )
        for row_index, frame in enumerate(frames):
            y = header_h + row_index * tile_h
            for camera_index, name in enumerate(names):
                image = decoded[name].get(
                    frame, Image.new("RGB", (tile_w, tile_h), (55, 0, 0))
                )
                canvas.paste(
                    fit_tile(image, tile_w, tile_h), (camera_index * tile_w, y)
                )
                draw.text(
                    (camera_index * tile_w + 5, y + 5),
                    name,
                    font=font(14, True),
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )
            x, ty = tile_w * len(names) + 10, y + 8
            for text_value, color in frame_text(packet, frame):
                for line in wrap(draw, text_value, text_w - 20, font(15)):
                    draw.text((x, ty), line, font=font(15), fill=COLORS[color])
                    ty += 20
            draw.line(
                (0, y + tile_h - 1, width, y + tile_h - 1),
                fill=(100, 100, 100),
            )
        canvas.save(out / f"page_{page_index + 1:03d}.jpg", quality=90)
    return pages


def render_trace(packet: dict[str, Any], out: Path) -> None:
    inv = packet["inventory"]
    episode_dir = Path(inv["label_store"]) / "episodes" / inv["episode_id"]
    rate, n = float(inv["native_rate_hz"]), int(inv["frames"])
    timeline = np.arange(n) / rate
    if inv["store"] == "fmb":
        pose = np.load(episode_dir / "tcp_pose.npy", mmap_mode="r")
        speed = np.r_[
            0.0, np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1) * rate * 100
        ]
        gripper_path = episode_dir / "gripper_pose.npy"
        gripper = (
            np.load(gripper_path, mmap_mode="r").reshape(n, -1)[:, 0]
            if gripper_path.is_file()
            else None
        )
        ylabel = "TCP speed cm/s"
    else:
        state = np.load(episode_dir / "state.npy", mmap_mode="r")
        if inv["family"] == "molmoact":
            speed = np.r_[
                0.0,
                np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1) * rate * 100,
            ]
            ylabel = "TCP speed cm/s"
        else:
            speed = np.r_[
                0.0,
                np.linalg.norm(np.diff(state[:, :-1], axis=0), axis=1) * rate,
            ]
            ylabel = "joint L2 speed /s"
        gripper = state[:, -1] if state.shape[1] else None
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeline, speed, color="black", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("seconds")
    ax2 = ax.twinx()
    if gripper is not None:
        ax2.plot(timeline, gripper, color="tab:blue", lw=0.7, alpha=0.7)
        ax2.set_ylabel("source-native gripper", color="tab:blue")
    for atom in packet["channels"]["subtask_atoms.jsonl"]:
        x = int(atom["start_timestep"]) / rate
        ax.axvline(x, color="tab:green", alpha=0.35)
    for row in packet["channels"]["quality_spans.jsonl"]:
        color = "gold" if int(row["quality"]) == 5 else "tab:orange"
        ax.axvspan(
            int(row["from_index"]) / rate,
            int(row["to_index"]) / rate,
            color=color,
            alpha=0.16,
        )
    for row in packet["channels"]["mistakes_v2.jsonl"]:
        ax.axvspan(
            int(row["from_index"]) / rate,
            int(row["to_index"]) / rate,
            color="tab:red",
            alpha=0.25,
        )
    ax.set_title(
        f"{inv['episode_id']} source-native trace "
        "(green=atom, orange/gold=quality span, red=mistake)"
    )
    fig.tight_layout()
    fig.savefig(out / "trace.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--step-s", type=float, default=0.5)
    parser.add_argument("--per-page", type=int, default=6)
    args = parser.parse_args()
    base = Path(args.work) / "evidence" / f"round_{args.round:02d}"
    total = 0
    for episode_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        packet = json.loads(
            (episode_dir / "effective_labels.json").read_text(encoding="utf-8")
        )
        pages = render_pages(packet, episode_dir, args.step_s, args.per_page)
        render_trace(packet, episode_dir)
        total += pages
        print(episode_dir.name, pages)
    print("pages", total)


if __name__ == "__main__":
    main()

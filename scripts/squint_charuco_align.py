#!/usr/bin/env python3
"""Generate a ChArUco board and convert detections into Squint camera poses."""

from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

DEFAULT_MARKER_CENTER = (0.18524555, 0.110, 0.0)


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary: str

    @property
    def width_m(self) -> float:
        return self.squares_x * self.square_length_m

    @property
    def height_m(self) -> float:
        return self.squares_y * self.square_length_m

    def make_board(self) -> Any:
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            _aruco_dictionary(self.dictionary),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "dictionary": _dictionary_attr_name(self.dictionary),
            "width_m": self.width_m,
            "height_m": self.height_m,
        }


def _dictionary_attr_name(name: str) -> str:
    name = name.strip().upper()
    if not name.startswith("DICT_"):
        name = f"DICT_{name}"
    return name


def _aruco_dictionary(name: str) -> Any:
    attr_name = _dictionary_attr_name(name)
    dictionary_id = getattr(cv2.aruco, attr_name, None)
    if dictionary_id is None:
        valid = sorted(key for key in dir(cv2.aruco) if key.startswith("DICT_"))
        raise ValueError(f"Unknown ArUco dictionary {name!r}. Known dictionaries include: {valid[:12]}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _parse_named_args(entries: list[str] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Expected NAME=VALUE, got {entry!r}")
        name, value = entry.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Expected non-empty NAME=VALUE, got {entry!r}")
        values[name] = value
    return values


def _load_config_defaults(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if "experiments" in raw:
        defaults = raw.get("defaults", {}) or {}
        first = (raw.get("experiments") or [{}])[0] or {}
        raw = {**defaults, **first}

    cameras: dict[str, str] = {}
    camera1_name = raw.get("camera1_name")
    camera1_path = raw.get("camera1_path")
    camera2_name = raw.get("camera2_name")
    camera2_path = raw.get("camera2_path")
    if camera1_name and camera1_path:
        cameras[str(camera1_name)] = str(camera1_path)
    if camera2_name and camera2_path:
        cameras[str(camera2_name)] = str(camera2_path)
    return {
        "cameras": cameras,
        "width": raw.get("camera_width"),
        "height": raw.get("camera_height"),
        "fps": raw.get("camera_fps"),
        "fourcc": raw.get("camera_fourcc"),
    }


def _board_spec_from_args(args: argparse.Namespace) -> BoardSpec:
    return BoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length_m,
        marker_length_m=args.marker_length_m,
        dictionary=args.dictionary,
    )


def _add_board_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-length-m", type=float, default=0.030)
    parser.add_argument("--marker-length-m", type=float, default=0.022)
    parser.add_argument("--dictionary", default="DICT_4X4_50")


def _board_image(spec: BoardSpec, dpi: int, margin_mm: float) -> tuple[np.ndarray, float, float]:
    board = spec.make_board()
    board_w_px = max(1, round((spec.width_m / 0.0254) * dpi))
    board_h_px = max(1, round((spec.height_m / 0.0254) * dpi))
    margin_px = max(0, round((margin_mm / 25.4) * dpi))
    board_img = board.generateImage((board_w_px, board_h_px))
    if board_img.ndim == 2:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    canvas = np.full(
        (board_h_px + 2 * margin_px, board_w_px + 2 * margin_px, 3),
        255,
        dtype=np.uint8,
    )
    canvas[margin_px : margin_px + board_h_px, margin_px : margin_px + board_w_px] = board_img

    if margin_px >= 30:
        red = (40, 40, 220)
        green = (40, 160, 40)
        black = (0, 0, 0)
        x_y = max(16, margin_px // 2)
        y_x = max(16, margin_px // 2)
        arrow_len = min(board_w_px // 3, max(80, margin_px * 3))
        cv2.arrowedLine(canvas, (margin_px, x_y), (margin_px + arrow_len, x_y), red, 3, tipLength=0.15)
        cv2.putText(canvas, "+X", (margin_px + arrow_len + 8, x_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
        arrow_len_y = min(board_h_px // 3, max(80, margin_px * 3))
        cv2.arrowedLine(canvas, (y_x, margin_px), (y_x, margin_px + arrow_len_y), green, 3, tipLength=0.15)
        cv2.putText(canvas, "+Y", (max(2, y_x - 8), margin_px + arrow_len_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
        cv2.putText(
            canvas,
            "origin: pattern top-left",
            (margin_px, canvas.shape[0] - max(10, margin_px // 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            black,
            1,
        )

    total_w_mm = spec.width_m * 1000.0 + 2 * margin_mm
    total_h_mm = spec.height_m * 1000.0 + 2 * margin_mm
    return canvas, total_w_mm, total_h_mm


def cmd_print_board(args: argparse.Namespace) -> int:
    spec = _board_spec_from_args(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"charuco_{spec.squares_x}x{spec.squares_y}_{round(spec.square_length_m * 1000)}mm"
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    meta_path = out_dir / f"{name}.json"

    image, width_mm, height_mm = _board_image(spec, args.dpi, args.margin_mm)
    cv2.imwrite(str(png_path), image)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode ChArUco PNG")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:.3f}mm" '
        f'height="{height_mm:.3f}mm" viewBox="0 0 {image.shape[1]} {image.shape[0]}">\n'
        f'  <image href="data:image/png;base64,{b64}" x="0" y="0" '
        f'width="{image.shape[1]}" height="{image.shape[0]}"/>\n'
        "</svg>\n"
    )
    svg_path.write_text(svg)
    meta = {
        "saved_at": _now(),
        "print_size_mm": [width_mm, height_mm],
        "margin_mm": args.margin_mm,
        "dpi": args.dpi,
        "board": spec.as_dict(),
        "notes": [
            "Print the SVG at 100 percent scale.",
            "For the default pose convention, put the printed side up; +X follows the red arrow and +Y follows the green arrow.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {svg_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {meta_path}")
    print(f"Board area: {spec.width_m * 1000:.1f} x {spec.height_m * 1000:.1f} mm")
    print(f"Printed page/artboard: {width_mm:.1f} x {height_mm:.1f} mm")
    return 0


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _video_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def _capture_frames(
    source: str,
    *,
    frames: int,
    width: int | None,
    height: int | None,
    fps: int | None,
    fourcc: str | None,
    interval_s: float,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(_video_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {source}")
    try:
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))

        for _ in range(5):
            cap.read()
            time.sleep(0.03)

        captured: list[np.ndarray] = []
        for _ in range(frames):
            ok, frame = cap.read()
            if ok and frame is not None:
                captured.append(frame)
            if interval_s > 0:
                time.sleep(interval_s)
        return captured
    finally:
        cap.release()


def _detect_charuco(image: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    board = spec.make_board()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        raise RuntimeError("No ChArUco corners detected")
    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    else:
        chessboard_corners = board.getChessboardCorners()
        object_points = chessboard_corners[charuco_ids.reshape(-1)].reshape(-1, 1, 3)
        image_points = charuco_corners.reshape(-1, 1, 2)
    return object_points.astype(np.float32), image_points.astype(np.float32), charuco_corners, charuco_ids


def _load_intrinsics(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        raw = json.load(f)
    matrix = raw.get("camera_matrix", raw.get("K"))
    dist = raw.get("dist_coeffs", raw.get("distortion_coefficients", [0, 0, 0, 0, 0]))
    if matrix is None:
        raise ValueError(f"Intrinsic file lacks camera_matrix/K: {path}")
    return np.asarray(matrix, dtype=np.float64), np.asarray(dist, dtype=np.float64).reshape(-1, 1)


def _approx_intrinsics(image: np.ndarray, horizontal_fov_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    f = (0.5 * width) / math.tan(math.radians(horizontal_fov_degrees) * 0.5)
    matrix = np.array(
        [[f, 0.0, (width - 1) * 0.5], [0.0, f, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return matrix, np.zeros((5, 1), dtype=np.float64)


def _board_world_transform(
    spec: BoardSpec,
    center: tuple[float, float, float],
    yaw_degrees: float,
    board_z_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_degrees)
    x_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    if board_z_axis == "into-table":
        y_axis = np.array([math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        y_axis = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    center_w = np.asarray(center, dtype=np.float64)
    center_b = np.array([spec.width_m * 0.5, spec.height_m * 0.5, 0.0], dtype=np.float64)
    origin_w = center_w - rotation @ center_b
    return rotation, origin_w


def _estimate_pose_from_frame(
    image: np.ndarray,
    spec: BoardSpec,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board_rotation_world: np.ndarray,
    board_origin_world: np.ndarray,
    target_distance_m: float,
    min_corners: int,
) -> dict[str, Any]:
    object_points, image_points, _charuco_corners, charuco_ids = _detect_charuco(image, spec)
    if len(charuco_ids) < min_corners:
        raise RuntimeError(f"Only {len(charuco_ids)} ChArUco corners detected; need at least {min_corners}")

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed")
    rotation_camera_board, _ = cv2.Rodrigues(rvec)
    translation_camera_board = tvec.reshape(3)

    camera_pos_board = -rotation_camera_board.T @ translation_camera_board
    camera_forward_board = rotation_camera_board.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    eye = board_origin_world + board_rotation_world @ camera_pos_board
    forward = board_rotation_world @ camera_forward_board
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    board_normal_world = board_rotation_world @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    denom = float(np.dot(board_normal_world, forward))
    distance_to_board = target_distance_m
    if abs(denom) > 1e-9:
        ray_distance = float(np.dot(board_normal_world, board_origin_world - eye) / denom)
        if ray_distance > 0:
            distance_to_board = ray_distance
    target = eye + forward * distance_to_board

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    reproj_error = float(np.sqrt(np.mean(np.sum(reproj * reproj, axis=1))))

    return {
        "eye": eye,
        "target": target,
        "forward": forward,
        "target_distance_m": distance_to_board,
        "corner_count": int(len(charuco_ids)),
        "reprojection_error_px": reproj_error,
        "rvec": rvec.reshape(3),
        "tvec": translation_camera_board,
    }


def _camera_inputs(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    config_defaults = _load_config_defaults(args.config)
    cameras = dict(config_defaults.get("cameras") or {})
    cameras.update(_parse_named_args(args.camera))
    images = _parse_named_args(args.image)
    return cameras, images, config_defaults


def _effective_capture_setting(args: argparse.Namespace, config_defaults: dict[str, Any], key: str) -> Any:
    value = getattr(args, key)
    if value is not None:
        return value
    return config_defaults.get(key)


def _load_frames_for_name(
    name: str,
    *,
    cameras: dict[str, str],
    images: dict[str, str],
    args: argparse.Namespace,
    config_defaults: dict[str, Any],
) -> tuple[list[np.ndarray], list[str]]:
    if name in images:
        paths = sorted(glob.glob(images[name]))
        if not paths:
            path = Path(images[name])
            if path.exists():
                paths = [str(path)]
        if not paths:
            raise FileNotFoundError(f"No image paths matched for {name}: {images[name]}")
        return [_read_image(path) for path in paths], paths

    source = cameras.get(name)
    if not source:
        raise ValueError(f"No --image or --camera source found for {name}")
    frames = _capture_frames(
        source,
        frames=args.frames,
        width=_effective_capture_setting(args, config_defaults, "width"),
        height=_effective_capture_setting(args, config_defaults, "height"),
        fps=_effective_capture_setting(args, config_defaults, "fps"),
        fourcc=_effective_capture_setting(args, config_defaults, "fourcc"),
        interval_s=args.capture_interval_s,
    )
    return frames, [f"{source}#{index}" for index in range(len(frames))]


def cmd_calibrate_intrinsics(args: argparse.Namespace) -> int:
    spec = _board_spec_from_args(args)
    cameras, images, config_defaults = _camera_inputs(args)
    names = sorted(set(cameras) | set(images))
    if not names:
        raise ValueError("Provide --camera NAME=SOURCE, --image NAME=GLOB, or --config with camera paths")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        frames, labels = _load_frames_for_name(
            name,
            cameras=cameras,
            images=images,
            args=args,
            config_defaults=config_defaults,
        )
        object_points_by_frame: list[np.ndarray] = []
        image_points_by_frame: list[np.ndarray] = []
        used_labels: list[str] = []
        image_size: tuple[int, int] | None = None
        frame_dir = out_dir / f"{name}_frames"
        if args.save_frames:
            frame_dir.mkdir(parents=True, exist_ok=True)

        for index, (frame, label) in enumerate(zip(frames, labels, strict=False)):
            image_size = (frame.shape[1], frame.shape[0])
            try:
                object_points, image_points, _corners, ids = _detect_charuco(frame, spec)
            except RuntimeError:
                continue
            if len(ids) < args.min_corners:
                continue
            object_points_by_frame.append(object_points)
            image_points_by_frame.append(image_points)
            used_labels.append(label)
            if args.save_frames:
                cv2.imwrite(str(frame_dir / f"{index:04d}.png"), frame)

        if image_size is None:
            raise RuntimeError(f"No frames available for {name}")
        if len(object_points_by_frame) < args.min_frames:
            raise RuntimeError(
                f"{name}: only {len(object_points_by_frame)} usable frames; need at least {args.min_frames}"
            )

        rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            object_points_by_frame,
            image_points_by_frame,
            image_size,
            None,
            None,
        )
        payload = {
            "saved_at": _now(),
            "camera": name,
            "image_size": list(image_size),
            "rms_reprojection_error_px": float(rms),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
            "frames_used": len(object_points_by_frame),
            "frame_labels": used_labels,
            "board": spec.as_dict(),
        }
        path = out_dir / f"{name}_intrinsics.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{name}: wrote {path} using {len(object_points_by_frame)} frames, rms={rms:.3f}px")
    return 0


def cmd_estimate_poses(args: argparse.Namespace) -> int:
    spec = _board_spec_from_args(args)
    cameras, images, config_defaults = _camera_inputs(args)
    intrinsics = _parse_named_args(args.intrinsics)
    names = sorted(set(cameras) | set(images) | set(intrinsics))
    if not names:
        raise ValueError("Provide --camera NAME=SOURCE, --image NAME=GLOB, or --config with camera paths")

    board_center = tuple(float(value) for value in args.board_world_center)
    board_rotation_world, board_origin_world = _board_world_transform(
        spec,
        board_center,
        args.board_yaw_degrees,
        args.board_z_axis,
    )

    output_cameras: dict[str, Any] = {}
    for name in names:
        frames, labels = _load_frames_for_name(
            name,
            cameras=cameras,
            images=images,
            args=args,
            config_defaults=config_defaults,
        )
        if not frames:
            raise RuntimeError(f"No frames available for {name}")
        if name in intrinsics:
            camera_matrix, dist_coeffs = _load_intrinsics(intrinsics[name])
            intrinsics_source = intrinsics[name]
        else:
            camera_matrix, dist_coeffs = _approx_intrinsics(frames[0], args.approximate_horizontal_fov_degrees)
            intrinsics_source = f"approx_hfov_{args.approximate_horizontal_fov_degrees:g}"
            print(
                f"WARNING: {name} has no intrinsic file; using approximate "
                f"{args.approximate_horizontal_fov_degrees:g} deg horizontal FOV"
            )

        estimates: list[dict[str, Any]] = []
        failures = 0
        for frame in frames:
            try:
                estimates.append(
                    _estimate_pose_from_frame(
                        frame,
                        spec,
                        camera_matrix,
                        dist_coeffs,
                        board_rotation_world,
                        board_origin_world,
                        args.target_distance_m,
                        args.min_corners,
                    )
                )
            except RuntimeError:
                failures += 1

        if len(estimates) < args.min_valid_frames:
            raise RuntimeError(
                f"{name}: only {len(estimates)} valid pose frames; need at least {args.min_valid_frames}"
            )

        eyes = np.stack([estimate["eye"] for estimate in estimates], axis=0)
        forwards = np.stack([estimate["forward"] for estimate in estimates], axis=0)
        mean_eye = eyes.mean(axis=0)
        targets = np.stack([estimate["target"] for estimate in estimates], axis=0)
        mean_forward = forwards.mean(axis=0)
        mean_forward = mean_forward / max(np.linalg.norm(mean_forward), 1e-9)
        mean_target = targets.mean(axis=0)
        mean_error = float(np.mean([estimate["reprojection_error_px"] for estimate in estimates]))
        output_cameras[name] = {
            "eye": [float(value) for value in mean_eye],
            "target": [float(value) for value in mean_target],
            "valid_frames": len(estimates),
            "failed_frames": failures,
            "mean_reprojection_error_px": mean_error,
            "mean_corner_count": float(np.mean([estimate["corner_count"] for estimate in estimates])),
            "intrinsics": intrinsics_source,
            "frame_labels": labels[: len(frames)],
        }
        print(
            f"{name}: eye={output_cameras[name]['eye']} target={output_cameras[name]['target']} "
            f"valid={len(estimates)} reproj={mean_error:.3f}px"
        )

    output = {
        "saved_at": _now(),
        "type": "squint_charuco_camera_poses",
        "env_id_hint": "SO101PlaceCubeMarker-v1",
        "board": {
            **spec.as_dict(),
            "world_center": list(board_center),
            "world_origin": [float(value) for value in board_origin_world],
            "world_rotation_board_to_world": board_rotation_world.tolist(),
            "yaw_degrees": args.board_yaw_degrees,
            "z_axis": args.board_z_axis,
        },
        "cameras": output_cameras,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    print_parser = subparsers.add_parser("print-board", help="Generate printable ChArUco board files")
    _add_board_args(print_parser)
    print_parser.add_argument("--output-dir", default="outputs/camera_alignment")
    print_parser.add_argument("--name", default="")
    print_parser.add_argument("--dpi", type=int, default=300)
    print_parser.add_argument("--margin-mm", type=float, default=15.0)
    print_parser.set_defaults(func=cmd_print_board)

    calib_parser = subparsers.add_parser("calibrate-intrinsics", help="Calibrate camera intrinsics from ChArUco frames")
    _add_board_args(calib_parser)
    calib_parser.add_argument("--config", default=None)
    calib_parser.add_argument("--camera", action="append", default=[], help="Camera source as NAME=/dev/videoN")
    calib_parser.add_argument("--image", action="append", default=[], help="Image glob as NAME='captures/*.png'")
    calib_parser.add_argument("--out-dir", default="outputs/camera_alignment/intrinsics")
    calib_parser.add_argument("--frames", type=int, default=40)
    calib_parser.add_argument("--capture-interval-s", type=float, default=0.25)
    calib_parser.add_argument("--min-corners", type=int, default=8)
    calib_parser.add_argument("--min-frames", type=int, default=12)
    calib_parser.add_argument("--width", type=int, default=None)
    calib_parser.add_argument("--height", type=int, default=None)
    calib_parser.add_argument("--fps", type=int, default=None)
    calib_parser.add_argument("--fourcc", default=None)
    calib_parser.add_argument("--save-frames", action="store_true")
    calib_parser.set_defaults(func=cmd_calibrate_intrinsics)

    estimate_parser = subparsers.add_parser("estimate-poses", help="Estimate Squint eye/target camera poses")
    _add_board_args(estimate_parser)
    estimate_parser.add_argument("--config", default=None)
    estimate_parser.add_argument("--camera", action="append", default=[], help="Camera source as NAME=/dev/videoN")
    estimate_parser.add_argument("--image", action="append", default=[], help="Image glob as NAME='captures/*.png'")
    estimate_parser.add_argument("--intrinsics", action="append", default=[], help="Intrinsics JSON as NAME=path.json")
    estimate_parser.add_argument("--out", default="outputs/camera_alignment/charuco_camera_poses.json")
    estimate_parser.add_argument("--frames", type=int, default=20)
    estimate_parser.add_argument("--capture-interval-s", type=float, default=0.1)
    estimate_parser.add_argument("--min-corners", type=int, default=8)
    estimate_parser.add_argument("--min-valid-frames", type=int, default=3)
    estimate_parser.add_argument(
        "--target-distance-m",
        type=float,
        default=1.0,
        help="Fallback look-at distance if the optical ray does not intersect the board plane",
    )
    estimate_parser.add_argument("--board-world-center", type=float, nargs=3, default=DEFAULT_MARKER_CENTER)
    estimate_parser.add_argument("--board-yaw-degrees", type=float, default=0.0)
    estimate_parser.add_argument("--board-z-axis", choices=["into-table", "up"], default="into-table")
    estimate_parser.add_argument("--approximate-horizontal-fov-degrees", type=float, default=60.0)
    estimate_parser.add_argument("--width", type=int, default=None)
    estimate_parser.add_argument("--height", type=int, default=None)
    estimate_parser.add_argument("--fps", type=int, default=None)
    estimate_parser.add_argument("--fourcc", default=None)
    estimate_parser.set_defaults(func=cmd_estimate_poses)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

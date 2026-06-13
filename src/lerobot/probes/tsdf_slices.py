"""TSDF slice visualizer (depth_tsdf_design.md §9 phasing item 1, roadmap WS2).

For sampled frames of one recorded episode, builds the TSDF grid from the
sidecar depth PNG exactly as training does (build_tsdf_grid + the policy's
tsdf_config, placeholder calibration and all) and renders a contact sheet:

  row 1   RGB | raw depth, grid-box footprint overlaid | φ and known side
          views (xz @ y-mid, yz @ x-mid)
  row 2   φ xy-slices marching through z (unknown cells gray)
  row 3   known mask at the same slices

What to look for: the zero-crossing band (white in RdBu) must sit where the
depth map puts the surface and move coherently with the camera; holes and
frustum edges must read unknown (gray), never fake surface. With real
calibration this becomes the extrinsic acceptance check / flatness test
(design §7.1); with placeholders it still catches units, projection and
masking bugs by eye.

Pure data-path probe: no GPU, no checkpoint, no encoder — grid in, pixels out.

Run:
    uv run python -m lerobot.probes.tsdf_slices --config config_rl.yaml
"""

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.video_utils import encode_video_frames
from lerobot.policies.depth_tsdf.modeling_depth_tsdf import build_tsdf_grid
from lerobot.probes.utils import build_episode_index
from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png
from lerobot.utils.utils import init_logging

NUM_Z_SLICES = 6


@dataclass
class TsdfSlicesConfig(TrainRLServerPipelineConfig):
    slices_episode: int = 0
    slices_num_sheets: int = 8  # evenly spaced standalone contact sheets
    slices_video: bool = True  # also encode an mp4 over the episode
    slices_stride: int = 2  # render every Nth frame for the video
    slices_output_dir: str = "outputs/probes/tsdf_slices"


def _box_edges_px(cfg_t) -> list[tuple[np.ndarray, np.ndarray]]:
    """The grid box's 12 edges projected into the raw depth image, in pixels.

    Catches a wrong extrinsic at a glance: the footprint must sit where the
    grid is supposed to look. Edges with an endpoint behind the camera are
    dropped rather than clipped.
    """
    fx, fy, cx, cy = cfg_t.intrinsics
    side = cfg_t.grid_size * cfg_t.voxel_size_mm
    box_min = np.array(cfg_t.box_min_mm)
    corners = np.array([box_min + side * np.array([i, j, k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    t_c_from_g = np.linalg.inv(np.array(cfg_t.t_g_from_c).reshape(4, 4))
    q = corners @ t_c_from_g[:3, :3].T + t_c_from_g[:3, 3]
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") != 1:  # corner indices differ in exactly one axis bit
                continue
            if q[a, 2] <= 0 or q[b, 2] <= 0:
                continue
            ua = (fx * q[a, 0] / q[a, 2] + cx, fy * q[a, 1] / q[a, 2] + cy)
            ub = (fx * q[b, 0] / q[b, 2] + cx, fy * q[b, 1] / q[b, 2] + cy)
            edges.append((np.array([ua[0], ub[0]]), np.array([ua[1], ub[1]])))
    return edges


def _imshow_phi(ax, phi: np.ndarray, known: np.ndarray, title: str) -> None:
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("0.65")
    ax.imshow(np.ma.masked_where(known < 0.5, phi), cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _imshow_known(ax, known: np.ndarray, title: str) -> None:
    ax.imshow(known, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _render_sheet(rgb: np.ndarray | None, depth_raw: np.ndarray, grid: torch.Tensor, cfg_t, suptitle: str):
    phi, known = grid[0, 0].numpy(), grid[0, 1].numpy()
    n = phi.shape[0]
    mid = n // 2
    z0 = cfg_t.box_min_mm[2]
    voxel = cfg_t.voxel_size_mm

    # Fixed-size figure (no tight bbox): video frames must all share dimensions.
    fig, axes = plt.subplots(3, NUM_Z_SLICES, figsize=(18, 9), dpi=100, constrained_layout=True)
    fig.suptitle(suptitle, fontsize=10)

    ax = axes[0, 0]
    if rgb is not None:
        ax.imshow(rgb)
    ax.set_title("RGB", fontsize=8)
    ax.axis("off")

    ax = axes[0, 1]
    depth_mm = depth_raw.astype(np.float32) * cfg_t.depth_units_mm
    holes = depth_raw == 0
    vmax = np.percentile(depth_mm[~holes], 99) if (~holes).any() else 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    ax.imshow(np.ma.masked_where(holes, depth_mm), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    for xs, ys in _box_edges_px(cfg_t):
        ax.plot(xs, ys, color="red", linewidth=1.0)
    ax.set_xlim(0, depth_raw.shape[1])
    ax.set_ylim(depth_raw.shape[0], 0)
    ax.set_title("depth (mm, holes black, box red)", fontsize=8)
    ax.axis("off")

    _imshow_phi(axes[0, 2], phi[:, mid, :], known[:, mid, :], "φ xz @ y-mid (x↓ z→)")
    _imshow_known(axes[0, 3], known[:, mid, :], "known xz @ y-mid")
    _imshow_phi(axes[0, 4], phi[mid, :, :], known[mid, :, :], "φ yz @ x-mid (y↓ z→)")
    _imshow_known(axes[0, 5], known[mid, :, :], "known yz @ x-mid")

    for col, k in enumerate(np.linspace(0, n - 1, NUM_Z_SLICES).round().astype(int)):
        z_mm = z0 + (k + 0.5) * voxel
        _imshow_phi(axes[1, col], phi[:, :, k], known[:, :, k], f"φ xy @ z={z_mm:.0f}mm")
        _imshow_known(axes[2, col], known[:, :, k], f"known @ z={z_mm:.0f}mm")
    return fig


def _frame_rgb(dataset, global_idx: int, depth_key: str) -> np.ndarray | None:
    frame = dataset[global_idx]
    key = f"observation.images.{depth_key}"
    if key not in frame:
        image_keys = [k for k in frame if k.startswith("observation.images.")]
        if not image_keys:
            return None
        key = image_keys[0]
    return frame[key].permute(1, 2, 0).clamp(0, 1).numpy()


@parser.wrap()
def cli(cfg: TsdfSlicesConfig):
    init_logging()
    cfg_t = getattr(cfg.policy, "tsdf_config", None)
    if cfg_t is None:
        raise SystemExit("policy.tsdf_config is null in this config — nothing to visualize.")

    from lerobot.datasets.factory import make_dataset

    dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None

    ep_to_indices = build_episode_index(dataset)
    if cfg.slices_episode not in ep_to_indices:
        raise SystemExit(f"Episode {cfg.slices_episode} not in dataset (have {sorted(ep_to_indices)}).")
    episode_indices = ep_to_indices[cfg.slices_episode]

    out_dir = Path(cfg.slices_output_dir) / f"episode-{cfg.slices_episode:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "video_frames"

    stride = max(1, cfg.slices_stride)
    render_positions = (
        list(range(0, len(episode_indices), stride))
        if cfg.slices_video
        else np.linspace(0, len(episode_indices) - 1, min(cfg.slices_num_sheets, len(episode_indices)))
        .round()
        .astype(int)
        .tolist()
    )
    sheet_positions = set(
        np.array(render_positions)[
            np.linspace(0, len(render_positions) - 1, min(cfg.slices_num_sheets, len(render_positions)))
            .round()
            .astype(int)
        ].tolist()
    )
    if cfg.slices_video:
        frames_dir.mkdir(exist_ok=True)

    t_g_from_c = torch.tensor(cfg_t.t_g_from_c, dtype=torch.float32).reshape(4, 4)
    rendered = 0
    for seq, pos in enumerate(render_positions):
        global_idx = episode_indices[pos]
        fr_idx = dataset.hf_dataset[global_idx]["frame_index"].item()
        try:
            depth_raw = load_depth_png(dataset.root, f"{cfg_t.depth_key}.depth", cfg.slices_episode, fr_idx)
        except FileNotFoundError:
            logging.warning(f"frame {fr_idx}: no depth sidecar, skipping.")
            continue
        grid = build_tsdf_grid(
            torch.from_numpy(depth_raw.astype(np.float32)).unsqueeze(0),
            intrinsics=tuple(cfg_t.intrinsics),
            t_g_from_c=t_g_from_c,
            box_min_mm=tuple(cfg_t.box_min_mm),
            voxel_size_mm=cfg_t.voxel_size_mm,
            grid_size=cfg_t.grid_size,
            truncation_mm=cfg_t.truncation_mm,
            depth_units_mm=cfg_t.depth_units_mm,
        )
        depth_mm = depth_raw.astype(np.float32) * cfg_t.depth_units_mm
        nonzero = depth_mm[depth_raw > 0]
        suptitle = (
            f"ep {cfg.slices_episode} frame {fr_idx} — "
            f"depth med {np.median(nonzero):.0f}mm, holes {(depth_raw == 0).mean():.1%}, "
            f"grid known {grid[0, 1].mean().item():.1%}"
        )
        fig = _render_sheet(_frame_rgb(dataset, global_idx, cfg_t.depth_key), depth_raw, grid, cfg_t, suptitle)
        if cfg.slices_video:
            fig.savefig(frames_dir / f"frame-{seq:06d}.png")
        if pos in sheet_positions:
            fig.savefig(out_dir / f"sheet-frame-{fr_idx:06d}.png")
        plt.close(fig)
        rendered += 1

    if rendered == 0:
        raise SystemExit("No frames rendered — does this episode have depth sidecars?")
    logging.info(f"{rendered} frames rendered, sheets in {out_dir}")

    if cfg.slices_video:
        fps = max(1, round(dataset.fps / stride))
        video_path = out_dir / "tsdf_slices.mp4"
        encode_video_frames(frames_dir, video_path, fps, overwrite=True)
        shutil.rmtree(frames_dir)
        logging.info(f"Video: {video_path} ({fps} fps)")


def main() -> None:
    # Same pre-parse machinery as rl_offline: register policy configs, strip inactive-model YAML fields.
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    import lerobot.rl.pi05.rl_pi05  # noqa: F401 — registers PI05RLConfig
    from lerobot.robots import so_follower  # noqa: F401 — registers so101_follower
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import so_leader  # noqa: F401 — registers so101_leader

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    main()

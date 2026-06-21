"""Configuration for back-projected point-map depth tokens (depth_pointmap_design.md).

Model-agnostic: policies embed this as ``pointmap_config: DepthPointmapConfig | None``
(the RTCConfig pattern). ``None`` means depth-free — no encoder is built, no depth
ships, forward cost is unchanged. How the emitted tokens enter a specific policy
(the attention seam) lives with that policy (depth_pointmap_design.md), not here.

Replaces the gripper-frame TSDF voxel config (depth_tsdf). No box, no extrinsic,
no voxel grid: the only setup-specific input is the camera intrinsics. Units are
millimeters throughout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DepthPointmapConfig:
    """Back-projected point-map depth (depth_pointmap_design.md).

    The wrist depth image is back-projected to a 4-channel metric point map
    ``[X, Y, Z, m]`` in the camera frame, cut into ``patch_size`` × ``patch_size``
    non-overlapping patches, and each patch becomes one token.
    """

    # Bare camera name; depth rides observations as observation.depth.{depth_key}.
    depth_key: str = "wrist"
    # raw uint16 × depth_units_mm = mm (D405 Z16 = 0.1 mm/level).
    depth_units_mm: float = 0.1

    # Camera intrinsics (fx, fy, cx, cy) of the raw depth stream, in pixels.
    # PLACEHOLDER D405 ballpark for the raw 640×480 stream; calibrate once.
    intrinsics: tuple[float, float, float, float] = (382.0, 382.0, 320.0, 240.0)

    # Depth image size (H, W). Both must be divisible by patch_size.
    image_size: tuple[int, int] = (480, 640)
    patch_size: int = 40  # 480/40=12, 640/40=16 → 192 tokens

    # Valid-depth band. z_min ≈ D405 near limit; z_max is a soft far-plane (the
    # only "extent" parameter, tuned in practice). Pixels outside → mask 0.
    z_min_mm: float = 70.0
    z_max_mm: float = 800.0

    # Divisor applied to recentered patch coordinates so CNN inputs are O(1).
    coord_scale_mm: float = 25.0
    # Condition the per-patch CNN on the patch's absolute depth z̄ (range-dependent
    # noise, error ∝ z²) by appending it as a constant 5th channel.
    include_centroid_depth: bool = True

    # Fourier position encoding wavelength count. Bounds are derived:
    # λ_min = patch_size·z_min/fx (near token spacing), λ_max = 2·z_max (scene).
    num_wavelengths: int = 8

    # Modality dropout p_drop: swap to the learned null bank at train time.
    dropout_prob: float = 0.25

    # --- MoT co-evolving depth stream (depth_pointmap_design.md Part B) ------
    # The encoder's tokens co-evolve through M light transformer blocks (depth
    # self-attention + cross-attention to the wrist-cam KV), read per-layer by the
    # action expert via a gated additive SDPA + a zero-value sink (revised §3.3).
    # These gate Phase 3 only; the Phase-2 A.3 read ignores them.
    stream_width: int = 512  # d_d — depth stream width (lean, design D4)
    stream_num_heads: int = 8  # heads for the depth self/cross-attention
    stream_layers: int | None = None  # M; None ⇒ one depth block per action-expert layer (M = L)
    stream_mlp_ratio: float = 4.0

    def __post_init__(self) -> None:
        if not self.depth_key:
            raise ValueError("DepthPointmapConfig requires a depth_key (bare camera name).")
        if self.depth_units_mm <= 0:
            raise ValueError(f"depth_units_mm must be > 0, got {self.depth_units_mm}.")
        h, w = self.image_size
        if self.patch_size <= 0 or h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by patch_size {self.patch_size}."
            )
        if not 0 < self.z_min_mm < self.z_max_mm:
            raise ValueError(f"need 0 < z_min < z_max, got ({self.z_min_mm}, {self.z_max_mm}).")
        if self.coord_scale_mm <= 0:
            raise ValueError(f"coord_scale_mm must be > 0, got {self.coord_scale_mm}.")
        if self.num_wavelengths < 1:
            raise ValueError(f"num_wavelengths must be ≥ 1, got {self.num_wavelengths}.")
        if not 0 <= self.dropout_prob < 1:
            raise ValueError(f"dropout_prob must be in [0, 1), got {self.dropout_prob}.")
        if self.stream_width <= 0 or self.stream_width % self.stream_num_heads:
            raise ValueError(
                f"stream_width {self.stream_width} must be > 0 and divisible by "
                f"stream_num_heads {self.stream_num_heads}."
            )
        if self.stream_layers is not None and self.stream_layers < 1:
            raise ValueError(f"stream_layers must be ≥ 1 or None, got {self.stream_layers}.")
        if self.stream_mlp_ratio <= 0:
            raise ValueError(f"stream_mlp_ratio must be > 0, got {self.stream_mlp_ratio}.")

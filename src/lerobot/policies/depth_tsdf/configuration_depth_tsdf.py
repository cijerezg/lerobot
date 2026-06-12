"""Configuration for gripper-frame TSDF depth tokens (depth_tsdf_design.md).

Model-agnostic: policies embed this as ``tsdf_config: DepthTsdfConfig | None``
(the RTCConfig pattern). ``None`` means depth-free — no encoder is built, the
delivery plumbing ships no depth, forward cost is unchanged. How the emitted
memory tokens enter a specific policy (the attention seam) lives with that
policy, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DepthTsdfConfig:
    """Gripper-frame truncated-signed-distance-function depth (design §3, §5–6).

    Distances are millimeters throughout.
    """

    # Bare camera name; depth rides observations as observation.depth.{depth_key}.
    depth_key: str = "wrist"
    # raw uint16 × depth_units_mm = mm. Verify once against a recorded sidecar PNG
    # (roadmap WS1); the builder consumes metric depth, never normalized [0,1].
    depth_units_mm: float = 0.1

    # Grid geometry (design §3.2): N³ voxels of pitch δ anchored at the TCP;
    # the box spans box_min + N·δ per axis.
    grid_size: int = 48  # must be divisible by 8 (3 stride-2 trunk stages)
    voxel_size_mm: float = 2.0
    box_min_mm: tuple[float, float, float] = (-48.0, -48.0, -16.0)
    truncation_mm: float = 8.0

    # Modality dropout p_drop (design §6.3): swap to the learned null bank.
    dropout_prob: float = 0.25

    # PLACEHOLDER calibration (roadmap WS0) — swap for calibrated values in this
    # exact format. Intrinsics (fx, fy, cx, cy): D405 ballpark for the raw
    # 640×480 depth stream. Extrinsic T_{G←C}: row-major 4×4, translation mm;
    # placeholder puts the camera 80 mm behind the TCP looking along the
    # gripper approach axis.
    intrinsics: tuple[float, float, float, float] = (382.0, 382.0, 320.0, 240.0)
    t_g_from_c: tuple[float, ...] = field(
        default=(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, -80.0,
            0.0, 0.0, 0.0, 1.0,
        )
    )  # fmt: skip

    def __post_init__(self) -> None:
        if not self.depth_key:
            raise ValueError("DepthTsdfConfig requires a depth_key (bare camera name).")
        if self.depth_units_mm <= 0:
            raise ValueError(f"depth_units_mm must be > 0, got {self.depth_units_mm}.")
        if self.grid_size < 8 or self.grid_size % 8 != 0:
            raise ValueError(f"grid_size must be a positive multiple of 8, got {self.grid_size}.")
        if self.voxel_size_mm <= 0:
            raise ValueError(f"voxel_size_mm must be > 0, got {self.voxel_size_mm}.")
        if self.truncation_mm <= 0:
            raise ValueError(f"truncation_mm must be > 0, got {self.truncation_mm}.")
        if not 0 <= self.dropout_prob < 1:
            raise ValueError(f"dropout_prob must be in [0, 1), got {self.dropout_prob}.")
        if len(self.t_g_from_c) != 16:
            raise ValueError(f"t_g_from_c must be 16 row-major floats, got {len(self.t_g_from_c)}.")

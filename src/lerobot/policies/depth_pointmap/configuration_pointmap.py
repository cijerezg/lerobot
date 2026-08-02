"""Configuration for back-projected point-map depth tokens (depth_pointmap_design.md).

Model-agnostic: policies embed this as ``pointmap_config: DepthPointmapConfig | None``
(the RTCConfig pattern). ``None`` means depth-free — no encoder is built, no depth
ships, forward cost is unchanged. How the emitted tokens enter a specific policy
(the attention seam) lives with that policy (depth_pointmap_design.md), not here.

No box, no extrinsic, no voxel grid: the only setup-specific input is the camera
intrinsics. Units are millimeters throughout.
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
    # Factory calibration of D405 427622270837, 640×480 z16 (rectified, zero distortion).
    intrinsics: tuple[float, float, float, float] = (394.9832, 394.9832, 322.5604, 238.6966)

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
    # Anti-laziness RGB dropout (depth_redesign_options.md §4.3): at train time,
    # mask the depth camera's <im_patch> span out of the attention mask with this
    # probability — nothing attends the span, no gradient flows through its vision
    # path, and the depth stream's wrist bridge is killed per-row from the same
    # mask. Those samples are solvable only through depth. Independent of the
    # depth modality dropout above. 0 disables.
    rgb_dropout_prob: float = 0.15

    # Patch-CNN trunk widths (the two hidden stages; the third stage outputs the
    # stream width). Bumped from (32, 64) 2026-07-25 — depth capacity is cheap next
    # to the 6B backbone.
    cnn_hidden_channels: tuple[int, int] = (128, 256)

    # Gradient-checkpointing granularity for the patch CNN, in patch rows (a row is
    # one patch of one sample, so a batch of B contributes B·N rows). The trunk is
    # recomputed one chunk at a time, which caps the recompute peak at a value set by
    # this number instead of letting it scale with the batch: the trunk holds
    # full-resolution feature maps for every frame, and at the (128, 256) widths above
    # that is the single largest activation in the whole model. Every op in it (convs,
    # GroupNorm, the same-pixel temporal attention, the final spatial mean) is
    # independent across rows, so splitting them is exact. 0 = one chunk (no cap).
    # Only consulted when the policy enables gradient checkpointing.
    encoder_chunk_rows: int = 384

    # --- MoT co-evolving depth stream (depth_pointmap_design.md Part B) ------
    # The encoder's tokens co-evolve through M light transformer blocks (depth
    # self-attention + cross-attention to the wrist-cam KV), read per-layer by the
    # action expert as extra columns in its context softmax with a learned bias.
    # These settings affect the co-evolving stream only.
    stream_width: int = 512  # d_d — depth stream width (lean, design D4)
    stream_num_heads: int = 8  # heads for the depth self/cross-attention
    stream_layers: int | None = None  # M; None ⇒ one depth block per action-expert layer (M = L)
    stream_mlp_ratio: float = 4.0

    # Temporal history (depth_history_design.md): past depth frames ride the shared
    # patch CNN and are fused into the current frame by same-pixel temporal attention
    # after every CNN block; past rows are dropped before pooling, so the encoder
    # always emits N tokens (0 = current frame only, no fusion modules built).
    # Synced from memory.history_num_samples / history_window_seconds by
    # MolmoAct2RLConfig.__post_init__ when the depth key is in memory.history_keys.
    # Past frames are NOT re-projected into the current camera frame (the wrist
    # moves) — frames are told apart by the sinusoidal e(Δt) stamp only.
    history_num_samples: int = 0
    history_window_seconds: float = 5.0

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
        if not 0 <= self.rgb_dropout_prob < 1:
            raise ValueError(f"rgb_dropout_prob must be in [0, 1), got {self.rgb_dropout_prob}.")
        if not self.cnn_hidden_channels or any(c <= 0 for c in self.cnn_hidden_channels):
            raise ValueError(f"cnn_hidden_channels must be positive, got {self.cnn_hidden_channels}.")
        if self.encoder_chunk_rows < 0:
            raise ValueError(f"encoder_chunk_rows must be >= 0, got {self.encoder_chunk_rows}.")
        if self.history_num_samples < 0:
            raise ValueError(f"history_num_samples must be >= 0, got {self.history_num_samples}.")
        if self.history_window_seconds <= 0:
            raise ValueError(f"history_window_seconds must be > 0, got {self.history_window_seconds}.")
        if self.stream_width <= 0 or self.stream_width % self.stream_num_heads:
            raise ValueError(
                f"stream_width {self.stream_width} must be > 0 and divisible by "
                f"stream_num_heads {self.stream_num_heads}."
            )
        if self.stream_layers is not None and self.stream_layers < 1:
            raise ValueError(f"stream_layers must be ≥ 1 or None, got {self.stream_layers}.")
        if self.stream_mlp_ratio <= 0:
            raise ValueError(f"stream_mlp_ratio must be > 0, got {self.stream_mlp_ratio}.")

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
    patch_size: int = 20  # 24×32 = 768 fine tokens
    # Mirror Molmo's image adapter: attention-pool non-overlapping 2×2 groups,
    # yielding the same 12×16 = 192 prefix-token layout as RGB.
    pooling_size: tuple[int, int] = (2, 2)

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

    # Optional modality dropout: swap to the learned null bank at train time.
    # Disabled by default; keep the mechanism available for controlled ablations.
    dropout_prob: float = 0.0
    # Anti-laziness RGB dropout (depth_redesign_options.md §4.3): at train time,
    # mask the depth camera's <im_patch> span out of the attention mask with this
    # probability — nothing attends the span and no gradient flows through its
    # vision path, so those samples are solvable only through depth. Independent of
    # the depth modality dropout above. Disabled by default; keep the mechanism
    # available for controlled ablations.
    rgb_dropout_prob: float = 0.0

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

    # Match Molmo's ViT hidden size. The actor sends these through a separate,
    # seven-block depth copy of Molmo's visual path before pooling/projecting them.
    token_width: int = 1152
    visual_num_blocks: int = 7
    # One-based outputs to concatenate. The pooler consumes [block7; block3],
    # matching Molmo's [late; earlier] concatenation order.
    visual_feature_taps: tuple[int, int] = (3, 7)
    # RGB block indices used only to initialize the independent depth blocks.
    visual_source_indices: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24)
    # Elementwise soft bound on the complete token injected into the VLM prefix:
    # output_bound * tanh((projected_feature + depth_marker) / output_bound).
    # This is locally identity-like around zero while preventing the depth path or
    # its trainable marker from growing the fusion input without limit.
    output_bound: float = 128.0

    # CRITIC-ONLY (rl_molmoact2.py): the critic still runs its own co-evolving
    # DepthStreamBlocks over its own encoder's tokens. The actor no longer does —
    # its depth goes through the VLM prefix. These two knobs exist solely for that
    # remaining critic path and go away when it is migrated to the same seam.
    stream_num_heads: int = 8
    stream_mlp_ratio: float = 4.0
    # Preserve the critic's existing, cheaper point-map stream while the actor uses
    # the new fine visual grid. These knobs do not affect the actor path.
    critic_patch_size: int = 40
    critic_token_width: int = 512

    # Temporal history (depth_history_design.md): past depth frames ride the shared
    # patch CNN and are fused into the current frame by same-pixel temporal attention
    # after every CNN block; past rows are dropped before pooling, so the encoder
    # always emits N tokens (0 = current frame only, no fusion modules built).
    # Synced from memory.history_offsets_seconds / history_num_samples /
    # history_window_seconds by MolmoAct2RLConfig.__post_init__ when the depth key is
    # in memory.history_keys. Past frames are NOT re-projected into the current camera
    # frame (the wrist moves) — frames are told apart by the sinusoidal e(Δt) stamp
    # only, so the stamps must carry the real slot ages: history_times_seconds when
    # set (oldest → newest, e.g. [6.0, 4.0, 2.0]), else the uniform ladder derived
    # from the window/count pair.
    history_num_samples: int = 0
    history_window_seconds: float = 5.0
    history_times_seconds: list[float] | None = None

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
        fine_h, fine_w = self.fine_grid_size
        pool_h, pool_w = self.pooling_size
        if pool_h <= 0 or pool_w <= 0 or fine_h % pool_h or fine_w % pool_w:
            raise ValueError(
                f"fine depth grid {self.fine_grid_size} must be divisible by "
                f"pooling_size {self.pooling_size}."
            )
        if self.visual_num_blocks <= 0:
            raise ValueError(f"visual_num_blocks must be > 0, got {self.visual_num_blocks}.")
        if len(self.visual_source_indices) != self.visual_num_blocks:
            raise ValueError("visual_source_indices must contain one source per depth block.")
        if len(set(self.visual_source_indices)) != len(self.visual_source_indices):
            raise ValueError("visual_source_indices must be unique.")
        if (
            len(self.visual_feature_taps) != 2
            or self.visual_feature_taps[0] >= self.visual_feature_taps[1]
            or not all(1 <= tap <= self.visual_num_blocks for tap in self.visual_feature_taps)
        ):
            raise ValueError(
                "visual_feature_taps must contain two increasing, valid one-based block numbers."
            )
        if self.output_bound <= 0:
            raise ValueError(f"output_bound must be > 0, got {self.output_bound}.")
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
        if self.history_times_seconds is not None:
            self.history_times_seconds = [abs(float(t)) for t in self.history_times_seconds]
            if len(self.history_times_seconds) != self.history_num_samples:
                raise ValueError(
                    f"history_times_seconds has {len(self.history_times_seconds)} entries but "
                    f"history_num_samples is {self.history_num_samples}."
                )
        if self.token_width <= 0 or self.token_width % self.stream_num_heads:
            raise ValueError(
                f"token_width {self.token_width} must be > 0 and divisible by "
                f"stream_num_heads {self.stream_num_heads} (the critic's stream blocks)."
            )
        if self.stream_mlp_ratio <= 0:
            raise ValueError(f"stream_mlp_ratio must be > 0, got {self.stream_mlp_ratio}.")
        if self.critic_patch_size <= 0 or h % self.critic_patch_size or w % self.critic_patch_size:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by critic_patch_size "
                f"{self.critic_patch_size}."
            )
        if self.critic_token_width <= 0 or self.critic_token_width % self.stream_num_heads:
            raise ValueError(
                f"critic_token_width {self.critic_token_width} must be > 0 and divisible by "
                f"stream_num_heads {self.stream_num_heads}."
            )

    @property
    def fine_grid_size(self) -> tuple[int, int]:
        h, w = self.image_size
        return h // self.patch_size, w // self.patch_size

    @property
    def pooled_grid_size(self) -> tuple[int, int]:
        fine_h, fine_w = self.fine_grid_size
        pool_h, pool_w = self.pooling_size
        return fine_h // pool_h, fine_w // pool_w

    @property
    def num_fine_tokens(self) -> int:
        h, w = self.fine_grid_size
        return h * w

    @property
    def num_pooled_tokens(self) -> int:
        h, w = self.pooled_grid_size
        return h * w

"""Back-projected point-map depth tokens (depth_pointmap_design.md).

Pipeline:

  depth image ──back_project──▶ 4-channel point map [X, Y, Z, m] (camera frame, mm)
            ──patchify──▶ 192 non-overlapping P×P patches
            ──per patch──▶ token = f (local-shape 2D CNN, recentered) + g (Fourier
                           PE of the absolute centroid)

A single depth frame is a heightfield (one Z per pixel), so the image plane is its
natural dense domain: the CNN is 2D in *structure* (dense, hole-friendly via the
mask channel, no quantization) but 3D in *content* (each pixel carries its metric
position). Recentering each patch to its centroid removes position but keeps metric
scale (a near patch has small Δ, a far one large), so f is translation-invariant
local shape; the absolute centroid goes to the position encoding g instead.

Emits (B, N, d_mem) fine-grid tokens. The actor sends them through its copied visual path into the VLM
prefix on DEPTH_TOKEN placeholders; the critic still runs them through its own
DepthStreamBlocks (modeling_stream.py). Units: millimeters.

Short-term history (depth_history_design.md): past frames ride the same CNN as extra
batch rows and are fused into the current frame by same-pixel temporal attention
(TemporalFusion) after every block — the CNN analog of the MEM video encoder's
temporal layers. Past rows are dropped before pooling, so N never changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from torch import Tensor, nn

from .configuration_pointmap import DepthPointmapConfig


def back_project(
    depth: Tensor,
    *,
    intrinsics: tuple[float, float, float, float],
    depth_units_mm: float,
    z_min_mm: float,
    z_max_mm: float,
) -> Tensor:
    """Depth image → metric point map in the camera frame (design §1).

    Args:
        depth: (H, W), (B, H, W) or (B, 1, H, W) raw depth; uint16 or float. 0 = hole.
        intrinsics: (fx, fy, cx, cy) of the raw depth stream, in pixels.
        depth_units_mm: raw value × this = mm.
        z_min_mm, z_max_mm: valid-depth band; outside → mask 0 (near deadzone, far cutoff).

    Returns:
        (B, 4, H, W) float32 = [X, Y, Z, m]. Z is the depth; X=(u-cx)Z/fx,
        Y=(v-cy)Z/fy; m is the validity mask. Invalid pixels have X=Y=Z=0.
    """
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)  # single live frame (H, W) → (1, H, W)
    if depth.ndim == 4:
        depth = depth.squeeze(1)
    if depth.ndim != 3:
        raise ValueError(f"back_project expects (B, H, W) depth, got {tuple(depth.shape)}.")
    z = depth.to(torch.float32) * depth_units_mm  # (B, H, W) mm
    _, height, width = z.shape
    fx, fy, cx, cy = intrinsics

    m = (z >= z_min_mm) & (z <= z_max_mm)
    mf = m.to(torch.float32)
    vv, uu = torch.meshgrid(
        torch.arange(height, device=z.device, dtype=torch.float32),
        torch.arange(width, device=z.device, dtype=torch.float32),
        indexing="ij",
    )
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return torch.stack([x * mf, y * mf, z * mf, mf], dim=1)


def patchify(pointmap: Tensor, patch_size: int) -> Tensor:
    """(B, C, H, W) → (B, N, C, P, P), non-overlapping, row-major patch order."""
    b, c, h, w = pointmap.shape
    p = patch_size
    nh, nw = h // p, w // p
    x = pointmap.reshape(b, c, nh, p, nw, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, nh * nw, c, p, p)


def fourier_position_encoding(
    centers_mm: Tensor, *, lambda_max_mm: float, lambda_min_mm: float, num_wavelengths: int
) -> Tensor:
    """Geometric sin/cos ladder over metric positions (design §4).

    centers_mm: (T, 3) → (T, 6 * num_wavelengths). A wavelength λ_k is the distance
    over which sinusoid k completes one cycle.
    """
    device = centers_mm.device
    ratio = lambda_min_mm / lambda_max_mm
    steps = torch.arange(num_wavelengths, device=device, dtype=torch.float32) / max(num_wavelengths - 1, 1)
    wavelengths = lambda_max_mm * ratio**steps
    angles = 2 * torch.pi * centers_mm.to(torch.float32).unsqueeze(-1) / wavelengths  # (T, 3, L)
    return torch.cat([angles.sin(), angles.cos()], dim=-1).reshape(centers_mm.shape[0], -1)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, channels), channels)


def _sinusoidal_seconds_embedding(times_s: Tensor, dim: int) -> Tensor:
    """e(Δt) with e(0) = 0 — the same convention as the MEM video encoder
    (modeling_molmoact2._sinusoidal_seconds_embedding): standard sinusoidal PE
    shifted by PE(0), so the current frame (Δt = 0) carries a zero stamp.
    times_s: (T,) seconds in the past → (T, dim), float32."""
    times_s = times_s.to(torch.float32)
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=times_s.device) * (-math.log(10000.0) / half)
    )
    angles = times_s[:, None] * freqs[None, :]
    emb = torch.zeros((times_s.shape[0], dim), dtype=torch.float32, device=times_s.device)
    emb[:, 0::2] = torch.sin(angles)
    emb[:, 1::2] = torch.cos(angles) - 1.0
    return emb


class TemporalFusion(nn.Module):
    """Same-pixel temporal attention + MLP after one CNN block (depth_history_design.md §2).

    The input holds the same patch at T time slices, oldest → newest with the
    current frame last. Each slice is stamped with sinusoidal e(Δt) (attention is
    permutation-invariant, so it must be told frame ages; e(0) = 0), then every
    pixel of the current frame queries that pixel across all frames in one softmax —
    itself included, which is the abstain route: when no past frame is informative
    the mass parks on the current frame. Past frames are keys/values only and pass
    through unchanged; the fused current frame is refined by a pixelwise MLP. All
    projections are 1×1 convs (per-pixel linear maps shared across pixels/patches).
    """

    MLP_RATIO = 4

    def __init__(self, channels: int, times_s: Tensor) -> None:
        super().__init__()
        self.num_heads = max(1, channels // 64)
        if channels % self.num_heads:
            self.num_heads = 1
        self.head_dim = channels // self.num_heads
        self.norm_attn = _group_norm(channels)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.o_proj = nn.Conv2d(channels, channels, 1)
        self.norm_mlp = _group_norm(channels)
        hidden = channels * self.MLP_RATIO
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, channels, 1)
        )
        # (T_h+1, C) e(Δt) stamp; derived from the config, so not checkpointed.
        self.register_buffer(
            "time_embed", _sinusoidal_seconds_embedding(times_s, channels), persistent=False
        )

    def forward(self, x: Tensor, history_on: Tensor | None) -> Tensor:
        """x: (M, T, C, H, W), current frame last, T ≤ len(time_embed) (T = 1 when
        the window is missing — cold deque / plain eval). history_on: (M,) bool or
        None; False masks the sample's past keys (the shared MEM history-dropout
        draw), which computes exactly the T = 1 op. Returns x with the current
        slice fused, past slices untouched."""
        m, t, c, h, w = x.shape
        stamped = x + self.time_embed[-t:].to(x.dtype).view(1, t, c, 1, 1)
        normed = self.norm_attn(stamped.reshape(m * t, c, h, w))
        k = self.k_proj(normed).view(m, t, self.num_heads, self.head_dim, h * w)
        v = self.v_proj(normed).view(m, t, self.num_heads, self.head_dim, h * w)
        q = self.q_proj(normed.view(m, t, c, h, w)[:, -1])
        q = q.view(m, self.num_heads, self.head_dim, h * w)
        scores = torch.einsum("mhdp,mthdp->mthp", q, k) / math.sqrt(self.head_dim)
        if history_on is not None and t > 1:
            past = torch.zeros(t, dtype=torch.bool, device=x.device)
            past[:-1] = True
            blocked = past.view(1, t, 1, 1) & ~history_on.view(m, 1, 1, 1)
            scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1)
        fused = torch.einsum("mthp,mthdp->mhdp", weights, v).reshape(m, c, h, w)
        current = x[:, -1] + self.o_proj(fused)
        current = current + self.mlp(self.norm_mlp(current))
        return torch.cat([x[:, :-1], current.unsqueeze(1)], dim=1)


class PatchResidualBlock2d(nn.Module):
    """Stride-2 residual downsampling block, GroupNorm/SiLU (design §3)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
        self.norm1 = _group_norm(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.norm2 = _group_norm(c_out)
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class PatchShapeCNN(nn.Module):
    """Per-patch 2D CNN over the recentered point map → one feature vector.

    Shared across all patches (a conv shares its filters by construction). Applied
    to (M, C_in, P, P) and global-average-pooled to (M, d_out). For the actor's
    P=20, the three stride-2 blocks downsample 20→10→5→3 before pooling. `blocks` is a ModuleList
    (same state-dict keys as the former Sequential) so the encoder's history path
    can interleave a TemporalFusion after each block; `out_channels` tells it the
    per-block widths.
    """

    def __init__(self, in_channels: int, hidden: tuple[int, ...], d_out: int) -> None:
        super().__init__()
        dims = (in_channels, *hidden, d_out)
        self.out_channels = dims[1:]
        self.blocks = nn.ModuleList(
            PatchResidualBlock2d(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=(-1, -2))


class DepthPointmapEncoder(nn.Module):
    """Point map → depth-stream input tokens (design §3–5).

    Input : (B, 4, H, W) from build via back_project.
    Output: (B, N, d_mem) tokens, N = (H/P)(W/P), where d_mem is config.token_width.
    The actor adapts these straight into the VLM prefix; the critic co-evolves them,
    which owns the joint-softmax read projections and bias.

    null_tokens is the learned per-patch bank substituted for empty patches (all
    pixels invalid) and under whole-sample modality dropout / depth-missing.
    """

    def __init__(
        self, config: DepthPointmapConfig, *, d_mem: int, gradient_checkpointing: bool = False
    ) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        height, width = config.image_size
        self.num_tokens = config.num_fine_tokens

        in_channels = 4 + (1 if config.include_centroid_depth else 0)
        self.cnn = PatchShapeCNN(in_channels, config.cnn_hidden_channels, d_mem)

        # PE bounds derived so they track the far cutoff (design §4): λ_min = near
        # token spacing P·z_min/fx, λ_max = 2·z_max (must span the scene or alias).
        fx = config.intrinsics[0]
        self.lambda_min_mm = config.patch_size * config.z_min_mm / fx
        self.lambda_max_mm = 2.0 * config.z_max_mm
        self.pos_proj = nn.Linear(6 * config.num_wavelengths, d_mem)

        self.modality_embed = nn.Parameter(torch.randn(d_mem) * 0.02)
        self.null_tokens = nn.Parameter(torch.randn(self.num_tokens, d_mem) * 0.02)
        # Short-term history (depth_history_design.md): one TemporalFusion per CNN
        # block, fusing past frames into the current one inside the trunk. Only
        # created when history is on, so historyless checkpoints load unchanged.
        self.temporal_fusions: nn.ModuleList | None = None
        if config.history_num_samples > 0:
            stride_s = config.history_window_seconds / config.history_num_samples
            times_s = torch.tensor(
                [stride_s * (config.history_num_samples - i) for i in range(config.history_num_samples)]
                + [0.0]
            )  # seconds in the past, oldest → newest, current (0 s) last
            self.temporal_fusions = nn.ModuleList(
                TemporalFusion(c, times_s) for c in self.cnn.out_channels
            )

    def null_memory(self, batch_size: int) -> Tensor:
        """Null bank for the token assembly — (B, N, d_mem); history never changes
        the token count, so there are no history slots to null."""
        return self.null_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def _fuse_trunk(self, x: Tensor, history_on: Tensor | None) -> Tensor:
        """History trunk: (M, T, C_in, P, P) → (M, d_mem). One TemporalFusion after
        each CNN block; past rows are dropped at the final pooling."""
        for block, fusion in zip(self.cnn.blocks, self.temporal_fusions, strict=True):
            m, t = x.shape[:2]
            y = block(x.reshape(m * t, *x.shape[2:]))
            x = fusion(y.view(m, t, *y.shape[1:]), history_on)
        return x[:, -1].mean(dim=(-1, -2))

    def _plain_trunk(self, x: Tensor, history_on: Tensor | None) -> Tensor:  # noqa: ARG002
        """Historyless trunk: (M, C_in, P, P) → (M, d_mem)."""
        return self.cnn(x)

    def _run_trunk(self, trunk, x: Tensor, history_on: Tensor | None) -> Tensor:
        """Run a trunk over the patch-row axis in gradient-checkpointed chunks.

        The trunk is the model's largest activation by a wide margin (full-resolution
        feature maps × every history frame, in float32), and it is the one module that
        used to run without checkpointing. Chunking is exact — every op inside is
        independent across patch rows — and it is what keeps the recompute peak from
        scaling with the batch. See ``encoder_chunk_rows``."""
        if not (self.gradient_checkpointing and self.training and torch.is_grad_enabled()):
            return trunk(x, history_on)
        rows = self.config.encoder_chunk_rows or x.shape[0]
        return torch.cat(
            [
                torch.utils.checkpoint.checkpoint(
                    trunk,
                    x[start : start + rows],
                    None if history_on is None else history_on[start : start + rows],
                    use_reentrant=False,
                )
                for start in range(0, x.shape[0], rows)
            ],
            dim=0,
        )

    def _patch_inputs(self, pointmap: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(B, 4, H, W) point map → per-patch CNN input, centroids, empty mask:
        cnn_in (B, N, C_in, P, P), centroid (B, N, 3) mm, empty (B, N) bool."""
        cfg = self.config
        p = cfg.patch_size
        patches = patchify(pointmap, p)  # (B, N, 4, P, P)
        b, n = patches.shape[:2]
        coords = patches[:, :, :3]  # (B, N, 3, P, P)
        mask = patches[:, :, 3:4]  # (B, N, 1, P, P)

        count = mask.sum(dim=(-1, -2))  # (B, N, 1)
        centroid = (coords * mask).sum(dim=(-1, -2)) / count.clamp(min=1.0)  # (B, N, 3)
        empty = count.squeeze(-1) == 0  # (B, N)

        delta = (coords - centroid[..., None, None]) * mask / cfg.coord_scale_mm
        cnn_in = torch.cat([delta, mask], dim=2)  # (B, N, 4, P, P)
        if cfg.include_centroid_depth:
            zbar = (centroid[:, :, 2:3] / cfg.coord_scale_mm)[..., None, None]  # (B, N, 1, 1, 1)
            cnn_in = torch.cat([cnn_in, zbar.expand(b, n, 1, p, p)], dim=2)
        return cnn_in, centroid, empty

    def forward(
        self,
        pointmap: Tensor,
        history_pointmaps: Tensor | None = None,
        history_on: Tensor | None = None,
    ) -> Tensor:
        """Point maps → (B, N, d_mem) tokens. history_pointmaps (B, T_h, 4, H, W),
        oldest → newest, rides the shared CNN blocks and is fused into the current
        frame by the temporal attention (built iff history_num_samples > 0); the
        token count is N regardless. history_on (B,) bool masks a sample's past
        keys (history dropout). A fully empty current patch falls back to the newest
        valid historical centroid and fused feature; it routes to the null bank when
        history is masked/missing or every frame's patch is empty."""
        cfg = self.config
        pointmap = pointmap.to(self.pos_proj.weight.dtype)
        cnn_in, centroid, empty = self._patch_inputs(pointmap)
        b, n = cnn_in.shape[:2]
        history_enabled = None
        if history_on is not None:
            history_enabled = history_on.to(device=pointmap.device, dtype=torch.bool)
            if history_enabled.shape != (b,):
                raise ValueError(
                    f"history_on must have shape ({b},), got {tuple(history_enabled.shape)}."
                )

        if self.temporal_fusions is None:
            f = self._run_trunk(
                self._plain_trunk, cnn_in.reshape(b * n, *cnn_in.shape[2:]), None
            ).reshape(b, n, -1)
        else:
            x = cnn_in.unsqueeze(2)  # (B, N, 1, C_in, P, P) — current frame last
            if history_pointmaps is not None:
                t_h = history_pointmaps.shape[1]
                if t_h != cfg.history_num_samples:
                    raise ValueError(
                        f"history_pointmaps has {t_h} frames, expected "
                        f"history_num_samples={cfg.history_num_samples}."
                    )
                hist_in, hist_centroid, hist_empty = self._patch_inputs(
                    history_pointmaps.to(device=pointmap.device, dtype=pointmap.dtype).reshape(
                        b * t_h, *history_pointmaps.shape[2:]
                    )
                )
                hist_centroid = hist_centroid.reshape(b, t_h, n, 3)
                hist_empty = hist_empty.reshape(b, t_h, n)
                history_valid = ~hist_empty
                if history_enabled is not None:
                    history_valid = history_valid & history_enabled[:, None, None]
                has_valid_history = history_valid.any(dim=1)
                latest_from_end = (
                    history_valid.flip(dims=(1,)).to(dtype=torch.int64).argmax(dim=1)
                )
                latest_idx = t_h - 1 - latest_from_end
                fallback_centroid = torch.gather(
                    hist_centroid,
                    dim=1,
                    index=latest_idx[:, None, :, None].expand(-1, 1, -1, 3),
                ).squeeze(1)
                recover_from_history = empty & has_valid_history
                centroid = torch.where(recover_from_history[..., None], fallback_centroid, centroid)
                empty = empty & ~has_valid_history
                hist_in = hist_in.reshape(b, t_h, n, *hist_in.shape[2:]).transpose(1, 2)
                x = torch.cat([hist_in, x], dim=2)  # (B, N, T_h+1, C_in, P, P)
            x = x.reshape(b * n, *x.shape[2:])  # (M, T, C_in, P, P)
            on = None
            if history_enabled is not None:
                on = history_enabled.repeat_interleave(n)
            f = self._run_trunk(self._fuse_trunk, x, on).reshape(b, n, -1)  # past rows dropped inside

        pe = fourier_position_encoding(
            centroid.reshape(b * n, 3),
            lambda_max_mm=self.lambda_max_mm,
            lambda_min_mm=self.lambda_min_mm,
            num_wavelengths=cfg.num_wavelengths,
        )
        g = self.pos_proj(pe.to(self.pos_proj.weight.dtype)).reshape(b, n, -1)

        token = f + g + self.modality_embed
        null = self.null_tokens.unsqueeze(0).to(token.dtype)
        return torch.where(empty[..., None], null, token)

    def memory_from_batch(
        self, batch: dict[str, Tensor], *, batch_size: int, device: torch.device
    ) -> Tensor:
        """Depth tokens from a policy batch: back-project → encode (design §1, §5).

        Consumes raw metric depth from observation.depth.{depth_key} (no [0,1]
        normalizer on this path). The history window
        history.depth.{depth_key}.depth (B, T_h, H, W), oldest → newest, is
        back-projected per frame and fused inside the CNN (depth_history_design.md);
        `history_images_mask` — the one shared MEM history-dropout draw — masks a
        sample's temporal keys. A missing window computes the same op as a fully
        masked one (T = 1 attention). Padded slots at episode start arrive
        repeat-padded from the buffer clamp and are used as-is (repeat-pad v1, like
        the video encoder); the _is_pad sidecar is deliberately ignored. Swaps in
        the learned null bank under modality dropout at train time and whenever
        depth is missing, keeping shapes static.

        Returns fine-grid memory (B, N, d_mem) — the actor visual path's input.
        """
        cfg = self.config
        depth = batch.get(f"observation.depth.{cfg.depth_key}")
        if depth is None:
            memory = self.null_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            pointmap = self._backproject(torch.as_tensor(depth).to(device=device))
            history_pm = history_on = None
            if self.temporal_fusions is not None:
                window = batch.get(f"history.depth.{cfg.depth_key}.depth")
                if window is not None:
                    window = torch.as_tensor(window).to(device=device)
                    if window.shape[1] != cfg.history_num_samples:
                        raise ValueError(
                            f"depth history window has {window.shape[1]} frames, expected "
                            f"history_num_samples={cfg.history_num_samples}."
                        )
                    history_pm = self._backproject(
                        window.reshape(batch_size * cfg.history_num_samples, *window.shape[2:])
                    ).reshape(batch_size, cfg.history_num_samples, *pointmap.shape[1:])
                    mask = batch.get("history_images_mask")
                    if mask is not None:
                        history_on = torch.as_tensor(mask).to(device=device, dtype=torch.bool)
            memory = self(pointmap, history_pm, history_on)
        if self.training and cfg.dropout_prob > 0:
            dropped = torch.rand(memory.shape[0], device=memory.device) < cfg.dropout_prob
            memory = torch.where(
                dropped[:, None, None], self.null_memory(memory.shape[0]).to(memory.dtype), memory
            )
        return memory

    def _backproject(self, depth: Tensor) -> Tensor:
        cfg = self.config
        return back_project(
            depth,
            intrinsics=tuple(cfg.intrinsics),
            depth_units_mm=cfg.depth_units_mm,
            z_min_mm=cfg.z_min_mm,
            z_max_mm=cfg.z_max_mm,
        )

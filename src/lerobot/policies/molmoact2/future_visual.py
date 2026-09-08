"""Training-only future prediction in an aligned temporal-PCA feature space.

The policy's RGB encoder and token layout are unchanged. A frozen single-frame
ViT supplies the target. Its weights and the PCA basis refresh only at optimizer
boundaries. Raw calibration pairs are retained so PCA is always fitted using the
new target weights, never a mixture of feature spaces from different targets.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from .configuration_molmoact2 import FutureVisualLossConfig


def normalize_taps(features: Tensor, tap_width: int, eps: float = 1e-6) -> Tensor:
    shape = features.shape
    taps = features.float().reshape(*shape[:-1], -1, tap_width)
    return F.layer_norm(taps, (tap_width,), eps=eps).reshape(shape)


def temporal_pca(differences: Tensor, rank: int, previous: Tensor | None = None):
    """Uncentered change energy, with Procrustes-aligned output coordinates.

    Not whitening: small-variance directions must not be amplified. Alignment
    rotates within the selected subspace without changing the energy it retains.
    """
    delta = differences.reshape(-1, differences.shape[-1]).float()
    if delta.shape[0] < rank:
        return None
    with torch.autocast(device_type=delta.device.type, enabled=False):
        covariance = delta.T @ delta / delta.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        energy = eigenvalues.clamp_min(0).sum()
        if not torch.isfinite(energy) or energy <= 1e-8:
            return None
        basis = eigenvectors[:, -rank:].flip(1)
        if previous is not None:
            u, _, vh = torch.linalg.svd(basis.T @ previous.float(), full_matrices=False)
            basis = basis @ (u @ vh)
        else:
            # Deterministic signs for the initial basis.
            maxima = basis.abs().argmax(dim=0)
            signs = basis[maxima, torch.arange(rank, device=basis.device)].sign()
            basis = basis * signs
        retained = eigenvalues[-rank:].clamp_min(0).sum() / energy
        return basis, energy, retained


def weighted_future_l1(
    prediction: Tensor, target: Tensor, valid: Tensor, cameras: Tensor,
    mistake: Tensor, *, mistake_weight: float, weight: float, eps: float = 1e-6,
) -> Tensor:
    """Per-example contributions whose mean equals the valid weighted loss."""
    prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],), eps=eps)
    target = F.layer_norm(target.detach().float(), (target.shape[-1],), eps=eps)
    per_camera = (prediction - target).abs().mean(dim=(-1, -2))
    cameras = cameras.to(device=prediction.device, dtype=torch.bool)
    per_sample = (per_camera * cameras).sum(-1) / cameras.sum(-1).clamp_min(1)
    valid = valid.to(device=prediction.device, dtype=torch.bool) & cameras.any(-1)
    weights = valid.float() * (1 + (mistake_weight - 1) * mistake.float().to(prediction.device))
    # Sum, rather than mean, normalization keeps episode-tail samples from diluting
    # the auxiliary. All-invalid batches stay graph-connected with exactly zero loss.
    return weight * prediction.shape[0] * weights * per_sample / weights.sum().clamp_min(eps)


class FuturePredictor(nn.Module):
    def __init__(self, feature_dim: int, grid: tuple[int, int], cfg: FutureVisualLossConfig):
        super().__init__()
        self.input = nn.Linear(feature_dim, cfg.predictor_width)
        self.position = nn.Parameter(torch.zeros(1, math.prod(grid), cfg.predictor_width))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                cfg.predictor_width, cfg.predictor_heads, 4 * cfg.predictor_width,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(cfg.predictor_layers)
        ])
        self.output = nn.Linear(cfg.predictor_width, cfg.latent_dim)

    def forward(self, features: Tensor) -> Tensor:
        b, cameras, patches, _ = features.shape
        x = self.input(features.float().reshape(b * cameras, patches, -1)) + self.position
        for block in self.blocks:
            x = block(x)
        return self.output(x).reshape(b, cameras, patches, -1)


class FutureVisualObjective(nn.Module):
    def __init__(self, vision_backbone: nn.Module, num_cameras: int, cfg: FutureVisualLossConfig):
        super().__init__()
        self.cfg = cfg
        self.taps = tuple(int(i) for i in vision_backbone.vit_layers)
        self.tap_width = int(vision_backbone.image_vit.config.hidden_size)
        self.feature_dim = self.tap_width * len(self.taps)
        if cfg.latent_dim > self.feature_dim:
            raise ValueError("future_visual_loss.latent_dim exceeds the concatenated ViT width.")
        if getattr(vision_backbone, "num_prefix_tokens", 0):
            raise ValueError("future_visual_loss currently requires the no-CLS Molmo ViT.")
        self.target_vit = copy.deepcopy(vision_backbone.image_vit).requires_grad_(False).eval()
        grid = tuple(self.target_vit.config.image_num_patch)
        self.predictor = FuturePredictor(self.feature_dim, grid, cfg)
        device = next(self.target_vit.parameters()).device
        self.predictor.to(device)
        self.register_buffer("basis", torch.zeros(self.feature_dim, cfg.latent_dim, device=device))
        self.register_buffer("ready", torch.tensor(False, device=device))
        self.register_buffer("optimizer_steps", torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer("last_target_update", torch.tensor(-1, dtype=torch.long, device=device))
        self.register_buffer("last_pca_update", torch.tensor(-1, dtype=torch.long, device=device))
        self.register_buffer("temporal_energy", torch.tensor(0.0, device=device))
        self.register_buffer("retained_energy", torch.tensor(0.0, device=device))
        self.register_buffer("calibration_count", torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer("calibration_seen", torch.tensor(0, dtype=torch.long, device=device))
        patch_dim = self.target_vit.patch_embedding.in_features
        shape = (cfg.calibration_pairs, num_cameras, math.prod(grid), patch_dim)
        # Fixed shapes support safetensors resume and prevent reallocations in DDP.
        self.register_buffer("calibration_current", torch.zeros(shape, dtype=torch.bfloat16, device=device))
        self.register_buffer("calibration_future", torch.zeros(shape, dtype=torch.bfloat16, device=device))
        self.register_buffer(
            "calibration_cameras", torch.zeros(shape[:2], dtype=torch.bool, device=device)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_vit.eval()
        return self

    @staticmethod
    def _distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    @torch.no_grad()
    def encode_target(self, images: Tensor) -> Tensor:
        """True single-frame path, with bounded temporary activation memory."""
        self.target_vit.eval()
        b, cameras, patches, width = images.shape
        flat = images.reshape(-1, patches, width)
        dtype = next(self.target_vit.parameters()).dtype
        outputs = []
        for chunk in flat.split(self.cfg.target_encode_batch_size):
            # Mirror MolmoAct2VisionBackbone.forward's canonical SigLIP pixel
            # grid, including the rounding after the processor's bf16 cast.
            pixels = chunk.float()
            if chunk.dtype != torch.uint8:
                pixels = ((pixels + 1.0) * 0.5 * 255.0).round().clamp(0, 255)
            pixels = pixels / 255.0 * 2.0 - 1.0
            x = self.target_vit.patch_embedding(pixels.to(dtype=dtype))
            x = self.target_vit.add_pos_emb(x, self.target_vit.config.image_num_patch)
            selected = {}
            for index, block in enumerate(self.target_vit.transformer.resblocks):
                x = block(x)
                if index in self.taps:
                    selected[index] = x
            outputs.append(torch.cat([selected[index] for index in self.taps], dim=-1))
        return torch.cat(outputs).reshape(b, cameras, patches, self.feature_dim)

    @torch.no_grad()
    def _remember(self, current: Tensor, future: Tensor, valid: Tensor, cameras: Tensor):
        """A bounded reservoir spanning training batches, maintained on rank zero."""
        for row in (valid.bool() & cameras.any(-1)).nonzero(as_tuple=True)[0].tolist():
            self.calibration_seen.add_(1)
            seen = int(self.calibration_seen)
            slot = seen - 1 if seen <= self.cfg.calibration_pairs else int(
                torch.randint(seen, (), device=current.device)
            )
            if slot >= self.cfg.calibration_pairs:
                continue
            self.calibration_current[slot].copy_(current[row])
            self.calibration_future[slot].copy_(future[row])
            self.calibration_cameras[slot].copy_(cameras[row])
            self.calibration_count.fill_(min(seen, self.cfg.calibration_pairs))

    @torch.no_grad()
    def prepare(self, source_vit: nn.Module, current: Tensor, future: Tensor, valid: Tensor, cameras: Tensor):
        """Called before the training forward, identically on every DDP rank."""
        primary = not self._distributed() or dist.get_rank() == 0
        if primary:
            self._remember(current, future, valid, cameras)
        self._refresh(source_vit)

    @torch.no_grad()
    def _refresh(self, source_vit: nn.Module):
        primary = not self._distributed() or dist.get_rank() == 0
        step = int(self.optimizer_steps)
        interval = self.cfg.target_update_steps
        if int(self.last_target_update) < 0 or (
            step % interval == 0 and step != int(self.last_target_update)
        ):
            # Student parameters have already been synchronized by DDP.
            self.target_vit.load_state_dict(source_vit.state_dict(), strict=True)
            self.target_vit.requires_grad_(False).eval()
            self.last_target_update.fill_(step)
        needs_fit = not bool(self.ready) or int(self.last_pca_update) != int(self.last_target_update)
        if not needs_fit:
            return
        if primary and int(self.calibration_count):
            count = int(self.calibration_count)
            current_features = normalize_taps(
                self.encode_target(self.calibration_current[:count]), self.tap_width, self.cfg.eps
            )
            future_features = normalize_taps(
                self.encode_target(self.calibration_future[:count]), self.tap_width, self.cfg.eps
            )
            n = current_features.shape[-2]
            positions = torch.linspace(
                0, n - 1, min(n, self.cfg.calibration_patches_per_camera),
                device=current_features.device,
            ).long()
            differences = (future_features - current_features)[:, :, positions]
            differences = differences[self.calibration_cameras[:count]]
            result = temporal_pca(
                differences, self.cfg.latent_dim,
                self.basis if int(self.last_pca_update) >= 0 else None,
            )
            if result is not None:
                basis, energy, retained = result
                self.basis.copy_(basis)
                self.temporal_energy.copy_(energy)
                self.retained_energy.copy_(retained)
                self.ready.fill_(True)
                self.last_pca_update.copy_(self.last_target_update)
            else:
                # Do not train against a stale projection after a degenerate refresh.
                # Preserve the old basis for alignment when calibration recovers.
                self.ready.fill_(False)
                self.temporal_energy.zero_()
                self.retained_energy.zero_()
        if self._distributed():
            # Ignore these buffers in DDP's automatic broadcast; refresh only here.
            for value in (self.basis, self.ready, self.temporal_energy,
                          self.retained_energy, self.last_pca_update):
                dist.broadcast(value, src=0)

    @torch.no_grad()
    def optimizer_step(self, source_vit: nn.Module):
        self.optimizer_steps.add_(1)
        if int(self.optimizer_steps) % self.cfg.target_update_steps == 0:
            self._refresh(source_vit)

    def forward(self, current_features: Tensor, future: Tensor, valid: Tensor,
                cameras: Tensor, mistake: Tensor, reduction: str, current_images: Tensor | None = None):
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}.")
        features = normalize_taps(current_features, self.tap_width, self.cfg.eps)
        prediction = self.predictor(features)
        with torch.no_grad():
            target = normalize_taps(self.encode_target(future), self.tap_width, self.cfg.eps)
            target = target @ self.basis.float()
        active = valid.bool() & self.ready
        per_sample = weighted_future_l1(
            prediction, target, active, cameras, mistake,
            mistake_weight=self.cfg.mistake_weight, weight=self.cfg.weight, eps=self.cfg.eps,
        )
        with torch.no_grad():
            sample_weights = (active & cameras.any(-1)).float() * (
                1 + (self.cfg.mistake_weight - 1) * mistake.float()
            )
            valid_targets = target[active[:, None] & cameras.bool()].flatten(0, 1)
            target_std = valid_targets.float().std(dim=0, unbiased=False).mean() if valid_targets.numel() else 0.0
        metrics = {
            "future_visual_loss": per_sample.detach().mean().item(),
            "future_visual_ready": float(self.ready),
            "future_visual_weight_sum": sample_weights.sum().item(),
            "future_visual_valid_fraction": (active & cameras.any(-1)).float().mean().item(),
            "future_visual_temporal_energy": float(self.temporal_energy),
            "future_visual_pca_retained_energy": float(self.retained_energy),
            "future_visual_target_update_step": float(self.last_target_update),
            "future_visual_target_std": float(target_std),
        }
        if current_images is not None:
            with torch.no_grad():
                present = normalize_taps(
                    self.encode_target(current_images), self.tap_width, self.cfg.eps
                ) @ self.basis.float()
                for name, baseline in (("persistence", present), ("zero", torch.zeros_like(target))):
                    metrics[f"future_visual_{name}_loss"] = weighted_future_l1(
                        baseline, target, active, cameras, mistake,
                        mistake_weight=self.cfg.mistake_weight, weight=self.cfg.weight, eps=self.cfg.eps,
                    ).mean().item()
        return (per_sample.mean() if reduction == "mean" else per_sample), metrics

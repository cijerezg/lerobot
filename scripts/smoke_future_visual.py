"""Bounded synthetic smoke against local trained ViT weights; writes no checkpoint.

From the workspace root:
  .venv/bin/python lerobot/scripts/smoke_future_visual.py --base outputs/MolmoAct2 \
    --checkpoint outputs/molmoact2_rebot_nohistory_v1/checkpoints/001200/pretrained_model/model.safetensors
"""

import argparse
import json
import time
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from lerobot.policies.molmoact2.configuration_molmoact2 import FutureVisualLossConfig
from lerobot.policies.molmoact2.future_visual import FutureVisualObjective
from lerobot.policies.molmoact2.modeling_molmoact2 import _patch_memory_efficient_vision_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    config = AutoConfig.from_pretrained(args.base, trust_remote_code=True, local_files_only=True)
    cls = get_class_from_dynamic_module(
        "modeling_molmoact2.MolmoAct2VisionBackbone", args.base, local_files_only=True
    )
    config.vit_config._attn_implementation = "sdpa"
    config.adapter_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        vision = cls(config.vit_config, config.adapter_config)
    prefix = "model.model.vision_backbone."
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        state = {key.removeprefix(prefix): checkpoint.get_tensor(key)
                 for key in checkpoint.keys() if key.startswith(prefix)}
    vision.load_state_dict(state, strict=True, assign=True)
    vision = vision.to(device="cuda", dtype=torch.bfloat16)
    del state
    _patch_memory_efficient_vision_backbone(
        SimpleNamespace(vision_backbone=vision), gradient_checkpointing=True, temporal_layer_stride=4
    )
    cfg = FutureVisualLossConfig(enabled=True, target_update_steps=2, calibration_pairs=2)
    aux = FutureVisualObjective(vision, 2, cfg)
    patches = vision.image_vit.config.image_num_patch[0] ** 2
    pixels = vision.image_vit.patch_embedding.in_features
    current = (torch.rand(1, 2, patches, pixels, device="cuda") * 2 - 1).bfloat16()
    future = torch.roll(current, shifts=27, dims=2)
    valid = torch.ones(1, dtype=torch.bool, device="cuda")
    cameras = torch.ones(1, 2, dtype=torch.bool, device="cuda")
    # One spatial patch per pooled token is sufficient to exercise the genuine
    # backbone forward, its pixel canonicalization, and our feature capture seam.
    pooling = torch.arange(2 * patches, device="cuda").reshape(1, -1, 1)
    vision.eval()
    with torch.no_grad():
        vision._lerobot_capture_future_features = True
        vision(current, pooling)
        expected = vision._lerobot_future_features
        target = aux.encode_target(current)
        torch.testing.assert_close(target, expected, atol=0.03, rtol=0.01)
    del expected, target
    torch.cuda.synchronize()
    start = time.monotonic()
    aux.prepare(vision.image_vit, current, future, valid, cameras)
    torch.cuda.synchronize()
    calibration_seconds = time.monotonic() - start
    assert aux.ready
    vision.train()
    history = current.unsqueeze(2).clone().detach().requires_grad_()
    vision._lerobot_history = (history, torch.tensor([2.0], device="cuda"), cameras)
    vision._lerobot_capture_future_features = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vision(current, pooling)
        features = vision._lerobot_future_features
        vision._lerobot_future_features = None
        loss, metrics = aux(features, future, valid, cameras, torch.ones(1, device="cuda"), "mean")
    loss.backward()
    assert torch.isfinite(loss) and history.grad.abs().sum() > 0
    assert all(p.grad is None for p in aux.target_vit.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in vision.image_vit.parameters())
    optimizer = torch.optim.SGD(aux.predictor.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.step()
        aux.optimizer_step(vision.image_vit)
    assert aux.last_target_update == 2 and aux.last_pca_update == 2
    aux.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, validation = aux(
            features.detach(), future, valid, cameras, torch.ones(1, device="cuda"), "mean",
            current_images=current,
        )
    assert all(torch.isfinite(torch.tensor(value)) for value in validation.values())
    print(json.dumps({
        "feature_shape": list(features.shape), "executed_layers": len(vision.image_vit.transformer.resblocks),
        "taps": vision.vit_layers, "latent_dim": cfg.latent_dim,
        "calibration_seconds": round(calibration_seconds, 3),
        "history_gradient_l1": float(history.grad.float().abs().sum()),
        "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "train": metrics, "validation": validation,
    }, indent=2))


if __name__ == "__main__":
    main()

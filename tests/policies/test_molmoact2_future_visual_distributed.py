"""Exercise the auxiliary's explicit refresh collectives alongside DDP."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from tests.policies.test_molmoact2_future_visual import batch, objective


class AuxiliaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.future_visual = objective()
        self._ddp_params_and_buffers_to_ignore = {
            f"future_visual.{name}" for name, _ in self.future_visual.named_buffers()
        }

    def forward(self, current, future, valid, cameras):
        features = self.backbone.vision_backbone.encode_image(current)
        loss, _ = self.future_visual(features, future, valid, cameras, torch.zeros(4), "mean")
        return loss


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=30)
    )
    try:
        torch.manual_seed(71 + rank)
        model = AuxiliaryModel()
        wrapped = DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        aux = model.future_visual
        source = model.backbone.vision_backbone.image_vit
        for step in range(3):
            args = batch()
            aux.prepare(source, *args)
            optimizer.zero_grad()
            loss = wrapped(*args)
            assert torch.isfinite(loss) and aux.ready
            loss.backward()
            optimizer.step()
            aux.optimizer_step(source)
            basis_by_rank = [torch.empty_like(aux.basis) for _ in range(2)]
            dist.all_gather(basis_by_rank, aux.basis)
            torch.testing.assert_close(basis_by_rank[0], basis_by_rank[1], rtol=0, atol=0)
            assert aux.optimizer_steps == step + 1
            assert aux.last_target_update == (2 if step >= 1 else 0)
            assert int(aux.calibration_count) == (min(3 * (step + 1), 4) if rank == 0 else 0)
            assert all(p.grad is None for p in aux.target_vit.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires the local Gloo backend")
def test_distributed_refresh_keeps_basis_in_sync_without_broadcasting_reservoir(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)

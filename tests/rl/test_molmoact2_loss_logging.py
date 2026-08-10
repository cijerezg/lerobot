from __future__ import annotations

import torch

from lerobot.probes.objective import wandb_scalars
from lerobot.rl.molmoact2.val_loss import ValLoss


class _FakePolicy:
    def __init__(self, *, auxiliary: bool = True) -> None:
        self.training = True
        self.auxiliary = auxiliary

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def forward(self, batch, **kwargs):
        marker = float(batch["marker"].item())
        metrics = {
            "action_flow_loss": marker,
            "discrete_ce_loss": marker + 2.0,
        }
        if self.auxiliary:
            metrics.update(
                action_auxiliary_loss=2.0 * marker,
                discrete_auxiliary_loss=2.0 * marker + 2.0,
            )
        return torch.tensor(0.0), metrics


def _val_loss() -> ValLoss:
    val_loss = ValLoss.__new__(ValLoss)
    val_loss._device = "cpu"
    val_loss._counts = [2, 1]
    val_loss._batches = [
        {
            "marker": torch.tensor(1.0),
            "_flow_timesteps": torch.zeros(1, 1),
            "_flow_noise": torch.zeros(1, 1, 1, 1),
        },
        {
            "marker": torch.tensor(4.0),
            "_flow_timesteps": torch.zeros(1, 1),
            "_flow_noise": torch.zeros(1, 1, 1, 1),
        },
    ]
    return val_loss


def test_val_loss_logs_enabled_auxiliaries_with_frame_weighting() -> None:
    policy = _FakePolicy(auxiliary=True)

    metrics = _val_loss()(policy)

    assert metrics == {
        "val_loss_flow": 2.0,
        "val_loss_discrete_ce": 4.0,
        "val_loss_action_aux": 4.0,
        "val_loss_discrete_aux": 6.0,
    }
    assert policy.training


def test_val_loss_omits_disabled_auxiliaries() -> None:
    metrics = _val_loss()(_FakePolicy(auxiliary=False))

    assert set(metrics) == {"val_loss_flow", "val_loss_discrete_ce"}


def test_objective_wandb_scalars_include_headlines_only() -> None:
    summary = {
        "loss_flow": {"val": 1.0},
        "loss_action_aux": {"val": 2.0, "train": 1.5, "gap": 0.5, "z": 3.0},
        "loss_discrete_ce": {"val": 3.0},
        "loss_discrete_aux": {"val": 4.0},
        "action_aux_path_mse": {"val": 5.0},
        "discrete_aux_ordinal_ce": {"val": 6.0},
        "discrete": {
            "val": {"top1": 0.4},
            "train": {"top1": 0.8},
        },
    }

    scalars = wandb_scalars(summary)

    assert scalars == {
        "objective_z_action_aux": 3.0,
        "objective_val_fast_top1": 0.4,
    }



def test_training_wandb_metrics_use_compact_allowlist() -> None:
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer

    metrics = {
        "Optimization step": 12,
        "loss_flow": 0.2,
        "loss_discrete_z": 0.0004,
        "action_aux/path_relative_mean": 0.7,
        "action_aux/path_relative_p50": 0.6,
        "action_aux/path_mse_mean": 0.03,
        "actor_loss_histogram": object(),
        "depth_block3_token_rms": 1.2,
    }

    assert MolmoAct2Trainer._wandb_metrics(metrics) == {
        "Optimization step": 12,
        "loss_flow": 0.2,
        "action_aux/path_relative_mean": 0.7,
    }


def test_depth_optimizer_inherits_joint_lr_when_override_is_null() -> None:
    from types import SimpleNamespace

    from torch import nn

    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer

    class _Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Linear(2, 2)
            self.pointmap_encoder = nn.Linear(2, 2)

    policy = _Policy()
    cfg = SimpleNamespace(policy=SimpleNamespace(optimizer_lr=3e-5, depth_lr=None))
    groups = MolmoAct2Trainer._split_depth_group(
        policy,
        cfg,
        [{"name": "policy", "params": list(policy.parameters()), "lr": cfg.policy.optimizer_lr}],
    )

    by_name = {group["name"]: group for group in groups}
    assert by_name["policy"]["lr"] == 3e-5
    assert by_name["depth"]["lr"] == 3e-5
    depth_ids = {id(parameter) for parameter in policy.pointmap_encoder.parameters()}
    assert {id(parameter) for parameter in by_name["depth"]["params"]} == depth_ids
